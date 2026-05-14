//! End-to-end integration test on the CPU NdArray backend.
//!
//! Runs the full pipeline (TZA load → tile plan → UNet forward → write output)
//! on a small synthetic image without touching any GPU. This proves all
//! glue is correct; the wgpu backend swaps in transparently.

use std::path::PathBuf;

use burn::backend::NdArray;
use oidn_rs::{Filter, Image, PixelFormat, Quality, RtFilter};

fn weights_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data").join("weights");
    if p.is_dir() { Some(p) } else { None }
}

#[test]
fn denoise_small_hdr_color_only_ndarray() {
    let Some(dir) = weights_dir() else {
        eprintln!("skipping: weights submodule not initialised");
        return;
    };

    type B = NdArray<f32>;
    let device = <B as burn::tensor::backend::BackendTypes>::Device::default();

    // Generate a synthetic noisy 64×64 HDR colour image (a smooth gradient
    // with additive noise — pixel values in [0, 4]).
    let (w, h) = (64usize, 64usize);
    let mut color = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let g = 0.5 + 0.3 * ((x as f32 + y as f32) / (w as f32 + h as f32));
            let n = ((x * 17 + y * 31) % 19) as f32 * 0.05;
            let i = (y * w + x) * 3;
            color[i]     = g + n;
            color[i + 1] = g + n * 0.7;
            color[i + 2] = g + n * 0.4;
        }
    }

    let mut output = vec![0.0f32; w * h * 3];
    let in_img = Image::from_rgb_f32(&color, w, h);

    let mut filter = RtFilter::<B>::builder(&device, &dir)
        .hdr(true)
        .quality(Quality::High)
        .input_scale(Some(1.0)) // skip autoexposure to keep test deterministic
        .build();
    filter.set_color(&in_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    filter.execute().expect("execute");

    // Move output into the local buffer.
    let (raw, ow, oh, fmt) = filter.take_output().unwrap();
    assert_eq!((ow, oh, fmt), (w, h, PixelFormat::Rgb32f));
    let out_pixels: &[f32] = bytemuck::cast_slice(&raw);
    output.copy_from_slice(out_pixels);

    // Sanity checks: finite, non-trivial output, broadly close in luminance to
    // the input (denoise shouldn't shift mean intensity by orders of magnitude).
    let mean_in: f32 = color.iter().sum::<f32>() / color.len() as f32;
    let mean_out: f32 = output.iter().sum::<f32>() / output.len() as f32;
    for x in &output { assert!(x.is_finite(), "non-finite output value"); }
    assert!((mean_out - mean_in).abs() < 1.0,
            "output mean ({mean_out}) drifted too far from input ({mean_in})");
}

#[test]
fn rt_filter_picks_correct_model_key() {
    let Some(dir) = weights_dir() else { return; };

    type B = NdArray<f32>;
    let device = <B as burn::tensor::backend::BackendTypes>::Device::default();

    // hdr + albedo + normal → rt_hdr_alb_nrm
    let (w, h) = (32usize, 32usize);
    let buf = vec![0.5f32; w * h * 3];
    let img = Image::from_rgb_f32(&buf, w, h);

    let mut filter = RtFilter::<B>::builder(&device, &dir)
        .hdr(true)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&img);
    filter.set_albedo(&img);
    filter.set_normal(&img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_hdr_alb_nrm");
}
