//! LDR pipeline end-to-end test on wgpu — verifies the sRGB transfer
//! codepath plus the `rt_ldr` model route.

use std::path::PathBuf;

use oidn_rs::prelude::wgpu_prelude::*;
use oidn_rs::prelude::*;

fn weights_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("weights");
    if p.is_dir() { Some(p) } else { None }
}

fn make_clean_ldr(w: usize, h: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            // sRGB-ish gradient values strictly inside [0, 1].
            let r = (x as f32 / w as f32).clamp(0.0, 1.0);
            let g = (y as f32 / h as f32).clamp(0.0, 1.0);
            let b = ((x + y) as f32 / (w + h) as f32).clamp(0.0, 1.0);
            let i = (y * w + x) * 3;
            buf[i] = r * 0.9 + 0.05;
            buf[i + 1] = g * 0.9 + 0.05;
            buf[i + 2] = b * 0.9 + 0.05;
        }
    }
    buf
}

fn add_noise(clean: &[f32], magnitude: f32) -> Vec<f32> {
    let mut out = clean.to_vec();
    for (i, v) in out.iter_mut().enumerate() {
        let mut n = (i as u32).wrapping_mul(2654435761);
        n ^= n >> 13;
        n = n.wrapping_mul(0x85ebca6b);
        n ^= n >> 16;
        let r = (n as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *v = (*v + r * magnitude).clamp(0.0, 1.0);
    }
    out
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let s: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (s / n).sqrt()
}

#[test]
fn denoise_ldr_srgb_wgpu_reduces_noise() {
    let Some(dir) = weights_dir() else {
        eprintln!("skipping: weights submodule not initialised");
        return;
    };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (256usize, 256usize);
    let clean = make_clean_ldr(w, h);
    let noisy = add_noise(&clean, 0.08);

    let in_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(false) // LDR path
        .srgb(false) // default: input is sRGB-encoded, runner picks SRGB transfer
        .quality(Quality::High)
        .build();
    filter.set_color(&in_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_ldr");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    let denoised = out.to_vec();

    for x in &denoised {
        assert!(x.is_finite());
    }

    let rmse_noisy = rmse(&noisy, &clean);
    let rmse_denoised = rmse(&denoised, &clean);
    eprintln!(
        "LDR sRGB: rmse noisy={rmse_noisy:.5} denoised={rmse_denoised:.5} improvement={:.2}x",
        rmse_noisy / rmse_denoised.max(1e-12)
    );
    assert!(
        rmse_denoised < rmse_noisy,
        "LDR denoiser did not reduce error: noisy={rmse_noisy} denoised={rmse_denoised}"
    );
}

#[test]
fn denoise_ldr_explicit_linear_route_wgpu() {
    // hdr=false, srgb=true → input already linear, network applies Linear transfer.
    // Still routes to rt_ldr model.
    let Some(dir) = weights_dir() else {
        return;
    };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (128usize, 128usize);
    let clean = make_clean_ldr(w, h);
    let noisy = add_noise(&clean, 0.05);

    let in_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(false)
        .srgb(true)
        .build();
    filter.set_color(&in_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_ldr");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out {
        assert!(x.is_finite());
    }
}
