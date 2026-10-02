//! LDR pipeline end-to-end test on wgpu — verifies the sRGB transfer
//! codepath plus the `rt_ldr` model route.

use std::path::PathBuf;

use oidn_rs::prelude::wgpu_prelude::*;
use oidn_rs::prelude::*;
#[path = "../../oidn-cli/src/support.rs"]
pub mod support;
use support::{add_noise, metrics};

fn weights_dir() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("weights");
    assert!(p.is_dir(), "required shipped weights are missing");
    p
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

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_ldr_srgb_wgpu_reduces_noise() {
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (256usize, 256usize);
    let clean = make_clean_ldr(w, h);
    let noisy = add_noise(&clean, 0.08)
        .unwrap()
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0))
        .collect::<Vec<_>>();

    let in_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(false) // LDR path
        .srgb(false) // Linear input; the runner applies the sRGB forward transfer.
        .quality(Quality::High)
        .build();
    filter.set_color(&in_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_ldr");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    let denoised = out.to_vec();

    for x in &denoised {
        assert!(x.is_finite());
    }

    let rmse_noisy = metrics(&noisy, &clean).unwrap().rmse;
    let rmse_denoised = metrics(&denoised, &clean).unwrap().rmse;
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
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_ldr_explicit_linear_route_wgpu() {
    // hdr=false, srgb=true: input is already sRGB encoded; network transfer is Linear.
    // Still routes to rt_ldr model.
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (128usize, 128usize);
    let clean = make_clean_ldr(w, h);
    let noisy = add_noise(&clean, 0.05)
        .unwrap()
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0))
        .collect::<Vec<_>>();

    let in_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(false)
        .srgb(true)
        .build();
    filter.set_color(&in_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_ldr");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite());
    }
}
