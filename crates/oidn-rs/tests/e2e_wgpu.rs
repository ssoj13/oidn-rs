//! End-to-end integration tests on the actual wgpu backend.
//!
//! Each test exercises a different slice of the pipeline on a real GPU:
//! - colour-only HDR denoise on a small tile (smoke test)
//! - colour + albedo + normal AOV path (`rt_hdr_alb_nrm` model)
//! - larger 512×512 tile (still single-tile; sanity check on dimensions)
//! - actual noise-reduction check: denoised RMSE vs clean reference must be
//!   smaller than noisy RMSE vs clean reference, proving the network does
//!   real work rather than passing the signal through unchanged.

use std::path::PathBuf;

use oidn_rs::prelude::wgpu_prelude::*;
use oidn_rs::prelude::*;
#[path = "../../oidn-cli/src/support.rs"]
pub mod support;
use support::{add_noise, make_clean, make_normal, metrics};

fn weights_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is the crate root (`crates/oidn-rs`), regardless of
    // how cargo test was invoked. Going up two levels lands at the workspace
    // root where `data/` lives.
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("weights");
    assert!(p.is_dir(), "required shipped weights are missing");
    p
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_small_hdr_color_only_wgpu() {
    let dir = weights_dir();

    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let clean = make_clean(w, h).unwrap();
    let noisy = add_noise(&clean, 0.15).unwrap();

    let in_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(true)
        .quality(Quality::High)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&in_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    filter.execute().expect("execute");

    let (raw, ow, oh, fmt) = filter.take_output().unwrap();
    assert_eq!((ow, oh, fmt), (w, h, PixelFormat::Rgb32f));
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");

    for x in out {
        assert!(x.is_finite());
    }
    let mean_in: f32 = noisy.iter().sum::<f32>() / noisy.len() as f32;
    let mean_out: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!(
        (mean_out - mean_in).abs() < 1.0,
        "wgpu output mean drift too large: in={mean_in} out={mean_out}"
    );

    eprintln!("64x64 colour-only OK — input mean={mean_in:.4}, output mean={mean_out:.4}");
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_with_albedo_normal_wgpu() {
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let clean = make_clean(w, h).unwrap();
    let noisy = add_noise(&clean, 0.2).unwrap();

    // Synthetic AOVs: albedo = clean colour clamped, normal = constant up-vector.
    let albedo: Vec<f32> = clean.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let normal = make_normal(w, h).unwrap();

    let color_img = Image::from_rgb_f32(&noisy, w, h);
    let albedo_img = Image::from_rgb_f32(&albedo, w, h);
    let normal_img = Image::from_rgb_f32(&normal, w, h);

    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(true)
        .quality(Quality::High)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&color_img).unwrap();
    filter.set_albedo(&albedo_img).unwrap();
    filter.set_normal(&normal_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");

    // Must have routed to the 9-channel model.
    assert_eq!(filter.model_key().unwrap().0, "rt_hdr_alb_nrm");

    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite());
    }

    eprintln!("64x64 color+albedo+normal OK on rt_hdr_alb_nrm");
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_512x512_wgpu() {
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (512usize, 512usize);
    let clean = make_clean(w, h).unwrap();
    let noisy = add_noise(&clean, 0.1).unwrap();

    let in_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(true)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&in_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite());
    }
    eprintln!("512x512 OK");
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoiser_actually_reduces_noise_wgpu() {
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (256usize, 256usize);
    let clean = make_clean(w, h).unwrap();
    let noisy = add_noise(&clean, 0.12).unwrap();

    let noisy_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(true)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&noisy_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    let denoised = out.to_vec();

    let rmse_noisy = metrics(&noisy, &clean).unwrap().rmse;
    let rmse_denoised = metrics(&denoised, &clean).unwrap().rmse;

    eprintln!(
        "RMSE vs clean: noisy={rmse_noisy:.5}  denoised={rmse_denoised:.5}  improvement={:.2}x",
        rmse_noisy / rmse_denoised.max(1e-12)
    );

    assert!(
        rmse_denoised < rmse_noisy,
        "denoiser did not reduce error: noisy={rmse_noisy} denoised={rmse_denoised}"
    );
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_albedo_only_wgpu() {
    // AOV-only filter: only albedo provided, no colour.
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    // Albedo is in [0, 1].
    let albedo: Vec<f32> = make_clean(w, h)
        .unwrap()
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0))
        .collect();
    let albedo_img = Image::from_rgb_f32(&albedo, w, h);

    // Default Quality::High prefers `_large` when available — OIDN spec
    // (see _ref/oidn/core/unet_filter.cpp:450).
    let mut filter = RtFilter::builder(&device.handle, &dir).build();
    filter.set_albedo(&albedo_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_alb_large");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite());
    }
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_normal_only_wgpu() {
    // AOV-only filter: only normal provided.
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let mut normal = vec![0.0f32; w * h * 3];
    // Wave-like normal — varies across image so the network has structure to work with.
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let nx = ((x as f32 / w as f32) * 2.0 - 1.0) * 0.5;
            let ny = ((y as f32 / h as f32) * 2.0 - 1.0) * 0.5;
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            normal[i] = nx;
            normal[i + 1] = ny;
            normal[i + 2] = nz;
        }
    }
    let normal_img = Image::from_rgb_f32(&normal, w, h);

    let mut filter = RtFilter::builder(&device.handle, &dir).build();
    filter.set_normal(&normal_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_nrm_large");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite());
    }
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_with_clean_aux_wgpu() {
    // cleanAux=true routes to *_calb_cnrm model (clean albedo + clean normal).
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let clean = make_clean(w, h).unwrap();
    let noisy = add_noise(&clean, 0.12).unwrap();

    // Synthetic "already denoised" AOVs.
    let albedo: Vec<f32> = clean.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let normal = make_normal(w, h).unwrap();

    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(true)
        .clean_aux(true)
        .quality(oidn_rs::Quality::Balanced) // Balanced ⇒ base only, easier to assert key.
        .input_scale(Some(1.0))
        .build();
    filter
        .set_color(&Image::from_rgb_f32(&noisy, w, h))
        .unwrap();
    filter
        .set_albedo(&Image::from_rgb_f32(&albedo, w, h))
        .unwrap();
    filter
        .set_normal(&Image::from_rgb_f32(&normal, w, h))
        .unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_hdr_calb_cnrm");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite());
    }

    let denoised = out.to_vec();
    let rmse_noisy = metrics(&noisy, &clean).unwrap().rmse;
    let rmse_denoised = metrics(&denoised, &clean).unwrap().rmse;
    eprintln!("cleanAux: noisy rmse={rmse_noisy:.5} denoised rmse={rmse_denoised:.5}");
    assert!(
        rmse_denoised < rmse_noisy,
        "cleanAux denoiser did not reduce error"
    );
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn quality_fast_routes_to_small_wgpu() {
    // Quality::Fast prefers _small variant when available.
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let clean = make_clean(w, h).unwrap();
    let noisy = add_noise(&clean, 0.1).unwrap();

    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(true)
        .quality(oidn_rs::Quality::Fast)
        .input_scale(Some(1.0))
        .build();
    filter
        .set_color(&Image::from_rgb_f32(&noisy, w, h))
        .unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_hdr_small");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite());
    }
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_lightmap_hdr_wgpu() {
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let color = make_clean(w, h).unwrap(); // positive HDR-ish irradiance
    let color_img = Image::from_rgb_f32(&color, w, h);

    let mut filter = RtLightmapFilter::builder(&device.handle, &dir)
        .directional(false)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&color_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rtlightmap_hdr");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite(), "non-finite output from rtlightmap_hdr");
    }
}

#[test]
#[ignore = "requires explicit GPU verification; run with --ignored"]
fn denoise_lightmap_directional_wgpu() {
    let dir = weights_dir();
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    // Directional lightmap stores signed irradiance gradients — values can
    // be negative.
    let mut color = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            color[i] = (x as f32 / w as f32) * 2.0 - 1.0;
            color[i + 1] = (y as f32 / h as f32) * 2.0 - 1.0;
            color[i + 2] = ((x + y) as f32 / (w + h) as f32) * 2.0 - 1.0;
        }
    }
    let color_img = Image::from_rgb_f32(&color, w, h);

    let mut filter = RtLightmapFilter::builder(&device.handle, &dir)
        .directional(true)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&color_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rtlightmap_dir");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    assert!(out.iter().any(|v| v.abs() > 1e-6), "output lost all signal");
    for x in out {
        assert!(x.is_finite(), "non-finite output from rtlightmap_dir");
    }
}
