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

use oidn_rs::prelude::*;

fn weights_dir() -> Option<PathBuf> {
    // CARGO_MANIFEST_DIR is the crate root (`crates/oidn-rs`), regardless of
    // how cargo test was invoked. Going up two levels lands at the workspace
    // root where `data/` lives.
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data").join("weights");
    if p.is_dir() { Some(p) } else { None }
}

/// Smooth radial gradient — used as the ground truth for noise-reduction tests.
fn make_clean(w: usize, h: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; w * h * 3];
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let rmax = (cx * cx + cy * cy).sqrt();
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt() / rmax;
            let v = 0.7 + 0.25 * (1.0 - r);
            let i = (y * w + x) * 3;
            buf[i]     = v;
            buf[i + 1] = v * 0.9;
            buf[i + 2] = v * 0.7;
        }
    }
    buf
}

/// Deterministic per-pixel hash-noise (no `rand` dep). Magnitude controls the
/// amount of noise added on top of the clean image.
fn add_noise(clean: &[f32], magnitude: f32) -> Vec<f32> {
    let mut out = clean.to_vec();
    for (i, v) in out.iter_mut().enumerate() {
        // xorshift-ish from index → uniform-ish f32 in [-1, 1]
        let mut n = (i as u32).wrapping_mul(2654435761);
        n ^= n >> 13;
        n = n.wrapping_mul(0x85ebca6b);
        n ^= n >> 16;
        let r = (n as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *v += r * magnitude;
    }
    out
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let s: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (s / n).sqrt()
}

#[test]
fn denoise_small_hdr_color_only_wgpu() {
    let Some(dir) = weights_dir() else {
        eprintln!("skipping: weights submodule not initialised");
        return;
    };

    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let clean = make_clean(w, h);
    let noisy = add_noise(&clean, 0.15);

    let in_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, &dir)
        .hdr(true)
        .quality(Quality::High)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&in_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    filter.execute().expect("execute");

    let (raw, ow, oh, fmt) = filter.take_output().unwrap();
    assert_eq!((ow, oh, fmt), (w, h, PixelFormat::Rgb32f));
    let out: &[f32] = bytemuck::cast_slice(&raw);

    for x in out { assert!(x.is_finite()); }
    let mean_in: f32 = noisy.iter().sum::<f32>() / noisy.len() as f32;
    let mean_out: f32 = out.iter().sum::<f32>() / out.len() as f32;
    assert!((mean_out - mean_in).abs() < 1.0,
            "wgpu output mean drift too large: in={mean_in} out={mean_out}");

    eprintln!("64x64 colour-only OK — input mean={mean_in:.4}, output mean={mean_out:.4}");
}

#[test]
fn denoise_with_albedo_normal_wgpu() {
    let Some(dir) = weights_dir() else { return; };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let clean = make_clean(w, h);
    let noisy = add_noise(&clean, 0.2);

    // Synthetic AOVs: albedo = clean colour clamped, normal = constant up-vector.
    let albedo: Vec<f32> = clean.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let mut normal = vec![0.0f32; w * h * 3];
    for px in normal.chunks_exact_mut(3) {
        px[0] = 0.0;
        px[1] = 1.0;
        px[2] = 0.0;
    }

    let color_img  = Image::from_rgb_f32(&noisy,  w, h);
    let albedo_img = Image::from_rgb_f32(&albedo, w, h);
    let normal_img = Image::from_rgb_f32(&normal, w, h);

    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, &dir)
        .hdr(true)
        .quality(Quality::High)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&color_img);
    filter.set_albedo(&albedo_img);
    filter.set_normal(&normal_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");

    // Must have routed to the 9-channel model.
    assert_eq!(filter.model_key().unwrap().0, "rt_hdr_alb_nrm");

    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite()); }

    eprintln!("64x64 color+albedo+normal OK on rt_hdr_alb_nrm");
}

#[test]
fn denoise_512x512_wgpu() {
    let Some(dir) = weights_dir() else { return; };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (512usize, 512usize);
    let clean = make_clean(w, h);
    let noisy = add_noise(&clean, 0.1);

    let in_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, &dir)
        .hdr(true)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&in_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite()); }
    eprintln!("512x512 OK");
}

#[test]
fn denoiser_actually_reduces_noise_wgpu() {
    let Some(dir) = weights_dir() else { return; };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (256usize, 256usize);
    let clean = make_clean(w, h);
    let noisy = add_noise(&clean, 0.12);

    let noisy_img = Image::from_rgb_f32(&noisy, w, h);
    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, &dir)
        .hdr(true)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&noisy_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    let denoised = out.to_vec();

    let rmse_noisy    = rmse(&noisy,    &clean);
    let rmse_denoised = rmse(&denoised, &clean);

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
fn denoise_albedo_only_wgpu() {
    // AOV-only filter: only albedo provided, no colour.
    let Some(dir) = weights_dir() else { return; };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    // Albedo is in [0, 1].
    let albedo: Vec<f32> = make_clean(w, h).into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let albedo_img = Image::from_rgb_f32(&albedo, w, h);

    // Default Quality::High prefers `_large` when available — OIDN spec
    // (see _ref/oidn/core/unet_filter.cpp:450).
    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, &dir).build();
    filter.set_albedo(&albedo_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_alb_large");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite()); }
}

#[test]
fn denoise_normal_only_wgpu() {
    // AOV-only filter: only normal provided.
    let Some(dir) = weights_dir() else { return; };
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
            normal[i]     = nx;
            normal[i + 1] = ny;
            normal[i + 2] = nz;
        }
    }
    let normal_img = Image::from_rgb_f32(&normal, w, h);

    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, &dir).build();
    filter.set_normal(&normal_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_nrm_large");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite()); }
}

#[test]
fn denoise_with_clean_aux_wgpu() {
    // cleanAux=true routes to *_calb_cnrm model (clean albedo + clean normal).
    let Some(dir) = weights_dir() else { return; };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let clean = make_clean(w, h);
    let noisy = add_noise(&clean, 0.12);

    // Synthetic "already denoised" AOVs.
    let albedo: Vec<f32> = clean.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let mut normal = vec![0.0f32; w * h * 3];
    for px in normal.chunks_exact_mut(3) {
        px[0] = 0.0; px[1] = 1.0; px[2] = 0.0;
    }

    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, &dir)
        .hdr(true)
        .clean_aux(true)
        .quality(oidn_rs::Quality::Balanced)  // Balanced ⇒ base only, easier to assert key.
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&Image::from_rgb_f32(&noisy,  w, h));
    filter.set_albedo(&Image::from_rgb_f32(&albedo, w, h));
    filter.set_normal(&Image::from_rgb_f32(&normal, w, h));
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_hdr_calb_cnrm");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite()); }

    let denoised = out.to_vec();
    let rmse_noisy    = rmse(&noisy,    &clean);
    let rmse_denoised = rmse(&denoised, &clean);
    eprintln!("cleanAux: noisy rmse={rmse_noisy:.5} denoised rmse={rmse_denoised:.5}");
    assert!(rmse_denoised < rmse_noisy, "cleanAux denoiser did not reduce error");
}

#[test]
fn quality_fast_routes_to_small_wgpu() {
    // Quality::Fast prefers _small variant when available.
    let Some(dir) = weights_dir() else { return; };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let clean = make_clean(w, h);
    let noisy = add_noise(&clean, 0.1);

    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, &dir)
        .hdr(true)
        .quality(oidn_rs::Quality::Fast)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&Image::from_rgb_f32(&noisy, w, h));
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rt_hdr_small");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite()); }
}

#[test]
fn denoise_lightmap_hdr_wgpu() {
    use oidn_rs::RtLightmapFilter;

    let Some(dir) = weights_dir() else { return; };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    let color = make_clean(w, h);   // positive HDR-ish irradiance
    let color_img = Image::from_rgb_f32(&color, w, h);

    let mut filter = RtLightmapFilter::<WgpuBackend>::builder(&device.handle, &dir)
        .directional(false)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&color_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rtlightmap_hdr");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite(), "non-finite output from rtlightmap_hdr"); }
}

#[test]
fn denoise_lightmap_directional_wgpu() {
    use oidn_rs::RtLightmapFilter;

    let Some(dir) = weights_dir() else { return; };
    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (64usize, 64usize);
    // Directional lightmap stores signed irradiance gradients — values can
    // be negative.
    let mut color = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            color[i]     = (x as f32 / w as f32) * 2.0 - 1.0;
            color[i + 1] = (y as f32 / h as f32) * 2.0 - 1.0;
            color[i + 2] = ((x + y) as f32 / (w + h) as f32) * 2.0 - 1.0;
        }
    }
    let color_img = Image::from_rgb_f32(&color, w, h);

    let mut filter = RtLightmapFilter::<WgpuBackend>::builder(&device.handle, &dir)
        .directional(true)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&color_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit");
    assert_eq!(filter.model_key().unwrap().0, "rtlightmap_dir");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite(), "non-finite output from rtlightmap_dir"); }
}
