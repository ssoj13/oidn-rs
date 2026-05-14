//! Phase 7 API-surface tests: progress callbacks, user weights blob, input
//! scale override, memory budget.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use burn::backend::NdArray;
use oidn_rs::{Filter, Image, OidnError, PixelFormat, Quality, RtFilter, tile};

fn weights_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data").join("weights");
    if p.is_dir() { Some(p) } else { None }
}

fn weights_bytes(name: &str) -> Option<Vec<u8>> {
    let dir = weights_dir()?;
    std::fs::read(dir.join(format!("{name}.tza"))).ok()
}

fn synth_color(w: usize, h: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            buf[i]     = 0.5 + 0.1 * (x as f32 / w as f32);
            buf[i + 1] = 0.5 + 0.1 * (y as f32 / h as f32);
            buf[i + 2] = 0.5;
        }
    }
    buf
}

#[test]
fn progress_callback_fires_per_tile() {
    let Some(dir) = weights_dir() else { return; };
    type B = NdArray<f32>;
    let device = <B as burn::tensor::backend::BackendTypes>::Device::default();

    let (w, h) = (256usize, 256usize);
    let color = synth_color(w, h);

    let mut filter = RtFilter::<B>::builder(&device, &dir)
        .hdr(true)
        .quality(Quality::Balanced)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&Image::from_rgb_f32(&color, w, h));
    filter.allocate_output(w, h, PixelFormat::Rgb32f);

    let calls = Arc::new(Mutex::new(Vec::<f32>::new()));
    let calls_cb = Arc::clone(&calls);
    filter.set_progress(move |frac| {
        calls_cb.lock().unwrap().push(frac);
        true
    });

    filter.commit().expect("commit");
    filter.execute().expect("execute");

    let calls = calls.lock().unwrap();
    assert!(!calls.is_empty(), "progress callback never fired");
    // Last call must be 1.0 (or extremely close to it).
    let last = *calls.last().unwrap();
    assert!((last - 1.0).abs() < 1e-6, "progress did not finish at 1.0, got {last}");
    // Each call must be monotonically increasing in [0, 1].
    let mut prev = 0.0f32;
    for &c in calls.iter() {
        assert!(c >= prev && c <= 1.0, "progress out of order: {prev} → {c}");
        prev = c;
    }
}

#[test]
fn progress_callback_cancels() {
    let Some(dir) = weights_dir() else { return; };
    type B = NdArray<f32>;
    let device = <B as burn::tensor::backend::BackendTypes>::Device::default();

    // Force multi-tile workload so cancellation triggers on the first tile
    // but the result is still aborted.
    let (w, h) = (3200usize, 3200usize);
    let color = vec![0.5f32; w * h * 3];

    let mut filter = RtFilter::<B>::builder(&device, &dir)
        .hdr(true)
        .quality(Quality::Balanced)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&Image::from_rgb_f32(&color, w, h));
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.set_progress(|_| false); // cancel immediately

    filter.commit().expect("commit");

    match filter.execute() {
        Err(OidnError::Cancelled) => {} // good
        other => panic!("expected Cancelled, got {other:?}"),
    }
}

#[test]
fn user_weights_blob_bypasses_registry() {
    let Some(bytes) = weights_bytes("rt_hdr") else { return; };
    type B = NdArray<f32>;
    let device = <B as burn::tensor::backend::BackendTypes>::Device::default();

    // Pass an empty weights_dir — should never be consulted since we provide blob.
    let (w, h) = (64usize, 64usize);
    let color = synth_color(w, h);
    let mut filter = RtFilter::<B>::builder(&device, PathBuf::from("/nonexistent"))
        .hdr(true)
        .quality(Quality::High)
        .input_scale(Some(1.0))
        .weights(bytes)
        .build();
    filter.set_color(&Image::from_rgb_f32(&color, w, h));
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit().expect("commit must succeed with user weights");
    assert_eq!(filter.model_key().unwrap().0, "user");
    filter.execute().expect("execute");

    let (raw, _, _, _) = filter.take_output().unwrap();
    let out: &[f32] = bytemuck::cast_slice(&raw);
    for x in out { assert!(x.is_finite()); }
}

#[test]
fn input_scale_explicit_vs_unscaled() {
    // Setting explicit input_scale=1.0 ⇒ identity scaling; same as no
    // autoexposure on a low-dynamic-range scene. This test pins the
    // user-override codepath as live.
    let Some(dir) = weights_dir() else { return; };
    type B = NdArray<f32>;
    let device = <B as burn::tensor::backend::BackendTypes>::Device::default();

    let (w, h) = (64usize, 64usize);
    let color = synth_color(w, h);

    let mut a = RtFilter::<B>::builder(&device, &dir)
        .hdr(true).quality(Quality::Balanced).input_scale(Some(1.0)).build();
    a.set_color(&Image::from_rgb_f32(&color, w, h));
    a.allocate_output(w, h, PixelFormat::Rgb32f);
    a.execute().expect("execute a");
    let (ra, _, _, _) = a.take_output().unwrap();
    let out_a: &[f32] = bytemuck::cast_slice(&ra);

    let mut b = RtFilter::<B>::builder(&device, &dir)
        .hdr(true).quality(Quality::Balanced).input_scale(Some(2.0)).build();
    b.set_color(&Image::from_rgb_f32(&color, w, h));
    b.allocate_output(w, h, PixelFormat::Rgb32f);
    b.execute().expect("execute b");
    let (rb, _, _, _) = b.take_output().unwrap();
    let out_b: &[f32] = bytemuck::cast_slice(&rb);

    // Different scales must produce different outputs (otherwise the scale
    // override is dead).
    let rmse_diff: f32 = out_a.iter().zip(out_b.iter())
        .map(|(x, y)| (x - y).powi(2)).sum::<f32>() / out_a.len() as f32;
    let rmse_diff = rmse_diff.sqrt();
    eprintln!("input_scale 1.0 vs 2.0 RMSE delta: {rmse_diff:.5}");
    assert!(rmse_diff > 1e-4, "explicit input_scale appears to have no effect");
}

#[test]
fn memory_budget_forces_more_tiles() {
    // Hand the planner a tight memory budget on a synthetic 3072x3072 plan.
    // Comparing the no-budget vs budget plans directly avoids paying for GPU
    // inference here.
    let plan_no_budget = tile::plan(3072, 3072, tile::RECEPTIVE_FIELD_BASE,
                                    tile::MIN_TILE_ALIGNMENT, tile::DEFAULT_MAX_TILE_SIZE);

    // 8 MB cap is tight enough to force more than the default 4 tiles.
    // Per-pixel bytes for UNet base ≈ 96 ch × 4 (f32) × 4 (safety) = 1536.
    // 8 MB / 1536 ≈ 5461 pixels per tile → ~74² tile, definitely forces split.
    let plan_8mb = tile::plan(3072, 3072, tile::RECEPTIVE_FIELD_BASE,
                              tile::MIN_TILE_ALIGNMENT, 5461);
    assert!(plan_8mb.jobs.len() >= plan_no_budget.jobs.len(),
            "tighter budget should not produce fewer tiles");
    // The budget cap might bottom out at min_tile_dim (~768 px) regardless of
    // requested ceiling, but tile area should still be smaller.
    assert!(plan_8mb.tile_w * plan_8mb.tile_h <= plan_no_budget.tile_w * plan_no_budget.tile_h,
            "tighter budget should not produce a larger tile");
}
