//! Multi-tile correctness on real wgpu backend.
//!
//! At 3072×3072 the image (9.4M pixels) exceeds DEFAULT_MAX_TILE_SIZE
//! (2160² = 4.66M), so the tile planner must split into multiple jobs.
//! The planner test checks that default geometry. The GPU pass uses an explicit
//! logical budget to exercise the same tile loop with smaller working tiles.
//! Large topology additionally compares actual two-tile and single-tile outputs.

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use oidn_rs::prelude::wgpu_prelude::*;
use oidn_rs::prelude::*;
use oidn_rs::tile;

fn weights_dir() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("weights");
    assert!(
        p.is_dir(),
        "required shipped weights directory is missing: {}",
        p.display()
    );
    p
}

/// Smooth radial gradient — no high-frequency content so any tile-seam
/// artifact would jump out as a discontinuity in row-mean variance.
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
            let v = 0.6 + 0.3 * (1.0 - r);
            let i = (y * w + x) * 3;
            buf[i] = v;
            buf[i + 1] = v * 0.9;
            buf[i + 2] = v * 0.7;
        }
    }
    buf
}

#[test]
fn plan_actually_tiles_at_3072() {
    // Cheap sanity check on the planner itself before paying for GPU.
    let plan = tile::plan(
        3072,
        3072,
        tile::RECEPTIVE_FIELD_BASE,
        tile::MIN_TILE_ALIGNMENT,
        tile::DEFAULT_MAX_TILE_SIZE,
    )
    .unwrap();
    assert!(
        plan.jobs.len() > 1,
        "expected multi-tile plan at 3072x3072, got {} jobs",
        plan.jobs.len()
    );
    eprintln!(
        "3072×3072 → {} tiles of {}×{}",
        plan.jobs.len(),
        plan.tile_w,
        plan.tile_h
    );

    // Tiles must collectively cover every pixel exactly once.
    let total: i64 = plan
        .jobs
        .iter()
        .map(|j| (j.output_dst.w as i64) * (j.output_dst.h as i64))
        .sum();
    assert_eq!(
        total,
        3072i64 * 3072,
        "tiles must cover full image without gaps or overlap"
    );
}

#[test]
#[ignore = "requires an explicit GPU verification lane; run with --ignored"]
fn denoise_3072_multi_tile_wgpu() {
    assert_eq!(
        std::env::var("OIDN_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "explicit GPU verification requires OIDN_REQUIRE_GPU=1"
    );
    let dir = weights_dir();

    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (3072usize, 3072usize);
    let clean = make_clean(w, h);

    let in_img = Image::from_rgb_f32(&clean, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
        .hdr(true)
        .quality(Quality::High)
        .max_memory_mb(512)
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&in_img).unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let progress_calls = Arc::clone(&calls);
    filter.set_progress(move |_| {
        progress_calls.fetch_add(1, Ordering::Relaxed);
        true
    });
    filter.commit().expect("commit");
    filter.execute().expect("execute");
    assert!(
        calls.load(Ordering::Relaxed) > 1,
        "GPU execution must use multiple tiles"
    );

    let (raw, ow, oh, fmt) = filter.take_output().unwrap();
    assert_eq!((ow, oh, fmt), (w, h, PixelFormat::Rgb32f));
    let out: &[f32] = bytemuck::cast_slice(&raw);

    // 1) All finite.
    for x in out {
        assert!(x.is_finite(), "non-finite output value");
    }

    // Reject zero/constant output before assessing row continuity.
    let mean_in = clean.iter().sum::<f32>() / clean.len() as f32;
    let mean_out = out.iter().sum::<f32>() / out.len() as f32;
    assert!(
        (mean_out - mean_in).abs() < mean_in * 0.5,
        "mean signal was lost: input={mean_in}, output={mean_out}"
    );
    let center = out[((h / 2) * w + w / 2) * 3];
    let edge = out[(h / 2) * w * 3];
    assert!(
        center > edge + 0.05,
        "radial contrast was lost: center={center}, edge={edge}"
    );

    // 2) No tile-seam discontinuities: row means should vary smoothly because
    // the input is smooth. We compute mean luminance per row, then look at the
    // largest absolute first-difference. For a smooth gradient this should be
    // small; a tile seam would show a spike.
    let row_means: Vec<f32> = (0..h)
        .map(|y| {
            let mut s = 0.0f32;
            for x in 0..w {
                let i = (y * w + x) * 3;
                s += (out[i] + out[i + 1] + out[i + 2]) / 3.0;
            }
            s / w as f32
        })
        .collect();

    let mut max_jump = 0.0f32;
    let mut max_jump_y = 0;
    for y in 1..h {
        let j = (row_means[y] - row_means[y - 1]).abs();
        if j > max_jump {
            max_jump = j;
            max_jump_y = y;
        }
    }

    let mean_overall: f32 = row_means.iter().sum::<f32>() / h as f32;
    eprintln!(
        "3072×3072 multi-tile OK — mean luminance {:.4}, max row jump {:.5} at y={}",
        mean_overall, max_jump, max_jump_y
    );

    // This bounds row-average continuity for this smooth fixture; it is not
    // a claim about noise amplitude or every possible local seam.
    assert!(
        max_jump < 0.01,
        "row-mean jump {max_jump} at y={max_jump_y} suggests a tile seam"
    );
}

#[test]
#[ignore = "requires an explicit GPU verification lane; run with --ignored"]
fn large_two_tiles_match_full_image_wgpu() {
    assert_eq!(std::env::var("OIDN_REQUIRE_GPU").as_deref(), Ok("1"));
    let device = WgpuDevice::new().expect("wgpu init");
    let (w, h) = (769usize, 16usize);
    let dir = weights_dir();
    let archive = std::fs::read(dir.join("rt_hdr_calb_cnrm_large.tza"))
        .expect("required shipped Large archive");
    let tensors = oidn_tza::parse(&archive).unwrap();
    let descriptor = oidn_model::ModelDescriptor::from_tza(&tensors).unwrap();
    assert_eq!(descriptor.receptive_field(), tile::RECEPTIVE_FIELD_LARGE);
    let plan = tile::plan(
        w as i32,
        h as i32,
        descriptor.receptive_field(),
        descriptor.alignment(),
        1,
    )
    .unwrap();
    assert_eq!(plan.jobs.len(), 2);
    assert_eq!(plan.overlap, 112);

    let mut color = Vec::with_capacity(w * h * 3);
    let mut albedo = Vec::with_capacity(w * h * 3);
    let mut normal = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let bright = w / 4 <= x && x < 3 * w / 4 && h / 4 <= y && y < 3 * h / 4;
            let value = if bright {
                12.0 + 18.0 * ((x * 13 + y * 17) % 5) as f32 / 4.0
            } else {
                0.08 + 0.04 * ((x * 19 + y * 7) % 3) as f32 / 2.0
            };
            color.extend([value, value * 0.7, value * 0.3]);
            albedo.extend([
                0.2 + 0.5 * x as f32 / (w - 1) as f32,
                0.3 + 0.2 * y as f32 / (h - 1) as f32,
                0.4,
            ]);
            let nx = ((x % 5) as f32 - 2.0) * 0.15;
            let ny = ((y % 3) as f32 - 1.0) * 0.2;
            normal.extend([nx, ny, (1.0 - nx * nx - ny * ny).sqrt()]);
        }
    }
    let mut outputs = Vec::new();
    for tiled in [false, true] {
        let mut builder = RtFilter::builder(&device.handle, &dir)
            .hdr(true)
            .clean_aux(true)
            .quality(Quality::High)
            .input_scale(Some(0.02));
        if tiled {
            builder = builder.max_memory_mb(0);
        }
        let mut filter = builder.build();
        filter
            .set_color(&Image::from_rgb_f32(&color, w, h))
            .unwrap();
        filter
            .set_albedo(&Image::from_rgb_f32(&albedo, w, h))
            .unwrap();
        filter
            .set_normal(&Image::from_rgb_f32(&normal, w, h))
            .unwrap();
        filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let progress_calls = Arc::clone(&calls);
        filter.set_progress(move |_| {
            progress_calls.fetch_add(1, Ordering::Relaxed);
            true
        });
        filter.execute().unwrap();
        assert_eq!(filter.model_key().unwrap().name(), "rt_hdr_calb_cnrm_large");
        assert_eq!(
            calls.load(Ordering::Relaxed),
            if tiled { 2 } else { 1 },
            "actual execution must use the requested single/two-tile path"
        );
        let (bytes, ow, oh, format) = filter.take_output().unwrap();
        assert_eq!((ow, oh, format), (w, h, PixelFormat::Rgb32f));
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|v| *v > 1.0), "HDR signal was lost");
        outputs.push(values);
    }
    let max_error = outputs[0]
        .iter()
        .zip(&outputs[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 1e-4,
        "Large two-tile/full max error: {max_error}"
    );
}
