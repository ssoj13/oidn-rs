//! Multi-tile correctness on real wgpu backend.
//!
//! At 3072×3072 the image (9.4M pixels) exceeds DEFAULT_MAX_TILE_SIZE
//! (2160² = 4.66M), so the tile planner must split into multiple jobs.
//! This test verifies the tile loop in `unet_runner::run` stitches the
//! output back together without seams.

use std::path::PathBuf;

use oidn_rs::prelude::wgpu_prelude::*;
use oidn_rs::prelude::*;
use oidn_rs::tile;

fn weights_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("weights");
    if p.is_dir() { Some(p) } else { None }
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
    );
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
fn denoise_3072_multi_tile_wgpu() {
    let Some(dir) = weights_dir() else {
        eprintln!("skipping: weights submodule not initialised");
        return;
    };

    let device = WgpuDevice::new().expect("wgpu init");

    let (w, h) = (3072usize, 3072usize);
    let clean = make_clean(w, h);

    let in_img = Image::from_rgb_f32(&clean, w, h);
    let mut filter = RtFilter::builder(&device.handle, &dir)
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

    // 1) All finite.
    for x in out {
        assert!(x.is_finite(), "non-finite output value");
    }

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

    // Threshold: 0.01 is generous — at the seam between two adjacent tiles
    // on a smooth image, the network's output noise alone would be < 0.001.
    // A real seam (no overlap) would show jumps of 0.05+.
    assert!(
        max_jump < 0.01,
        "row-mean jump {max_jump} at y={max_jump_y} suggests a tile seam"
    );
}
