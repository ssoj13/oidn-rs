//! Unit tests for color and tile modules — both pure-CPU, no dependencies.

use oidn_rs::color::{
    TransferFunction, TransferState, pu_forward, pu_inverse, srgb_forward, srgb_inverse,
};
use oidn_rs::tile;

#[test]
fn srgb_round_trip() {
    for i in 0..1000 {
        let y = i as f32 / 1000.0;
        let r = srgb_inverse(srgb_forward(y));
        assert!((r - y).abs() < 1e-5, "sRGB round-trip failed: y={y} -> {r}");
    }
}

#[test]
fn pu_round_trip() {
    // Sample across the practical HDR range used by OIDN.
    for i in 0..1000 {
        let y = (i as f32 / 1000.0) * 10.0; // 0..10
        let r = pu_inverse(pu_forward(y));
        assert!(
            (r - y).abs() / (y + 1e-3) < 1e-3,
            "PU round-trip failed: y={y} -> {r}"
        );
    }
}

#[test]
fn transfer_state_pu_with_scale() {
    let mut tf = TransferState::new(TransferFunction::PU);
    tf.set_input_scale(0.5);
    let v = 1.0;
    let f = tf.forward(v);
    let i = tf.inverse(f);
    // Inverse should restore the original (modulo float precision).
    assert!(
        (i - v).abs() < 1e-2,
        "PU forward/inverse with scale failed: {v} -> {f} -> {i}"
    );
}

#[test]
fn tile_plan_covers_small_image() {
    let p = tile::plan(
        256,
        256,
        tile::RECEPTIVE_FIELD_BASE,
        tile::MIN_TILE_ALIGNMENT,
        tile::DEFAULT_MAX_TILE_SIZE,
    );
    assert_eq!(p.jobs.len(), 1, "256x256 should fit in one tile");
    let job = p.jobs[0];
    assert_eq!(job.output_dst.x, 0);
    assert_eq!(job.output_dst.y, 0);
    assert_eq!(job.output_dst.w, 256);
    assert_eq!(job.output_dst.h, 256);
}

#[test]
fn tile_plan_covers_4k() {
    // 3840×2160 is a single tile under the default 2160² budget (after rounding).
    let p = tile::plan(
        3840,
        2160,
        tile::RECEPTIVE_FIELD_BASE,
        tile::MIN_TILE_ALIGNMENT,
        tile::DEFAULT_MAX_TILE_SIZE,
    );
    let total = oidn_rs::tile::total_output_pixels(&p);
    assert_eq!(
        total,
        3840i64 * 2160,
        "tile plan must cover the entire image exactly"
    );
}

#[test]
fn tile_plan_covers_1024() {
    let p = tile::plan(
        1024,
        1024,
        tile::RECEPTIVE_FIELD_BASE,
        tile::MIN_TILE_ALIGNMENT,
        tile::DEFAULT_MAX_TILE_SIZE,
    );
    let total = oidn_rs::tile::total_output_pixels(&p);
    assert_eq!(total, 1024i64 * 1024);
}
