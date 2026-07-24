//! Round-trip tests for the broadened `PixelFormat` enum.
//!
//! For each format (1ch, 2ch, 3ch in both f32 and f16), build an Image,
//! convert to internal RGB, write back through ImageMut and check we
//! recover the original values modulo broadcast / collapse rules.

use half::f16;
use oidn_rs::{Image, ImageMut, PixelFormat};

fn run_rgb_f32_round_trip() {
    let (w, h) = (4usize, 3usize);
    let mut src = vec![0.0f32; w * h * 3];
    for (i, v) in src.iter_mut().enumerate() {
        *v = (i as f32) * 0.1;
    }

    let img = Image::from_rgb_f32(&src, w, h);
    let rgb_f32 = img.to_rgb_f32();
    assert_eq!(rgb_f32.len(), w * h * 3);
    for (a, b) in rgb_f32.iter().zip(src.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    let mut dst = vec![0.0f32; w * h * 3];
    let mut dst_img = ImageMut::from_rgb_f32(&mut dst, w, h);
    dst_img.write_rgb_f32(&rgb_f32);
    assert_eq!(dst, src);
}

#[test]
fn rgb_f32_round_trip() {
    run_rgb_f32_round_trip();
}

#[test]
fn rgb_f16_round_trip() {
    let (w, h) = (4usize, 3usize);
    let mut src = vec![f16::ZERO; w * h * 3];
    for (i, v) in src.iter_mut().enumerate() {
        *v = f16::from_f32(i as f32 * 0.1);
    }

    let img = Image::from_rgb_f16(&src, w, h);
    let rgb_f32 = img.to_rgb_f32();

    let mut dst = vec![f16::ZERO; w * h * 3];
    let mut dst_img = ImageMut::from_rgb_f16(&mut dst, w, h);
    dst_img.write_rgb_f32(&rgb_f32);

    for (a, b) in dst.iter().zip(src.iter()) {
        assert!(
            (a.to_f32() - b.to_f32()).abs() < 1e-3,
            "f16 round-trip drift"
        );
    }
}

#[test]
fn r_f32_broadcasts_to_rgb_and_collapses_on_write() {
    let (w, h) = (2usize, 2usize);
    let src = vec![0.25f32, 0.5, 0.75, 1.0]; // w*h = 4 luminance values

    let img = Image::from_r_f32(&src, w, h);
    assert_eq!(img.format.channels(), 1);
    let rgb = img.to_rgb_f32();
    // Each pixel must have all 3 channels equal to the luminance.
    for x in 0..w * h {
        assert_eq!(rgb[x * 3], src[x]);
        assert_eq!(rgb[x * 3 + 1], src[x]);
        assert_eq!(rgb[x * 3 + 2], src[x]);
    }

    // Write a known RGB through a 1-ch destination — must keep only the red channel
    // (matches the broadcast rule in reverse).
    let denoised_rgb = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    let mut dst = vec![0.0f32; w * h];
    let mut dst_img = ImageMut::from_r_f32(&mut dst, w, h);
    dst_img.write_rgb_f32(&denoised_rgb);
    assert_eq!(dst, vec![0.1, 0.4, 0.7, 1.0]);
}

#[test]
fn rg_f32_replicates_green_into_blue_and_drops_blue_on_write() {
    let (w, h) = (2usize, 2usize);
    let src = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let img = Image::from_rg_f32(&src, w, h);
    assert_eq!(img.format.channels(), 2);
    let rgb = img.to_rgb_f32();
    // Matches `_ref/oidn/core/image_accessor.h::get3` for `C==2`:
    // `vec3<T>(pixel[0], pixel[1], pixel[1])`.
    for x in 0..w * h {
        assert_eq!(rgb[x * 3], src[x * 2]);
        assert_eq!(rgb[x * 3 + 1], src[x * 2 + 1]);
        assert_eq!(rgb[x * 3 + 2], src[x * 2 + 1]);
    }

    let denoised_rgb = vec![
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.0,
    ];
    let mut dst = vec![0.0f32; w * h * 2];
    let mut dst_img = ImageMut::from_rg_f32(&mut dst, w, h);
    dst_img.write_rgb_f32(&denoised_rgb);
    assert_eq!(dst, vec![0.9, 0.8, 0.6, 0.5, 0.3, 0.2, 0.05, 0.025]);
}

#[test]
fn pixel_format_size_table() {
    use PixelFormat::*;
    assert_eq!(R32f.pixel_size(), 4);
    assert_eq!(R16f.pixel_size(), 2);
    assert_eq!(Rg32f.pixel_size(), 8);
    assert_eq!(Rg16f.pixel_size(), 4);
    assert_eq!(Rgb32f.pixel_size(), 12);
    assert_eq!(Rgb16f.pixel_size(), 6);
    assert!(R16f.is_f16());
    assert!(!R32f.is_f16());
}
