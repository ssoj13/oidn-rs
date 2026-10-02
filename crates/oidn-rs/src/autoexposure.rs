//! Autoexposure — port of `_ref/oidn/core/autoexposure.h` and the GPU kernel
//! in `_ref/oidn/devices/gpu/gpu_autoexposure.h`.
//!
//! Two implementations are kept:
//!
//! - [`compute_scale`]: host-side mathematical oracle.
//! - [`compute_scale_tensor`]: Burn-tensor variant. Runs the bin reduction
//!   on the device of `rgb_chw`; only the final two scalars (`sum_log`
//!   and `count`) cross to host. Use this from any tensor-native
//!   pipeline.
//!
//! Both share the same constants and algorithm:
//! 1. Downsample to bins of size up to [`MAX_BIN_SIZE`] × [`MAX_BIN_SIZE`]
//!    via per-bin luminance mean.
//! 2. Reject bins whose mean falls below `EPS`.
//! 3. Geometric mean over the surviving bins.
//! 4. `scale = KEY / max(geom_mean, EPS)`.
//!
//! ## Luminance space (`acescg-autoexposure` feature)
//!
//! The estimator collapses each pixel to a single luminance value, so the
//! channel weights it uses have to match the colour space the pixels live in.
//!
//! - **Default — Rec.709.** Weights `(0.212671, 0.715160, 0.072169)`, identical
//!   to [`crate::color::luminance`] and the upstream OIDN reference. Correct for
//!   sRGB / Rec.709 input and the right default for a general-purpose denoiser.
//! - **`acescg-autoexposure` — ACEScg (AP1).** Weights `(0.2722287, 0.6740818,
//!   0.0536895)`, the Y row of the AP1→XYZ matrix. Use when the denoiser input is
//!   ACEScg: measuring an AP1 image with Rec.709 weights skews the exposure
//!   estimate, because the same RGB triple carries different luminance in the two
//!   spaces.
//!
//! It is a compile-time feature rather than a runtime parameter on purpose: a
//! pipeline's working space is fixed, the weights are `const` (so the per-pixel
//! multiply folds away), and both the CPU ([`compute_scale`]) and tensor
//! ([`compute_scale_tensor`]) paths read the same constants with no parameter to
//! thread through. [`crate::color::luminance`] stays on Rec.709 regardless, so
//! other consumers of it are unaffected.
//!
//! Used by `vfx-rs`'s `pt-denoise-oidn` (a path tracer working internally in
//! ACEScg), which turns the feature on through its git dependency on this crate.

use crate::error::OidnError;
use burn::tensor::{Bool, Int, Tensor, TensorData, module::avg_pool2d};

/// Bin geometry from `_ref/oidn/devices/gpu/gpu_autoexposure.h:21`.
pub const MAX_BIN_SIZE: usize = 16;
/// Key value from autoexposure paper — `_ref/oidn/core/autoexposure.h`.
pub const KEY: f32 = 0.18;
/// Eps used when the image has zero usable pixels.
pub const EPS: f32 = 1e-8;

// Luminance weights for the autoexposure estimator. Both the CPU and tensor
// paths below share these constants so they stay within parity tolerance.
//
// Default: Rec.709 — identical to [`crate::color::luminance`] and the OIDN
// reference. The CPU path computes `LUM_R*r + LUM_G*g + LUM_B*b`, byte-for-byte
// what `luminance()` returns.
#[cfg(not(feature = "acescg-autoexposure"))]
const LUM_R: f32 = 0.212671;
#[cfg(not(feature = "acescg-autoexposure"))]
const LUM_G: f32 = 0.715160;
#[cfg(not(feature = "acescg-autoexposure"))]
const LUM_B: f32 = 0.072169;

// With `acescg-autoexposure`: ACEScg (AP1) luminance weights — the Y row of the
// AP1->XYZ matrix. Use when the denoiser input is ACEScg, so autoexposure
// measures luminance in the same space the image lives in.
#[cfg(feature = "acescg-autoexposure")]
const LUM_R: f32 = 0.2722287;
#[cfg(feature = "acescg-autoexposure")]
const LUM_G: f32 = 0.6740818;
#[cfg(feature = "acescg-autoexposure")]
const LUM_B: f32 = 0.0536895;

/// Balanced bin bounds shared by the host oracle and device estimator.
/// Matches LOCAL2.5 GPUAutoexposureDownsampleKernel integer boundaries.
fn bin_range(index: usize, length: usize, bins: usize) -> std::ops::Range<usize> {
    index * length / bins..(index + 1) * length / bins
}

/// Compute exposure for interleaved RGB values using all balanced bins.
/// NaNs become zero; each component is clamped before luminance, as in OIDN.
/// Invalid dimensions or sample counts return an error.
pub fn compute_scale(rgb_hwc: &[f32], width: usize, height: usize) -> Result<f32, OidnError> {
    let count = crate::image::validate_dimensions(width, height, 3)?;
    if rgb_hwc.len() != count {
        return Err(OidnError::InvalidArgument(
            "autoexposure sample count mismatch",
        ));
    }
    if width == 0 || height == 0 {
        return Ok(1.0);
    }
    let bins_w = width.div_ceil(MAX_BIN_SIZE);
    let bins_h = height.div_ceil(MAX_BIN_SIZE);
    let mut sum_log = 0.0f64;
    let mut count = 0usize;
    for by in 0..bins_h {
        let ys = bin_range(by, height, bins_h);
        for bx in 0..bins_w {
            let xs = bin_range(bx, width, bins_w);
            let mut sum = 0.0f32;
            for y in ys.clone() {
                for x in xs.clone() {
                    let i = (y * width + x) * 3;
                    let value = |v: f32| {
                        if v.is_nan() {
                            0.0
                        } else {
                            v.clamp(0.0, f32::MAX)
                        }
                    };
                    sum += LUM_R * value(rgb_hwc[i])
                        + LUM_G * value(rgb_hwc[i + 1])
                        + LUM_B * value(rgb_hwc[i + 2]);
                }
            }
            let avg = sum / (ys.len() * xs.len()) as f32;
            if avg > EPS {
                sum_log += f64::from(avg).ln();
                count += 1;
            }
        }
    }
    Ok(if count == 0 {
        1.0
    } else {
        KEY / (sum_log / count as f64).exp() as f32
    })
}

/// Device-native balanced-bin exposure; only the final sum/count cross to host.
/// Indexed rows/columns pack exact floor boundaries into zero-padded 16x16
/// pooling cells, so tiny images and both partial edges contribute. This avoids
/// a kernel per bin and does not download image pixels.
pub fn compute_scale_tensor(rgb_chw: Tensor<4>) -> Result<f32, OidnError> {
    let [n, c, h, w] = rgb_chw.dims();
    crate::image::validate_dimensions(w, h, c)?;
    if n != 1 || c != 3 {
        return Err(OidnError::InvalidArgument("autoexposure expects [1,3,H,W]"));
    }
    if h == 0 || w == 0 {
        return Ok(1.0);
    }
    let device = rgb_chw.device();
    let rgb = rgb_chw
        .clone()
        .mask_fill(rgb_chw.is_nan(), 0.0)
        .clamp(0.0, f32::MAX);
    let lum = rgb.clone().narrow(1, 0, 1).mul_scalar(LUM_R)
        + rgb.clone().narrow(1, 1, 1).mul_scalar(LUM_G)
        + rgb.narrow(1, 2, 1).mul_scalar(LUM_B);
    let bins_h = h.div_ceil(MAX_BIN_SIZE);
    let bins_w = w.div_ceil(MAX_BIN_SIZE);
    let axis = |length: usize, bins: usize| {
        let mut indices = Vec::with_capacity(bins * MAX_BIN_SIZE);
        let mut padding = Vec::with_capacity(bins * MAX_BIN_SIZE);
        let mut lengths = Vec::with_capacity(bins);
        for b in 0..bins {
            let range = bin_range(b, length, bins);
            lengths.push(range.len());
            for i in 0..MAX_BIN_SIZE {
                indices.push((range.start + i.min(range.len() - 1)) as i64);
                padding.push(i >= range.len());
            }
        }
        (indices, padding, lengths)
    };
    let (rows, row_padding, row_lengths) = axis(h, bins_h);
    let (cols, col_padding, col_lengths) = axis(w, bins_w);
    let ph = rows.len();
    let pw = cols.len();
    let rows = Tensor::<1, Int>::from_data(TensorData::new(rows, [ph]), &device);
    let cols = Tensor::<1, Int>::from_data(TensorData::new(cols, [pw]), &device);
    let row_padding =
        Tensor::<4, Bool>::from_data(TensorData::new(row_padding, [1, 1, ph, 1]), &device);
    let col_padding =
        Tensor::<4, Bool>::from_data(TensorData::new(col_padding, [1, 1, 1, pw]), &device);
    let packed = lum
        .select(2, rows)
        .select(3, cols)
        .mask_fill(row_padding, 0.0)
        .mask_fill(col_padding, 0.0);
    let means = avg_pool2d(
        packed,
        [MAX_BIN_SIZE; 2],
        [MAX_BIN_SIZE; 2],
        [0; 2],
        false,
        false,
    );
    let mut corrections = Vec::with_capacity(bins_h * bins_w);
    for rh in row_lengths {
        for &cw in &col_lengths {
            corrections.push((MAX_BIN_SIZE * MAX_BIN_SIZE) as f32 / (rh * cw) as f32);
        }
    }
    let means = means
        * Tensor::<4>::from_data(
            TensorData::new(corrections, [1, 1, bins_h, bins_w]),
            &device,
        );
    let valid = means.clone().greater_elem(EPS);
    let sum_log = means
        .clamp_min(EPS)
        .log()
        .mask_fill(valid.clone().bool_not(), 0.0)
        .sum()
        .into_scalar::<f32>();
    let count = valid.float().sum().into_scalar::<f32>();
    Ok(if count < 0.5 {
        1.0
    } else {
        KEY / (sum_log / count).exp()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn balanced_bins_cover_every_pixel() {
        for length in 1usize..130 {
            let bins = length.div_ceil(MAX_BIN_SIZE);
            let covered: Vec<_> = (0..bins).flat_map(|b| bin_range(b, length, bins)).collect();
            assert_eq!(covered, (0..length).collect::<Vec<_>>());
            assert!((0..bins).all(|b| bin_range(b, length, bins).len() <= MAX_BIN_SIZE));
        }
    }

    #[test]
    fn exact_edges_tiny_and_signed_component_parity() {
        let device = Device::ndarray();
        for (w, h) in [(1, 1), (8, 8), (33, 19), (32, 19), (64, 48)] {
            let mut hwc = vec![0.0; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    hwc[i] = if x == w - 1 { 80.0 } else { -10.0 };
                    hwc[i + 1] = 0.1 + y as f32 * 0.01;
                    hwc[i + 2] = if y == h - 1 { 3.0 } else { f32::NAN };
                }
            }
            let expected = compute_scale(&hwc, w, h).unwrap();
            let chw = crate::image_tensor::hwc_to_chw(&hwc, 3, h, w).unwrap();
            let tensor = Tensor::from_data(TensorData::new(chw, [1, 3, h, w]), &device);
            let actual = compute_scale_tensor(tensor).unwrap();
            assert!(
                (expected - actual).abs() / expected < 2e-5,
                "{w}x{h}: {expected} != {actual}"
            );
        }
    }

    #[test]
    fn constant_and_dark_oracles() {
        let device = Device::ndarray();
        for value in [0.0, 2.0] {
            let hwc = vec![value; 8 * 8 * 3];
            let expected = if value == 0.0 { 1.0 } else { KEY / value };
            assert!((compute_scale(&hwc, 8, 8).unwrap() - expected).abs() < 1e-6);
            let tensor = Tensor::full([1, 3, 8, 8], value, &device);
            assert!((compute_scale_tensor(tensor).unwrap() - expected).abs() < 1e-6);
        }
        assert!(compute_scale(&[], usize::MAX, 1).is_err());
        assert!(compute_scale(&[1.0], 1, 1).is_err());
    }
}
