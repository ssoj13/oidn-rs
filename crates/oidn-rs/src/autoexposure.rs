//! Autoexposure — port of `_ref/oidn/core/autoexposure.h` and the GPU kernel
//! in `_ref/oidn/devices/gpu/gpu_autoexposure.h`.
//!
//! Two implementations are kept:
//!
//! - [`compute_scale`]: CPU reference. Used by the legacy
//!   [`Image`](crate::image::Image)-driven `unet_runner` path until
//!   sub-tasks I.2 + I.4 retire the host roundtrip.
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

use burn::prelude::ElementConversion;
use burn::tensor::{Bool, Tensor, backend::Backend, module::avg_pool2d};

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

/// Compute the autoexposure scale factor for an HDR colour image laid out as
/// HWC `f32` triples. Returns `1.0` if the image is empty or all zero.
///
/// Algorithm (mirrors `GPUAutoexposureDownsampleKernel` + `GPUAutoexposureReduceKernel`):
/// 1. Downsample to bins of size up to `MAX_BIN_SIZE` × `MAX_BIN_SIZE` by
///    averaging luminance over each bin.
/// 2. Reject bins whose log-luminance falls outside the `[1e-8, ∞)` band.
/// 3. Mean of the valid bins' log-luminance → geometric mean of luminance.
/// 4. `scale = KEY / max(geometric_mean, eps)`.
pub fn compute_scale(rgb_hwc: &[f32], width: usize, height: usize) -> f32 {
    debug_assert_eq!(rgb_hwc.len(), width * height * 3);
    if width == 0 || height == 0 {
        return 1.0;
    }

    let bins_w = width.div_ceil(MAX_BIN_SIZE);
    let bins_h = height.div_ceil(MAX_BIN_SIZE);
    let mut sum_log = 0.0f64;
    let mut count = 0i64;

    for by in 0..bins_h {
        let y0 = by * MAX_BIN_SIZE;
        let y1 = (y0 + MAX_BIN_SIZE).min(height);
        for bx in 0..bins_w {
            let x0 = bx * MAX_BIN_SIZE;
            let x1 = (x0 + MAX_BIN_SIZE).min(width);

            let mut sum = 0.0f32;
            let mut n = 0u32;
            for y in y0..y1 {
                let row = &rgb_hwc[(y * width + x0) * 3..(y * width + x1) * 3];
                for px in row.chunks_exact(3) {
                    let lum = LUM_R * px[0] + LUM_G * px[1] + LUM_B * px[2];
                    if lum.is_finite() {
                        sum += lum;
                        n += 1;
                    }
                }
            }

            if n > 0 {
                let avg = sum / n as f32;
                if avg > EPS {
                    sum_log += (avg as f64).ln();
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        return 1.0;
    }

    let geom_mean = (sum_log / count as f64).exp() as f32;
    KEY / geom_mean.max(EPS)
}

/// Tensor-native autoexposure — operates entirely on `rgb_chw.device()`.
///
/// `rgb_chw` must be a `[1, 3, H, W]` (NCHW) `f32` tensor. Returns the
/// scalar scale; only `sum_log` and `count` (two scalars) ever cross the
/// device→host boundary. The bin reduction uses Burn's `avg_pool2d` with
/// stride = kernel = [`MAX_BIN_SIZE`], which floors the bin grid to whole
/// 16-pixel cells (drops at most 15 boundary pixels per axis — irrelevant
/// for a statistical estimator).
pub fn compute_scale_tensor<B: Backend>(rgb_chw: Tensor<B, 4>) -> f32 {
    let dims = rgb_chw.dims();
    debug_assert_eq!(dims[0], 1, "compute_scale_tensor expects batch size 1");
    debug_assert_eq!(dims[1], 3, "compute_scale_tensor expects 3 colour channels (CHW)");
    let h = dims[2];
    let w = dims[3];
    if h < MAX_BIN_SIZE || w < MAX_BIN_SIZE {
        // No full bin fits → unity scale (matches CPU path falling through
        // to `count == 0`).
        return 1.0;
    }

    // 1. Luminance via weighted channel reduce: take each plane, scale,
    //    sum. Slice keeps things zero-copy on most backends and avoids a
    //    broadcast tensor allocation for the weight vector.
    let r = rgb_chw.clone().narrow(1, 0, 1);
    let g = rgb_chw.clone().narrow(1, 1, 1);
    let b = rgb_chw.narrow(1, 2, 1);
    let lum: Tensor<B, 4> =
        r.mul_scalar(LUM_R) + g.mul_scalar(LUM_G) + b.mul_scalar(LUM_B);

    // 2. avg_pool2d over MAX_BIN_SIZE × MAX_BIN_SIZE → bin grid.
    //    count_include_pad=false; padding=0 (we already floored above so
    //    every pooled cell is full).
    let binned: Tensor<B, 4> = avg_pool2d(
        lum,
        [MAX_BIN_SIZE, MAX_BIN_SIZE],
        [MAX_BIN_SIZE, MAX_BIN_SIZE],
        [0, 0],
        false, // count_include_pad: no padding ⇒ irrelevant, keep deterministic
        false, // ceil_mode: we already floored dims to whole bins above
    );

    // 3. Reject bins with luminance ≤ EPS, take log of survivors.
    let valid_mask: Tensor<B, 4, Bool> = binned.clone().greater_elem(EPS);
    let valid_f = valid_mask.clone().float();
    // clamp_min(EPS) keeps log() finite for the rejected bins; mask
    // zeroes their contribution to the sum, so any finite value is fine.
    let log_binned = binned.clamp_min(EPS).log();
    let masked_log = log_binned.mul(valid_f.clone());

    // 4. Pull only the two reduced scalars to host.
    let sum_log = masked_log.sum().into_scalar().elem::<f32>();
    let count = valid_f.sum().into_scalar().elem::<f32>();
    if count < 0.5 {
        return 1.0;
    }
    let geom_mean = (sum_log / count).exp();
    KEY / geom_mean.max(EPS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    /// CPU and tensor implementations must agree within 1% on a synthetic
    /// HDR-ish gradient. Plan-stated tolerance (I.3 acceptance).
    #[test]
    fn cpu_vs_tensor_parity_gradient() {
        let device = NdArrayDevice::default();
        // 64×48 gradient: luminance ranges roughly [0.01, 10.0]. Plenty of
        // dynamic range; 12 full 16×16 bins fit.
        let w = 64;
        let h = 48;
        let mut hwc = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                // Engineer a value that produces visibly different
                // luminances across the image to avoid pathological cases.
                let v = 0.01_f32 + (x as f32 / w as f32) * 9.99;
                let idx = (y * w + x) * 3;
                hwc[idx]     = v;
                hwc[idx + 1] = v * 0.9;
                hwc[idx + 2] = v * 0.5;
            }
        }
        let cpu_scale = compute_scale(&hwc, w, h);

        // HWC → CHW for the tensor path.
        let chw = crate::image_tensor::hwc_to_chw(&hwc, 3, h, w);
        let t = Tensor::<B, 4>::from_data(TensorData::new(chw, [1, 3, h, w]), &device);
        let gpu_scale = compute_scale_tensor(t);

        let rel = (cpu_scale - gpu_scale).abs() / cpu_scale.abs().max(1e-6);
        assert!(
            rel < 0.01,
            "tensor autoexposure deviated >1%: cpu={cpu_scale} gpu={gpu_scale} rel={rel}",
        );
    }

    /// All-dark image: both paths should fall through to scale = 1.0
    /// (CPU returns 1.0 when `count == 0`; tensor returns 1.0 when
    /// `count < 0.5`).
    #[test]
    fn cpu_vs_tensor_parity_dark() {
        let device = NdArrayDevice::default();
        let (w, h) = (32, 32);
        let hwc = vec![0.0f32; w * h * 3];
        assert_eq!(compute_scale(&hwc, w, h), 1.0);

        let chw = crate::image_tensor::hwc_to_chw(&hwc, 3, h, w);
        let t = Tensor::<B, 4>::from_data(TensorData::new(chw, [1, 3, h, w]), &device);
        assert_eq!(compute_scale_tensor(t), 1.0);
    }
}
