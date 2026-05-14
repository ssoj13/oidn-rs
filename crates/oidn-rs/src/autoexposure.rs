//! Autoexposure — port of `_ref/oidn/core/autoexposure.h` and the GPU kernel
//! in `_ref/oidn/devices/gpu/gpu_autoexposure.h`.
//!
//! For v0.1 we do this on the CPU once per `execute()` call — the cost is
//! O(W*H) and dominated by the GPU inference anyway. A GPU implementation
//! using Burn ops can replace this transparently in Phase 6.

use crate::color::luminance;

/// Bin geometry from `_ref/oidn/devices/gpu/gpu_autoexposure.h:21`.
const MAX_BIN_SIZE: usize = 16;
/// Key value from autoexposure paper — `_ref/oidn/core/autoexposure.h`.
const KEY: f32 = 0.18;
/// Eps used when the image has zero usable pixels.
const EPS: f32 = 1e-8;

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
                    let lum = luminance(px[0], px[1], px[2]);
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
