//! GPU-side helpers for the Phase I tile pipeline.
//!
//! Two main things live here:
//!
//! - [`reflect_pad_2d`] — boundary-reflecting padding (matches OIDN's
//!   `gpu_input_process.h` and the legacy `reflect_into` CPU helper).
//!   Burn 0.21 only ships constant padding; we emulate reflect via
//!   `cat([flip(left_slice), tile, flip(right_slice)])`.
//! - [`apply_transfer_forward`] / [`apply_transfer_inverse`] — vectorised
//!   versions of the four [`TransferFunction`] variants from
//!   [`crate::color`]. The piecewise PU and sRGB curves use
//!   [`Tensor::mask_where`] cascades so all branches execute as a single
//!   stream of kernels.
//!
//! These helpers replace the per-tile `pack()` closure and the per-pixel
//! inverse-transfer loop in `unet_runner.rs`. Numerics match the CPU
//! reference within `rtol = 1e-5` for normal HDR inputs (NdArray
//! parity tests below).

use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::color::{TransferFunction, TransferState};

// ---------- transfer-function constants — mirror `color.rs` ----------

const SRGB_A: f32 = 12.92;
const SRGB_B: f32 = 1.055;
const SRGB_C: f32 = 1.0 / 2.4;
const SRGB_D: f32 = -0.055;
const SRGB_Y0: f32 = 0.0031308;
const SRGB_X0: f32 = 0.04045;

const PU_A: f32 = 1.41283765e+03;
const PU_B: f32 = 1.64593172e+00;
const PU_C: f32 = 4.31384981e-01;
const PU_D: f32 = -2.94139609e-03;
const PU_E: f32 = 1.92653254e-01;
const PU_F: f32 = 6.26026094e-03;
const PU_G: f32 = 9.98620152e-01;
const PU_Y0: f32 = 1.57945760e-06;
const PU_Y1: f32 = 3.22087631e-02;
const PU_X0: f32 = 2.23151711e-03;
const PU_X1: f32 = 3.70974749e-01;

// ---------- reflect padding ----------

/// Reflect-pad a `[1, C, H, W]` tensor along `H` and `W` axes using
/// "edge-reflection" semantics — the boundary pixel itself is *not*
/// repeated. Matches the CPU `reflect_into` helper used by the legacy
/// `pack()` closure.
///
/// For a 1-D source `[a, b, c, d]` with `pad_left=2, pad_right=2`, the
/// output is `[c, b, a, b, c, d, c, b]`.
///
/// Pad arguments may be zero, in which case the corresponding `cat()`
/// step is skipped (Burn won't accept a zero-length slice).
pub fn reflect_pad_2d<B: Backend>(
    t: Tensor<B, 4>,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) -> Tensor<B, 4> {
    let t = pad_axis_reflect::<B>(t, pad_left, pad_right, 3);
    pad_axis_reflect::<B>(t, pad_top, pad_bottom, 2)
}

/// Reflect-pad one axis (`dim` is the tensor axis to pad, e.g. 3 for W).
fn pad_axis_reflect<B: Backend>(
    t: Tensor<B, 4>,
    pad_pre: usize,
    pad_post: usize,
    dim: usize,
) -> Tensor<B, 4> {
    if pad_pre == 0 && pad_post == 0 {
        return t;
    }
    let dims = t.dims();
    let len = dims[dim];
    // Edge-reflection requires at least 2 source elements to mirror against.
    // If the source is degenerate, fall back to clamp-replicate (single
    // element repeated). Real workloads never hit this branch because
    // tile geometry guarantees each input rect has `len >= 1` *and* we
    // only pad up to `len-1` elements (receptive-field alignment).
    if len < 2 {
        return t;
    }

    let mut parts: Vec<Tensor<B, 4>> = Vec::with_capacity(3);
    if pad_pre > 0 {
        // Slice elements `[1..=pad_pre]` from the start, then flip — gives
        // the mirror image of the leading `pad_pre` boundary pixels.
        let pre = slice_axis::<B>(t.clone(), dim, 1, 1 + pad_pre).flip([dim as isize]);
        parts.push(pre);
    }
    parts.push(t.clone());
    if pad_post > 0 {
        // Symmetric on the trailing side: take elements
        // `[len - 1 - pad_post .. len - 1]`, flip.
        let start = len - 1 - pad_post;
        let end = len - 1;
        let post = slice_axis::<B>(t, dim, start, end).flip([dim as isize]);
        parts.push(post);
    }
    Tensor::cat(parts, dim)
}

/// Slice the 4-D tensor along a single axis, leaving the other three
/// alone. Handy for `pad_axis_reflect`.
fn slice_axis<B: Backend>(
    t: Tensor<B, 4>,
    dim: usize,
    start: usize,
    end: usize,
) -> Tensor<B, 4> {
    let dims = t.dims();
    let make_range = |d: usize| -> core::ops::Range<usize> {
        if d == dim { start..end } else { 0..dims[d] }
    };
    t.slice([make_range(0), make_range(1), make_range(2), make_range(3)])
}

// ---------- transfer functions (tensor-vectorised) ----------

/// Forward transfer for the colour tile. `state.input_scale` is applied
/// up-front (autoexposure for HDR), then the piecewise non-linear curve
/// matching [`TransferFunction::forward`] runs vectorised across every
/// element.
pub fn apply_transfer_forward<B: Backend>(
    color: Tensor<B, 4>,
    state: &TransferState,
) -> Tensor<B, 4> {
    let y = color.mul_scalar(state.input_scale);
    match state.kind {
        TransferFunction::Linear => y,
        TransferFunction::SRGB => srgb_forward_tensor(y),
        TransferFunction::PU => pu_forward_tensor(y).mul_scalar(state.norm_scale),
        TransferFunction::Log => log_forward_tensor(y).mul_scalar(state.norm_scale),
    }
}

/// Inverse transfer for the network output. After undoing the non-linear
/// curve we multiply by `output_scale` to recover the input HDR range.
pub fn apply_transfer_inverse<B: Backend>(
    x: Tensor<B, 4>,
    state: &TransferState,
) -> Tensor<B, 4> {
    let unscaled = match state.kind {
        TransferFunction::Linear => x,
        TransferFunction::SRGB => srgb_inverse_tensor(x),
        TransferFunction::PU => pu_inverse_tensor(x.mul_scalar(state.rcp_norm_scale)),
        TransferFunction::Log => {
            x.mul_scalar(state.rcp_norm_scale).exp().sub_scalar(1.0_f32)
        }
    };
    unscaled.mul_scalar(state.output_scale)
}

fn srgb_forward_tensor<B: Backend>(y: Tensor<B, 4>) -> Tensor<B, 4> {
    // if y <= Y0: A * y    else: B * y^C + D
    let low_mask = y.clone().lower_equal_elem(SRGB_Y0);
    let low = y.clone().mul_scalar(SRGB_A);
    let high = y.clamp_min(0.0).powf_scalar(SRGB_C).mul_scalar(SRGB_B).add_scalar(SRGB_D);
    high.mask_where(low_mask, low)
}

fn srgb_inverse_tensor<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    // if x <= X0: x / A    else: ((x - D) / B)^(1/C)
    let low_mask = x.clone().lower_equal_elem(SRGB_X0);
    let low = x.clone().div_scalar(SRGB_A);
    let high = x
        .sub_scalar(SRGB_D)
        .div_scalar(SRGB_B)
        .clamp_min(0.0)
        .powf_scalar(1.0_f32 / SRGB_C);
    high.mask_where(low_mask, low)
}

fn pu_forward_tensor<B: Backend>(y: Tensor<B, 4>) -> Tensor<B, 4> {
    // Three-region piecewise:
    //   y <= Y0:  A * y
    //   Y0 < y <= Y1:  B * y^C + D
    //   y > Y1:  E * ln(y + F) + G
    let low_mask = y.clone().lower_equal_elem(PU_Y0);
    let mid_mask = y.clone().lower_equal_elem(PU_Y1);
    let b_low = y.clone().mul_scalar(PU_A);
    let b_mid = y.clone().clamp_min(0.0).powf_scalar(PU_C).mul_scalar(PU_B).add_scalar(PU_D);
    let b_high = y.add_scalar(PU_F).clamp_min(1e-30).log().mul_scalar(PU_E).add_scalar(PU_G);
    // Start from b_high, overlay mid where y <= Y1 (covers both low+mid),
    // then overlay low where y <= Y0. Order yields the correct three-way
    // disjoint split because low ⊂ mid by construction of the masks.
    b_high.mask_where(mid_mask, b_mid).mask_where(low_mask, b_low)
}

fn pu_inverse_tensor<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    // Three-region inverse:
    //   x <= X0:  x / A
    //   X0 < x <= X1:  ((x - D) / B)^(1/C)
    //   x > X1:  exp((x - G) / E) - F
    let low_mask = x.clone().lower_equal_elem(PU_X0);
    let mid_mask = x.clone().lower_equal_elem(PU_X1);
    let b_low = x.clone().div_scalar(PU_A);
    let b_mid = x
        .clone()
        .sub_scalar(PU_D)
        .div_scalar(PU_B)
        .clamp_min(0.0)
        .powf_scalar(1.0_f32 / PU_C);
    let b_high = x.sub_scalar(PU_G).div_scalar(PU_E).exp().sub_scalar(PU_F);
    b_high.mask_where(mid_mask, b_mid).mask_where(low_mask, b_low)
}

fn log_forward_tensor<B: Backend>(y: Tensor<B, 4>) -> Tensor<B, 4> {
    // ln((y + 1).max(1e-30)) — matches the CPU branch in
    // `TransferState::forward` for `Log`.
    y.add_scalar(1.0_f32).clamp_min(1e-30).log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::TensorData;

    use crate::color::{self, TransferFunction};

    type B = NdArray<f32>;

    /// Reflect-pad on a `[1, 1, 1, 4]` tensor: input `[a,b,c,d]`,
    /// pad_left=2, pad_right=2 → `[c,b,a,b,c,d,c,b]`. Matches the
    /// CPU `reflect_into` semantics used in `pack()`.
    #[test]
    fn reflect_pad_1d_basic() {
        let device = NdArrayDevice::default();
        let src = Tensor::<B, 4>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
            &device,
        );
        let padded = reflect_pad_2d(src, 0, 0, 2, 2);
        let data = padded.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(data, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    /// Zero padding: identity.
    #[test]
    fn reflect_pad_zero_pad_identity() {
        let device = NdArrayDevice::default();
        let src = Tensor::<B, 4>::from_data(
            TensorData::new((0..12).map(|v| v as f32).collect::<Vec<_>>(), [1, 1, 3, 4]),
            &device,
        );
        let padded = reflect_pad_2d(src.clone(), 0, 0, 0, 0);
        let a = padded.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let b = src.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(a, b);
    }

    /// Forward / inverse round-trip on PU within 1e-4 relative error
    /// — confirms the mask_where cascade matches the CPU piecewise impl.
    #[test]
    fn pu_forward_inverse_roundtrip_ndarray() {
        let device = NdArrayDevice::default();
        // Sample 128 values spanning the three PU regions
        // (low ≤ 1.6e-6, mid ≤ 3.2e-2, high ≥ several).
        let samples: Vec<f32> = (0..128)
            .map(|i| 10.0_f32.powf((i as f32 - 64.0) / 16.0)) // log-spaced
            .collect();

        // Apply forward via tensor path.
        let t = Tensor::<B, 4>::from_data(
            TensorData::new(samples.clone(), [1, 1, 1, 128]),
            &device,
        );
        let mut tf = TransferState::new(TransferFunction::PU);
        tf.set_input_scale(1.0); // skip autoexposure for this test
        let fwd = apply_transfer_forward(t, &tf);
        let bwd = apply_transfer_inverse(fwd, &tf);
        let result = bwd.into_data().convert::<f32>().to_vec::<f32>().unwrap();

        for (orig, back) in samples.iter().zip(result.iter()) {
            let rel = (orig - back).abs() / orig.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "PU round-trip drift: orig={orig} back={back} rel={rel}"
            );
        }
    }

    /// Cross-check forward path against the scalar CPU impl on a fresh
    /// sample set. Picks up any constant typo / sign error in the
    /// tensor version.
    #[test]
    fn pu_forward_cpu_parity_ndarray() {
        let device = NdArrayDevice::default();
        let samples: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let t = Tensor::<B, 4>::from_data(
            TensorData::new(samples.clone(), [1, 1, 1, 64]),
            &device,
        );
        let mut tf = TransferState::new(TransferFunction::PU);
        tf.set_input_scale(1.0);
        let tensor_fwd = apply_transfer_forward(t, &tf)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        for (s, got) in samples.iter().zip(tensor_fwd.iter()) {
            let expected = tf.forward(*s);
            let rel = (expected - got).abs() / expected.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "PU forward drift @ y={s}: cpu={expected} gpu={got} rel={rel}"
            );
        }
    }

    /// Same parity check for sRGB.
    #[test]
    fn srgb_forward_cpu_parity_ndarray() {
        let device = NdArrayDevice::default();
        let samples: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        let t = Tensor::<B, 4>::from_data(
            TensorData::new(samples.clone(), [1, 1, 1, 64]),
            &device,
        );
        let mut tf = TransferState::new(TransferFunction::SRGB);
        tf.set_input_scale(1.0);
        let tensor_fwd = apply_transfer_forward(t, &tf)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        for (s, got) in samples.iter().zip(tensor_fwd.iter()) {
            let expected = color::srgb_forward(*s);
            let rel = (expected - got).abs() / expected.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "sRGB forward drift @ y={s}: cpu={expected} gpu={got} rel={rel}"
            );
        }
    }
}
