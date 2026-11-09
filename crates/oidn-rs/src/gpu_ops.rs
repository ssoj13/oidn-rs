//! GPU-side helpers for the Phase I tile pipeline.
//!
//! - `preprocess_input` / `postprocess_color` wrap the transfer-function
//!   curves with the sanitise + clamp + scale sequence that mirrors
//!   `_ref/oidn/devices/cpu/cpu_input_process.isph` and
//!   `cpu_output_process.isph`. The piecewise PU and sRGB curves use
//!   [`Tensor::mask_where`] cascades so all branches execute as a single
//!   stream of kernels.
//! - [`apply_transfer_forward`] / [`apply_transfer_inverse`] expose the
//!   non-linear curves alone (no scale, no clamp); the new wrapper helpers
//!   compose them with the surrounding ops in the exact reference order.
//!
//! Tile borders are zero-padded (not reflected) via `slice_assign` in
//! `unet_runner`, matching the reference's `cpu_input_process` zero-pad.

use burn::tensor::Tensor;

use crate::color::{
    PU_A, PU_B, PU_C, PU_D, PU_E, PU_F, PU_G, PU_X0, PU_X1, PU_Y0, PU_Y1, SRGB_A, SRGB_B, SRGB_C,
    SRGB_D, SRGB_X0, SRGB_Y0, TransferFunction, TransferState, Y_MAX,
};

// ---------- preprocess / postprocess wrappers ----------

/// Replace NaNs with zero; infinities are clamped by the caller. Matches the
/// `nan_to_zero` helper called at the top of every reference input/output
/// kernel (`_ref/oidn/devices/cpu/cpu_input_process.isph:31`,
/// `cpu_output_process.isph:38`).
fn nan_to_zero(t: Tensor<4>) -> Tensor<4> {
    t.clone().mask_fill(t.is_nan(), 0.0)
}

/// Preprocess the colour tile before the network.
///
/// Mirrors `_ref/oidn/devices/cpu/cpu_input_process.isph:31-51`:
/// `*input_scale -> nan_to_zero -> clamp(lo, hi) -> [snorm remap] ->
/// forward transfer`.
///
/// Optional strict nonfinite sanitation happens upstream once per frame.
/// NaN replacement after scaling remains mandatory, including when strict
/// sanitation is disabled.
pub(crate) fn preprocess_input(
    color: Tensor<4>,
    input_scale: f32,
    hdr: bool,
    snorm: bool,
    transfer: &TransferState,
) -> Tensor<4> {
    let t = nan_to_zero(color.mul_scalar(input_scale));
    let lo = if snorm { -1.0_f32 } else { 0.0_f32 };
    let hi = if hdr { f32::MAX } else { 1.0_f32 };
    let t = t.clamp(lo, hi);
    // Primary normals and directional lightmaps use signed normalized input.
    let t = if snorm {
        t.mul_scalar(0.5_f32).add_scalar(0.5_f32)
    } else if hdr {
        // Post-exposure white-floor-preserving, hue-conserving ceiling compression
        // into `[0, Y_MAX]` across the channel axis (`dim = 1` of `[N, 3, H, W]`),
        // keeping `PU(y) * norm_scale` / `Log(y) * norm_scale` inside `[0, 1]`.
        let white = t.clone().min_dim(1);
        let white_clamped = white.clamp_max(Y_MAX);
        let max_c = t.clone().max_dim(1);
        let chroma = t - white_clamped.clone();
        let chroma_max = (max_c - white_clamped.clone()).clamp_min(1e-6_f32);
        let headroom = (Y_MAX - white_clamped.clone()).clamp_min(0.0_f32);
        let scale = headroom.div(chroma_max).clamp(0.0_f32, 1.0_f32);
        (white_clamped + chroma * scale).clamp(0.0_f32, Y_MAX)
    } else {
        t
    };
    apply_transfer_forward(t, transfer)
}

/// Postprocess the network output before slicing back into the accumulator.
///
/// Mirrors `_ref/oidn/devices/cpu/cpu_output_process.isph:37-69`:
/// `nan_to_zero -> clamp(0, +inf) -> inverse transfer -> [scalar mean] ->
/// [snorm demap] ->
/// [ldr clamp] -> *output_scale`.
pub(crate) fn postprocess_color(
    network_output: Tensor<4>,
    transfer: &TransferState,
    hdr: bool,
    snorm: bool,
    output_scale: f32,
    output_channels: usize,
) -> Tensor<4> {
    let t = nan_to_zero(network_output);
    let t = t.clamp(0.0_f32, f32::MAX);
    let t = apply_transfer_inverse(t, transfer);
    // Reference reduces after inverse transfer, before signed decode/clamp.
    let t = if output_channels == 1 {
        t.clone()
            .narrow(1, 0, 1)
            .add(t.clone().narrow(1, 1, 1))
            .add(t.narrow(1, 2, 1))
            .mul_scalar(1.0 / 3.0)
            .repeat_dim(1, 3)
    } else {
        t
    };
    // Decode signed primary output before the non-HDR upper clamp.
    let t = if snorm {
        t.mul_scalar(2.0_f32)
            .sub_scalar(1.0_f32)
            .clamp_min(-1.0_f32)
    } else {
        t
    };
    let t = if !hdr { t.clamp_max(1.0_f32) } else { t };
    let t = nan_to_zero(t.mul_scalar(output_scale));
    if snorm && !hdr {
        t.clamp(-f32::MAX, f32::MAX)
    } else {
        t.clamp(0.0_f32, f32::MAX)
    }
}

// ---------- transfer functions (tensor-vectorised) ----------

/// Forward transfer curve only. No `input_scale`, no clamp — the wrapping
/// `preprocess_input` is responsible for ordering those ops to match the
/// CPU reference.
pub fn apply_transfer_forward(color: Tensor<4>, state: &TransferState) -> Tensor<4> {
    match state.kind {
        TransferFunction::Linear => color,
        TransferFunction::SRGB => srgb_forward_tensor(color),
        TransferFunction::PU => pu_forward_tensor(color).mul_scalar(state.norm_scale),
        TransferFunction::Log => log_forward_tensor(color).mul_scalar(state.norm_scale),
    }
}

/// Inverse transfer curve only. No `output_scale`, no clamp — see
/// `postprocess_color` for the full reference-ordered sequence.
pub fn apply_transfer_inverse(x: Tensor<4>, state: &TransferState) -> Tensor<4> {
    match state.kind {
        TransferFunction::Linear => x,
        TransferFunction::SRGB => srgb_inverse_tensor(x),
        TransferFunction::PU => pu_inverse_tensor(x.mul_scalar(state.rcp_norm_scale)),
        TransferFunction::Log => x
            .mul_scalar(state.rcp_norm_scale)
            .clamp_max(88.0_f32)
            .exp()
            .sub_scalar(1.0_f32),
    }
}

fn srgb_forward_tensor(y: Tensor<4>) -> Tensor<4> {
    // if y <= Y0: A * y    else: B * y^C + D
    let low_mask = y.clone().lower_equal_elem(SRGB_Y0);
    let low = y.clone().mul_scalar(SRGB_A);
    let high = y
        .clamp_min(0.0)
        .powf_scalar(SRGB_C)
        .mul_scalar(SRGB_B)
        .add_scalar(SRGB_D);
    high.mask_where(low_mask, low)
}

fn srgb_inverse_tensor(x: Tensor<4>) -> Tensor<4> {
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

fn pu_forward_tensor(y: Tensor<4>) -> Tensor<4> {
    // Three-region piecewise:
    //   y <= Y0:  A * y
    //   Y0 < y <= Y1:  B * y^C + D
    //   y > Y1:  E * ln(y + F) + G
    let low_mask = y.clone().lower_equal_elem(PU_Y0);
    let mid_mask = y.clone().lower_equal_elem(PU_Y1);
    let b_low = y.clone().mul_scalar(PU_A);
    let b_mid = y
        .clone()
        .clamp_min(0.0)
        .powf_scalar(PU_C)
        .mul_scalar(PU_B)
        .add_scalar(PU_D);
    let b_high = y
        .add_scalar(PU_F)
        .clamp_min(1e-30)
        .log()
        .mul_scalar(PU_E)
        .add_scalar(PU_G);
    // Start from b_high, overlay mid where y <= Y1 (covers both low+mid),
    // then overlay low where y <= Y0. Order yields the correct three-way
    // disjoint split because low ⊂ mid by construction of the masks.
    b_high
        .mask_where(mid_mask, b_mid)
        .mask_where(low_mask, b_low)
}

fn pu_inverse_tensor(x: Tensor<4>) -> Tensor<4> {
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
    // Clamp exponent argument below `ln(f32::MAX) ≈ 88.72` so anomalous U-Net
    // activations (`>= 5.78` normalized) never overflow `f32::exp` to `+Inf`.
    let b_high = x
        .sub_scalar(PU_G)
        .div_scalar(PU_E)
        .clamp_max(88.0_f32)
        .exp()
        .sub_scalar(PU_F);
    b_high
        .mask_where(mid_mask, b_mid)
        .mask_where(low_mask, b_low)
}

fn log_forward_tensor(y: Tensor<4>) -> Tensor<4> {
    // ln((y + 1).max(1e-30)) — matches the CPU branch in
    // `TransferState::forward` for `Log`.
    y.add_scalar(1.0_f32).clamp_min(1e-30).log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    use crate::color::{self, TransferFunction};

    #[test]
    fn signed_primary_sanitizes_after_scale_and_clamps_both_ends() {
        let device = Device::ndarray();
        let source = Tensor::from_data(
            TensorData::new(
                vec![f32::NEG_INFINITY, -2.0, f32::NAN, f32::INFINITY],
                [1, 1, 1, 4],
            ),
            &device,
        );
        let tf = TransferState::new(TransferFunction::Linear);
        let result = preprocess_input(source, 2.0, false, true, &tf)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(result, [0.0, 0.0, 0.5, 1.0]);
        let output =
            Tensor::from_data(TensorData::new(vec![0.5, 1.5, 0.25], [1, 3, 1, 1]), &device);
        let result = postprocess_color(output, &tf, false, true, 2.0, 3)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(result, [0.0, 2.0, -1.0]);
    }

    #[test]
    fn scalar_reduction_precedes_clamp_and_output_scale() {
        let device = Device::ndarray();
        let output =
            Tensor::from_data(TensorData::new(vec![0.5, 1.5, 0.25], [1, 3, 1, 1]), &device);
        let tf = TransferState::new(TransferFunction::Linear);
        let result = postprocess_color(output, &tf, false, false, 2.0, 1)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(result, [1.5; 3]);
    }

    #[test]
    fn scalar_reduction_uses_inverse_transfer_domain() {
        let device = Device::ndarray();
        let values = [0.2, 0.7, 1.2];
        let output = Tensor::from_data(TensorData::new(values.to_vec(), [1, 3, 1, 1]), &device);
        let tf = TransferState::new(TransferFunction::SRGB);
        let expected =
            (values.into_iter().map(color::srgb_inverse).sum::<f32>() / 3.0).min(1.0) * 0.5;
        let result = postprocess_color(output, &tf, false, false, 0.5, 1)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(result.iter().all(|value| (value - expected).abs() < 1e-6));
    }

    /// Forward / inverse round-trip on PU within 1e-4 relative error
    /// — confirms the mask_where cascade matches the CPU piecewise impl.
    #[test]
    fn pu_forward_inverse_roundtrip_ndarray() {
        let device = Device::ndarray();
        // Sample 128 values spanning the three PU regions
        // (low ≤ 1.6e-6, mid ≤ 3.2e-2, high ≥ several).
        let samples: Vec<f32> = (0..128)
            .map(|i| 10.0_f32.powf((i as f32 - 64.0) / 16.0)) // log-spaced
            .collect();

        let t = Tensor::<4>::from_data(TensorData::new(samples.clone(), [1, 1, 1, 128]), &device);
        let tf = TransferState::new(TransferFunction::PU);
        let fwd = apply_transfer_forward(t, &tf);
        let bwd = apply_transfer_inverse(fwd, &tf);
        let result = bwd.into_data().convert::<f32>().try_to_vec::<f32>().unwrap();

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
        let device = Device::ndarray();
        let samples: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let t = Tensor::<4>::from_data(TensorData::new(samples.clone(), [1, 1, 1, 64]), &device);
        let tf = TransferState::new(TransferFunction::PU);
        let tensor_fwd = apply_transfer_forward(t, &tf)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap();
        for (s, got) in samples.iter().zip(tensor_fwd.iter()) {
            // CPU `forward` multiplies by input_scale (=1.0 by default); the
            // tensor path no longer does, so compare against the raw curve.
            let expected = color::pu_forward(*s) * tf.norm_scale;
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
        let device = Device::ndarray();
        let samples: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        let t = Tensor::<4>::from_data(TensorData::new(samples.clone(), [1, 1, 1, 64]), &device);
        let tf = TransferState::new(TransferFunction::SRGB);
        let tensor_fwd = apply_transfer_forward(t, &tf)
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
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

    #[test]
    fn hdr_post_exposure_preserves_scaled_values_above_raw_ymax() {
        let device = Device::ndarray();
        let raw = [200_000.0_f32, 180_000.0, 100_000.0];
        let color = Tensor::<4>::from_data(TensorData::new(raw.to_vec(), [1, 3, 1, 1]), &device);
        let mut tf = TransferState::new(TransferFunction::PU);
        tf.set_input_scale(0.01);

        let encoded = preprocess_input(color, tf.input_scale, true, false, &tf);
        let enc_vals = encoded.clone().into_data().try_to_vec::<f32>().unwrap();
        assert!(
            enc_vals.iter().all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)),
            "encoded PU input must remain within [0, 1]: {enc_vals:?}"
        );

        let decoded = postprocess_color(encoded, &tf, true, false, tf.output_scale, 3)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (expected, actual) in raw.iter().zip(decoded.iter()) {
            let rel = (expected - actual).abs() / expected;
            assert!(
                rel < 1e-4,
                "post-exposure HDR values <= Y_MAX must round-trip uncrushed: expected={expected} actual={actual} rel={rel}"
            );
        }
    }

    #[test]
    fn hdr_post_exposure_compresses_above_ymax_preserving_hue_and_white() {
        let device = Device::ndarray();
        let tf = TransferState::new(TransferFunction::PU);

        // Chromatic highlight `[0, 90_000, 100_000]` -> `[0, 58_953.6, 65_504.0]` (exact 0:9:10 ratio).
        let chromatic = Tensor::<4>::from_data(
            TensorData::new(vec![0.0_f32, 90_000.0, 100_000.0], [1, 3, 1, 1]),
            &device,
        );
        let enc_chroma = preprocess_input(chromatic, 1.0, true, false, &tf);
        let enc_chroma_vals = enc_chroma.clone().into_data().try_to_vec::<f32>().unwrap();
        assert!(
            enc_chroma_vals
                .iter()
                .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)),
            "compressed PU input must stay in [0, 1]: {enc_chroma_vals:?}"
        );
        let dec_chroma = postprocess_color(enc_chroma, &tf, true, false, 1.0, 3)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(dec_chroma[0].abs() < 1e-5, "zero channel drifted: {}", dec_chroma[0]);
        assert!(
            (dec_chroma[2] - Y_MAX).abs() / Y_MAX < 1e-4,
            "peak channel must land at Y_MAX: {}",
            dec_chroma[2]
        );
        assert!(
            (dec_chroma[1] / dec_chroma[2] - 0.9).abs() < 1e-4,
            "0:9:10 hue ratio was not preserved: {dec_chroma:?}"
        );

        // Pure white highlight `[100_000; 3]` -> `[65_504; 3]` with zero 0/0 NaNs.
        let white =
            Tensor::<4>::from_data(TensorData::new(vec![100_000.0_f32; 3], [1, 3, 1, 1]), &device);
        let enc_white = preprocess_input(white, 1.0, true, false, &tf);
        let enc_white_vals = enc_white.clone().into_data().try_to_vec::<f32>().unwrap();
        assert!(
            enc_white_vals
                .iter()
                .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)),
            "compressed white PU input must stay in [0, 1]: {enc_white_vals:?}"
        );
        let dec_white = postprocess_color(enc_white, &tf, true, false, 1.0, 3)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for &c in &dec_white {
            assert!(
                c.is_finite() && (c - Y_MAX).abs() / Y_MAX < 1e-4,
                "pure white must compress to Y_MAX without NaN: {dec_white:?}"
            );
        }

        // White floor + excess chroma `[10_000, 55_000, 100_000]` -> `[10_000, 37_752, 65_504]`.
        let pastel = Tensor::<4>::from_data(
            TensorData::new(vec![10_000.0_f32, 55_000.0, 100_000.0], [1, 3, 1, 1]),
            &device,
        );
        let enc_pastel = preprocess_input(pastel, 1.0, true, false, &tf);
        let dec_pastel = postprocess_color(enc_pastel, &tf, true, false, 1.0, 3)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let expected_pastel = [10_000.0_f32, 37_752.0, Y_MAX];
        for (expected, actual) in expected_pastel.iter().zip(dec_pastel.iter()) {
            let rel = (expected - actual).abs() / expected;
            assert!(
                rel < 1e-4,
                "white-floor chroma compression mismatch: expected={expected} actual={actual} rel={rel}"
            );
        }
    }

    #[test]
    fn pu_inverse_and_postprocess_stay_finite_on_extreme_activations() {
        let device = Device::ndarray();
        let mut tf = TransferState::new(TransferFunction::PU);
        tf.set_input_scale(0.01); // output_scale = 100.0

        let extreme_unnorm =
            Tensor::<4>::from_data(TensorData::new(vec![10.0 * tf.rcp_norm_scale], [1, 1, 1, 1]), &device);
        let inv = pu_inverse_tensor(extreme_unnorm)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            inv[0].is_finite() && inv[0] > Y_MAX,
            "pu_inverse_tensor must clamp exp argument and remain finite: {}",
            inv[0]
        );

        for output_scale in [1.0_f32, tf.output_scale] {
            let net_out = Tensor::<4>::from_data(
                TensorData::new(vec![10.0_f32, -5.0, f32::NAN], [1, 3, 1, 1]),
                &device,
            );
            let post = postprocess_color(net_out, &tf, true, false, output_scale, 3)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            assert!(
                post.iter().all(|v| v.is_finite() && *v >= 0.0),
                "HDR postprocess_color must sanitize non-finites/negatives at scale={output_scale}: {post:?}"
            );
            assert!(post[0] > Y_MAX);
            assert_eq!(post[1], 0.0);
            assert_eq!(post[2], 0.0);
        }
    }
}
