//! Transfer functions — direct port of `_ref/oidn/core/color.{h,cpp}`.
//!
//! Operates on `f32` (network input/output precision is decided per-backend by
//! Burn). All four OIDN modes are reproduced (`Linear`, `SRGB`, `PU`, `Log`)
//! with the same constants, the same `inputScale`/`outputScale`/`normScale`
//! semantics, and the same per-channel application (`forward`/`inverse` on
//! each of R,G,B independently).
//!
//! Constants are reproduced verbatim from `color.h` to keep numerical drift
//! against the reference at machine epsilon.

#![allow(clippy::excessive_precision)]

/// Maximum representable HDR luminance — equal to `half::MAX`, used for
/// computing `normScale` so the PU range maps roughly into `[0, 1]`.
pub const Y_MAX: f32 = 65504.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunction {
    Linear,
    SRGB,
    PU,
    Log,
}

/// Runtime state for a transfer function — `inputScale`/`outputScale` are
/// updated per-frame from autoexposure (HDR) or set to 1.0 (LDR).
#[derive(Debug, Clone, Copy)]
pub struct TransferState {
    pub kind: TransferFunction,
    pub input_scale: f32,
    pub output_scale: f32,
    pub norm_scale: f32,
    pub rcp_norm_scale: f32,
}

impl TransferState {
    pub fn new(kind: TransferFunction) -> Self {
        let mut s = Self {
            kind,
            input_scale: 1.0,
            output_scale: 1.0,
            norm_scale: 1.0,
            rcp_norm_scale: 1.0,
        };
        // Mirror `TransferFunction::TransferFunction(Type)` ctor from color.cpp:
        // normScale = 1 / max-component(forward(yMax)).
        let scaled = forward_one(kind, Y_MAX, 1.0);
        let xmax = scaled.max(forward_one(kind, Y_MAX, 1.0));
        s.norm_scale = if xmax > 0.0 { 1.0 / xmax } else { 1.0 };
        s.rcp_norm_scale = if s.norm_scale != 0.0 { 1.0 / s.norm_scale } else { 1.0 };
        s
    }

    pub fn set_input_scale(&mut self, scale: f32) {
        self.input_scale = scale;
        self.output_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    }

    /// Apply forward transfer with current scaling — `(y * inputScale)` then
    /// non-linear curve, with the PU/Log result multiplied by `normScale`.
    pub fn forward(&self, y: f32) -> f32 {
        let y = y * self.input_scale;
        match self.kind {
            TransferFunction::Linear => y,
            TransferFunction::SRGB => srgb_forward(y),
            TransferFunction::PU => pu_forward(y) * self.norm_scale,
            TransferFunction::Log => (y + 1.0).max(1e-30).ln() * self.norm_scale,
        }
    }

    /// Apply inverse transfer and remove `outputScale` — undo what `forward` did.
    pub fn inverse(&self, x: f32) -> f32 {
        let unscaled_norm = match self.kind {
            TransferFunction::Linear => x,
            TransferFunction::SRGB => srgb_inverse(x),
            TransferFunction::PU => pu_inverse(x * self.rcp_norm_scale),
            TransferFunction::Log => (x * self.rcp_norm_scale).exp() - 1.0,
        };
        unscaled_norm * self.output_scale
    }
}

#[inline]
fn forward_one(kind: TransferFunction, y: f32, input_scale: f32) -> f32 {
    let y = y * input_scale;
    match kind {
        TransferFunction::Linear => y,
        TransferFunction::SRGB => srgb_forward(y),
        TransferFunction::PU => pu_forward(y),
        TransferFunction::Log => (y + 1.0).max(1e-30).ln(),
    }
}

// ---------- sRGB constants from color.h ----------
pub(crate) const SRGB_A: f32 = 12.92;
pub(crate) const SRGB_B: f32 = 1.055;
pub(crate) const SRGB_C: f32 = 1.0 / 2.4;
pub(crate) const SRGB_D: f32 = -0.055;
pub(crate) const SRGB_Y0: f32 = 0.0031308;
pub(crate) const SRGB_X0: f32 = 0.04045;

#[inline]
pub fn srgb_forward(y: f32) -> f32 {
    if y <= SRGB_Y0 { SRGB_A * y } else { SRGB_B * y.powf(SRGB_C) + SRGB_D }
}

#[inline]
pub fn srgb_inverse(x: f32) -> f32 {
    if x <= SRGB_X0 { x / SRGB_A } else { ((x - SRGB_D) / SRGB_B).powf(1.0 / SRGB_C) }
}

// ---------- PU constants from color.h ----------
pub(crate) const PU_A: f32 = 1.41283765e+03;
pub(crate) const PU_B: f32 = 1.64593172e+00;
pub(crate) const PU_C: f32 = 4.31384981e-01;
pub(crate) const PU_D: f32 = -2.94139609e-03;
pub(crate) const PU_E: f32 = 1.92653254e-01;
pub(crate) const PU_F: f32 = 6.26026094e-03;
pub(crate) const PU_G: f32 = 9.98620152e-01;
pub(crate) const PU_Y0: f32 = 1.57945760e-06;
pub(crate) const PU_Y1: f32 = 3.22087631e-02;
pub(crate) const PU_X0: f32 = 2.23151711e-03;
pub(crate) const PU_X1: f32 = 3.70974749e-01;

#[inline]
pub fn pu_forward(y: f32) -> f32 {
    if y <= PU_Y0 {
        PU_A * y
    } else if y <= PU_Y1 {
        PU_B * y.powf(PU_C) + PU_D
    } else {
        PU_E * (y + PU_F).max(1e-30).ln() + PU_G
    }
}

#[inline]
pub fn pu_inverse(x: f32) -> f32 {
    if x <= PU_X0 {
        x / PU_A
    } else if x <= PU_X1 {
        ((x - PU_D) / PU_B).powf(1.0 / PU_C)
    } else {
        ((x - PU_G) / PU_E).exp() - PU_F
    }
}

#[inline]
pub fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.212671 * r + 0.715160 * g + 0.072169 * b
}
