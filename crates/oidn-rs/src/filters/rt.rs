//! RT filter — public API + glue, equivalent to `_ref/oidn/core/rt_filter.cpp`.
//!
//! In burn 0.22 the backend is dynamic (device-selected), so the same filter
//! type runs on CPU (`Device::ndarray()`) for tests and on wgpu
//! (`Device::wgpu(..)`) in the CLI — the choice is the device, not a type
//! parameter.
//!
//! Supports two network topologies via the `Net` enum dispatcher:
//! Base/Small `UNet` and Large/XL `UNetLarge`. A validated archive descriptor
//! derives topology, all channel widths and RF; filenames are selection metadata.
//!
//! ## Two parallel I/O modes
//!
//! - **Legacy `Image<'_>` mode.** [`RtFilter::set_color`] / [`RtFilter::set_albedo`]
//!   / [`RtFilter::set_normal`] take byte-backed images; [`RtFilter::allocate_output`]
//!   reserves a host-side buffer; [`RtFilter::take_output`] returns those bytes.
//!   Used by the CLI and the test fixtures.
//! - **Tensor mode.** [`RtFilter::set_color_tensor`] etc. take a Burn
//!   `Tensor<4>` (`[1, 3, H, W]` NCHW). [`RtFilter::allocate_output_tensor`]
//!   declares the output shape; [`RtFilter::take_output_tensor`] returns the
//!   denoised accumulator. The shared tensor core also serves immutable
//!   [`CommittedRtFilter`], used by squarebob's wgpu bridge.
//!
//! [`Filter::execute`] dispatches between the two paths based on which
//! input slot is populated. Mixing modes within one `commit() / execute()`
//! cycle is not supported — pick one set of setters per call.

use std::path::PathBuf;

use burn::tensor::{Device, Tensor};
use oidn_model::Net;

use crate::{
    color::TransferFunction,
    error::OidnError,
    filter::{Filter, Quality},
    filters::unet_runner::{self, ProgressFn, RunOptions},
    image::{Image, OwnedImage, PixelFormat},
    registry::{ModelKey, select_rt},
    tile::TilePlan,
    weights::{self, SourcePolicy},
};

pub struct RtFilterBuilder<'b> {
    device: &'b Device,
    weights_dir: PathBuf,
    weight_source: SourcePolicy,
    hdr: bool,
    srgb: bool,
    clean_aux: bool,
    quality: Quality,
    user_input_scale: Option<f32>,
    user_weights: Option<Vec<u8>>,
    max_memory_mb: Option<i32>,
    nan_to_zero: bool,
}

impl<'b> RtFilterBuilder<'b> {
    pub fn new(device: &'b Device, weights_dir: impl Into<PathBuf>) -> Self {
        let weights_dir = weights_dir.into();
        let weight_source = if weights_dir.as_os_str().is_empty() {
            SourcePolicy::EmbeddedFirst
        } else {
            SourcePolicy::DiskFirst
        };
        Self {
            device,
            weights_dir,
            weight_source,
            hdr: false,
            srgb: false,
            clean_aux: false,
            quality: Quality::High,
            user_input_scale: None,
            user_weights: None,
            max_memory_mb: None,
            nan_to_zero: true,
        }
    }

    /// Choose embedded/filesystem source precedence without hiding I/O errors.
    pub fn weight_source(mut self, policy: SourcePolicy) -> Self {
        self.weight_source = policy;
        self
    }

    pub fn hdr(mut self, v: bool) -> Self {
        self.hdr = v;
        self
    }
    pub fn srgb(mut self, v: bool) -> Self {
        self.srgb = v;
        self
    }
    pub fn clean_aux(mut self, v: bool) -> Self {
        self.clean_aux = v;
        self
    }
    pub fn quality(mut self, q: Quality) -> Self {
        self.quality = q;
        self
    }
    pub fn input_scale(mut self, s: Option<f32>) -> Self {
        self.user_input_scale = s;
        self
    }

    /// Use the caller-supplied TZA blob instead of looking up a model in
    /// `weights_dir`. Bypasses built-in candidate selection, while retaining mode validation —
    /// callers are responsible for matching the blob's channel counts to
    /// the input set. Variant (`UNet` vs `UNetLarge`) is auto-detected from
    /// tensor names.
    pub fn weights(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.user_weights = Some(bytes.into());
        self
    }

    /// Best-effort logical memory budget in MiB. Accounts for parameters,
    /// full-image input/output and a conservative activation estimate. Backend
    /// workspaces/allocator overhead are additional; the minimum tile may exceed
    /// an infeasible budget and is logged. Negative values restore the default.
    pub fn max_memory_mb(mut self, mb: i32) -> Self {
        self.max_memory_mb = (mb >= 0).then_some(mb);
        self
    }

    /// Optional stronger policy replacing NaN and infinities before scaling.
    /// Default true. When disabled, the shared kernels still apply native
    /// NaN-only sanitation after scale, followed by range clamps.
    pub fn nan_to_zero(mut self, v: bool) -> Self {
        self.nan_to_zero = v;
        self
    }

    pub fn build(self) -> RtFilter<'b> {
        RtFilter {
            device: self.device,
            weights_dir: self.weights_dir,
            weight_source: self.weight_source,
            hdr: self.hdr,
            srgb: self.srgb,
            clean_aux: self.clean_aux,
            quality: self.quality,
            user_input_scale: self.user_input_scale,
            user_weights: self.user_weights,
            max_memory_mb: self.max_memory_mb,
            nan_to_zero: self.nan_to_zero,
            color: None,
            albedo: None,
            normal: None,
            output: None,
            color_tensor: None,
            albedo_tensor: None,
            normal_tensor: None,
            output_tensor: None,
            output_tensor_dims: None,
            net: None,
            plan: None,
            model_key: None,
            progress: None,
            committed: false,
            last_committed_dims: None,
            input_signature: [None; 3],
        }
    }
}

pub struct RtFilter<'b> {
    device: &'b Device,
    weights_dir: PathBuf,
    weight_source: SourcePolicy,
    hdr: bool,
    srgb: bool,
    clean_aux: bool,
    quality: Quality,
    user_input_scale: Option<f32>,
    user_weights: Option<Vec<u8>>,
    max_memory_mb: Option<i32>,
    nan_to_zero: bool,

    // --- Legacy Image<'_> path ---
    color: Option<OwnedImage>,
    albedo: Option<OwnedImage>,
    normal: Option<OwnedImage>,
    output: Option<OwnedImage>,

    // --- Tensor path (zero host-roundtrip; Phase I.5/I.6) ---
    color_tensor: Option<Tensor<4>>,
    albedo_tensor: Option<Tensor<4>>,
    normal_tensor: Option<Tensor<4>>,
    /// Populated by `execute()` in tensor mode; consumed by
    /// [`RtFilter::take_output_tensor`].
    output_tensor: Option<Tensor<4>>,
    /// `(width, height)` declared by [`RtFilter::allocate_output_tensor`].
    /// Doubles as the tile-plan / shape source when the tensor path is
    /// active.
    output_tensor_dims: Option<(usize, usize)>,

    net: Option<Net>,
    plan: Option<TilePlan>,
    model_key: Option<ModelKey>,
    progress: Option<Box<ProgressFn<'static>>>,
    committed: bool,
    /// Output dims/format from the most recent successful `commit()` in
    /// legacy mode. Tensor mode tracks its own dims via
    /// `output_tensor_dims`; both feed [`Self::output_dims`].
    last_committed_dims: Option<(usize, usize, PixelFormat)>,
    input_signature: [Option<[usize; 4]>; 3],
}

/// Immutable RT denoise state for tensor-native callers.
///
/// This owns the expensive committed state (`Net` weights + tile plan) but no
/// per-pass input/output tensor slots. Reuse this across frames/passes, and
/// pass fresh tensors to [`Self::execute_tensors`] each time.
pub struct CommittedRtFilter<'b> {
    device: &'b Device,
    hdr: bool,
    transfer: TransferFunction,
    user_input_scale: Option<f32>,
    nan_to_zero: bool,
    has_color: bool,
    has_albedo: bool,
    has_normal: bool,
    width: usize,
    height: usize,
    net: Net,
    plan: TilePlan,
    model_key: ModelKey,
}

struct RtCommitArtifacts {
    net: Net,
    plan: TilePlan,
    model_key: ModelKey,
}

impl<'b> RtFilter<'b> {
    pub fn builder(device: &'b Device, weights_dir: impl Into<PathBuf>) -> RtFilterBuilder<'b> {
        RtFilterBuilder::new(device, weights_dir)
    }

    // ----- Legacy Image-based inputs -----

    /// Replace the color image. Note: this does *not* invalidate the
    /// committed model/plan — when only pixel content changes (same
    /// dimensions, same input set), `execute()` reuses the cached UNet
    /// and tile plan. Only mode/quality/dims changes need a fresh
    /// `commit()`.
    pub fn set_color(&mut self, img: &Image<'_>) -> Result<(), OidnError> {
        let image = OwnedImage::from(img)?;
        let needs_invalidate = self.color.as_ref().is_none_or(|old| {
            (old.width, old.height, old.format) != (img.width, img.height, img.format)
        });
        self.color = Some(image);
        if needs_invalidate {
            self.committed = false;
        }
        Ok(())
    }
    pub fn set_albedo(&mut self, img: &Image<'_>) -> Result<(), OidnError> {
        let image = OwnedImage::from(img)?;
        let needs_invalidate = self.albedo.as_ref().is_none_or(|old| {
            (old.width, old.height, old.format) != (img.width, img.height, img.format)
        });
        self.albedo = Some(image);
        if needs_invalidate {
            self.committed = false;
        }
        Ok(())
    }
    pub fn set_normal(&mut self, img: &Image<'_>) -> Result<(), OidnError> {
        let image = OwnedImage::from(img)?;
        let needs_invalidate = self.normal.as_ref().is_none_or(|old| {
            (old.width, old.height, old.format) != (img.width, img.height, img.format)
        });
        self.normal = Some(image);
        if needs_invalidate {
            self.committed = false;
        }
        Ok(())
    }

    // ----- Tensor-native inputs (zero host roundtrip) -----

    /// Tensor-native colour input. Shape `[1, 3, H, W]` (NCHW), `f32`.
    /// The tensor is stored by reference (Burn tensors are cheap to
    /// `clone()`); no data crosses to host. `execute()` runs the
    /// tensor-native pipeline when *any* `set_*_tensor` was used.
    pub fn set_color_tensor(&mut self, t: Tensor<4>) -> Result<(), OidnError> {
        if t.device() != *self.device {
            return Err(OidnError::InvalidArgument(
                "input tensor must be on the selected device",
            ));
        }
        let d = t.dims();
        crate::image::validate_dimensions(d[3], d[2], d[1])?;
        if d[0] != 1 || d[1] != 3 {
            return Err(OidnError::InvalidArgument("input tensor expects [1,3,H,W]"));
        }
        let needs_invalidate = self.input_signature[0] != Some(d);
        self.color_tensor = Some(t);
        if needs_invalidate {
            self.committed = false;
        }
        Ok(())
    }

    /// Tensor-native albedo input. See [`RtFilter::set_color_tensor`].
    pub fn set_albedo_tensor(&mut self, t: Tensor<4>) -> Result<(), OidnError> {
        if t.device() != *self.device {
            return Err(OidnError::InvalidArgument(
                "input tensor must be on the selected device",
            ));
        }
        let d = t.dims();
        crate::image::validate_dimensions(d[3], d[2], d[1])?;
        if d[0] != 1 || d[1] != 3 {
            return Err(OidnError::InvalidArgument("input tensor expects [1,3,H,W]"));
        }
        let needs_invalidate = self.input_signature[1] != Some(d);
        self.albedo_tensor = Some(t);
        if needs_invalidate {
            self.committed = false;
        }
        Ok(())
    }

    /// Tensor-native normal input. See [`RtFilter::set_color_tensor`].
    pub fn set_normal_tensor(&mut self, t: Tensor<4>) -> Result<(), OidnError> {
        if t.device() != *self.device {
            return Err(OidnError::InvalidArgument(
                "input tensor must be on the selected device",
            ));
        }
        let d = t.dims();
        crate::image::validate_dimensions(d[3], d[2], d[1])?;
        if d[0] != 1 || d[1] != 3 {
            return Err(OidnError::InvalidArgument("input tensor expects [1,3,H,W]"));
        }
        let needs_invalidate = self.input_signature[2] != Some(d);
        self.normal_tensor = Some(t);
        if needs_invalidate {
            self.committed = false;
        }
        Ok(())
    }

    /// Take ownership of the denoised output as a `[1, 3, H, W]` (NCHW)
    /// `f32` Burn tensor.
    ///
    /// Returns `None` if `execute()` has not been called yet or the
    /// output slot was already consumed. Re-invoking the filter at the
    /// same shape uses the retained output declaration and fresh input handles;
    /// [`RtFilter::allocate_output_tensor`] remains idempotent.
    pub fn take_output_tensor(&mut self) -> Option<Tensor<4>> {
        self.output_tensor.take()
    }

    // ----- Output allocation (legacy + tensor) -----

    pub fn allocate_output(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<(), OidnError> {
        // Skip `committed = false` when the requested output dims/format
        // match the previously-committed ones. `take_output()` leaves
        // `self.output = None` even when the renderer wants to denoise
        // again at the same dims — without this check we'd rebuild the
        // UNet and tile plan every single call.
        let same_dims = self.last_committed_dims == Some((width, height, format));
        self.output = Some(OwnedImage::empty(width, height, format)?);
        // Clear any tensor-mode shape so the dispatcher picks the legacy
        // path next time.
        self.output_tensor_dims = None;
        if !same_dims {
            self.committed = false;
        }
        Ok(())
    }

    /// Declare the output shape for tensor-mode execution. No tensor is
    /// allocated up-front — `execute()` builds the accumulator with
    /// `Tensor::zeros([1, 3, h, w], device)` and hands it back via
    /// [`RtFilter::take_output_tensor`].
    pub fn allocate_output_tensor(&mut self, width: usize, height: usize) -> Result<(), OidnError> {
        crate::image::validate_dimensions(width, height, 3)?;
        let same_dims = self.output_tensor_dims == Some((width, height));
        self.output_tensor_dims = Some((width, height));
        // Drop the legacy buffer; we're going tensor.
        self.output = None;
        if !same_dims {
            self.committed = false;
        }
        Ok(())
    }

    pub fn take_output(&mut self) -> Option<(Vec<u8>, usize, usize, PixelFormat)> {
        let o = self.output.take()?;
        Some((o.data, o.width, o.height, o.format))
    }

    /// Returns the model key chosen at `commit()` time (after quality-based
    /// upgrade has resolved to a `_large` / `_small` variant if applicable).
    pub fn model_key(&self) -> Option<&ModelKey> {
        self.model_key.as_ref()
    }

    /// Build an immutable tensor-native filter for repeated denoise passes.
    ///
    /// Unlike caching [`RtFilter`] itself, the returned object does not retain
    /// color/albedo/normal/output tensor handles between calls. This preserves
    /// the expensive committed model and tile plan without carrying mutable
    /// per-pass state across GPU submissions.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_tensor_model(
        &self,
        width: usize,
        height: usize,
        has_color: bool,
        has_albedo: bool,
        has_normal: bool,
    ) -> Result<CommittedRtFilter<'b>, OidnError> {
        if !has_color && !has_albedo && !has_normal {
            return Err(OidnError::Unset("color/albedo/normal"));
        }
        let artifacts =
            self.build_commit_artifacts(width, height, has_color, has_albedo, has_normal)?;
        Ok(CommittedRtFilter {
            device: self.device,
            hdr: self.hdr,
            transfer: self.transfer_kind(has_color, has_normal),
            user_input_scale: self.user_input_scale,
            nan_to_zero: self.nan_to_zero,
            has_color,
            has_albedo,
            has_normal,
            width,
            height,
            net: artifacts.net,
            plan: artifacts.plan,
            model_key: artifacts.model_key,
        })
    }

    /// Toggle NaN/Inf input sanitisation. See
    /// [`RtFilterBuilder::nan_to_zero`] for rationale.
    pub fn set_nan_to_zero(&mut self, v: bool) {
        self.nan_to_zero = v;
    }

    /// Install a progress callback. Receives `[0.0, 1.0]` after each
    /// processed tile; returning `false` aborts execution with
    /// `OidnError::Cancelled`.
    pub fn set_progress<F: FnMut(f32) -> bool + 'static>(&mut self, callback: F) {
        self.progress = Some(Box::new(callback));
    }

    /// True when at least one tensor input slot is populated. Used by
    /// [`Filter::execute`] to dispatch between the two pipelines.
    fn tensor_mode(&self) -> bool {
        self.output_tensor_dims.is_some()
            || self.color_tensor.is_some()
            || self.albedo_tensor.is_some()
            || self.normal_tensor.is_some()
    }

    /// `(width, height)` of the active output target, regardless of mode.
    fn output_dims(&self) -> Option<(usize, usize)> {
        if let Some(o) = &self.output {
            Some((o.width, o.height))
        } else {
            self.output_tensor_dims
        }
    }

    /// Reference: `_ref/oidn/core/rt_filter.cpp:55-68` — transfer kind depends
    /// on primary role and mode flags. Normal-only or explicitly encoded sRGB
    /// input uses Linear; HDR color uses PU; linear LDR/albedo uses SRGB.
    fn transfer_kind(&self, has_color: bool, has_normal: bool) -> TransferFunction {
        if self.srgb || (!has_color && has_normal) {
            TransferFunction::Linear
        } else if self.hdr {
            TransferFunction::PU
        } else {
            TransferFunction::SRGB
        }
    }

    fn build_commit_artifacts(
        &self,
        out_w: usize,
        out_h: usize,
        has_color: bool,
        has_albedo: bool,
        has_normal: bool,
    ) -> Result<RtCommitArtifacts, OidnError> {
        super::validate_execution(out_w, out_h, self.user_input_scale)?;
        crate::registry::validate_rt(has_color, has_albedo, has_normal, self.hdr, self.srgb)?;
        let (stem, bytes) = if let Some(bytes) = &self.user_weights {
            ("user".to_owned(), bytes.clone())
        } else {
            let base_key = select_rt(
                has_color,
                has_albedo,
                has_normal,
                self.hdr,
                self.srgb,
                self.clean_aux,
                self.quality,
            )?;
            let resolved = weights::resolve(
                &base_key,
                self.quality,
                Some(&self.weights_dir),
                self.weight_source,
            )?
            .ok_or_else(|| OidnError::MissingModel(self.weights_dir.join(base_key.filename())))?;
            log::debug!(
                "resolved model={} source={:?}",
                resolved.stem,
                resolved.source
            );
            (resolved.stem, resolved.bytes)
        };
        let in_channels =
            (usize::from(has_color) + usize::from(has_albedo) + usize::from(has_normal)) * 3;
        let (net, plan) = super::build_commit_artifacts(
            self.device,
            &bytes,
            in_channels,
            out_w,
            out_h,
            self.max_memory_mb,
            self.user_input_scale,
        )?;

        Ok(RtCommitArtifacts {
            net,
            plan,
            model_key: ModelKey::new(stem),
        })
    }
}

impl<'b> CommittedRtFilter<'b> {
    /// Returns the model key chosen when this committed filter was built.
    pub fn model_key(&self) -> &ModelKey {
        &self.model_key
    }

    /// Output dimensions this committed filter was built for.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Toggle NaN/Inf input sanitisation at runtime. The flag is read
    /// on every [`Self::execute_tensors`] call, so this can be
    /// flipped between passes without rebuilding the committed model.
    pub fn set_nan_to_zero(&mut self, v: bool) {
        self.nan_to_zero = v;
    }

    /// Override the autoexposure input scale at runtime. `None`
    /// reverts to OIDN's built-in autoexposure (recomputed each
    /// pass); `Some(s)` clamps it to a fixed value — the recommended
    /// path for physical-camera pipelines that own exposure
    /// themselves.
    pub fn set_input_scale(&mut self, scale: Option<f32>) {
        self.user_input_scale = scale;
    }

    /// Run one tensor-native denoise pass with fresh per-pass inputs.
    ///
    /// The input presence must match the layout used at commit time. The
    /// committed object keeps no references to these tensors after returning.
    pub fn execute_tensors(
        &self,
        color: Option<Tensor<4>>,
        albedo: Option<Tensor<4>>,
        normal: Option<Tensor<4>>,
        progress: Option<&mut ProgressFn<'_>>,
    ) -> Result<Tensor<4>, OidnError> {
        validate_tensor_slot(
            self.has_color,
            color.as_ref(),
            self.width,
            self.height,
            "color_tensor",
        )?;
        validate_tensor_slot(
            self.has_albedo,
            albedo.as_ref(),
            self.width,
            self.height,
            "albedo_tensor",
        )?;
        validate_tensor_slot(
            self.has_normal,
            normal.as_ref(),
            self.width,
            self.height,
            "normal_tensor",
        )?;
        unet_runner::run_tensors(
            &self.net,
            self.device,
            &self.plan,
            color,
            albedo,
            normal,
            self.width,
            self.height,
            RunOptions {
                transfer: self.transfer,
                hdr: self.hdr,
                signed: !self.has_color && self.has_normal,
                input_scale: self.user_input_scale,
                sanitize_nonfinite: self.nan_to_zero,
                output_channels: 3,
            },
            progress,
        )
    }
}

fn validate_tensor_slot(
    required: bool,
    tensor: Option<&Tensor<4>>,
    width: usize,
    height: usize,
    name: &'static str,
) -> Result<(), OidnError> {
    match (required, tensor) {
        (true, None) => Err(OidnError::Unset(name)),
        (false, Some(_)) => Err(OidnError::Inconsistent(name)),
        (false, None) => Ok(()),
        (true, Some(t)) => {
            let d = t.dims();
            if d == [1, 3, height, width] {
                Ok(())
            } else {
                Err(OidnError::Inconsistent(name))
            }
        }
    }
}

impl<'b> Filter for RtFilter<'b> {
    fn set_progress(&mut self, cb: Box<dyn FnMut(f32) -> bool + 'static>) -> Result<(), OidnError> {
        // The inherent `set_progress` boxes any `F: FnMut(f32) -> bool +
        // 'static`; the trait method already takes a box, so store it
        // without re-boxing.
        self.progress = Some(cb);
        Ok(())
    }

    fn commit(&mut self) -> Result<(), OidnError> {
        self.committed = false;
        if let Some(output) = &self.output {
            output.view().validate()?;
        }
        let any_input_legacy =
            self.color.is_some() || self.albedo.is_some() || self.normal.is_some();
        let any_input_tensor = self.tensor_mode();
        if any_input_legacy && any_input_tensor {
            return Err(OidnError::InvalidArgument(
                "host and tensor modes cannot be mixed",
            ));
        }
        if !any_input_legacy && !any_input_tensor {
            return Err(OidnError::Unset("color/albedo/normal"));
        }

        if self.hdr && self.srgb {
            return Err(OidnError::InvalidArgument(
                "hdr and srgb are mutually exclusive",
            ));
        }

        // Channel count is taken from whichever side (legacy or tensor)
        // is populated. Mixing both modes for the same slot is undefined;
        // we treat any input slot as a present channel triple.
        let has_color = self.color.is_some() || self.color_tensor.is_some();
        let has_albedo = self.albedo.is_some() || self.albedo_tensor.is_some();
        let has_normal = self.normal.is_some() || self.normal_tensor.is_some();
        let (out_w, out_h) = self.output_dims().ok_or(OidnError::Unset("output"))?;
        // Cross-check that every populated input matches the declared
        // output geometry — applies to both modes.
        let check_dims = |w: usize, h: usize, name: &'static str| -> Result<(), OidnError> {
            if w != out_w || h != out_h {
                Err(OidnError::Inconsistent(name))
            } else {
                Ok(())
            }
        };
        if let Some(c) = &self.color {
            c.view().validate()?;
            check_dims(c.width, c.height, "color")?;
        }
        if let Some(a) = &self.albedo {
            a.view().validate()?;
            check_dims(a.width, a.height, "albedo")?;
        }
        if let Some(n) = &self.normal {
            n.view().validate()?;
            check_dims(n.width, n.height, "normal")?;
        }
        if let Some(t) = &self.color_tensor {
            let d = t.dims();
            check_dims(d[3], d[2], "color_tensor")?;
        }
        if let Some(t) = &self.albedo_tensor {
            let d = t.dims();
            check_dims(d[3], d[2], "albedo_tensor")?;
        }
        if let Some(t) = &self.normal_tensor {
            let d = t.dims();
            check_dims(d[3], d[2], "normal_tensor")?;
        }

        let artifacts =
            self.build_commit_artifacts(out_w, out_h, has_color, has_albedo, has_normal)?;
        self.model_key = Some(artifacts.model_key);
        self.net = Some(artifacts.net);
        self.plan = Some(artifacts.plan);

        self.input_signature = [
            self.color_tensor.as_ref().map(Tensor::dims),
            self.albedo_tensor.as_ref().map(Tensor::dims),
            self.normal_tensor.as_ref().map(Tensor::dims),
        ];
        self.committed = true;
        // last_committed_dims is only meaningful for the legacy path
        // (which queries it via `allocate_output`); tensor-mode dims
        // live in `output_tensor_dims` and are cached identically.
        if let Some(o) = &self.output {
            self.last_committed_dims = Some((o.width, o.height, o.format));
        }
        Ok(())
    }

    fn execute(&mut self) -> Result<(), OidnError> {
        if !self.committed {
            self.commit()?;
        }
        let net = self.net.as_ref().ok_or(OidnError::Unset("model"))?;
        let plan = self.plan.as_ref().ok_or(OidnError::Unset("plan"))?;

        let has_color = self.color.is_some() || self.input_signature[0].is_some();
        let has_normal = self.normal.is_some() || self.input_signature[2].is_some();
        let transfer = self.transfer_kind(has_color, has_normal);
        let options = RunOptions {
            transfer,
            hdr: self.hdr,
            signed: !has_color && has_normal,
            input_scale: self.user_input_scale,
            sanitize_nonfinite: self.nan_to_zero,
            output_channels: 3,
        };

        if self.tensor_mode() {
            let (out_w, out_h) = self
                .output_tensor_dims
                .ok_or(OidnError::Unset("output_tensor"))?;
            let progress: Option<&mut ProgressFn<'_>> = self.progress.as_deref_mut();
            let result = unet_runner::run_tensors(
                net,
                self.device,
                plan,
                self.color_tensor.clone(),
                self.albedo_tensor.clone(),
                self.normal_tensor.clone(),
                out_w,
                out_h,
                options,
                progress,
            )?;
            self.output_tensor = Some(result);
            // Per-pass handles are released; committed layout signatures remain.
            // Backend submission/resource ownership governs device lifetimes.
            self.color_tensor = None;
            self.albedo_tensor = None;
            self.normal_tensor = None;
            Ok(())
        } else {
            let output = self.output.as_mut().ok_or(OidnError::Unset("output"))?;
            let color = self.color.as_ref().map(|i| i.view());
            let albedo = self.albedo.as_ref().map(|i| i.view());
            let normal = self.normal.as_ref().map(|i| i.view());

            let mut out_view = output.view_mut();
            let progress: Option<&mut ProgressFn<'_>> = self.progress.as_deref_mut();
            unet_runner::run(
                net,
                self.device,
                plan,
                color.as_ref(),
                albedo.as_ref(),
                normal.as_ref(),
                &mut out_view,
                options,
                progress,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_contracts_fail_before_model_io() {
        let device = Device::ndarray();
        for (hdr, srgb, color, albedo, normal) in [
            (true, true, true, false, false),
            (true, false, false, true, false),
            (false, true, false, false, true),
            (false, false, false, true, true),
        ] {
            let filter = RtFilter::builder(&device, "missing-model-directory")
                .hdr(hdr)
                .srgb(srgb)
                .weights([0u8])
                .build();
            assert!(matches!(
                filter.commit_tensor_model(16, 16, color, albedo, normal),
                Err(OidnError::InvalidArgument(_))
            ));
        }
        let filter = RtFilter::builder(&device, "missing-model-directory").build();
        assert!(matches!(
            filter.commit_tensor_model(usize::MAX, 16, true, false, false),
            Err(OidnError::InvalidArgument(_))
        ));
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let filter = RtFilter::builder(&device, "missing-model-directory")
                .input_scale(Some(scale))
                .build();
            assert!(matches!(
                filter.commit_tensor_model(16, 16, true, false, false),
                Err(OidnError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn auxiliary_primary_transfer_matches_native_modes() {
        let device = Device::ndarray();
        let filter = RtFilter::builder(&device, "").build();
        assert_eq!(filter.transfer_kind(false, false), TransferFunction::SRGB);
        assert_eq!(filter.transfer_kind(false, true), TransferFunction::Linear);
        let filter = RtFilter::builder(&device, "").srgb(true).build();
        assert_eq!(filter.transfer_kind(false, false), TransferFunction::Linear);
    }

    #[test]
    fn fresh_tensor_handles_reuse_parameters_and_shape_changes_rebuild() {
        let device = Device::ndarray();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/weights");
        let mut filter = RtFilter::builder(&device, dir)
            .hdr(true)
            .quality(Quality::Fast)
            .input_scale(Some(1.0))
            .build();
        let id = |filter: &RtFilter<'_>| match filter.net.as_ref().unwrap() {
            Net::Base(model) => model.enc_conv0.weight.id,
            Net::Large(model) => model.enc_conv1a.weight.id,
        };
        filter
            .set_color_tensor(Tensor::full([1, 3, 16, 16], 0.5, &device))
            .unwrap();
        filter.allocate_output_tensor(16, 16).unwrap();
        filter.execute().unwrap();
        let first = id(&filter);
        filter.take_output_tensor().unwrap();
        filter
            .set_color_tensor(Tensor::full([1, 3, 16, 16], 0.6, &device))
            .unwrap();
        filter.allocate_output_tensor(16, 16).unwrap();
        filter.execute().unwrap();
        assert_eq!(
            id(&filter),
            first,
            "same layout must retain model parameters"
        );
        filter
            .set_color_tensor(Tensor::full([1, 3, 16, 32], 0.6, &device))
            .unwrap();
        filter.allocate_output_tensor(32, 16).unwrap();
        filter.execute().unwrap();
        assert_ne!(id(&filter), first, "geometry change must rebuild artifacts");
    }
}
