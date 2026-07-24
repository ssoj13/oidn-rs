//! RT filter — public API + glue, equivalent to `_ref/oidn/core/rt_filter.cpp`.
//!
//! In burn 0.22 the backend is dynamic (device-selected), so the same filter
//! type runs on CPU (`Device::ndarray()`) for tests and on wgpu
//! (`Device::wgpu(..)`) in the CLI — the choice is the device, not a type
//! parameter.
//!
//! Supports three network topologies via the `Net` enum dispatcher:
//! base / small `UNet` and the wider `UNetLarge`. Variant is detected from
//! the model filename suffix (`_large` / `_small`) after quality-based
//! candidate selection.
//!
//! ## Two parallel I/O modes
//!
//! - **Legacy `Image<'_>` mode.** [`Self::set_color`] / [`Self::set_albedo`]
//!   / [`Self::set_normal`] take byte-backed images; [`Self::allocate_output`]
//!   reserves a host-side buffer; [`Self::take_output`] returns those bytes.
//!   Used by the CLI and the test fixtures.
//! - **Tensor mode.** [`Self::set_color_tensor`] etc. take a Burn
//!   `Tensor<4>` (`[1, 3, H, W]` NCHW). [`Self::allocate_output_tensor`]
//!   declares the output shape; [`Self::take_output_tensor`] returns the
//!   denoised accumulator. Used by squarebob's wgpu bridge to keep pixels
//!   on-device end-to-end.
//!
//! [`Filter::execute`] dispatches between the two paths based on which
//! input slot is populated. Mixing modes within one `commit() / execute()`
//! cycle is not supported — pick one set of setters per call.

use std::path::PathBuf;

use burn::tensor::{Device, Tensor};
use oidn_model::{Net, UNet, UNetLarge, Variant, load_tza, load_tza_large};

use crate::{
    color::TransferFunction,
    error::OidnError,
    filter::{Filter, Quality},
    filters::unet_runner::{self, ProgressFn},
    image::{Image, ImageMut, PixelFormat},
    registry::{ModelKey, quality_candidates, select_rt},
    tile::{self, DEFAULT_MAX_TILE_SIZE, MIN_TILE_ALIGNMENT, RECEPTIVE_FIELD_BASE, TilePlan},
};

pub struct RtFilterBuilder<'b> {
    device: &'b Device,
    weights_dir: PathBuf,
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
        Self {
            device,
            weights_dir: weights_dir.into(),
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
    /// `weights_dir`. Bypasses [`crate::registry::select_rt`] completely —
    /// callers are responsible for matching the blob's channel counts to
    /// the input set. Variant (`UNet` vs `UNetLarge`) is auto-detected from
    /// tensor names.
    pub fn weights(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.user_weights = Some(bytes.into());
        self
    }

    /// Memory budget in MB. The tile planner will subdivide the image until
    /// the largest intermediate tensor fits below this budget. Pass `-1` (or
    /// don't call this) to use the default `DEFAULT_MAX_TILE_SIZE` cap.
    pub fn max_memory_mb(mut self, mb: i32) -> Self {
        self.max_memory_mb = (mb >= 0).then_some(mb);
        self
    }

    /// Enable replacement of non-finite (`NaN` / ±`Inf`) input samples
    /// with `0` before clamp / transfer. Mirrors the reference C++
    /// OIDN kernel contract (`nan_to_zero` at the head of every
    /// `getInput` / `getAlbedo` / `getNormal` body). Default: `true`
    /// — strongly recommended; disabling it lets bad path-tracer
    /// samples poison the entire output through PU/exp expansion.
    pub fn nan_to_zero(mut self, v: bool) -> Self {
        self.nan_to_zero = v;
        self
    }

    pub fn build(self) -> RtFilter<'b> {
        RtFilter {
            device: self.device,
            weights_dir: self.weights_dir,
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
        }
    }
}

pub struct RtFilter<'b> {
    device: &'b Device,
    weights_dir: PathBuf,
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
    output: Option<OwnedImageMut>,

    // --- Tensor path (zero host-roundtrip; Phase I.5/I.6) ---
    color_tensor: Option<Tensor<4>>,
    albedo_tensor: Option<Tensor<4>>,
    normal_tensor: Option<Tensor<4>>,
    /// Populated by `execute()` in tensor mode; consumed by
    /// [`Self::take_output_tensor`].
    output_tensor: Option<Tensor<4>>,
    /// `(width, height)` declared by [`Self::allocate_output_tensor`].
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

/// Owned copy of an image's bytes plus geometry. Storing borrowed lifetimes
/// across `commit()` / `execute()` becomes invasive; copying once at set time
/// is simpler and the cost is dominated by GPU transfer anyway.
struct OwnedImage {
    data: Vec<u8>,
    width: usize,
    height: usize,
    row_stride: usize,
    format: PixelFormat,
}

struct OwnedImageMut {
    data: Vec<u8>,
    width: usize,
    height: usize,
    row_stride: usize,
    format: PixelFormat,
}

impl OwnedImage {
    fn from(img: &Image<'_>) -> Self {
        Self {
            data: img.data.to_vec(),
            width: img.width,
            height: img.height,
            row_stride: img.row_stride,
            format: img.format,
        }
    }
    fn view(&self) -> Image<'_> {
        Image {
            data: &self.data,
            width: self.width,
            height: self.height,
            row_stride: self.row_stride,
            format: self.format,
        }
    }
}

impl OwnedImageMut {
    fn empty(width: usize, height: usize, format: PixelFormat) -> Self {
        let row_stride = width * format.pixel_size();
        Self {
            data: vec![0u8; row_stride * height],
            width,
            height,
            row_stride,
            format,
        }
    }
    fn view_mut(&mut self) -> ImageMut<'_> {
        ImageMut {
            data: &mut self.data,
            width: self.width,
            height: self.height,
            row_stride: self.row_stride,
            format: self.format,
        }
    }
}

impl<'b> RtFilter<'b> {
    pub fn builder(
        device: &'b Device,
        weights_dir: impl Into<PathBuf>,
    ) -> RtFilterBuilder<'b> {
        RtFilterBuilder::new(device, weights_dir)
    }

    // ----- Legacy Image-based inputs -----

    /// Replace the color image. Note: this does *not* invalidate the
    /// committed model/plan — when only pixel content changes (same
    /// dimensions, same input set), `execute()` reuses the cached UNet
    /// and tile plan. Only mode/quality/dims changes need a fresh
    /// `commit()`.
    pub fn set_color(&mut self, img: &Image<'_>) {
        let needs_invalidate = self.color.is_none();
        self.color = Some(OwnedImage::from(img));
        if needs_invalidate {
            self.committed = false;
        }
    }
    pub fn set_albedo(&mut self, img: &Image<'_>) {
        let needs_invalidate = self.albedo.is_none();
        self.albedo = Some(OwnedImage::from(img));
        if needs_invalidate {
            self.committed = false;
        }
    }
    pub fn set_normal(&mut self, img: &Image<'_>) {
        let needs_invalidate = self.normal.is_none();
        self.normal = Some(OwnedImage::from(img));
        if needs_invalidate {
            self.committed = false;
        }
    }

    // ----- Tensor-native inputs (zero host roundtrip) -----

    /// Tensor-native colour input. Shape `[1, 3, H, W]` (NCHW), `f32`.
    /// The tensor is stored by reference (Burn tensors are cheap to
    /// `clone()`); no data crosses to host. `execute()` runs the
    /// tensor-native pipeline when *any* `set_*_tensor` was used.
    pub fn set_color_tensor(&mut self, t: Tensor<4>) {
        debug_assert_tensor_chw_3(&t, "set_color_tensor");
        let needs_invalidate =
            self.color_tensor.is_none() || tensor_dims_changed(&self.color_tensor, &t);
        self.color_tensor = Some(t);
        if needs_invalidate {
            self.committed = false;
        }
    }

    /// Tensor-native albedo input. See [`Self::set_color_tensor`].
    pub fn set_albedo_tensor(&mut self, t: Tensor<4>) {
        debug_assert_tensor_chw_3(&t, "set_albedo_tensor");
        let needs_invalidate =
            self.albedo_tensor.is_none() || tensor_dims_changed(&self.albedo_tensor, &t);
        self.albedo_tensor = Some(t);
        if needs_invalidate {
            self.committed = false;
        }
    }

    /// Tensor-native normal input. See [`Self::set_color_tensor`].
    pub fn set_normal_tensor(&mut self, t: Tensor<4>) {
        debug_assert_tensor_chw_3(&t, "set_normal_tensor");
        let needs_invalidate =
            self.normal_tensor.is_none() || tensor_dims_changed(&self.normal_tensor, &t);
        self.normal_tensor = Some(t);
        if needs_invalidate {
            self.committed = false;
        }
    }

    /// Take ownership of the denoised output as a `[1, 3, H, W]` (NCHW)
    /// `f32` Burn tensor.
    ///
    /// Returns `None` if `execute()` has not been called yet or the
    /// output slot was already consumed. Re-invoking the filter at the
    /// same shape requires a fresh [`Self::allocate_output_tensor`]
    /// (idempotent — cached model and plan are reused when shape
    /// matches).
    pub fn take_output_tensor(&mut self) -> Option<Tensor<4>> {
        self.output_tensor.take()
    }

    // ----- Output allocation (legacy + tensor) -----

    pub fn allocate_output(&mut self, width: usize, height: usize, format: PixelFormat) {
        // Skip `committed = false` when the requested output dims/format
        // match the previously-committed ones. `take_output()` leaves
        // `self.output = None` even when the renderer wants to denoise
        // again at the same dims — without this check we'd rebuild the
        // UNet and tile plan every single call.
        let same_dims = self.last_committed_dims == Some((width, height, format));
        self.output = Some(OwnedImageMut::empty(width, height, format));
        // Clear any tensor-mode shape so the dispatcher picks the legacy
        // path next time.
        self.output_tensor_dims = None;
        if !same_dims {
            self.committed = false;
        }
    }

    /// Declare the output shape for tensor-mode execution. No tensor is
    /// allocated up-front — `execute()` builds the accumulator with
    /// `Tensor::zeros([1, 3, h, w], device)` and hands it back via
    /// [`Self::take_output_tensor`].
    pub fn allocate_output_tensor(&mut self, width: usize, height: usize) {
        let same_dims = self.output_tensor_dims == Some((width, height));
        self.output_tensor_dims = Some((width, height));
        // Drop the legacy buffer; we're going tensor.
        self.output = None;
        if !same_dims {
            self.committed = false;
        }
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
            transfer: self.transfer_kind(has_color),
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
        self.color_tensor.is_some() || self.albedo_tensor.is_some() || self.normal_tensor.is_some()
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
    /// on both mode flags and which inputs are present. When no color image is
    /// supplied (albedo-only or normal-only filtering), the transfer is always
    /// `Linear` regardless of `hdr`/`srgb`.
    fn transfer_kind(&self, has_color: bool) -> TransferFunction {
        if !has_color {
            return TransferFunction::Linear;
        }
        if self.hdr {
            TransferFunction::PU
        } else if self.srgb {
            TransferFunction::Linear
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
        // User-supplied weights override the registry/quality lookup entirely.
        let (stem, bytes): (String, Vec<u8>) = if let Some(bytes) = self.user_weights.clone() {
            ("user".to_string(), bytes)
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

            // Quality-based candidates: try _large / _small first, fall back to base.
            let candidates = quality_candidates(&base_key, self.quality);

            let mut chosen: Option<(String, Vec<u8>)> = None;
            let mut last_path: Option<PathBuf> = None;
            for stem in &candidates {
                let p = self.weights_dir.join(format!("{stem}.tza"));
                last_path = Some(p.clone());
                match std::fs::read(&p) {
                    Ok(bytes) => {
                        chosen = Some((stem.clone(), bytes));
                        break;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(OidnError::Io(e)),
                }
            }
            chosen.ok_or_else(|| {
                OidnError::MissingModel(
                    last_path.unwrap_or_else(|| self.weights_dir.join(base_key.filename())),
                )
            })?
        };

        let tensors = oidn_tza::parse(&bytes)?;
        let in_channels =
            has_color as usize * 3 + has_albedo as usize * 3 + has_normal as usize * 3;
        let out_channels = 3;

        // Variant detection: when we're loading user weights, we can't trust
        // the filename, so consult tensor names directly. Otherwise the
        // resolved stem suffix is authoritative.
        let variant = if self.user_weights.is_some() {
            Variant::from_tensor_names(tensors.keys())
        } else {
            variant_from_stem(&stem)
        };
        let net = match variant {
            Variant::Base | Variant::Small => {
                let unet = UNet::new(in_channels, out_channels, variant, self.device);
                Net::Base(load_tza(unet, &tensors, self.device)?)
            }
            Variant::Large | Variant::XLarge => {
                let unet = UNetLarge::new(in_channels, out_channels, self.device);
                Net::Large(load_tza_large(unet, &tensors, self.device)?)
            }
        };

        let max_tile_pixels = match self.max_memory_mb {
            None => DEFAULT_MAX_TILE_SIZE,
            Some(mb) => {
                let bytes_per_pixel: i64 = match variant {
                    Variant::Large | Variant::XLarge => 256 * 4 * 4,
                    _ => 96 * 4 * 4,
                };
                let budget_bytes = (mb as i64) * 1024 * 1024;
                let cap = (budget_bytes / bytes_per_pixel).clamp(1, i32::MAX as i64) as i32;
                cap.min(DEFAULT_MAX_TILE_SIZE)
            }
        };
        let plan = tile::plan(
            out_w as i32,
            out_h as i32,
            RECEPTIVE_FIELD_BASE,
            MIN_TILE_ALIGNMENT,
            max_tile_pixels,
        );

        Ok(RtCommitArtifacts {
            net,
            plan,
            model_key: ModelKey::new(stem),
        })
    }
}

fn tensor_dims_changed(slot: &Option<Tensor<4>>, new: &Tensor<4>) -> bool {
    match slot {
        None => true,
        Some(existing) => existing.dims() != new.dims(),
    }
}

fn debug_assert_tensor_chw_3(t: &Tensor<4>, who: &'static str) {
    let dims = t.dims();
    debug_assert_eq!(dims[0], 1, "{who}: batch size must be 1, got {:?}", dims);
    debug_assert_eq!(dims[1], 3, "{who}: must be 3-channel CHW, got {:?}", dims);
}

/// Pick the variant (UNet topology) from the resolved model filename stem.
/// `_large` ⇒ Large; `_small` ⇒ Small; otherwise Base.
fn variant_from_stem(stem: &str) -> Variant {
    if stem.ends_with("_large") {
        Variant::Large
    } else if stem.ends_with("_small") {
        Variant::Small
    } else {
        Variant::Base
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
            self.transfer,
            // NB: positional — see run_tensors signature.
            //   hdr, user_input_scale, nan_to_zero, progress
            self.hdr,
            self.user_input_scale,
            self.nan_to_zero,
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
            debug_assert_tensor_chw_3(t, name);
            let d = t.dims();
            if d[3] == width && d[2] == height {
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
        let any_input_legacy =
            self.color.is_some() || self.albedo.is_some() || self.normal.is_some();
        let any_input_tensor = self.tensor_mode();
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
        let artifacts =
            self.build_commit_artifacts(out_w, out_h, has_color, has_albedo, has_normal)?;
        self.model_key = Some(artifacts.model_key);
        self.net = Some(artifacts.net);
        self.plan = Some(artifacts.plan);

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
            check_dims(c.width, c.height, "color")?;
        }
        if let Some(a) = &self.albedo {
            check_dims(a.width, a.height, "albedo")?;
        }
        if let Some(n) = &self.normal {
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

        let has_color = self.color.is_some() || self.color_tensor.is_some();
        let transfer = self.transfer_kind(has_color);

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
                transfer,
                self.hdr,
                self.user_input_scale,
                self.nan_to_zero,
                progress,
            )?;
            self.output_tensor = Some(result);
            // Release input tensor handles now that `run_tensors` has
            // consumed (cloned) them. Keeping them on `self` across
            // calls forces the caller's buffers to live until the next
            // `set_*_tensor` reassignment. With CubeCL's lazy kernel
            // submission, that retention overlaps the caller's buffer
            // pool with our UNet reads — a classic read-after-free if
            // the caller recycles the buffer before the GPU drains.
            // Dropping refs here lets the caller (and the pool) treat
            // the input buffers as "owned only as long as execute()
            // ran" — the safe contract.
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
                transfer,
                self.hdr,
                self.user_input_scale,
                self.nan_to_zero,
                progress,
            )
        }
    }
}
