//! RT filter — public API + glue, equivalent to `_ref/oidn/core/rt_filter.cpp`.
//!
//! Generic over Burn `Backend` so the same filter type can run on CPU
//! (`burn::backend::NdArray`) for tests and on `WgpuBackend` in the CLI.
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
//!   `Tensor<B, 4>` (`[1, 3, H, W]` NCHW). [`Self::allocate_output_tensor`]
//!   declares the output shape; [`Self::take_output_tensor`] returns the
//!   denoised accumulator. Used by squarebob's wgpu bridge to keep pixels
//!   on-device end-to-end.
//!
//! [`Filter::execute`] dispatches between the two paths based on which
//! input slot is populated. Mixing modes within one `commit() / execute()`
//! cycle is not supported — pick one set of setters per call.

use std::path::PathBuf;

use burn::tensor::{Tensor, backend::Backend};
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

pub struct RtFilterBuilder<'b, B: Backend> {
    device: &'b B::Device,
    weights_dir: PathBuf,
    hdr: bool,
    srgb: bool,
    directional: bool,
    clean_aux: bool,
    quality: Quality,
    user_input_scale: Option<f32>,
    user_weights: Option<Vec<u8>>,
    max_memory_mb: Option<i32>,
}

impl<'b, B: Backend> RtFilterBuilder<'b, B> {
    pub fn new(device: &'b B::Device, weights_dir: impl Into<PathBuf>) -> Self {
        Self {
            device,
            weights_dir: weights_dir.into(),
            hdr: false,
            srgb: false,
            directional: false,
            clean_aux: false,
            quality: Quality::High,
            user_input_scale: None,
            user_weights: None,
            max_memory_mb: None,
        }
    }

    pub fn hdr(mut self, v: bool) -> Self { self.hdr = v; self }
    pub fn srgb(mut self, v: bool) -> Self { self.srgb = v; self }
    pub fn directional(mut self, v: bool) -> Self { self.directional = v; self }
    pub fn clean_aux(mut self, v: bool) -> Self { self.clean_aux = v; self }
    pub fn quality(mut self, q: Quality) -> Self { self.quality = q; self }
    pub fn input_scale(mut self, s: Option<f32>) -> Self { self.user_input_scale = s; self }

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

    pub fn build(self) -> RtFilter<'b, B> {
        RtFilter {
            device: self.device,
            weights_dir: self.weights_dir,
            hdr: self.hdr,
            srgb: self.srgb,
            directional: self.directional,
            clean_aux: self.clean_aux,
            quality: self.quality,
            user_input_scale: self.user_input_scale,
            user_weights: self.user_weights,
            max_memory_mb: self.max_memory_mb,
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

pub struct RtFilter<'b, B: Backend> {
    device: &'b B::Device,
    weights_dir: PathBuf,
    hdr: bool,
    srgb: bool,
    directional: bool,
    clean_aux: bool,
    quality: Quality,
    user_input_scale: Option<f32>,
    user_weights: Option<Vec<u8>>,
    max_memory_mb: Option<i32>,

    // --- Legacy Image<'_> path ---
    color: Option<OwnedImage>,
    albedo: Option<OwnedImage>,
    normal: Option<OwnedImage>,
    output: Option<OwnedImageMut>,

    // --- Tensor path (zero host-roundtrip; Phase I.5/I.6) ---
    color_tensor: Option<Tensor<B, 4>>,
    albedo_tensor: Option<Tensor<B, 4>>,
    normal_tensor: Option<Tensor<B, 4>>,
    /// Populated by `execute()` in tensor mode; consumed by
    /// [`Self::take_output_tensor`].
    output_tensor: Option<Tensor<B, 4>>,
    /// `(width, height)` declared by [`Self::allocate_output_tensor`].
    /// Doubles as the tile-plan / shape source when the tensor path is
    /// active.
    output_tensor_dims: Option<(usize, usize)>,

    net: Option<Net<B>>,
    plan: Option<TilePlan>,
    model_key: Option<ModelKey>,
    progress: Option<Box<ProgressFn<'static>>>,
    committed: bool,
    /// Output dims/format from the most recent successful `commit()` in
    /// legacy mode. Tensor mode tracks its own dims via
    /// `output_tensor_dims`; both feed [`Self::output_dims`].
    last_committed_dims: Option<(usize, usize, PixelFormat)>,
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
        Image { data: &self.data, width: self.width, height: self.height,
                row_stride: self.row_stride, format: self.format }
    }
}

impl OwnedImageMut {
    fn empty(width: usize, height: usize, format: PixelFormat) -> Self {
        let row_stride = width * format.pixel_size();
        Self { data: vec![0u8; row_stride * height], width, height, row_stride, format }
    }
    fn view_mut(&mut self) -> ImageMut<'_> {
        ImageMut { data: &mut self.data, width: self.width, height: self.height,
                   row_stride: self.row_stride, format: self.format }
    }
}

impl<'b, B: Backend> RtFilter<'b, B> {
    pub fn builder(device: &'b B::Device, weights_dir: impl Into<PathBuf>) -> RtFilterBuilder<'b, B> {
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
        if needs_invalidate { self.committed = false; }
    }
    pub fn set_albedo(&mut self, img: &Image<'_>) {
        let needs_invalidate = self.albedo.is_none();
        self.albedo = Some(OwnedImage::from(img));
        if needs_invalidate { self.committed = false; }
    }
    pub fn set_normal(&mut self, img: &Image<'_>) {
        let needs_invalidate = self.normal.is_none();
        self.normal = Some(OwnedImage::from(img));
        if needs_invalidate { self.committed = false; }
    }

    // ----- Tensor-native inputs (zero host roundtrip) -----

    /// Tensor-native colour input. Shape `[1, 3, H, W]` (NCHW), `f32`.
    /// The tensor is stored by reference (Burn tensors are cheap to
    /// `clone()`); no data crosses to host. `execute()` runs the
    /// tensor-native pipeline when *any* `set_*_tensor` was used.
    pub fn set_color_tensor(&mut self, t: Tensor<B, 4>) {
        debug_assert_tensor_chw_3::<B>(&t, "set_color_tensor");
        let needs_invalidate =
            self.color_tensor.is_none() || tensor_dims_changed(&self.color_tensor, &t);
        self.color_tensor = Some(t);
        if needs_invalidate { self.committed = false; }
    }

    /// Tensor-native albedo input. See [`Self::set_color_tensor`].
    pub fn set_albedo_tensor(&mut self, t: Tensor<B, 4>) {
        debug_assert_tensor_chw_3::<B>(&t, "set_albedo_tensor");
        let needs_invalidate =
            self.albedo_tensor.is_none() || tensor_dims_changed(&self.albedo_tensor, &t);
        self.albedo_tensor = Some(t);
        if needs_invalidate { self.committed = false; }
    }

    /// Tensor-native normal input. See [`Self::set_color_tensor`].
    pub fn set_normal_tensor(&mut self, t: Tensor<B, 4>) {
        debug_assert_tensor_chw_3::<B>(&t, "set_normal_tensor");
        let needs_invalidate =
            self.normal_tensor.is_none() || tensor_dims_changed(&self.normal_tensor, &t);
        self.normal_tensor = Some(t);
        if needs_invalidate { self.committed = false; }
    }

    /// Take ownership of the denoised output as a `[1, 3, H, W]` (NCHW)
    /// `f32` Burn tensor.
    ///
    /// Returns `None` if `execute()` has not been called yet or the
    /// output slot was already consumed. Re-invoking the filter at the
    /// same shape requires a fresh [`Self::allocate_output_tensor`]
    /// (idempotent — cached model and plan are reused when shape
    /// matches).
    pub fn take_output_tensor(&mut self) -> Option<Tensor<B, 4>> {
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
    pub fn model_key(&self) -> Option<&ModelKey> { self.model_key.as_ref() }

    /// Install a progress callback. Receives `[0.0, 1.0]` after each
    /// processed tile; returning `false` aborts execution with
    /// `OidnError::Cancelled`.
    pub fn set_progress<F: FnMut(f32) -> bool + 'static>(&mut self, callback: F) {
        self.progress = Some(Box::new(callback));
    }

    /// True when at least one tensor input slot is populated. Used by
    /// [`Filter::execute`] to dispatch between the two pipelines.
    fn tensor_mode(&self) -> bool {
        self.color_tensor.is_some()
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
}

fn tensor_dims_changed<B: Backend>(slot: &Option<Tensor<B, 4>>, new: &Tensor<B, 4>) -> bool {
    match slot {
        None => true,
        Some(existing) => existing.dims() != new.dims(),
    }
}

fn debug_assert_tensor_chw_3<B: Backend>(t: &Tensor<B, 4>, who: &'static str) {
    let dims = t.dims();
    debug_assert_eq!(dims[0], 1, "{who}: batch size must be 1, got {:?}", dims);
    debug_assert_eq!(dims[1], 3, "{who}: must be 3-channel CHW, got {:?}", dims);
}

/// Pick the variant (UNet topology) from the resolved model filename stem.
/// `_large` ⇒ Large; `_small` ⇒ Small; otherwise Base.
fn variant_from_stem(stem: &str) -> Variant {
    if stem.ends_with("_large") { Variant::Large }
    else if stem.ends_with("_small") { Variant::Small }
    else { Variant::Base }
}

impl<'b, B: Backend> Filter for RtFilter<'b, B> {
    fn commit(&mut self) -> Result<(), OidnError> {
        let any_input_legacy = self.color.is_some() || self.albedo.is_some() || self.normal.is_some();
        let any_input_tensor = self.tensor_mode();
        if !any_input_legacy && !any_input_tensor {
            return Err(OidnError::Unset("color/albedo/normal"));
        }

        // User-supplied weights override the registry/quality lookup entirely.
        let (stem, bytes): (String, Vec<u8>) = if let Some(bytes) = self.user_weights.clone() {
            ("user".to_string(), bytes)
        } else {
            // Channel-presence flags merge both modes: model selection only
            // cares about which inputs are wired, not which API was used.
            let has_color = self.color.is_some() || self.color_tensor.is_some();
            let has_albedo = self.albedo.is_some() || self.albedo_tensor.is_some();
            let has_normal = self.normal.is_some() || self.normal_tensor.is_some();
            let base_key = select_rt(
                has_color, has_albedo, has_normal,
                self.hdr, self.srgb, self.directional, self.clean_aux,
                self.quality,
            ).ok_or(OidnError::UnsupportedFeatures)?;

            // Quality-based candidates: try _large / _small first, fall back to base.
            let candidates = quality_candidates(&base_key, self.quality);

            let mut chosen: Option<(String, Vec<u8>)> = None;
            let mut last_path: Option<PathBuf> = None;
            for stem in &candidates {
                let p = self.weights_dir.join(format!("{stem}.tza"));
                last_path = Some(p.clone());
                match std::fs::read(&p) {
                    Ok(bytes) => { chosen = Some((stem.clone(), bytes)); break; }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(OidnError::Io(e)),
                }
            }
            chosen.ok_or_else(|| OidnError::MissingModel(
                last_path.unwrap_or_else(|| self.weights_dir.join(base_key.filename()))
            ))?
        };

        self.model_key = Some(ModelKey::new(stem.clone()));
        let tensors = oidn_tza::parse(&bytes)?;

        // Channel count is taken from whichever side (legacy or tensor)
        // is populated. Mixing both modes for the same slot is undefined;
        // we treat any input slot as a present channel triple.
        let has_color = self.color.is_some() || self.color_tensor.is_some();
        let has_albedo = self.albedo.is_some() || self.albedo_tensor.is_some();
        let has_normal = self.normal.is_some() || self.normal_tensor.is_some();
        let in_channels = has_color as usize * 3
            + has_albedo as usize * 3
            + has_normal as usize * 3;
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
                let unet = UNet::<B>::new(in_channels, out_channels, variant, self.device);
                Net::Base(load_tza(unet, &tensors, self.device)?)
            }
            Variant::Large | Variant::XLarge => {
                let unet = UNetLarge::<B>::new(in_channels, out_channels, self.device);
                Net::Large(load_tza_large(unet, &tensors, self.device)?)
            }
        };
        self.net = Some(net);

        let (out_w, out_h) = self.output_dims().ok_or(OidnError::Unset("output"))?;
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
        self.plan = Some(plan);

        // Cross-check that every populated input matches the declared
        // output geometry — applies to both modes.
        let check_dims = |w: usize, h: usize, name: &'static str| -> Result<(), OidnError> {
            if w != out_w || h != out_h { Err(OidnError::Inconsistent(name)) } else { Ok(()) }
        };
        if let Some(c) = &self.color  { check_dims(c.width, c.height, "color")?; }
        if let Some(a) = &self.albedo { check_dims(a.width, a.height, "albedo")?; }
        if let Some(n) = &self.normal { check_dims(n.width, n.height, "normal")?; }
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
        if !self.committed { self.commit()?; }
        let net  = self.net.as_ref().ok_or(OidnError::Unset("model"))?;
        let plan = self.plan.as_ref().ok_or(OidnError::Unset("plan"))?;

        let transfer = if self.directional { TransferFunction::Linear }
                       else if self.hdr     { TransferFunction::PU }
                       else if self.srgb    { TransferFunction::Linear }
                       else                 { TransferFunction::SRGB };

        if self.tensor_mode() {
            let (out_w, out_h) = self.output_tensor_dims.ok_or(OidnError::Unset("output_tensor"))?;
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
                progress,
            )?;
            self.output_tensor = Some(result);
            Ok(())
        } else {
            let output = self.output.as_mut().ok_or(OidnError::Unset("output"))?;
            let color  = self.color.as_ref().map(|i| i.view());
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
                progress,
            )
        }
    }
}
