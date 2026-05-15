//! RT filter — public API + glue, equivalent to `_ref/oidn/core/rt_filter.cpp`.
//!
//! Generic over Burn `Backend` so the same filter type can run on CPU
//! (`burn::backend::NdArray`) for tests and on `WgpuBackend` in the CLI.
//!
//! Supports three network topologies via the `Net` enum dispatcher:
//! base / small `UNet` and the wider `UNetLarge`. Variant is detected from
//! the model filename suffix (`_large` / `_small`) after quality-based
//! candidate selection.

use std::path::PathBuf;

use burn::tensor::{Tensor, backend::Backend};
use oidn_model::{Net, UNet, UNetLarge, Variant, load_tza, load_tza_large};

use crate::{
    color::TransferFunction,
    error::OidnError,
    filter::{Filter, Quality},
    filters::unet_runner::{self, ProgressFn},
    image::{Image, ImageMut, PixelFormat},
    image_tensor,
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

    color: Option<OwnedImage>,
    albedo: Option<OwnedImage>,
    normal: Option<OwnedImage>,
    output: Option<OwnedImageMut>,

    net: Option<Net<B>>,
    plan: Option<TilePlan>,
    model_key: Option<ModelKey>,
    progress: Option<Box<ProgressFn<'static>>>,
    committed: bool,
    /// Output dims/format from the most recent successful `commit()`.
    /// `allocate_output()` reuses the existing `committed` state when the
    /// caller asks for the same shape again — keeps the cached UNet/plan
    /// alive across `take_output()` + re-allocate cycles (e.g. the
    /// `pt-denoise-oidn` periodic-fire loop).
    last_committed_dims: Option<(usize, usize, crate::image::PixelFormat)>,
}

/// Owned copy of an image's bytes plus geometry. Storing borrowed lifetimes
/// across `commit()` / `execute()` becomes invasive; copying once at set time
/// is simpler and the cost is dominated by GPU transfer anyway.
struct OwnedImage {
    data: Vec<u8>,
    width: usize,
    height: usize,
    row_stride: usize,
    format: crate::image::PixelFormat,
}

struct OwnedImageMut {
    data: Vec<u8>,
    width: usize,
    height: usize,
    row_stride: usize,
    format: crate::image::PixelFormat,
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
    fn empty(width: usize, height: usize, format: crate::image::PixelFormat) -> Self {
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

    /// Tensor-native colour input. Shape `[1, 3, H, W]` (NCHW), `f32`.
    ///
    /// Equivalent to [`Self::set_color`] but accepts a Burn tensor
    /// directly. In Phase I.1 the tensor is host-roundtripped (CHW → HWC
    /// bytes) into the existing `OwnedImage` storage so the legacy CPU
    /// `unet_runner` path stays byte-for-byte identical. Sub-tasks I.2 and
    /// I.4 lift the roundtrip onto Burn ops; this signature is stable.
    pub fn set_color_tensor(&mut self, t: Tensor<B, 4>) {
        self.color = Some(tensor_to_owned_image(t));
        // Same invalidation rule as `set_color`: only force a re-commit
        // when an input slot transitions from None → Some. Re-uploading the
        // same shape every frame must not trash the cached UNet/plan.
        if self.color.as_ref().map(|i| (i.width, i.height)) != self.last_committed_dims.map(|(w, h, _)| (w, h)) {
            self.committed = false;
        }
    }

    /// Tensor-native albedo input. See [`Self::set_color_tensor`].
    pub fn set_albedo_tensor(&mut self, t: Tensor<B, 4>) {
        let needs_invalidate = self.albedo.is_none();
        self.albedo = Some(tensor_to_owned_image(t));
        if needs_invalidate { self.committed = false; }
    }

    /// Tensor-native normal input. See [`Self::set_color_tensor`].
    pub fn set_normal_tensor(&mut self, t: Tensor<B, 4>) {
        let needs_invalidate = self.normal.is_none();
        self.normal = Some(tensor_to_owned_image(t));
        if needs_invalidate { self.committed = false; }
    }

    /// Take ownership of the denoised output as a `[1, 3, H, W]` (NCHW)
    /// `f32` Burn tensor.
    ///
    /// Returns `None` if `execute()` has not been called yet or the output
    /// slot was already consumed. The internal output storage is cleared
    /// after this call — call [`Self::allocate_output`] again before the
    /// next denoise.
    ///
    /// In Phase I.1 this reads the legacy HWC byte buffer and uploads it
    /// to `device` via [`image_tensor::chw_vec_to_tensor`]. I.4 + I.5 will
    /// short-circuit the host path entirely.
    pub fn take_output_tensor(&mut self) -> Option<Tensor<B, 4>> {
        let o = self.output.take()?;
        let w = o.width;
        let h = o.height;
        // Decode bytes (any supported pixel format) into HWC f32 via the
        // existing `Image::to_rgb_f32` helper. Format coercion (f16, 1/2
        // channel broadcast) is handled there.
        let img = Image {
            data: &o.data,
            width: w,
            height: h,
            row_stride: o.row_stride,
            format: o.format,
        };
        let hwc = img.to_rgb_f32();
        let chw = image_tensor::hwc_to_chw(&hwc, 3, h, w);
        Some(image_tensor::chw_vec_to_tensor::<B>(chw, 3, h, w, self.device))
    }

    pub fn allocate_output(&mut self, width: usize, height: usize, format: crate::image::PixelFormat) {
        // Skip `committed = false` when the requested output dims/format
        // match the previously-committed ones. `take_output()` leaves
        // `self.output = None` even when the renderer wants to denoise
        // again at the same dims — without this check we'd rebuild the
        // UNet and tile plan every single call.
        let same_dims =
            self.last_committed_dims == Some((width, height, format));
        self.output = Some(OwnedImageMut::empty(width, height, format));
        if !same_dims {
            self.committed = false;
        }
    }

    pub fn take_output(&mut self) -> Option<(Vec<u8>, usize, usize, crate::image::PixelFormat)> {
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
}

/// Materialise a Burn tensor (`[1, C, H, W]`, CHW) as an `OwnedImage`
/// holding HWC f32 bytes. Used by the tensor-native input setters in
/// Phase I.1; will be removed once I.2/I.5 eliminate the host roundtrip.
fn tensor_to_owned_image<B: Backend>(t: Tensor<B, 4>) -> OwnedImage {
    let (chw, dims) = image_tensor::tensor_to_chw_vec(t);
    let c = dims[1];
    let h = dims[2];
    let w = dims[3];
    debug_assert_eq!(dims[0], 1, "set_*_tensor expects batch size 1");
    debug_assert_eq!(c, 3, "set_*_tensor expects 3 channels (broadcast 1/2ch upstream)");
    let hwc = image_tensor::chw_to_hwc(&chw, c, h, w);
    let bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&hwc).to_vec();
    OwnedImage {
        data: bytes,
        width: w,
        height: h,
        row_stride: w * PixelFormat::Rgb32f.pixel_size(),
        format: PixelFormat::Rgb32f,
    }
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
        let any_input = self.color.is_some() || self.albedo.is_some() || self.normal.is_some();
        if !any_input { return Err(OidnError::Unset("color/albedo/normal")); }

        // User-supplied weights override the registry/quality lookup entirely.
        let (stem, bytes): (String, Vec<u8>) = if let Some(bytes) = self.user_weights.clone() {
            ("user".to_string(), bytes)
        } else {
            let base_key = select_rt(
                self.color.is_some(),
                self.albedo.is_some(),
                self.normal.is_some(),
                self.hdr,
                self.srgb,
                self.directional,
                self.clean_aux,
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

        let in_channels = self.color.is_some() as usize * 3
            + self.albedo.is_some() as usize * 3
            + self.normal.is_some() as usize * 3;
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

        let out = self.output.as_ref().ok_or(OidnError::Unset("output"))?;
        let max_tile_pixels = match self.max_memory_mb {
            None => DEFAULT_MAX_TILE_SIZE,
            Some(mb) => {
                // Rough estimate of the dominant activation size, in bytes per
                // pixel of tile area. UNet base ec5 = 96 ch; UNet large ec5 =
                // 256 ch. Multiply by 4 (f32) and a safety factor of 4 to
                // account for double-buffered scratch + skip-connection
                // pool tensors held during decoder upsamples.
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
            out.width as i32,
            out.height as i32,
            RECEPTIVE_FIELD_BASE,
            MIN_TILE_ALIGNMENT,
            max_tile_pixels,
        );
        self.plan = Some(plan);

        let check_dims = |w: usize, h: usize, name: &'static str| -> Result<(), OidnError> {
            if w != out.width || h != out.height { Err(OidnError::Inconsistent(name)) } else { Ok(()) }
        };
        if let Some(c) = &self.color  { check_dims(c.width, c.height, "color")?; }
        if let Some(a) = &self.albedo { check_dims(a.width, a.height, "albedo")?; }
        if let Some(n) = &self.normal { check_dims(n.width, n.height, "normal")?; }

        self.committed = true;
        self.last_committed_dims = Some((out.width, out.height, out.format));
        Ok(())
    }

    fn execute(&mut self) -> Result<(), OidnError> {
        if !self.committed { self.commit()?; }
        let net   = self.net.as_ref().ok_or(OidnError::Unset("model"))?;
        let plan  = self.plan.as_ref().ok_or(OidnError::Unset("plan"))?;
        let output = self.output.as_mut().ok_or(OidnError::Unset("output"))?;

        let transfer = if self.directional { TransferFunction::Linear }
                       else if self.hdr     { TransferFunction::PU }
                       else if self.srgb    { TransferFunction::Linear }
                       else                 { TransferFunction::SRGB };

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
