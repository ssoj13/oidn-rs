//! Lightmap filter — port of `_ref/oidn/core/rtlightmap_filter.cpp`.
//!
//! Two modes:
//! - HDR (default): Log transfer (`color.h::TransferFunction::Type::Log`),
//!   network = `rtlightmap_hdr`.
//! - Directional: Linear transfer with snorm normalisation (input values can
//!   be negative), network = `rtlightmap_dir`. Used to denoise per-direction
//!   irradiance gradients.

use std::path::PathBuf;

use burn::tensor::Device;
use oidn_model::{Net, UNet, Variant, load_tza};

use crate::{
    color::TransferFunction,
    error::OidnError,
    filter::{Filter, Quality},
    filters::unet_runner::{self, ProgressFn},
    image::{Image, ImageMut, PixelFormat},
    registry::ModelKey,
    tile::{self, DEFAULT_MAX_TILE_SIZE, MIN_TILE_ALIGNMENT, RECEPTIVE_FIELD_BASE, TilePlan},
};

pub struct RtLightmapFilterBuilder<'b> {
    device: &'b Device,
    weights_dir: PathBuf,
    directional: bool,
    quality: Quality,
    user_input_scale: Option<f32>,
    user_weights: Option<Vec<u8>>,
}

impl<'b> RtLightmapFilterBuilder<'b> {
    pub fn new(device: &'b Device, weights_dir: impl Into<PathBuf>) -> Self {
        Self {
            device,
            weights_dir: weights_dir.into(),
            directional: false,
            quality: Quality::High,
            user_input_scale: None,
            user_weights: None,
        }
    }

    /// In directional mode the lightmap stores signed per-axis irradiance
    /// gradients; we use Linear transfer + snorm input handling instead of
    /// Log (matches `RTLightmapFilter::setInt("directional", ...)` semantics).
    pub fn directional(mut self, v: bool) -> Self {
        self.directional = v;
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

    /// Use the caller-supplied TZA blob instead of looking up
    /// `rtlightmap_hdr.tza` / `rtlightmap_dir.tza` in `weights_dir`. Mirrors
    /// [`crate::filters::rt::RtFilterBuilder::weights`] — bypasses the
    /// registry entirely, so callers must ensure the blob matches the
    /// chosen mode (HDR vs directional).
    pub fn weights(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.user_weights = Some(bytes.into());
        self
    }

    pub fn build(self) -> RtLightmapFilter<'b> {
        RtLightmapFilter {
            device: self.device,
            weights_dir: self.weights_dir,
            directional: self.directional,
            quality: self.quality,
            user_input_scale: self.user_input_scale,
            user_weights: self.user_weights,
            color: None,
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

pub struct RtLightmapFilter<'b> {
    device: &'b Device,
    weights_dir: PathBuf,
    directional: bool,
    quality: Quality,
    user_input_scale: Option<f32>,
    user_weights: Option<Vec<u8>>,

    color: Option<OwnedImage>,
    output: Option<OwnedImageMut>,

    net: Option<Net>,
    plan: Option<TilePlan>,
    model_key: Option<ModelKey>,
    progress: Option<Box<ProgressFn<'static>>>,
    committed: bool,
    /// Output (w, h, format) the last `commit()` validated against. Used by
    /// [`Self::allocate_output`] to preserve the cached UNet + tile plan
    /// when the renderer re-uses the same shape across frames. Mirrors the
    /// equivalent path on [`crate::filters::rt::RtFilter`].
    last_committed_dims: Option<(usize, usize, PixelFormat)>,
}

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

impl<'b> RtLightmapFilter<'b> {
    pub fn builder(
        device: &'b Device,
        weights_dir: impl Into<PathBuf>,
    ) -> RtLightmapFilterBuilder<'b> {
        RtLightmapFilterBuilder::new(device, weights_dir)
    }

    pub fn set_color(&mut self, img: &Image<'_>) {
        let needs_invalidate = self.color.is_none();
        self.color = Some(OwnedImage::from(img));
        if needs_invalidate {
            self.committed = false;
        }
    }

    /// Reserve a host-side output buffer at the requested shape. Identical
    /// shape + format as the previous commit leaves `committed` intact so
    /// the UNet weights and tile plan are reused across frames; only a
    /// genuine shape change forces a rebuild. Mirrors `RtFilter::allocate_output`.
    pub fn allocate_output(&mut self, width: usize, height: usize, format: PixelFormat) {
        let same_dims = self.last_committed_dims == Some((width, height, format));
        self.output = Some(OwnedImageMut::empty(width, height, format));
        if !same_dims {
            self.committed = false;
        }
    }

    pub fn take_output(&mut self) -> Option<(Vec<u8>, usize, usize, PixelFormat)> {
        let o = self.output.take()?;
        Some((o.data, o.width, o.height, o.format))
    }

    pub fn model_key(&self) -> Option<&ModelKey> {
        self.model_key.as_ref()
    }

    /// Install a progress callback. Receives `[0.0, 1.0]` after each
    /// processed tile; returning `false` aborts execution with
    /// `OidnError::Cancelled`. Mirrors `RtFilter::set_progress`.
    pub fn set_progress<F: FnMut(f32) -> bool + 'static>(&mut self, callback: F) {
        self.progress = Some(Box::new(callback));
    }

    fn select_model(&self) -> ModelKey {
        // rtlightmap_filter.cpp:19-20 — directional → rtlightmap_dir, otherwise rtlightmap_hdr.
        if self.directional {
            ModelKey::new("rtlightmap_dir")
        } else {
            ModelKey::new("rtlightmap_hdr")
        }
    }
}

impl<'b> Filter for RtLightmapFilter<'b> {
    fn set_progress(&mut self, cb: Box<dyn FnMut(f32) -> bool + 'static>) -> Result<(), OidnError> {
        // The inherent `set_progress` boxes any closure; here we already
        // have a boxed dyn — store it directly to avoid re-boxing.
        self.progress = Some(cb);
        Ok(())
    }

    fn commit(&mut self) -> Result<(), OidnError> {
        if self.color.is_none() {
            return Err(OidnError::Unset("color"));
        }

        let key = self.select_model();
        let _ = self.quality; // single-variant filter — no quality routing

        // User weights override the registry/file lookup, matching
        // `RtFilter`'s contract.
        let bytes: Vec<u8> = if let Some(user) = self.user_weights.clone() {
            user
        } else {
            let path = self.weights_dir.join(key.filename());
            std::fs::read(&path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    OidnError::MissingModel(path.clone())
                } else {
                    OidnError::Io(e)
                }
            })?
        };
        self.model_key = Some(key);
        let tensors = oidn_tza::parse(&bytes)?;

        // Lightmap models always take 3 colour channels → 3 channels out.
        let unet = UNet::new(3, 3, Variant::Base, self.device);
        let unet = load_tza(unet, &tensors, self.device)?;
        self.net = Some(Net::Base(unet));

        let out = self.output.as_ref().ok_or(OidnError::Unset("output"))?;
        let plan = tile::plan(
            out.width as i32,
            out.height as i32,
            RECEPTIVE_FIELD_BASE,
            MIN_TILE_ALIGNMENT,
            DEFAULT_MAX_TILE_SIZE,
        );
        self.plan = Some(plan);

        let color = self.color.as_ref().unwrap();
        if color.width != out.width || color.height != out.height {
            return Err(OidnError::Inconsistent("color"));
        }

        self.committed = true;
        self.last_committed_dims = Some((out.width, out.height, out.format));
        Ok(())
    }

    fn execute(&mut self) -> Result<(), OidnError> {
        if !self.committed {
            self.commit()?;
        }
        let net = self.net.as_ref().ok_or(OidnError::Unset("model"))?;
        let plan = self.plan.as_ref().ok_or(OidnError::Unset("plan"))?;
        let output = self.output.as_mut().ok_or(OidnError::Unset("output"))?;

        // rtlightmap_filter.cpp:24-30 — HDR uses Log, directional uses Linear.
        // Directional is also treated as snorm (signed input range).
        let (transfer, is_hdr) = if self.directional {
            (TransferFunction::Linear, false)
        } else {
            (TransferFunction::Log, true)
        };

        let color = self.color.as_ref().map(|i| i.view());

        let mut out_view = output.view_mut();
        let progress: Option<&mut ProgressFn<'_>> = self.progress.as_deref_mut();
        unet_runner::run(
            net,
            self.device,
            plan,
            color.as_ref(),
            None,
            None,
            &mut out_view,
            transfer,
            is_hdr,
            self.user_input_scale,
            true, // nan_to_zero: match reference contract by default
            progress,
        )
    }
}
