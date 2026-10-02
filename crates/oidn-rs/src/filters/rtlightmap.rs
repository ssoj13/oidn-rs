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
use oidn_model::Net;

use crate::{
    color::TransferFunction,
    error::OidnError,
    filter::{Filter, Quality},
    filters::unet_runner::{self, ProgressFn, RunOptions},
    image::{Image, OwnedImage, PixelFormat},
    registry::ModelKey,
    tile::TilePlan,
    weights::{self, SourcePolicy},
};

pub struct RtLightmapFilterBuilder<'b> {
    device: &'b Device,
    weights_dir: PathBuf,
    weight_source: SourcePolicy,
    max_memory_mb: Option<i32>,
    directional: bool,
    quality: Quality,
    user_input_scale: Option<f32>,
    user_weights: Option<Vec<u8>>,
    nan_to_zero: bool,
}

impl<'b> RtLightmapFilterBuilder<'b> {
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
            max_memory_mb: None,
            directional: false,
            quality: Quality::High,
            user_input_scale: None,
            user_weights: None,
            nan_to_zero: true,
        }
    }

    /// In directional mode the lightmap stores signed per-axis irradiance
    /// gradients; we use Linear transfer + snorm input handling instead of
    /// Log (matches `RTLightmapFilter::setInt("directional", ...)` semantics).
    /// Choose source precedence for the single lightmap model.
    pub fn weight_source(mut self, policy: SourcePolicy) -> Self {
        self.weight_source = policy;
        self
    }
    /// Best-effort memory budget in MiB; negative values restore the default.
    pub fn max_memory_mb(mut self, mb: i32) -> Self {
        self.max_memory_mb = (mb >= 0).then_some(mb);
        self
    }

    pub fn directional(mut self, v: bool) -> Self {
        self.directional = v;
        self
    }
    pub fn quality(mut self, q: Quality) -> Self {
        self.quality = q;
        self
    }
    /// Optional stronger policy replacing all nonfinite samples before scaling.
    /// When disabled, mandatory native NaN-only sanitation/clamping remains.
    pub fn nan_to_zero(mut self, enabled: bool) -> Self {
        self.nan_to_zero = enabled;
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
            weight_source: self.weight_source,
            max_memory_mb: self.max_memory_mb,
            directional: self.directional,
            quality: self.quality,
            user_input_scale: self.user_input_scale,
            user_weights: self.user_weights,
            nan_to_zero: self.nan_to_zero,
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
    weight_source: SourcePolicy,
    max_memory_mb: Option<i32>,
    directional: bool,
    quality: Quality,
    user_input_scale: Option<f32>,
    user_weights: Option<Vec<u8>>,
    nan_to_zero: bool,

    color: Option<OwnedImage>,
    output: Option<OwnedImage>,

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

impl<'b> RtLightmapFilter<'b> {
    pub fn builder(
        device: &'b Device,
        weights_dir: impl Into<PathBuf>,
    ) -> RtLightmapFilterBuilder<'b> {
        RtLightmapFilterBuilder::new(device, weights_dir)
    }

    pub fn set_color(&mut self, img: &Image<'_>) -> Result<(), OidnError> {
        let image = OwnedImage::from(img)?;
        let changed = self.color.as_ref().is_none_or(|old| {
            (old.width, old.height, old.format) != (img.width, img.height, img.format)
        });
        self.color = Some(image);
        if changed {
            self.committed = false;
        }
        Ok(())
    }

    /// Reserve a host-side output buffer at the requested shape. Identical
    /// shape + format as the previous commit leaves `committed` intact so
    /// the UNet weights and tile plan are reused across frames; only a
    /// genuine shape change forces a rebuild. Mirrors `RtFilter::allocate_output`.
    pub fn allocate_output(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<(), OidnError> {
        let same_dims = self.last_committed_dims == Some((width, height, format));
        self.output = Some(OwnedImage::empty(width, height, format)?);
        if !same_dims {
            self.committed = false;
        }
        Ok(())
    }

    pub fn take_output(&mut self) -> Option<(Vec<u8>, usize, usize, PixelFormat)> {
        let o = self.output.take()?;
        Some((o.data, o.width, o.height, o.format))
    }

    /// Update the optional stronger sanitation policy between passes.
    pub fn set_nan_to_zero(&mut self, enabled: bool) {
        self.nan_to_zero = enabled;
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
        self.committed = false;
        if self.color.is_none() {
            return Err(OidnError::Unset("color"));
        }

        let key = self.select_model();
        let _ = self.quality; // single-variant filter — no quality routing

        let out = self.output.as_ref().ok_or(OidnError::Unset("output"))?;
        out.view().validate()?;
        super::validate_execution(out.width, out.height, self.user_input_scale)?;
        let color = self.color.as_ref().ok_or(OidnError::Unset("color"))?;
        color.view().validate()?;
        if color.width != out.width || color.height != out.height {
            return Err(OidnError::Inconsistent("color"));
        }
        let bytes = if let Some(user) = &self.user_weights {
            user.clone()
        } else {
            // Native lightmap selection has one model regardless of quality.
            weights::resolve(
                &key,
                Quality::Balanced,
                Some(&self.weights_dir),
                self.weight_source,
            )?
            .ok_or_else(|| OidnError::MissingModel(self.weights_dir.join(key.filename())))?
            .bytes
        };
        let (net, plan) = super::build_commit_artifacts(
            self.device,
            &bytes,
            3,
            out.width,
            out.height,
            self.max_memory_mb,
            self.user_input_scale,
        )?;
        self.model_key = Some(key);
        self.net = Some(net);
        self.plan = Some(plan);

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
            RunOptions {
                transfer,
                hdr: is_hdr,
                signed: self.directional,
                input_scale: self.user_input_scale,
                sanitize_nonfinite: self.nan_to_zero,
                output_channels: 3,
            },
            progress,
        )
    }
}
