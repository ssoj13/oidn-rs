//! Prelude — convenience re-exports.
//!
//! The top-level glob is device-agnostic: it exposes traits, error
//! types, image / pixel helpers, and the filter front-ends (`RtFilter`,
//! `RtLightmapFilter`, `CommittedRtFilter`). The wgpu device wrapper this
//! crate ships is opt-in via the [`wgpu_prelude`] sub-module.

pub use crate::{
    CommittedRtFilter, Filter, Image, ImageMut, ModelKey, OidnError, PixelFormat, Quality,
    RtFilter, RtLightmapFilter,
};

/// Convenience re-exports for the wgpu-backed pipeline. Import with
/// `use oidn_rs::prelude::wgpu_prelude::*;` when you specifically want
/// the wgpu device wrapper this crate ships with.
pub mod wgpu_prelude {
    pub use crate::WgpuDevice;
}
