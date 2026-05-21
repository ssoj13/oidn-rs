//! Prelude — convenience re-exports.
//!
//! The top-level glob is backend-agnostic: it exposes traits, error
//! types, image / pixel helpers, and the filter front-ends (`RtFilter`,
//! `RtLightmapFilter`, `CommittedRtFilter`). Anything tied to a concrete
//! Burn backend (such as the wgpu device / backend aliases this crate
//! ships) is opt-in via the [`wgpu_prelude`] sub-module so generic
//! downstream code never accidentally pulls in `burn-wgpu`.

pub use crate::{
    CommittedRtFilter, Filter, Image, ImageMut, ModelKey, OidnError, PixelFormat, Quality,
    RtFilter, RtLightmapFilter,
};

/// Convenience re-exports for the wgpu-backed pipeline. Import with
/// `use oidn_rs::prelude::wgpu_prelude::*;` when you specifically want
/// the wgpu device and backend aliases this crate ships with.
pub mod wgpu_prelude {
    pub use crate::WgpuDevice;
    pub use crate::device::WgpuBackend;
}
