//! GPU device wrapper around burn's wgpu backend.
//!
//! In burn 0.22 the backend is dynamic (device-selected): a single
//! backend-erased [`burn::tensor::Device`] carries the backend identity, and
//! tensors/modules run on whichever device they were built with. The wgpu
//! variant is provided by burn's `wgpu` feature via [`Device::wgpu`]. We pick
//! `f32` precision by relying on the default device settings — wgpu's
//! `shader-f16` extension is still inconsistent across vendors as of 2026.

use burn::tensor::{Device, DeviceKind};

use crate::error::OidnError;

/// High-level wrapper owning the Burn wgpu device handle.
///
/// Use `WgpuDevice::new()` to pick the best adapter automatically (wgpu's
/// high-power heuristic, overridable via `CUBECL_WGPU_DEFAULT_DEVICE`), or
/// `WgpuDevice::with_handle(handle)` when integrating with an existing Burn
/// device (e.g. from a renderer that already owns a wgpu context).
#[derive(Debug, Clone)]
pub struct WgpuDevice {
    pub handle: Device,
}

impl WgpuDevice {
    /// Initialise a default wgpu device. Adapter selection follows wgpu's
    /// heuristics; the actual client is initialised on first tensor
    /// allocation. Returns `Result` for forward compatibility with
    /// adapter-selection failures.
    pub fn new() -> Result<Self, OidnError> {
        Ok(Self {
            handle: Device::wgpu(DeviceKind::DefaultDevice),
        })
    }

    /// Wrap an existing Burn device handle (e.g. from a renderer that already
    /// owns a wgpu context).
    pub fn with_handle(handle: Device) -> Self {
        Self { handle }
    }
}

impl Default for WgpuDevice {
    fn default() -> Self {
        Self::new().expect("default wgpu device init")
    }
}
