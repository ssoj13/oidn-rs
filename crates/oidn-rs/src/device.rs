//! GPU device wrapper around `burn-wgpu`.
//!
//! The `Backend` type alias picked here decides the precision used by the
//! network. We pick `f32` for portability — wgpu's `shader-f16` extension is
//! still inconsistent across vendors as of 2026.

use crate::error::OidnError;

/// Concrete Burn backend used by the wgpu pipeline.
pub type WgpuBackend = burn_wgpu::Wgpu<f32, i32>;
/// Burn device handle for `WgpuBackend`. `Device` is declared on `BackendTypes`
/// (a supertrait of `Backend`) — qualifying through `BackendTypes` avoids the
/// E0576 "associated type not found in `Backend`" we'd get otherwise.
pub type WgpuDeviceHandle = <WgpuBackend as burn::tensor::backend::BackendTypes>::Device;

/// High-level wrapper that owns a wgpu adapter selection plus the Burn device.
///
/// Use `WgpuDevice::new()` to pick a primary adapter automatically, or
/// `WgpuDevice::with_handle(handle)` if integrating with an existing wgpu
/// context.
#[derive(Debug, Clone)]
pub struct WgpuDevice {
    pub handle: WgpuDeviceHandle,
}

impl WgpuDevice {
    /// Initialise a default wgpu device. Burn defers actual adapter selection
    /// to first tensor allocation; returning a default-constructed device
    /// handle is sufficient. Returns `Result` for forward compatibility with
    /// adapter-selection failures.
    pub fn new() -> Result<Self, OidnError> {
        Ok(Self { handle: WgpuDeviceHandle::default() })
    }

    /// Wrap an existing Burn device handle (e.g. from a renderer that already
    /// owns a wgpu context).
    pub fn with_handle(handle: WgpuDeviceHandle) -> Self {
        Self { handle }
    }
}

impl Default for WgpuDevice {
    fn default() -> Self {
        Self::new().expect("default wgpu device init")
    }
}
