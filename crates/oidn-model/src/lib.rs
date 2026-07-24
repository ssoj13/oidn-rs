//! U-Net architecture definitions for Intel OIDN.
//!
//! Direct port of `_ref/oidn/training/model.py:UNet`. In burn 0.22 the backend
//! is dynamic (device-selected), so the same network runs on CPU (NdArray) for
//! tests and on wgpu for production simply by choosing the device.

#![forbid(unsafe_op_in_unsafe_fn)]

mod loader;
mod net;
mod unet;
mod unet_large;
mod variants;

pub use loader::{LoadError, load_tza, load_tza_large};
pub use net::Net;
pub use unet::UNet;
pub use unet_large::{ChannelConfigLarge, UNetLarge};
pub use variants::{ChannelConfig, Variant};
