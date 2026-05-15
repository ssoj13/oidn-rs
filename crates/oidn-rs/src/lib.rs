//! oidn-rs — pure Rust port of Intel Open Image Denoise.
//!
//! High-level architecture mirrors `_ref/oidn/core/`. Each top-level module
//! corresponds to a C++ file in the upstream reference implementation:
//!
//! | Module           | Reference                                      |
//! |------------------|------------------------------------------------|
//! | `device`         | `core/device.h`                                |
//! | `image`          | `core/image.h`                                 |
//! | `color`          | `core/color.{h,cpp}`                           |
//! | `tile`           | `core/tile.h` + `core/unet_filter.cpp` (geom)  |
//! | `autoexposure`   | `core/autoexposure.h`                          |
//! | `registry`       | `core/rt_filter.cpp` (model picker)            |
//! | `filter`         | `core/filter.h` (trait)                        |
//! | `filters::rt`    | `core/rt_filter.cpp`                           |
//! | `filters::rtlightmap` | `core/rtlightmap_filter.cpp`              |
//! | `filters::unet_runner` | `core/unet_filter.cpp`                   |

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod autoexposure;
pub mod color;
pub mod device;
pub mod error;
pub mod filter;
pub mod filters;
pub mod gpu_ops;
pub mod image;
pub mod image_tensor;
pub mod prelude;
pub mod registry;
pub mod tile;

pub use device::WgpuDevice;
pub use error::OidnError;
pub use filter::{Filter, Quality};
pub use filters::rt::RtFilter;
pub use filters::rtlightmap::RtLightmapFilter;
pub use image::{Image, ImageMut, PixelFormat};
pub use registry::ModelKey;
