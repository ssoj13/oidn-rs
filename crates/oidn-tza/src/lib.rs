//! TZA (Tensor Archive) parser.
//!
//! Direct port of the binary format defined in `_ref/oidn/core/tza.cpp`.
//! Layout: `[u16 magic=0x41D7][u8 majorV][u8 minorV][u64 tableOffset][... tensor blobs ...][u32 N][N tensor entries]`.
//! Each tensor entry: `[u16 nameLen][nameLen bytes][u8 ndim][u32 dim_i × ndim][char layout_i × ndim][char dtype][u64 dataOffset]`.

#![forbid(unsafe_op_in_unsafe_fn)]

mod error;
mod parser;
mod types;

pub use error::TzaError;
pub use parser::parse;
pub use types::{DType, Layout, Tensor, TensorDesc, TensorMap};
