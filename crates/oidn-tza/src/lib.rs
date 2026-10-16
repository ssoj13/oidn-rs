//! TZA (Tensor Archive) parser and writer.
//!
//! Direct port of the binary format defined in `_ref/oidn/core/tza.cpp` (reader) and `_ref/oidn/training/tza.py`
//! (writer). Layout: `[u16 magic=0x41D7][u8 majorV][u8 minorV][u64 tableOffset][... tensor blobs ...][u32 N][N tensor entries]`.
//! Each tensor entry: `[u16 nameLen][nameLen bytes][u8 ndim][u32 dim_i × ndim][char layout_i × ndim][char dtype][u64 dataOffset]`.
//!
//! The one weight format of our networks: Intel's shipped denoisers, and every checkpoint the ofx-rs ML store
//! converts (`write`) and loads by name (`parse`, then a named-tensor loader like `oidn-model`'s).

#![forbid(unsafe_op_in_unsafe_fn)]

mod error;
mod parser;
mod types;
mod writer;

pub use bytes::Bytes;
pub use error::TzaError;
pub use parser::parse;
pub use types::{DType, Layout, Tensor, TensorDesc, TensorMap};
pub use writer::write;
