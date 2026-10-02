use bytes::Bytes;
use std::collections::BTreeMap;

use crate::TzaError;

/// Tensor memory layout — port of `_ref/oidn/core/tensor_layout.h`.
/// We only support the subset that appears in shipped TZA files: `x` (1-D bias)
/// and `oihw` (4-D conv kernels). Blocked GPU layouts are runtime-only and
/// never appear in the archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// 1-D vector layout (used for biases).
    X,
    /// 4-D conv weight: `[out_channels, in_channels, kernel_h, kernel_w]`.
    Oihw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32,
    Float16,
}

impl DType {
    pub const fn byte_size(self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub dims: Vec<u32>,
    pub layout: Layout,
    pub dtype: DType,
}

impl TensorDesc {
    /// Validate the layout and dimensions and calculate their checked product.
    pub fn num_elements(&self) -> Result<usize, TzaError> {
        let (layout, expected) = match self.layout {
            Layout::X => ("x", 1),
            Layout::Oihw => ("oihw", 4),
        };
        if self.dims.len() != expected {
            return Err(TzaError::LayoutNdimMismatch {
                layout: layout.to_owned(),
                expected,
                got: self.dims.len(),
            });
        }
        self.dims
            .iter()
            .enumerate()
            .try_fold(1usize, |n, (axis, &value)| {
                if value == 0 {
                    return Err(TzaError::InvalidDimension { axis, value });
                }
                n.checked_mul(value as usize).ok_or(TzaError::SizeOverflow)
            })
    }

    pub fn byte_size(&self) -> Result<usize, TzaError> {
        self.num_elements()?
            .checked_mul(self.dtype.byte_size())
            .ok_or(TzaError::SizeOverflow)
    }
}

/// A tensor owning a shared immutable little-endian payload.
/// Parsed tensors borrow ranges of one owned archive allocation. Construct custom
/// payloads with `Vec<u8>::into()` and replace the whole payload to modify data.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub desc: TensorDesc,
    pub data: Bytes,
}

impl Tensor {
    /// Borrow an aligned little-endian f32 payload when its descriptor is valid.
    /// Use `to_f32_vec` for portable decoding independent of alignment.
    pub fn as_f32(&self) -> Option<&[f32]> {
        if !cfg!(target_endian = "little") || self.desc.dtype != DType::Float32 {
            return None;
        }
        self.validate().ok()?;
        bytemuck::try_cast_slice(&self.data).ok()
    }

    /// Borrow an aligned little-endian f16 payload when its descriptor is valid.
    pub fn as_f16(&self) -> Option<&[half::f16]> {
        if !cfg!(target_endian = "little") || self.desc.dtype != DType::Float16 {
            return None;
        }
        self.validate().ok()?;
        bytemuck::try_cast_slice(&self.data).ok()
    }

    /// Validate both the public descriptor and its raw payload length.
    pub fn validate(&self) -> Result<(), TzaError> {
        let expected = self.desc.byte_size()?;
        if self.data.len() != expected {
            return Err(TzaError::DataLengthMismatch {
                expected,
                got: self.data.len(),
            });
        }
        Ok(())
    }

    /// Iterate validated little-endian values without allocation or alignment assumptions.
    pub fn iter_f32(&self) -> Result<impl ExactSizeIterator<Item = f32> + '_, TzaError> {
        self.validate()?;
        let dtype = self.desc.dtype;
        Ok(self
            .data
            .chunks_exact(dtype.byte_size())
            .map(move |b| match dtype {
                DType::Float32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                DType::Float16 => half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32(),
            }))
    }

    /// Decode little-endian values through the same fallible iterator.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, TzaError> {
        Ok(self.iter_f32()?.collect())
    }
}

/// Sorted map of tensor name to tensor (BTreeMap for deterministic iteration).
pub type TensorMap = BTreeMap<String, Tensor>;
