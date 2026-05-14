use std::collections::BTreeMap;

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
    pub fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.byte_size()
    }
}

/// A single tensor with its raw data buffer (owned, copied out of the source).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub desc: TensorDesc,
    pub data: Vec<u8>,
}

impl Tensor {
    /// Reinterpret the data as f32 slice. Returns `None` if dtype is not f32.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self.desc.dtype {
            DType::Float32 => Some(bytemuck::cast_slice(&self.data)),
            DType::Float16 => None,
        }
    }

    /// Reinterpret the data as f16 slice. Returns `None` if dtype is not f16.
    pub fn as_f16(&self) -> Option<&[half::f16]> {
        match self.desc.dtype {
            DType::Float16 => Some(bytemuck::cast_slice(&self.data)),
            DType::Float32 => None,
        }
    }

    /// Decode the tensor as `Vec<f32>`, converting from f16 if needed.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.desc.dtype {
            DType::Float32 => self.as_f32().unwrap().to_vec(),
            DType::Float16 => self.as_f16().unwrap().iter().map(|h| h.to_f32()).collect(),
        }
    }
}

/// Sorted map of tensor name to tensor (BTreeMap for deterministic iteration).
pub type TensorMap = BTreeMap<String, Tensor>;
