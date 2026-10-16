use bytes::Bytes;
use std::collections::BTreeMap;

use crate::TzaError;

/// A tensor's layout as the archive stores it: one ASCII letter per axis (`training/tza.py` writes the string as
/// given, `ndim` bytes). Intel's shipped files use `x` (1-D bias) and `oihw` (4-D conv kernel); our converted
/// networks also carry e.g. `oi` (a linear weight), `iohw` (a transposed-conv kernel) and the empty layout of a
/// rank-0 scalar, so the layout is the string itself, not a closed set. Intel's C++ reader accepts only `x` / `oihw`;
/// a loader states which layout each named tensor must have (as `oidn-model` does).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    len: u8,
    axes: [u8; Layout::MAX_NDIM],
}

impl Layout {
    /// The highest rank a layout may name. TZA's `ndim` is a byte; no network of ours has a tensor above rank 8, so a
    /// longer layout is refused instead of carried.
    pub const MAX_NDIM: usize = 8;
    /// 1-D vector layout (used for biases).
    pub const X: Layout = Layout::from_ascii(b"x");
    /// 4-D conv weight: `[out_channels, in_channels, kernel_h, kernel_w]`.
    pub const OIHW: Layout = Layout::from_ascii(b"oihw");

    /// Copy a known-valid layout (the constants above); `new` validates.
    const fn from_ascii(name: &[u8]) -> Layout {
        let mut axes = [0u8; Layout::MAX_NDIM];
        let mut i = 0;
        while i < name.len() {
            axes[i] = name[i];
            i += 1;
        }
        Layout {
            len: name.len() as u8,
            axes,
        }
    }

    /// The layout named `name`: ASCII letters, at most [`Self::MAX_NDIM`] of them (the empty name is a scalar's).
    pub fn new(name: &str) -> Result<Layout, TzaError> {
        if name.len() > Layout::MAX_NDIM || !name.bytes().all(|b| b.is_ascii_alphabetic()) {
            return Err(TzaError::InvalidLayout {
                got: name.to_owned(),
            });
        }
        Ok(Layout::from_ascii(name.as_bytes()))
    }

    /// The layout string, one letter per axis.
    pub fn as_str(&self) -> &str {
        // Only `new` (ASCII-checked) and the ASCII constants build a `Layout`; the fallback is unreachable.
        self.axes
            .get(..usize::from(self.len))
            .and_then(|axes| std::str::from_utf8(axes).ok())
            .unwrap_or("")
    }

    /// The rank the layout names.
    pub fn ndim(&self) -> usize {
        usize::from(self.len)
    }
}

impl std::fmt::Debug for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
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
        if self.dims.len() != self.layout.ndim() {
            return Err(TzaError::LayoutNdimMismatch {
                layout: self.layout.as_str().to_owned(),
                expected: self.layout.ndim(),
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
