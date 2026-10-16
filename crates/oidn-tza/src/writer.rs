//! TZA writer: the archive Intel's `training/tza.py` `Writer` produces, so one format holds Intel's shipped denoiser
//! weights and every network we convert or train (the ofx-rs ML weight store converts downloaded `.pth` / `.pkl` /
//! `.safetensors` checkpoints into it once).

use crate::error::TzaError;
use crate::parser::{MAGIC, SUPPORTED_MAJOR};
use crate::types::{DType, TensorMap};

/// Payload and table alignment of `training/tza.py` (`_write_pad`): every tensor's payload and the table start on a
/// 64-byte boundary, so a mapped archive hands out aligned `f32` / `f16` slices (`Tensor::as_f32` borrows them).
const ALIGN: usize = 64;

/// The version `training/tza.py` writes (`VERSION = (2, 0)`).
const MINOR: u8 = 0;

/// Header: magic `u16`, major `u8`, minor `u8`, table offset `u64`.
const HEADER: usize = 12;

fn align(pos: usize) -> Result<usize, TzaError> {
    pos.checked_next_multiple_of(ALIGN)
        .ok_or(TzaError::SizeOverflow)
}

fn dtype_code(dtype: DType) -> u8 {
    match dtype {
        DType::Float32 => b'f',
        DType::Float16 => b'h',
    }
}

fn offset(pos: usize) -> Result<u64, TzaError> {
    u64::try_from(pos).map_err(|_| TzaError::SizeOverflow)
}

/// Serialise `map` as a TZA 2.0 archive into `out` (cleared first, its capacity kept): the header, each tensor's
/// little-endian payload at a 64-byte aligned offset in name order, then the table at a 64-byte aligned offset.
/// [`crate::parse`] of the result returns `map` (same names, descriptors and payload bytes). Refused before anything is
/// written: a tensor whose payload does not match its descriptor (`Tensor::validate`), a name longer than `u16::MAX`
/// bytes (`NameTooLong`).
pub fn write(map: &TensorMap, out: &mut Vec<u8>) -> Result<(), TzaError> {
    // Pass 1: validate and place every payload, so the header can name the table offset up front.
    let mut offsets = Vec::with_capacity(map.len());
    let mut pos = HEADER;
    for (name, tensor) in map {
        tensor.validate()?;
        if u16::try_from(name.len()).is_err() {
            return Err(TzaError::NameTooLong { len: name.len() });
        }
        pos = align(pos)?;
        offsets.push(pos);
        pos = pos
            .checked_add(tensor.data.len())
            .ok_or(TzaError::SizeOverflow)?;
    }
    let table = align(pos)?;
    let count = u32::try_from(map.len()).map_err(|_| TzaError::SizeOverflow)?;

    // Pass 2: emit. Padding is zero bytes, as `_write_pad` writes.
    out.clear();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&[SUPPORTED_MAJOR, MINOR]);
    out.extend_from_slice(&offset(table)?.to_le_bytes());
    for (tensor, &at) in map.values().zip(&offsets) {
        out.resize(at, 0);
        out.extend_from_slice(&tensor.data);
    }
    out.resize(table, 0);
    out.extend_from_slice(&count.to_le_bytes());
    for ((name, tensor), &at) in map.iter().zip(&offsets) {
        let desc = &tensor.desc;
        // `validate` checked dims.len() == layout.ndim() <= Layout::MAX_NDIM and the name length above.
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(desc.dims.len() as u8);
        for dim in &desc.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(desc.layout.as_str().as_bytes());
        out.push(dtype_code(desc.dtype));
        out.extend_from_slice(&offset(at)?.to_le_bytes());
    }
    Ok(())
}
