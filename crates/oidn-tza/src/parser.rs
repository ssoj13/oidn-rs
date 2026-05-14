use crate::error::TzaError;
use crate::types::{DType, Layout, Tensor, TensorDesc, TensorMap};

const MAGIC: u16 = 0x41D7;
const SUPPORTED_MAJOR: u8 = 2;

/// Cursor over the input buffer with bounds-checked reads.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn check(&self, need: usize) -> Result<(), TzaError> {
        if self.buf.len().saturating_sub(self.pos) < need {
            return Err(TzaError::OutOfBounds {
                offset: self.pos,
                need,
                have: self.buf.len().saturating_sub(self.pos),
            });
        }
        Ok(())
    }

    fn seek(&mut self, pos: usize) -> Result<(), TzaError> {
        if pos > self.buf.len() {
            return Err(TzaError::OutOfBounds {
                offset: pos,
                need: 0,
                have: self.buf.len(),
            });
        }
        self.pos = pos;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, TzaError> {
        self.check(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> Result<u16, TzaError> {
        self.check(2)?;
        let v = u16::from_le_bytes(self.buf[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32, TzaError> {
        self.check(4)?;
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64, TzaError> {
        self.check(8)?;
        let v = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], TzaError> {
        self.check(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

fn decode_layout(s: &str) -> Result<Layout, TzaError> {
    match s {
        "x" => Ok(Layout::X),
        "oihw" => Ok(Layout::Oihw),
        other => Err(TzaError::InvalidLayout { got: other.to_owned() }),
    }
}

fn expected_ndim(layout: Layout) -> usize {
    match layout {
        Layout::X => 1,
        Layout::Oihw => 4,
    }
}

fn decode_dtype(c: char) -> Result<DType, TzaError> {
    match c {
        'f' => Ok(DType::Float32),
        'h' => Ok(DType::Float16),
        other => Err(TzaError::InvalidDtype { got: other }),
    }
}

/// Parse a TZA archive from a byte buffer.
///
/// Returns a `TensorMap` (`BTreeMap<String, Tensor>`) containing every named
/// tensor, with data copied out of the source buffer (so the result is `'static`).
pub fn parse(bytes: &[u8]) -> Result<TensorMap, TzaError> {
    let mut cur = Cursor::new(bytes);

    let magic = cur.read_u16()?;
    if magic != MAGIC {
        return Err(TzaError::BadMagic { got: magic });
    }

    let major = cur.read_u8()?;
    let _minor = cur.read_u8()?;
    if major != SUPPORTED_MAJOR {
        return Err(TzaError::UnsupportedVersion { got: major });
    }

    let table_offset = cur.read_u64()? as usize;
    cur.seek(table_offset)?;

    let n_tensors = cur.read_u32()? as usize;
    let mut map = TensorMap::new();

    for _ in 0..n_tensors {
        // Name
        let name_len = cur.read_u16()? as usize;
        let name_bytes = cur.read_bytes(name_len)?.to_vec();
        let name = String::from_utf8(name_bytes)?;

        // Dims
        let ndim = cur.read_u8()? as usize;
        let mut dims = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            dims.push(cur.read_u32()?);
        }

        // Layout (`ndim` chars)
        let layout_bytes = cur.read_bytes(ndim)?;
        let layout_str = std::str::from_utf8(layout_bytes)
            .map_err(|_| TzaError::InvalidLayout { got: format!("{layout_bytes:?}") })?;
        let layout = decode_layout(layout_str)?;
        if expected_ndim(layout) != ndim {
            return Err(TzaError::LayoutNdimMismatch {
                layout: layout_str.to_owned(),
                expected: expected_ndim(layout),
                got: ndim,
            });
        }

        // Dtype
        let dtype_byte = cur.read_u8()?;
        let dtype = decode_dtype(dtype_byte as char)?;

        // Tensor data offset
        let data_offset = cur.read_u64()? as usize;

        let desc = TensorDesc { dims, layout, dtype };
        let byte_size = desc.byte_size();

        // Bounds-check raw data without consuming the cursor's position
        if data_offset
            .checked_add(byte_size)
            .map(|end| end > bytes.len())
            .unwrap_or(true)
        {
            return Err(TzaError::OutOfBounds {
                offset: data_offset,
                need: byte_size,
                have: bytes.len().saturating_sub(data_offset),
            });
        }
        let data = bytes[data_offset..data_offset + byte_size].to_vec();

        map.insert(name, Tensor { desc, data });
    }

    Ok(map)
}
