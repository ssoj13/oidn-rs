use thiserror::Error;

#[derive(Debug, Error)]
pub enum TzaError {
    #[error("tensor size exceeds the addressable range")]
    SizeOverflow,
    #[error("invalid tensor dimension at axis {axis}: {value}")]
    InvalidDimension { axis: usize, value: u32 },
    #[error("archive offset cannot be represented on this target: {0}")]
    InvalidOffset(u64),
    #[error("duplicate tensor name: {0:?}")]
    DuplicateName(String),
    #[error("tensor payload length mismatch: expected {expected}, got {got}")]
    DataLengthMismatch { expected: usize, got: usize },
    #[error("buffer too small at offset {offset}: need {need}, have {have}")]
    OutOfBounds {
        offset: usize,
        need: usize,
        have: usize,
    },

    #[error("invalid TZA magic: expected 0x41D7, got 0x{got:04X}")]
    BadMagic { got: u16 },

    #[error("unsupported TZA major version {got} (expected 2)")]
    UnsupportedVersion { got: u8 },

    #[error("invalid tensor layout: {got:?}")]
    InvalidLayout { got: String },

    #[error("invalid tensor dtype: {got:?}")]
    InvalidDtype { got: char },

    #[error("invalid utf-8 in tensor name")]
    BadName(#[from] std::string::FromUtf8Error),

    #[error("invalid layout/ndim mismatch: layout {layout:?} requires {expected} dims, got {got}")]
    LayoutNdimMismatch {
        layout: String,
        expected: usize,
        got: usize,
    },
}
