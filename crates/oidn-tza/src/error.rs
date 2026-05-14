use thiserror::Error;

#[derive(Debug, Error)]
pub enum TzaError {
    #[error("buffer too small at offset {offset}: need {need}, have {have}")]
    OutOfBounds { offset: usize, need: usize, have: usize },

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
    LayoutNdimMismatch { layout: String, expected: usize, got: usize },
}
