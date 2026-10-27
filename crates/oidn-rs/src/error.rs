//! Error types and result definitions for OIDN operations.
//!
//! # What this module is
//!
//! Defines [`OidnError`], the crate-wide error enum for `oidn-rs`.
//!
//! # Why it exists
//!
//! Centralizes all error conditions — parameter validation, feature mismatches,
//! I/O and weight deserialization, tensor runtime issues, and upstream C ABI
//! parity codes — into a single zero-allocation, typed error hierarchy.
//!
//! # How it works
//!
//! Uses `thiserror::Error` to derive formatted display messages and `From`
//! implementations for underlying errors (I/O, TZA parsing, model loading,
//! backend runtime errors). Variants mirroring Intel OIDN C ABI error codes
//! carry `&'static str` tags to preserve zero-allocation error reporting.
//!
//! # Where it is used
//!
//! Exposed at the crate root (`oidn::OidnError`) and returned by all fallible
//! filter execution, weight loading, and buffer operations across `oidn-rs`
//! and its consumer crates.

use thiserror::Error;

/// Crate-wide error type.
///
/// Marked `#[non_exhaustive]` so future variants — including any new error
/// codes added by upstream Intel OIDN — can be introduced without a breaking
/// change. Match exhaustively only when you intend to surface every case to
/// the user; otherwise add a `_` arm.
///
/// Variants prefixed `Unknown` / `InvalidOperation` / `OutOfMemory` /
/// `UnsupportedHardware` mirror the C ABI error codes in
/// `_ref/oidn/include/OpenImageDenoise/oidn.h` (`OIDNError`). They carry a
/// `&'static str` context tag rather than the C API's per-device string buffer
/// so we stay zero-allocation on the error path.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OidnError {
    #[error("filter parameter not set: {0}")]
    Unset(&'static str),

    #[error("input/output image dimensions or formats inconsistent: {0}")]
    Inconsistent(&'static str),

    #[error(
        "unsupported feature combination — see _ref/oidn/core/rt_filter.cpp:getWeights for valid sets"
    )]
    UnsupportedFeatures,

    #[error("model file not found: {0}")]
    MissingModel(std::path::PathBuf),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("tza: {0}")]
    Tza(#[from] oidn_tza::TzaError),

    #[error("model load: {0}")]
    Load(#[from] oidn_model::LoadError),

    /// Device-initialisation failure. Currently carries a `String` because
    /// `burn-wgpu` does not export a stable error type we can wrap; revisit
    /// once Burn's device API stabilises (tracked as V14).
    #[error("device init failed: {0}")]
    Device(String),

    /// A backend read failed during execution (for example, exposure reduction).
    #[error("tensor execution failed: {0}")]
    Execution(#[from] burn::tensor::TensorReadError),

    #[error("execution cancelled by progress callback")]
    Cancelled,

    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),

    /// C-ABI parity: `OIDN_ERROR_UNKNOWN`.
    #[error("unknown error: {0}")]
    Unknown(&'static str),

    /// C-ABI parity: `OIDN_ERROR_INVALID_OPERATION` — call sequence violated
    /// (e.g. `execute` before `commit`, or setter after `commit`).
    #[error("invalid operation: {0}")]
    InvalidOperation(&'static str),

    /// C-ABI parity: `OIDN_ERROR_OUT_OF_MEMORY`. Surfaced for device-side
    /// allocation failures; host-side allocation failures abort via the
    /// global allocator before we get a chance to construct this.
    #[error("out of memory: {0}")]
    OutOfMemory(&'static str),

    /// C-ABI parity: `OIDN_ERROR_UNSUPPORTED_HARDWARE` — adapter lacks a
    /// capability the chosen pipeline requires.
    #[error("unsupported hardware: {0}")]
    UnsupportedHardware(&'static str),
}
