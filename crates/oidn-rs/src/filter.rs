//! Filter trait + Quality enum — corresponds to `_ref/oidn/core/filter.h`.

use crate::error::OidnError;

/// Quality preset, matching the public C API enum
/// (`_ref/oidn/include/OpenImageDenoise/oidn.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// Maximum quality (default for new filters).
    #[default]
    High,
    /// Same network width as `High` for v0.1, future variant slot.
    Balanced,
    /// Smaller model for previews / interactive (falls back to base in v0.1).
    Fast,
}

/// Common interface for all OIDN filters. Equivalent to
/// `oidnCommitFilter` + `oidnExecuteFilter` in the C API.
pub trait Filter {
    /// Validate parameters and lock in the model + tile plan.
    fn commit(&mut self) -> Result<(), OidnError>;

    /// Run inference. Must be called after `commit()`.
    fn execute(&mut self) -> Result<(), OidnError>;
}
