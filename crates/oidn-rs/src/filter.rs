//! Filter trait + Quality enum — corresponds to `_ref/oidn/core/filter.h`.

use crate::error::OidnError;

/// Quality preset, matching the public C API enum
/// (`_ref/oidn/include/OpenImageDenoise/oidn.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// Maximum quality (default for new filters).
    #[default]
    High,
    /// Same network width as `High`; reserved for a future balanced
    /// variant once dedicated weights are trained.
    Balanced,
    /// Smaller model for previews / interactive; falls back to the
    /// base variant when no `_small` weights are shipped for a route.
    Fast,
}

/// Common interface for all OIDN filters. Equivalent to
/// `oidnCommitFilter` + `oidnExecuteFilter` in the C API.
pub trait Filter {
    /// Validate parameters and lock in the model + tile plan.
    fn commit(&mut self) -> Result<(), OidnError>;

    /// Run inference. Must be called after `commit()`.
    fn execute(&mut self) -> Result<(), OidnError>;

    /// Install a progress callback. Receives `[0.0, 1.0]` after each
    /// processed tile; returning `false` aborts execution with
    /// `OidnError::Cancelled`.
    ///
    /// The default implementation returns
    /// [`OidnError::UnsupportedFeatures`] so individual filter
    /// implementations can opt in. Mirrors the
    /// `oidnSetFilterProgressMonitorFunction` entrypoint in the C ABI.
    fn set_progress(
        &mut self,
        _cb: Box<dyn FnMut(f32) -> bool + 'static>,
    ) -> Result<(), OidnError> {
        Err(OidnError::UnsupportedFeatures)
    }
}
