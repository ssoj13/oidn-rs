use thiserror::Error;

#[derive(Debug, Error)]
pub enum OidnError {
    #[error("filter parameter not set: {0}")]
    Unset(&'static str),

    #[error("input/output image dimensions or formats inconsistent: {0}")]
    Inconsistent(&'static str),

    #[error("unsupported feature combination — see _ref/oidn/core/rt_filter.cpp:getWeights for valid sets")]
    UnsupportedFeatures,

    #[error("model file not found: {0}")]
    MissingModel(std::path::PathBuf),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("tza: {0}")]
    Tza(#[from] oidn_tza::TzaError),

    #[error("model load: {0}")]
    Load(#[from] oidn_model::LoadError),

    #[error("device init failed: {0}")]
    Device(String),

    #[error("execution cancelled by progress callback")]
    Cancelled,

    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
}
