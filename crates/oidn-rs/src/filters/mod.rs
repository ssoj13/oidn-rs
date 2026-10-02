pub mod rt;
pub mod rtlightmap;
pub mod unet_runner;

use crate::{
    error::OidnError,
    image,
    tile::{self, DEFAULT_MAX_TILE_SIZE, TilePlan},
};
use burn::tensor::Device;
use oidn_model::{ModelDescriptor, Net};

/// Shared validated model construction and topology-aware tile planning.
/// The logical memory estimate excludes backend workspaces/allocator overhead;
/// an infeasible budget uses the reference minimum-tile fallback and is reported.
fn build_commit_artifacts(
    device: &Device,
    bytes: &[u8],
    in_channels: usize,
    width: usize,
    height: usize,
    max_memory_mb: Option<i32>,
    input_scale: Option<f32>,
) -> Result<(Net, TilePlan), OidnError> {
    validate_execution(width, height, input_scale)?;
    let tensors = oidn_tza::parse(bytes)?;
    let descriptor = ModelDescriptor::from_tza(&tensors)?;
    if descriptor.in_channels() != in_channels || descriptor.out_channels() != 3 {
        return Err(OidnError::InvalidArgument(
            "archive channel layout does not match filter inputs",
        ));
    }
    let whole_image_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(in_channels + 3))
        .and_then(|elements| elements.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(descriptor.parameters_bytes()))
        .ok_or(OidnError::InvalidArgument("model memory estimate overflow"))?;
    let activation_bytes = descriptor.activation_bytes_per_pixel();
    let budget = max_memory_mb
        .map(|mb| {
            usize::try_from(mb)
                .ok()
                .and_then(|n| n.checked_mul(1024 * 1024))
                .ok_or(OidnError::InvalidArgument("invalid memory budget"))
        })
        .transpose()?;
    let max_pixels = budget
        .map(|bytes| {
            (bytes.saturating_sub(whole_image_bytes) / activation_bytes.max(1))
                .clamp(1, DEFAULT_MAX_TILE_SIZE as usize) as i32
        })
        .unwrap_or(DEFAULT_MAX_TILE_SIZE);
    let plan = tile::plan(
        width as i32,
        height as i32,
        descriptor.receptive_field(),
        descriptor.alignment(),
        max_pixels,
    )?;
    let tile_pixels = (plan.tile_w as usize)
        .checked_mul(plan.tile_h as usize)
        .ok_or(OidnError::InvalidArgument("tile estimate overflow"))?;
    let estimate = tile_pixels
        .checked_mul(activation_bytes)
        .and_then(|bytes| bytes.checked_add(whole_image_bytes))
        .ok_or(OidnError::InvalidArgument(
            "working memory estimate overflow",
        ))?;
    log::debug!(
        "model={:?} rf={} tile={}x{} logical_memory_bytes={} budget_bytes={:?}",
        descriptor.variant(),
        descriptor.receptive_field(),
        plan.tile_w,
        plan.tile_h,
        estimate,
        budget
    );
    if budget.is_some_and(|limit| estimate > limit) {
        log::warn!(
            "minimum tile exceeds requested logical memory budget: estimate_bytes={estimate} budget_bytes={budget:?}; backend allocations are additional"
        );
    }
    let net = descriptor.load(&tensors, device)?;
    Ok((net, plan))
}

/// Validate execution geometry/scale before filesystem or model work.
fn validate_execution(
    width: usize,
    height: usize,
    input_scale: Option<f32>,
) -> Result<(), OidnError> {
    if input_scale
        .is_some_and(|scale| !scale.is_finite() || scale <= 0.0 || !scale.recip().is_finite())
    {
        return Err(OidnError::InvalidArgument(
            "input scale must be positive and finite",
        ));
    }
    image::validate_dimensions(width, height, 3)?;
    if width == 0 || height == 0 {
        return Err(OidnError::InvalidArgument("empty execution geometry"));
    }
    Ok(())
}
