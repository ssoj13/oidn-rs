//! UNet inference pipeline — generic over Burn `Backend`.
//!
//! Phase I (I.2 + I.4) version: tile pack / unpack runs on the GPU via
//! Burn ops. The per-tile, per-pixel CPU loop from the original port
//! (visible in this file's history before commit `8ae2939`) is gone; the
//! pipeline now does, per `run()`:
//!
//! 1. Decode the input images (`Image::to_rgb_f32`) — still host work,
//!    will be lifted by I.5 once the wgpu↔Burn buffer bridge lands.
//! 2. Upload each source as a `[1, 3, H, W]` Burn tensor (one allocation
//!    per input, reused across all tiles).
//! 3. For each tile:
//!    - slice the source rectangle out of the uploaded tensor,
//!    - [`gpu_ops::reflect_pad_2d`] to the tile geometry,
//!    - [`gpu_ops::apply_transfer_forward`] on the colour channel,
//!    - albedo `clamp(0, 1)` / normal identity-or-clamp,
//!    - `Tensor::cat` channels into `[1, in_c, tile_h, tile_w]`,
//!    - `Net::forward`,
//!    - [`gpu_ops::apply_transfer_inverse`] on the network output,
//!    - `slice_assign` the cropped output region into a `[1, 3, H, W]`
//!      accumulator tensor.
//! 4. Pull the accumulator back to host once and write it into the
//!    legacy `ImageMut`. I.5 will replace this with a tensor-out path.

use burn::prelude::Backend;
use burn::tensor::Tensor;
use oidn_model::Net;

use crate::{
    autoexposure,
    color::{TransferFunction, TransferState},
    error::OidnError,
    gpu_ops,
    image::{Image, ImageMut},
    image_tensor,
    tile::{Rect, TilePlan},
};

/// Callback called with a `[0.0, 1.0]` progress fraction after each tile.
/// Returning `false` aborts execution with `OidnError::Cancelled`.
pub type ProgressFn<'a> = dyn FnMut(f32) -> bool + 'a;

/// One inference call: read tiles from the inputs, run the network, write
/// tiles into `output`. The caller supplies the loaded `UNet` model.
#[allow(clippy::too_many_arguments)]
pub fn run<B: Backend>(
    net: &Net<B>,
    device: &B::Device,
    plan: &TilePlan,
    color: Option<&Image<'_>>,
    albedo: Option<&Image<'_>>,
    normal: Option<&Image<'_>>,
    output: &mut ImageMut<'_>,
    transfer_kind: TransferFunction,
    hdr: bool,
    user_input_scale: Option<f32>,
    progress: Option<&mut ProgressFn<'_>>,
) -> Result<(), OidnError> {
    // Decode inputs to flat HWC f32 buffers up-front. Stays on host until
    // I.5 replaces `Image<'_>` callers with a tensor-native path.
    let color_buf = color.map(|img| img.to_rgb_f32());
    let albedo_buf = albedo.map(|img| img.to_rgb_f32());
    let normal_buf = normal.map(|img| img.to_rgb_f32());

    let w = output.width;
    let h = output.height;
    log::debug!(
        "unet_runner::run() {}x{} color={} albedo={} normal={} transfer={:?} hdr={} tiles={}",
        w, h,
        color_buf.is_some(), albedo_buf.is_some(), normal_buf.is_some(),
        transfer_kind, hdr, plan.jobs.len()
    );
    if let Some(c) = color_buf.as_deref() {
        let (cmin, cmax, cmean) = quick_stats(c);
        log::debug!("unet input color stats: min={cmin:.4} max={cmax:.4} mean={cmean:.4}");
    }

    // Autoexposure / transfer state — for HDR PU + Log we feed the colour
    // buffer through `autoexposure::compute_scale` on the host. The
    // tensor variant (`compute_scale_tensor`) exists but would force an
    // extra upload here; we already have the host buffer, so use the
    // cheap path. Either way the result is a single f32 scalar that ends
    // up baked into the `TransferState`.
    let mut tf = TransferState::new(transfer_kind);
    let scale = if let Some(s) = user_input_scale {
        s
    } else if hdr {
        if let Some(c) = color_buf.as_deref() { autoexposure::compute_scale(c, w, h) } else { 1.0 }
    } else {
        1.0
    };
    tf.set_input_scale(scale);
    log::debug!(
        "unet_runner: autoexposure scale={:.6} (hdr={}, user_scale={:?})",
        scale, hdr, user_input_scale
    );

    // Stage source tensors once. Each is [1, 3, H, W] f32 on `device`.
    let color_t = color_buf.as_ref().map(|buf| upload_hwc_as_chw_tensor::<B>(buf, w, h, device));
    let albedo_t = albedo_buf.as_ref().map(|buf| upload_hwc_as_chw_tensor::<B>(buf, w, h, device));
    let normal_t = normal_buf.as_ref().map(|buf| upload_hwc_as_chw_tensor::<B>(buf, w, h, device));

    // Output accumulator: starts at zeros, slice_assign per tile.
    let mut accum: Tensor<B, 4> = Tensor::zeros([1, 3, h, w], device);
    // snorm: when only a normal input is set (no colour), the normal
    // values are signed in `[-1, 1]` and should not be clamped to
    // `[0, 1]`. Matches the `pack(..., snorm=true)` branch in the
    // original CPU path.
    let snorm = normal.is_some() && color.is_none();

    let mut progress = progress;
    let total_jobs = plan.jobs.len();
    let tile_w = plan.tile_w as usize;
    let tile_h = plan.tile_h as usize;

    for (tile_idx, job) in plan.jobs.iter().enumerate() {
        // 1. Pull the source rectangle out of each input tensor and pad
        //    to (tile_h, tile_w) via reflection. `align_offset_{x,y}` is
        //    the leading padding amount on the left/top side; the
        //    trailing side is whatever's left after the rectangle fits.
        let src_x = job.input.x as usize;
        let src_y = job.input.y as usize;
        let src_w = job.input.w as usize;
        let src_h = job.input.h as usize;
        let pad_left = job.align_offset_x as usize;
        let pad_top = job.align_offset_y as usize;
        debug_assert!(pad_left + src_w <= tile_w);
        debug_assert!(pad_top + src_h <= tile_h);
        let pad_right = tile_w - src_w - pad_left;
        let pad_bottom = tile_h - src_h - pad_top;

        let mut channel_parts: Vec<Tensor<B, 4>> = Vec::with_capacity(3);

        if let Some(src) = color_t.as_ref() {
            let rect = src
                .clone()
                .slice([0..1, 0..3, src_y..src_y + src_h, src_x..src_x + src_w]);
            let padded = gpu_ops::reflect_pad_2d(rect, pad_top, pad_bottom, pad_left, pad_right);
            channel_parts.push(gpu_ops::apply_transfer_forward(padded, &tf));
        }
        if let Some(src) = albedo_t.as_ref() {
            let rect = src
                .clone()
                .slice([0..1, 0..3, src_y..src_y + src_h, src_x..src_x + src_w]);
            let padded = gpu_ops::reflect_pad_2d(rect, pad_top, pad_bottom, pad_left, pad_right);
            // Albedos are LDR reflectances; clamp the same way the CPU
            // path did (`(r/g/b).clamp(0.0, 1.0)`).
            channel_parts.push(padded.clamp(0.0, 1.0));
        }
        if let Some(src) = normal_t.as_ref() {
            let rect = src
                .clone()
                .slice([0..1, 0..3, src_y..src_y + src_h, src_x..src_x + src_w]);
            let padded = gpu_ops::reflect_pad_2d(rect, pad_top, pad_bottom, pad_left, pad_right);
            channel_parts.push(if snorm { padded } else { padded.clamp(0.0, 1.0) });
        }

        // 2. Concat along channel dim → [1, in_c, tile_h, tile_w].
        let input_tensor: Tensor<B, 4> = Tensor::cat(channel_parts, 1);

        // 3. Forward (UNet or UNetLarge, dispatched by `Net`).
        let output_tensor: Tensor<B, 4> = net.forward(input_tensor);

        // 4. Inverse transfer + crop + slice_assign into the accumulator.
        //    `Linear` has no inverse; the tensor passes through.
        let post = if matches!(transfer_kind, TransferFunction::Linear) {
            output_tensor
        } else {
            gpu_ops::apply_transfer_inverse(output_tensor, &tf)
        };

        let Rect { x: ox, y: oy, w: ow, h: oh } = job.output_src_in_tile;
        let Rect { x: dx, y: dy, w: _, h: _ } = job.output_dst;
        let cropped = post.slice([
            0..1,
            0..3,
            oy as usize..(oy + oh) as usize,
            ox as usize..(ox + ow) as usize,
        ]);
        accum = accum.slice_assign(
            [
                0..1,
                0..3,
                dy as usize..(dy + oh) as usize,
                dx as usize..(dx + ow) as usize,
            ],
            cropped,
        );

        // Tile-granularity progress reporting + cooperative cancellation.
        if let Some(cb) = progress.as_deref_mut() {
            let fraction = (tile_idx + 1) as f32 / total_jobs as f32;
            if !cb(fraction) { return Err(OidnError::Cancelled); }
        }
    }

    // 5. Pull the full accumulator back to host as HWC f32 once.
    //    I.5 swaps this for a direct accumulator → wgpu::Buffer copy.
    let (chw_vec, dims) = image_tensor::tensor_to_chw_vec(accum);
    debug_assert_eq!(dims, [1, 3, h, w]);
    let hwc_vec = image_tensor::chw_to_hwc(&chw_vec, 3, h, w);

    let (omin_post, omax_post, omean_post) = quick_stats(&hwc_vec);
    log::debug!(
        "unet_runner output (after inverse transfer): min={omin_post:.4} max={omax_post:.4} mean={omean_post:.4}"
    );
    output.write_rgb_f32(&hwc_vec);
    Ok(())
}

/// Stage a flat HWC `f32` slice as a `[1, 3, H, W]` Burn tensor on `device`.
fn upload_hwc_as_chw_tensor<B: Backend>(
    buf_hwc: &[f32],
    width: usize,
    height: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let chw = image_tensor::hwc_to_chw(buf_hwc, 3, height, width);
    image_tensor::chw_vec_to_tensor::<B>(chw, 3, height, width, device)
}

/// Quick min/max/mean over a flat f32 slice — used to trace the host
/// boundaries (decoded input + final output) during integration with
/// `env_logger`.
fn quick_stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in data {
        if v.is_finite() {
            if v < min { min = v; }
            if v > max { max = v; }
            sum += v as f64;
        }
    }
    (min, max, (sum / data.len() as f64) as f32)
}
