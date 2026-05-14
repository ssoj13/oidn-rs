//! UNet inference pipeline — generic over Burn `Backend`.
//!
//! Direct port of `_ref/oidn/core/unet_filter.cpp::execute` (without
//! sub-device sharding). Per tile we:
//!
//! 1. Pack the input rectangles (color + optional albedo + optional normal)
//!    into a `[1, in_channels, tile_h, tile_w]` Burn tensor, applying the
//!    HDR/sRGB transfer to the colour channels and any necessary normal
//!    rescaling. Padding outside the source rectangle is handled by reflection
//!    against the source content (matching `gpu_input_process.h`).
//! 2. Forward through the U-Net.
//! 3. Crop the output region (`output_src_in_tile`) and write it back to the
//!    destination buffer with the inverse transfer applied.

use burn::{
    prelude::*,
    tensor::TensorData,
};
use oidn_model::Net;

use crate::{
    autoexposure,
    color::{TransferFunction, TransferState},
    error::OidnError,
    image::{Image, ImageMut},
    tile::{Rect, TileJob, TilePlan},
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
    // Decode inputs to flat HWC f32 buffers up-front. This is cheap relative
    // to inference and keeps tile packing logic uniform.
    let color_buf = color.map(|img| img.to_rgb_f32());
    let albedo_buf = albedo.map(|img| img.to_rgb_f32());
    let normal_buf = normal.map(|img| img.to_rgb_f32());

    let w = output.width;
    let h = output.height;

    // Build transfer state. For HDR without an explicit scale, run autoexposure
    // on the colour buffer (mirrors unet_filter.cpp:171-189).
    let mut tf = TransferState::new(transfer_kind);
    let scale = if let Some(s) = user_input_scale {
        s
    } else if hdr {
        if let Some(c) = color_buf.as_deref() { autoexposure::compute_scale(c, w, h) } else { 1.0 }
    } else {
        1.0
    };
    tf.set_input_scale(scale);

    let in_channels = color.is_some() as usize * 3
        + albedo.is_some() as usize * 3
        + normal.is_some() as usize * 3;
    let in_c = in_channels as i32;
    let snorm = normal.is_some() && color.is_none();

    let mut output_buf: Vec<f32> = vec![0.0; w * h * 3];
    let mut progress = progress;
    let total_jobs = plan.jobs.len();

    for (tile_idx, job) in plan.jobs.iter().enumerate() {
        let tile_h = plan.tile_h as usize;
        let tile_w = plan.tile_w as usize;

        // Allocate flat NCHW input buffer for this tile (N=1 implicit).
        let mut tile_input: Vec<f32> = vec![0.0; (in_c as usize) * tile_h * tile_w];

        // Helper closure for packing one 3-channel image into a contiguous
        // CHW slice of `tile_input`, applying transfer/normal correction.
        let pack = |dst: &mut [f32],
                    src: &[f32],
                    job: &TileJob,
                    apply_transfer: bool,
                    snorm: bool| {
            for ty in 0..tile_h {
                for tx in 0..tile_w {
                    // Translate tile-local (tx, ty) to image space.
                    //
                    // The actual image content for this tile occupies the
                    // tile-buffer region `[align_offset, align_offset + input.w)`
                    // along each axis. Outside that region we pad by reflecting
                    // against the input rectangle (matches `gpu_input_process.h`).
                    let raw_sx = job.input.x + tx as i32 - job.align_offset_x;
                    let raw_sy = job.input.y + ty as i32 - job.align_offset_y;
                    let sx = reflect_into(
                        raw_sx,
                        job.input.x,
                        job.input.x + job.input.w - 1,
                    );
                    let sy = reflect_into(
                        raw_sy,
                        job.input.y,
                        job.input.y + job.input.h - 1,
                    );
                    let i = (sy * w as i32 + sx) as usize * 3;
                    let r = src[i];
                    let g = src[i + 1];
                    let b = src[i + 2];

                    let (r, g, b) = if apply_transfer {
                        (tf.forward(r), tf.forward(g), tf.forward(b))
                    } else if snorm {
                        // Normals shipped as [-1, 1] need no remap; albedos clamp [0,1].
                        (r, g, b)
                    } else {
                        (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
                    };

                    let stride_c = tile_h * tile_w;
                    let off = ty * tile_w + tx;
                    dst[off] = r;
                    dst[stride_c + off] = g;
                    dst[2 * stride_c + off] = b;
                }
            }
        };

        let mut channel_offset = 0;
        let chw_size = tile_h * tile_w * 3;

        if let Some(buf) = color_buf.as_deref() {
            pack(&mut tile_input[channel_offset..channel_offset + chw_size], buf, job, true, false);
            channel_offset += chw_size;
        }
        if let Some(buf) = albedo_buf.as_deref() {
            pack(&mut tile_input[channel_offset..channel_offset + chw_size], buf, job, false, false);
            channel_offset += chw_size;
        }
        if let Some(buf) = normal_buf.as_deref() {
            pack(&mut tile_input[channel_offset..channel_offset + chw_size], buf, job, false, snorm);
            channel_offset += chw_size;
        }
        debug_assert_eq!(channel_offset, in_c as usize * tile_h * tile_w);

        // Build Burn input tensor [1, in_c, tile_h, tile_w].
        let shape = [1, in_c as usize, tile_h, tile_w];
        let input_tensor = Tensor::<B, 4>::from_data(TensorData::new(tile_input, shape), device);

        // Forward pass — Net dispatches to UNet or UNetLarge internally.
        let output_tensor = net.forward(input_tensor);

        // Pull data back to host. Output shape: [1, 3, tile_h, tile_w].
        let out_data = output_tensor.into_data();
        let out_flat: Vec<f32> = out_data.convert::<f32>().to_vec().unwrap_or_default();

        // Crop output_src_in_tile and write into output_buf with inverse transfer.
        let stride_c = tile_h * tile_w;
        let Rect { x: ox, y: oy, w: ow, h: oh } = job.output_src_in_tile;
        let Rect { x: dx, y: dy, w: _, h: _ } = job.output_dst;

        for row in 0..oh as usize {
            for col in 0..ow as usize {
                let src_idx_row = (oy as usize + row) * tile_w + (ox as usize + col);
                let r = out_flat[src_idx_row];
                let g = out_flat[stride_c + src_idx_row];
                let b = out_flat[2 * stride_c + src_idx_row];

                let (r, g, b) = if matches!(transfer_kind, TransferFunction::Linear) {
                    (r, g, b)
                } else {
                    (tf.inverse(r), tf.inverse(g), tf.inverse(b))
                };

                let dst_pixel = (dy as usize + row) * w + (dx as usize + col);
                let base = dst_pixel * 3;
                output_buf[base] = r;
                output_buf[base + 1] = g;
                output_buf[base + 2] = b;
            }
        }

        // Tile-granularity progress reporting + cooperative cancellation.
        if let Some(cb) = progress.as_deref_mut() {
            let fraction = (tile_idx + 1) as f32 / total_jobs as f32;
            if !cb(fraction) { return Err(OidnError::Cancelled); }
        }
    }

    output.write_rgb_f32(&output_buf);
    Ok(())
}

/// Reflect-clamp `x` into `[lo, hi]`. For coordinates within the rectangle
/// this is the identity; outside, we mirror against the boundary so the
/// padding pixels look like the image content (matching OIDN's input border
/// handling).
#[inline]
fn reflect_into(x: i32, lo: i32, hi: i32) -> i32 {
    if hi <= lo { return lo; }
    let mut v = x;
    while v < lo || v > hi {
        if v < lo { v = 2 * lo - v; }
        if v > hi { v = 2 * hi - v; }
    }
    v
}
