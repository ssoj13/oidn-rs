//! UNet inference pipeline (burn 0.22 — dynamic, device-selected backend).
//!
//! Two entry points share a single tensor-native core:
//!
//! - [`run_tensors`] is the primary impl. Inputs and outputs are
//!   `[1, 3, H, W]` (NCHW) Burn tensors that already live on the target
//!   device. HDR exposure reads back reduced scalars; explicitly enabled tensor
//!   diagnostics read back pixels. Squarebob uses this numerical core.
//! - [`run`] is the legacy `Image<'_>` ↔ `ImageMut<'_>` entry point used
//!   by the CLI and tests. It does the host upload / download bookends
//!   around `run_tensors` so callers that don't have a wgpu pipeline
//!   keep working.

use burn::tensor::{Device, Tensor};
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

/// Shared execution semantics for both image and tensor frontends.
#[derive(Clone, Copy, Debug)]
pub struct RunOptions {
    pub transfer: TransferFunction,
    pub hdr: bool,
    pub signed: bool,
    pub input_scale: Option<f32>,
    /// Optional stronger policy; reference NaN sanitation remains mandatory.
    pub sanitize_nonfinite: bool,
    pub output_channels: usize,
}

/// Tensor-native UNet forward pass. All inputs (`color`, `albedo`,
/// `normal`) are `[1, 3, H, W]` (NCHW) `f32` tensors on `device`. The
/// returned tensor has the same shape. No data crosses to host except
/// the two scalars produced by HDR autoexposure (and only when both
/// `hdr` is true and `user_input_scale` is `None`).
///
/// `output_w` / `output_h` must match the input tensor dimensions (and
/// the tile plan's image size).
#[allow(clippy::too_many_arguments)]
pub fn run_tensors(
    net: &Net,
    device: &Device,
    plan: &TilePlan,
    color: Option<Tensor<4>>,
    albedo: Option<Tensor<4>>,
    normal: Option<Tensor<4>>,
    output_w: usize,
    output_h: usize,
    options: RunOptions,
    mut progress: Option<&mut ProgressFn<'_>>,
) -> Result<Tensor<4>, OidnError> {
    let RunOptions {
        transfer: transfer_kind,
        hdr,
        signed,
        input_scale: user_input_scale,
        sanitize_nonfinite: nan_to_zero,
        output_channels,
    } = options;
    if !(1..=3).contains(&output_channels) || (signed && hdr) {
        return Err(OidnError::InvalidArgument(
            "invalid output channels or signed HDR mode",
        ));
    }
    super::validate_execution(output_w, output_h, user_input_scale)?;
    crate::image::validate_dimensions(output_w, output_h, output_channels)?;
    if output_w == 0 || output_h == 0 {
        return Err(OidnError::InvalidArgument("empty execution geometry"));
    }
    plan.validate(output_w, output_h)?;
    let inputs = [&color, &albedo, &normal];
    let roles = inputs.iter().filter(|input| input.is_some()).count();
    if roles == 0 || net.in_channels() != roles * 3 || net.out_channels() != 3 {
        return Err(OidnError::InvalidArgument("model/input channel mismatch"));
    }
    for tensor in inputs.into_iter().flatten() {
        if tensor.device() != *device {
            return Err(OidnError::InvalidArgument(
                "input tensor must be on the selected device",
            ));
        }
        if tensor.dims() != [1, 3, output_h, output_w] {
            return Err(OidnError::InvalidArgument("input must match [1,3,H,W]"));
        }
    }
    // Optional stronger input policy executes once before exposure and packing.
    // Kernel-local native NaN-only sanitation after scale remains mandatory.
    let sanitize = |t: Tensor<4>| -> Tensor<4> {
        if nan_to_zero {
            let finite_mask = t.clone().is_finite();
            let zeros: Tensor<4> = Tensor::zeros(t.dims(), &t.device());
            t.mask_where(finite_mask.bool_not(), zeros)
        } else {
            t
        }
    };
    let color = color.map(sanitize);
    let albedo = albedo.map(sanitize);
    let normal = normal.map(sanitize);
    let w = output_w;
    let h = output_h;
    log::debug!(
        "unet_runner::run_tensors() {}x{} color={} albedo={} normal={} transfer={:?} hdr={} tiles={}",
        w,
        h,
        color.is_some(),
        albedo.is_some(),
        normal.is_some(),
        transfer_kind,
        hdr,
        plan.jobs.len()
    );

    // Autoexposure — when HDR and no user override, use the GPU variant
    // so we don't have to drag the colour buffer back to host. Only two
    // scalars (`sum_log` + `count`) end up readback.
    let mut tf = TransferState::new(transfer_kind);
    let scale = if let Some(s) = user_input_scale {
        s
    } else if hdr {
        match color.as_ref() {
            Some(c) => autoexposure::compute_scale_tensor(c.clone())?,
            None => 1.0,
        }
    } else {
        1.0
    };
    if !scale.is_finite() || scale <= 0.0 || !scale.recip().is_finite() {
        return Err(OidnError::InvalidArgument(
            "input scale must be positive and finite",
        ));
    }
    tf.set_input_scale(scale);
    log::debug!(
        "unet_runner: autoexposure scale={:.6} (hdr={}, user_scale={:?})",
        scale,
        hdr,
        user_input_scale
    );

    let trace_tensors = tensor_diagnostics_enabled();
    if trace_tensors {
        if let Some(t) = color.as_ref() {
            log_tensor_stats("unet input/color_chw", t);
        }
        if let Some(t) = albedo.as_ref() {
            log_tensor_stats("unet input/albedo_chw", t);
        }
        if let Some(t) = normal.as_ref() {
            log_tensor_stats("unet input/normal_chw", t);
        }
    }

    // Auxiliary normal channel matches reference OIDN getNormal()
    // (`devices/gpu/gpu_input_process.h:77`, `devices/cpu/cpu_input_process.isph:65-76`):
    // unconditional clamp(-1, 1) then linear remap to [0, 1]. The `snorm`
    // flag in the reference only gates channel 0 (the primary input for
    // directional/normal-only filters); auxiliary normals are *always*
    // fed to the network as unsigned [0, 1]. Both prior Rust rules (raw
    // pass-through and clamp(0,1)) diverged from the network's training
    // contract.
    log::debug!(
        "unet_runner: input contract color={} albedo={} normal={} tensor_stats={}",
        color.is_some(),
        albedo.is_some(),
        normal.is_some(),
        trace_tensors,
    );

    // Output accumulator: starts at zeros, slice_assign per tile.
    let mut accum: Tensor<4> = Tensor::zeros([1, 3, h, w], device);

    let auxiliary_transfer = TransferState::new(TransferFunction::Linear);
    let total_jobs = plan.jobs.len();
    let tile_w = plan.tile_w as usize;
    let tile_h = plan.tile_h as usize;

    for (tile_idx, job) in plan.jobs.iter().enumerate() {
        // 1. Pull the source rectangle out of each input tensor and place
        //    it into a tile-shaped zero buffer. `align_offset_{x,y}` is
        //    the leading offset on the left/top side. The padded region
        //    is filled with zeros to match the reference's input kernel
        //    (`_ref/oidn/devices/cpu/cpu_input_process.isph:88-93,120-125`),
        //    which writes zero outside the source rect rather than
        //    reflecting boundary pixels.
        let src_x = job.input.x as usize;
        let src_y = job.input.y as usize;
        let src_w = job.input.w as usize;
        let src_h = job.input.h as usize;
        let pad_left = job.align_offset_x as usize;
        let pad_top = job.align_offset_y as usize;
        debug_assert!(pad_left + src_w <= tile_w);
        debug_assert!(pad_top + src_h <= tile_h);

        // Transform only real source values. Padding stays exactly zero in
        // every role, including signed normals whose valid zero maps to 0.5.
        let mut channel_parts = Vec::with_capacity(3);
        for (index, src) in [color.as_ref(), albedo.as_ref(), normal.as_ref()]
            .into_iter()
            .enumerate()
        {
            let Some(src) = src else { continue };
            let rect = src
                .clone()
                .slice([0..1, 0..3, src_y..src_y + src_h, src_x..src_x + src_w]);
            let primary = channel_parts.is_empty();
            let processed = gpu_ops::preprocess_input(
                rect,
                if primary { scale } else { 1.0 },
                primary && hdr,
                if primary { signed } else { index == 2 },
                if primary { &tf } else { &auxiliary_transfer },
            );
            let dst = Tensor::<4>::zeros([1, 3, tile_h, tile_w], device);
            channel_parts.push(dst.slice_assign(
                [
                    0..1,
                    0..3,
                    pad_top..pad_top + src_h,
                    pad_left..pad_left + src_w,
                ],
                processed,
            ));
        }

        // 2. Concat along channel dim → [1, in_c, tile_h, tile_w].
        let input_tensor: Tensor<4> = Tensor::cat(channel_parts, 1);

        // 3. Forward.
        let output_tensor: Tensor<4> = net.forward(input_tensor);

        // 4. Postprocess (nan_to_zero -> clamp -> inverse transfer ->
        //    [ldr clamp] -> *output_scale) then crop + slice_assign.
        //    Handles Linear correctly: the inverse curve is identity and
        //    the surrounding clamp/scale still apply per reference.
        let post = gpu_ops::postprocess_color(
            output_tensor,
            &tf,
            hdr,
            signed,
            tf.output_scale,
            output_channels,
        );

        let Rect {
            x: ox,
            y: oy,
            w: ow,
            h: oh,
        } = job.output_src_in_tile;
        let Rect {
            x: dx,
            y: dy,
            w: _,
            h: _,
        } = job.output_dst;
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

        if let Some(cb) = progress.as_deref_mut() {
            let fraction = (tile_idx + 1) as f32 / total_jobs as f32;
            if !cb(fraction) {
                return Err(OidnError::Cancelled);
            }
        }
    }

    if trace_tensors {
        log_tensor_stats("unet output/accum_chw", &accum);
    }

    Ok(accum)
}

/// Legacy `Image<'_>` entry point. Decodes inputs to HWC `f32`, uploads
/// to NCHW tensors, runs [`run_tensors`], pulls the accumulator back to
/// host, writes to `output`.
///
/// The CLI and the test fixtures still use this. Squarebob's hot path
/// goes through [`run_tensors`] directly via the bridge in
/// `pt-denoise-oidn` and the tensor-native API on
/// [`RtFilter`](crate::filters::rt::RtFilter).
#[allow(clippy::too_many_arguments)]
pub fn run(
    net: &Net,
    device: &Device,
    plan: &TilePlan,
    color: Option<&Image<'_>>,
    albedo: Option<&Image<'_>>,
    normal: Option<&Image<'_>>,
    output: &mut ImageMut<'_>,
    options: RunOptions,
    progress: Option<&mut ProgressFn<'_>>,
) -> Result<(), OidnError> {
    let w = output.width;
    let h = output.height;

    output.validate()?;
    super::validate_execution(w, h, options.input_scale)?;
    plan.validate(w, h)?;
    for image in [color, albedo, normal].into_iter().flatten() {
        image.validate()?;
        if image.width != w || image.height != h {
            return Err(OidnError::Inconsistent("input/output dimensions"));
        }
    }
    let options = RunOptions {
        output_channels: output.format.channels(),
        ..options
    };
    let color_buf = color.map(|img| img.to_rgb_f32()).transpose()?;
    let albedo_buf = albedo.map(|img| img.to_rgb_f32()).transpose()?;
    let normal_buf = normal.map(|img| img.to_rgb_f32()).transpose()?;

    if let Some(c) = color_buf.as_deref() {
        let (cmin, cmax, cmean) = quick_stats(c);
        log::debug!("unet input color stats: min={cmin:.4} max={cmax:.4} mean={cmean:.4}");
    }

    let color_t = color_buf
        .as_deref()
        .map(|buf| upload_hwc_as_chw_tensor(buf, w, h, device))
        .transpose()?;
    let albedo_t = albedo_buf
        .as_deref()
        .map(|buf| upload_hwc_as_chw_tensor(buf, w, h, device))
        .transpose()?;
    let normal_t = normal_buf
        .as_deref()
        .map(|buf| upload_hwc_as_chw_tensor(buf, w, h, device))
        .transpose()?;

    let accum = run_tensors(
        net, device, plan, color_t, albedo_t, normal_t, w, h, options, progress,
    )?;

    let (chw_vec, dims) = image_tensor::tensor_to_chw_vec(accum)?;
    debug_assert_eq!(dims, [1, 3, h, w]);
    let hwc_vec = image_tensor::chw_to_hwc(&chw_vec, 3, h, w)?;

    let (omin_post, omax_post, omean_post) = quick_stats(&hwc_vec);
    log::debug!(
        "unet_runner output (after inverse transfer): min={omin_post:.4} max={omax_post:.4} mean={omean_post:.4}"
    );
    output.write_rgb_f32(&hwc_vec)?;
    Ok(())
}

/// Stage a flat HWC `f32` slice as a `[1, 3, H, W]` Burn tensor on `device`.
fn upload_hwc_as_chw_tensor(
    buf_hwc: &[f32],
    width: usize,
    height: usize,
    device: &Device,
) -> Result<Tensor<4>, OidnError> {
    let chw = image_tensor::hwc_to_chw(buf_hwc, 3, height, width)?;
    image_tensor::chw_vec_to_tensor(chw, 3, height, width, device)
}

fn quick_stats(data: &[f32]) -> (f32, f32, f32) {
    let stats = TensorStats::from_slice(data);
    (stats.min, stats.max, stats.mean)
}

fn tensor_diagnostics_enabled() -> bool {
    if log::log_enabled!(log::Level::Trace) {
        return true;
    }
    std::env::var("OIDN_TRACE_TENSORS")
        .map(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn log_tensor_stats(label: &str, tensor: &Tensor<4>) {
    let dims = tensor.dims();
    match tensor.clone().into_data().convert::<f32>().to_vec::<f32>() {
        Ok(data) => {
            let stats = TensorStats::from_slice(&data);
            log::trace!(
                "OIDN tensor stats {label}: dims={:?} finite={} nan={} inf={} neg={} gt1={} min={:.6} max={:.6} mean={:.6}",
                dims,
                stats.finite,
                stats.nan,
                stats.inf,
                stats.neg,
                stats.gt_one,
                stats.min,
                stats.max,
                stats.mean,
            );
        }
        Err(err) => {
            log::warn!("OIDN tensor stats {label}: dims={dims:?} readback failed: {err:?}");
        }
    }
}

struct TensorStats {
    finite: usize,
    nan: usize,
    inf: usize,
    neg: usize,
    gt_one: usize,
    min: f32,
    max: f32,
    mean: f32,
}

impl TensorStats {
    fn from_slice(data: &[f32]) -> Self {
        let mut finite = 0usize;
        let mut nan = 0usize;
        let mut inf = 0usize;
        let mut neg = 0usize;
        let mut gt_one = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;

        for &v in data {
            if v.is_nan() {
                nan += 1;
            } else if v.is_infinite() {
                inf += 1;
            } else {
                finite += 1;
                if v < 0.0 {
                    neg += 1;
                }
                if v > 1.0 {
                    gt_one += 1;
                }
                min = min.min(v);
                max = max.max(v);
                sum += v as f64;
            }
        }

        if finite == 0 {
            min = 0.0;
            max = 0.0;
        }

        Self {
            finite,
            nan,
            inf,
            neg,
            gt_one,
            min,
            max,
            mean: if finite == 0 {
                0.0
            } else {
                (sum / finite as f64) as f32
            },
        }
    }
}
