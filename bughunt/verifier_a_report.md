# Verifier A Report — Mechnikov fixes (2026-05-21)

## Verdicts per fix

- **H5 (UNet final ReLU)** — PASS. `crates/oidn-model/src/unet.rs:131` ends in `relu(self.dec_conv0.forward(x))`.
- **H1 (reflect→zero pad)** — PASS. Zero allocation + `slice_assign` at `crates/oidn-rs/src/filters/unet_runner.rs:155-172`; no `reflect_pad_2d` / `pad_axis_reflect` / `slice_axis` anywhere in `crates/oidn-rs/src` (grep clean).
- **H2 preprocess_input** — PASS. Op order at `gpu_ops.rs:45-62` matches `_ref/oidn/devices/cpu/cpu_input_process.isph:31-51`.
- **H3 postprocess_color** — PASS. Op order at `gpu_ops.rs:69-88` matches `_ref/oidn/devices/cpu/cpu_output_process.isph:37-69`.
- **H4 (scale removed from transfer fns)** — PASS. `apply_transfer_forward` (`gpu_ops.rs:95-105`) and `apply_transfer_inverse` (`gpu_ops.rs:109-121`) contain no `input_scale`/`output_scale` multiplication; only `norm_scale`/`rcp_norm_scale`, which is correct.
- **Color constants dedup** — PASS. `color.rs:96-124` declares `SRGB_*`, `PU_*`, `Y_MAX` as `pub(crate) const`; `gpu_ops.rs:19-22` imports them; grep for `const SRGB_|PU_|Y_MAX` in `gpu_ops.rs` returns nothing.
- **Call sites** — PASS. `unet_runner.rs:180` calls `preprocess_input` for colour, `:203` calls `postprocess_color`; albedo branch clamps `[0,1]` at `:184`; normal branch clamps `[-1,1]` then `*0.5 + 0.5` at `:189`.
- **Build** — PASS. `cargo check -p oidn-model` and `cargo check -p oidn-rs --lib` both finish clean (no errors, no warnings printed).

## Detailed findings

| id  | sev | file:line | observed | required |
|-----|-----|-----------|----------|----------|
| H5  | OK  | crates/oidn-model/src/unet.rs:131 | `relu(self.dec_conv0.forward(x))` | same |
| H1a | OK  | crates/oidn-rs/src/filters/unet_runner.rs:162 | `Tensor::zeros([1, channels, tile_h, tile_w], device)` | zero alloc |
| H1b | OK  | crates/oidn-rs/src/filters/unet_runner.rs:163-172 | `dst.slice_assign(...)` from source rect | same |
| H1c | OK  | crates/oidn-rs/src/gpu_ops.rs (whole file) | no `reflect_pad_2d`/`pad_axis_reflect`/`slice_axis` symbols | deleted |
| H2a | OK  | gpu_ops.rs:52 | `nan_to_zero(color)` | (a) |
| H2b | OK  | gpu_ops.rs:53 | `t.mul_scalar(input_scale)` | (b) |
| H2c | OK  | gpu_ops.rs:54-56 | `clamp(snorm ? -1 : 0, hdr ? f32::MAX : 1)` | (c) |
| H2d | OK  | gpu_ops.rs:60 | `if snorm { t.mul_scalar(0.5).add_scalar(0.5) }` — matches ref `value*0.5+0.5` (active branch, not a stub) | (d) |
| H2e | OK  | gpu_ops.rs:61 | `apply_transfer_forward(t, transfer)` | (e) |
| H3a | OK  | gpu_ops.rs:76 | `nan_to_zero(network_output)` | (a) |
| H3b | OK  | gpu_ops.rs:77 | `clamp(0.0, f32::MAX)` | (b) |
| H3c | OK  | gpu_ops.rs:78 | `apply_transfer_inverse(t, transfer)` | (c) |
| H3d | OK  | gpu_ops.rs:81-85 | `t.mul_scalar(2).sub_scalar(1).clamp_min(-1)` when snorm | (d) |
| H3e | OK  | gpu_ops.rs:86 | `if !hdr && !snorm { t.clamp_max(1.0) }` | (e) |
| H3f | OK  | gpu_ops.rs:87 | `t.mul_scalar(output_scale)` | (f) |
| H4f | OK  | gpu_ops.rs:95-105 | forward: only `norm_scale` multiplications, no `input_scale` | scale-free curve |
| H4i | OK  | gpu_ops.rs:109-121 | inverse: only `rcp_norm_scale`, no `output_scale` | scale-free curve |
| Dedup | OK | gpu_ops.rs:19-22 | `use crate::color::{PU_*, SRGB_*, ...}` | imported |
| Dedup | OK | color.rs:96-124 | `pub(crate) const SRGB_*`, `pub(crate) const PU_*`, `pub const Y_MAX` | visibility |
| Call  | OK | unet_runner.rs:180 | `gpu_ops::preprocess_input(padded, scale, hdr, false, &tf)` | colour preproc |
| Call  | OK | unet_runner.rs:184 | `padded.clamp(0.0, 1.0)` | albedo `[0,1]` |
| Call  | OK | unet_runner.rs:189 | `padded.clamp(-1.0, 1.0).mul_scalar(0.5).add_scalar(0.5)` | normal remap |
| Call  | OK | unet_runner.rs:203 | `gpu_ops::postprocess_color(output_tensor, &tf, hdr, false, tf.output_scale)` | colour postproc |
| Build | OK | -                                | `cargo check -p oidn-model` clean; `cargo check -p oidn-rs --lib` clean | both compile |

## Notes (non-blocking)

- `nan_to_zero` runs twice on the colour path (once upstream in `run_tensors:61-69`, once inside `preprocess_input`). Documented in the helper docstring (`gpu_ops.rs:42-44`) and explicitly noted in `unet_runner.rs:56-60`. Cheap, intentional — reference parity (every reference kernel does its own `nan_to_zero`). Not a defect.
- `preprocess_input` is called with `snorm=false` and the snorm branch is therefore dead at all current call sites; the implementation is nonetheless live code (not a stub), matching `cpu_input_process.isph:41-45`. Comment at `gpu_ops.rs:57-59` slightly misleading where it says "no-op for now" — the branch is actually live. LOW (doc-only) — leave for follow-up.
- `postprocess_color` snorm branch (`gpu_ops.rs:81-85`) is also live and reference-faithful. Same comment-vs-code mismatch at `:79-80`. LOW.
- `TransferState::forward`/`inverse` (the scalar CPU helpers at `color.rs:62-81`) still embed `input_scale`/`output_scale`. They are used by host-side scalar tests, not by the GPU path. No drift introduced. Out of scope for this verification.

## Overall verdict

**ACCEPT.** All six items match spec, op orders mirror the reference ISPH kernels exactly, dedup is complete, both target crates compile clean. Open items above are LOW-severity comment cleanups only.
