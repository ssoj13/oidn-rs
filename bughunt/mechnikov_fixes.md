# Mechnikov fixes report

## Fixes applied

- **H5** — `dec_conv0` wrapped in `relu(...)` as last expression of `UNet::forward`.
- **H1** — Per-tile `reflect_pad_2d` replaced with zero-pad via
  `Tensor::zeros([1, C, tile_h, tile_w], device)` + `slice_assign` of the
  source rect at `(pad_top, pad_left)`. `gpu_ops::reflect_pad_2d` and the
  `pad_axis_reflect` / `slice_axis` helpers deleted entirely (no callers
  remain). Old reflect-pad tests removed with them.
- **H2 + H3 + H4** — New `pub(crate)` helpers in `gpu_ops`:
  - `preprocess_input`: `nan_to_zero -> *input_scale -> clamp(lo, hi) ->
    [snorm remap] -> forward transfer`. `snorm` branch documented as
    unused-by-current-filters stub.
  - `postprocess_color`: `nan_to_zero -> clamp(0, +inf) -> inverse
    transfer -> [snorm demap] -> [ldr clamp] -> *output_scale`.
  - `apply_transfer_forward` / `apply_transfer_inverse` no longer
    multiply by `input_scale` / `output_scale` internally — scale ops
    moved into the wrapping helpers in reference order. `PU`/`Log`
    `norm_scale` stays inside the curve helpers (it's the curve's own
    normalisation, not autoexposure).
  - Linear branch goes through `postprocess_color` unconditionally; the
    `matches!(Linear) { ... } else { ... }` switch at the previous call
    site is gone.
  - Internal `nan_to_zero<B>` in `gpu_ops` is a single canonical impl
    reused by both wrappers; the whole-tensor sanitisation in
    `run_tensors` remains for upstream user-supplied non-finite samples
    and is now documented as the canonical sanitisation point.
- **Dedup** — sRGB / PU constants in `color.rs` promoted to
  `pub(crate)` and imported by `gpu_ops`. Duplicate `const` block at
  the top of `gpu_ops.rs` deleted.

## Files changed

- `crates/oidn-model/src/unet.rs` — 1 line (H5).
- `crates/oidn-rs/src/color.rs` — 18 lines (visibility change on 18
  constants, no behaviour change).
- `crates/oidn-rs/src/gpu_ops.rs` — rewritten (~280 → ~290 lines). Lost
  ~60 lines of reflect-pad helpers, gained ~70 lines of preprocess/
  postprocess wrappers. Net ≈ +10 lines.
- `crates/oidn-rs/src/filters/unet_runner.rs` — ~50 lines around the
  tile loop. Inner `zero_pad` closure introduced; `apply_transfer_*`
  call sites replaced with `preprocess_input` / `postprocess_color`.

## `cargo check` results

- `cargo check -p oidn-model` — **pass** (clean, 30.53s).
- `cargo check -p oidn-rs` (library only) — **pass** (clean, 0.79s
  incremental).
- `cargo check -p oidn-rs --tests` — **fail**, but failures are in
  out-of-scope test files (`tests/e2e_wgpu.rs`, `tests/e2e_ldr.rs`,
  `tests/multi_tile_wgpu.rs`) referencing `WgpuBackend` /
  `WgpuDevice` symbols that are gated behind features owned by other
  agents (`device.rs`, `prelude.rs`, `lib.rs`). Library code in scope
  compiles clean; no errors traceable to fixes H1/H2/H3/H4/H5/dedup.

## Open follow-ups

- The wgpu test compilation errors above pre-exist this branch's
  bughunt scope. They live in files I'm not permitted to edit
  (`tests/*`, `device.rs`, `prelude.rs`, `lib.rs`). Hand off to the
  agent owning prelude/device.
- `snorm` branches in both wrappers are no-op stubs — wire them up
  when a directional filter actually needs `value = value * 0.5 + 0.5`
  on a signed primary input. Documented inline.
- The `pu_forward_cpu_parity_ndarray` test compares against
  `pu_forward(s) * norm_scale` now that scaling moved out of the curve
  helper; matches the new contract.
