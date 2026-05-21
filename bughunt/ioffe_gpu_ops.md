# Ioffe — GPU Ops & Input/Output Process Parity Audit

- Agent: Ioffe (read-only)
- Date: 2026-05-21
- Scope: oidn-rs vs Intel OIDN reference (`devices/cpu/cpu_input_process*.*`, `devices/gpu/gpu_input_process.h`, `core/{input,output}_process.*`, `core/tensor_layout.h`).
- Method: source inspection only (no edits, no builds, no test runs).

## Verdict

Rust **PARTIAL** parity. Functional/numeric matching for the common HDR f32 path is achievable and explicitly tested for piecewise transfer curves and reflect padding. Two architectural mismatches dominate:

1. **Layout** — Rust never touches the blocked `Chw{8,16,32}c` / `OIhw…` layouts the reference uses on GPU. Burn always sees plain NCHW (`chw`). For the wgpu backend this is a deliberate simplification (cubecl chooses its own layout internally), so this is "parity by abstraction", not by exact memory layout. Acceptable for correctness on this backend, but loses the blocked-vec throughput the reference exploits and means any GPU op author touching raw buffers will not find a 1:1 layout map.
2. **Padding semantics** — reference `cpu_input_process.isph` and `gpu_input_process.h` **zero-pad** outside the source tile (kernels write `0` for `wDst < tile.wDstBegin`, etc.). The Rust pipeline **reflect-pads** every input (`gpu_ops::reflect_pad_2d` at `unet_runner.rs:167,174,181`). This is a real semantic divergence; it cannot be hand-waved as equivalent at tile borders, see "Issue I-1" below.

The HDR/transfer/autoexposure math, NaN sanitization, concat order, normal/albedo remap, and inverse transfer/sanitization are all faithfully ported.

## Per-kernel mapping

| Rust kernel / function | Reference kernel | Status | Location |
|---|---|---|---|
| `unet_runner::run_tensors` per-tile loop | `InputProcess::submitKernels` + `OutputProcess::submitKernels` (chained via `Graph`) | partial — flow matches; padding diverges; layout abstracted | rust `crates/oidn-rs/src/filters/unet_runner.rs:145-222` vs ref `devices/cpu/cpu_input_process.cpp:19-46`, `cpu_input_process.isph:78-136`, `devices/gpu/gpu_input_process.h:82-176` |
| `gpu_ops::apply_transfer_forward` (Linear/sRGB/PU/Log) | `TransferFunction.forward` from `core/color.*` invoked inside `getInput` (`cpu_input_process.isph:48`, `gpu_input_process.h:54`) | match — mask_where cascade replicates piecewise; round-trip test rtol < 1e-4 | rust `gpu_ops.rs:130-218` |
| `gpu_ops::apply_transfer_inverse` | `getOutputScale` + `transferFunc.inverse` in `cpu_output_process.isph:35-67` | match — same three-region branch | rust `gpu_ops.rs:145-212` |
| `gpu_ops::reflect_pad_2d` | reference does **not** reflect-pad inputs — it zero-pads | mismatch (see Issue I-1) | rust `gpu_ops.rs:58-107` vs ref `cpu_input_process.isph:89-93,121-125` |
| color slice + autoexposure | `getInput`: `value * inputScale → clamp(0, hdr ? +inf : 1) → forward` | match for HDR + non-snorm | `unet_runner.rs:92-103,168` vs `cpu_input_process.isph:31-51`; clamp+nan_to_zero done in Rust via `is_finite()` mask at `unet_runner.rs:64-76` |
| albedo branch `padded.clamp(0,1)` | `getAlbedo` = `clamp(nan_to_zero, 0, 1)` | match | `unet_runner.rs:175` vs `cpu_input_process.isph:54-62` |
| normal branch `clamp(-1,1).*0.5+0.5` | `getNormal` = `clamp(nan_to_zero, -1,1) * 0.5 + 0.5` | match (NaNs handled up-front by `sanitize` closure) | `unet_runner.rs:182-184` vs `cpu_input_process.isph:65-76` |
| concat order: color → albedo → normal | `Tensor_set3` writes to channels 0, 3, 6 (`isph:101-113`); `GPUInputProcessKernel` fills `values[0..3], [3..6], [6..9]` (`gpu_input_process.h:99-116`) | match | `unet_runner.rs:163-188` |
| `nan_to_zero` policy | reference always sanitizes via `nan_to_zero(value)` inside `getInput/getAlbedo/getNormal` | match when `nan_to_zero=true`; **mismatch** when caller passes `false` — reference has no opt-out (see Issue I-2) | `unet_runner.rs:64-76` |
| Output: detile + inverse transfer + write | `OutputProcess` per-row: `Tensor_get3 → clamp(nan_to_zero, 0, +inf) → inverse → snorm? → !hdr?min(1) → *outputScale → Image_set3` | mostly match; **missing** post-inverse `clamp(0, +inf)` sanitization on Rust path (see Issue I-3) | rust `unet_runner.rs:194-216` vs ref `cpu_output_process.isph:37-69` |
| `image_tensor::chw_to_hwc` / `hwc_to_chw` | reference relies on `ImageAccessor`/`TensorAccessor` directly, no host re-pack | match (semantically); host roundtrip is a perf overhead specific to legacy `run()` entry point | `image_tensor.rs:20-52` |
| GPU input layout (Rust): plain `[1,C,H,W]` f32 via Burn | reference: `Chw{8,16,32}c` or `hwc` blocked, per `tensorBlockC` engine pref | **architectural divergence** (acceptable — Burn handles its own layout) | `image_tensor.rs:74-83` vs `core/tensor_layout.h:11-33`, ref `input_process.cpp:17-23` |
| wgpu compute pipelines, dispatch dims, workgroup | Rust delegates entirely to cubecl-wgpu (no hand-written wgsl in these files) | n/a — there are no Rust-authored compute shaders here to compare against `gpu_input_process.h:82` (which uses subgroup transpose). Workgroup/subgroup tuning is cubecl-internal | n/a |

## Per-issue table

| # | Severity | Component | Issue | Where |
|---|---|---|---|---|
| I-1 | **High** | input padding | Rust **reflect-pads** every tile out to `tile_h × tile_w`; reference **zero-pads**. For non-edge tiles `pad_*` are 0 → no diff; for edge tiles the network sees mirrored content vs the trained-on zeros. Tile blending hides most artifacts but border pixels can differ measurably from reference output. | rust `gpu_ops.rs:46-107`, `unet_runner.rs:167,174,181` ↔ ref `cpu_input_process.isph:89-93,116-125,129-135` (zero-pad), `gpu_input_process.h:91-118` (default-zero `values[]`) |
| I-2 | Medium | NaN policy | `nan_to_zero=false` skips sanitation entirely (`unet_runner.rs:65-75`). Reference always runs `nan_to_zero` inside `getInput/getAlbedo/getNormal` and inside `OutputProcessKernel_run`. The Rust opt-out is a public API knob with no analog in OIDN. | `unet_runner.rs:64-76` ↔ `cpu_input_process.isph:39,59,70`, `cpu_output_process.isph:46` |
| I-3 | Medium | output sanitization | Rust path does **not** clamp the network output to `[0, +∞)` (or `min(1.f)` when `!hdr`) before/after inverse transfer; only relies on `is_finite` mask on **inputs** at the top of `run_tensors`. Reference clamps **on the way out** (`cpu_output_process.isph:46` + `cpu_output_process.isph:62-63`). Negative model outputs can leak into PU-inverse and become huge. | `unet_runner.rs:193-198` ↔ `cpu_output_process.isph:43-67` |
| I-4 | Medium | snorm path | `snorm` (signed-normalized colour input/output) is **not implemented** in Rust. Reference branches at `cpu_input_process.isph:41-46` and `cpu_output_process.isph:56-61`. Rust treats colour as if `snorm=false` always. Only relevant for `rtlightmap_dir` (directional lightmap) — tests at `e2e_wgpu.rs:367-400` exercise this filter; presumably snorm is folded into the filter type rather than the IO process here. Verify against `RtLightmapFilter` builder. | reference `cpu_input_process.isph:41-46` only; Rust has no equivalent flag in `run_tensors` |
| I-5 | Medium | output single-channel averaging | When `dst.C == 1` the reference averages the three channels (`cpu_output_process.isph:52-53`). Rust always writes 3 channels (`unet_runner.rs:139` allocates `[1,3,h,w]`; `run()` always calls `output.write_rgb_f32`). For 1-channel output filters this would silently differ. May be unused in current model zoo; flag for verification. | ref `cpu_output_process.isph:52-53` |
| I-6 | Low | `getInput` clamp range | Reference clamps `value` to `[0, +inf)` (HDR) or `[0, 1]` (LDR) **before** `forward` (`cpu_input_process.isph:39`). Rust's `gpu_ops::apply_transfer_forward` does `mul(input_scale)` then jumps straight to the transfer curve — there is no pre-clamp. PU forward has `clamp_min(0.0)` inside the mid branch (`gpu_ops.rs:188`); sRGB forward likewise (`gpu_ops.rs:164`). Negative HDR samples therefore take a different branch (Log/Linear paths don't clamp at all). | rust `gpu_ops.rs:130-141` ↔ ref `cpu_input_process.isph:39` |
| I-7 | Low | tensor channel padding | Reference rounds `C` up to `tensorBlockC` and **zero-fills** the padding lanes (`input_process.cpp:17-23`, `cpu_input_process.isph:116-117`, `gpu_input_process.h:91`). Rust never pads channel count — Burn convs declare in-channels at config time. Affects only blocked-vec backends; not an issue on Rust's NCHW path. | ref `input_process.cpp:17-23` |
| I-8 | Low | host roundtrip in `run()` | Legacy `run()` does `Image → Vec<f32> HWC → hwc_to_chw → upload`, then `tensor_to_chw_vec → chw_to_hwc → write`. Two host copies per filter call. `run_tensors` avoids it; the public CLI / tests still use `run()` (`unet_runner.rs:257-287`). | `unet_runner.rs:240-289` |
| I-9 | Low | tile_w/tile_h vs reference dst rounding | Reference's `dstPaddedDims` rounds **channels** only, not H/W. Rust pads H/W via reflect to `tile_h × tile_w`. The two are orthogonal — Rust's padding is tile-receptive-field padding, reference's is channel-vec-lane padding. Same name, different jobs. | rust `unet_runner.rs:156-159` vs ref `input_process.cpp:17-23` |
| I-10 | Info | tile transfer round-trip | Burn/`Tensor::slice` + `slice_assign` keeps every tile on device for the wgpu backend; no host roundtrip per tile (only the legacy `run()` does host bookends). Compute-pipeline correctness is delegated to cubecl. Confirmed by tests at `e2e_wgpu.rs:69-211`. | `unet_runner.rs:145-222` |
| I-11 | Info | UNet layer order | Rust `Unet::forward` and `UnetLarge::forward` reproduce the reference `build_graph` ordering exactly: `enc_conv0/enc_conv1 → pool1 → enc_conv2 → pool2 → enc_conv3 → pool3 → enc_conv4 → pool4 → enc_conv5{a,b} → upsample → concat(pool3) → dec_conv4{a,b} → upsample → concat(pool2) → dec_conv3 → … → concat(input) → dec_conv1 → dec_conv0`. Skip from input/inputProcess is preserved (`unet.rs:73` `dc2b + ic`, `unet_runner.rs:111` chain). | rust `crates/oidn-model/src/unet.rs:32-119`, `unet_large.rs:5-135` ↔ ref `unet_filter.cpp:470-528` |
| I-12 | Info | residual handling | Reference has no residual adds — only concat-skips. Rust mirrors this; no extra residual was introduced. | n/a |

## Performance / precision concerns

- **Reflect vs zero pad cost.** Reflect adds `flip + slice + cat` per axis per input tensor per tile — three tensor ops per padded edge, executed even when `pad_* == 0` is detected at higher level but not when one side is non-zero. Reference zero-pad is a memset on the dst. Both correctness *and* throughput would improve by using zeros (since cubecl `Tensor::zeros` of the full `[1,C,tile_h,tile_w]` then `slice_assign` of the source rect is one allocation + one copy).
- **Host roundtrip in `quick_stats` / `log_tensor_stats` / `TensorStats::from_slice`** (`unet_runner.rs:302-409`) pull the entire tensor to host when `OIDN_TRACE_TENSORS=1` or `log::Level::Trace`. This is fine for debug, but the call sites are guarded only by `if trace_tensors`. Verify the trace flag check isn't accidentally true in release/bench runs.
- **Output channel sanitization missing** (Issue I-3). PU/Log inverse can blow up unbounded for negative network outputs because there is no `max(0, x)` before `apply_transfer_inverse`. Reference clamps to `[0, +inf)` first.
- **f16 path.** Rust's tensor kernels are `B: Backend` generic and rely on `mul_scalar`, `powf_scalar`, `log`, `exp`, `clamp_min` from Burn. Burn supports f16 elementwise via cubecl on wgpu, but the constants in `gpu_ops.rs` are typed `f32` and applied via `mul_scalar(SRGB_C: f32)` etc.; Burn upcasts the scalar to the tensor element type, so an f16 tensor would silently downcast each constant. PU's `b_high = log(x + PU_F).clamp_min(1e-30)` has a constant `1e-30 < f16::MIN_POSITIVE (~6e-5)` — that clamp_min would be effectively a no-op in f16, letting `log(0)` reach the kernel. **Practical impact today is nil because the wgpu backend in this crate runs f32 only** (no `oidn_model::Net<WgpuBackend<f16>>` instantiated anywhere I can see). Flag for any future f16 wiring.
- **Async / fences.** All work is structured as Burn tensor expressions; there are no manual `submit` / `wait_idle` calls in these four files. cubecl handles command-buffer submission. Sync points only at host readbacks (`into_data().to_vec()` in `tensor_to_chw_vec` and the diagnostic helpers). No leaked GPU/CPU stalls in the hot path.

## Dead / unused code paths

- `pad_axis_reflect` degenerate-length fallback (`gpu_ops.rs:86-88`) — the comment admits real workloads never reach the `len < 2` branch. Harmless; flagged for completeness.
- `tensor_diagnostics_enabled()` (`unet_runner.rs:319-326`) — reads `OIDN_TRACE_TENSORS`; nothing in the Rust code sets it. Fine, intentional.
- `TensorStats` (`unet_runner.rs:352-409`) — only used by the trace path.
- No feature-flagged GPU stubs found in the audited files. No `#[cfg]` branches that would silently disable kernels.

## Open questions

1. Is the **reflect-pad** in Rust a deliberate quality choice (mirror is a better extrapolation than zeros at the receptive field's outer edge) or unintentional drift from the spec? The reference's choice of zero padding is itself documented (`cpu_input_process.isph:88`: `// Zero pad`). If intentional, document the deviation in `gpu_ops.rs` doc-comment. If unintentional, switch to zero pad to recover bit-parity at borders.
2. `snorm` (Issue I-4) is exercised by `denoise_lightmap_directional_wgpu` (`e2e_wgpu.rs:367-400`). Does `RtLightmapFilter` set up an internal forward/inverse that emulates `snorm` (i.e., feed signed values as `*0.5+0.5` before the network and undo after)? Needs inspection of `crates/oidn-rs/src/filters/rt_lightmap.rs` (outside this audit's scope).
3. Single-channel output (Issue I-5) — any model in the current zoo declare `dst.C == 1`? If not, this is dead spec; if yes, Rust returns a 3-channel tensor with all three channels equal? Or three independent channels (network would not match reference)?
4. The reference handles "main src is whichever of color/albedo/normal is non-null" (`input_process.h:40-44`, `cpu_input_process.cpp:26-28`) so albedo-only or normal-only flows route the auxiliary into channel 0. Rust mirrors this by routing the AOV-only filter to a different model file (`rt_alb_large`, `rt_nrm_large`), and `run_tensors` cat-orders by presence — but does `albedo` alone get the `getAlbedo` transform (clamp 0..1) or the `getInput` transform (forward + autoexposure)? Reference uses `getInput` on the **first** non-null AOV (`cpu_input_process.cpp:26`). Rust applies `apply_transfer_forward` only on `color`; albedo-only path skips transfer entirely (`unet_runner.rs:170-176`). For `rt_alb_large` (LDR, sRGB? linear?) this likely matches the model contract — but verify against `RtFilter::commit()` model-routing logic.

## Citations

- rust input/transfer pipeline: `crates/oidn-rs/src/gpu_ops.rs:46-218`, `crates/oidn-rs/src/filters/unet_runner.rs:64-228`, `crates/oidn-rs/src/image_tensor.rs:54-83`
- rust e2e tests: `crates/oidn-rs/tests/e2e_wgpu.rs:69-400`
- ref CPU input kernel: `oidn/devices/cpu/cpu_input_process.isph:11-136`, `cpu_input_process.cpp:19-46`
- ref GPU input kernel: `oidn/devices/gpu/gpu_input_process.h:18-276`
- ref output kernel: `oidn/devices/cpu/cpu_output_process.isph:11-71`, `cpu_output_process.cpp:11-43`
- ref shape/layout: `oidn/core/input_process.cpp:9-26`, `oidn/core/output_process.cpp:8-15`, `oidn/core/tensor_layout.h:11-415`
- ref graph build (UNet): `oidn/core/unet_filter.cpp:470-528`
- rust graph build (UNet): `crates/oidn-model/src/unet.rs:32-119`, `crates/oidn-model/src/unet_large.rs:5-135`
