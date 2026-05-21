# Kapitsa — Color / HDR transform and autoexposure parity audit

- Agent: Kapitsa
- Date: 2026-05-21
- Scope: read-only parity check between
  - Rust: `crates/oidn-rs/src/color.rs`, `crates/oidn-rs/src/autoexposure.rs`, `crates/oidn-rs/src/gpu_ops.rs`, `crates/oidn-rs/src/filters/unet_runner.rs`, `crates/oidn-rs/src/image_tensor.rs`, `crates/oidn-rs/tests/unit_color_tile.rs`
  - C++ reference: `_ref/oidn/core/color.{h,cpp}`, `_ref/oidn/core/autoexposure.h`, `_ref/oidn/devices/cpu/{color.ispc,color.isph,cpu_autoexposure.{cpp,ispc,h},cpu_input_process.isph,cpu_output_process.isph}`, `_ref/oidn/core/{input_process,output_process}.cpp`

## Verdict

- Transfer functions (Linear / sRGB / PU / Log) — **MATCH**: constants identical, formulas identical, normScale derivation identical, luminance weights identical.
- Autoexposure constants and algorithm — **MATCH** functionally. One implementation detail (log base) differs but is mathematically equivalent; one detail (per-pixel pre-clamp + NaN-to-zero sanitisation inside the bin) is **MISSING** in Rust.
- Pre-processing (`getInput` / `getAlbedo` / `getNormal`) — **DIVERGENT**:
  1. Reference clamps color after multiplying by `inputScale` and before forward transfer; Rust applies forward to raw scaled values, no clamp.
  2. Reference handles `snorm` color (remap to `[0,1]` and `[-1,1]`); Rust has no `snorm` path on the color channel.
  3. Reference zero-pads tile borders; Rust uses reflect-pad. Different network input distribution at tile boundaries.
- Post-processing (output_process) — **DIVERGENT**:
  1. Reference clamps CNN output to `[0, +inf)` and NaN→0 **before** inverse transfer; Rust passes raw network output directly into inverse.
  2. Reference applies LDR clamp `min(1.0)`, optional snorm remap, and `outputScale` **after** inverse. Rust folds `output_scale` inside `inverse()` and never applies the LDR clamp.
  3. Reference averages all 3 output channels to 1 when `dst.C == 1` (lightmap directional case). No such path in Rust output.

Severity summary: 3 high (P0), 2 medium (P1), 4 low (P2). Numerical drift on HDR inputs with negatives or NaN is significant; round-trip tests in Rust currently mask the gap because they feed clean ≥ 0 samples.

## Constant comparison table

| Constant | Rust value | Ref value | Rust location | Ref location | Status |
|---|---|---|---|---|---|
| `yMax` | `65504.0` | `65504.f` | `color.rs:16` | `color.h:20` | OK |
| sRGB `a` | `12.92` | `12.92f` | `color.rs:96` | `color.h:31`, `color.ispc:55` | OK |
| sRGB `b` | `1.055` | `1.055f` | `color.rs:97` | `color.h:32` | OK |
| sRGB `c` | `1.0/2.4` | `1.f/2.4f` | `color.rs:98` | `color.h:33` | OK |
| sRGB `d` | `-0.055` | `-0.055f` | `color.rs:99` | `color.h:34` | OK |
| sRGB `y0` | `0.0031308` | `0.0031308f` | `color.rs:100` | `color.h:35` | OK |
| sRGB `x0` | `0.04045` | `0.04045f` | `color.rs:101` | `color.h:36` | OK |
| PU `a` | `1.41283765e+03` | `1.41283765e+03f` | `color.rs:114` | `color.h:57` | OK |
| PU `b` | `1.64593172e+00` | `1.64593172e+00f` | `color.rs:115` | `color.h:58` | OK |
| PU `c` | `4.31384981e-01` | `4.31384981e-01f` | `color.rs:116` | `color.h:59` | OK |
| PU `d` | `-2.94139609e-03` | `-2.94139609e-03f` | `color.rs:117` | `color.h:60` | OK |
| PU `e` | `1.92653254e-01` | `1.92653254e-01f` | `color.rs:118` | `color.h:61` | OK |
| PU `f` | `6.26026094e-03` | `6.26026094e-03f` | `color.rs:119` | `color.h:62` | OK |
| PU `g` | `9.98620152e-01` | `9.98620152e-01f` | `color.rs:120` | `color.h:63` | OK |
| PU `y0` | `1.57945760e-06` | `1.57945760e-06f` | `color.rs:121` | `color.h:64` | OK |
| PU `y1` | `3.22087631e-02` | `3.22087631e-02f` | `color.rs:122` | `color.h:65` | OK |
| PU `x0` | `2.23151711e-03` | `2.23151711e-03f` | `color.rs:123` | `color.h:66` | OK |
| PU `x1` | `3.70974749e-01` | `3.70974749e-01f` | `color.rs:124` | `color.h:67` | OK |
| Luminance R | `0.212671` | `0.212671f` | `color.rs:150`, `autoexposure.rs:34` | `color.h:171`, `color.isph:40` | OK |
| Luminance G | `0.715160` | `0.715160f` | `color.rs:150`, `autoexposure.rs:35` | `color.h:171`, `color.isph:40` | OK |
| Luminance B | `0.072169` | `0.072169f` | `color.rs:150`, `autoexposure.rs:36` | `color.h:171`, `color.isph:40` | OK |
| Autoexposure `MAX_BIN_SIZE` | `16` | `16` | `autoexposure.rs:27` | `autoexposure.h:16` | OK |
| Autoexposure `KEY` | `0.18` | `0.18f` | `autoexposure.rs:29` | `autoexposure.h:17` | OK |
| Autoexposure `EPS` | `1e-8` | `1e-8f` | `autoexposure.rs:31` | `autoexposure.h:18` | OK |
| Log forward floor | `max(1e-30)` | none (`log(y+1)` directly) | `color.rs:68,91`, `gpu_ops.rs:217` | `color.ispc:158` | NOTE — Rust adds defensive floor; ref does not |
| PU forward floor | `max(1e-30)` (gpu_ops tensor only) | none | `gpu_ops.rs:189` | `color.ispc:120` | NOTE — same |

## Function-by-function mapping

### sRGB forward
- Rust `srgb_forward` (`color.rs:104-106`) ↔ ref `SRGB::forward` (`color.h:38-44`) and `srgbForward` (`color.ispc:62-68`).
- Branch on `y <= y0`; both implementations identical, including the use of `pow(y, c)` (not powf with float arg only — Rust uses `f32::powf`).
- **MATCH**.

### sRGB inverse
- Rust `srgb_inverse` (`color.rs:109-111`) ↔ ref `SRGB::inverse` (`color.h:46-52`).
- Branch on `x <= x0`; both use `pow((x - d)/b, 1/c)`. **MATCH**.
- Note: tensor-path `srgb_inverse_tensor` (`gpu_ops.rs:168-178`) introduces `clamp_min(0.0)` before `powf_scalar(1/C)` — this is a divergence in the high branch (the masked-out low branch result is discarded but the high tensor is still computed everywhere). Whether the inputs are guaranteed ≥ 0 depends on the upstream sanitisation. After the post-network missing-clamp issue (see Bug O-1), negative network output reaches this path and gets silently clamped to 0 instead of going through the linear low branch (`x / A`).

### PU forward
- Rust `pu_forward` (`color.rs:127-135`) ↔ ref `PU::forward` (`color.h:69-77`) and `puForward` (`color.ispc:113-121`).
- Three-segment formula identical, branch thresholds identical, constants identical.
- Tensor variant `pu_forward_tensor` (`gpu_ops.rs:180-194`) adds `clamp_min(0.0)` inside the mid branch and `clamp_min(1e-30)` inside the high branch before `log`. The mask cascade routes negative inputs to the linear branch `A*y` (which is correct), so these defensive clamps only affect dead branches. **MATCH**.

### PU inverse
- Rust `pu_inverse` (`color.rs:138-146`) ↔ ref `PU::inverse` (`color.h:79-87`). Same comments; **MATCH**.
- `pu_inverse_tensor` has `clamp_min(0.0)` on the mid branch — dead-branch defensive only. **MATCH**.

### Log forward
- Rust `TransferState::forward` Log branch (`color.rs:68`): `(y + 1.0).max(1e-30).ln() * normScale`.
- Ref Log branch (`color.h:139`, `color.ispc:158`): `log(y + 1.f) * normScale` — no `max(1e-30)` floor.
- **DIVERGENCE**: for `y < -1` (which only reaches here if the upstream pre-clamp is missing), Rust returns `ln(1e-30) * normScale` while ref returns `NaN`. With proper pre-clamping (Bug I-1) both behaviours are unreachable. See Bug C-1 (low severity).

### Log inverse
- Rust: `(x * rcpNormScale).exp() - 1.0` (`color.rs:78`).
- Ref: `exp(x * rcpNormScale) - 1.f` (`color.h:160`, `color.ispc:163`).
- **MATCH**.

### normScale init
- Rust `TransferState::new` (`color.rs:38-53`): `forward_one(kind, yMax, 1.0)`, then `1/xMax`.
- Ref ctor `TransferFunction::TransferFunction(Type)` (`color.cpp:10-16`): `reduce_max(forward(yMax))` where `forward` returns a vec3f; for Linear/sRGB the value isn't normalised so `normScale` stays 1 *because* it's initialised to 1 in the member-init line and never overwritten in those constructors in ISPC (`color.ispc:46-49,90-93`). In C++ header `TransferFunction::TransferFunction(Type type)` always runs the line in `color.cpp`, but `forward(vec3f(yMax))` for Linear returns `vec3f(yMax)` so `1/yMax` would be used. Yet the C++ semantics in `forward(vec3f)` for Linear (`color.h:130`) do *not* multiply by `normScale` — only PU/Log/Log do (lines 136/139). So `normScale` ends up unused for Linear/sRGB. Rust matches this: `forward()` only multiplies `norm_scale` for PU and Log (`color.rs:67-69`).
- Subtle: Rust `forward_one(kind, Y_MAX, 1.0)` for Linear returns `Y_MAX`, so `norm_scale` = `1 / Y_MAX ≈ 1.5e-5`. For sRGB it returns `srgb_forward(Y_MAX)` ≈ some large number. These values are stored but only consumed by PU/Log forward paths, so the dead value is harmless. **MATCH (functionally)**, but the Rust normScale value for Linear/sRGB is `1 / yMax`, not `1` like ISPC, while in C++ header it's also `1 / yMax`. Header parity: OK. (See Dead code section.)
- Line `color.rs:49`: `let xmax = scaled.max(forward_one(kind, Y_MAX, 1.0));` is redundant — it computes the same value twice and takes max. Likely a leftover from an attempted `reduce_max` mimic on a vec3 result. Effect-neutral but dead. See Bug C-3.

### Autoexposure compute
- Reference algorithm (`cpu_autoexposure.cpp:31-65` + `cpu_autoexposure.ispc:8-25`):
  1. Bin grid sized `ceil_div(H, 16) × ceil_div(W, 16)`.
  2. Per-bin **bounds**: `beginH = i*H/numBinsH`, `endH = (i+1)*H/numBinsH` — i.e. proportional division, *not* fixed 16-pixel strides. Last bin can be slightly larger or smaller than 16.
  3. Per-pixel: `c = clamp(nan_to_zero(c), 0.f, pos_max)`, accumulate `luminance(c)`.
  4. Bin mean `L = sum / ((endH-beginH)*(endW-beginW))`.
  5. If `L > eps`: accumulate `log2(L)`, increment count.
  6. Final: `scale = (count > 0) ? key / exp2(sum/count) : 1.f`.
- Rust CPU path (`autoexposure.rs:47-94`):
  1. Bin grid same shape via `div_ceil`.
  2. Per-bin bounds use **fixed-16 strides clamped to image size**: `y0 = by*16`, `y1 = min(y0+16, H)`. Last bin is always ≤16; ref's last bin is `H - (numBinsH-1)*H/numBinsH` which can be > or < 16. Numerical difference in last bin only.
  3. Per-pixel: filter on `lum.is_finite()` — non-finite skipped (count `n` only counts finite). **No clamp to ≥0 and no negative→0**, so negative luminance contributes to `sum`, and a bin mean can be negative.
  4. Bin mean uses `n` (finite count), not the geometric bin area, so a bin with one inf gives wrong mean.
  5. `if avg > EPS`: accumulate `ln(avg)`, count++.
  6. Final: `scale = key / max(exp(sum/count), eps)` (natural log/exp).
- Tensor path (`autoexposure.rs:104-153`):
  1. Drops to unity if `H < 16` or `W < 16` (any axis under one bin). Ref still produces an answer for, e.g., 8x256.
  2. `avg_pool2d` with stride = kernel = 16, **no padding** — last < 16-pixel strip is dropped entirely. Ref includes it.
  3. No NaN clamp, no negative clamp.
- See Bugs A-1..A-4 for the consequences.

### Albedo prep
- Rust (`unet_runner.rs:170-176`): full-tensor NaN→0 sanitisation (line 78), then per-tile slice+reflect-pad, then `padded.clamp(0.0, 1.0)`.
- Ref (`cpu_input_process.isph:54-62` `getAlbedo`): per-pixel `clamp(nan_to_zero(value), 0.f, 1.f)`. No reflect-pad — zero-pad at tile borders (`cpu_input_process.isph:88-93,120-125`).
- **DIVERGENCE B-1** on tile boundary policy (reflect vs zero). Constants match.

### Normal prep
- Rust (`unet_runner.rs:177-185`): NaN→0, then `clamp(-1, 1)`, then `* 0.5 + 0.5`. Matches ref `getNormal()` (`cpu_input_process.isph:65-76`) ordering: clamp first, then remap.
- Same reflect vs zero-pad divergence (B-1).
- **MATCH (semantics)**.

### Color prep (full path)
- Ref `getInput` (`cpu_input_process.isph:31-51`):
  1. `value = Image_get3()`
  2. `value *= inputScale`  (autoexposure)
  3. `value = clamp(nan_to_zero(value), snorm ? -1 : 0, hdr ? pos_max : 1)`
  4. if snorm: `value = value * 0.5 + 0.5`
  5. `value = transferFunc.forward(value)`  — **without** input scale (raw `forward`, not `forward()` with embedded scale)
- Rust path (`unet_runner.rs:64-72` + `gpu_ops.rs:130-141`):
  1. Full-tensor NaN→0 sanitisation (good).
  2. No clamp to `[0, pos_max]` or `[0, 1]`. Negative values pass through.
  3. No snorm path — color is always assumed unsigned.
  4. `apply_transfer_forward` multiplies by `state.input_scale` then runs forward.
- Same end result for the *transfer* step (because Rust's forward formula already includes input scale and ref's forward formula explicitly does not — but ref multiplies separately first, so the input to the curve is `y*inputScale` in both cases).
- **Bugs I-1, I-2**: missing post-scale clamp; missing snorm-color contract.

## Per-issue table

| ID | Severity | Rust file:line | Ref file:line | Description | Fix |
|---|---|---|---|---|---|
| I-1 | **HIGH** | `unet_runner.rs:163-168`, `gpu_ops.rs:130-141` | `cpu_input_process.isph:35-39` | Color input never clamped to `[0, hdr ? +inf : 1]` after applying inputScale and before forward transfer. Negatives bypass the sanitiser and reach `srgb_forward` / `pu_forward` / `log_forward`. | After the input-scale multiply (or before `apply_transfer_forward`), add `clamp(0, if hdr { f32::MAX } else { 1.0 })`. Also clamp away `+inf` for HDR (`pos_max`, not `f32::MAX`). |
| I-2 | LOW | `filters/unet_runner.rs:62-72` (whole-tensor sanitise) | `cpu_input_process.isph:31-51` | Sanitisation is whole-tensor, not per-pixel inside the kernel. Functionally equivalent because the network sees the whole tile; just note Rust does an extra pass. | None — performance only. |
| I-3 | MEDIUM | `unet_runner.rs:163-168` | `cpu_input_process.isph:11-28` | No `snorm` flag on color channel. Reference can run on directional/normal-only networks via `snorm = true` (`-1..1` color, remap to `[0,1]`). Not yet supported in oidn-rs API surface. | Plumb `snorm` through filter config or document explicit limitation. |
| O-1 | **HIGH** | `unet_runner.rs:191-198`, `gpu_ops.rs:145-158` | `cpu_output_process.isph:42-49` | CNN output is fed straight into `apply_transfer_inverse` without `clamp(nan_to_zero(value), 0, pos_max)`. Negative network outputs end up running through `PU::inverse` low-branch (negative linear), `srgb_inverse` low-branch (negative linear), or `log_inverse` (exp of negative → small positive). Ref always clamps these to zero before inverse. | In `apply_transfer_inverse` (and / or the call site), prepend a `t.clamp(0.0, f32::INFINITY)` *and* NaN→0 for transfer kinds other than Linear. |
| O-2 | **HIGH** | `unet_runner.rs:191-216` | `cpu_output_process.isph:55-63` | LDR-mode (`hdr = false`) does not clamp the inverse output to `[0, 1]`. Ref applies `min(value, 1.f)` after inverse. Rust will return >1 values for LDR sRGB. | After `apply_transfer_inverse`, when `!hdr`, `accum.clamp_max(1.0)`. Apply `value*2-1`, `max(-1)` if `snorm`. |
| O-3 | MEDIUM | `gpu_ops.rs:145-158` (`unscaled.mul_scalar(state.output_scale)`) | `cpu_output_process.isph:35,66` | `output_scale` is applied *inside* `apply_transfer_inverse`. Ref applies it *outside* `transferFunc.inverse` and *after* hdr/snorm clamps. Result identical for HDR linear case, but interacts with the missing LDR clamp (O-2): currently scale acts on un-clamped values; with O-2 fix the order must be `inverse → clamp → snorm-remap → multiply outputScale`. | Restructure `apply_transfer_inverse` to leave outputScale out, then apply it after the clamp/snorm step. Or fold all three into a single `postprocess_color` function that mirrors the reference op-by-op. |
| O-4 | LOW | `unet_runner.rs:191-198` | `cpu_output_process.isph:51-53` | When `dst.C == 1` (directional lightmap output) ref averages `(x+y+z)/3` after inverse. No equivalent in Rust output path. | Add a 1-channel collapse before image write when filter is RT lightmap / directional. |
| B-1 | MEDIUM | `unet_runner.rs:167,174,181`, `gpu_ops.rs:58-107` | `cpu_input_process.isph:88-93,120-125` | Tile-boundary policy mismatch: Rust reflect-pads tile edges; reference zero-pads tile destination borders. The network was trained against zero-padded borders, so reflect-pad changes the input distribution at the edges and can produce sub-pixel artefacts at tile seams. | Replace `reflect_pad_2d` with zero-pad for the tile-edge padding, matching `getInput`/`getAlbedo`/`getNormal` zero-fill in the input kernel. |
| A-1 | MEDIUM | `autoexposure.rs:67-76` | `cpu_autoexposure.ispc:13-22` | Per-pixel sanitisation differs. Ref does `clamp(nan_to_zero(c), 0, pos_max)` before luminance; Rust simply skips non-finite pixels via `is_finite()` and lets negative values contribute to `sum`. | In `compute_scale`, replace the finite filter with `let mut s = px[i]; if !s.is_finite() { s = 0.0; } s = s.clamp(0.0, f32::MAX);` — same in tensor variant via `t.clamp(0, f32::MAX)` before `avg_pool2d`. |
| A-2 | LOW | `autoexposure.rs:79` (`avg = sum / n`) | `cpu_autoexposure.ispc:24` (`reduce_add(L) / ((endH-beginH)*(endW-beginW))`) | Rust divides by finite-pixel count; ref divides by full bin area. With NaN/Inf in input the two differ. After fixing A-1, this becomes moot because Rust no longer drops pixels. | After applying A-1's clamp+NaN-to-zero, change denominator to the actual bin area `(y1-y0)*(x1-x0)`. |
| A-3 | LOW | `autoexposure.rs:81,92` (`ln` / `exp`) | `cpu_autoexposure.cpp:53,64` (`log2` / `exp2`) | Geometric mean is computed via natural log in Rust, base-2 log in ref. Algebraically equivalent (`exp(mean(ln x)) == exp2(mean(log2 x))`) within float precision; harmless. | None — mathematically equivalent. Optionally switch to `log2`/`exp2` for byte-exact parity with the ISA path. |
| A-4 | MEDIUM | `autoexposure.rs:53-60` | `cpu_autoexposure.cpp:43-46` | Bin partitioning differs. Ref uses proportional split `i*H/numBins` (last bin can be larger than 16). Rust uses fixed-16 strides with the last bin clamped to image edge. For image sizes that aren't a multiple of 16 the last-bin mean is computed over a different pixel population. | Switch to `beginH = i*H/numBinsH; endH = (i+1)*H/numBinsH;` to match ref byte-for-byte. |
| A-5 | MEDIUM | `autoexposure.rs:110-114` | `cpu_autoexposure.cpp:43-46` | Tensor variant returns unity scale if `H < 16` or `W < 16`. Ref always returns a finite scale for non-empty images. | Pad with zeros up to 16 along each short axis, or fall back to the CPU `compute_scale` for small images. |
| A-6 | LOW | `autoexposure.rs:128-135` | `cpu_autoexposure.cpp:43-46` | `avg_pool2d` with stride=kernel=16 drops residual columns/rows on the right/bottom edges (no padding, `count_include_pad=false`). Ref includes those pixels in the corresponding last bins. | Use `ceil_mode = true` + `count_include_pad = false`, or explicitly handle the remainder strip. |
| C-1 | LOW | `color.rs:68,91`, `gpu_ops.rs:217` | `color.ispc:158`, `color.h:139` | Log forward floors `y+1` at `1e-30` before `ln`. Reference never floors. With proper input clamping (Bug I-1) the y+1 ≥ 1 invariant holds and the floor is irrelevant. Without I-1, the two diverge for y < -1. | Either: keep floor and accept divergence as "more robust"; or remove and rely on the I-1 clamp. Note as defensive code. |
| C-2 | LOW | `gpu_ops.rs:189` | `color.ispc:120` | Tensor PU forward floors `y + PU_F` at `1e-30` before `log` for the high branch. Reference does not. Branch is unreachable for `y < 0` because `low_mask` routes those into the linear branch; for `0 ≤ y` the value is ≥ `PU_F` ≈ 6.26e-3 > 1e-30, so the floor never fires. Dead defensive code. | Remove `clamp_min(1e-30)`; or leave with a comment. |
| C-3 | LOW | `color.rs:48-50` | `color.cpp:13-16` | `let xmax = scaled.max(forward_one(kind, Y_MAX, 1.0));` computes the same value twice. Likely a stale attempt to mimic `reduce_max(vec3f)`. The result is identical to `scaled`, so the `.max()` is dead. | Replace with `let xmax = forward_one(kind, Y_MAX, 1.0);`. Add comment explaining that vec3 collapse is unnecessary because the per-channel formula has equal results when applied to a constant vec3. |
| C-4 | LOW | `color.rs:38-53` | `color.h:91`, `color.cpp:10-16`, `color.ispc:44-49` | For Linear/sRGB, Rust still computes `norm_scale = 1 / forward_one(yMax)` (`1/65504` and `1/srgb_forward(65504)`). The C++ header runs the same line; the ISPC ctor for Linear/sRGB skips `TransferFunction_initNormalization` and leaves `normScale = 1`. None of `forward()`/`inverse()` actually uses `normScale` for Linear/sRGB, so the value is dead. **Inconsistent with ISPC but consistent with C++ header.** | Optional: skip the init for Linear/sRGB to match ISPC, or leave the dead value documented. |

## Numerical precision concerns

- **Autoexposure log base** (A-3): Rust uses `ln/exp`, ref uses `log2/exp2`. Geometric mean of identical bin means yields identical results in real arithmetic; in float they will differ by `~1 ulp` per term. Negligible.
- **Float vs double accumulation** in `compute_scale`: Rust accumulates `sum_log` as `f64`. Ref accumulates `sum` as `float` (TBB reduce sums floats). Rust is *more* numerically stable; not a bug, expected to give slightly different scale on degenerate inputs (very large bin counts).
- **`f32::powf` vs `pow` in ISPC**: identical math.lib bindings on modern Intel and clang-spurious-difference at most 1 ulp. Round-trip tests in `unit_color_tile.rs:7-24` already cover this.
- **NaN handling in `clamp`**: `gpu_ops::apply_transfer_forward` does not clamp NaN. The whole-tensor sanitiser at `unet_runner.rs:64-72` covers NaN in the input, but if any tensor op in between introduces NaN (e.g. division during reflect_pad — none currently), it slips through.
- **`pos_max`**: Ref clamps to `pos_max` (largest finite float) to nuke `+inf` to a finite value pre-transfer. Rust never does. Combined with I-1: `+inf * inputScale = +inf → ln(+inf+1) = +inf`. Downstream network sees `+inf` and produces NaN. (Real path requires inputs to not be `+inf`; standard renderers shouldn't, but pathological HDR exrs can.)
- **f16 conversion paths**: not covered by Rust code under review. `cpu_input_process_f16.ispc` and `cpu_output_process_f16.ispc` exist in ref; Rust uses `f32` throughout (`color.rs:1-10` comment). If the Burn backend ever runs the network in f16, the Rust side must add rounding helpers (`round-to-nearest-even` standard). Not blocking. Notation only.

## Dead / unused code

- `color.rs:49` — duplicate `forward_one` call inside `max` (see Bug C-3).
- `color.rs:38-53` for Linear/sRGB — `norm_scale` computed and stored but never consulted (see Bug C-4).
- `gpu_ops.rs:189` `clamp_min(1e-30)` on PU high branch — unreachable (see Bug C-2).
- `gpu_ops.rs:217` `clamp_min(1e-30)` on Log forward — defensive, only fires when I-1 is unfixed and a pre-clamped input is negative below -1.
- `gpu_ops.rs:163-164,188,210` `clamp_min(0.0)` inside dead branches of the `mask_where` cascade.

## Open questions

1. **Snorm color contract.** The reference supports `snorm = true` color (signed normalized, e.g. denoising raw normal-only renders). Is this within scope for `oidn-rs` Phase I? If yes, plumb it through; if no, document the limitation and reject snorm filter configs at the public API boundary.
2. **Tile padding policy.** Switching from reflect to zero will change network outputs at tile borders. Are the bundled weights (RT, RTLightmap) trained against zero-padded tiles? Confirm against `_ref/oidn/training/`. If yes (very likely — ref always zero-pads), bug B-1 is a real seam-artefact bug, not a stylistic one.
3. **Lightmap directional collapse.** When does `dst.C == 1` fire in the Rust path? `RtLightmapFilter` exposes a directional mode; verify whether the filter's tensor-mode output already collapses, or whether it relies on this kernel-level fold.
4. **Pos-max constant.** Reference clamps to `pos_max` (typically `FLT_MAX` ≈ 3.4e38). If switching Rust to a matching clamp, use `f32::MAX`. Confirm the Burn `clamp` op preserves the sign of `+inf` → `f32::MAX` mapping (it should — `clamp` is `min(max(x, lo), hi)`).
5. **Autoexposure on `H or W < 16` images.** Is the tensor-path unity-scale fallback (`autoexposure.rs:110-114`) acceptable, or do we need the CPU path's per-bin partial coverage? Likely OK in production (no one denoises <16-pixel images) but worth confirming.
6. **Output snorm.** Same as #1 for the output side. The reference handles `value = value*2 - 1; max(value, -1)` for snorm output — Rust has no path for this.

## Provenance / scope notes

- The Rust audit deliberately covers both `color.rs` (scalar) and `gpu_ops.rs` (tensor) because they implement the same transfer functions twice. Constants in `gpu_ops.rs:26-44` duplicate `color.rs:96-124`; if either set drifts, parity is lost silently. Consider exposing the constants once from `color.rs` and using them in `gpu_ops.rs` (`pub(crate) const`).
- All citations verified against `_ref/oidn` upstream at the path supplied. No reference file outside the listed set was consulted.
