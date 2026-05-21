# Landau — Strict parity audit, Rust U-Net vs C++ OIDN reference (2026-05-21)

## Verdict

**DIVERGENT — 1 high-severity activation mismatch on base/small UNet output, plus several minor gaps. UNetLarge is topology-correct. Weight naming matches reference exactly.**

Scope of audit (READ-ONLY): U-Net topology, conv/pool/upsample ops, concat-skip semantics, weight loader naming, dtype handling, variant detection. Auxiliary HDR/albedo/normal channel composition lives in `oidn-runtime` (out of scope here — runtime tests use a Rust caller-supplied `in_channels`, which is correct against C++ where the runtime computes `inputC = 3 + 3 + 3` in `unet_filter.cpp:542-544`).

---

## Weight-name comparison: Rust loader vs reference TZA

Reference TZA names follow Python `state_dict` (Intel `_ref/oidn/training/model.py:55-103, 161-208`) and are looked up by the C++ graph builder in `_ref/oidn/core/unet_filter.cpp:468-531` via `addConv("<name>", …)`, which prepends `.weight` / `.bias` in `_ref/oidn/core/graph.cpp:82-83, 151-152`.

### UNet (Base / Small)

| Rust loader name (loader.rs:127-142) | C++ addConv call (unet_filter.cpp) | Match |
|---|---|---|
| `enc_conv0`  | line 470 `enc_conv0`  | OK |
| `enc_conv1`  | line 472 `enc_conv1`  | OK |
| `enc_conv2`  | line 474 `enc_conv2`  | OK |
| `enc_conv3`  | line 476 `enc_conv3`  | OK |
| `enc_conv4`  | line 478 `enc_conv4`  | OK |
| `enc_conv5a` | line 480 `enc_conv5a` | OK |
| `enc_conv5b` | line 481 `enc_conv5b` | OK |
| `dec_conv4a` | line 483 `dec_conv4a` (addConcatConv) | OK |
| `dec_conv4b` | line 484 `dec_conv4b` | OK |
| `dec_conv3a` | line 486 `dec_conv3a` (addConcatConv) | OK |
| `dec_conv3b` | line 487 `dec_conv3b` | OK |
| `dec_conv2a` | line 489 `dec_conv2a` (addConcatConv) | OK |
| `dec_conv2b` | line 490 `dec_conv2b` | OK |
| `dec_conv1a` | line 492 `dec_conv1a` (addConcatConv) | OK |
| `dec_conv1b` | line 493 `dec_conv1b` | OK |
| `dec_conv0`  | line 495 `dec_conv0`  | OK |

### UNetLarge

| Rust loader name (loader.rs:158-176) | C++ addConv call (unet_filter.cpp) | Match |
|---|---|---|
| `enc_conv1a` | line 502 | OK |
| `enc_conv1b` | line 503 | OK |
| `enc_conv2a` | line 505 | OK |
| `enc_conv2b` | line 506 | OK |
| `enc_conv3a` | line 508 | OK |
| `enc_conv3b` | line 509 | OK |
| `enc_conv4a` | line 511 | OK |
| `enc_conv4b` | line 512 | OK |
| `enc_conv5a` | line 514 | OK |
| `enc_conv5b` | line 515 | OK |
| `dec_conv4a` | line 517 | OK |
| `dec_conv4b` | line 518 | OK |
| `dec_conv3a` | line 520 | OK |
| `dec_conv3b` | line 521 | OK |
| `dec_conv2a` | line 523 | OK |
| `dec_conv2b` | line 524 | OK |
| `dec_conv1a` | line 526 | OK |
| `dec_conv1b` | line 527 | OK |
| `dec_conv1c` | line 528 | OK |

All weight/bias name lookups in `loader.rs` use the layer field-name directly (`{layer}.weight` / `{layer}.bias`, loader.rs:109-110), which matches the C++ pattern `name + ".weight"` / `name + ".bias"` (graph.cpp:82-83). **No naming gap.**

---

## Per-issue table

| ID | Sev | Rust file:line | Ref file:line | Description | Fix |
|---|---|---|---|---|---|
| U1 | **HIGH** | `crates/oidn-model/src/unet.rs:131` | `_ref/oidn/core/unet_filter.cpp:495` | UNet output `dec_conv0` is invoked *without* ReLU in Rust (`self.dec_conv0.forward(x)`), but the C++ runtime fuses `Activation::ReLU` onto it. The Python *training* code (model.py:153) also omits the final ReLU, so Rust matches training but DIVERGES from the C++ runtime. Effect: every negative value in the final Rust output stays negative, while the shipping OIDN runtime would clamp it to 0. For HDR mode the runtime then runs the inverse transfer function on a non-negative input. Mismatch is real and produces visibly different pixels on noisy edges. | Wrap the final conv in `relu(...)` to match the runtime exactly: `relu(self.dec_conv0.forward(x))`. See unet_large.rs:143 for the analogous correct pattern. |
| U2 | LOW | `crates/oidn-model/src/unet.rs:48`, `unet_large.rs:57` | `_ref/oidn/core/pool.cpp:11-14` | Rust uses Burn `MaxPool2d` with kernel `[2,2]` stride `[2,2]` and the default `PaddingConfig2d::Valid`. Ref also enforces `H%2==0 && W%2==0` (pool.cpp:11) and emits `[C, H/2, W/2]`. Functionally equivalent, but Burn's default uses an implicit `dilation=1`, `ceil_mode=false`. Ref hard-asserts `srcDesc.getH() % 2 != 0` → throw; Rust silently rounds down via Burn. | Add a debug-assert in `forward` that input H,W are multiples of 16 (consistent with `unet.rs:90` doc comment claim) so a misaligned tensor doesn't silently produce a sub-pixel mis-aligned output. |
| U3 | LOW | `crates/oidn-model/src/unet.rs:137-143`, `unet_large.rs:148-154` | `_ref/oidn/core/upsample.cpp:14-15` | Rust uses `interpolate(x, [h*2, w*2], Nearest)`. Ref output desc is `{C, H*2, W*2}` with no smoothing — identical semantics. | None — parity OK. |
| U4 | LOW | `crates/oidn-model/src/unet.rs:112,117,122,127`, `unet_large.rs:125,130,135,140` | `_ref/oidn/core/unet_filter.cpp:483,486,489,492 / 517,520,523,526` | Skip-connection concat order. Rust does `Tensor::cat(vec![x, poolN], 1)` (i.e. *decoder-first*, then skip). Ref `addConcatConv(name, x, poolN, …)` (src1=decoder, src2=skip). Reorder of the channel concat is the same convention (decoder-up channels first, then skip), confirmed by graph.cpp:167-170 (`finalWeightDims = … src1Paddec + src2Padded`). | None — parity OK. |
| U5 | INFO | `crates/oidn-model/src/loader.rs:57-60, 86-89` | `_ref/oidn/core/graph.cpp:91-106` | Rust converts f16→f32 once in `into_param4/into_param1` and stores f32 weights. Ref keeps device-native dtype (`device->getWeightDataType()`). For an NdArray/CPU CG-grade port this is fine; for a future wgpu backend, decide whether to keep f16 storage to save VRAM. | None now; revisit when wgpu backend lands. |
| U6 | LOW | `crates/oidn-model/src/variants.rs:27-32` | `_ref/oidn/core/unet_filter.cpp:263` | `Variant::from_tensor_names` only distinguishes Base vs Large via presence of `enc_conv1b.weight`. It cannot distinguish Base from Small, nor Large from XL — those need to be inferred from `enc_conv0.weight` channel count (Base: 32; Small: 32 but with `enc_conv2.weight[O]==32` instead of 48). Ref doesn't auto-detect either: it uses the `Model::small/base/large` slot derived from `quality`. Currently the caller must pass the correct `Variant`/channel set externally. | Either accept the caller-supplied variant (current behaviour, fine) or extend `from_tensor_names` to also peek at `enc_conv2.weight` shape to disambiguate Small vs Base. Document explicitly in `variants.rs:26` doc-comment. |
| U7 | LOW | `crates/oidn-model/src/variants.rs:53-67` | n/a | `ChannelConfig::for_variant` panics on `Large`/`XLarge`. Safe (other path uses `UNetLarge`), but the panic message is the only signal — if a future refactor accidentally hits this branch via `Net::Base(…)` while loading large weights, the user sees a panic deep inside a const fn. | Replace panic with a `Result` or surface it through `LoadError` so the runtime never panics on malformed input. |
| U8 | INFO | `crates/oidn-model/src/unet_large.rs:39-43` | `_ref/oidn/training/model.py:167-176` | XL channel widths match: 96/128/192/256/384, dec 256/192/128/96. OK. | None. |
| U9 | INFO | `crates/oidn-model/src/loader.rs:107` | n/a | `b_len = … unwrap_or(w_shape[0])` — falls back to output-channel count if bias is absent. Conv was constructed with `with_bias(true)` (unet.rs:20, unet_large.rs:23), so bias is always Some — unreachable branch in practice. | None (harmless), could be `expect("conv always built with bias")`. |
| U10 | LOW | `crates/oidn-model/src/loader.rs:121-146 / 152-180` | n/a | `load_tza` and `load_tza_large` are full-field updates — they reconstruct the entire `UNet` struct field-by-field. If `UNet` ever gains a new conv field the compiler will catch it via missing-field error, but **`pool`** and **`in_channels`** are hand-copied. Acceptable as long as the topology stays static. | None; relies on Burn `#[derive(Module)]` invariants. |
| U11 | INFO | `crates/oidn-model/src/net.rs:14-18` | n/a | `#[allow(clippy::large_enum_variant)]` on `Net` — UNetLarge is significantly bigger than UNet, but only one variant is alive at a time per filter instance. Comment explains rationale. | None. |
| U12 | INFO | `crates/oidn-model/src/unet.rs:73` | `_ref/oidn/training/model.py:101` | `dec_conv1a` in_channels = `dc2b + ic` (i.e. uses the *raw* input channel count of the filter, e.g. 9 for hdr_alb_nrm). The skip is `concat(x, inputProcess)` in ref (unet_filter.cpp:492) — the input image after the input-process op, which preserves channel count. Matches. | None. |

---

## Architectural diff (ASCII)

### UNet (Base / Small) — Rust vs Ref

Rust (`crates/oidn-model/src/unet.rs:91-132`) vs Ref (`_ref/oidn/core/unet_filter.cpp:468-498`):

```
                                                Rust                     Ref C++ runtime
  input ─ relu(enc_conv0) ───────────────────── input                   input
                │                                                       enc_conv0 (ReLU)         <- "no pool" head
                relu(enc_conv1)                                         enc_conv1 (ReLU+Pool)  ── pool1
                │
                pool 2x2 (MaxPool) ── pool1
                │
                relu(enc_conv2)                                         enc_conv2 (ReLU+Pool)  ── pool2
                pool 2x2          ── pool2
                relu(enc_conv3)                                         enc_conv3 (ReLU+Pool)  ── pool3
                pool 2x2          ── pool3
                relu(enc_conv4)                                         enc_conv4 (ReLU+Pool)  ── pool4
                pool 2x2          ── pool4 (anon)
  ── bottleneck ──
                relu(enc_conv5a)                                        enc_conv5a (ReLU)
                relu(enc_conv5b)                                        enc_conv5b (ReLU+Upsample)
  ── decoder ──
                upsample2x
                cat([x, pool3], 1)                                      concat_conv(dec_conv4a, x, pool3, ReLU)
                relu(dec_conv4a)
                relu(dec_conv4b)                                        dec_conv4b (ReLU+Upsample)
                upsample2x
                cat([x, pool2], 1)                                      concat_conv(dec_conv3a, x, pool2, ReLU)
                relu(dec_conv3a)
                relu(dec_conv3b)                                        dec_conv3b (ReLU+Upsample)
                upsample2x
                cat([x, pool1], 1)                                      concat_conv(dec_conv2a, x, pool1, ReLU)
                relu(dec_conv2a)
                relu(dec_conv2b)                                        dec_conv2b (ReLU+Upsample)
                upsample2x
                cat([x, input], 1)                                      concat_conv(dec_conv1a, x, input, ReLU)
                relu(dec_conv1a)
                relu(dec_conv1b)                                        dec_conv1b (ReLU)
  ── output ──
                dec_conv0(x)                                            dec_conv0 (ReLU)    <<< U1: HIGH
```

Same op count (16 convs, 4 pool, 4 upsample, 4 concat). Same skip wiring (pool1, pool2, pool3, input). Same channel widths. Different output activation (U1).

### UNetLarge — Rust vs Ref

Rust (`crates/oidn-model/src/unet_large.rs:101-144`) vs Ref (`_ref/oidn/core/unet_filter.cpp:500-531`):

```
  input ─ relu(enc_conv1a)
          relu(enc_conv1b)
          pool 2x2 ── pool1
          relu(enc_conv2a)
          relu(enc_conv2b)
          pool 2x2 ── pool2
          relu(enc_conv3a)
          relu(enc_conv3b)
          pool 2x2 ── pool3
          relu(enc_conv4a)
          relu(enc_conv4b)
          pool 2x2 ── pool4 (anon)
  ── bottleneck ──
          relu(enc_conv5a)
          relu(enc_conv5b)
  ── decoder ── (same upsample+cat pattern as UNet)
          relu(dec_conv1c(x))   <<< matches Ref (ReLU)
```

Topology-correct. 19 convs (10 enc + 8 dec + 1 output). Channel widths match `_ref/oidn/training/model.py:178-186` for BASE.

---

## Dead / unused / unfinished code

| Item | Location | Status |
|---|---|---|
| `Variant::XLarge` | variants.rs:18 | Reserved for future Intel weight releases. No TZA exists yet; runtime never picks this variant. Comment says so explicitly. Keep — harmless. |
| `ChannelConfigLarge::XL` | unet_large.rs:39-43 | Used only by `UNetLarge::new_xl` (unet_large.rs:94-96), which in turn is used only by `tests/unet_large.rs::unet_large_xl_forward_shape`. Production code never invokes XL. Keep as a smoke test, but mark as `#[cfg(test)]`-only or document that XL is dormant. |
| `ChannelConfig::for_variant(Large|XLarge)` panic branch | variants.rs:63-65 | Dead by contract — never called because Large path uses `UNetLarge`. Replace with `unreachable!()` or a `Result`. |
| `b_len` fallback in `load_conv` | loader.rs:107 | Conv is always built `with_bias(true)`, so the `.unwrap_or(w_shape[0])` branch is dead. Replace with `expect`. |
| TZA `enc_conv1b.weight` detection | variants.rs:29 | Used by `tests/unet_large.rs:45` and matches `_ref/oidn/core/unet_filter.cpp:263`. Live. |
| Test `weights_path` skip-when-missing branches | tests/unet_small.rs:8-15, unet_large.rs:8-15, load_real_weights.rs:9-12 | Tests silently `return` if weights are absent — they are de-facto no-ops on a clean checkout. Acceptable, but a CI run won't catch a regression unless the weights are checked in. |

---

## Dtype handling — f16 vs f32

`loader.rs:57-60, 86-89`: switches on `src.desc.dtype` and converts each `f16` element via `h.to_f32()` (presumably `half::f16::to_f32`). Result is always stored as f32 in Burn `Tensor`. **Correct** for the NdArray CPU backend (which is f32 by default per the test type `NdArray<f32>` at e.g. `tests/unet_shapes.rs:8`). For a future wgpu/f16 backend, this conversion path should be re-evaluated to preserve memory savings — non-blocking.

`fastMath` from the C++ ref (graph.cpp:23, unet_filter.cpp:262) — the Rust port has no equivalent flag. `quality != High` in the runtime selects `fastMath = true`, which lowers conv accumulation precision on some hardware paths. Rust currently never lowers precision; this is a feature gap (not a bug), worth flagging in `Variant`/`Net` builder.

---

## Per-variant input channel summary

Reference C++ computes input channels in `_ref/oidn/core/unet_filter.cpp:541-544`:
```
inputC = (color ? 3 : 0) + (albedo ? 3 : 0) + (normal ? 3 : 0);
```
Output channel count is propagated from `output->getC()` via `checkParams` (unet_filter.cpp:369). Always 3 for color/HDR/LDR/dir, 3 for normal, 3 for albedo. **Note**: directional (`dir`) is broadcast to 3 channels too (line 542 comment "always broadcast to 3 channels"), so single-channel lightmaps go through the same 3-out U-Net.

Rust mirrors this by taking `in_channels` and `out_channels` as constructor params (`unet.rs:54`, `unet_large.rs:62`). The variant-to-channel mapping is the caller's responsibility (presumably `oidn-runtime`). Tests cover 3-in/3-out (unet_shapes.rs:11) and 9-in/3-out (unet_shapes.rs:33, tests/unet_small.rs:55, tests/unet_large.rs:81), matching `rt_hdr_alb_nrm` / `rt_hdr_calb_cnrm_large`. **No lightmap (mono-output) test**, but the model itself supports 1 or 3 output channels by construction.

---

## Open questions

1. **U1 final ReLU**: Should Rust match the *runtime* C++ (ReLU on `dec_conv0`) or the *training* Python (no ReLU)? Pre-trained weights were trained with the Python definition, so the network expects no final ReLU. But the shipping runtime fuses ReLU regardless. The two produce different outputs whenever `dec_conv0` predicts a slightly negative value (rare for HDR, occasional for LDR near very dark regions). Recommend matching the C++ runtime (ReLU) since that's what published OIDN benchmarks use, and document the deviation from training in a code comment.
2. Is `oidn-runtime` (out of scope here) responsible for: (a) auto-exposure scaling, (b) transfer function (`TransferFunction` in unet_filter.h:11), (c) input/output normalization (`snorm` for normal-only mode at unet_filter.cpp:551), (d) tile overlap (`receptiveFieldBase = 174` / `receptiveFieldLarge = 202` at unet_filter.h:35-36)? None of these are in the `oidn-model` crate audited here.
3. The `fastMath` quality knob — currently no Rust equivalent. Worth adding once a backend that benefits from it (wgpu, CUDA) is wired up.
4. Cached const-tensor reuse (graph.cpp:528-544) — Rust loads weights once per `UNet` instance and clones if needed. Acceptable for CPU; revisit when multi-engine support arrives.
5. `Variant` Small vs Base auto-detection (U6) — should `from_tensor_names` peek at `enc_conv2.weight` channel count, or should it stay caller-driven?

---

End of report.
