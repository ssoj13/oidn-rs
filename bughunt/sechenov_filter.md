# Sechenov RT filter pipeline parity audit

- Agent: sechenov
- Date: 2026-05-21
- Mode: READ-ONLY (no source edits)
- Scope: RT + RTLightmap filter surface vs `_ref/oidn` (offload mirror at `C:/projects/projects.rust.cg.offload/oidn`)

## Verdict

PARTIAL PARITY with several concrete gaps.

Core inference flow (UNet topology, weight selection table, tile loop, autoexposure pre-pass, transfer-function routing) matches the reference at the algorithmic level. The Rust port loses, or only partially exposes, several public-surface affordances of the reference RT filter:

- No runtime mode-toggle API (no `set_hdr` / `set_srgb` / `set_cleanAux` post-build; everything is captured by the builder).
- No `unsetImage`, no input image replacement that drops back to *no* input.
- No `cleanAux` semantics beyond model selection — the reference uses `cleanAux` *only* for model selection too, but the Rust port does not expose the *chain* (prefilter-then-RT) idiom; see notes in issues table.
- `directional` is wired on `RtFilter` (registry returns `rtlightmap_dir`!) — this is an out-of-band path that does not match the reference RT filter at all; the reference dispatches `directional` only on the `RTLightmap` filter.
- HDR-mode check vs `srgb` mutual exclusion and `directional` ↔ `hdr/srgb` mutual exclusion is NOT validated in commit() — `unet_filter.cpp:377-380` throws here.
- `Quality::Default` (the C enum's `OIDN_QUALITY_DEFAULT`) has no Rust analogue; defaults to `High` (matches `defaultQuality` in `unet_filter.h:48`, but the explicit "default" sentinel is missing from public API).
- No `set_image("name", ...)` string-keyed API, no `getInt`/`getFloat` introspection.
- Progress callback exists on `RtFilter` only; `RtLightmapFilter` has *no* progress hook.
- No in-place tiled filtering (`inplace` + `outputTemp` + `imageCopy` from `unet_filter.cpp:120-125, 583-590, 641-646`).
- Cancellation support: Rust returns `OidnError::Cancelled` if callback returns false — matches reference contract semantically (reference uses progress return-value to early-out via Exception). OK.

## Option-surface matrix (RT)

| Ref option (`rt_filter.cpp` / `unet_filter.cpp`) | Rust support | Location | Notes |
|---|---|---|---|
| `setImage("color", ...)` | YES (`set_color`) | `rt.rs:305` | typed setter, no string key |
| `setImage("albedo", ...)` | YES (`set_albedo`) | `rt.rs:312` | |
| `setImage("normal", ...)` | YES (`set_normal`) | `rt.rs:319` | |
| `setImage("output", ...)` | YES (`allocate_output`) | `rt.rs:379` | output is allocated by the filter; no caller-supplied image |
| `unsetImage("color"/...)` | NO | — | cannot drop a previously-set input; the slot is sticky for the lifetime of the filter |
| `setData("weights", blob)` | YES (`weights(bytes)`) | `rt.rs:103` | builder-only; cannot replace after build |
| `updateData("weights")` | NO | — | reference allows in-place edit + dirtyParam recomputation |
| `unsetData("weights")` | NO | — | no way to clear user weights once set |
| `setInt("hdr", v)` | YES (`hdr(v)`) | `rt.rs:73` | builder-only |
| `setInt("srgb", v)` | YES (`srgb(v)`) | `rt.rs:77` | builder-only |
| `setInt("cleanAux", v)` | YES (`clean_aux(v)`) | `rt.rs:85` | builder-only, used only for weight selection |
| `setInt("quality", v)` | YES (`quality(q)`) | `rt.rs:89` | `Default` sentinel missing |
| `setInt("maxMemoryMB", v)` | PARTIAL (`max_memory_mb(mb)`) | `rt.rs:111` | semantics differ — see issue #5 |
| `setFloat("inputScale", v)` | YES (`input_scale(s)`) | `rt.rs:93` | maps `None` → autoexposure; reference uses `NaN` sentinel |
| `getInt("tileAlignment")` | NO | — | no introspection |
| `getInt("tileOverlap")` | NO | — | no introspection |
| `getInt("hdr"/"srgb"/"cleanAux"/"quality"/"maxMemoryMB")` | NO | — | no introspection |
| `getFloat("inputScale")` | NO | — | no introspection |
| `setProgressMonitorFunction(fn, userPtr)` | YES (`set_progress`) | `rt.rs:466` | `'static` closure rather than fn ptr + user pointer; OK |
| `commit()` | YES | `rt.rs:725` | |
| `execute(SyncMode)` | YES (`execute()`) | `rt.rs:787` | SyncMode parameter absent — Burn calls block on readback, async semantics not exposed |
| `cleanAux` prefilter chain | PARTIAL | n/a | weight selection works; orchestration (filter albedo→normal→RT) is left to caller |
| Tile size override | INDIRECT | `rt.rs:111` | only via `maxMemoryMB`; reference also has no direct override but uses `defaultMaxTileSize` |
| Weights selection (`Quality::High` → `_large` then fallback to base; `Quality::Fast` → `_small` fallback; `Quality::Balanced` → base) | YES | `registry.rs:65` | matches `unet_filter.cpp:446-459` |
| Device parameter | YES | `rt.rs:43` | Burn `B::Device` |
| `nan_to_zero` (input sanitisation) | YES (extra) | `rt.rs:122,459` | matches reference kernel contract (`devices/cpu/cpu_input_process.isph` head) explicitly; not a public-API option in the reference |
| `directional` on `RtFilter` | YES (extra) | `rt.rs:81` | NOT PRESENT on reference `RTFilter`; see issue #1 |

## Option-surface matrix (RTLightmap)

| Ref option | Rust support | Location | Notes |
|---|---|---|---|
| `setImage("color", ...)` | YES (`set_color`) | `rtlightmap.rs:132` | |
| `setImage("output", ...)` | YES (`allocate_output`) | `rtlightmap.rs:134` | |
| `unsetImage(...)` | NO | — | |
| `setInt("directional", v)` (also flips `hdr`) | YES (`directional(v)`) | `rtlightmap.rs:47` | builder-only, `hdr` is implicit |
| `setInt("quality", v)` | ACCEPTED, IGNORED | `rtlightmap.rs:157` | single-variant filter; `let _ = self.quality;` — quality has no effect on weights (matches reference, which selects from the single `models.hdr`/`models.dir` blob) |
| `setInt("maxMemoryMB", v)` | NO | — | reference path inherits this from `UNetFilter::setInt`; Rust hardcodes `DEFAULT_MAX_TILE_SIZE` (`rtlightmap.rs:180`) |
| `setFloat("inputScale", v)` | YES (`input_scale`) | `rtlightmap.rs:49` | |
| Progress | NO | — | not exposed (passes `None` at `rtlightmap.rs:222`) |
| HDR-default | YES | `rtlightmap.rs` builder default | `directional=false` → HDR mode with Log transfer |
| Weights (`rtlightmap_hdr` vs `rtlightmap_dir`) | YES | `rtlightmap.rs:148` | matches `rtlightmap_filter.cpp:19-20` |
| Transfer routing (Log vs Linear) | YES | `rtlightmap.rs:201-205` | matches `rtlightmap_filter.cpp:24-30` |

## Input-combo matrix (weight selection)

Reference truth table from `unet_filter.cpp::getWeights` 394-466 + `rt_filter.cpp::RTFilter()` 33-61.

| Combo (color/alb/nrm) | hdr | srgb | cleanAux | Ref selects (base, Quality=Balanced) | Rust selects (`registry.rs:select_rt`) | Match? |
|---|---|---|---|---|---|---|
| C, –, – | true | – | – | `rt_hdr` | `rt_hdr` (line 37) | OK |
| C, –, – | false | true | – | `rt_ldr` | `rt_ldr` (line 38) | OK |
| C, –, – | false | false | – | `rt_ldr` | `rt_ldr` (line 39) | OK |
| C, A, – | true | – | – | `rt_hdr_alb` | `rt_hdr_alb` (line 42) | OK |
| C, A, – | false | – | – | `rt_ldr_alb` | `rt_ldr_alb` (line 43) | OK |
| C, A, N | true | – | false | `rt_hdr_alb_nrm` | `rt_hdr_alb_nrm` (line 45) | OK |
| C, A, N | true | – | true | `rt_hdr_calb_cnrm` | `rt_hdr_calb_cnrm` (line 46) | OK |
| C, A, N | false | – | false | `rt_ldr_alb_nrm` | `rt_ldr_alb_nrm` (line 47) | OK |
| C, A, N | false | – | true | `rt_ldr_calb_cnrm` | `rt_ldr_calb_cnrm` (line 48) | OK |
| –, A, – | false | – | – | `rt_alb` | `rt_alb` (line 50) | OK |
| –, A, – | true | – | – | THROW "hdr mode not supported for albedo filtering" (`unet_filter.cpp:423`) | accepts and returns `rt_alb` (line 50) | DIVERGENT (issue #2) |
| –, –, N | false | false | – | `rt_nrm` | `rt_nrm` (line 51) | OK |
| –, –, N | true | – | – | THROW "hdr/srgb not supported for normal filtering" (`unet_filter.cpp:428`) | falls through → `None` → `UnsupportedFeatures` (line 51 pattern misses `hdr=true`) | OK-ish (different error; behaviour acceptable) |
| –, A, N | – | – | – | THROW "invalid combination of input features" (`unet_filter.cpp:434`) | falls through → `UnsupportedFeatures` | OK |
| Quality::High routing | — | — | — | prefer `*_large` blob if present, else base | `quality_candidates(base, High) = [base_large, base]` (`registry.rs:68`) | OK |
| Quality::Fast routing | — | — | — | prefer `*_small`, else base | `quality_candidates(base, Fast) = [base_small, base]` (`registry.rs:70`) | OK |
| Quality::Balanced | — | — | — | base only | `[base]` (`registry.rs:69`) | OK |
| RT-lightmap | — | — | — | always HDR=true (`rtlightmap_filter.cpp:16`), `directional` flips to `rtlightmap_dir` and `hdr=false` (`rtlightmap_filter.cpp:60-62`) | `rtlightmap.rs:148` `directional → rtlightmap_dir`, else `rtlightmap_hdr` | OK at weight level — but Rust path's `hdr` flag is implicit (no separate `is_hdr`/`directional` interlock; see issue #6) |

## Per-issue table

| ID | Sev | Rust file:line | Ref file:line | Description | Fix |
|---|---|---|---|---|---|
| 1 | HIGH | `rt.rs:47,81-83` and `registry.rs:40` | `rt_filter.cpp:73-117` (RTFilter::setInt has no `directional`) | `RtFilter` exposes a `directional()` builder method, and `select_rt` routes `(color, –, –, hdr=false, srgb=*, directional=true)` to `rtlightmap_dir`. The reference RTFilter has NO `directional` knob; `rtlightmap_dir` is owned by the *RTLightmap* filter. Cross-feeding a lightmap weight blob through the RT filter is a category error — channel layout matches (3 in) but the network was trained for log-irradiance, not RT colour. | Either delete `directional` from `RtFilter`/`RtFilterBuilder` and the `(_,_,_,_,_,true,_)` arm in `select_rt`, OR document it as an internal escape hatch and gate it from public API. |
| 2 | MED | `rt.rs:497-543`, `registry.rs:50` | `unet_filter.cpp:419-431` | Reference throws on `(albedo only, hdr=true)` and on `(normal only, hdr || srgb)`. Rust accepts `(–, A, –, hdr=true)` silently and returns `rt_alb` — the aux filters were never trained for HDR/PU and the result will be garbage. | In `select_rt`, add explicit early errors mirroring lines 423/428. Surface as `OidnError::UnsupportedFeatures` or a new variant. |
| 3 | MED | `rt.rs:485-495` | `rt_filter.cpp:63-71` | Transfer-function selection mismatch on `(color absent, normal present)`. Reference returns `Linear` for `(!color && normal)` (line 65); Rust `transfer_kind()` does not consult input presence at all — it returns `SRGB` (default branch) when only normal is set with `hdr=false, srgb=false`. The downstream snorm handling in `run_tensors` does pick a different code path for the channel, but the `TransferFunction` plumbed in is still `SRGB` instead of `Linear`. | Pass `has_color`/`has_normal` flags into `transfer_kind()` or compute the kind at `build_commit_artifacts` time using the same combo bits as `select_rt`. |
| 4 | MED | `rt.rs:725-785` | `unet_filter.cpp:346-380` (checkParams) + 377-380 | Rust commit() does not enforce `directional && (hdr || srgb)` mutual exclusion, nor `hdr && srgb` mutual exclusion. Reference throws on both. The exclusion happens to be unreachable in some configurations because the builder set them mutually only via separate calls, but the public surface still permits all three. | Add a check in `RtFilter::commit()` (or before `build_commit_artifacts`): error if `(directional && (hdr || srgb))` or `(hdr && srgb)`. |
| 5 | MED | `rt.rs:569-580` | `unet_filter.cpp:300-326` | `maxMemoryMB` semantics differ. Reference iteratively shrinks tiles in `init()` so the *constructed graph* fits within the budget (calls `buildModel(maxMemoryByteSize)` in the loop). Rust converts MB → pixel-cap via a constant `bytes_per_pixel` estimate (96 or 256 × 4 × 4) at line 572-575, then caps tile size in the planner. The estimate is coarse: it ignores skip connections, scratch arena, transfer LUTs, output-temp for in-place tiling. Real memory usage can exceed the budget. | Either document the heuristic explicitly or implement a build-and-shrink loop similar to the reference. |
| 6 | LOW | `rtlightmap.rs:201-205` | `rtlightmap_filter.cpp:56-67` | Rust derives `is_hdr` from `!directional`, but does not expose an `hdr` setter so the two are perfectly anti-correlated. Reference allows `setInt("directional", v)` to flip `hdr` as a side effect (and `setInt("hdr", v)` is rejected at the UNetFilter layer since RTLightmap does not override it). Net behaviour matches but the path is fragile — if `directional` is later toggled without rebuilding, `is_hdr` would be wrong. Acceptable given builder-only ergonomics, but worth a doc-comment. | Add `///` doc on `directional()` explaining the implicit HDR flip. |
| 7 | LOW | `rt.rs:725-785` | `unet_filter.cpp:115-143` (dirty / dirtyParam bookkeeping) | Reference uses `dirty`/`dirtyParam` to skip recomputation when only pixel content changed. Rust replicates this partially via `needs_invalidate` in `set_color`/`set_albedo`/`set_normal` (rt.rs:306-324) and `same_dims` check in `allocate_output` (rt.rs:385). Tensor-mode `set_*_tensor` also tracks shape (rt.rs:336-362). Coverage is decent but inconsistent: changing `hdr`/`srgb`/`cleanAux`/`quality` on the *builder* by definition produces a new filter, so this is mostly fine — there is no setInt-style runtime mutator. | None; document that runtime tuning requires rebuilding. |
| 8 | LOW | `rtlightmap.rs:132-137` | `unet_filter.cpp:120-125,141-143` | `RtLightmapFilter::set_color`/`allocate_output` unconditionally set `committed = false`. The `RtFilter` has the `same_dims` optimisation; the lightmap filter does not. Performance regression for repeated lightmap passes at the same resolution. | Mirror `RtFilter::allocate_output` dim-cache optimisation. |
| 9 | LOW | `rt.rs` (no progress on RTLightmap), `rtlightmap.rs:222` | `unet_filter.cpp:153-169` (uses `progressFunc` if present) | RTLightmap filter cannot accept a progress callback (passes `None` to `unet_runner::run`). Reference inherits progress from `Filter::setProgressMonitorFunction`. | Add a `set_progress` to `RtLightmapFilter` mirroring `RtFilter::set_progress`. |
| 10 | LOW | `rt.rs:495` | `rt_filter.cpp:65` | When `hdr=false, srgb=false, directional=false, color=true`, Rust transfer falls to `TransferFunction::SRGB`. Reference returns `TransferFunction::Type::SRGB` only via the trailing `else` (line 70). Matches. (Listed for confirmation, not a defect.) | None. |
| 11 | INFO | `rt.rs:799` | `unet_filter.cpp:145-251` | `execute()` has no `SyncMode` equivalent. Rust calls block on Burn ops. For wgpu/cubeCL the readback is synchronous so this is OK in practice. | Document the divergence. |
| 12 | INFO | `rt.rs:506-544` | `unet_filter.cpp:441-460` | User weights bypass the registry and feature combo entirely — caller is responsible for channel-count match. Reference does the same (`if (userWeightsBlob) weightsBlob = userWeightsBlob;` at line 441). Match. | None. |
| 13 | LOW | `rt.rs:553-557` | `unet_filter.cpp:263` (`largeModel = constTensors->find("enc_conv1b.weight") != constTensors->end();`) | Rust does variant detection by *filename suffix* (`variant_from_stem`) when the registry resolved the blob, and only by tensor names when user-supplied. The reference *always* uses tensor-name detection. For built-in registry blobs this is consistent only because the stems' suffixes correctly mirror the contents; if a packager swaps a `_large` filename's payload, Rust would mis-instantiate. | Always detect variant from tensor names; treat the stem suffix only as a hint / for telemetry. |
| 14 | LOW | `rt.rs:435-455`, `rt.rs:622-699` | n/a | `CommittedRtFilter` API (immutable, no per-pass state) is a Rust-only addition for tensor pipelines. Behaviour matches the same commit path. Not a parity issue but expands surface area beyond reference. | None. |
| 15 | LOW | `rtlightmap.rs:157` | `unet_filter.cpp:445-459` | `let _ = self.quality;` — the lightmap path drops `quality`. The reference also has only `models.hdr.base` / `models.dir.base` (no _small/_large variants for lightmap, see `rtlightmap_filter.cpp:19-20`), so `getWeights` returns the same blob regardless of quality. Match. | None. |
| 16 | MED | `rt.rs` (no in-place support), `unet_runner.rs` | `unet_filter.cpp:120-125, 583-590, 641-646` | Reference detects buffer overlap between any input image and the output image, allocates a temporary output, runs the tile loop into it, then `imageCopy` into the user buffer. Rust filter owns its output buffer (`OwnedImageMut`) so the overlap case literally cannot occur via the legacy API; tensor-mode path receives a fresh accumulator. Acceptable architectural divergence, but a tensor caller could accidentally re-use the same buffer as input AND target (Burn does not protect against this either). | Document the divergence; consider adding a debug-assert in tensor mode. |
| 17 | LOW | `rt.rs:564` | `unet_filter.cpp:262` (`const bool fastMath = quality != Quality::High;`) | Reference flips a `fastMath` flag on the graph builder for non-`High` quality. Rust does not differentiate kernel-level math precision by quality — Burn / cubeCL kernels are fixed precision. | None at this layer; revisit if Burn exposes per-op fast-math toggles. |

## Validation / setup phase observations

- Rust `RtFilter::commit()` (`rt.rs:725-785`) does check input/output dimension consistency and at-least-one-input presence — matches reference checkParams() for those clauses.
- Pixel-format validation: reference checks `isSupportedFormat` for Float3/Half3/Float2/Half2/Float/Half (`unet_filter.cpp:353-358`). Rust accepts whatever `PixelFormat` is plumbed; type-system narrows it but there is no explicit allow-list check.
- Output channel-count match (`input->getC() != output->getC()` at `unet_filter.cpp:369`): no equivalent in Rust because output is always 3 channels by construction (hardcoded `out_channels = 3` at `rt.rs:548`).
- HDR ↔ srgb ↔ directional exclusivity: NOT enforced in Rust (issue #4).

## Execute phase observations

- Tile loop order (`for h { for w }`) matches between Rust `unet_runner::run_tensors` (`unet_runner.rs:145`) iterating `plan.jobs` (TilePlan generates jobs in H-major then W-major order; verify in `tile.rs`) and `unet_filter.cpp:201-241`.
- Autoexposure pre-pass: Rust gates on `hdr && color.is_some()` (`unet_runner.rs:95-99`) — matches `unet_filter.cpp:172-185` which also gates on `hdr && color`. Match.
- Inverse transfer skipped for `Linear` (`unet_runner.rs:194-198`) — reference TransferFunction::Linear is a no-op forward+inverse by definition, so behaviour matches.

## cleanAux semantics

Reference: `cleanAux` in `getWeights` (`unet_filter.cpp:411-415`) only selects the `*_calb_cnrm` weight set; the *prefiltering chain* (denoise albedo with `rt_alb`, denoise normal with `rt_nrm`, then run RT with `cleanAux=true`) is the caller's job in both libraries — there is no auto-chain on the C++ side either. The training docs assume the caller orchestrates the prefilter. So Rust matches the reference on cleanAux *as a flag*. No issue.

## Quality vs model variant mapping

`registry.rs:65-72` is exact mirror of `unet_filter.cpp:446-459`:
- `Quality::High` → `_large` then base; reference: `model->large ? model->large : model->base` (line 451).
- `Quality::Balanced` → base; reference: `model->base` (line 454).
- `Quality::Fast` → `_small` then base; reference: `model->small ? model->small : model->base` (line 457).
- `Quality::Default` C-enum value: no Rust analogue; `Quality::default()` returns `High` (`filter.rs:11`), which matches `defaultQuality = Quality::High` (`unet_filter.h:48`).

## inputScale auto vs manual

`unet_runner.rs:93-103`:
- `Some(s)` → use `s` directly.
- `None` + `hdr` + color tensor present → call `autoexposure::compute_scale_tensor`.
- `None` + non-HDR → fixed `1.0`.

Reference `unet_filter.cpp:171-189`:
- `math::isnan(inputScale)` (i.e. unset) + `hdr` → run `autoexposure->submit(...)`, set scale from autoexposure result.
- `math::isnan(inputScale)` + non-HDR → `setInputScale(1)`.
- Otherwise → set to user value.

Match (Rust uses `Option<f32>` instead of NaN sentinel — equivalent).

## Progress / cancel

- `RtFilter::set_progress` (`rt.rs:466`) accepts `FnMut(f32) -> bool + 'static`; returning false yields `OidnError::Cancelled` (`unet_runner.rs:220`).
- Reference uses `ProgressMonitorFunction func(void* userPtr, double n)` returning `bool`; same cancel semantics via Exception in `Progress::update`.
- `RtLightmapFilter` has no progress wiring (issue #9).

## Error reporting paths

Rust `OidnError` (in `error.rs`, not opened here but observable from uses): variants seen — `Unset(&str)`, `Inconsistent(&str)`, `UnsupportedFeatures`, `MissingModel(PathBuf)`, `Io(io::Error)`, `Cancelled`. Reference uses C++ `Exception` with `Error` codes (`InvalidArgument`, `InvalidOperation`, `OutOfMemory`, `Cancelled`, ...). No 1:1 mapping but coverage is comparable.

Concrete gaps:
- Rust does not report a distinct error for invalid format (reference has `InvalidOperation` "unsupported input image format").
- Rust does not report `InvalidOperation` when (`directional && hdr`), (`hdr && srgb`), or (`albedo-only && hdr`) — see issues #2, #4.

## Dead / unused / unfinished code

No `TODO` / `FIXME` / `XXX` markers found in `crates/oidn-rs/src/filters/*` (grep with Grep tool returned no matches).

Other observations:
- `rtlightmap.rs:157` `let _ = self.quality;` — explicit unused-field suppression. Intentional and documented inline. OK.
- `Quality::Balanced` enum doc (`filter.rs:13`) says "Same network width as High for v0.1, future variant slot." That description is now incorrect after `registry.rs:69` was implemented: `Quality::Balanced` correctly maps to *base* (not large), so it differs from `High` (which prefers large). Update stale doc-comment.
- `rt.rs:54` builder field `nan_to_zero: bool` — default true, mirrors reference contract. Not present as a public option in `_ref/oidn`; this is a Rust extension. Documented.
- `RtFilter` retains `weights_dir: PathBuf` on the built struct (`rt.rs:161`) so future re-commits can re-resolve. Reference does not need this because weights are baked-in blobs.

## Open questions

1. Should `RtFilter::directional` be removed (issue #1) or kept as an internal toggle for the lightmap-from-RT path? The reference does not allow this combination at all.
2. Is the `maxMemoryMB` pixel-cap heuristic (issue #5) considered "close enough" for the Rust port, or should we port the iterative build-and-shrink loop?
3. Is the absence of an `unsetImage` / runtime mode-toggle API a deliberate design choice tied to Rust's builder pattern, or a porting gap to fill?
4. The reference allows multiple subdevices (`device->getNumSubdevices()`); the Rust port assumes a single device. Document the divergence or add multi-device support later?
5. `SyncMode` parameter on `execute()` — should the Rust port expose async submission for wgpu backends that support it (eventually)?
6. The Rust `Quality::High` enum carries `#[default]` (`filter.rs:9-11`). Reference enum `Quality::Default` is a separate sentinel that resolves to `High` via `setInt("quality")` (`unet_filter.cpp:48`). Worth exposing `Quality::Default` in Rust for API symmetry, or keep the simpler model?

End of report.
