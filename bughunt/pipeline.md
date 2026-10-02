# Pipeline audit — 2026-10-02

## Scope and evidence

Read in full: `crates/oidn-rs/src/filters/{mod,rt,rtlightmap,unet_runner}.rs`, `gpu_ops.rs`, `color.rs`, and `autoexposure.rs`, including in-file tests. No builds, tests, source edits, process termination, or rollbacks were performed. All file/shell operations used filesystem MCP; memory and sequential-thinking MCP were used. GitNexus tools were absent from the available inventory, so exact source reads and filesystem searches were used; no graph-based completeness claim is made.

The configured local C++ checkout `C:/projects/projects.rust.cg.offload/oidn` and repository `_ref` are absent. Official upstream **v2.4.1** was fetched through filesystem HTTP from `https://raw.githubusercontent.com/RenderKit/oidn/v2.4.1/`. The nine pipeline reference files are preserved under `bughunt/reference-v2.4.1`; the existing unet_filter.cpp was hash-verified identical before reusing it. The historical `reference-v2.3.3` folder is retained to avoid affecting other agents. Citations below use **REF** for official v2.4.1 paths (remote links are authoritative), not the absent local checkout. This prevents quoting stale architecture notes as evidence.

Confidence labels: **confirmed** means the code and reference demonstrably differ; it does not mean a user's visual regression has been reproduced. **Hypothesis** means runtime evidence is still required.

## Current dataflow and codepaths

```text
RtFilter::commit                      RtFilter::commit_tensor_model
  -> build_commit_artifacts             -> build_commit_artifacts
      -> registry OR user bytes             (different validation coverage)
      -> parse / variant / load Net
      -> tile::plan(RF_BASE)             CommittedRtFilter::execute_tensors
  -> mutable input dimension checks       -> validate_tensor_slot
  -> committed=true                        |
       |                                   |
       +----------------+------------------+
                        v
                 unet_runner::run_tensors
legacy Image run: decode HWC -> upload CHW --^
                        |
       optional whole-frame nonfinite replacement
                        |
       HDR exposure on raw signed RGB (negative values retained)
                        |
       per tile: slice -> ZERO PAD -> preprocess color
                                  -> clamp albedo
                                  -> remap normal [-1,1] to [0,1]
                        |
              concat -> Net::forward
                        |
     postprocess_color(snorm=false ALWAYS) -> crop -> accumulator
                        |
       tensor return OR host CHW->HWC -> ImageMut write
```

Source anchors: RT artifacts `rt.rs:496`; dispatch `rt.rs:798`; immutable dispatch `rt.rs:650`; host adapter `unet_runner.rs:266`; sanitization/exposure `unet_runner.rs:55`, `unet_runner.rs:86`; tile packing `unet_runner.rs:144`; output `unet_runner.rs:212`.

Expected primary input is **color, otherwise albedo, otherwise normal**. Only extra albedo/normal alongside color are auxiliary features. REF `devices/gpu/gpu_input_process.h:244-246` chooses that primary input, REF `core/unet_filter.cpp:551-564` chooses signed mode and uses it for both input/output.

## Confirmed findings (11)

### P1 — Normal padding becomes 0.5 rather than zero (high, plausible image-edge noise mechanism)

Rust `unet_runner.rs:161-178` inserts a raw source tile into a zero buffer; `unet_runner.rs:192-199` remaps the entire padded normal tensor with `*0.5+0.5`. Every out-of-source normal sample therefore becomes 0.5. The reference starts all channels at zero and calls getNormal only inside the valid source rectangle: REF [gpu_input_process.h:90-118](https://github.com/RenderKit/oidn/blob/v2.4.1/devices/gpu/gpu_input_process.h#L90). Correct remapping inside the rectangle is REF `gpu_input_process.h:69-79`; it must not affect padding.

Conditions: normal input present and tile has alignment/outer-image padding. Affects color+albedo+normal and normal-only. This is a mathematical input-contract violation, with potential border artifacts; visual magnitude is unmeasured.

History: `09bd9a0` (2026-05-16) introduced the correct normal remap but applied it after reflection. `91a261e` (2026-05-21) replaced reflection with zero-padding while leaving remapping after padding, introducing the specific 0 -> 0.5 error. This is an actionable regression window.

Systemic solution: preprocess only the source rectangle before writing into a zero tile, for every primary/auxiliary mode. Keep one zero-padding implementation; do not add a normal-only border patch.

### P2 — Albedo-only is treated as auxiliary and bypasses primary input scale and transfer (high)

Rust `rt.rs:479-493` returns Linear for every input set without color. In addition, `unet_runner.rs:188-190` only clamps albedo, never calling primary preprocessing. Upstream [rt_filter.cpp:63-70](https://github.com/RenderKit/oidn/blob/v2.4.1/core/rt_filter.cpp#L63) selects Linear for srgb or normal-only, otherwise PU for HDR or SRGB by default. Albedo-only with default srgb=false therefore uses SRGB. REF `gpu_input_process.h:37-54` applies input scale and primary transfer, since albedo is the primary image (REF `gpu_input_process.h:244`).

Consequences: albedo-only default mode feeds linear values to a network expecting sRGB-transformed values. With an explicit input scale, Rust skips multiplication but still applies reciprocal output scale (`unet_runner.rs:90-100,212`), so the pair is inconsistent even if srgb=true.

History: `912aecf` (2026-05-21) changed no-color transfer to Linear. Its commit message mistakenly claims rt_filter.cpp:65 makes every no-color input Linear; the actual condition includes normal. That explanation must be corrected in project docs. Earlier primary/auxiliary routing was already incomplete.

Systemic solution: share primary-input selection; derive transfer from actual primary kind and mode flags; process albedo-only using existing preprocess_input, and use auxiliary clamp only when color exists.

### P3 — Normal-only never decodes output to signed normals (high)

Rust normal-only input is remapped at `unet_runner.rs:192-199`, but `unet_runner.rs:212` hardcodes snorm=false for output. Normal-only produces unsigned output instead of signed [-1,1] normals, and its primary input scale is ignored.

REF [unet_filter.cpp:551-564](https://github.com/RenderKit/oidn/blob/v2.4.1/core/unet_filter.cpp#L551) uses `directional || (!color && normal)`; [gpu_output_process.h:58-69](https://github.com/RenderKit/oidn/blob/v2.4.1/devices/gpu/gpu_output_process.h#L58) applies `2*x-1`, lower clamp, non-HDR upper clamp, then output scale.

This can corrupt a workflow that denoises normal AOVs separately and then passes them to a clean-aux color filter. Whether the user's renderer follows that workflow remains unverified.

Systemic solution: carry signed primary mode through the shared runner and both bookend operations; keep auxiliary normal handling separate by role, not separate pipelines.

### P4 — Directional lightmaps lose signed input and signed output (high)

`rtlightmap.rs:294-319` selects Linear/non-HDR for directional data, but neither run nor run_tensors accepts signed mode. Color preprocessing at `unet_runner.rs:186` uses snorm=false, so negative irradiance gradients clamp to zero. Output at `unet_runner.rs:212` also uses snorm=false. Public docs promise signed normalization at `rtlightmap.rs:46-48`, but it is absent.

REF [rtlightmap_filter.cpp:56-62](https://github.com/RenderKit/oidn/blob/v2.4.1/core/rtlightmap_filter.cpp#L56) sets hdr=!directional; REF `core/unet_filter.cpp:551-564` propagates signed mode; REF `gpu_input_process.h:45-50` remaps signed primary input; REF `gpu_output_process.h:59-66` decodes output.

Systemic solution: add a signed-mode argument to existing shared runner, derived once from committed mode, and use existing gpu_ops signed branches. Do not implement another lightmap runner.

### P5 — Tensor autoexposure excludes boundary pixels and returns unity for nonempty small images (high)

`autoexposure.rs:152-155` returns 1 when either axis <16. Fixed 16x16 floor pooling at `autoexposure.rs:169-176` drops remaining rows/columns. Reference [autoexposure.h:26-31](https://github.com/RenderKit/oidn/blob/v2.4.1/core/autoexposure.h#L26) creates ceil(image/16) bins and [gpu_autoexposure.h:29-32,59-62](https://github.com/RenderKit/oidn/blob/v2.4.1/devices/gpu/gpu_autoexposure.h#L29) partitions all pixels by `bin*dimension/bin_count`.

The CPU implementation `autoexposure.rs:97-102` also uses different geometry: fixed cells with a short last cell, instead of balanced reference cells. Sharing constants does not make these algorithms equivalent.

Deterministic code-derived counterexamples, not executed: uniform 8x8 RGB=10 has CPU/reference scale 0.018 but tensor scale 1; in 17x16 images with luminous column 16, tensor exposure completely omits that column. Even uniform images are covered only by full 16-pixel cells. Existing tests at `autoexposure.rs:208-209,242` use multiples of16 and cannot detect these discrepancies.

Both public I/O paths currently invoke tensor exposure: `unet_runner.rs:94,302`. The CPU function is no longer the legacy runtime path despite its rustdoc.

History: `c357622` (2026-05-15) introduced floor pooling; `b3c9c62` (2026-05-15) routed legacy and tensor execution through it. These are concrete exposure-regression windows.

Systemic solution: one bin geometry/source sanitation contract used by CPU oracle and tensor implementation. Compute exact balanced bin boundaries; maintain device-local reductions. Simply turning pooling ceil_mode on still does not match balanced reference bins.

### P6 — Negative RGB affects autoexposure although network input later clamps it (medium/high)

The runner's whole-image sanitization only replaces nonfinite values (`unet_runner.rs:60-71`). Exposure computes luminance before any nonnegative clamp (`autoexposure.rs:161-164`); CPU exposure also sums signed luminance (`autoexposure.rs:109-119`). Actual network color later clamps at `gpu_ops.rs:53-55`. Reference [gpu_autoexposure.h:40-42](https://github.com/RenderKit/oidn/blob/v2.4.1/devices/gpu/gpu_autoexposure.h#L40) clamps RGB componentwise to [0,FLT_MAX] before luminance.

Counterexample: a16x16 image with half RGB=-1 and half RGB=+1 gives Rust mean0 -> scale1, whereas reference mean0.5 -> scale0.36. This is a code-derived result, not runtime reproduction. Negative RGB can occur in transforms, but its presence in the user's inputs is unconfirmed.

Systemic solution: same componentwise sanitize/clamp convention before exposure in both implementations, preserving the intentional ACEScg luminance feature.

### P7 — Large RT models use base receptive field in tile planning (high for multi-tile Large)

`rt.rs:555-563` distinguishes base and Large networks, but `rt.rs:578-583` always passes RECEPTIVE_FIELD_BASE. REF [unet_filter.cpp:263-268](https://github.com/RenderKit/oidn/blob/v2.4.1/core/unet_filter.cpp#L263) derives receptive field from parsed model topology. This is also independently covered by the geometry/model agents.

Systemic solution: model metadata from the actual parsed/loaded topology must determine architecture, channel counts and receptive field in all loaders. Avoid RF choices duplicated in filter modules.

### P8 — Validation differs between mutable/immutable RT APIs and custom/default weights (medium/high)

`rt.rs:738-742` rejects hdr+srgb in mutable commit; `rt.rs:421-448` immutable commit lacks that check. `rt.rs:504-516` skips registry validation entirely for caller weights, so invalid feature combinations can bypass established rejection. Reference [unet_filter.cpp:254-260](https://github.com/RenderKit/oidn/blob/v2.4.1/core/unet_filter.cpp#L254) checks parameters before model construction, and REF `unet_filter.cpp:420-443` checks auxiliary mode restrictions before user-weight selection.

Batch/channel validation uses debug_assert only (`rt.rs:601-605,710-716`), and mutable tensor commit checks only W/H (`rt.rs:775-785`). A release-mode batch>1 tensor can silently select batch0 (`unet_runner.rs:163`); extra channels are silently truncated to3 (`unet_runner.rs:164`); too few channels can panic downstream. Public run_tensors also trusts all shapes/plan.

Systemic solution: one runtime validator invoked by all commit and execution entry points, before expensive weight load; Result for invalid shapes/modes; validate custom weights structurally without routing them through a particular model filename.

### P9 — Cached state depends on input handles, causing rebuilds and missed shape invalidation (medium)

Image setters invalidate only on first presence (`rt.rs:299-317`, `rtlightmap.rs:182-187`), not geometry change. After successful commit, replacing an input with a different shape while output remains fixed skips cross-checks in execute, and host upload uses output geometry (`unet_runner.rs:292-300`). Same total pixel count with different W/H can silently reinterpret pixels; different counts can panic.

Mutable tensor execute clears handles (`rt.rs:839-841`); next setter sees None and invalidates (`rt.rs:329-334`), rebuilding weights every pass despite comments promising reuse (`rt.rs:363-366`). The immutable API avoids this state retention but has the validation gap above. No assertion about GPU read-after-free is established by comments at `rt.rs:829-838`; proving that requires backend ownership/lifetime investigation.

Systemic solution: separate committed input layout/geometry metadata from per-pass handles, sharing immutable committed artifacts across both façades. Same-shape values should reuse artifacts; actual layout/geometry changes must validate/invalidate.

### P10 — Sanitization switch is inconsistent across channels and repeated (medium)

`unet_runner.rs:61` gates whole-frame sanitation with nan_to_zero, but color preprocess unconditionally sanitizes (`gpu_ops.rs:51`) and output unconditionally sanitizes (`gpu_ops.rs:79`). Aux inputs only receive the gated pass. Thus false does not disable color sanitation and can change exposure semantics before primary sanitation. True sanitizes color once for the whole frame and again per tile.

Systemic solution: define the user flag's exact policy and apply it once at a shared input contract boundary, with reference-required output sanitation separate. Preserve the feature but remove redundant passes.

### P11 — Single-channel filter output selects red instead of averaging inverse-transformed RGB (high for scalar images)

Confirmed in both official CPU and GPU paths: REF [cpu_output_process.isph:48-66](https://github.com/RenderKit/oidn/blob/v2.4.1/devices/cpu/cpu_output_process.isph#L48) and [gpu_output_process.h:51-69](https://github.com/RenderKit/oidn/blob/v2.4.1/devices/gpu/gpu_output_process.h#L51) perform inverse transfer first, then for one-channel destinations compute arithmetic mean `(R+G+B)/3`, then signed demap/non-HDR clamp/output scale. RGB output from the CNN is not guaranteed to have equal channels even when input was broadcast from a scalar.

Rust always postprocesses three independent channels (`unet_runner.rs:212`, `gpu_ops.rs:79-96`), preserves them in the accumulator (`unet_runner.rs:226-240`), downloads all three (`unet_runner.rs:318-320`), and directly writes through `output.write_rgb_f32` (`unet_runner.rs:326`). `image.rs:268-270,285-286` explicitly selects red for a single-channel format. Therefore R32f/R16f **filter** outputs discard two CNN predictions instead of matching upstream channel averaging.

The direct generic `ImageMut::write_rgb_f32` contract and its test `tests/formats.rs:73-79` are internally consistent. Do not globally change this accessor to average: the averaging belongs to denoiser output processing. Other image writers may intentionally select red. This distinction avoids an unrelated API behavior change.

Code-derived example, not executed: with Linear transfer, hdr=false, scale1 and CNN RGB=(0,0,3), upstream averages to1 and clamps to1; Rust clamps to(0,0,1) and scalar write returns0. Averaging after the final clamp would return1/3 and would also be wrong. Actual visual noise causality for the user's render remains unproven; ordinary three-channel output is unaffected by this finding.

Systemic solution: thread destination channel count (or existing output layout metadata) through the shared runner into `postprocess_color`, reuse its inverse stage, and average there **before** signed demap/clamp/scale when output has one channel. Keep generic image accessor selection semantics unchanged and preserve tensor-native [1,3,H,W] API unless a separate scalar tensor output is requested. Do not implement a second host-only averaging pipeline.

Verification proposal for the separate approved pass: scalar f32 and f16 destinations; deliberately unequal CNN channel values; Linear/SRGB/PU/Log transfer; output values above the LDR cap; scale overrides; signed mode once wired. Check arithmetic mean after inverse, not luminance weighting and not mean of encoded or already-clamped output.

## Deduplication, unused-code and unfinished-feature review

- No TODO/FIXME/todo!/unimplemented!/allow(dead_code) was found in owned production sources. Absence of markers does not establish feature completeness.
- `OwnedImage`/`OwnedImageMut` and allocation/view methods are near-identical in `rt.rs:225-281` and `rtlightmap.rs:115-171`. Move ownership/geometry into the existing image module, preserving all supported formats and stride information. Validation must precede allocation to prevent overflow/panic paths.
- Lightmap model selection at `rtlightmap.rs:218-225` and file loading at `rtlightmap.rs:246-256` duplicate registry/weight-resolution responsibilities. Share established registry/resolver APIs rather than adding a third naming convention.
- `compute_scale` is public and used by parity tests; repository search found no production caller. Preserve it as CPU oracle/public API, repair algorithm and stale rustdoc (`autoexposure.rs:6-8`) instead of deleting it.
- Signed gpu_ops branches (`gpu_ops.rs:56-60,82-95`) are present but every production call passes false; this is unfinished feature wiring, not dead code. Output signed branch also skips upper1 clamp (`gpu_ops.rs:91`), unlike REF `gpu_output_process.h:65-66`. Wire and repair these branches.
- Constants are correctly shared between scalar and tensor transfer implementations (`color.rs:99-136`, `gpu_ops.rs:18-21`). Scalar/tensor algorithms legitimately differ by execution representation; preserving scalar oracle checks is better than forced abstractions.
- TransferState constructor unnecessarily computes the same forward curve twice (`color.rs:48-49`). Remove the duplicate calculation in an approved refactor; no noise mechanism is established.
- `quick_stats` (`unet_runner.rs:341-359`) and TensorStats (`unet_runner.rs:411-459`) duplicate finite-statistics logic with different denominators and all-invalid handling. Reuse one implementation. Legacy quick_stats is computed even if debug logging is disabled (`unet_runner.rs:287-290,322`).
- `OIDN_TRACE_TENSORS` can force full tensor host readback (`unet_runner.rs:366-378`), while successful output is trace-level only (`unet_runner.rs:381`). A debug-only logger may therefore incur reads without showing requested statistics. Clarify diagnostics contract and use shared stats.
- Lightmap quality intentionally maps to a single variant (`rtlightmap.rs:242`), consistent with REF `rtlightmap_filter.cpp:19-20`; do not remove it or invent non-existent model files.
- Image and tensor runners already share one core (`unet_runner.rs:302`). Retain that architectural success while repairing input-role/mode contracts.
- Weight resolver integration and channel/format concerns are coordinated with other agents; this report does not authorize deleting any public surface.

## Priority and verification plan for approval

- [x] Read every owned source and in-file test.
- [x] Compare transfer constants/order, input roles, padding and output to official v2.4.1.
- [x] Compare exposure bins/sanitation with official reference.
- [x] Search unfinished markers and identify public/oracle surfaces before proposing deletion.
- [x] Inspect git history read-only; identify concrete regression windows.
- [x] Save confirmed findings/decision and relation in memory MCP.
- [ ] Acquire a frozen noisy input and matching albedo/normal, exact model key, backend, exposure, dimensions and last known-good output/commit.
- [ ] Implement one mode/layout/geometry contract used by RT mutable/immutable and lightmap.
- [ ] Apply primary preprocessing and auxiliary preprocessing before zero-padding; propagate signed mode through shared runner.
- [ ] Repair exact exposure geometry and shared sanitation; preserve ACEScg feature.
- [ ] Use actual model metadata for tile receptive field and loader integration.
- [ ] Add single-channel averaging after inverse transfer in shared output processing; preserve generic ImageMut direct-write contract.
- [ ] Unify validation and committed cache semantics.
- [ ] In a separate approved verification pass, add semantic parity fixtures: small/odd/edge-lit exposure, negative/nonfinite samples, normal-only signed output, albedo-only with scale/srgb, signed lightmap, padded normal tiles, release invalid shapes and cache reuse.
- [ ] Compare CPU and wgpu using the same weights and frozen input, then compare against upstream OIDN. Include per-stage output diagnostics and tiled/single-tile differences; avoid relying only on round-trip transfer tests.
- [ ] Bisect identified history windows on isolated checkouts if before/after fixtures indicate them. No rollback in the shared working tree.

Noise causality remains **unproven**. P1 and P7 predict boundary/tile-associated defects; P5/P6 predict exposure-dependent denoising changes; P2/P3 predict feature-prefilter defects; P4 applies to lightmaps. Uniform residual noise throughout an otherwise valid full-size color render still requires investigating weights/topology/backend and renderer bridge contracts. The Burn0.22 migration message claims numerics unchanged; it is not evidence of numerical parity.
