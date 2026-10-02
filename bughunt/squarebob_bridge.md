# Squarebob OIDN bridge audit

Date: 2026-10-02. Static source review only; no builds, tests, production edits, or symptom reproduction by this agent. Renderer checkout: C:/projects/projects.rust.cg/cglibs/squarebob-rs, HEAD5f73509 during review. OIDN checkout: C:/projects/projects.rust.cg/cglibs/oidn-rs, baseline bebbfc5. Other agents may update dependency manifests/lockfiles concurrently; preserve their work.

Citation prefixes:
- SB = C:/projects/projects.rust.cg/cglibs/squarebob-rs.
- OI = C:/projects/projects.rust.cg/cglibs/oidn-rs.
- CU = C:/Users/joss1/.cargo/git/checkouts/cubecl-058c47895211d464/31c5506, pinned revision31c5506cfc1dd9d350124910fb51cee8b54a31e1 (SB/Cargo.toml:89).
- Bridge = SB/crates/pt-denoise-oidn/src/lib.rs.

## Scope and checked sources

- [x] Read squarebob AGENTS.md before research.
- [x] Read the entire bridge source (1041 lines), including the separately reread middle section.
- [x] Read bridge Cargo.toml and relevant root dependency pins.
- [x] Read app denoiser settings and invocation/display paths.
- [x] Check mode, quality, exposure, clamp, buffer ownership, normalization, row padding, and filter caching.
- [x] Trace renderer source/output accessors and display composition.
- [x] Check pinned CubeCL external-resource submission and allocation ownership.
- [x] Compare relevant OIDN audit defects with modes actually reachable from this bridge.
- [x] Identify existing test and proposed controlled runtime verification.
- [ ] Runtime checks and numerical symptom reproduction: parent-owned testing pass.

GitNexus MCP was available to this agent. graph_status initially reported indexed e662a71 vs HEAD5f73509. reanalyze(repo=squarebob-rs,scope=compare,base_ref=e662a71,progress_tail_limit=2) updated14 files; subsequent status was fresh with no unstaged changes at that point. query(repo=squarebob-rs,query="OIDN denoise progressive accumulator render sampling albedo normals") returned actual bridge/App definitions. Ranking also included irrelevant processes; direct filesystem MCP source reads remain the evidence. This successful MCP path is distinct from the parent's timed-out CLI catalog scan. No graph completeness claim.

## Actual dataflow

```text
PT accumulation / AOV sums
  |
  +--> normalized Rgba32Float PT output texture
  |      render-3d/lib.rs1141 -> pt-megakernel/compute.rs1893
  +--> active backend albedo/normal vec4 buffers
         render-3d/lib.rs1153,1159 -> compute.rs1918,1927
  |
App current_spp / periodic-final-manual trigger (treemap_view.rs1491-1532)
  |
OidnDenoiser::denoise(ctx, encoder, color_tex, AOV buffers, current_spp)
  |
shared CubeCL Device/Queue -> tensor zeros -> get_resource
  |
external copy: texture -> HWC RGBA padded rows
external copy: AOV buffers -> HWC vec4 tensors
  | queue.submit
  +--> trim padded columns -> sample-dependent luminance clamp
  |                           -> HWC RGB to NCHW RGB
  +--> AOV RGB / max(W,1) -> NCHW RGB
  |
weights resolve -> cached bytes -> immutable CommittedRtFilter
  |                          fresh tensors each pass
  +--> input scale: env override > physical exposure > autoexposure
  |
execute_tensors -> CHW RGB -> HWC RGBA(alpha=1) -> row padding
  |
get_resource(flush pending CubeCL streams, allocation pin)
  |
external copy buffer -> separate result_texture -> device.poll
  |
App result_view -> composite_overlay -> render_view -> display
```

The bridge writes its own result_texture (Bridge:649-656), while the next input is the renderer's output_texture (SB/src/app/treemap_view.rs:1538-1547). Display composition targets state.targets.render_view, not the PT color texture (SB/crates/render-3d/src/lib.rs:1217-1222). These inspected paths contain no denoised-to-raw feedback. This limited bridge conclusion does not replace the parent's full accumulator/shader audit.

## B1 — Confirmed sample-dependent denoiser input; strong highlight-specific diagnostic candidate

The app defaults to full AOV, Base quality, auto=true, interval128, clamp10, adaptive=true, Physical camera (SB/crates/render-shared/src/lib.rs:1128-1135). New OidnDenoiser itself defaults to clamp0 (Bridge:170); the app explicitly forwards its clamp/adaptive values on each pass (SB/src/app/treemap_view.rs:1579-1581).

Bridge:457-468 computes:

```text
t = clamp(current_spp / 256, 0, 1)
s = t*t*(3 - 2*t)
effective = min(2,user_clamp) + (user_clamp - min(2,user_clamp))*s
```

Constants256 and2 are Bridge:42,48. For app default10, effective threshold is6 at128SPP and10 from256SPP onward. The actual operation scales RGB together by min(1,threshold/max(Rec.709 luminance,1e-6)), preserving finite RGB ratios (Bridge:860-897), before OIDN preprocessing/exposure (Bridge:469-485,515-518,627).

Therefore identical HDR color/AOV inputs denoised with current_spp128 and256 can yield different images solely due to this policy. Bright outliers clipped heavily at the first default preview are allowed more energy at the next. This matches the location/time shape of the reported symptom sufficiently to justify an isolated experiment; it is not proof of causality or an established algorithmic bug. After256SPP this specific threshold is constant, so it cannot alone explain indefinite growth beyond that point.

Correct verification: hold a captured color/AOV snapshot, model, exposure and dimensions fixed; run the bridge at128 vs256SPP with adaptive enabled, then repeat with adaptive disabled and fixed clamp10, then clamp0. Separately repeat identical snapshot/SPP many times. Changing raw progressive input and spp simultaneously cannot separate a network or memory defect from this intentional input change. Do not remove clamping or features merely because the visual difference exists.

## B2 — Confirmed final-pass scheduling defect after a periodic preview

SB/src/app/treemap_view.rs:1510-1513 permits auto_final only when oidn_denoised_this_accumulation is false. Every successful pass, including periodic previews, sets that flag true at1609. Periodic trigger1519-1522 requires current_spp-last_interval_spp >= interval. No explicit final-target guard appears in that periodic condition despite comment1515.

Source-derived example: target300SPP, interval128, previews at128 and256. After the256 preview the flag is true and last_interval=256. At final300, auto_final is false and delta44<128, so final300 is not denoised. The displayed denoised image remains the earlier256SPP result. A target512 with exact128 increments can still trigger at512 through the periodic condition; this does not invalidate the nonmultiple-target defect.

Systematic proposal: one source of truth for successfully denoised sample count plus accumulation identity; distinguish completion at current target from any earlier preview. Final scheduling should compare the completed count/identity against current final count, and periodic scheduling should use the same count. Preserve manual, periodic and final functionality. Existing state also identifies resets only by current_spp decreasing1502-1506; reset correctness when counts are equal requires parent renderer-generation audit.

## B3 — Confirmed changing display snapshots, not continuous denoise

The app only calls OIDN on manual/final/interval events (treemap_view.rs:1509-1532). Successful output persists as denoised display state1608-1612 and is composited at1209-1215 on later frames. Raw accumulation can advance while the denoised preview remains from the previous pass. Thus compare images by both raw accumulationSPP and denoised snapshotSPP; otherwise the apparent quality timeline mixes two different times. This is expected preview architecture, except B2 final scheduling.

## B4 — Confirmed model provenance loss propagates known Fast override defect

The bridge selects color/HDR/non-sRGB/noisy-aux registry models (Bridge:535-543), resolves (stem,bytes), logs stem, then caches only bytes548-552. Builder receives bytes through weights597, losing selected model descriptor/stem. OIDN override topology inference is already audited in OI/bughunt/model_weights.md and pipeline.md: Small/XL widths are not derived correctly by the override route.

Consequently the UI Fast choice can take this failing override route; this is a mode-specific model loading/shape problem, not a proven cause of worsening noise in the default Base mode. High with this bridge's noisy-aux stems ordinarily falls back to Base because the shipped Large models serve clean auxiliaries, as documented in OI/bughunt/model_weights.md. Do not label every High run as Large or apply Large-overlap causality without observing the resolved stem.

Systematic proposal: resolver returns validated descriptor/provenance together with bytes; both cache and filter construction consume it through the existing resolution/commit path. No duplicate bridge-side model classifier.

## B5 — Exposure selection is explicit; autoexposure defects have conditional scope

Physical camera supplies Some(effective_exposure_multiplier) (SB/src/app/treemap_view.rs:1587-1591), while Manual supplies None. Bridge:515-518 parses OIDN_INPUT_SCALE and gives it priority over the caller. Nonfinite strings accepted by f32 parsing and nonpositive scales are not rejected in this bridge; shared OIDN scale validation should own the contract, with diagnostic error rather than silently invalid arithmetic.

Scale is applied inside OIDN and inverted on output; display exposure is separately applied by composite_overlay (SB/crates/render-3d/src/lib.rs:1210). That source structure alone does not establish accidental double exposure.

Bridge filter cache key includes scale bits562-569, yet scale also has a runtime setter619. A scale change rebuilds the network despite comments613-616 saying no rebuild. This is confirmed redundant invalidation/performance work; not evidence of image noise.

Autoexposure P5/P6 is reachable in Manual mode without env scale. It is bypassed by the normal Physical-camera override. A controlled comparison must record camera mode, actual scale and env, rather than globally blame autoexposure.

## B6 — Direct GPU copy order is supported by the checked pinned runtime; present race unproven

Production make_burn_device clones renderer Instance/Adapter/Device/Queue (Bridge:715-727). It does not create a second GPU context. Color/AOV external-copy commands submit at422. Input allocations remain owned by tensors subsequently consumed by OIDN (Bridge:809-826,485-493,627).

Pinned CubeCL:
- CU/crates/cubecl-runtime/src/client.rs:353-365 get_resource performs submit_blocking to the server.
- CU/crates/cubecl-wgpu/src/compute/server.rs:357-371 executes relevant streams and returns ManagedResource with a cloned allocation binding.
- CU/crates/cubecl-runtime/src/storage/base.rs:99-105 explicitly retains that binding to keep the underlying suballocation alive; cloning a raw wgpu buffer alone does not have the same pool-allocation guarantee.
- CU/crates/cubecl-runtime/src/stream/scheduler.rs:186-220 executes pending schedules; CU/crates/cubecl-wgpu/src/compute/schedule.rs:225-231 flushes the backend.
- CU/crates/cubecl-wgpu/src/compute/stream.rs:520-541 submits pending kernels to its shared queue.

Output-copy helper retains cube and ManagedResource while encoding and submitting the external copy (Bridge:979-1012). The copy is submitted after get_resource flushes prior kernels; same-queue order therefore supports compute-before-copy. The helper releases local ownership after submission, then denoise polls before the next bridge call (676).

Bridge comments658-675 assert a historical buffer reuse race causing growing speckle. Those comments are not a reproduction or proof of a current race. No confirmed current ordering defect was found in this reviewed path. Concurrent clients, asynchronous pool cleanup and GPU numerical differences still require runtime tests; do not introduce a global wait-based workaround on the basis of the comment alone.

poll result is discarded676, so a polling error can still lead to Ok690. Preserve device errors in a systematic shared error path; this is an observability/failure-reporting defect, not proof of silently corrupted pixels.

## B7 — Debug standalone-device branch is incompatible with direct shared-resource copies

OIDN_STANDALONE_DEVICE creates an independent Burn WGPU device (Bridge:707-711). Subsequent copy commands still use the renderer encoder/queue for tensor buffers (392-422 and953-1012). Buffers from the independent device cannot be used by the renderer device's copy encoder. Therefore this branch is not a valid end-to-end A/B isolation method for the zero-copy bridge. Use a supported host transfer bridge for that diagnostic, or test shared-device and independent-device numerical paths outside direct-copy integration. This unused/debug branch must not be mistaken for the normal production path.

## Existing OIDN findings: actual bridge scope

| OIDN finding | Bridge relevance |
|---|---|
| P1 normal tile padding | Reachable full-AOV default; source-derived boundary/tile candidate, not noise proof. |
| P2 albedo-only primary | Not reached: this bridge always includes color. |
| P3 normal-only signed output | Not reached: this bridge always includes color. |
| P4 directional lightmap | Not reached: bridge builds RT HDR, not RTLightmap. |
| P5/P6 autoexposure | Manual camera / absent explicit scale only. |
| P7 Large receptive field | Only if actual loaded model is Large and tiled; High noisy-aux fallback normally Base. |
| P8 input validation | Bridge constructs expected batch1/RGB tensors, but explicit scale still needs validation. |
| P9 mutable cached handles | Bridge uses immutable committed model with fresh tensor arguments627; no mutable input-handle cache here. |
| P10 sanitation | Reachable via runtime nan_protect; pre-OIDN clamp can already perform Inf*0, so test nonfinite path by stage. |
| P11 grayscale reduction | Not reached: bridge outputs RGB then RGBA, not scalar host images. |

Full paired Rust/reference evidence remains in OI/bughunt/pipeline.md and local_pipeline.md. No findings should be relabeled as confirmed causes of this renderer symptom.

## Verification entry points and useful commands

Checked public API:
- OidnDenoiser::new(ctx,width,height,weights_dir) and resize(ctx,width,height).
- set_mode218, set_quality222, set_input_clamp180, set_adaptive_clamp192, set_nan_protect186, set_external_input_scale201.
- denoise(&mut self,ctx,encoder,color_tex,albedo_buf,normal_buf,current_spp)249-257 consumes the encoder.
- result_view234; result_texture is private. A harness must composite/read a harness-owned render target or use another existing supported export path rather than assume public texture access.

Existing bridge unit test only checks mode/AOV requirements (Bridge:1034-1040). From SB workspace:

```powershell
cargo test -p pt-denoise-oidn --lib tests::mode_aov_requirements -- --exact
```

This test does not verify images, GPU resource lifetimes, adaptive clamp, model loading or final scheduling.

A meaningful parent-owned harness should cover:
1. Frozen HDR snapshot + valid per-pixel AOV sums/counts, repeated sameSPP/model/scale, compare image hashes and max/mean differences.
2. Same frozen snapshot at128 vs256SPP, adaptive on/off and clamp0; capture effective clamp trace already present atBridge475.
3. Actual progressive captures at128/256/512/... with raw and denoised images, highlight/detail/flat-region metrics separately.
4. Color / ColorAlbedo / fullAOV, Base / Fast / High, same dimensions/exposure; record resolved stem549 and model605.
5. Physical explicit scale vs Manual autoexposure, and aligned/nonaligned widths; valid edge/tile dimensions.
6. NaN/+Inf/-Inf inputs tested before and after clamp/sanitation; GPU error propagation, not just finite outputs.
7. Target300 interval128 confirms last preview256; final scheduling regression test for exact-multiple and nonmultiple targets after a fix.

OIDN_TRACE_TENSORS is parsed atBridge1020-1027 and provides intrusive full-tensor diagnostics; leave it disabled for timing/stability baselines. OIDN_INPUT_SCALE overrides camera-derived scale515-518. Do not use OIDN_STANDALONE_DEVICE for the direct-copy harness (B7).

## Proposed ordered actions

1. Reproduce frozen-input stability and adaptive-clamp differences first; retain raw snapshots and full settings.
2. Fix B2 scheduling using one completed-sample/accumulation identity state, with meaningful final/preview tests.
3. Integrate validated model descriptor/provenance globally (B4), and shared scale/error validation (B5/B6).
4. Apply approved OIDN normal-padding/transfer and autoexposure fixes with paired reference tests, according to actual reachable mode.
5. Keep physical exposure, AOV normalization, clamping and preview functionality; no evidence justifies feature deletion.
6. Update the parent plan with measured results before naming a root cause.
