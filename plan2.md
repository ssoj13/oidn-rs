# plan2.md — render noise investigation and systemic repair plan

Date: 2026-10-02. Baseline: `bebbfc5ec26c589121e2b0878ff55a6f669b43a6`. Status: **static audit complete; user approval pending; implementation and runtime validation not started**.

The audit confirms several violations of OIDN input, exposure, tiling, and output contracts. It does **not** establish the cause of the user's observed noise. Each candidate below has an explicit input/mode scope. Source-derived counterexamples are not executed reproductions. No Rust code was changed, no feature was deleted, and no builds/tests were run.

Historical `plan1.md` exists in git `7fc4eff` and was deleted in `b999415`; this is the next historical plan number, even though no plan file existed at audit start.

## Contents

- [Evidence and coverage](#evidence-and-coverage)
- [Noise candidates and history](#noise-candidates-and-history)
- [Confirmed findings](#confirmed-findings)
- [Architecture and single sources of truth](#architecture-and-single-sources-of-truth)
- [Implementation and validation checklist](#implementation-and-validation-checklist)
- [Retained functionality and unresolved decisions](#retained-functionality-and-unresolved-decisions)
- [Local v2.5.0 follow-up](#local-v250-follow-up--static-evidence-only)
- [Tool limitations and resumption](#tool-limitations-and-resumption)

## Evidence and coverage

Detailed, durable reports:

| Report | Source responsibility |
| --- | --- |
| [Pipeline](bughunt/pipeline.md) | RT mutable/immutable, lightmap, shared runner, tensor ops, color, exposure |
| [Model/weights](bughunt/model_weights.md) | Both U-Net topologies, all loaders, TZA parser/types/tests, archive inventory/history |
| [Geometry/API](bughunt/geometry_api.md) | Image/tensor boundaries, tile planner, registry/resolver, device/error/public API |
| [CLI/verification](bughunt/cli_verification.md) | CLI I/O/flags/comparison, example benchmarks, integration tests, CI/bootstrap/docs |
| [Astra causal review](bughunt/astra_review.md) | Independent focused review of progressive highlight noise, causal triage, comparison controls, and architecture contracts |

All 43 Rust source files were assigned across the audit team, with production code, in-file tests, integration tests, manifests, and relevant callsites inspected. The model/TZA report includes 17 owned files. The archive inventory contains 23 files; names, sizes, embedding coverage, and committed asset history were inspected. The later local-reference follow-up computed all 23 Rust SHA256 values; SHA matching against pinned upstream archive bytes remains **not performed** because the native weights submodule is uninitialized.

C++/Python reference citations refer to official [OIDN v2.4.1](https://github.com/RenderKit/oidn/tree/v2.4.1), preserved under `bughunt/reference-v2.4.1`, matching Rust `crates/oidn-rs/src/lib.rs:58`. The originally configured `C:/projects/projects.rust.cg.offload/oidn` checkout and `_ref` are absent. The user subsequently supplied an available local checkout at `D:/Projects/vfx.ref/oidn`, verified clean at tag v2.5.0, commit `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7`. Original citations below remain pinned to v2.4.1; local v2.5.0 follow-up evidence is labeled separately. The historical directory name `reference-v2.3.3` is not version evidence and is not this plan's citation root.

Rust paths in the tables are workspace-relative. Under `crates/oidn-rs/src/`, abbreviations `rt.rs`, `rtlightmap.rs`, and `unet_runner.rs` mean `filters/<file>`; other façade names are directly under `src/`. Model paths mean `crates/oidn-model/src/`, TZA paths mean `crates/oidn-tza/src/`, and CLI paths mean `crates/oidn-cli/src/`. Reference paths are relative to the pinned reference root. Detailed reports provide expanded context and authoritative upstream links.

Completed audit checklist:

- [x] Establish baseline and inspect read-only history without rollback.
- [x] Read every assigned source/test/configuration file and relevant callsites.
- [x] Compare runtime topology, activations, concat/skip order, parameters, transfer and padding contracts.
- [x] Compare exposure bin geometry and source sanitation.
- [x] Inspect all 23 archive names and embedding arms.
- [x] Inspect all public/CLI entry paths and state invalidation.
- [x] Search TODO/FIXME/HACK/todo!/unimplemented!/dead-code annotations; no explicit source marker found.
- [x] Inspect uncalled/exported surfaces before proposing removal; classify unfinished wiring rather than delete.
- [x] Consolidate independent RF, Small-loading, cache, and pipeline findings.
- [x] Write current ASCII codepaths in [AGENTS.md](AGENTS.md), actual/proposed Mermaid in [DIAGRAMS.md](DIAGRAMS.md).
- [ ] User approval for implementation and a separate verification pass.
- [ ] Reproduce the user's image regression.
- [ ] Certify pixel-level CPU/GPU/native parity.

## Noise candidates and history

| Candidate | Applies when | Predicted signature / historical window |
| --- | --- | --- |
| P1 normal padding | Normal input plus outer/alignment padding | Incorrect border context; `91a261e` replaced reflection with zero padding while retaining subsequent0.5 remap |
| G1/P7 Large RF | Actual Large model, multiple tiles | Seam-local context loss; overlap96 instead of112px. Color-only High falls back to Base in shipped inventory |
| P5 exposure bins | HDR color, no explicit scale, small/odd dimensions or edge-lit content | Global scale/denoising change, dropped edge contribution; `c357622` floor pooling and `b3c9c62` common runtime wiring |
| P6 signed RGB exposure | Negative RGB components before primary clamp | Exposure inconsistent with input actually seen by network |
| P2 albedo-only | Separate albedo denoise; default linear input or explicit scale | Incorrect primary transfer/scale; `912aecf` incorrectly made every no-color route Linear |
| P3 normal-only | Normal prefilter then reuse as clean auxiliary AOV | Unsigned output instead of signed normals |
| C2 file encoding | CLI encoded PNG/JPEG input/output | Encoding-domain mismatch; does not explain an exclusively tensor-native renderer path |
| P4 directional | Directional lightmap | Negative gradients destroyed; mode-specific |
| Backend migration hypothesis | Otherwise valid renderer inputs after Burn0.22/WGPU30 migration `bebbfc5` | Requires matched CPU/GPU/last-good numerical comparisons; unchanged Rust topology does not prove backend parity |

The user clarified the observed symptom: noise occurs in highlights and fine details and grows during one progressive rendering run as rendering continues. This is temporal growth within a run, not a claim about successive code versions. Whether the denoiser receives fresh raw accumulated radiance or a previously denoised result remains unanswered. The renderer bridge outside this workspace was not audited.

This localization and time dependence change the diagnostic order; they do not identify a cause. First capture same-run raw inputs/AOVs and denoised outputs at increasing sample counts, preserving highlight crops and frame/sample timestamps. Record per-pass flags/input scale, exposure result, dimensions, actual resolved model/source/hash, backend/adapter and dependency versions. The shared runner recalculates automatic scale per execution (`unet_runner.rs:89-100`) and creates a fresh output accumulator (`:137-138`); this repository does not establish temporal output feedback. Repeated denoising, renderer buffer ownership/order, and raw-versus-denoised reuse require bridge evidence. Boundary/RF candidates remain conditional on normal padding, actual Large model, and spatial correspondence to image edges or tile seams.

### Astra review and progressive diagnostic order

[Astra's consultation](bughunt/astra_review.md) supports the semantic findings and shared-runner architecture, while separating source defects from the confirmed same-run highlight/fine-detail symptom. This was an independent focused review, not another exhaustive audit. HDR mode and raw-versus-denoised feedback remain unconfirmed; display highlights alone do not establish HDR input. No runtime cause was demonstrated.

Proposed diagnostic order, all pending the separate verification approval:

1. Capture early/late raw color/AOVs, denoised and displayed outputs with a bright-detail ROI and dim control. Record samples/timestamps, mode, working space, sum-versus-mean convention, AOV sample counts, scale, model/source/hash, tile plan, device/dependencies and errors/fallbacks.
2. Identify where noise grows: raw accumulation, denoised output, or display composition. Establish actual producer/consumer paths and whether denoised output feeds the raw accumulator.
3. Replay frozen early/late inputs repeatedly outside the live renderer with one immutable filter, fixed positive finite scale/device/model, then a fresh filter. Stable versus changing replay results separate evolving inputs from state/integration/backend investigation; neither outcome alone identifies a culprit.
4. Compare live reused buffers with owned snapshots at an explicit GPU submission/completion boundary. A copying/synchronized diagnostic is an isolation experiment, not a permanent workaround.
5. Record exposure over sample count and compare fixed-scale/automatic-scale replay; inspect AOV evolution, ranges and NaN/+Inf/-Inf at raw, preprocessed, CNN, inverse-transfer and presented stages. Match ACEScg/Rec.709 policy. Changing clean_aux/color-only also changes model selection and is a combined-workflow experiment.
6. Compare CPU/WGPU/native with matched buffers, weights, scale and semantics; localize the first divergent stage and report highlight/detail errors in both network-output and radiance spaces.
7. Test padding/RF only with spatial correlation and real route metadata. Hold model/scale/alignment/context fixed for tiled/untiled comparison; a smaller crop with newly estimated exposure is not controlled.
8. Replay the last-good development revision and bisect only after a frozen snapshot distinguishes implementations. Historical onset and within-run growth are separate evidence axes.

The fresh TransferState/accumulator and feed-forward U-Net have no explicit temporal blend (`unet_runner.rs:89-100,137-138`, model `unet.rs:104-142`); they do not establish safe bridge lifetime, queue completion or deterministic device execution. Mutable tensor cache rebuilding is unwanted work, not proof of accumulating image error.

If HDR/PU is active, scalar constants and high-branch formulas agree (Rust `color.rs:126-136`, `gpu_ops.rs:179-196`; LOCAL2.5 `core/color.h:77-106`). The exponential inverse can amplify absolute highlight radiance error; this is mathematical sensitivity, not a discovered defect. The 65504 normalization anchor (Rust `color.rs:14-16,38-55`; LOCAL2.5 `core/color.cpp:10-15`) is not a hard input clamp. Measure both network and final-radiance error before assigning a cause.

**Additional comparison scope — native quality arithmetic policy:** LOCAL2.5 `core/unet_filter.cpp:262,280` selects fastMath when quality is not High; `core/graph.cpp:114,203` forwards it, and `devices/cuda/cuda_conv.cu:28,34-48` uses it for requested accumulator/kernel selection. The same policy exists in pinned v2.4.1. Rust quality currently selects archive/preset (`rt.rs:504-564`, `registry.rs:98-103`) without a convolution quality argument (model `unet.rs:23-27`); lightmap discards quality (`rtlightmap.rs:242`). Record native device/quality/precision in parity experiments, initially prefer explicit native High with model bytes fixed. This is a design/parity gap and experimental confounder, not evidence that CUDA policy explains WGPU noise. Do not blindly add a native fastMath switch or promise bit-exact cross-backend output.

## Confirmed findings

Priorities describe implementation/verification order, not proven contribution to the reported noise. Duplicate observations share one repair item.

| Priority / IDs | Confirmed defect and evidence | Proposed systemic solution |
| --- | --- | --- |
| P0 P1 | Raw normal padding0 becomes0.5 after remap: `unet_runner.rs:161-199`; REF `devices/gpu/gpu_input_process.h:69-79,90-118` leaves out-of-source channels zero | Preprocess the valid rectangle by role, then use one zero-padding path |
| P0 P2 | Albedo-only bypasses primary scale/transfer: `rt.rs:479-493`, runner188-190; REF `core/rt_filter.cpp:63-70`, devices/gpu/gpu_input_process.h:37-54,244-246 | Select primary role once; reuse existing primary preprocess for albedo-only |
| P0 P3/P4 | Signed normal/directional input-output is unwired: runner186,192-212; lightmap294-319; REF `core/unet_filter.cpp:551-564`, devices/gpu/gpu_input_process.h:45-50, devices/gpu/gpu_output_process.h:58-69 | Pass signed mode through existing shared runner and both bookends; repair signed non-HDR upper clamp at `gpu_ops.rs:91` |
| P0 P5/P6 | Exposure drops partial cells/small images and includes negative RGB: `autoexposure.rs:97-119,152-176`; REF core/autoexposure.h:26-31, devices/gpu/gpu_autoexposure.h:29-32,40-42,59-62 | Share exact balanced bin boundaries and componentwise sanitation between CPU oracle and device implementation |
| P0 G1/P7 | RT always plans BASE RF: `rt.rs:578-584`, tile9-13,88; REF core/unet_filter.cpp:263-268, unet_filter.h35-37 | Actual parsed model descriptor supplies RF; Large overlap112, Base/Small96 |
| P0 P11 | Single-channel filter output writes red instead of reference mean after inverse transfer: runner326, `image.rs:285-286`; REF devices/gpu/gpu_output_process.h:54-56 | Pass destination semantics into existing output processing; retain generic image accessor behavior |
| P1 MW1/C1 | CLI Fast drops selected stem and constructs Small as Base: CLI main267-283,342-344; model variants27-33,57-81; rt550-558; REF training/model.py:61-84, core/unet_filter.cpp:254-280 | Ordinary CLI uses filter resolver; custom archives infer topology and all widths from validated schema |
| P1 MW2 | XL has constructor/widths but override path constructs Large BASE widths: model unet_large59-69,136-137; rt560-562; REF training/model.py:167-176 | Preserve XL and infer executable widths; no filename-only variant guessing |
| P1 G3 | Public resolver, RT and lightmap have divergent source/error policies: weights136-152, rt519-539, lightmap246-257; REF core/unet_filter.cpp:438-459 | Extend existing resolver with explicit policy and Result/provenance; skip only NotFound |
| P1 P8/G4 | Mutable/immutable/custom validation differs; release tensor contracts unchecked: rt421-448,504-516,601-605,710-716,738-742,775-785; tensor helpers21,40,81; REF core/unet_filter.cpp:254-260,420-443 | One mode/layout/geometry validator before load and at execution; reject invalid batch/channels/shapes |
| P1 P9/G7 | Host geometry change does not invalidate; consumed tensor handles force same-shape reload: rt299-355,839-841; lightmap182-187; runner292-300; REF core/unet_filter.cpp:170-251 | Separate committed input signature from per-pass handles; reuse immutable artifacts through both façades |
| P1 G2 | Public descriptors/debug-only length checks permit overlap/panic/overflow: `crates/oidn-rs/src/image.rs:60-75,84,100,113,170-181,275-307`; REF core/image.cpp:14-34,54-65 | Existing image layer validates checked occupied extent/stride/narrowing; fallible unaligned byte handling |
| P1 MW3/MW4 | Tensor dimensions/offset narrowing unchecked; public decode can panic: TZA types38-43,49-76, parser120,159,166-172; model loader50-97; REF core/tza.cpp:18-23,66-70,93-98 | One validated descriptor and fallible little-endian byte decode, reused by both loader ranks |
| P1 C2 | CLI RGB conversion does not apply file color transfer: main263-265,333-338; io86-89,106-115; REF apps/oidnDenoise269-283, apps/utils/image_io411-439 | Add semantic/encoding arguments to existing load/save; reuse scalar sRGB functions once at boundaries |
| P1 C3 | PFM/PHM scale magnitude ignored, CRLF payload corrupted; unchecked header arithmetic: io170-184,217-231,275-302; REF apps/utils/image_io.cpp:87-104,187-210 | One validated header/payload path parameterized by sample width; preserve endian and row flip |
| P1 C4/C7/MW6 | MSE can accept NaN; tests admit wrong/zero output and miss seams/exposure: main438-454; tests multi_tile_wgpu67-76,112-148, e2e_ndarray64-73; model tests cited in MW6; REF apps/oidnDenoise435-445 has different documented metric | Preserve MSE contract, reject invalid samples/thresholds, use f64 arithmetic; golden corpus and precise spatial/stage metrics |
| P2 P10 | Sanitation flag gates auxiliary/exposure pass but color/output sanitize unconditionally: runner60-71; gpu_ops51,79 | Explicit sanitation policy at shared input boundary; reference-required output sanitation separately |
| P2 G5 | maxmem heuristic is best effort and excludes actual live allocations: rt566-576, tile97-99,130-132; REF core/unet_filter.cpp:300-305,321-324 also has minimal fallback | Report achieved plan/estimate and lower bound; backend-informed planning without silently changing quality |
| P2 C5 | RT directional accepted but ignored; lightmap maxmem omitted and mode flags misleading: main124-125,141-143,246-250,339-341,393-399; REF apps/oidnDenoise.cpp:338-361 | Family-specific validation/forwarding; shared save/compare path |
| P2 C6 | Zero iterations/panics/duplicated metrics and false-success summaries: main552-553; example135-168,409-465,519-528 | Positive inputs, structured errors, shared fixtures/metrics/timestamp, failure if no successful row |
| P2 C7 | Default device falsely called CPU; GPU tests in claimed no-GPU lane: all_models_smoke2-4,38; CI47-51, e2e_wgpu73-78 | Explicit `Device::ndarray()`; dedicated GPU lanes and mandatory asset checks |
| P2 C8 | Bootstrap installs wrong bin from virtual root; profile/verbose ineffective: bootstrap119-121,224-226,260-261; CLI Cargo15-17 | Preserve target manifest directory and actual bin name, correct profile, propagate metadata/verbosity failures |
| P3 MW5/Pipeline cleanup | Duplicate dtype conversion/conv/upsample, OwnedImage, stats, fixtures, forward calculation: model loader64-67,93-96, unet23-27,147-153, unet_large26-30,191-197; rt225-281/lightmap115-171; runner341-459; color48-49 | Reuse existing shared helpers/types; preserve topology-specific ordering and CPU oracle |
| P3 G6/C9 | Stale Backend/24-model/CPU-roundtrip/exact-parity claims and unused dependencies: tensor3-7,69-73; filter12-14; device30-33; Cargo54-55; README7-12,44,95-100 | Current AGENTS/DIAGRAMS updated now; approved pass updates public docs and reviews dependency intent |

Public tensor names with duplicate entries currently replace earlier values in Rust (`parser.rs:182`) whereas native emplace keeps first (`core/tza.cpp:99`). Reject ambiguous duplicate names during the same archive-validation repair. Zero/overflow dimensions and 32-bit offset conversion need explicit errors.

## Architecture and single sources of truth

The common numerical runner already exists; reuse it. A validated execution context within existing modules can carry primary/auxiliary roles, transfer/signed mode, sanitation policy, dimensions and destination semantics; avoid adding ambiguous adjacent booleans to the already long runner interface. Burn0.22 dynamic `Device` dispatch, host adapters, tensor-native RT, immutable committed execution, lightmap, all image formats, cancellation/progress, explicit scales, diagnostic features, and ACEScg exposure remain supported.

Current codepath is represented in [AGENTS.md](AGENTS.md) and [DIAGRAMS.md](DIAGRAMS.md). Proposed contract:

```text
Host Image / Tensor + mode flags
       |
one validated role/encoding/shape/geometry contract
       |
registry candidate route -> existing resolver(source policy)
       -> resolved bytes + stem + source provenance
       -> validated TZA schema
       -> model descriptor (topology, widths, channels, RF)
                  |                      |
         construct/load Net       plan(actual RF, budget)
                  +-----------+----------+
                              |
                  immutable committed artifacts
                              |
per-pass handles -> exact exposure -> valid source preprocessing
                              |
                   zero-pad -> common forward
                              |
inverse transfer -> destination reduction -> signed decode/clamp/scale
                              |
                      crop -> stitch
                              |
                Tensor result / host Image write
```

Archive tensor shapes determine executable dimensions, not the model's training semantics: same-shaped weights cannot establish HDR/LDR, albedo/normal or clean/noisy auxiliary identity. Preserve selected model identity, provenance and explicit custom-weight semantics. Derive RF only after validating the known topology's kernel/stride/pooling assumptions.

Do not add independent mode decisions in RT, lightmap, CLI, and tile code. Extend existing functions with explicit semantic arguments when needed. Share ownership/layout with the existing image module. Scalar and device implementations can remain distinct where execution representation requires it, while sharing constants, geometry, and an exact documented mathematical contract.

Verified exclusions: successful loads replace every convolution parameter; no retained random parameter was found. Base and Large runtime op order, pooling skips, nearest upsampling and final ReLU agree with native v2.4.1 (`unet.rs:104-142`, `unet_large.rs:147-187`; REF core/unet_filter.cpp:468-530). Do not remove final ReLU merely because training Python differs. Legacy example same-shape output allocation does not by itself reload the model (`rt.rs:373-386,792-793`); the tensor-handle cache defect is a separate path.

## Implementation and validation checklist

All unchecked actions below are **proposed, pending approval**. Approval does not turn a source-derived counterexample into measured evidence.

### 1. Freeze reproducible evidence

- [ ] Record actual renderer scene/AOVs, outputs, settings, dimensions, adapter, dependencies and last-good commit.
- [ ] Capture early/middle/late raw accumulated color, matching albedo/normal and denoised output from the same progressive run, with sample count/timestamps and highlight/fine-detail crops.
- [x] Trace Squarebob renderer-to-filter input/output ownership and ordering: the inspected path receives raw accumulated radiance and writes a separate denoised texture; no feedback was found. See [Squarebob bridge audit](bughunt/squarebob_bridge.md), B3/B6 and its exact source anchors. Other integrations remain unverified.
- [ ] Record automatic exposure per pass; propose paired fixed-scale and automatic-scale comparisons with identical raw buffers to isolate changing exposure from changing Monte Carlo input.
- [ ] Repeat one frozen input/AOV set with fixed mode/model/scale/backend and compare outputs; a changing result would require investigation of integration/backend state, while a stable result redirects investigation to changing inputs or settings. These runtime comparisons are not executed in the audit.
- [ ] Record resolved model/stem/source plus archive hashes; compare all 23 shipped assets against pinned upstream revision.
- [ ] Freeze asymmetric native v2.4.1/CPU/WGPU baseline corpus before repairs; separate raw-buffer numerics from CLI encoding.
- [ ] Compare primary packing/exposure/intermediate-layer tensors to locate first divergence; record native quality/device/arithmetic policy and measure network-space plus highlight radiance-space error.

### 2. Repair shared numerical semantics

- [ ] Use one primary-role and signed-mode contract across all RT/lightmap entry paths.
- [ ] Preprocess valid source before zero-padding; keep padded channels exactly zero.
- [ ] Correct albedo-only transfer and scale, normal-only signed output, directional input/output, signed upper clamp.
- [ ] Implement exact balanced exposure bins and matching nonnegative source sanitation; preserve ACEScg.
- [ ] Carry destination1-channel reduction into postprocessing after inverse transfer without changing generic accessor semantics.
- [ ] Define sanitation flag semantics and remove redundant work.

### 3. Unify model resolution, validation and cache

- [ ] Infer topology/widths/channels/RF from validated archives; preserve Small/Base/Large/XL.
- [ ] Route RT/lightmap/CLI through existing canonical resolver with explicit precedence/provenance.
- [ ] Validate modes before overrides and all tensor/image boundaries before expensive load.
- [ ] Make archive size/offset/payload/duplicate-name validation and decoding fallible and reusable.
- [ ] Separate committed signature from mutable frame handles; correct host geometry invalidation and tensor model reuse.
- [ ] Describe best-effort maxmem and report achievable tile/byte planning.

### 4. Repair file/CLI/tooling contracts and deduplicate

- [ ] Explicit color/albedo/normal file semantics; encode/decode exactly once.
- [ ] Shared validated PFM/PHM header/payload/scale path.
- [ ] Finite f64 MSE/PSNR metrics, threshold validation and reference requirement.
- [ ] Family-specific CLI controls; preserve lightmap behavior and supported formats.
- [ ] Positive benchmark inputs and meaningful failure status.
- [ ] Correct bootstrap target/profile/verbosity handling.
- [ ] Reuse ownership, dtype decode, conv/upsample, stats, fixtures and timestamp helpers.
- [ ] Update README/API/rustdoc/CI claims and review tracing/memmap2/model half dependency intent.

### 5. Run a separate meaningful verification pass

- [ ] Release invalid-shape/mode/descriptor tests; zero/overflow/unaligned/short/padded buffers; malformed archives and duplicate names.
- [ ] Every shipped archive schema/load and Small/XL custom routes; resolver precedence/NotFound vs permission failure.
- [ ] 8x8 uniformHDR,17x16 edge-lit, odd balanced bins, negative/NaN/+Inf/-Inf exposure and preprocessing; distinguish Rust all-nonfinite replacement from native NaN-only replacement followed by clamp; explicit-scale comparison.
- [ ] Albedo-only srgb/scale, normal-only signed range, signed lightmap, zero padded normal tensor inspection.
- [ ] Single-channel denoise compares mean of inverse-transferred RGB, distinct from red accessor behavior.
- [ ] Exact per-pixel tile coverage; horizontal/vertical seams and corners; true Large tiled/untiled comparisons.
- [ ] Host/tensor, CPU/WGPU, native OIDN comparison with mode-specific numerical tolerances and asymmetric inputs.
- [ ] Repeated same-layout mutable frames load one model; real input signature changes rebuild or reject coherently.
- [ ] CLI encoded-image roundtrip, PFM/PHM scale/endian/CRLF/binary whitespace, NaN metrics, zero-iteration failures.
- [ ] Explicit CPU tests; dedicated GPU lanes; required assets cannot silently skip quality verification.
- [ ] Run appropriate format/lint/build/tests only after approval; record exact commands and results.
- [ ] If frozen evidence implicates a historical window, bisect isolated checkouts without rollback of shared work.

## Retained functionality and unresolved decisions

No exported function/error variant/constructor is dead merely because no internal callsite exists. Preserve CPU exposure as oracle/public functionality, tensor layout helpers, signed gpu_ops branches as unfinished wiring, XL constructors, all host adapters, progress/cancellation, and current format support.

Resolver precedence must become explicit: proposed ordinary policy is disk-first for a supplied directory, then enabled embedded data on NotFound only; custom bytes remain explicit override. Final API policy should be documented and reviewed before implementation. Preserve documented Rust MSE rather than silently replace it with native's different comparator. Define HDR PSNR peak explicitly. Decide sanitation flag contract and device-initialization error guarantees based on actual backend behavior, without claiming deferred initialization has already succeeded.

Guideline checks from [Rust API guidelines](https://github.com/rust-lang/api-guidelines): C-VALIDATE, C-GOOD-ERR, C-FAILURE, C-STRUCT-PRIVATE, and C-CALLER-CONTROL apply to descriptor validation, fallible APIs, and explicit policy. Compatibility is not required, but feature preservation is.

## Local v2.5.0 follow-up — static evidence only

The user-provided reference `D:/Projects/vfx.ref/oidn` is a clean v2.5.0 checkout at `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7`. Detailed follow-ups: [local pipeline](bughunt/local_pipeline.md) and [local model/weights](bughunt/local_model_weights.md). **LOCAL2.5** anchors below resolve relative to that checkout; original table/reference citations remain pinned to v2.4.1.

- [x] Verify local tag, commit, clean status, and supplied path availability.
- [x] Recheck P1–P11 against local input/output/exposure/RT/lightmap source.
- [x] Recheck Base/Small/Large/XL topology, widths, selection, TZA and fusion changes.
- [x] Inspect the native weight submodule availability; read all 23 Rust binary headers and compute all 23 SHA256 values.
- [ ] Compare those hashes with actual pinned native archive bytes.
- [ ] Reproduce render noise and execute native/CPU/WGPU comparisons after approval.

All P1–P11 remain applicable. Exact paired Rust/LOCAL2.5 anchors are in the local reports. Representative confirmations: normal padding `unet_runner.rs:161-199` vs LOCAL2.5 `devices/gpu/gpu_input_process.h:90-118`; exposure `autoexposure.rs:152-176` vs LOCAL2.5 `core/autoexposure.h:29-31` and `devices/gpu/gpu_autoexposure.h:29-32,59-62`; Large RF `rt.rs:578-583` vs LOCAL2.5 `core/unet_filter.cpp:263-268`. Scalar output averaging remains required after inverse transfer and before signed decode/clamp/scale (LOCAL2.5 `devices/gpu/gpu_output_process.h:51-69` and `devices/cpu/cpu_output_process.isph:48-66`), rather than changing generic image accessors.

**P10 extension:** Rust `gpu_ops.rs:29-32,51-55` and `unet_runner.rs:60-64` replace all nonfinite samples; LOCAL2.5 `core/math.h:77-79` and `devices/cpu/math.isph:102-106` replace only NaN. Native primary order is scale -> sanitize -> clamp (LOCAL2.5 `devices/gpu/gpu_input_process.h:41-54`, `devices/cpu/cpu_input_process.isph:35-48`), while Rust sanitizes before scale. Positive infinity can saturate to1/FLT_MAX, and signed auxiliary normal infinities saturate to endpoints (LOCAL2.5 input_process `devices/gpu/gpu_input_process.h:45,64,74-77`); default Rust turns them into0, then normal remap yields0.5. This extends the sanitation repair and NaN/+Inf/-Inf/overflow fixtures; a deliberate stricter policy must be named, not claimed as exact native parity. No evidence establishes these values in the user's renderer.

Local training topology/widths, model selection and TZA format are unchanged from v2.4.1. Native v2.5 moved upsampling fusion from the producer post-op to decoder input (LOCAL2.5 `core/unet_filter.cpp:481-492`, `core/graph.cpp:151-155,166-170,312-317`), preserving logical nearest upsample -> concat -> convolution. Scalar transfer contracts also remain unchanged despite SIMD/template additions. This does not prove backend numerical identity and does not implicate fusion as the noise cause.

**Asset evidence and limit:** all 23 shipped Rust archives have real binary TZA magic/version headers, not LFS pointer text; sizes and SHA256 baseline are tabulated in [local model/weights](bughunt/local_model_weights.md). Native `weights/` is empty; LOCAL2.5 `.gitmodules:1-3` defines the external archive repository and `git submodule status weights` reports uninitialized gitlink `28883d1769d5930e13cf7f1676dd852bd81ed9e7`. That gitlink is unchanged between v2.4.1 and v2.5.0, but no native archive bytes were available. Header/hash checks do not establish full parser validity, upstream byte equality, or image quality. No submodule initialization/download, source edit, build, or test was performed. The native image corpus may use supplied v2.5.0 as an additional labeled reference; retain the pinned v2.4.1 baseline if comparing version changes.

This is a follow-up to the same plan2 audit. Priorities and approval boundary remain unchanged; the root cause of observed noise remains unproven.

## Tool limitations and resumption

Filesystem MCP performed file operations and read-only history; `mem_` and `seq_think` captured audit decisions/relations. GitHub MCP supplied official sources/API-guideline evidence and Context7 corroborated image layout vs color-transfer semantics. GitNexus and fetch MCP were unavailable; freshness/impact/reindex could not execute. No Rust symbol was edited and no graph coverage is claimed.

Missing files at the original configured reference path and deleted historical reports are input availability issues; the newly supplied `D:/Projects/vfx.ref/oidn` checkout is available. This follow-up remains part of plan2, not a new implementation plan. Reproducible filesystem HTTP/download failures exposing internal stack traces were logged, preserving prior entries, in external `filesystem-mcp-rs/BUG_CDX.md`. Missing docs-write style guide/project formatter was logged in external `oh-my-harness/BUG3.md`; inline skill guidance and manual full reread/whitespace checks are the fallback. ReadLints is not exposed, so no markdown-lint-clean claim is made.

Resumption order: read this plan, then the detailed report for each unchecked repair, then verify current source/history before editing. Recheck working-tree changes from other contributors. Update the corresponding checkbox immediately after evidence exists; never mark runtime parity or root cause proven from static inspection. Stop here for user approval of production fixes. The later renderer testing and SSH dependency update request separately authorizes the verification follow-up below.

## Squarebob integration follow-up — 2026-10-02

The user explicitly authorized testing `C:/projects/projects.rust.cg/cglibs/squarebob-rs` and updating all of its SSH GitHub refs. This authorization supersedes the earlier audit-only restriction for these verification/dependency tasks, but does not authorize the proposed production denoiser changes. Detailed checked source: [Squarebob bridge audit](bughunt/squarebob_bridge.md). Renderer integration findings and dependency updates are tracked in Squarebob's next numbered plan, `plan16.md`.

- [x] Read the complete OIDN bridge, progressive scheduling, display composition and relevant pinned CubeCL ownership/submission code.
- [x] Refresh Squarebob's stale GitNexus graph before the agent's codepath queries; no production symbol edited.
- [x] Update all 11 unique SSH GitHub dependencies; six resolved commits advanced and five were already current. Preserve explicit fscan revision policy. HTTPS pins remain unchanged.
- [x] Resolve the Windows allocator interface mismatch through Cargo's own lock resolution, without registry edits or source patches; `cargo test -p pt-denoise-oidn --lib --locked --quiet` passed (1 test).
- [x] Add a standalone diagnostic example using the actual shared-device bridge and frozen HDR/AOV buffers; production behavior unchanged.
- [x] Compile and execute the production-bridge diagnostic on RTX3080Ti/Vulkan. All outputs finite; fixed-clamp SPP1 vs256 and32 additional frozen-input repeats were bit-identical. Adaptive-clamp ROI standard deviation changed from0.003034 at128SPP to0.065254 at256SPP; maximal RGB change between adaptive1 and256 was8.448264122. Logs: `bughunt/squarebob_probe_series.stdout.log` and `.stderr.log`. These are synthetic-input measurements, not the user's scene reproduction.
- [x] Verify the updated dependency workspace: `cargo check --workspace --locked --quiet` and `cargo build --locked --bin squarebob --quiet` both passed (exit0, empty stdout/stderr). Locked metadata resolution and the bridge GPU diagnostic also passed. See `bughunt/squarebob_workspace.stdout.log`/`.stderr.log` and `bughunt/squarebob_build.stdout.log`/`.stderr.log`.
- [ ] Capture the user's actual scene and prove its first divergent stage.

**B1, checked mechanism:** Squarebob's default adaptive pre-OIDN luminance clamp is sample dependent: effective limit 6 at128SPP and10 at256SPP for the default user limit10 (`squarebob-rs/crates/pt-denoise-oidn/src/lib.rs:457-470`, defaults `crates/render-shared/src/lib.rs:1128-1135`). Identical raw/AOV inputs can therefore reach the network differently as SPP advances. This is a targeted highlight/detail candidate, not a demonstrated cause of the user's render; its threshold stops changing at256SPP.

**B2, confirmed scheduling defect:** Any successful periodic preview sets `oidn_denoised_this_accumulation=true` (`squarebob-rs/src/app/treemap_view.rs:1609`), blocking the final-only trigger (`1510-1513`). A nonmultiple target such as300 with interval128 can retain the256SPP preview instead of denoising the completed300SPP input. Proposed repair uses one successful-denoise sample count plus accumulation identity, preserving manual/periodic/final behavior; awaiting approval.

**Evidence limits:** Physical camera provides an explicit scale and normally bypasses the audited autoexposure defects. The bridge uses immutable committed filters with fresh inputs, so the mutable-host handle cache issue does not explain this path. Current shared-queue order and pinned allocation ownership support the reviewed direct-copy path; historical race comments are not a current reproduction. Fast override model provenance, scale validation, ignored polling errors and standalone-device debug incompatibility remain separate checked findings. Do not relabel them or the OIDN parity defects as the user's root cause without measured scene evidence.
