# plan4 — Systematic OIDN contract repairs

Updated: 2026-10-02. Status: **33 confirmed findings repaired and verified; publication approved and in progress**. Continues [plan2](plan2.md) and [plan3](plan3.md). Historical source findings and test results are preserved.

## Authorization and evidence boundary

The user instructed “тогда надо всё системно исправлять”. This explicitly authorizes the proposed systematic OIDN source repairs and the verification needed to complete them. Earlier report-approval and audit-only restrictions describe prior phases and no longer block this implementation. No further approval is needed for the repairs already authorized. After reviewing the completed repairs and verification, the user explicitly instructed “пушни всё в main”, authorizing publication of the OIDN repairs and the Squarebob consumer update. Publication is in progress; confirmed commit and remote receipts will be recorded when available.

Reference: **LOCAL2.5**, `D:/Projects/vfx.ref/oidn`, v2.5.0 `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7`. Original v2.4.1 report citations remain original. A confirmed contract defect is not proof of the user's progressive highlight-noise cause.

## Contents

- [Work sequence](#work-sequence)
- [Complete issue ledger](#complete-issue-ledger)
- [Current implementation and migration map](#current-implementation-and-migration-map)
- [Retained functionality](#retained-functionality)
- [Shared repair contract](#shared-repair-contract)
- [Verification receipts](#verification-receipts)
- [Squarebob consumer verification](#squarebob-consumer-verification)
- [Resumption](#resumption)

## Work sequence

- [x] Read the complete preceding plans and issue reports; preserve historical evidence.
- [x] Enumerate every original finding below, including duplicate IDs.
- [x] Complete model/archive, geometry, numerical runner/facade, resolver and CLI repairs.
- [x] Verify runtime boundary contracts and supported model/mode routes in CPU/public/native fixtures; the final supported feature matrix is recorded below. Release-mode execution remains outside this pass.
- [x] Run CPU/WGPU/native numerical comparisons against frozen matched inputs (108/108 finite; receipt below).
- [x] Check host/tensor equivalence, tile seams, repeated immutable/mutable executions (public 6/6 and bounded Large seam receipt).
- [x] Run workspace formatting/build/test/feature checks appropriate to final edits.
- [x] Reconcile current README, ASCII/Mermaid architecture and source anchors; reread complete documents and pass bounded Markdown lint.
- [x] Review all source diffs and record exact commands/results before delivery.

## Complete issue ledger

All 33 confirmed findings have implementation and verification evidence. Duplicate IDs share a systemic repair, not separate special-case code. Closure is bounded by the tested modes, fixtures and backends; it does not establish the cause of the user's scene noise.

| Done | ID | Required repair | Owner | Closure evidence |
| --- | --- | --- | --- | --- |
| [x] | P1 | Valid-source preprocessing before zero padding | runner | `unet_runner.rs:207`; unaligned full-AOV native error4.665127→0.000125885 |
| [x] | P2 | Albedo-only primary scale and transfer | runner | `pipeline_contracts.rs:307` public SRGB/scale oracle passed |
| [x] | P3 | Normal-only signed input/output | runner | `pipeline_contracts.rs:131` host/tensor signed test passed |
| [x] | P4 | Directional lightmap signed bookends | runner | `pipeline_contracts.rs:210` negative/scalar directional test passed |
| [x] | P5 | Exact balanced exposure bins including tiny/odd images | exposure | `autoexposure.rs:78,137`; odd33x19 and tiny8x8 matched-native receipts below |
| [x] | P6 | Nonnegative RGB exposure sanitation | exposure | final feature lib22/22: balanced-bin and negative/NaN component CPU/tensor oracle passed |
| [x] | P7 | Descriptor-derived Large RF | model/runner | descriptor RF202; native/Rust2tiles; tiled/full max0 at769x16 |
| [x] | P8 | Common mode and tensor boundary validation | runner | public invalid mode/shape/scale/custom-weight contract test passed |
| [x] | P9 | Independent committed signatures and frame handles | runner | core cache parameter-ID/layout invalidation test plus public fresh A/B/A test passed |
| [x] | P10 | Explicit sanitation policy and ordering | runner | core signed NaN/±Inf after-scale clamp oracle passed; explicit strict policy documented |
| [x] | P11 | Scalar destination reduction after inverse transfer | runner | `pipeline_contracts.rs:173,210` RGB mean/signed public tests passed |
| [x] | MW1 | Small schema-derived executable widths | model | descriptor all23 archives plus Small model tests; model15/15 |
| [x] | MW2 | XL schema-derived widths and topology | model | `tests/descriptor.rs:188`; XL/custom topology construction/load passed |
| [x] | MW3 | Checked TZA dimensions/offsets/payload/duplicates | model | `oidn-tza/tests/validation.rs:24,47,59`; TZA6/6 |
| [x] | MW4 | Fallible public tensor decoding | model | `oidn-tza/tests/validation.rs:83`; unaligned/short/iterator/endian tests passed |
| [x] | MW5 | Shared model primitives and dtype conversion | model | descriptor/loader shared path; existing Base/Small/Large/XL tests passed |
| [x] | MW6 | Meaningful model fidelity tests | verification | descriptor asymmetric custom-width numerical oracle; matched-native matrix |
| [x] | G1 | Large overlap (same repair as P7) | model/runner | Large6/6 finite; bounded2tile seam fixture max0 |
| [x] | G2 | Validated image descriptors and occupied extents | geometry | geometry 11/11 and formats 5/5 passed; overflow/extent/stride contracts |
| [x] | G3 | Canonical resolver with source/error policy | root | source-policy/conflicting embedded-vs-disk feature tests passed; skip NotFound only |
| [x] | G4 | Fallible tile/layout helpers | geometry | latest geometry11 + color/tile6 passed in final feature matrix, including checked pixel-count malicious-plan regression |
| [x] | G5 | Best-effort max-memory disclosure/achieved plan | root | `filters/mod.rs:33-82`; Large maxmem0 retained native minimum and reported achieved 2tile plan |
| [x] | G6 | Current public docs and dependency intent | docs/root | README/architecture reread and lint passed; unused façade tracing/memmap2 removed, CLI tracing retained |
| [x] | G7 | Model cache identity (same repair as P9) | runner | persistent signatures and actual parameter-ID cache regression passed |
| [x] | C1 | Ordinary CLI resolver rather than custom bytes | CLI/root | process test and actual3x2 CPU Fast canonical Small run passed |
| [x] | C2 | Role-specific file transfer at boundaries | CLI | `oidn-cli/src/io.rs:422` PNG role-aware decoding and HDR range test passed |
| [x] | C3 | Shared validated PFM/PHM reader | CLI | `oidn-cli/src/io.rs:359,396`;6CLI units passed |
| [x] | C4 | Finite f64 MSE and threshold validation | CLI | `support.rs:43,118`; invalid sample/range and process rejection tests passed |
| [x] | C5 | Family-specific CLI flags and maxmem | CLI | `main.rs:266`;3process tests and native full/tiled CLI runs passed |
| [x] | C6 | Positive benchmarks and honest failures | CLI | benchmark3/3 passed; zero/nonfinite input rejected; shared finite metrics and failure reporting |
| [x] | C7 | CPU/GPU lane and fidelity coverage | CLI/root | final96 CPU/feature tests plus explicit14/14 GPU; actual3072 multi-tile and Large full/tiled oracle |
| [x] | C8 | Correct bootstrap bin/profile/metadata failures | CLI | bootstrap4/4 tests passed; target/package/profile/metadata cases |
| [x] | C9 | README/config claim reconciliation | docs/CLI | 23 archives, dynamic Device, explicit source policy and bounded backend/performance claims verified |

Original source/line evidence: [pipeline P1–P11](bughunt/pipeline.md), [model MW1–MW6](bughunt/model_weights.md), [geometry G1–G7](bughunt/geometry_api.md), [CLI C1–C9](bughunt/cli_verification.md), and paired [LOCAL2.5 pipeline](bughunt/local_pipeline.md)/[model](bughunt/local_model_weights.md). Those line numbers are historical; current implementation anchors are listed below.

## Current implementation and migration map

The current source implements one model-construction path for RT and lightmap, one tensor runner for host/mutable/immutable fronts, one weight resolver, one validated host image owner, one tensor payload decoder, and shared model primitives. [AGENTS](AGENTS.md#current-repaired-source-map--2026-10-02) contains the ASCII codepath; [DIAGRAMS](DIAGRAMS.md#current-systematic-repair-flow--verified-scope) contains Mermaid.

| Issue group | Current implementation anchor | Contract and retained behavior |
| --- | --- | --- |
| P1–P4 | `crates/oidn-rs/src/filters/unet_runner.rs:207`, `filters/rt.rs:482`, `gpu_ops.rs:42` | First present role is primary; transform valid source before zero padding; albedo uses primary scale/transfer; normal-only/directional uses signed bookends |
| P5/P6 | `crates/oidn-rs/src/autoexposure.rs:78,85,137` | Balanced floor-boundary bins cover all pixels, including tiny/odd images; component sanitation precedes luminance |
| P7/G1/G5 | `crates/oidn-model/src/descriptor.rs:98,119`, `crates/oidn-rs/src/filters/mod.rs:16` | Actual topology supplies RF; logical memory planning reports infeasible native minimum-tile fallback |
| P8 | `crates/oidn-rs/src/filters/unet_runner.rs:72,88`, `filters/mod.rs:89` | Shape/device/model/output/scale checks before numerical operations; validation precedes weight I/O/model allocation |
| P9/G7 | `crates/oidn-rs/src/filters/rt.rs:208,730` | Persistent signatures survive consumption of fresh frame handles; geometry/role change invalidates artifacts |
| P10 | `crates/oidn-rs/src/filters/unet_runner.rs:98`, `gpu_ops.rs:42` | Optional strict nonfinite replacement before exposure; mandatory native NaN-only sanitation after scale |
| P11 | `crates/oidn-rs/src/gpu_ops.rs:68` | Scalar RGB mean after inverse transfer, before signed decoding/clamp/output scale |
| MW1–MW6 | `crates/oidn-model/src/descriptor.rs:30,141`, `crates/oidn-tza/src/parser.rs:109`, `types.rs:114` | Schema-derived widths, mutation-safe loading, checked aliased ranges, one archive backing allocation, fallible LE decoding |
| G2/G4 | `crates/oidn-rs/src/image.rs:89,230,338`, `image_tensor.rs:12,44`, `tile.rs:85,230,248` | Checked occupied extents/strides/layouts, fallible helpers, exact tile partition validation |
| G3 | `crates/oidn-rs/src/weights.rs:116,133,142` | Explicit source precedence before quality fallback; preserved provenance; only NotFound permits fallback |
| G6/C9 | [README](README.md#library-use), [current architecture](AGENTS.md#current-repaired-source-map--2026-10-02) | Dynamic device dispatch, deferred WGPU handle, explicit CPU, 23 archives and honest verification scope |
| C1/C5/C7 | `crates/oidn-cli/src/main.rs:27,33,266,313` | Canonical filter route; explicit CPU/WGPU selection; family flag validation; dedicated GPU lane |
| C2/C3 | `crates/oidn-cli/src/io.rs:35,58,359,396`, `main.rs:303,320` | Color/albedo transfer by role; normals/directional remain numeric data; shared checked PFM/PHM parser |
| C4/C6 | `crates/oidn-cli/src/support.rs:43,118`, `crates/oidn-rs/examples/bench.rs:115` | Finite f64 metrics, positive inputs/iterations, shared fixtures and honest benchmark failure reporting |
| C8 | `bootstrap.py`, `crates/oidn-cli/tests/test_bootstrap.py:19` | Correct package/bin/profile path and noncached metadata failures; bootstrap 4/4 tests passed |

Public migrations follow the official Rust API guideline [C-VALIDATE: functions validate their arguments](https://rust-lang.github.io/api-guidelines/dependability.html#functions-validate-their-arguments-c-validate): runtime image, tensor, mode, scale and archive contracts fail through `Result` at shared boundaries. No unchecked public loading bypass was introduced. Compatibility shims were not requested:

- Propagate `Result` from host/tensor setters, output allocation, generic image constructors, image conversions, tensor-layout helpers, exposure helpers, TZA sample decoding, and `tile::total_output_pixels(&plan)`. Typed image convenience constructors retain their asserting contract.
- `weights::resolve(key, quality, directory, policy)` returns `Result<Option<ResolvedWeights>>`; use its `bytes`, `stem`, and `source`. A supplied nonempty builder directory defaults to DiskFirst; an empty directory defaults to EmbeddedFirst. Explicit policy remains available.
- TZA `Tensor.data` is shared immutable `Bytes`; convert owned vectors with `.into()`. Valid aliased/overlapping archive payloads and unused trailing bytes remain accepted without copy amplification.
- Shared `run`/`run_tensors` take `RunOptions` for transfer/HDR/signed/scale/sanitation/output channels. Immutable `commit_tensor_model`/`execute_tensors` remains the renderer route.
- Explicit scale must be finite and positive with a finite reciprocal; invalid geometry/modes fail before I/O or model loading. Empty host descriptors are representable; executing an empty image is rejected.
- Retain the native minimum tile when a logical memory budget is infeasible and report the achieved estimate. Backend workspace and allocator overhead are excluded, so this is not an OS/device allocation cap.

Declared MSRV was corrected from 1.85 to 1.95 in workspace `Cargo.toml:8`, matching [pinned Burn `93e63e8`, Cargo.toml:18](https://github.com/tracel-ai/burn/blob/93e63e8e9cf215ffb870ed62132961588a774096/crates/burn/Cargo.toml#L18). Metadata also declares 1.95 for `cubecl-zspace`. The actual verification compiler is `rustc 1.99.0 (b940084d7 2026-09-28)`; this pass did not test the minimum 1.95 toolchain. No dependency-version upgrade was made for this correction.

### Paired LOCAL2.5 contract anchors

These reference anchors describe the intended runtime semantics, not identical backend arithmetic. Prefix all native paths with `D:/Projects/vfx.ref/oidn/`; Rust paths are relative to this workspace.

| Contract | Current Rust | LOCAL2.5 counterpart |
| --- | --- | --- |
| Valid-source padding and primary scale/transfer | `filters/unet_runner.rs:207`, `gpu_ops.rs:42` | `devices/gpu/gpu_input_process.h:41-54,90-118,244-251`; CPU `devices/cpu/cpu_input_process.isph:35-48,88-93` |
| Primary transfer and signed modes | `filters/rt.rs:482`, `gpu_ops.rs:42,68` | `core/rt_filter.cpp:63-70`, `core/unet_filter.cpp:551-564`, `core/rtlightmap_filter.cpp:58-61` |
| Balanced exposure and component sanitation | `autoexposure.rs:78,85,137` | `core/autoexposure.h:29-31`, `devices/gpu/gpu_autoexposure.h:29-42,59-62` |
| Scalar reduction and signed output order | `gpu_ops.rs:68` | `devices/gpu/gpu_output_process.h:48-72`; CPU `devices/cpu/cpu_output_process.isph:48-66` |
| Schema-derived widths/topology and RF | `oidn-model/src/descriptor.rs:30,98,141` | `core/graph.cpp:87-114,173-203`, `core/unet_filter.cpp:263-268`, `core/unet_filter.h:35-37` |
| Archive format and checked runtime boundaries | `oidn-tza/src/parser.rs:109`, `filters/mod.rs:89` | `core/tza.cpp:32-42,66-99`, `core/unet_filter.cpp:372-380,420-443` |

Rust shorthand in this table uses `crates/oidn-rs/src/` unless a crate is explicitly named. Existing [local pipeline](bughunt/local_pipeline.md) and [local model](bughunt/local_model_weights.md) preserve the original paired audit evidence; their “pending” statements describe that historical static pass.

## Retained functionality

Preserve Base/Small/Large/XL, mutable host and tensor façades, immutable committed RT, lightmap/directional and auxiliary-only modes, 1/2/3-channel f16/f32 images, row strides, cancellation/progress, explicit scale, ACEScg, diagnostics and all supported file formats. No public code is dead merely because no in-tree caller exists. Reuse the shared runner, existing resolver and existing loader rather than introduce parallel paths.

## Shared repair contract

```text
Host image / NCHW tensor + flags
 -> common role/mode/geometry validation
 -> canonical resolver -> validated TZA schema -> model descriptor
                                              | topology,widths,RF
 -> committed artifacts + tile plan <----------+
 -> fresh frame handles / balanced exposure
 -> preprocess VALID source by role -> place in ZERO tile
 -> shared Net forward
 -> inverse transfer -> destination reduction -> signed decode/clamp/scale
 -> crop/stitch -> tensor return or validated host adapter
```

This is the implemented shared contract; the source map above provides current anchors. All ledger entries are closed by the receipts below. Actual user-scene diagnosis remains a separate, unresolved investigation.

## Historical measurements retained

- [x] All 23 Rust archive sizes/SHA256s equal isolated pinned native assets.
- [x] Native CPU/CUDA smoke and 90/90 synthetic matrix runs produced finite output.
- [x] Pre-repair explicit-scale color-only max CPU/WGPU error0.0003814697265625; aligned full-AOV32x32 max0.000213623046875.
- [x] Pre-repair odd/tiny autoexposure and unaligned full-AOV cases showed substantial differences; these are the repair baseline, not post-fix results.
- [x] Squarebob CPU/PQ display repair was separately published; see plan3 publication receipt.
- [ ] Capture and replay the user's actual progressive scene/AOVs to identify its first divergent stage.
- [x] Establish bounded post-repair numerical results for frozen color/AOV/exposure and Large fixtures; auxiliary-only/signed/scalar public CPU contracts also passed.
- [ ] Establish general quality/convergence claims beyond the bounded fixtures.

Exact historical measurements/provenance: [native runtime](bughunt/native_runtime.md), [Astra numerics](bughunt/astra_numerics.md), [Squarebob verification](bughunt/squarebob_pq_verification.md). Adaptive-clamp spatial standard deviation restores fixture contrast and is not a noise-only metric. Cross-backend precision differences remain explicit.

## Verification receipts

Verified post-repair receipts (2026-10-02):

| Check | Exact command or artifact | Result |
| --- | --- | --- |
| Model/TZA | `cargo test -p oidn-model -p oidn-tza --locked --quiet`; [stdout](bughunt/model_final_test.stdout.log), [stderr](bughunt/model_final_test.stderr.log) | 21passed: model15,TZA6;171.278s overall; empty stderr |
| CLI unit/process | [stdout](bughunt/native-verification/cli_final_test.stdout.log), [stderr](bughunt/native-verification/cli_final_test.stderr.log) | 6unit+3process passed; process suite0.21s; empty stderr |
| CLI binary/CPU route | `cli_final_build.*`, `cli_final_help.*`, `cli_cpu_bench.*` under [native-verification](bughunt/native-verification/) | fresh binary/help passed; actualCPUFast3x2,iters1,`rt_hdr_small` |
| Documentation lint | `bunx --yes markdownlint-cli plan4.md README.md AGENTS.md DIAGRAMS.md --disable MD013 MD024 MD031 MD032 MD040 MD041`; [stderr](bughunt/repair_docs_lint.stderr.log) | Passed; empty logs. Historical long lines, duplicate headings, fence/list spacing and unlabeled fences are exempted |
| Bootstrap | [stderr](bughunt/native-verification/bootstrap_final_test.stderr.log) | 4/4passed in0.002s |
| Public CPU contracts | `cargo test --locked -p oidn-rs --test pipeline_contracts -- --test-threads=1`; [stdout](bughunt/native-verification/postfix/pipeline_contracts.stdout.log), [stderr](bughunt/native-verification/postfix/pipeline_contracts.stderr.log) | 6/6 passed in4.58s;252.608s total cold compile |
| Core CPU retry | [stdout](bughunt/core_cpu_final_retry.stdout.log), [stderr](bughunt/core_cpu_final_retry.stderr.log) | 51/51 passed: lib20, all-models1, API5, e2eCPU2, formats5, geometry11, planner1, color/tile6; 2GPU ignored; 38.400s; empty stderr |
| Formatting | `cargo fmt --all -- --check`; `bughunt/final_fmt_retry.*` | Passed in0.359s; empty logs after import-order correction |
| Final workspace feature matrix | `cargo test --workspace --all-targets --locked --features oidn-rs/embed-all,oidn-rs/acescg-autoexposure --quiet -- --test-threads=1`; [stdout](bughunt/final_workspace_features.stdout.log), [stderr](bughunt/final_workspace_features.stderr.log) | 96 passed, 14 ignored; 107.115s; empty stderr; includes latest CLI6+3, bench3, model/TZA21, geometry/color17 and public6 |
| Required GPU lane | `OIDN_REQUIRE_GPU=1 cargo test --locked -p oidn-rs --test e2e_wgpu --test e2e_ldr --test multi_tile_wgpu --quiet -- --ignored --test-threads=1`; [stdout](bughunt/final_gpu.stdout.log), [stderr](bughunt/final_gpu.stderr.log) | 14/14 passed; 19.674s; actual3072 multi-tile with maxmem512; Large High RF202 two/one callbacks and full/tiled error <=1e-4 |
| Benchmark checks | [stdout](bughunt/final_bench.stdout.log), [stderr](bughunt/final_bench.stderr.log) | 3/3 passed; rerun within final feature matrix |
| Final strict all-feature Clippy | `cargo clippy --workspace --all-targets --all-features --locked --quiet -- -D warnings`; `bughunt/final_clippy_all.*` | Exit0; 2.624s; empty logs |
| Strict rustdoc | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --locked --no-deps --quiet`; `bughunt/final_rustdoc_retry.*` | Exit0; 2.639s; empty logs after link correction |
| Doctests | `cargo test --workspace --locked --doc --quiet`; `bughunt/final_doctests.*` | Exit0; zero available doctests; no example-execution claim |
| Final formatting/diff | `cargo fmt --all -- --check`; `bughunt/final_fmt_all.*`; `git diff --check`, `bughunt/final_diff_all.*` | Both exit0; 0.336s and 0.093s; empty logs |
| Refreshed CLI binary | `cargo build -p oidn-cli --bin oidn-rs --locked --quiet`; `native-verification/cli_refreshed_build.*` | Exit0; 13.732s; empty logs |
| Frozen native matrix | `python bughunt/native-verification/postfix/compare_runtime.py --rust target/debug/oidn-rs.exe --rust-device both` | 108/108finite,302958values;78.217s |
| Large seam | same script with `--rust-device wgpu --seams` | 6/6finite;4.617s; actual2tiles768x16,RF202 |

The exact native commands, binary SHA256, geometry and metrics are in [postfix receipt](bughunt/native-verification/postfix/receipt.json), [matrix](bughunt/native-verification/postfix/comparison.json), and [seams](bughunt/native-verification/postfix/seams.json). Rust binary SHA256 is `1a898451b3d81f57507a3900a9e3d227827bb6f833ec95893fca3c7ffb1e0693`.

| Frozen comparison: RustWGPU vs nativeCPU max absolute RGB | Before | After |
| --- | ---: | ---: |
| Odd33x19 automatic exposure | 1.7249717712 | 0.0001411438 |
| Tiny8x8 automatic exposure | 3.2188835144 | 0.0000286102 |
| FullAOV33x19 explicit scale1 | 0.9775581360 | 0.0001964569 |
| FullAOV33x19 explicit scale0.02 | 4.6651268005 | 0.0001258850 |
| FullAOV32x19 explicit scale0.02 | 3.6860580444 | 0.0001068115 |

Across this matrix RustWGPU/nativeCPU maximum is0.0003814697265625; RustCPU/nativeCPU maximum is0.00031280517578125. Large clean-aux High769x16 at scale0.02 has tiled/full maximum0 and RustWGPU/nativeCPU maximum0.000152587890625. NativeCUDA/nativeCPU maximum1.2774772644042969 demonstrates that native backend arithmetic also differs; it is not a clean-target quality score.

Initial strict rustdoc failed on module-level `Self::set_*` links in `filters/rt.rs:14-20` and public-to-private helper links in `gpu_ops.rs:3`. [Initial stderr](bughunt/final_rustdoc.stderr.log) preserves the failure. Documentation links were qualified as `RtFilter::...`, and private helper references became plain code text; the strict retry passed. Final source-comment/topology and formatting corrections are covered by the successful all-feature Clippy and final formatting receipts above.

Initial final-core pass is not a clean receipt: `planned_rectangles_cover_each_pixel_once_with_valid_crops` failed at `crates/oidn-rs/tests/geometry_boundaries.rs:130` with `InvalidArgument("invalid tile plan dimensions or metadata")`; that target reported 10 passed and 1 failed. [Core stdout](bughunt/core_cpu_final.stdout.log) and [stderr](bughunt/core_cpu_final.stderr.log) preserve the failure. The root fixed the overly strict `pad >= tile_size` metadata check to `pad > tile_size` at `crates/oidn-rs/src/tile.rs:259-262` (current source after the checked-count insertion). LOCAL2.5 `core/unet_filter.cpp:289-292` permits equality for a tiny dimension below a larger tile alignment: tile size is rounded to16, then padding is its remainder modulo tile alignment. This preserves the native geometry instead of changing the planner. The complete same CPU suite then passed 51/51, including geometry11/11; the original failure remains as caught-and-fixed regression evidence.

Final boundary review also found that the public helper `tile::total_output_pixels(&plan)` could overflow an unchecked i64 sum for three malicious i32MAX rectangles, or return negative areas. The existing helper now returns `Result<i64, OidnError>`, checks positive dimensions and checked addition, and has a malicious-plan regression. The final feature matrix reran geometry11/color-tile6 successfully after this change; the prior 51-test receipt describes the earlier source pass. Descriptor planning arithmetic now uses `div_ceil`; model/TZA21 also passed again in the final feature matrix.

Supported feature tests, explicit GPU regressions, benchmark tests, strict Clippy, strict rustdoc and source review passed. The matched-native binary hash above belongs to that numerical pass; a refreshed CLI binary subsequently built successfully. Original progressive scene/AOV replay, exhaustive tile geometries, release-mode execution, other hardware, physical convergence and universal bitwise equality remain unverified. Publication approval has been received for this concrete report and the verified repairs; commit and remote receipts are pending while delivery is in progress.

### Squarebob consumer verification

The renderer bridge now uses canonical `RtFilter::builder` resolution rather than a second byte cache/manual `.weights` route. An explicit weights directory uses DiskFirst; absence uses EmbeddedOnly. Its committed cache key retains role/quality/dimensions and excludes runtime scale; each pass updates scale and sanitation without rebuilding the model. Current anchors: `../squarebob-rs/crates/pt-denoise-oidn/src/lib.rs:516,528,533,559`; OIDN `crates/oidn-rs/src/filters/rt.rs:571`.

A detached, self-created Squarebob worktree at `f0c64c7` received only the bridge source, SHA256 `0179E091EAB3A6A3AD0A81A9CBDE423A8D7462CE14FA730FC8C41AF5EDAF5504`. A command-only Git patch selected this local OIDN façade:

```text
cargo check --workspace --all-targets --quiet --config patch."ssh://git@github.com/ssoj13/oidn-rs.git".oidn-rs.path="C:/projects/projects.rust.cg/cglibs/oidn-rs/crates/oidn-rs" --config build.target-dir="C:/projects/projects.rust.cg/cglibs/squarebob-rs/target"
```

Check passed in1.653s with empty logs: [stdout](bughunt/native-verification/bob_local_contract_check_verified.stdout.log), [stderr](bughunt/native-verification/bob_local_contract_check_verified.stderr.log). The freshly linked `target/debug/examples/oidn_highlight_probe.exe` then passed in4.545s on RTX3080Ti/Vulkan, frozen96x64 HDR/full-AOV/Balanced/scale0.02. Repeated output and fixed-clamp SPP1/256 output were identical; 32 additional repeats were identical. Adaptive SPP1/256 changed by max8.448264122, reproducing the previous bounded input-policy diagnostic, not the user's scene. [Probe stdout](bughunt/native-verification/bob_local_contract_probe_verified.stdout.log) and [stderr](bughunt/native-verification/bob_local_contract_probe_verified.stderr.log) record results; stderr contains the existing Bandicam Vulkan layer API1.2/1.3 warning.

The isolated worktree was removed after verification. Original Squarebob lock SHA256 remained `F600F1F5390E0D46066076690A7A93D26F900AA66942F277030C25453F83FCBF`; no permanent path dependency or original lock edit was introduced. The user has now approved publishing OIDN and updating Squarebob's permanent Git revision. These delivery steps are in progress; the unchanged-lock receipt above describes the completed isolated verification. Earlier published Squarebob PQ/CPU fixes remain recorded in plan3.

## Resumption

Read this ledger, inspect the current working tree and each relevant report, then review the repaired contracts and remaining scene-diagnosis scope. Source owners are working concurrently: preserve their edits. Record completion immediately when evidence exists. GitNexus freshness/impact/reindex must be attempted when available; direct source/caller checks are the documented fallback when tools cannot complete. No scene-root-cause or numerical-parity claim follows from static inspection.
