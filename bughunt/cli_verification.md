# CLI, verification and tooling audit — 2026-10-02

Baseline: clean working tree at `bebbfc5ec26c589121e2b0878ff55a6f669b43a6`. Static audit only; no builds/tests executed. Reference: official RenderKit/oidn tag v2.4.1 fetched through GitHub MCP and preserved in `bughunt/reference-v2.4.1/apps/`. These findings do not establish which defect caused the user's observed render noise.

## Coverage

- [x] Both CLI sources, all five Cargo manifests, Cargo.lock dependency versions, rust-toolchain, CI, bootstrap.py, README, CHANGELOG, AGENTS, DIAGRAMS, gitignore and gitattributes.
- [x] All eight façade integration-test files and example bench.rs.
- [x] Complete Rust-source inventory (43 files) assigned between root and three source-audit agents; crate model/TZA tests assigned to model agent.
- [x] Explicit TODO/FIXME/HACK/todo!/unimplemented!/dead_code search: no matches. Broader legacy/temporary/future search: live adapters and unfinished XLarge contract require retention, not deletion.
- [x] Git history of CLI, pipeline migration and historical plan reports.
- [ ] Runtime reproduction and native/golden comparisons: separate approval stage.

## C1 — HIGH: CLI converts normal weight lookup into custom-archive loading

`crates/oidn-cli/src/main.rs:267-283` resolves an ordinary RT model, discards its stem and passes bytes through `.weights(bytes)` at `main.rs:342-344`. `filters/rt.rs:550-554` then uses tensor-name detection instead of the actual Small/Base variant. `oidn-model/src/variants.rs:27-33` can only return Base/Large. Small models share Base names but have different widths (`variants.rs:57-81`), so normal `denoise --hdr --quality fast --weights-dir data/weights` reaches the wrong constructor and fails with ShapeMismatch. This is a source-path proof, not a runtime execution. Cross-check: model agent independently confirmed expected enc_conv2.weight mismatch. Upstream `core/unet_filter.cpp:254-279` reads channels from the archive and derives graph metadata, rather than constructing fixed widths from names.

**Systemic solution:** keep ordinary resolution in the filter; infer topology AND widths from validated archive shapes for actual custom blobs; use one canonical resolver shared by RT/lightmap/CLI and preserve stem/provenance in metadata. Do not add a special Fast-only workaround.

## C2 — HIGH: CLI file color space is disconnected from filter color space

`main.rs:263-265` loads color, albedo and normal with the same flag-free loader. `io.rs:86-89` converts pixel layout to RGB f32; `io.rs:106-115` quantizes output without calling the existing sRGB transfer functions. Filter options `main.rs:333-338` never reach I/O. Consequently sRGB PNG/JPEG used with linear mode gets processed as if encoded values were linear; linear EXR output written to PNG stays linear numeric data. Albedo PNG lacks its required independent decode, while normal data must remain numeric.

**Reference:** `apps/oidnDenoise.cpp:269-283` decodes albedo with srgb=false, leaves normals numeric and loads color using srgb; `apps/utils/image_io.cpp:411-439` performs decode/encode at file boundaries. **Dependency cross-check:** lock pins image0.25.9 (`Cargo.lock:2744-2747`); locally read image source `src/images/dynimage.rs:277-280,295-299`, `src/images/buffer.rs:1565-1581` and `src/metadata/cicp.rs:1165-1173` confirms RGB-to-RGB conversion casts channels without transfer conversion. Context7 corroborated distinction between layout casting and explicit color-space transformation.

**Systemic solution:** extend existing load/save APIs with input semantic/encoding arguments and reuse `oidn_rs::color::{srgb_forward,srgb_inverse}`. Apply encoding exactly once at boundaries; albedo/normal semantics must be explicit. Maintain current HDR float save behavior and support all existing formats.

## C3 — MEDIUM: PFM/PHM parser loses scale and corrupts CRLF payload boundary

`io.rs:275-277` keeps only scale sign; `io.rs:180-184,226-231` never multiply samples by scale magnitude. A scale -2 with sample0.5 returns0.5 instead of1.0. Upstream `apps/utils/image_io.cpp:87-104,187-210` multiplies by absolute scale. `io.rs:301-302` consumes only one whitespace character: final CRLF consumes CR and leaves LF as the first pixel byte. Header accepts PF and PH in either loader (`io.rs:164-171,211-218,268-271`), although sample widths differ. Width/height and payload multiplications (`io.rs:170-172,217-219`) are unchecked; nonfinite/zero scale is accepted. Valid row flipping and endian decoding should be retained.

**Solution:** one shared validated header/payload path parameterized by sample width; validate expected magic, finite nonzero scale, checked positive dimensions, exact header end; preserve endian support; apply magnitude on both formats. Use asymmetric small file fixtures later, including CRLF and first binary byte that happens to be whitespace.

## C4 — MEDIUM: reference comparator can report success for NaNs

`main.rs:438-454` computes f32 difference/square before f64 accumulation, ignores nonfinite operands and compares NaN MSE with >threshold (false). maxerror NaN/negative and maxerror without ref lack argument validation (`main.rs:153-159,244-259`). This undermines a quality gate. Upstream comparison is a different per-pixel contract at `apps/oidnDenoise.cpp:435-445`; Rust intentionally documents MSE, so do not silently change it.

**Solution:** retain documented MSE, validate finite inputs and nonnegative finite threshold, accumulate f64 differences/squares, fail on any invalid sample, require ref when threshold supplied; share one metrics implementation with the benchmark. PSNR needs an explicitly stated peak for HDR (`main.rs:445-448` currently assumes1).

## C5 — MEDIUM: accepted CLI controls are silently ineffective

`main.rs:124-125` accepts directional for RT but `run_rt` never forwards/rejects it. `maxmem` (`main.rs:141-143`) is forwarded only for RT (`main.rs:339-341`), omitted in lightmap builder (`main.rs:393-399`). Lightmap also requires hdr/ldr choice globally even though construction ignores that choice (`main.rs:246-250,393-396`), risking false expectations. Upstream forwards maxMemoryMB to both families (`apps/oidnDenoise.cpp:338-361`). Threads already explicitly announces its no-op, so it is not dead/unimplemented functionality to delete.

**Solution:** family-specific validation and forwarding. Preserve capabilities, report unsupported choices explicitly, and share post-execute save/compare code between run_rt/run_rtlightmap (`main.rs:365-380,409-421`).

## C6 — MEDIUM: benchmarks accept zero iterations and have duplicated helpers

`main.rs:552-553` divides/indexes empty timing vector for bench --iters0. Example `bench.rs:409-424` also indexes empty timings. Example modes/quality/resolution parsing uses panics (`bench.rs:135-168`). Synthetic generators and RMSE duplicate `e2e_wgpu.rs:29-68`, `e2e_ldr.rs:18-51`, `bench.rs:219-279`; two epoch helper bodies are identical (`bench.rs:446-465`). The example can exit successfully after every run failed (`bench.rs:519-528`).

**Solution:** validate positive iteration/dimension/count inputs, use meaningful errors, share fixtures/metrics across tests and example, one timestamp helper, fail summary if no successful rows. **Rejected suspicion:** output reallocation in example403-417 does NOT force net reload for identical dimensions; independently checked by geometry agent against rt.rs373-386,792-793.

## C7 — HIGH verification gap: current tests cannot certify render quality parity

- `multi_tile_wgpu.rs:112-148` checks only row means: vertical seams, local checkerboard errors, and constant-zero output can escape. Planner area-sum assertion `multi_tile_wgpu.rs:67-76` cannot establish exact coverage without detecting overlapping rectangles/gaps individually.
- Signed normal and directional tests check only finite output (`e2e_wgpu.rs:292-296,439-442`).
- HDR CPU smoke permits zero output for this scene (`e2e_ndarray.rs:64-73`: input mean below1, difference<1). It is not a proof of glue correctness claimed by lines4-5.
- Many quality tests force input_scale1 (`e2e_wgpu.rs:88,139,175,204,324,365`); no full pipeline native exposure parity check.
- `all_models_smoke.rs:38` uses Default::default despite claiming CPU at lines2-4. Burn0.22 dispatch can choose GPU when wgpu feature enabled. Migration commit bebbfc5 specifically documents fixing this hazard in oidn-model tests, but this façade test remains.
- CI claims no-GPU lanes but runs all workspace GPU tests with shipped weights (`.github/workflows/ci.yml:47-51` and `e2e_wgpu.rs:73-78`). Availability of weight directory is the only skip condition; no explicit GPU-lane isolation.

**Solution:** explicit Device::ndarray for CPU; dedicated GPU lanes; exact pixel coverage; signed-range assertions; single/multi-tile pixel comparisons around both seam directions and corners; native OIDN2.4.1 image corpus for every input family/quality/scale, tensor vs host parity and CPU vs GPU tolerance. Smoke and noise-reduction tests remain useful but cannot substitute for golden comparisons.

## C8 — MEDIUM: bootstrap install cannot identify a bin target correctly

`bootstrap.py:119-121` returns package names, not executable target names; install224-226 uses --path . at a virtual workspace root and --bin oidn-cli, while actual target is oidn-rs in `crates/oidn-cli/Cargo.toml:15-17`. Debug option branch is empty (`bootstrap.py:226`), verbose is parsed and discarded (`bootstrap.py:260-261`); metadata failures become empty configuration (`bootstrap.py:94-99`) that can misleadingly report success with no targets.

**Solution:** retain discovered package manifest parent AND target name; install from actual crate directory with correct profile; validate metadata errors, propagate requested verbosity. No C++ counterpart (project-specific tooling).

## C9 — Documentation/configuration drift

README7-12,32,74,95-100 and AGENTS describe generic Backend and 24 models; current source uses dynamic device dispatch and agents count23 weight files. README44 claims embedded default, but CLI dependency enables no embed feature (`oidn-cli/Cargo.toml:21`, `oidn-rs/Cargo.toml:24`). README11 points to deleted plan1; git7fc4eff contains it, b999415 deletes it. Historical next plan number is2, even though current root has no plan*.md. .gitignore7 says Git LFS while .gitattributes has no LFS rule. DIAGRAMS claims all divergences closed; new confirmed semantic failures contradict an unqualified present-tense claim.

**Solution:** update AGENTS/DIAGRAMS now as audit artifacts, preserve history clearly; defer README/API examples to approved implementation pass. Retain existing public host/tensor adapters, reserved XLarge, CPU reference functions and registry capabilities; no code removal solely because no in-tree caller exists.

## Proposed validation sequence (not executed)

1. Scene provenance: user's backend, settings, weights hash, last-good commit, dimensions and input AOVs.
2. Native2.4.1 vs RustCPU vs RustGPU for identical raw buffers and weights; separate file encoding from tensor numerics.
3. Fix primary semantics/padding/RF/autoexposure globally; validate every shared route.
4. Fix canonical archive loading/resolution and invalid buffer contracts.
5. Add focused regression fixtures for I/O/metrics and backend isolation; run required checks only in validation pass.

## Tool limitations

GitNexus and fetch MCP were not exposed in tool inventory, so graph freshness/impact/reindex could not run. No Rust symbol edited in audit; filesystem MCP source reads and read-only git history used. Filesystem HTTP allowlist rejection and missing download parent produced full internal backtraces: reporting defect recorded preserving entries in external BUG_CDX.md. Missing local reference and deleted historical report are expected missing files, not server defects. API-guidelines checklist fetched by GitHub MCP; applicable C-VALIDATE, C-GOOD-ERR, C-FAILURE, C-STRUCT-PRIVATE, C-CALLER-CONTROL and C-INTERMEDIATE inform proposed solutions.
