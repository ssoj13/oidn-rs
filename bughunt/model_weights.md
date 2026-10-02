# Model and TZA audit — 2026-10-02

## Scope and evidence

Read every source file, manifest and integration test in `crates/oidn-model` and `crates/oidn-tza` (17 files). Inspected the RT model construction boundary, CLI weight handoff, all 23 shipped archive names/sizes, and git history. No builds or tests were run; no production code was edited. GitNexus tools were unavailable in the exposed inventory; filesystem MCP reads/searches and read-only git commands were used.

The documented local C++ checkout is absent. Official OIDN v2.4.1 source was downloaded to `bughunt/reference-v2.4.1`; all C++/Python citations below refer to that directory. Upstream links: https://github.com/RenderKit/oidn/blob/v2.4.1/core/unet_filter.cpp and https://github.com/RenderKit/oidn/blob/v2.4.1/training/model.py . The historical `reference-v2.3.3` directory name does not establish source version; that directory is retained without being used as this report's citation root. Current cited reference files are official v2.4.1 under `reference-v2.4.1`.

No confirmed cause of the user's observed noise can be assigned from this code-reading pass. The findings below establish structural defects and measurement gaps, not a reproduced image regression.

## Confirmed findings

### MW1 — Small custom weights, including the normal CLI Fast path, are constructed as Base (high)

`variants.rs:27-33` uses tensor names to detect only Large versus Base. Small and Base have the same layer names but different widths (`variants.rs:57-81`; reference `training/model.py:61-84`). `filters/rt.rs:550-558` uses this detection for every byte override. Loading shipped `rt_hdr_small.tza` through `.weights(bytes)` constructs Base and fails a shape check in `loader.rs:57-62`; the first differing convolution is `enc_conv2` (Base output48 versus Small32).

This affects ordinary CLI Fast requests: `oidn-cli/src/main.rs:267-283` resolves the small archive, discards its stem, and passes its bytes as user weights at `main.rs:342-344`. It is an execution error, not evidence of noisy successful output.

Systematic solution: infer topology and all channel widths from a validated tensor-map schema, independent of filename; construct and load one model description. Centralize weight resolution in the filter and avoid converting normal resolution into an override in the CLI. Native OIDN identifies topology from names (`core/unet_filter.cpp:263`) but its graph receives the complete tensor map (`:280`) rather than a hardcoded width preset.

### MW2 — XL is exposed but not supported by the RT byte-loading route (medium)

`unet_large.rs:59-69,136-137` retains XL widths/constructor, matching reference `training/model.py:167-176`. `Variant::from_tensor_names` always returns Large for either Large/XL. `filters/rt.rs:560-562` always calls `UNetLarge::new`, including its XLarge match arm, selecting BASE widths. XL custom archives therefore fail shape checks. Preserve the feature; infer widths from validated tensors. The shipped asset inventory has no XL file, so this is an exposed unfinished route, not dead code.

### MW3 — Checked archive offsets do not protect unchecked tensor size multiplication (medium)

`oidn-tza/src/types.rs:38-43` computes the dimension product and byte size with unchecked multiplication. `parser.rs:166` invokes this before the checked offset addition at `:169-172`. Four dimensions of `u32::MAX` in an otherwise valid oihw table can overflow usize: debug panic or release wrapping, with incorrect byte bounds. On 32-bit targets `parser.rs:120,159` also truncates u64 offsets with `as usize`. Zero dimensions are accepted.

Reference parser `core/tza.cpp:66-70,93-95` similarly depends on tensor-descriptor byte sizing; native permissiveness is not a reason to retain a Rust panic contract. Solution: one checked TensorDesc validation/size path; reject overflow/invalid dimensions and use fallible offset conversions before slicing. Add malformed-archive tests in a later verification pass.

### MW4 — Public raw tensor representation makes Result loaders panic (medium)

`types.rs:49-51` exposes arbitrary descriptor/data pairs; `as_f32/as_f16` call `bytemuck::cast_slice` at `:58,66`. An f32 Tensor with a one-byte data Vec is a valid constructible Rust value yet decoding panics. Loaders validate descriptor shape/layout (`loader.rs:50-62,79-91`) but not data length, then decode at `:64-68,93-97`; their Result signature does not prevent malformed-public-input panics. Vec<u8> also does not promise the stronger alignment required by these casts, and native-endian casts are inconsistent with explicit little-endian parser reads on big-endian hosts.

Reference `core/tza.cpp:18-23` uses memcpy for scalar reads and `:93-98` checks archive tensor extent; neither establishes a safe aligned typed Rust Vec contract.

Solution: validated Tensor construction and a shared fallible little-endian decode method that checks payload length and decodes unaligned byte chunks. Reuse that method for both parameter ranks. Alignment/big-endian concerns are portability hazards, not demonstrated faults on this Windows renderer.

### MW5 — Duplicate decoding and model primitives (low)

Three separate dtype conversion implementations exist: `types.rs:72-76`, `loader.rs:64-67`, `loader.rs:93-96`. Identical conv3 definitions and nearest upsampling exist at `unet.rs:23-27,147-153` and `unet_large.rs:26-30,191-197`. Reuse existing helpers in a common module rather than creating parallel model-specific functions. Reference primitives are already shared (`training/model.py:32-49`). Keep topology-specific layer ordering explicit.

### MW6 — Existing tests cannot certify image fidelity (high measurement gap)

`load_real_weights.rs:32-42` checks finite and nonzero output for constant input; its 9-channel test only checks finiteness (`:60-65`). Small checks shape/finiteness/nonzero (`unet_small.rs:18-23,39-46`); Large does likewise (`unet_large.rs:18-32,52-59,89-93`). `unet_shapes.rs:11-22` uses zero input. These assertions admit wrong interpolation, incorrect parameters, feature ordering and substantial image artifacts. No pixel-level reference, intermediate-layer comparison or GPU-vs-CPU comparison exists in these owned tests.

Real-weight tests silently return when files are absent (`load_real_weights.rs:15-18`, `unet_large.rs:37-39`, `unet_small.rs:28-30`). Parser test `parse_all_weights.rs:19-21` likewise skips the entire archive pass and still advertises a stale submodule setup message; its count check is only >=20 (`:37`) against the actual23 archives. Its only layer-shape spot-check is rt_hdr (`:64-75`). `load_real_weights.rs:9` alone uses a relative current-working-directory path while other tests use CARGO_MANIFEST_DIR.

Solution after approval: required asset manifest/hash checks; exact schema/load checks for every archive, explicit opt-in skip policy; golden asymmetric inputs and native output; per-layer taps on first divergence; CPU/WGPU numerical comparisons. Do not claim shape/finite checks prove quality.

## Verified parity and exclusions

- Base/small widths and convolution shapes match reference `training/model.py:61-103` against `variants.rs:57-81`/`unet.rs:71-86`.
- Large/XL widths and shapes match reference `training/model.py:167-208` against `unet_large.rs:47-69,108-126`.
- Runtime op ordering, pooled skip tensors, concat order and four nearest upsamples match `core/unet_filter.cpp:468-497,500-530` against `unet.rs:104-142`/`unet_large.rs:147-187`.
- Final Base ReLU is correct for shipping runtime (`core/unet_filter.cpp:495`, Rust `unet.rs:142`) even though training Python omits it (`model.py:153`). Large final ReLU also matches (`core/unet_filter.cpp:528`, Rust `unet_large.rs:187`). Do not remove it based solely on Python.
- Loader explicitly replaces every convolution weight/bias for both topologies (`loader.rs:134-149,165-183`), preserving OIHW ordering and f16 to f32 numerical conversion (`:64-68,93-97`). No retained random parameter was found in successful loads.
- Parser magic/version/layout/dtype decoding matches `core/tza.cpp:32-42,73-90` against `parser.rs:109-117,139-156`.
- Duplicate archive tensor names silently replace earlier tensors in Rust (`parser.rs:182`), whereas native emplace keeps the first (`core/tza.cpp:99`). Reject duplicates rather than adopting either ambiguous behavior.
- No TODO/FIXME appears in the owned sources. `to_f32_vec` has no internal callsites but is exported functionality; reuse it with a validated contract rather than delete it.
- `oidn-model/Cargo.toml:21` declares half directly although owned model source uses the re-exported TZA type; likely unnecessary direct dependency, minor cleanup.
- 23 .tza archives exist with realistic blob sizes (634568–7698439 bytes). Read-only `git diff --numstat eeb9c37 HEAD -- data/weights` returned no changes. This excludes committed post-LFS archive replacement, not uncommitted corruption or upstream mismatch.

## History and noise investigation

Only four commits touch the owned crates/assets: initial0df2cfc; eeb9c37 LFS-to-regular-blobs; 91a261e May21 runtime-parity fix; bebbfc5 July24 Burn0.22/WGPU30 migration. The May fix changed Base final ReLU (`unet.rs:142`) as required by native runtime. The migration diff inspected here changes backend generic types, Device signatures and formatting; topology/decoding arithmetic is unchanged. Dependencies can change numerical behavior despite unchanged application operators; this remains a hypothesis requiring recorded sample/backend comparison, not a proven migration defect.

Cross-boundary RF concern: `filters/rt.rs:581` supplies BASE receptive field unconditionally, while native `core/unet_filter.cpp:266-268` branches Large/Base. Geometry agent owns detailed impact. This can affect Large multi-tile borders and belongs in the highest-priority image experiment.

## Dataflow and codepath

```text
resolved bytes / override bytes
    -> parser::parse
       LE header + named tensor table
       -> TensorDesc size -> bounds -> owned raw payload
    -> RT variant selection (stem OR tensor names; divergent!)
    -> fixed preset UNet / UNetLarge construction
    -> load_tza / load_tza_large
       fetch pair -> validate shape/layout -> decode -> replace Param
    -> Net::forward
       conv+ReLU -> pools1/2/3 -> bottleneck
       upsample+concat(pool3/2/1/input) -> decoder+ReLU
    -> postprocess + tile crop (pipeline/geometry audit)
```

Proposed canonical path: bytes -> validated archive schema -> model descriptor(topology,widths,RF) -> construct+load -> forward. Filename remains selection metadata; tensor schema determines executable architecture. Same descriptor supplies tiling and diagnostics.

## Checklist and follow-up

- [x] Read all17 owned crate files including all tests/manifests.
- [x] Compare both topology definitions, all widths, activations and loading lists with official v2.4.1.
- [x] Check parser formats, offsets, sizes and public decode invariants.
- [x] Inspect23 asset names/sizes and committed archive history.
- [x] Search callsites, duplicate paths and TODO/FIXME.
- [x] Confirm CLI Fast crossing with direct code reads.
- [x] Store audit conclusions in MCP memory.
- [ ] Hash all23 assets against a pinned upstream asset revision (not performed).
- [ ] Execute malformed-archive/public-Tensor cases after approval.
- [ ] Record actual problematic input, previous smooth output, exact quality/model/backend/dependency hashes.
- [ ] Compare native2.4.1, Rust CPU and Rust WGPU outputs on same inputs.
- [ ] Compare intermediate tensors at first mismatch; tiled/untiled asymmetric9-channel images.
- [ ] Implement descriptor/resolver/decode consolidation after approval, preserving Small/XL features.

Tool failure encountered: missing destination core directory caused http_download to expose an internal stack trace twice; root notified and logs centrally in filesystem-mcp-rs BUG_CDX.md. Creating the parent and retrying succeeded. Missing original reference source reads were expected user errors, not defects.
