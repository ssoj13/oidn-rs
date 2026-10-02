# Geometry, image boundaries, weights, and public API audit

Date: 2026-10-02. Read-only source audit; no builds/tests were run and no production source was edited.

## Scope and evidence

Fully read: `crates/oidn-rs/src/{device,error,filter,image,image_tensor,lib,prelude,registry,tile,weights}.rs`, `crates/oidn-rs/Cargo.toml`. Read related commit/execution boundary sections in `filters/rt.rs` and `filters/rtlightmap.rs`, plus `tests/{formats,unit_color_tile,api_surface,multi_tile_wgpu}.rs`. Source inventory and callsite searches used filesystem MCP. GitNexus tools are absent in the available tool inventory; direct source/callsite inspection was the fallback.

The advertised local reference `C:/projects/projects.rust.cg.offload/oidn/core/*` is absent. Searches for `unet_filter.cpp` under the Rust workspace and offload root found none. This is missing input, not a filesystem server defect. Official upstream source was fetched using GitHub MCP at tag `v2.4.1`, consistent with `crates/oidn-rs/src/lib.rs:58`:
- [unet_filter.cpp](https://github.com/RenderKit/oidn/blob/v2.4.1/core/unet_filter.cpp), blob eca3fd82f1789c16b2ffbe8d0623dfb35b671936.
- [unet_filter.h](https://github.com/RenderKit/oidn/blob/v2.4.1/core/unet_filter.h).
- [image.cpp](https://github.com/RenderKit/oidn/blob/v2.4.1/core/image.cpp), blob b2173633142b7957c5d8ec953d65165c47180f9e.
- [image_accessor.h](https://github.com/RenderKit/oidn/blob/v2.4.1/core/image_accessor.h).
- [Rust API guidelines C-VALIDATE](https://github.com/rust-lang/api-guidelines/blob/master/src/dependability.md#functions-validate-their-arguments-c-validate).

C++ citations below refer to those fetched tag files, not an unverified local checkout.

## Findings

### G1 — High: large-model tile overlap is selected as base-model overlap

Rust: `filters/rt.rs:550-563` distinguishes Base/Small and Large/XLarge, but `filters/rt.rs:578-584` unconditionally passes `RECEPTIVE_FIELD_BASE` to the tile planner. `tile.rs:9-13` defines RF=174/202 and alignment16, and `tile.rs:88` derives overlap.

Reference: `core/unet_filter.cpp:263-268` detects the large model from weight tensor names and selects RF accordingly; `core/unet_filter.h:35-37` specifies174/202/16.

With alignment16 the present code discards96px at internal tile boundaries, while Large requires112px. Thus16px of insufficient-context output survives each internal boundary. This is a proven contract defect and plausible source of seam-local artifacts; it is not proof that it explains the user's unspecified render noise. It applies to large auxiliary-only models and `rt_hdr_calb_cnrm_large`, including custom large weights. Ordinary `rt_hdr` and noisy-aux routes have no large blob in the current23-weight inventory.

Proposed solution: carry receptive field as model metadata from the actual loaded topology into the existing shared commit artifact path. Use one source of model variant/topology truth; do not branch independently by guessed filename in tile code. Compare tiled versus whole-frame output for large models, with seam-local and interior errors separately.

Existing `tests/multi_tile_wgpu.rs:93-96` uses color-only `rt_hdr` High, which falls back to Base because no `rt_hdr_large.tza` exists. Its row-mean-only metric `tests/multi_tile_wgpu.rs:112-148` cannot demonstrate large-model correctness or localized vertical seams.

### G2 — High: public image descriptors bypass essential layout validation

Rust: all `Image`/`ImageMut` descriptor fields are public (`image.rs:60-75`). Constructors check length only in debug builds (`image.rs:84,100,113` and other typed constructors). `to_rgb_f32` and `write_rgb_f32` directly slice rows (`image.rs:170-181,275-307`) and cast byte slices to aligned typed slices. No checked-size/stride validator exists in the inspected crate.

Reference: `core/image.cpp:14-15` rejects excessive dimensions; `core/image.cpp:20-34` rejects undersized pixel/row strides; `core/image.cpp:54-65` validates buffer bounds.

Deterministic source-derived cases (not executed):
- An R32f descriptor with width2,height2,row_stride4 over three f32 values reads rows [a,b] and [b,c], silently overlapping rows; reference rejects row_stride<8.
- A short byte buffer panics at row slicing instead of returning InvalidArgument.
- A valid-sized byte slice starting at a non-f32-aligned address, or a stride not divisible by element alignment, can panic in bytemuck casts.
- Very large dimensions overflow products before allocation or wrap later conversion to i32 (`filters/rt.rs:579-580`, `filters/rtlightmap.rs:268-269`).

Proposed solution: centralize checked geometry/layout validation in the existing image layer and call it before copying/allocating/committing. Private validated fields prevent invalid descriptors; checked arithmetic must cover final occupied row extent, tensor element counts, and i32 narrowing. Either support unaligned byte buffers by decoding values from bytes or explicitly enforce/document alignment once. Return structured errors before expensive model loading. Do not add separate per-filter validators.

This can corrupt image geometry for malformed integration descriptors; no evidence connects it to ordinary typed contiguous RGB render inputs.

### G3 — Medium: weight resolution remains duplicated with incompatible error/source policies

Rust: `weights.rs:136-152` implements embedded-first per-candidate resolution and discards every filesystem error at147. `filters/rt.rs:519-539` duplicates candidate traversal but uses disk only and propagates non-NotFound errors at531-532. `filters/rtlightmap.rs:246-257` has a third path. CLI calls the public resolver (`crates/oidn-cli/src/main.rs:280`).

Reference: `core/unet_filter.cpp:438-459` has one quality routing point.

Consequences: enabling embed-* does not automatically make direct RT/Lightmap builder disk lookups use embedded data; public resolver can silently fall back from unreadable preferred large weights to base weights, changing quality and concealing permission/I/O failures. Embedded-first behavior can shadow disk updates for the same stem. These are observed code policies; intended source precedence requires an explicit decision.

Proposed solution: extend the existing resolver with a source-policy argument and Result return, preserve explicit custom-weight override, skip only NotFound, and return resolved stem/topology/source provenance. Route all library and CLI loading through it. Consolidate available model metadata and embedding coverage rather than add another candidate loop.

### G4 — Medium: public tile and tensor helpers enforce contracts only implicitly/debug-only

Rust: `tile.rs:81-101` takes unconstrained signed dimensions, alignment, RF and budget. Alignment0 divides by zero at58/92; negative dimensions produce nonsensical jobs; size products overflow at101/137. `image_tensor.rs:21,40,81` validates lengths only in debug builds; conversion assumes N=1 only in documentation at54 while accepting arbitrary `Tensor<4>` at57. Excess values can be silently ignored by layout conversion in release, undersized inputs panic.

Reference: tile arguments are internal state after `core/unet_filter.cpp:257` checkParams and image descriptor validation `core/image.cpp:14-15`; exposing them as Rust public APIs requires its own boundary validation. There is no direct C++ API equivalent for the CHW/HWC helpers.

Proposed solution: validate cheap boundary invariants in release with checked arithmetic and Result, or constrain argument types/visibility. Enforce actual NCHW contracts once at the public tensor boundary. Do not remove reusable layout helpers: production callsites exist in `filters/unet_runner.rs:318,337-338`.

### G5 — Medium: max-memory setting is an estimate, not a verified cap

Rust: `filters/rt.rs:566-576` estimates1536 or4096bytes per pixel and caps at DEFAULT_MAX_TILE_SIZE, while `tile.rs:97-99,130-132` stops shrinking at a minimum dimension and accepts an over-budget tile. It neither accounts for live tensor/weight allocations nor probes an actual graph memory budget.

Reference: `core/unet_filter.cpp:300-305` separates max pixels from memory bytes and calls buildModel(memory budget); at321-324 it attempts an unconstrained minimal model when it cannot split further. Therefore upstream itself has a documented-by-code lower limit rather than an absolute allocation cap; Rust's estimate introduces a further semantic gap.

The tests acknowledge this lower limit (`tests/api_surface.rs:219-223`), so do not claim that cap violation alone is a newly discovered upstream regression. Proposed solution: report achieved plan/estimated or measured bytes, explicitly describe best-effort behavior, and implement backend-informed allocation planning where possible. Keep quality unchanged. This is resource semantics, not demonstrated image noise.

### G6 — Low: documentation and dependency residue obscure the actual architecture

- `image_tensor.rs:3-7` claims host roundtrip/CPU runner for tensor inputs, whereas active tensor dispatch is `filters/rt.rs:813-827`. `image_tensor.rs:69-73` mentions upcoming phases after they have been implemented.
- `filter.rs:12-14` says Balanced has the same width as High and is reserved, while quality routing already selects base vs large (`registry.rs:101-103`; reference `core/unet_filter.cpp:450-454`).
- Registry docs attribute feature validation to checkParams (`registry.rs:52-53`), but reference checks those combinations in getWeights at422-434. Rust uses InvalidArgument where upstream uses InvalidOperation; compatibility is not requested, but the claimed exact parity is inaccurate.
- `device.rs:30-33` always returns Ok and deferred device initialization cannot surface through the advertised Result. `error.rs:41-45,62-71` claims failure cases not constructed anywhere in the searched crate.
- `crates/oidn-rs/Cargo.toml:54-55` declares tracing/memmap2 with no source references in the crate; it uses log instead. Do not delete public error variants or helpers merely because internal construction is absent: they are public API/future functionality. Verify dependency intent in the approved implementation pass.
- The physical inventory contains23 TZA blobs, matching `weights.rs:3`, with embedding arms for all23; older AGENTS architecture says24. No missing embedded arm was found.

### G7 — Medium: mutable tensor execution invalidates the cached model every pass

Rust: tensor setters compute needs_invalidate from slot presence at `filters/rt.rs:329-333,340-344,351-355`. Successful tensor execution clears all input slots at `filters/rt.rs:839-841`, so feeding the next same-shape frame makes each setter mark committed=false and `execute` commits again at799-800. This defeats the shape-cache promise in `filters/rt.rs:364-366`; `CommittedRtFilter::execute_tensors` at650-695 already provides the intended immutable model reuse path. There is no corresponding required rebuild in reference `core/unet_filter.cpp:170-251` execute, which submits the existing graph.

Proposed solution: cache the committed input signature independently of per-pass tensor handles; use the existing immutable committed artifact representation for both frontends. Slot consumption must not erase topology/shape identity. Verify repeated same-layout frames perform one model load while presence/dim changes rebuild or return explicit errors.

Double-check requested by orchestrator: the legacy example benchmark does reuse the model. `take_output` at403-405 leaves last_committed_dims intact, `allocate_output` at373-386 compares that signature, and `commit` at792-793 saves it. Consequently `examples/bench.rs:403-417` does not rebuild solely because it reallocates the same legacy output.

## Dataflow and trust boundaries

```text
Typed floats / public byte descriptor / Tensor<4>
              |
              v
Image geometry + layout validation [currently incomplete]
              |
     feature flags + quality
              |
              v
Registry base route -> ordered candidates
              |
 custom blob / embedded / disk [currently multiple policies]
              |
              v
TZA parse -> actual network topology
              |
              +-> RF metadata -> tile plan [Large currently uses Base RF]
              |
              v
legacy HWC -> CHW upload ----+---- tensor-native NCHW
                            |
                     common tile runner
                            |
                crop using output_src_in_tile
                            |
                  stitch at output_dst
                            |
           NCHW output / HWC host readback
                            |
            validated typed/byte output write
```

## Coverage and next-pass checklist

- [x] Read every owned production file and Cargo manifest.
- [x] Inspect image format broadcasting/collapse for all1/2/3-channel f16/f32 formats.
- [x] Verify broadcasting against reference get3/set3 (`image_accessor.h:30-51,56-91`): correct.
- [x] Compare tile job coordinates against reference `unet_filter.cpp:201-231`: same source/crop/destination formulas for valid normal parameters.
- [x] Compare RF/alignment/constants and quality routing to official v2.4.1.
- [x] Search TODO/FIXME/unimplemented/dead-code annotations: none in owned production files; unfinished-phase docs and future error cases identified above.
- [x] Inspect helper callsites before declaring anything dead; no public function removed.
- [x] Audit all23 weight stems against embedding table.
- [x] Record architecture, evidence, limitations and proposed unified fixes.
- [ ] Obtain user's actual input/output render and renderer flags/model key; determine spatial noise signature.
- [ ] Approval pass: unify topology/RF and all weight policies, centralize validated image/tensor boundaries.
- [ ] Separate verification pass: Base/Small/Large single/tiled equivalence; randomized rectangle coverage rather than area sums alone; both horizontal and vertical seam-local metrics.
- [ ] Separate verification pass: typed and unaligned/padded/short byte buffers, malformed dims/channels/batch, zero/oversized geometry, resolver I/O failures and source precedence.
- [ ] Review public error/device guarantees and stale documentation with documentation owner.
