# AGENTS.md — oidn-rs architecture notes for orchestrators

The current repaired source map is at the end of this document and tracked in [plan4.md](plan4.md). The following source map describes the historical static audit on 2026-10-02, baseline `bebbfc5`. Historical May 2026 ownership and topology notes remain below. Confirmed open defects and implementation proposals are tracked in [plan2.md](plan2.md); diagrams are in [DIAGRAMS.md](DIAGRAMS.md). No production code was changed and no builds/tests were run in this audit.

---

## Crate layout

```
oidn-rs/                   workspace root
├── crates/
│   ├── oidn-tza/          .tza weight-archive parser (host-only, no Burn)
│   ├── oidn-model/        Burn U-Net (base + large variants) + loader
│   ├── oidn-rs/           public façade: device, filter, color, autoexp, tile, gpu_ops, image
│   └── oidn-cli/          thin binary: `denoise`, `bench`, `probe` subcommands
├── data/weights/          .tza blobs (23 models)
├── tests/                 (currently empty; integration tests live in crates/*/tests)
└── bughunt/                this audit's per-agent reports
```

Historical May 2026 audit ownership (report files are absent from the current tree; retrieve them from git `7fc4eff`):

- `oidn-tza` — Mendeleev (`bughunt/mendeleev_tza.md`)
- `oidn-model` + `oidn-model/loader.rs` — Landau (`bughunt/landau_unet.md`)
- `crates/oidn-rs/src/color.rs`, `autoexposure.rs` — Kapitsa (`bughunt/kapitsa_color.md`)
- `crates/oidn-rs/src/image.rs`, `tile.rs`, `image_tensor.rs` — Pavlov (`bughunt/pavlov_tile.md`)
- `crates/oidn-rs/src/filters/*.rs` — Sechenov (`bughunt/sechenov_filter.md`)
- `crates/oidn-rs/src/gpu_ops.rs`, `filters/unet_runner.rs` — Ioffe (`bughunt/ioffe_gpu_ops.md`)
- `crates/oidn-rs/src/{lib,device,filter,error,registry,weights,prelude}.rs` — Vavilov (`bughunt/vavilov_api.md`)
- `crates/oidn-cli/**`, `examples/bench.rs`, integration tests — Kurchatov (`bughunt/kurchatov_cli.md`)

---

## Historical baseline high-level dataflow

Burn 0.22 uses dynamic device dispatch: `Device` selects CPU/WGPU, and tensors are `Tensor<4>`, without a backend type parameter (`device.rs:3-7,30-33`; `unet_runner.rs:14,41-54`). Explicit CPU selection is `Device::ndarray()`. `WgpuDevice::new()` creates a deferred handle, not a verified initialized adapter.

```text
flags + input roles + quality + weight source + dimensions
                         |
                 RT build_commit_artifacts (rt.rs:496)
                   /                    \
    mutable RtFilter::commit        commit_tensor_model
       (rt.rs:728 onward)           -> CommittedRtFilter
                   \                    /
         registry/override -> parse -> construct/load Net
                         |
               tile::plan (currently RF_BASE)
                         |
legacy Image HWC ---------------- tensor-native NCHW
    decode -> CHW upload                 |
              \                         /
                 run_tensors (unet_runner.rs:41)
                    | sanitize / exposure
                    | per-tile pack -> forward -> postprocess
                    | crop output_src_in_tile -> stitch output_dst
              +-----+----------------------+
              |                            |
          NCHW Tensor                CHW readback -> HWC
                                    -> ImageMut / take_output
```

RT mutable/immutable mode and shape checks currently differ; custom weights can bypass registry validation. See `pipeline.md` P8/P9, `geometry_api.md` G7.

```text
RtLightmapFilter::commit (rtlightmap.rs:218-275)
    -> directional ? rtlightmap_dir : rtlightmap_hdr
    -> override OR its own disk lookup -> parse/load Base -> tile plan
    -> execute (rtlightmap.rs:294-319) -> host run -> shared run_tensors
    -> output image
       [directional selects Linear/non-HDR, but signed mode is not wired]
```

Tensor input stays on its device through the normal numerical path, except exposure scalar readbacks. Enabled tensor diagnostics can read back complete tensors (`unet_runner.rs:94,366-378`). The host adapter necessarily uploads/downloads image values (`:292-318`).

Autoexposure executes for HDR color when no explicit input scale is supplied (`unet_runner.rs:90-100`); both host and tensor frontends use tensor exposure. Current floor-pooling excludes partial edge cells and returns unity for either dimension below 16 (`autoexposure.rs:152-176`).

---

## Historical baseline codepath: tile loop with open defects

```text
for job in plan.jobs:
    source rectangle -> zero_pad(raw tile)
      color  -> preprocess_input(scale, hdr, snorm=false, transfer)
      albedo -> clamp(0,1)
      normal -> clamp(-1,1) -> *0.5+0.5
    concat color | albedo | normal -> net.forward
    postprocess_color(..., snorm=false)
    crop output_src_in_tile -> slice_assign output_dst
```

Source: `unet_runner.rs:144-240`. This is the actual path, not a claim of full reference parity. Normal remapping changes raw zero padding into 0.5 (`:161-199`), unlike reference `devices/gpu/gpu_input_process.h:90-118`. Proposed approved path: slice -> preprocess valid source by primary/auxiliary role -> place into a zero tile. Signed primary mode must reach both preprocessing and postprocessing (`pipeline.md` P1-P4).

Tile constants are `RF_BASE=174`, `RF_LARGE=202`, alignment16, default max pixels2160² (`tile.rs:9-19`; reference `core/unet_filter.h:35-37`). Intended overlap is96/112px respectively. RT currently always passes BASE RF (`rt.rs:578-584`), even for Large; reference selects RF from topology (`core/unet_filter.cpp:263-268`). Rectangle formulas match reference for valid ordinary parameters; this does not establish Large overlap or public-input validation parity.

---

## Historical baseline codepath: weight resolution

```text
registry::select_rt -> ModelKey -> quality candidates
      |                    |
      |             public weights::resolve (embedded-first; hides I/O errors)
      |                    -> CLI drops stem -> .weights(bytes)
      v
RT artifacts: explicit bytes OR separate disk candidate loop
      -> parse
      -> variant from stem OR tensor names
      -> fixed-width UNet / UNetLarge -> load parameters
Lightmap: another disk lookup -> parse/load Base
```

Anchors: `registry.rs:101-103`, `weights.rs:136-152`, `rt.rs:504-563`, `rtlightmap.rs:246-257`, CLI `main.rs:267-283,342-344`. Ordinary CLI Fast selects Small bytes but override detection distinguishes only Base/Large (`variants.rs:27-33`), causing shape mismatch. XL widths remain exposed but are not selected correctly by that route. No feature is approved for removal.

High tries `_large` then base; Fast tries `_small` then base; Balanced uses base. Lightmap uses one model per mode, consistent with reference `rtlightmap_filter.cpp:19-20`. A Large suffix is unavailable for ordinary color-only/noisy-aux models in the shipped 23-archive inventory; High does not universally mean Large.

Base stems by valid role: color HDR/LDR -> `rt_hdr`/`rt_ldr`; add `_alb` or `_alb_nrm` for noisy auxiliary inputs; clean albedo+normal -> `_calb_cnrm`; albedo-only -> `rt_alb`; normal-only -> `rt_nrm`; lightmap -> `rtlightmap_hdr`/`rtlightmap_dir`. Selection names do not establish transfer semantics: albedo-only currently has a confirmed primary-transfer defect (`pipeline.md` P2).

Proposed single source: existing resolver with explicit source policy -> resolved bytes/stem/provenance -> validated archive schema -> model descriptor (topology,widths,RF) -> construct/load and tile plan. Filename is selection metadata; validated tensors determine executable dimensions.

---

## U-Net topology (verified against OIDN v2.4.1)

```
   input
     │
   enc_conv0 ─ ReLU
     │
   enc_conv1 ─ ReLU
     │
    pool 2×2 ──── pool1 ────────────────────────────────────────────┐
     │                                                              │
   enc_conv2 ─ ReLU                                                 │
     │                                                              │
    pool 2×2 ──── pool2 ──────────────────────────────────────┐     │
     │                                                        │     │
   enc_conv3 ─ ReLU                                           │     │
     │                                                        │     │
    pool 2×2 ──── pool3 ────────────────────────────────┐     │     │
     │                                                  │     │     │
   enc_conv4 ─ ReLU                                     │     │     │
     │                                                  │     │     │
    pool 2×2                                            │     │     │
     │                                                  │     │     │
   enc_conv5a ─ ReLU                                    │     │     │
   enc_conv5b ─ ReLU ─ upsample 2× (nearest)            │     │     │
     │                                                  │     │     │
   concat ◄────────────────────────────────────────────┘     │     │
     │                                                        │     │
   dec_conv4a ─ ReLU                                          │     │
   dec_conv4b ─ ReLU ─ upsample 2×                            │     │
     │                                                        │     │
   concat ◄──────────────────────────────────────────────────┘     │
     │                                                              │
   dec_conv3a ─ ReLU                                                │
   dec_conv3b ─ ReLU ─ upsample 2×                                  │
     │                                                              │
   concat ◄────────────────────────────────────────────────────────┘
     │
   dec_conv2a ─ ReLU
   dec_conv2b ─ ReLU ─ upsample 2×
     │
   concat ◄── input
     │
   dec_conv1a ─ ReLU
   dec_conv1b ─ ReLU
     │
   dec_conv0  ─ ReLU
     │
   output
```

Base ordering is verified at `unet.rs:104-142` against reference `core/unet_filter.cpp:468-497`. Large ordering is verified at `unet_large.rs:147-187` against reference `:500-530`; its final `dec_conv1c` includes ReLU at Rust187/reference528. Width presets for Base/Small and Large/XL match reference `training/model.py:61-103,167-208`; loading Small/XL through custom routes remains defective.

---

## Coordinate / tensor conventions

- Host images support1/2/3-channel f16/f32 typed/byte representations and row strides (`image.rs:60-75,170-181,275-307`). Arbitrary pixel stride is not represented. Descriptor validation is incomplete; see `geometry_api.md` G2.
- Burn numerical input contract is `[1,3,H,W]` per role; concatenated model input has3/6/9 channels. Public tensor/layout helpers do not consistently validate batch/channels in release; see `pipeline.md` P8.
- TZA supports f16/f32 payloads; loaders convert into f32 at `oidn-model/src/loader.rs:64-68,93-97`. Public byte decoding needs fallible length/alignment/endian validation (`model_weights.md` MW3/MW4).
- Tile job carries three rectangles (`tile.rs`) and two alignment offsets: `input` (src region), `output_src_in_tile` (which subrect of the tile-output tensor to keep), `output_dst` (where to write in the user image), plus `align_offset_x/y`.

---

## Historical May report names and current audit links

- TZA / weights: see `bughunt/mendeleev_tza.md`
- U-Net architecture: see `bughunt/landau_unet.md`
- Color transforms / autoexposure: see `bughunt/kapitsa_color.md`
- Tile / image / buffer: see `bughunt/pavlov_tile.md`
- Filter pipeline (RT, RTLightmap): see `bughunt/sechenov_filter.md`
- GPU ops + input/output process: see `bughunt/ioffe_gpu_ops.md`
- Public API surface: see `bughunt/vavilov_api.md`
- CLI + integration tests: see `bughunt/kurchatov_cli.md`
- Historical plan1: present in git `7fc4eff`, deleted in `b999415`; current approval plan: [plan2.md](plan2.md).
- Current detailed audits: [pipeline](bughunt/pipeline.md), [model/weights](bughunt/model_weights.md), [geometry/API](bughunt/geometry_api.md), [CLI/verification](bughunt/cli_verification.md).

---

## Conventions for future agents

1. Cite source claims with Rust and reference `file:line` pairs when a counterpart exists. The historical `_ref/oidn` and `C:/projects/projects.rust.cg.offload/oidn` checkout are absent. Original audit citations use official OIDN v2.4.1 files under `bughunt/reference-v2.4.1` and upstream tag URLs. The user subsequently supplied `D:/Projects/vfx.ref/oidn`, clean tag v2.5.0 at `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7`; label its evidence as LOCAL2.5 and do not relabel the pinned v2.4.1 citations. Do not infer a reference version from the historical `reference-v2.3.3` folder name.
2. Do not run tests or builds during audit work. Audit is a code-reading task; verification belongs in a separate pass.
3. Write reports under `bughunt/<agent_name>_<area>.md` so the orchestrator can survive context compaction.
4. Prefer parallelism: dispatch up to 8 agents simultaneously, each on a disjoint slice.

## Local reference follow-up

The supplied `D:/Projects/vfx.ref/oidn` checkout is clean v2.5.0 at `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7`. Read [local pipeline](bughunt/local_pipeline.md) and [local model/weights](bughunt/local_model_weights.md) for paired current Rust/LOCAL2.5 evidence. All P1–P11 remain applicable. Logical model topology/widths and TZA format remain unchanged from v2.4.1; graph fusion APIs changed without a discovered logical-order difference. This does not prove numerical parity.

Sanitation is an additional contract difference: Rust `gpu_ops.rs:29-32,51-55` replaces all nonfinite values before scale; LOCAL2.5 `core/math.h:77-79`, `devices/gpu/gpu_input_process.h:41-54` replaces only NaN after scale, then clamps infinities. Preserve explicit policy and test NaN/+Inf/-Inf rather than claim exact sanitation parity. Native weights are unavailable locally: `.gitmodules:1-3` describes an uninitialized external submodule at `28883d1769d5930e13cf7f1676dd852bd81ed9e7`. All 23 Rust headers and SHA256 values were recorded, but native byte equality/runtime quality are unverified. Original reference citations elsewhere in this file remain v2.4.1.

## Audit boundary and tool fallback

The initial static audit did not reproduce the user's noise observation or edit Rust symbols. Its confirmed source defects remain mode-specific candidates, not an assigned root cause. GitNexus/fetch MCP were unavailable during that original pass; direct filesystem MCP source/callsite reads and read-only history were the documented fallback. Later, the user explicitly authorized renderer testing and dependency updates in squarebob-rs. That separate verification pass supersedes the audit-only test/build restriction for the authorized diagnostic work; production repairs still require report approval. Read [plan2.md](plan2.md), [the bridge report](bughunt/squarebob_bridge.md), and [Squarebob plan16](../squarebob-rs/plan16.md). When graph tools are available, follow freshness/impact/reindex rules. Preserve other contributors' edits and existing tool-failure reports.

## Squarebob renderer follow-up — 2026-10-02

[Squarebob plan16](../squarebob-rs/plan16.md) records the SSH dependency maintenance, passed bridge mode unit test, and initial successful diagnostic GPU results. Frozen input repeated at 256 SPP and fixed-clamp SPP 1/256 were identical; adaptive SPP 1/256 changed by max absolute RGB 8.448264122 on RTX 3080 Ti/Vulkan. This synthetic result confirms input-policy influence, not reproduction of the user's scene. Another 32 fixed-clamp runs at 256 SPP were identical; workspace check passed with empty stdout/stderr. Actual squarebob binary linking also passed with empty stdout/stderr, as recorded in plan16. These results cover the frozen pattern/device/configuration, not general race absence or scene noise. [The bridge audit](bughunt/squarebob_bridge.md) supplies exact source and pinned CubeCL evidence. The renderer uses immutable committed RT state with fresh tensors, color/HDR/noisy auxiliaries, and RGB output; auxiliary-only, lightmap, and scalar-output findings are outside this bridge's current modes.

```text
Squarebob PT normalized HDR color + AOV sums/counts + current SPP
 -> shared-device external input copies
 -> trim row padding / sample-dependent luminance clamp
 -> AOV RGB/max(W,1) / NCHW RGB
 -> resolved weight bytes -> committed RT model -> fresh execute_tensors
 -> env scale > Physical-camera scale > Manual autoexposure
 -> HWC RGBA(alpha1) -> CubeCL get_resource(flush + allocation pin)
 -> separate result_texture -> poll -> result_view
 -> display composite to render_view
```

The bridge's default app clamp rises from 6 at 128 SPP to 10 at 256 SPP (`../squarebob-rs/crates/pt-denoise-oidn/src/lib.rs:457-468`; defaults `../squarebob-rs/crates/render-shared/src/lib.rs:1128-1135`). It changes highlight input and requires a frozen-input comparison before causal attribution. Earlier periodic success also blocks final-denoise scheduling (`../squarebob-rs/src/app/treemap_view.rs:1510-1522,1609-1612`); target 300/interval 128 can leave a 256 SPP snapshot.

Normal shared-device output-copy ordering is supported by checked CubeCL source, not contradicted by a reproduced race. The historical buffer-race comment is not proof of a present issue. This bridge writes separate result texture and display target (`../squarebob-rs/crates/pt-denoise-oidn/src/lib.rs:649-656`; `../squarebob-rs/crates/render-3d/src/lib.rs:1217-1222`), with no denoised-to-raw feedback in the inspected integration path. Actual GPU measurements and renderer symptom reproduction have separate gates in plan16.

## Native numerical and PQ follow-up — 2026-10-02

Current evidence continues [OIDN plan3](../oidn-rs/plan3.md), [Squarebob plan17](../squarebob-rs/plan17.md), [native runtime](../oidn-rs/bughunt/native_runtime.md) and [Astra numerics](../oidn-rs/bughunt/astra_numerics.md). Earlier audit/no-runtime statements describe their historical pass; later diagnostic testing was explicitly authorized. PQ display production changes have separate explicit authorization; unrelated denoiser/scheduling repairs remain unapproved proposals.

All 23 local Rust archive sizes/SHA256 values now match isolated pinned native archives at28883d1769d5930e13cf7f1676dd852bd81ed9e7. This supersedes the earlier asset-availability limitation without modifying the supplied native checkout. 90/90 synthetic nativeCPU/CUDA/RustWGPU executions passed with finite output. Explicit-scale color-only Rust/nativeCPU max absolute error0.0003814697265625; aligned full-AOV32x32 max0.000213623046875. Odd/tiny autoexposure and unaligned normal-padding comparisons differ substantially. Scope is synthetic CLI host inputs, not all modes or the user's scene.

The checker's adaptive-clamp spatial-SD change restores fixture contrast after clipping, so it is not a noise-only metric. Native CUDA half storage and quality-dependent accumulator policy differ from checked Burn f32 Direct convolution; default fusion/autotune was not enabled in Astra's checked target. The actual progressive highlight-noise cause remains unproven.

The current Squarebob display source uses `src/display_host.rs` plus canonical `egui-display` (`06acf66506583e2cef07450ac304d8ae0414b59d`). Actual negotiated output HDR/white is propagated before app UI/rendering; ColorSettings runtime fields select OCIO display-reference decoding and Rec.709 light relative to reference white. The shared presenter owns final SDR/PQ/HLG/scRGB encoding once. An unsupported requested PQ mode falls back with an explicit actual-state error. Initial workspace check passed, while final color/shader/host/binary/actual-window validation is still in progress; see plan17.

```text
scene-linear renderer / OIDN result
 -> exposure + OCIO selected view/look/LUT
 -> display-reference XYZ D65 -> Rec.709 display light / reference white
 -> float extended-sRGB GUI/renderer canvas
 -> shared egui-display PresentPass
 -> negotiated SDR/PQ/HLG/scRGB surface
```

Anchors in Squarebob: `src/main.rs:187`, `src/display_host.rs:212-231,309-310,389-455,520-584`, `src/app/mod.rs:59-64`, `crates/color-pipeline/src/lib.rs:193-198,772-808,875-887`. Canonical presenter anchors: `egui-widgets-rs/crates/egui-display/src/present.rs:25,127-149,202-215,612,766`. Compilation and float readback do not certify an actual HDR monitor surface or physical luminance.

## CPU/PQ verification closure — 2026-10-02

This closure supersedes earlier pending CPU/PQ test statements and qualifies earlier no-feedback conclusions as GPU-path evidence. Historical pass results and source anchors remain preserved above.

The user explicitly authorized the old CPU display repair and push to main in addition to PQ integration. Unrelated denoiser/scheduling proposals remain unapproved. [Squarebob plan17](../squarebob-rs/plan17.md), [Astra post-fix review](bughunt/astra_pq_review.md), and [OIIO CPU review](bughunt/oiio_cpu_display.md) record the checked contracts and results.

The historical no-feedback conclusion described the inspected GPU bridge; CPU display had existing raw-buffer feedback/exposure-order and denoised-view defects. The shared composition repair reads raw PT/OIDN, applies CPU exposure before OCIO, and writes a separate reusable display scratch. Source: Squarebob `crates/render-3d/src/lib.rs:1190-1236`, `crates/pt-megakernel/src/compute.rs:5383-5495`, and both callers `src/app/treemap_view.rs:699-708,1193-1205`. No matching defect was found in the scoped OCIO/OIIO library inspection.

Color6/6, host3/3, render-core3 plus GPU1, extended-sRGB/full-PresentPass signal and CPU immutability/order probes passed. The signal probe covered gray plus six saturated-primary cases (max PQ primary error0.000542;203nits0.580652;1000nits0.751720). Final locked workspace/all-target check passed9.381s; actual binary build passed33.406s, both empty logs. CPU probe preserved PT/external raw bytes and repeated output, with exposure2 before OCIO matching the processor oracle. Actual GUI retry and confirmed remote push remain pending; these measurements do not certify physical monitor luminance. Context7 supplied official winit0.30 changelog evidence; unavailable/timed-out GitNexus requests used direct-source fallback.

## Final window and persistence verification — 2026-10-02

This supersedes the preceding pending GUI retry and final-check status. Actual PBR, PT GPU, PT CPU and newly saved CPU-state restart windows completed successfully. Logs confirm actual HDR10(PQ), `Rgb10a2Unorm/Bt2100Pq`; full UI capture `squarebob_pq_ui_controls.png` was inspected by the root reviewer. CPU/GPU controls are visible; all five output modes, reference-white Auto/manual and HLG peak controls exist in the checked host. Auto white is240nits. OS-reported peak603nits/full-frame150nits/headroom2.51/10bits are metadata, not physical luminance measurements. Restart preserved CPU color path, ISO240, f/1, shutter1 and effective scale2.

App state now uses a typed RON payload inside the existing outer storage map, while DisplayPrefs stays a JSON string. `Squarebob src/app/state.rs:142-149` accepts RON plus valid legacy JSON; `Squarebob src/app/mod.rs:782-834` saves the complete typed state. `Squarebob src/app/persistence_tests.rs:5-43` verifies full-state roundtrip including nonfinite dock rectangle sentinels, finite floating-window position/size, topology and CPU/camera settings. Previously corrupted JSON null-rectangle snapshots report decode failure; no geometry is guessed or stripped to migrate them.

Final binary unit suite:34/34 passed in0.13s (`squarebob_pq_bin_tests.stdout.log`). Final locked workspace/all-target check passed in6.313s with empty logs; binary build passed in16.140s with empty logs. Earlier measurements remain historical. Final Clippy exited0 in10.685s, with464 stderr lines of warnings (`pq_final_clippy.stderr.log`); this is not a warning-free result. The official Rust API checklist was consulted for getter naming, validation, meaningful errors and intermediate results; Context7 supplied official winit0.30 documentation.

Manual clipboard/accessibility, multi-monitor transitions, all alternative output modes and physical luminance remain unverified. Actual user-scene progressive noise remains unresolved. Production work and review are complete in the verified scope. Main publication is explicitly authorized; its confirmed commit/remote receipt is recorded separately in OIDN plan3.

## Authorized systemic OIDN repair — 2026-10-02

The user explicitly instructed systematic implementation of the outstanding OIDN fixes. [plan4.md](plan4.md) is the current issue ledger and resumption entry point. Earlier approval-pending/audit-only language above remains historical; it does not block these approved source repairs and their required verification. Final source closure and new numerical receipts are still pending. Preserve all concurrent contributors' edits.

```text
validated roles/modes + host/tensor geometry
 -> canonical resolution -> fallible archive schema -> descriptor(widths,RF)
 -> shared committed model and validated tile plan
 -> fresh pass handles -> balanced exposure
 -> valid-source preprocessing -> zero tile -> shared forward
 -> inverse transfer -> scalar destination mean -> signed decode/scale
 -> crop/stitch -> device tensor or validated host output
```

This was the approved repair contract before implementation receipts. Historical source anchors above describe their stated baseline. The updated current-source map and bounded post-fix results follow below; the actual progressive user-scene cause remains unresolved.

## Current repaired source map — 2026-10-02

This section supersedes the historical defect diagrams and source claims above for the current working tree. Source implementation is present; the final regression/feature/lint gates remain tracked in [plan4.md](plan4.md). Native frozen comparisons now pass within the recorded tolerance. They do not reproduce the user's progressive scene.

```text
RT roles/modes + host image or [1,3,H,W] tensor + output geometry
 -> validate_execution / validate_rt / image descriptor validation
 -> override bytes OR weights::resolve(policy, quality)
 -> TZA checked parse -> ModelDescriptor(widths, topology, RF)
 -> shared build_commit_artifacts -> Net + validated TilePlan
                 |                         |
      persistent input signature          | RF174/202, overlap96/112
                 |                         |
      fresh per-frame handles ------------+
 -> optional strict nonfinite sanitation -> HDR exposure or explicit scale
 -> slice VALID source -> preprocess primary/auxiliary -> insert into ZERO tile
 -> concatenate roles -> shared Net::forward
 -> NaN0 / positive clamp -> inverse transfer -> scalar mean if required
 -> signed decode / LDR clamp -> output scale
 -> crop output_src_in_tile -> stitch output_dst
 -> NCHW tensor OR validated CHW/HWC host adapter
```

Lightmap uses the same model construction and runner: Log/HDR for ordinary lightmaps, Linear/signed for directional data. It resolves its single model with Balanced candidates; the quality field does not select an unsupported alternate lightmap topology. RT custom weights still support validated color+normal layouts even when no built-in archive exists for that role combination.

| Responsibility | Current source anchors |
| --- | --- |
| Shared geometry/scale and model/tile construction | `crates/oidn-rs/src/filters/mod.rs:16,89` |
| Source priority, quality fallback, provenance, I/O propagation | `crates/oidn-rs/src/weights.rs:116,133,142` |
| Schema-derived channels/topology/RF and mutation-safe loading | `crates/oidn-model/src/descriptor.rs:30,98,119,141` |
| Archive ownership and fallible little-endian decoding | `crates/oidn-tza/src/parser.rs:109`, `types.rs:102,114,127` |
| RT commit entry and persistent frame signatures | `crates/oidn-rs/src/filters/rt.rs:208,418,492,730` |
| Exact runtime tensor/model validation | `crates/oidn-rs/src/filters/unet_runner.rs:33,72,88` |
| Balanced-bin CPU oracle and tensor exposure | `crates/oidn-rs/src/autoexposure.rs:78,85,137` |
| Transform valid source before padding | `crates/oidn-rs/src/filters/unet_runner.rs:207` |
| Native primary/postprocessing order | `crates/oidn-rs/src/gpu_ops.rs:42,68` |
| Fallible host descriptors, layouts and exact tile partition | `crates/oidn-rs/src/image.rs:89,230,338`, `image_tensor.rs:12,44`, `tile.rs:85,230,248` |
| CLI CPU selection and family validation | `crates/oidn-cli/src/main.rs:27,33,266` |

API migration: image setters/output allocation, generic image constructors, image/layout/exposure helpers, public tensor decoding and `tile::total_output_pixels(&plan)` return `Result`. Typed convenience image constructors retain their asserting contract. TZA payloads use shared immutable `Bytes`; use `.into()` when constructing payloads from a `Vec<u8>`. Direct weight resolution returns `Result<Option<ResolvedWeights>>` and takes `SourcePolicy`. Shared runner semantics travel in `RunOptions`; immutable RT commit/execute remains the normal renderer route.

A supplied nonempty filter weights directory defaults to DiskFirst; an empty directory defaults to EmbeddedFirst. Source precedence dominates quality fallback. The default stronger `nan_to_zero(true)` policy replaces all nonfinite input before exposure; disabling it retains mandatory native NaN-only sanitation after scale and lets infinities reach range clamping. Memory budgets are best-effort logical estimates, with explicit minimum-tile fallback; backend workspace/allocator overhead is additional.

Final workspace/all-target tests with `embed-all,acescg-autoexposure` passed 96 tests, with 14 GPU tests intentionally ignored. The explicit required-GPU lane then passed all 14, including a 3072x3072 actual multi-tile run and Large High RF202 full/two-tile comparison. Strict Clippy and strict rustdoc passed. Earlier focused model/TZA21, CLI6+3, bootstrap4, publicCPU6 and coreCPU51 receipts remain in plan4. The tiny-dimension pad-equality validator regression was caught and corrected against LOCAL2.5, then the complete CPU suite passed. The post-fix matrix completed 108/108 finite runs; Large completed 6/6 with actual two-tile RF202 planning and tiled/full maximum zero for the 769x16 fixture. Odd/tiny exposure and unaligned-AOV maximum errors fell to 0.000141144, 0.0000286102 and 0.000125885 respectively. Exact receipts, numerical-pass binary hash and limitations are in plan4 and [postfix receipt](bughunt/native-verification/postfix/receipt.json). These are bounded numerical measurements, not general quality or scene-noise claims.

The repaired Squarebob bridge resolves weights only at canonical commit (`../squarebob-rs/crates/pt-denoise-oidn/src/lib.rs:528,533`) and updates scale at runtime (`:559`), excluding scale from its role/quality/geometry cache key (`:516`). A temporary isolated local Git-patch workspace/all-target check and fresh GPU probe passed; repeat32 and fixed-clamp SPP1/256 differences were zero. The original renderer lock was preserved, the temporary worktree removed, and no permanent dependency bump or OIDN publication is claimed. See plan4 for exact hashes, logs and the existing Vulkan-layer warning.
