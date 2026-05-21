# AGENTS.md — oidn-rs architecture notes for orchestrators

This file documents the crates, dataflow and codepaths discovered during the parity audit (2026-05-21). It is the bird's-eye map an LLM agent needs before touching the codebase.

For per-issue findings see `bughunt/*.md`. For the prioritised fix list see `plan1.md`.

---

## Crate layout

```
oidn-rs/                   workspace root
├── crates/
│   ├── oidn-tza/          .tza weight-archive parser (host-only, no Burn)
│   ├── oidn-model/        Burn U-Net (base + large variants) + loader
│   ├── oidn-rs/           public façade: device, filter, color, autoexp, tile, gpu_ops, image
│   └── oidn-cli/          thin binary: `denoise`, `bench`, `probe` subcommands
├── data/weights/          .tza blobs (24 models)
├── tests/                 (workspace-level e2e tests)
└── bughunt/                this audit's per-agent reports
```

Audit ownership (one agent per bullet):

- `oidn-tza` — Mendeleev (`bughunt/mendeleev_tza.md`)
- `oidn-model` + `oidn-model/loader.rs` — Landau (`bughunt/landau_unet.md`)
- `crates/oidn-rs/src/color.rs`, `autoexposure.rs` — Kapitsa (`bughunt/kapitsa_color.md`)
- `crates/oidn-rs/src/image.rs`, `tile.rs`, `image_tensor.rs` — Pavlov (`bughunt/pavlov_tile.md`)
- `crates/oidn-rs/src/filters/*.rs` — Sechenov (`bughunt/sechenov_filter.md`)
- `crates/oidn-rs/src/gpu_ops.rs`, `filters/unet_runner.rs` — Ioffe (`bughunt/ioffe_gpu_ops.md`)
- `crates/oidn-rs/src/{lib,device,filter,error,registry,weights,prelude}.rs` — Vavilov (`bughunt/vavilov_api.md`)
- `crates/oidn-cli/**`, `examples/bench.rs`, integration tests — Kurchatov (`bughunt/kurchatov_cli.md`)

---

## High-level dataflow

```
                        ┌───────────────────────────────────────────────────────┐
                        │                       USER INPUT                      │
                        │   color image, [albedo], [normal], hdr/srgb/cleanAux   │
                        └────────────────────────────┬──────────────────────────┘
                                                     │
                                                     ▼
        ┌────────────────────────────────────────────────────────────────────┐
        │  RtFilter::builder(device, weights_dir)                            │
        │      .hdr(_).srgb(_).clean_aux(_).quality(_).input_scale(_)        │
        │      .set_color(_).set_albedo(_).set_normal(_)                     │
        │      .commit()  →  CommittedRtFilter<B>                            │
        └────────────────────────────┬───────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────────┐
              │  registry::select_rt(color, alb, nrm, hdr, srgb, │
              │                     directional, clean_aux)      │
              │  → ModelKey   ("rt_hdr_alb_nrm_large", ...)      │
              └──────────────────────────────────────────────────┘
                                     │
        ┌────────────────────────────┴───────────────────────────┐
        ▼                                                        ▼
  weights::resolve(key, dir, embed)              tile::plan(image_dims, rf)
  → bytes (mmap or embed)                          → TilePlan { jobs, tile_h, tile_w }
        │                                                        │
        ▼                                                        │
  oidn_tza::parse(bytes)                                         │
  → TensorMap                                                    │
        │                                                        │
        ▼                                                        │
  oidn_model::Net::from_tza(&map)                                │
  → UNet | UNetLarge                                             │
        │                                                        │
        └──────────────────────────────┬─────────────────────────┘
                                       ▼
              ┌──────────────────────────────────────────────────┐
              │     filters::unet_runner::run_tensors(...)       │
              │     ┌────────────────────────────────────────┐   │
              │     │ per-tile loop:                         │   │
              │     │ 1. slice src tile                      │   │
              │     │ 2. PAD (currently reflect — BUG H1)    │   │
              │     │ 3. forward transfer (color path)       │   │
              │     │ 4. concat color|albedo|normal          │   │
              │     │ 5. net.forward(input_tensor)           │   │
              │     │ 6. inverse transfer (MISSING CLAMPS H2)│   │
              │     │ 7. crop + slice_assign into accum      │   │
              │     └────────────────────────────────────────┘   │
              └──────────────────────────────────────────────────┘
                                       │
                                       ▼
                            Owned output image (Vec<f32>, HWC)
                                       │
                                       ▼
                            User callback / take_output()
```

Autoexposure side-pass (only when `hdr && color.is_some() && input_scale.is_none()`):
```
   color tensor ──► autoexposure::compute_scale_tensor ──► f32
                                                            │
                                                            ▼ (set as state.input_scale)
                                                  used inside apply_transfer_forward
```

---

## Codepath: tile loop (matches `_ref/oidn/core/unet_filter.cpp:198-238`)

```
for job in plan.jobs:                      // H-major then W-major
    src = (color, albedo, normal) sliced to job.input (Rect)
    pad to tile_h × tile_w (zero-pad in ref; REFLECT in Rust — H1)
    input_tensor = concat(forward_transfer(color), clamp_alb, clamp_remap_nrm)
    output_tensor = net.forward(input_tensor)
    output_post = postprocess(output_tensor)   // missing clamps — H2/H3
    crop output_post by job.output_src_in_tile
    accum.slice_assign(job.output_dst, crop)
```

Tile geometry (matches ref byte-for-byte, see `pavlov_tile.md`):
- `RF_BASE = 174`, `RF_LARGE = 202`, `MIN_TILE_ALIGNMENT = 16`, `DEFAULT_MAX_TILE_PIXELS = 2160²`.
- `tile_overlap = round_up(RF/2, tile_alignment)` → 96 px (base) / 112 px (large).

---

## Codepath: weight resolution

```
RtFilterBuilder.build() → RtFilter
   .commit() →
       registry::select_rt(...) → ModelKey
       weights::resolve(key, dir, embedded) → Vec<u8>
       oidn_tza::parse(bytes) → TensorMap
       oidn_model::Net::from_tza(map, in_channels, out_channels)
           → UNet (small/base) or UNetLarge (base/XL)
       Quality::High first tries _large suffix, then base.
       Quality::Fast first tries _small, then base.
       Quality::Balanced is base only.
```

Truth table (Sechenov input-combo matrix):

| color | albedo | normal | hdr | srgb | clean_aux | weight stem (base, Balanced) |
|---|---|---|---|---|---|---|
| ✓ | – | – | ✓ | – | – | `rt_hdr` |
| ✓ | – | – | – | ✓ | – | `rt_ldr` |
| ✓ | – | – | – | – | – | `rt_ldr` |
| ✓ | ✓ | – | ✓ | – | – | `rt_hdr_alb` |
| ✓ | ✓ | – | – | – | – | `rt_ldr_alb` |
| ✓ | ✓ | ✓ | ✓ | – | – | `rt_hdr_alb_nrm` |
| ✓ | ✓ | ✓ | ✓ | – | ✓ | `rt_hdr_calb_cnrm` |
| ✓ | ✓ | ✓ | – | – | – | `rt_ldr_alb_nrm` |
| ✓ | ✓ | ✓ | – | – | ✓ | `rt_ldr_calb_cnrm` |
| – | ✓ | – | – | – | – | `rt_alb` |
| – | – | ✓ | – | – | – | `rt_nrm` |
| RTLightmap, hdr=true     | `rtlightmap_hdr` |
| RTLightmap, directional  | `rtlightmap_dir` |

Quality multiplier appended:
- High → try `<stem>_large`, fall back to `<stem>`.
- Fast → try `<stem>_small`, fall back to `<stem>`.
- Balanced → `<stem>` only.
- (lightmap quality is ignored — no `_large`/`_small` blobs in the ref either.)

---

## U-Net topology (matches `_ref/oidn/core/unet_filter.cpp:470-528`)

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
   dec_conv0  ◄── ReLU in ref runtime;  ✗ MISSING in Rust unet.rs:131 (BUG H5)
     │
   output
```

UNetLarge differs by: doubled depth per stage (`enc_conv1a/1b`, `2a/2b`, …, `5a/5b`, decoder `1a/1b/1c`). UNetLarge's `dec_conv1c` already includes ReLU (`unet_large.rs:143`).

---

## Coordinate / tensor conventions

- Image memory: HWC, row-major, `Vec<f32>`. Row stride = `width * pixel_size`. No `pixel_stride` support yet (BUG M7).
- Burn tensor layout: `[N=1, C, H, W]`. Backend chooses internal blocked layout (cubecl handles it on wgpu).
- f16 vs f32: TZA weights are f16, loader converts to f32 inside `oidn-model/src/loader.rs:57-60, 86-89` (Landau U5).
- Tile job carries five rectangles (`tile.rs`): `input` (src region), `output_src_in_tile` (which subrect of the tile-output tensor to keep), `output_dst` (where to write in the user image), plus `align_offset_x/y`.

---

## Per-agent quick links

- TZA / weights: see `bughunt/mendeleev_tza.md`
- U-Net architecture: see `bughunt/landau_unet.md`
- Color transforms / autoexposure: see `bughunt/kapitsa_color.md`
- Tile / image / buffer: see `bughunt/pavlov_tile.md`
- Filter pipeline (RT, RTLightmap): see `bughunt/sechenov_filter.md`
- GPU ops + input/output process: see `bughunt/ioffe_gpu_ops.md`
- Public API surface: see `bughunt/vavilov_api.md`
- CLI + integration tests: see `bughunt/kurchatov_cli.md`
- Consolidated plan: see `plan1.md`

---

## Conventions for future agents

1. Read `plan1.md` first to see the current open-issue inventory.
2. Cite every claim with a `file:line` pair from both the Rust source and the C++ reference (`_ref/oidn` at `C:/projects/projects.rust.cg.offload/oidn`).
3. Do not run tests or builds during audit work. Audit is a code-reading task; verification belongs in a separate pass.
4. Write reports under `bughunt/<agent_name>_<area>.md` so the orchestrator can survive context compaction.
5. Prefer parallelism: dispatch up to 8 agents simultaneously, each on a disjoint slice.
