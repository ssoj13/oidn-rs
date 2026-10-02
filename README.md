# oidn-rs

Pure Rust port of [Intel Open Image Denoise](https://www.openimagedenoise.org/)
running on [Burn](https://burn.dev/) + [wgpu](https://wgpu.rs/), targeting any
GPU vendor (NVIDIA / AMD / Intel / Apple) through a single backend.

**Status:** all 33 confirmed contract findings are repaired and verified in the
recorded CPU/feature/GPU scope; the issue ledger and exact receipts are in
[plan4.md](plan4.md). Both `UNet` (base/small)
and `UNetLarge` (large/XL) topologies are implemented. All 23 shipped weight
archives match the pinned native archives byte for byte. Historical numerical
results are in [plan3.md](plan3.md); post-repair results and their fixture/device
limits are in [plan4.md](plan4.md).

The May 2026 audit reported ~11× synthetic-noise RMSE reduction and closure
of 12 HIGH-severity findings. Those are historical measurements, not a
current claim of seamless tiling or complete native parity: the October
audit identified additional defects in [plan2.md](plan2.md). Historical
`plan1.md` is available in git `7fc4eff`.

## Workspace layout

```
crates/
├─ oidn-tza/    standalone TZA tensor archive parser (zero ML deps)
├─ oidn-model/  U-Net definitions on Burn (dynamic Device dispatch)
├─ oidn-rs/     runtime — filters, tiling, color, autoexposure
└─ oidn-cli/    command-line binary (`oidn-rs probe | denoise | bench`)
```

The original port follows the upstream
[RenderKit/oidn](https://github.com/RenderKit/oidn) tree (v2.4.1); shared
validation, descriptors, and adapters are organized around the Rust API.
Source comments cite upstream paths such as `core/tza.cpp` and
`training/model.py`; those are paths inside the Intel repository.

## Quickstart

The declared Rust dependency floor is 1.95, matching the pinned Burn dependency.
This repair pass used rustc 1.99.0; it did not test the minimum toolchain.

Trained Intel weights ship in this repo at `data/weights/*.tza` as regular
git blobs (23 model archives). A plain clone is enough — no
Git LFS setup required.

```sh
git clone https://github.com/ssoj13/oidn-rs.git
cd oidn-rs
cargo build --release --workspace

# Print the tensor list of a weights blob
cargo run -p oidn-cli --release -- probe data/weights/rt_hdr.tza

# RT requires --hdr or --ldr (mutually exclusive).
# A default build embeds no weights; supply the shipped weights directory.
cargo run -p oidn-cli --release -- denoise --hdr --weights-dir data/weights -i noisy.exr -o clean.exr
cargo run -p oidn-cli --release -- denoise --hdr --weights-dir data/weights \
    -i color.exr --albedo albedo.exr --normal normal.exr -o out.exr

# LDR PNG/JPG with sRGB encoding hint
cargo run -p oidn-cli --release -- denoise --ldr --srgb --weights-dir data/weights -i in.png -o out.png

# Reference golden formats are first-class
cargo run -p oidn-cli --release -- denoise --hdr --weights-dir data/weights -i in.pfm -o out.pfm  # f32
cargo run -p oidn-cli --release -- denoise --hdr --weights-dir data/weights -i in.phm -o out.phm  # f16

# RTLightmap directional mode
cargo run -p oidn-cli --release -- denoise --filter RTLightmap --dir --weights-dir data/weights \
    -i lightmap.exr -o clean.exr

# CPU execution uses NdArray and does not initialize a GPU.
cargo run -p oidn-cli --release -- --device cpu denoise --hdr --quality fast \
    --weights-dir data/weights -i in.pfm -o out.pfm

# Enumerate wgpu adapters
cargo run -p oidn-cli --release -- list-devices

# Benchmark on a synthetic scene
cargo run -p oidn-cli --release -- bench --weights-dir data/weights --resolution 1024x1024 --iters 10
```

## Library use

```rust
use oidn_rs::prelude::*;                    // backend-agnostic types
use oidn_rs::prelude::wgpu_prelude::*;      // WgpuDevice

let device = WgpuDevice::new()?;
let mut filter = RtFilter::builder(&device.handle, "data/weights")
    .hdr(true)
    .quality(Quality::High)
    .build();

filter.set_color(&Image::from_rgb_f32(&color, w, h))?;
filter.set_albedo(&Image::from_rgb_f32(&albedo, w, h))?; // optional
filter.allocate_output(w, h, PixelFormat::Rgb32f)?;
filter.commit()?;
filter.execute()?;

let (raw, _, _, _) = filter.take_output().unwrap();
```

The prelude is device-agnostic by default. Wgpu types live in the
`wgpu_prelude` submodule; select CPU through
`burn::tensor::Device::ndarray()` without generic backend parameters.

For lightmaps, use `RtLightmapFilter` instead (HDR Log transfer, or directional
mode with Linear transfer and signed input/output).

Image setters and output allocation return `Result`; propagate errors with `?`.
Generic byte-backed `Image::new`/`ImageMut::new`, image conversion, tensor-layout
helpers, exposure helpers, and `tile::total_output_pixels(&plan)` also return `Result`. Typed `Image::from_*`
convenience constructors retain their asserting contract. Tensor inputs must
have shape `[1, 3, H, W]` on the selected device. Explicit input scale must be
positive and finite, with a finite reciprocal.

A supplied nonempty weights directory defaults to `SourcePolicy::DiskFirst`;
an empty directory defaults to `EmbeddedFirst`. Override this with
`.weight_source(SourcePolicy::EmbeddedOnly)` or another explicit policy.
Source priority is applied before quality fallback, and I/O errors other than
`NotFound` are reported. Direct `weights::resolve` now returns
`Result<Option<ResolvedWeights>>`, preserving bytes, stem, and provenance.
Executable widths and receptive field come from validated tensors rather than
filenames, including custom Base/Small/Large/XL widths.

`max_memory_mb` is a best-effort logical planning budget. The estimate includes
parameters, whole-image buffers, and tile activations; backend workspaces and
allocator overhead are additional. If the native minimum tile exceeds the
budget, execution retains that tile and reports the achieved estimate.

Both filters default to `nan_to_zero(true)`, replacing all nonfinite input
values before exposure. Set it to `false` to retain native NaN-only sanitation
after scale; infinities then reach the native range clamp. This is an explicit
input policy, not a promise of bitwise backend equality.

`RtFilter` and `RtLightmapFilter` use Burn 0.22 dynamic device dispatch.
Use an explicit `Device::ndarray()` for CPU verification. `WgpuDevice::new()`
creates a deferred device handle; it does not prove adapter initialization
has succeeded before the first tensor allocation.

## Supported models

The 23 shipped `.tza` archives use these routes. High falls back to Base
when a Large archive is unavailable; topology alone does not imply identical
arithmetic or image quality across backends:

| Filter | Quality::High → | Balanced → | Fast → |
| --- | --- | --- | --- |
| RT (color, HDR) | `rt_hdr` | `rt_hdr` | `rt_hdr_small` |
| RT (color+albedo, HDR) | `rt_hdr_alb` | `rt_hdr_alb` | `rt_hdr_alb_small` |
| RT (color+albedo+normal, HDR) | `rt_hdr_alb_nrm` | `rt_hdr_alb_nrm` | `rt_hdr_alb_nrm_small` |
| RT (cleanAux, HDR) | `rt_hdr_calb_cnrm_large` | `rt_hdr_calb_cnrm` | `rt_hdr_calb_cnrm_small` |
| RT (LDR, all combos) | `rt_ldr*` | `rt_ldr*` | `rt_ldr*_small` |
| RT albedo prefilter | `rt_alb_large` | `rt_alb` | `rt_alb` |
| RT normal prefilter | `rt_nrm_large` | `rt_nrm` | `rt_nrm` |
| Lightmap (HDR) | `rtlightmap_hdr` | — | — |
| Lightmap (directional) | `rtlightmap_dir` | — | — |

Quality routing matches Intel OIDN semantics
(see `core/unet_filter.cpp:446-459` in upstream).

## Performance

Historical measurements on this machine (Windows 11, default wgpu DX12 backend, `--quality balanced`,
`rt_hdr` model, 10 timed iterations after a warm-up):

| Resolution | Pixels | Median latency | Throughput |
| --- | --- | --- | --- |
| 256×256 | 0.07 MP | **21.9 ms** | 2.99 MP/s |
| 1024×1024 | 1.05 MP | **302 ms** | 3.47 MP/s |
| 2048×2048 | 4.19 MP | **1217 ms** | 3.45 MP/s |

These historical runs reached about 3.5 MP/s on the tested hardware.
The earlier project comparison reported native CUDA/CUTLASS inference about
an order of magnitude faster; it is not a current benchmark or a general
speed ratio. Rebenchmark the current source/backend on the target device.

## How it differs from upstream OIDN

Historical project size estimates were ~14000 LOC for native OIDN and ~2000
LOC for this port. These are not current source counts. The port delegates
network and tensor operations to Burn:

- Conv, pool, upsample, concat, and ReLU use Burn backend operations. GPU
  dispatch uses Burn/CubeCL; the implementation is not tied to a claim that
  every operation compiles through WGSL.
- Shared Burn model definitions run through dynamic CPU (NdArray) or WGPU
  dispatch. Backend arithmetic and device support are verified separately;
  future backends are not certified by this implementation.
- **GPU-first execution**: per-tile input prep, transfer functions, the
  network, and output postprocessing all stay in Burn tensors — no
  full-image host roundtrip in ordinary tensor execution. Automatic exposure
  reads scalar results; host adapters and enabled diagnostics perform readbacks.
- **Embedded weights**: cargo features (`embed-hdr`, `embed-ldr`,
  `embed-aov`, `embed-aux-clean`, `embed-lightmap`, `embed-all`) bake the
  `.tza` blobs into the binary so deployment is a single executable. The
  on-disk `--weights-dir` path provides explicit filesystem resolution;
  read errors are reported rather than treated as missing models.
- Weights load directly from Intel's `.tza` archive (the
  [oidn-weights](https://github.com/RenderKit/oidn-weights) repo), no
  PyTorch or ONNX intermediate step.
- No C-ABI / no `libOpenImageDenoise.dll` shim. Consumers depend on the
  Rust crate directly. `OidnError` is `#[non_exhaustive]` so future
  variants ship as minor versions.
- `pub const OIDN_REFERENCE_VERSION: (u32, u32, u32) = (2, 4, 1);`
  identifies the original upstream snapshot this port tracks. Subsequent
  contract checks against local v2.5.0 are recorded separately in plan4; this
  constant does not claim numerical parity.

## Testing

Most integration tests need `data/weights/*.tza` to run real-weight
scenarios. A normal clone contains the required archives. CPU and GPU tests
must select their devices explicitly; an unavailable GPU is not evidence
that its numerical checks passed. Final repaired-source test commands and
results are tracked in [plan4.md](plan4.md).

```sh
cargo test --workspace --locked
# GPU numerical tests run only in an explicit GPU lane.
OIDN_REQUIRE_GPU=1 cargo test -p oidn-rs --test e2e_wgpu --test e2e_ldr --test multi_tile_wgpu -- --ignored --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

Historical suite inventory (2026-05-21): 25 integration tests across
`e2e_wgpu` (10), `e2e_ldr`, `e2e_ndarray`, `multi_tile_wgpu`, `formats`,
`unit_color_tile`, `all_models_smoke`, `api_surface`, plus per-crate unit
tests and TZA parser sweeps. That historical pass does not certify the current
repair changes or the user's progressive rendering scene.

## License

Apache-2.0, matching upstream Intel OIDN. Algorithms and architecture
adapted from [RenderKit/oidn](https://github.com/RenderKit/oidn)
(Apache-2.0, Copyright 2018-2025 Intel Corporation).
