# oidn-rs

Pure Rust port of [Intel Open Image Denoise](https://www.openimagedenoise.org/)
running on [Burn](https://burn.dev/) + [wgpu](https://wgpu.rs/), targeting any
GPU vendor (NVIDIA / AMD / Intel / Apple) through a single backend.

**Status:** every shipped TZA model loads and runs, both `UNet` (base/small)
and `UNetLarge` (large) topologies are implemented, multi-tile inference
seamless on real GPU, denoiser verified to reduce synthetic noise by **~11×**
RMSE on wgpu.

## Workspace layout

```
crates/
├─ oidn-tza/    standalone TZA tensor archive parser (zero ML deps)
├─ oidn-model/  U-Net definitions on Burn (generic over Backend)
├─ oidn-rs/     runtime — filters, tiling, color, autoexposure
└─ oidn-cli/    command-line binary (`oidn-rs probe | denoise | bench`)
```

Each Rust source file mirrors a specific C++ file in the upstream
[RenderKit/oidn](https://github.com/RenderKit/oidn) tree (v2.4.1). Comments
in the code reference upstream paths like `core/tza.cpp`, `training/model.py`,
etc. — those are paths inside the Intel repository, not local paths.

## Quickstart

Trained Intel weights ship in this repo at `data/weights/*.tza` as regular
git blobs (~48 MB across 24 model variants). A plain clone is enough — no
Git LFS setup required.

```sh
git clone https://github.com/ssoj13/oidn-rs.git
cd oidn-rs
cargo build --release --workspace

# Print the tensor list of a weights blob
cargo run -p oidn-cli --release -- probe data/weights/rt_hdr.tza

# Denoise an HDR EXR (works with optional --albedo / --normal AOVs).
# `--weights-dir data/weights` is the default; pass it explicitly if you
# put the weights elsewhere.
cargo run -p oidn-cli --release -- denoise -i noisy.exr -o clean.exr
cargo run -p oidn-cli --release -- denoise \
    -i color.exr --albedo albedo.exr --normal normal.exr -o out.exr

# Benchmark on a synthetic scene
cargo run -p oidn-cli --release -- bench --resolution 1024x1024 --iters 10
```

## Library use

```rust
use oidn_rs::prelude::*;

let device = WgpuDevice::new()?;
let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, "data")
    .hdr(true)
    .quality(Quality::High)
    .build();

filter.set_color(&Image::from_rgb_f32(&color, w, h));
filter.set_albedo(&Image::from_rgb_f32(&albedo, w, h)); // optional
filter.allocate_output(w, h, PixelFormat::Rgb32f);
filter.commit()?;
filter.execute()?;

let (raw, _, _, _) = filter.take_output().unwrap();
```

For lightmaps, use `RtLightmapFilter` instead (HDR Log transfer, or directional
mode with Linear transfer and signed input).

The same `RtFilter<B>` / `RtLightmapFilter<B>` works generically over any Burn
backend — tests use `burn::backend::NdArray<f32>` for fast CPU verification.

## Supported models

All 24 shipped `.tza` weight files load and run:

| Filter | Quality::High → | Balanced → | Fast → |
|---|---|---|---|
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

Measured on this machine (Windows 11, default wgpu DX12 backend, `--quality balanced`,
`rt_hdr` model, 10 timed iterations after a warm-up):

| Resolution | Pixels | Median latency | Throughput |
|---|---|---|---|
| 256×256 | 0.07 MP | **21.9 ms** | 2.99 MP/s |
| 1024×1024 | 1.05 MP | **302 ms** | 3.47 MP/s |
| 2048×2048 | 4.19 MP | **1217 ms** | 3.45 MP/s |

Throughput plateaus around ~3.5 MP/s on the test hardware — that's the
sustained per-pixel inference cost. Native Intel OIDN on CUDA via CUTLASS is
roughly an order of magnitude faster on equivalent NVIDIA hardware; we trade
that for cross-vendor portability without writing a single GPU kernel
ourselves.

## How it differs from upstream OIDN

OIDN ships ~14000 LOC of C++ + ISPC + CUDA/SYCL/HIP/Metal device backends.
oidn-rs ships ~2000 LOC of Rust by **delegating all the GPU math to Burn**:

- Conv / pool / upsample / concat / ReLU come from `burn-wgpu`'s WGSL kernels
  — we don't write GPU code ourselves.
- The U-Net architecture is described once as a Burn `Module` and runs
  unchanged on CPU (NdArray), wgpu (Vulkan / DX12 / Metal / WebGPU), and
  any future Burn backend (CUDA / ROCm).
- Weights load directly from Intel's `.tza` archive (the
  [oidn-weights](https://github.com/RenderKit/oidn-weights) repo), no
  PyTorch or ONNX intermediate step.
- No C-ABI / no `libOpenImageDenoise.dll` shim. Consumers depend on the
  Rust crate directly.

## Testing

Most integration tests need `data/weights/*.tza` to run real-weight
scenarios — without them they skip silently with
`eprintln!("skipping: weights not initialised")`. A normal clone already
contains the weights.

```sh
cargo test --workspace --release
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

## License

Apache-2.0, matching upstream Intel OIDN. Algorithms and architecture
adapted from [RenderKit/oidn](https://github.com/RenderKit/oidn)
(Apache-2.0, Copyright 2018-2025 Intel Corporation).
