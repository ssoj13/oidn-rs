# Changelog

All notable behaviour-affecting changes to `oidn-rs`.

Format follows the same project-local convention as squarebob-rs: concise
dated sections, with refactors included only when they affect API,
diagnostics, performance, or correctness.

## 2026-05-21 — Parity audit against Intel OIDN v2.4.1

Eight-agent read-only audit followed by a fix sweep. Full record in
`plan1.md`, per-agent reports under `bughunt/`, architecture handoff in
`AGENTS.md`, mermaid topology in `DIAGRAMS.md`.

### Fixed (12 HIGH parity divergences)
- **dec_conv0 ReLU** — UNet final layer now wraps in `relu(...)` matching
  the shipping C++ runtime (`unet_filter.cpp:495`). UNetLarge already had
  the correct ReLU on `dec_conv1c`.
- **Zero-pad tile borders** — per-tile input prep now zero-pads via
  `Tensor::zeros + slice_assign` instead of reflect-pad. The network was
  trained against zero borders
  (`cpu_input_process.isph:88-93`,
  `gpu_input_process.h:91`).
- **Output sanitisation chain** — new pub(crate) `preprocess_input` and
  `postprocess_color` wrappers in `gpu_ops` mirror reference op order:
  in: nan_to_zero → *input_scale → clamp[0|-1, hdr?fmax:1] → forward;
  out: nan_to_zero → clamp(0, fmax) → inverse → ldr-clamp → *output_scale.
  Closes negative-output PU/Log/sRGB inverse blow-ups and missing LDR
  `min(1)` clamp.
- **2-channel image broadcast** — `Image::to_rgb_f32` now replicates G
  into B (matches `image_accessor.h:39`); was zero-padding blue silently.
- **RT `directional` removed** — the knob silently cross-fed RTLightmap
  weights through the RT pipeline. Directional lightmaps go via
  `RtLightmapFilter::builder(...).directional(true)` only.
- **Invalid input combinations rejected** — `registry::select_rt` now
  returns `Result<ModelKey, OidnError>` and refuses `(albedo-only, hdr=true)`,
  `(normal-only, hdr || srgb)`, `(albedo + normal without color)`, mirroring
  `unet_filter.cpp:423/428/434`. `RtFilter::commit()` rejects mutually
  exclusive `hdr && srgb`.
- **`transfer_kind` honours input presence** — returns `Linear` when no
  color is present (matches `rt_filter.cpp:65`); previously defaulted to
  sRGB.
- **CLI parity with `oidnDenoise`** — full flag set: `--hdr`/`--ldr`
  (required, mutually exclusive), `--srgb`, `--clean_aux`, `--input_scale`,
  `--quality` with `default/h/high/b/balanced/f/fast` aliases, `--filter`,
  `--dir`, `--weights` single-file override, `--threads` (no-op on GPU),
  `--maxmem`, `-v/--verbose` (0..3), `--ref`/`--maxerror` compare,
  `list-devices` subcommand, `probe --json`.
- **PFM and PHM I/O** — reference golden formats now natively supported
  on load and save (float32 PFM, float16 PHM). Endianness via scale sign.
- **`save_image` keeps HDR precision** — `.hdr`/`.tiff`/`.tif` write
  float pixel data; quantisation to RGB8 only happens for `.png`/`.jpg`/
  `.bmp`. Unknown extensions error out.
- **`tracing_subscriber_init` actually installs a subscriber** — was an
  `eprintln!` stub; library `tracing` macros now reach a real sink.
- **`OidnError` C-ABI parity** — adds `Unknown`, `InvalidOperation`,
  `OutOfMemory`, `UnsupportedHardware` variants. Enum is now
  `#[non_exhaustive]`.

### Added
- `RtLightmapFilter` reaches feature parity with `RtFilter`: builder
  `weights()`, `set_progress()` wired through to the runner, and the
  `allocate_output` recommit optimisation.
- `Filter::set_progress` trait method with default returning
  `UnsupportedFeatures`; both concrete filters override.
- `pub const OIDN_REFERENCE_VERSION: (u32, u32, u32) = (2, 4, 1)`
  in the library root identifies the upstream snapshot.
- `prelude::wgpu_prelude` submodule for backend-specific re-exports;
  top-level `prelude` is now backend-agnostic.
- sRGB/PU/Log/Y_MAX constants live in `color.rs` as `pub(crate)` and are
  imported by `gpu_ops.rs` (single source of truth).

### Removed
- `RtFilter::directional()` builder method and the matching field; the
  knob made no sense for the RT filter.
- `gpu_ops::reflect_pad_2d` / `pad_axis_reflect` / `slice_axis` helpers
  (no longer needed after the zero-pad switch).
- Compat re-exports of `WgpuDevice` / `WgpuBackend` from the top-level
  `prelude` — they live in `prelude::wgpu_prelude` exclusively.

### Compatibility
- Public-API breaks: `registry::select_rt` signature change (returns
  `Result`, no `directional` arg), `RtFilter` loses `directional`,
  `prelude` no longer leaks `WgpuDevice` / `WgpuBackend` at the top
  level. Update imports to `use oidn_rs::prelude::wgpu_prelude::*;`.
- CLI break: `--hdr` is no longer default; one of `--hdr` or `--ldr`
  must be passed explicitly (matches `oidnDenoise`).

## 2026-05-16 — RT normal AOV contract and tensor diagnostics

### Fixed
- Fixed RT normal AOV preprocessing in `filters::unet_runner` to match the
  reference C++ implementation. Auxiliary normal inputs now go through the
  canonical `clamp(-1, 1)` → linear remap to `[0, 1]` (i.e.
  `n * 0.5 + 0.5`) on every code path, regardless of whether `color` /
  `albedo` are also supplied. This mirrors `getNormal()` in
  `oidn/devices/gpu/gpu_input_process.h:77` and
  `oidn/devices/cpu/cpu_input_process.isph:65-76`, which clamp and remap
  unconditionally.
- Reverts an earlier attempt to forward raw signed `[-1, 1]` normals,
  which produced out-of-distribution inputs to the U-Net (which was
  trained on the `[0, 1]`-remapped signal) and caused rainbow-coloured
  artifacts along surface edges plus extreme HDR outliers after the
  inverse PU transform.
- Note on the `snorm` flag in the reference: it only gates *channel 0*
  (the primary input for directional / normal-only filters such as
  RTLightmap). It never touches the auxiliary normal feature in RT
  Color+Albedo+Normal configurations.

### Added
- Added DEBUG-level U-Net input-contract logging: color/albedo/normal
  presence, HDR transfer mode, autoexposure scale, tile count, and whether
  normal input is treated as signed.
- Added TRACE-level tensor statistics for `run_tensors` inputs and output:
  shape, finite/NaN/Inf counts, negative-value count, `> 1.0` count, and
  min/max/mean.
- TRACE logging now enables tensor diagnostics automatically; the
  `OIDN_TRACE_TENSORS=1` environment variable remains available as an
  explicit override for callers that do not control logger filters.

### Compatibility
- Existing `RtFilter` and `CommittedRtFilter` APIs are unchanged.
- The fix applies to both tensor-native and legacy `Image<'_>` entry
  points because both paths share `unet_runner::run_tensors`.
