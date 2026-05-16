# Changelog

All notable behaviour-affecting changes to `oidn-rs`.

Format follows the same project-local convention as squarebob-rs: concise
dated sections, with refactors included only when they affect API,
diagnostics, performance, or correctness.

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
