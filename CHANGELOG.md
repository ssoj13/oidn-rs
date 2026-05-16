# Changelog

All notable behaviour-affecting changes to `oidn-rs`.

Format follows the same project-local convention as squarebob-rs: concise
dated sections, with refactors included only when they affect API,
diagnostics, performance, or correctness.

## 2026-05-16 — RT normal AOV contract and tensor diagnostics

### Fixed
- Fixed RT normal AOV packing in `filters::unet_runner`: normal inputs are
  now preserved as signed `[-1, 1]` direction vectors whenever a normal
  tensor/image is supplied.
- Removed the previous implicit rule that only treated normals as signed
  when no color input was present. That rule corrupted `Color + Albedo +
  Normal` runs by clamping negative normal components to zero before the
  U-Net saw them.

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
