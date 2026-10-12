# Changelog

Notable changes to `oidn-rs`.

## Unreleased

### Changed
- burn's `ndarray` backend left the runtime graph of a consumer: the workspace burn dependency is `std` only, `oidn-rs`
  takes `wgpu` as a dependency and `ndarray` as a dev-dependency (its CPU unit tests), `oidn-cli` takes `ndarray` for
  `--device cpu`. A renderer that links `oidn-rs` (render-rs `pt-denoise-oidn`) no longer compiles or links the CPU
  backend.

### Added
- **`autoexposure::Primaries` (Rec.709 / ACEScg), a run-time choice** replacing the `acescg-autoexposure`
  cargo feature: `compute_scale` / `compute_scale_tensor` take it, `RtFilterBuilder::autoexposure_primaries`
  and `RtFilter::set_autoexposure_primaries` set it (default Rec.709, the OIDN reference). One build now serves
  a Rec.709 and an ACEScg consumer; the feature is removed (callers that enabled it pass `Primaries::AcesCg`).

### Changed
- **oidn-cli EXR goes through exr-core** (exr-rs, our 1:1 OpenEXR port) instead
  of crates.io `exr`. Reading `.exr` now shares the one `image::open` load path
  (exr-image's hooks decode it with exr-core; float RGB unclamped), so the
  separate EXR loader is gone; writing is float32 R, G, B, ZIP, untagged (the
  buffer may be colour, albedo or normals). `image` drops its default features
  (which pulled `exr`) and takes its format set from exr-image.
- `autoexposure::compute_scale` (CPU path) now folds the `LUM_*` luminance
  constants inline instead of calling `color::luminance`, so both the CPU and
  tensor autoexposure paths honour the feature-selected weights. Behaviour is
  unchanged on the default (Rec.709) build; `color::luminance` itself is
  untouched and remains Rec.709 for other callers.
