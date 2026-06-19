# Changelog

Notable changes to `oidn-rs`.

## Unreleased

### Added
- **`acescg-autoexposure` cargo feature.** Measures autoexposure luminance with
  ACEScg (AP1) weights `(0.2722287, 0.6740818, 0.0536895)` — the Y row of the
  AP1→XYZ matrix — instead of the default Rec.709 weights.

  *Why:* autoexposure reduces the image to a single luminance statistic, so its
  channel weights must match the colour space of the pixels. Measuring an ACEScg
  image with Rec.709 weights skews the exposure estimate. The default stays
  Rec.709 (byte-identical to `color::luminance` and the upstream OIDN reference),
  so general-purpose consumers are unaffected; pipelines that denoise in ACEScg
  opt in.

  *Why a feature and not a runtime parameter:* the working space is fixed for a
  given pipeline, the weights are `const` (the `mul_scalar` folds away), and both
  the CPU and tensor paths read the same constants with no extra plumbing.

  *Where used:* `vfx-rs`'s `pt-denoise-oidn` (a path tracer working internally in
  ACEScg) enables it through the git dependency. This replaces a hard-coded AP1
  fork that previously lived in a vendored in-tree copy of the crate.

### Changed
- `autoexposure::compute_scale` (CPU path) now folds the `LUM_*` luminance
  constants inline instead of calling `color::luminance`, so both the CPU and
  tensor autoexposure paths honour the feature-selected weights. Behaviour is
  unchanged on the default (Rec.709) build; `color::luminance` itself is
  untouched and remains Rec.709 for other callers.
