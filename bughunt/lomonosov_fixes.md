# Lomonosov — image broadcast + filter routing fixes

## Fixes applied

### H6 — 2-channel image broadcast (replicate G into B)
- `crates/oidn-rs/src/image.rs`: `to_rgb_f32` 2-channel branch now writes
  `out[dst_off + 2] = src[src_off + 1]` (was `0.0`). Matches
  `_ref/oidn/core/image_accessor.h::get3` for `C==2`.
- Header doc-comment (`//! shorter formats broadcast …`) and `to_rgb_f32`
  doc-comment updated to describe replicate-G semantics.
- `write_rgb_f32` audited — the `C==2` path already drops blue
  (`pixel[0]=x; pixel[1]=y`), matching `set3` for `C==2`. No change.
- `crates/oidn-rs/tests/formats.rs`: test renamed
  `rg_f32_replicates_green_into_blue_and_drops_blue_on_write`; now asserts
  `rgb[x*3+2] == src[x*2+1]`.

### H7 — Remove `directional` from RtFilter
- `crates/oidn-rs/src/filters/rt.rs`:
  - Removed `directional: bool` field from `RtFilterBuilder` and `RtFilter`.
  - Removed `.directional(v)` builder method.
  - Removed `directional` from `select_rt(...)` call site.
  - Removed the directional branch from `transfer_kind` (see M3).
- `crates/oidn-rs/src/filters/rtlightmap.rs`: untouched — `RtLightmapFilter`
  retains its own `directional` knob (its native home).

### M1 — `select_rt` returns `Result`, rejects invalid combos
- `crates/oidn-rs/src/registry.rs`:
  - Signature changed: `pub fn select_rt(...) -> Result<ModelKey, OidnError>`.
  - `directional` parameter dropped from arg list.
  - Removed the `rtlightmap_dir` arm (was cross-feeding the lightmap weights).
  - Explicit `InvalidArgument` errors for the three combinations called out
    by `_ref/oidn/core/unet_filter.cpp:423,428,434`:
    - `(albedo-only, hdr=true)` → `"hdr mode not supported for albedo-only filtering"`
    - `(normal-only, hdr || srgb)` → `"hdr/srgb not supported for normal-only filtering"`
    - `(albedo + normal without color)` → `"invalid combination of input features"`
  - All other no-match cases fall through to `Err(OidnError::UnsupportedFeatures)`.
- `crates/oidn-rs/src/filters/rt.rs::build_commit_artifacts`: uses `?` on
  `select_rt(...)` directly.

### M2 — Mutual exclusion in `RtFilter::commit()`
- `crates/oidn-rs/src/filters/rt.rs::commit()`: early guard
  `if self.hdr && self.srgb` → `Err(InvalidArgument("hdr and srgb are mutually exclusive"))`.
  Runs before `build_commit_artifacts`. No `directional` check (knob removed in H7).

### M3 — `transfer_kind` accounts for input presence
- `crates/oidn-rs/src/filters/rt.rs`: signature changed to
  `fn transfer_kind(&self, has_color: bool) -> TransferFunction`. Returns
  `TransferFunction::Linear` when `!has_color` (matches
  `_ref/oidn/core/rt_filter.cpp:65`). Updated both call sites
  (`commit_tensor_model`, `execute`) to pass `has_color`.

## Files changed
- `crates/oidn-rs/src/image.rs`
- `crates/oidn-rs/src/filters/rt.rs`
- `crates/oidn-rs/src/registry.rs`
- `crates/oidn-rs/tests/formats.rs`

`crates/oidn-rs/src/filters/rtlightmap.rs` reviewed; no change required.

## `cargo check -p oidn-rs --tests` results
- `oidn-rs` **lib compiles cleanly** (no warnings, no errors).
- `formats` test target compiles cleanly.
- Pre-existing failures in `tests/e2e_ldr.rs`, `tests/e2e_wgpu.rs`,
  `tests/multi_tile_wgpu.rs`: `WgpuDevice` / `WgpuBackend` unresolved.
  Cause is unrelated to this scope — `prelude.rs` was refactored in
  parallel to split wgpu types into `prelude::wgpu_prelude`, but the
  e2e tests still import via `oidn_rs::prelude::*`. **Out of my edit
  scope** (those three test files are not in my allowlist). Flagged
  below.

## Open follow-ups
1. **E2E tests need `use oidn_rs::prelude::wgpu_prelude::*;`** added
   alongside the existing `use oidn_rs::prelude::*;` line. Owner of
   `prelude.rs` split (not in my scope) should fix or restore.
2. **`weights.rs:121` doc-comment example** still shows the old
   `select_rt(... directional, clean_aux, quality)?` signature. Doc-only,
   outside my scope.
3. **Single-tier directional path** — `RtLightmapFilter::directional`
   still routes to `rtlightmap_dir`; that's the intended home. No work
   needed unless we want a `quality` upgrade path there too.
