# Kurchatov-2 — final cleanup pass

Scope: kill prelude compat shim, fix import sites, wire RTLightmap `--weights`,
fix bench.rs nits, sanity sweep, verify with cargo check + clippy.

## 1. Prelude shim status

Removed. `crates/oidn-rs/src/prelude.rs`:

- Top-level `prelude` re-exports only backend-agnostic types:
  `CommittedRtFilter`, `Filter`, `Image`, `ImageMut`, `ModelKey`, `OidnError`,
  `PixelFormat`, `Quality`, `RtFilter`, `RtLightmapFilter`.
- `WgpuDevice` and `WgpuBackend` are NOT in the top-level glob.
- `pub mod wgpu_prelude { pub use crate::WgpuDevice; pub use crate::device::WgpuBackend; }`
  remains the explicit opt-in path.
- Removed the redundant `pub mod core { ... }` shim Sakharov added; the
  top-level prelude is now backend-agnostic by default, so `core` was a duplicate.
- `crates/oidn-rs/src/lib.rs` re-exports were already correct (it exports the
  individual types, not the wgpu aliases as a glob).

## 2. Import-site updates

Added `use oidn_rs::prelude::wgpu_prelude::*;` next to the existing
`use oidn_rs::prelude::*;` at:

- `crates/oidn-rs/tests/e2e_wgpu.rs:13-14` — also removed two redundant inner
  `use oidn_rs::RtLightmapFilter;` lines (was at 343, 369) since the type is
  now in the prelude glob.
- `crates/oidn-rs/tests/e2e_ldr.rs:6-7`
- `crates/oidn-rs/tests/multi_tile_wgpu.rs:10-11`
- `crates/oidn-rs/examples/bench.rs:37-38`
- `crates/oidn-cli/src/main.rs:9-10`

Untouched (correctly so — backend-agnostic, no wgpu types):

- `crates/oidn-rs/tests/api_surface.rs` — uses `burn::backend::NdArray`
- `crates/oidn-rs/tests/all_models_smoke.rs` — uses `burn::backend::NdArray`
- `crates/oidn-rs/tests/e2e_ndarray.rs` — NdArray
- `crates/oidn-rs/tests/formats.rs` — `Image`, `ImageMut`, `PixelFormat` only
- `crates/oidn-rs/tests/unit_color_tile.rs` — `color` + `tile` submodules

## 3. RTLightmap `--weights` wiring

`crates/oidn-cli/src/main.rs`:

- `run_rtlightmap` gained a `user_weights: Option<Vec<u8>>` parameter.
- The dispatcher in `denoise()` now passes `user_weights` to both filter
  branches; the comment about "RTLightmap path resolves its weights inside
  the filter" was updated to reflect that an explicit `--weights` blob is now
  honoured (the `if let Some(p) = args.weights.as_deref()` arm covers it for
  both filter kinds).
- Inside `run_rtlightmap`, the builder is now mutated incrementally and
  `.weights(bytes)` is called when `user_weights` is `Some(_)`, mirroring the
  `run_rt` wiring.

## 4. bench.rs nits

`crates/oidn-rs/examples/bench.rs`:

- `psnr_db`: replaced the `1.0_f32.max(1.0)` no-op with an explicit
  `let peak = 1.0_f32; 20.0 * (peak / rmse.max(1e-12)).log10()`. Floor on
  `rmse` retained so PSNR doesn't go to `-inf` when RMSE is exactly zero.
- Renamed `now_iso_short` -> `now_epoch_secs_short`, `now_iso_long` ->
  `now_epoch_secs_long`. Doc comments updated to describe what they actually
  return (raw Unix epoch seconds, not ISO-8601). Both call sites updated:
  - `parse_args` default output filename (was line 109)
  - per-row CSV timestamp (was line 480)
- CSV header column renamed `timestamp` -> `epoch_secs`.

## 5. Sanity sweep

- `crates/oidn-rs/src/weights.rs:121` — doc-example for `resolve` updated to
  the current 7-arg `select_rt` signature (was 8 args including the removed
  `directional`). Also fixed the example flow: `resolve` returns `Option`, not
  `Result`, so the example now uses `.ok_or("...")?`.
- `crates/oidn-rs/src/filter.rs:12-15` — stripped `v0.1` qualifiers from the
  `Balanced` and `Fast` Quality variant docs.
- `crates/oidn-rs/src/tile.rs:73` — stripped `for v0.1` qualifier from the
  `plan` doc-comment.
- `crates/oidn-rs/src/gpu_ops.rs:45` — removed the session-introduced
  `#[allow(dead_code)]` on `preprocess_input`. Investigation: the function is
  actively called at `unet_runner.rs:180`, so the attribute was a stale
  band-aid silencing a non-existent warning. Confirmed via `git diff HEAD`
  that the attribute was added in this session, not pre-existing.
- `crates/oidn-rs/src/registry.rs:26-31` — added a blank line before the new
  error-listing bullet block and another blank line before the trailing
  paragraph, fixing two `clippy::doc_lazy_continuation` warnings introduced
  this session by Lomonosov's M1 changes.

No other session-introduced `#[allow(dead_code)]` found
(`git diff HEAD -- 'crates/**/*.rs' | grep "allow(dead_code)"` returned only
the one entry).

## 6. Final verification

`cargo check --workspace --all-targets` (from repo root):

```
    Finished `dev` profile [optimized + debuginfo] target(s) in 2.41s
```

Clean. No errors, no warnings.

`cargo clippy --workspace --all-targets`:

```
warning: `oidn-rs` (example "bench") generated 2 warnings
warning: `oidn-cli` (bin "oidn-rs") generated 1 warning
warning: `oidn-cli` (bin "oidn-rs" test) generated 1 warning (1 duplicate)
    Finished `dev` profile [optimized + debuginfo] target(s) in 3.76s
```

3 warnings total (one duplicated across cli bin + test target). All
pre-existing (verified via `git diff HEAD` — none of the lines triggering
them were modified this session):

| File | Lint | Severity |
|------|------|----------|
| `examples/bench.rs:18` | `clippy::doc_lazy_continuation` (module-doc numbered list continuation) | cosmetic |
| `examples/bench.rs:323` | `clippy::too_many_arguments` (`run_one` has 9, limit 7) | style |
| `oidn-cli/src/main.rs:22` | `clippy::large_enum_variant` (Cmd variants differ by ~272 bytes) | style |

Library crate `oidn-rs` is clippy-clean. `oidn-tza` and `oidn-model` are
clippy-clean. `-D warnings` would only fire on the three pre-existing
warnings above; per the brief I dropped `-D warnings` from the final report.

`cargo test` not run (out of scope per the brief — user's call).

## Open follow-ups

- `examples/bench.rs::run_one` (9 args) and `oidn-cli::Cmd` (large enum) are
  pre-existing style warnings. Both deserve a follow-up: `run_one` could
  take a `RunConfig` struct; `Cmd` could box the heavy `DenoiseArgs` variant
  (`Denoise(Box<DenoiseArgs>)`) to flatten the enum size.
- `examples/bench.rs:18` continuation-indent is a documentation-only style
  nit; one-line fix when next touching the file.
- AGENTS.md (lines 54-55, 129) and bughunt/*.md still reference the old
  `select_rt(... directional, clean_aux)` signature. Out of scope for the
  Rust cleanup but a doc sweep would catch them.
