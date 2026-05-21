# Verifier C report — API hygiene (H12, V09, V13, V14, M9/V03, trait extension)

Date: 2026-05-21
Scope: read-only verification of Sakharov's fixes plus Kurchatov-2's prelude cleanup.
Build: `cargo check -p oidn-rs --tests` — PASS (clean, no errors/warnings emitted).

## Verdicts per claim

| Fix | Verdict | Justification |
|-----|---------|---------------|
| H12 — `#[non_exhaustive]` + 4 new variants on `OidnError` | PASS | All 4 variants present with `&'static str` payloads; enum carries `#[non_exhaustive]`; legacy variants intact. (`crates/oidn-rs/src/error.rs:15-69`) |
| V09 — `OIDN_REFERENCE_VERSION` constant | PASS (could-not-cross-check) | Constant exists with doc comment at `crates/oidn-rs/src/lib.rs:43-50`. Could not verify the literal `(2, 4, 1)` against upstream because `_ref/oidn/cmake/oidn_version.cmake` is not present in this worktree (no files matching the pattern). |
| V13 — prelude split (no `WgpuDevice`/`WgpuBackend` at top level) | PASS | Top-level `pub use` (`crates/oidn-rs/src/prelude.rs:10-13`) re-exports only backend-agnostic names; wgpu types live in `wgpu_prelude` sub-module (`prelude.rs:18-21`). |
| V14 — `OidnError::Device(String)` kept + documented | PASS | Variant kept as `Device(String)` with comment explaining the Burn rationale and V14 tracking note (`error.rs:39-43`). |
| M9/V03a — `RtLightmapFilterBuilder::weights(impl Into<Vec<u8>>)` | PASS | Method present, stores `user_weights` (`rtlightmap.rs:58-61`); consumed in `build()` (`rtlightmap.rs:70`) and overrides registry lookup in `commit` (`rtlightmap.rs:213-224`). |
| M9/V03b — `RtLightmapFilter::set_progress` wired to runner | PASS | Inherent method stores closure (`rtlightmap.rs:184-186`); trait impl stores already-boxed dyn (`rtlightmap.rs:194-203`); `execute()` passes `self.progress.as_deref_mut()` into `unet_runner::run` (`rtlightmap.rs:270-284`). The previous hard-coded `None` is gone — the only `None`s in the call are the `albedo`/`normal` slots, which are correct for a colour-only lightmap path. |
| M9/V03c — `allocate_output` preserves `committed=true` on shape match | PASS | `same_dims` short-circuits the invalidation (`rtlightmap.rs:166-172`); identical pattern to RtFilter at `rt.rs:371-385`. |
| Trait extension — `Filter::set_progress` with default | PASS | Trait method signature matches spec, default returns `OidnError::UnsupportedFeatures` (`filter.rs:37-42`); both filter types override (`rt.rs:720-729`, `rtlightmap.rs:195-203`). |
| Build — `cargo check -p oidn-rs --tests` | PASS | Finished `dev` profile clean in 1.30s, no diagnostics. |

## Detailed findings

| Id | Severity | File:line | Observed | Expected |
|----|----------|-----------|----------|----------|
| H12-a | — | `crates/oidn-rs/src/error.rs:16` | `#[non_exhaustive]` on enum | Present |
| H12-b | — | `error.rs:52-54` | `Unknown(&'static str)` with `#[error("unknown error: {0}")]` | Present |
| H12-c | — | `error.rs:55-58` | `InvalidOperation(&'static str)` with `#[error]` | Present |
| H12-d | — | `error.rs:60-64` | `OutOfMemory(&'static str)` with `#[error]` | Present |
| H12-e | — | `error.rs:66-69` | `UnsupportedHardware(&'static str)` with `#[error]` | Present |
| H12-f | — | `error.rs:18-49` | Legacy variants (`Unset`, `Inconsistent`, `UnsupportedFeatures`, `MissingModel`, `Io`, `Tza`, `Load`, `Device`, `Cancelled`, `InvalidArgument`) all intact | Present |
| V09-a | — | `lib.rs:50` | `pub const OIDN_REFERENCE_VERSION: (u32, u32, u32) = (2, 4, 1);` | Present |
| V09-b | LOW | `lib.rs:43-49` | Doc comment cites `_ref/oidn/cmake/oidn_version.cmake` but file not present in this worktree | Could not cross-verify the literal `2.4.1` against upstream cmake. Non-blocking, doc-only concern. |
| V13-a | — | `prelude.rs:10-13` | Top-level glob = `{CommittedRtFilter, Filter, Image, ImageMut, ModelKey, OidnError, PixelFormat, Quality, RtFilter, RtLightmapFilter}` — no wgpu types | Matches spec |
| V13-b | — | `prelude.rs:18-21` | `wgpu_prelude` sub-module re-exports `WgpuDevice` + `WgpuBackend` | Matches spec |
| V13-c | LOW | `lib.rs:35` | Crate root still `pub use device::WgpuDevice` | Acceptable — the cleanup target was the *prelude*, not the crate root. `WgpuDevice` at crate root is required for `prelude::wgpu_prelude::WgpuDevice` to resolve as written. |
| V14-a | — | `error.rs:39-43` | Variant retained as `Device(String)`; comment notes Burn rationale and V14 tracking | Matches spec |
| Trait-a | — | `filter.rs:37-42` | Default impl returns `Err(OidnError::UnsupportedFeatures)`; signature `Box<dyn FnMut(f32) -> bool + 'static>` | Matches spec |
| Trait-b | — | `rt.rs:719-729` | `impl Filter for RtFilter` overrides `set_progress`, stores box directly | Matches spec |
| Trait-c | — | `rtlightmap.rs:194-203` | `impl Filter for RtLightmapFilter` overrides `set_progress`, stores box directly | Matches spec |
| Trait-d | LOW | `rt.rs:458-460`, `rtlightmap.rs:184-186` | Inherent `set_progress<F: FnMut(f32) -> bool + 'static>(...)` still present alongside the trait method | Coexistence intentional — inherent method takes any closure and boxes it; trait method takes a pre-boxed dyn. Both wire into the same `progress` field. No conflict at call sites. |
| M9-a | — | `rtlightmap.rs:31` + `:58-61` + `:70` + `:213-224` | Builder field `user_weights`, setter `weights(impl Into<Vec<u8>>)`, consumed in `build()` and `commit()` overriding registry lookup | Matches spec |
| M9-b | — | `rtlightmap.rs:270-284` | `execute()` extracts `self.progress.as_deref_mut()` and passes it into `unet_runner::run` as the trailing `progress` argument | Matches spec — runner now sees the user callback, no more hard-coded `None`. |
| M9-c | — | `rtlightmap.rs:166-172` | `allocate_output` checks `last_committed_dims == Some((width, height, format))` and skips `committed = false` on match | Matches spec |
| M9-d | — | `rtlightmap.rs:249` | `commit()` records `last_committed_dims = Some((out.width, out.height, out.format))` so the next `allocate_output` can compare | Required corollary — present |
| Build | — | `cargo check -p oidn-rs --tests` | Clean build, no warnings | PASS |

## Notes

- The trailing `None` arguments at `rtlightmap.rs:276-277` are the `albedo` / `normal` input slots, not the progress slot. Lightmap is colour-only, so these are correct.
- Both filters' trait impls deliberately avoid re-boxing the incoming `Box<dyn FnMut>` — micro-optimisation, no behavioural difference vs the inherent `set_progress`.
- V09 could not be cross-checked because the `_ref/` tree is absent from this checkout. The number `2.4.1` is plausible (matches the public Intel OIDN 2.4.x line) but the literal cannot be confirmed against upstream cmake from inside this worktree.

## Overall verdict

ACCEPT.

All BLOCKER and HIGH expectations are met:
- New `OidnError` variants present with correct payloads and `#[non_exhaustive]` on the enum.
- Prelude no longer shims `WgpuDevice` / `WgpuBackend` at the top level.
- `RtLightmapFilter::set_progress` is wired through to `unet_runner::run` (previous hard-coded `None` removed).
- `allocate_output` preserves committed state on shape-match for both filter types.
- `Filter::set_progress` trait method exists with the documented default, overridden by both concrete filters.
- `cargo check -p oidn-rs --tests` passes clean.

Only LOW-severity observations remain (inability to literal-check the version constant against the absent `_ref/` tree, harmless inherent+trait `set_progress` coexistence, intentional crate-root `WgpuDevice` re-export needed for the `wgpu_prelude` path to resolve). None warrant rework.
