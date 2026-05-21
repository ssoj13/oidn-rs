# Sakharov fixes — public API hygiene pass

## Applied

- H12 — `OidnError` parity + `#[non_exhaustive]`
- V14 — documented `Device(String)` rationale (no Burn device error type to wrap; revisit when Burn stabilises)
- V13 — backend-agnostic prelude (sub-module split; main prelude kept compatible)
- V09 — `OIDN_REFERENCE_VERSION` constant
- Filter trait extension — `Filter::set_progress` default + `RtFilter` / `RtLightmapFilter` overrides
- M9 / V03 — `RtLightmapFilter` parity with `RtFilter`:
  - `RtLightmapFilterBuilder::weights(impl Into<Vec<u8>>)` for user TZA blobs
  - `RtLightmapFilter::set_progress(F: FnMut(f32) -> bool + 'static)` inherent, plus trait impl
  - `same_dims` recommit optimisation on `allocate_output` (tracks `last_committed_dims`)
  - `set_color` only invalidates on first set, mirroring `RtFilter::set_color`

## Files changed

- `crates/oidn-rs/src/error.rs` — H12 + V14
- `crates/oidn-rs/src/lib.rs` — V09
- `crates/oidn-rs/src/prelude.rs` — V13 (`core` + `wgpu_prelude` sub-modules)
- `crates/oidn-rs/src/filter.rs` — `Filter::set_progress` default
- `crates/oidn-rs/src/filters/rt.rs` — added `Filter::set_progress` impl that stores
  the boxed callback directly (alongside the inherent generic `set_progress`)
- `crates/oidn-rs/src/filters/rtlightmap.rs` — M9 / V03 parity:
  - new fields: `user_weights`, `progress`, `last_committed_dims`
  - new builder methods: `weights`
  - new filter methods: `set_progress` (inherent + trait), updated `allocate_output`,
    updated `commit` to honour user weights + cache `last_committed_dims`,
    updated `execute` to pass progress callback to `unet_runner::run`
- `crates/oidn-rs/src/device.rs` — unchanged (kept as-is; see V14 follow-up)

## Cargo check

`cargo check -p oidn-rs --tests` → clean (single run, no warnings reported in tail).

## V13 — prelude compatibility note

Spec asked for a strict move of `WgpuDevice` / `WgpuBackend` into a
`wgpu_prelude` sub-module. A strict move breaks `crates/oidn-rs/tests/*` which
`use oidn_rs::prelude::*` and then reference both types unqualified. Those test
files are outside Sakharov's edit scope.

Compromise shipped: the historic top-level prelude still re-exports both wgpu
aliases (so existing call sites — including the in-tree tests — keep
compiling), AND a backend-agnostic `prelude::core` sub-module plus an explicit
`prelude::wgpu_prelude` sub-module are now exposed so new generic call sites
can opt into a backend-clean import without touching the legacy surface.

If a hard break is preferred, a follow-up commit can drop the top-level wgpu
re-exports and update the four test files
(`multi_tile_wgpu.rs`, `e2e_ldr.rs`, `e2e_wgpu.rs`, plus `examples/bench.rs`
and `crates/oidn-cli/src/main.rs`) to import from `wgpu_prelude`.

## V14 — `OidnError::Device(String)` status

Searched for a wrap-able Burn error type (`DeviceError` / `burn_wgpu::Error`):
none exposed. `burn-wgpu` panics or returns `Result<_, String>` itself for
device init. Variant kept as `Device(String)`; rustdoc updated to record the
rationale and reference V14. No new dependencies added.

## Follow-ups (out of scope for Sakharov)

- V15 — Seal `ModelKey` field (`registry.rs` is owned by Lomonosov).
- Tests + CLI + bench example currently rely on top-level `prelude::*` wgpu
  re-exports. If the project wants the strict backend-agnostic prelude, update
  those call sites to `use oidn_rs::prelude::wgpu_prelude::*;` and remove the
  top-level wgpu pubs.
- The version constant currently hard-codes `(2, 4, 1)` because the
  `_ref/oidn/` snapshot is not checked into this worktree. When `_ref/oidn` is
  vendored back in, cross-check against `_ref/oidn/cmake/oidn_version.cmake`.
- `CommittedRtFilter` does not expose `set_progress` symmetrically — its
  `execute_tensors` already accepts a `&mut ProgressFn<'_>` argument. Consider
  whether the trait method should be widened or whether documentation
  suffices.
