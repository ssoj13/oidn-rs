# Popov fixes

## Defect 1: wgpu version drift (MEDIUM) — FIXED

Pinned `oidn-cli` to wgpu 29 to match what `cubecl-wgpu 0.10` (via `burn-wgpu 0.21`) pulls in.

Changed lines:
- `crates/oidn-cli/Cargo.toml:30` — `wgpu = "26"` -> `wgpu = "29"`.
- `crates/oidn-cli/Cargo.toml:31` — added `pollster = "0.4"` (needed because wgpu 29's
  `Instance::enumerate_adapters` became async).
- `crates/oidn-cli/src/main.rs:469-473` — adapted `list_devices()` to the wgpu 29 API:
  - `InstanceDescriptor::default()` -> `InstanceDescriptor::new_without_display_handle()`
    (the descriptor no longer derives `Default`).
  - `Instance::new(&desc)` -> `Instance::new(desc)` (now takes the descriptor by value).
  - `enumerate_adapters(...)` is now a future, so wrapped in `pollster::block_on(...)`.
  - Added a short comment explaining the wgpu 29 API shift.

`cargo tree -p oidn-cli | grep wgpu` shows only ONE major:

```
cubecl-wgpu v0.10.0 -> wgpu v29.0.3
oidn-cli   -> wgpu v29.0.3 (direct)
```

No `wgpu v26.*` anywhere in the tree. `cargo tree --duplicates` lists no wgpu duplicates.

## Defect 2: PHM magic mismatch (HIGH) — FIXED

Changed lines in `crates/oidn-cli/src/io.rs`:

- L128-131: doc comment now distinguishes PFM (`PF`/`Pf`, f32) from PHM (`PH`/`Ph`, f16).
- L184: comment typo — "(`PF`)" -> "(`PH`)" in the `load_phm` mono-rejection error.
- L209: `save_phm` magic — `b"PF\n"` -> `b"PH\n"` (correct PHM color magic).
- L228-235: `read_pfm_header` magic table extended:
  - `PF` | `PH` -> 3 channels (color, either f32 or f16, dispatched by the caller).
  - `Pf` | `Ph` -> 1 channel (mono — rejected by both `load_pfm` and `load_phm`
    because mono is not supported on either format here).
  - Comment added clarifying the scheme.

The dispatch in `load_rgb_f32` / `save_rgb_f32` (io.rs:13-29) already routes
`.phm` -> `load_phm`/`save_phm` correctly; no change needed there.

Mono PHM is rejected on save implicitly: `save_phm` always emits color (`PH\n`)
because the in-memory buffer is always HWC RGB f32 — there is no mono code path
to reject. On load, `load_phm` rejects `Ph` (1 channel) the same way `load_pfm`
rejects `Pf`.

## cargo check result

```
$ cargo check -p oidn-cli
   Checking oidn-cli v0.1.0
    Finished `dev` profile [optimized + debuginfo] target(s) in 1.93s
```

Zero warnings, zero errors.

## Leftover concerns

- Adding `pollster` adds one small dep, but it is the canonical sync-wgpu helper
  and is already pulled transitively. Alternative would be exposing a
  `list_devices`-style helper from `oidn-rs` that hides the async wgpu surface;
  out of scope for this fix.
- The `read_pfm_header` is shared between PFM and PHM. If we ever add 1-channel
  support to one but not the other, the caller's channel check must stay strict
  (currently both reject `!= 3`, which is correct).
- No PHM round-trip test exists in this crate; the previous `PF\n`-on-write bug
  went unnoticed because writes and reads both used `PF`. A self-round-trip
  test for `.phm` would have caught it. Not added here (out of scope, brief
  forbids `cargo test`).
