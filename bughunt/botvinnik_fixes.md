# Botvinnik CLI fixes

## Scope applied

H8, H9, H10, H11 plus the optional `probe --json` and `--list-devices` features.
No library code touched.

## Files changed

- `crates/oidn-cli/Cargo.toml`
- `crates/oidn-cli/src/io.rs`
- `crates/oidn-cli/src/main.rs`

## New dependencies

- `tracing-subscriber = { version = "0.3", features = ["env-filter"] }`
- `wgpu = "26"` (matches the version `burn-wgpu` 0.21 already pulls in; verified
  with `cargo tree`).

No other deps added. `half`, `image`, `exr`, `bytemuck` were already present.

## Fix details

### H11 — tracing subscriber

`tracing_subscriber_init` in `main.rs` now installs a real `fmt` subscriber with
an `EnvFilter`. The denoise subcommand has a `-v <0..3>` flag that overrides
`RUST_LOG`. `try_init` is used so re-entry from `--list-devices`/`bench` is
harmless.

### H8 — PFM and PHM I/O

`io.rs` gained `load_pfm`, `save_pfm`, `load_phm`, `save_phm` plus a shared
`read_pfm_header` parser and `read_token` helper. Specifics:

- 3-channel `PF` only; 1-channel `Pf` header is parsed but `load_pfm`/`load_phm`
  refuse it with a clear error. (Mono support deferred; documented in the file
  header.)
- Scale sign drives endianness: negative → little-endian (default writer
  behaviour, matches OIDN reference); positive → big-endian.
- Rows are flipped at load and save time so the in-memory buffer is always
  top-to-bottom HWC RGB f32, identical to the EXR/`image` code paths.
- PHM uses `half::f16::{from_le_bytes,from_be_bytes,from_f32,to_le_bytes}`.
- Both routines wired through `load_rgb_f32`/`save_rgb_f32` dispatch on lowercased
  extension. `.exr` keeps the existing path.

### H9 — `save_image` preserves HDR

`save_image` now branches on extension:

- `.png` / `.jpg` / `.jpeg` / `.bmp` → existing `to_rgb8()` quantisation path.
- `.hdr` → `image::codecs::hdr::HdrEncoder::encode` over `Rgb<f32>` samples.
- `.tif` / `.tiff` → save the `Rgb32FImage` directly (the `image` crate's TIFF
  codec writes float samples without quantisation).
- Anything else returns an explicit error listing supported extensions.

### H10 — CLI flag surface

`DenoiseArgs` (in `main.rs`) now exposes:

`-i/--input`, `-o/--output`, `--albedo`, `--normal`, `--hdr`, `--ldr`, `--srgb`,
`--clean_aux` (with `--clean-aux` alias), `--input_scale` (alias `--input-scale`),
`-q/--quality` (custom `value_parser` accepting
`default|high|h|balanced|b|fast|f`), `-f/--filter <RT|RTLightmap>`,
`--dir`/`--directional`, `--weights_dir`, `--weights`, `--threads`,
`--maxmem`, `-n/--iters`, `-v/--verbose`, `--ref` (alias `--reference`),
`--maxerror`.

Validation done at the top of `denoise()`:

- `--hdr` and `--srgb` are `conflicts_with` in clap (rejected at parse time).
- One of `--hdr` or `--ldr` must be set (matches `_ref/oidn/apps/oidnDenoise.cpp:121-130`).
- `--clean_aux` requires both `--albedo` and `--normal`.
- `--filter RTLightmap` rejects `--albedo`/`--normal`.

Top-level subcommands added: `list-devices` (wgpu adapter enumeration) and
`probe --json`. `probe` JSON is hand-rolled to avoid an extra dep.

Weight resolution order in the denoise path:

1. `--weights <FILE>` → read bytes from disk and pass via
   `RtFilterBuilder::weights(Vec<u8>)`.
2. Otherwise use `oidn_rs::registry::select_rt` + `oidn_rs::weights::resolve`
   with the user `--weights_dir` (if any) as filesystem fallback. Embedded
   weights are picked up automatically when the matching `embed-*` feature
   is built.
3. `RTLightmap` path resolves inside the library via `weights_dir`; no
   embedded shortcut (RTLightmap doesn't accept user weights in the library
   API). Documented in code.

The `bench` subcommand now uses the new `parse_quality_clap` parser too and
accepts `--threads` as a documented no-op.

### Cleanup

- `bench`'s `parse_quality` was replaced by `parse_quality_clap` so it accepts
  the same aliases as `denoise`.
- `probe` got `--json`; the human-readable path is unchanged when the flag is
  absent.

## Build verification

`cargo check -p oidn-cli` from the workspace root:

```
Checking oidn-cli v0.1.0 (...)
Finished `dev` profile [optimized + debuginfo] target(s) in 1.75s
```

No warnings, no errors. Initial run flagged two errors against my first draft:

- `select_rt` takes 7 args (not 8 — no `directional`) and returns
  `Result<ModelKey, OidnError>` not `Option`. Fixed: only call `select_rt`
  for `--filter RT`; for `RTLightmap` let the library resolve via
  `weights_dir`; use `?` on the `Result`.

## Known limitations / follow-ups

- `--threads` is a documented no-op on wgpu; we log an `info!` if the user
  passes it. A real CPU/NdArray path needs generic plumbing through the
  binary — out of scope.
- `--list-devices` uses `wgpu::Instance::enumerate_adapters(Backends::all())`.
  Works on Windows / Linux / macOS host (`wgpu` crate compiled in). Output
  format is `[idx] name (vendor) backend=... device_type=...`.
- 1-channel PFM/PHM (`Pf` magic) is parsed but loading errors out — the colour
  pipeline only handles 3-channel images.
- RTLightmap with `--weights` user blob: not wired. The library
  `RtLightmapFilterBuilder` doesn't expose a `.weights(bytes)` setter; reading
  it would require a library change. Out of scope per the task.
- `examples/bench.rs` cleanup (`now_iso_*`, `psnr_db` `1.0_f32.max(1.0)`) is
  outside the editable file set; left untouched.
- PFM/PHM endianness: we emit `-1.0` scale (little-endian) on write. Big-endian
  reads are supported on load. We do not support writing big-endian PFM —
  fine for parity with OIDN's reference writer.
- `compare_against_reference` requires the reference image to match the output
  resolution exactly; no resampling is attempted.
