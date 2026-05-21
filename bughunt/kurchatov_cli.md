# Kurchatov — CLI & Integration / E2E Parity Audit

- Agent: Kurchatov
- Date: 2026-05-21
- Scope: read-only parity audit of `oidn-cli`, `oidn-rs/examples/bench.rs`,
  `oidn-rs/tests/e2e_*.rs` against reference apps in
  `C:/projects/projects.rust.cg.offload/oidn/apps/`.

## Verdict

PARTIAL parity. The Rust CLI implements only a small subset of the reference
`oidnDenoise` flag surface (no device selection, type selection, srgb,
clean_aux, input_scale, weights file override, verbose, threads, affinity,
maxmem, inplace, buffer storage, reference compare, list-devices). The image
I/O backend has a fundamentally different format set: PFM/PHM/PPM (the
reference's native, OIIO-free defaults) are missing entirely; the Rust CLI
substitutes EXR and the `image` crate (PNG/JPG/etc.), so no reference golden
PFM image can be loaded or compared by the Rust CLI without conversion.
The `examples/bench.rs` is internally consistent but does NOT exercise the
same combinatoric benchmark grid as `oidnBenchmark` (no LDR mode and no
clean-aux mode in its sweep). Test coverage of model-routing combinations is
strong, but there are no CLI-level integration tests (no test ever spawns
the `oidn-rs` binary), no PFM/EXR golden-image comparison tests, and no
test exercises the `--weights` override path or the LDR/sRGB + albedo/normal
combination through the public Filter API.

## CLI flag matrix (oidnDenoise → oidn-rs `denoise` subcommand)

Reference flags from `apps/oidnDenoise.cpp:23-38` and parse loop
`apps/oidnDenoise.cpp:108-211`. Rust flags from
`crates/oidn-cli/src/main.rs:27-52`.

| ref flag | Rust flag | status | notes |
|---|---|---|---|
| `-d/--device [0-9]+\|default\|cpu\|sycl\|cuda\|hip\|metal` (oidnDenoise.cpp:111-118) | — | MISSING | Rust always uses `WgpuDevice::new()` (main.rs:121). No way to force CPU/NdArray, no physical-device index. |
| `-f/--filter RT\|RTLightmap` (oidnDenoise.cpp:119-120) | — | MISSING | CLI is hard-wired to `RtFilter` (main.rs:127). Lightmap filter exists in the lib (used only in `e2e_wgpu.rs:341,367`) but is unreachable from the binary. |
| `--hdr color.pfm` (oidnDenoise.cpp:121-125) | `-i/--input` + `--hdr` (main.rs:30,45) | DIVERGENT | Ref couples filename and HDR mode in one flag. Rust splits them; default `hdr=true` (main.rs:45-46) silently flips LDR inputs into HDR mode. |
| `--ldr color.pfm` (oidnDenoise.cpp:126-130) | `--hdr false` (no dedicated `--ldr`) | DIVERGENT | Must invert the `--hdr` bool. Counter-intuitive given Rust's default is `hdr=true`. |
| `--srgb` (oidnDenoise.cpp:131-132) | — | MISSING | No way to mark already-linear input or already-sRGB output. The library supports it (e.g. `e2e_ldr.rs:106-108`) but the CLI does not. |
| `--dir directional.pfm` (oidnDenoise.cpp:133-137) | — | MISSING | Directional RTLightmap input cannot be selected. |
| `--alb/--albedo` (oidnDenoise.cpp:138-139) | `--albedo` (main.rs:34) | PRESENT | Long form only; ref accepts `--alb`. |
| `--nrm/--normal` (oidnDenoise.cpp:140-141) | `--normal` (main.rs:38) | PRESENT | Long form only; ref accepts `--nrm`. |
| `-o/--out/--output` (oidnDenoise.cpp:142-143) | `-o/--output` (main.rs:42) | PRESENT | `--out` shortcut absent. |
| `-r/--ref/--reference` (oidnDenoise.cpp:144-145) | — | MISSING | No reference-output compare. `--maxerror` likewise missing (oidnDenoise.cpp:198-199). |
| `--is/--input_scale value` (oidnDenoise.cpp:146-147) | — | MISSING | `RtFilterBuilder::input_scale` exists (used in bench.rs:350) but is not exposed via CLI. |
| `--clean_aux` (oidnDenoise.cpp:148-149) | — | MISSING | `clean_aux(true)` exists in the library (e2e_wgpu.rs:291) but CLI cannot request `*_calb_cnrm` routing. |
| `-t/--type float\|half` (oidnDenoise.cpp:150-159) | — | MISSING | No fp16/fp32 selector. |
| `-q/--quality default\|h/high\|b/balanced\|f/fast` (oidnDenoise.cpp:160-173) | only on `bench` subcommand (main.rs:65, parse main.rs:160-167) | PARTIAL | `denoise` hard-codes `Quality::High` (main.rs:129). Aliases h/b/f are not accepted (main.rs:161-165). No `default`. |
| `-w/--weights weights.tza` (oidnDenoise.cpp:174-175) | `--weights-dir <dir>` (main.rs:50) | DIVERGENT | Ref takes a single TZA file blob (`loadFile`+`setData("weights", ...)`, oidnDenoise.cpp:316-321,363-364). Rust requires a *directory* and selects internally by model key. Override-a-single-file semantics are not reproducible. |
| `--threads n` (oidnDenoise.cpp:178-179) | — | MISSING | No thread cap. |
| `--affinity 0\|1` (oidnDenoise.cpp:180-181) | — | MISSING | |
| `--maxmem MB` (oidnDenoise.cpp:182-183) | — | MISSING | |
| `--inplace` (oidnDenoise.cpp:184-185) | — | MISSING | Library always allocates fresh output (main.rs:141). |
| `--buffer host\|device\|managed` (oidnDenoise.cpp:186-197) | — | MISSING | Wgpu backend hides storage class. |
| `-n times_to_run` (oidnDenoise.cpp:176-177) | — on `denoise`; `-n/--iters` on `bench` (main.rs:62) | PARTIAL | Ref re-runs `denoise` (with hash-determinism check, oidnDenoise.cpp:381-426). Rust `denoise` runs once; only `bench` iterates and skips the hash-stability check. |
| `-v/--verbose 0-3` (oidnDenoise.cpp:200-201) | — | MISSING | No verbose level; `tracing_subscriber_init` only prints `RUST_LOG` value (main.rs:84-89). |
| `--ld/--list_devices` (oidnDenoise.cpp:202-203) | — | MISSING | No way to enumerate adapters. |
| `-h/--help` (oidnDenoise.cpp:204-208) | clap-generated `--help` | DIVERGENT | clap auto-help; help text does not mirror reference layout. |
| — | `probe <PATH>` (main.rs:21-24) | EXTRA | Rust-only: dumps tensor list from a `.tza` file. No reference equivalent (closest is `oidnTest`). |

### oidnBenchmark → `oidn-rs bench`

Reference flags from `apps/oidnBenchmark.cpp:35-43`, parse `apps/oidnBenchmark.cpp:285-374`.

| ref flag | Rust flag | status | notes |
|---|---|---|---|
| `-d/--device` (oidnBenchmark.cpp:288-295) | — | MISSING | |
| `-r/--run regex` (oidnBenchmark.cpp:296-297) | — | MISSING | No bench-name regex filter. |
| `-n times_to_run` (oidnBenchmark.cpp:298-303) | `-n/--iters` (main.rs:62) | DIVERGENT | Same intent, but ref's `n` is split into warmup+timed (oidnBenchmark.cpp:185-191); Rust uses a fixed 1 warmup + N timed (main.rs:205-213). |
| `-s/--size W H` (oidnBenchmark.cpp:304-310) | `-r/--resolution WxH` (main.rs:58) | DIVERGENT | Two positional ints vs. single `WxH` string. |
| `-t/--type` (oidnBenchmark.cpp:311-319) | — | MISSING | |
| `-q/--quality` (oidnBenchmark.cpp:321-334) | `-q/--quality` (main.rs:66) | DIVERGENT | Same set, but Rust does not accept short aliases (`h`/`b`/`f`) and has no `default` (main.rs:161-165). |
| `--threads` / `--affinity` / `--maxmem` / `--inplace` / `--buffer` (oidnBenchmark.cpp:335-360) | — | MISSING | |
| `-v/--verbose` (oidnBenchmark.cpp:361-362) | — | MISSING | |
| `-l/--list` (oidnBenchmark.cpp:363-364) | — | MISSING | |
| `--ld/--list_devices` (oidnBenchmark.cpp:365-366) | — | MISSING | |
| — | `--weights-dir` (main.rs:70) | EXTRA | Reference loads weights from device defaults; Rust always points at a TZA directory. |
| Benchmark grid: RT × {hdr,ldr,calb/cnrm} × 3 resolutions (oidnBenchmark.cpp:246-271) | single synthetic HDR scene (main.rs:180-191) | DIVERGENT | Rust's `bench` subcommand runs one mode at one resolution per invocation. No LDR, no clean-aux, no RTLightmap, no batched suite. Cooldown sleep (oidnBenchmark.cpp:427-431) absent. |

## Image format support matrix

Reference `loadImage`/`saveImage` dispatch in `apps/utils/image_io.cpp:372-409`.
Rust `load_rgb_f32`/`save_rgb_f32` in `crates/oidn-cli/src/io.rs:5-17`.

| format | ref load (image_io.cpp) | ref save (image_io.cpp) | Rust load (io.rs) | Rust save (io.rs) | notes |
|---|---|---|---|---|---|
| PFM (.pfm, float32) | yes (`loadImagePFM`, line 381) | yes (`saveImagePFM`, line 398) | NO | NO | Native OIDN test/golden format. Rust CLI cannot read or write it. |
| PHM (.phm, float16) | yes (line 383) | yes (line 400) | NO | NO | Half-precision PFM variant. Not supported. |
| PPM (.ppm) | NO (only OIIO fallback) | yes (line 402, `saveImagePPM`) | NO (handled via `image` crate fallback) | only via `image` crate `save()` fallback (io.rs:60-70) | Untested, format presence depends on `image` features. |
| EXR (.exr) | only via OIIO (line 386, conditional `OIDN_USE_OPENIMAGEIO`) | only via OIIO (line 405) | yes (`load_exr` / `save_exr`, io.rs:19-52) | yes | Rust has first-class EXR via `exr` crate; ref needs OIIO build. |
| PNG / JPG / TIFF / HDR | only via OIIO (line 386) | only via OIIO (line 405) | yes (via `image::open` / `Rgb32FImage::save`, io.rs:54-70) | save is **8-bit only**: `to_rgb8()` (io.rs:68) drops HDR precision regardless of extension | Saving a `.hdr` or `.tiff` via the Rust CLI silently quantises to 8 bits. |
| sRGB inverse on load / sRGB forward on save (image_io.cpp:411-441) | yes, branching on extension (`isSrgbImage`) | yes, branching on extension | NO | NO | The Rust CLI never applies sRGB↔linear conversion based on extension. Combined with the missing `--srgb` flag, LDR PNG/JPG inputs are treated as already-linear, which silently produces a colour-space mismatch vs. the C++ reference. |

## Test coverage gaps

| combination | tested? | location | notes |
|---|---|---|---|
| RT, HDR, colour only (NdArray) | YES | `tests/e2e_ndarray.rs:21-71` | Smoke. Mean-drift check only, no RMSE assertion vs. clean. |
| RT, HDR, colour only (wgpu) | YES | `tests/e2e_wgpu.rs:69-104` | 64×64 smoke. |
| RT, HDR, colour+albedo+normal (wgpu) | YES | `tests/e2e_wgpu.rs:106-149` | Asserts `rt_hdr_alb_nrm` routing. |
| RT, HDR, 512×512 (single-tile sanity) | YES | `tests/e2e_wgpu.rs:151-174` | Finite-only assertion. |
| RT, HDR, RMSE-reduction proof | YES | `tests/e2e_wgpu.rs:176-211` | Real noise-reduction assert. |
| RT, albedo-only | YES | `tests/e2e_wgpu.rs:213-236` | Asserts `rt_alb_large` (quality=High default). |
| RT, normal-only | YES | `tests/e2e_wgpu.rs:238-270` | Asserts `rt_nrm_large`. |
| RT, cleanAux=true + albedo+normal | YES | `tests/e2e_wgpu.rs:272-312` | Asserts `rt_hdr_calb_cnrm`. |
| RT, quality=Fast routes to `_small` | YES | `tests/e2e_wgpu.rs:314-338` | Asserts `rt_hdr_small`. |
| RT, quality=Balanced (base) | PARTIAL | only inside cleanAux test (e2e_wgpu.rs:292) | No explicit base-vs-large test for the non-cleanAux path. |
| RTLightmap, HDR | YES | `tests/e2e_wgpu.rs:340-364` | Asserts `rtlightmap_hdr`. |
| RTLightmap, directional | YES | `tests/e2e_wgpu.rs:366-400` | Asserts `rtlightmap_dir`. |
| LDR sRGB-input route (rt_ldr) | YES | `tests/e2e_ldr.rs:52-90` | RMSE reduction asserted. |
| LDR explicit linear route (srgb=true) | YES | `tests/e2e_ldr.rs:92-117` | Finite-only. |
| LDR + albedo + normal (rt_ldr_alb_nrm) | NO | — | Model exists in OIDN but no test routes to it. |
| LDR + cleanAux (rt_ldr_calb_cnrm) | NO | — | Untested. |
| Golden-image regression (PFM/EXR vs ref) | NO | — | No test loads a reference image and compares pixel-wise against expected output. Reference relies on `--ref` + `compareImage` (oidnDenoise.cpp:431-454). |
| Hash-stability across multiple runs | NO | — | Reference FNV-1a hash check (oidnDenoise.cpp:402-426) has no Rust analogue. |
| `--weights` single-file override | NO | — | No test feeds a custom TZA blob (cf. oidnDenoise.cpp:316-321). |
| `input_scale` honoured (no autoexposure) | INDIRECT | used in tests (e2e_ndarray.rs:51, e2e_wgpu.rs:86,131, etc.) | Asserted only as a side-effect of determinism; no test checks scale arithmetic. |
| CLI binary smoke test | NO | — | No `tests/cli_*.rs` or `assert_cmd` test spawns the `oidn-rs` binary. Cargo.toml has no `dev-dependencies` (Cargo.toml:19-29). |
| EXR ↔ PFM round-trip | NO | — | I/O paths in `io.rs` have no unit tests. |
| Image format dispatch errors (unsupported ext) | NO | — | `load_rgb_f32` silently falls into the `image` crate branch (io.rs:6-9); no test asserts behaviour for unknown extensions. |

## Per-issue table

| # | severity | file:line | issue |
|---|---|---|---|
| 1 | HIGH | `crates/oidn-cli/src/io.rs:5-17` | No PFM/PHM I/O. Reference golden assets are PFM and cannot be ingested or produced by the Rust CLI. |
| 2 | HIGH | `crates/oidn-cli/src/io.rs:60-70` | `save_image` always calls `to_rgb8()` (line 68), forcing 8-bit output for every non-EXR extension. Saving `.hdr`, `.tiff`, `.tga` silently quantises. |
| 3 | HIGH | `crates/oidn-cli/src/main.rs:127-130` | `denoise` hard-codes `Quality::High`; no CLI knob. Diverges from reference default (`Quality::Default`, oidnDenoise.cpp:82). |
| 4 | HIGH | `crates/oidn-cli/src/main.rs:45-46` | `--hdr` defaults to `true` and is a bool flag without dedicated `--ldr` counterpart. Reference treats the filename flag (`--hdr` vs `--ldr`) as mutually exclusive and explicit (oidnDenoise.cpp:121-130). Rust default silently routes LDR inputs through the HDR model. |
| 5 | HIGH | `crates/oidn-cli/src/main.rs:121` | Device selection is hard-wired to wgpu. No `--device cpu` to force NdArray, no physical-device index. |
| 6 | HIGH | `crates/oidn-cli/src/io.rs:6-9` | No sRGB inverse on load and no sRGB forward on save (cf. ref `isSrgbImage` / `srgbInverse` / `srgbForward`, image_io.cpp:411-441). Combined with #4, PNG inputs are treated as already-linear. |
| 7 | MED | `crates/oidn-cli/src/main.rs:46-52` | `--clean_aux`, `--input_scale`, `--srgb`, `--type`, `--weights` (file), `--inplace`, `--threads`, `--affinity`, `--maxmem`, `--buffer`, `--verbose`, `--ref`, `--maxerror`, `--list_devices`, `--filter`, `--dir` all absent. |
| 8 | MED | `crates/oidn-cli/src/main.rs:160-167` | Quality parser rejects ref aliases (`h`/`b`/`f`) and `default`. Bench-only flag, not on `denoise`. |
| 9 | MED | `crates/oidn-cli/src/main.rs:170-227` | `bench` subcommand runs a single (mode, resolution, quality) combo. Reference `oidnBenchmark` iterates over a built-in matrix (hdr/ldr × clean/noisy aux × 3 sizes for RT plus RTLightmap, oidnBenchmark.cpp:246-271). `examples/bench.rs` partially covers this but only for hdr+colour/albedo/normal, no ldr, no cleanAux, no RTLightmap (`examples/bench.rs:103-105`). |
| 10 | MED | `crates/oidn-cli/src/main.rs:205-213` | Bench uses a fixed single warmup + N timed iterations. Reference auto-tunes (oidnBenchmark.cpp:193-206) for ≥0.5 s. No cooldown between runs (oidnBenchmark.cpp:427-431). |
| 11 | MED | `crates/oidn-cli/src/main.rs:74-89` | `tracing_subscriber_init` does not actually install a subscriber. It only prints `eprintln!("oidn-rs (log={level})")` (line 88). All `tracing::info!`/`debug!` macros from the library are no-ops at runtime, so `--verbose` would be impossible to wire even if added. |
| 12 | MED | `crates/oidn-cli/Cargo.toml:19-29` | No `dev-dependencies`, no `assert_cmd`/`predicates`. No CLI integration test fixture exists. |
| 13 | MED | `crates/oidn-cli/src/main.rs:78-81` | Exit code: any error returns `ExitCode::FAILURE` (== 1). Matches reference (`return 1` on caught exception, oidnDenoise.cpp:472-473). But missing-argument case in ref prints usage and exits 1 (oidnDenoise.cpp:99-103); Rust uses clap, which exits 2 on parse error. Subtle CI-script breakage risk. |
| 14 | LOW | `crates/oidn-cli/src/main.rs:113-120` | `denoise()` swallows all errors as `Box<dyn Error>` and prints `error: {e}` via `eprintln!`. Reference prefixes with `Error: ` (oidnDenoise.cpp:472). Trivial, but breaks `grep ^Error:` style log parsing. |
| 15 | LOW | `crates/oidn-cli/src/main.rs:107-110` | `probe` prints `name dims layout dtype` per line. Format is undocumented (no `--json`), so no machine-readable consumer exists. |
| 16 | LOW | `crates/oidn-rs/examples/bench.rs:253-257` | `psnr_db` computes `20*log10(1.max(1.0) / rmse)` — the `1.0_f32.max(1.0)` is a no-op; the comment claims peak signal ≈ 1.0 but values reach 0.95. Numerical drift small but reported PSNR is mis-scaled vs. true peak. |
| 17 | LOW | `crates/oidn-rs/examples/bench.rs:414-434` | `now_iso_short` / `now_iso_long` both return raw epoch seconds despite the names promising ISO formatting. The CSV `timestamp` column is therefore not ISO-8601. |
| 18 | LOW | `crates/oidn-rs/tests/e2e_ndarray.rs:21-71` | Smoke test asserts only finite-ness and mean drift `< 1.0`. Does NOT assert noise reduction (cf. `e2e_wgpu.rs:176-211`). NdArray path has weaker validation than wgpu. |
| 19 | LOW | `crates/oidn-cli/src/main.rs:131-139` | After `set_albedo`/`set_normal`, `commit()` is called but failure case is not specifically reported (just a `?`). No way to know which aux input is rejected. |

## Open questions

1. Should the Rust CLI gain PFM read/write so reference golden assets (and
   `_ref/oidn/training/result/*.pfm` style fixtures) can be compared
   pixel-wise without going through OpenEXR conversion?
2. Is dropping `to_rgb8()` for HDR file extensions in `io.rs:save_image`
   intentional? `.hdr` and `.tiff` are valid HDR sinks that the `image`
   crate supports as float; current code path forces 8-bit.
3. Should `--quality`, `--clean_aux`, `--input_scale`, `--srgb`,
   `--filter RT|RTLightmap` and `--device cpu|wgpu` be added to `denoise`
   to reach reference-level controllability? The library already supports
   them — only the CLI surface is missing.
4. Is the lack of CLI-level integration tests (no `assert_cmd`) a deliberate
   choice (library-only validation through `tests/e2e_*`) or an oversight?
5. Should `examples/bench.rs` move to (or duplicate as) a subcommand that
   sweeps the full oidnBenchmark grid (RT × {hdr,ldr,clean-aux} × RTLightmap
   × multiple sizes) for like-for-like numbers vs. the C++ reference?
6. Should `tracing_subscriber_init` (`main.rs:84`) actually install a
   subscriber, given the library uses `tracing` (`Cargo.toml:28`)? Right now
   `--verbose` would be inert.
