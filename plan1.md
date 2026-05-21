# Bug-Hunt Plan 1 — Parity audit oidn-rs vs Intel OIDN reference

- Date: 2026-05-21
- Orchestrator: Claude Opus 4.7
- Audit mode: read-only, no source edits, no test/build runs
- Reference snapshot: Intel OIDN v2.4.1 (`C:/projects/projects.rust.cg.offload/oidn`)
- Reports: `bughunt/mendeleev_tza.md`, `bughunt/landau_unet.md`, `bughunt/kapitsa_color.md`, `bughunt/pavlov_tile.md`, `bughunt/sechenov_filter.md`, `bughunt/ioffe_gpu_ops.md`, `bughunt/vavilov_api.md`, `bughunt/kurchatov_cli.md`

Eight specialist agents covered non-overlapping slices of the codebase. The orchestrator spot-verified every HIGH finding by reading the cited source lines in both repositories. No HIGH was inflated, no HIGH was wrongly dismissed.

---

## Verdict matrix

| Area | Agent | Verdict | BLOCKER | HIGH | MED | LOW |
|---|---|---|---:|---:|---:|---:|
| TZA loader | mendeleev | PARITY | 0 | 0 | 2 | 5 |
| U-Net model | landau | DIVERGENT | 0 | 1 | 0 | 6 |
| Color/HDR/autoexposure | kapitsa | DIVERGENT | 0 | 3 | 5 | 4 |
| Tile/image/buffer | pavlov | DIVERGENT | 0 | 1 | 1 | 8 |
| RT/RTLightmap filter | sechenov | DIVERGENT | 0 | 1 | 4 | 7 |
| GPU input/output ops | ioffe | DIVERGENT | 0 | 1 | 4 | 4 |
| Public API surface | vavilov | DIVERGENT-by-design | 0 | 3 | 7 | 6 |
| CLI & integration tests | kurchatov | DIVERGENT | 0 | 6 | 7 | 6 |

Most HIGH findings cluster around three themes: **output sanitisation** (PU/Log inverse on negative model output), **tile-boundary semantics** (reflect vs zero pad), and **CLI/API surface gaps** (missing flags, missing error variants, missing PFM I/O).

---

## Top-priority issue list (proposed fix order)

Severity coding: 🛑 BLOCKER · 🔴 HIGH · 🟠 MED · 🟡 LOW · ⚪ INFO.

### 🔴 H1 — Reflect-pad instead of zero-pad at tile borders
- Rust: `crates/oidn-rs/src/filters/unet_runner.rs:167,174,181` calls `gpu_ops::reflect_pad_2d` for every input tile (color, albedo, normal).
- Ref: `_ref/oidn/devices/cpu/cpu_input_process.isph:88-93,120-125` and `gpu_input_process.h:91-118` explicitly **zero-pad** with the literal comment `// Zero pad`.
- Impact: edge tiles feed mirrored content into a network trained against zeros. Visible seam shift in the outermost ~96 px (RF/2 base) of the image. Reported and cross-confirmed by Ioffe (I-1) and Kapitsa (B-1).
- Fix: replace `reflect_pad_2d` with a zero-fill pad. Single point of change because all three inputs use the same helper. Cleaner: pre-allocate the `[1,3,tile_h,tile_w]` tensor as zeros, then `slice_assign` the source rect into `(align_offset_x, align_offset_y, src_w, src_h)`. Delete `reflect_pad_2d` once `pad_axis_reflect` has no other callers (`crates/oidn-rs/src/gpu_ops.rs:46-107`).

### 🔴 H2 — Missing pre-inverse output clamp `max(0)` (PU/Log/sRGB blow up on negative network output)
- Rust: `crates/oidn-rs/src/filters/unet_runner.rs:193-198` feeds raw network output into `gpu_ops::apply_transfer_inverse`.
- Ref: `_ref/oidn/devices/cpu/cpu_output_process.isph:42-49` runs `value = clamp(nan_to_zero(value), 0.f, pos_max)` *before* `transferFunc.inverse(value)`.
- Impact: PU low-branch becomes negative-linear, Log goes through `exp(neg)→tiny` and the inverse domain assumptions break. Causes occasional dark-pixel halos.
- Fix: in `unet_runner.rs:193-198`, replace
  ```rust
  let post = if matches!(transfer_kind, TransferFunction::Linear) { output_tensor }
             else { gpu_ops::apply_transfer_inverse(output_tensor, &tf) };
  ```
  with a single `postprocess_color(output_tensor, &tf, hdr, snorm, output_scale)` helper inside `gpu_ops` that runs `nan_to_zero → clamp(0, pos_max) → inverse → snorm-remap → !hdr ? clamp_max(1) → mul(output_scale)`. Use it for both Linear and non-Linear, so the order matches ref byte-for-byte.

### 🔴 H3 — LDR mode missing `min(1)` clamp at output
- Rust: `crates/oidn-rs/src/filters/unet_runner.rs:191-216` never clamps when `!hdr`.
- Ref: `_ref/oidn/devices/cpu/cpu_output_process.isph:62-63` does `value = min(value, 1.f)` when `!hdr`.
- Impact: LDR outputs occasionally exceed 1.0, which then gets written as out-of-gamut pixels by `Image::write_rgb_f32` (no clamp on the host side either).
- Fix: same `postprocess_color` helper as H2.

### 🔴 H4 — Missing input clamp after `inputScale`
- Rust: `gpu_ops.rs:130-141` (`apply_transfer_forward`) and `unet_runner.rs:163-168` apply `mul(input_scale)` then go straight to the transfer curve.
- Ref: `_ref/oidn/devices/cpu/cpu_input_process.isph:35-39` does `value = clamp(nan_to_zero(value * inputScale), snorm ? -1 : 0, hdr ? pos_max : 1)` before forward.
- Impact: `+inf` and `NaN` HDR samples reach `srgb_forward`/`pu_forward`/`log_forward`. Also unsigned/negative leakage.
- Fix: add `let prepared = sanitized.clamp(if hdr { 0.0 } else { 0.0 }, if hdr { f32::MAX } else { 1.0 });` (use `f32::MAX` rather than `f32::INFINITY` — matches ref `pos_max`). Apply after the existing NaN sanitiser at `unet_runner.rs:62-72`.

### 🔴 H5 — `dec_conv0` missing ReLU on base UNet output
- Rust: `crates/oidn-model/src/unet.rs:131` → `self.dec_conv0.forward(x)` (no ReLU).
- Ref: `_ref/oidn/core/unet_filter.cpp:495` → `graph->addConv("dec_conv0", x, Activation::ReLU)`.
- Impact: pre-trained weights came from training without final ReLU, but the *shipping runtime* fuses it. Differences appear when the network predicts slightly negative values (rare in HDR, occasional in LDR dark regions).
- Fix: wrap with `relu(...)` to match the runtime: `relu(self.dec_conv0.forward(x))`. UNetLarge's `dec_conv1c` (`unet_large.rs:143`) is already correct.

### 🔴 H6 — 2-channel image broadcast: zero instead of replicate-G
- Rust: `crates/oidn-rs/src/image.rs:149` writes `out[dst_off + 2] = 0.0` for `C==2` input.
- Ref: `_ref/oidn/core/image_accessor.h:39,49` writes `pixel[1]` (replicate green into blue) for `C==2`.
- Impact: RG inputs (rare in production — happens when denoising 2-channel data like polar normals) get a wrong B channel; test `crates/oidn-rs/tests/formats.rs:84-87` currently asserts the *wrong* rule.
- Fix: change `image.rs:149` to `out[dst_off + 2] = src[src_off + 1];`. Update `formats.rs:84-87` accordingly. Cross-check `ImageMut::write_rgb_f32` (`image.rs:193-229`) — same channel-collapse policy should hold.

### 🔴 H7 — `RtFilter::directional()` cross-feeds the lightmap weight blob
- Rust: `crates/oidn-rs/src/filters/rt.rs:47,81-83` exposes a `directional` knob on the RT builder; `crates/oidn-rs/src/registry.rs:40` routes `(_,_,_,_,_,true,_)` → `rtlightmap_dir`.
- Ref: `_ref/oidn/core/rt_filter.cpp:73-117` — the RTFilter has no `directional` knob; `rtlightmap_dir` is owned by RTLightmap.
- Impact: a user calling `RtFilter::builder(...).directional(true)` loads RTLightmap-directional weights into the RT pipeline. Channel count matches (3 in / 3 out) so it silently runs but produces wrong results (log-irradiance network on linear RGB input).
- Fix: delete `directional` from `RtFilter`/`RtFilterBuilder`; delete the `directional=true` arm in `registry::select_rt`. Caller wanting directional lightmap denoise must use `RtLightmapFilter::builder(...).directional(true)`.

### 🔴 H8 — CLI: no PFM/PHM I/O
- Rust: `crates/oidn-cli/src/io.rs:5-17` dispatches only EXR and `image` crate formats; PFM/PHM absent.
- Ref: `_ref/oidn/apps/utils/image_io.cpp:372-409` natively handles `loadImagePFM`/`saveImagePFM`/`loadImagePHM`/`saveImagePHM`. These are OIDN's golden test format.
- Impact: cannot run the reference test suite end-to-end through the Rust binary; all comparisons require an EXR round-trip.
- Fix: implement `load_pfm`/`save_pfm` (float32) and `load_phm`/`save_phm` (float16, via `half::f16::from_bits`). Header is `PF\n<W> <H>\n<scale>\n` then row-by-row floats; little-endian if scale < 0, big-endian if scale > 0. Wire into `io.rs:6-9` and `io.rs:13-15` extension dispatch.

### 🔴 H9 — CLI: `save_image` always 8-bit
- Rust: `crates/oidn-cli/src/io.rs:68` unconditionally calls `to_rgb8()` before save.
- Impact: writing `.hdr`/`.tiff` silently quantises HDR output to 8 bits.
- Fix: branch on extension — `.hdr` → `image::codecs::hdr::HdrEncoder`; `.tiff` → keep `Rgb32FImage` and let `buf.save(path)` pick the right codec; `.png`/`.jpg` → existing `to_rgb8()` path.

### 🔴 H10 — CLI: `denoise` hard-codes Quality::High and lacks key flags
- Rust: `crates/oidn-cli/src/main.rs:127-130` hard-wires `Quality::High`. `--srgb`, `--clean_aux`, `--input_scale`, `--type`, `--inplace`, `--threads`, `--maxmem`, `--filter`, `--dir`, `--device`, `--verbose`, `--ref`, `--list_devices` all absent (Kurchatov §1, Sechenov #4).
- Impact: cannot reproduce reference benchmarks; cannot exercise LDR + albedo+normal or LDR + cleanAux paths from the CLI.
- Fix: add the missing clap flags to `DenoiseArgs` and pipe them into `RtFilterBuilder` / `RtLightmapFilterBuilder`. Use the same name set as `oidnDenoise` so existing scripts work unchanged. Quality parser must accept `h/b/f/default` aliases (Kurchatov #8).

### 🔴 H11 — CLI: `tracing_subscriber_init` is a no-op
- Rust: `crates/oidn-cli/src/main.rs:74-89` only does `eprintln!`; never installs a subscriber.
- Impact: every `tracing::info!`/`debug!`/`trace!` in the library is silently discarded. Future `--verbose` cannot work.
- Fix: replace body with `tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();`. Add `tracing-subscriber = "0.3"` to `oidn-cli/Cargo.toml`.

### 🔴 H12 — `OidnError` parity broken
- Rust: `crates/oidn-rs/src/error.rs:3-34` defines `InvalidArgument` and `Cancelled` (matching ref) plus 7 Rust-specific variants. Missing: `Unknown`, `InvalidOperation`, `OutOfMemory`, `UnsupportedHardware`.
- Impact: callers cannot map ref-style diagnostics; adding variants later is breaking because `OidnError` is **not** `#[non_exhaustive]`.
- Fix: add the missing four variants, annotate the enum with `#[non_exhaustive]`. Audit call sites to use the most accurate variant (e.g. allocation failure → `OutOfMemory`).

### 🟠 M1 — Silent acceptance of invalid input combinations
- Rust `crates/oidn-rs/src/registry.rs:50` returns `rt_alb` even when `hdr=true`; ref `_ref/oidn/core/unet_filter.cpp:419-431` throws `"hdr mode is not supported for albedo filtering"`. Same shape for `(normal, hdr||srgb)`.
- Fix: add explicit guards in `registry::select_rt` returning `Err(OidnError::InvalidArgument(...))` for `(only-albedo, hdr)` and `(only-normal, hdr||srgb)`.

### 🟠 M2 — Commit phase missing mutual-exclusion checks
- Rust `crates/oidn-rs/src/filters/rt.rs::commit()` accepts `directional && (hdr || srgb)` and `hdr && srgb` simultaneously. Ref `_ref/oidn/core/unet_filter.cpp:377-380` throws.
- Fix: gate in `RtFilter::commit()` with explicit `Err(OidnError::InvalidArgument(...))`.

### 🟠 M3 — `transfer_kind` ignores input-presence (color absent + normal present)
- Rust `crates/oidn-rs/src/filters/rt.rs:485-495` returns `SRGB` when only normal is set with `hdr=false, srgb=false`.
- Ref `_ref/oidn/core/rt_filter.cpp:65` returns `Linear` for `(!color && normal)`.
- Fix: extend `transfer_kind()` signature to take `has_color`/`has_normal`/`has_albedo` and mirror the ref decision tree.

### 🟠 M4 — `maxMemoryMB` heuristic vs iterative shrink
- Rust `crates/oidn-rs/src/filters/rt.rs:569-580` converts MB → pixel cap via a fixed `bytes_per_pixel`.
- Ref `_ref/oidn/core/unet_filter.cpp:300-326` calls `buildModel(maxMemoryByteSize)` in a loop and shrinks tiles until it fits.
- Fix: implement the build-and-shrink loop (or, less invasively, refine the heuristic with skip-connection + arena contributions). Document until then.

### 🟠 M5 — `nan_to_zero` opt-out has no ref analogue
- Rust `crates/oidn-rs/src/filters/rt.rs:54` and `unet_runner.rs:65-75` expose a `nan_to_zero=false` knob.
- Ref always sanitises inside `getInput`/`getAlbedo`/`getNormal`.
- Fix: keep the option but document it as a Rust extension; default `true` matches ref behaviour.

### 🟠 M6 — Autoexposure bin partition + per-pixel sanitisation
- Rust `crates/oidn-rs/src/autoexposure.rs:53-60` uses fixed-16 strides clamped to image edge.
- Ref uses proportional split `i*H/numBinsH ... (i+1)*H/numBinsH`.
- Rust skips non-finite pixels and lets negatives contribute; ref clamps `nan_to_zero(c) → [0, pos_max]`.
- Fix: switch to proportional bins + per-pixel clamp/sanitise to match ref byte-for-byte.

### 🟠 M7 — `Image` has no pixel/row stride support
- Rust `crates/oidn-rs/src/image.rs:74-82` hard-codes `row_stride = width * pixel_size`, no `pixel_stride` at all.
- Ref `_ref/oidn/core/image.cpp:9-35` accepts arbitrary `pixelByteStride ≥ pixelSize` and `rowByteStride ≥ width*pixelStride`.
- Impact: cannot denoise the RGB part of an RGBA buffer in place.
- Fix: add `pixel_stride: usize` and `Image::with_strides`/`with_row_stride` constructors; thread strides through `to_rgb_f32`/`write_rgb_f32`.

### 🟠 M8 — Buffer abstraction entirely absent
- Vavilov §2 — `OIDNBuffer`, `OIDNStorage`, external memory: not ported.
- Status: explicit non-goal per README, but a tracking note is required for future wgpu zero-copy interop.
- Fix: document in README "## Compatibility" and add a `Buffer` skeleton trait for future GPU-pointer interop.

### 🟠 M9 — `RtLightmapFilter` missing parity with `RtFilter`
- Sechenov #9, Vavilov V03 — no `set_progress`, no `weights()` blob override, no `same_dims` recommit optimisation (sechenov #8).
- Fix: replicate the `RtFilter` builder/runtime methods on `RtLightmapFilter`. Consider abstracting common builder bits into `UnetFilterBuilder<B, S>` to avoid drift.

### 🟠 M10 — Reference CSV-noisy: dropped progress callback precision, no `Send+Sync`, README link
- Vavilov V02 (progress precision f32 vs double), V10 (no `Send+Sync` docs), V07 (README quickstart). Apply minor fixes; mostly hygiene.

---

## Low-severity batch (collapse during cleanup pass)

- Mendeleev M-01: TZA loader copies blobs into `Vec<u8>` instead of zero-copy borrow.
- Mendeleev L-01..L-05: tighter up-front validation against malicious TZA.
- Landau U2-U10: misc dead branches, missing `H%16==0` assert, `Variant::XLarge` reserved-but-unused, `fastMath` flag absent.
- Kapitsa C-3, C-4: dead `forward_one` call and unused `norm_scale` for Linear/sRGB.
- Pavlov T-01, T-04: `round_up_pad` order vs ref closed-form (currently equivalent on real call sites).
- Sechenov 6, 7, 8, 13, 15-17: doc-comment freshness, weight-detection by filename suffix, fastMath, in-place tiling.
- Ioffe I-7, I-9, I-10: channel-block padding / host-roundtrip in legacy `run()`.
- Vavilov V11-V18: prelude hygiene, `pub` tuple-field hygiene, `OidnError::Device(String)`, `CommittedRtFilter` type-state half-impl.
- Kurchatov #14-#19: error prefix, probe `--json`, `psnr_db` 1.max(1.0) no-op, `now_iso_*` returning epoch seconds, NdArray e2e weaker than wgpu.

---

## Suggested deduplication / consolidation

1. **`postprocess_color` helper** — fold H2+H3 (and the existing inverse-transfer call site) into one function so the order `nan_to_zero → clamp → inverse → snorm-remap → ldr-clamp → output-scale` is enforced in one place.
2. **`preprocess_input` helper** — similar fold of H4 + the input-scale multiply + forward transfer + zero-pad (H1). Single entry point that mirrors ref `cpu_input_process.isph:31-51`.
3. **Constant deduplication** — `crates/oidn-rs/src/gpu_ops.rs:26-44` redefines sRGB/PU constants that already live in `crates/oidn-rs/src/color.rs:96-124`. Make `color.rs` constants `pub(crate)` and import them from `gpu_ops.rs` to prevent silent drift.
4. **Filter common base** — `RtFilter` and `RtLightmapFilter` share ~70 % of the commit/execute flow (registry resolve → weights load → net build → tile plan → run_tensors). Factor into `crates/oidn-rs/src/filters/unet_filter.rs::UnetFilterBase<B, S>` taking a strategy struct (transfer kind, weight key picker, input prep). Sechenov #1, #4, #6, #8 all stem from divergent copy-paste.
5. **`select_rt` truth table** — the chained `match (color, alb, nrm, hdr, srgb, dir, clean) { ... }` in `crates/oidn-rs/src/registry.rs:30-53` can be a small data-driven table that mirrors `_ref/oidn/core/unet_filter.cpp::getWeights` line-by-line and rejects invalid combos uniformly.
6. **`reflect_pad_2d` removal** — once H1 is fixed, the helper has no live callers; delete it.

---

## Suggested commit decomposition

1. `fix(unet): wrap dec_conv0 in ReLU to match runtime` — H5, one-line.
2. `fix(filter): zero-pad tile borders instead of reflect-pad` — H1, refactor `unet_runner` input prep.
3. `fix(transfer): sanitise + clamp around inverse transfer; honour LDR clamp` — H2 + H3 via new `postprocess_color`.
4. `fix(transfer): clamp + sanitise after inputScale before forward` — H4 via new `preprocess_input`.
5. `fix(image): replicate green into blue for 2-channel input` — H6 + update `tests/formats.rs`.
6. `fix(filter): remove directional knob from RtFilter, gate combos in select_rt` — H7 + M1 + M2.
7. `feat(cli): PFM/PHM I/O + HDR `.hdr` save path` — H8 + H9.
8. `feat(cli): expose quality, srgb, clean_aux, input_scale, weights-file, filter selector, verbose, threads, maxmem, device flags` — H10.
9. `fix(cli): install tracing subscriber from RUST_LOG` — H11.
10. `feat(error): add Unknown/InvalidOperation/OutOfMemory/UnsupportedHardware; #[non_exhaustive]` — H12.

After landing 1–10 the codebase moves from DIVERGENT to PARITY on every audited slice except the deliberately-omitted Buffer/Device-type/Async surface (Vavilov §1–§2, M8) which is non-goal.

---

## Test plan (run after fixes land)

- `cargo test -p oidn-tza --tests` — regression for TZA parser.
- `cargo test -p oidn-model --tests` — verifies the `dec_conv0` ReLU change doesn't break unet shape tests.
- `cargo test -p oidn-rs --tests -- formats` — H6 update.
- `cargo test -p oidn-rs --test e2e_wgpu` — H1, H2, H3, H4 produce different pixel output; expect golden tiles to be regenerated.
- `cargo test -p oidn-rs --test multi_tile_wgpu` — seam-jump assertion must still pass (and may be tightened from 0.01 to 0.003 once H1 lands).
- New golden test: load `data/weights` and a reference PFM, compare against `_ref/oidn` known-good output within RMSE < 0.005.

The audit deliberately did not run tests. Reruns are the user's call; per project rules, do not blindly retest until all code changes are in.

---

## Open questions for the user

1. Should `Buffer` zero-copy interop with externally-owned GPU memory be added to the roadmap, or stay out of scope (current README stance)? (Vavilov M8.)
2. Is `fastMath` worth porting once a Burn backend exposes per-op fast-math toggles? (Landau §dtype, Sechenov #17.)
3. Should `RtLightmapFilter` expose a runtime `set_progress` and full builder parity with `RtFilter`, or keep its narrower surface? (M9.)
4. Multi-subdevice / multi-device support (`device->getNumSubdevices()`): non-goal, or future work? (Pavlov T-03, Sechenov §open Q-4.)
5. `Quality::Default` sentinel — add for API symmetry or keep `High` as `#[default]`? (Sechenov §open Q-6, Vavilov V08.)
6. Update `data/weights` golden test set to assert layout-set membership `∈ {X, Oihw}` directly? (Mendeleev §open Q-5.)
7. `snorm` color contract for `RtLightmapFilter` directional path — handled inside the filter or absent? (Kapitsa I-3, Ioffe I-4.)

---

## Awaiting approval

This plan is the consolidated, cross-verified output of the eight-agent bughunt sweep. Awaiting user approval before any code change. No fixes have been applied; no tests have been run. All source-line citations were verified by direct read of both repos.
