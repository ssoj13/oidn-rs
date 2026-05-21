# Verifier B — Lomonosov fixes verification (2026-05-21)

## Verdict summary

| ID | Verdict   | One-line justification                                                                                                          |
|----|-----------|---------------------------------------------------------------------------------------------------------------------------------|
| H6 | PASS      | `to_rgb_f32` 2-channel branch writes `src[src_off+1]` to dst[+2]; 1ch replicates `v`; 3ch untouched.                            |
| H6t| PASS      | `formats.rs::rg_f32_replicates_green_into_blue_and_drops_blue_on_write` asserts `rgb[x*3+2] == src[x*2+1]`.                     |
| H7 | PASS      | Zero hits for `directional` anywhere in `filters/rt.rs`; builder method, field, and `select_rt` arg all gone.                   |
| H7L| PASS      | `RtLightmapFilter::directional(...)` still present at `rtlightmap.rs:49`; mode-selection logic preserved.                       |
| M1 | PASS      | `select_rt` returns `Result<ModelKey, OidnError>` and rejects the three invalid combos with `InvalidArgument`.                  |
| M2 | PASS      | `commit()` rejects `hdr && srgb` at `rt.rs:739-743`, before `build_commit_artifacts`.                                           |
| M3 | PASS      | `transfer_kind(has_color)` returns `Linear` when `!has_color` (rt.rs:481-484); call sites updated.                              |
| 8  | PASS      | `cargo check -p oidn-rs --lib` finishes cleanly (no errors, no warnings).                                                       |

Overall verdict: **ACCEPT**.

---

## Detailed findings

| ID | Severity | File:line                                                                  | Observed                                                                                                                                                    | Expected                                                                                              |
|----|----------|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| H6 | —        | `crates/oidn-rs/src/image.rs:148-152`                                      | `out[dst_off+2] = src[src_off + 1]` in the 2-channel match arm.                                                                                             | Replicate G into B per `_ref/oidn/core/image_accessor.h:39,49`.                                       |
| H6 | —        | `crates/oidn-rs/src/image.rs:141-147`                                      | 1ch branch writes `v` to all 3 dst channels.                                                                                                                 | Replicate scalar to RGB per `image_accessor.h:41,51`.                                                  |
| H6 | —        | `crates/oidn-rs/src/image.rs:153-157`                                      | 3ch branch unchanged: copies src[0..2] verbatim.                                                                                                             | Direct copy; matches `image_accessor.h:37,47`.                                                         |
| H6t| —        | `crates/oidn-rs/tests/formats.rs:77-100`                                   | Test renamed; line 89 asserts `rgb[x*3+2] == src[x*2+1]`. Write-back at line 99 asserts blue-dropped destination `[0.9,0.8, 0.6,0.5, 0.3,0.2, 0.05,0.025]`. | Replicate-G read + drop-B on write.                                                                    |
| H7 | —        | `crates/oidn-rs/src/filters/rt.rs` (whole file)                            | `grep "directional"` returns no matches.                                                                                                                     | No `directional` field, method, or arg in any RtFilter context.                                       |
| H7L| —        | `crates/oidn-rs/src/filters/rtlightmap.rs:28,49,67,86,190,261`             | `RtLightmapFilter[Builder]::directional` field + setter retained; routes to `rtlightmap_dir` and Linear transfer.                                          | Preserved — this is its correct home (`_ref/oidn/core/rtlightmap_filter.cpp:19-30`).                  |
| M1 | —        | `crates/oidn-rs/src/registry.rs:35-43`                                     | `pub fn select_rt(has_color, has_albedo, has_normal, hdr, srgb, clean_aux, quality) -> Result<ModelKey, OidnError>`. No `directional` parameter.            | Signature returns `Result`; `directional` removed.                                                    |
| M1 | —        | `crates/oidn-rs/src/registry.rs:48-52`                                     | `(!color && albedo && !normal && hdr)` → `Err(InvalidArgument("hdr mode not supported for albedo-only filtering"))`.                                       | Matches `_ref/oidn/core/unet_filter.cpp:423` rejection (string differs slightly — MED at most).        |
| M1 | —        | `crates/oidn-rs/src/registry.rs:53-57`                                     | `(!color && !albedo && normal && (hdr || srgb))` → `Err(InvalidArgument("hdr/srgb not supported for normal-only filtering"))`.                              | Matches `_ref/oidn/core/unet_filter.cpp:428`.                                                          |
| M1 | —        | `crates/oidn-rs/src/registry.rs:58-62`                                     | `(!color && albedo && normal)` → `Err(InvalidArgument("invalid combination of input features"))`.                                                            | Matches `_ref/oidn/core/unet_filter.cpp:434`.                                                          |
| M1 | —        | `crates/oidn-rs/src/registry.rs:64-81`                                     | Fall-through table; unmatched arms → `Err(UnsupportedFeatures)`. No `Ok(rt_alb)` silent return for HDR+albedo.                                              | Strict.                                                                                                |
| M1c| —        | `crates/oidn-rs/src/filters/rt.rs:506-514`                                 | Single call site invokes `select_rt(has_color, has_albedo, has_normal, self.hdr, self.srgb, self.clean_aux, self.quality)?` with no `directional`.        | Updated.                                                                                               |
| M2 | —        | `crates/oidn-rs/src/filters/rt.rs:739-743`                                 | `if self.hdr && self.srgb { return Err(InvalidArgument("hdr and srgb are mutually exclusive")); }` precedes the `build_commit_artifacts` call at line 753.   | Guard runs before registry/weights lookup.                                                            |
| M3 | —        | `crates/oidn-rs/src/filters/rt.rs:481-492`                                 | `transfer_kind(&self, has_color: bool)`. Body: `if !has_color → Linear; if hdr → PU; if srgb → Linear; else SRGB`.                                          | C++ ref `_ref/oidn/core/rt_filter.cpp:65` is `if (srgb \|\| (!color && normal)) Linear …`. See note. |
| M3c| —        | `crates/oidn-rs/src/filters/rt.rs:436,807`                                 | Both call sites (`commit_tensor_model`, `execute`) now compute `has_color = self.color.is_some() \|\| self.color_tensor.is_some()` and pass it in.          | Updated; no dangling no-arg call.                                                                      |
| 8  | —        | repo root                                                                  | `cargo check -p oidn-rs --lib` → `Finished dev profile [optimized + debuginfo] target(s) in 0.75s`. Clean.                                                  | Library compiles.                                                                                      |

### Note on M3 vs the C++ reference

C++ `RTFilter::newTransferFunc()` returns `Linear` for `srgb || (!color && normal)`, `PU` for `hdr`, else `SRGB`.

Rust currently returns `Linear` for `!has_color` (i.e. albedo-only OR normal-only), and otherwise mirrors the chain. This is a **strict superset** of the C++ Linear branch:

- `(!color && normal)` → Linear in both. Match.
- `(!color && albedo)` → Linear in Rust; the C++ doesn't enter `newTransferFunc()` for that path because the registry-equivalent `checkParams` already rejects `(albedo-only, hdr=true)` and the `(albedo-only, !hdr)` case yields `SRGB` in C++. The audit explicitly asked for `Linear` here (M3 spec line: "For (no color, only albedo) → Linear"). Rust meets the spec; **divergence from C++ is intentional per the task brief**, so verdict stays PASS. Severity LOW informational only.
- `(color, srgb)` → C++ Linear, Rust Linear. Match.
- `(color, hdr)` → C++ PU, Rust PU. Match.
- `(color, !hdr, !srgb)` → C++ SRGB, Rust SRGB. Match.

No behavior loss; no spurious SRGB for the previously-buggy `(only-normal, !hdr, !srgb)` case.

### Notes on Lomonosov's open follow-ups

- The report flags `weights.rs:121` doc-comment drift and pre-existing e2e test import breakage (`WgpuDevice` / `WgpuBackend`). Both are outside the scope of items H6/H7/M1/M2/M3 and out of this verification's mandate. No impact on ACCEPT verdict.
- I did not re-run integration tests; only `cargo check -p oidn-rs --lib` per item 8.

---

## Overall verdict

**ACCEPT.** All 5 claimed fixes (H6, H7, M1, M2, M3) are present in source, match the cited reference behavior (or the spec's intentional extension thereof), and the library compiles cleanly.
