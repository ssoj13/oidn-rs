# Mendeleev — TZA loader parity audit (Rust vs C++ reference)

Date: 2026-05-21
Agent: Mendeleev (strict parity auditor)
Scope: Rust `crates/oidn-tza/` vs C++ `_ref/oidn/core/tza.*` + `tensor*`

---

## Summary verdict

**PARITY** for the on-disk TZA binary format and the actually-used (file-resident) layout/dtype set. The Rust loader is byte-for-byte equivalent to `parseTZA` for every input the C++ pipeline accepts from disk.

Justification (one line): magic, version, table-offset, per-tensor encoding, dtype set, and the two layouts ever emitted into TZA (`x`, `oihw`) are all matched exactly; remaining C++ surface area (`paddedDims`, blocked GPU layouts, `chw`/`hwc`/`ohwi` accessors, `HostTensor`/`DeviceTensor` hierarchy, `isValid()` rank check) belongs to the runtime tensor-engine layer that is intentionally out of scope for an archive parser.

Caveat: a few minor robustness deltas exist (overflow checks, unread `minor`, `ndim==0` handling); see issue table.

---

## File-by-file findings

### `crates/oidn-tza/src/lib.rs`
- Doc comment at L3-5 accurately describes the layout matching `tza.cpp` L27-103.
- Re-exports `parse`, `TzaError`, `DType`, `Layout`, `Tensor`, `TensorDesc`, `TensorMap`. Public surface is intentionally narrower than C++ (no `HostTensor`/`DeviceTensor`/`Buffer`/`Engine` notion); appropriate for an archive-only crate.
- `#![forbid(unsafe_op_in_unsafe_fn)]` is good hygiene; no `unsafe` is used.

### `crates/oidn-tza/src/parser.rs`
- `MAGIC = 0x41D7` (L4) matches `tza.cpp:34` exactly.
- `SUPPORTED_MAJOR = 2` (L5) matches `tza.cpp:41`.
- Cursor implementation (L8-75) is bounds-checked on every read; mirrors `checkBounds`/`read<T>` in `tza.cpp:10-25`.
- Endianness: explicit `u16::from_le_bytes` / `u32::from_le_bytes` / `u64::from_le_bytes` (L50, L57, L64). C++ uses `memcpy` into typed value (`tza.cpp:22`) — i.e. **host endianness**. Rust is correct on all OIDN target platforms (x86_64, ARM64 — all LE); this is a *robustness improvement*, not a divergence, since OIDN ships only LE binaries. UNCERTAIN whether C++ would mis-parse on a BE host — moot in practice.
- Magic check (L107-110) matches `tza.cpp:33-35`.
- Version: reads major+minor, ignores minor (L113 `_minor`), rejects major != 2 (L114-116). Matches `tza.cpp:38-42` semantics including `UNUSED(minorVersion)`.
- Table offset: `u64` read + seek (L118-119) matches `tza.cpp:45-46`. Both implementations compute the table pointer from the start of the buffer (not from current cursor).
- Per-tensor loop (L124-175) matches `tza.cpp:53-100` field order: nameLen(u16), name bytes, ndim(u8), dims(u32×ndim), layout(char×ndim), dtype(char), dataOffset(u64).
- Layout string parsed via UTF-8 (L138-141) then matched (`x` / `oihw`); C++ uses `std::string(input, input+ndims)` raw bytes (`tza.cpp:74-80`). Equivalent for ASCII layout codes — both reject all other layouts on read.
- `expected_ndim(layout)` (L85-90) + `LayoutNdimMismatch` check (L142-148) is an extra safety check Rust adds. C++ does **not** validate ndim against layout at TZA-parse time; it relies on `TensorDesc::isValid()` which is invoked in the `TensorDesc(dims, layout, dataType)` constructor at `tensor_desc.h:33-37` via `assert`. Rust's behavior is stricter (errors out where C++ may assert in debug or silently allow ndim mismatch in release). **Tighter, parity-preserving for normal data.**
- Dtype decoding (L92-98) — `'f'`→Float32, `'h'`→Float16 — matches `tza.cpp:84-90`.
- Data-offset bounds check (L161-171) mirrors `checkBounds(tensorData, bufferEnd, getByteSize())` at `tza.cpp:95` and uses `checked_add` to guard overflow. Slightly stronger than C++ (which can wrap on `tensorOffset + byteSize` if attacker-controlled, although `ptrdiff_t` comparison is generally safe on 64-bit).
- Data is **copied** (`to_vec()` at L172) producing a `'static` owned buffer. C++ uses `HostTensor` with `shared=true` and a non-owning pointer into the source buffer (`tza.cpp:98`, `tensor.cpp:136-139`). See Issue M-01.
- `BTreeMap` (L122, `types.rs:81`) gives deterministic iteration; C++ uses `std::unordered_map` (`tensor.h:117`). Functionally equivalent for lookup; Rust is more debug-friendly. Lookup characteristics differ (O(log n) vs O(1) amortized) — irrelevant for ~30 tensors.

### `crates/oidn-tza/src/types.rs`
- `Layout` enum (L7-13) only contains `X` and `Oihw`. C++ enum `TensorLayout` (`tensor_layout.h:11-33`) lists 15 variants. Of those, **only `x` and `oihw` are emitted by `_ref/oidn/training/`** to TZA; the rest are runtime-only blocked variants used by the inference engine. Rust correctly omits them from the archive parser. See Issue L-01.
- `DType` enum (L15-19): `Float32`, `Float16`. C++ `DataType` likely also has `UInt8`/`Int32`/`Undefined` (not in scope files, but TZA only handles `'f'` and `'h'`). Matches the TZA on-disk set.
- `byte_size()` (L22-27): 4 / 2 — correct.
- `TensorDesc` (L30-45): holds `dims`, `layout`, `dtype`. **Missing `paddedDims`** which exists in C++ `tensor_desc.h:21`. C++ sets `paddedDims = dims` at parse time (`tza.cpp:70`), so for TZA contents the two are always identical. Padded dims are only used by runtime blocked-layout storage allocations. See Issue L-02.
- `num_elements()` and `byte_size()` (L38-44) compute over `dims` directly. C++ `getNumElements()` uses `dims` (`tensor_desc.h:122-130`) but `getByteSize()` uses `paddedDims` (L133-141). For TZA-resident tensors where padded==logical, results are equal. Identical behavior in scope.
- `Tensor` (L48-77): owns `Vec<u8>`. Adds convenience: `as_f32`, `as_f16`, `to_f32_vec` (with on-the-fly f16→f32 conversion). C++ has no equivalent decoder helper in the parser path; conversion happens in `network.cpp` or downstream engines. Reasonable additive functionality; no parity violation.
- `TensorMap = BTreeMap<String, Tensor>` (L81) vs C++ `std::unordered_map<std::string, Ref<Tensor>>` (`tensor.h:117`). Type/semantic equivalent (string→tensor), differs in ordering and ownership model (Rust value, C++ ref-counted pointer).

### `crates/oidn-tza/src/error.rs`
- `OutOfBounds { offset, need, have }` — covers `checkBounds` failures (`tza.cpp:10-14`). Richer than C++'s single `"invalid or corrupted weights blob"` string.
- `BadMagic`, `UnsupportedVersion`, `InvalidLayout`, `InvalidDtype`, `BadName`, `LayoutNdimMismatch` — all C++ failure paths are covered.
- `BadName(FromUtf8Error)`: C++ does not validate UTF-8; it builds `std::string` from raw bytes. Rust is stricter. Real TZA names are ASCII (PyTorch state_dict keys like `enc_conv0.weight`), so divergence cannot manifest on real archives. INFO-only.

### `crates/oidn-tza/tests/parse_all_weights.rs`
- Locates `data/weights/*.tza` and parses each (L14-35). Threshold `>= 20` covers the ~24 shipped models.
- `rt_hdr_has_expected_layer_set` (L38-64): verifies tensor count (32), names, shape `[32,3,3,3]`, layout `Oihw`, dtype `Float16`. Good golden test. UNCERTAIN whether all 24 weight files have been actually verified to contain only `x`+`oihw`; the broad test would have caught a missing layout because it `unwrap`s parse — implicit coverage.

### `_ref/oidn/core/tza.cpp` cross-reference points
- `tza.cpp:33-35`  magic ↔ `parser.rs:107-110`
- `tza.cpp:38-42`  version ↔ `parser.rs:112-116`
- `tza.cpp:45-46`  tableOffset+seek ↔ `parser.rs:118-119`
- `tza.cpp:49`     numTensors u32 ↔ `parser.rs:121`
- `tza.cpp:58-61`  name u16+bytes ↔ `parser.rs:126-128`
- `tza.cpp:64-69`  ndim u8 + dims u32×n ↔ `parser.rs:131-135`
- `tza.cpp:74-80`  layout chars + match ↔ `parser.rs:138-148`
- `tza.cpp:84-90`  dtype char ↔ `parser.rs:151-152`
- `tza.cpp:93-96`  data offset + bounds ↔ `parser.rs:155, 161-172`

### `_ref/oidn/core/tensor.h` / `tensor.cpp`
- `HostTensor` (`tensor.h:87-101`, `tensor.cpp:131-153`): zero-copy aliasing of source buffer (`shared=true`). Rust copies. See Issue M-01.
- `Tensor::getHash()` / `dump()` (`tensor.cpp:47-125`): wrapped in `#if 0` — dead code in reference. Not expected in Rust port. INFO.
- `DeviceTensor` and `Engine`/`Buffer` interactions: out of scope for an archive parser — Rust correctly omits.

### `_ref/oidn/core/tensor_layout.h`
- 15 layout variants (L11-33); only `x` and `oihw` ever appear in TZA archives (the rest are constructed at runtime by the inference engine for blocked GPU storage). Rust's two-variant enum is the exact set needed by the TZA parser.

---

## Per-issue table

| ID    | Severity | Rust location                          | Ref location                                    | Description                                                                                                                                                          | Recommended fix                                                                                                                  |
|-------|----------|----------------------------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| M-01  | MED      | `parser.rs:172` (`to_vec()`)           | `tza.cpp:98`; `tensor.cpp:136-139`              | C++ aliases the source buffer (`shared=true`, zero-copy). Rust copies every tensor blob, doubling peak memory at load. Acceptable for ~30MB weight files.            | Optional: expose a borrowed variant `parse_borrowed<'a>(&'a [u8]) -> ...<Tensor<'a>>`. Keep `parse` as the owned API.             |
| M-02  | MED      | `parser.rs:131` (ndim read, no bound)  | `tza.cpp:64`                                    | `ndim: u8` accepted as-is (0..=255). C++ has no explicit upper bound either, but `TensorDesc::isValid()` (`tensor_desc.h:39-50`) constrains rank via layout info.    | Reject `ndim == 0` early and enforce `ndim == expected_ndim(layout)` (already done at L142 — moved earlier would short-circuit a malicious huge ndim before the dim-vec allocation). |
| L-01  | LOW      | `types.rs:7-13`                        | `tensor_layout.h:11-33`                         | Rust enum only has `X` and `Oihw`. C++ enum has 13 more (blocked/runtime). For archive parsing this is correct; future TZA writers might emit `chw`/`hwc`/`ohwi`.    | Document in `types.rs` doc comment which subset is intentionally supported (already partially done at L3-6).                     |
| L-02  | LOW      | `types.rs:30-35` (`TensorDesc`)        | `tensor_desc.h:18-32` (`paddedDims`)            | Rust drops `paddedDims`. For TZA contents `paddedDims == dims` always; downstream Rust runtime will need to compute padded dims when allocating blocked storage.     | When the inference engine is ported, add `paddedDims` to a runtime descriptor (likely in a different crate); leave parser as-is. |
| L-03  | LOW      | `parser.rs:131`                        | `tza.cpp:64-69`                                 | Rust allocates `Vec::with_capacity(ndim)` from an attacker-controlled u8. Capped at 255 entries × 4 bytes = harmless, but allocation precedes layout validation.     | Reorder: read all dims into a fixed-size array, validate layout/ndim, then construct `dims` vec. Or accept current order (safe). |
| L-04  | LOW      | `parser.rs:155` (`as usize`)           | `tza.cpp:93` (`uint64_t`)                       | `data_offset = u64 as usize` truncates on 32-bit targets. TZA files >4 GiB impossible in practice; not a real risk for OIDN models.                                  | Add `try_from` + dedicated `OffsetTooLarge` error for hygiene, or document the 32-bit limitation.                                |
| L-05  | LOW      | `parser.rs:121` (`n_tensors`)          | `tza.cpp:49`                                    | `numTensors = u32 as usize` not validated against remaining table length before the per-tensor loop. A bogus huge value would simply exhaust the buffer in-loop.    | Optional: sanity-cap (e.g. `numTensors * MIN_ENTRY_BYTES <= bytes.len() - table_offset`).                                        |
| I-01  | INFO     | `error.rs:21` (`BadName(FromUtf8)`)    | `tza.cpp:60` (`std::string` raw)                | Rust enforces UTF-8 on tensor names; C++ does not. Real names are ASCII.                                                                                              | No change.                                                                                                                       |
| I-02  | INFO     | `parser.rs:113` (`_minor`)             | `tza.cpp:39-40` (`UNUSED(minorVersion)`)        | Rust discards minor; matches C++. Future format extension may want to expose it.                                                                                     | Optional: store `(major, minor)` in result metadata.                                                                              |
| I-03  | INFO     | `types.rs:81` (`BTreeMap`)             | `tensor.h:117` (`unordered_map`)                | Different map type. No correctness impact; iteration order differs from C++.                                                                                          | No change.                                                                                                                       |
| I-04  | INFO     | `parser.rs:152` (`dtype_byte as char`) | `tza.cpp:84`                                    | Casting `u8` to `char` only handles ASCII; same as C++ which reads `char` directly. No issue.                                                                         | No change.                                                                                                                       |

Note: items previously considered HIGH (endianness, dtype-set, magic) were verified PARITY after reading both implementations side-by-side.

---

## Suggested deduplication / consolidation opportunities

1. `expected_ndim()` (`parser.rs:85-90`) could move into `Layout::expected_ndim()` in `types.rs` so other crates can query the relationship without duplicating the match.
2. `decode_layout` / `decode_dtype` are only used once each; folding them into `Layout::from_str` (`FromStr` impl) and `DType::try_from(u8)` would expose them publicly and replace the ad-hoc helpers.
3. The bounds-check pattern `data_offset.checked_add(byte_size)...` could be lifted into `Cursor::bounds_check_absolute(offset, len)` for reuse if more fields are added.
4. Per Issue M-01, expose `Tensor::data` as a borrowed slice variant; the owned `Vec<u8>` can be reduced to one allocation via `Box<[u8]>` to save Vec metadata.

---

## FIXMEs / TODOs / dead code spotted in scope

- Rust source: no `TODO`/`FIXME`/`XXX` markers, no commented-out code, no dead functions. Clean.
- Doc comment in `lib.rs:3` references `_ref/oidn/core/tza.cpp`; ensure that path stays valid (currently `_ref/oidn` is `C:/projects/projects.rust.cg.offload/oidn`). Path string is informational only.
- C++ ref: `tensor.cpp:47-125` (`getHash`/`dump`) is fenced with `#if 0` — intentional dead code in the reference; not expected in the Rust port. No action.
- C++ ref: `Tensor::operator ispc::TensorAccessor1D()` etc. (`tensor.h:64-68`) are ISPC bindings — runtime concerns, intentionally out of scope.

---

## Open questions for the orchestrator

1. Should the Rust crate expose a zero-copy `parse_borrowed` API (Issue M-01)? Owned-by-default is ergonomic for callers but doubles working set during model load. Recommendation: defer until the inference crate exists and we know its lifetime model.
2. When the inference engine is ported, where should `paddedDims` live? Suggested: in a separate `oidn-runtime` crate's `RuntimeTensorDesc`, leaving `oidn-tza::TensorDesc` as the on-disk descriptor only.
3. Is there appetite for round-trip TZA *writing* (currently only `parse` is exposed)? C++ has no writer in `tza.h`; a Python writer lives in `_ref/oidn/training/`. Not required for parity with C++ core.
4. Do we want stronger up-front validation (Issues M-02, L-03, L-05) as a hardening pass against malicious/corrupted files, given that TZA blobs may be downloaded from URLs?
5. Should `Layout` and `DType` round-trip via `Display` so the error messages emit the same strings the format uses on disk (`"x"`, `"oihw"`, `'f'`, `'h'`)? Currently `InvalidLayout { got: String }` echoes the offending bytes only.

UNCERTAIN: confirmation that all 24 shipped `.tza` files use only `x` + `oihw` layouts. Evidence is indirect (the broad `parse_all_shipped_tza_files` test passes per crate intent), but a direct enumeration was not performed in this read-only audit. Recommendation: extend the existing test to assert `desc.layout` ∈ {X, Oihw} for every tensor in every file to convert this from indirect to direct coverage.
