# Vavilov — Public API Surface Parity Audit (oidn-rs vs Intel OIDN C++ reference)

- Agent: Vavilov
- Date: 2026-05-21
- Scope: Device, Filter, Buffer, Error, Weights, Quality, Progress, Versioning, Cargo features
- Method: Read + Grep only; no edits / no builds
- Reference snapshot: Intel OIDN v2.4.1 (cmake/oidn_version.cmake:4-6)

## Verdict

**API parity: PARTIAL / DIVERGENT BY DESIGN.**

oidn-rs is not a 1:1 binding of the OIDN C ABI. It is a *Rust-idiomatic re-implementation* on top of Burn/wgpu that intentionally drops the C-handle world (`oidnNewDevice`, `oidnRetain*`, `oidnRelease*`, `oidnSetFilterImage(..., "color", ...)`) in favour of typed builders (`RtFilter::builder(...).hdr(true).build()`). README explicitly states this (README.md:115-129).

Consequences (all observed, with citations below):
- No `Buffer` type at all (sharedBuffer / externalMemory unsupported)
- No `Device` for CPU/CUDA/HIP/Metal/SYCL — only `WgpuDevice` (device.rs:22) which is not even used by the `RtFilter` (it takes a raw `B::Device`, rt.rs:291)
- `OidnError` variants do not map 1:1 to `OIDNError` (error.rs:3-34)
- No version query, no error callback, no `oidnGetDeviceError`, no async execution, no `oidnSyncDevice`
- Filter has typed setters (`set_color`, `set_albedo`, `set_normal`) but no generic string-name `setFilterImage` or `setFilterData("weights", ...)`
- Quality enum is missing `Default` (filter.rs:8-16)
- Progress callback uses `f32` while ref uses `double` (rt.rs:466 vs oidn.h:386)
- README "Library use" snippet imports a path that does not match the implemented signature

The port is functional for its declared scope (RT + RTLightmap denoising on Burn) but anyone porting C/C++ OIDN code expecting C-ABI parity will find most of the surface absent.

---

## 1. Device API matrix

| Reference symbol (oidn.h) | Rust counterpart | Status | Location |
|---|---|---|---|
| `OIDNDeviceType` enum (Default/CPU/SYCL/CUDA/HIP/Metal) | — (no enum) | MISSING | oidn.h:80-90 |
| `oidnNewDevice(OIDNDeviceType)` | `WgpuDevice::new()` | PARTIAL — wgpu-only, no type selection | device.rs:31 |
| `oidnGetNumPhysicalDevices` / `oidnGetPhysicalDevice*` | — | MISSING | oidn.h:55-74 |
| `oidnNewDeviceByID/UUID/LUID/PCIAddress` | — | MISSING | oidn.h:131-141 |
| `oidnNewSYCLDevice/CUDADevice/HIPDevice/MetalDevice` | — | MISSING | oidn.h:147-162 |
| `oidnIsCPUDeviceSupported / IsCUDA / IsHIP / IsMetal / IsSYCL` | — | MISSING | oidn.h:111-125 |
| `oidnRetainDevice` / `oidnReleaseDevice` | (Rust drop/clone) | N/A — RAII handled by `WgpuDevice: Clone` | device.rs:21 |
| `oidnSetDeviceBool("verbose", …)` | — | MISSING (no verbose toggle) | oidn.h:171, device.cpp:206 |
| `oidnSetDeviceInt("numThreads", …)` | — | MISSING | apps/oidnDenoise.cpp:245 |
| `oidnSetDeviceInt("setAffinity", …)` | — | MISSING | apps/oidnDenoise.cpp:247 |
| `oidnGetDeviceInt("version" / "versionMajor"/…)` | — | MISSING | device.cpp:184-191 |
| `oidnGetDeviceInt("externalMemoryTypes")` | — | MISSING | device.cpp:198 |
| `oidnGetDeviceInt("systemMemorySupported" / "managedMemorySupported")` | — | MISSING | device.cpp:194-197 |
| `oidnSetDeviceErrorFunction(func, userPtr)` | — | MISSING (no per-device error sink) | oidn.h:219 |
| `oidnGetDeviceError(device, &message)` | — | MISSING — Rust returns `Result<_, OidnError>` per-call | oidn.h:225 |
| `oidnCommitDevice` | — | NOT IMPLEMENTED (no device-level commit; the filter `commit()` does both) | device.cpp:222 |
| `oidnSyncDevice` | — | MISSING (no wait_for_completion API) | oidn.h:232 |

## 2. Buffer API matrix

| Reference symbol | Rust counterpart | Status | Location |
|---|---|---|---|
| `OIDNBuffer` handle | — | MISSING ENTIRELY | oidn.h:314 |
| `oidnNewBuffer / NewBufferWithStorage` | — | MISSING | oidn.h:317-320 |
| `oidnNewSharedBuffer(device, devPtr, size)` | — | MISSING | oidn.h:323 |
| `oidnNewSharedBufferFromFD / FromWin32Handle / FromMetal` | — | MISSING | oidn.h:326-337 |
| `OIDNStorage` (HOST/DEVICE/MANAGED) | — | MISSING | oidn.h:259-269 |
| `OIDNExternalMemoryTypeFlag` (10 variants) | — | MISSING | oidn.h:274-310 |
| `oidnReadBuffer / WriteBuffer / Async` | — | MISSING | oidn.h:351-363 |
| `oidnGetBufferData` / `Size` / `Storage` | — | MISSING | oidn.h:340-348 |
| `oidnRetainBuffer / ReleaseBuffer` | — | N/A (Rust RAII) | oidn.h:366-369 |

Note: oidn-rs goes directly from `Image<'_>` slices (CPU side) → Burn `Tensor` (device side), bypassing the OIDN buffer abstraction entirely. Acceptable architectural choice but means **zero-copy interop with externally-owned GPU memory is not possible** — anyone trying to denoise a render target produced by an existing wgpu/Vulkan/D3D12 pipeline must round-trip through host memory.

## 3. Filter API matrix

| Reference symbol | Rust counterpart | Status | Location |
|---|---|---|---|
| `oidnNewFilter(device, "RT")` | `RtFilter::<B>::builder(device, weights_dir).build()` | RENAMED (typed) | rt.rs:291 |
| `oidnNewFilter(device, "RTLightmap")` | `RtLightmapFilter::<B>::builder(...)` | RENAMED (typed) | rtlightmap.rs:128 |
| `oidnNewFilter(device, "...")` arbitrary name | — | MISSING (no dynamic filter registry) | api.cpp filter creation table |
| `oidnSetFilterImage("color", …)` | `set_color(&Image)` | RENAMED | rt.rs:305 |
| `oidnSetFilterImage("albedo", …)` | `set_albedo(&Image)` | RENAMED | rt.rs:312 |
| `oidnSetFilterImage("normal", …)` | `set_normal(&Image)` | RENAMED | rt.rs:319 |
| `oidnSetFilterImage("output", …)` | `allocate_output(w, h, fmt)` + `take_output()` | DIVERGENT — Rust always allocates internally | rt.rs:379, 409 |
| `oidnSetSharedFilterImage` | — | MISSING (no zero-copy image) | oidn.h:410 |
| `oidnUnsetFilterImage(name)` | — | MISSING | oidn.h:417 |
| `oidnSetFilterBool("hdr", v)` | `.hdr(v)` builder | RENAMED | rt.rs:73, rt_filter.cpp:107 |
| `oidnSetFilterBool("srgb", v)` | `.srgb(v)` builder | RENAMED | rt.rs:77, rt_filter.cpp:109 |
| `oidnSetFilterBool("cleanAux", v)` | `.clean_aux(v)` builder | RENAMED | rt.rs:85, rt_filter.cpp:111 |
| `oidnSetFilterBool("directional", v)` | `.directional(v)` | RENAMED | rt.rs:81 |
| `oidnSetFilterInt("quality", q)` | `.quality(Quality)` | RENAMED | rt.rs:89, unet_filter.cpp:45 |
| `oidnSetFilterInt("maxMemoryMB", n)` | `.max_memory_mb(i32)` | RENAMED | rt.rs:111, unet_filter.cpp:55 |
| `oidnSetFilterFloat("inputScale", s)` | `.input_scale(Option<f32>)` | RENAMED | rt.rs:93, unet_filter.cpp:89 |
| `oidnGetFilterBool/Int/Float(name)` | — | MISSING (no getter parity; `model_key()` is closest, rt.rs:416) | oidn.h:451-487 |
| `oidnSetSharedFilterData("weights", ptr, size)` | `.weights(impl Into<Vec<u8>>)` builder | RENAMED — but **owned `Vec<u8>` copy** not zero-copy | rt.rs:103, unet_filter.cpp:15-35 |
| `oidnUpdateFilterData("weights")` | — | MISSING — must rebuild filter | oidn.h:430 |
| `oidnUnsetFilterData(name)` | — | MISSING | oidn.h:433 |
| `oidnSetFilterProgressMonitorFunction(func, userPtr)` | `set_progress(impl FnMut(f32) -> bool)` | DIVERGENT — `f32` vs `double`, no `userPtr` (closure captures), only on `RtFilter` not `RtLightmapFilter` | rt.rs:466 vs oidn.h:386, 496 |
| `oidnCommitFilter(filter)` | `Filter::commit(&mut self)` | RENAMED | filter.rs:22, rt.rs:724 |
| `oidnExecuteFilter(filter)` | `Filter::execute(&mut self)` | RENAMED | filter.rs:25 |
| `oidnExecuteFilterAsync(filter)` | — | MISSING (no async path) | oidn.h:507 |
| `oidnExecuteSYCLFilterAsync` | — | MISSING (no SYCL) | oidn.h:512 |
| `oidnRetainFilter / oidnReleaseFilter` | — (RAII) | N/A | oidn.h:395-398 |

### Filter parameters NOT reachable from Rust
- `weight` (alternate spelling, see unet_filter.cpp:55) — only `weights` is recognised in Rust builder
- Progress monitor on `RtLightmapFilter` — rtlightmap.rs has no `set_progress` (filter.rs:25 only requires `commit`/`execute`, no progress)
- Per-filter `set_progress` after commit — only pre-commit on `RtFilter`; no way to install progress mid-flight on the `CommittedRtFilter` (rt.rs:622-651)

## 4. Error code matrix

Reference (`oidn.h:93-102`):

| OIDN C enum | Numeric | Rust `OidnError` variant | Status |
|---|---:|---|---|
| `OIDN_ERROR_NONE` | 0 | — (Rust uses `Ok(())`) | OK — Result convention |
| `OIDN_ERROR_UNKNOWN` | 1 | — | **MISSING** |
| `OIDN_ERROR_INVALID_ARGUMENT` | 2 | `InvalidArgument(&'static str)` | OK | error.rs:32 |
| `OIDN_ERROR_INVALID_OPERATION` | 3 | — | **MISSING** (could fold into `Unset`/`Inconsistent` but not 1:1) |
| `OIDN_ERROR_OUT_OF_MEMORY` | 4 | — | **MISSING** |
| `OIDN_ERROR_UNSUPPORTED_HARDWARE` | 5 | — | **MISSING** |
| `OIDN_ERROR_CANCELLED` | 6 | `Cancelled` | OK | error.rs:30 |

Rust-specific variants (not in C ABI):
- `Unset(&'static str)` — error.rs:5
- `Inconsistent(&'static str)` — error.rs:8
- `UnsupportedFeatures` — error.rs:11
- `MissingModel(PathBuf)` — error.rs:14
- `Io(std::io::Error)` — error.rs:17
- `Tza(oidn_tza::TzaError)` — error.rs:20
- `Load(oidn_model::LoadError)` — error.rs:23
- `Device(String)` — error.rs:26

Verdict on errors: **NOT 1:1** — neither the OIDN canonical 7 nor any compatibility numeric mapping. A consumer porting C++ code that branches on `OIDNError` cannot do so. Recommend either (a) honest mapping (`Unknown`, `InvalidOperation`, `OutOfMemory`, `UnsupportedHardware`) or (b) explicit doc note that error taxonomy is intentionally Rust-native.

## 5. Feature flag matrix (CMake vs Cargo)

| CMake option (CMakeLists.txt:27-46) | Cargo feature (oidn-rs/Cargo.toml:23-36) | Status |
|---|---|---|
| `OIDN_DEVICE_CPU` (ON) | — | Implicit — Burn `ndarray` backend covers CPU but not exposed as a feature |
| `OIDN_DEVICE_SYCL` (OFF) | — | MISSING |
| `OIDN_DEVICE_CUDA` (OFF) | — | MISSING (Burn supports CUDA, not enabled in workspace deps) |
| `OIDN_DEVICE_HIP` (OFF) | — | MISSING |
| `OIDN_DEVICE_METAL` (OFF) | — | wgpu backend implicitly covers Metal (no explicit feature) |
| `OIDN_FILTER_RT` (ON) | — (always compiled) | OK — implicit |
| `OIDN_FILTER_RTLIGHTMAP` (ON) | — (always compiled) | OK — implicit |
| — | `embed-hdr` / `embed-ldr` / `embed-aov` / `embed-aux-clean` / `embed-lightmap` / `embed-all` | Rust-only — weight embedding, not in ref |

Notable: Cargo features only control **which TZA blobs are baked into the binary**, not which device or filter backends exist. Device backend selection happens at the Burn generic-parameter level (`RtFilter::<NdArray<f32>>::...` vs `RtFilter::<Wgpu>::...`) — not via a feature flag, which is the correct Rust-idiomatic pattern but worth documenting in README.

## 6. Weights API

| Ref behaviour | Rust behaviour | Status |
|---|---|---|
| `oidnSetSharedFilterData("weights", ptr, size)` — pass any `.tza` blob, ref-borrowed, host-readable | `.weights(impl Into<Vec<u8>>)` — **owned copy** | DIVERGENT — extra alloc, no zero-copy |
| Built-in registry picks weights from compiled set (`OIDN_FILTER_RT/RTLIGHTMAP`) | `registry::select_rt(...)` + `weights::resolve(...)` reading `data/weights/*.tza` from disk or `include_bytes!` embedded blob | DIVERGENT but explicit (registry.rs, weights.rs) |
| `oidnUpdateFilterData("weights")` to swap weights post-commit | — | MISSING |
| Quality routing (unet_filter.cpp:446-459) | `quality_candidates()` (registry.rs:65-72) | OK — explicitly mirrored |

`Quality` enum parity:
- Ref (oidn.h:376-383): `Default=0`, `Fast=4`, `Balanced=5`, `High=6`
- Rust (filter.rs:7-16): `High` (default), `Balanced`, `Fast`
- **MISSING: `Default` variant.** Rust just makes `High` the `#[default]`. Acceptable.
- **Numeric values not preserved** — fine because no FFI.

## 7. Progress callback parity

| Aspect | Reference (oidn.h:386, 496-497) | Rust (rt.rs:466) |
|---|---|---|
| Signature | `bool(void* userPtr, double n)` | `FnMut(f32) -> bool + 'static` |
| Precision | `double` | **`f32`** — narrower |
| User data | `void* userPtr` | Closure captures (better) |
| Return semantics | `false` → cancel | Same — confirmed in `tests/api_surface.rs:94-101` |
| Available on RtFilter | yes | yes (rt.rs:466) |
| Available on RtLightmapFilter | yes (same C API) | **NO** (rtlightmap.rs has no `set_progress`) |

## 8. Version / runtime query

| Reference | Rust |
|---|---|
| `OIDN_VERSION_MAJOR/MINOR/PATCH/STRING` macros (config.h.in:6-11) — value 2.4.1 | None |
| `oidnGetDeviceInt("version" / "versionMajor"/...)` (device.cpp:184-191) | None |

Audit: Rust crate exposes its own version via `Cargo.toml` only (`version = "0.1.0"`). No public `VERSION` constant, no runtime version query equivalent. Worth adding a `pub const OIDN_REFERENCE_VERSION: (u32,u32,u32) = (2,4,1);` to document which upstream snapshot the port targets.

## 9. Thread-safety / Send + Sync

Reference documents (oidn.h:222-225 — per-thread global errors; api.cpp:24-28 — error sink shared across threads guarded inside Device) that:
- Device methods are thread-safe after commit
- `oidnGetDeviceError` reads a thread-local slot
- Filters can be invoked on different threads provided the device was committed first

Rust side:
- `RtFilter<'b, B: Backend>` holds `&'b B::Device`, `Vec<u8>` weights, `Option<OwnedImage>` etc. — `Send`/`Sync` is **not asserted** anywhere
- `set_progress` stores `Box<dyn FnMut(f32) -> bool + 'static>` (no `Send` bound) — means filter cannot be sent across threads with a progress callback installed even if `B::Device: Send + Sync`
- No documentation re: thread-safety guarantees

Verdict: **Send/Sync semantics undocumented and likely accidentally tighter than ref.** This is an API hygiene gap, not a correctness bug.

## 10. Per-issue table

| # | Severity | Issue | Location |
|---|---|---|---|
| V01 | High | No `OidnError` variants for `Unknown`, `InvalidOperation`, `OutOfMemory`, `UnsupportedHardware` — error parity claim cannot be honoured | error.rs:3-34 |
| V02 | High | Progress callback signature uses `f32` instead of OIDN-standard `double`; precision loss is observable for very long inferences | rt.rs:466 vs oidn.h:386 |
| V03 | High | `RtLightmapFilter` is missing `set_progress` — feature regression vs `RtFilter` and ref C API | rtlightmap.rs:127-150 |
| V04 | Med | `.weights(impl Into<Vec<u8>>)` copies the entire blob (10–48 MB) — ref takes `void*`/`size` with no copy | rt.rs:103 |
| V05 | Med | No equivalent of `oidnSetSharedFilterImage` / `oidnSetSharedFilterData` (zero-copy device pointer) — every input/output round-trips through `Vec<u8>` / `Tensor` host alloc | rt.rs:305-321, 379 |
| V06 | Med | `WgpuDevice` is declared but **not consumed by any filter** — `RtFilter::builder` takes `&B::Device` directly, so `WgpuDevice` is dead weight in the public API. Confusing for users | device.rs:22, rt.rs:291 |
| V07 | Med | README "Library use" snippet on README.md:54-69 uses `&device.handle` for `RtFilter::builder` but elsewhere docs show `&dir`/`&device` mismatched. Verify quickstart compiles | README.md:58 |
| V08 | Med | No `Quality::Default` (ref OIDN exposes it for "let the library decide"). Rust forces caller to pick High/Balanced/Fast. Minor ergonomic gap | filter.rs:7-16, oidn.h:378 |
| V09 | Med | No version constant or build-time string identifying which upstream OIDN version was ported | (absent) |
| V10 | Med | No `Send + Sync` annotations / docs on `RtFilter`, `RtLightmapFilter`, `WgpuDevice` — thread-safety contract undefined | rt.rs:159, device.rs:22 |
| V11 | Low | Cargo features for embedded weights are documented in `src/weights.rs` but not summarised in README — users have to read source to discover `embed-all` | weights.rs:9-22, README.md |
| V12 | Low | No `oidnUnsetFilter*` / "reset filter param" pattern — once set on builder, can't unset. Acceptable for builder model but flag for docs | rt.rs:73-127 |
| V13 | Low | `prelude.rs` re-exports `WgpuDevice` and `WgpuBackend` (wgpu types) even in builds where Burn-wgpu is irrelevant; making prelude backend-agnostic would be cleaner | prelude.rs:1-5 |
| V14 | Low | `OidnError::Device(String)` carries a `String` rather than a typed error — defeats `thiserror` source chaining | error.rs:26 |
| V15 | Low | `ModelKey(pub String)` — public tuple field exposed; consider sealing | registry.rs:13 |
| V16 | Low | `set_color`, `set_albedo`, `set_normal` mutate self and silently invalidate prior commit (`self.committed = false`) — works but should be documented as part of trait contract | rt.rs:305-321 |
| V17 | Low | `CommittedRtFilter` exists as a distinct type (rt.rs:206, 622) but is unused by the `Filter` trait — exposes a type-state pattern that's only half-implemented (state collapses back inside `RtFilter::execute`) | rt.rs:206, filter.rs:20-26 |
| V18 | Info | README claims "every shipped TZA model loads and runs" — supported by `all_models_smoke.rs` smoke test; not a parity issue, recorded for traceability | README.md:7-10 |

## 11. Public API ergonomics (Rust API Guidelines)

| Guideline | Status | Note |
|---|---|---|
| C-NAMING — snake_case | OK | `set_color`, `clean_aux`, `max_memory_mb` |
| C-CONV — `Result<T, E>` for fallible ops | OK | `commit()`/`execute()` both return `Result` (filter.rs:22-25) |
| C-CTOR — `new` returns `Self` | PARTIAL | `WgpuDevice::new() -> Result<Self>` (device.rs:31). `OidnError` from `new` is unreachable today (always `Ok`); could be `Self` directly |
| C-BUILDER — typed builder | OK | `RtFilterBuilder` returns moved `self` on each chain step (rt.rs:73-127) |
| C-SEND-SYNC — explicit `Send + Sync` | MISSING | See V10 |
| C-DEREF / smart-pointer abuse | OK | None used |
| C-DEBUG / C-COMMON-TRAITS | PARTIAL | `OidnError: Debug + thiserror::Error` (OK); `RtFilter` not `Debug` |
| C-FAILURE — error type is non-exhaustive | NOT FOLLOWED | `OidnError` lacks `#[non_exhaustive]` so adding variants is breaking (error.rs:3) |
| C-CALLER-CONTROL — generic over types caller picks | OK | `B: Backend` parameter is canonical Burn idiom |
| C-OWN-SUFFIX / drop semantics | OK | All resources are Rust-owned; no manual `release_*` |
| C-NEWTYPE — `ModelKey` wraps `String` | PARTIAL | `pub String` tuple field — should be private with `as_str` |
| C-RE-EXPORT — `prelude` | PARTIAL | Reasonable picks but `WgpuBackend`/`WgpuDevice` leak wgpu into preludes meant to be backend-generic. Consider `prelude::wgpu` submodule |

## 12. Public reexports — what's exported vs hidden

`lib.rs:35-41` exports: `WgpuDevice`, `OidnError`, `Filter`, `Quality`, `CommittedRtFilter`, `RtFilter`, `RtLightmapFilter`, `Image`, `ImageMut`, `PixelFormat`, `ModelKey`.

`prelude.rs` adds: `WgpuBackend`.

Not exported but possibly needed by users:
- `registry::select_rt` / `quality_candidates` — only accessible via full path (`oidn_rs::registry::select_rt`). The module is `pub mod registry;` so accessible, but no prelude entry
- `weights::resolve` / `weights::embedded` — same situation
- `tile::*` constants used in `api_surface.rs:172-179` — `tile::RECEPTIVE_FIELD_BASE`, `tile::MIN_TILE_ALIGNMENT`, `tile::DEFAULT_MAX_TILE_SIZE`, `tile::plan`. None in prelude
- `image::Image::from_rgb_f32` — used everywhere, only reachable through `Image` re-export. OK.
- `RtFilterBuilder`, `RtLightmapFilterBuilder` — not re-exported. Most users only need `RtFilter::builder()` (returns the builder type), so OK in practice
- `autoexposure`, `color`, `gpu_ops`, `image_tensor` modules — `pub mod` but no items in prelude; treat as advanced API

## 13. README claims vs reality

| Claim | Reality |
|---|---|
| "every shipped TZA model loads and runs" (README.md:7) | Backed by `tests/all_models_smoke.rs` |
| "Burn + wgpu" + cross-vendor | True — `WgpuBackend = burn_wgpu::Wgpu<f32, i32>` (device.rs:10) |
| Quickstart `RtFilter::<WgpuBackend>::builder(&device.handle, "data")` (README.md:58) | Signature matches `rt.rs:291` — OK |
| "the same `RtFilter<B>` / `RtLightmapFilter<B>` works generically over any Burn backend — tests use `burn::backend::NdArray<f32>`" (README.md:75-77) | Consistent with `tests/api_surface.rs:38` |
| "Library use" shows `RtFilter::<WgpuBackend>::builder(&device.handle, "data")` but does NOT mention `embed-*` features as an alternative to `"data"` path | Minor doc gap |
| Performance table @ "default wgpu DX12 backend" | Not verified — not an API parity concern |

## 14. Open questions for follow-up

1. Is the OIDN C ABI parity an *explicit non-goal*? README hints yes ("No C-ABI / no `libOpenImageDenoise.dll` shim"). If so, recommend adding `## Compatibility` section enumerating which surfaces are intentionally absent (Buffer, Device.sync, ErrorFunction, AsyncExecute, SharedBuffer, ExternalMemory).
2. Should `OidnError` map ref error codes 1:1 (add `Unknown`, `InvalidOperation`, `OutOfMemory`, `UnsupportedHardware`) for downstream diagnostic UX? Or keep Rust-native taxonomy and document the mapping?
3. Why is `WgpuDevice` exported when no filter consumes it? Either (a) wire it through `RtFilter::builder` or (b) delete it / move to examples.
4. Should `RtLightmapFilter` have feature parity with `RtFilter` (progress callback, `weights()` blob, `nan_to_zero`)? Currently missing all three (rtlightmap.rs:33-49).
5. Should `Filter` trait include `set_progress` so the abstraction holds for both filter types?
6. Should `OidnError` be `#[non_exhaustive]` to preserve forward compatibility (C-FAILURE)?
7. Adding `pub const OIDN_REFERENCE_VERSION: (u32,u32,u32) = (2,4,1);` — yes/no?

---

**End of audit.**
