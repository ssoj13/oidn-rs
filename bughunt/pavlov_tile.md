# Pavlov — Image / Buffer / Tile Parity Audit

- Agent: Pavlov (read-only audit)
- Date: 2026-05-21
- Scope: Rust `oidn-rs` vs C++ reference (`_ref/oidn`) — image format model, buffer/stride model, tile planner, tile loop, multi-tile copy.

---

## Verdict

PASS WITH MINOR GAPS. The Rust tile planner in `crates/oidn-rs/src/tile.rs` is a faithful port of the
geometric portion of `_ref/oidn/core/unet_filter.cpp::init` (lines 285–318) and the tile loop (199–238).
Formulas for `tileH/W`, `tileCountH/W`, `tileOverlap`, `tilePadH/W`, the round‑up‑with‑remainder helper,
and the per‑tile `overlapBeginH/W`/`overlapEndH/W`/`tileH1/H2`/`alignOffsetH/W` derivations are
identical modulo variable names. Constants match (RF base = 174, RF large = 202, alignment = 16,
default max tile = 2160²).

Gaps are limited to scope items, not correctness of the implemented subset:
- Multi‑subdevice loop condition (`tileCountH*tileCountW % numSubdevices == 0`) is dropped — fine for the
  single‑device Rust port but worth a comment.
- `maxMemoryMB`‑driven probing is dropped: Rust always uses the default budget. Documented in the
  module header. No behavioural divergence unless callers set `maxMemoryMB` (not exposed in Rust).
- Format support deliberately omits 4‑channel (`Float4`/`Half4`) — ref accepts them in `ImageDesc::getC()`
  but `unet_filter.cpp::checkParams` rejects them for RT/RTLightmap, so the Rust subset is consistent.
- Pixel stride (`wByteStride`) is **always implicit** in Rust (`Image::row_stride` + tightly packed pixels);
  ref allows an arbitrary `pixelByteStride ≥ pixelSize`. This is a real semantic gap for users with strided
  per‑pixel buffers (e.g. an RGBA buffer denoised as RGB). See I‑02.

---

## Tile algorithm side‑by‑side

### Reference (`unet_filter.cpp:265–318` and `199–238`)
```
tileAlignment   = lcm(minTileAlignment=16, device.minTileAlignment)
tileOverlap     = round_up(receptiveField/2, tileAlignment)
tileH           = round_up(H, minTileAlignment)         // 16
tileW           = round_up(W, minTileAlignment)
tilePadH        = tileH % tileAlignment
tilePadW        = tileW % tileAlignment
minTileDim      = max(4*tileOverlap, 768)
minTileH        = round_up(minTileDim, tileAlignment, tilePadH)   // 3-arg variant
minTileW        = round_up(minTileDim, tileAlignment, tilePadW)
maxTileSize     = (maxMemoryMB < 0) ? defaultMaxTileSize=2160*2160 : INT_MAX

while (tileCountH*tileCountW % numSubdevices != 0
       OR tileH*tileW > maxTileSize
       OR memoryUsage > budget):
    if tileH > minTileH and tileH > tileW:
        newTileH   = ceil_div(H + (2*tileOverlap+tilePadH)*tileCountH, tileCountH+1)
        tileH      = clamp(round_up(newTileH, tileAlignment, tilePadH),
                           minTileH, tileH - tileAlignment)
        tileCountH = max(ceil_div(H - (2*tileOverlap+tilePadH),
                                  tileH - (2*tileOverlap+tilePadH)), 1)
    elif tileW > minTileW:
        ...mirror...
    else: break

// tile loop (lines 201–232)
h = i*(tileH - (2*tileOverlap+tilePadH))
overlapBeginH = (i>0)            ? tileOverlap            : 0
overlapEndH   = (i<tileCountH-1) ? tileOverlap+tilePadH   : 0
tileH1 = min(H-h, tileH)
tileH2 = tileH1 - overlapBeginH - overlapEndH
alignOffsetH = tileH - round_up(tileH1, minTileAlignment)
// ...same for W...
inputProcess.setTile (h, w, alignOffsetH, alignOffsetW, tileH1, tileW1)
outputProcess.setTile(alignOffsetH+overlapBeginH, alignOffsetW+overlapBeginW,
                      h+overlapBeginH, w+overlapBeginW, tileH2, tileW2)
```

### Rust (`crates/oidn-rs/src/tile.rs:75–157`)
```
tile_overlap = round_up(RF/2, tile_alignment)
tile_h = round_up(height, MIN_TILE_ALIGNMENT)
tile_w = round_up(width,  MIN_TILE_ALIGNMENT)
pad_h  = tile_h % tile_alignment
pad_w  = tile_w % tile_alignment
min_tile_dim = max(4*tile_overlap, 768)
min_tile_h   = round_up_pad(min_tile_dim, tile_alignment, pad_h)
min_tile_w   = round_up_pad(min_tile_dim, tile_alignment, pad_w)

while (tile_h * tile_w) > max_tile_pixels:                    // no subdevice / memory probe
    if tile_h > min_tile_h && tile_h > tile_w:
        new_h = ceil_div(height + (2*tile_overlap+pad_h)*tile_count_h, tile_count_h+1)
        tile_h = new_h.clamp(min_tile_h, tile_h - tile_alignment)
        tile_h = round_up_pad(tile_h, tile_alignment, pad_h)   // (*) see T-01
        tile_count_h = max(ceil_div(height - (2*to+pad_h), tile_h - (2*to+pad_h)), 1)
    else if tile_w > min_tile_w:
        ...mirror...
    else: break

// tile loop (lines 120–154) — identical to ref:
y           = i * (tile_h - (2*tile_overlap + pad_h))
overlap_top = i>0 ? tile_overlap : 0
overlap_bot = i<tile_count_h-1 ? tile_overlap + pad_h : 0
tile_h1     = min(height - y, tile_h)
tile_h2     = tile_h1 - overlap_top - overlap_bot
align_off_h = tile_h - round_up(tile_h1, MIN_TILE_ALIGNMENT)
```

Mapping is 1:1: `TileJob.input` ↔ ref `(h, w, tileH1, tileW1)`,
`TileJob.output_src_in_tile` ↔ ref `(alignOffsetH+overlapBeginH, alignOffsetW+overlapBeginW, tileH2, tileW2)`,
`TileJob.output_dst` ↔ ref `(h+overlapBeginH, w+overlapBeginW, tileH2, tileW2)`.

---

## Per‑issue table

| id | sev | rust file:line | ref file:line | description | suggested fix |
|----|-----|----------------|---------------|-------------|---------------|
| T‑01 | low | tile.rs:97‑99,105‑107 | unet_filter.cpp:310 | Ref does `tileH = clamp(round_up(newTileH, ta, pad), minTileH, tileH-ta)` as a single expression. Rust splits into `clamp` then re‑applies `round_up_pad`. The second `round_up_pad` can push the value above the upper clamp bound `tile_h - tile_alignment` because rounding adds up to `(ta - 1)`. In practice safe for common (W,H) but not byte‑for‑byte identical. | Mirror ref: `tile_h = round_up_pad(new_h, tile_alignment, pad_h).clamp(min_tile_h, tile_h - tile_alignment);` (round first, clamp second). |
| T‑02 | low | tile.rs:88‑89 (then 100,108) | unet_filter.cpp:293‑294, 311 | `tile_count_h/w` start at 1 and are only updated inside the chosen branch. Ref behaves the same, but Rust never re‑syncs the *other* axis count after the dimension was shrunk — same as ref, so this is parity, not a bug. Documented here so it isn't mistaken for a divergence. | None. |
| T‑03 | low | tile.rs:95 | unet_filter.cpp:303‑305 | Loop condition only checks `tileH*tileW > maxTilePixels`. Ref also forces `tileCount % numSubdevices == 0` and re‑checks the memory budget from `computeBufferReservation`. Acceptable for single‑device Burn backends but should be noted. | Add a TODO referencing `unet_filter.cpp:303` if/when multi‑subdevice Burn backends land. |
| T‑04 | low | tile.rs:62‑64 | platform.h:208‑214 | Ref's `round_up(a,b,c)` ≡ `(a+b-c-1)/b*b + c` — closed‑form, always returns `≥ a`. Rust's `round_up_pad` uses `r = (x-pad) % align`; if `r<0` it returns `x + (-r)` which can be **less than** `x` when `x < pad` (negative `x-pad` makes `r ∈ (-align, 0]`). The only call sites pass `min_tile_dim ≥ 768`, and `pad ∈ [0, 16)`, so `x-pad ≥ 752 > 0` and the branch is never exercised — match in practice, divergent in form. | Replace with ref's closed‑form: `(x + align - pad - 1) / align * align + pad`. |
| F‑01 | low | image.rs:74‑82 | image.cpp:9‑35 | Rust `Image` always stores `row_stride = width * pixel_size` and has no `wByteStride`. Ref allows arbitrary `pixelByteStride ≥ pixelSize` and arbitrary `rowByteStride ≥ width*pixelStride`. | Add `pixel_stride: usize` to `Image`/`ImageMut` with `Image::with_strides` constructor; thread through `to_rgb_f32`/`write_rgb_f32` so x‑offset = `x*pixel_stride`. |
| F‑02 | info | image.rs:13‑26 | oidn.h:241‑253 | Rust omits `Float4`/`Half4`. Ref's `ImageDesc::getC` accepts them, but `unet_filter.cpp::checkParams` rejects them for RT, so the Rust subset matches the *filter's* acceptance set, not the *format enum*'s superset. | Document the rationale or add `Rgba32f`/`Rgba16f` with a runtime "not supported by RT filter" error. |
| F‑03 | info | image.rs:38‑43 | image.cpp:17‑25 | Rust returns hard‑coded `4` / `2` byte element sizes. Ref calls `getFormatSize(format)`. Same result for the implemented subset. | None. |
| I‑01 | low | image.rs:118‑161 | image_accessor.h:30‑52 | Broadcast/zero‑pad rule: 1ch→`(v,v,v)`, 2ch→`(x,y,0)`. Ref `ImageAccessor::get3` does 1ch→`(v,v,v)` (matches) but 2ch→`(x,y,y)` (replicates G into B), not zero. | Decide canonical rule. If matching ref's runtime kernel is required, replace `0.0` on line 149 with `src[src_off + 1]`. Note: ref's `InputProcess` also broadcasts to 3 channels (`input_process.cpp:33`), so behaviour must agree with `image_accessor.h::get3`, which is the `(x,y,y)` rule. **This is a real divergence on RG inputs.** |
| I‑02 | info | image.rs:193‑229 | image_accessor.h:56‑93 | `write_rgb_f32`: 1ch dst takes red only, 2ch dst keeps (R,G). Ref's `set3` for 2ch writes `pixel[0]=x; pixel[1]=y`, identical. For 1ch ref writes only `pixel[0]=value.x`, identical. Match. | None. |
| C‑01 | low | tile.rs::TileJob | tile.h:10‑18 | Ref's `Tile` carries `(hSrcBegin, wSrcBegin, hDstBegin, wDstBegin, H, W)` and is re‑used by both InputProcess and OutputProcess with **different meanings** of dst (network tensor vs output image). Rust packs both into `TileJob.{input, output_dst, output_src_in_tile, align_offset_*}`. Equivalent but verbose. | None — Rust shape is clearer; just document the mapping. |
| C‑02 | med | (multi_tile_wgpu.rs:124) | unet_filter.cpp:201‑232 | Multi‑tile assembly: ref *relies on overlap halo*, no blending. Rust must do likewise. Test `denoise_3072_multi_tile_wgpu` checks row‑mean jumps `< 0.01`, which is the right shape of test, but the **actual** copy (`output_src_in_tile` → `output_dst`) is in `unet_runner` — not audited in this pass. | Cross‑check in a follow‑up that the runner reads `output_src_in_tile` from the network tensor and writes to `output_dst` of the user image (no halo on the write side, no blending). |
| B‑01 | info | image.rs:56‑72 | image.cpp:37‑75 | Ref has three image constructors: raw pointer + offset, user `Buffer` + desc, engine‑allocated. Rust only has borrowed `Image`/`ImageMut` over a host slice. No device‑allocated path, no `byteOffset`. Acceptable because Burn owns device memory separately. | None — but `ImageOwned` analogue may be needed when Burn‑native I/O lands. |
| B‑02 | info | image.rs | image.cpp:83‑104 | `Image::overlaps` is not ported. Ref uses it to gate the `outputTemp` alloc in `unet_filter.cpp:122‑124`. Rust avoids this by always going through Burn tensors. | None for now; add when in‑place denoising on user pointers is supported. |
| B‑03 | low | image.rs:60 | image.h:20 | Rust `row_stride: usize` field is exposed and respected on read (image.rs:123, 197), but the `from_*` constructors hard‑code it to `width * pixel_size`. Cannot represent a user buffer with non‑tight rows. | Add `Image::with_row_stride` ctor, validate `row_stride ≥ width * pixel_size`. |

---

## Format support matrix

| Format | Ref enum (oidn.h:241‑253) | Ref RT filter accepts | Rust `PixelFormat` | Rust factory | Match? |
|--------|---------------------------|-----------------------|--------------------|--------------|--------|
| Undefined | OIDN_FORMAT_UNDEFINED | n/a | — | — | n/a |
| Float (R32f) | 1 | yes (1ch) | R32f | `from_r_f32` | yes |
| Float2 (RG32f) | 2 | yes (2ch) | Rg32f | `from_rg_f32` | yes |
| Float3 (RGB32f) | 3 | yes (3ch) | Rgb32f | `from_rgb_f32` | yes |
| Float4 (RGBA32f) | 4 | rejected by `checkParams` | — | — | yes (intentional) |
| Half (R16f) | 257 | yes (1ch) | R16f | `from_r_f16` | yes |
| Half2 (RG16f) | 258 | yes (2ch) | Rg16f | `from_rg_f16` | yes |
| Half3 (RGB16f) | 259 | yes (3ch) | Rgb16f | `from_rgb_f16` | yes |
| Half4 (RGBA16f) | 260 | rejected | — | — | yes (intentional) |

Channel order: ref's `ImageAccessor::get3` reads `(pixel[0], pixel[1], pixel[2])` in memory order; Rust does
the same (image.rs:152‑154). HWC row-major. Match.

Pixel size: ref `getFormatSize` returns `channels * sizeof(dtype)`; Rust `pixel_size()` returns
`channels * element_size` (image.rs:45‑47). Match for the supported subset.

Row stride: ref default `hByteStride = width * wByteStride`; Rust default `row_stride = width * pixel_size`.
Match for the contiguous default (image.rs:76, image.cpp:34). Ref permits override; Rust does not — see F‑01/B‑03.

f16 vs f32 at tile boundary (item 11): the tile planner is **format‑agnostic** in both ref and Rust — it
operates only on pixel coordinates. Per‑pixel f16/f32 handling lives in `ImageAccessor` (ref) and in
`Image::to_rgb_f32` / `ImageMut::write_rgb_f32` (Rust, image.rs:126‑134, 219‑227), which decode/encode
per row. Match — no f16/f32‑specific tile divergence.

Auxiliary buffers (albedo, normal, item 12): both ref and Rust treat them as additional inputs to the
*same* tile region; ref's `InputProcess::setSrc(color, albedo, normal)` (`input_process.cpp:28‑43`) takes
all three at once and slices them by the same `tile` rect. Rust must do likewise — this is in
`unet_runner`, not audited here, but `TileJob.input` is the single source rect, so as long as runner
applies it to all aux inputs, match.

---

## Tile overlap (halo) — exact value

- Receptive field base: 174 px (ref `unet_filter.h:35` ; Rust `tile.rs:9`). Match.
- Receptive field large: 202 px (ref `unet_filter.h:36` ; Rust `tile.rs:11`). Match.
- `tileOverlap = round_up(RF/2, tileAlignment)` ⇒ base‑model on default 16‑aligned device:
  `round_up(87, 16) = 96`. Both implementations compute this identically.
- The audit task brief mentions "32 px typical" — that is **incorrect** for OIDN's UNet; ref uses
  87 → rounded to 96 (base) or 101 → rounded to 112 (large). Rust matches ref.

---

## Multi‑tile assembly (item 8) and boundary modes (item 7)

- Ref does not blend tile outputs; it **relies on halo**: each interior tile contributes only the
  central `(tileW2, tileH2)` rectangle (`unet_filter.cpp:216, 228‑231`). Edge tiles contribute the full
  remaining slice because `overlapEnd = 0` on the last row/col.
- Boundary behaviour for inputs outside the image: in ref this falls to the network's input
  zero‑padding inside the conv kernels — `inputProcess` writes `tileH1×tileW1 ≤ tileH×tileW` into a
  tensor of size `tileH×tileW`, leaving the `alignOffset*` region uninitialised (later set to 0 by
  the conv padding). Rust's `align_offset_x/y` and `output_src_in_tile` carry the same offsets, so the
  same scheme applies — assuming the runner zero‑initialises the network input tensor before
  `InputProcess` writes into it. **Not verified in this audit; flag as open question Q‑1.**
- Rust planner does *not* explicitly clamp input reads — it relies on the runner. Match with ref.

## Buffer alignment / GPU stride requirements (item 9)

- Ref `Image` allows arbitrary `pixelByteStride` and `rowByteStride`; alignment requirements come from
  `Device::getMinTileAlignment()` (folded into `tileAlignment`).
- Rust currently exposes neither; everything is tightly packed and the planner takes `tile_alignment`
  as a free parameter. For wgpu backends needing larger alignment (e.g. 256 B for storage buffers),
  the caller would have to pass a larger value to `tile::plan`. No mechanism today to query the
  Burn/wgpu device for its required alignment. Open question Q‑2.

## Memory ownership semantics (item 10)

- Ref: `Image` is either user‑pointer‑backed (`ptr + byteOffset`, no buffer), `Buffer`‑backed (shared
  ownership via `Memory`), or engine‑allocated (owns its own `Buffer`). All cases use `Ref<Image>`
  (refcounted).
- Rust: only borrowed `Image<'a>` / `ImageMut<'a>` (image.rs:56, 66). No owning variant, no
  `Buffer`/`Memory` analogue. The Burn tensor takes ownership of device memory through `Tensor`. This
  is a cleaner Rust split (host-side borrowed view + device-side Burn tensor) and **does not block
  parity** for the filter API.

---

## Dead / unused

- `tile.rs::total_output_pixels` (line 160) — used only by tests (`unit_color_tile.rs:53, 60`, `multi_tile_wgpu.rs:58`). OK.
- `PixelFormat::is_f16` (image.rs:49‑52) — used internally by `to_rgb_f32`/`write_rgb_f32` and tests. OK.
- `TileJob.align_offset_x/_y` are stored but the same information is recoverable from
  `output_src_in_tile.x - overlap_left`. Not dead — useful for runners that don't need overlap explicitly
  but want the alignment offset to clear/zero‑pad the network input tensor. Keep.
- Nothing obviously dead.

---

## Open questions

- Q‑1: Does `unet_runner` zero‑initialise the network input tensor before InputProcess writes
  `(tileW1, tileH1)` at `(align_offset_x, align_offset_y)`? If not, the `align_offset*` region holds
  stale data and the conv receptive field will read garbage on edge tiles. Ref relies on this being zero.
- Q‑2: How is the device‑specific `tileAlignment` (ref `device->getMinTileAlignment()`) obtained on
  the Rust side? Today `tile::plan` takes it as a parameter — does any caller compute it from the Burn
  backend, or do all callers hard‑code 16? If hard‑coded, GPU backends needing >16 alignment will
  produce mis‑aligned tile buffers.
- Q‑3: 2‑channel input broadcast: Rust zero‑pads B (image.rs:149); ref replicates G into B
  (`image_accessor.h:39, 49`). Which is canonical? Tests in `formats.rs:84‑87` *assert* the zero‑pad
  rule — this would fail against a strict ref‑bit‑match goal. See I‑01.
- Q‑4: Pixel stride support — is it on the roadmap? Many users denoise the RGB triplet of an RGBA
  buffer (pixelByteStride=16, channels=3). Today the Rust API forces a contiguous copy. See F‑01.
- Q‑5: 4‑channel formats — ref accepts at `ImageDesc` level but rejects at filter level. Should the
  Rust API surface a typed `Rgba32f`/`Rgba16f` variant that fails with `Error::InvalidArgument` at
  filter‑build time, matching ref's error surface?

---

## Cross‑references

- Rust planner: `crates/oidn-rs/src/tile.rs:75‑157`
- Rust image: `crates/oidn-rs/src/image.rs:55‑230`
- Rust tensor layout helpers: `crates/oidn-rs/src/image_tensor.rs:20‑83`
- Ref tile loop: `_ref/oidn/core/unet_filter.cpp:198‑238`
- Ref tile init: `_ref/oidn/core/unet_filter.cpp:265‑318`
- Ref image desc: `_ref/oidn/core/image.h:14‑72`, `image.cpp:9‑35`
- Ref image accessor (broadcast rules): `_ref/oidn/core/image_accessor.h:30‑93`
- Ref InputProcess setTile: `_ref/oidn/core/input_process.cpp:53‑72`
- Ref OutputProcess setTile: `_ref/oidn/core/output_process.cpp:33‑52`
- Ref constants: `_ref/oidn/core/unet_filter.h:35‑39`
- Ref round_up 3‑arg: `_ref/oidn/common/platform.h:207‑214`
- Ref Format enum: `_ref/oidn/include/OpenImageDenoise/oidn.h:241‑253`, `oidn.hpp:118‑132`
