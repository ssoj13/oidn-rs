# Local OIDN 2.5.0 model and archive follow-up — 2026-10-02

## Provenance and scope

Reference checkout: `D:/Projects/vfx.ref/oidn`, HEAD `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7` (read-only git rev-parse). Git status was clean. No builds/tests, source modifications, submodule initialization or downloads were performed. This report supersedes the reference availability limitation in `model_weights.md`, while retaining its static findings.

Compared local2.5 source with current Rust and read-only `git diff v2.4.1 HEAD` for training/model.py, core/tza.cpp, core/unet_filter.cpp, core/graph.cpp, core/rt_filter.cpp, core/rtlightmap_filter.cpp and weights.

## Results

1. **Topology and channel widths are unchanged.** Local `training/model.py:61-103` matches Rust `crates/oidn-model/src/variants.rs:57-81` and `unet.rs:71-86`; local `training/model.py:167-208` matches Rust `unet_large.rs:47-69,108-126`. Git reports no2.4.1→2.5 change to training/model.py. Local runtime base layers `core/unet_filter.cpp:468-497` match Rust `unet.rs:104-142`; local runtime Large layers `:500-530` match Rust `unet_large.rs:147-187`. Final ReLU remains required at local `:495,528`, Rust `:142,187`. Python's Base final layer still lacks ReLU at local `training/model.py:153`; shipping runtime remains the authority for inference.

2. **Native2.5 moved upsampling fusion to the decoder input, without changing mathematical topology.** Native2.4.1 `bughunt/reference-v2.4.1/core/unet_filter.cpp:481,484,487,490` requested Upsample as convolution post-op. Local2.5 `core/unet_filter.cpp:481,484,487,490` ends these layers without a post-op and `:483,486,489,492` explicitly calls addUpsampleConcatConv on the next layer. Local `core/graph.cpp:312-317` requests Fusion::UpsampleSrc0; unsupported fused paths split into nearest upsample then concat convolution at `:166-170`. Pool wrappers preserve conv→pool (`:151-155`). Rust already performs independent upsample→concat→conv at `unet.rs:122-140`/`unet_large.rs:168-187`, matching the same logical sequence. Native fusion/backend numeric performance can differ, but no static operator-order regression was found.

3. **Model/quality selection is unchanged.** No git delta to local `core/rt_filter.cpp` or `core/rtlightmap_filter.cpp`. Local `core/rt_filter.cpp:37-59` declares the same21 RT models, including three Large and eight Small; local `core/rtlightmap_filter.cpp:19-20` declares two lightmap models. Current Rust shipped23 filenames match this set. Local `core/unet_filter.cpp:399-435` feature selection and `:449-458` quality fallback match Rust `registry.rs:70-86,100-103`. Native takes explicit user bytes at `:441-443`, then builds from tensor dimensions (`core/graph.cpp:87-114,173-203`).

4. **The confirmed Rust Small/XL construction bug remains against2.5.** Local `core/unet_filter.cpp:263` names detect Large topology, while native graph derives actual widths from each weight at `core/graph.cpp:98-114,188-203`. Rust `variants.rs:27-33` maps Small→Base and XL→Large and `filters/rt.rs:550-562` instantiates fixed presets. Normal CLI Fast resolves Small then uses override bytes (`oidn-cli/src/main.rs:267-283,342-344`); current loading therefore fails shape validation instead of executing a Small model. This is a confirmed API/CLI defect, not successful-output noise evidence.

5. **Large receptive-field mismatch remains against2.5.** Local constants174/202/16 at `core/unet_filter.h:35-37`; local `core/unet_filter.cpp:266-268` selects Large RF before overlap. Rust `filters/rt.rs:581` always passes BASE. Geometry report owns detailed impact. Source evidence suggests multi-tile Large artifacts are plausible; the user's image regression still needs reproduction.

6. **TZA format did not change.** Git has no2.4.1→2.5 delta to core/tza.cpp. Local `core/tza.cpp:32-42,66-90,93-99` uses the same version2, dims, oihw/x, h/f and offsets. Rust `oidn-tza/src/parser.rs:109-120,133-159` agrees. Earlier unchecked size multiplication (`types.rs:38-43` called at `parser.rs:166`), raw public-Tensor payload panic (`types.rs:49-66`) and duplicate-name ambiguity (`parser.rs:182` vs local `core/tza.cpp:99`) are not explained by a native format upgrade.

## Asset comparison availability

**Upstream SHA256 comparison cannot be performed locally:** `D:/Projects/vfx.ref/oidn/weights` is empty, recursive search finds no .tza anywhere in the reference checkout, and `git submodule status weights` returns `-28883d1769d5930e13cf7f1676dd852bd81ed9e7 weights` (leading minus: uninitialized). `.gitmodules:1-3` identifies weights as a separate oidn-weights repository. Read-only git diff confirms the pinned weights gitlink did not change between2.4.1 and local2.5, but that is not an archive byte comparison.

The23 Rust files are **actual binary TZA data, not Git LFS pointer text**: filesystem MCP read_binary of each12-byte header yields prefix Base64 `10ECA`, representing D7 41 02 00 (TZA magic/version2.0), with realistic sizes below. No reference files exist to classify as LFS pointers or real assets. Local binary headers plus hashes do not prove upstream correspondence, complete parser validity or numerical quality.

All23 local SHA256 hashes were computed through filesystem MCP file_hash to establish the current audit baseline:

| Archive | Bytes | Local SHA256 |
|---|---:|---|
| rtlightmap_dir.tza | 1830792 | `3fa8be25ea8ef4db87d2db5b85df63cfb251ca736d1075b787c56a6cc9c8541c` |
| rtlightmap_hdr.tza | 1830792 | `5f9757cb6a3d6c966cbd50dcd0cbffef88fe78f80cb7bf45ca99bcbd4ff1292d` |
| rt_alb.tza | 1830792 | `752c9a8d57884d660ae375a7c853cad8a44f77d559bb282f0ef39f8e758601a4` |
| rt_alb_large.tza | 7684615 | `a8efbe6eed511f41c8f5ff9145923214cefb0703cf91a3b00a1dc131ade1600a` |
| rt_hdr.tza | 1830792 | `a097e8fe9b1ba71ec43cde3e6df3fcf54fa4f8f2f4dd41eac01d0a76a16261e3` |
| rt_hdr_alb.tza | 1835976 | `a8e4fbbd3a794e0098f88e23882faf5459840bda5b0ff39f22bcaaf2cfd03b5a` |
| rt_hdr_alb_nrm.tza | 1841160 | `e586ef2ff48d7fbb7611986405220ed8fc5c13ca79bfc40be4dc742fbf959e1a` |
| rt_hdr_alb_nrm_small.tza | 641480 | `980ea307e7825cae23ed8b1addf8de62ece84b61094795a5631ebf1c91b2e1a7` |
| rt_hdr_alb_small.tza | 638024 | `1f3839dfc949e969bc5ad4d600e85bfa96187e0cb5ffee2f4715c791e6221fba` |
| rt_hdr_calb_cnrm.tza | 1841160 | `d0057057bd76c64e3e67b449728ef8f4fd2f11ccfb74c1e59b9d50dafd50ee0e` |
| rt_hdr_calb_cnrm_large.tza | 7698439 | `08130cdc4f33b7a513d0073757f089ac830f900613bb5bad72ca42c52b934465` |
| rt_hdr_calb_cnrm_small.tza | 641480 | `a6c43bfdbff01bdd8a9c777e139720f9e58b872a50c8db07079bddd007482dc5` |
| rt_hdr_small.tza | 634568 | `c9171947f2bceb4367725b7a0d5b4e0d663ac44108386af42f1b49e4361952e7` |
| rt_ldr.tza | 1830792 | `ad35b8fe765d453ade41861e90ee1da66e0b5fe35136de7763b64098a682e3ab` |
| rt_ldr_alb.tza | 1835976 | `d7402ccf5dc632bb2a6b14917f0bc143315765e29dd2227533ce3e75546d0e4c` |
| rt_ldr_alb_nrm.tza | 1841160 | `eae7f05c02ed49c0b267454818013736a0bdd90483dae2ffd7a120acff87cc0b` |
| rt_ldr_alb_nrm_small.tza | 641480 | `919609e048fe78fa1d60afe562f10daf1552872542f6eda6703d48e1c99426f1` |
| rt_ldr_alb_small.tza | 638024 | `1deb0b14b52368140e92cdff59469e37ff2922cda78d0872aa0a78234b879fb1` |
| rt_ldr_calb_cnrm.tza | 1841160 | `165d86fe5d13de1f623d45eb9cdd2202d4b8a0e1500b136c464739020cf8f4ce` |
| rt_ldr_calb_cnrm_small.tza | 641480 | `3777cc3446e9cee4cfc0f577b717068466bc72d8a344748a01266882dd7a398a` |
| rt_ldr_small.tza | 634568 | `d50ad6ca693ec8f2bdfac5d4657c4cb7c0763b20a9768e66f0402a00046b704c` |
| rt_nrm.tza | 1830792 | `fb8f7f5571332b602406f88a0b570d809dd3c06aebb530f5d048828f57490bc7` |
| rt_nrm_large.tza | 7684615 | `8ac7705ec5ac303116b40ad2f0880cd741ed1f4280828bf2f18f7635c3377257` |

## Dataflow and decision

```text
2.5 weights gitlink28883d... -> uninitialized local directory -> no native blob hashes
Rust23 TZA blobs -> verified header + SHA256 baseline -> parser -> fixed model presets
Native bytes -> parsed tensor dimensions -> graph widths -> fused logical U-Net
Rust bytes -> names/stem -> preset widths -> load validation -> explicit logical U-Net
```

Use the user-provided2.5 checkout as native code reference. Preserve logical Rust topology; centralize a validated model descriptor so tensors determine widths, topology and RF. After approval, compare native/Rust CPU/Rust WGPU outputs on the same saved inputs and compare local assets against initialized pinned weights. Do not attribute noise to2.5 fusion restructuring or weight changes without numerical/byte evidence.

## Checklist

- [x] Verify local HEAD and clean working tree.
- [x] Compare base/small/large/XL widths and logical operator sequence.
- [x] Compare2.4.1→2.5 diffs for model selection, graph fusion and TZA.
- [x] Confirm unresolved current Rust custom-weight and RF defects.
- [x] Inspect local reference weights submodule availability.
- [x] Read23 Rust binary headers and compute23 SHA256 hashes.
- [ ] Compare23 hashes to real pinned native archives: unavailable locally.
- [ ] Runtime image/parity verification: deferred, no audit builds/tests.
