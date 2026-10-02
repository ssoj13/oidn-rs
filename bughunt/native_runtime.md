# Native runtime numerical verification — 2026-10-02

This report extends the static audit and [Astra numerical review](astra_numerics.md). The user authorized renderer diagnostic testing and renewed native investigation. Measured synthetic comparisons below do not reproduce the user's actual progressive scene.

## Provenance and reproducibility

Reference source: clean `D:/Projects/vfx.ref/oidn`, tag v2.5.0, commit `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7`. Official Windows release runtime was downloaded into isolated audit artifacts, not built from the local source. [provenance.json](native-verification/provenance.json) records acquisition commands, ZIP SHA256 `6ae0474ef7606d68647c1e2c2842832d6af01128ee84a523d30368a91013b707`, executable/DLL hashes, observed version2.5.0, devices and limits. Reference source and its uninitialized weights directory were not modified.

The external weights repository was separately checked out at pinned gitlink `28883d1769d5930e13cf7f1676dd852bd81ed9e7`. **All 23 Rust archive sizes and SHA256 values equal the corresponding pinned native archive bytes.** [weight_hashes.json](native-verification/weight_hashes.json) contains both paths and hashes for every archive. This supersedes the earlier inability to compare native weights in the uninitialized reference directory; it does not establish all-model runtime parity.

Official runtime device listing identifies RTX3080Ti CUDA SM8.6 and Ryzen9 5900X CPU AVX2. [native_smoke.json](native-verification/native_smoke.json) records successful CPU/CUDA executions with explicit `rt_hdr.tza`, balanced quality and scale1, finite 3072-channel-value output. This establishes readiness, not accuracy. The Rust executable is the Squarebob target-directory `oidn-rs.exe`, using its resolved dependency configuration.

## Harness contract and measured coverage

[compare_runtime.py](native-verification/compare_runtime.py) contains fixture construction, exact CLI commands, PFM parsing and comparison logic. Main and alignment invocations produced [comparison.json](native-verification/comparison.json) and [alignment_comparison.json](native-verification/alignment_comparison.json).

- Main matrix:70 executions; alignment follow-up:20 executions. **90/90 exited0 with every output value finite.**
- Main fixtures: uniform32x32, ramp32x32, fine64x48, odd edge-lit33x19, tiny8x8 and full-AOV fine33x19.
- Alignment fixtures: full-AOV32x32 and32x19. Explicit scales1 and0.02; odd/tiny additionally automatic scale.
- Each configuration runs Rust WGPU Balanced, native CPU Balanced/High and native CUDA Balanced/High. Identical explicit Base archive bytes prevent native High from silently selecting another model.
- Files are little-endian RGB f32 PFM with LF header, unit magnitude scale and row conversion. This deliberately avoids the separately audited CRLF/scale-magnitude and encoded-file defects.
- Metrics compare finite pairs: max/mean absolute and relative channel errors with denominator floor1e-6. Highlight mask is raw input maxRGB>=5; dark mask maxRGB<1. These masks/metrics are diagnostic definitions, not clean-target noise measures.
- The Rust CLI uses mutable host-image adapter, not Squarebob's immutable renderer bridge. No long-frame multi-tile, Small/XL, scalar output, auxiliary-only, lightmap, nonfinite or actual-scene corpus is covered.

## Rust WGPU versus native CPU Balanced

Maximum absolute channel difference:

| Fixture / dimensions | Scale1 | Scale0.02 | Automatic scale |
| --- | ---: | ---: | ---: |
| Uniform32x32, color-only |0.000085831 |0.000038147 |not run |
| Ramp32x32, color-only |0.000381470 |0.000225067 |not run |
| Fine64x48, color-only |0.000162125 |0.000083923 |not run |
| Odd edge-lit33x19, color-only |0.000165939 |0.000078201 |1.724971771 |
| Tiny8x8, color-only |0.000055313 |0.000035286 |3.218883514 |
| Fine33x19, full AOV |0.977558136 |4.665126801 |not run |
| Fine32x32, full AOV aligned |0.000213623 |0.000112534 |not run |
| Fine32x19, full AOV one-axis unaligned |0.550964355 |3.686058044 |not run |

For the measured color-only explicit-scale corpus, max error is0.0003814697265625 and max relative channel error is approximately1.16e-5. This supports close numerical agreement under these bounded conditions; it is not a universal tolerance. The aligned full-AOV fixture similarly has max error0.000213623046875 at scale1 and0.0001125335693359375 at0.02.

Odd/tiny automatic-scale discrepancies reproduce a numerical difference under conditions predicted by P5: Rust `crates/oidn-rs/src/autoexposure.rs:152-176` drops partial cells and returns unity below16, while LOCAL2.5 `core/autoexposure.h:29-31` / `devices/gpu/gpu_autoexposure.h:29-32,59-62` use balanced all-pixel cells. Explicit scale bypasses this path and reduces the same-fixture differences substantially.

Full-AOV differences strongly depend on spatial alignment: both-axis aligned32x32 agrees closely with CPU, while32x19 and33x19 do not. This is consistent with P1's invalid normal padding: Rust `filters/unet_runner.rs:161-199` remaps raw padded0 to0.5; LOCAL2.5 `devices/gpu/gpu_input_process.h:90-118` leaves outside-source channels0. It is controlled numerical support, but this experiment does not independently capture packed tensors or prove P1 is the sole difference. It does not reproduce the user's scene or confirm Large RF defects; these Base fixtures do not exercise Large multi-tile overlap.

## CUDA and quality differences

Rust WGPU versus native CUDA Balanced has larger differences even in aligned color-only fixtures: uniform scale1 max0.314720154, ramp scale1 max1.277641296, fine scale1 max0.686485291. Aligned full-AOV scale1 max0.832857132. These cannot automatically be called Rust defects.

For every measured fixture/scale, native CPU High and Balanced outputs were identical. Native CUDA High and Balanced were not generally identical despite the same archive: examples uniform scale1 max0.281738281 and ramp scale1 max1.219005585. This demonstrates arithmetic-policy influence in the tested official CUDA runtime; it does not establish which output best reconstructs a clean target.

Checked source explains relevant precision distinctions: LOCAL2.5 `devices/cuda/cuda_device.cpp:215-219` selects Float16 tensors/weights; `core/unet_filter.cpp:262,280` passes quality-derived fastMath; `devices/cuda/cuda_conv.cu:28-48` constrains accumulator/kernel choice; `devices/cuda/cutlass_conv.h:55-77` distinguishes output/epilogue and accumulator types. Native High can retain half intermediate storage. Astra directly inspected pinned Burn default f32 Direct-convolution dispatch and feature graph; no fusion/autotune feature was enabled in that checked Squarebob default configuration. See [Astra](astra_numerics.md) for full dependency paths, source anchors and arithmetic limitations. Do not infer TF32 or blame changing autotuned kernels without evidence.

## Signal, noise and highlight interpretation

The earlier Squarebob checker SD comparison is **not a measured noise increase**. The synthetic RGB checker has luminances6.62464 and39.74784. Adaptive threshold6 at128SPP collapses both levels to the same chromaticity/luminance before inference; threshold10 at256SPP restores contrast. Its output spatial SD can therefore rise because real fixture structure returns. Frozen fixed-clamp repeats remain useful repeatability evidence, but no noisy/clean target was defined. Exact source/formula/SD analysis is in [astra_numerics.md](astra_numerics.md), with prior measurements in [Squarebob plan16](../../squarebob-rs/plan16.md).

PU constants and branch formulas match the native scalar contract; the exponential inverse increases absolute radiance sensitivity in highlights (Rust `gpu_ops.rs:179-196`, `color.rs:126-136`; LOCAL2.5 `core/color.h:77-106`). The 65504 normalization reference is not a hard clamp. Matched weights plus close f32 CPU/WGPU output narrow speculation about corrupted archives or gross color-only operator mismatch; they do not prove actual-scene convergence, half/f32 quality equivalence or all-mode correctness.

## Next verification gates

- [x] Pin official source/runtime provenance and preserve exact commands/logs.
- [x] Verify all 23 archive bytes against pinned native weights.
- [x] Execute90 finite matched-output runs and preserve comparison artifacts.
- [x] Compare alignment and explicit-versus-automatic scale conditions.
- [x] Review precision/confounders and checker signal interpretation with Astra.
- [ ] Capture actual progressive raw/AOV/post-clamp/denoised/display snapshots at128/256/above256SPP, with settings, input counts, scale and presented snapshotSPP.
- [ ] Use clean targets plus separately controlled noise/seeds; measure residual error/bias/variance rather than raw spatial SD.
- [ ] Capture valid preprocessed/packed tensors and first differing intermediate layer under matched storage precision.
- [ ] Verify true Large single/multi-tile, all public modes, malformed/nonfinite contracts and actual renderer bridge separately.
- [ ] Implement unrelated denoiser repairs only after their scope is approved; PQ display changes have separate explicit authorization.

The user's progressive highlight-noise cause remains unproven. This report records measured contracts and limits, not a denoiser quality certification.

## CPU/PQ verification closure — 2026-10-02

The user explicitly authorized the old CPU display repair and push to main in addition to PQ integration. Unrelated denoiser/scheduling proposals remain unapproved. [Squarebob plan17](../../squarebob-rs/plan17.md), [Astra post-fix review](astra_pq_review.md), and [OIIO CPU review](oiio_cpu_display.md) record the checked contracts and results.

The historical no-feedback conclusion described the inspected GPU bridge; CPU display had existing raw-buffer feedback/exposure-order and denoised-view defects. The shared composition repair reads raw PT/OIDN, applies CPU exposure before OCIO, and writes a separate reusable display scratch. Source: Squarebob `crates/render-3d/src/lib.rs:1190-1236`, `crates/pt-megakernel/src/compute.rs:5383-5495`, and both callers `src/app/treemap_view.rs:699-708,1193-1205`. No matching defect was found in the scoped OCIO/OIIO library inspection.

Color6/6, host3/3, render-core3 plus GPU1, extended-sRGB/full-PresentPass signal and CPU immutability/order probes passed. The signal probe covered gray plus six saturated-primary cases (max PQ primary error0.000542;203nits0.580652;1000nits0.751720). Final locked workspace/all-target check passed9.381s; actual binary build passed33.406s, both empty logs. CPU probe preserved PT/external raw bytes and repeated output, with exposure2 before OCIO matching the processor oracle. Actual GUI retry and confirmed remote push remain pending; these measurements do not certify physical monitor luminance. Context7 supplied official winit0.30 changelog evidence; unavailable/timed-out GitNexus requests used direct-source fallback.

## Final window and persistence verification — 2026-10-02

This supersedes the preceding pending GUI retry and final-check status. Actual PBR, PT GPU, PT CPU and newly saved CPU-state restart windows completed successfully. Logs confirm actual HDR10(PQ), `Rgb10a2Unorm/Bt2100Pq`; full UI capture `squarebob_pq_ui_controls.png` was inspected by the root reviewer. CPU/GPU controls are visible; all five output modes, reference-white Auto/manual and HLG peak controls exist in the checked host. Auto white is240nits. OS-reported peak603nits/full-frame150nits/headroom2.51/10bits are metadata, not physical luminance measurements. Restart preserved CPU color path, ISO240, f/1, shutter1 and effective scale2.

App state now uses a typed RON payload inside the existing outer storage map, while DisplayPrefs stays a JSON string. `Squarebob src/app/state.rs:142-149` accepts RON plus valid legacy JSON; `Squarebob src/app/mod.rs:782-834` saves the complete typed state. `Squarebob src/app/persistence_tests.rs:5-43` verifies full-state roundtrip including nonfinite dock rectangle sentinels, finite floating-window position/size, topology and CPU/camera settings. Previously corrupted JSON null-rectangle snapshots report decode failure; no geometry is guessed or stripped to migrate them.

Final binary unit suite:34/34 passed in0.13s (`squarebob_pq_bin_tests.stdout.log`). Final locked workspace/all-target check passed in6.313s with empty logs; binary build passed in16.140s with empty logs. Earlier measurements remain historical. Final Clippy exited0 in10.685s, with464 stderr lines of warnings (`pq_final_clippy.stderr.log`); this is not a warning-free result. The official Rust API checklist was consulted for getter naming, validation, meaningful errors and intermediate results; Context7 supplied official winit0.30 documentation.

Manual clipboard/accessibility, multi-monitor transitions, all alternative output modes and physical luminance remain unverified. Actual user-scene progressive noise remains unresolved. The authorized main push is prepared, awaiting the confirmed commit/remote receipt.
