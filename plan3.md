# plan3 — Native numerical verification and PQ display integration

Updated: 2026-10-02. Continues plan2 and Squarebob plan16 without replacing their evidence. The user renewed exhaustive research against `D:/Projects/vfx.ref/oidn`, and separately authorized implementing PQ display in Squarebob if missing. This latter instruction authorizes display production changes; unrelated denoiser repairs remain proposals pending approval.

## Scope and ownership

- Root: orchestration, numerical evidence review, Squarebob color/display contract integration and validation.
- Pipeline agent: progressive renderer source/history and shared `egui-display` integration investigation; implementation ownership assigned explicitly before edits.
- Native agent: isolated official native runtime, pinned weight identity and numerical comparison artifacts.
- Astra: independent numerical/precision/causal analysis.

## Checklist

- [x] Verify the supplied native checkout is still clean.
- [x] Re-read prior source reports and retain mode-specific evidence limits.
- [x] Compare actual bytes of all 23 Rust weights against native pinned archives: all SHA256 values and sizes match.
- [x] Provision official native OIDN 2.5.0 runtime in isolated audit artifacts; CPU/CUDA smoke passes with finite output.
- [x] Execute matched synthetic Rust WGPU/native CPU/CUDA matrix and alignment follow-up:90/90 successful finite runs; exact commands and outputs preserved. Actual-scene/all-mode parity remains unchecked.
- [>] Inspect progressive accumulation, AOV routing/reset, sampling and history for mechanisms outside adaptive-clamp changes.
- [x] Review native/Burn arithmetic, PU sensitivity and checker signal/noise interpretation with Astra.
- [x] Verify the pre-change Squarebob SDR baseline and blocked HDR OCIO views; its unused PQ helper did not provide PQ presentation.
- [x] Verify exr-view and Playa reuse `egui-display`: float extended-sRGB canvas -> negotiated SDR/PQ/HLG/scRGB surface.
- [x] Implement shared PQ-capable presentation and color contract; bounded tests and native HDR10/PQ windows passed, and publication is confirmed. See Squarebob plan17 and the receipt below.
- [x] Align and verify bounded color pipeline display-light/canvas contracts, including OCIO HDR decoding, luminance units and GPU/CPU order.
- [x] Validate source changes with focused unit, shader/GPU signal and CPU immutability tests, final locked workspace check and binary build.
- [x] Update both architecture maps and Squarebob plan17 with exact source anchors and measured limitations.
- [x] Deliver findings, measured limitations, architecture maps and the confirmed publication receipt.
- [ ] Obtain approval for still-proposed unrelated production denoiser repairs.

## Evidence and boundaries

See `bughunt/native-verification/provenance.json`, `weight_hashes.json`, and `native_smoke.json` for native acquisition and hashes. Reference source remains untouched. Same weight bytes do not establish equal arithmetic: native CUDA has half tensors, and requested quality controls accumulation policy; Rust uses its checked Burn backend configuration.

The prior checkerboard diagnostic proves sample-dependent input/output changes and frozen-input repeatability. It does not measure scene noise. At clamp6 both synthetic checker levels have luminance above6 and equal chromaticity, so clipping removes their contrast before the network; clamp10 restores contrast from the lower level. Updated Astra analysis will retain this explicit causal qualification.

GitNexus MCP is absent in this turn; CLI help is available, but catalog-backed queries previously timed out. Direct source and caller reads are the fallback when a graph request cannot complete. No absent-tool or expected nonexistent-path error is mislabeled a server defect.

## Measured numerical follow-up

[Native runtime report](bughunt/native_runtime.md) consolidates provenance, all 23 equal archive sizes/SHA256s,90 finite outputs and alignment/scale controls from `comparison.json`/`alignment_comparison.json`. Color-only explicit-scale Rust WGPU versus nativeCPU max0.0003814697265625; aligned full-AOV32x32 max0.000213623046875. Odd/tiny autoexposure and unaligned normal-padding cases diverge much more. Results cover a finite synthetic CLI corpus, not actual renderer scene convergence or all-mode accuracy.

[Astra numerics](bughunt/astra_numerics.md) corrects the earlier checker interpretation: raising the adaptive clamp restores legitimate fixture contrast; raw spatial SD is not error against a clean target. The default threshold plateaus at256SPP. Native CUDA half-storage/quality accumulation differs from checked Burn f32 Direct dispatch; its default graph had no fusion/autotune features. PU inverse can magnify absolute highlight radiance errors without a new formula defect.

## Historical display implementation checkpoint — superseded by closure below

[Squarebob plan17](../squarebob-rs/plan17.md) tracks shared `egui-display` native host, actual negotiated output state, runtime HDR/white context and OCIO display-reference XYZ->Rec.709 light conversion. PQ is encoded by the canonical shared presenter once after float canvas composition. Initial workspace check passed; host/color/shader/binary/actual-window validation is still in progress. Mark implementation complete only after those gates pass. Native diagnostic source/runtime acquisition does not authorize unrelated denoiser production repairs.

## CPU/PQ verification closure — 2026-10-02

The user explicitly authorized the old CPU display repair and push to main in addition to PQ integration. Unrelated denoiser/scheduling proposals remain unapproved. [Squarebob plan17](../squarebob-rs/plan17.md), [Astra post-fix review](bughunt/astra_pq_review.md), and [OIIO CPU review](bughunt/oiio_cpu_display.md) record the checked contracts and results.

The historical no-feedback conclusion described the inspected GPU bridge; CPU display had existing raw-buffer feedback/exposure-order and denoised-view defects. The shared composition repair reads raw PT/OIDN, applies CPU exposure before OCIO, and writes a separate reusable display scratch. Source: Squarebob `crates/render-3d/src/lib.rs:1190-1236`, `crates/pt-megakernel/src/compute.rs:5383-5495`, and both callers `src/app/treemap_view.rs:699-708,1193-1205`. No matching defect was found in the scoped OCIO/OIIO library inspection.

Color6/6, host3/3, render-core3 plus GPU1, extended-sRGB/full-PresentPass signal and CPU immutability/order probes passed. The signal probe covered gray plus six saturated-primary cases (max PQ primary error0.000542;203nits0.580652;1000nits0.751720). Final locked workspace/all-target check passed9.381s; actual binary build passed33.406s, both empty logs. CPU probe preserved PT/external raw bytes and repeated output, with exposure2 before OCIO matching the processor oracle. Actual GUI retry and confirmed remote push remain pending; these measurements do not certify physical monitor luminance. Context7 supplied official winit0.30 changelog evidence; unavailable/timed-out GitNexus requests used direct-source fallback.

## Final window and persistence verification — 2026-10-02

This supersedes the preceding pending GUI retry and final-check status. Actual PBR, PT GPU, PT CPU and newly saved CPU-state restart windows completed successfully. Logs confirm actual HDR10(PQ), `Rgb10a2Unorm/Bt2100Pq`; full UI capture `squarebob_pq_ui_controls.png` was inspected by the root reviewer. CPU/GPU controls are visible; all five output modes, reference-white Auto/manual and HLG peak controls exist in the checked host. Auto white is240nits. OS-reported peak603nits/full-frame150nits/headroom2.51/10bits are metadata, not physical luminance measurements. Restart preserved CPU color path, ISO240, f/1, shutter1 and effective scale2.

App state now uses a typed RON payload inside the existing outer storage map, while DisplayPrefs stays a JSON string. `Squarebob src/app/state.rs:142-149` accepts RON plus valid legacy JSON; `Squarebob src/app/mod.rs:782-834` saves the complete typed state. `Squarebob src/app/persistence_tests.rs:5-43` verifies full-state roundtrip including nonfinite dock rectangle sentinels, finite floating-window position/size, topology and CPU/camera settings. Previously corrupted JSON null-rectangle snapshots report decode failure; no geometry is guessed or stripped to migrate them.

Final binary unit suite:34/34 passed in0.13s (`squarebob_pq_bin_tests.stdout.log`). Final locked workspace/all-target check passed in6.313s with empty logs; binary build passed in16.140s with empty logs. Earlier measurements remain historical. Final Clippy exited0 in10.685s, with464 stderr lines of warnings (`pq_final_clippy.stderr.log`); this is not a warning-free result. The official Rust API checklist was consulted for getter naming, validation, meaningful errors and intermediate results; Context7 supplied official winit0.30 documentation.

Manual clipboard/accessibility, multi-monitor transitions, all alternative output modes and physical luminance remain unverified. Actual user-scene progressive noise remains unresolved. The authorized main push is prepared, awaiting the confirmed commit/remote receipt.

## Confirmed Squarebob publication — 2026-10-02

The orchestrator committed and successfully pushed the explicitly authorized CPU/PQ display and persistence work as `5465cb083769bbcd27f04f41b4e46457a4f51d31` to `origin HEAD:main` (`5f73509..5465cb0`). Independent `git ls-remote origin refs/heads/main` equals the full local HEAD hash, and `git status --porcelain=v1` is empty. This supersedes earlier pending-publication checkpoints. No OIDN commit or source repair was included; unrelated denoiser/scheduling proposals and actual-scene noise investigation remain open.

- [x] Commit the reviewed, tested Squarebob changes.
- [x] Push main and independently verify the remote commit.
- [x] Verify the published Squarebob working tree is clean.
