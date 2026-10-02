# Astra review of Squarebob PQ implementation

Date: 2026-10-02. Independent read-only review of the current worktree. No production changes, builds, tests or GUI launches by this reviewer. Other agents own implementation and runtime verification. Source line numbers below identify the inspected snapshot and can move during their fixes.

## Decision

The GPU OCIO -> extended-sRGB float canvas -> shared presenter contract is coherent in the inspected source. I found no confirmed double-PQ encoding, missing Rec.709-to-Rec.2020 conversion, or erroneous HDR 100/reference-white scale in this path.

The initial review found two existing CPU OCIO integration defects: physical-camera exposure occurred after the view transform, and denoised display bypassed CPU OCIO. Both were present in committed HEAD before the PQ work. The orchestrator subsequently repaired the shared source-stage path; the post-fix source review below confirms both routes are addressed. Runtime validation remains separate. The shared DisplayPrefs RON incompatibility found by the implementation agent is addressed by the current JSON-inside-RON storage change; the source mechanism is verified below.

Actual window negotiation, monitor movement/OS-HDR transitions, persistence/restart, and interaction still require the orchestrator's planned GUI validation. Successful float readback or compilation alone cannot certify the actual monitor signal.

## Checked sources

- `SB/`: `C:/projects/projects.rust.cg/cglibs/squarebob-rs/`.
- `EW/`: `C:/projects/projects.rust.cg/cglibs/egui-widgets-rs/`; shared presenter pin `06acf66506583e2cef07450ac304d8ae0414b59d` appears at `SB/Cargo.lock:3059-3061`.
- `EXR/`: `C:/projects/projects.rust.cg/cglibs/exr-rs/`.
- `PLAYA/`: `C:/projects/projects.rust.cg/cglibs/playa/`.
- `EGUI/`: `C:/Users/joss1/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/egui-wgpu-0.36.2/`; egui 0.36.2 registry pin at `SB/Cargo.lock:3029-3032`.
- Read-only history baseline: Squarebob HEAD `5f73509160629d6eed8e93c6b19355851ec6a727`.

GitNexus was unavailable to this review; direct source/callsite reads were used. No graph freshness is claimed.

## Initial findings (addressed by the later source revision)

### PQ-R1: CPU OCIO applies physical exposure in display light (high impact within this mode)

**Trigger:** OCIO CPU codepath with Physical camera and effective exposure different from 1, especially an absolute HDR view.

The CPU pass reads raw `output_texture`, applies only `pipeline.apply_cpu_to_surface_linear`, uploads it back, then invokes `composite_overlay(None, opts)` (`SB/crates/pt-megakernel/src/compute.rs:5393-5437`; `SB/crates/render-3d/src/lib.rs:1264-1269`). That overlay sets the effective camera exposure (`:1210`), which `blit.wgsl:244-246` multiplies into the already transformed display light. In the GPU route, the same multiplication precedes the OCIO LUT lookup (`blit.wgsl:246,293-304`).

For a nonlinear view V and camera multiplier e, CPU therefore displays e*V(raw), while GPU displays V(e*raw). These are generally unequal. For absolute HDR views the late multiplication also changes the intended nits after the display transform. `ColorPipeline::apply_cpu_to_surface_linear` only applies its processor, sanitation, transport decoding and output scale; it does not consume camera exposure (`SB/crates/color-pipeline/src/lib.rs:898-910`).

**Important narrowing:** legacy EV/WB lanes are currently identity constants: `Render3DOptions::blit_color_lane` returns EV=0, WB=1, gamut-compression=0 (`SB/crates/render-shared/src/lib.rs:834-842`). The active finding is physical-camera exposure (`:849-853`), not an asserted live EV/WB mismatch. If these lanes become active later, pre-view operations should remain shared semantically.

**Historical attribution:** `git show HEAD:crates/render-3d/src/lib.rs` retains the same `:1210,1264-1269` order. This is pre-existing in HEAD, not introduced by the current PQ canvas changes. History output: `C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1790968699972_b7705bf0-8642-477d-9d14-c541b92bb6ab_stdout.log`.

### PQ-R2: CPU OCIO denoised display bypasses the view transform (high impact within this mode)

**Trigger:** OCIO CPU selected and a successful OIDN result displayed.

CPU OCIO selects blit tag 0 (`SB/crates/color-pipeline/src/lib.rs:293-299`). The CPU pass transforms the PT output texture, but the later denoised overlay takes the separate raw `denoised_view` (`SB/src/app/treemap_view.rs:1175-1185,1210-1216`). `composite_overlay` uses that view directly (`SB/crates/render-3d/src/lib.rs:1215-1222`), and tag 0 simply uses scene values (`SB/crates/pt-megakernel/src/blit.wgsl:266`). Thus the denoised image receives exposure and canvas encoding without the selected CPU OCIO view/look/LUT. Selection refresh repeats the same source selection (`SB/src/app/treemap_view.rs:691-705`).

This is a source-established route difference; no screenshot reproduction was performed here. The same denoised overlay call exists in HEAD `src/app/treemap_view.rs`, so it too predates the current PQ patch. Read-only history output: `C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1790968734604_56df1b14-bd3e-4713-882d-93bedc733ded_stdout.log`.

### Smallest coherent repair boundary

A local multiplier change only fixes R1's first render; it does not fix denoised or selection-refresh routes. Keep one explicit distinction between raw scene-linear input and already transformed display-light output:

1. Select the raw source: current normalized PT output or raw OIDN result.
2. CPU path applies the effective camera multiplier before its existing OCIO processor, once.
3. Write the transformed result to a separate display-light staging target; preserve the raw source.
4. Reuse the existing composite operation with neutral exposure and tag 0 for that transformed target; retain the common `display_encode`.
5. Selection/hover refresh reuses the transformed target without another OCIO application. Rebuild its contents when source, camera exposure, OCIO settings, or negotiated HDR/white changes.

Extending existing `apply_cpu_color_pass` / readback helper with an explicit source and reusing `composite_overlay` is preferable to copying the full overlay implementation. A stage enum or explicit parameter is safer than inferring “already transformed” merely from `source.is_none()`. Do not put denoised display light back into OIDN's input or raw accumulation. Do not silently switch the requested CPU mode to GPU LUT processing.

Current callers to preserve: `treemap_view.rs:705` (selection refresh), `:1181` (CPU after PT rendering), `:1216` (denoised display), and `render-3d/src/lib.rs:1269` (CPU reblit). If an in-place PT output remains temporarily, confirm a fresh raw resolve precedes every CPU transform; repeated presentation must not apply OCIO twice.

Meaningful regression cases: non-unit Physical exposure through a nonlinear SDR and HDR view, CPU versus GPU within LUT approximation tolerance; raw/denoised selection under CPU OCIO; selection refresh without new SPP; and changing reference white while the last denoised frame remains displayed.

## Verified GPU display contract

| Boundary | Evidence | Review |
|---|---|---|
| Selected OCIO output -> display reference | `SB/crates/color-pipeline/src/lib.rs:782-807`; `EXR/crates/exr-view/src/ocio.rs:239-259` | Uses selected view's effective color space, its to-display transform or inverse from-display transform; rejects scene-referred non-data output in HDR rather than guessing. |
| Display XYZ D65 -> Rec.709 light | `SB/crates/color-pipeline/src/lib.rs:808-826`; `EXR/crates/exr-view/src/ocio.rs:260-266` | Uses the same conversion matrix and no additional chromatic adaptation. Data views intentionally pass through as display light. |
| Absolute HDR units | `SB/crates/color-pipeline/src/lib.rs:827-829`; `EXR/crates/exr-view/src/ocio.rs:296-303` | HDR encoding is scaled by 100/reference-white. SDR views on HDR output remain relative to reference white, matching exr-view policy. |
| CPU processor / LUT values | `SB/crates/color-pipeline/src/lib.rs:530-541,898-910` | Both apply the same output scale and finite-value handling. R1/R2 concern integration before/after this processor, not this shared scale. |
| Display-light transport | `SB/crates/render-core/src/display_transfer.wgsl:1-7`; `SB/crates/pt-megakernel/src/blit.wgsl:308-311` | Sign-preserving extended-sRGB, without clipping highlights to 1. The transport curve is paired with presenter decoding. |
| Float target and egui | `SB/crates/render-core/src/lib.rs:5`; `SB/src/display_host.rs:218-226`; `EGUI/src/renderer.rs:409-414`; `EGUI/src/egui.wgsl:151-164` | Rgba16Float is non-sRGB and selects egui's gamma framebuffer shader. Texture values are expected gamma encoded; the host disables egui dithering so the presenter controls final quantization. |
| Final signal | `EW/crates/egui-display/src/present.rs:277-301,378-380` | Decodes canvas to light, converts Rec.709 to Rec.2020, multiplies by reference white and applies PQ. Hence HDR's prior 100/white scale gives the intended absolute nits. |
| SDR surfaces | `EW/crates/egui-display/src/present.rs:120-125,153-159,293-300` | Non-sRGB target receives code values; an sRGB target receives decoded light so hardware encodes once. No extra PQ in SDR. |

Example: a neutral HDR display-reference value 10 represents 1000 nits. The OCIO tail writes approximately 10*100/white Rec.709 light; transport encoding and decoding cancel; presenter multiplication by white restores 1000 nits before PQ. Changing reference white should not change this absolute HDR patch, while relative SDR UI white should follow the selected white. This is an algebraic check, not a physical luminance measurement.

The SDR branch preserves selected encoded code values through sRGB decode followed by transport encode (`SB/crates/color-pipeline/src/lib.rs:508-514,667-678`). This is compatibility transport rather than a claim that every external display color space is actually sRGB. HDR decoding uses the config's true display-reference transform. A custom OCIO v1 scene-referred view can be rejected in HDR by design; the error must remain visible.

Raster PBR and skybox now explicitly encode to the same float canvas (`SB/crates/render-3d/shaders/cube_pbr.wgsl:337`, `skybox.wgsl:82`). These are not evidence that raster rendering applies the full PT OCIO view. Transparent raster blending still occurs in the display target; this review does not certify scene-linear compositing parity.

## Host lifecycle and persistence

The inspected host closely follows Playa's redraw/negotiation sequence (`PLAYA/crates/playa-app/src/display_host.rs:267-399`). It preserves the relevant source paths:

- Shared GPU device/queue setup from main (`SB/src/main.rs:164-187`); renderer rebuilt for the float canvas (`display_host.rs:218-226`).
- Hidden initial window until AccessKit attaches, restored geometry, later visibility (`display_host.rs:181-205,227-243,275-277`).
- Actual output negotiation before app frame, published DisplayState, then app runtime HDR/white sync (`display_host.rs:308-321`; `src/app/render_loop.rs:111`; `src/app/mod.rs:60-65`). Runtime HDR/white settings are skipped by serde and included in processor build hash (`crates/color-pipeline/src/lib.rs:193-198,315-316`).
- Root close/cancel-close, platform output, clipboard actions and repaint deadlines (`display_host.rs:314-382`); texture-delta frees even on failed/skipped acquisition (`:470-484`).
- Lost/outdated/suboptimal surface recovery, minimize/zero-size guard, same-queue canvas submission before presentation (`:492-579`).
- Accessibility events, resize/focus/move refresh, scheduled autosave and exit hooks (`:641-766`).
- Whole-window screenshot uses shared presenter capture in SDR for existing screenshot API (`:587-629`). It intentionally cannot establish HDR highlight headroom.

The RON issue is real at the schema boundary: `EW/crates/egui-display/src/settings.rs:21-26` deserializes the enum via `serde_json::Value` and falls back to Sdr8 if conversion fails. The old direct `eframe::get_value/set_value<DisplayPrefs>` used RON for this inner value. Current source instead stores a JSON string under `display_output` inside the original outer RON string map (`SB/src/display_host.rs:125-131,295-303`). Existing unrelated eframe keys and `squarebob_state` remain in that map (`:49-82`; `src/app/mod.rs:822-824`). This resolves the identified codec mismatch without replacing the whole settings file format. Tests and restart results are owned by the implementation agent, not claimed as executed here.

Potential compatibility limitation: an experimental `display_output` inner value already saved as RON will fail the new JSON parse and use default Hdr10. Since this key is newly introduced by the current work, this is not evidence of loss of a pre-existing released preference. If such snapshots must be migrated, use an explicit compatibility decoder rather than the problematic shared RON enum route.

The host embeds viewports into the root. No claim is made that it implements all possible eframe NativeOptions, native multiwindow, mobile suspend/resume, or device-loss rebuild behavior. These are outside the checked desktop feature set. Runtime inspection must still verify that actual surface negotiation and event handling work on the user's adapter/monitor.

## Remaining validation gates

Keep the already assigned checks focused:

1. Verify the repaired CPU scene-versus-display stage handling at runtime for R1/R2, including denoised and selection refresh.
2. Run the final source after all edits through existing color/float-canvas/host checks and binary linking; record exact results separately.
3. In the actual window, confirm requested versus actual output, format/color space, reference white, fallback explanation, moving between monitors and OS HDR changes.
4. Confirm non-default saved display preferences survive restart alongside previous application settings; test close/cancel-close, resize, clipboard, accessibility and screenshots.
5. Check known absolute HDR patches plus SDR UI/gray and saturated colors through the actual presenter. A clipped SDR capture is not a PQ measurement.

No conclusion here assigns the original progressive denoising symptom to presentation or repairs unrelated OIDN/scheduling defects.


## Post-fix source review

The orchestrator supplied a subsequent authorized CPU repair in the same session. I re-read the changed implementations and every production callsite. This supersedes the open-source-defect status of R1/R2 above; their original evidence and historical attribution are retained.

- `SB/crates/render-3d/src/lib.rs:1190-1217` now accepts a raw texture source and the color pipeline. CPU OCIO multiplies raw pixels by effective camera exposure before calling the processor. It processes either the default PT texture or the explicit OIDN result through the same branch.
- The later blit uses exposure 1 for CPU output (`:1220-1229`); CPU's existing tag 0 and identity EV/WB lanes preserve display light until common transport encoding. GPU inputs keep the prior camera exposure/LUT route.
- `SB/crates/pt-megakernel/src/compute.rs:5383-5398` selects and validates the raw Rgba32Float source and dimensions. `:5413-5425` only copies/reads it. Transformed pixels go into a distinct reusable `cpu_color_output` texture (`:5450-5495`), resized when needed; the source is not overwritten.
- The OIDN texture accessor returns the raw result (`SB/crates/pt-denoise-oidn/src/lib.rs:234-236`), whose format is Rgba32Float with COPY_SRC (`:784-787`).
- Both remaining production callers use the shared route: selection refresh `SB/src/app/treemap_view.rs:699-708`, and main raw/denoised PT presentation `:1193-1205`. The old early CPU-only pass has been removed. Main frame ensures the current processor/LUT before rendering (`:1147-1154`) and chooses the OIDN result only after the denoise stage (`:1184-1205`).
- Repeated CPU selection refresh recomputes from the immutable raw source. This is slower than caching display light, but does not repeatedly transform the previous transformed result. Correctness does not require introducing a cache now.

No remaining source-level bypass or compound CPU transform was found in these inspected callsites. Tests should still exercise raw-source equality before/after CPU presentation, repeated same-source presentation, denoised override, and non-unit physical exposure. This reviewer did not execute them.

Small nonblocking responsiveness observation: `SB/src/app/mod.rs:60-66` updates runtime HDR/white without explicitly marking the 3D display dirty. PT normally re-composites on its ticks (`treemap_view.rs:1076-1082`), but marking display dirty on actual HDR/white change would guarantee immediate update while throttled. This is not a demonstrated permanent stale-image bug.
