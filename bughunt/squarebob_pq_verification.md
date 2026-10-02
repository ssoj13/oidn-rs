# Squarebob native PQ and persistence verification

Date: 2026-10-02. Authorized follow-up implementation and runtime checks; this supplements the initial static noise audit. Parent owns final Squarebob plan17 and publication. No OIDN production numerical findings were repaired in this pass.

## Final implementation boundary

Squarebob's generic winit host uses the same egui-display PresentPass, Output capability negotiation, DisplayPrefs, and DisplayState as Playa. Its actual canvas is RGBA16Float and contains extended sRGB gamma RGB; egui samples that canvas and the shared final present stage converts to the negotiated signal. No local duplicate PQ math was introduced.

Host ownership: ../squarebob-rs/src/display_host.rs; main.rs; App integration/settings; Cargo manifest/lock; examples/display_probe.rs. Parent owns central transfer, render targets/shaders, and singular fallible PT composition. render-core ownership belongs to the dependency agent.

The initial native window exposed epaint's required TexturesDelta consumption contract: both set and free lists must be drained, including failed acquisition/minimized frames. The host now drains both lists. Initial failure logs are retained as squarebob_pq_window_pbr_initial.*. The startup logging filter now uses the actual binary module name; actual negotiated output is visible with -v.

## Actual window measurements and artifacts

RTX 3080 Ti / Vulkan / NVIDIA 616.64. Actual surface pair was Rgb10a2Unorm / Bt2100Pq, reported by the running shared host. This is separate from the headless encoded-signal tests.

| Case | Evidence |
|---|---|
| Raster PBR + native egui window | squarebob_pq_window_pbr.stderr.log; renderer squarebob_pq_window_pbr.png; full UI squarebob_pq_ui_pbr.png |
| PT GPU OCIO | squarebob_pq_window_pt_gpu.stderr.log; squarebob_pq_window_pt_gpu.png; squarebob_pq_ui_pt_gpu.png |
| PT CPU OCIO, nonunity Physical exposure | squarebob_pq_cpu_save.stderr.log; squarebob_pq_window_pt_cpu.png |
| Restart from the newly saved typed CPU state | squarebob_pq_cpu_restart.stderr.log; squarebob_pq_window_pt_cpu_restart.png; squarebob_pq_ui_cpu_restart.png |
| Expanded Display output controls / selected CPU | squarebob_pq_controls.stderr.log; squarebob_pq_ui_controls.png |

The expanded native screenshot shows Active HDR10 (PQ), Auto SDR white 240 nits, OS-reported peak 603 nits, full-frame 150 nits, SDR white 240 nits, headroom 2.51, and 10 bits/color. CPU codepath is selected. These are software capability/OS reports, not photometer measurements. OS screenshot appearance does not prove physical monitor luminance or PQ calibration.

CPU/GPU PT fixtures use Ocio mode with the selected ACES SDR view, Physical camera f-number 1, ISO 240, shutter 1 second, EV compensation 0: multiplier 2. Post-run options were checked. OIDN was disabled to isolate display composition. CLI --samples 16 / --pt-spp 1, megakernel, fixed twelve-file synthetic treemap. Low-sample scene appearance does not reproduce the user's progressive noise regression.

Diagnostics used SQUAREBOB_STORAGE_PATH pointing to owned RON files under bughunt; normal default storage is unchanged. Owned windows exit using the existing timed screenshot lifecycle. No unrelated process was killed. The scan cache for the owned fixture is incidental existing behavior; user rendering state was not replaced.

The raster PBR snapshot displays existing OCIO settings UI, but the repaired CPU/GPU OCIO composition contract is scoped to PT. Do not infer raster CPU OCIO support from visible settings.

## Lossless application persistence

A real saved JSON snapshot contained null rectangle coordinates in the remembered DockLayout. These originated from nonfinite egui rectangle sentinels. JSON restore then rejected null as f32 and silently discarded the entire App state. This prevented the first injected CPU diagnostic settings from restoring; those runs are not counted as CPU verification.

App saves the full typed PersistState as RON under the existing squarebob_state storage key. PersistState::decode accepts typed RON and valid legacy JSON. No tab topology, floating-window geometry, or app fields are stripped; no lost infinity signs are guessed for already-corrupt legacy JSON. Such invalid records produce explicit diagnostics containing both codec failures.

The shared DisplayPrefs payload remains a JSON string inside outer RON because its custom enum decoder failed direct RON roundtrip. The outer SessionStorage format/keys remain compatible with existing eframe storage.

Regression coverage uses actual eframe::App::save with App::default, CPU OCIO, nondefault Physical camera, default nonfinite DockLayout sentinels, and an additional finite-position/size floating window. The entire typed reserialized snapshot is identical after decoding; hidden AE/tab topology and camera/codepath are checked. Valid legacy JSON with optional dock fields absent also decodes. Native CPU save then restart used the newly written full RON, retaining CPU, ISO240, dock layout, and infinity sentinels without restore warnings.

Source anchors: ../squarebob-rs/src/app/mod.rs:90 (decode), :831 (save); src/app/state.rs:142 (codec); src/app/persistence_tests.rs:5 (actual save/roundtrip); src/display_host.rs:49 (outer storage).

## Passed verification

- Shared PresentPass headless signal test: gray 0/36.54/100/203/1000 nits plus saturated Rec.709 RGB primaries at 203/1000 nits. PQ primary conversion and transfer use the checked shared oracle. Maximum primary PQ error 0.000542; gray maximum 0.000368. PQ headroom retained; SDR high values clipped; scRGB checks passed. Logs squarebob_pq_signal.*.
- Parent's actual PT blit / Renderer3D pipeline probe: signed value and headroom preserved, maximum gamma-half error 0.00057745. Extended CPU probe verifies immutable raw PT/external denoiser source, repeated processing from source, and exposure applied once before OCIO.
- All 34 Squarebob binary unit tests passed: cargo test --locked --quiet --bin squarebob, logs squarebob_pq_bin_tests.*, 19.665 seconds.
- Final workspace all-targets check passed: cargo check --locked --quiet --workspace --all-targets, squarebob_pq_final_check.*, 6.313 seconds, empty stdout/stderr.
- Final binary build passed: cargo build --locked --quiet --bin squarebob, squarebob_pq_build.*, 16.140 seconds, empty stdout/stderr.
- Final git diff --check passed. Git line-ending conversion notices are not whitespace failures.
- Parent's earlier workspace Clippy passed with existing warnings; no claim of warning-free workspace.

## Limits and handoff

Native capability negotiation, visible controls, source preservation, encoded signal, and CPU state restart are verified on this device/configuration. General HDR monitor behavior, every platform/viewport hook, CPU/GPU numerical equality of approximate LUTs, and the user's scene-noise cause are not established.

GitNexus MCP was unavailable in this phase; parent CLI impact attempts timed out in catalog scanning. Direct source/callsite checks replace unavailable graph output; no successful impact rating or reindex is claimed.

The initial static OIDN findings and unfinished broader progressive shader/history audit remain separately tracked in pipeline.md, local_pipeline.md, squarebob_bridge.md, and progressive_renderer.md. OIIO source audit found no library feedback/hidden-camera-exposure defect: see oiio_cpu_display.md.

- [x] Reuse shared Playa display implementation.
- [x] Verify actual PQ surface negotiation and visible white/output controls.
- [x] Verify CPU/GPU PT window routes with nonunity Physical exposure.
- [x] Preserve raw PT and denoiser input through singular composition.
- [x] Verify full typed persistence and real CPU restart.
- [x] Complete final unit/check/build/diff validation.
- [x] Release Cargo slot to parent.
- [x] Parent publication confirmed below; user-facing limitations remain recorded.

## Confirmed Squarebob publication — 2026-10-02

The orchestrator committed and successfully pushed the explicitly authorized CPU/PQ display and persistence work as `5465cb083769bbcd27f04f41b4e46457a4f51d31` to `origin HEAD:main` (`5f73509..5465cb0`). Independent `git ls-remote origin refs/heads/main` equals the full local HEAD hash, and `git status --porcelain=v1` is empty. This supersedes earlier pending-publication checkpoints. No OIDN commit or source repair was included; unrelated denoiser/scheduling proposals and actual-scene noise investigation remain open.

- [x] Commit the reviewed, tested Squarebob changes.
- [x] Push main and independently verify the remote commit.
- [x] Verify the published Squarebob working tree is clean.
