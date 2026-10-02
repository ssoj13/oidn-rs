# Progressive renderer audit — checkpoint, 2026-10-02

This report extends [squarebob_bridge.md](squarebob_bridge.md). It does not repeat its complete bridge inspection and does not assign a cause to the user's scene. The raw PT audit was interrupted by the user's explicit request to implement Playa-equivalent winit/PQ display output. This checkpoint records inspected source and remaining work so the broader audit can resume after that authorized implementation. No PT accumulation, RNG, guiding, backend or denoiser repair was made by this agent.

Repository: `C:/projects/projects.rust.cg/cglibs/squarebob-rs`. References below are relative to that root unless explicitly prefixed. Original pipeline comparisons remain pinned OIDN v2.4.1; the separately supplied native reference is clean LOCAL2.5 at `D:/Projects/vfx.ref/oidn`, SHA `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7`. Reference OIDN does not define the application's path tracer, so renderer-specific findings require its own evidence and reproduction.

## Scope and verification boundary

- [x] Read backend dispatch wrappers and entire shared renderer frame routine.
- [x] Trace camera uniform construction, reset order, dispatch batches, raw output and AOV selection.
- [x] Read shared RNG, shader numerical contracts, wavefront raygen/finalize.
- [x] Read wavefront shade main and selected megakernel accumulation/jitter/AOV sections.
- [x] Inspect backend setter and R2/filter history with read-only git.
- [x] Compare Squarebob display transport with EXR/Playa and shared egui-display.
- [ ] Finish all megakernel light transport, GGX/MIS/roughness/normal/transmission numerical paths.
- [ ] Finish wavefront intersection/shade helpers, ReSTIR and guiding passes.
- [ ] Complete adaptive sampling identity/count/cap/reset/export coverage.
- [ ] Capture raw progressive HDR/AOV snapshots from a real scene at 128, 256, 512 and 1024 actual samples.
- [ ] Compare scene snapshots to native OIDN using matching weights/configuration.
- [ ] Establish the user's actual backend, sampler, scene, exposure and denoiser scheduling.
- [ ] Identify a demonstrated last-good regression; commit chronology alone is insufficient.

GitNexus MCP is absent in this current tool inventory. A stale graph is not trusted. The parent attempted CLI impact using the absolute repository and encountered the previously documented catalog scan timeout. Direct filesystem MCP source/callsite reads and read-only history are the explicit fallback. The GUI host affects all render modes; no successful graph impact rating is claimed.

## Verified dataflow

```text
App Render3DOptions + camera + scene + CPU update budget
  -> Renderer3D::prepare_scene (render-3d/src/lib.rs:467)
       -> choose backend (477), ensure targets (488), cache/layout, scene dirty (531)
  -> shared PT render_frame (pt/megakernel/render.rs)
       -> lazy PathTraceCompute / resize / upload scene
       -> camera uniform constructed with frame_count+1 (251-276)
       -> camera/slice/animation resets (311/325/363)
       -> one camera-buffer write for this CPU frame (389)
       -> configure ReSTIR/adaptive/guiding/spectral/environment
       -> dispatch batch (440-481)
          megakernel:
             integer variance.count (bvh_traverse.wgsl:968)
             RNG(pixel,camera.frame,count) (972-978)
             jitter(camera.frame) -> Gaussian weight (266-298,980-981)
             RGB weighted sum / weight sum; matching weighted AOVs
             Welford integer sample count -> normalized Rgba32Float output
          wavefront:
             raygen -> intersect -> shade -> finalize
             primary-hit AOV RGB sums / integer count
             raygen seed depends on fixed camera frame (raygen.wgsl:94)
       -> output_texture raw HDR, active AOV accessors
  -> App OIDN scheduling -> pt-denoise-oidn (see bridge report)
       -> fresh color/AOV tensors -> immutable committed RT
       -> separate denoised result texture
  -> display composite -> extended-sRGB float producer (authorized PQ follow-up)
  -> egui float canvas -> shared PresentPass
       -> actual supported surface format + color space -> queue present
```

The backend facade is `crates/render-3d/src/pt/mod.rs:18-40`; the spectral wrapper delegates to the shared routine with adjusted spectral options, rather than an independent accumulator. The GPU-only wrapper `crates/render-3d/src/pt/megakernel/render_no_readback.rs` uses the same frame routine. GPU submission is `render.rs:519`; CPU readback, when requested, is `:507-530`.

## Confirmed renderer defects, with reachability limits

### R1 — backend identity and AOV source diverge after wavefront-to-megakernel switch

`crates/render-3d/src/pt/megakernel/render.rs:460-476` enables wavefront only in the wavefront branch. The megakernel branch never calls the same setter with false. The repository-wide inspected callsites have no other disable call. `crates/render-3d/src/lib.rs:477` selects the desired facade backend but does not synchronize `PathTraceCompute.wavefront_config.enabled`.

AOV accessors `crates/pt-megakernel/src/compute.rs:1918-1930` return wavefront buffers while that flag stays true. The megakernel shader bind group instead binds its own albedo and normal buffers at `:4823,4827`. Thus the active raw RGB can come from megakernel while the denoiser receives the wavefront AOVs. The first-megakernel reset at `:5143-5144` clears the accessor-selected wavefront buffers, not the megakernel shader's own buffers.

This is a source-confirmed identity/reset defect when switching on an existing compute instance. It does not explain an uninterrupted megakernel run without such a transition. Systematic proposal: synchronize the requested backend through the existing setter before dispatch, and have one authoritative backend identity govern dispatch, AOV access, and reset ownership. Clear exactly the buffers the next backend writes; validate WF->mega->WF transitions and AOV/sample counts.

### R2 — wavefront setter cannot re-enable an already created pipeline

`compute.rs:1845-1851` sets enabled=true only when the pipeline is absent. After an explicit disable, re-enabling an existing pipeline does not set the flag true. The transition reset still occurs at `:1853-1855`, making the selected buffer identity inconsistent with the requested state.

Read-only history command `git log -3 -L 1838,1858:crates/pt-megakernel/src/compute.rs` shows this structure already in initial commit `2ce275f` (2026-04-19); `9f83b1e` (2026-07-15) added fallible construction but retained the state logic. This history is not evidence of the user's later regression. Repair should reuse the existing setter: initialize resources if needed, then assign the requested flag consistently and reset only on a real transition.

### R3 — tiled wavefront diffuse PDF is inconsistent with the actual sampling distribution

`crates/pt-megakernel/src/compute.rs:3450-3473` sets both guide_enabled=0 and guide_product=0 for tiled execution but retains the configured guide_weight. Default `pathguide/config.rs:24-26` has product sampling=true, guide_weight=0.5 and warmup8; tiling overrides the product flag.

In `crates/pt-wavefront/src/wavefront/shade.wgsl:377-379`, the direction is a cosine sample, PDF starts zero, and disabled guiding performs no alternate sample. Fallback `:405-407` nonetheless uses `(1-guide_weight)*cos/PI`. Throughput `:441-443` divides the diffuse BRDF by that underweighted PDF. With weight0.5, a positive-cosine diffuse sample receives twice the ordinary cosine-sampling throughput before roulette compensation. The issue repeats per applicable diffuse bounce.

This algebra is a confirmed sampling/PDF mismatch for the tiled wavefront configuration. It is not attributed to default megakernel or to the user's scene. Repair the existing sampling/PDF contract for disabled guiding, unavailable guided samples, product sampling, and actual mixtures; verify energy and variance on simple diffuse scenes. A true mixture must evaluate the density of the distribution actually sampled.

## Confirmed sampling behavior requiring reproduction

### R4 — CPU batches reuse the camera frame for subpixel sampling

`render.rs:251-276` constructs one camera uniform and `:389` writes it once before the sample loop `:460-479`. The CPU frame counter advances inside `compute.rs:5151` for each megakernel dispatch, but the camera buffer is not updated per dispatch.

Megakernel R2 jitter `bvh_traverse.wgsl:266-273,980` depends on camera.frame_count alone. Therefore all sub-samples of a CPU batch share the same subpixel location in R2 mode. Wavefront raygen `wavefront/raygen.wgsl:94-98` seeds the same pixel with camera.frame_count in both frame/sample slots, likewise reusing primary jitter within the batch. This establishes correlation in primary sampling, not that all complete light paths are identical (other passes/history may vary).

Crucial double-check: current megakernel RNG sample identity is the integer `variance[pixel].count` (`bvh_traverse.wgsl:968,975`), **not** truncated Gaussian accumulator weight. A suspected weighted-count RNG defect is not present in that current megakernel code.

History: R2 helper first appears in `f493354` (2026-05-08); Gaussian filter is added in `48450bd` (2026-05-16), preserving R2 frame dependence. Full camera/batching history and a samples-per-update1 versus25 convergence comparison remain necessary. Use a single explicit per-pixel sample ordinal for sampling across both backends; preserve reconstruction weights separately.

## Accumulator and AOV facts that were checked

Megakernel `bvh_traverse.wgsl:294-298` uses positive Gaussian reconstruction weights. Final accumulation `:1508-1533` stores weighted RGB and weight sum, increments the independent Welford integer count, and normalizes RGB by weight sum. Primary misses add zero auxiliary RGB with the same sample weight at `:1007`; primary hits add albedo/normal with that weight at `:1045-1050`. These selected sections are consistent with the bridge dividing AOV RGB by W, and do not by themselves prove a count bug.

Wavefront `wavefront/finalize.wgsl:24-33` increments an integer sample denominator on termination and normalizes its raw color. Its output clamp0..100 differs from megakernel's per-path luminance clamp1000 (`bvh_traverse.wgsl:771-776`); parity and clipping policy need full transport review. The absence of a current finite check inside that clamp is only a candidate: Inf*0 can be NaN mathematically, but upstream guards/reachability are not completely inspected.

Wavefront sample-map initialization is checked: `compute.rs:1772-1775` fills adaptive positive maxSPP or u32::MAX. The raygen zero-limit interpretation is therefore not currently demonstrated as a reachable app bug.

## Correct interpretation of the frozen bridge experiment

The prior successful diagnostic remains useful evidence: repeated fixed inputs at256SPP, including32 additional fixed-clamp runs, returned exactly equal outputs on the tested device/configuration. It does not prove general race absence.

Do not label its increasing bright-region standard deviation as residual noise or a clamp defect. Its checker has RGB=(v,0.8v,0.6v), whose bridge luminance is 0.8281004*v: low8 has luminance6.6248032 and high48 has39.7488192. At128SPP the adaptive clamp6 maps **both** colors to the same proportional RGB. At256SPP clamp10 retains the low cell and clamps only the high cell, restoring checker contrast **before the network**. Increased output variation can therefore be legitimate restored detail. The reported max absolute RGB difference8.448264122 proves sample-dependent input-policy influence only. The adaptive clamp is constant after256SPP; ongoing growth after that point requires another explanation or changing scene inputs/scheduling.

## Headless capture requirements

The current CLI screenshot is display RGBA8, not a raw HDR/AOV progressive dump: `src/app/screenshot.rs:116-143` calls render-target readback. Its actual flags are `--mode 3d --path-trace --samples N --pt-spp N --screenshot SECONDS --screenshot-path FILE --exit-after-screenshot DIR` (`src/cli.rs:141-153,253-255`). A GUI run validates lifecycle but cannot supply raw PT evidence by itself.

Existing exact compute entry points: `PathTraceCompute::new` (`compute.rs:845`), `update_camera(queue,&PtCameraUniform)` (`:4890`), `dispatch(encoder,queue)->bool` (`:5109`), `output_texture` (`:1893`), and backend-authoritative AOV accessors (`:1918,1927`). Scene construction, upload and readback signatures must be read in full before writing a raw scene harness. Update the camera sample ordinal per accepted dispatch in a diagnostic harness, and compare with production batching as a separate experiment. Record actual per-pixel count/weight, raw finite status, exposure, normalized AOVs and denoiser outputs independently.

## Authorized PQ follow-up

The user explicitly requested the same window output as Playa using winit. Pipeline ownership covers Cargo/main/new generic display host/App UI integration; root owns color-pipeline and rendering transport; dependency agent owns shared float readback. This authorization does not approve R1-R4 or the original denoiser production repairs.

Shared library provenance: `egui-widgets-rs` local HEAD and newly locked source both `06acf66506583e2cef07450ac304d8ae0414b59d`. The checked `egui-display/src/lib.rs:1-3` supplies presentation but explicitly leaves windows/persistence to the host. Its `present.rs:25` canvas is Rgba16Float with extended-sRGB gamma values; `:127-149` negotiates a supported format/color-space pair for PQ; `:202-215` defines the common absolute-nits PQ encoder. Generic host adaptation preserves the application's used root-window input, IME/clipboard/drop handling, AccessKit, repaint, title/close commands, persistent app/window/egui state, autosave/exit/raw-input hooks and asynchronous egui screenshots. Frame-dependent eframe logic is not implemented by Bob, and its previously passed Frame argument was unused. Embedded viewports remain enabled; independent native child viewport support is not claimed.

A passed workspace source check predates the parallel float-readback update. Targeted host persistence/format tests and headless actual PresentPass GPU signal probe are in progress; real window surface negotiation and full UI validation must be recorded separately in the current implementation plan. Current status must be refreshed from logs rather than inferred from this checkpoint.

## Remaining audit sequence

1. Finish source reads listed above, including every numeric guard and PDF along material branches.
2. Capture a fixed real scene with guiding/ReSTIR/adaptive controlled, then vary one setting at a time.
3. Record batch1 versusbatch25 and mega/WF transitions with actual sample/AOV identities.
4. Compare native and Rust denoisers on identical saved raw inputs and matching weight bytes.
5. Investigate historical changes only after an observed failure has an exact codepath.
6. Propose shared repairs with tests and update the approval plan; do not delete unfinished features.
