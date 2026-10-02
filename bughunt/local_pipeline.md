# Local OIDN v2.5.0 pipeline follow-up — 2026-10-02

## Provenance and limits

User supplied reference checkout: `D:/Projects/vfx.ref/oidn`. Filesystem MCP `git rev-parse HEAD` independently returned **f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7** (the supplied detached v2.5.0 reference). Read local input/output/exposure kernels and relevant filter code, re-read relevant current Rust code, and compared ten files against the preserved official v2.4.1 snapshot. No builds/tests, source changes, rollback or external writes; this document is the only newly written file for this follow-up.

**All P1–P11 remain applicable.** P8–P10 include Rust-internal API/contract problems, not only reference parity violations. No frozen renderer input/output or executed reproduction was supplied, so none is established as the user's noise cause. The earlier report remains an historical v2.4.1 audit; this report supplies local-reference citations.

Path notation, always expand before using:
- **R** = `C:/projects/projects.rust.cg/cglibs/oidn-rs/crates/oidn-rs/src/`.
- **L** = `D:/Projects/vfx.ref/oidn/`.

## Local expected dataflow

```text
validate params -> select weights -> parse actual topology -> RF174/202 tile plan
                                  |
select primary: color > albedo > normal
derive transfer: srgb OR normal-only -> Linear; hdr -> PU; otherwise SRGB
derive snorm: directional OR normal-only
                                  |
HDR exposure: component sanitize/clamp -> balanced ceil bins -> log mean -> scale
                                  |
each tile: zeros outside valid source rectangle
           primary: scale -> NaN->0 -> clamp -> optional signed remap -> transfer
           aux alb: NaN->0 -> clamp[0,1]
           aux nrm: NaN->0 -> clamp[-1,1] -> remap[0,1]
                                  |
CNN -> NaN->0 -> clamp[0,FLT_MAX] -> inverse transfer
    -> scalar destination? arithmetic RGB mean
    -> signed demap if snorm -> nonHDR upper clamp1 -> output scale -> write
```

Local anchors: `L/core/unet_filter.cpp:254-268`, `L/devices/gpu/gpu_input_process.h:244-251`, `L/core/rt_filter.cpp:63-70`, `L/core/unet_filter.cpp:551-564`, `L/devices/gpu/gpu_autoexposure.h:29-42`, `L/devices/gpu/gpu_output_process.h:48-72`.

## Finding-by-finding confirmation

| Finding | Current Rust evidence | Local reference evidence | Status against v2.5.0 |
|---|---|---|---|
| P1 normal padding becomes0.5 | `R/filters/unet_runner.rs:161-178` pads raw normal; `R/filters/unet_runner.rs:192-199` remaps whole tile | `L/devices/gpu/gpu_input_process.h:90-118` leaves out-of-source values0; `L/devices/cpu/cpu_input_process.isph:88-93,120-124` writes literal0 | Confirmed; valid normal remap must happen before zero padding. |
| P2 albedo-only bypasses primary scale/transfer | `R/filters/rt.rs:483-493` makes all no-color cases Linear; `R/filters/unet_runner.rs:188-190` only clamps albedo | `L/core/rt_filter.cpp:65-70` makes only srgb/normal-only Linear; `L/devices/gpu/gpu_input_process.h:244` chooses albedo as primary; `L/devices/gpu/gpu_input_process.h:41-54` scales/transfers primary | Confirmed; default albedo-only requires SRGB and explicit input scale applies. |
| P3 normal-only output lacks signed demap | `R/filters/unet_runner.rs:192-199` remaps input, `R/filters/unet_runner.rs:212` output snorm=false | `L/core/unet_filter.cpp:551-564` derives snorm for normal-only; `L/devices/gpu/gpu_output_process.h:59-69` decodes signed output | Confirmed; also primary scale omission from P2 applies to normal-only. |
| P4 directional lightmap loses signs | `R/filters/rtlightmap.rs:294-319` selects Linear/nonHDR but no snorm parameter; `R/filters/unet_runner.rs:186,212` passes false | `L/core/rtlightmap_filter.cpp:58-61` sets hdr=!directional; `L/core/unet_filter.cpp:551` sets snorm; `L/devices/gpu/gpu_input_process.h:45-50` signed remap | Confirmed; negative gradients are clamped to0 in current Rust. |
| P5 exposure drops edges/small images | `R/autoexposure.rs:152-155,169-176` uses floor16 pooling and unity on either axis<16; CPU fixed cells `R/autoexposure.rs:97-102` | `L/core/autoexposure.h:29-31` ceil bins; `L/devices/gpu/gpu_autoexposure.h:29-32,59-62` balanced all-pixel boundaries | Confirmed; CPU oracle geometry also differs. Turning pooling ceil_mode on alone does not match balanced bins. |
| P6 exposure includes negative RGB | `R/filters/unet_runner.rs:60-71,94` sanitizes nonfinite only before exposure; `R/autoexposure.rs:109-119,161-164` computes signed luminance | `L/devices/gpu/gpu_autoexposure.h:40-42` componentwise sanitize/clamp[0,FLT_MAX] before luminance | Confirmed; preserve intentional ACEScg coefficients while repairing sanitation/geometry. |
| P7 Large uses base RF | `R/filters/rt.rs:555-563,578-583` loads Large but plans with RF_BASE | `L/core/unet_filter.cpp:263-268` topology-based RF; `L/core/unet_filter.h:35-37` base174, large202, alignment16 | Confirmed; large overlap112 rather than base96. |
| P8 inconsistent validation/custom weights | `R/filters/rt.rs:421-433` no hdr/srgb check vs `R/filters/rt.rs:738-742`; user bytes skip registry `R/filters/rt.rs:504-516`; release shape assertions only `R/filters/rt.rs:601-605,710-716` | `L/core/unet_filter.cpp:254-260` checks before load; `L/core/unet_filter.cpp:372-380` shape/mode checks; `L/core/unet_filter.cpp:420-443` auxiliary restrictions precede user-weight override | Confirmed Rust-internal and reference-contract problems. Backend tensor batch/channel rules require Rust-specific runtime validation; C++ has no equivalent tensor façade. |
| P9 cache invalidation/handle retention | `R/filters/rt.rs:299-317`, `R/filters/rtlightmap.rs:182-187` image setters ignore changed geometry; `R/filters/rt.rs:839-841` clears tensors then `R/filters/rt.rs:329-334` invalidates on None | `L/core/rt_filter.cpp:73-86`, `L/core/rtlightmap_filter.cpp:32-41` mark changes dirty; `L/core/unet_filter.cpp:372-375` checks image geometry | Confirmed Rust-internal stale validation/unwanted rebuild issues. Do not claim C++ and Rust cache APIs must be identical; ensure promised same-layout reuse and actual geometry validation. |
| P10 sanitation policy is inconsistent/repeated | `R/filters/unet_runner.rs:61-71` optional frame pass; `R/gpu_ops.rs:29-32,51,79` unconditional helper, replacing ALL nonfinite samples | `L/core/math.h:77-79` replaces only NaN; `L/devices/gpu/gpu_input_process.h:41-45` scales BEFORE sanitize; `L/devices/gpu/gpu_output_process.h:48-49` sanitizes then clamps | Confirmed internal switch inconsistency; local evidence adds infinity and operation-order differences (below). |
| P11 scalar filter output takes red | `R/filters/unet_runner.rs:212,318-326` no reduction; `R/image.rs:268-270,285-286` direct writer selects red | `L/devices/gpu/gpu_output_process.h:51-69`; `L/devices/cpu/cpu_output_process.isph:48-66` arithmetic mean after inverse before demap/clamp/scale | Confirmed for R16f/R32f filter output; keep generic direct accessor contract, fix shared filter postprocessor. |

## P10 extension: infinity handling and operation order

The previous Rust docs claim exact reference parity while defining nonfinite replacement as NaN **and** infinity (`R/gpu_ops.rs:25-32`; `R/filters/unet_runner.rs:55-64`). Local helper explicitly returns `isnan(x) ? 0.f : x` (`D:/Projects/vfx.ref/oidn/core/math.h:77-79`); CPU likewise (`D:/Projects/vfx.ref/oidn/devices/cpu/math.isph:102-106`). Infinity then passes to clamp: positive infinity becomes1 for albedo and nonHDR primary, and FLT_MAX for HDR primary; auxiliary signed normal infinities saturate to +/-1 before remapping (`L/devices/gpu/gpu_input_process.h:45,64,74-77`). Default Rust maps these infinities to0 before clamp; auxiliary normals then map0 to0.5 instead of saturation0/1.

Reference primary input is **scale -> sanitize -> clamp -> transfer**, both GPU `L/devices/gpu/gpu_input_process.h:41-54` and CPU `L/devices/cpu/cpu_input_process.isph:35-48`. Rust does sanitize -> scale -> clamp (`R/gpu_ops.rs:51-55`) after a separate optional frame sanitation pass. Hence bad scales or finite multiplication overflow are handled at different stages. Output NaN/infinity semantics also differ (`R/gpu_ops.rs:79-80` vs `L/devices/gpu/gpu_output_process.h:48-49`).

These are confirmed semantics, not proof that input infinities occur in the renderer. Decide explicitly whether a stricter user policy is desired; preserve it under a clear parameter, but do not describe it as exact upstream nan_to_zero behavior. Share one input policy and keep reference-required output sanitation separately defined.

## Differences between local v2.5.0 and audited v2.4.1

Text comparison through filesystem MCP found these seven files identical:
- `core/rt_filter.cpp`, `core/rtlightmap_filter.cpp`, `core/autoexposure.h`.
- `devices/gpu/gpu_input_process.h`, `devices/gpu/gpu_output_process.h`, `devices/gpu/gpu_autoexposure.h`.
- `devices/cpu/cpu_output_process.isph`.

Read-only `git diff v2.4.1 HEAD -- core/unet_filter.cpp core/color.cpp core/color.h` established:
- UNet graph-building API changed at `L/core/unet_filter.cpp:472-491,503-524`: fused `addConvPool` and `addUpsampleConcatConv` replace older PostOp/concat forms. RF selection, mode validation and snorm locations remain unchanged. Model/topology semantic verification is the model agent's responsibility; this follow-up does not infer runtime numeric identity from renamed fusion APIs.
- `L/core/color.cpp:13` now constructs `vec3f(yMax)` explicitly.
- `L/core/color.h:54-71,109-130` adds SYCL SIMD curve overloads and `L/core/color.h:169-214` templates vec3 transfer dispatch. Scalar SRGB/PU constants and curves remain unchanged (`L/core/color.h:31-51,77-106`), so no P1–P11 finding is invalidated.

## Systemic resolution and next evidence

The prior proposals remain correct for this local reference: shared primary/mode/layout contract; preprocess valid source before zero-fill; propagate snorm through existing runner; destination channels drive arithmetic mean at the correct output stage; exact balanced exposure bins and policy; actual model metadata drives RF; shared runtime validation; cache metadata separate from per-pass handles. No parallel denoiser pipeline or generic accessor behavior change is needed.

- [x] Verify local checkout commit.
- [x] Compare P1–P11 with local exact citations.
- [x] Check v2.4.1 differences in all ten referenced pipeline files.
- [x] Read CPU input/output and NaN helper as independent corroboration.
- [ ] Implement only after user approval.
- [ ] Obtain frozen before/after image and AOVs with model/backend/mode/scale metadata.
- [ ] Verify in separate approved test pass on CPU and wgpu, then local native OIDN with identical inputs.

P1/P7 remain candidates for edge/tile artifacts; P5/P6 for exposure-dependent denoising; P2/P3 for separate AOV prefiltering; P4 for directional lightmaps; P11 for scalar output. Infinity handling requires nonfinite sample evidence. No build/test was run and no visual regression cause is asserted.
