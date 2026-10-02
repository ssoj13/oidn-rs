# Astra numerical review: progressive highlight detail

Date: 2026-10-02. Scope: source review and analytical derivation; no production edits, builds, or GPU runs by this reviewer. Existing GPU measurements are attributed to Squarebob plan16. Read-only `cargo tree --locked --offline -e features -i burn-cubecl` completed successfully in Squarebob to establish the effective default feature graph. GitNexus tools were unavailable in this consultation; no graph freshness claim is made.

## Evidence and source notation

The user reports localized noise in highlights/fine details growing during one progressive render. Growth beyond 256 SPP is not established. The inspected Squarebob path explicitly selects HDR; this does not retroactively prove every image described by the user was HDR. This report extends [astra_review.md](astra_review.md), [squarebob_bridge.md](squarebob_bridge.md), and [Squarebob plan16](../../squarebob-rs/docs/plans/plan16.md).

Source prefixes below identify exact checked trees:

- `OI/`: `C:/projects/projects.rust.cg/cglibs/oidn-rs/`.
- `SB/`: `C:/projects/projects.rust.cg/cglibs/squarebob-rs/`.
- `LOCAL2.5/`: `D:/Projects/vfx.ref/oidn/`, supplied clean v2.5.0, commit `f7ae1bf07b3201aaa8cfe04d71f5243f8e0f2bb7`.
- `B/`: `C:/Users/joss1/.cargo/git/checkouts/burn-6c277d792b0d5d7a/93e63e8/`, full revision `93e63e8e9cf215ffb870ed62132961588a774096`.
- `CU/`: `C:/Users/joss1/.cargo/git/checkouts/cubecl-058c47895211d464/31c5506/`, revision `31c5506cfc1dd9d350124910fb51cee8b54a31e1`.
- `CK/`: `C:/Users/joss1/.cargo/git/checkouts/cubek-21eb4731b65c1fbd/55567e4/`, revision `55567e4ff2708499b0a329e57655208c2a77f71a` (`SB/Cargo.lock:2167-2169`).

These dependency paths are local pin evidence, not claims about newer upstream releases. The feature graph resolves OIDN dependency commit `bebbfc5e`; checkout documentation changes do not by themselves change that renderer dependency.

## Main conclusions

1. The synthetic checker demonstrates a large deterministic response to changed input policy. Its spatial SD does not establish denoiser-created noise: the adaptive clamp removes the checker before the network at 128 SPP and lets contrast return at 256 SPP.
2. Native CUDA and current Burn WGPU have materially different precision/storage/accumulation paths. Identical logical topology and model weights do not imply numerical equality.
3. The inspected default Squarebob graph has neither Burn fusion nor convolution autotune enabled. Its source dispatch selects Direct convolution with default f32 tensors. A theory involving changing autotuned tensor-core kernels is unsupported for this configuration.
4. The inverse PU curve can amplify a small normalized network error into a visibly larger absolute highlight error. This is a mathematical sensitivity, not evidence that it caused this scene's progression.
5. First compare identical saved bridge inputs across native and Burn with a fixed explicit scale and identified weights. Then separate restored real detail from residual stochastic error using a known clean target. Current measurements do not justify assigning the user's scene to a Burn defect or prescribing half precision as a repair.

## What the checker actually proves

The probe sets RGB to `(v, 0.8v, 0.6v)`, with bright-region checker levels `v=8` and `v=48` (`SB/examples/oidn_highlight_probe.rs:177-185`). The bridge computes luminance with `0.2126, 0.7152, 0.0722` and multiplies RGB by `min(max_lum / max(lum, 1e-6), 1)` (`SB/crates/pt-denoise-oidn/src/lib.rs:879-894`). Therefore their luminances before clamping are:

```text
k = 0.2126 + 0.8*0.7152 + 0.6*0.0722 = 0.82808
low  = 8*k  = 6.62464
high = 48*k = 39.74784
```

The default adaptive threshold is `2 + (10-2)*smoothstep(clamp(SPP/256,0,1))`: it is 6 at 128 SPP and 10 at 256 and above (`SB/crates/pt-denoise-oidn/src/lib.rs:457-470`; defaults `SB/crates/render-shared/src/lib.rs:1128-1135`).

At threshold 6, both levels clip to the same chromaticity and luminance 6; the checker disappears algebraically before denoising, apart from finite-precision rounding. At threshold 10, the low level remains 6.62464 and the high level becomes 10. The network now receives spatial contrast. The measured highlight output SD increase from 0.003034 at 128 SPP to 0.065254 at 256 SPP is consequently not a noise-only comparison. The statistics code measures spatial population SD, not error against a clean image (`SB/examples/oidn_highlight_probe.rs:135-157`).

Plan16 also records exact equality for fixed-clamp SPP 1 versus 256 and 32 repeated fixed-clamp executions through the cached bridge on RTX 3080 Ti/Vulkan. This supports determinism for those frozen inputs and that configuration. It does not prove general race absence or reproduce the user's scene.

The SPP-dependent clamp plateaus at 256. If the actual symptom keeps growing well beyond 256 with the same settings, this particular threshold schedule cannot itself keep increasing: inspect changing raw radiance/AOVs, scale, scheduling/display snapshots and noise realization. Its earlier alteration of the image remains relevant.

## End-to-end numerical correspondence

| Stage | Checked Rust/Burn path | Checked LOCAL2.5 path | Consequence |
|---|---|---|---|
| Scale selection | Fresh transfer state each call; explicit scale precedes HDR exposure (`OI/crates/oidn-rs/src/filters/unet_runner.rs:89-100`). Bridge environment override precedes supplied physical-camera scale (`SB/crates/pt-denoise-oidn/src/lib.rs:515-518`). | Transfer stores scale and reciprocal (`core/color.h:139-144`). | Freeze the actual numeric scale for backend comparisons; manual autoexposure is a separate experiment. |
| Sanitation and transform | All nonfinite input becomes zero before scale; then clamp and transfer (`OI/crates/oidn-rs/src/gpu_ops.rs:29-32,51-64`). | Multiply by scale first, replace NaN only, clamp infinities, then transfer (`devices/gpu/gpu_input_process.h:37-54`; `core/math.h:77-79`). | Finite ordinary inputs can be paired directly; extreme/nonfinite cases have an existing contract difference. |
| PU normalization | `1/PU(65504)`, same constants and piecewise equations (`OI/crates/oidn-rs/src/color.rs:14-16,38-55,126-157`; tensor operations `gpu_ops.rs:108-124,151-196`). | `core/color.cpp:10-15`; `core/color.h:77-106,180-206`. | Same formulas do not guarantee identical device exp/log/pow rounding. 65504 is the normalization reference, not an input clamp here. |
| Pack/storage | Model loader decodes half weights to f32 tensors (`OI/crates/oidn-model/src/loader.rs:64-68,93-97`). WGPU default float dtype is f32 (`B/crates/burn-backend/src/lib.rs:69-87`). | CUDA tensor and weight dtype Float16, HWC/OHWI and channel block 8 (`devices/cuda/cuda_device.cpp:215-219`); input process casts transformed floats to destination elements (`devices/gpu/gpu_input_process.h:120-175`). | Native CUDA rounds packed activations to half before the first convolution. Comparing this to unquantized Burn input must expect a difference. |
| Graph and convolution | 3x3 biased conv, explicit padding 1 (`OI/crates/oidn-model/src/unet.rs:23-27`); ReLU/pool/nearest/concat sequence `:104-153`. | Base operations `core/unet_filter.cpp:468-497`; graph creates typed, channel-padded weight/bias tensors and reorders them (`core/graph.cpp:87-140`). | Logical correspondence is necessary, not a numerical proof; compare decoded/reordered values independently. |
| Quality and accumulation | RT quality participates in model candidate selection; model construction has no equivalent native fastMath quality switch (`OI/crates/oidn-rs/src/filters/rt.rs:504-563`). | `fastMath = quality != High` (`core/unet_filter.cpp:262`). CUDA demands at least Float32 accumulation for High; other qualities allow the source dtype minimum and select among compatible kernels (`devices/cuda/cuda_conv.cu:28-48`). | Balanced does not mean it always accumulates in half; High does not mean all intermediates become f32. Compare the same archive before interpreting a quality toggle. |
| ReLU/output storage | Explicit Burn ReLU after conv; tensor dtype preserved (`OI/crates/oidn-model/src/unet.rs:104-142`). | CUTLASS ReLU epilogue uses Element as output and epilogue compute type, with separate ElementAccumulator (`devices/cuda/cutlass_conv.h:55-77`). Convolution output dtype follows source (`core/conv.cpp:42`). | Native High still rounds each stored output to half. Fused bias/ReLU and separate f32 operations need not round identically. |
| Inverse transform | Sanitize network output, inverse transfer, output scale (`OI/crates/oidn-rs/src/gpu_ops.rs:79-96,118-124,179-196`). | `devices/gpu/gpu_output_process.h:48-69`. | Capture before and after inverse PU; final radiance error alone obscures its origin. |

Native CPU is a useful second baseline but its name does not identify its precision. The supplied source defaults to Float32 and has an AMX FP16 branch (`LOCAL2.5/devices/cpu/cpu_device.cpp:157-182`). Record ISA/backend and actual tensor dtype. Do not call a native GPU versus Burn GPU difference a Rust error without this context.

### Current Burn dispatch and rounding

The successful read-only feature command produced 97 lines with no `fusion` or `autotune` feature. Its log is `C:/Users/joss1/.filesystem-mcp-rs/tmp/run_command_1790967352717_2bbb7333-4873-4be0-8795-5debdfbaa182_stdout.log:1-97`. This applies to the checked default Squarebob target, not every possible Cargo feature selection.

Burn convolution delegates with its default strategy (`B/crates/burn-cubecl/src/ops/module.rs:60-66`). The default is Direct when autotune is absent (`kernel/conv/base.rs:23-43,82-97`). NCHW is permuted for NHWC kernel execution and returned to NCHW (`:53-65`). Direct initializes its accumulator from bias (`kernel/conv/direct.rs:63-65`), performs bounds-checked cross-correlation, and sums input/weight products (`:145-151,180-214`). Output dtype follows input (`:223-265`). With the inspected default tensors, this is a source-level f32 path.

The generic ndarray implementation starts from zero and adds bias after products (`B/crates/burn-ndarray/src/ops/conv.rs:139,151-196`); the module can select SIMD separately (`ops/module.rs:52-65`). Different accumulation and bias order already invalidate bitwise CPU/GPU equality as a universal criterion. Shader compiler contraction/vectorization may further affect it; the source-level loop is not an instruction trace.

If autotune is explicitly enabled later, it can select Direct, im2col or implicit GEMM strategies (`B/crates/burn-cubecl/src/kernel/conv/forward/tune.rs:26-81`). Its key includes dtype, dimensions and convolution options (`:114-164`); that is a configuration change to record, not evidence for today's symptom. Cubek's matmul element selection also distinguishes operand/stage/output/accumulator types (`CK/crates/cubek-matmul/src/definition/spec.rs:276-295`). Do not infer TF32 solely from “NVIDIA GPU.”

CubeCL's SPIR-V arithmetic lowers exp/log and pow separately (`CU/crates/cubecl-spirv/src/arithmetic.rs:360-376,613-662`). Fast-math decoration depends on supported capabilities and requested modes (`compiler.rs:532-566`). Native OIDN's convolution `fastMath` flag is not a global promise about transcendental accuracy. Neither implementation's source establishes exact driver transcendental results; capture generated shader/kernel identity if the first mismatch lies there.

## Why small differences can be prominent in highlights

For the high branch of PU inverse, let `u` be normalized network output, `r=PU(65504)`, and `s` the positive input scale. The output channel is:

```text
c = (exp((r*u - g)/e) - f) / s
dc/du = (r/e) * (c + f/s)
```

Using the checked constants gives approximately `r=3.135119` and `r/e=16.2734`. These are analytical evaluations of source constants, not GPU measurements. In sufficiently bright values, a normalized perturbation of 0.0001 corresponds locally to roughly 0.163% radiance change. Its absolute value grows with brightness. Half activation rounding, convolution cancellation, and transcendental differences can therefore become much more visible in highlights than their raw normalized magnitudes suggest. This does not prove they are large enough to explain the user's image.

The network contains biases and ReLUs, and PU is nonlinear. Changing input scale and multiplying the output by its reciprocal does not make the complete denoiser scale-equivariant. Changing a clamp loses information before either implementation; no output rescaling restores it. Thus input policy and scale can alter which fine structures survive even with a correct model.

For this probe, clamp 10 and input scale 0.02 bound the largest color channel at about 0.241523 before PU and 0.232792 after normalized PU. This particular ordinary input is nowhere near half input overflow. Intermediate network values must still be measured; do not extrapolate that bound through learned convolutions. Conversely, extreme finite values or a bad scale can overflow after the early sanitation point, and inverse PU can overflow after network-output sanitation. Record finite counts at each stage instead of hiding all bad values in the final image.

## Noise, signal, and progression

Spatial variation is not synonymous with noise. A clean checker or fine specular structure has variance; removing it can make an image smooth while making reconstruction worse. The present checker contains no declared noisy-versus-clean pair, so its output SD cannot quantify denoising quality.

For independent unbiased finite-variance Monte Carlo samples of a fixed quantity, variance of the sample mean falls as 1/N in expectation. A single progressive realization need not improve monotonically; rare bright paths can become visible later. Adaptive clipping changes bias and the effective signal with N, and a nonlinear denoiser changes the relation further. Therefore “more visible detail at higher SPP” is compatible with genuine detail recovery, newly encountered fireflies, or denoiser artifacts. Source inspection alone does not choose among these.

The shared runner creates transfer state and output accumulation anew per call (`OI/crates/oidn-rs/src/filters/unet_runner.rs:89-100,137-138`). Combined with the bridge and repeated-input results, there is currently stronger evidence for an input-dependent effect than an internal image-history mechanism. This narrows the next experiment; it does not establish a general absence of asynchronous bugs.

## Next paired diagnostic corpus

First save one actual-scene snapshot at several progressive points, including 128, 256, and above 256 if the symptom persists. Save raw normalized RGB, auxiliary RGB plus their counts, post-clamp RGB, exact scale, settings, weight hash, selected model, device/compiler and presented snapshot SPP. Compare the raw and denoised highlight ROIs at the same display transform.

Use the following small controlled corpus before expanding random coverage:

| Input | Controlled comparisons | Question |
|---|---|---|
| Frozen current checker | Fixed clamp 6, 10 and disabled; same explicit scale; feed identical post-clamp arrays to each backend | Does the existing result follow restored input contrast? |
| Uniform clean HDR patches with seeded zero-mean noise | Multiple seeds; fixed mean; nested sample averages at increasing N; fixed clamp/scale | Does residual error decrease, and does either backend introduce structured variation? |
| Clean fine patterns, thin bright lines, edges and smooth highlight gradients | Save exact clean target; add separately controlled noise; sweep contrast around clamp thresholds | Is apparent roughness residual noise, ringing, or correct detail recovery? |
| Sparse bright impulses over smooth radiance | Hold impulse position/energy fixed, then vary occurrence separately | Are fireflies/clipping mistaken for temporal state? |
| Brightness and scale sweep | Cross PU branch boundaries and the actual scene's operating range; scale around the actual chosen value | Is there a transfer discontinuity, saturation or precision-dependent threshold? |
| Same valid crop and translated pattern | Initially aligned dimensions with one tile; later odd dimensions and multiple tiles | Does error follow content or padding/tile boundaries? |

Keep actual full-AOV/noisy-aux semantics first. An AOV-disabled run also changes model selection; interpret it as a separate model comparison, not a pure arithmetic toggle. Freeze both AOVs and color to separate input evolution from execution state.

Run native CUDA High and Balanced using identical custom bytes if supported by the harness, plus native CPU with recorded ISA and Burn CPU/WGPU. Native High can select a different archive automatically, so a raw quality-button comparison confounds model and arithmetic. Weight equality is a prerequisite being checked by the parallel native investigation; this report does not claim native archive equality.

## First-divergence capture and tolerances

Capture at these semantic boundaries in a diagnostic harness, with layout/dtype metadata and finite/min/max statistics:

1. Actual RGB/AOV values after upload/stride trimming and application clamp; explicit scale value.
2. Valid transformed channels before spatial padding; packed input after native destination conversion. Compare native half packing both to Burn's original f32 values and to explicitly half-rounded copies, so expected storage quantization is identified.
3. Each convolution before/after ReLU where accessible, then each pool, nearest upsample and concat. Native fused operators may require capture at equivalent semantic outputs rather than matching operation names.
4. Final normalized network output before inverse PU, after inverse PU and reciprocal scale, then cropped/stiched RGB and displayed texture.

For the first materially divergent convolution, use exactly the same captured input tensor and decoded weights on each backend. This prevents previous-layer differences from being blamed on the current operator. Check one selected output's dot product in higher precision. Record kernel, precision, padding, channel order, bias order and fused activation. For native CUDA, quantize to its actual source dtype before comparing operator arithmetic.

Use structural equality for shape, channel order, crop rectangles, expected padding and identical decoded weight elements. Require finite output for the ordinary finite corpus. Preserve NaN/+Inf/-Inf as explicit policy tests with different documented expected behavior.

Do not choose one absolute HDR tolerance for every stage. Report max and RMS absolute error, relative error with a stated luminance floor, and normalized-PU error. For the clean/noisy corpus report error to the clean target, ROI bias and variance across seeds, edge/contrast preservation, and spatial residual structure. Raw spatial SD alone is insufficient.

Establish tolerances from the measured baseline for each backend/dtype pair and retain outliers, rather than declaring an arbitrary epsilon to mean parity. At a half store, rounding error can be assessed against half ULPs for finite non-overflowing values; it is not a bound on the entire network. A dot-product bound based on the actual sum of absolute products and accumulation precision is more meaningful than relative error near a cancelling output. Final inverse-PU tolerance should be propagated through its local derivative, then checked against a practical image-error target. Matching topology does not supply that target.

The repeated frozen Burn run already gives a useful strict diagnostic expectation for this device/configuration: bitwise equality. Failure to retain it after a change warrants investigation, but passing it says nothing by itself about accuracy. Native-half versus Burn-f32 comparisons are expected to be nonzero; isolate and classify the first difference before declaring a bug.

## Architectural recommendation

Keep a reproducible diagnostic manifest separate from production inference: input hashes, model archive hash and descriptor, effective scale/clamp, dtype, Cargo features, backend/kernel identity, tile plan and stage captures. Reuse the existing transfer/model descriptor work proposed in plan2; do not add a second numerical preprocessing implementation to the renderer.

A shared semantic preprocessing specification and independent golden vectors are preferable to deduplicating all code before evidence is collected. Preserve backend-specific storage/accumulation metadata. The immediate useful work is a small native/Burn replay harness and actual-scene snapshots; a broad pipeline rewrite would obscure which change affected the symptom.
