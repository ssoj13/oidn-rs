# DIAGRAMS.md — oidn-rs flow & topology in Mermaid

Renders for the parity audit (2026-05-21). See `AGENTS.md` for the ASCII counterparts.

---

## Crate dependency graph

```mermaid
graph LR
    cli[oidn-cli<br/>denoise · bench · probe] --> facade[oidn-rs<br/>RtFilter · RtLightmapFilter · color · tile · autoexp · gpu_ops]
    facade --> model[oidn-model<br/>UNet · UNetLarge · loader]
    facade --> tza[oidn-tza<br/>TZA parser]
    model --> tza
    facade --> burn[burn 0.21<br/>backend = Wgpu OR NdArray]
    cli --> exr[exr crate]
    cli --> img[image crate]
    style facade fill:#ffd,stroke:#664
    style tza fill:#dfd,stroke:#363
```

---

## End-to-end dataflow

```mermaid
flowchart TB
    A[Caller: color + optional albedo/normal images] --> B[RtFilter::builder<br/>.hdr/.srgb/.clean_aux/.quality/.input_scale]
    B --> C[builder.commit]
    C --> D{registry::select_rt}
    D -- ModelKey --> E[weights::resolve<br/>fs OR embedded]
    E -- Vec u8 --> F[oidn_tza::parse]
    F -- TensorMap --> G[oidn_model::Net::from_tza]
    G -- UNet / UNetLarge --> H[unet_runner::run_tensors]
    A --> I[tile::plan<br/>tile_h · tile_w · jobs]
    I --> H
    H --> J[Owned output image]
    J --> K[take_output / user callback]

    subgraph T[per-tile loop]
        T1[1 slice src rect] --> T2[2 zero-pad<br/>Tensor::zeros + slice_assign]
        T2 --> T3[3 preprocess_input<br/>nan→0 · *scale · clamp · forward]
        T3 --> T4[4 concat color · albedo · normal]
        T4 --> T5[5 net.forward]
        T5 --> T6[6 postprocess_color<br/>nan→0 · clamp · inverse · ldr-clamp · *out_scale]
        T6 --> T7[7 crop + slice_assign into accum]
        T7 --> T1
    end
    H --> T1
```

---

## U-Net topology (base)

```mermaid
flowchart TB
    in([input N×C×H×W])
    in --> ec0[enc_conv0<br/>+ReLU]
    ec0 --> ec1[enc_conv1<br/>+ReLU]
    ec1 --> p1[pool 2×2]
    p1 --> ec2[enc_conv2<br/>+ReLU]
    ec2 --> p2[pool 2×2]
    p2 --> ec3[enc_conv3<br/>+ReLU]
    ec3 --> p3[pool 2×2]
    p3 --> ec4[enc_conv4<br/>+ReLU]
    ec4 --> p4[pool 2×2]
    p4 --> ec5a[enc_conv5a<br/>+ReLU]
    ec5a --> ec5b[enc_conv5b<br/>+ReLU]
    ec5b --> u4[upsample 2×]
    u4 --> cc4(concat with pool3)
    p3 -. skip .-> cc4
    cc4 --> dc4a[dec_conv4a<br/>+ReLU]
    dc4a --> dc4b[dec_conv4b<br/>+ReLU]
    dc4b --> u3[upsample 2×]
    u3 --> cc3(concat with pool2)
    p2 -. skip .-> cc3
    cc3 --> dc3a[dec_conv3a<br/>+ReLU]
    dc3a --> dc3b[dec_conv3b<br/>+ReLU]
    dc3b --> u2[upsample 2×]
    u2 --> cc2(concat with pool1)
    p1 -. skip .-> cc2
    cc2 --> dc2a[dec_conv2a<br/>+ReLU]
    dc2a --> dc2b[dec_conv2b<br/>+ReLU]
    dc2b --> u1[upsample 2×]
    u1 --> cc1(concat with input)
    in -. skip .-> cc1
    cc1 --> dc1a[dec_conv1a<br/>+ReLU]
    dc1a --> dc1b[dec_conv1b<br/>+ReLU]
    dc1b --> dc0[dec_conv0<br/>+ReLU]
    dc0 --> out([output N×C×H×W])
```

---

## Tile geometry

```mermaid
flowchart LR
    img((Source image<br/>H×W))
    img --> plan[tile::plan<br/>RF=174 base / 202 large<br/>align=16, max_pixels=2160²]
    plan --> jobs[List of TileJob:<br/>input rect · output_src_in_tile · output_dst]
    jobs --> tloop[per-tile loop]

    subgraph job[TileJob fields]
        f1[input: Rect on src image]
        f2[output_src_in_tile: Rect on tile-output tensor]
        f3[output_dst: Rect on dst image]
        f4[align_offset_x / y]
    end
```

`tileOverlap = round_up(RF/2, align)`  →  base: 96 px, large: 112 px.

---

## Transfer-function decision tree (RT filter)

```mermaid
flowchart TB
    start{which transfer?}
    start -- "hdr=true" --> chk_color{"color present?"}
    chk_color -- "yes" --> pu[PU<br/>perceptual quantizer]
    chk_color -- "no, only nrm" --> linear[Linear]
    start -- "hdr=false, srgb=true" --> linear2[Linear]
    start -- "hdr=false, srgb=false" --> chk2{"only normal?"}
    chk2 -- "yes" --> linear3[Linear<br/>ref rt_filter.cpp:65]
    chk2 -- "no" --> srgb[sRGB]
```

`transfer_kind(has_color: bool, hdr: bool, srgb: bool)` in `filters/rt.rs` mirrors the reference decision tree exactly. `has_color=false` short-circuits to `Linear` for normal-only / albedo-only routes (previously defaulted to sRGB).

---

## Audit baseline — issue density (historical, pre-fix)

```mermaid
quadrantChart
    title oidn-rs parity audit baseline (2026-05-21, pre-fix)
    x-axis "fewer issues" --> "more issues"
    y-axis "lower severity" --> "higher severity"
    quadrant-1 "critical hotspots"
    quadrant-2 "mostly polish"
    quadrant-3 "minor stuff"
    quadrant-4 "single nasty"
    "TZA loader (mendeleev)": [0.15, 0.18]
    "U-Net model (landau)": [0.32, 0.75]
    "Color/HDR (kapitsa)": [0.78, 0.82]
    "Tile/image (pavlov)": [0.55, 0.65]
    "Filters (sechenov)": [0.7, 0.7]
    "GPU ops (ioffe)": [0.65, 0.78]
    "Public API (vavilov)": [0.62, 0.6]
    "CLI (kurchatov)": [0.85, 0.85]
```

All 12 HIGH and the listed MED items are now closed — see commits
`91a261e`, `912aecf`, `f3022a5`, `af69f9d` on `main`.

---

## Fix delivery (status: DONE)

```mermaid
flowchart LR
    H5[H5 dec_conv0 ReLU] --> COMMIT1[91a261e<br/>fix core]
    H1[H1 zero-pad tiles] --> COMMIT1
    H2[H2/H3 output sanitise + LDR clamp] --> COMMIT1
    H4[H4 input clamp after scale] --> COMMIT1
    H6[H6 RG → B replicate] --> COMMIT1
    H7[H7 drop directional from RT] --> COMMIT2[912aecf<br/>fix filter]
    M1[M1 reject invalid combos] --> COMMIT2
    M2[M2 hdr/srgb mutex] --> COMMIT2
    M3[M3 transfer_kind input-presence] --> COMMIT2
    H12[H12 OidnError parity + non_exhaustive] --> COMMIT3[f3022a5<br/>feat api]
    V09[V09 OIDN_REFERENCE_VERSION] --> COMMIT3
    V13[V13 prelude split] --> COMMIT3
    M9[M9 RtLightmap parity] --> COMMIT3
    H8[H8 PFM/PHM I/O] --> COMMIT4[af69f9d<br/>feat cli]
    H9[H9 HDR-precision save] --> COMMIT4
    H10[H10 full flag set] --> COMMIT4
    H11[H11 tracing subscriber] --> COMMIT4
```

---

## Memory & ownership conventions

```mermaid
flowchart LR
    user[User memory<br/>Vec f32 HWC] --> image[Image / ImageMut<br/>borrowed slice]
    image --> rt[RtFilter::set_color/...]
    rt -- inside run_tensors --> burn[Burn Tensor B,4<br/>device-owned]
    burn -- detile + crop --> accum[OwnedImageMut<br/>Rust-owned Vec f32]
    accum -- take_output --> user
```

No `Buffer` abstraction (V vavilov §2). Zero-copy with externally-owned GPU memory is currently impossible (M8); every input round-trips through host RAM.

---

End of diagrams.
