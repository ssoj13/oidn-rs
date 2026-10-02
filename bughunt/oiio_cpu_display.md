# OIIO CPU display integration audit

Date: 2026-10-02. Scope: source audit of the Squarebob CPU display feedback / repeated physical-camera exposure issue. Evidence was supplied by the squarebob_dependencies agent and integrated by the pipeline agent. This is not a complete OIIO correctness certificate.

## Provenance and tools

Local repository: C:/projects/projects.rust.cg/cglibs/oiio-rs, clean main at fa4936bcc4fd9f34d9dded823b144189c33d5265, origin git@github.com:ssoj13/oiio-rs.git. Bob's vfx-ocio dependency points to that repository (../squarebob-rs/Cargo.toml:143). The library is crates/ocio/vfx-ocio within oiio-rs.

The audit read the ancestor cglibs AGENTS and relevant processor, vfx-io/imagebuf, and vfx-warp instructions; no root OIIO AGENTS exists. GitNexus MCP was unavailable. Direct source/callsite inspection was the fallback; no graph risk or freshness result is claimed. No OIIO source edits, builds, tests, commits, or pushes were performed.

All OIIO anchors below are relative to C:/projects/projects.rust.cg/cglibs/oiio-rs.

## Confirmed contracts

- Plain Processor::apply_rgb walks its immutable compiled operations once on the caller's supplied pixels (crates/ocio/vfx-ocio/src/processor/apply.rs:39). apply_rgba has the same contract (:60). There is no processor image history, hidden physical-camera multiplier, or implicit second exposure.
- The configuration compiles the display chain at crates/ocio/vfx-ocio/src/config/factory.rs:378 and :530. display_view_chain (:398) assembles source, looks, view, and display. Destination view colorspace is resolved at :493; :516 appends the destination reference-to-colorspace transform. Processor output therefore has the selected configuration's destination transfer and units; it is not inherently display-linear, scRGB, or RGBA8.
- OcioDisplayParams has no exposure parameter (crates/oiio/vfx-io/src/imagebufalgo/ocio.rs:95). ociodisplay_into accepts immutable source (:390), creates a fresh result (:444), applies the processor once (:449), and copies into destination (:472). The row helper (:706) uses RGB scratch and calls Processor::apply_rgb once (:768). The source remains unchanged.
- ColorConfig compiles selected transforms, context, and direction without adding camera exposure (crates/oiio/vfx-io/src/colorconfig.rs:991, :1079). Colorspace encoding metadata getters (crates/ocio/vfx-ocio/src/colorspace.rs:379; crates/oiio/vfx-io/src/colorconfig.rs:487) classify the color space; they do not themselves transform pixels.
- Explicit ExposureContrast uses EV and 2^EV (crates/ocio/vfx-ocio/src/processor/apply.rs:435, :446). Tests exist at crates/ocio/vfx-ocio/src/processor/tests.rs:11 and crates/ocio/vfx-ocio/tests/ocio_cpu_tables.rs:111; these were not executed in this read-only lane.
- DynamicProcessor applies explicit adjustments after the base processor by default (crates/ocio/vfx-ocio/src/dynamic.rs:429), exposes set_apply_before (:505), and evaluates base/adjustments once (:567). This is distinct from the plain Processor used by Squarebob.
- vfx-warp's display path takes immutable Image, applies explicit EV once before tone mapping, and returns separate RGBA8 (crates/stool/vfx-warp/src/display.rs:223, :225, :714). No source feedback was found.

## Decision and host repair boundary

No OIIO library defect was confirmed for the investigated feedback/double-camera-exposure integration problem. Changing or pushing oiio-rs merely alongside the host repair is unjustified.

The authorized Squarebob repair belongs to its display composition: select immutable raw PT or separate OIDN output; apply physical exposure once before the CPU display processor; decode/scale the selected configuration-defined output into display light; upload separate reusable display scratch; encode the common float canvas transport; present through shared egui-display. GPU and CPU OCIO routes share this source-selection/composition boundary.

```text
raw PT / separate OIDN output (immutable scene radiance)
 -> host physical exposure once
 -> plain OIIO Processor once
 -> configuration-defined destination RGB
 -> explicit host output-transfer decode / white scaling
 -> separate display scratch
 -> shared float canvas transfer
 -> egui-display PresentPass -> negotiated native output
```

## Historical CPU attribution and paired application evidence

The earlier [bridge audit](squarebob_bridge.md) no-feedback conclusion covered the inspected GPU composition path, not CPU display immutability. [Astra post-fix review](astra_pq_review.md) confirms CPU physical exposure after the nonlinear view and a denoised CPU view bypass existed in historical Squarebob HEAD; they were not introduced by PQ integration. The authorized application repair shares `../../squarebob-rs/crates/render-3d/src/lib.rs:1190-1236` across both callers (`src/app/treemap_view.rs:699-708,1193-1205` in Squarebob). Raw source is read without overwrite; separate reusable display scratch is written at `../../squarebob-rs/crates/pt-megakernel/src/compute.rs:5383-5495`. The CPU immutability/order probe and real CPU restart passed; see [final verification](squarebob_pq_verification.md).

## Checklist

- [x] Inspect plain processor, compiled display chain, destination transfer, explicit EV.
- [x] Inspect immutable imagebuf display source and destination ownership.
- [x] Distinguish DynamicProcessor from the caller's plain Processor.
- [x] Inspect vfx-warp independent display path.
- [x] Record existing tests without claiming execution.
- [x] Confirm no justified OIIO edit for this specific integration issue.
- [x] Hand off host repair and runtime verification to Squarebob owners.
