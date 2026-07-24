//! Tile planner — port of the geometric portion of
//! `_ref/oidn/core/unet_filter.cpp::init` (lines 254-335).
//!
//! Produces a list of input/output tile rectangles that, when independently
//! denoised and stitched, fully cover the source image with the correct
//! receptive-field overlap.

/// Receptive field of the base UNet (`_ref/oidn/core/unet_filter.h:30`).
pub const RECEPTIVE_FIELD_BASE: i32 = 174;
/// Receptive field of UNetLarge (`_ref/oidn/core/unet_filter.h:31`).
pub const RECEPTIVE_FIELD_LARGE: i32 = 202;
/// Spatial alignment required by the network (`_ref/oidn/core/unet_filter.h:32`).
pub const MIN_TILE_ALIGNMENT: i32 = 16;
/// Default upper bound on tile pixel count (`_ref/oidn/core/unet_filter.h:34`).
pub const DEFAULT_MAX_TILE_SIZE: i32 = 2160 * 2160;

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// One scheduled tile: where to read input, where to write output, and where
/// the output region is within the network's tile buffer.
#[derive(Debug, Clone, Copy)]
pub struct TileJob {
    /// Top-left of the input region in source-image coordinates. Width/height
    /// equal the network input size (including overlap padding).
    pub input: Rect,
    /// Top-left in source-image coordinates of where the *output* region
    /// belongs.
    pub output_dst: Rect,
    /// Position inside the network's tile buffer of the region that should
    /// be copied to `output_dst`. `(x, y)` are the offsets into the network
    /// output, `(w, h)` is the output region size.
    pub output_src_in_tile: Rect,
    /// Offset (in tile-buffer coordinates) where the source image content was
    /// placed. Used by InputProcess to know where padding starts.
    pub align_offset_x: i32,
    pub align_offset_y: i32,
}

#[derive(Debug, Clone)]
pub struct TilePlan {
    pub tile_w: i32,
    pub tile_h: i32,
    pub overlap: i32,
    pub pad_w: i32,
    pub pad_h: i32,
    pub jobs: Vec<TileJob>,
}

#[inline]
fn round_up(x: i32, align: i32) -> i32 {
    debug_assert!(align > 0);
    let r = x % align;
    if r == 0 { x } else { x + (align - r) }
}

/// Round up to a multiple of `align`, but also keep the result aligned so that
/// `(x - pad) % align == 0` — corresponds to the 3-arg `round_up` helper used
/// in `unet_filter.cpp:289-298`.
#[inline]
fn round_up_pad(x: i32, align: i32, pad: i32) -> i32 {
    let r = (x - pad) % align;
    if r <= 0 { x + (-r) } else { x + (align - r) }
}

#[inline]
fn ceil_div(a: i32, b: i32) -> i32 {
    (a + b - 1) / b
}

/// Plan the tiling for an image of size `(W, H)` and a given receptive field.
///
/// Mirrors the loop structure of `UNetFilter::init` (unet_filter.cpp:265-326)
/// without the memory-budget probing — we either fit the whole image in one
/// tile (when ≤ `DEFAULT_MAX_TILE_SIZE`) or shrink dimensions until it does.
pub fn plan(
    width: i32,
    height: i32,
    receptive_field: i32,
    tile_alignment: i32,
    max_tile_pixels: i32,
) -> TilePlan {
    let tile_overlap = round_up(receptive_field / 2, tile_alignment);

    let mut tile_h = round_up(height, MIN_TILE_ALIGNMENT);
    let mut tile_w = round_up(width, MIN_TILE_ALIGNMENT);
    let pad_h = tile_h % tile_alignment;
    let pad_w = tile_w % tile_alignment;
    let mut tile_count_h: i32 = 1;
    let mut tile_count_w: i32 = 1;

    let min_tile_dim = std::cmp::max(4 * tile_overlap, 768);
    let min_tile_h = round_up_pad(min_tile_dim, tile_alignment, pad_h);
    let min_tile_w = round_up_pad(min_tile_dim, tile_alignment, pad_w);

    while (tile_h * tile_w) > max_tile_pixels {
        if tile_h > min_tile_h && tile_h > tile_w {
            let new_h = ceil_div(
                height + (2 * tile_overlap + pad_h) * tile_count_h,
                tile_count_h + 1,
            );
            tile_h = new_h.clamp(min_tile_h, tile_h - tile_alignment);
            tile_h = round_up_pad(tile_h, tile_alignment, pad_h);
            tile_count_h = std::cmp::max(
                ceil_div(
                    height - (2 * tile_overlap + pad_h),
                    tile_h - (2 * tile_overlap + pad_h),
                ),
                1,
            );
        } else if tile_w > min_tile_w {
            let new_w = ceil_div(
                width + (2 * tile_overlap + pad_w) * tile_count_w,
                tile_count_w + 1,
            );
            tile_w = new_w.clamp(min_tile_w, tile_w - tile_alignment);
            tile_w = round_up_pad(tile_w, tile_alignment, pad_w);
            tile_count_w = std::cmp::max(
                ceil_div(
                    width - (2 * tile_overlap + pad_w),
                    tile_w - (2 * tile_overlap + pad_w),
                ),
                1,
            );
        } else {
            // Cannot divide further — accept current size.
            break;
        }
    }

    // Generate jobs — direct port of `UNetFilter::execute` tile loop (lines 199-241).
    let mut jobs = Vec::with_capacity((tile_count_h * tile_count_w) as usize);
    for i in 0..tile_count_h {
        let y = i * (tile_h - (2 * tile_overlap + pad_h));
        let overlap_top = if i > 0 { tile_overlap } else { 0 };
        let overlap_bottom = if i < tile_count_h - 1 {
            tile_overlap + pad_h
        } else {
            0
        };
        let tile_h1 = std::cmp::min(height - y, tile_h);
        let tile_h2 = tile_h1 - overlap_top - overlap_bottom;
        let align_offset_h = tile_h - round_up(tile_h1, MIN_TILE_ALIGNMENT);

        for j in 0..tile_count_w {
            let x = j * (tile_w - (2 * tile_overlap + pad_w));
            let overlap_left = if j > 0 { tile_overlap } else { 0 };
            let overlap_right = if j < tile_count_w - 1 {
                tile_overlap + pad_w
            } else {
                0
            };
            let tile_w1 = std::cmp::min(width - x, tile_w);
            let tile_w2 = tile_w1 - overlap_left - overlap_right;
            let align_offset_w = tile_w - round_up(tile_w1, MIN_TILE_ALIGNMENT);

            jobs.push(TileJob {
                input: Rect {
                    x,
                    y,
                    w: tile_w1,
                    h: tile_h1,
                },
                output_dst: Rect {
                    x: x + overlap_left,
                    y: y + overlap_top,
                    w: tile_w2,
                    h: tile_h2,
                },
                output_src_in_tile: Rect {
                    x: align_offset_w + overlap_left,
                    y: align_offset_h + overlap_top,
                    w: tile_w2,
                    h: tile_h2,
                },
                align_offset_x: align_offset_w,
                align_offset_y: align_offset_h,
            });
        }
    }

    TilePlan {
        tile_w,
        tile_h,
        overlap: tile_overlap,
        pad_w,
        pad_h,
        jobs,
    }
}

/// Convenience: total pixel count covered by all output rectangles.
pub fn total_output_pixels(plan: &TilePlan) -> i64 {
    plan.jobs
        .iter()
        .map(|j| (j.output_dst.w as i64) * (j.output_dst.h as i64))
        .sum()
}
