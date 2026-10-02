//! Tile planner — port of the geometric portion of
//! `_ref/oidn/core/unet_filter.cpp::init` (lines 254-335).
//!
//! Produces a list of input/output tile rectangles that, when independently
//! denoised and stitched, fully cover the source image with the correct
//! receptive-field overlap.

/// Network geometry constants shared with the validated model descriptor.
pub use oidn_model::{MIN_TILE_ALIGNMENT, RECEPTIVE_FIELD_BASE, RECEPTIVE_FIELD_LARGE};
/// Default upper bound on tile pixel count (`_ref/oidn/core/unet_filter.h:34`).
pub const DEFAULT_MAX_TILE_SIZE: i32 = 2160 * 2160;

use crate::{
    error::OidnError,
    image::{MAX_IMAGE_DIMENSION, validate_dimensions},
};

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
fn round_up(x: i64, align: i64) -> i64 {
    debug_assert!(align > 0);
    let r = x % align;
    if r == 0 { x } else { x + (align - r) }
}

/// Round up to a multiple of `align`, but also keep the result aligned so that
/// `(x - pad) % align == 0` — corresponds to the 3-arg `round_up` helper used
/// in `unet_filter.cpp:289-298`.
#[inline]
fn round_up_pad(x: i64, align: i64, pad: i64) -> i64 {
    let r = (x - pad) % align;
    if r <= 0 { x + (-r) } else { x + (align - r) }
}

#[inline]
fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// Plan the tiling for an image of size `(W, H)` and a given receptive field.
///
/// Mirrors the loop structure of `UNetFilter::init` (unet_filter.cpp:265-326)
/// Returns an error for invalid or oversized geometry and invalid planning parameters.
/// The pixel budget is best effort when the minimum tile size prevents further division.
/// This is implemented
/// without the memory-budget probing — we either fit the whole image in one
/// tile (when ≤ `DEFAULT_MAX_TILE_SIZE`) or shrink dimensions until it does.
pub fn plan(
    width: i32,
    height: i32,
    receptive_field: i32,
    tile_alignment: i32,
    max_tile_pixels: i32,
) -> Result<TilePlan, OidnError> {
    if width <= 0 || height <= 0 {
        return Err(OidnError::InvalidArgument(
            "tile dimensions must be positive",
        ));
    }
    validate_dimensions(width as usize, height as usize, 3)?;
    if receptive_field <= 0
        || receptive_field > MAX_IMAGE_DIMENSION as i32
        || tile_alignment < MIN_TILE_ALIGNMENT
        || tile_alignment % MIN_TILE_ALIGNMENT != 0
        || tile_alignment > MAX_IMAGE_DIMENSION as i32
        || max_tile_pixels <= 0
    {
        return Err(OidnError::InvalidArgument(
            "invalid tile receptive field, alignment, or pixel budget",
        ));
    }
    // Wider arithmetic also covers public parameters and intermediate overlap products.
    let (width, height, receptive_field, tile_alignment, max_tile_pixels) = (
        i64::from(width),
        i64::from(height),
        i64::from(receptive_field),
        i64::from(tile_alignment),
        i64::from(max_tile_pixels),
    );
    let tile_overlap = round_up(receptive_field / 2, tile_alignment);

    let mut tile_h = round_up(height, i64::from(MIN_TILE_ALIGNMENT));
    let mut tile_w = round_up(width, i64::from(MIN_TILE_ALIGNMENT));
    let pad_h = tile_h % tile_alignment;
    let pad_w = tile_w % tile_alignment;
    let mut tile_count_h: i64 = 1;
    let mut tile_count_w: i64 = 1;

    let min_tile_dim = std::cmp::max(4 * tile_overlap, 768);
    let min_tile_h = round_up_pad(min_tile_dim, tile_alignment, pad_h);
    let min_tile_w = round_up_pad(min_tile_dim, tile_alignment, pad_w);

    while (tile_h * tile_w) > max_tile_pixels {
        if tile_h >= min_tile_h + tile_alignment && tile_h > tile_w {
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
        } else if tile_w >= min_tile_w + tile_alignment {
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
        let align_offset_h = tile_h - round_up(tile_h1, i64::from(MIN_TILE_ALIGNMENT));

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
            let align_offset_w = tile_w - round_up(tile_w1, i64::from(MIN_TILE_ALIGNMENT));

            jobs.push(TileJob {
                input: Rect {
                    x: x as i32,
                    y: y as i32,
                    w: (tile_w1) as i32,
                    h: (tile_h1) as i32,
                },
                output_dst: Rect {
                    x: (x + overlap_left) as i32,
                    y: (y + overlap_top) as i32,
                    w: (tile_w2) as i32,
                    h: (tile_h2) as i32,
                },
                output_src_in_tile: Rect {
                    x: (align_offset_w + overlap_left) as i32,
                    y: (align_offset_h + overlap_top) as i32,
                    w: (tile_w2) as i32,
                    h: (tile_h2) as i32,
                },
                align_offset_x: align_offset_w as i32,
                align_offset_y: align_offset_h as i32,
            });
        }
    }

    let plan = TilePlan {
        tile_w: tile_w as i32,
        tile_h: tile_h as i32,
        overlap: tile_overlap as i32,
        pad_w: pad_w as i32,
        pad_h: pad_h as i32,
        jobs,
    };
    plan.validate(width as usize, height as usize)?;
    Ok(plan)
}

/// Sum output rectangle areas, rejecting invalid dimensions and overflow.
/// Exact nonoverlapping coverage is checked separately by [`TilePlan::validate`].
pub fn total_output_pixels(plan: &TilePlan) -> Result<i64, OidnError> {
    plan.jobs.iter().try_fold(0_i64, |total, job| {
        let Rect { w, h, .. } = job.output_dst;
        if w <= 0 || h <= 0 {
            return Err(OidnError::InvalidArgument(
                "invalid output rectangle dimensions",
            ));
        }
        total
            .checked_add(i64::from(w) * i64::from(h))
            .ok_or(OidnError::InvalidArgument("output pixel count overflow"))
    })
}

impl TilePlan {
    /// Validate public plan geometry and exact, nonoverlapping destination coverage.
    ///
    /// Uses a sweep over destination rectangle edges, without allocating an image-sized mask.
    pub fn validate(&self, width: usize, height: usize) -> Result<(), OidnError> {
        validate_dimensions(width, height, 3)?;
        if width == 0
            || height == 0
            || self.tile_w <= 0
            || self.tile_h <= 0
            || self.tile_w % MIN_TILE_ALIGNMENT != 0
            || self.tile_h % MIN_TILE_ALIGNMENT != 0
            || self.overlap < 0
            || self.pad_w < 0
            || self.pad_h < 0
            // For a single small tile, native pad = tile_size % alignment
            // can equal tile_size when alignment exceeds that dimension.
            || self.pad_w > self.tile_w
            || self.pad_h > self.tile_h
            || self.jobs.is_empty()
        {
            return Err(OidnError::InvalidArgument(
                "invalid tile plan dimensions or metadata",
            ));
        }
        validate_dimensions(self.tile_w as usize, self.tile_h as usize, 3)?;
        let inside = |rect: Rect, w: i64, h: i64| {
            rect.x >= 0
                && rect.y >= 0
                && rect.w > 0
                && rect.h > 0
                && i64::from(rect.x) + i64::from(rect.w) <= w
                && i64::from(rect.y) + i64::from(rect.h) <= h
        };
        let mut events = Vec::with_capacity(
            self.jobs
                .len()
                .checked_mul(2)
                .ok_or(OidnError::InvalidArgument("tile event count overflow"))?,
        );
        for (index, job) in self.jobs.iter().enumerate() {
            let source = job.output_src_in_tile;
            let destination = job.output_dst;
            if !inside(job.input, width as i64, height as i64)
                || !inside(destination, width as i64, height as i64)
                || !inside(source, i64::from(self.tile_w), i64::from(self.tile_h))
                || job.align_offset_x < 0
                || job.align_offset_y < 0
                || i64::from(job.align_offset_x) + i64::from(job.input.w) > i64::from(self.tile_w)
                || i64::from(job.align_offset_y) + i64::from(job.input.h) > i64::from(self.tile_h)
                || (source.w, source.h) != (destination.w, destination.h)
                || source.x < job.align_offset_x
                || source.y < job.align_offset_y
                || i64::from(source.x) + i64::from(source.w)
                    > i64::from(job.align_offset_x) + i64::from(job.input.w)
                || i64::from(source.y) + i64::from(source.h)
                    > i64::from(job.align_offset_y) + i64::from(job.input.h)
                || i64::from(source.x)
                    != i64::from(job.align_offset_x) + i64::from(destination.x)
                        - i64::from(job.input.x)
                || i64::from(source.y)
                    != i64::from(job.align_offset_y) + i64::from(destination.y)
                        - i64::from(job.input.y)
            {
                return Err(OidnError::InvalidArgument(
                    "tile rectangles or alignment offsets are inconsistent",
                ));
            }
            events.push((i64::from(destination.y), true, index));
            events.push((
                i64::from(destination.y) + i64::from(destination.h),
                false,
                index,
            ));
        }
        events.sort_unstable(); // End events precede start events at the same height.
        let mut active = std::collections::BTreeSet::<(i64, i64, usize)>::new();
        let (mut previous_y, mut covered_width) = (0_i64, 0_i64);
        for (y, starts, index) in events {
            if y > previous_y && covered_width != width as i64 {
                return Err(OidnError::InvalidArgument(
                    "tile destinations leave uncovered pixels",
                ));
            }
            previous_y = y;
            let rect = self.jobs[index].output_dst;
            let interval = (
                i64::from(rect.x),
                i64::from(rect.x) + i64::from(rect.w),
                index,
            );
            if starts {
                if active
                    .range(..interval)
                    .next_back()
                    .is_some_and(|other| other.1 > interval.0)
                    || active
                        .range(interval..)
                        .next()
                        .is_some_and(|other| other.0 < interval.1)
                {
                    return Err(OidnError::InvalidArgument("tile destinations overlap"));
                }
                active.insert(interval);
                covered_width += i64::from(rect.w);
            } else {
                active.remove(&interval);
                covered_width -= i64::from(rect.w);
            }
        }
        if previous_y != height as i64 || covered_width != 0 || !active.is_empty() {
            return Err(OidnError::InvalidArgument(
                "tile destinations do not cover image height",
            ));
        }
        Ok(())
    }
}
