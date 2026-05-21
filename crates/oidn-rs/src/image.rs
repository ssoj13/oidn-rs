//! Image buffer abstractions.
//!
//! Mirrors the subset of `_ref/oidn/include/OpenImageDenoise/oidn.h::Format`
//! that the RT/RTLightmap filters accept (`unet_filter.cpp:checkParams`):
//! `Float`, `Half` (1-channel), `Float2`, `Half2` (2-channel), `Float3`,
//! `Half3` (3-channel). The internal pipeline always operates on 3 channels;
//! shorter formats broadcast (1ch → replicate to RGB, 2ch → replicate G into
//! B per `image_accessor.h::get3`), and outputs collapse the same way on
//! write-back.

use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 1 × f32 (luminance / mask).
    R32f,
    /// 1 × f16.
    R16f,
    /// 2 × f32 (e.g. UV).
    Rg32f,
    /// 2 × f16.
    Rg16f,
    /// 3 × f32 per pixel, HWC layout.
    Rgb32f,
    /// 3 × f16 per pixel, HWC layout.
    Rgb16f,
}

impl PixelFormat {
    pub const fn channels(self) -> usize {
        match self {
            PixelFormat::R32f | PixelFormat::R16f => 1,
            PixelFormat::Rg32f | PixelFormat::Rg16f => 2,
            PixelFormat::Rgb32f | PixelFormat::Rgb16f => 3,
        }
    }

    /// Bytes per element of the underlying dtype.
    pub const fn element_size(self) -> usize {
        match self {
            PixelFormat::R32f | PixelFormat::Rg32f | PixelFormat::Rgb32f => 4,
            PixelFormat::R16f | PixelFormat::Rg16f | PixelFormat::Rgb16f => 2,
        }
    }

    pub const fn pixel_size(self) -> usize {
        self.channels() * self.element_size()
    }

    pub const fn is_f16(self) -> bool {
        matches!(self, PixelFormat::R16f | PixelFormat::Rg16f | PixelFormat::Rgb16f)
    }
}

/// Borrowed read-only image.
#[derive(Debug, Clone, Copy)]
pub struct Image<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
    pub format: PixelFormat,
}

/// Borrowed mutable image (for outputs).
#[derive(Debug)]
pub struct ImageMut<'a> {
    pub data: &'a mut [u8],
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
    pub format: PixelFormat,
}

fn contiguous_image<'a>(data: &'a [u8], width: usize, height: usize, format: PixelFormat) -> Image<'a> {
    debug_assert_eq!(data.len(), width * height * format.pixel_size());
    Image { data, width, height, row_stride: width * format.pixel_size(), format }
}

fn contiguous_image_mut<'a>(data: &'a mut [u8], width: usize, height: usize, format: PixelFormat) -> ImageMut<'a> {
    debug_assert_eq!(data.len(), width * height * format.pixel_size());
    ImageMut { data, width, height, row_stride: width * format.pixel_size(), format }
}

impl<'a> Image<'a> {
    /// 3 × f32 contiguous HWC image.
    pub fn from_rgb_f32(data: &'a [f32], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height * 3);
        contiguous_image(bytemuck::cast_slice(data), width, height, PixelFormat::Rgb32f)
    }
    /// 3 × f16 contiguous HWC image.
    pub fn from_rgb_f16(data: &'a [f16], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height * 3);
        contiguous_image(bytemuck::cast_slice(data), width, height, PixelFormat::Rgb16f)
    }
    /// 2 × f32 contiguous HWC image.
    pub fn from_rg_f32(data: &'a [f32], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height * 2);
        contiguous_image(bytemuck::cast_slice(data), width, height, PixelFormat::Rg32f)
    }
    /// 2 × f16 contiguous HWC image.
    pub fn from_rg_f16(data: &'a [f16], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height * 2);
        contiguous_image(bytemuck::cast_slice(data), width, height, PixelFormat::Rg16f)
    }
    /// 1 × f32 contiguous luminance image.
    pub fn from_r_f32(data: &'a [f32], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height);
        contiguous_image(bytemuck::cast_slice(data), width, height, PixelFormat::R32f)
    }
    /// 1 × f16 contiguous luminance image.
    pub fn from_r_f16(data: &'a [f16], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height);
        contiguous_image(bytemuck::cast_slice(data), width, height, PixelFormat::R16f)
    }

    /// Decode to a `Vec<f32>` in 3-channel HWC order, broadcasting 1ch to all
    /// three channels and replicating green into blue for 2ch
    /// (matches `_ref/oidn/core/image_accessor.h::get3` for `C==2`).
    pub fn to_rgb_f32(&self) -> Vec<f32> {
        let n = self.width * self.height * 3;
        let mut out = vec![0.0f32; n];
        let ch = self.format.channels();
        for y in 0..self.height {
            let row_bytes = &self.data[y * self.row_stride..y * self.row_stride + self.width * self.format.pixel_size()];
            // Read into a temporary f32 array of `ch` values per pixel.
            let mut src = vec![0.0f32; self.width * ch];
            if self.format.is_f16() {
                let half_row: &[f16] = bytemuck::cast_slice(row_bytes);
                for (d, s) in src.iter_mut().zip(half_row.iter()) {
                    *d = s.to_f32();
                }
            } else {
                let f32_row: &[f32] = bytemuck::cast_slice(row_bytes);
                src.copy_from_slice(f32_row);
            }
            // Broadcast into the 3-channel destination.
            for x in 0..self.width {
                let dst_off = (y * self.width + x) * 3;
                let src_off = x * ch;
                match ch {
                    1 => {
                        let v = src[src_off];
                        out[dst_off]     = v;
                        out[dst_off + 1] = v;
                        out[dst_off + 2] = v;
                    }
                    2 => {
                        out[dst_off]     = src[src_off];
                        out[dst_off + 1] = src[src_off + 1];
                        out[dst_off + 2] = src[src_off + 1];
                    }
                    3 => {
                        out[dst_off]     = src[src_off];
                        out[dst_off + 1] = src[src_off + 1];
                        out[dst_off + 2] = src[src_off + 2];
                    }
                    _ => unreachable!(),
                }
            }
        }
        out
    }
}

impl<'a> ImageMut<'a> {
    pub fn from_rgb_f32(data: &'a mut [f32], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height * 3);
        contiguous_image_mut(bytemuck::cast_slice_mut(data), width, height, PixelFormat::Rgb32f)
    }
    pub fn from_rgb_f16(data: &'a mut [f16], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height * 3);
        contiguous_image_mut(bytemuck::cast_slice_mut(data), width, height, PixelFormat::Rgb16f)
    }
    pub fn from_rg_f32(data: &'a mut [f32], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height * 2);
        contiguous_image_mut(bytemuck::cast_slice_mut(data), width, height, PixelFormat::Rg32f)
    }
    pub fn from_rg_f16(data: &'a mut [f16], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height * 2);
        contiguous_image_mut(bytemuck::cast_slice_mut(data), width, height, PixelFormat::Rg16f)
    }
    pub fn from_r_f32(data: &'a mut [f32], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height);
        contiguous_image_mut(bytemuck::cast_slice_mut(data), width, height, PixelFormat::R32f)
    }
    pub fn from_r_f16(data: &'a mut [f16], width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width * height);
        contiguous_image_mut(bytemuck::cast_slice_mut(data), width, height, PixelFormat::R16f)
    }

    /// Write a 3-channel HWC f32 buffer into this image, collapsing into the
    /// destination format. For 1ch: take the red channel only (it matches
    /// the broadcast input rule). For 2ch: drop blue.
    pub fn write_rgb_f32(&mut self, src_rgb: &[f32]) {
        debug_assert_eq!(src_rgb.len(), self.width * self.height * 3);
        let ch = self.format.channels();
        for y in 0..self.height {
            let dst_row = &mut self.data[y * self.row_stride..y * self.row_stride + self.width * self.format.pixel_size()];
            let src_row = &src_rgb[y * self.width * 3..(y + 1) * self.width * 3];

            // Build the per-row destination as f32, then cast if format is f16.
            let mut dst_f32 = vec![0.0f32; self.width * ch];
            for x in 0..self.width {
                let s = x * 3;
                let d = x * ch;
                match ch {
                    1 => { dst_f32[d] = src_row[s]; }
                    2 => {
                        dst_f32[d]     = src_row[s];
                        dst_f32[d + 1] = src_row[s + 1];
                    }
                    3 => {
                        dst_f32[d]     = src_row[s];
                        dst_f32[d + 1] = src_row[s + 1];
                        dst_f32[d + 2] = src_row[s + 2];
                    }
                    _ => unreachable!(),
                }
            }
            if self.format.is_f16() {
                let dst_half: &mut [f16] = bytemuck::cast_slice_mut(dst_row);
                for (d, v) in dst_half.iter_mut().zip(dst_f32.iter()) {
                    *d = f16::from_f32(*v);
                }
            } else {
                let dst_f: &mut [f32] = bytemuck::cast_slice_mut(dst_row);
                dst_f.copy_from_slice(&dst_f32);
            }
        }
    }
}
