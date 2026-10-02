//! Image buffer abstractions.
//!
//! Mirrors the subset of `_ref/oidn/include/OpenImageDenoise/oidn.h::Format`
//! that the RT/RTLightmap filters accept (`unet_filter.cpp:checkParams`):
//! `Float`, `Half` (1-channel), `Float2`, `Half2` (2-channel), `Float3`,
//! `Half3` (3-channel). The internal pipeline always operates on 3 channels;
//! shorter formats broadcast (1ch → replicate to RGB, 2ch → replicate G into
//! B per `image_accessor.h::get3`), and outputs collapse the same way on
//! write-back.

use crate::error::OidnError;
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
        matches!(
            self,
            PixelFormat::R16f | PixelFormat::Rg16f | PixelFormat::Rgb16f
        )
    }
}

/// Borrowed read-only image. Conversion validates public descriptors before reading.
///
/// Typed `from_*` constructors panic on invalid dimensions or length; use `new` for fallible byte input.
#[derive(Debug, Clone, Copy)]
pub struct Image<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
    pub format: PixelFormat,
}

/// Borrowed mutable image (for outputs). Writes validate descriptors before modifying bytes.
///
/// Typed `from_*` constructors panic on invalid dimensions or length; use `new` for fallible byte input.
#[derive(Debug)]
pub struct ImageMut<'a> {
    pub data: &'a mut [u8],
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
    pub format: PixelFormat,
}

/// Maximum supported image dimension, matching OIDN ImageDesc.
pub const MAX_IMAGE_DIMENSION: usize = 65_536;

/// Validate spatial dimensions and return the checked number of elements.
///
/// Empty images are valid descriptors; execution APIs may require nonempty inputs.
pub fn validate_dimensions(
    width: usize,
    height: usize,
    channels: usize,
) -> Result<usize, OidnError> {
    if channels == 0
        || channels > i32::MAX as usize
        || width > MAX_IMAGE_DIMENSION
        || height > MAX_IMAGE_DIMENSION
    {
        return Err(OidnError::InvalidArgument(
            "invalid image dimensions or channel count",
        ));
    }
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or(OidnError::InvalidArgument(
            "image element count exceeds supported range",
        ))
}

fn validate_layout(
    len: usize,
    width: usize,
    height: usize,
    row_stride: usize,
    format: PixelFormat,
) -> Result<usize, OidnError> {
    validate_dimensions(width, height, format.channels())?;
    let row_bytes = width
        .checked_mul(format.pixel_size())
        .ok_or(OidnError::InvalidArgument("image row size overflow"))?;
    if row_stride < row_bytes {
        return Err(OidnError::InvalidArgument(
            "row stride is smaller than image row size",
        ));
    }
    let extent = if width == 0 || height == 0 {
        0
    } else {
        (height - 1)
            .checked_mul(row_stride)
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or(OidnError::InvalidArgument("image occupied extent overflow"))?
    };
    if len < extent {
        return Err(OidnError::InvalidArgument(
            "image buffer is smaller than occupied extent",
        ));
    }
    Ok(extent)
}

fn contiguous_image<'a>(
    data: &'a [u8],
    width: usize,
    height: usize,
    format: PixelFormat,
) -> Image<'a> {
    let row_stride = width
        .checked_mul(format.pixel_size())
        .expect("image row size overflow");
    let image =
        Image::new(data, width, height, row_stride, format).expect("invalid contiguous image");
    assert_eq!(
        data.len(),
        row_stride
            .checked_mul(height)
            .expect("image byte size overflow"),
        "contiguous image length mismatch"
    );
    image
}

fn contiguous_image_mut<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
    format: PixelFormat,
) -> ImageMut<'a> {
    let row_stride = width
        .checked_mul(format.pixel_size())
        .expect("image row size overflow");
    assert_eq!(
        data.len(),
        row_stride
            .checked_mul(height)
            .expect("image byte size overflow"),
        "contiguous image length mismatch"
    );
    ImageMut::new(data, width, height, row_stride, format).expect("invalid contiguous image")
}
impl<'a> Image<'a> {
    /// 3 × f32 contiguous HWC image.
    pub fn from_rgb_f32(data: &'a [f32], width: usize, height: usize) -> Self {
        contiguous_image(
            bytemuck::cast_slice(data),
            width,
            height,
            PixelFormat::Rgb32f,
        )
    }
    /// 3 × f16 contiguous HWC image.
    pub fn from_rgb_f16(data: &'a [f16], width: usize, height: usize) -> Self {
        contiguous_image(
            bytemuck::cast_slice(data),
            width,
            height,
            PixelFormat::Rgb16f,
        )
    }
    /// 2 × f32 contiguous HWC image.
    pub fn from_rg_f32(data: &'a [f32], width: usize, height: usize) -> Self {
        contiguous_image(
            bytemuck::cast_slice(data),
            width,
            height,
            PixelFormat::Rg32f,
        )
    }
    /// 2 × f16 contiguous HWC image.
    pub fn from_rg_f16(data: &'a [f16], width: usize, height: usize) -> Self {
        contiguous_image(
            bytemuck::cast_slice(data),
            width,
            height,
            PixelFormat::Rg16f,
        )
    }
    /// 1 × f32 contiguous luminance image.
    pub fn from_r_f32(data: &'a [f32], width: usize, height: usize) -> Self {
        contiguous_image(bytemuck::cast_slice(data), width, height, PixelFormat::R32f)
    }
    /// 1 × f16 contiguous luminance image.
    pub fn from_r_f16(data: &'a [f16], width: usize, height: usize) -> Self {
        contiguous_image(bytemuck::cast_slice(data), width, height, PixelFormat::R16f)
    }

    /// Create a validated byte image. Values use native byte order; alignment is unrestricted.
    pub fn new(
        data: &'a [u8],
        width: usize,
        height: usize,
        row_stride: usize,
        format: PixelFormat,
    ) -> Result<Self, OidnError> {
        let image = Self {
            data,
            width,
            height,
            row_stride,
            format,
        };
        image.validate()?;
        Ok(image)
    }

    /// Validate the descriptor and return its occupied byte extent, excluding final row padding.
    pub fn validate(&self) -> Result<usize, OidnError> {
        validate_layout(
            self.data.len(),
            self.width,
            self.height,
            self.row_stride,
            self.format,
        )
    }

    /// Decode RGB in HWC order. One channel broadcasts; two channels replicate green into blue.
    pub fn to_rgb_f32(&self) -> Result<Vec<f32>, OidnError> {
        self.validate()?;
        let len = validate_dimensions(self.width, self.height, 3)?;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| OidnError::OutOfMemory("RGB image decode"))?;
        out.resize(len, 0.0);
        let channels = self.format.channels();
        let element_size = self.format.element_size();
        for y in 0..self.height {
            for x in 0..self.width {
                let offset = y * self.row_stride + x * self.format.pixel_size();
                let rgb = &mut out[(y * self.width + x) * 3..][..3];
                for (c, value) in rgb.iter_mut().enumerate() {
                    let c = c.min(channels - 1);
                    let bytes = &self.data[offset + c * element_size..][..element_size];
                    *value = if self.format.is_f16() {
                        f16::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]])).to_f32()
                    } else {
                        f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                    };
                }
            }
        }
        Ok(out)
    }
}
impl<'a> ImageMut<'a> {
    pub fn from_rgb_f32(data: &'a mut [f32], width: usize, height: usize) -> Self {
        contiguous_image_mut(
            bytemuck::cast_slice_mut(data),
            width,
            height,
            PixelFormat::Rgb32f,
        )
    }
    pub fn from_rgb_f16(data: &'a mut [f16], width: usize, height: usize) -> Self {
        contiguous_image_mut(
            bytemuck::cast_slice_mut(data),
            width,
            height,
            PixelFormat::Rgb16f,
        )
    }
    pub fn from_rg_f32(data: &'a mut [f32], width: usize, height: usize) -> Self {
        contiguous_image_mut(
            bytemuck::cast_slice_mut(data),
            width,
            height,
            PixelFormat::Rg32f,
        )
    }
    pub fn from_rg_f16(data: &'a mut [f16], width: usize, height: usize) -> Self {
        contiguous_image_mut(
            bytemuck::cast_slice_mut(data),
            width,
            height,
            PixelFormat::Rg16f,
        )
    }
    pub fn from_r_f32(data: &'a mut [f32], width: usize, height: usize) -> Self {
        contiguous_image_mut(
            bytemuck::cast_slice_mut(data),
            width,
            height,
            PixelFormat::R32f,
        )
    }
    pub fn from_r_f16(data: &'a mut [f16], width: usize, height: usize) -> Self {
        contiguous_image_mut(
            bytemuck::cast_slice_mut(data),
            width,
            height,
            PixelFormat::R16f,
        )
    }

    /// Create a validated output byte image. Values use native byte order; alignment is unrestricted.
    pub fn new(
        data: &'a mut [u8],
        width: usize,
        height: usize,
        row_stride: usize,
        format: PixelFormat,
    ) -> Result<Self, OidnError> {
        let image = Self {
            data,
            width,
            height,
            row_stride,
            format,
        };
        image.validate()?;
        Ok(image)
    }

    /// Validate the output descriptor before any write.
    pub fn validate(&self) -> Result<usize, OidnError> {
        validate_layout(
            self.data.len(),
            self.width,
            self.height,
            self.row_stride,
            self.format,
        )
    }

    /// Write RGB HWC values. The generic accessor retains red for one channel and drops blue for two.
    ///
    /// Filter-specific scalar reduction must be applied by output processing before this accessor.
    pub fn write_rgb_f32(&mut self, src_rgb: &[f32]) -> Result<(), OidnError> {
        self.validate()?;
        if src_rgb.len() != validate_dimensions(self.width, self.height, 3)? {
            return Err(OidnError::InvalidArgument(
                "RGB source length does not match output dimensions",
            ));
        }
        let element_size = self.format.element_size();
        for y in 0..self.height {
            for x in 0..self.width {
                let offset = y * self.row_stride + x * self.format.pixel_size();
                for c in 0..self.format.channels() {
                    let value = src_rgb[(y * self.width + x) * 3 + c];
                    let bytes = &mut self.data[offset + c * element_size..][..element_size];
                    if self.format.is_f16() {
                        bytes.copy_from_slice(&f16::from_f32(value).to_bits().to_ne_bytes());
                    } else {
                        bytes.copy_from_slice(&value.to_ne_bytes());
                    }
                }
            }
        }
        Ok(())
    }
}

/// Shared owned host storage for filter inputs and outputs.
pub(crate) struct OwnedImage {
    pub(crate) data: Vec<u8>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) row_stride: usize,
    pub(crate) format: PixelFormat,
}

impl OwnedImage {
    pub(crate) fn from(image: &Image<'_>) -> Result<Self, OidnError> {
        let extent = image.validate()?;
        let mut data = Vec::new();
        data.try_reserve_exact(extent)
            .map_err(|_| OidnError::OutOfMemory("host image copy"))?;
        data.extend_from_slice(&image.data[..extent]);
        Ok(Self {
            data,
            width: image.width,
            height: image.height,
            row_stride: image.row_stride,
            format: image.format,
        })
    }

    pub(crate) fn empty(
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<Self, OidnError> {
        validate_dimensions(width, height, format.channels())?;
        let row_stride = width
            .checked_mul(format.pixel_size())
            .ok_or(OidnError::InvalidArgument("image row size overflow"))?;
        let len = row_stride
            .checked_mul(height)
            .ok_or(OidnError::InvalidArgument("image byte size overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|_| OidnError::OutOfMemory("host output image allocation"))?;
        data.resize(len, 0);
        Ok(Self {
            data,
            width,
            height,
            row_stride,
            format,
        })
    }

    pub(crate) fn view(&self) -> Image<'_> {
        Image {
            data: &self.data,
            width: self.width,
            height: self.height,
            row_stride: self.row_stride,
            format: self.format,
        }
    }

    pub(crate) fn view_mut(&mut self) -> ImageMut<'_> {
        ImageMut {
            data: &mut self.data,
            width: self.width,
            height: self.height,
            row_stride: self.row_stride,
            format: self.format,
        }
    }
}

#[cfg(test)]
mod owned_tests {
    use super::*;

    #[test]
    fn copy_owns_only_occupied_extent_and_validates_before_allocating() {
        let values = [1.0_f32, 2.0, 3.0, 99.0];
        let source =
            Image::new(bytemuck::cast_slice(&values), 1, 1, 12, PixelFormat::Rgb32f).unwrap();
        let owned = OwnedImage::from(&source).unwrap();
        assert_eq!(owned.data.len(), 12);
        assert_eq!(owned.view().to_rgb_f32().unwrap(), [1.0, 2.0, 3.0]);
        let invalid = Image {
            data: &[],
            width: 1,
            height: 1,
            row_stride: 12,
            format: PixelFormat::Rgb32f,
        };
        assert!(OwnedImage::from(&invalid).is_err());
    }

    #[test]
    fn owned_output_is_checked_and_writable() {
        assert!(OwnedImage::empty(usize::MAX, 1, PixelFormat::Rgb32f).is_err());
        let mut output = OwnedImage::empty(1, 1, PixelFormat::Rgb16f).unwrap();
        output.view_mut().write_rgb_f32(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(output.view().to_rgb_f32().unwrap(), [1.0, 2.0, 3.0]);
    }
}
