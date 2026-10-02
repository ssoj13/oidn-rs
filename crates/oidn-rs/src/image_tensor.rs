//! Checked host layout adapters for NCHW tensors and HWC image values.
//! Tensor-native filtering avoids these host conversions; image adapters use them for upload/readback.

use crate::{error::OidnError, image::validate_dimensions};
use burn::tensor::{Device, Tensor, TensorData};

/// Convert a flat NCHW `f32` slice into HWC layout.
///
/// Length contract: `chw.len() == channels * height * width`. Returns a
/// freshly allocated `Vec<f32>` of the same length in HWC order
/// `[(y, x, c)]`.
pub fn chw_to_hwc(
    chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, OidnError> {
    let len = validate_dimensions(width, height, channels)?;
    if chw.len() != len {
        return Err(OidnError::InvalidArgument(
            "CHW buffer length does not match dimensions",
        ));
    }
    let mut hwc = vec![0.0f32; len];
    if len == 0 {
        return Ok(hwc);
    }
    let stride_c = height * width;
    for c in 0..channels {
        let plane = &chw[c * stride_c..(c + 1) * stride_c];
        for y in 0..height {
            let src_row = &plane[y * width..(y + 1) * width];
            for x in 0..width {
                hwc[(y * width + x) * channels + c] = src_row[x];
            }
        }
    }
    Ok(hwc)
}

/// Convert a flat HWC `f32` slice into NCHW layout.
///
/// Inverse of [`chw_to_hwc`]; same length contract.
pub fn hwc_to_chw(
    hwc: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, OidnError> {
    let len = validate_dimensions(width, height, channels)?;
    if hwc.len() != len {
        return Err(OidnError::InvalidArgument(
            "HWC buffer length does not match dimensions",
        ));
    }
    let mut chw = vec![0.0f32; len];
    if len == 0 {
        return Ok(chw);
    }
    let stride_c = height * width;
    for y in 0..height {
        for x in 0..width {
            let src_off = (y * width + x) * channels;
            for c in 0..channels {
                chw[c * stride_c + y * width + x] = hwc[src_off + c];
            }
        }
    }
    Ok(chw)
}

/// Pull a `[1, C, H, W]` Burn tensor onto the host as a `Vec<f32>` in CHW
/// order. Returns the data plus the original `[N, C, H, W]` dims so the
/// caller doesn't have to query them separately after the move.
pub fn tensor_to_chw_vec(t: Tensor<4>) -> Result<(Vec<f32>, [usize; 4]), OidnError> {
    let dims = t.dims();
    if dims[0] != 1 {
        return Err(OidnError::InvalidArgument("tensor batch size must be one"));
    }
    validate_dimensions(dims[3], dims[2], dims[1])?;
    let v = t
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| OidnError::InvalidArgument("tensor data cannot be converted to f32"))?;
    if v.len() != validate_dimensions(dims[3], dims[2], dims[1])? {
        return Err(OidnError::Inconsistent("tensor data length"));
    }
    Ok((v, dims))
}

/// Build a `[1, C, H, W]` Burn tensor from a flat CHW `f32` buffer.
///
/// Used by [`RtFilter::take_output_tensor`](crate::filters::rt::RtFilter::take_output_tensor)
/// and the host image adapter. The data is uploaded to
/// `device` via Burn's [`TensorData`] machinery; on a wgpu device this
/// goes through the backend staging path.
pub fn chw_vec_to_tensor(
    data: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor<4>, OidnError> {
    if data.len() != validate_dimensions(width, height, channels)? {
        return Err(OidnError::InvalidArgument(
            "tensor buffer length does not match dimensions",
        ));
    }
    Ok(Tensor::<4>::from_data(
        TensorData::new(data, [1, channels, height, width]),
        device,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Layout helpers are inverses of each other and produce the documented
    /// `(y, x, c)` ordering.
    #[test]
    fn chw_hwc_roundtrip_3ch_2x2() {
        // C=3, H=2, W=2. Each value is unique so any permutation bug shows.
        let chw = vec![
            // R-plane (row-major H×W)
            1.0, 2.0, 3.0, 4.0, // G-plane
            5.0, 6.0, 7.0, 8.0, // B-plane
            9.0, 10.0, 11.0, 12.0,
        ];
        let hwc = chw_to_hwc(&chw, 3, 2, 2).unwrap();
        // Expect interleaved RGB per pixel in row-major scan.
        let expected_hwc = vec![
            1.0, 5.0, 9.0, // (y=0, x=0)
            2.0, 6.0, 10.0, // (y=0, x=1)
            3.0, 7.0, 11.0, // (y=1, x=0)
            4.0, 8.0, 12.0, // (y=1, x=1)
        ];
        assert_eq!(hwc, expected_hwc);

        let back = hwc_to_chw(&hwc, 3, 2, 2).unwrap();
        assert_eq!(back, chw);
    }

    #[test]
    fn chw_hwc_roundtrip_1ch_3x4() {
        // C=1: HWC and CHW differ only in length contract; values stay in-place.
        let chw: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let hwc = chw_to_hwc(&chw, 1, 3, 4).unwrap();
        assert_eq!(hwc, chw);
        let back = hwc_to_chw(&hwc, 1, 3, 4).unwrap();
        assert_eq!(back, chw);
    }

    /// Tensor build → host pull round-trip must preserve both data and
    /// dims, regardless of backend memory layout assumptions.
    #[test]
    fn tensor_chw_vec_roundtrip_ndarray() {
        let device = Device::ndarray();
        let original = vec![
            // C=3, H=2, W=2 — same shape as the layout test above.
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let t = chw_vec_to_tensor(original.clone(), 3, 2, 2, &device).unwrap();
        let (back, dims) = tensor_to_chw_vec(t).unwrap();
        assert_eq!(dims, [1, 3, 2, 2]);
        assert_eq!(back, original);
    }
}
