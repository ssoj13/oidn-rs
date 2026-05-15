//! Tensor-based I/O for filters — foundation of the Phase I GPU-only pipeline.
//!
//! The legacy [`crate::image::Image`] / [`crate::image::ImageMut`] API
//! continues to work; filters that accept tensor inputs currently
//! host-roundtrip them through `Vec<f32>` (CHW → HWC) before feeding the
//! existing CPU `unet_runner`. Sub-tasks I.2–I.5 lift that roundtrip onto
//! Burn tensor ops without changing the public API.
//!
//! Layout convention: tensors are `[1, C, H, W]` (NCHW). The legacy
//! [`Image::to_rgb_f32`](crate::image::Image::to_rgb_f32) helper produces
//! HWC. The two layout helpers in this module translate between them.

use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Convert a flat NCHW `f32` slice into HWC layout.
///
/// Length contract: `chw.len() == channels * height * width`. Returns a
/// freshly allocated `Vec<f32>` of the same length in HWC order
/// `[(y, x, c)]`.
pub fn chw_to_hwc(chw: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    debug_assert_eq!(chw.len(), channels * height * width);
    let mut hwc = vec![0.0f32; channels * height * width];
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
    hwc
}

/// Convert a flat HWC `f32` slice into NCHW layout.
///
/// Inverse of [`chw_to_hwc`]; same length contract.
pub fn hwc_to_chw(hwc: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    debug_assert_eq!(hwc.len(), channels * height * width);
    let mut chw = vec![0.0f32; channels * height * width];
    let stride_c = height * width;
    for y in 0..height {
        for x in 0..width {
            let src_off = (y * width + x) * channels;
            for c in 0..channels {
                chw[c * stride_c + y * width + x] = hwc[src_off + c];
            }
        }
    }
    chw
}

/// Pull a `[1, C, H, W]` Burn tensor onto the host as a `Vec<f32>` in CHW
/// order. Returns the data plus the original `[N, C, H, W]` dims so the
/// caller doesn't have to query them separately after the move.
pub fn tensor_to_chw_vec<B: Backend>(t: Tensor<B, 4>) -> (Vec<f32>, [usize; 4]) {
    let dims = t.dims();
    let v = t
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("Tensor<B, 4> → Vec<f32> conversion failed (NCHW must be f32-compatible)");
    (v, dims)
}

/// Build a `[1, C, H, W]` Burn tensor from a flat CHW `f32` buffer.
///
/// Used by [`RtFilter::take_output_tensor`](crate::filters::rt::RtFilter::take_output_tensor)
/// and the upcoming I.2 GPU pre-process path. The data is uploaded to
/// `device` via Burn's [`TensorData`] machinery; for `WgpuBackend` this
/// goes through cubecl-wgpu's staging path today (I.5 will swap that for
/// an in-place wrap once the cubecl public API allows it).
pub fn chw_vec_to_tensor<B: Backend>(
    data: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    debug_assert_eq!(data.len(), channels * height * width);
    Tensor::<B, 4>::from_data(TensorData::new(data, [1, channels, height, width]), device)
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
            1.0, 2.0,
            3.0, 4.0,
            // G-plane
            5.0, 6.0,
            7.0, 8.0,
            // B-plane
            9.0, 10.0,
            11.0, 12.0,
        ];
        let hwc = chw_to_hwc(&chw, 3, 2, 2);
        // Expect interleaved RGB per pixel in row-major scan.
        let expected_hwc = vec![
            1.0, 5.0, 9.0,   // (y=0, x=0)
            2.0, 6.0, 10.0,  // (y=0, x=1)
            3.0, 7.0, 11.0,  // (y=1, x=0)
            4.0, 8.0, 12.0,  // (y=1, x=1)
        ];
        assert_eq!(hwc, expected_hwc);

        let back = hwc_to_chw(&hwc, 3, 2, 2);
        assert_eq!(back, chw);
    }

    #[test]
    fn chw_hwc_roundtrip_1ch_3x4() {
        // C=1: HWC and CHW differ only in length contract; values stay in-place.
        let chw: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let hwc = chw_to_hwc(&chw, 1, 3, 4);
        assert_eq!(hwc, chw);
        let back = hwc_to_chw(&hwc, 1, 3, 4);
        assert_eq!(back, chw);
    }

    /// Tensor build → host pull round-trip must preserve both data and
    /// dims, regardless of backend memory layout assumptions.
    #[test]
    fn tensor_chw_vec_roundtrip_ndarray() {
        use burn::backend::NdArray;
        use burn::backend::ndarray::NdArrayDevice;
        type B = NdArray<f32>;
        let device = NdArrayDevice::default();
        let original = vec![
            // C=3, H=2, W=2 — same shape as the layout test above.
            1.0_f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let t = chw_vec_to_tensor::<B>(original.clone(), 3, 2, 2, &device);
        let (back, dims) = tensor_to_chw_vec(t);
        assert_eq!(dims, [1, 3, 2, 2]);
        assert_eq!(back, original);
    }
}
