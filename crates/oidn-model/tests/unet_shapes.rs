//! Forward-pass shape sanity check on the CPU NdArray backend — fast, no GPU.

use burn::backend::NdArray;
use burn::prelude::*;

#[test]
fn unet_base_forward_shape() {
    type B = NdArray<f32>;
    let device = Default::default();

    let unet = oidn_model::UNet::<B>::new(3, 3, oidn_model::Variant::Base, &device);

    // 256×256 must work — multiple of the 16-pixel alignment requirement.
    let input = Tensor::<B, 4>::zeros([1, 3, 256, 256], &device);
    let out = unet.forward(input);
    assert_eq!(out.dims(), [1, 3, 256, 256]);

    // Sanity: forward on zero input must not produce NaN/Inf (only matters for
    // randomly initialised weights, but verifies no degenerate ops).
    let data = out.into_data();
    let v: Vec<f32> = data.convert::<f32>().to_vec().unwrap();
    for x in &v {
        assert!(x.is_finite(), "non-finite value in output");
    }
}

#[test]
fn unet_with_aux_features() {
    // 9 input channels: color (3) + albedo (3) + normal (3).
    type B = NdArray<f32>;
    let device = Default::default();

    let unet = oidn_model::UNet::<B>::new(9, 3, oidn_model::Variant::Base, &device);
    let input = Tensor::<B, 4>::zeros([1, 9, 128, 128], &device);
    let out = unet.forward(input);
    assert_eq!(out.dims(), [1, 3, 128, 128]);
}
