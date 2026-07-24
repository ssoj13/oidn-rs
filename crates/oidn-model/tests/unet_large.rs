//! UNetLarge shape + real-weight load tests on the CPU NdArray backend.

use std::path::PathBuf;

use burn::prelude::*;

fn weights_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("weights")
        .join(format!("{name}.tza"));
    if p.is_file() { Some(p) } else { None }
}

#[test]
fn unet_large_base_forward_shape() {
    let device = burn::tensor::Device::ndarray();
    let unet = oidn_model::UNetLarge::new(3, 3, &device);
    let input = Tensor::<4>::zeros([1, 3, 128, 128], &device);
    let out = unet.forward(input);
    assert_eq!(out.dims(), [1, 3, 128, 128]);
}

#[test]
fn unet_large_xl_forward_shape() {
    let device = burn::tensor::Device::ndarray();
    let unet = oidn_model::UNetLarge::new_xl(3, 3, &device);
    let input = Tensor::<4>::zeros([1, 3, 96, 96], &device);
    let out = unet.forward(input);
    assert_eq!(out.dims(), [1, 3, 96, 96]);
}

#[test]
fn load_rt_alb_large() {
    let Some(path) = weights_path("rt_alb_large") else {
        return;
    };
    let device = burn::tensor::Device::ndarray();

    let bytes = std::fs::read(&path).unwrap();
    let tensors = oidn_tza::parse(&bytes).unwrap();
    assert_eq!(
        oidn_model::Variant::from_tensor_names(tensors.keys()),
        oidn_model::Variant::Large
    );

    let unet = oidn_model::UNetLarge::new(3, 3, &device);
    let unet = oidn_model::load_tza_large(unet, &tensors, &device).unwrap();

    let input = Tensor::<4>::ones([1, 3, 64, 64], &device) * 0.5;
    let out = unet.forward(input).into_data();
    let v: Vec<f32> = out.convert::<f32>().to_vec().unwrap();
    for x in &v {
        assert!(x.is_finite());
    }
    let mean_abs: f32 = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
    assert!(mean_abs > 0.0, "rt_alb_large produced all-zero output");
}

#[test]
fn load_rt_nrm_large() {
    let Some(path) = weights_path("rt_nrm_large") else {
        return;
    };
    let device = burn::tensor::Device::ndarray();

    let bytes = std::fs::read(&path).unwrap();
    let tensors = oidn_tza::parse(&bytes).unwrap();

    let unet = oidn_model::UNetLarge::new(3, 3, &device);
    let _unet = oidn_model::load_tza_large(unet, &tensors, &device).unwrap();
}

#[test]
fn load_rt_hdr_calb_cnrm_large_9ch() {
    let Some(path) = weights_path("rt_hdr_calb_cnrm_large") else {
        return;
    };
    let device = burn::tensor::Device::ndarray();

    let bytes = std::fs::read(&path).unwrap();
    let tensors = oidn_tza::parse(&bytes).unwrap();

    // 9-ch input (colour + albedo + normal).
    let unet = oidn_model::UNetLarge::new(9, 3, &device);
    let unet = oidn_model::load_tza_large(unet, &tensors, &device).unwrap();
    let input = Tensor::<4>::ones([1, 9, 64, 64], &device) * 0.25;
    let out = unet.forward(input).into_data();
    let v: Vec<f32> = out.convert::<f32>().to_vec().unwrap();
    for x in &v {
        assert!(x.is_finite());
    }
}
