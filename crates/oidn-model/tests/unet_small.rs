//! UNet Small variant — same topology as Base but narrower channels.

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::prelude::*;

fn weights_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data").join("weights")
        .join(format!("{name}.tza"));
    if p.is_file() { Some(p) } else { None }
}

#[test]
fn unet_small_forward_shape() {
    type B = NdArray<f32>;
    let device = Default::default();
    let unet = oidn_model::UNet::<B>::new(3, 3, oidn_model::Variant::Small, &device);
    let input = Tensor::<B, 4>::zeros([1, 3, 128, 128], &device);
    let out = unet.forward(input);
    assert_eq!(out.dims(), [1, 3, 128, 128]);
}

#[test]
fn load_rt_hdr_small() {
    let Some(path) = weights_path("rt_hdr_small") else { return; };
    type B = NdArray<f32>;
    let device = Default::default();

    let bytes = std::fs::read(&path).unwrap();
    let tensors = oidn_tza::parse(&bytes).unwrap();

    let unet = oidn_model::UNet::<B>::new(3, 3, oidn_model::Variant::Small, &device);
    let unet = oidn_model::load_tza(unet, &tensors, &device).unwrap();

    let input = Tensor::<B, 4>::ones([1, 3, 64, 64], &device) * 0.5;
    let out = unet.forward(input).into_data();
    let v: Vec<f32> = out.convert::<f32>().to_vec().unwrap();
    for x in &v { assert!(x.is_finite()); }
    let mean_abs: f32 = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
    assert!(mean_abs > 0.0);
}

#[test]
fn load_rt_hdr_alb_nrm_small_9ch() {
    let Some(path) = weights_path("rt_hdr_alb_nrm_small") else { return; };
    type B = NdArray<f32>;
    let device = Default::default();

    let bytes = std::fs::read(&path).unwrap();
    let tensors = oidn_tza::parse(&bytes).unwrap();
    let unet = oidn_model::UNet::<B>::new(9, 3, oidn_model::Variant::Small, &device);
    let _unet = oidn_model::load_tza(unet, &tensors, &device).unwrap();
}
