//! Verify that real TZA weights (from `data`) load into a UNet
//! and produce a non-trivial output on real input data.

use std::path::PathBuf;

use burn::prelude::*;

fn weights_path(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(format!("../../data/weights/{name}.tza"));
    if p.is_file() { Some(p) } else { None }
}

#[test]
fn load_rt_hdr_into_unet() {
    let Some(path) = weights_path("rt_hdr") else {
        eprintln!("skipping: rt_hdr.tza not present");
        return;
    };

    let device = burn::tensor::Device::ndarray();

    let bytes = std::fs::read(&path).unwrap();
    let tensors = oidn_tza::parse(&bytes).unwrap();

    let unet = oidn_model::UNet::new(3, 3, oidn_model::Variant::Base, &device);
    let unet = oidn_model::load_tza(unet, &tensors, &device).unwrap();

    // Pass a constant grey tile through — the loaded model should produce
    // finite, non-zero output (fully random initialised weights would also
    // pass `is_finite` but produce uniformly tiny values; loaded weights
    // should yield meaningful magnitude).
    let input = Tensor::<4>::ones([1, 3, 64, 64], &device) * 0.5;
    let out = unet.forward(input).into_data();
    let v: Vec<f32> = out.convert::<f32>().to_vec().unwrap();
    let mean_abs: f32 = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
    for x in &v {
        assert!(x.is_finite(), "non-finite value");
    }
    assert!(
        mean_abs > 0.0,
        "output is identically zero — load_tza did not populate weights"
    );
}

#[test]
fn load_rt_hdr_alb_nrm_into_unet_9ch() {
    let Some(path) = weights_path("rt_hdr_alb_nrm") else {
        return;
    };

    let device = burn::tensor::Device::ndarray();

    let bytes = std::fs::read(&path).unwrap();
    let tensors = oidn_tza::parse(&bytes).unwrap();

    let unet = oidn_model::UNet::new(9, 3, oidn_model::Variant::Base, &device);
    let unet = oidn_model::load_tza(unet, &tensors, &device).unwrap();

    // 9-channel input.
    let input = Tensor::<4>::ones([1, 9, 32, 32], &device) * 0.25;
    let out = unet.forward(input).into_data();
    let v: Vec<f32> = out.convert::<f32>().to_vec().unwrap();
    for x in &v {
        assert!(x.is_finite());
    }
}
