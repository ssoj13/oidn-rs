//! Smoke test: every shipped `.tza` parses, picks the right variant, loads
//! into a UNet of the right topology, and produces finite forward output on
//! the CPU NdArray backend. Catches structural regressions in variant
//! detection or layer naming across the full 24-file weights set.

use std::path::PathBuf;

use burn::backend::NdArray;
use burn::prelude::*;
use oidn_model::{UNet, UNetLarge, Variant};

fn weights_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data").join("weights");
    if p.is_dir() { Some(p) } else { None }
}

/// Number of input channels expected by the model — inferred from the
/// `enc_conv0.weight` / `enc_conv1a.weight` shape.
fn infer_in_channels(tensors: &oidn_tza::TensorMap) -> usize {
    let first = tensors.get("enc_conv0.weight")
        .or_else(|| tensors.get("enc_conv1a.weight"))
        .expect("model has neither enc_conv0 nor enc_conv1a");
    // oihw layout — [out, in, kh, kw], so dims[1] is the input channel count.
    first.desc.dims[1] as usize
}

#[test]
fn all_shipped_models_load_and_forward() {
    let Some(dir) = weights_dir() else {
        eprintln!("skipping: weights submodule not initialised");
        return;
    };

    type B = NdArray<f32>;
    let device = Default::default();

    let mut count = 0usize;
    let mut by_variant = [0usize, 0usize, 0usize]; // base, small, large

    for entry in std::fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("tza") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();

        let bytes = std::fs::read(&path).unwrap();
        let tensors = oidn_tza::parse(&bytes)
            .unwrap_or_else(|e| panic!("parse failed for {stem}: {e}"));

        let in_ch = infer_in_channels(&tensors);
        let out_ch = 3;

        // Variant detection: filename suffix is authoritative.
        // `_large` ⇒ UNetLarge; `_small` ⇒ UNet Small; otherwise UNet Base.
        if stem.ends_with("_large") {
            by_variant[2] += 1;
            let unet = UNetLarge::<B>::new(in_ch, out_ch, &device);
            let unet = oidn_model::load_tza_large(unet, &tensors, &device)
                .unwrap_or_else(|e| panic!("load_tza_large failed for {stem}: {e}"));
            let input = Tensor::<B, 4>::ones([1, in_ch, 64, 64], &device) * 0.5;
            let out = unet.forward(input).into_data();
            let v: Vec<f32> = out.convert::<f32>().to_vec().unwrap();
            for x in &v { assert!(x.is_finite(), "non-finite output for {stem}"); }
        } else {
            let variant = if stem.ends_with("_small") {
                by_variant[1] += 1;
                Variant::Small
            } else {
                by_variant[0] += 1;
                Variant::Base
            };
            let unet = UNet::<B>::new(in_ch, out_ch, variant, &device);
            let unet = oidn_model::load_tza(unet, &tensors, &device)
                .unwrap_or_else(|e| panic!("load_tza failed for {stem}: {e}"));
            let input = Tensor::<B, 4>::ones([1, in_ch, 64, 64], &device) * 0.5;
            let out = unet.forward(input).into_data();
            let v: Vec<f32> = out.convert::<f32>().to_vec().unwrap();
            for x in &v { assert!(x.is_finite(), "non-finite output for {stem}"); }
        }

        count += 1;
        eprintln!("OK: {stem}");
    }

    eprintln!(
        "Total: {count}, by variant: base={}, small={}, large={}",
        by_variant[0], by_variant[1], by_variant[2]
    );
    assert!(count >= 20, "expected ~24 weight files, got {count}");
    assert!(by_variant[2] >= 3, "expected at least 3 _large variants, got {}", by_variant[2]);
    assert!(by_variant[1] >= 6, "expected at least 6 _small variants, got {}", by_variant[1]);
}
