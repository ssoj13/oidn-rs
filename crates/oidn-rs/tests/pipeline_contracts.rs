//! Public pipeline contracts across image and tensor frontends.
//! Required shipped archives are test assets, not optional skip conditions.

use std::path::PathBuf;

use burn::tensor::{Device, Tensor, TensorData};
use oidn_rs::image_tensor::{chw_to_hwc, hwc_to_chw, tensor_to_chw_vec};
use oidn_rs::{Filter, Image, PixelFormat, Quality, RtFilter, RtLightmapFilter};

fn weights() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/weights");
    assert!(
        path.is_dir(),
        "required shipped archives: {}",
        path.display()
    );
    path
}

fn tensor(values: &[f32], w: usize, h: usize, device: &Device) -> Tensor<4> {
    Tensor::from_data(
        TensorData::new(hwc_to_chw(values, 3, h, w).unwrap(), [1, 3, h, w]),
        device,
    )
}

fn pixels(t: Tensor<4>) -> Vec<f32> {
    let (values, [_, c, h, w]) = tensor_to_chw_vec(t).unwrap();
    chw_to_hwc(&values, c, h, w).unwrap()
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "nonfinite pixel {i}: {a}, {b}"
        );
        assert!(
            (a - b).abs() <= 2e-5 * b.abs().max(1.0),
            "pixel {i}: {a} != {b}"
        );
    }
}

fn fixture(w: usize, h: usize, signed: bool) -> Vec<f32> {
    (0..w * h)
        .flat_map(|i| {
            let x = (i % w) as f32 / w as f32;
            let y = (i / w) as f32 / h as f32;
            if signed {
                [-0.8 + x * 0.2, -0.6 + y * 0.2, -0.3]
            } else {
                [0.1 + x * 0.7, 0.2 + y * 0.3, 0.4]
            }
        })
        .collect()
}

#[test]
fn host_mutable_and_immutable_tensors_agree_with_fresh_frames() {
    let device = Device::ndarray();
    let (w, h) = (9, 7);
    let values = fixture(w, h, false);
    let mut host = RtFilter::builder(&device, weights())
        .hdr(true)
        .quality(Quality::Fast)
        .input_scale(Some(0.2))
        .build();
    host.set_color(&Image::from_rgb_f32(&values, w, h)).unwrap();
    host.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    host.execute().unwrap();
    let expected = decode(&host.take_output().unwrap().0);

    let mut mutable = RtFilter::builder(&device, weights())
        .hdr(true)
        .quality(Quality::Fast)
        .input_scale(Some(0.2))
        .build();
    let committed = mutable
        .commit_tensor_model(w, h, true, false, false)
        .unwrap();
    let mut first: Option<Vec<f32>> = None;
    for factor in [1.0, 2.0, 1.0] {
        let frame: Vec<f32> = values.iter().map(|v| v * factor).collect();
        mutable
            .set_color_tensor(tensor(&frame, w, h, &device))
            .unwrap();
        mutable.allocate_output_tensor(w, h).unwrap();
        mutable.execute().unwrap();
        let actual = pixels(mutable.take_output_tensor().unwrap());
        let source = tensor(&frame, w, h, &device);
        let immutable = pixels(
            committed
                .execute_tensors(Some(source.clone()), None, None, None)
                .unwrap(),
        );
        close(&actual, &immutable);
        assert_eq!(
            pixels(source),
            frame,
            "immutable execution changed the caller's input tensor"
        );
        if factor == 1.0 {
            close(&actual, &expected);
            if let Some(previous) = &first {
                close(&actual, previous);
            } else {
                first = Some(actual);
            }
        } else {
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .any(|(a, b)| (a - b).abs() > 1e-3),
                "fresh changed frame must affect output"
            );
        }
    }
}

#[test]
fn auxiliary_only_frontends_agree_and_normal_output_is_signed() {
    let device = Device::ndarray();
    let (w, h) = (8, 7);
    for normal in [false, true] {
        let values = fixture(w, h, normal);
        let mut host = RtFilter::builder(&device, weights())
            .quality(Quality::Balanced)
            .input_scale(Some(0.5))
            .build();
        let image = Image::from_rgb_f32(&values, w, h);
        if normal {
            host.set_normal(&image).unwrap();
        } else {
            host.set_albedo(&image).unwrap();
        }
        host.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
        host.execute().unwrap();
        let actual = decode(&host.take_output().unwrap().0);
        let immutable = RtFilter::builder(&device, weights())
            .quality(Quality::Balanced)
            .input_scale(Some(0.5))
            .build()
            .commit_tensor_model(w, h, false, !normal, normal)
            .unwrap();
        let t = tensor(&values, w, h, &device);
        let result = if normal {
            immutable.execute_tensors(None, None, Some(t), None)
        } else {
            immutable.execute_tensors(None, Some(t), None, None)
        }
        .unwrap();
        close(&actual, &pixels(result));
        if normal {
            assert!(
                actual.iter().any(|v| *v < -0.1),
                "signed normal output was lost"
            );
        }
    }
}

#[test]
fn scalar_destination_is_rgb_mean_after_inverse_transfer() {
    let device = Device::ndarray();
    let (w, h) = (8, 7);
    for normal in [false, true] {
        let values = fixture(w, h, normal);
        let mut outputs = Vec::new();
        for format in [PixelFormat::Rgb32f, PixelFormat::R32f] {
            let mut filter = RtFilter::builder(&device, weights())
                .hdr(!normal)
                .quality(Quality::Balanced)
                .input_scale(Some(0.25))
                .build();
            let image = Image::from_rgb_f32(&values, w, h);
            if normal {
                filter.set_normal(&image).unwrap();
            } else {
                filter.set_color(&image).unwrap();
            }
            filter.allocate_output(w, h, format).unwrap();
            filter.execute().unwrap();
            outputs.push(decode(&filter.take_output().unwrap().0));
        }
        assert!(
            outputs[0]
                .iter()
                .all(|v| v.is_finite() && (!normal || (*v >= -4.0 && *v < 4.0))),
            "RGB oracle must be unsaturated"
        );
        let mean: Vec<f32> = outputs[0]
            .as_chunks::<3>()
            .0
            .iter()
            .map(|rgb| (rgb[0] + rgb[1] + rgb[2]) / 3.0)
            .collect();
        close(&outputs[1], &mean);
    }
}

#[test]
fn directional_lightmap_keeps_signed_output_and_scalar_reduction() {
    let device = Device::ndarray();
    let (w, h) = (8, 7);
    let values = fixture(w, h, true);
    let mut outputs = Vec::new();
    for format in [PixelFormat::Rgb32f, PixelFormat::R32f] {
        let mut filter = RtLightmapFilter::builder(&device, weights())
            .directional(true)
            .input_scale(Some(0.5))
            .build();
        filter
            .set_color(&Image::from_rgb_f32(&values, w, h))
            .unwrap();
        filter.allocate_output(w, h, format).unwrap();
        filter.execute().unwrap();
        outputs.push(decode(&filter.take_output().unwrap().0));
    }
    assert!(
        outputs[0].iter().any(|v| *v < -0.1),
        "directional output must retain negative irradiance gradients"
    );
    assert!(
        outputs[0].iter().all(|v| *v >= -2.0 && *v < 2.0),
        "RGB oracle must be unsaturated"
    );
    let mean: Vec<f32> = outputs[0]
        .as_chunks::<3>()
        .0
        .iter()
        .map(|rgb| (rgb[0] + rgb[1] + rgb[2]) / 3.0)
        .collect();
    close(&outputs[1], &mean);
}

#[test]
fn custom_weights_do_not_bypass_mode_scale_or_geometry_validation() {
    let device = Device::ndarray();
    let bytes = std::fs::read(weights().join("rt_hdr.tza")).unwrap();
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        let filter = RtFilter::builder(&device, weights())
            .hdr(true)
            .weights(bytes.clone())
            .input_scale(Some(scale))
            .build();
        assert!(
            filter
                .commit_tensor_model(8, 8, true, false, false)
                .is_err(),
            "invalid scale {scale}"
        );
    }
    let filter = RtFilter::builder(&device, weights())
        .hdr(true)
        .srgb(true)
        .weights(bytes.clone())
        .build();
    assert!(
        filter
            .commit_tensor_model(8, 8, true, false, false)
            .is_err()
    );
    let filter = RtFilter::builder(&device, weights())
        .hdr(true)
        .weights(bytes.clone())
        .build();
    assert!(
        filter
            .commit_tensor_model(8, 8, false, true, false)
            .is_err()
    );
    assert!(
        filter
            .commit_tensor_model(0, 8, true, false, false)
            .is_err()
    );
    assert!(
        filter.commit_tensor_model(8, 8, true, true, false).is_err(),
        "3-channel archive cannot serve 6-channel inputs"
    );
    let mut committed = filter
        .commit_tensor_model(8, 8, true, false, false)
        .unwrap();
    for shape in [[2, 3, 8, 8], [1, 2, 8, 8], [1, 3, 7, 8]] {
        assert!(
            committed
                .execute_tensors(Some(Tensor::zeros(shape, &device)), None, None, None)
                .is_err()
        );
    }
    assert!(committed.execute_tensors(None, None, None, None).is_err());
    committed.set_input_scale(Some(f32::NAN));
    assert!(
        committed
            .execute_tensors(Some(Tensor::zeros([1, 3, 8, 8], &device)), None, None, None)
            .is_err()
    );
}

#[test]
fn albedo_primary_transfer_and_scale_follow_public_contract() {
    let device = Device::ndarray();
    let (w, h) = (8, 7);
    let values = fixture(w, h, false);
    let gamma: Vec<f32> = values
        .iter()
        .copied()
        .map(oidn_rs::color::srgb_forward)
        .collect();
    let mut outputs = Vec::new();
    for (source, scale, srgb) in [(&values, 1.0, false), (&gamma, 1.0, true)] {
        let mut filter = RtFilter::builder(&device, weights())
            .quality(Quality::Balanced)
            .input_scale(Some(scale))
            .srgb(srgb)
            .build();
        filter
            .set_albedo(&Image::from_rgb_f32(source, w, h))
            .unwrap();
        filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
        filter.execute().unwrap();
        outputs.push(decode(&filter.take_output().unwrap().0));
    }
    assert!(
        outputs[1].iter().all(|v| *v < 1.0),
        "gamma oracle must be unsaturated"
    );
    let decoded: Vec<f32> = outputs[1]
        .iter()
        .copied()
        .map(oidn_rs::color::srgb_inverse)
        .collect();
    close(&outputs[0], &decoded);

    // Equal valid preprocessed input must yield equal output before inverse scale.
    let doubled: Vec<f32> = values.iter().map(|v| v * 2.0).collect();
    let mut filter = RtFilter::builder(&device, weights())
        .quality(Quality::Balanced)
        .input_scale(Some(0.5))
        .build();
    filter
        .set_albedo(&Image::from_rgb_f32(&doubled, w, h))
        .unwrap();
    filter.allocate_output(w, h, PixelFormat::Rgb32f).unwrap();
    filter.execute().unwrap();
    let scaled: Vec<f32> = decode(&filter.take_output().unwrap().0)
        .iter()
        .map(|v| v * 0.5)
        .collect();
    close(&scaled, &outputs[0]);
}
