use burn::tensor::{Device, Tensor, TensorData};
use oidn_model::{
    ChannelConfig, ChannelConfigLarge, LoadError, ModelDescriptor, UNet, UNetLarge, Variant,
};
use oidn_tza::{DType, Layout, Tensor as ArchiveTensor, TensorDesc, TensorMap};
use std::path::PathBuf;

fn shipped(stem: &str) -> TensorMap {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/weights")
        .join(format!("{stem}.tza"));
    oidn_tza::parse(&std::fs::read(path).expect("required shipped archive")).unwrap()
}

#[test]
fn all_shipped_archives_infer_loadable_channels_and_topology() {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/weights");
    let mut count = 0;
    let device = Device::ndarray();
    for file in std::fs::read_dir(directory).unwrap() {
        let path = file.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("tza") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let map = oidn_tza::parse(&std::fs::read(&path).unwrap()).unwrap();
        let desc = ModelDescriptor::from_tza(&map).unwrap();
        let expected = if stem.ends_with("_small") {
            Variant::Small
        } else if stem.ends_with("_large") {
            Variant::Large
        } else {
            Variant::Base
        };
        assert_eq!(desc.variant(), expected, "{stem}");
        assert_eq!(desc.out_channels(), 3, "{stem}");
        assert_eq!(desc.alignment(), 16);
        assert_eq!(
            desc.parameters_bytes(),
            map.values()
                .map(|t| t.desc.num_elements().unwrap() * 4)
                .sum::<usize>()
        );
        assert!(desc.activation_bytes_per_pixel() > (desc.in_channels() + 3) * 4);
        assert_eq!(
            desc.receptive_field(),
            if expected == Variant::Large { 202 } else { 174 }
        );
        let net = desc.load(&map, &device).unwrap();
        assert_eq!(net.in_channels(), desc.in_channels(), "{stem}");
        assert_eq!(net.out_channels(), 3, "{stem}");
        count += 1;
    }
    assert_eq!(count, 23, "required complete shipped model inventory");
}

#[test]
fn malformed_schema_and_parameters_fail_before_device_loading() {
    let original = shipped("rt_hdr_small");
    assert_eq!(
        ModelDescriptor::from_tza(&original).unwrap().variant(),
        Variant::Small
    );
    let mut bad = original.clone();
    let t = bad.get_mut("enc_conv0.bias").unwrap();
    t.data = t.data.slice(..t.data.len() - 1);
    assert!(matches!(
        ModelDescriptor::from_tza(&bad),
        Err(LoadError::InvalidTensor(_))
    ));
    let descriptor = ModelDescriptor::from_tza(&original).unwrap();
    assert!(matches!(
        descriptor.load(&bad, &Device::ndarray()),
        Err(LoadError::InvalidTensor(_))
    ));
    let mut bad = original.clone();
    bad.remove("dec_conv0.bias");
    assert!(matches!(
        ModelDescriptor::from_tza(&bad),
        Err(LoadError::MissingTensor(_))
    ));
    let mut bad = original.clone();
    let t = bad.get_mut("enc_conv1.weight").unwrap();
    // A valid raw tensor with an inconsistent graph edge.
    t.desc.dims[1] -= 1;
    t.data = t.data.slice(..t.desc.byte_size().unwrap());
    assert!(matches!(
        ModelDescriptor::from_tza(&bad),
        Err(LoadError::ShapeMismatch { .. })
    ));
    let mut bad = original.clone();
    bad.insert("unknown.bias".into(), bad["enc_conv0.bias"].clone());
    assert!(matches!(
        ModelDescriptor::from_tza(&bad),
        Err(LoadError::InvalidSchema(_))
    ));
    let mut bad = original;
    let t = bad.get_mut("dec_conv0.bias").unwrap();
    let mut data = t.data.to_vec();
    data[..2].copy_from_slice(&0x7e00u16.to_le_bytes()); // f16 NaN
    t.data = data.into();
    assert!(matches!(
        ModelDescriptor::from_tza(&bad),
        Err(LoadError::NonFiniteTensor(_))
    ));
}

fn parameter(map: &mut TensorMap, name: &str, dims: Vec<u32>, layout: Layout) {
    let desc = TensorDesc {
        dims,
        layout,
        dtype: DType::Float16,
    };
    let data = vec![0; desc.byte_size().unwrap()];
    map.insert(
        name.into(),
        ArchiveTensor {
            desc,
            data: data.into(),
        },
    );
}

fn layer(map: &mut TensorMap, name: &str, input: usize, output: usize) {
    parameter(
        map,
        &format!("{name}.weight"),
        vec![output as u32, input as u32, 3, 3],
        Layout::Oihw,
    );
    parameter(map, &format!("{name}.bias"), vec![output as u32], Layout::X);
}

fn base_map(c: ChannelConfig) -> TensorMap {
    let mut map = TensorMap::new();
    for (name, input, output) in [
        ("enc_conv0", 3, c.ec1),
        ("enc_conv1", c.ec1, c.ec1),
        ("enc_conv2", c.ec1, c.ec2),
        ("enc_conv3", c.ec2, c.ec3),
        ("enc_conv4", c.ec3, c.ec4),
        ("enc_conv5a", c.ec4, c.ec5),
        ("enc_conv5b", c.ec5, c.ec5),
        ("dec_conv4a", c.ec5 + c.ec3, c.dc4),
        ("dec_conv4b", c.dc4, c.dc4),
        ("dec_conv3a", c.dc4 + c.ec2, c.dc3),
        ("dec_conv3b", c.dc3, c.dc3),
        ("dec_conv2a", c.dc3 + c.ec1, c.dc2a),
        ("dec_conv2b", c.dc2a, c.dc2b),
        ("dec_conv1a", c.dc2b + 3, c.dc1a),
        ("dec_conv1b", c.dc1a, c.dc1b),
        ("dec_conv0", c.dc1b, 3),
    ] {
        layer(&mut map, name, input, output);
    }
    map
}

fn large_map(c: ChannelConfigLarge) -> TensorMap {
    let mut map = TensorMap::new();
    for (name, input, output) in [
        ("enc_conv1a", 3, c.ec1),
        ("enc_conv1b", c.ec1, c.ec1),
        ("enc_conv2a", c.ec1, c.ec2),
        ("enc_conv2b", c.ec2, c.ec2),
        ("enc_conv3a", c.ec2, c.ec3),
        ("enc_conv3b", c.ec3, c.ec3),
        ("enc_conv4a", c.ec3, c.ec4),
        ("enc_conv4b", c.ec4, c.ec4),
        ("enc_conv5a", c.ec4, c.ec5),
        ("enc_conv5b", c.ec5, c.ec5),
        ("dec_conv4a", c.ec5 + c.ec3, c.dc4),
        ("dec_conv4b", c.dc4, c.dc4),
        ("dec_conv3a", c.dc4 + c.ec2, c.dc3),
        ("dec_conv3b", c.dc3, c.dc3),
        ("dec_conv2a", c.dc3 + c.ec1, c.dc2),
        ("dec_conv2b", c.dc2, c.dc2),
        ("dec_conv1a", c.dc2 + 3, c.dc1),
        ("dec_conv1b", c.dc1, c.dc1),
        ("dec_conv1c", c.dc1, 3),
    ] {
        layer(&mut map, name, input, output);
    }
    map
}

#[test]
fn xl_and_custom_widths_are_preserved() {
    let xl = large_map(ChannelConfigLarge::XL);
    let desc = ModelDescriptor::from_tza(&xl).unwrap();
    assert_eq!(desc.variant(), Variant::XLarge);
    assert_eq!(desc.receptive_field(), 202);
    // Construction and loading are tested too, not just filename-free classification.
    let device = Device::ndarray();
    let _net = desc.load(&xl, &device).unwrap();
    let c = ChannelConfig {
        ec1: 2,
        ec2: 3,
        ec3: 4,
        ec4: 5,
        ec5: 6,
        dc4: 7,
        dc3: 8,
        dc2a: 9,
        dc2b: 10,
        dc1a: 11,
        dc1b: 12,
    };
    let mut map = base_map(c);
    // Route each RGB channel through the original-input skip using exact
    // center-tap identity weights. This independently checks concat order,
    // OIHW decoding, custom widths and final inference ReLU on asymmetric data.
    for (name, offset) in [("dec_conv1a", c.dc2b), ("dec_conv1b", 0), ("dec_conv0", 0)] {
        let tensor = map.get_mut(&format!("{name}.weight")).unwrap();
        let input_channels = tensor.desc.dims[1] as usize;
        let mut data = tensor.data.to_vec();
        for channel in 0..3 {
            let index = (channel * input_channels + offset + channel) * 9 + 4;
            data[2 * index..2 * index + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        tensor.data = data.into();
    }
    let desc = ModelDescriptor::from_tza(&map).unwrap();
    let net = desc.load(&map, &device).unwrap();
    let samples: Vec<f32> = (0..3 * 16 * 16).map(|n| (n as f32 - 80.0) / 16.0).collect();
    let output = net
        .forward(Tensor::<4>::from_data(
            TensorData::new(samples.clone(), [1, 3, 16, 16]),
            &device,
        ))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let expected: Vec<f32> = samples.iter().map(|&v| v.max(0.0)).collect();
    assert_eq!(output, expected);
    assert_eq!(net.in_channels(), 3);
    // Legacy loaders must also reject a malformed public payload without a panic.
    let mut broken = map;
    let t = broken.get_mut("enc_conv0.bias").unwrap();
    t.data = t.data.slice(..t.data.len() - 1);
    let model = UNet::new_with(3, 3, c, &device);
    assert!(oidn_model::load_tza(model, &broken, &device).is_err());
    let tiny = ChannelConfigLarge {
        ec1: 2,
        ec2: 2,
        ec3: 2,
        ec4: 2,
        ec5: 2,
        dc4: 2,
        dc3: 2,
        dc2: 2,
        dc1: 2,
    };
    let large = UNetLarge::new_with(3, 3, tiny, &device);
    assert!(oidn_model::load_tza_large(large, &broken, &device).is_err());
}
