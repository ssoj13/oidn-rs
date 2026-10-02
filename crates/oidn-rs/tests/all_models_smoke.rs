//! Every shipped archive loads through the executable descriptor on explicit CPU.
//! Structural and nonzero-finite output coverage; not native numerical parity.
use burn::tensor::{Device, Tensor};
use oidn_model::{ModelDescriptor, Variant};
use std::path::PathBuf;

#[test]
fn all_shipped_models_load_and_forward() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/weights");
    assert!(dir.is_dir(), "required shipped weights are missing");
    let device = Device::ndarray();
    let mut count = 0;
    let mut variants = [0; 4];
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("tza") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let tensors = oidn_tza::parse(&bytes).unwrap();
        let descriptor = ModelDescriptor::from_tza(&tensors).unwrap();
        let index = match descriptor.variant() {
            Variant::Base => 0,
            Variant::Small => 1,
            Variant::Large => 2,
            Variant::XLarge => 3,
        };
        variants[index] += 1;
        let net = descriptor.load(&tensors, &device).unwrap();
        let output = net.forward(Tensor::<4>::full(
            [1, descriptor.in_channels(), 16, 16],
            0.5,
            &device,
        ));
        assert_eq!(output.dims(), [1, 3, 16, 16]);
        let values = output.into_data().to_vec::<f32>().unwrap();
        assert!(
            values.iter().all(|x| x.is_finite()),
            "{path:?}: nonfinite output"
        );
        assert!(
            values.iter().any(|x| *x > 1e-6),
            "{path:?}: all-zero output"
        );
        count += 1;
    }
    assert_eq!(count, 23, "required model inventory changed");
    assert!(
        variants[1] >= 6 && variants[2] >= 3,
        "Small/Large routes not covered"
    );
}
