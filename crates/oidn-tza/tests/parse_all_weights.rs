//! Verify that the TZA parser successfully reads every shipped weights file.
//!
//! Skipped silently if the `data` submodule isn't initialised.

use std::path::PathBuf;

fn weights_dir() -> Option<PathBuf> {
    // Tests run from the crate root (crates/oidn-tza/), so weights live up two levels.
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("..").join("data").join("weights");
    if p.is_dir() { Some(p) } else { None }
}

#[test]
fn parse_all_shipped_tza_files() {
    let Some(dir) = weights_dir() else {
        eprintln!("skipping: data not initialised (run `git submodule update --init`)");
        return;
    };

    let mut count = 0usize;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("tza") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let map = oidn_tza::parse(&bytes)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));
        assert!(!map.is_empty(), "{} has no tensors", path.display());
        count += 1;
    }
    assert!(count >= 20, "expected ~24 tza files, found {count}");
}

#[test]
fn rt_hdr_has_expected_layer_set() {
    let Some(dir) = weights_dir() else { return; };
    let bytes = std::fs::read(dir.join("rt_hdr.tza")).unwrap();
    let map = oidn_tza::parse(&bytes).unwrap();

    // 16 conv layers × {weight, bias} = 32 tensors.
    assert_eq!(map.len(), 32, "rt_hdr should have 32 tensors");

    // Sample of names that must exist (they match PyTorch state_dict naming).
    for n in &[
        "enc_conv0.weight", "enc_conv0.bias",
        "enc_conv5b.weight", "dec_conv4a.weight",
        "dec_conv1b.bias",  "dec_conv0.weight",
    ] {
        assert!(map.contains_key(*n), "missing tensor {n}");
    }

    // Spot-check shapes (must match _ref/oidn/training/model.py:UNet base config).
    let w = &map["enc_conv0.weight"];
    assert_eq!(w.desc.dims, vec![32, 3, 3, 3], "enc_conv0.weight shape mismatch");
    assert_eq!(w.desc.layout, oidn_tza::Layout::Oihw);
    assert_eq!(w.desc.dtype, oidn_tza::DType::Float16);

    let b = &map["enc_conv0.bias"];
    assert_eq!(b.desc.dims, vec![32], "enc_conv0.bias shape mismatch");
    assert_eq!(b.desc.layout, oidn_tza::Layout::X);
}
