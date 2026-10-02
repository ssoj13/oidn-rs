//! Process-level CLI contracts; these tests never initialize a GPU.
use std::process::Command;

#[test]
fn probe_json_preserves_escaped_tensor_names() {
    let name = "tensor\"\\\n\t";
    let mut archive = Vec::new();
    archive.extend_from_slice(&0x41d7_u16.to_le_bytes());
    archive.extend_from_slice(&[2, 0]);
    archive.extend_from_slice(&12_u64.to_le_bytes());
    archive.extend_from_slice(&1_u32.to_le_bytes());
    archive.extend_from_slice(&(name.len() as u16).to_le_bytes());
    archive.extend_from_slice(name.as_bytes());
    archive.push(1);
    archive.extend_from_slice(&1_u32.to_le_bytes());
    archive.extend_from_slice(b"xf");
    let offset = archive.len() + 8;
    archive.extend_from_slice(&(offset as u64).to_le_bytes());
    archive.extend_from_slice(&1_f32.to_le_bytes());
    let path = std::env::temp_dir().join(format!("oidn-probe-{}.tza", std::process::id()));
    std::fs::write(&path, archive).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_oidn-rs"))
        .arg("probe")
        .arg(&path)
        .arg("--json")
        .output();
    std::fs::remove_file(&path).unwrap();
    let output = output.unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["name"], name);
    assert_eq!(value["dims"], serde_json::json!([1]));
}

#[test]
fn invalid_iterations_and_family_flags_exit_before_input_loading() {
    for args in [
        vec!["bench", "--iters", "0"],
        vec![
            "denoise",
            "-i",
            "missing.pfm",
            "-o",
            "out.pfm",
            "--hdr",
            "--iters",
            "0",
        ],
        vec![
            "denoise",
            "-i",
            "missing.pfm",
            "-o",
            "out.pfm",
            "--hdr",
            "--dir",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_oidn-rs"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(
            error.contains("--iters must be positive") || error.contains("directional"),
            "{error}"
        );
    }
}

#[test]
fn cpu_fast_uses_ordinary_weight_resolution() {
    let directory = std::env::temp_dir().join(format!("oidn-cli-fast-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let input = directory.join("input.pfm");
    let output_path = directory.join("output.pfm");
    let mut pixels = b"PF\n3 2\n-1.0\n".to_vec();
    for value in [0.25_f32, 0.5, 2.0].into_iter().cycle().take(18) {
        pixels.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&input, pixels).unwrap();
    let weights = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/weights");
    assert!(
        weights.join("rt_hdr_small.tza").is_file(),
        "shipped Fast weights required"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_oidn-rs"))
        .args([
            "denoise",
            "--device",
            "cpu",
            "--hdr",
            "--quality",
            "fast",
            "--input-scale",
            "1",
            "--weights-dir",
        ])
        .arg(weights)
        .arg("-i")
        .arg(&input)
        .arg("-o")
        .arg(&output_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result = std::fs::read(&output_path).unwrap();
    let header = b"PF\n3 2\n-1.0\n";
    assert!(result.starts_with(header));
    let values: Vec<f32> = result[header.len()..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    assert_eq!(values.len(), 18);
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v > 0.0));
    std::fs::remove_dir_all(directory).unwrap();
}
