//! The TZA writer against the parser and against the bytes Intel's `training/tza.py` writes.
//!
//! Seeded mutant (run once, then revert): `ALIGN = 1` in `src/writer.rs` -> `the_bytes_are_those_tza_py_writes`
//! FAILS (payload at 12 instead of 64), while the parse round trips still pass: the byte oracle is what pins the format.

use std::path::PathBuf;

use oidn_tza::{DType, Layout, Tensor, TensorDesc, TensorMap, TzaError, parse, write};

fn tensor(dims: &[u32], layout: &str, dtype: DType, data: Vec<u8>) -> Tensor {
    Tensor {
        desc: TensorDesc {
            dims: dims.to_vec(),
            layout: Layout::new(layout).unwrap(),
            dtype,
        },
        data: data.into(),
    }
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Intel's two layouts plus the ones our converted networks carry: a linear weight (`oi`) and a rank-0 scalar.
fn sample() -> TensorMap {
    let mut map = TensorMap::new();
    map.insert(
        "conv.weight".into(),
        tensor(&[2, 1, 1, 1], "oihw", DType::Float32, f32s(&[1.0, -2.5])),
    );
    // f16 1.0 = 0x3C00, -2.0 = 0xC000 (little-endian).
    map.insert(
        "conv.bias".into(),
        tensor(&[2], "x", DType::Float16, vec![0x00, 0x3c, 0x00, 0xc0]),
    );
    map.insert(
        "fc.weight".into(),
        tensor(&[1, 3], "oi", DType::Float32, f32s(&[0.5, 0.25, -1.0])),
    );
    map.insert(
        "scale".into(),
        tensor(&[], "", DType::Float32, f32s(&[0.125])),
    );
    map
}

fn assert_same(a: &TensorMap, b: &TensorMap) {
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>()
    );
    for (name, x) in a {
        let y = &b[name];
        assert_eq!(x.desc.dims, y.desc.dims, "{name}");
        assert_eq!(x.desc.layout, y.desc.layout, "{name}");
        assert_eq!(x.desc.dtype, y.desc.dtype, "{name}");
        assert_eq!(x.data, y.data, "{name}");
    }
}

/// A written map parses back to itself, values included. Oracle: the map the test built.
#[test]
fn written_archives_parse_back_to_the_same_map() {
    let map = sample();
    let mut bytes = Vec::new();
    write(&map, &mut bytes).unwrap();
    let back = parse(&bytes).unwrap();
    assert_same(&map, &back);
    assert_eq!(back["conv.bias"].to_f32_vec().unwrap(), [1.0, -2.0]);
    assert_eq!(back["fc.weight"].desc.layout.as_str(), "oi");
    assert_eq!(back["scale"].desc.layout.ndim(), 0);
    assert_eq!(back["scale"].to_f32_vec().unwrap(), [0.125]);
}

/// The bytes of a one-tensor archive equal the ones `training/tza.py` writes for it: header (magic, version 2.0,
/// table offset), zero padding to 64, the payload at 64, zero padding to 128, the table. Oracle: the archive
/// assembled by hand from `tza.py`'s `_write_header` / `_write_pad` / `write` / `_write_table`.
#[test]
fn the_bytes_are_those_tza_py_writes() {
    let mut map = TensorMap::new();
    map.insert("b".into(), tensor(&[1], "x", DType::Float32, f32s(&[1.0])));
    let mut expected = vec![0xd7, 0x41, 2, 0];
    expected.extend_from_slice(&128u64.to_le_bytes());
    expected.resize(64, 0);
    expected.extend_from_slice(&1.0f32.to_le_bytes());
    expected.resize(128, 0);
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.extend_from_slice(&1u16.to_le_bytes());
    expected.push(b'b');
    expected.push(1);
    expected.extend_from_slice(&1u32.to_le_bytes());
    expected.extend_from_slice(b"xf");
    expected.extend_from_slice(&64u64.to_le_bytes());

    // A reused buffer is cleared, not appended to.
    let mut bytes = b"stale".to_vec();
    write(&map, &mut bytes).unwrap();
    assert_eq!(bytes, expected);
}

/// Every shipped Intel archive survives parse -> write -> parse unchanged (names, descriptors, payload bytes).
/// Oracle: the parse of Intel's own file.
#[test]
fn every_shipped_archive_round_trips() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/weights");
    assert!(dir.is_dir(), "required shipped weights directory is absent");
    let mut count = 0usize;
    let mut bytes = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("tza") {
            continue;
        }
        let original = parse(&std::fs::read(&path).unwrap()).unwrap();
        write(&original, &mut bytes).unwrap();
        assert_same(&original, &parse(&bytes).unwrap());
        count += 1;
    }
    assert_eq!(count, 23, "required shipped TZA inventory");
}

/// A payload that does not match its descriptor and an over-long name are refused, and `out` is left untouched.
#[test]
fn invalid_tensors_are_refused_before_writing() {
    let mut map = sample();
    map.insert(
        "short".into(),
        tensor(&[3], "x", DType::Float32, f32s(&[1.0])),
    );
    let mut bytes = b"kept".to_vec();
    assert!(matches!(
        write(&map, &mut bytes),
        Err(TzaError::DataLengthMismatch { expected: 12, got: 4 })
    ));
    assert_eq!(bytes, b"kept");

    let mut map = TensorMap::new();
    map.insert(
        "n".repeat(usize::from(u16::MAX) + 1),
        tensor(&[1], "x", DType::Float32, f32s(&[1.0])),
    );
    assert!(matches!(
        write(&map, &mut bytes),
        Err(TzaError::NameTooLong { len: 65536 })
    ));
    assert_eq!(bytes, b"kept");
}

/// Layouts are the archive's letter strings: Intel's constants, any ASCII-letter name up to `MAX_NDIM`, the empty
/// scalar layout; digits and over-long names are refused.
#[test]
fn layouts_are_the_archive_strings() {
    assert_eq!(Layout::new("oihw").unwrap(), Layout::OIHW);
    assert_eq!(Layout::new("x").unwrap(), Layout::X);
    assert_eq!(Layout::new("iohw").unwrap().ndim(), 4);
    assert_eq!(Layout::new("").unwrap().ndim(), 0);
    assert_eq!(format!("{:?}", Layout::OIHW), "\"oihw\"");
    for bad in ["o1", "oi hw", "abcdefghi"] {
        assert!(
            matches!(Layout::new(bad), Err(TzaError::InvalidLayout { .. })),
            "{bad:?} accepted"
        );
    }
}
