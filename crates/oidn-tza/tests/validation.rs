use oidn_tza::{DType, Layout, Tensor, TensorDesc, TzaError, parse};

fn archive(names: &[&str], dims: &[u32], dtype: u8, payload: &[u8]) -> Vec<u8> {
    let table = 12 + payload.len();
    let mut bytes = vec![0xd7, 0x41, 2, 0];
    bytes.extend_from_slice(&(table as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&(names.len() as u32).to_le_bytes());
    for name in names {
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(dims.len() as u8);
        for dim in dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(if dims.len() == 1 { b"x" } else { b"oihw" });
        bytes.push(dtype);
        bytes.extend_from_slice(&12u64.to_le_bytes());
    }
    bytes
}

#[test]
fn malformed_shapes_offsets_and_duplicates_return_errors() {
    assert!(matches!(
        parse(&archive(&["a", "a"], &[1], b'f', &[0; 4])),
        Err(TzaError::DuplicateName(_))
    ));
    assert!(matches!(
        parse(&archive(&["a"], &[0], b'f', &[])),
        Err(TzaError::InvalidDimension { .. })
    ));
    assert!(matches!(
        parse(&archive(&["a"], &[u32::MAX; 4], b'h', &[])),
        Err(TzaError::SizeOverflow)
    ));
    let mut huge_offset = archive(&["a"], &[1], b'f', &[0; 4]);
    huge_offset[4..12].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(parse(&huge_offset).is_err());
    let mut huge_data_offset = archive(&["a"], &[1], b'f', &[0; 4]);
    let end = huge_data_offset.len();
    huge_data_offset[end - 8..end].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(parse(&huge_data_offset).is_err());
}

#[test]
fn every_truncation_of_a_valid_archive_is_an_error() {
    let bytes = archive(&["a"], &[2], b'h', &[0, 0x3c, 0, 0xc0]);
    for end in 0..bytes.len() {
        assert!(parse(&bytes[..end]).is_err(), "accepted prefix {end}");
    }
    assert_eq!(
        parse(&bytes).unwrap()["a"].to_f32_vec().unwrap(),
        [1.0, -2.0]
    );
}

#[test]
fn aliased_payloads_share_owned_storage_and_allow_trailing_bytes() {
    let mut bytes = archive(&["a", "b", "c"], &[2], b'h', &[0, 0x3c, 0, 0xc0]);
    bytes.extend_from_slice(b"valid unused trailer");
    let map = parse(&bytes).unwrap();
    assert_eq!(map["a"].data.as_ptr(), map["b"].data.as_ptr());
    assert_eq!(map["a"].data.as_ptr(), map["c"].data.as_ptr());
    let retained = map["a"].clone();
    drop(map);
    drop(bytes);
    assert_eq!(retained.to_f32_vec().unwrap(), [1.0, -2.0]);

    // Distinct ranges can overlap too, without duplicating their backing bytes.
    let mut bytes = archive(&["a", "b"], &[1], b'h', &[0, 0x3c, 0, 0xc0]);
    let end = bytes.len();
    bytes[end - 8..].copy_from_slice(&14u64.to_le_bytes());
    let map = parse(&bytes).unwrap();
    assert_eq!(
        map["b"].data.as_ptr() as usize,
        map["a"].data.as_ptr() as usize + 2
    );
    assert_eq!(map["b"].to_f32_vec().unwrap(), [-2.0]);
}

#[test]
fn public_tensor_payloads_are_fallible_and_little_endian() {
    let desc = TensorDesc {
        dims: vec![2],
        layout: Layout::X,
        dtype: DType::Float32,
    };
    let tensor = Tensor {
        desc: desc.clone(),
        data: [1.0f32.to_le_bytes(), (-2.5f32).to_le_bytes()]
            .concat()
            .into(),
    };
    assert_eq!(tensor.to_f32_vec().unwrap(), [1.0, -2.5]);
    let mut values = tensor.iter_f32().unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(values.next(), Some(1.0));
    assert_eq!(values.len(), 1);
    assert_eq!(values.next(), Some(-2.5));
    assert_eq!(values.next(), None);
    let mut backing = vec![0u8; 12];
    let offset = (0..4)
        .find(|&n| !(backing.as_ptr() as usize + n).is_multiple_of(4))
        .unwrap();
    backing[offset..offset + 8].copy_from_slice(&tensor.data);
    let unaligned = Tensor {
        desc: desc.clone(),
        data: oidn_tza::Bytes::from(backing).slice(offset..offset + 8),
    };
    assert!(unaligned.as_f32().is_none());
    assert_eq!(unaligned.to_f32_vec().unwrap(), [1.0, -2.5]);
    for size in [0, 1, 4, 9] {
        let malformed = Tensor {
            desc: desc.clone(),
            data: vec![0; size].into(),
        };
        assert!(matches!(
            malformed.to_f32_vec(),
            Err(TzaError::DataLengthMismatch { .. })
        ));
        assert!(malformed.as_f32().is_none());
        assert!(malformed.iter_f32().is_err());
    }
    let wrong_rank = TensorDesc {
        dims: vec![1, 1],
        ..desc
    };
    assert!(matches!(
        wrong_rank.byte_size(),
        Err(TzaError::LayoutNdimMismatch { .. })
    ));
}
