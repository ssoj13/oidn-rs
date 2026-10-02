//! Geometry and memory-layout boundary regressions, including release validation.
use burn::tensor::{Device, Tensor, TensorData};
use half::f16;
use oidn_rs::{Image, ImageMut, PixelFormat, image, image_tensor, tile};

#[test]
fn arbitrary_byte_alignment_and_final_padding_are_supported() {
    let values = [1.25_f32, -2.5, 3.75, 4.5, 5.25, -6.0];
    let mut bytes = vec![0xa5; 1 + 12 + 1 + 12];
    for (index, value) in values.iter().enumerate() {
        let offset = 1 + (index / 3) * 13 + (index % 3) * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    }
    let input = Image::new(&bytes[1..], 1, 2, 13, PixelFormat::Rgb32f).unwrap();
    assert_eq!(input.validate().unwrap(), 25);
    assert_eq!(input.to_rgb_f32().unwrap(), values);
    let mut target = vec![0xa5; bytes.len()];
    let mut output = ImageMut::new(&mut target[1..], 1, 2, 13, PixelFormat::Rgb32f).unwrap();
    output.write_rgb_f32(&values).unwrap();
    assert_eq!(target, bytes);
}

#[test]
fn half_bytes_roundtrip_and_two_channel_broadcast() {
    let values = [f16::from_f32(-1.5), f16::from_f32(2.0)];
    let mut bytes = vec![0xcc; 5];
    bytes[1..3].copy_from_slice(&values[0].to_bits().to_ne_bytes());
    bytes[3..5].copy_from_slice(&values[1].to_bits().to_ne_bytes());
    let input = Image::new(&bytes[1..], 1, 1, 4, PixelFormat::Rg16f).unwrap();
    let rgb = input.to_rgb_f32().unwrap();
    assert_eq!(rgb, [-1.5, 2.0, 2.0]);
    let mut target = vec![0xcc; 5];
    ImageMut::new(&mut target[1..], 1, 1, 4, PixelFormat::Rg16f)
        .unwrap()
        .write_rgb_f32(&rgb)
        .unwrap();
    assert_eq!(target, bytes);
}

#[test]
fn malformed_public_descriptors_return_errors_without_writing() {
    let source = [0_u8; 12];
    let descriptor = Image {
        data: &source,
        width: 2,
        height: 2,
        row_stride: 4,
        format: PixelFormat::R32f,
    };
    assert!(descriptor.to_rgb_f32().is_err());
    assert!(Image::new(&source, 2, 2, 8, PixelFormat::R32f).is_err());
    assert!(Image::new(&source, 1, 3, usize::MAX, PixelFormat::R32f).is_err());
    let mut target = [0xa5; 12];
    let before = target;
    let mut output = ImageMut {
        data: &mut target,
        width: 2,
        height: 2,
        row_stride: 4,
        format: PixelFormat::R32f,
    };
    assert!(output.write_rgb_f32(&[0.0; 12]).is_err());
    assert_eq!(target, before);
}

#[test]
fn source_length_validation_is_transactional() {
    let mut target = [0xa5; 12];
    ImageMut::new(&mut target, 1, 1, 12, PixelFormat::Rgb32f)
        .unwrap()
        .write_rgb_f32(&[1.0, 2.0])
        .unwrap_err();
    assert_eq!(target, [0xa5; 12]);
}

#[test]
fn checked_geometry_handles_empty_and_oversized_inputs() {
    assert_eq!(image::validate_dimensions(0, 4, 3).unwrap(), 0);
    assert!(image::validate_dimensions(usize::MAX, 2, 3).is_err());
    assert!(image::validate_dimensions(65_536, 65_536, 3).is_err());
    assert!(image::validate_dimensions(1, 1, 0).is_err());
    let input = Image::new(&[], 0, 4, 0, PixelFormat::Rgb32f).unwrap();
    assert!(input.to_rgb_f32().unwrap().is_empty());
}

#[test]
fn typed_constructor_checks_are_present_in_release() {
    assert!(std::panic::catch_unwind(|| Image::from_rgb_f32(&[1.0], 1, 1)).is_err());
}

#[test]
fn layout_helpers_reject_short_excess_and_overflowing_buffers() {
    for values in [&[1.0][..], &[1.0, 2.0, 3.0, 4.0][..]] {
        assert!(image_tensor::hwc_to_chw(values, 3, 1, 1).is_err());
        assert!(image_tensor::chw_to_hwc(values, 3, 1, 1).is_err());
    }
    assert!(image_tensor::hwc_to_chw(&[], usize::MAX, usize::MAX, 2).is_err());
    assert!(image_tensor::chw_to_hwc(&[], 0, 1, 1).is_err());
}

#[test]
fn tensor_helpers_reject_batch_and_length_mismatch() {
    let device = Device::ndarray();
    assert!(image_tensor::chw_vec_to_tensor(vec![1.0], 3, 1, 1, &device).is_err());
    let batch = Tensor::<4>::from_data(TensorData::new(vec![1.0; 6], [2, 3, 1, 1]), &device);
    assert!(image_tensor::tensor_to_chw_vec(batch).is_err());
}

#[test]
fn planner_rejects_invalid_parameters_and_oversized_geometry() {
    for (w, h, rf, align, pixels) in [
        (0, 1, 174, 16, 1000),
        (-1, 1, 174, 16, 1000),
        (1, 1, 174, 0, 1000),
        (1, 1, 174, 17, 1000),
        (1, 1, 0, 16, 1000),
        (1, 1, 174, 16, 0),
        (i32::MAX, 1, 174, 16, 1000),
        (65_536, 65_536, 174, 16, i32::MAX),
    ] {
        assert!(tile::plan(w, h, rf, align, pixels).is_err());
    }
}

#[test]
fn planned_rectangles_cover_each_pixel_once_with_valid_crops() {
    for (w, h) in [(1, 1), (17, 31), (801, 817), (1601, 801)] {
        for rf in [tile::RECEPTIVE_FIELD_BASE, tile::RECEPTIVE_FIELD_LARGE] {
            for alignment in [16, 32, 64] {
                let plan = tile::plan(w, h, rf, alignment, 768 * 768).unwrap();
                plan.validate(w as usize, h as usize).unwrap();
                let mut coverage = vec![0_u8; (w * h) as usize];
                assert_eq!(plan.overlap % alignment, 0);
                assert!(plan.overlap * 2 >= rf);
                for job in plan.jobs {
                    let input = job.input;
                    let src = job.output_src_in_tile;
                    let dst = job.output_dst;
                    assert!(
                        input.x >= 0
                            && input.y >= 0
                            && input.x + input.w <= w
                            && input.y + input.h <= h
                    );
                    assert!(
                        src.x >= 0
                            && src.y >= 0
                            && src.x + src.w <= plan.tile_w
                            && src.y + src.h <= plan.tile_h
                    );
                    assert_eq!((src.w, src.h), (dst.w, dst.h));
                    assert!(dst.x >= 0 && dst.y >= 0 && dst.x + dst.w <= w && dst.y + dst.h <= h);
                    for y in dst.y..dst.y + dst.h {
                        for x in dst.x..dst.x + dst.w {
                            coverage[(y * w + x) as usize] += 1;
                        }
                    }
                }
                assert!(coverage.iter().all(|&count| count == 1));
            }
        }
    }
}

#[test]
fn public_plans_reject_overlap_gaps_and_overflowing_rectangles() {
    let plan = tile::plan(1601, 801, 174, 16, 768 * 768).unwrap();
    let mut invalid_area = plan.clone();
    invalid_area.jobs[0].output_dst.w = -1;
    assert!(tile::total_output_pixels(&invalid_area).is_err());
    let mut overflowing_area = plan.clone();
    let mut huge = plan.jobs[0];
    huge.output_dst.w = i32::MAX;
    huge.output_dst.h = i32::MAX;
    overflowing_area.jobs = vec![huge; 3];
    assert!(tile::total_output_pixels(&overflowing_area).is_err());
    let mut overlapping = plan.clone();
    overlapping.jobs.push(plan.jobs[0]);
    assert!(overlapping.validate(1601, 801).is_err());
    let mut gap = plan.clone();
    gap.jobs.remove(0);
    assert!(gap.validate(1601, 801).is_err());
    let mut overflow = plan.clone();
    overflow.jobs[0].input.x = i32::MAX;
    overflow.jobs[0].input.w = i32::MAX;
    assert!(overflow.validate(1601, 801).is_err());
    let mut padding = plan.clone();
    padding.jobs[0].align_offset_x = i32::MAX;
    assert!(padding.validate(1601, 801).is_err());
    let mut crop = plan.clone();
    crop.jobs[0].output_src_in_tile.x += 1;
    assert!(crop.validate(1601, 801).is_err());
}
