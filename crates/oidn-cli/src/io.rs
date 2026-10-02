//! Image I/O helpers — EXR, HDR, TIFF (float), PFM (float32) and PHM (float16).
//!
//! All paths converge on a flat HWC `f32` RGB buffer. Quantisation to 8-bit
//! is deliberately gated to the LDR file extensions (`png`/`jpg`/`jpeg`/`bmp`)
//! so callers writing HDR EXRs or float TIFFs never lose dynamic range.

use crate::support::{Error, samples};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;

/// Packed RGB samples and their positive dimensions.
pub type RgbImage = (Vec<f32>, usize, usize);

/// Required buffer encoding; normal/data images never undergo color conversion.
#[derive(Clone, Copy, Debug)]
pub enum Encoding {
    Linear,
    Srgb,
    Data,
}

fn encoded(path: &Path) -> bool {
    !matches!(
        path.extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("exr" | "pfm" | "phm" | "hdr")
    )
}

pub fn load_rgb_f32(path: &Path, encoding: Encoding) -> Result<RgbImage, Error> {
    let (mut pixels, w, h) = match path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("exr") => load_exr(path),
        Some("pfm") => load_pfm(path, false),
        Some("phm") => load_pfm(path, true),
        _ => load_image(path),
    }?;
    if pixels.len() != samples(w, h, 3)? {
        return Err("invalid RGB image layout".into());
    }
    if matches!(encoding, Encoding::Linear) && encoded(path) {
        pixels
            .iter_mut()
            .for_each(|v| *v = oidn_rs::color::srgb_inverse(*v));
    }
    Ok((pixels, w, h))
}

pub fn save_rgb_f32(
    path: &Path,
    pixels: &[f32],
    w: usize,
    h: usize,
    encoding: Encoding,
) -> Result<(), Error> {
    if pixels.len() != samples(w, h, 3)? {
        return Err("invalid RGB image layout".into());
    }
    u32::try_from(w)?;
    u32::try_from(h)?;
    let converted;
    let pixels = if matches!(encoding, Encoding::Linear) && encoded(path) {
        converted = pixels
            .iter()
            .copied()
            .map(oidn_rs::color::srgb_forward)
            .collect::<Vec<_>>();
        &converted
    } else {
        pixels
    };
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("exr") => save_exr(path, pixels, w, h),
        Some("pfm") => save_pfm(path, pixels, w, h, false),
        Some("phm") => save_pfm(path, pixels, w, h, true),
        _ => save_image(path, pixels, w, h),
    }
}

fn load_exr(path: &Path) -> Result<RgbImage, Error> {
    use exr::prelude::*;

    let img = read_first_rgba_layer_from_file(
        path,
        |resolution, _channels: &RgbaChannels| {
            let pixels: Vec<(f32, f32, f32, f32)> =
                vec![(0.0, 0.0, 0.0, 1.0); resolution.width() * resolution.height()];
            (pixels, resolution.width(), resolution.height())
        },
        |(pixels, w, _h), pos, (r, g, b, a): (f32, f32, f32, f32)| {
            pixels[pos.y() * *w + pos.x()] = (r, g, b, a);
        },
    )?;

    let (pixels, w, h) = img.layer_data.channel_data.pixels;
    let mut flat = Vec::with_capacity(w * h * 3);
    for (r, g, b, _a) in pixels {
        flat.push(r);
        flat.push(g);
        flat.push(b);
    }
    Ok((flat, w, h))
}

fn save_exr(
    path: &Path,
    pixels: &[f32],
    w: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use exr::prelude::*;
    debug_assert_eq!(pixels.len(), w * h * 3);
    write_rgb_file(path, w, h, |x, y| {
        let idx = (y * w + x) * 3;
        (pixels[idx], pixels[idx + 1], pixels[idx + 2])
    })?;
    Ok(())
}

fn load_image(path: &Path) -> Result<RgbImage, Error> {
    let img = image::open(path)?.to_rgb32f();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((img.into_raw(), w, h))
}

/// HDR-preserving image save. Branches on extension; refuses to silently
/// quantise float pixels into an 8-bit container.
fn save_image(
    path: &Path,
    pixels: &[f32],
    w: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(pixels.len(), w * h * 3);
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("png") | Some("jpg") | Some("jpeg") | Some("bmp") => {
            // LDR quantisation path — explicit 8-bit destination.
            let mut buf = image::Rgb32FImage::new(w as u32, h as u32);
            for (i, px) in pixels.as_chunks::<3>().0.iter().enumerate() {
                let x = (i % w) as u32;
                let y = (i / w) as u32;
                buf.put_pixel(x, y, image::Rgb([px[0], px[1], px[2]]));
            }
            let buf8 = image::DynamicImage::ImageRgb32F(buf).to_rgb8();
            buf8.save(path)?;
            Ok(())
        }
        Some("hdr") => {
            // Radiance .hdr — RGBE encoder takes Rgb<f32>.
            use image::codecs::hdr::HdrEncoder;
            let mut data = Vec::with_capacity(w * h);
            for px in pixels.as_chunks::<3>().0 {
                data.push(image::Rgb([px[0], px[1], px[2]]));
            }
            let f = File::create(path)?;
            let enc = HdrEncoder::new(BufWriter::new(f));
            enc.encode(&data, w, h)?;
            Ok(())
        }
        Some("tif") | Some("tiff") => {
            // TIFF supports float samples directly via the `image` codec.
            let mut buf = image::Rgb32FImage::new(w as u32, h as u32);
            for (i, px) in pixels.as_chunks::<3>().0.iter().enumerate() {
                let x = (i % w) as u32;
                let y = (i / w) as u32;
                buf.put_pixel(x, y, image::Rgb([px[0], px[1], px[2]]));
            }
            buf.save(path)?;
            Ok(())
        }
        other => Err(format!(
            "unsupported output extension {:?}; supported: exr, pfm, phm, hdr, tif/tiff, png, jpg/jpeg, bmp",
            other,
        )
        .into()),
    }
}

// --------------------------------------------------------------------------
// PFM / PHM
//
// Header:
//   PF\n            (RGB float32, PFM) or  Pf\n  (grayscale, not supported here)
//   PH\n            (RGB float16, PHM) or  Ph\n  (grayscale, not supported here)
//   <W> <H>\n
//   <scale>\n       (negative → little-endian, positive → big-endian)
//   <raw float pixels, bottom-to-top, row-major, RGB triplets>
//
// OIDN writes negative scale (little-endian). The on-disk pixel order is
// bottom-to-top — we flip rows on load/save so the in-memory buffer is
// always top-to-bottom (matches every other loader in this CLI).
// --------------------------------------------------------------------------

fn load_pfm(path: &Path, half: bool) -> Result<RgbImage, Error> {
    let mut reader = BufReader::new(File::open(path)?);
    let (w, h, scale) = read_pfm_header(&mut reader, half)?;
    let count = samples(w, h, 3)?;
    let width = if half { 2 } else { 4 };
    let size = count
        .checked_mul(width)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or("PFM/PHM payload size overflow")?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(size)?;
    raw.resize(size, 0);
    reader.read_exact(&mut raw)?;
    let mut flat = Vec::new();
    flat.try_reserve_exact(count)?;
    flat.resize(count, 0.0);
    for y in 0..h {
        let src = (h - 1 - y) * w * 3 * width;
        let dst = y * w * 3;
        for x in 0..w * 3 {
            let o = src + x * width;
            let value = if half {
                let bytes = [raw[o], raw[o + 1]];
                let v = if scale < 0.0 {
                    f16::from_le_bytes(bytes)
                } else {
                    f16::from_be_bytes(bytes)
                };
                v.to_f32()
            } else {
                let bytes = [raw[o], raw[o + 1], raw[o + 2], raw[o + 3]];
                if scale < 0.0 {
                    f32::from_le_bytes(bytes)
                } else {
                    f32::from_be_bytes(bytes)
                }
            };
            flat[dst + x] = value * scale.abs();
        }
    }
    Ok((flat, w, h))
}

fn save_pfm(path: &Path, pixels: &[f32], w: usize, h: usize, half: bool) -> Result<(), Error> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(if half { b"PH\n" } else { b"PF\n" })?;
    writer.write_all(format!("{w} {h}\n-1.0\n").as_bytes())?;
    for y in 0..h {
        let src = (h - 1 - y) * w * 3;
        for &value in &pixels[src..src + w * 3] {
            if half {
                writer.write_all(&f16::from_f32(value).to_le_bytes())?;
            } else {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

/// Parse a validated RGB header, consuming exactly the final LF or CRLF.
/// Binary payload bytes, including whitespace, are never skipped.
fn read_pfm_header<R: Read>(reader: &mut R, half: bool) -> Result<(usize, usize, f32), Error> {
    let magic = read_token(reader, false)?;
    if magic != if half { "PH" } else { "PF" } {
        return Err(format!(
            "expected RGB {} magic, found {magic}",
            if half { "PHM (PH)" } else { "PFM (PF)" }
        )
        .into());
    }
    let w = read_token(reader, false)?.parse()?;
    let h = read_token(reader, false)?.parse()?;
    samples(w, h, 3)?;
    let scale: f32 = read_token(reader, true)?.parse()?;
    if !scale.is_finite() || scale == 0.0 {
        return Err("PFM/PHM scale must be finite and nonzero".into());
    }
    Ok((w, h, scale))
}

fn read_token<R: Read>(reader: &mut R, final_token: bool) -> Result<String, Error> {
    let mut token = Vec::new();
    let mut byte = [0_u8];
    for skipped in 0..=64 {
        reader.read_exact(&mut byte)?;
        if !byte[0].is_ascii_whitespace() {
            token.push(byte[0]);
            break;
        }
        if skipped == 64 {
            return Err("PFM/PHM header whitespace is too long".into());
        }
    }
    loop {
        reader.read_exact(&mut byte)?;
        if byte[0].is_ascii_whitespace() {
            break;
        }
        if token.len() >= 64 {
            return Err("PFM/PHM header token is too long".into());
        }
        token.push(byte[0]);
    }
    if final_token {
        let mut spaces = 0;
        while byte[0] != b'\n' {
            if byte[0] == b'\r' {
                reader.read_exact(&mut byte)?;
                if byte[0] != b'\n' {
                    return Err("PFM/PHM header requires LF or CRLF".into());
                }
                break;
            }
            if !matches!(byte[0], b' ' | b'\t') || spaces >= 64 {
                return Err("invalid PFM/PHM scale line terminator".into());
            }
            spaces += 1;
            reader.read_exact(&mut byte)?;
        }
    }
    Ok(String::from_utf8(token)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new(ext: &str) -> Self {
            Self(std::env::temp_dir().join(format!(
                "oidn-cli-{}-{}.{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed),
                ext
            )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    #[test]
    fn pfm_phm_endian_scale_flip_and_crlf_preserve_payload() {
        for half in [false, true] {
            for little in [false, true] {
                let temp = Temp::new(if half { "phm" } else { "pfm" });
                let scale = if little { -2.0 } else { 2.0 };
                let magic = if half { "PH" } else { "PF" };
                let mut bytes = format!("{magic}\r\n1 2\r\n{scale}\r\n").into_bytes();
                let first = if half {
                    f16::from_bits(0x380a).to_f32()
                } else {
                    f32::from_bits(0x3f00000a)
                };
                let disk = [first, 0.25, 0.75, 1.0, 1.5, 2.0];
                for value in disk {
                    if half {
                        let v = f16::from_f32(value);
                        bytes.extend_from_slice(&if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        });
                    } else {
                        bytes.extend_from_slice(&if little {
                            value.to_le_bytes()
                        } else {
                            value.to_be_bytes()
                        });
                    }
                }
                std::fs::write(&temp.0, bytes).unwrap();
                let (pixels, w, h) = load_rgb_f32(&temp.0, Encoding::Data).unwrap();
                assert_eq!((w, h), (1, 2));
                assert_eq!(pixels, [2.0, 3.0, 4.0, first * 2.0, 0.5, 1.5]);
            }
        }
    }
    #[test]
    fn invalid_float_headers_and_truncated_payload_are_errors() {
        for header in [
            "PH\n1 1\n-1\n",
            "Pf\n1 1\n-1\n",
            "PF\n0 1\n-1\n",
            "PF\n1 1\n0\n",
            "PF\n1 1\nNaN\n",
            "PF\n1 1\ninf\n",
            "PF\n1 1\n-1",
            "PF\n1 1\n-1\rX",
        ] {
            assert!(
                read_pfm_header(&mut header.as_bytes(), false).is_err(),
                "{header:?}"
            );
        }
        assert!(read_pfm_header(&mut b"PF\n1 1\n-1\n".as_slice(), true).is_err());
        let padded = format!("{}PF\n1 1\n-1\n", " ".repeat(65));
        assert!(read_pfm_header(&mut padded.as_bytes(), false).is_err());
        let header = format!("PF\n{} 2\n-1\n", usize::MAX);
        assert!(read_pfm_header(&mut header.as_bytes(), false).is_err());
        let temp = Temp::new("pfm");
        std::fs::write(&temp.0, b"PF\n1 1\n-1\n\x00").unwrap();
        assert!(load_rgb_f32(&temp.0, Encoding::Linear).is_err());
    }
    #[test]
    fn file_encoding_is_role_aware_and_hdr_preserves_range() {
        let temp = Temp::new("png");
        save_rgb_f32(&temp.0, &[0.18, 0.5, 1.0], 1, 1, Encoding::Linear).unwrap();
        let encoded_pixels = load_rgb_f32(&temp.0, Encoding::Srgb).unwrap().0;
        assert!((encoded_pixels[0] - 0.4613561).abs() < 0.005);
        let linear = load_rgb_f32(&temp.0, Encoding::Linear).unwrap().0;
        assert!((linear[0] - 0.18).abs() < 0.005);
        assert_eq!(
            load_rgb_f32(&temp.0, Encoding::Data).unwrap().0,
            encoded_pixels
        );
        let temp = Temp::new("pfm");
        save_rgb_f32(&temp.0, &[-0.5, 2.0, 100.0], 1, 1, Encoding::Linear).unwrap();
        assert_eq!(
            load_rgb_f32(&temp.0, Encoding::Linear).unwrap().0,
            [-0.5, 2.0, 100.0]
        );
        assert!(save_rgb_f32(&temp.0, &[1.0], 1, 1, Encoding::Linear).is_err());
    }
}
