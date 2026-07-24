//! Image I/O helpers — EXR, HDR, TIFF (float), PFM (float32) and PHM (float16).
//!
//! All paths converge on a flat HWC `f32` RGB buffer. Quantisation to 8-bit
//! is deliberately gated to the LDR file extensions (`png`/`jpg`/`jpeg`/`bmp`)
//! so callers writing HDR EXRs or float TIFFs never lose dynamic range.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use half::f16;

pub fn load_rgb_f32(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("exr") => load_exr(path),
        Some("pfm") => load_pfm(path),
        Some("phm") => load_phm(path),
        _ => load_image(path),
    }
}

pub fn save_rgb_f32(
    path: &Path,
    pixels: &[f32],
    w: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("exr") => save_exr(path, pixels, w, h),
        Some("pfm") => save_pfm(path, pixels, w, h),
        Some("phm") => save_phm(path, pixels, w, h),
        _ => save_image(path, pixels, w, h),
    }
}

fn load_exr(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
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

fn load_image(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
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
            for (i, px) in pixels.chunks_exact(3).enumerate() {
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
            for px in pixels.chunks_exact(3) {
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
            for (i, px) in pixels.chunks_exact(3).enumerate() {
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

fn load_pfm(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let (channels, w, h, little_endian) = read_pfm_header(&mut reader)?;
    if channels != 3 {
        return Err("only 3-channel PFM (`PF`) is supported".into());
    }
    let mut raw = vec![0u8; w * h * 3 * 4];
    reader.read_exact(&mut raw)?;
    let mut flat = vec![0.0f32; w * h * 3];
    for y in 0..h {
        // Flip vertically: row `y` in memory ↔ row `h-1-y` on disk.
        let src = (h - 1 - y) * w * 3 * 4;
        let dst = y * w * 3;
        for x in 0..(w * 3) {
            let o = src + x * 4;
            let bytes = [raw[o], raw[o + 1], raw[o + 2], raw[o + 3]];
            flat[dst + x] = if little_endian {
                f32::from_le_bytes(bytes)
            } else {
                f32::from_be_bytes(bytes)
            };
        }
    }
    Ok((flat, w, h))
}

fn save_pfm(
    path: &Path,
    pixels: &[f32],
    w: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(pixels.len(), w * h * 3);
    let mut writer = BufWriter::new(File::create(path)?);
    // Negative scale → little-endian, matches reference OIDN writer.
    writer.write_all(b"PF\n")?;
    writer.write_all(format!("{w} {h}\n").as_bytes())?;
    writer.write_all(b"-1.0\n")?;
    for y in 0..h {
        let src = (h - 1 - y) * w * 3;
        for x in 0..(w * 3) {
            writer.write_all(&pixels[src + x].to_le_bytes())?;
        }
    }
    Ok(())
}

fn load_phm(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let (channels, w, h, little_endian) = read_pfm_header(&mut reader)?;
    if channels != 3 {
        return Err("only 3-channel PHM (`PH`) is supported".into());
    }
    let mut raw = vec![0u8; w * h * 3 * 2];
    reader.read_exact(&mut raw)?;
    let mut flat = vec![0.0f32; w * h * 3];
    for y in 0..h {
        let src = (h - 1 - y) * w * 3 * 2;
        let dst = y * w * 3;
        for x in 0..(w * 3) {
            let o = src + x * 2;
            let bytes = [raw[o], raw[o + 1]];
            let v = if little_endian {
                f16::from_le_bytes(bytes)
            } else {
                f16::from_be_bytes(bytes)
            };
            flat[dst + x] = v.to_f32();
        }
    }
    Ok((flat, w, h))
}

fn save_phm(
    path: &Path,
    pixels: &[f32],
    w: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(pixels.len(), w * h * 3);
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"PH\n")?;
    writer.write_all(format!("{w} {h}\n").as_bytes())?;
    writer.write_all(b"-1.0\n")?;
    for y in 0..h {
        let src = (h - 1 - y) * w * 3;
        for x in 0..(w * 3) {
            let v = f16::from_f32(pixels[src + x]);
            writer.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Parse `PF\n<W> <H>\n<scale>\n` from `reader`. Returns
/// `(channels, width, height, little_endian)`. After this call the reader
/// is positioned at the first byte of pixel data.
fn read_pfm_header<R: Read>(
    reader: &mut R,
) -> Result<(usize, usize, usize, bool), Box<dyn std::error::Error>> {
    let magic = read_token(reader)?;
    // PF/Pf → 32-bit float (PFM); PH/Ph → 16-bit half (PHM). Lowercase variants
    // are 1-channel grayscale; the loaders above reject them and only accept
    // their 3-channel uppercase counterparts.
    let channels = match magic.as_str() {
        "PF" | "PH" => 3,
        "Pf" | "Ph" => 1,
        other => return Err(format!("unknown PFM/PHM magic `{other}`").into()),
    };
    let w: usize = read_token(reader)?.parse()?;
    let h: usize = read_token(reader)?.parse()?;
    let scale: f32 = read_token(reader)?.parse()?;
    let little_endian = scale < 0.0;
    Ok((channels, w, h, little_endian))
}

/// Read one whitespace-terminated ASCII token from the stream.
/// PFM headers are line-based but tolerate any whitespace as separator —
/// the closing newline of the scale line is consumed here so that the
/// next byte is the first pixel byte.
fn read_token<R: Read>(reader: &mut R) -> Result<String, Box<dyn std::error::Error>> {
    let mut s = String::new();
    let mut b = [0u8; 1];
    // skip leading whitespace
    loop {
        if reader.read(&mut b)? == 0 {
            return Err("unexpected EOF in PFM/PHM header".into());
        }
        if !b[0].is_ascii_whitespace() {
            s.push(b[0] as char);
            break;
        }
    }
    loop {
        if reader.read(&mut b)? == 0 {
            break;
        }
        if b[0].is_ascii_whitespace() {
            break;
        }
        s.push(b[0] as char);
    }
    Ok(s)
}
