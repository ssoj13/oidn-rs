//! EXR and PNG I/O helpers. Returns flat HWC `f32` buffers.

use std::path::Path;

pub fn load_rgb_f32(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    match path.extension().and_then(|s| s.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("exr") => load_exr(path),
        _ => load_image(path),
    }
}

pub fn save_rgb_f32(path: &Path, pixels: &[f32], w: usize, h: usize) -> Result<(), Box<dyn std::error::Error>> {
    match path.extension().and_then(|s| s.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("exr") => save_exr(path, pixels, w, h),
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

fn save_exr(path: &Path, pixels: &[f32], w: usize, h: usize) -> Result<(), Box<dyn std::error::Error>> {
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

fn save_image(path: &Path, pixels: &[f32], w: usize, h: usize) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(pixels.len(), w * h * 3);
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
