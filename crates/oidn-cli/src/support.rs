//! Shared deterministic fixtures and finite quality metrics for CLI and benchmarks.
pub type Error = Box<dyn std::error::Error>;

pub fn samples(w: usize, h: usize, channels: usize) -> Result<usize, Error> {
    if w == 0 || h == 0 || channels == 0 {
        return Err("image dimensions and channels must be positive".into());
    }
    w.checked_mul(h)
        .and_then(|n| n.checked_mul(channels))
        .filter(|n| *n <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or_else(|| "image sample count overflows addressable storage".into())
}

pub fn resolution(value: &str) -> Result<(usize, usize), Error> {
    let (w, h) = value
        .split_once('x')
        .ok_or("resolution must be WIDTHxHEIGHT")?;
    let (w, h) = (w.parse()?, h.parse()?);
    samples(w, h, 3)?;
    Ok((w, h))
}

pub struct Metrics {
    pub mse: f64,
    pub rmse: f64,
    pub max_error: f64,
}

impl Metrics {
    /// PSNR with an explicit finite positive peak (one for the synthetic fixture).
    pub fn psnr(&self, peak: f64) -> Result<f64, Error> {
        if !peak.is_finite() || peak <= 0.0 {
            return Err("PSNR peak must be finite and positive".into());
        }
        Ok(if self.mse == 0.0 {
            f64::INFINITY
        } else {
            20.0 * (peak / self.rmse).log10()
        })
    }
}

pub fn metrics(actual: &[f32], expected: &[f32]) -> Result<Metrics, Error> {
    if actual.is_empty() || actual.len() != expected.len() {
        return Err("quality metrics require equal nonempty sample buffers".into());
    }
    let mut sum = 0.0_f64;
    let mut max_error = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() {
            return Err("quality metrics reject nonfinite input or output samples".into());
        }
        let d = f64::from(a) - f64::from(b);
        sum += d * d;
        max_error = max_error.max(d.abs());
    }
    let mse = sum / actual.len() as f64;
    Ok(Metrics {
        mse,
        rmse: mse.sqrt(),
        max_error,
    })
}

pub fn make_clean(w: usize, h: usize) -> Result<Vec<f32>, Error> {
    let n = samples(w, h, 3)?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(n)?;
    buf.resize(n, 0.0);
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let rmax = (cx * cx + cy * cy).sqrt();
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt() / rmax;
            let v = 0.7 + 0.25 * (1.0 - r);
            let i = (y * w + x) * 3;
            buf[i..i + 3].copy_from_slice(&[v, v * 0.9, v * 0.7]);
        }
    }
    Ok(buf)
}

pub fn add_noise(clean: &[f32], magnitude: f32) -> Result<Vec<f32>, Error> {
    if !magnitude.is_finite() || magnitude < 0.0 {
        return Err("noise magnitude must be finite and nonnegative".into());
    }
    let mut out = Vec::new();
    out.try_reserve_exact(clean.len())?;
    out.extend_from_slice(clean);
    for (i, v) in out.iter_mut().enumerate() {
        let mut n = (i as u32).wrapping_mul(2654435761);
        n ^= n >> 13;
        n = n.wrapping_mul(0x85ebca6b);
        n ^= n >> 16;
        *v += ((n as f32 / u32::MAX as f32) * 2.0 - 1.0) * magnitude;
    }
    Ok(out)
}

pub fn make_normal(w: usize, h: usize) -> Result<Vec<f32>, Error> {
    let n = samples(w, h, 3)?;
    let mut buf = Vec::new();
    buf.try_reserve_exact(n)?;
    buf.resize(n, 0.0);
    for px in buf.as_chunks_mut::<3>().0 {
        px[1] = 1.0;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metrics_reject_invalid_samples_and_preserve_f64_range() {
        assert!(metrics(&[f32::NAN], &[0.0]).is_err());
        assert!(metrics(&[0.0], &[f32::INFINITY]).is_err());
        assert!(metrics(&[], &[]).is_err());
        assert!(metrics(&[0.0], &[0.0, 1.0]).is_err());
        let m = metrics(&[f32::MAX], &[-f32::MAX]).unwrap();
        assert!(m.mse.is_finite() && m.mse > f64::from(f32::MAX));
        assert!(
            metrics(&[1.0], &[1.0])
                .unwrap()
                .psnr(1.0)
                .unwrap()
                .is_infinite()
        );
    }
    #[test]
    fn checked_fixture_geometry_and_repeatability() {
        assert!(resolution("0x16").is_err());
        assert!(samples(usize::MAX, 2, 3).is_err());
        let clean = make_clean(3, 2).unwrap();
        assert_eq!(
            add_noise(&clean, 0.1).unwrap(),
            add_noise(&clean, 0.1).unwrap()
        );
        assert_eq!(make_normal(1, 1).unwrap(), [0.0, 1.0, 0.0]);
    }
}
