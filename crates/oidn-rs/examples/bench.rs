//! OIDN bench — runs the full RT filter through a grid of
//! `(resolution × mode × quality)` and writes latency + RMSE/PSNR
//! statistics to a CSV.
//!
//! Builds against the Wgpu backend so the numbers reflect the real
//! production path (Burn-wgpu through cubecl on whatever GPU
//! `wgpu::Instance::default()` picks). For each combination we:
//!
//! 1. Generate a deterministic synthetic HDR image — a smooth radial
//!    gradient plus hash-noise (`add_noise`), shared with the CLI bench.
//! 2. Build the filter, set inputs, allocate output, commit.
//! 3. Run `--warmup` iterations to amortise model load + tile-plan
//!    compute + first-touch GPU allocations.
//! 4. Time `--iters` iterations end-to-end through `filter.execute()`
//!    and `filter.take_output()`. The take is included because that's
//!    what the production loop pays.
//! 5. Compute RMSE vs the clean reference, then PSNR =
//!    `20 * log10(1.0 / rmse)` (peak signal is 1.0 for our gradient).
//!
//! Usage:
//!
//! ```
//! cargo run --release --example bench -- \
//!     --weights-dir data/weights \
//!     --output bench-2026-05-15.csv
//! ```
//!
//! See `parse_args` below for the full flag set.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use oidn_rs::prelude::wgpu_prelude::*;
use oidn_rs::prelude::*;
#[path = "../../oidn-cli/src/support.rs"]
pub mod support;
use support::{add_noise, make_clean, make_normal, metrics};

// ---------------------- CLI ----------------------

struct Cfg {
    weights_dir: PathBuf,
    resolutions: Vec<(usize, usize)>,
    modes: Vec<Mode>,
    qualities: Vec<Quality>,
    iters: usize,
    warmup: usize,
    output: PathBuf,
    noise_magnitude: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Color,
    ColorAlbedo,
    ColorAlbedoNormal,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Color => "color",
            Self::ColorAlbedo => "color_albedo",
            Self::ColorAlbedoNormal => "color_albedo_normal",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        match s {
            "color" | "c" => Some(Self::Color),
            "color_albedo" | "ca" => Some(Self::ColorAlbedo),
            "color_albedo_normal" | "can" => Some(Self::ColorAlbedoNormal),
            _ => None,
        }
    }
}

fn quality_str(q: Quality) -> &'static str {
    match q {
        Quality::Fast => "fast",
        Quality::Balanced => "balanced",
        Quality::High => "high",
    }
}

fn parse_quality(s: &str) -> Option<Quality> {
    match s {
        "fast" | "small" => Some(Quality::Fast),
        "balanced" | "base" => Some(Quality::Balanced),
        "high" | "large" => Some(Quality::High),
        _ => None,
    }
}

/// Defaults: the three modes squarebob exposes × the three Burn-wgpu
/// quality selections × four common resolutions (320×240, 1280×720,
/// 1920×1080, 3840×2160). The smallest size lets the bench complete
/// in seconds even on slow CI; larger sizes test the tile planner.
fn default_cfg() -> Cfg {
    Cfg {
        // Squarebob ships weights under `data/oidn-weights`; oidn-rs
        // tests use `data/weights`. Try both at parse time.
        weights_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/weights"),
        resolutions: vec![(320, 240), (1280, 720), (1920, 1080), (3840, 2160)],
        modes: vec![Mode::Color, Mode::ColorAlbedo, Mode::ColorAlbedoNormal],
        qualities: vec![Quality::Fast, Quality::Balanced, Quality::High],
        iters: 10,
        warmup: 2,
        output: PathBuf::from(format!("bench-{}.csv", now_epoch_secs())),
        noise_magnitude: 0.12,
    }
}

fn parse_args(args: &[String]) -> Result<Cfg, support::Error> {
    let mut cfg = default_cfg();
    let mut weights_explicit = false;
    let mut i = 1;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> Result<&str, support::Error> {
            args.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("flag {flag} requires a value").into())
        };
        match flag {
            "--weights-dir" => {
                cfg.weights_dir = PathBuf::from(value()?);
                weights_explicit = true;
                i += 2;
            }
            "--output" | "-o" => {
                cfg.output = PathBuf::from(value()?);
                i += 2;
            }
            "--iters" => {
                cfg.iters = value()?.parse()?;
                i += 2;
            }
            "--warmup" => {
                cfg.warmup = value()?.parse()?;
                i += 2;
            }
            "--noise" => {
                cfg.noise_magnitude = value()?.parse()?;
                i += 2;
            }
            "--resolutions" => {
                cfg.resolutions = value()?
                    .split(',')
                    .map(support::resolution)
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "--modes" => {
                cfg.modes = value()?
                    .split(',')
                    .map(|s| Mode::parse(s).ok_or_else(|| format!("unknown mode {s}")))
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "--qualities" => {
                cfg.qualities = value()?
                    .split(',')
                    .map(|s| parse_quality(s).ok_or_else(|| format!("unknown quality {s}")))
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown flag {other}; use --help").into());
            }
        }
    }
    // Allow `data/oidn-weights/` as a fallback so the bench works both
    // from the oidn-rs workspace and from squarebob's bundled weights.
    if !weights_explicit && !cfg.weights_dir.is_dir() {
        let fallback = PathBuf::from("../../data/oidn-weights");
        if fallback.is_dir() {
            cfg.weights_dir = fallback;
        }
    }
    if cfg.iters == 0 {
        return Err("--iters must be positive".into());
    }
    if !cfg.noise_magnitude.is_finite() || cfg.noise_magnitude < 0.0 {
        return Err("--noise must be finite and nonnegative".into());
    }
    Ok(cfg)
}

fn print_help() {
    println!(
        "oidn-bench — sweep (resolution × mode × quality) and dump CSV

OPTIONS
  --weights-dir PATH     directory containing rt_*.tza weights
                         (default: workspace data/weights, falls back to
                          ../../data/oidn-weights)
  --output, -o FILE      CSV output path (default: bench-<epoch_secs>.csv)
  --resolutions LIST     comma-separated WxH, e.g. 1280x720,1920x1080
  --modes LIST           comma-separated; values: color, color_albedo,
                         color_albedo_normal (or c / ca / can)
  --qualities LIST       comma-separated; values: fast, balanced, high
                         (aliases: small / base / large)
  --iters N              timing iterations per combination (default 10)
  --warmup N             non-timed warmup iterations (default 2)
  --noise F              per-pixel noise magnitude added to the synthetic
                         gradient (default 0.12)
  -h, --help             print this and exit"
    );
}

// ---------------------- Bench core ----------------------

#[derive(Debug)]
struct Row {
    width: usize,
    height: usize,
    mode: Mode,
    quality: Quality,
    iters: usize,
    lat_min: f32,
    lat_med: f32,
    lat_max: f32,
    rmse_noisy: f32,
    rmse_denoised: f32,
    psnr_in_db: f32,
    psnr_out_db: f32,
    model: String,
}

impl Row {
    fn header() -> &'static str {
        "epoch_secs,width,height,mode,quality,iters,lat_min_ms,lat_med_ms,lat_max_ms,\
         rmse_noisy,rmse_denoised,psnr_in_db,psnr_out_db,improvement_x,model"
    }
    fn to_csv(&self, timestamp: &str) -> String {
        let improvement = if self.rmse_denoised > 0.0 {
            self.rmse_noisy / self.rmse_denoised
        } else {
            f32::NAN
        };
        format!(
            "{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.5},{:.5},{:.2},{:.2},{:.3},{}",
            timestamp,
            self.width,
            self.height,
            self.mode.as_str(),
            quality_str(self.quality),
            self.iters,
            self.lat_min,
            self.lat_med,
            self.lat_max,
            self.rmse_noisy,
            self.rmse_denoised,
            self.psnr_in_db,
            self.psnr_out_db,
            improvement,
            self.model,
        )
    }
    fn brief(&self) -> String {
        format!(
            "{:>5}x{:<5} {:>20} {:>9}  iters={:>3}  lat={:>6.1}/{:>6.1}/{:>6.1}ms  \
             PSNR {:>5.2}→{:>5.2}dB  {:>5.2}× rmse  [{}]",
            self.width,
            self.height,
            self.mode.as_str(),
            quality_str(self.quality),
            self.iters,
            self.lat_min,
            self.lat_med,
            self.lat_max,
            self.psnr_in_db,
            self.psnr_out_db,
            self.rmse_noisy / self.rmse_denoised.max(1e-12),
            self.model,
        )
    }
}

fn run_one(
    device: &WgpuDevice,
    cfg: &Cfg,
    (w, h): (usize, usize),
    mode: Mode,
    quality: Quality,
) -> Result<Row, Box<dyn std::error::Error>> {
    let weights_dir = &cfg.weights_dir;
    let iters = cfg.iters;
    let warmup = cfg.warmup;
    let noise = cfg.noise_magnitude;
    if iters == 0 {
        return Err("--iters must be positive".into());
    }
    let clean = make_clean(w, h)?;
    let noisy = add_noise(&clean, noise)?;
    let color_img = Image::from_rgb_f32(&noisy, w, h);

    let albedo: Vec<f32> = if matches!(mode, Mode::ColorAlbedo | Mode::ColorAlbedoNormal) {
        clean.iter().map(|v| v.clamp(0.0, 1.0)).collect()
    } else {
        Vec::new()
    };
    let normal = if matches!(mode, Mode::ColorAlbedoNormal) {
        make_normal(w, h)?
    } else {
        Vec::new()
    };

    let mut filter = RtFilter::builder(&device.handle, weights_dir)
        .hdr(true)
        .quality(quality)
        .weight_source(oidn_rs::weights::SourcePolicy::DiskFirst)
        // Pin the autoexposure so latency isn't dominated by scale
        // chatter between iterations.
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&color_img)?;
    if !albedo.is_empty() {
        let img = Image::from_rgb_f32(&albedo, w, h);
        filter.set_albedo(&img)?;
    }
    if !normal.is_empty() {
        let img = Image::from_rgb_f32(&normal, w, h);
        filter.set_normal(&img)?;
    }
    filter.allocate_output(w, h, PixelFormat::Rgb32f)?;
    filter.commit()?;
    let model = filter
        .model_key()
        .map(|k| k.0.clone())
        .unwrap_or_else(|| "<unknown>".into());

    // Run warmup + the first timed iteration to also produce the
    // output buffer for RMSE/PSNR. After each `take_output()` we
    // re-allocate so the next `execute()` has somewhere to write.
    for _ in 0..warmup {
        filter.execute()?;
        let _ = filter.take_output();
        filter.allocate_output(w, h, PixelFormat::Rgb32f)?;
    }

    let mut latencies = Vec::with_capacity(iters);
    let mut last_output: Option<Vec<u8>> = None;
    for _ in 0..iters {
        let t0 = Instant::now();
        filter.execute()?;
        let (raw, ow, oh, format) = filter.take_output().ok_or("take_output: empty")?;
        if ow != w
            || oh != h
            || format != PixelFormat::Rgb32f
            || raw.len()
                != support::samples(w, h, 3)?
                    .checked_mul(4)
                    .ok_or("output byte size overflow")?
        {
            return Err("benchmark received an invalid RGB32f output".into());
        }
        latencies.push(t0.elapsed().as_secs_f32() * 1000.0);
        last_output = Some(raw);
        filter.allocate_output(w, h, PixelFormat::Rgb32f)?;
    }
    latencies.sort_by(f32::total_cmp);
    let lat_min = latencies[0];
    let lat_max = *latencies.last().unwrap();
    let lat_med = latencies[latencies.len() / 2];

    let raw = last_output.ok_or("no successful timed iteration")?;
    let out: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_ne_bytes)
        .collect();
    let input_metrics = metrics(&noisy, &clean)?;
    let output_metrics = metrics(&out, &clean)?;
    let rmse_in = input_metrics.rmse as f32;
    let rmse_out = output_metrics.rmse as f32;

    Ok(Row {
        width: w,
        height: h,
        mode,
        quality,
        iters,
        lat_min,
        lat_med,
        lat_max,
        rmse_noisy: rmse_in,
        rmse_denoised: rmse_out,
        psnr_in_db: input_metrics.psnr(1.0)? as f32,
        psnr_out_db: output_metrics.psnr(1.0)? as f32,
        model,
    })
}

fn now_epoch_secs() -> String {
    // Raw Unix epoch seconds — good enough for a filename suffix
    // without pulling in `chrono` / `time` just to format an ISO
    // string. The same timestamp representation is used for filenames and CSV rows.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}", secs)
}

// ---------------------- main ----------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args(&std::env::args().collect::<Vec<_>>())?;

    if !cfg.weights_dir.is_dir() {
        eprintln!(
            "error: weights directory not found: {}\n\
             Pass --weights-dir or place TZA files at workspace data/weights \
             (oidn-rs convention) or ../../data/oidn-weights (squarebob).",
            cfg.weights_dir.display()
        );
        std::process::exit(1);
    }

    println!("OIDN bench");
    println!("  weights: {}", cfg.weights_dir.display());
    println!("  output:  {}", cfg.output.display());
    println!(
        "  grid:    {} resolutions × {} modes × {} qualities = {} combos",
        cfg.resolutions.len(),
        cfg.modes.len(),
        cfg.qualities.len(),
        cfg.resolutions.len() * cfg.modes.len() * cfg.qualities.len(),
    );
    println!("  iters:   {} (after {} warmup)", cfg.iters, cfg.warmup);

    let device = WgpuDevice::new()?;
    let mut out = File::create(&cfg.output)?;
    writeln!(out, "{}", Row::header())?;
    let mut succeeded = 0_usize;
    let mut failed = 0_usize;

    for &(w, h) in &cfg.resolutions {
        for &mode in &cfg.modes {
            for &quality in &cfg.qualities {
                match run_one(&device, &cfg, (w, h), mode, quality) {
                    Ok(row) => {
                        succeeded += 1;
                        let ts = now_epoch_secs();
                        writeln!(out, "{}", row.to_csv(&ts))?;
                        out.flush()?;
                        println!("{}", row.brief());
                    }
                    Err(e) => {
                        failed += 1;
                        eprintln!("  ! skipped {}×{} {:?} {:?}: {}", w, h, mode, quality, e);
                    }
                }
            }
        }
    }

    if succeeded == 0 {
        return Err(format!("all {failed} benchmark combinations failed").into());
    }
    println!(
        "\nWrote {} ({succeeded} successful, {failed} failed)",
        cfg.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn benchmark_rejects_invalid_args_without_panics_or_gpu() {
        for pair in [
            ["--iters", "0"],
            ["--iters", "oops"],
            ["--resolutions", "0x8"],
            ["--resolutions", "wrong"],
            ["--modes", "missing"],
            ["--qualities", "wrong"],
            ["--noise", "NaN"],
            ["--noise", "-0.1"],
        ] {
            let argv = ["bench", pair[0], pair[1]].map(String::from);
            assert!(parse_args(&argv).is_err(), "{pair:?}");
        }
        assert!(parse_args(&["bench".into(), "--iters".into()]).is_err());
        let argv = [
            "bench",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--resolutions",
            "3x2",
        ]
        .map(String::from);
        assert_eq!(parse_args(&argv).unwrap().resolutions, [(3, 2)]);
    }
}
