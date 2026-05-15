//! OIDN bench — runs the full RT filter through a grid of
//! `(resolution × mode × quality)` and writes latency + RMSE/PSNR
//! statistics to a CSV.
//!
//! Builds against the Wgpu backend so the numbers reflect the real
//! production path (Burn-wgpu through cubecl on whatever GPU
//! `wgpu::Instance::default()` picks). For each combination we:
//!
//! 1. Generate a deterministic synthetic HDR image — a smooth radial
//!    gradient plus hash-noise (`add_noise`). Same generator as
//!    `tests/e2e_wgpu.rs` so any drift here would also fail those
//!    tests.
//! 2. Build the filter, set inputs, allocate output, commit.
//! 3. Run `--warmup` iterations to amortise model load + tile-plan
//!    compute + first-touch GPU allocations.
//! 4. Time `--iters` iterations end-to-end through `filter.execute()`
//!    + `filter.take_output()`. The take is included because that's
//!    what the production loop pays.
//! 5. Compute RMSE vs the clean reference, then PSNR =
//!    `20 * log10(1.0 / rmse)` (peak signal is 1.0 for our gradient).
//!
//! Usage:
//!
//! ```
//! cargo run --release --example bench -- \
//!     --weights-dir ../../data/weights \
//!     --output bench-2026-05-15.csv
//! ```
//!
//! See `parse_args` below for the full flag set.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use oidn_rs::prelude::*;

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
/// model topologies × four common resolutions (320×240, 1280×720,
/// 1920×1080, 3840×2160). The smallest size lets the bench complete
/// in seconds even on slow CI; larger sizes test the tile planner.
fn default_cfg() -> Cfg {
    Cfg {
        // Squarebob ships weights under `data/oidn-weights`; oidn-rs
        // tests use `data/weights`. Try both at parse time.
        weights_dir: PathBuf::from("../../data/weights"),
        resolutions: vec![(320, 240), (1280, 720), (1920, 1080), (3840, 2160)],
        modes: vec![Mode::Color, Mode::ColorAlbedo, Mode::ColorAlbedoNormal],
        qualities: vec![Quality::Fast, Quality::Balanced, Quality::High],
        iters: 10,
        warmup: 2,
        output: PathBuf::from(format!("bench-{}.csv", now_iso_short())),
        noise_magnitude: 0.12,
    }
}

fn parse_args() -> Cfg {
    let mut cfg = default_cfg();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = || -> &str {
            args.get(i + 1).map(|s| s.as_str()).unwrap_or_else(|| {
                eprintln!("error: flag `{flag}` requires a value");
                std::process::exit(1)
            })
        };
        match flag {
            "--weights-dir" => { cfg.weights_dir = PathBuf::from(value()); i += 2; }
            "--output" | "-o" => { cfg.output = PathBuf::from(value()); i += 2; }
            "--iters" => { cfg.iters = value().parse().expect("--iters: integer"); i += 2; }
            "--warmup" => { cfg.warmup = value().parse().expect("--warmup: integer"); i += 2; }
            "--noise" => { cfg.noise_magnitude = value().parse().expect("--noise: float"); i += 2; }
            "--resolutions" => {
                cfg.resolutions = value()
                    .split(',')
                    .map(|s| {
                        let (w, h) = s.split_once('x').expect("resolution: WIDTHxHEIGHT");
                        (w.parse().expect("width"), h.parse().expect("height"))
                    })
                    .collect();
                i += 2;
            }
            "--modes" => {
                cfg.modes = value()
                    .split(',')
                    .map(|s| Mode::parse(s).unwrap_or_else(|| panic!("unknown mode `{s}`")))
                    .collect();
                i += 2;
            }
            "--qualities" => {
                cfg.qualities = value()
                    .split(',')
                    .map(|s| parse_quality(s).unwrap_or_else(|| panic!("unknown quality `{s}`")))
                    .collect();
                i += 2;
            }
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            other => {
                eprintln!("error: unknown flag `{other}`. Use --help.");
                std::process::exit(1);
            }
        }
    }
    // Allow `data/oidn-weights/` as a fallback so the bench works both
    // from the oidn-rs workspace and from squarebob's bundled weights.
    if !cfg.weights_dir.is_dir() {
        let fallback = PathBuf::from("../../data/oidn-weights");
        if fallback.is_dir() {
            cfg.weights_dir = fallback;
        }
    }
    cfg
}

fn print_help() {
    println!(
"oidn-bench — sweep (resolution × mode × quality) and dump CSV

OPTIONS
  --weights-dir PATH     directory containing rt_*.tza weights
                         (default: ../../data/weights, falls back to
                          ../../data/oidn-weights)
  --output, -o FILE      CSV output path (default: bench-<date>.csv)
  --resolutions LIST     comma-separated WxH, e.g. 1280x720,1920x1080
  --modes LIST           comma-separated; values: color, color_albedo,
                         color_albedo_normal (or c / ca / can)
  --qualities LIST       comma-separated; values: fast, balanced, high
                         (aliases: small / base / large)
  --iters N              timing iterations per combination (default 10)
  --warmup N             non-timed warmup iterations (default 2)
  --noise F              per-pixel noise magnitude added to the synthetic
                         gradient (default 0.12)
  -h, --help             print this and exit");
}

// ---------------------- Synthetic data ----------------------

/// Smooth radial gradient — ground truth for PSNR. Same generator as
/// `tests/e2e_wgpu.rs::make_clean`, so the bench output is comparable
/// across runs and CI machines.
fn make_clean(w: usize, h: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; w * h * 3];
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
            buf[i] = v;
            buf[i + 1] = v * 0.9;
            buf[i + 2] = v * 0.7;
        }
    }
    buf
}

/// Deterministic hash-noise (no `rand` dep). Same generator as in
/// `tests/e2e_wgpu.rs::add_noise`.
fn add_noise(clean: &[f32], magnitude: f32) -> Vec<f32> {
    let mut out = clean.to_vec();
    for (i, v) in out.iter_mut().enumerate() {
        let mut n = (i as u32).wrapping_mul(2654435761);
        n ^= n >> 13;
        n = n.wrapping_mul(0x85ebca6b);
        n ^= n >> 16;
        let r = (n as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *v += r * magnitude;
    }
    out
}

/// Constant up-vector normal map — matches the AOV pattern from the
/// `denoise_with_albedo_normal_wgpu` test.
fn make_normal(w: usize, h: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; w * h * 3];
    for px in buf.chunks_exact_mut(3) {
        px[0] = 0.0;
        px[1] = 1.0;
        px[2] = 0.0;
    }
    buf
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() as f32;
    let s: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (s / n).sqrt()
}

fn psnr_db(rmse: f32) -> f32 {
    // Peak signal ≈ 1.0 for our synthetic gradient (values land in
    // ~[0.5, 0.95]). Clamp to avoid -inf when rmse is 0.
    20.0 * (1.0_f32.max(1.0) / rmse.max(1e-12)).log10()
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
        "timestamp,width,height,mode,quality,iters,lat_min_ms,lat_med_ms,lat_max_ms,\
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
            self.width, self.height,
            self.mode.as_str(),
            quality_str(self.quality),
            self.iters,
            self.lat_min, self.lat_med, self.lat_max,
            self.rmse_noisy, self.rmse_denoised,
            self.psnr_in_db, self.psnr_out_db,
            improvement,
            self.model,
        )
    }
    fn brief(&self) -> String {
        format!(
            "{:>5}x{:<5} {:>20} {:>9}  iters={:>3}  lat={:>6.1}/{:>6.1}/{:>6.1}ms  \
             PSNR {:>5.2}→{:>5.2}dB  {:>5.2}× rmse  [{}]",
            self.width, self.height,
            self.mode.as_str(),
            quality_str(self.quality),
            self.iters,
            self.lat_min, self.lat_med, self.lat_max,
            self.psnr_in_db, self.psnr_out_db,
            self.rmse_noisy / self.rmse_denoised.max(1e-12),
            self.model,
        )
    }
}

fn run_one(
    device: &WgpuDevice,
    weights_dir: &Path,
    w: usize,
    h: usize,
    mode: Mode,
    quality: Quality,
    iters: usize,
    warmup: usize,
    noise: f32,
) -> Result<Row, Box<dyn std::error::Error>> {
    let clean = make_clean(w, h);
    let noisy = add_noise(&clean, noise);
    let color_img = Image::from_rgb_f32(&noisy, w, h);

    let albedo: Vec<f32> = if matches!(mode, Mode::ColorAlbedo | Mode::ColorAlbedoNormal) {
        clean.iter().map(|v| v.clamp(0.0, 1.0)).collect()
    } else {
        Vec::new()
    };
    let normal = if matches!(mode, Mode::ColorAlbedoNormal) {
        make_normal(w, h)
    } else {
        Vec::new()
    };

    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, weights_dir)
        .hdr(true)
        .quality(quality)
        // Pin the autoexposure so latency isn't dominated by scale
        // chatter between iterations.
        .input_scale(Some(1.0))
        .build();
    filter.set_color(&color_img);
    if !albedo.is_empty() {
        let img = Image::from_rgb_f32(&albedo, w, h);
        filter.set_albedo(&img);
    }
    if !normal.is_empty() {
        let img = Image::from_rgb_f32(&normal, w, h);
        filter.set_normal(&img);
    }
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
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
        filter.allocate_output(w, h, PixelFormat::Rgb32f);
    }

    let mut latencies = Vec::with_capacity(iters);
    let mut last_output: Option<Vec<u8>> = None;
    for _ in 0..iters {
        let t0 = Instant::now();
        filter.execute()?;
        let (raw, _w, _h, _fmt) = filter.take_output().ok_or("take_output: empty")?;
        latencies.push(t0.elapsed().as_secs_f32() * 1000.0);
        last_output = Some(raw);
        filter.allocate_output(w, h, PixelFormat::Rgb32f);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lat_min = latencies[0];
    let lat_max = *latencies.last().unwrap();
    let lat_med = latencies[latencies.len() / 2];

    let raw = last_output.expect("at least one iteration");
    let out: &[f32] = bytemuck::cast_slice(&raw);
    let rmse_in = rmse(&noisy, &clean);
    let rmse_out = rmse(out, &clean);

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
        psnr_in_db: psnr_db(rmse_in),
        psnr_out_db: psnr_db(rmse_out),
        model,
    })
}

fn now_iso_short() -> String {
    // YYYYMMDD-HHMMSS without external time deps. Good enough for a
    // file suffix; the CSV `timestamp` column uses a full ISO string.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Trivial Y/M/D math via Unix epoch days. Avoid pulling chrono
    // just for a filename suffix.
    format!("{}", secs)
}

fn now_iso_long() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Same simplification — we tag rows with epoch seconds so the
    // CSV is sortable without parsing.
    format!("{}", secs)
}

// ---------------------- main ----------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args();

    if !cfg.weights_dir.is_dir() {
        eprintln!(
            "error: weights directory not found: {}\n\
             Pass --weights-dir or place TZA files at ../../data/weights \
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

    for &(w, h) in &cfg.resolutions {
        for &mode in &cfg.modes {
            for &quality in &cfg.qualities {
                match run_one(
                    &device, &cfg.weights_dir,
                    w, h, mode, quality,
                    cfg.iters, cfg.warmup, cfg.noise_magnitude,
                ) {
                    Ok(row) => {
                        let ts = now_iso_long();
                        writeln!(out, "{}", row.to_csv(&ts))?;
                        out.flush()?;
                        println!("{}", row.brief());
                    }
                    Err(e) => {
                        eprintln!(
                            "  ! skipped {}×{} {:?} {:?}: {}",
                            w, h, mode, quality, e
                        );
                    }
                }
            }
        }
    }

    println!("\nWrote {}", cfg.output.display());
    Ok(())
}
