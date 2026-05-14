//! `oidn-rs` — command-line denoiser using the oidn-rs library.

mod io;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use oidn_rs::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "oidn-rs", version, about = "Pure Rust port of Intel OIDN")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print the tensor list of a TZA weights file.
    Probe {
        /// Path to a `.tza` file from the oidn-weights submodule.
        path: PathBuf,
    },

    /// Denoise an EXR (or PNG) image.
    Denoise {
        /// Noisy colour input.
        #[arg(short, long)]
        input: PathBuf,

        /// Optional auxiliary albedo image (same resolution as `--input`).
        #[arg(long)]
        albedo: Option<PathBuf>,

        /// Optional auxiliary world-space normal image.
        #[arg(long)]
        normal: Option<PathBuf>,

        /// Output path (.exr or .png based on extension).
        #[arg(short, long)]
        output: PathBuf,

        /// Treat the input as HDR (PU transfer, autoexposure).
        #[arg(long, default_value_t = true)]
        hdr: bool,

        /// Directory containing `.tza` weight files (defaults to `./data`,
        /// produced by `git clone https://github.com/RenderKit/oidn-weights.git data`).
        #[arg(long, default_value = "data/weights")]
        weights_dir: PathBuf,
    },

    /// Benchmark denoising throughput on a synthetic HDR scene.
    Bench {
        /// Resolution as `WIDTHxHEIGHT` (e.g. `1920x1080`).
        #[arg(short, long, default_value = "1024x1024")]
        resolution: String,

        /// Number of timed iterations after one warm-up run.
        #[arg(short = 'n', long, default_value_t = 10)]
        iters: u32,

        /// Quality preset.
        #[arg(short, long, default_value = "balanced")]
        quality: String,

        /// Directory of `.tza` weight files (defaults to `./data`).
        #[arg(long, default_value = "data/weights")]
        weights_dir: PathBuf,
    },
}

fn main() -> ExitCode {
    tracing_subscriber_init();
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn tracing_subscriber_init() {
    // Respect RUST_LOG, default to INFO. Avoid pulling in the full
    // tracing-subscriber crate to keep dep count down.
    let level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into());
    eprintln!("oidn-rs (log={level})");
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.cmd {
        Cmd::Probe { path } => probe(&path),
        Cmd::Denoise { input, albedo, normal, output, hdr, weights_dir } => {
            denoise(&input, albedo.as_deref(), normal.as_deref(), &output, hdr, &weights_dir)
        }
        Cmd::Bench { resolution, iters, quality, weights_dir } => {
            bench(&resolution, iters, &quality, &weights_dir)
        }
    }
}

fn probe(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let tensors = oidn_tza::parse(&bytes)?;
    println!("# {} ({} tensors)", path.display(), tensors.len());
    for (name, t) in &tensors {
        println!("{:32} dims={:?} layout={:?} dtype={:?}", name, t.desc.dims, t.desc.layout, t.desc.dtype);
    }
    Ok(())
}

fn denoise(
    input: &std::path::Path,
    albedo: Option<&std::path::Path>,
    normal: Option<&std::path::Path>,
    output: &std::path::Path,
    hdr: bool,
    weights_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::new()?;

    let (color_pixels, w, h) = io::load_rgb_f32(input)?;
    let albedo_pixels = albedo.map(io::load_rgb_f32).transpose()?;
    let normal_pixels = normal.map(io::load_rgb_f32).transpose()?;

    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, weights_dir)
        .hdr(hdr)
        .quality(Quality::High)
        .build();

    let color_img = Image::from_rgb_f32(&color_pixels, w, h);
    filter.set_color(&color_img);

    let albedo_img = albedo_pixels.as_ref().map(|(buf, w, h)| Image::from_rgb_f32(buf, *w, *h));
    if let Some(img) = &albedo_img { filter.set_albedo(img); }

    let normal_img = normal_pixels.as_ref().map(|(buf, w, h)| Image::from_rgb_f32(buf, *w, *h));
    if let Some(img) = &normal_img { filter.set_normal(img); }

    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit()?;
    eprintln!("model: {}", filter.model_key().unwrap().0);
    filter.execute()?;

    let (raw, ow, oh, fmt) = filter.take_output().ok_or("no output")?;
    debug_assert_eq!(fmt, PixelFormat::Rgb32f);
    let out_pixels: &[f32] = bytemuck::cast_slice(&raw);
    io::save_rgb_f32(output, out_pixels, ow, oh)?;

    eprintln!("wrote {}", output.display());
    Ok(())
}

fn parse_resolution(s: &str) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let (w, h) = s.split_once('x').ok_or("resolution must be WxH (e.g. 1024x1024)")?;
    Ok((w.parse()?, h.parse()?))
}

fn parse_quality(s: &str) -> Result<Quality, Box<dyn std::error::Error>> {
    match s.to_ascii_lowercase().as_str() {
        "high" => Ok(Quality::High),
        "balanced" => Ok(Quality::Balanced),
        "fast" => Ok(Quality::Fast),
        other => Err(format!("unknown quality `{other}` (expected high/balanced/fast)").into()),
    }
}

fn bench(
    resolution: &str,
    iters: u32,
    quality: &str,
    weights_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = parse_resolution(resolution)?;
    let q = parse_quality(quality)?;
    let device = WgpuDevice::new()?;

    // Generate a deterministic noisy HDR image.
    let mut color = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = 0.5 + 0.3 * ((x + y) as f32 / (w + h) as f32);
            let n = ((x * 17 + y * 31) % 19) as f32 * 0.04;
            let i = (y * w + x) * 3;
            color[i]     = v + n;
            color[i + 1] = v + n * 0.7;
            color[i + 2] = v + n * 0.4;
        }
    }
    let color_img = Image::from_rgb_f32(&color, w, h);

    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, weights_dir)
        .hdr(true)
        .quality(q)
        .build();
    filter.set_color(&color_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit()?;

    eprintln!("bench: {w}x{h}, quality={quality}, model={}", filter.model_key().unwrap().0);

    // Warm-up run (excluded from timing — wgpu pipeline + shader compile
    // happens here on most backends).
    filter.execute()?;

    let mut times_ms = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        filter.execute()?;
        let dt = t0.elapsed();
        times_ms.push(dt.as_secs_f64() * 1000.0);
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times_ms.first().copied().unwrap_or(0.0);
    let max = times_ms.last().copied().unwrap_or(0.0);
    let avg: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let med = times_ms[times_ms.len() / 2];
    let mp = (w * h) as f64 / 1_000_000.0;

    println!("resolution={w}x{h} ({mp:.2} MP) quality={quality} iters={iters}");
    println!("  min={min:>8.2} ms  median={med:>8.2} ms  avg={avg:>8.2} ms  max={max:>8.2} ms");
    println!("  throughput @ median: {:.2} MP/s", mp / (med / 1000.0));

    Ok(())
}
