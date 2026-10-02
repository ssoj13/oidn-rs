//! `oidn-rs` — command-line denoiser using the oidn-rs library.

mod io;
pub mod support;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn::tensor::Device;
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use io::Encoding;
use oidn_rs::prelude::wgpu_prelude::*;
use oidn_rs::prelude::*;
use oidn_rs::weights::SourcePolicy;

#[derive(Parser, Debug)]
#[command(name = "oidn-rs", version, about = "Pure Rust port of Intel OIDN")]
struct Cli {
    /// Execution backend; CPU never initializes a GPU.
    #[arg(long, global = true, value_enum, default_value_t = DeviceChoice::Wgpu)]
    device: DeviceChoice,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DeviceChoice {
    Cpu,
    Wgpu,
}

impl DeviceChoice {
    fn create(self) -> Result<Device, support::Error> {
        let device = match self {
            Self::Cpu => Device::ndarray(),
            Self::Wgpu => WgpuDevice::new()?.handle,
        };
        tracing::info!("backend: {:?}", self);
        Ok(device)
    }
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print the tensor list of a TZA weights file.
    Probe {
        /// Path to a `.tza` file from the oidn-weights submodule.
        path: PathBuf,

        /// Emit one JSON object per tensor instead of the human-readable table.
        #[arg(long, action = ArgAction::SetTrue)]
        json: bool,
    },

    /// Denoise an image (EXR / PFM / PHM / HDR / TIFF / PNG / JPG / BMP).
    Denoise(Box<DenoiseArgs>),

    /// Benchmark denoising throughput on a synthetic HDR scene.
    Bench {
        /// Resolution as `WIDTHxHEIGHT` (e.g. `1920x1080`).
        #[arg(short, long, default_value = "1024x1024")]
        resolution: String,

        /// Number of timed iterations after one warm-up run.
        #[arg(short = 'n', long, default_value_t = 10)]
        iters: u32,

        /// Quality preset.
        #[arg(short, long, default_value = "balanced", value_parser = parse_quality_clap)]
        quality: Quality,

        /// Directory of `.tza` weight files (defaults to `./data/weights`).
        #[arg(long, default_value = "data/weights")]
        weights_dir: PathBuf,

        /// Accepted for parity with `oidnDenoise`; ignored on the wgpu backend.
        #[arg(long)]
        threads: Option<u32>,
    },

    /// List wgpu adapters visible on this system.
    ListDevices,
}

/// `FilterKind` mirrors the `-f` / `--filter` argument of the reference
/// `oidnDenoise` CLI.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum FilterKind {
    #[value(name = "RT", alias = "rt")]
    Rt,
    #[value(name = "RTLightmap", alias = "rtlightmap", alias = "rt_lightmap")]
    RtLightmap,
}

#[derive(Args, Debug)]
struct DenoiseArgs {
    /// Noisy colour input.
    #[arg(short, long)]
    input: PathBuf,

    /// Output path.
    #[arg(short, long)]
    output: PathBuf,

    /// Optional auxiliary albedo image.
    #[arg(long)]
    albedo: Option<PathBuf>,

    /// Optional auxiliary world-space normal image.
    #[arg(long)]
    normal: Option<PathBuf>,

    /// Input is HDR — PU transfer + autoexposure.
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["srgb", "ldr"])]
    hdr: bool,

    /// Input is LDR (linear in [0, 1]).
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "hdr")]
    ldr: bool,

    /// LDR color buffer stays sRGB; albedo is decoded independently and normals stay numeric.
    #[arg(long, action = ArgAction::SetTrue)]
    srgb: bool,

    /// Auxiliary albedo / normal images are already denoised (the "clean aux"
    /// model variant). Requires both `--albedo` and `--normal`.
    #[arg(long = "clean_aux", alias = "clean-aux", action = ArgAction::SetTrue)]
    clean_aux: bool,

    /// Explicit input scale; if omitted the filter computes its own
    /// autoexposure value (HDR) or defaults to 1.0 (LDR).
    #[arg(long = "input_scale", alias = "input-scale")]
    input_scale: Option<f32>,

    /// Quality preset. Accepts `default`/`high`/`h`/`balanced`/`b`/`fast`/`f`.
    #[arg(short, long, default_value = "default", value_parser = parse_quality_clap)]
    quality: Quality,

    /// Filter family. `RT` (default) is the standard ray-tracing denoiser;
    /// `RTLightmap` is the lightmap variant.
    #[arg(short = 'f', long, value_enum, default_value_t = FilterKind::Rt)]
    filter: FilterKind,

    /// Use the directional lightmap network (only meaningful with
    /// `--filter RTLightmap`).
    #[arg(long = "dir", alias = "directional", action = ArgAction::SetTrue)]
    directional: bool,

    /// Directory containing `.tza` weight files. Ignored if `--weights` is
    /// passed; falls back to embedded weights when omitted.
    #[arg(long)]
    weights_dir: Option<PathBuf>,

    /// Path to a single `.tza` blob — overrides `--weights_dir` and the
    /// embedded weight lookup.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Accepted for parity with `oidnDenoise`; ignored on the wgpu backend.
    #[arg(long)]
    threads: Option<u32>,

    /// Maximum memory budget in MB for either family; negative disables the limit.
    #[arg(long, allow_negative_numbers = true)]
    maxmem: Option<i32>,

    /// Re-run the filter N times for hash-stability checks. Default 1.
    #[arg(short = 'n', long, default_value_t = 1)]
    iters: u32,

    /// Tracing verbosity: 0=warn, 1=info, 2=debug, 3=trace. Overrides `RUST_LOG`.
    #[arg(short = 'v', long, default_value_t = 1)]
    verbose: u8,

    /// Optional reference image — prints MSE / PSNR / MaxError versus the output.
    #[arg(long = "ref", alias = "reference")]
    reference: Option<PathBuf>,

    /// MSE threshold; exit non-zero if the output diverges from `--ref` beyond it.
    #[arg(long)]
    maxerror: Option<f32>,
}

fn main() -> ExitCode {
    // Verbose flag (if present on the denoise subcommand) overrides the
    // default RUST_LOG; we parse the CLI first to peek at it before init.
    let cli = Cli::parse();
    let verbosity = match &cli.cmd {
        Cmd::Denoise(a) => Some(a.verbose),
        _ => None,
    };
    tracing_subscriber_init(verbosity);
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn tracing_subscriber_init(verbose: Option<u8>) {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = if let Some(v) = verbose {
        let level = match v {
            0 => "warn",
            1 => "info",
            2 => "debug",
            _ => "trace",
        };
        EnvFilter::new(level)
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.cmd {
        Cmd::Probe { path, json } => probe(&path, json),
        Cmd::Denoise(args) => denoise(*args, cli.device),
        Cmd::Bench {
            resolution,
            iters,
            quality,
            weights_dir,
            threads,
        } => {
            if threads == Some(0) {
                return Err("--threads must be positive".into());
            }
            if threads.is_some() {
                tracing::info!("--threads is not supported by this frontend");
            }
            bench(&resolution, iters, quality, &weights_dir, cli.device)
        }
        Cmd::ListDevices => list_devices(),
    }
}

fn probe(path: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let tensors = oidn_tza::parse(&bytes)?;
    if !json {
        println!("# {} ({} tensors)", path.display(), tensors.len());
    }
    for (name, t) in &tensors {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "name": name,
                    "dims": t.desc.dims,
                    "layout": format!("{:?}", t.desc.layout),
                    "dtype": format!("{:?}", t.desc.dtype),
                })
            );
        } else {
            println!(
                "{:32} dims={:?} layout={:?} dtype={:?}",
                name, t.desc.dims, t.desc.layout, t.desc.dtype
            );
        }
    }
    Ok(())
}

fn validate(args: &DenoiseArgs) -> Result<(), support::Error> {
    if args.iters == 0 {
        return Err("--iters must be positive".into());
    }
    if args.input_scale.is_some_and(|v| !v.is_finite() || v <= 0.0) {
        return Err("--input-scale must be finite and positive".into());
    }
    if args.maxerror.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return Err("--maxerror must be finite and nonnegative".into());
    }
    if args.maxerror.is_some() && args.reference.is_none() {
        return Err("--maxerror requires --ref".into());
    }
    if args.threads == Some(0) {
        return Err("--threads must be positive".into());
    }
    if args.filter == FilterKind::Rt {
        if !args.hdr && !args.ldr {
            return Err("RT requires one of --hdr or --ldr".into());
        }
        if args.directional {
            return Err("--directional requires --filter RTLightmap".into());
        }
    } else {
        if args.ldr || args.srgb {
            return Err("RTLightmap is intrinsically HDR (or signed directional); --ldr/--srgb are unsupported".into());
        }
        if args.albedo.is_some() || args.normal.is_some() || args.clean_aux {
            return Err("RTLightmap does not accept auxiliary images".into());
        }
    }
    if args.clean_aux && (args.albedo.is_none() || args.normal.is_none()) {
        return Err("--clean-aux requires both --albedo and --normal".into());
    }
    Ok(())
}

fn encoding(args: &DenoiseArgs) -> Encoding {
    if args.directional {
        Encoding::Data
    } else if args.srgb {
        Encoding::Srgb
    } else {
        Encoding::Linear
    }
}

fn denoise(args: DenoiseArgs, backend: DeviceChoice) -> Result<(), Box<dyn std::error::Error>> {
    validate(&args)?;
    if args.threads.is_some() {
        tracing::info!("--threads is not supported by this frontend");
    }
    let device = backend.create()?;

    let (color_pixels, w, h) = io::load_rgb_f32(&args.input, encoding(&args))?;
    let albedo_pixels = args
        .albedo
        .as_deref()
        .map(|p| io::load_rgb_f32(p, Encoding::Linear))
        .transpose()?;
    let normal_pixels = args
        .normal
        .as_deref()
        .map(|p| io::load_rgb_f32(p, Encoding::Data))
        .transpose()?;

    // Ordinary model selection belongs to the filter's shared resolver.
    let user_weights = args.weights.as_deref().map(std::fs::read).transpose()?;
    let weights_dir = args
        .weights_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/weights"));

    match args.filter {
        FilterKind::Rt => run_rt(
            &device,
            &weights_dir,
            user_weights,
            &args,
            (&color_pixels, w, h),
            albedo_pixels.as_ref(),
            normal_pixels.as_ref(),
        )?,
        FilterKind::RtLightmap => run_rtlightmap(
            &device,
            &weights_dir,
            user_weights,
            &args,
            (&color_pixels, w, h),
        )?,
    }

    Ok(())
}

fn run_rt(
    device: &Device,
    weights_dir: &Path,
    user_weights: Option<Vec<u8>>,
    args: &DenoiseArgs,
    (color, w, h): (&[f32], usize, usize),
    albedo: Option<&io::RgbImage>,
    normal: Option<&io::RgbImage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = RtFilter::builder(device, weights_dir)
        .hdr(args.hdr)
        .srgb(args.srgb)
        .clean_aux(args.clean_aux)
        .quality(args.quality)
        .input_scale(args.input_scale)
        .weight_source(if args.weights_dir.is_some() {
            SourcePolicy::DiskFirst
        } else {
            SourcePolicy::EmbeddedFirst
        });
    if let Some(mb) = args.maxmem {
        builder = builder.max_memory_mb(mb);
    }
    if let Some(bytes) = user_weights {
        builder = builder.weights(bytes);
    }
    let mut filter = builder.build();

    let color_img = Image::from_rgb_f32(color, w, h);
    filter.set_color(&color_img)?;

    let albedo_img = albedo.map(|(buf, w, h)| Image::from_rgb_f32(buf, *w, *h));
    if let Some(img) = &albedo_img {
        filter.set_albedo(img)?;
    }
    let normal_img = normal.map(|(buf, w, h)| Image::from_rgb_f32(buf, *w, *h));
    if let Some(img) = &normal_img {
        filter.set_normal(img)?;
    }

    filter.allocate_output(w, h, PixelFormat::Rgb32f)?;
    filter.commit()?;
    if let Some(k) = filter.model_key() {
        tracing::info!("model: {}", k.0);
    }

    for i in 0..args.iters {
        let t0 = std::time::Instant::now();
        filter.execute()?;
        tracing::info!("iter {}: {:.2} ms", i, t0.elapsed().as_secs_f64() * 1000.0);
    }

    let output = filter.take_output().ok_or("no output")?;
    save_output(args, output)
}

fn run_rtlightmap(
    device: &Device,
    weights_dir: &Path,
    user_weights: Option<Vec<u8>>,
    args: &DenoiseArgs,
    (color, w, h): (&[f32], usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = RtLightmapFilter::builder(device, weights_dir)
        .directional(args.directional)
        .quality(args.quality)
        .input_scale(args.input_scale)
        .weight_source(if args.weights_dir.is_some() {
            SourcePolicy::DiskFirst
        } else {
            SourcePolicy::EmbeddedFirst
        });
    if let Some(mb) = args.maxmem {
        builder = builder.max_memory_mb(mb);
    }
    if let Some(bytes) = user_weights {
        builder = builder.weights(bytes);
    }
    let mut filter = builder.build();

    let color_img = Image::from_rgb_f32(color, w, h);
    filter.set_color(&color_img)?;
    filter.allocate_output(w, h, PixelFormat::Rgb32f)?;
    filter.commit()?;
    if let Some(k) = filter.model_key() {
        tracing::info!("model: {}", k.0);
    }
    for i in 0..args.iters {
        let t0 = std::time::Instant::now();
        filter.execute()?;
        tracing::info!("iter {}: {:.2} ms", i, t0.elapsed().as_secs_f64() * 1000.0);
    }
    let output = filter.take_output().ok_or("no output")?;
    save_output(args, output)
}

fn save_output(
    args: &DenoiseArgs,
    (raw, w, h, format): (Vec<u8>, usize, usize, PixelFormat),
) -> Result<(), support::Error> {
    if format != PixelFormat::Rgb32f
        || raw.len()
            != support::samples(w, h, 3)?
                .checked_mul(4)
                .ok_or("output byte length overflow")?
    {
        return Err("filter returned an invalid RGB32f output".into());
    }
    let pixels: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_ne_bytes)
        .collect();
    io::save_rgb_f32(&args.output, &pixels, w, h, encoding(args))?;
    if let Some(reference) = args.reference.as_deref() {
        compare_against_reference(reference, &pixels, w, h, args.maxerror, encoding(args))?;
    }
    eprintln!("wrote {}", args.output.display());
    Ok(())
}

fn compare_against_reference(
    ref_path: &Path,
    out_pixels: &[f32],
    w: usize,
    h: usize,
    maxerror: Option<f32>,
    encoding: Encoding,
) -> Result<(), Box<dyn std::error::Error>> {
    let (ref_pixels, rw, rh) = io::load_rgb_f32(ref_path, encoding)?;
    if rw != w || rh != h {
        return Err(format!("reference image is {rw}x{rh}, output is {w}x{h}",).into());
    }
    let metrics = support::metrics(out_pixels, &ref_pixels)?;
    let mse = metrics.mse;
    let maxe = metrics.max_error;
    let psnr = metrics.psnr(1.0)?;
    println!("compare: mse={mse:.6e} psnr={psnr:.2} dB (peak=1) max={maxe:.6e}");
    if maxerror.is_some_and(|threshold| mse > f64::from(threshold)) {
        return Err(format!("MSE {mse:.6e} exceeds --maxerror").into());
    }
    Ok(())
}

fn parse_quality_clap(s: &str) -> Result<Quality, String> {
    match s.to_ascii_lowercase().as_str() {
        // `default` aliases match reference oidnDenoise.cpp behaviour:
        // empty / "default" → highest available.
        "default" | "high" | "h" => Ok(Quality::High),
        "balanced" | "b" => Ok(Quality::Balanced),
        "fast" | "f" => Ok(Quality::Fast),
        other => Err(format!(
            "unknown quality `{other}` (expected default|high|h|balanced|b|fast|f)"
        )),
    }
}

fn parse_resolution(s: &str) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    support::resolution(s)
}

fn list_devices() -> Result<(), Box<dyn std::error::Error>> {
    // wgpu 29: `InstanceDescriptor` no longer impls Default and
    // `enumerate_adapters` returns a future — block on it synchronously.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapters: Vec<wgpu::Adapter> =
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if adapters.is_empty() {
        println!("(no wgpu adapters found)");
        return Ok(());
    }
    for (i, a) in adapters.iter().enumerate() {
        let info = a.get_info();
        println!(
            "[{i}] {} ({:?}) backend={:?} device_type={:?}",
            info.name, info.vendor, info.backend, info.device_type
        );
    }
    Ok(())
}

fn bench(
    resolution: &str,
    iters: u32,
    quality: Quality,
    weights_dir: &Path,
    backend: DeviceChoice,
) -> Result<(), Box<dyn std::error::Error>> {
    if iters == 0 {
        return Err("--iters must be positive".into());
    }
    let (w, h) = parse_resolution(resolution)?;
    let device = backend.create()?;

    let color = support::add_noise(&support::make_clean(w, h)?, 0.12)?;
    let color_img = Image::from_rgb_f32(&color, w, h);

    let mut filter = RtFilter::builder(&device, weights_dir)
        .hdr(true)
        .quality(quality)
        .weight_source(SourcePolicy::DiskFirst)
        .build();
    filter.set_color(&color_img)?;
    filter.allocate_output(w, h, PixelFormat::Rgb32f)?;
    filter.commit()?;

    eprintln!(
        "bench: {w}x{h}, quality={:?}, model={}",
        quality,
        filter
            .model_key()
            .map(|k| k.0.as_str())
            .unwrap_or("<unknown>")
    );

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

    times_ms.sort_by(f64::total_cmp);
    let min = times_ms.first().copied().unwrap_or(0.0);
    let max = times_ms.last().copied().unwrap_or(0.0);
    let avg: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let med = times_ms[times_ms.len() / 2];
    let mp = (w * h) as f64 / 1_000_000.0;

    println!(
        "resolution={w}x{h} ({mp:.2} MP) quality={:?} iters={iters}",
        quality
    );
    println!("  min={min:>8.2} ms  median={med:>8.2} ms  avg={avg:>8.2} ms  max={max:>8.2} ms");
    println!("  throughput @ median: {:.2} MP/s", mp / (med / 1000.0));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(extra: &[&str]) -> DenoiseArgs {
        let mut argv = vec!["oidn-rs", "denoise", "-i", "in.pfm", "-o", "out.pfm"];
        argv.extend_from_slice(extra);
        let Cmd::Denoise(args) = Cli::try_parse_from(argv).unwrap().cmd else {
            panic!("denoise command");
        };
        *args
    }
    #[test]
    fn family_validation_and_invalid_quality_gates_precede_gpu_work() {
        assert!(validate(&args(&["--hdr"])).is_ok());
        assert!(validate(&args(&["--hdr", "--maxmem", "-1"])).is_ok());
        assert_eq!(
            Cli::try_parse_from(["oidn-rs", "--device", "cpu", "bench"])
                .unwrap()
                .device,
            DeviceChoice::Cpu
        );
        assert_eq!(
            Cli::try_parse_from(["oidn-rs", "bench", "--device", "cpu"])
                .unwrap()
                .device,
            DeviceChoice::Cpu
        );
        assert!(validate(&args(&["--filter", "RTLightmap"])).is_ok());
        assert!(validate(&args(&["--hdr", "--dir"])).is_err());
        assert!(validate(&args(&["--filter", "RTLightmap", "--ldr"])).is_err());
        assert!(validate(&args(&["--hdr", "--iters", "0"])).is_err());
        assert!(validate(&args(&["--hdr", "--input-scale", "NaN"])).is_err());
        assert!(validate(&args(&["--hdr", "--maxerror", "NaN", "--ref", "ref.pfm"])).is_err());
        assert!(validate(&args(&["--hdr", "--maxerror", "1"])).is_err());
        assert!(
            Cli::try_parse_from([
                "oidn-rs", "denoise", "-i", "in.pfm", "-o", "out.pfm", "--hdr", "--ldr"
            ])
            .is_err()
        );
    }
}
