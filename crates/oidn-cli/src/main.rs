//! `oidn-rs` — command-line denoiser using the oidn-rs library.

mod io;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use oidn_rs::prelude::*;
use oidn_rs::prelude::wgpu_prelude::*;
use oidn_rs::registry::select_rt;
use oidn_rs::weights;

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

        /// Emit one JSON object per tensor instead of the human-readable table.
        #[arg(long, action = ArgAction::SetTrue)]
        json: bool,
    },

    /// Denoise an image (EXR / PFM / PHM / HDR / TIFF / PNG / JPG / BMP).
    Denoise(DenoiseArgs),

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
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "srgb")]
    hdr: bool,

    /// Input is LDR (linear in [0, 1]).
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "hdr")]
    ldr: bool,

    /// Input already in sRGB (skip the linear→sRGB conversion before display).
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

    /// Maximum memory budget in MB; routed to `RtFilterBuilder::max_memory_mb`.
    #[arg(long)]
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
        Cmd::Denoise(args) => denoise(args),
        Cmd::Bench { resolution, iters, quality, weights_dir, threads } => {
            if threads.is_some() {
                tracing::info!("--threads is a no-op on the wgpu backend");
            }
            bench(&resolution, iters, quality, &weights_dir)
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
            // Hand-rolled JSON to avoid a serde_json dep for one trivial use.
            let dims = t
                .desc
                .dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "{{\"name\":\"{name}\",\"dims\":[{dims}],\"layout\":\"{:?}\",\"dtype\":\"{:?}\"}}",
                t.desc.layout, t.desc.dtype
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

fn denoise(args: DenoiseArgs) -> Result<(), Box<dyn std::error::Error>> {
    // ---- argument validation (matches oidnDenoise.cpp:121-130 contract) ----
    if !args.hdr && !args.ldr {
        return Err(
            "must specify one of --hdr or --ldr (matches reference oidnDenoise behaviour)".into(),
        );
    }
    if args.clean_aux && (args.albedo.is_none() || args.normal.is_none()) {
        return Err("--clean_aux requires both --albedo and --normal".into());
    }
    if args.filter == FilterKind::RtLightmap && (args.albedo.is_some() || args.normal.is_some()) {
        return Err("--filter RTLightmap does not accept --albedo / --normal".into());
    }
    if args.threads.is_some() {
        tracing::info!("--threads is a no-op on the wgpu backend");
    }

    let device = WgpuDevice::new()?;

    let (color_pixels, w, h) = io::load_rgb_f32(&args.input)?;
    let albedo_pixels = args.albedo.as_deref().map(io::load_rgb_f32).transpose()?;
    let normal_pixels = args.normal.as_deref().map(io::load_rgb_f32).transpose()?;

    // Resolve weights: explicit blob > weights_dir > embedded fallback.
    let user_weights: Option<Vec<u8>> = if let Some(p) = args.weights.as_deref() {
        Some(std::fs::read(p)?)
    } else if args.filter == FilterKind::Rt {
        let base = select_rt(
            true,
            albedo_pixels.is_some(),
            normal_pixels.is_some(),
            args.hdr,
            args.srgb,
            args.clean_aux,
            args.quality,
        )?;
        weights::resolve(&base, args.quality, args.weights_dir.as_deref()).map(|(stem, bytes)| {
            tracing::info!("resolved model stem `{stem}` ({} bytes)", bytes.len());
            bytes
        })
    } else {
        // RTLightmap: explicit blob handled in the `if let Some(...)` arm
        // above; otherwise the filter resolves its weights from
        // `weights_dir` at commit time.
        None
    };
    let weights_dir = args.weights_dir.clone().unwrap_or_else(|| PathBuf::from("data/weights"));

    match args.filter {
        FilterKind::Rt => run_rt(
            &device,
            &weights_dir,
            user_weights,
            &args,
            &color_pixels,
            albedo_pixels.as_ref(),
            normal_pixels.as_ref(),
            w,
            h,
        )?,
        FilterKind::RtLightmap => run_rtlightmap(
            &device,
            &weights_dir,
            user_weights,
            &args,
            &color_pixels,
            w,
            h,
        )?,
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_rt(
    device: &WgpuDevice,
    weights_dir: &Path,
    user_weights: Option<Vec<u8>>,
    args: &DenoiseArgs,
    color: &[f32],
    albedo: Option<&(Vec<f32>, usize, usize)>,
    normal: Option<&(Vec<f32>, usize, usize)>,
    w: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = RtFilter::<WgpuBackend>::builder(&device.handle, weights_dir)
        .hdr(args.hdr)
        .srgb(args.srgb)
        .clean_aux(args.clean_aux)
        .quality(args.quality)
        .input_scale(args.input_scale);
    if let Some(mb) = args.maxmem {
        builder = builder.max_memory_mb(mb);
    }
    if let Some(bytes) = user_weights {
        builder = builder.weights(bytes);
    }
    let mut filter = builder.build();

    let color_img = Image::from_rgb_f32(color, w, h);
    filter.set_color(&color_img);

    let albedo_img = albedo.map(|(buf, w, h)| Image::from_rgb_f32(buf, *w, *h));
    if let Some(img) = &albedo_img {
        filter.set_albedo(img);
    }
    let normal_img = normal.map(|(buf, w, h)| Image::from_rgb_f32(buf, *w, *h));
    if let Some(img) = &normal_img {
        filter.set_normal(img);
    }

    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit()?;
    if let Some(k) = filter.model_key() {
        tracing::info!("model: {}", k.0);
    }

    for i in 0..args.iters.max(1) {
        let t0 = std::time::Instant::now();
        filter.execute()?;
        tracing::info!("iter {}: {:.2} ms", i, t0.elapsed().as_secs_f64() * 1000.0);
    }

    let (raw, ow, oh, fmt) = filter.take_output().ok_or("no output")?;
    debug_assert_eq!(fmt, PixelFormat::Rgb32f);
    let out_pixels: &[f32] = bytemuck::cast_slice(&raw);
    io::save_rgb_f32(&args.output, out_pixels, ow, oh)?;

    if let Some(ref_path) = args.reference.as_deref() {
        compare_against_reference(ref_path, out_pixels, ow, oh, args.maxerror)?;
    }

    eprintln!("wrote {}", args.output.display());
    Ok(())
}

fn run_rtlightmap(
    device: &WgpuDevice,
    weights_dir: &Path,
    user_weights: Option<Vec<u8>>,
    args: &DenoiseArgs,
    color: &[f32],
    w: usize,
    h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = RtLightmapFilter::<WgpuBackend>::builder(&device.handle, weights_dir)
        .directional(args.directional)
        .quality(args.quality)
        .input_scale(args.input_scale);
    if let Some(bytes) = user_weights {
        builder = builder.weights(bytes);
    }
    let mut filter = builder.build();

    let color_img = Image::from_rgb_f32(color, w, h);
    filter.set_color(&color_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit()?;
    if let Some(k) = filter.model_key() {
        tracing::info!("model: {}", k.0);
    }
    for i in 0..args.iters.max(1) {
        let t0 = std::time::Instant::now();
        filter.execute()?;
        tracing::info!("iter {}: {:.2} ms", i, t0.elapsed().as_secs_f64() * 1000.0);
    }
    let (raw, ow, oh, fmt) = filter.take_output().ok_or("no output")?;
    debug_assert_eq!(fmt, PixelFormat::Rgb32f);
    let out_pixels: &[f32] = bytemuck::cast_slice(&raw);
    io::save_rgb_f32(&args.output, out_pixels, ow, oh)?;
    if let Some(ref_path) = args.reference.as_deref() {
        compare_against_reference(ref_path, out_pixels, ow, oh, args.maxerror)?;
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
) -> Result<(), Box<dyn std::error::Error>> {
    let (ref_pixels, rw, rh) = io::load_rgb_f32(ref_path)?;
    if rw != w || rh != h {
        return Err(format!(
            "reference image is {rw}x{rh}, output is {w}x{h}",
        )
        .into());
    }
    let mut sse = 0.0f64;
    let mut maxe = 0.0f32;
    for (a, b) in out_pixels.iter().zip(ref_pixels.iter()) {
        let d = a - b;
        sse += (d * d) as f64;
        if d.abs() > maxe {
            maxe = d.abs();
        }
    }
    let mse = (sse / out_pixels.len() as f64) as f32;
    let psnr = if mse > 0.0 { 10.0 * (1.0 / mse).log10() } else { f32::INFINITY };
    println!("compare: mse={mse:.6e} psnr={psnr:.2} dB max={maxe:.6e}");
    if let Some(thr) = maxerror {
        if mse > thr {
            return Err(format!("MSE {mse:.6e} exceeds --maxerror {thr:.6e}").into());
        }
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
    let (w, h) = s.split_once('x').ok_or("resolution must be WxH (e.g. 1024x1024)")?;
    Ok((w.parse()?, h.parse()?))
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
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = parse_resolution(resolution)?;
    let device = WgpuDevice::new()?;

    // Generate a deterministic noisy HDR image.
    let mut color = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = 0.5 + 0.3 * ((x + y) as f32 / (w + h) as f32);
            let n = ((x * 17 + y * 31) % 19) as f32 * 0.04;
            let i = (y * w + x) * 3;
            color[i] = v + n;
            color[i + 1] = v + n * 0.7;
            color[i + 2] = v + n * 0.4;
        }
    }
    let color_img = Image::from_rgb_f32(&color, w, h);

    let mut filter = RtFilter::<WgpuBackend>::builder(&device.handle, weights_dir)
        .hdr(true)
        .quality(quality)
        .build();
    filter.set_color(&color_img);
    filter.allocate_output(w, h, PixelFormat::Rgb32f);
    filter.commit()?;

    eprintln!(
        "bench: {w}x{h}, quality={:?}, model={}",
        quality,
        filter.model_key().unwrap().0
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

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times_ms.first().copied().unwrap_or(0.0);
    let max = times_ms.last().copied().unwrap_or(0.0);
    let avg: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let med = times_ms[times_ms.len() / 2];
    let mp = (w * h) as f64 / 1_000_000.0;

    println!("resolution={w}x{h} ({mp:.2} MP) quality={:?} iters={iters}", quality);
    println!("  min={min:>8.2} ms  median={med:>8.2} ms  avg={avg:>8.2} ms  max={max:>8.2} ms");
    println!("  throughput @ median: {:.2} MP/s", mp / (med / 1000.0));

    Ok(())
}
