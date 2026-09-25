// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sample application: a folder of photographs in, a 3D Gaussian Splatting
//! scene out, through the public `brain` SDK.
//!
//! No camera poses, no EXIF and no weights: structure from motion recovers
//! the cameras and their lenses from the photographs, multi-view stereo
//! measures the surfaces they show, and the fit grows a radiance field from
//! that dense start. Writes the scene as a standard splat PLY, the recovered
//! cameras, and a render of the first camera to compare with its photograph.
//!
//! ```text
//! make samples/reconstruction/splat/run ARGS="--photos ~/captures/scene --out out/scene.ply"
//! ```
//!
//! Swedish Embedded AB implements photogrammetry pipelines, from a folder of
//! photographs to calibrated cameras and a radiance field, for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// What the sample was asked to do, after argument parsing.
struct Args {
    photos: PathBuf,
    out: String,
    max_width: u32,
    iterations: Option<usize>,
    max_gaussians: Option<usize>,
    sparse: bool,
    camera_model: bool,
}

const USAGE: &str = "\
sample-reconstruction-splat - photographs to a 3D Gaussian Splatting scene

USAGE:
    sample-reconstruction-splat --photos DIR [--out PATH] [--max-width N]
                                [--iterations N] [--max-gaussians N]
                                [--sparse] [--camera-model]

OPTIONS:
    --photos DIR         the photographs (.jpg/.jpeg/.png) of one static scene
    --out PATH           the scene PLY (default: out/reconstruction.ply); the
                         cameras, a summary and a render of the first camera
                         are written next to it
    --max-width N        widest training image; photographs are halved exactly
                         until they fit (default: 2048)
    --iterations N       optimizer steps (default: from the photograph count)
    --max-gaussians N    the scene's gaussian budget (default: from the stereo)
    --sparse             start from structure from motion's points instead of
                         multi-view stereo
    --camera-model       fit per-photograph exposure and white balance too
";

fn parse() -> Result<Args, String> {
    let mut photos = None;
    let mut a = Args {
        photos: PathBuf::new(),
        out: "out/reconstruction.ply".into(),
        max_width: 2048,
        iterations: None,
        max_gaussians: None,
        sparse: false,
        camera_model: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag}: missing value"));
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--photos" => photos = Some(PathBuf::from(value()?)),
            "--out" => a.out = value()?,
            "--max-width" => a.max_width = value()?.parse().map_err(|e| format!("--max-width: {e}"))?,
            "--iterations" => a.iterations = Some(value()?.parse().map_err(|e| format!("--iterations: {e}"))?),
            "--max-gaussians" => a.max_gaussians = Some(value()?.parse().map_err(|e| format!("--max-gaussians: {e}"))?),
            "--sparse" => a.sparse = true,
            "--camera-model" => a.camera_model = true,
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    a.photos = photos.ok_or("--photos DIR is required: a folder of photographs of one static scene")?;
    Ok(a)
}

/// The photographs in `dir`, in name order.
fn photographs(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| ["jpg", "jpeg", "png"].contains(&x.to_ascii_lowercase().as_str()))
        })
        .collect();
    paths.sort();
    if paths.len() < 2 {
        return Err(format!("{}: {} photographs; a reconstruction needs at least two, and a good one dozens", dir.display(), paths.len()));
    }
    Ok(paths)
}

fn run(a: &Args) -> brain::Result<()> {
    let paths = photographs(&a.photos).map_err(brain::Error::Backend)?;
    println!("{} photographs from {}", paths.len(), a.photos.display());

    let mut b = brain::Reconstruction::builder()
        .photo_files(&paths)
        .max_width(a.max_width)
        .dense(!a.sparse)
        .camera_model(a.camera_model)
        .log(|line| println!("{line}"))
        .progress(|it, loss| {
            if it % 250 == 0 {
                println!("  step {it:6}: loss {loss:.5}");
            }
            true
        });
    if let Some(n) = a.iterations {
        b = b.iterations(n);
    }
    if let Some(n) = a.max_gaussians {
        b = b.max_gaussians(n);
    }
    let scene = b.run()?;

    println!(
        "{} of {} photographs placed (reprojection error {:.2} px), {} gaussians",
        scene.views(),
        paths.len(),
        scene.reprojection_rms_px(),
        scene.len()
    );
    for (i, p) in paths.iter().enumerate() {
        if !scene.registered().contains(&i) {
            println!("  not placed: {}", p.display());
        }
    }
    if let Some(dir) = Path::new(&a.out).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(brain::Error::Io)?;
    }
    let cameras = format!("{}.cameras.json", a.out);
    let preview = format!("{}.view0.png", a.out);
    let bundle = format!("{}.bundle", a.out);
    scene.save_ply(&a.out)?;
    scene.save_cameras(&cameras)?;
    scene.save(&bundle)?;
    scene.render(0)?.save(&preview)?;
    println!(
        "scene -> {}\ncameras -> {cameras}\neverything the pipeline measured -> {bundle}/\nfirst camera's view -> {preview} (compare with {})",
        a.out,
        paths[scene.registered()[0]].display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
