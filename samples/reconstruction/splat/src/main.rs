// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sample application: a folder of photographs in, a 3D Gaussian Splatting
//! scene out, through the public `brain` SDK.
//!
//! No camera poses, no EXIF and no weights: structure from motion recovers
//! the cameras from the photographs, and the fit grows the sparse points it
//! triangulated into the scene. Writes the scene as a standard splat PLY, the
//! recovered cameras, and a render of the first camera to compare with its
//! photograph.
//!
//! ```text
//! make samples/reconstruction/splat/run ARGS="--photos ~/captures/can --out can.ply"
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
    width: u32,
    iterations: usize,
    max_gaussians: usize,
}

const USAGE: &str = "\
sample-reconstruction-splat - photographs to a 3D Gaussian Splatting scene

USAGE:
    sample-reconstruction-splat --photos DIR [--out PATH] [--width N]
                                [--iterations N] [--max-gaussians N]

OPTIONS:
    --photos DIR         the photographs (.jpg/.jpeg/.png), one camera, static scene
    --out PATH           the scene PLY (default: out/reconstruction.ply); the
                         cameras and a render of the first one are written next to it
    --width N            training resolution across (default: 1024)
    --iterations N       optimizer steps (default: 3000)
    --max-gaussians N    the scene's gaussian budget (default: 500000)
";

fn parse() -> Result<Args, String> {
    let mut photos = None;
    let mut a = Args { photos: PathBuf::new(), out: "out/reconstruction.ply".into(), width: 1024, iterations: 3000, max_gaussians: 500_000 };
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
            "--width" => a.width = value()?.parse().map_err(|e| format!("--width: {e}"))?,
            "--iterations" => a.iterations = value()?.parse().map_err(|e| format!("--iterations: {e}"))?,
            "--max-gaussians" => a.max_gaussians = value()?.parse().map_err(|e| format!("--max-gaussians: {e}"))?,
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
    let photos = paths.iter().map(brain::Image::open).collect::<brain::Result<Vec<_>>>()?;
    println!("{} photographs from {}", photos.len(), a.photos.display());

    let every = (a.iterations / 20).max(1);
    let scene = brain::Reconstruction::builder()
        .photos(photos)
        .width(a.width)
        .iterations(a.iterations)
        .max_gaussians(a.max_gaussians)
        .progress(move |it, loss| {
            if it % every == 0 {
                println!("  step {it:5}: loss {loss:.4}");
            }
            true
        })
        .run()?;

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
    scene.save_ply(&a.out)?;
    scene.save_cameras(&cameras)?;
    scene.render(0)?.save(&preview)?;
    println!("scene -> {}\ncameras -> {cameras}\nfirst camera's view -> {preview} (compare with {})", a.out, paths[scene.registered()[0]].display());
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
