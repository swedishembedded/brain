// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `view` workflow's command line: `render`'s flags that shape a frame,
//! read through `render`'s own manifest so a value is typed and bounded as it
//! is there, plus `--frames` and `--shots`.
//!
//! Swedish Embedded AB implements real-time 3D scene viewers on its clients'
//! own hardware. If your team needs expertise in interactive rendering of
//! captured scenes then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::PathBuf;

use splat::types::{cameras_from_json, Camera, RenderOpts};

use crate::view::{fov_of, start_camera, Viewer};

/// Window size the viewer opens at when none is given.
const DEFAULT_SIZE: (u32, u32) = (1280, 720);
/// Background of the viewer, darker than a render's black-on-nothing.
const DEFAULT_BG: [f32; 3] = [0.02, 0.02, 0.03];
/// `render` flags that make no sense for a live window.
const NOT_FOR_VIEW: [&str; 3] = ["naive", "bench", "normalize_depth"];

fn take_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            let v = args.get(i + 1).cloned().ok_or_else(|| format!("{flag} needs a value"))?;
            args.drain(i..=i + 1);
            Ok(Some(v))
        }
        None => Ok(None),
    }
}

fn vec3(csv: &str, name: &str) -> Result<[f32; 3], String> {
    let v: Vec<f32> = csv.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    <[f32; 3]>::try_from(v).map_err(|_| format!("{name} wants x,y,z"))
}

pub fn view(call: &invoke::WorkflowCall) -> Result<(), String> {
    let mut rest: Vec<String> = call.args.to_vec();
    let frames = take_value(&mut rest, "--frames")?.map(|v| v.parse::<u64>().map_err(|_| format!("--frames {v}: not a count"))).transpose()?;
    let shots = PathBuf::from(take_value(&mut rest, "--shots")?.unwrap_or_else(|| ".".to_string()));
    if rest.first().is_some_and(|a| !a.starts_with('-')) {
        let scene = rest.remove(0);
        rest.splice(0..0, ["--scene".to_string(), scene]);
    }
    let render = call.provider.action("render").ok_or("the model has no render action")?.spec();
    let parsed = invoke::args::check(call.program, &render, &rest).map_err(|_| "see the flag error above".to_string())?;
    let inv = &parsed.invocation;
    if let Some(flag) = NOT_FOR_VIEW.iter().find(|f| inv.get_bool(f).is_some() || inv.get_i64(f).is_some()) {
        return Err(format!("--{} does not apply to a live window", flag.replace('_', "-")));
    }
    let input = |name: &str| parsed.inputs.iter().find(|(n, _)| n == name).map(|(_, p)| p.clone());
    let path = input("scene").ok_or("give the scene first: view <scene.ply>")?;
    let scene = splat::ply::read(&path)?;

    let start: Option<Camera> = match input("cameras") {
        Some(file) => {
            let cams = cameras_from_json(&std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?)?;
            let index = inv.get_i64("view").unwrap_or(0).max(0) as usize;
            Some(*cams.get(index).ok_or_else(|| format!("--view {index}: {file} holds {} cameras", cams.len()))?)
        }
        None => None,
    };
    let width = inv.get_i64("width").map_or(DEFAULT_SIZE.0, |v| v as u32);
    let height = inv.get_i64("height").map_or(DEFAULT_SIZE.1, |v| v as u32);
    // a camera from a file sets the field of view unless one is given
    let fov = inv.get_f64("fov").map(|v| v as f32).or_else(|| start.map(|c| fov_of(c.fy, c.height))).unwrap_or(60.0);
    let start = match (start, inv.get_str("eye"), inv.get_str("target")) {
        (Some(cam), None, None) => Some(cam),
        (None, Some(eye), Some(target)) => {
            let up = inv.get_str("up").map(|u| vec3(&u, "--up")).transpose()?.unwrap_or([0.0, -1.0, 0.0]);
            Some(Camera::look_at(vec3(&eye, "--eye")?, vec3(&target, "--target")?, up, fov, width, height))
        }
        (None, None, None) => None,
        _ => return Err("--eye and --target go together, and neither combines with --cameras".to_string()),
    };
    let inria = inv.get_bool("inria_lowpass").unwrap_or(false);
    let defaults = RenderOpts::default();
    let opts = RenderOpts {
        bg: inv.get_str("bg").map(|b| vec3(&b, "--bg")).transpose()?.unwrap_or(DEFAULT_BG),
        mode: if inv.get_bool("depth").unwrap_or(false) { splat::types::Mode::Depth } else { splat::types::Mode::Color },
        antialiased: inv.get_bool("antialiased").unwrap_or(defaults.antialiased) && !inria,
        eps2d: inv.get_f64("eps2d").map_or(defaults.eps2d, |v| v as f32),
        ..defaults
    };
    Viewer {
        title: format!("brain splat - {path}"),
        width,
        height,
        fov,
        opts,
        isect_cap: inv.get_i64("isect_cap").map_or(0, |v| v as usize),
        start: start_camera(&scene, start, width, height, fov),
        scene: &scene,
        max_frames: frames,
        shots_dir: shots,
    }
    .run()
}
