// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Level a scene that came out tipped, without re-running anything.
//!
//! `orient`'s camera-derived up is only as good as the path the photographer
//! walked; the surface the capture is OF is a better witness. This reads the
//! scene's own dominant surface and takes the residual tilt out, which costs
//! one pass over the gaussians rather than another forward pass.
//!
//! Usage: level_scene <in.ply> <cameras.json> <out.ply>

use splat::types::Camera;

fn read_cameras(path: &str) -> Vec<Camera> {
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    j.as_array().unwrap().iter().map(|c| Camera {
        c2w: c["c2w"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<f32>>().try_into().unwrap(),
        fx: c["fx"].as_f64().unwrap() as f32, fy: c["fy"].as_f64().unwrap() as f32,
        cx: c["cx"].as_f64().unwrap() as f32, cy: c["cy"].as_f64().unwrap() as f32,
        width: c["width"].as_u64().unwrap() as u32, height: c["height"].as_u64().unwrap() as u32,
    }).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!("usage: level_scene <in.ply> <cameras.json> <out.ply>");
        std::process::exit(2);
    }
    let s = splat::ply::read(&a[0]).unwrap();
    let cams = read_cameras(&a[1]);
    let before = splat::orient::dominant_normal(&s, [0.0, -1.0, 0.0]);
    let (out, cm) = splat::orient::level(&s, &cams);
    let after = splat::orient::dominant_normal(&out, [0.0, -1.0, 0.0]);
    let tilt = |n: Option<[f64; 3]>| {
        n.map(|v| v[1].abs().clamp(0.0, 1.0).acos().to_degrees()).unwrap_or(f64::NAN)
    };
    println!("tilt of the dominant surface: {:.2} deg -> {:.2} deg", tilt(before), tilt(after));
    splat::ply::write(&a[2], &out).unwrap();
    let js: Vec<serde_json::Value> = cm
        .iter()
        .map(|c| serde_json::json!({
            "c2w": c.c2w.to_vec(), "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
            "width": c.width, "height": c.height,
        }))
        .collect();
    std::fs::write(format!("{}.cameras.json", a[2]), serde_json::to_string_pretty(&js).unwrap()).unwrap();
    println!("{}: {} gaussians", a[2], out.len());
}
