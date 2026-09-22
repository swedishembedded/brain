// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resize a folder of frames to the grid a model ran them at.
//!
//! Scoring a scene means comparing a render against the photograph the model
//! actually saw, which is the photograph AFTER the model's own resize - not
//! the file on disk. Using a different resampler here would charge the scene
//! for the difference between two filters, so this uses the same torch-parity
//! bicubic the forward pass does.
//!
//! Usage: resize_frames <in-dir> <out-dir> <width> <height>

use worldmirror2::preprocess::resize_bicubic_torch;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("usage: resize_frames <in-dir> <out-dir> <width> <height>");
        std::process::exit(2);
    }
    let (ow, oh): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    std::fs::create_dir_all(&a[1]).ok();
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&a[0])
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ppm" || x == "png" || x == "jpg" || x == "jpeg"))
        .collect();
    paths.sort();
    for (i, p) in paths.iter().enumerate() {
        let img = imaging::load(p).unwrap();
        let (w, h) = (img.w as usize, img.h as usize);
        // CHW f32 in [0,1], which is the layout the resize is defined on
        let mut chw = vec![0.0f32; 3 * w * h];
        for c in 0..3 {
            for j in 0..w * h {
                chw[c * w * h + j] = img.px[j * 3 + c] as f32 / 255.0;
            }
        }
        let out = resize_bicubic_torch(&chw, 3, h, w, oh, ow);
        let mut px = vec![0u8; ow * oh * 3];
        for c in 0..3 {
            for j in 0..ow * oh {
                px[j * 3 + c] = (out[c * ow * oh + j].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        let dst = format!("{}/frame_{i:03}.ppm", a[1]);
        imaging::save(&dst, &imaging::Rgb8 { w: ow as u32, h: oh as u32, px }).unwrap();
    }
    println!("{} frame(s) -> {}/ at {ow}x{oh}", paths.len(), a[1]);
}
