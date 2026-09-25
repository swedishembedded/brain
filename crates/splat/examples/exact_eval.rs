// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Score a scene against the photographs it was built from, through the exact
//! cameras rather than a `look_at` approximation.
//!
//! A `look_at` rebuild of a camera is not the camera: forcing fx == fy alone
//! understated fitted scenes here by 18 dB, which is enough to make a working
//! change look like a regression and a broken one look fine. Reconstruction
//! quality is only a number worth quoting if it is measured through the pose
//! and intrinsics the scene was actually built on, so this reads them back.
use splat::quality::{psnr, sharpness_ratio};
use splat::renderer::{GpuSplats, Renderer};
use splat::types::RenderOpts;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (ply, cj, dir) = (&a[0], &a[1], &a[2]);
    // "inria" selects the uncompensated dilation at 0.3, "mip" the compensated
    // filter at 0.1, and a bare number an explicit uncompensated kernel size.
    let sel = a.get(3).cloned().unwrap_or_else(|| "inria".into());
    let s = splat::ply::read(ply).unwrap();
    let cams = splat::types::cameras_from_json(&std::fs::read_to_string(cj).unwrap()).unwrap();
    let g = gpu_core::Gpu::new(splat::PIPELINES);
    let ks = splat::Kernels::at(0);
    let o = match sel.as_str() {
        "inria" => RenderOpts { eps2d: 0.3, antialiased: false, ..Default::default() },
        "mip" => RenderOpts { eps2d: 0.1, antialiased: true, ..Default::default() },
        v => RenderOpts { eps2d: v.parse().expect("eps2d"), antialiased: false, ..Default::default() },
    };
    let mut ps = vec![];
    let mut ss = vec![];
    for (i, cam) in cams.into_iter().enumerate() {
        let p = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "ppm" || x == "png" || x == "jpg" || x == "jpeg"))
            .collect::<std::collections::BTreeSet<_>>().into_iter().nth(i).unwrap();
        let img = imaging::load(&p).unwrap();
        let want: Vec<f32> = img.px.iter().map(|&v| v as f32 / 255.0).collect();
        let mut r = Renderer::new(&g, ks, s.len(), cam.width, cam.height, s.len() * 24);
        let gs = GpuSplats::upload(&g, &s);
        r.render(&g, &gs, &cam, &o);
        let rgba = r.read_rgba(&g, cam.width, cam.height);
        let got: Vec<f32> = rgba.chunks_exact(4).flat_map(|q| [q[0], q[1], q[2]]).collect();
        if let Ok(d) = std::env::var("EXACT_EVAL_DUMP") {
            let px: Vec<u8> = got.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8).collect();
            let mut f = std::fs::File::create(format!("{d}/render_{i:02}.ppm")).unwrap();
            use std::io::Write;
            write!(f, "P6\n{} {}\n255\n", cam.width, cam.height).unwrap();
            f.write_all(&px).unwrap();
        }
        let (db, sh) = (psnr(&got, &want), sharpness_ratio(&got, &want, cam.width as usize, cam.height as usize));
        println!("  view {i}: PSNR {db:6.2} dB   sharpness {sh:.3}");
        ps.push(db); ss.push(sh);
    }
    println!("  MEAN : PSNR {:6.2} dB   sharpness {:.3}", ps.iter().sum::<f64>()/ps.len() as f64, ss.iter().sum::<f64>()/ss.len() as f64);
}
