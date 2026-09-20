// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end gate: photographs in, a scene out, and the scene has to LOOK
//! like the photographs it came from.
//!
//! Every other test here checks that the model runs - outputs are finite,
//! different frames differ, a tiny config does not read out of bounds. A
//! reconstruction that has collapsed into a smear passes all of them, because
//! a smear is finite and frame-dependent too. This one renders the assembled
//! scene from each camera the model RECOVERED and compares it with the
//! photograph that camera took.
//!
//! Two measures, because either alone can be satisfied by a failure:
//!
//! * PSNR catches geometry and colour going wrong, and is noisy across views -
//!   on the reference fixture it ranges 15.8 to 22.4 dB for the same scene,
//!   because how much a view is dominated by flat background varies.
//! * [`sharpness_ratio`] catches the reconstruction going soft, which PSNR
//!   barely registers, and is far steadier: 0.431 to 0.484 across those same
//!   views. A smear cannot hide in it.
//!
//! This is not hypothetical. Assembly took its opacity from the raw head
//! channel instead of the learned merge weight, and every default
//! reconstruction rendered at 12.5 dB carrying 0.071 - a milky smear that
//! nothing in the suite noticed. Both bounds below reject that by a wide
//! margin.
//!
//! The bands are a REGRESSION gate, not a claim of excellence. A feed-forward
//! reconstruction carries well under half the high-frequency content of the
//! photographs it was built from, and the band says so out loud rather than
//! rounding it up.
//!
//! Swedish Embedded AB implements multi-view reconstruction pipelines whose
//! output quality is gated automatically. If your team needs 3D reconstruction
//! that cannot silently degrade, you can procure our services by sending an
//! email to info@swedishembedded.com.

use splat::quality::{psnr, sharpness_ratio};
use splat::renderer::{GpuSplats, Renderer};
use splat::types::RenderOpts;
use worldmirror2::config::MirrorConfig;
use worldmirror2::gaussians::{assemble, AssembleOpts};
use worldmirror2::model::Mirror;

/// Calibrated on the committed fixture, feed-forward only (no `splat fit`).
/// Measured: 15.8-22.4 dB, sharpness 0.431-0.484. The broken assembly this
/// gate was written after scored 12.5 dB and 0.071.
const ACCURACY_FLOOR_DB: f64 = 14.5;
const SHARPNESS_BAND: std::ops::Range<f64> = 0.36..0.58;

fn checkpoint() -> Option<String> {
    let Some(p) = std::env::var("BRAIN_WORLDMIRROR2_WEIGHTS").ok().filter(|s| !s.is_empty()) else {
        brain_testutil::skip("set BRAIN_WORLDMIRROR2_WEIGHTS to an imported mirror.safetensors");
        return None;
    };
    if !std::path::Path::new(&p).exists() {
        brain_testutil::skip(&format!("{p} not found"));
        return None;
    }
    Some(p)
}

/// The reference photographs, already at the model's own 518x392 grid so the
/// test measures the reconstruction rather than a resampling of it.
fn photographs() -> Option<Vec<imaging::Rgb8>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/worldmirror2");
    let Ok(rd) = std::fs::read_dir(dir) else {
        brain_testutil::skip(&format!("{dir} absent (reference photographs are not committed here)"));
        return None;
    };
    let mut paths: Vec<_> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ppm" || x == "jpg" || x == "png"))
        .collect();
    paths.sort();
    if paths.len() < 2 {
        brain_testutil::skip(&format!("{dir} holds {} image(s); need at least 2", paths.len()));
        return None;
    }
    Some(paths.iter().map(|p| imaging::load(p).expect("readable reference photograph")).collect())
}

#[test]
fn the_scene_looks_like_the_photographs_it_was_built_from() {
    let (Some(weights), Some(photos)) = (checkpoint(), photographs()) else { return };
    let cfg = MirrorConfig::default();
    let (w, h) = (photos[0].w, photos[0].h);
    assert!(
        photos.iter().all(|p| (p.w, p.h) == (w, h)),
        "reference photographs differ in size; they must all be at the model's grid"
    );
    assert_eq!(
        (w as usize % cfg.patch, h as usize % cfg.patch),
        (0, 0),
        "{w}x{h} is not a multiple of the {}px patch grid", cfg.patch
    );

    // CHW, [0,1], frames concatenated - the layout `Mirror::forward` takes.
    let s = photos.len();
    let hw = (w * h) as usize;
    let mut frames = Vec::with_capacity(s * hw * 3);
    for p in &photos {
        for c in 0..3 {
            for i in 0..hw {
                frames.push(p.px[i * 3 + c] as f32 / 255.0);
            }
        }
    }

    let init = worldmirror2::import::load_weights(&weights, &cfg).expect("checkpoint loads");
    let pipes: Vec<(&str, &str)> =
        worldmirror2::model::PIPELINES.iter().chain(splat::PIPELINES.iter()).copied().collect();
    let gpu = gpu_core::Gpu::new(&pipes);
    let (hp, wp) = (h as usize / cfg.patch, w as usize / cfg.patch);
    let mut model = Mirror::new(gpu, cfg, &init, 0);
    drop(init);
    model.forward(&frames, s, hp, wp);
    let opts = AssembleOpts { min_opacity: 0.02, max_depth: 0.0, ..Default::default() };
    let (splats, cams, _) = assemble(model.gpu(), &model, &frames, s, w, h, &opts);
    assert_eq!(cams.len(), s, "one recovered camera per photograph");
    eprintln!("assembled {} gaussians", splats.len());
    for (i, c) in cams.iter().enumerate() {
        eprintln!(
            "cam{i}: fx {:.2} fy {:.2} cx {:.1} cy {:.1} {}x{} eye [{:.4} {:.4} {:.4}]",
            c.fx, c.fy, c.cx, c.cy, c.width, c.height, c.c2w[3], c.c2w[7], c.c2w[11]
        );
    }

    // Render on a Gpu built from splat's pipelines ALONE. The model's Gpu was
    // built from worldmirror2's list with splat's appended, so splat's kernels
    // do not start at slot 0 there and `Kernels::at(0)` would bind the wrong
    // layouts - which is what every other caller avoids by never sharing the
    // two. Dropping the model first also returns the checkpoint's memory
    // before a million gaussians are uploaded.
    drop(model);
    let g = &gpu_core::Gpu::new(splat::PIPELINES);
    let ks = splat::Kernels::at(0);
    // Every (gaussian, tile) pair must fit, or a dropped tail would show up as
    // missing detail and be blamed on the reconstruction.
    let mut ren = Renderer::new(g, ks, splats.len(), w, h, splats.len() * 16);
    let gs = GpuSplats::upload(g, &splats);
    let ro = RenderOpts::default();

    let (mut worst_db, mut worst_sharp) = (f64::INFINITY, (f64::INFINITY, 0.0f64));
    for (i, cam) in cams.iter().enumerate() {
        // the camera the model RECOVERED, not a look_at approximation of it
        ren.render(g, &gs, cam, &ro);
        let rgba = ren.read_rgba(g, w, h);
        let rgb: Vec<f32> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        let want: Vec<f32> = photos[i].px.iter().map(|&v| v as f32 / 255.0).collect();

        let db = psnr(&rgb, &want);
        let sharp = sharpness_ratio(&rgb, &want, w as usize, h as usize);

        // A number is enough to fail on and not enough to diagnose with, so
        // set BRAIN_DUMP_DIR and the frames come out side by side.
        if let Ok(dir) = std::env::var("BRAIN_DUMP_DIR") {
            let _ = std::fs::create_dir_all(&dir);
            let u8s = |v: &[f32]| -> Vec<u8> { v.iter().map(|x| (x.clamp(0.0, 1.0) * 255.0) as u8).collect() };
            for (tag, px) in [("render", u8s(&rgb)), ("photo", u8s(&want))] {
                let img = imaging::Rgb8::new(w, h, px).expect("frame");
                imaging::save(format!("{dir}/{i:02}-{tag}.png"), &img).expect("dump");
            }
        }
        eprintln!("photograph {i}: {db:.2} dB, sharpness {sharp:.3}");
        worst_db = worst_db.min(db);
        worst_sharp = (worst_sharp.0.min(sharp), worst_sharp.1.max(sharp));

        assert!(
            db > ACCURACY_FLOOR_DB,
            "photograph {i}: the scene renders at {db:.1} dB from the camera the model recovered \
             for it. The geometry or the colours are wrong, not merely soft."
        );
        assert!(
            SHARPNESS_BAND.contains(&sharp),
            "photograph {i}: the render carries {sharp:.3}x the photograph's high-frequency \
             content ({db:.1} dB). Below {} is a reconstruction that has gone soft - the failure \
             PSNR does not see; above {} is speckle or aliasing.",
            SHARPNESS_BAND.start, SHARPNESS_BAND.end
        );
    }
    eprintln!(
        "worldmirror2: {s} photographs, {} gaussians, worst {worst_db:.1} dB, sharpness {:.3}-{:.3}",
        splats.len(), worst_sharp.0, worst_sharp.1
    );
}
