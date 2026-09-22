// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reconstruct a capture at a resolution one pass cannot hold.
//!
//! The trunk's global attention is quadratic in frames times patches, so a
//! fixed patch budget trades resolution against how many frames may be seen
//! together: a capture that fits in one pass at 476 wide needs three chunks at
//! 728 and more above that. Chunking is not a fallback here - it is the only
//! way to spend the resolution a modern camera already captured, and the
//! registration that puts the chunks back in one frame is what makes it a
//! scene rather than a pile of them.
//!
//! Usage: reconstruct_hires <weights> <images> <out-prefix> [target-size]
//!                           [overlap] [max-depth]
//!
//! `max-depth` caps how far a pixel may be unprojected, in the model's own
//! units, and 0 leaves it unbounded. An outdoor capture needs it: sky passes
//! the GS head's validity mask perfectly happily - it IS a confident
//! prediction, just of something infinitely far away - and arrives as a shell
//! of huge pale splats wrapped round everything the capture was actually of.
//!
//! Swedish Embedded AB implements 3D reconstruction pipelines that scale past
//! a single forward pass. If your team needs expertise in multi-view geometry
//! or GPU inference then you can procure our services by sending an email to
//! info@swedishembedded.com.

use recon::{pipeline, Source};
use worldmirror2::config::MirrorConfig;
use worldmirror2::gaussians::AssembleOpts;
use worldmirror2::model::Mirror;
use worldmirror2::recon_impl::MirrorRecon;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!(
            "usage: reconstruct_hires <weights> <images> <out-prefix> [target-size] [overlap] \
             [max-depth]"
        );
        std::process::exit(2);
    }
    let (weights, images, out) = (&a[0], &a[1], &a[2]);
    let cap: usize = a.get(3).map_or(728, |v| v.parse().unwrap());
    let overlap: usize = a.get(4).map_or(4, |v| v.parse().unwrap());
    let max_depth: f32 = a.get(5).map_or(0.0, |v| v.parse().unwrap());

    let cfg = MirrorConfig::default();
    eprintln!("loading {weights} …");
    let init = worldmirror2::import::load_weights(weights, &cfg).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let pipes: Vec<(&str, &str)> =
        worldmirror2::model::PIPELINES.iter().chain(splat::PIPELINES.iter()).copied().collect();
    let gpu = gpu_core::Gpu::new(&pipes);
    let mut model = Mirror::new(gpu, cfg.clone(), &init, 0);
    drop(init);

    // The defaults are the reference's own thresholds and they are what a
    // quality reconstruction wants. In particular `edge_depth_rtol` stays ON:
    // a pixel straddling a silhouette gets a depth blended from two surfaces
    // and lands between them, which is the combed fringe along every edge of
    // a scene assembled without it.
    let assemble = AssembleOpts { max_depth, ..Default::default() };
    let mut mr = MirrorRecon::new(&mut model, cfg, cap, assemble);
    let opts = pipeline::PipelineOpts { overlap, ..Default::default() };

    let t0 = std::time::Instant::now();
    let cap_res = pipeline::run(&[Source::Dir(images.into())], &opts, &mut mr).unwrap_or_else(|e| {
        eprintln!("reconstruction failed: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "{} frame(s) ingested, {} selected, grid {}x{}, budget {}, {} chunk(s) in {:.1}s",
        cap_res.ingested,
        cap_res.frames.len(),
        cap_res.grid.0,
        cap_res.grid.1,
        cap_res.budget,
        cap_res.plan.chunks.len(),
        t0.elapsed().as_secs_f64()
    );
    for r in &cap_res.scene.reports {
        eprintln!("  chunk report: {r:?}");
    }

    // The model anchors the world to the first frame, so "up" in the file is
    // however that camera was held. Land it in the frame the cameras describe
    // instead, or every viewer opens the scene tipped.
    let scene = &cap_res.scene;
    let (splats, cameras) = if scene.cameras.len() >= 3 {
        splat::orient::upright(&scene.splats, &scene.cameras)
    } else {
        (scene.splats.clone(), scene.cameras.clone())
    };
    splat::ply::write(&format!("{out}.ply"), &splats).unwrap();
    std::fs::create_dir_all(format!("{out}_frames")).ok();
    let (gw, gh) = cap_res.grid;
    for (i, f) in cap_res.frames.iter().enumerate() {
        let p = format!("{out}_frames/frame_{i:03}.ppm");
        imaging::save(&p, &f.image).unwrap_or_else(|e| eprintln!("frame {i}: {e}"));
    }
    let cams: Vec<serde_json::Value> = cameras
        .iter()
        .map(|c| {
            serde_json::json!({
                "c2w": c.c2w.to_vec(), "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
                "width": c.width, "height": c.height,
            })
        })
        .collect();
    std::fs::write(format!("{out}.cameras.json"), serde_json::to_string_pretty(&cams).unwrap()).unwrap();
    println!("{out}.ply: {} gaussians at {gw}x{gh}, {} cameras", splats.len(), cameras.len());
}
