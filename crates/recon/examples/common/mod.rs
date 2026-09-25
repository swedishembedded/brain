// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the reconstruction evaluations share: their command line, the fit
//! configuration it selects, and rendering a scene the way the fit saw it.
//!
//! Swedish Embedded AB implements photogrammetry and 3D reconstruction
//! pipelines and the evaluation that keeps them honest. If your team needs
//! that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use splat::opt::FitCfg;
use splat::types::RenderOpts;

/// Positional arguments, `--name value` options and bare `--name` switches.
pub struct Flags {
    pub positional: Vec<String>,
    named: Vec<(String, String)>,
}

impl Flags {
    pub fn from_env() -> Flags {
        let mut positional = Vec::new();
        let mut named = Vec::new();
        let mut it = std::env::args().skip(1).peekable();
        while let Some(v) = it.next() {
            match v.strip_prefix("--") {
                Some(name) => {
                    let value = it.next_if(|n| !n.starts_with("--")).unwrap_or_default();
                    named.push((name.to_string(), value));
                }
                None => positional.push(v),
            }
        }
        Flags { positional, named }
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.named.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    pub fn parse<T: std::str::FromStr>(&self, name: &str) -> Option<T> {
        self.get(name).map(|v| v.parse().unwrap_or_else(|_| panic!("--{name} {v}: not a valid value")))
    }

    pub fn arg<T: std::str::FromStr>(&self, i: usize, default: T) -> T {
        self.positional.get(i).map_or(default, |v| v.parse().unwrap_or_else(|_| panic!("argument {i} `{v}` is not valid")))
    }
}

/// The options every evaluation takes on top of the preset.
pub const FIT_USAGE: &str = "[--budget gaussians] [--sh degree] [--isp off|exposure|full] [--pose-after f] \
                             [--intrinsics-after f] [--distortion w] [--normal w] [--geometry-after f] \
                             [--lr-position s] [--max-scale-px px] [--densify heuristic|mcmc|hybrid] \
                             [--mip on|off] [--noise l] [--opacity-reg l] [--batch n] [--pyramid n] [--coarse f]";

/// [`FitCfg::from_sparse_points`] for `views` training views, with the
/// command line's overrides.
pub fn fit_cfg(flags: &Flags, iters: usize, views: usize) -> FitCfg {
    let budget = flags.parse("budget").unwrap_or(300_000);
    let mut cfg = FitCfg { log_every: (iters / 10).max(1), ..FitCfg::from_sparse_points(iters, budget, views) };
    if let Some(v) = flags.parse("sh") {
        cfg.sh_degree = v;
    }
    if let Some(v) = flags.parse("pose-after") {
        cfg.camera.pose_after = v;
    }
    if let Some(v) = flags.parse("intrinsics-after") {
        cfg.camera.intrinsics_after = v;
    }
    if let Some(v) = flags.parse("distortion") {
        cfg.distortion_weight = v;
    }
    if let Some(v) = flags.parse("normal") {
        cfg.normal_consistency_weight = v;
    }
    if let Some(v) = flags.parse("geometry-after") {
        cfg.geometry_after = v;
    }
    if let Some(v) = flags.parse("lr-position") {
        cfg.lr_position = v;
    }
    if let Some(v) = flags.parse("max-scale-px") {
        cfg.max_scale_pixels = v;
    }
    if let Some(v) = flags.parse("noise") {
        cfg.noise = v;
    }
    if let Some(v) = flags.parse("opacity-reg") {
        cfg.opacity_reg = v;
    }
    if let Some(v) = flags.parse("batch") {
        cfg.batch = v;
    }
    if let Some(v) = flags.parse("pyramid") {
        cfg.pyramid = v;
    }
    if let Some(v) = flags.parse("coarse") {
        cfg.coarse = v;
    }
    match flags.get("mip") {
        None => {}
        Some("on") => cfg.antialiased = true,
        Some("off") => cfg.antialiased = false,
        Some(o) => panic!("--mip {o}: on or off"),
    }
    match flags.get("densify") {
        None => {}
        Some("heuristic") => cfg.strategy = splat::opt::Densify::Heuristic,
        Some("mcmc") => cfg.strategy = splat::opt::Densify::Mcmc,
        Some("hybrid") => cfg.strategy = splat::opt::Densify::Hybrid,
        Some(o) => panic!("--densify {o}: heuristic, mcmc or hybrid"),
    }
    match flags.get("isp") {
        None => {}
        Some("off") => cfg.isp = None,
        Some("exposure") => {
            let mut i = cfg.isp.unwrap_or_default();
            i.vignetting_after = f32::INFINITY;
            i.response_after = f32::INFINITY;
            cfg.isp = Some(i);
        }
        Some("full") => cfg.isp = Some(Default::default()),
        Some(o) => panic!("--isp {o}: off, exposure or full"),
    }
    println!(
        "fit: {iters} iterations, budget {budget}, sh {}, poses after {}, intrinsics after {}, isp {}",
        cfg.sh_degree,
        cfg.camera.pose_after,
        cfg.camera.intrinsics_after,
        match cfg.isp {
            None => "off".to_string(),
            Some(i) => format!("vignetting after {}, response after {}", i.vignetting_after, i.response_after),
        }
    );
    cfg
}

/// How the fit rendered: its Mip filter setting, along rays.
pub fn fitted_opts(cfg: &FitCfg) -> RenderOpts {
    RenderOpts { antialiased: cfg.antialiased, eps2d: cfg.eps2d, ray: true, ..Default::default() }
}

/// Side-by-side columns of equally sized RGB images, as 8-bit.
pub fn montage(columns: &[&[f32]], w: u32, h: u32) -> imaging::Rgb8 {
    let (w, h, n) = (w as usize, h as usize, columns.len());
    let mut px = vec![0u8; w * n * h * 3];
    for (col, img) in columns.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    px[(y * w * n + col * w + x) * 3 + c] = (img[(y * w + x) * 3 + c].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
    }
    imaging::Rgb8 { w: (w * n) as u32, h: h as u32, px }
}
