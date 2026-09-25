// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the reconstruction evaluations share: their command line, the fit
//! configuration it selects, and rendering a scene the way the fit saw it.
//!
//! Swedish Embedded AB implements photogrammetry and 3D reconstruction
//! pipelines and the evaluation that keeps them honest. If your team needs
//! that, you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::Gpu;
use splat::opt::FitCfg;
use splat::renderer::{rgba_to_rgb, GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;

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
pub const FIT_USAGE: &str = "[--budget gaussians] [--sh degree] [--isp off|exposure|full] [--pose-lr step] \
                             [--distortion w] [--normal w] [--geometry-after f] [--position-budget r] \
                             [--scale-budget r] [--max-scale-px px] [--densify heuristic|mcmc|hybrid]";

/// [`FitCfg::from_sparse_points`] for `views` training views, with the
/// command line's overrides.
pub fn fit_cfg(flags: &Flags, iters: usize, views: usize) -> FitCfg {
    let budget = flags.parse("budget").unwrap_or(300_000);
    let mut cfg = FitCfg { log_every: (iters / 10).max(1), ..FitCfg::from_sparse_points(iters, budget, views) };
    if let Some(v) = flags.parse("sh") {
        cfg.sh_degree = v;
    }
    if let Some(v) = flags.parse("pose-lr") {
        cfg.pose_lr = v;
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
    if let Some(v) = flags.parse("position-budget") {
        cfg.position_budget = v;
    }
    if let Some(v) = flags.parse("scale-budget") {
        cfg.scale_budget = v;
    }
    if let Some(v) = flags.parse("max-scale-px") {
        cfg.max_scale_pixels = v;
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
        "fit: {iters} iterations, budget {budget}, sh {}, pose lr {}, isp {}",
        cfg.sh_degree,
        cfg.pose_lr,
        match cfg.isp {
            None => "off".to_string(),
            Some(i) => format!("vignetting after {}, response after {}", i.vignetting_after, i.response_after),
        }
    );
    cfg
}

/// `s` from `c`, SH evaluated for that camera, as RGB `[w*h*3]`.
pub fn render(g: &Gpu, s: &Splats, c: &Camera, o: &RenderOpts) -> Vec<f32> {
    let mut r = Renderer::new(g, Kernels::at(0), s.len().max(1), c.width, c.height, 0).growable();
    let gs = GpuSplats::upload(g, s);
    if let Some(col) = splat::sh::shade(s, c.eye()) {
        g.write_f32(&gs.colors, &col);
    }
    r.render(g, &gs, c, o);
    rgba_to_rgb(&r.read_rgba(g, c.width, c.height))
}

/// How the fit rendered: its Mip filter setting.
pub fn fitted_opts(cfg: &FitCfg) -> RenderOpts {
    RenderOpts { antialiased: cfg.antialiased, eps2d: cfg.eps2d, ..Default::default() }
}

/// PSNR over the pixels whose `mask` (`[w*h]`, `None` = all) is at least 0.5.
pub fn psnr_masked(a: &[f32], b: &[f32], mask: Option<&[f32]>) -> f64 {
    let (mut se, mut n) = (0.0f64, 0usize);
    for (p, (x, y)) in a.chunks_exact(3).zip(b.chunks_exact(3)).enumerate() {
        if mask.is_some_and(|m| m[p] < 0.5) {
            continue;
        }
        se += (0..3).map(|c| ((x[c] - y[c]) as f64).powi(2)).sum::<f64>();
        n += 3;
    }
    if se <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (n as f64 / se).log10()
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
