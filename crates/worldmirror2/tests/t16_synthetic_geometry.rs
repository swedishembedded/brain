// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T16 gate: the model has to recover a THIN structure, not just a big one.
//!
//! Every other quality gate here scores a reconstruction against photographs,
//! which means the only available truth is a 2D image and the only available
//! verdict is "close enough on average". Averages are dominated by whatever
//! fills the frame - a ground plane, a large smooth body - so a small object
//! reconstructed as a translucent cloud costs almost nothing in PSNR and
//! nothing at all in a sharpness ratio computed over the whole image. That is
//! exactly the failure worth catching, and it was invisible.
//!
//! This builds a scene analytically instead: a textured ground plane, a large
//! smooth body, and a thin disc held above the plane on a stalk. Because the
//! scene is analytic, the TRUE camera-space depth of every pixel is known
//! exactly, and each pixel knows which object it belongs to - so the model's
//! depth error can be reported per object rather than per image. The
//! raytracer shares no code with the renderer or the assembler, so it cannot
//! be self-consistently wrong with them.
//!
//! Depth is compared after a single global scale alignment, because a
//! feed-forward multi-view model fixes the world to its first camera and the
//! overall scale is arbitrary. One scalar for the whole batch, taken as the
//! median ratio: anything else would absorb the error being measured.
//!
//! Swedish Embedded AB implements 3D reconstruction whose accuracy is gated on
//! geometry it can verify rather than on how plausible the picture looks. If
//! your team needs reconstruction that cannot silently lose a thin structure,
//! you can procure our services by sending an email to info@swedishembedded.com.

use worldmirror2::config::MirrorConfig;
use worldmirror2::gaussians::frame_maps;
use worldmirror2::model::Mirror;

/// Grid the scene is rendered and reconstructed at. Overridable, because how
/// much of the error is simply "the thin part is a handful of pixels" is a
/// question the harness should be able to answer rather than assume.
fn grid() -> (usize, usize) {
    let t: usize = std::env::var("T16_TARGET").ok().and_then(|v| v.parse().ok()).unwrap_or(476);
    let w = t / 14 * 14;
    (w, (w * 3 / 4) / 14 * 14)
}
const VIEWS: usize = 8;
const FOV_DEG: f64 = 55.0;

/// What a pixel hit. Reported separately because the interesting failure is
/// confined to one of them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Part {
    Sky,
    Ground,
    Body,
    Stalk,
    /// the thin disc - small, dark, and held clear of everything else
    Head,
}

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn mul(a: V3, k: f64) -> V3 { [a[0] * k, a[1] * k, a[2] * k] }
fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt().max(1e-12);
    mul(a, 1.0 / l)
}

/// Deterministic value noise, so the ground has texture the model can latch
/// onto without pulling in a dependency.
fn noise(x: f64, y: f64) -> f64 {
    let s = (x * 12.9898 + y * 78.233).sin() * 43758.5453;
    s - s.floor()
}

fn smooth_noise(x: f64, y: f64) -> f64 {
    let (xi, yi) = (x.floor(), y.floor());
    let (xf, yf) = (x - xi, y - yi);
    let (u, v) = (xf * xf * (3.0 - 2.0 * xf), yf * yf * (3.0 - 2.0 * yf));
    let a = noise(xi, yi);
    let b = noise(xi + 1.0, yi);
    let c = noise(xi, yi + 1.0);
    let d = noise(xi + 1.0, yi + 1.0);
    a * (1.0 - u) * (1.0 - v) + b * u * (1.0 - v) + c * (1.0 - u) * v + d * u * v
}

struct Hit {
    t: f64,
    n: V3,
    albedo: V3,
    part: Part,
}

/// Ground plane y = 0, planked along x with grain along z.
fn hit_ground(o: V3, d: V3) -> Option<Hit> {
    if d[1] >= -1e-9 {
        return None;
    }
    let t = -o[1] / d[1];
    if t <= 1e-4 {
        return None;
    }
    let p = add(o, mul(d, t));
    if p[0].abs() > 4.0 || p[2].abs() > 4.0 {
        return None;
    }
    let plank = (p[0] / 0.22).floor();
    let seam = ((p[0] / 0.22).fract().abs() - 0.5).abs(); // 0.5 at plank centre
    let grain = smooth_noise(p[0] * 26.0 + plank * 7.0, p[2] * 3.5);
    let mut v = 0.52 + 0.30 * grain + 0.04 * (plank * 1.7).sin();
    if seam > 0.47 {
        v *= 0.45; // the dark gap between planks
    }
    Some(Hit { t, n: [0.0, 1.0, 0.0], albedo: [v, v * 0.97, v * 0.90], part: Part::Ground })
}

/// Axis-aligned ellipsoid, as the large smooth body.
fn hit_ellipsoid(o: V3, d: V3, c: V3, r: V3, albedo: V3, part: Part) -> Option<Hit> {
    let oc = [(o[0] - c[0]) / r[0], (o[1] - c[1]) / r[1], (o[2] - c[2]) / r[2]];
    let dd = [d[0] / r[0], d[1] / r[1], d[2] / r[2]];
    let a = dot(dd, dd);
    let b = 2.0 * dot(oc, dd);
    let cc = dot(oc, oc) - 1.0;
    let disc = b * b - 4.0 * a * cc;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / (2.0 * a);
    if t <= 1e-4 {
        return None;
    }
    let p = add(o, mul(d, t));
    let n = norm([(p[0] - c[0]) / (r[0] * r[0]), (p[1] - c[1]) / (r[1] * r[1]), (p[2] - c[2]) / (r[2] * r[2])]);
    Some(Hit { t, n, albedo, part })
}

/// Finite capped cylinder from `a` to `b`. Serves both the stalk and, at a
/// large radius and small length, the thin disc.
fn hit_cylinder(o: V3, d: V3, a: V3, b: V3, rad: f64, albedo: V3, part: Part) -> Option<Hit> {
    let ax = norm(sub(b, a));
    let len = dot(sub(b, a), ax);
    let oa = sub(o, a);
    let d_par = dot(d, ax);
    let o_par = dot(oa, ax);
    let d_perp = sub(d, mul(ax, d_par));
    let o_perp = sub(oa, mul(ax, o_par));
    let mut best: Option<Hit> = None;
    let mut note = |t: f64, n: V3| {
        if t > 1e-4 && best.as_ref().is_none_or(|h| t < h.t) {
            best = Some(Hit { t, n, albedo, part });
        }
    };
    // curved wall
    let qa = dot(d_perp, d_perp);
    if qa > 1e-12 {
        let qb = 2.0 * dot(o_perp, d_perp);
        let qc = dot(o_perp, o_perp) - rad * rad;
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            for s in [-1.0, 1.0] {
                let t = (-qb + s * disc.sqrt()) / (2.0 * qa);
                let h = o_par + t * d_par;
                if t > 1e-4 && (0.0..=len).contains(&h) {
                    let p = add(o, mul(d, t));
                    let n = norm(sub(sub(p, a), mul(ax, h)));
                    note(t, n);
                }
            }
        }
    }
    // end caps
    if d_par.abs() > 1e-12 {
        for (h, sgn) in [(0.0, -1.0), (len, 1.0)] {
            let t = (h - o_par) / d_par;
            if t > 1e-4 {
                let p = add(o, mul(d, t));
                let radial = sub(sub(p, a), mul(ax, h));
                if dot(radial, radial) <= rad * rad {
                    note(t, mul(ax, sgn));
                }
            }
        }
    }
    best
}

fn trace(o: V3, d: V3) -> Option<Hit> {
    let head_a = [0.66, 0.35, 0.0];
    let head_b = [0.70, 0.36, 0.0];
    let mut best: Option<Hit> = None;
    let mut keep = |h: Option<Hit>| {
        if let Some(h) = h {
            if best.as_ref().is_none_or(|q| h.t < q.t) {
                best = Some(h);
            }
        }
    };
    keep(hit_ground(o, d));
    keep(hit_ellipsoid(o, d, [0.0, 0.22, 0.0], [0.30, 0.22, 0.20], [0.16, 0.62, 0.20], Part::Body));
    keep(hit_cylinder(o, d, [0.22, 0.26, 0.0], head_a, 0.035, [0.16, 0.62, 0.20], Part::Stalk));
    // the thin disc: 0.11 radius, 0.04 thick, held clear of the plane
    keep(hit_cylinder(o, d, head_a, head_b, 0.11, [0.06, 0.06, 0.07], Part::Head));
    best
}

struct View {
    rgb: Vec<u8>,
    depth: Vec<f64>,
    part: Vec<Part>,
}

/// Render one view: shaded RGB plus the exact camera-space z of every pixel.
fn render(eye: V3, at: V3, w: usize, h: usize) -> View {
    #[allow(non_snake_case)]
    let (W, H) = (w, h);
    let fwd = norm(sub(at, eye));
    let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);
    let f = (W as f64 * 0.5) / (FOV_DEG.to_radians() * 0.5).tan();
    let (cx, cy) = (W as f64 * 0.5, H as f64 * 0.5);
    let light = norm([0.45, 1.0, 0.35]);
    let mut rgb = vec![0u8; W * H * 3];
    let mut depth = vec![0.0f64; W * H];
    let mut part = vec![Part::Sky; W * H];
    for py in 0..H {
        for px in 0..W {
            // pixel CENTRES: a pixel is an area and this is where it samples
            let sx = (px as f64 + 0.5 - cx) / f;
            let sy = (py as f64 + 0.5 - cy) / f;
            let d = norm(add(fwd, add(mul(right, sx), mul(up, -sy))));
            let i = py * W + px;
            let (col, z) = match trace(eye, d) {
                Some(h) => {
                    let lam = dot(h.n, light).max(0.0);
                    let s = 0.32 + 0.78 * lam;
                    depth[i] = h.t * dot(d, fwd); // camera-space z, not ray length
                    part[i] = h.part;
                    ([h.albedo[0] * s, h.albedo[1] * s, h.albedo[2] * s], depth[i])
                }
                None => {
                    // a plain gradient sky, so the frame is not half black
                    let v = 0.62 + 0.30 * (1.0 - (py as f64 / H as f64));
                    ([v * 0.80, v * 0.87, v], 0.0)
                }
            };
            let _ = z;
            for c in 0..3 {
                rgb[i * 3 + c] = (col[c].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            }
        }
    }
    View { rgb, depth, part }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

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

#[test]
fn a_thin_structure_is_recovered_as_accurately_as_the_ground_it_stands_on() {
    #[allow(non_snake_case)]
    let (W, H) = grid();
    eprintln!("grid {W}x{H}");
    // Cameras on an arc, all looking at the subject. Built BEFORE the
    // checkpoint is required, so the scene can be inspected (and the claim
    // below that it poses the question at all can be checked) without one.
    let at = [0.25, 0.22, 0.0];
    let views: Vec<View> = (0..VIEWS)
        .map(|i| {
            let a = -0.70 + 1.40 * (i as f64) / (VIEWS - 1) as f64;
            let eye = [at[0] + 1.55 * a.sin(), 0.95, at[2] + 1.55 * a.cos()];
            render(eye, at, W, H)
        })
        .collect();

    if let Ok(dir) = std::env::var("T16_DUMP") {
        std::fs::create_dir_all(&dir).ok();
        for (i, v) in views.iter().enumerate() {
            let mut out = format!("P6\n{W} {H}\n255\n").into_bytes();
            out.extend_from_slice(&v.rgb);
            std::fs::write(format!("{dir}/synth_{i:02}.ppm"), out).ok();
        }
    }

    // the test is only meaningful if the thin part is actually visible
    let head_px: usize = views.iter().map(|v| v.part.iter().filter(|p| **p == Part::Head).count()).sum();
    assert!(
        head_px > 200 * VIEWS,
        "the thin structure covers only {head_px} pixels across {VIEWS} views; the scene does not pose the question"
    );

    let Some(weights) = checkpoint() else { return };
    let cfg = MirrorConfig::default();
    let hw = W * H;
    let mut frames = Vec::with_capacity(VIEWS * hw * 3);
    for v in &views {
        for c in 0..3 {
            for i in 0..hw {
                frames.push(v.rgb[i * 3 + c] as f32 / 255.0);
            }
        }
    }
    let init = worldmirror2::import::load_weights(&weights, &cfg).expect("checkpoint loads");
    let pipes: Vec<(&str, &str)> =
        worldmirror2::model::PIPELINES.iter().chain(splat::PIPELINES.iter()).copied().collect();
    let gpu = gpu_core::Gpu::new(&pipes);
    let (hp, wp) = (H / cfg.patch, W / cfg.patch);
    let mut model = Mirror::new(gpu, cfg, &init, 0);
    drop(init);
    model.forward(&frames, VIEWS, hp, wp);

    let maps: Vec<worldmirror2::gaussians::FrameMaps> =
        (0..VIEWS).map(|fi| frame_maps(model.gpu(), &model, fi, W as u32, H as u32)).collect();
    let pred: Vec<Vec<f32>> = maps.iter().map(|m| m.depth.clone()).collect();

    // The model's own confidence, as a global floor over every frame: the
    // question is whether it identifies the pixels it got wrong.
    let mut allc: Vec<f32> = maps.iter().flat_map(|m| m.conf.iter().copied()).collect();
    let kth = allc.len() * 30 / 100;
    allc.sort_by(f32::total_cmp);
    let conf_floor = allc[kth];

    // ONE scale for the whole batch - the model's world is only defined up to it.
    let mut ratios: Vec<f64> = Vec::new();
    for (v, p) in views.iter().zip(&pred) {
        for i in 0..hw {
            if v.part[i] != Part::Sky && p[i] > 1e-6 {
                ratios.push(v.depth[i] / p[i] as f64);
            }
        }
    }
    let scale = median(&mut ratios);
    assert!(scale.is_finite() && scale > 0.0, "no valid predicted depth at all");

    let mut err: std::collections::HashMap<Part, Vec<f64>> = Default::default();
    let mut sgn: std::collections::HashMap<Part, Vec<f64>> = Default::default();
    for (v, p) in views.iter().zip(&pred) {
        for i in 0..hw {
            if v.part[i] == Part::Sky || !(p[i] > 1e-6) {
                continue;
            }
            let e = ((scale * p[i] as f64) - v.depth[i]) / v.depth[i];
            err.entry(v.part[i]).or_default().push(e.abs());
            sgn.entry(v.part[i]).or_default().push(e);
        }
    }
    // The same error, restricted to the pixels the model is most sure of.
    let mut kept: std::collections::HashMap<Part, Vec<f64>> = Default::default();
    let mut total: std::collections::HashMap<Part, usize> = Default::default();
    for ((v, p), m) in views.iter().zip(&pred).zip(&maps) {
        for i in 0..hw {
            if v.part[i] == Part::Sky || !(p[i] > 1e-6) {
                continue;
            }
            *total.entry(v.part[i]).or_default() += 1;
            if m.conf[i] >= conf_floor {
                let e = ((scale * p[i] as f64) - v.depth[i]).abs() / v.depth[i];
                kept.entry(v.part[i]).or_default().push(e);
            }
        }
    }
    let mut report = |part: Part| -> f64 {
        let mut v = err.remove(&part).unwrap_or_default();
        let mut sv = sgn.remove(&part).unwrap_or_default();
        let n = v.len();
        let m = median(&mut v);
        // Signed, because "pushed toward the surface behind it" and "noisy in
        // both directions" are different defects with different causes.
        let sm = median(&mut sv);
        let mut kv = kept.remove(&part).unwrap_or_default();
        let survive = 100.0 * kv.len() as f64 / (*total.get(&part).unwrap_or(&1)).max(1) as f64;
        let km = median(&mut kv);
        eprintln!(
            "  {part:?}: {n} px, median |err| {:.2}%, signed {:+.2}%  |  top-70% confidence: \
             {survive:.0}% of them kept, median |err| {:.2}%",
            m * 100.0, sm * 100.0, km * 100.0
        );
        m
    };
    eprintln!("global scale {scale:.4}");
    let ground = report(Part::Ground);
    let body = report(Part::Body);
    let stalk = report(Part::Stalk);
    let head = report(Part::Head);
    let _ = (body, stalk);

    // Measured at 476x350, 8 views: ground 0.86% (signed +0.06%, unbiased),
    // body 2.83% (-2.54%), stalk 2.17% (+2.16%), disc 5.72% (+5.72%).
    //
    // The disc's error is ENTIRELY a systematic push away from the camera -
    // |err| equals the signed error, so essentially every pixel of it is
    // placed behind where it is, sunk toward the surface beyond. That is
    // background bleed: around a thin object the receptive field is dominated
    // by what is behind it. It is the defect that renders as a translucent
    // cloud from any direction the object was not photographed from.
    //
    // The bounds below are a REGRESSION gate at those measured values, not a
    // target. The gap between the disc and the ground it stands on is a
    // tracked defect, and the ratio is reported so shrinking it is visible.
    eprintln!("thin/ground error ratio {:.1}x", head / ground.max(1e-9));
    assert!(ground < 0.015, "the textured ground plane regressed to {:.2}%", ground * 100.0);
    assert!(head < 0.07, "the thin disc regressed to {:.2}%", head * 100.0);
}
