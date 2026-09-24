// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scale-invariant keypoints and descriptors, implemented from Lowe,
//! "Distinctive Image Features from Scale-Invariant Keypoints" (IJCV 2004):
//!
//! 1. a gaussian scale space, `S = 3` intervals per octave from `σ0 = 1.6`,
//!    the input assumed pre-blurred by `0.5`;
//! 2. extrema of the difference-of-gaussians in 3x3x3 neighbourhoods,
//!    refined to sub-sample accuracy by a quadratic fit (§4), rejected when
//!    low-contrast or edge-like (principal-curvature ratio above 10, §4.1);
//! 3. one or more orientations per keypoint from a 36-bin gradient histogram
//!    (every peak within 80% of the highest, §5);
//! 4. the 4x4x8 descriptor of gradient orientations relative to that
//!    orientation, trilinearly binned under a gaussian window, normalized,
//!    clamped at 0.2 and renormalized (§6).
//!
//! Descriptors are returned as RootSIFT (Arandjelović & Zisserman, CVPR
//! 2012): L1-normalized then square-rooted, so the Euclidean distance between
//! two of them is the Hellinger kernel of the originals - a better matching
//! metric at no cost, and each descriptor has unit L2 norm, which lets a
//! matcher compare by dot product.
//!
//! The SIFT patent (US 6,711,293) expired in March 2020.

/// A detected feature in the coordinates of the image passed in, CONTINUOUS
/// pixel coordinates: pixel `i` spans `[i, i+1)` and its centre is `i + 0.5`,
/// the convention the camera model's principal point uses.
#[derive(Clone, Copy, Debug)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    /// Scale (gaussian sigma) in input pixels.
    pub sigma: f32,
    /// Orientation in radians.
    pub angle: f32,
    /// |DoG| response at the refined extremum.
    pub response: f32,
}

pub const DESC: usize = 128;

/// Detection settings.
#[derive(Clone, Copy, Debug)]
pub struct SiftCfg {
    /// Keep at most this many keypoints, strongest first.
    pub max_features: usize,
    /// Lowe's contrast threshold on the refined DoG value, for images in
    /// [0,1], before division by the interval count.
    pub contrast: f32,
    /// Principal-curvature ratio above which an extremum is an edge.
    pub edge_ratio: f32,
}

impl Default for SiftCfg {
    fn default() -> Self {
        SiftCfg { max_features: 8000, contrast: 0.04, edge_ratio: 10.0 }
    }
}

const S: usize = 3;
const SIGMA0: f32 = 1.6;
const INPUT_BLUR: f32 = 0.5;

struct Plane {
    w: usize,
    h: usize,
    px: Vec<f32>,
}

impl Plane {
    fn at(&self, x: usize, y: usize) -> f32 {
        self.px[y * self.w + x]
    }
}

fn kernel(sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    k
}

/// Separable gaussian blur, border replicated.
fn blur(p: &Plane, sigma: f32) -> Plane {
    let k = kernel(sigma);
    let r = (k.len() / 2) as isize;
    let (w, h) = (p.w, p.h);
    let mut tmp = vec![0.0f32; w * h];
    backend_cpu::par::rows_mut(&mut tmp, w, |y, row| {
        let line = &p.px[y * w..y * w + w];
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let sx = (x as isize + j as isize - r).clamp(0, w as isize - 1) as usize;
                acc += kv * line[sx];
            }
            *o = acc;
        }
    });
    let mut out = vec![0.0f32; w * h];
    backend_cpu::par::rows_mut(&mut out, w, |y, row| {
        for (j, kv) in k.iter().enumerate() {
            let sy = (y as isize + j as isize - r).clamp(0, h as isize - 1) as usize;
            let src = &tmp[sy * w..sy * w + w];
            for x in 0..w {
                row[x] += kv * src[x];
            }
        }
    });
    Plane { w, h, px: out }
}

fn half(p: &Plane) -> Plane {
    let (w, h) = (p.w / 2, p.h / 2);
    let mut px = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            px[y * w + x] = p.at(2 * x, 2 * y);
        }
    }
    Plane { w, h, px }
}

/// Detect keypoints and compute RootSIFT descriptors on a grayscale image
/// `gray` (`w*h`, values in [0,1]). Returns keypoints and descriptors
/// `[k*128]`.
pub fn detect(gray: &[f32], w: usize, h: usize, cfg: &SiftCfg) -> (Vec<Keypoint>, Vec<f32>) {
    assert_eq!(gray.len(), w * h);
    let base = Plane { w, h, px: gray.to_vec() };
    let first = (SIGMA0 * SIGMA0 - INPUT_BLUR * INPUT_BLUR).sqrt();
    let mut cur = blur(&base, first);
    let k = 2f32.powf(1.0 / S as f32);
    let steps: Vec<f32> = (1..S + 3)
        .map(|i| {
            let prev = SIGMA0 * k.powi(i as i32 - 1);
            let next = prev * k;
            (next * next - prev * prev).sqrt()
        })
        .collect();
    let mut found: Vec<(Keypoint, usize, f32)> = Vec::new(); // keypoint, octave, octave-local sigma
    let mut octaves: Vec<Vec<Plane>> = Vec::new();
    let mut octave = 0usize;
    while cur.w >= 16 && cur.h >= 16 {
        let mut gauss = vec![cur];
        for s in &steps {
            let next = blur(gauss.last().unwrap(), *s);
            gauss.push(next);
        }
        let dog: Vec<Plane> = gauss
            .windows(2)
            .map(|p| Plane { w: p[0].w, h: p[0].h, px: p[1].px.iter().zip(&p[0].px).map(|(a, b)| a - b).collect() })
            .collect();
        let scale = (1u32 << octave) as f32;
        for kp in extrema(&dog, cfg) {
            // (x, y, interval, response) in octave coordinates
            let local_sigma = SIGMA0 * 2f32.powf(kp.2 / S as f32);
            let g = &gauss[(kp.2.round() as usize).clamp(0, S + 2)];
            for angle in orientations(g, kp.0, kp.1, local_sigma) {
                found.push((
                    Keypoint { x: kp.0 * scale + 0.5, y: kp.1 * scale + 0.5, sigma: local_sigma * scale, angle, response: kp.3 },
                    octave,
                    local_sigma,
                ));
            }
        }
        let next = half(&gauss[S]);
        octaves.push(gauss);
        cur = next;
        octave += 1;
    }
    found.sort_by(|a, b| b.0.response.total_cmp(&a.0.response));
    found.truncate(cfg.max_features);
    let desc: Vec<Vec<f32>> = backend_cpu::par::map(found.len(), |i| {
        let (kp, o, ls) = found[i];
        let scale = (1u32 << o) as f32;
        let interval = (S as f32 * (ls / SIGMA0).log2()).round() as usize;
        descriptor(&octaves[o][interval.clamp(0, S + 2)], (kp.x - 0.5) / scale, (kp.y - 0.5) / scale, ls, kp.angle)
    });
    let mut flat = Vec::with_capacity(found.len() * DESC);
    for d in desc {
        flat.extend(d);
    }
    (found.into_iter().map(|f| f.0).collect(), flat)
}

/// DoG extrema of one octave, refined: `(x, y, interval, |response|)`.
fn extrema(dog: &[Plane], cfg: &SiftCfg) -> Vec<(f32, f32, f32, f32)> {
    let (w, h) = (dog[0].w, dog[0].h);
    let pre = 0.5 * cfg.contrast / S as f32;
    let mut out = Vec::new();
    for s in 1..=S {
        for y in 5..h.saturating_sub(5) {
            for x in 5..w.saturating_sub(5) {
                let v = dog[s].at(x, y);
                if v.abs() <= pre {
                    continue;
                }
                let mut is_max = true;
                let mut is_min = true;
                'n: for ds in 0..3 {
                    for dy in 0..3 {
                        for dx in 0..3 {
                            if ds == 1 && dy == 1 && dx == 1 {
                                continue;
                            }
                            let u = dog[s + ds - 1].at(x + dx - 1, y + dy - 1);
                            is_max &= v > u;
                            is_min &= v < u;
                            if !is_max && !is_min {
                                break 'n;
                            }
                        }
                    }
                }
                if !(is_max || is_min) {
                    continue;
                }
                if let Some(r) = refine(dog, x, y, s, cfg) {
                    out.push(r);
                }
            }
        }
    }
    out
}

/// Quadratic sub-sample refinement (Lowe §4) and the contrast and edge tests.
fn refine(dog: &[Plane], mut x: usize, mut y: usize, mut s: usize, cfg: &SiftCfg) -> Option<(f32, f32, f32, f32)> {
    let (w, h) = (dog[0].w, dog[0].h);
    for _ in 0..5 {
        let d = |ds: isize, dy: isize, dx: isize| dog[(s as isize + ds) as usize].at((x as isize + dx) as usize, (y as isize + dy) as usize);
        let g = [(d(0, 0, 1) - d(0, 0, -1)) / 2.0, (d(0, 1, 0) - d(0, -1, 0)) / 2.0, (d(1, 0, 0) - d(-1, 0, 0)) / 2.0];
        let v = d(0, 0, 0);
        let hxx = d(0, 0, 1) + d(0, 0, -1) - 2.0 * v;
        let hyy = d(0, 1, 0) + d(0, -1, 0) - 2.0 * v;
        let hss = d(1, 0, 0) + d(-1, 0, 0) - 2.0 * v;
        let hxy = (d(0, 1, 1) - d(0, 1, -1) - d(0, -1, 1) + d(0, -1, -1)) / 4.0;
        let hxs = (d(1, 0, 1) - d(1, 0, -1) - d(-1, 0, 1) + d(-1, 0, -1)) / 4.0;
        let hys = (d(1, 1, 0) - d(1, -1, 0) - d(-1, 1, 0) + d(-1, -1, 0)) / 4.0;
        let hm = [hxx as f64, hxy as f64, hxs as f64, hxy as f64, hyy as f64, hys as f64, hxs as f64, hys as f64, hss as f64];
        let det = crate::linalg::det(&hm);
        if det.abs() < 1e-12 {
            return None;
        }
        // offset = -H^-1 g
        let inv = inverse3(&hm, det);
        let off: [f64; 3] = std::array::from_fn(|r| -(inv[r * 3] * g[0] as f64 + inv[r * 3 + 1] * g[1] as f64 + inv[r * 3 + 2] * g[2] as f64));
        if off.iter().all(|o| o.abs() < 0.5) {
            let val = v as f64 + 0.5 * (g[0] as f64 * off[0] + g[1] as f64 * off[1] + g[2] as f64 * off[2]);
            if val.abs() < (cfg.contrast / S as f32) as f64 {
                return None;
            }
            let tr = (hxx + hyy) as f64;
            let dt = (hxx * hyy - hxy * hxy) as f64;
            let r = cfg.edge_ratio as f64;
            if dt <= 0.0 || tr * tr / dt >= (r + 1.0) * (r + 1.0) / r {
                return None;
            }
            return Some(((x as f64 + off[0]) as f32, (y as f64 + off[1]) as f32, (s as f64 + off[2]) as f32, val.abs() as f32));
        }
        let step = |p: usize, o: f64| (p as isize + o.round() as isize).max(0) as usize;
        x = step(x, off[0]);
        y = step(y, off[1]);
        s = step(s, off[2]);
        if s < 1 || s > S || x < 5 || y < 5 || x >= w - 5 || y >= h - 5 {
            return None;
        }
    }
    None
}

fn inverse3(m: &[f64; 9], det: f64) -> [f64; 9] {
    [
        (m[4] * m[8] - m[5] * m[7]) / det,
        (m[2] * m[7] - m[1] * m[8]) / det,
        (m[1] * m[5] - m[2] * m[4]) / det,
        (m[5] * m[6] - m[3] * m[8]) / det,
        (m[0] * m[8] - m[2] * m[6]) / det,
        (m[2] * m[3] - m[0] * m[5]) / det,
        (m[3] * m[7] - m[4] * m[6]) / det,
        (m[1] * m[6] - m[0] * m[7]) / det,
        (m[0] * m[4] - m[1] * m[3]) / det,
    ]
}

fn gradient(p: &Plane, x: usize, y: usize) -> (f32, f32) {
    let dx = p.at(x + 1, y) - p.at(x - 1, y);
    let dy = p.at(x, y + 1) - p.at(x, y - 1);
    ((dx * dx + dy * dy).sqrt(), dy.atan2(dx))
}

/// Dominant orientations (Lowe §5).
fn orientations(g: &Plane, x: f32, y: f32, sigma: f32) -> Vec<f32> {
    const BINS: usize = 36;
    let sw = 1.5 * sigma;
    let r = (3.0 * sw).round() as isize;
    let (cx, cy) = (x.round() as isize, y.round() as isize);
    let mut hist = [0.0f32; BINS];
    for dy in -r..=r {
        for dx in -r..=r {
            let (px, py) = (cx + dx, cy + dy);
            if px < 1 || py < 1 || px >= g.w as isize - 1 || py >= g.h as isize - 1 {
                continue;
            }
            let (m, a) = gradient(g, px as usize, py as usize);
            let wgt = (-((dx * dx + dy * dy) as f32) / (2.0 * sw * sw)).exp();
            let b = ((a + std::f32::consts::PI) / std::f32::consts::TAU * BINS as f32) as usize % BINS;
            hist[b] += wgt * m;
        }
    }
    for _ in 0..2 {
        let prev = hist;
        for i in 0..BINS {
            hist[i] = 0.25 * prev[(i + BINS - 1) % BINS] + 0.5 * prev[i] + 0.25 * prev[(i + 1) % BINS];
        }
    }
    let top = hist.iter().cloned().fold(0.0f32, f32::max);
    let mut out = Vec::new();
    for i in 0..BINS {
        let (l, c, rr) = (hist[(i + BINS - 1) % BINS], hist[i], hist[(i + 1) % BINS]);
        if c > l && c > rr && c >= 0.8 * top && top > 0.0 {
            let off = 0.5 * (l - rr) / (l - 2.0 * c + rr);
            let b = (i as f32 + 0.5 + off) / BINS as f32;
            out.push(b * std::f32::consts::TAU - std::f32::consts::PI);
        }
    }
    out
}

/// The 4x4x8 descriptor (Lowe §6), returned as RootSIFT.
fn descriptor(g: &Plane, x: f32, y: f32, sigma: f32, angle: f32) -> Vec<f32> {
    const NB: usize = 4;
    const NO: usize = 8;
    let hist_w = 3.0 * sigma;
    let radius = (hist_w * std::f32::consts::SQRT_2 * (NB as f32 + 1.0) * 0.5).round() as isize;
    let (c, s) = (angle.cos(), angle.sin());
    let mut hist = [0.0f32; NB * NB * NO];
    let (cx, cy) = (x.round() as isize, y.round() as isize);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            // rotate into the keypoint frame, in histogram-bin units
            let rx = (c * dx as f32 + s * dy as f32) / hist_w;
            let ry = (-s * dx as f32 + c * dy as f32) / hist_w;
            let bx = rx + NB as f32 / 2.0 - 0.5;
            let by = ry + NB as f32 / 2.0 - 0.5;
            if bx <= -1.0 || by <= -1.0 || bx >= NB as f32 || by >= NB as f32 {
                continue;
            }
            let (px, py) = (cx + dx, cy + dy);
            if px < 1 || py < 1 || px >= g.w as isize - 1 || py >= g.h as isize - 1 {
                continue;
            }
            let (m, a) = gradient(g, px as usize, py as usize);
            let wgt = (-(rx * rx + ry * ry) / (2.0 * (0.5 * NB as f32).powi(2))).exp();
            let mut o = (a - angle) / std::f32::consts::TAU * NO as f32;
            o = o.rem_euclid(NO as f32);
            let (x0, y0, o0) = (bx.floor(), by.floor(), o.floor());
            let (fx, fy, fo) = (bx - x0, by - y0, o - o0);
            for (iy, wy) in [(y0 as isize, 1.0 - fy), (y0 as isize + 1, fy)] {
                if iy < 0 || iy >= NB as isize {
                    continue;
                }
                for (ix, wx) in [(x0 as isize, 1.0 - fx), (x0 as isize + 1, fx)] {
                    if ix < 0 || ix >= NB as isize {
                        continue;
                    }
                    for (io, wo) in [(o0 as usize % NO, 1.0 - fo), ((o0 as usize + 1) % NO, fo)] {
                        hist[(iy as usize * NB + ix as usize) * NO + io] += m * wgt * wx * wy * wo;
                    }
                }
            }
        }
    }
    let n = hist.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    for v in &mut hist {
        *v = (*v / n).min(0.2);
    }
    let l1: f32 = hist.iter().sum::<f32>().max(1e-12);
    hist.iter().map(|v| (v / l1).sqrt()).collect()
}
