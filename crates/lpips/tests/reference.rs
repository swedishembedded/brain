// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LPIPS on the device against the reference.
//!
//! Two independent references:
//!
//! * an f64 transcription of `lpips/lpips.py` and torchvision's AlexNet in
//!   this file (scaling layer, conv-ReLU-pool trunk, channel unit
//!   normalization with the reference's `+1e-10` on the norm, squared
//!   difference, 1x1 heads, spatial mean, sum over taps). It shares no math
//!   with `lpips::model` on purpose - an oracle that calls the code it
//!   checks proves nothing (AGENTS.md's gradcheck-oracle exception) - only
//!   the weight loading, which `lpips::import` validates by shape;
//! * the official `lpips` package's own distances on the LPIPS repository's
//!   example images, dumped by `tools/goldens/lpips_dump_reference.py` into
//!   `testdata/lpips/`.
//!
//! Every test needs the two weight files in the model store (the same tool
//! puts them there) and skips, by name, without them.

use gpu_core::Gpu;
use lpips::config::{head_weight, trunk_bias, trunk_weight, EPS, POOL_K, POOL_STRIDE, SCALE, SHIFT, TRUNK};
use lpips::import::Tensors;
use lpips::Lpips;

const DUMPER: &str = "tools/goldens/lpips_dump_reference.py";

fn weights() -> Option<Tensors> {
    let found = lpips::spec::resolve().and_then(|(trunk, heads)| lpips::import::read(&trunk, &heads));
    match found {
        Ok(w) => Some(w),
        Err(e) => {
            brain_testutil::skip(&format!("LPIPS weights not in the model store ({e}); run {DUMPER}"));
            None
        }
    }
}

fn metric(w: &Tensors) -> Lpips {
    let g: Gpu = gpu_core::testgpu::dev(lpips::PIPELINES);
    Lpips::new(g, w).expect("lpips on the test device")
}

// ---------------------------------------------------------------------------
// The f64 reference.
// ---------------------------------------------------------------------------

/// A `[c, h, w]` map.
struct Map {
    c: usize,
    h: usize,
    w: usize,
    v: Vec<f64>,
}

fn conv_relu(x: &Map, wt: &[f32], b: &[f32], cout: usize, k: usize, stride: usize, pad: usize) -> Map {
    let (ho, wo) = ((x.h + 2 * pad - k) / stride + 1, (x.w + 2 * pad - k) / stride + 1);
    let mut v = vec![0.0; cout * ho * wo];
    for o in 0..cout {
        for oy in 0..ho {
            for ox in 0..wo {
                let mut acc = b[o] as f64;
                for ci in 0..x.c {
                    for ky in 0..k {
                        let iy = (oy * stride + ky) as isize - pad as isize;
                        if iy < 0 || iy >= x.h as isize {
                            continue;
                        }
                        for kx in 0..k {
                            let ix = (ox * stride + kx) as isize - pad as isize;
                            if ix < 0 || ix >= x.w as isize {
                                continue;
                            }
                            acc += wt[((o * x.c + ci) * k + ky) * k + kx] as f64 * x.v[(ci * x.h + iy as usize) * x.w + ix as usize];
                        }
                    }
                }
                v[(o * ho + oy) * wo + ox] = acc.max(0.0);
            }
        }
    }
    Map { c: cout, h: ho, w: wo, v }
}

fn max_pool(x: &Map) -> Map {
    let (k, s) = (POOL_K as usize, POOL_STRIDE as usize);
    let (ho, wo) = ((x.h - k) / s + 1, (x.w - k) / s + 1);
    let mut v = vec![f64::NEG_INFINITY; x.c * ho * wo];
    for c in 0..x.c {
        for oy in 0..ho {
            for ox in 0..wo {
                for ky in 0..k {
                    for kx in 0..k {
                        let o = &mut v[(c * ho + oy) * wo + ox];
                        *o = o.max(x.v[(c * x.h + oy * s + ky) * x.w + ox * s + kx]);
                    }
                }
            }
        }
    }
    Map { c: x.c, h: ho, w: wo, v }
}

/// The five post-ReLU taps of AlexNet on an interleaved `[0,1]` RGB image.
fn taps(t: &Tensors, img: &[f32], w: usize, h: usize) -> Vec<Map> {
    let mut v = vec![0.0; 3 * w * h];
    for p in 0..w * h {
        for c in 0..3 {
            let x = 2.0 * img[p * 3 + c] as f64 - 1.0;
            v[c * w * h + p] = (x - SHIFT[c] as f64) / SCALE[c] as f64;
        }
    }
    let mut x = Map { c: 3, h, w, v };
    let mut out = Vec::new();
    for (i, c) in TRUNK.iter().enumerate() {
        if c.pooled {
            x = max_pool(&x);
        }
        x = conv_relu(&x, &t[&trunk_weight(i)].1, &t[&trunk_bias(i)].1, c.cout as usize, c.k as usize, c.stride as usize, c.pad as usize);
        out.push(Map { c: x.c, h: x.h, w: x.w, v: x.v.clone() });
    }
    out
}

/// torch's adaptive average-pool cell `[lo, hi)` of output `o` of `n` over `len`.
fn cell(o: usize, n: usize, len: usize) -> (usize, usize) {
    (o * len / n, ((o + 1) * len).div_ceil(n))
}

/// LPIPS per tap, f64: `lpips.LPIPS.forward` with `retPerLayer=True`, the
/// spatial average weighted by the mask's area average over each cell.
fn reference(t: &Tensors, a: &[f32], b: &[f32], w: usize, h: usize, mask: Option<&[f32]>) -> [f64; 5] {
    let (fa, fb) = (taps(t, a, w, h), taps(t, b, w, h));
    let mut out = [0.0; 5];
    for (i, (ma, mb)) in fa.iter().zip(&fb).enumerate() {
        let lin = &t[&head_weight(i)].1;
        let hw = ma.h * ma.w;
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..hw {
            let norm = |m: &Map| (0..m.c).map(|c| m.v[c * hw + p].powi(2)).sum::<f64>().sqrt() + EPS;
            let (na, nb) = (norm(ma), norm(mb));
            let d: f64 = (0..ma.c).map(|c| lin[c] as f64 * (ma.v[c * hw + p] / na - mb.v[c * hw + p] / nb).powi(2)).sum();
            let wgt = match mask {
                None => 1.0,
                Some(m) => {
                    let ((y0, y1), (x0, x1)) = (cell(p / ma.w, ma.h, h), cell(p % ma.w, ma.w, w));
                    let s: f64 = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (y, x))).map(|(y, x)| m[y * w + x] as f64).sum();
                    s / ((y1 - y0) * (x1 - x0)) as f64
                }
            };
            num += wgt * d;
            den += wgt;
        }
        out[i] = num / den;
    }
    out
}

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

/// A textured test image: smooth gradients, oriented stripes and hard
/// edges, so every tap has structure to respond to.
fn texture(w: usize, h: usize, phase: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(3 * w * h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32, y as f32);
            let checker = if ((x / 9) + (y / 7)) % 2 == 0 { 0.15 } else { -0.15 };
            for c in 0..3 {
                let wave = 0.25 * (0.31 * fx + 0.17 * fy + phase + c as f32 * 1.3).sin();
                let ramp = 0.2 * (fx / w as f32 - fy / h as f32);
                v.push((0.5 + wave + ramp + checker).clamp(0.0, 1.0));
            }
        }
    }
    v
}

/// A separable gaussian blur of standard deviation `sigma` pixels, clamped at
/// the border. Gaussian rather than box: a box filter's response has
/// sidelobes, so a wider box can pass MORE of a periodic pattern than a
/// narrower one, and "more blur" would not mean less detail.
fn blur(img: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let taps: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let norm: f32 = taps.iter().sum();
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0; src.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0.0;
                    for (t, d) in taps.iter().zip(-r..=r) {
                        let (sx, sy) = if horizontal {
                            ((x as isize + d).clamp(0, w as isize - 1) as usize, y)
                        } else {
                            (x, (y as isize + d).clamp(0, h as isize - 1) as usize)
                        };
                        acc += t * src[(sy * w + sx) * 3 + c];
                    }
                    out[(y * w + x) * 3 + c] = acc / norm;
                }
            }
        }
        out
    };
    pass(&pass(img, true), false)
}

/// `img` plus uniform noise of amplitude `a`, clamped to `[0, 1]`.
fn noisy(img: &[f32], a: f32, seed: u64) -> Vec<f32> {
    let mut rng = data::rng::Lcg::new(seed);
    img.iter().map(|v| (v + rng.scaled(a)).clamp(0.0, 1.0)).collect()
}

fn close(device: f64, host: f64, what: &str) {
    let tol = 2e-4 * host.abs() + 1e-6;
    assert!((device - host).abs() <= tol, "{what}: device {device:.7} vs reference {host:.7} (tolerance {tol:.1e})");
}

// ---------------------------------------------------------------------------
// The spec.
// ---------------------------------------------------------------------------

/// The device agrees with the f64 reference, tap by tap, on a non-square
/// pair - with and without a mask that covers part of the frame unevenly.
#[test]
fn the_device_agrees_with_an_f64_transcription_of_the_reference() {
    let Some(t) = weights() else { return };
    let mut m = metric(&t);
    let (w, h) = (72, 48);
    let a = texture(w, h, 0.0);
    let b = noisy(&texture(w, h, 2.0), 0.15, 7);
    let mask: Vec<f32> = (0..w * h).map(|p| if p % w >= w / 3 { 1.0 } else { 0.25 * ((p / w) % 2) as f32 }).collect();
    for mask in [None, Some(&mask[..])] {
        let dev = m.distance(&a, &b, w as u32, h as u32, mask).expect("distance");
        let host = reference(&t, &a, &b, w, h, mask);
        for (i, (d, r)) in dev.layers.iter().zip(&host).enumerate() {
            close(*d, *r, &format!("tap {i}, masked {}", mask.is_some()));
        }
        close(dev.total, host.iter().sum(), "total");
        assert!(dev.total > 0.05, "a visibly different pair is far apart: {}", dev.total);
    }
}

/// An image is at distance exactly zero from itself, masked or not.
#[test]
fn identical_images_are_at_distance_zero() {
    let Some(t) = weights() else { return };
    let mut m = metric(&t);
    let (w, h) = (64, 40);
    let a = texture(w, h, 1.1);
    let mask = vec![0.5; w * h];
    for mask in [None, Some(&mask[..])] {
        let d = m.distance(&a, &a, w as u32, h as u32, mask).expect("distance");
        assert_eq!(d.total, 0.0, "{d:?}");
    }
}

/// More blur and more noise are both further from the original.
#[test]
fn the_distance_grows_with_blur_and_with_noise() {
    let Some(t) = weights() else { return };
    let mut m = metric(&t);
    let (w, h) = (96, 80);
    let a = texture(w, h, 0.0);
    let by_blur: Vec<f64> = [0.5, 1.0, 2.0, 3.0].iter().map(|&s| m.distance(&a, &blur(&a, w, h, s), w as u32, h as u32, None).expect("distance").total).collect();
    let by_noise: Vec<f64> = [0.02, 0.05, 0.1, 0.2].iter().map(|&s| m.distance(&a, &noisy(&a, s, 3), w as u32, h as u32, None).expect("distance").total).collect();
    for (what, v) in [("blur", &by_blur), ("noise", &by_noise)] {
        assert!(v.windows(2).all(|p| p[1] > p[0]), "{what}: {v:?} is not increasing");
        assert!(v[0] > 0.0, "{what}: the mildest step is already a distance: {v:?}");
    }
}

/// The official `lpips` package's distances between the LPIPS repository's
/// own example images, reproduced on the device and by the f64 reference.
#[test]
fn the_official_example_distances_are_reproduced() {
    let Some(t) = weights() else { return };
    let dir = brain_testutil::testdata_path("lpips");
    let Some(src) = brain_testutil::golden::Source::open(&dir, DUMPER) else { return };
    let channels: Vec<(String, i64)> = TRUNK.iter().enumerate().map(|(i, c)| (format!("relu{}", i + 1), c.cout as i64)).collect();
    let named: Vec<(&str, i64)> = channels.iter().map(|(n, c)| (n.as_str(), *c)).collect();
    if !src.require(&named) {
        return;
    }
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest")).expect("manifest json");
    let load = |name: &str| {
        let img = imaging::load(dir.join(name)).expect("example image");
        (img.to_hwc_unit(), img.w, img.h)
    };
    let (reference_img, w, h) = load("ex_ref.png");
    let mut m = metric(&t);
    for other in ["ex_p0.png", "ex_p1.png"] {
        let pair = &manifest["pairs"][format!("ex_ref.png|{other}")];
        let official = pair["lpips"].as_f64().expect("official distance");
        let (img, ow, oh) = load(other);
        assert_eq!((ow, oh), (w, h));
        let dev = m.distance(&reference_img, &img, w, h, None).expect("distance");
        let host = reference(&t, &reference_img, &img, w as usize, h as usize, None);
        for i in 0..5 {
            let official_tap = pair["layers"][i].as_f64().expect("official tap");
            close(dev.layers[i], official_tap, &format!("{other} tap {i} on the device"));
            close(host[i], official_tap, &format!("{other} tap {i} by the f64 reference"));
        }
        close(dev.total, official, &format!("{other} on the device"));
        eprintln!("lpips(ex_ref, {other}): device {:.6}, official {official:.6}", dev.total);
    }
}

/// Inputs the metric cannot score are refused, not scored.
#[test]
fn unusable_inputs_are_refused() {
    let Some(t) = weights() else { return };
    let mut m = metric(&t);
    let small = texture(30, 64, 0.0);
    assert!(m.distance(&small, &small, 30, 64, None).is_err(), "narrower than AlexNet can pool");
    let a = texture(40, 40, 0.0);
    assert!(m.distance(&a, &a[3..], 40, 40, None).is_err(), "sizes disagree");
    assert!(m.distance(&a, &a, 40, 40, Some(&vec![0.0; 1600])).is_err(), "an empty mask");
    assert!(m.distance(&a, &a, 40, 40, Some(&vec![1.0; 1599])).is_err(), "a mask of the wrong size");
}
