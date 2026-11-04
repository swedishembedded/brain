// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Putting two reconstructions of the same place into one scene.
//!
//! A feed-forward pass can only hold so many frames at once, so a long capture
//! has to be reconstructed in chunks - and each chunk comes back in ITS OWN
//! world: anchored to whichever frame led it, and at whatever scale that
//! chunk's content happened to normalise to. Two chunks of the same walk are
//! therefore related by a SIMILARITY, seven degrees of freedom, and stacking
//! their gaussians without solving for it gives two scenes at two sizes
//! pointing two ways.
//!
//! The frames the chunks share are the correspondence: each shared frame has a
//! pose in both worlds. The catch is that overlapping frames of a real capture
//! are consecutive frames of one smooth sweep, so their camera centres are
//! nearly collinear - and a fit that uses only centres is then free to spin
//! the whole scene about the line through them. That degenerate case is the
//! one worth testing, because it is the normal case.
//!
//! Swedish Embedded AB implements 3D reconstruction that scales past one
//! model's context window. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use splat::align::{apply_sim3, sim3_from_cameras, transform_c2w_sim3, Sim3};
use splat::quality::psnr;
use splat::renderer::{GpuSplats, Renderer};
use splat::types::{Camera, RenderOpts, Splats};
use splat::Kernels;
use gpu_core::Gpu;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

fn scene(n: usize) -> Splats {
    let mut r = Lcg(0xA11C);
    let mut s = Splats::default();
    for _ in 0..n {
        s.means.extend_from_slice(&[(r.next() - 0.5) * 2.0, (r.next() - 0.5) * 2.0, 3.0 + r.next()]);
        let q = [r.next() - 0.5, r.next() - 0.5, r.next() - 0.5, r.next() - 0.5];
        let l = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-6);
        s.quats.extend_from_slice(&[q[0] / l, q[1] / l, q[2] / l, q[3] / l]);
        s.scales.extend_from_slice(&[0.02 + 0.08 * r.next(), 0.03, 0.02 + 0.05 * r.next()]);
        s.opacities.push(0.6 + 0.35 * r.next());
        s.colors.extend_from_slice(&[r.next(), r.next(), r.next()]);
    }
    s
}

/// `span` radians of arc: a small span is what consecutive overlap frames of a
/// smooth sweep actually look like.
fn sweep(n: usize, start: f64, span: f64) -> Vec<[f64; 16]> {
    (0..n)
        .map(|i| {
            let a = start + span * i as f64 / (n.max(2) - 1) as f64;
            let e = [1.7 * a.cos(), -0.8, 3.0 + 1.7 * a.sin()];
            let f = {
                let d = [-e[0], -0.1 - e[1], 3.0 - e[2]];
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                [d[0] / l, d[1] / l, d[2] / l]
            };
            let up = [0.0, -1.0, 0.0];
            let rr = {
                let c = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]];
                let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                [c[0] / l, c[1] / l, c[2] / l]
            };
            let d2 = [f[1] * rr[2] - f[2] * rr[1], f[2] * rr[0] - f[0] * rr[2], f[0] * rr[1] - f[1] * rr[0]];
            [rr[0], d2[0], f[0], e[0], rr[1], d2[1], f[1], e[1], rr[2], d2[2], f[2], e[2], 0.0, 0.0, 0.0, 1.0]
        })
        .collect()
}

/// A chunk's own world: some rotation, some offset, and a scale all its own.
fn own_world() -> Sim3 {
    let (c, s) = (0.8f64, 0.6f64); // 36.9 degrees about a slanted axis
    let ax = [0.5773502691896258, -0.5773502691896258, 0.5773502691896258];
    let mut r = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = if i == j { c } else { 0.0 } + (1.0 - c) * ax[i] * ax[j];
        }
    }
    let k = [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] += s * k[i][j];
        }
    }
    Sim3 { s: 2.7, r, t: [0.4, -1.3, 2.2] }
}

fn cam(m: &[f64; 16], w: u32, h: u32) -> Camera {
    Camera::pinhole(std::array::from_fn(|i| m[i] as f32), 190.0, 190.0, w as f32 / 2.0, h as f32 / 2.0, w, h)
}

fn recentre_err(a: &[[f64; 16]], b: &[[f64; 16]], m: &Sim3) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let t = transform_c2w_sim3(y, m);
            ((t[3] - x[3]).powi(2) + (t[7] - x[7]).powi(2) + (t[11] - x[11]).powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
fn a_chunk_with_its_own_scale_and_heading_lands_on_the_first() {
    let a = sweep(6, 0.0, 1.2);
    let w = own_world();
    let b: Vec<[f64; 16]> = a.iter().map(|m| transform_c2w_sim3(m, &inverse(&w))).collect();
    let got = sim3_from_cameras(&a, &b).expect("six shared cameras is plenty");
    assert!((got.s - w.s).abs() < 1e-4 * w.s, "scale recovered as {} not {}", got.s, w.s);
    let e = recentre_err(&a, &b, &got);
    assert!(e < 1e-4, "shared cameras land {e:.2e} from where they should");
}

#[test]
fn three_overlap_frames_of_a_smooth_sweep_align_despite_being_collinear() {
    // 3 consecutive frames spanning 3 degrees: the centres are collinear to
    // well within any tolerance, so only the cameras' HEADINGS pin the roll.
    let a = sweep(3, 0.5, 0.05);
    let spread = {
        let c: Vec<[f64; 3]> = a.iter().map(|m| [m[3], m[7], m[11]]).collect();
        let d = [c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let v = [c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]];
        let cr = [v[1] * d[2] - v[2] * d[1], v[2] * d[0] - v[0] * d[2], v[0] * d[1] - v[1] * d[0]];
        (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt() / l
    };
    assert!(spread < 1e-3, "this test is pointless unless the centres are collinear (got {spread:.2e})");

    let w = own_world();
    let b: Vec<[f64; 16]> = a.iter().map(|m| transform_c2w_sim3(m, &inverse(&w))).collect();
    let got = sim3_from_cameras(&a, &b).expect("three shared cameras must be enough");
    assert!((got.s - w.s).abs() < 1e-3 * w.s, "scale recovered as {} not {}", got.s, w.s);
    let e = recentre_err(&a, &b, &got);
    assert!(e < 1e-3, "collinear overlap left the alignment {e:.2e} off");
}

#[test]
fn a_merged_chunk_looks_the_same_as_it_did_in_its_own_world() {
    let g = Gpu::new_cpu(splat::PIPELINES);
    let ks = Kernels::at(0);
    let (w, h) = (96u32, 96u32);
    let a = sweep(5, 0.0, 1.0);
    let world = own_world();
    let inv = inverse(&world);

    // the same physical scene as chunk B saw it: its own frame, its own size
    let truth = scene(500);
    let as_b = apply_sim3(&truth, &inv);
    let b: Vec<[f64; 16]> = a.iter().map(|m| transform_c2w_sim3(m, &inv)).collect();

    let m = sim3_from_cameras(&a, &b).expect("aligns");
    let merged = apply_sim3(&as_b, &m);

    let o = RenderOpts::default();
    let mut ren = Renderer::new(&g, ks, truth.len(), w, h, truth.len() * 16);
    let want = {
        let gs = GpuSplats::upload(&g, &truth);
        ren.render(&g, &gs, &cam(&a[2], w, h), &o);
        ren.read_rgba(&g, w, h)
    };
    let got = {
        let gs = GpuSplats::upload(&g, &merged);
        ren.render(&g, &gs, &cam(&a[2], w, h), &o);
        ren.read_rgba(&g, w, h)
    };
    let db = psnr(&want, &got);
    assert!(db > 40.0, "a merged chunk renders {db:.1} dB from where it belongs");
}

fn inverse(m: &Sim3) -> Sim3 {
    let mut r = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = m.r[j * 3 + i];
        }
    }
    let s = 1.0 / m.s;
    let t = [
        -s * (r[0] * m.t[0] + r[1] * m.t[1] + r[2] * m.t[2]),
        -s * (r[3] * m.t[0] + r[4] * m.t[1] + r[5] * m.t[2]),
        -s * (r[6] * m.t[0] + r[7] * m.t[1] + r[8] * m.t[2]),
    ];
    Sim3 { s, r, t }
}

/// Concatenating scenes keeps every gaussian's view-dependent colour: a
/// scene with harmonics merged with one without renders, gaussian for
/// gaussian, as each did alone.
#[test]
fn concatenation_keeps_view_dependent_colour() {
    let one = |x: f32, sh: Option<(u32, Vec<f32>)>| Splats {
        means: vec![x, 0.0, 3.0],
        quats: vec![1.0, 0.0, 0.0, 0.0],
        scales: vec![0.1; 3],
        opacities: vec![0.9],
        colors: vec![0.3, 0.4, 0.5],
        sh_rest: sh,
    };
    let a = one(0.0, Some((1, (0..9).map(|v| v as f32 * 0.01).collect())));
    let b = one(1.0, None);
    let m = splat::align::concat(&[a.clone(), b.clone()]);
    let eye = [0.4, -0.3, 0.0];
    let got = splat::sh::shade(&m, eye).expect("the merged scene has harmonics");
    let want_a = splat::sh::shade(&a, eye).expect("a has harmonics");
    assert_eq!(&got[..3], &want_a[..], "the gaussian with harmonics");
    assert_eq!(&got[3..], &b.colors[..], "the gaussian without them is its flat colour");
}
