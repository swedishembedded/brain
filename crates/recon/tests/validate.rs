// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Validating a reconstruction in 3D (`recon::validate`), on a scene whose
//! truth is known: a textured ball resting on a floor.
//!
//! - Observational support says how far a view's surface is from anything
//!   that observed it: the angle it reports is the geometric one, a surface
//!   no training view saw is unsupported, and a correct scene's training
//!   views support each other with no free-space violation.
//! - A floater - a surface where the training views saw empty space - is
//!   flagged as a free-space violation.
//! - Held-out designs select what they say and deviation is the angle it
//!   says.
//!
//! Swedish Embedded AB implements 3D reconstruction validated in 3D. If your
//! team needs expertise in radiance-field evaluation then you can procure our
//! services by sending an email to info@swedishembedded.com.

mod common;

use recon::eval::Viewer;
use recon::photogrammetry::pipelines;
use recon::validate::{deviation, split, Orbit, Split, DIAG, SUPPORT};
use splat::types::{Camera, RenderOpts, Splats};

/// A camera `dist` from the ball's centre at azimuth `az` and elevation `el`
/// (degrees; azimuth 0 looks along +z), aimed at the centre.
fn orbit_cam(az: f32, el: f32, dist: f32, w: u32, h: u32) -> Camera {
    let (c, _) = common::BALL;
    let (a, e) = (az.to_radians(), el.to_radians());
    let eye = [c[0] + dist * e.cos() * a.sin(), c[1] - dist * e.sin(), c[2] - dist * e.cos() * a.cos()];
    Camera::look_at(eye, c, [0.0, -1.0, 0.0], 50.0, w, h)
}

fn device() -> Option<gpu_core::Gpu> {
    let pipes: &'static [(&'static str, &'static str)] = Box::leak(pipelines().into_boxed_slice());
    let g = gpu_core::testgpu::dev(pipes);
    if !g.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("the per-pixel support test runs where the reconstruction does: on a GPU");
        return None;
    }
    Some(g)
}

fn viewer(g: &gpu_core::Gpu, s: &Splats, w: u32, h: u32) -> Viewer {
    Viewer::new(g, s, &[], RenderOpts { ray: true, ..Default::default() }, w, h)
}

/// The angle at `x` between the directions to `a` and to `b`, degrees.
fn angle_at(x: [f64; 3], a: [f32; 3], b: [f32; 3]) -> f64 {
    let d = |e: [f32; 3]| {
        let v: [f64; 3] = std::array::from_fn(|k| e[k] as f64 - x[k]);
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        v.map(|c| c / l)
    };
    let (u, v) = (d(a), d(b));
    (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0).acos().to_degrees()
}

#[test]
fn support_is_the_angle_to_what_observed_each_surface() {
    let Some(g) = device() else { return };
    let truth = common::floor_and_ball();
    let (w, h) = (160u32, 120u32);
    // training views on a frontal arc only
    let train: Vec<Camera> = [-40.0f32, -20.0, 0.0, 20.0, 40.0].iter().map(|&az| orbit_cam(az, 25.0, 2.6, w, h)).collect();
    let mut v = viewer(&g, &truth, w, h);
    let obs = v.observe(&train);

    // a novel view between two of them
    let near = orbit_cam(10.0, 25.0, 2.6, w, h);
    let d = v.diagnose(&near, Some(&obs), None);
    let centre = (h / 2 * w + w / 2) as usize;
    let px = &d.pixels[centre * DIAG..(centre + 1) * DIAG];
    assert!(px[0] > 0.99, "the ball covers the centre pixel");
    // the surface point the centre pixel renders, from its median range
    let (o, dir) = splat::reference::pixel_ray(&near, (w / 2) as f64 + 0.5, (h / 2) as f64 + 0.5).expect("a ray");
    let c2w = near.c2w;
    let world = |v: [f64; 3], t: f64| -> [f64; 3] {
        std::array::from_fn(|i| (0..3).map(|k| c2w[i * 4 + k] as f64 * v[k]).sum::<f64>() + t * c2w[i * 4 + 3] as f64)
    };
    let (eye, ray) = (world(o, 1.0), world(dir, 0.0));
    let x: [f64; 3] = std::array::from_fn(|k| eye[k] + ray[k] * px[2] as f64);
    let want = train.iter().map(|c| angle_at(x, near.eye(), c.eye())).fold(f64::INFINITY, f64::min);
    let got = d.support[centre * SUPPORT + 1] as f64;
    assert!((got - want).abs() < 0.5, "support angle {got:.2} deg on the device, {want:.2} from the geometry");
    assert!(d.support[centre * SUPPORT] >= 3.0, "{} training views support the centre", d.support[centre * SUPPORT]);
    let s = d.stats(None);
    assert!(s.support[0] < 0.1, "{:.0}% of a view inside the arc is unsupported", 100.0 * s.support[0]);
    assert!(s.free_space < 0.01, "a correct scene has free-space violations at {:.1}% of the pixels", 100.0 * s.free_space);

    // from the side, behind the arc: each ball pixel's surface point is observed exactly when
    // its normal faces some training camera (the ball is convex and nothing
    // else is in the way)
    let back = orbit_cam(130.0, 10.0, 2.6, w, h);
    let d = v.diagnose(&back, Some(&obs), None);
    let (c, rad) = common::BALL;
    let (mut unseen, mut unseen_flagged, mut seen, mut seen_supported) = (0usize, 0usize, 0usize, 0usize);
    for p in 0..(w * h) as usize {
        if d.pixels[p * DIAG] < 0.99 {
            continue;
        }
        let (o, dir) = splat::reference::pixel_ray(&back, (p % w as usize) as f64 + 0.5, (p / w as usize) as f64 + 0.5).expect("a ray");
        let m = back.c2w;
        let world = |v: [f64; 3], t: f64| -> [f64; 3] { std::array::from_fn(|i| (0..3).map(|k| m[i * 4 + k] as f64 * v[k]).sum::<f64>() + t * m[i * 4 + 3] as f64) };
        let (eye, ray) = (world(o, 1.0), world(dir, 0.0));
        let x: [f64; 3] = std::array::from_fn(|k| eye[k] + ray[k] * d.pixels[p * DIAG + 2] as f64);
        let n: [f64; 3] = std::array::from_fn(|k| (x[k] - c[k] as f64) / rad as f64);
        if (n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0).abs() > 0.1 {
            continue; // the floor
        }
        let facing = train
            .iter()
            .map(|t| {
                let e = t.eye();
                let v: [f64; 3] = std::array::from_fn(|k| e[k] as f64 - x[k]);
                let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                (n[0] * v[0] + n[1] * v[1] + n[2] * v[2]) / l
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let supported = d.support[p * SUPPORT] > 0.0;
        if facing < -0.15 {
            unseen += 1;
            unseen_flagged += !supported as usize;
        } else if facing > 0.5 {
            // seen within 60 degrees of its normal: at grazing incidence one
            // training pixel spans more range than the consistency test's 1%
            seen += 1;
            seen_supported += supported as usize;
        }
    }
    assert!(unseen > 200 && seen > 100, "{unseen} unseen and {seen} seen ball pixels from behind");
    assert!(unseen_flagged as f64 > 0.95 * unseen as f64, "{unseen_flagged} of {unseen} pixels on the ball's unseen side are unsupported");
    assert!(seen_supported as f64 > 0.9 * seen as f64, "{seen_supported} of {seen} pixels on the ball's observed side are supported");

    // training views of the correct scene support each other
    let d = v.diagnose(&train[2], Some(&obs), Some(2));
    let s = d.stats(None);
    assert!(s.support[0] < 0.1 && s.free_space < 0.01, "a training view against the others: {:.0}% unsupported, {:.1}% free-space violations", 100.0 * s.support[0], 100.0 * s.free_space);
}

#[test]
fn a_floater_is_a_free_space_violation() {
    let Some(g) = device() else { return };
    let truth = common::floor_and_ball();
    let (w, h) = (96u32, 72u32);
    let train: Vec<Camera> = [-30.0f32, -10.0, 10.0, 30.0].iter().map(|&az| orbit_cam(az, 25.0, 2.6, w, h)).collect();
    let obs = viewer(&g, &truth, w, h).observe(&train);
    // a dense blob halfway between the cameras and the ball
    let (c, _) = common::BALL;
    let mid = orbit_cam(0.0, 25.0, 1.3, w, h).eye();
    let mut blob = Splats::default();
    for k in 0..40 {
        let a = k as f32 * 2.4;
        blob.means.extend_from_slice(&[mid[0] + 0.08 * a.cos() * (k as f32 / 40.0).sqrt(), mid[1] + 0.08 * a.sin() * (k as f32 / 40.0).sqrt(), mid[2]]);
        blob.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        blob.scales.extend_from_slice(&[0.05; 3]);
        blob.opacities.push(0.95);
        blob.colors.extend_from_slice(&[0.9, 0.1, 0.1]);
    }
    let with = splat::align::concat(&[truth.clone(), blob]);
    let novel = orbit_cam(5.0, 25.0, 2.6, w, h);
    let d = viewer(&g, &with, w, h).diagnose(&novel, Some(&obs), None);
    let e = novel.eye();
    let to_ball = ((e[0] - c[0]).powi(2) + (e[1] - c[1]).powi(2) + (e[2] - c[2]).powi(2)).sqrt();
    let (mut floater, mut flagged) = (0usize, 0usize);
    for p in 0..(w * h) as usize {
        // pixels whose surface is the blob, well in front of the ball
        if d.pixels[p * DIAG] > 0.5 && d.pixels[p * DIAG + 2] < 0.7 * to_ball {
            floater += 1;
            flagged += (d.support[p * SUPPORT + 2] > 0.0) as usize;
        }
    }
    assert!(floater > 50, "the floater covers {floater} pixels");
    assert!(flagged as f64 > 0.9 * floater as f64, "{flagged} of the floater's {floater} pixels are free-space violations");
}

#[test]
fn held_out_designs_select_what_they_say() {
    let (w, h) = (32u32, 24u32);
    let mut cams = Vec::new();
    for el in [30.0f32, 60.0] {
        for k in 0..8 {
            cams.push(orbit_cam(k as f32 * 45.0, el, 2.6, w, h));
        }
    }
    let orbit = Orbit::of(&cams);
    let az: Vec<f64> = cams.iter().map(|c| orbit.angles(c).0).collect();
    // azimuths are measured from the orbit's own frame: relative ones hold
    for k in 1..8 {
        let step = (az[k] - az[k - 1]).rem_euclid(360.0);
        assert!((step - 45.0).abs() < 0.5 || (step - 315.0).abs() < 0.5, "azimuth step {step}");
    }
    let wedge = split(&cams, Split::Wedge { centre: az[3], width: 50.0 });
    assert_eq!(wedge.iter().filter(|&&x| x).count(), 2, "a 50 degree wedge holds out one camera per ring");
    assert!(wedge[3] && wedge[11]);
    let upper = split(&cams, Split::Band { low: 45.0, high: 90.0 });
    assert_eq!(upper, (0..16).map(|i| i >= 8).collect::<Vec<_>>(), "the upper ring");
    // a held-out camera's deviation: its nearest trained neighbour, 45
    // degrees round the same ring (the other ring at this azimuth is 30
    // degrees away and held out too)
    let train: Vec<Camera> = cams.iter().zip(&wedge).filter(|(_, &h)| !h).map(|(c, _)| *c).collect();
    let dev = deviation(&cams[3], &train, orbit.centre);
    let (c, _) = common::BALL;
    let want = train.iter().map(|t| angle_at(c.map(|v| v as f64), cams[3].eye(), t.eye())).fold(f64::INFINITY, f64::min);
    assert!((want - 45.0 * 30f64.to_radians().cos()).abs() < 3.0, "the nearest trained camera is the ring neighbour, {want:.1} degrees away");
    assert!((dev - want).abs() < 0.5, "deviation {dev:.2} against {want:.2}");
}
