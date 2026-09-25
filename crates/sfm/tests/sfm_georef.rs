// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Putting a reconstruction into the world: satellite fixes to local
//! east-north-up metres, a robust similarity from camera centres onto those
//! fixes (wrong fixes and all), the refusal to trust fixes whose error is as
//! large as the capture, and the direction of gravity from nothing but how
//! the photographs were held.
//!
//! The fixes are built from the published meridian and parallel arc lengths
//! per degree of latitude, independent of the ellipsoid code under test.
//!
//! Swedish Embedded AB implements georeferenced photogrammetry for its
//! clients. If your team needs reconstructions in metres and on the map, you
//! can procure our services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use sfm::camera::Pose;
use sfm::georef::{enu, gravity_from_cameras, solve_enu, GeorefCfg, Sim3, Wgs84};
use sfm::linalg::{add, dot, exp_so3, mm, mv, norm, scale, sub, transpose, M3, V3};

const LAT: f64 = 59.43;
const LON: f64 = 17.99;

/// Metres per degree of latitude and of longitude at `lat` (WGS84, the
/// series published with the ellipsoid; centimetre-accurate).
fn metres_per_degree(lat: f64) -> (f64, f64) {
    let p = lat.to_radians();
    let north = 111_132.954 - 559.822 * (2.0 * p).cos() + 1.175 * (4.0 * p).cos();
    let east = 111_412.84 * p.cos() - 93.5 * (3.0 * p).cos() + 0.118 * (5.0 * p).cos();
    (north, east)
}

/// The fix of a point `e` metres east, `n` north and `u` up of the origin.
fn fix(e: f64, n: f64, u: f64) -> Wgs84 {
    let (mn, me) = metres_per_degree(LAT);
    Wgs84 { latitude_deg: LAT + n / mn, longitude_deg: LON + e / me, altitude_m: Some(40.0 + u) }
}

fn origin() -> Wgs84 {
    Wgs84 { latitude_deg: LAT, longitude_deg: LON, altitude_m: Some(40.0) }
}

#[test]
fn east_north_up_is_metres_on_the_ground() {
    for (e, n, u) in [(0.0, 111.4, 0.0), (56.6, 0.0, 0.0), (-30.0, 20.0, 12.5)] {
        let got = enu(&origin(), &fix(e, n, u));
        // curvature drops a point 100 m away by under a millimetre
        assert!(norm(sub(got, [e, n, u])) < 0.02, "({e}, {n}, {u}) m came out as {got:?}");
    }
}

fn random_rotation(rng: &mut Lcg) -> M3 {
    exp_so3([3.0 * rng.signed() as f64, 3.0 * rng.signed() as f64, 3.0 * rng.signed() as f64])
}

/// A walk of camera centres over 60 x 40 m of ground, reconstructed in an
/// arbitrary frame (rotated, 1/20 scale, shifted), with metre-level fix
/// noise, twice that in altitude, and three fixes forty metres out: the
/// similarity puts every camera within a metre and a half of where it
/// really was, and names the three wrong fixes.
#[test]
fn a_robust_similarity_lands_the_cameras_in_metres() {
    let mut rng = Lcg::new(3);
    let truth: Vec<V3> = (0..24)
        .map(|i| {
            let t = i as f64 / 23.0;
            [60.0 * t - 30.0, 40.0 * (t * 5.0).sin(), 3.0 * (t * 7.0).cos()]
        })
        .collect();
    let world_to_local = Sim3 { s: 0.05, r: random_rotation(&mut rng), t: [3.0, -1.0, 7.0] };
    let local: Vec<Option<V3>> = truth.iter().map(|x| Some(world_to_local.apply(*x))).collect();
    let mut gps = Vec::new();
    for (i, x) in truth.iter().enumerate() {
        let off = if [4, 11, 19].contains(&i) { [40.0, -35.0, 5.0] } else { [0.0; 3] };
        gps.push(Some(fix(x[0] + off[0] + rng.signed() as f64, x[1] + off[1] + rng.signed() as f64, x[2] + off[2] + 2.0 * rng.signed() as f64)));
    }
    let fit = solve_enu(&local, &gps, None, &GeorefCfg::default()).expect("the fixes span the capture");
    for (i, (l, t)) in local.iter().zip(&truth).enumerate() {
        let got = fit.sim3.apply(l.unwrap());
        // the truth is east-north-up of `origin()`, the fit of its own
        let want = add(*t, enu(&fit.origin, &origin()));
        let e = norm(sub(got, want));
        assert!(e < 1.5, "camera {i} is {e:.2} m from where it was taken");
    }
    let flagged: Vec<usize> = fit.inliers.iter().enumerate().filter(|(_, &ok)| !ok).map(|(i, _)| i).collect();
    assert_eq!(flagged, vec![4, 11, 19]);
    assert!((fit.sim3.s * world_to_local.s - 1.0).abs() < 0.02, "scale off by {:.3}", fit.sim3.s * world_to_local.s);
}

/// A phone that fixes its position from the network reports the same fix,
/// to the arc-second, for a whole walk round a table: the fixes' scatter is
/// as large as the capture, so they say nothing about its scale and are
/// refused rather than applied.
#[test]
fn fixes_as_coarse_as_the_capture_are_refused() {
    let local: Vec<Option<V3>> = (0..16).map(|i| {
        let a = i as f64 * 0.4;
        Some([1.2 * a.cos(), 0.1, 1.2 * a.sin()])
    }).collect();
    // four arc-second cells (~30 m) visited in turn
    let cells = [(0.0, 0.0), (-31.0, -3.5), (-3.0, -1.3), (-4.4, -1.4)];
    let gps: Vec<Option<Wgs84>> = (0..16).map(|i| {
        let (n, e) = cells[(i * 4 / 16).min(3)];
        Some(fix(e, n, 0.0))
    }).collect();
    let got = solve_enu(&local, &gps, None, &GeorefCfg::default());
    assert!(got.is_err(), "coarse fixes were not refused: {got:?}");
}

/// Camera-to-world rotation of a camera with yaw about the vertical, then
/// pitch, then roll about its own axis, in a world whose up is `up` (+X
/// right, +Y down, +Z forward in the camera).
fn held(world: &M3, yaw: f64, pitch: f64, roll: f64) -> Pose {
    // in a frame with up = -Y: a level camera looking along +Z is the
    // identity
    let level_to_frame = mm(&exp_so3([0.0, yaw, 0.0]), &exp_so3([pitch, 0.0, 0.0]));
    let c2w_frame = mm(&level_to_frame, &exp_so3([0.0, 0.0, roll]));
    let c2w = mm(world, &c2w_frame);
    Pose { r: transpose(&c2w), t: [0.0; 3] }
}

/// Photographs taken walking down a street - not an orbit, nothing for an
/// orbit-plane fit to find - looking left and right, up and down, never
/// quite level, and two of them turned on their side: the cameras' own
/// horizontal axes still say which way gravity points, to within a couple of
/// degrees.
#[test]
fn gravity_comes_from_how_the_photographs_were_held() {
    let mut rng = Lcg::new(17);
    let world = random_rotation(&mut rng);
    let up = mv(&world, [0.0, -1.0, 0.0]);
    let mut poses = Vec::new();
    for i in 0..20 {
        let roll = if i == 6 || i == 13 { 1.5 } else { 4f64.to_radians() * rng.signed() as f64 };
        let mut p = held(&world, 1.2 * rng.signed() as f64, 0.6 * rng.signed() as f64, roll);
        p.t = scale(mv(&p.r, mv(&world, [0.0, 0.0, 1.5 * i as f64])), -1.0);
        poses.push(p);
    }
    let g = gravity_from_cameras(&poses).expect("gravity");
    let err = dot(g.up, up).clamp(-1.0, 1.0).acos().to_degrees();
    assert!(err < 2.0, "up is {err:.2} deg off");
    assert!(!g.inliers[6] && !g.inliers[13], "the sideways photographs were trusted");

    // all looking the same way down a corridor, pitched only: the horizontal
    // axes are all one axis, and the answer comes from the cameras' own
    // vertical instead
    let poses: Vec<Pose> = (0..10).map(|_| held(&world, 0.02 * rng.signed() as f64, 0.15 * rng.signed() as f64, 3f64.to_radians() * rng.signed() as f64)).collect();
    let g = gravity_from_cameras(&poses).expect("gravity");
    let err = dot(g.up, up).clamp(-1.0, 1.0).acos().to_degrees();
    assert!(err < 5.0, "corridor: up is {err:.2} deg off");
}
