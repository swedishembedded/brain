// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The camera models against their own definitions: projection inverts
//! unprojection, every analytic Jacobian agrees with a central difference of
//! the function it differentiates, and a polynomial lens refuses to fold
//! geometry from outside its valid disc back into the image.

use camera::{Intrinsics, Lens};

fn lenses() -> Vec<Intrinsics> {
    let (w, h) = (640u32, 480u32);
    let base = Intrinsics { fx: 410.0, fy: 405.0, cx: 322.5, cy: 238.0, lens: Lens::Pinhole, width: w, height: h };
    vec![
        base,
        Intrinsics { lens: Lens::radial(-0.08, 0.05), ..base },
        Intrinsics {
            lens: Lens::Brown { k: [-0.21, 0.09, -0.02, 0.03, 0.01, -0.005], p: [1.2e-3, -8e-4], s: [3e-4, -1e-4, 2e-4, 1e-4] },
            ..base
        },
        Intrinsics { fx: 200.0, fy: 200.0, lens: Lens::Fisheye { k: [0.02, -0.01, 0.003, -0.0005] }, ..base },
        Intrinsics::equirect(1024, 512),
    ]
}

fn pixels(k: &Intrinsics) -> Vec<[f64; 2]> {
    let mut v = Vec::new();
    for j in 0..9 {
        for i in 0..11 {
            v.push([(i as f64 + 0.37) / 11.0 * k.width as f64, (j as f64 + 0.61) / 9.0 * k.height as f64]);
        }
    }
    v
}

#[test]
fn projection_inverts_unprojection_across_the_frame() {
    for k in lenses() {
        let mut checked = 0;
        for px in pixels(&k) {
            let Some(d) = k.unproject(px) else { continue };
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((n - 1.0).abs() < 1e-12, "{}: ray through {px:?} is not unit ({n})", k.lens.name());
            let back = k.project(d).unwrap_or_else(|| panic!("{}: ray through {px:?} does not project", k.lens.name()));
            let err = (back[0] - px[0]).hypot(back[1] - px[1]);
            assert!(err < 1e-7, "{}: {px:?} -> {d:?} -> {back:?} ({err:.2e} px)", k.lens.name());
            checked += 1;
        }
        assert!(checked > 80, "{}: only {checked} pixels unprojected", k.lens.name());
    }
}

#[test]
fn direction_jacobians_match_central_differences() {
    for k in lenses() {
        for px in pixels(&k).into_iter().step_by(7) {
            let Some(d) = k.unproject(px) else { continue };
            // off the unit sphere too: the Jacobian is of the map on R³
            let d = [d[0] * 1.7, d[1] * 1.7, d[2] * 1.7];
            let (_, j) = k.project_jac(d).unwrap();
            for c in 0..3 {
                let h = 1e-6;
                let (mut a, mut b) = (d, d);
                a[c] += h;
                b[c] -= h;
                let (pa, pb) = (k.project(a).unwrap(), k.project(b).unwrap());
                for r in 0..2 {
                    let fd = (pa[r] - pb[r]) / (2.0 * h);
                    assert!((fd - j[r][c]).abs() < 1e-4 * fd.abs().max(1.0), "{}: d u{r}/d d{c} analytic {} vs {fd}", k.lens.name(), j[r][c]);
                }
            }
        }
    }
}

#[test]
fn calibration_jacobians_match_central_differences() {
    for k in lenses() {
        let p0 = k.params();
        for px in pixels(&k).into_iter().step_by(13) {
            let Some(d) = k.unproject(px) else { continue };
            let (_, _, jp) = k.project_param_jac(d).unwrap();
            assert_eq!(jp.len(), p0.len());
            for (i, g) in jp.iter().enumerate() {
                let h = 1e-6 * p0[i].abs().max(1e-3);
                let (mut a, mut b) = (p0.clone(), p0.clone());
                a[i] += h;
                b[i] -= h;
                let (pa, pb) = (k.with_params(&a).project(d).unwrap(), k.with_params(&b).project(d).unwrap());
                for r in 0..2 {
                    let fd = (pa[r] - pb[r]) / (2.0 * h);
                    assert!((fd - g[r]).abs() < 1e-4 * fd.abs().max(1.0), "{} {}: analytic {} vs {fd}", k.lens.name(), k.param_names()[i], g[r]);
                }
            }
            // and the ray's own motion under the calibration, holding the pixel
            let (_, cols) = k.unproject_param_jac(px).unwrap();
            for (i, g) in cols.iter().enumerate() {
                let h = 1e-6 * p0[i].abs().max(1e-3);
                let (mut a, mut b) = (p0.clone(), p0.clone());
                a[i] += h;
                b[i] -= h;
                let (da, db) = (k.with_params(&a).unproject(px).unwrap(), k.with_params(&b).unproject(px).unwrap());
                for r in 0..3 {
                    let fd = (da[r] - db[r]) / (2.0 * h);
                    assert!((fd - g[r]).abs() < 1e-4 * fd.abs().max(1e-2), "{} ray / {}: analytic {} vs {fd}", k.lens.name(), k.param_names()[i], g[r]);
                }
            }
        }
    }
}

/// A barrel lens strong enough to fold: past the fold a point further off
/// axis lands CLOSER to the centre, so trusting the polynomial there would
/// draw geometry from outside the field of view into the middle of the frame.
#[test]
fn a_folding_lens_refuses_what_lies_past_its_fold() {
    let k = Intrinsics { lens: Lens::radial(-0.35, 0.0), ..Intrinsics::pinhole(400.0, 640, 480) };
    let r = k.valid_radius();
    // r (1 - 0.35 r²) peaks at r = 1/sqrt(1.05)
    assert!((r - 1.0 / 1.05f64.sqrt()).abs() < 1e-6, "fold at {r}");
    assert!(k.project([0.9 * r, 0.0, 1.0]).is_some());
    assert!(k.project([1.1 * r, 0.0, 1.0]).is_none(), "a point past the fold must not project");
    // behind a perspective camera nothing projects
    assert!(k.project([0.0, 0.0, -1.0]).is_none());
    // while a fisheye sees past 90 degrees
    let f = Intrinsics { lens: Lens::Fisheye { k: [0.0; 4] }, ..Intrinsics::pinhole(200.0, 640, 480) };
    assert!(f.project([1.0, 0.0, -0.2]).is_some());
}

#[test]
fn resizing_scales_every_pixel_coordinate_exactly() {
    for k in lenses() {
        let s = k.resized(k.width / 2, k.height / 2);
        for px in pixels(&k).into_iter().step_by(5) {
            let Some(d) = k.unproject(px) else { continue };
            let a = k.project(d).unwrap();
            let b = s.project(d).unwrap();
            assert!((b[0] - a[0] * 0.5).abs() < 1e-9 && (b[1] - a[1] * 0.5).abs() < 1e-9, "{}", k.lens.name());
        }
    }
}

#[test]
fn json_round_trips_every_model_and_reads_old_pinhole_entries() {
    for k in lenses() {
        let back = Intrinsics::from_json(&k.to_json()).unwrap();
        assert_eq!(back, k);
    }
    let old = serde_json::json!({"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480});
    assert_eq!(Intrinsics::from_json(&old).unwrap().lens, Lens::Pinhole);
}

#[test]
fn sampling_rate_is_the_focal_length_on_axis() {
    let k = Intrinsics::pinhole(500.0, 640, 480);
    let r = k.pixels_per_radian([0.0, 0.0, 1.0]).unwrap();
    assert!((r - 500.0).abs() < 1e-9, "{r}");
    // off axis a pinhole samples more densely per radian
    assert!(k.pixels_per_radian([0.5, 0.0, 1.0]).unwrap() > 500.0);
}
