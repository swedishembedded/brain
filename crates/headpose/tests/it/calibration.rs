// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibration: the laptop camera mount maps camera measurements into the
//! screen frame correctly, a profile is strict, and the two user-assisted
//! corrections fix exactly the error they are for.

use headpose::calibration::Profile;
use headpose::{fit_head, Calibration, CameraMount, Display, FaceModel};
use sfm::camera::Pose;

use crate::{rotation, webcam};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn a_laptop_camera_sees_the_viewer_mirrored_and_from_above_the_glass() {
    let d = Display::flat(0.344, 0.194).unwrap();
    let m = CameraMount::above(&d, 0.008);
    // straight ahead of the lens, 0.5 m out
    let p = m.to_screen([0.0, 0.0, 0.5]);
    assert!(close(p[0], 0.0, 1e-12) && close(p[1], 0.097 + 0.008, 1e-12) && close(p[2], 0.5, 1e-12));
    // image right is the viewer's left; image down is down
    let r = m.to_screen([0.1, 0.0, 0.5]);
    assert!(r[0] < p[0]);
    let dn = m.to_screen([0.0, 0.1, 0.5]);
    assert!(dn[1] < p[1]);
    // pitching the camera down tips its axis toward the viewer's lap
    let pitched = m.pitched(10.0).to_screen([0.0, 0.0, 0.5]);
    assert!(pitched[1] < p[1] && close(pitched[1] - p[1], -0.5 * 10f64.to_radians().sin(), 1e-12));
}

#[test]
fn a_profile_is_strict_about_fields_and_units() {
    let good = r#"{"display_width_m":0.344,"display_height_m":0.194,"camera_gap_m":0.008,
        "camera":{"kind":"fov","width":640,"height":480,"hfov_deg":60.0},"ipd_m":0.063}"#;
    let p: Profile = serde_json::from_str(good).unwrap();
    let c = p.calibration().unwrap();
    assert!(close(c.camera.fx, 320.0 / 30f64.to_radians().tan(), 1e-9));
    let back: Profile = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(back, p);

    let unknown = good.replace("\"ipd_m\"", "\"ipd_mm\":63,\"ipd_m\"");
    assert!(serde_json::from_str::<Profile>(&unknown).is_err(), "unknown fields are refused");
    let millimetres: Profile = serde_json::from_str(&good.replace("0.344", "344")).unwrap();
    let e = millimetres.calibration().unwrap_err().to_string();
    assert!(e.contains("display_width_m"), "a unit mistake names the field: {e}");
}

#[test]
fn recentring_absorbs_a_horizontal_mount_error_and_keeps_the_height() {
    let mut c = Calibration::laptop(0.344, 0.194, 0.008, 640, 480, 60.0).unwrap();
    // the camera is really 2 cm right of centre, so a centred viewer measures 2 cm left
    let measured = c.to_screen([0.02, 0.05, 0.55]);
    assert!(measured[0] < -0.019);
    c.recentre(measured);
    let fixed = c.to_screen([0.02, 0.05, 0.55]);
    assert!(close(fixed[0], 0.0, 1e-12) && close(fixed[1], measured[1], 1e-12) && close(fixed[2], measured[2], 1e-12));
    let p = Profile {
        display_width_m: 0.344,
        display_height_m: 0.194,
        camera_gap_m: 0.008,
        camera_pitch_deg: 0.0,
        camera_offset_x_m: 0.0,
        camera: headpose::calibration::CameraProfile::Fov { width: 640, height: 480, hfov_deg: 60.0 },
        ipd_m: 0.063,
    };
    let saved = p.updated(&c);
    assert_eq!(saved.calibration().unwrap().mount, c.mount, "the correction survives a save and load");
}

#[test]
fn a_measured_distance_rescales_the_face_until_depth_is_right() {
    // the viewer's real IPD is 70 mm; the default model assumes 63 mm, so
    // every depth reads 10% short
    let truth = FaceModel::canonical(0.070);
    let pose = Pose { r: rotation(5.0, -3.0, 0.0), t: [0.01, 0.03, 0.6] };
    let k = webcam();
    let mut pts = vec![[320.0, 240.0]; 66];
    let mut conf = vec![0.0; 66];
    for &(i, p) in &truth.points {
        pts[i] = k.project(pose.to_cam(p)).unwrap();
        conf[i] = 0.9;
    }
    let mut c = Calibration::laptop(0.344, 0.194, 0.008, 640, 480, 60.0).unwrap();
    let measured = fit_head(&pts, &conf, &k, &FaceModel::canonical(c.ipd_m), None, 0.5, 2.0).unwrap().cyclopean_eye()[2];
    assert!(close(measured, 0.6 * 63.0 / 70.0, 1e-6));
    c.set_distance(measured, 0.6).unwrap();
    assert!(close(c.ipd_m, 0.070, 1e-9));
    let refit = fit_head(&pts, &conf, &k, &FaceModel::canonical(c.ipd_m), None, 0.5, 2.0).unwrap().cyclopean_eye()[2];
    assert!(close(refit, 0.6, 1e-6));
    assert!(c.set_distance(0.6, 0.06).is_err(), "an IPD no adult has is refused");
}

#[test]
fn a_camera_mode_with_another_aspect_keeps_square_pixels_or_is_refused() {
    let fov: Profile = serde_json::from_str(r#"{"display_width_m":0.344,"display_height_m":0.194,"camera_gap_m":0.008,
        "camera":{"kind":"fov","width":640,"height":480,"hfov_deg":60.0},"ipd_m":0.063}"#).unwrap();
    // the driver delivered 16:9 instead of the 4:3 the profile describes
    let c = fov.calibration_for(1280, 720).unwrap();
    assert!(close(c.camera.fx, c.camera.fy, 1e-12), "pixels stay square: fx {} fy {}", c.camera.fx, c.camera.fy);
    assert!(close(c.camera.fx, 640.0 / 30f64.to_radians().tan(), 1e-9), "the field of view is kept across the width");
    let k = camera::Intrinsics::pinhole(500.0, 640, 480);
    let cal: Profile = serde_json::from_value(serde_json::json!({"display_width_m":0.344,"display_height_m":0.194,"camera_gap_m":0.008,
        "camera":{"kind":"intrinsics","intrinsics":k.to_json()},"ipd_m":0.063})).unwrap();
    assert!(close(cal.calibration_for(1280, 960).unwrap().camera.fx, 1000.0, 1e-9), "same aspect: scaled");
    assert!(cal.calibration_for(1280, 720).is_err(), "another aspect: refused");
}
