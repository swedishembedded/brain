// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Everything that relates the webcam's measurements to the glass: the lens,
//! where the camera sits relative to the display, the display itself, and
//! the viewer's interpupillary distance.
//!
//! These are FACTS about one machine and one person, so they are data a
//! profile holds, never constants buried in a renderer. A profile is a strict
//! serde record ([`Profile`]); unknown fields and out-of-range values are
//! refused when it is read, naming the field.
//!
//! Two user-assisted corrections are provided, matching the two errors an
//! uncalibrated laptop actually shows:
//!
//! * [`Calibration::recentre`] - "I am sitting centred in front of the
//!   screen": the residual offset, horizontal and vertical, is the camera's
//!   aiming error (a laptop lid tips the camera with it), absorbed as the
//!   camera's pitch and yaw.
//! * [`Calibration::set_distance`] - "my eyes are this far from the glass":
//!   monocular depth is only as good as the face's assumed size, so a
//!   measured distance rescales the interpupillary distance.

use camera::Intrinsics;
use serde::{Deserialize, Serialize};
use sfm::linalg::{add, dot, mm, mtv, mv, norm, normalize, scale, sub, M3};

use crate::display::Display;
use crate::{GeometryError, V3};

/// The webcam's rigid placement in the screen frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraMount {
    /// Optical centre, screen frame, metres.
    pub position: V3,
    /// Camera frame to screen frame (row-major 3x3).
    pub rotation: M3,
}

impl CameraMount {
    /// A webcam in the bezel `gap_m` above the top edge of `display`,
    /// horizontally centred, looking out along the display normal - the
    /// laptop arrangement. The camera's image right is the viewer's left and
    /// its image down is the viewer's down, so the rotation is a half turn
    /// about the optical axis relative to the screen frame's (right, up,
    /// out) axes.
    pub fn above(display: &Display, gap_m: f64) -> CameraMount {
        let (r, u, n) = (display.right(), display.up(), display.normal());
        let top_centre = display.point(0.5, 1.0);
        // columns are the camera axes expressed in the screen frame
        let (cx, cy, cz) = (scale(r, -1.0), scale(u, -1.0), n);
        let rotation = [cx[0], cy[0], cz[0], cx[1], cy[1], cz[1], cx[2], cy[2], cz[2]];
        CameraMount { position: add(top_centre, scale(u, gap_m)), rotation }
    }

    /// The same mount with the camera aimed `pitch_deg` downward (positive
    /// looks toward the viewer's lap) and then `yaw_deg` sideways (positive
    /// looks toward the viewer's right): `rotation * Ry(yaw) * Rx(pitch)` in
    /// the camera's own axes, no roll.
    pub fn aimed(self, pitch_deg: f64, yaw_deg: f64) -> CameraMount {
        let (sp, cp) = pitch_deg.to_radians().sin_cos();
        let (sy, cy) = yaw_deg.to_radians().sin_cos();
        // the new camera axes in the old camera frame (columns). Pitch: +Z
        // tips toward +Y (down). Yaw: +Z turns toward -X, which is the
        // viewer's right because the camera faces them.
        let rx: M3 = [1.0, 0.0, 0.0, 0.0, cp, sp, 0.0, -sp, cp];
        let ry: M3 = [cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy];
        CameraMount { rotation: mm(&self.rotation, &mm(&ry, &rx)), ..self }
    }

    /// The `(pitch_deg, yaw_deg)` of this mount relative to `base`, the
    /// inverse of [`Self::aimed`] for a mount built by it.
    pub fn aim_relative_to(&self, base: &CameraMount) -> (f64, f64) {
        // the optical axis in the base camera's frame is Ry * Rx * (0,0,1)
        // = (-sin(yaw) cos(pitch), sin(pitch), cos(yaw) cos(pitch))
        let z = mtv(&base.rotation, [self.rotation[2], self.rotation[5], self.rotation[8]]);
        (z[1].clamp(-1.0, 1.0).asin().to_degrees(), (-z[0]).atan2(z[2]).to_degrees())
    }

    /// A camera-frame point in the screen frame.
    pub fn to_screen(&self, p_cam: V3) -> V3 {
        add(mv(&self.rotation, p_cam), self.position)
    }
}

/// One machine's and one viewer's calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub camera: Intrinsics,
    pub mount: CameraMount,
    pub display: Display,
    /// The viewer's interpupillary distance, metres.
    pub ipd_m: f64,
}

/// Bounds a profile is validated against. Wide enough for any real laptop or
/// monitor and any adult; narrow enough that a unit mistake (millimetres
/// typed as metres) is caught.
const DISPLAY_SIZE_M: (f64, f64) = (0.05, 3.0);
const IPD_M: (f64, f64) = (0.045, 0.080);
const CAMERA_GAP_M: (f64, f64) = (-0.2, 0.2);
const CAMERA_PITCH_DEG: (f64, f64) = (-45.0, 45.0);
const CAMERA_YAW_DEG: (f64, f64) = (-45.0, 45.0);
const HFOV_DEG: (f64, f64) = (20.0, 140.0);

impl Calibration {
    /// A laptop with no measurements beyond the display's size: a pinhole
    /// camera of `hfov_deg` horizontal field of view at `width` x `height`,
    /// centred `camera_gap_m` above the glass, and the canonical IPD.
    pub fn laptop(display_w_m: f64, display_h_m: f64, camera_gap_m: f64, width: u32, height: u32, hfov_deg: f64) -> Result<Calibration, GeometryError> {
        Profile {
            display_width_m: display_w_m,
            display_height_m: display_h_m,
            camera_gap_m,
            camera_pitch_deg: 0.0,
            camera_yaw_deg: 0.0,
            camera_offset_x_m: 0.0,
            camera: CameraProfile::Fov { width, height, hfov_deg },
            ipd_m: crate::face::CANONICAL_IPD_M,
        }
        .calibration()
    }

    /// A camera-frame point in the screen frame.
    pub fn to_screen(&self, p_cam: V3) -> V3 {
        self.mount.to_screen(p_cam)
    }

    /// Re-aim the camera so `measured_eye_screen`, where the viewer's
    /// cyclopean eye measures while they sit centred in front of the
    /// display, reads as exactly on the display's normal through its centre.
    ///
    /// The error is modelled as the camera's aim (pitch and yaw about its
    /// optical centre), not its position: a lid angle or a tilted bezel
    /// camera shifts the measurement in proportion to the viewer's distance,
    /// so an aim fixed at one distance stays right when they lean back,
    /// where a position shift would not. The camera-to-eye distance is
    /// kept, so the target is the point on the centre normal at that
    /// distance from the camera; roll stays zero. The new aim replaces the
    /// old one; it does not accumulate.
    pub fn recentre(&mut self, measured_eye_screen: V3) -> Result<(), GeometryError> {
        let cam = self.mount.position;
        let d = norm(sub(measured_eye_screen, cam));
        let (centre, n) = (self.display.centre(), self.display.normal());
        let q = sub(cam, centre);
        let qn = dot(q, n);
        let disc = d * d - dot(q, q) + qn * qn;
        if !(d > 0.05 && disc > 0.0) {
            return Err(GeometryError::OutOfRange(format!("the eye {measured_eye_screen:?} is too close to the camera to recentre on")));
        }
        let target = add(centre, scale(n, qn + disc.sqrt()));
        let base = self.unaimed();
        // v: the measured eye direction in the camera's own frame; w: where
        // it has to point, in the unaimed camera's frame. Solve
        // Ry(yaw) Rx(pitch) v = w.
        let v = normalize(mtv(&self.mount.rotation, sub(measured_eye_screen, cam)));
        let w = normalize(mtv(&base.rotation, sub(target, cam)));
        // Rx keeps x and mixes (y, z); Ry keeps y: so cos(p) vy + sin(p) vz = wy
        let (rho, phi) = (v[1].hypot(v[2]), v[2].atan2(v[1]));
        if rho < 1e-9 || (w[1] / rho).abs() > 1.0 {
            return Err(GeometryError::OutOfRange("no camera aim puts that eye on the centre normal".into()));
        }
        let a = (w[1] / rho).acos();
        // the branch nearer zero tilt: a lid tips the camera by tens of degrees, not by half turns
        let pitch = [phi + a, phi - a].into_iter().map(wrap_pi).min_by(|x, y| x.abs().total_cmp(&y.abs())).unwrap_or(0.0);
        let (sp, cp) = pitch.sin_cos();
        let u = [v[0], cp * v[1] + sp * v[2], -sp * v[1] + cp * v[2]];
        let yaw = wrap_pi(w[2].atan2(w[0]) - u[2].atan2(u[0]));
        let (pitch_deg, yaw_deg) = (pitch.to_degrees(), yaw.to_degrees());
        check("camera_pitch_deg", pitch_deg, CAMERA_PITCH_DEG)?;
        check("camera_yaw_deg", yaw_deg, CAMERA_YAW_DEG)?;
        self.mount = base.aimed(pitch_deg, yaw_deg);
        Ok(())
    }

    /// The mount with its aim removed: the bezel position, looking along
    /// the display normal.
    fn unaimed(&self) -> CameraMount {
        let level = CameraMount::above(&self.display, 0.0);
        CameraMount { position: self.mount.position, ..level }
    }

    /// Rescale the interpupillary distance so a viewer measured at
    /// `measured_distance_m` from the glass reads as `true_distance_m`.
    /// Monocular depth scales linearly with the face's assumed size, so this
    /// is exact for the depth along the camera axis.
    pub fn set_distance(&mut self, measured_distance_m: f64, true_distance_m: f64) -> Result<(), GeometryError> {
        if !(measured_distance_m > 0.05 && true_distance_m > 0.05) {
            return Err(GeometryError::OutOfRange(format!("distances must be positive, got {measured_distance_m} and {true_distance_m}")));
        }
        let ipd = self.ipd_m * true_distance_m / measured_distance_m;
        check("ipd_m", ipd, IPD_M)?;
        self.ipd_m = ipd;
        Ok(())
    }
}

/// How the lens is described in a profile.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CameraProfile {
    /// Uncalibrated: a pinhole from a field of view.
    Fov { width: u32, height: u32, hfov_deg: f64 },
    /// Calibrated: the `camera::Intrinsics` record as JSON (any lens model
    /// the workspace camera model supports).
    Intrinsics { intrinsics: serde_json::Value },
}

/// The persisted calibration: flat, human-editable, every length in metres.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub display_width_m: f64,
    pub display_height_m: f64,
    /// Optical centre above the top edge of the visible area (negative:
    /// below it, a camera in the glass).
    pub camera_gap_m: f64,
    /// Downward tilt of the camera relative to the display.
    /// [`Calibration::recentre`] writes this.
    #[serde(default)]
    pub camera_pitch_deg: f64,
    /// Sideways turn of the camera, toward the viewer's right.
    /// [`Calibration::recentre`] writes this.
    #[serde(default)]
    pub camera_yaw_deg: f64,
    /// Horizontal offset of the camera from the display centre, toward the
    /// viewer's right: a measured fact for a camera that is not centred.
    #[serde(default)]
    pub camera_offset_x_m: f64,
    pub camera: CameraProfile,
    pub ipd_m: f64,
}

fn check(name: &str, v: f64, (lo, hi): (f64, f64)) -> Result<(), GeometryError> {
    if v.is_finite() && v >= lo && v <= hi {
        Ok(())
    } else {
        Err(GeometryError::OutOfRange(format!("{name} = {v} is outside [{lo}, {hi}]")))
    }
}

impl Profile {
    /// Validate and build the calibration this profile describes.
    pub fn calibration(&self) -> Result<Calibration, GeometryError> {
        check("display_width_m", self.display_width_m, DISPLAY_SIZE_M)?;
        check("display_height_m", self.display_height_m, DISPLAY_SIZE_M)?;
        check("camera_gap_m", self.camera_gap_m, CAMERA_GAP_M)?;
        check("camera_pitch_deg", self.camera_pitch_deg, CAMERA_PITCH_DEG)?;
        check("camera_yaw_deg", self.camera_yaw_deg, CAMERA_YAW_DEG)?;
        check("camera_offset_x_m", self.camera_offset_x_m, (-self.display_width_m, self.display_width_m))?;
        check("ipd_m", self.ipd_m, IPD_M)?;
        let camera = match &self.camera {
            CameraProfile::Fov { width, height, hfov_deg } => {
                check("camera.hfov_deg", *hfov_deg, HFOV_DEG)?;
                if *width == 0 || *height == 0 {
                    return Err(GeometryError::OutOfRange(format!("camera.width x height = {width} x {height}")));
                }
                let f = *width as f64 / 2.0 / (hfov_deg.to_radians() / 2.0).tan();
                Intrinsics::pinhole(f, *width, *height)
            }
            CameraProfile::Intrinsics { intrinsics } => {
                Intrinsics::from_json(intrinsics).map_err(|e| GeometryError::OutOfRange(format!("camera.intrinsics: {e}")))?
            }
        };
        let display = Display::flat(self.display_width_m, self.display_height_m)?;
        let mut mount = CameraMount::above(&display, self.camera_gap_m);
        mount.position = add(mount.position, scale(display.right(), self.camera_offset_x_m));
        let mount = mount.aimed(self.camera_pitch_deg, self.camera_yaw_deg);
        Ok(Calibration { camera, mount, display, ipd_m: self.ipd_m })
    }

    /// The calibration for a camera delivering `width` x `height` frames,
    /// which need not be the size the profile describes (a driver may give a
    /// different mode than asked for). A field-of-view lens is rebuilt at the
    /// delivered size with square pixels; a calibrated lens is rescaled only
    /// when the aspect ratio is unchanged, because a crop or a different
    /// sensor mode is a different lens, not a scaled one.
    pub fn calibration_for(&self, width: u32, height: u32) -> Result<Calibration, GeometryError> {
        let mut c = self.calibration()?;
        c.camera = match &self.camera {
            CameraProfile::Fov { hfov_deg, .. } => {
                let f = width as f64 / 2.0 / (hfov_deg.to_radians() / 2.0).tan();
                Intrinsics::pinhole(f, width, height)
            }
            CameraProfile::Intrinsics { .. } => {
                let (a, b) = (c.camera.width as f64 / c.camera.height as f64, width as f64 / height as f64);
                if (a / b - 1.0).abs() > 0.01 {
                    return Err(GeometryError::OutOfRange(format!(
                        "the calibrated lens is {}x{} but the camera delivers {width}x{height}: a different aspect is a different lens, recalibrate it",
                        c.camera.width, c.camera.height
                    )));
                }
                c.camera.resized(width, height)
            }
        };
        Ok(c)
    }

    /// The profile of a calibration built from this one, after
    /// [`Calibration::recentre`] re-aimed the camera and
    /// [`Calibration::set_distance`] rescaled the face. The position and
    /// the lens record are carried from `self`.
    pub fn updated(&self, c: &Calibration) -> Profile {
        let (camera_pitch_deg, camera_yaw_deg) = c.mount.aim_relative_to(&CameraMount::above(&c.display, self.camera_gap_m));
        Profile { camera_pitch_deg, camera_yaw_deg, ipd_m: c.ipd_m, ..self.clone() }
    }
}

/// `a` in radians folded into (-pi, pi].
fn wrap_pi(a: f64) -> f64 {
    let t = a.rem_euclid(std::f64::consts::TAU);
    if t > std::f64::consts::PI {
        t - std::f64::consts::TAU
    } else {
        t
    }
}
