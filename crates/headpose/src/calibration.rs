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
//!   screen": the residual horizontal offset is the camera's mounting error,
//!   absorbed into the mount.
//! * [`Calibration::set_distance`] - "my eyes are this far from the glass":
//!   monocular depth is only as good as the face's assumed size, so a
//!   measured distance rescales the interpupillary distance.

use camera::Intrinsics;
use serde::{Deserialize, Serialize};
use sfm::linalg::{add, dot, mm, mv, scale, M3};

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

    /// The same mount with the camera pitched `degrees` downward (positive
    /// looks toward the viewer's lap), about its own X axis.
    pub fn pitched(self, degrees: f64) -> CameraMount {
        let (s, c) = degrees.to_radians().sin_cos();
        // the new camera axes in the old camera frame (columns): +Z tips
        // toward +Y (down), +Y toward -Z
        let rx: M3 = [1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c];
        CameraMount { rotation: mm(&self.rotation, &rx), ..self }
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

    /// Absorb a viewer's horizontal offset into the mount, given where their
    /// cyclopean eye measures while they sit centred in front of the display.
    /// Only the display-horizontal component is corrected: people sit at
    /// whatever height they sit at, so vertical offset is real, not error.
    pub fn recentre(&mut self, measured_eye_screen: V3) {
        let r = self.display.right();
        let off = dot(add(measured_eye_screen, scale(self.display.centre(), -1.0)), r);
        self.mount.position = add(self.mount.position, scale(r, -off));
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
    #[serde(default)]
    pub camera_pitch_deg: f64,
    /// Horizontal offset of the camera from the display centre, toward the
    /// viewer's right. [`Calibration::recentre`] writes this.
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
        let mut mount = CameraMount::above(&display, self.camera_gap_m).pitched(self.camera_pitch_deg);
        mount.position = add(mount.position, scale(display.right(), self.camera_offset_x_m));
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

    /// The profile of a calibration built from a flat display, after
    /// [`Calibration::recentre`] / [`Calibration::set_distance`] moved it.
    /// Pitch and the lens record are carried from `self`.
    pub fn updated(&self, c: &Calibration) -> Profile {
        let base = CameraMount::above(&c.display, self.camera_gap_m);
        let offset = dot(add(c.mount.position, scale(base.position, -1.0)), c.display.right());
        Profile { camera_offset_x_m: offset, ipd_m: c.ipd_m, ..self.clone() }
    }
}
