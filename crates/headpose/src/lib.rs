// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Head-coupled perspective ("fish tank VR"): where a viewer's eyes are in
//! front of a physical screen, and the projection that turns that screen into
//! a window onto a fixed virtual world.
//!
//! Swedish Embedded AB implements camera-based tracking and calibrated
//! geometry for products that have to react to a person in front of a device.
//! If your team needs expertise in turning a webcam into a measuring
//! instrument, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What this crate owns, and what it does not
//!
//! It owns the geometry between "2D face landmarks in a camera image" and "a
//! view and projection matrix for this frame":
//!
//! ```text
//! landmarks (px) --fitting--> head pose in the camera frame
//!                --calibration (camera mount)--> eyes in the screen frame
//!                --estimator--> filtered, predicted to presentation time
//!                --projection--> off-axis view + projection through the display
//! ```
//!
//! It does NOT run a neural network (a landmark model is a separate crate
//! whose output this crate consumes as plain pixels), open a camera, or
//! render. The lens model is `camera::Intrinsics` and the pose solver is
//! `sfm::pnp`; neither is re-implemented here.
//!
//! # Frames and units
//!
//! Every length is in METRES and every time in SECONDS of one monotonic clock
//! the caller chooses (the capture timestamps and the presentation times it
//! passes in must share it).
//!
//! * **camera frame**: `camera`'s convention, +X right in the image, +Y down,
//!   +Z forward out of the lens.
//! * **head frame**: origin at the midpoint of the two pupils, +X toward the
//!   subject's LEFT (the image's right when they face the camera), +Y down,
//!   +Z back into the head. A face looking straight into the camera has a head
//!   frame parallel to the camera frame.
//! * **screen frame**: origin wherever the [`Display`] corners say (the flat
//!   default puts it at the centre of the visible area), +X right as the
//!   viewer sees it, +Y up, +Z out of the glass toward the viewer. The
//!   virtual scene is placed in this frame, so it stays fixed to the laptop
//!   while the viewer moves.

pub mod calibration;
pub mod display;
pub mod estimator;
pub mod face;
pub mod fitting;
pub mod projection;

pub use calibration::{Calibration, CameraMount};
pub use display::Display;
pub use estimator::{EstimatorConfig, Rejected, TrackedViewer, TrackingStatus, ViewerEstimator};
pub use face::FaceModel;
pub use fitting::{fit_head, FitError, HeadFit};
pub use projection::{off_axis, OffAxis};

/// A point or direction in 3D, in metres.
pub type V3 = [f64; 3];

/// Why a geometric configuration was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum GeometryError {
    /// A display whose corners do not span a rectangle.
    DegenerateDisplay(String),
    /// An eye on the far side of, or in, the display plane.
    EyeBehindDisplay { distance_m: f64 },
    /// Clip planes that cannot form a frustum.
    BadClipPlanes { near: f64, far: f64 },
    /// A value outside the range it was checked against.
    OutOfRange(String),
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryError::DegenerateDisplay(why) => write!(f, "display corners do not form a rectangle: {why}"),
            GeometryError::EyeBehindDisplay { distance_m } => {
                write!(f, "the eye is {distance_m:.3} m from the display plane on the wrong side; it must be in front of the glass")
            }
            GeometryError::BadClipPlanes { near, far } => write!(f, "clip planes need 0 < near < far, got near={near} far={far}"),
            GeometryError::OutOfRange(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for GeometryError {}
