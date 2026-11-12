// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A canonical metric face: the 3D positions, in the head frame, of the
//! landmarks a rigid fit may trust.
//!
//! # Which landmarks, and why so few
//!
//! The landmark index scheme is the 68-point one (iBUG 300-W) that most
//! landmark models emit, or its 66-point variant that drops the two inner
//! mouth corners (60 and 64); the indices used here are below 60, so they
//! mean the same point in both. Only points on bone-backed, expression-free
//! parts of the face are listed: the four eye corners and the nose. The jaw
//! contour is excluded on purpose - its landmarks slide along the silhouette
//! as the head turns, so they name a different 3D point in every pose - and
//! so are the brows and mouth, which move with expression.
//!
//! # Where the numbers come from
//!
//! Published adult anthropometric averages, rounded, with the pupils as the
//! reference: interpupillary distance 63 mm, biocular (outer canthus) width
//! about 90 mm, intercanthal (inner canthus) width about 32 mm with the
//! canthi set back from the corneal plane, a nose from nasion to subnasale of
//! about 52 mm whose tip stands about 22 mm proud of the pupils. They are a
//! generic face, not anyone's: what makes the fit metric is that the whole
//! model is scaled to the viewer's interpupillary distance
//! ([`FaceModel::canonical`]), which [`crate::Calibration`] lets a user
//! measure or calibrate.

use crate::V3;

/// The interpupillary distance the canonical coordinates are written at.
pub const CANONICAL_IPD_M: f64 = 0.063;

/// `(landmark index, head-frame position in millimetres)` at
/// [`CANONICAL_IPD_M`]. +X toward the subject's left, +Y down, +Z into the
/// head; the origin is the midpoint of the pupils.
const RIGID_MM: &[(usize, V3)] = &[
    // eye corners: 36/39 the subject's right eye (image left), 42/45 the left
    (36, [-45.0, -2.0, 10.0]),
    (39, [-16.0, 0.0, 5.0]),
    (42, [16.0, 0.0, 5.0]),
    (45, [45.0, -2.0, 10.0]),
    // nasal bridge, nasion to tip, evenly spaced as the 68-point annotation
    // places them
    (27, [0.0, -2.0, 2.0]),
    (28, [0.0, 14.0, -6.0]),
    (29, [0.0, 30.0, -13.0]),
    (30, [0.0, 46.0, -22.0]),
    // the lower edge of the nose, subject's right ala to left
    (31, [-13.0, 49.0, -6.0]),
    (32, [-7.0, 51.0, -10.0]),
    (33, [0.0, 52.0, -12.0]),
    (34, [7.0, 51.0, -10.0]),
    (35, [13.0, 49.0, -6.0]),
];

/// The subject's right and left pupil centres at [`CANONICAL_IPD_M`], mm.
const RIGHT_PUPIL_MM: V3 = [-31.5, 0.0, 0.0];
const LEFT_PUPIL_MM: V3 = [31.5, 0.0, 0.0];

/// A face model at one viewer's scale.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceModel {
    /// `(landmark index, head-frame position in metres)`.
    pub points: Vec<(usize, V3)>,
    /// The subject's right pupil, head frame, metres.
    pub right_eye: V3,
    /// The subject's left pupil, head frame, metres.
    pub left_eye: V3,
}

impl FaceModel {
    /// The canonical face scaled so its pupils are `ipd_m` apart.
    pub fn canonical(ipd_m: f64) -> FaceModel {
        let s = ipd_m / CANONICAL_IPD_M / 1000.0;
        let m = |p: V3| [p[0] * s, p[1] * s, p[2] * s];
        FaceModel {
            points: RIGID_MM.iter().map(|&(i, p)| (i, m(p))).collect(),
            right_eye: m(RIGHT_PUPIL_MM),
            left_eye: m(LEFT_PUPIL_MM),
        }
    }

    /// The highest landmark index the model reads; a landmark set must have
    /// more points than this.
    pub fn max_index(&self) -> usize {
        self.points.iter().map(|p| p.0).max().unwrap_or(0)
    }

    /// The model's two outer eye corners, the widest rigid pair, used to
    /// seed a fit's depth.
    pub(crate) fn outer_corners(&self) -> (V3, V3) {
        let get = |i| self.points.iter().find(|p| p.0 == i).map(|p| p.1).expect("canonical model lists both outer eye corners");
        (get(36), get(45))
    }
}
