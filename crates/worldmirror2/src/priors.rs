// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Encoding known camera poses and intrinsics into the tokens the trunk
//! reserves for them.
//!
//! WorldMirror is an "any-prior" model: the trunk carries one pose token and
//! one intrinsics token per frame, and training drops each independently with
//! probability 0.5 by ZEROING it. A run with no priors is therefore not a
//! different code path, it is the all-zero case - which is what this crate
//! did, and all it could do, because nothing ever filled those rows.
//!
//! Supplying them is worth more than the accuracy it buys. It is the only way
//! to ask which half of the pipeline a reconstruction failure lives in: give
//! the model the poses and, if the result comes good, the geometry was being
//! lost in camera prediction rather than in depth or assembly.
//!
//! Swedish Embedded AB implements conditional 3D reconstruction pipelines that
//! can accept whatever a capture rig already knows. If your team needs
//! reconstruction that improves when handed calibration rather than ignoring
//! it, you can procure our services by sending an email to
//! info@swedishembedded.com.

/// One frame's known camera, in the convention this crate uses everywhere
/// else: `c2w` row-major 4x4, rigid, with intrinsics in pixels.
#[derive(Clone, Copy, Debug)]
pub struct CameraPrior {
    pub c2w: [f32; 16],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// Scalar-LAST `xyzw` quaternion of a row-major 3x3 rotation.
///
/// Scalar-last because that is the convention the camera head predicts in and
/// [`crate::gaussians::decode_cameras`] reads back; a pose prior has to speak
/// the same dialect as the output it is meant to constrain.
pub fn quat_xyzw(r: &[f32; 9]) -> [f32; 4] {
    let t = r[0] + r[4] + r[8];
    if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [(r[7] - r[5]) / s, (r[2] - r[6]) / s, (r[3] - r[1]) / s, 0.25 * s]
    } else if r[0] > r[4] && r[0] > r[8] {
        let s = (1.0 + r[0] - r[4] - r[8]).sqrt() * 2.0;
        [0.25 * s, (r[1] + r[3]) / s, (r[2] + r[6]) / s, (r[7] - r[5]) / s]
    } else if r[4] > r[8] {
        let s = (1.0 + r[4] - r[0] - r[8]).sqrt() * 2.0;
        [(r[1] + r[3]) / s, 0.25 * s, (r[5] + r[7]) / s, (r[2] - r[6]) / s]
    } else {
        let s = (1.0 + r[8] - r[0] - r[4]).sqrt() * 2.0;
        [(r[2] + r[6]) / s, (r[5] + r[7]) / s, 0.25 * s, (r[3] - r[1]) / s]
    }
}

/// The 7-vector each pose token is projected from: `[quaternion(4),
/// translation(3)]`, with the translation normalized ACROSS THE SEQUENCE by
/// `(t - centroid) / max_distance_to_centroid`.
///
/// The normalization is what makes a pose prior usable at all. A
/// reconstruction is only defined up to a similarity, so absolute camera
/// positions in metres carry a scale the model has no way to honour;
/// centring on the camera centroid and dividing by the furthest camera's
/// distance produces the same numbers whatever units the rig measured in.
/// It also means the priors for a sequence must be encoded TOGETHER - one
/// frame's token depends on where the others are.
pub fn pose_vectors(cams: &[CameraPrior]) -> Vec<[f32; 7]> {
    let n = cams.len();
    if n == 0 {
        return Vec::new();
    }
    let centres: Vec<[f32; 3]> = cams.iter().map(|c| [c.c2w[3], c.c2w[7], c.c2w[11]]).collect();
    let mut centroid = [0.0f32; 3];
    for p in &centres {
        for k in 0..3 {
            centroid[k] += p[k] / n as f32;
        }
    }
    let alpha = centres
        .iter()
        .map(|p| ((p[0] - centroid[0]).powi(2) + (p[1] - centroid[1]).powi(2) + (p[2] - centroid[2]).powi(2)).sqrt())
        .fold(0.0f32, f32::max)
        .max(1e-8);

    cams.iter()
        .zip(&centres)
        .map(|(c, p)| {
            let r = [c.c2w[0], c.c2w[1], c.c2w[2], c.c2w[4], c.c2w[5], c.c2w[6], c.c2w[8], c.c2w[9], c.c2w[10]];
            let q = quat_xyzw(&r);
            [q[0], q[1], q[2], q[3], (p[0] - centroid[0]) / alpha, (p[1] - centroid[1]) / alpha, (p[2] - centroid[2]) / alpha]
        })
        .collect()
}

/// The 4-vector each intrinsics token is projected from: focal lengths and
/// principal point normalized by the image size, so the token says "how wide
/// is this lens" rather than "how many pixels is this sensor".
pub fn intrinsic_vector(c: &CameraPrior, width: u32, height: u32) -> [f32; 4] {
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    [c.fx / w, c.fy / h, c.cx / w, c.cy / h]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(px: f32, py: f32, pz: f32) -> CameraPrior {
        let mut c2w = [0.0f32; 16];
        c2w[0] = 1.0; c2w[5] = 1.0; c2w[10] = 1.0; c2w[15] = 1.0;
        c2w[3] = px; c2w[7] = py; c2w[11] = pz;
        CameraPrior { c2w, fx: 400.0, fy: 410.0, cx: 256.0, cy: 192.0 }
    }

    /// Identity rotation is the identity quaternion, scalar LAST.
    #[test]
    fn the_quaternion_is_scalar_last() {
        let q = quat_xyzw(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!((q[3] - 1.0).abs() < 1e-6, "w should be 1.0 in slot 3, got {q:?}");
        assert!(q[..3].iter().all(|v| v.abs() < 1e-6), "xyz should be zero, got {q:?}");
        // a 180-degree turn about Y puts the magnitude in y, not in w
        let q = quat_xyzw(&[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]);
        assert!((q[1].abs() - 1.0).abs() < 1e-5 && q[3].abs() < 1e-5, "expected +-y, got {q:?}");
    }

    /// The whole point of the normalization: the same rig in different units,
    /// or shifted anywhere in the world, must produce the same tokens.
    #[test]
    fn pose_tokens_are_invariant_to_the_units_and_origin_of_the_capture() {
        let metres = [cam(0.0, 0.0, 0.0), cam(1.0, 0.0, 0.0), cam(0.0, 2.0, 0.0)];
        let millimetres_elsewhere =
            [cam(500.0, -300.0, 70.0), cam(1500.0, -300.0, 70.0), cam(500.0, 1700.0, 70.0)];
        let a = pose_vectors(&metres);
        let b = pose_vectors(&millimetres_elsewhere);
        for (x, y) in a.iter().zip(&b) {
            for k in 0..7 {
                assert!((x[k] - y[k]).abs() < 1e-5, "token differs in slot {k}: {x:?} vs {y:?}");
            }
        }
        // and the furthest camera sits exactly on the unit sphere
        let far = a.iter().map(|v| (v[4] * v[4] + v[5] * v[5] + v[6] * v[6]).sqrt()).fold(0.0f32, f32::max);
        assert!((far - 1.0).abs() < 1e-5, "the furthest camera should normalize to 1.0, got {far}");
    }

    /// A single camera has no scale to speak of and must not divide by zero.
    #[test]
    fn one_camera_is_not_a_division_by_zero() {
        let v = pose_vectors(&[cam(3.0, 4.0, 5.0)]);
        assert_eq!(v.len(), 1);
        assert!(v[0].iter().all(|x| x.is_finite()), "{:?}", v[0]);
        assert!(v[0][4..].iter().all(|x| x.abs() < 1e-6), "a lone camera is its own centroid");
    }

    #[test]
    fn intrinsics_are_normalized_by_the_image_size() {
        let v = intrinsic_vector(&cam(0.0, 0.0, 0.0), 512, 384);
        assert!((v[0] - 400.0 / 512.0).abs() < 1e-6);
        assert!((v[1] - 410.0 / 384.0).abs() < 1e-6);
        assert!((v[2] - 0.5).abs() < 1e-6);
        assert!((v[3] - 0.5).abs() < 1e-6);
    }
}
