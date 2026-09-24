// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Geometry regularizers: the terms that make a fit put its gaussians ON a
//! surface rather than merely somewhere that renders the right colour.
//!
//! An RGB loss is satisfied by a stack of semi-transparent layers along a ray
//! as happily as by one opaque surface, and by gaussians whose flat axis
//! points anywhere. Both render correctly from the training views and fall
//! apart from any other: layers become fog, misoriented discs become fur. The
//! two terms here are from 2D Gaussian Splatting (Huang et al., SIGGRAPH
//! 2024), implemented from the paper:
//!
//! * **Depth distortion** `Σ_ij w_i w_j (z_i - z_j)²` over each ray's
//!   compositing weights `w = Tα`. Expanded it is `2(A·M2 - M1²)` with
//!   `A = Σw`, `M1 = Σ w z`, `M2 = Σ w z²` - three quantities the rasterizer
//!   already composites, so it costs one extra pass instead of a new kernel.
//!   (2DGS uses `|z_i - z_j|` on NDC depth; the squared form is the one whose
//!   moments composite, and the fit's scene normalization keeps depth O(1).)
//! * **Normal consistency** `Σ_i w_i (1 - n_iᵀ N)`, which is `A - N_pᵀ N`
//!   with `N_p = Σ w_i n_i` the composited normal - LINEAR in what the
//!   rasterizer composites. `N` is the normal of the rendered depth map itself
//!   (held fixed, the 2DGS formulation) or a supervised prior normal.
//!
//! A gaussian's normal is its shortest axis, turned to face the camera.
//!
//! Both are evaluated through AUXILIARY render passes: the same rasterizer
//! with each gaussian's colour replaced by a per-gaussian feature (its normal,
//! or its squared depth), so the forward composites exactly the sums above and
//! the backward returns d/d(feature) in the colour slot and the alpha-chain
//! gradient into the geometry itself. The feature gradients are carried back
//! to means and rotations on the host here.
//!
//! Swedish Embedded AB implements surface-accurate 3D reconstruction for its
//! clients. If your team needs splat scenes whose geometry holds up away from
//! the capture path, you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::types::Camera;

/// Words per gaussian in the fit's packed geometry `{mean, scale, quat}`.
pub const GEO: usize = 10;

/// Unit quaternion (w, x, y, z) of a raw one, and its norm.
fn unit(q: &[f32]) -> ([f32; 4], f32) {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-8);
    ([q[0] / n, q[1] / n, q[2] / n, q[3] / n], n)
}

/// Column `k` of the rotation of unit quaternion `q` - the gaussian's k-th
/// axis in world space, in the convention `splat_project.wgsl` builds its
/// covariance with.
pub fn axis(q: [f32; 4], k: usize) -> [f32; 3] {
    let [w, x, y, z] = q;
    match k {
        0 => [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
        1 => [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
        _ => [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
    }
}

/// d axis(q, k) / d q, as `[component][w x y z]`.
fn axis_jacobian(q: [f32; 4], k: usize) -> [[f32; 4]; 3] {
    let [w, x, y, z] = q;
    match k {
        0 => [
            [0.0, 0.0, -4.0 * y, -4.0 * z],
            [2.0 * z, 2.0 * y, 2.0 * x, 2.0 * w],
            [-2.0 * y, 2.0 * z, -2.0 * w, 2.0 * x],
        ],
        1 => [
            [-2.0 * z, 2.0 * y, 2.0 * x, -2.0 * w],
            [0.0, -4.0 * x, 0.0, -4.0 * z],
            [2.0 * x, 2.0 * w, 2.0 * z, 2.0 * y],
        ],
        _ => [
            [2.0 * y, 2.0 * z, 2.0 * w, 2.0 * x],
            [-2.0 * x, -2.0 * w, 2.0 * z, 2.0 * y],
            [0.0, -4.0 * x, -4.0 * y, 0.0],
        ],
    }
}

fn shortest(s: &[f32]) -> usize {
    if s[0] <= s[1] && s[0] <= s[2] {
        0
    } else if s[1] <= s[2] {
        1
    } else {
        2
    }
}

fn to_cam(vm: &[f32; 12], p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| vm[r * 4] * p[0] + vm[r * 4 + 1] * p[1] + vm[r * 4 + 2] * p[2] + vm[r * 4 + 3])
}

fn rot_cam(vm: &[f32; 12], v: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| vm[r * 4] * v[0] + vm[r * 4 + 1] * v[1] + vm[r * 4 + 2] * v[2])
}

/// Camera-space depth of every gaussian's mean - the `z` the rasterizer
/// composites into its depth output.
pub fn depths(geo: &[f32], cam: &Camera) -> Vec<f32> {
    let vm = cam.viewmat();
    geo.chunks_exact(GEO).map(|g| to_cam(&vm, [g[0], g[1], g[2]])[2]).collect()
}

/// Per-gaussian features for the distortion pass: `(z², 0, 0)`.
pub fn distortion_features(z: &[f32]) -> Vec<f32> {
    z.iter().flat_map(|&v| [v * v, 0.0, 0.0]).collect()
}

/// Every gaussian's normal in camera space (its shortest axis, turned to
/// face the camera), interleaved `[N*3]`.
pub fn normals(geo: &[f32], cam: &Camera) -> Vec<f32> {
    let vm = cam.viewmat();
    let mut out = Vec::with_capacity(geo.len() / GEO * 3);
    for g in geo.chunks_exact(GEO) {
        let (q, _) = unit(&g[6..10]);
        let n = rot_cam(&vm, axis(q, shortest(&g[3..6])));
        let c = to_cam(&vm, [g[0], g[1], g[2]]);
        let s = if n[0] * c[0] + n[1] * c[1] + n[2] * c[2] > 0.0 { -1.0 } else { 1.0 };
        out.extend_from_slice(&[s * n[0], s * n[1], s * n[2]]);
    }
    out
}

/// Accumulate into `d_geo` (`[N*GEO]`, the packed layout) the gradient of the
/// loss with respect to the geometry, given its gradient with respect to the
/// distortion features `d_feat` (`[N*3]`, only the first slot used).
pub fn distortion_backward(geo: &[f32], cam: &Camera, d_feat: &[f32], d_geo: &mut [f32]) {
    let vm = cam.viewmat();
    for (i, g) in geo.chunks_exact(GEO).enumerate() {
        let z = to_cam(&vm, [g[0], g[1], g[2]])[2];
        let dz = 2.0 * z * d_feat[i * 3];
        for k in 0..3 {
            d_geo[i * GEO + k] += dz * vm[8 + k];
        }
    }
}

/// Accumulate into `d_geo` the gradient with respect to the raw quaternions,
/// given the gradient with respect to the camera-space normals `d_n`
/// (`[N*3]`). The facing flip and the choice of shortest axis are held fixed,
/// as they are piecewise constant.
pub fn normals_backward(geo: &[f32], cam: &Camera, d_n: &[f32], d_geo: &mut [f32]) {
    let vm = cam.viewmat();
    for (i, g) in geo.chunks_exact(GEO).enumerate() {
        let (q, qn) = unit(&g[6..10]);
        let k = shortest(&g[3..6]);
        let a = axis(q, k);
        let n = rot_cam(&vm, a);
        let c = to_cam(&vm, [g[0], g[1], g[2]]);
        let s = if n[0] * c[0] + n[1] * c[1] + n[2] * c[2] > 0.0 { -1.0 } else { 1.0 };
        // world-space gradient of the axis: s * R_w2c^T d_n
        let dn = &d_n[i * 3..i * 3 + 3];
        let da: [f32; 3] = std::array::from_fn(|col| s * (vm[col] * dn[0] + vm[4 + col] * dn[1] + vm[8 + col] * dn[2]));
        let j = axis_jacobian(q, k);
        let mut dq = [0.0f32; 4];
        for (c, row) in j.iter().enumerate() {
            for t in 0..4 {
                dq[t] += da[c] * row[t];
            }
        }
        // through the normalization q = raw / |raw|
        let dot = dq[0] * q[0] + dq[1] * q[1] + dq[2] * q[2] + dq[3] * q[3];
        for t in 0..4 {
            d_geo[i * GEO + 6 + t] += (dq[t] - q[t] * dot) / qn;
        }
    }
}

/// Upstream gradients of one auxiliary pass: `dimg` is `[W*H*4]` (feature
/// channels + alpha), `ddepth` `[W*H]` is dLoss/d(accumulated depth).
pub struct AuxGrad {
    pub loss: f64,
    pub dimg: Vec<f32>,
    pub ddepth: Option<Vec<f32>>,
}

/// Depth distortion of a frame, `weight · Σ_p m_p·2(A·M2 - M1²) / wsum`,
/// from the distortion pass's output `rgba` (`M2` in the red channel, `A` in
/// alpha) and its expected depth `depth` (`M1 = depth·A`).
pub fn distortion_loss(rgba: &[f32], depth: &[f32], weights: Option<&[f32]>, wsum: f64, weight: f32) -> AuxGrad {
    let px = depth.len();
    let mut dimg = vec![0.0f32; px * 4];
    let mut ddepth = vec![0.0f32; px];
    let mut loss = 0.0f64;
    let k = (2.0 * weight as f64 / wsum) as f32;
    for p in 0..px {
        let m = weights.map_or(1.0, |w| w[p]);
        let a = rgba[p * 4 + 3];
        if m == 0.0 || a < crate::renderer::MIN_DEPTH_ALPHA {
            continue;
        }
        let m2 = rgba[p * 4];
        let m1 = depth[p] * a;
        loss += (k * m * (a * m2 - m1 * m1)) as f64;
        dimg[p * 4] = k * m * a;
        dimg[p * 4 + 3] = k * m * m2;
        ddepth[p] = -2.0 * k * m * m1;
    }
    AuxGrad { loss, dimg, ddepth: Some(ddepth) }
}

/// The normal of the rendered depth map at every pixel, camera space, turned
/// to face the camera; zero where the depth is not defined on the pixel and
/// its four neighbours.
pub fn depth_normals(depth: &[f32], alpha: &[f32], cam: &Camera) -> Vec<f32> {
    let (w, h) = (cam.width as usize, cam.height as usize);
    let mut out = vec![0.0f32; w * h * 3];
    let point = |x: usize, y: usize| -> [f32; 3] {
        let d = depth[y * w + x];
        [d * (x as f32 + 0.5 - cam.cx) / cam.fx, d * (y as f32 + 0.5 - cam.cy) / cam.fy, d]
    };
    let ok = |x: usize, y: usize| alpha[y * w + x] > 0.5 && depth[y * w + x] > 0.0;
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if !(ok(x, y) && ok(x - 1, y) && ok(x + 1, y) && ok(x, y - 1) && ok(x, y + 1)) {
                continue;
            }
            let (l, r, u, d) = (point(x - 1, y), point(x + 1, y), point(x, y - 1), point(x, y + 1));
            let dx = [r[0] - l[0], r[1] - l[1], r[2] - l[2]];
            let dy = [d[0] - u[0], d[1] - u[1], d[2] - u[2]];
            let mut n = crate::types::cross3(dx, dy);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len < 1e-12 {
                continue;
            }
            let c = point(x, y);
            let s = if n[0] * c[0] + n[1] * c[1] + n[2] * c[2] > 0.0 { -1.0 / len } else { 1.0 / len };
            n = [n[0] * s, n[1] * s, n[2] * s];
            out[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&n);
        }
    }
    out
}

/// Normal agreement, `weight · Σ_p m_p (A_p - N_pᵀ T_p) / wsum` over pixels
/// where the target normal `T` (`[W*H*3]`, zero = no target) is defined, from
/// the normals pass's output `rgba` (`N_p` in rgb, `A` in alpha). Accumulates
/// into `dimg`.
pub fn normal_loss(rgba: &[f32], target: &[f32], weights: Option<&[f32]>, wsum: f64, weight: f32, dimg: &mut [f32]) -> f64 {
    let px = rgba.len() / 4;
    let k = (weight as f64 / wsum) as f32;
    let mut loss = 0.0f64;
    for p in 0..px {
        let t = &target[p * 3..p * 3 + 3];
        let m = weights.map_or(1.0, |w| w[p]);
        if m == 0.0 || (t[0] == 0.0 && t[1] == 0.0 && t[2] == 0.0) {
            continue;
        }
        let n = &rgba[p * 4..p * 4 + 3];
        let a = rgba[p * 4 + 3];
        loss += (k * m * (a - (n[0] * t[0] + n[1] * t[1] + n[2] * t[2]))) as f64;
        for c in 0..3 {
            dimg[p * 4 + c] -= k * m * t[c];
        }
        dimg[p * 4 + 3] += k * m;
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera {
        Camera::look_at([0.3, -0.2, -0.5], [0.0, 0.0, 3.0], [0.0, -1.0, 0.0], 50.0, 16, 12)
    }

    fn scene() -> Vec<f32> {
        let mut g = Vec::new();
        for i in 0..5 {
            let f = i as f32;
            g.extend_from_slice(&[0.1 * f - 0.2, 0.05 * f, 3.0 + 0.1 * f]);
            g.extend_from_slice(&[0.2 + 0.01 * f, 0.1, 0.05 + 0.04 * f]);
            g.extend_from_slice(&[0.9 + 0.1 * f, 0.2 - 0.1 * f, 0.3, -0.1 + 0.05 * f]);
        }
        g
    }

    /// The host chain rules are the derivatives of the features they claim to
    /// differentiate: checked by central differences of `normals` / `depths`.
    #[test]
    fn feature_gradients_match_finite_differences() {
        let (geo, c) = (scene(), cam());
        let n = geo.len() / GEO;
        let up: Vec<f32> = (0..n * 3).map(|i| ((i * 7 % 5) as f32 - 2.0) / 2.0).collect();
        let fnorm = |g: &[f32]| -> f64 { normals(g, &c).iter().zip(&up).map(|(a, b)| (a * b) as f64).sum() };
        let fdist = |g: &[f32]| -> f64 {
            distortion_features(&depths(g, &c)).iter().zip(&up).map(|(a, b)| (a * b) as f64).sum()
        };
        let mut dn = vec![0.0f32; geo.len()];
        normals_backward(&geo, &c, &up, &mut dn);
        let mut dd = vec![0.0f32; geo.len()];
        distortion_backward(&geo, &c, &up, &mut dd);
        for (f, d, what) in [(&fnorm as &dyn Fn(&[f32]) -> f64, &dn, "normals"), (&fdist, &dd, "distortion")] {
            for i in 0..geo.len() {
                let eps = 1e-3;
                let (mut a, mut b) = (geo.clone(), geo.clone());
                a[i] += eps;
                b[i] -= eps;
                let fd = (f(&a) - f(&b)) / (2.0 * eps as f64);
                assert!((fd - d[i] as f64).abs() < 1e-2 * fd.abs().max(1.0), "{what} word {i}: fd {fd} vs {}", d[i]);
            }
        }
    }

    /// A fronto-parallel plane's depth map has the normal facing the camera.
    #[test]
    fn a_plane_facing_the_camera_has_the_camera_axis_as_its_normal() {
        let c = Camera::look_at([0.0; 3], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 50.0, 8, 8);
        let d = vec![2.0f32; 64];
        let a = vec![1.0f32; 64];
        let n = depth_normals(&d, &a, &c);
        let p = (3 * 8 + 3) * 3;
        assert!((n[p + 2] + 1.0).abs() < 1e-5 && n[p].abs() < 1e-5 && n[p + 1].abs() < 1e-5, "{:?}", &n[p..p + 3]);
    }
}
