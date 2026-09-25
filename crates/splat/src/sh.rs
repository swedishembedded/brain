// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! View-dependent colour, host side.
//!
//! A gaussian's colour is `base + sum_k Y_k(d) * sh[k]` along the direction it
//! is seen from. Evaluating that is per-gaussian and cheap, so rendering does
//! it here rather than in the rasterizer, which keeps the hot inner loop
//! reading a plain colour array whether or not a scene has harmonics.
//!
//! `splat_sh.wgsl` computes the same thing on device for the fit, and the two
//! are held to each other by test rather than by inspection.
//!
//! Swedish Embedded AB implements radiance-field renderers that reproduce a
//! glossy surface as a glossy surface. If your team needs that, you can
//! procure our services by sending an email to info@swedishembedded.com.

use crate::types::{Camera, RenderOpts, Splats};

/// Real spherical harmonics up to degree 3, in the order Inria's PLY stores
/// them. Writes `k` values and leaves the rest zero.
// The constants are written digit-for-digit as in `splat_sh.wgsl`, so the two
// implementations can be compared by eye; f32 rounds the extra digits away.
#[allow(clippy::excessive_precision)]
pub fn basis(dir: [f32; 3], k: usize) -> [f32; 15] {
    let mut y = [0.0f32; 15];
    if k == 0 {
        return y;
    }
    let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if l <= 1e-12 {
        return y;
    }
    let (x, yy, z) = (dir[0] / l, dir[1] / l, dir[2] / l);
    y[0] = -0.4886025119029199 * yy;
    y[1] = 0.4886025119029199 * z;
    y[2] = -0.4886025119029199 * x;
    if k > 3 {
        let (xx, y2, zz) = (x * x, yy * yy, z * z);
        y[3] = 1.0925484305920792 * x * yy;
        y[4] = -1.0925484305920792 * yy * z;
        y[5] = 0.31539156525252005 * (2.0 * zz - xx - y2);
        y[6] = -1.0925484305920792 * x * z;
        y[7] = 0.5462742152960396 * (xx - y2);
        if k > 8 {
            y[8] = -0.5900435899266435 * yy * (3.0 * xx - y2);
            y[9] = 2.890611442640554 * x * yy * z;
            y[10] = -0.4570457994644658 * yy * (4.0 * zz - xx - y2);
            y[11] = 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * y2);
            y[12] = -0.4570457994644658 * x * (4.0 * zz - xx - y2);
            y[13] = 1.445305721320277 * z * (xx - y2);
            y[14] = -0.5900435899266435 * x * (xx - 3.0 * y2);
        }
    }
    y
}

/// Coefficients per channel for an SH degree.
pub fn coeffs(degree: u32) -> usize {
    match degree {
        0 => 0,
        1 => 3,
        2 => 8,
        _ => 15,
    }
}

/// The colours a scene shows from `eye`. Returns `None` for a scene with no
/// harmonics, whose colours are already what it shows from everywhere.
pub fn shade(s: &Splats, eye: [f32; 3]) -> Option<Vec<f32>> {
    let (deg, rest) = s.sh_rest.as_ref()?;
    let k = coeffs(*deg);
    if k == 0 || rest.len() != s.len() * 3 * k {
        return None;
    }
    let mut out = vec![0.0f32; s.len() * 3];
    for i in 0..s.len() {
        let y = basis(
            [s.means[i * 3] - eye[0], s.means[i * 3 + 1] - eye[1], s.means[i * 3 + 2] - eye[2]],
            k,
        );
        for c in 0..3 {
            let b = i * 3 * k + c * k;
            let mut acc = s.colors[i * 3 + c];
            for (t, yt) in y.iter().take(k).enumerate() {
                acc += yt * rest[b + t];
            }
            out[i * 3 + c] = acc.max(0.0);
        }
    }
    Some(out)
}

/// Render a scene as it looks from `cam`, shading through its harmonics first.
pub fn render_rgb(
    g: &gpu_core::Gpu,
    ks: crate::Kernels,
    ren: &mut crate::renderer::Renderer,
    gs: &crate::renderer::GpuSplats,
    s: &Splats,
    cam: &Camera,
    o: &RenderOpts,
) -> Vec<f32> {
    let _ = ks;
    let eye = [cam.c2w[3], cam.c2w[7], cam.c2w[11]];
    let shaded = shade(s, eye).map(|c| crate::renderer::GpuSplats {
        n: gs.n,
        means: gs.means.clone(),
        quats: gs.quats.clone(),
        scales: gs.scales.clone(),
        opacities: gs.opacities.clone(),
        colors: g.storage_init("sh.colors", &c),
        filter3d: gs.filter3d.clone(),
    });
    ren.render(g, shaded.as_ref().unwrap_or(gs), cam, o);
    ren.read_rgba(g, cam.width, cam.height)
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Splats;

    /// The kernel the fit optimises through and the function rendering shades
    /// with are two implementations of one formula. If they drift, a scene
    /// looks different from the thing that was fitted, and nothing else in the
    /// suite would say so.
    #[test]
    fn the_device_basis_and_the_host_basis_are_the_same_function() {
        let g = gpu_core::Gpu::new_cpu(crate::PIPELINES);
        let ks = crate::Kernels::at(0);
        let (n, deg) = (64usize, 3u32);
        let k = coeffs(deg);

        let mut s = Splats::default();
        let mut rest = vec![0.0f32; n * 3 * k];
        for i in 0..n {
            let a = i as f32 * 0.911;
            s.means.extend_from_slice(&[a.sin() * 2.0, a.cos() * 1.3, 2.0 + (0.7 * a).sin()]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[0.05, 0.05, 0.05]);
            s.opacities.push(0.5);
            s.colors.extend_from_slice(&[0.4, 0.5, 0.6]);
            for t in 0..3 * k {
                rest[i * 3 * k + t] = 0.25 * ((i * 7 + t * 13) as f32 * 0.37).sin();
            }
        }
        s.sh_rest = Some((deg, rest.clone()));
        let eye = [0.3f32, -0.2, -1.1];

        let want = shade(&s, eye).expect("has harmonics");
        let means = g.storage_init("m", &s.means);
        let base = g.storage_init("b", &s.colors);
        let shb = g.storage_init("s", &rest);
        let out = g.storage(3 * n as u64);
        let dummy = g.storage(1);
        let dummy2 = g.storage(1);
        let step = g.step(
            ks.splat_sh,
            &[&means, &base, &shb, &out, &dummy, &dummy2],
            &[
                n as u32,
                k as u32,
                0,
                0,
                gpu_core::f(eye[0]),
                gpu_core::f(eye[1]),
                gpu_core::f(eye[2]),
                0,
            ],
            n as u32,
        );
        g.submit(&[], &[step]);
        let got = g.read(&out, 3 * n);

        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "host and device shading differ by {worst:.2e}");
        assert!(
            want.iter().any(|&v| v > 0.0) && want.iter().zip(&s.colors.iter().cycle().take(want.len()).collect::<Vec<_>>()).any(|(a, b)| (a - *b).abs() > 1e-3),
            "the harmonics changed nothing, so this compares two constants"
        );
    }
}
