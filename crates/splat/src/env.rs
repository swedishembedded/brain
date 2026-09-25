// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The environment a scene is seen against: sky, distant buildings, anything
//! too far away for the photographs to place.
//!
//! A splat scene has no "far away". Whatever a view shows past its gaussians
//! the fit must explain with gaussians, and the only ones it has are the
//! scene's, so a sky becomes a smear of large gaussians just behind the
//! scene - floaters from every other viewpoint, and wrong wherever parallax
//! should have vanished. An environment is what is left once parallax has:
//! radiance as a function of DIRECTION alone, composited behind every pixel
//! with the transmittance the gaussians leave,
//!
//! ```text
//!   C = C_splats + (1 - alpha) E(d),   E(d) = max(0, sum_k Y_k(d) c_k)
//! ```
//!
//! a real spherical-harmonic expansion of degree up to 8 in the pixel's own
//! world-space ray direction (through the lens, like the render). It is
//! fitted with the scene (`FitCfg::environment`); a standard splat viewer has
//! no environment, so [`EnvMap::to_splats`] bakes one into a shell of distant
//! gaussians for export.
//!
//! Swedish Embedded AB implements 3D reconstruction of outdoor captures,
//! including the sky and the distant scenery around them, for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu};

use crate::renderer::{dispatched_groups, ray_view_params};
use crate::types::{Camera, RenderOpts, Splats};
use crate::Kernels;

/// The highest expansion degree the kernels evaluate.
pub const MAX_DEGREE: u32 = 8;

/// Coefficients per channel of a degree-`degree` expansion.
pub fn coeffs(degree: u32) -> usize {
    ((degree + 1) * (degree + 1)) as usize
}

/// The real spherical-harmonic basis at unit direction `d`, index
/// `l*l + l + m` - `lib/sh_env.wgsl`'s formula, in f64.
pub fn basis(d: [f64; 3], degree: u32) -> Vec<f64> {
    let lmax = degree.min(MAX_DEGREE) as usize;
    let mut y = vec![0.0f64; (lmax + 1) * (lmax + 1)];
    let (mut cm, mut sm, mut qmm) = (1.0f64, 0.0f64, 1.0f64);
    for m in 0..=lmax {
        if m > 0 {
            let c2 = d[0] * cm - d[1] * sm;
            sm = d[0] * sm + d[1] * cm;
            cm = c2;
            qmm = -qmm * (2 * m - 1) as f64;
        }
        let (mut q2, mut q1) = (0.0f64, 0.0f64);
        for l in m..=lmax {
            let q = if l == m {
                qmm
            } else if l == m + 1 {
                (2 * m + 1) as f64 * d[2] * qmm
            } else {
                ((2 * l - 1) as f64 * d[2] * q1 - (l + m - 1) as f64 * q2) / (l - m) as f64
            };
            q2 = q1;
            q1 = q;
            let ratio: f64 = ((l - m + 1)..=(l + m)).map(|j| 1.0 / j as f64).product();
            let k = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI) * ratio).sqrt();
            let at = l * l + l;
            if m == 0 {
                y[at] = k * q;
            } else {
                y[at + m] = std::f64::consts::SQRT_2 * k * q * cm;
                y[at - m] = std::f64::consts::SQRT_2 * k * q * sm;
            }
        }
    }
    y
}

/// Radiance by direction: `coeffs(degree)` RGB coefficients, interleaved
/// `[k][channel]`.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvMap {
    pub degree: u32,
    pub coeffs: Vec<f32>,
}

impl EnvMap {
    /// Black everywhere.
    pub fn new(degree: u32) -> EnvMap {
        assert!(degree <= MAX_DEGREE, "an environment of degree {degree}: at most {MAX_DEGREE}");
        EnvMap { degree, coeffs: vec![0.0; 3 * coeffs(degree)] }
    }

    /// `rgb` in every direction.
    pub fn uniform(degree: u32, rgb: [f32; 3]) -> EnvMap {
        let mut e = EnvMap::new(degree);
        let y00 = basis([0.0, 0.0, 1.0], 0)[0] as f32;
        for (dc, v) in e.coeffs.iter_mut().zip(rgb) {
            *dc = v / y00;
        }
        e
    }

    /// The radiance along unit world direction `d`.
    pub fn eval(&self, d: [f32; 3]) -> [f32; 3] {
        let y = basis(d.map(f64::from), self.degree);
        let mut e = [0.0f64; 3];
        for (k, yk) in y.iter().enumerate() {
            for (c, ec) in e.iter_mut().enumerate() {
                *ec += yk * self.coeffs[k * 3 + c] as f64;
            }
        }
        e.map(|v| v.max(0.0) as f32)
    }

    /// The environment as a shell of `count` gaussians at `radius` around
    /// `centre`, each coloured with the radiance in its direction and sized to
    /// close the shell - what a viewer without an environment can render.
    pub fn to_splats(&self, centre: [f32; 3], radius: f32, count: usize) -> Splats {
        let mut s = Splats::default();
        // spacing of a Fibonacci sphere of `count` points
        let spacing = radius * (4.0 * std::f32::consts::PI / count.max(1) as f32).sqrt();
        for i in 0..count {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = i as f32 * 2.399_963_2;
            let d = [r * phi.cos(), r * phi.sin(), z];
            s.means.extend_from_slice(&[centre[0] + radius * d[0], centre[1] + radius * d[1], centre[2] + radius * d[2]]);
            s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            s.scales.extend_from_slice(&[0.6 * spacing; 3]);
            s.opacities.push(0.99);
            s.colors.extend_from_slice(&self.eval(d));
        }
        s
    }
}

/// An environment's coefficients on the device, and the scratch its
/// gradient is reduced in.
pub struct EnvDevice {
    degree: u32,
    coeffs: DeviceBuffer,
    partial: DeviceBuffer,
    partial_cap: usize,
    /// Bound as the unused `dimg` of a composite: one buffer may not be
    /// bound writable twice in one dispatch.
    unused: DeviceBuffer,
}

/// Pixel blocks the coefficient gradient is split over.
const GRAD_BLOCKS: usize = 256;

impl EnvDevice {
    pub fn new(gpu: &Gpu, env: &EnvMap) -> EnvDevice {
        let d = EnvDevice { degree: env.degree, coeffs: gpu.storage(env.coeffs.len() as u64), partial: gpu.storage(1), partial_cap: 0, unused: gpu.storage(4) };
        d.upload(gpu, env);
        d
    }

    /// Replace the coefficients (same degree).
    pub fn upload(&self, gpu: &Gpu, env: &EnvMap) {
        assert_eq!(env.degree, self.degree, "an environment keeps its degree");
        gpu.write_f32(&self.coeffs, &env.coeffs);
    }

    fn params(&self, cam: &Camera, o: &RenderOpts, tail: [u32; 4]) -> Vec<u32> {
        let mut p = ray_view_params(0, cam, &RenderOpts { ray: true, ..*o }).to_vec();
        p.extend_from_slice(&tail);
        p
    }

    /// Composite the environment behind `img` (`[W*H*4]`, a render over
    /// black), in place.
    pub fn composite(&self, gpu: &Gpu, ks: &Kernels, img: &DeviceBuffer, cam: &Camera, o: &RenderOpts) {
        let px = cam.width * cam.height;
        let s = gpu.step(ks.splat_env, &[&self.coeffs, img, &self.unused], &self.params(cam, o, [self.degree, 0, 0, 0]), px);
        gpu.submit(&[], &[s]);
    }

    /// The backward of [`Self::composite`] against `dimg` (dL/d(rgb, alpha)
    /// of the composited frame `img`): moves the environment's share into
    /// `dimg`'s alpha channel, in place, for the rasterizer's backward, and
    /// returns dL/dcoefficients.
    pub fn backward(&mut self, gpu: &Gpu, ks: &Kernels, img: &DeviceBuffer, dimg: &DeviceBuffer, cam: &Camera, o: &RenderOpts) -> Vec<f64> {
        let n_pix = (cam.width * cam.height) as usize;
        let entries = 3 * coeffs(self.degree);
        let chunks = entries.div_ceil(16);
        let blocks = GRAD_BLOCKS.min(n_pix.div_ceil(64)).max(1);
        let groups = blocks * chunks;
        let words = 16 * dispatched_groups(64 * groups);
        if words > self.partial_cap {
            self.partial = gpu.storage(words as u64);
            self.partial_cap = words;
        }
        // the coefficients' gradient reads dL/dC before the alpha share is
        // written into the same buffer
        let g = gpu.dispatch(
            ks.splat_env_grad,
            &[&self.coeffs, img, dimg, &self.partial],
            &self.params(cam, o, [self.degree, blocks as u32, 0, 0]),
            gpu_core::Dispatch::Workgroups(groups as u32),
        );
        let a = gpu.step(ks.splat_env, &[&self.coeffs, img, dimg], &self.params(cam, o, [self.degree, 1, 0, 0]), n_pix as u32);
        gpu.submit(&[], &[g, a]);
        let part = gpu.read(&self.partial, 16 * groups);
        let mut grad = vec![0.0f64; entries];
        for (w, chunk) in part.chunks_exact(16).enumerate() {
            let base = (w / blocks) * 16;
            for (j, v) in chunk.iter().enumerate() {
                if base + j < entries {
                    grad[base + j] += *v as f64;
                }
            }
        }
        grad
    }
}
