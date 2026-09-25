// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A scene under optimization, resident on the device: raw parameters, their
//! Adam moments, the activated buffers the renderer draws, and the gradients
//! the backward writes.
//!
//! Raw parameters, per gaussian: `{mean (3), log scale (3), quaternion (4),
//! opacity logit}` packed 11 wide and stepped by `splat_adam.wgsl`, plus a
//! base colour (3) and the higher spherical-harmonic bands, stepped by the
//! generic `adamw.wgsl`.
//!
//! The optimizer state OUTLIVES a change of topology: [`DeviceScene::remap`]
//! carries every surviving gaussian's moments to its new slot, gives a split's
//! or a clone's children their parent's (a log scale's moments stay valid
//! when the scale is divided, which is part of why the scale is logarithmic),
//! and starts fresh only the samples density control placed from nothing.
//! The global Adam timestep runs across density rounds. The fit this replaced
//! reset every moment at every round, which measured ~3.5% worse final loss
//! from the lost momentum alone - paid exactly when refinement should
//! accelerate.
//!
//! Swedish Embedded AB implements GPU-resident optimizers for 3D
//! reconstruction. If your team needs training loops that never leave the
//! device then you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::{f, DeviceBuffer, Gpu};

use crate::renderer::{GpuSplats, SplatGrads};
use crate::types::Splats;
use crate::Kernels;

/// Words per gaussian of the raw geometry: mean, log scale, quaternion, logit.
pub(crate) const RAW: usize = 11;

/// Opacities are held inside (OPACITY_EPS, 1 - OPACITY_EPS), a logit of
/// about +-9.2: past it a sigmoid's gradient is gone and the gaussian can
/// never come back.
pub(crate) const OPACITY_EPS: f32 = 1e-4;

pub(crate) fn logit(o: f32) -> f32 {
    let o = o.clamp(OPACITY_EPS, 1.0 - OPACITY_EPS);
    (o / (1.0 - o)).ln()
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The step sizes of one [`DeviceScene::step`], already scheduled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rates {
    pub position: f32,
    pub scale: f32,
    pub rotation: f32,
    pub opacity: f32,
    pub color: f32,
    /// Higher SH bands, as 3DGS: a twentieth of the base colour's rate.
    pub sh: f32,
    /// `position` is in units of each gaussian's own largest axis rather than
    /// of the scene.
    pub relative: bool,
    /// SGLD amplitude (3DGS-MCMC's lambda_noise); 0 = plain descent.
    pub noise: f32,
    /// L1 weights on opacity and on scale (3DGS-MCMC's lambda_o, lambda_S).
    pub opacity_reg: f32,
    pub scale_reg: f32,
}

/// Shape bounds [`DeviceScene::step`] enforces, in linear terms.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub min_scale: f32,
    /// Largest longest-to-middle axis ratio; <= 1 = unbounded.
    pub max_needle: f32,
    /// Largest middle-to-shortest axis ratio; <= 1 = unbounded.
    pub max_flat: f32,
}

pub(crate) struct DeviceScene {
    pub n: usize,
    /// Higher SH coefficients per channel (0, 3, 8 or 15).
    pub ksh: usize,
    pub sh_degree: u32,
    geo: DeviceBuffer,
    m_geo: DeviceBuffer,
    v_geo: DeviceBuffer,
    /// Per-gaussian log-scale ceiling.
    smax: DeviceBuffer,
    pub base: DeviceBuffer,
    m_base: DeviceBuffer,
    v_base: DeviceBuffer,
    pub sh: DeviceBuffer,
    m_sh: DeviceBuffer,
    v_sh: DeviceBuffer,
    desc_base: DeviceBuffer,
    desc_sh: DeviceBuffer,
    unit_coef: DeviceBuffer,
    hparams: DeviceBuffer,
    /// What the renderer draws: means and quaternions are copies of the raw
    /// ones, scales and opacities activated, colours shaded per view.
    pub means: DeviceBuffer,
    pub scales: DeviceBuffer,
    pub quats: DeviceBuffer,
    pub opac: DeviceBuffer,
    pub col_view: DeviceBuffer,
    pub filter3d: DeviceBuffer,
    pub grads: SplatGrads,
    /// dL/d(base colour) and dL/d(SH bands), split out of the rendered
    /// colour's gradient by `splat_sh`'s VJP.
    pub d_base: DeviceBuffer,
    pub d_sh: DeviceBuffer,
    /// Per-gaussian cap on its SH coefficients per channel
    /// ([`DeviceScene::set_sh_limit`]); the host copy is what
    /// [`DeviceScene::snapshot`] zeroes the coefficients past.
    sh_limit: DeviceBuffer,
    sh_limit_host: Vec<u32>,
}

/// A host snapshot of everything [`DeviceScene`] carries, for a topology
/// change.
pub(crate) struct Snapshot {
    pub scene: Splats,
    geo_m: Vec<f32>,
    geo_v: Vec<f32>,
    base_m: Vec<f32>,
    base_v: Vec<f32>,
    sh_m: Vec<f32>,
    sh_v: Vec<f32>,
    sh_limit: Vec<u32>,
}

pub(crate) fn ksh_of(degree: u32) -> usize {
    match degree {
        0 => 0,
        1 => 3,
        2 => 8,
        _ => 15,
    }
}

impl DeviceScene {
    /// Upload `s` at `sh_degree` (bands it does not carry start at zero),
    /// with fresh optimizer state.
    pub fn new(gpu: &Gpu, s: &Splats, sh_degree: u32) -> DeviceScene {
        let n = s.len();
        let ksh = ksh_of(sh_degree);
        let zeros = |k: usize| vec![0.0f32; k];
        let snap = Snapshot {
            scene: s.clone(),
            geo_m: zeros(RAW * n),
            geo_v: zeros(RAW * n),
            base_m: zeros(3 * n),
            base_v: zeros(3 * n),
            sh_m: zeros(3 * n * ksh),
            sh_v: zeros(3 * n * ksh),
            sh_limit: vec![ksh as u32; n],
        };
        Self::from_snapshot(gpu, &snap, sh_degree)
    }

    fn from_snapshot(gpu: &Gpu, snap: &Snapshot, sh_degree: u32) -> DeviceScene {
        let s = &snap.scene;
        let n = s.len();
        let ksh = ksh_of(sh_degree);
        let mut raw = Vec::with_capacity(RAW * n);
        for i in 0..n {
            raw.extend_from_slice(&s.means[i * 3..i * 3 + 3]);
            raw.extend(s.scales[i * 3..i * 3 + 3].iter().map(|v| v.max(1e-12).ln()));
            raw.extend_from_slice(&s.quats[i * 4..i * 4 + 4]);
            raw.push(logit(s.opacities[i]));
        }
        let sh: Vec<f32> = match (&s.sh_rest, ksh) {
            (_, 0) => Vec::new(),
            (Some((d, r)), _) if ksh_of(*d) == ksh && r.len() == 3 * n * ksh => r.clone(),
            _ => vec![0.0; 3 * n * ksh],
        };
        let buf = |name: &str, v: &[f32]| gpu.storage_init(name, if v.is_empty() { &[0.0] } else { v });
        let desc = |numel: usize| {
            let words = kernels::adamw_desc(numel.max(1), 1.0);
            let d = gpu.storage(words.len() as u64);
            gpu.write(&d, &words);
            d
        };
        let unit_coef = gpu.storage(1);
        gpu.write(&unit_coef, &[f(1.0)]);
        let wide = |k: usize| gpu.storage(k.max(1) as u64);
        DeviceScene {
            n,
            ksh,
            sh_degree,
            geo: buf("fit.geo", &raw),
            m_geo: buf("fit.m_geo", &snap.geo_m),
            v_geo: buf("fit.v_geo", &snap.geo_v),
            smax: buf("fit.smax", &vec![f32::MAX.ln(); n]),
            base: buf("fit.base", &s.colors),
            m_base: buf("fit.m_base", &snap.base_m),
            v_base: buf("fit.v_base", &snap.base_v),
            sh: buf("fit.sh", &sh),
            m_sh: buf("fit.m_sh", &snap.sh_m),
            v_sh: buf("fit.v_sh", &snap.sh_v),
            desc_base: desc(3 * n),
            desc_sh: desc(3 * n * ksh),
            unit_coef,
            hparams: gpu.uniform_dynamic(8),
            means: wide(3 * n),
            scales: wide(3 * n),
            quats: wide(4 * n),
            opac: wide(n),
            col_view: wide(3 * n),
            filter3d: buf("fit.filter3d", &vec![0.0; n]),
            grads: SplatGrads::new(gpu, n.max(1)),
            d_base: wide(3 * n),
            d_sh: wide(3 * n * ksh),
            sh_limit: {
                let b = gpu.storage(n.max(1) as u64);
                if n > 0 {
                    gpu.write(&b, &snap.sh_limit);
                }
                b
            },
            sh_limit_host: snap.sh_limit.clone(),
        }
    }

    /// Everything, back on the host.
    pub fn snapshot(&self, gpu: &Gpu) -> Snapshot {
        let n = self.n;
        let raw = gpu.read(&self.geo, RAW * n);
        let mut s = Splats::default();
        for i in 0..n {
            let r = &raw[i * RAW..i * RAW + RAW];
            s.means.extend_from_slice(&r[0..3]);
            s.scales.extend(r[3..6].iter().map(|v| v.exp()));
            s.quats.extend_from_slice(&r[6..10]);
            s.opacities.push(sigmoid(r[10]));
        }
        s.colors = gpu.read(&self.base, 3 * n);
        if self.ksh > 0 {
            // what a gaussian's cap holds out renders as zero, so it is zero
            let mut rest = gpu.read(&self.sh, 3 * n * self.ksh);
            for (i, &cap) in self.sh_limit_host.iter().enumerate() {
                for c in 0..3 {
                    let row = &mut rest[(i * 3 + c) * self.ksh..(i * 3 + c + 1) * self.ksh];
                    row[(cap as usize).min(self.ksh)..].fill(0.0);
                }
            }
            s.sh_rest = Some((self.sh_degree, rest));
        }
        let k = 3 * n * self.ksh;
        Snapshot {
            scene: s,
            geo_m: gpu.read(&self.m_geo, RAW * n),
            geo_v: gpu.read(&self.v_geo, RAW * n),
            base_m: gpu.read(&self.m_base, 3 * n),
            base_v: gpu.read(&self.v_base, 3 * n),
            sh_m: if k > 0 { gpu.read(&self.m_sh, k) } else { Vec::new() },
            sh_v: if k > 0 { gpu.read(&self.v_sh, k) } else { Vec::new() },
            sh_limit: self.sh_limit_host.clone(),
        }
    }

    /// The scene alone.
    pub fn download(&self, gpu: &Gpu) -> Splats {
        self.snapshot(gpu).scene
    }

    /// Replace the scene by `next` - the same scene after density control -
    /// carrying optimizer state along `origin` (per gaussian of `next`, the
    /// gaussian of `old` whose state it continues; `None` = fresh).
    pub fn remap(gpu: &Gpu, old: &Snapshot, next: &Splats, origin: &[Option<usize>], sh_degree: u32) -> DeviceScene {
        assert_eq!(origin.len(), next.len(), "one origin per gaussian of the new scene");
        let ksh = ksh_of(sh_degree);
        let gather = |src: &[f32], width: usize| -> Vec<f32> {
            let mut out = vec![0.0f32; origin.len() * width];
            if src.is_empty() {
                return out;
            }
            for (j, o) in origin.iter().enumerate() {
                if let Some(i) = o {
                    out[j * width..j * width + width].copy_from_slice(&src[i * width..i * width + width]);
                }
            }
            out
        };
        let snap = Snapshot {
            scene: next.clone(),
            geo_m: gather(&old.geo_m, RAW),
            geo_v: gather(&old.geo_v, RAW),
            base_m: gather(&old.base_m, 3),
            base_v: gather(&old.base_v, 3),
            sh_m: gather(&old.sh_m, 3 * ksh),
            sh_v: gather(&old.sh_v, 3 * ksh),
            // a new sample has no views of its own yet: flat colour until
            // the next round measures them
            sh_limit: origin.iter().map(|o| o.map_or(0, |i| old.sh_limit.get(i).copied().unwrap_or(ksh as u32))).collect(),
        };
        Self::from_snapshot(gpu, &snap, sh_degree)
    }

    /// Per-gaussian ceiling on every axis, in world units.
    pub fn set_max_scale(&self, gpu: &Gpu, max: &[f32]) {
        assert_eq!(max.len(), self.n);
        let l: Vec<f32> = max.iter().map(|v| if *v > 0.0 { v.ln() } else { f32::MAX.ln() }).collect();
        gpu.write_f32(&self.smax, &l);
    }

    /// Cap each gaussian's SH coefficients per channel at `limit[i]` (0, 3, 8
    /// or 15; values past the scene's own degree are clamped to it).
    pub fn set_sh_limit(&mut self, gpu: &Gpu, limit: &[u32]) {
        assert_eq!(limit.len(), self.n);
        self.sh_limit_host = limit.iter().map(|v| (*v).min(self.ksh as u32)).collect();
        gpu.write(&self.sh_limit, &self.sh_limit_host);
    }

    /// Mip-Splatting's 3D filter, as a variance per gaussian.
    pub fn set_filter3d(&self, gpu: &Gpu, variance: &[f32]) {
        assert_eq!(variance.len(), self.n);
        gpu.write_f32(&self.filter3d, variance);
    }

    /// Write the renderer's inputs from the raw parameters.
    pub fn activate(&self, gpu: &Gpu, ks: &Kernels) {
        let s = gpu.step(ks.splat_activate, &[&self.geo, &self.means, &self.scales, &self.quats, &self.opac], &[self.n as u32], self.n as u32);
        gpu.submit(&[], &[s]);
    }

    /// Clear every gradient the backward accumulates into.
    pub fn zero_grads(&self, gpu: &Gpu) {
        let g = &self.grads;
        gpu.submit(&[&g.d_gauss, &g.d_opac, &g.d_colors, &g.d_absgrad, &g.d_sumgrad, &self.d_base, &self.d_sh], &[]);
    }

    /// Shade every gaussian's colour as seen from `eye` into `col_view`
    /// (a no-op for flat colour, which renders `base` directly), with the
    /// highest `skip` SH coefficients held out.
    pub fn shade(&self, gpu: &Gpu, ks: &Kernels, eye: [f32; 3], skip: u32) {
        if self.ksh == 0 {
            return;
        }
        let e = gpu.step(
            ks.splat_sh,
            &[&self.means, &self.base, &self.sh, &self.col_view, &self.d_base, &self.d_sh, &self.sh_limit],
            &[self.n as u32, self.ksh as u32, 0, skip, f(eye[0]), f(eye[1]), f(eye[2]), 0],
            self.n as u32,
        );
        gpu.submit(&[], &[e]);
    }

    /// Split the rendered colour's gradient between the base colour and the
    /// SH bands, for the direction the last [`Self::shade`] used, and clear
    /// it for the next view.
    pub fn shade_vjp(&self, gpu: &Gpu, ks: &Kernels, eye: [f32; 3], skip: u32) {
        if self.ksh == 0 {
            return;
        }
        let e = gpu.step(
            ks.splat_sh,
            &[&self.means, &self.base, &self.sh, &self.grads.d_colors, &self.d_base, &self.d_sh, &self.sh_limit],
            &[self.n as u32, self.ksh as u32, 1, skip, f(eye[0]), f(eye[1]), f(eye[2]), 0],
            self.n as u32,
        );
        gpu.submit(&[], &[e]);
        gpu.submit(&[&self.grads.d_colors], &[]);
    }

    /// The scene as the renderer takes it.
    pub fn splats(&self) -> GpuSplats {
        GpuSplats {
            n: self.n,
            means: self.means.clone(),
            quats: self.quats.clone(),
            scales: self.scales.clone(),
            opacities: self.opac.clone(),
            colors: if self.ksh > 0 { self.col_view.clone() } else { self.base.clone() },
            filter3d: Some(self.filter3d.clone()),
        }
    }

    /// One optimizer step at global timestep `t` (1-based).
    pub fn step(&self, gpu: &Gpu, ks: &Kernels, t: usize, r: &Rates, b: &Bounds) {
        let (b1, b2) = (0.9f32, 0.999f32);
        let (bc1, bc2) = (1.0 - b1.powi(t as i32), 1.0 - b2.powi(t as i32));
        let ln = |v: f32| if v > 1.0 { v.ln() } else { 0.0 };
        let params = [
            self.n as u32,
            t as u32,
            r.relative as u32,
            f(r.opacity_reg),
            f(r.position),
            f(r.scale),
            f(r.rotation),
            f(r.opacity),
            f(b1),
            f(b2),
            f(1e-15),
            f(bc1),
            f(bc2),
            0,
            f(r.noise),
            f(b.min_scale.max(1e-30).ln()),
            f(ln(b.max_needle)),
            f(ln(b.max_flat)),
            f(logit(0.0)),
            f(logit(1.0)),
            f(r.scale_reg),
        ];
        let geo = gpu.step(
            ks.splat_adam,
            &[&self.geo, &self.grads.d_gauss, &self.grads.d_opac, &self.m_geo, &self.v_geo, &self.smax],
            &params,
            self.n as u32,
        );
        gpu.submit(&[], &[geo]);
        let adam = |lr: f32, bufs: [&DeviceBuffer; 4], desc: &DeviceBuffer, numel: usize| {
            gpu.write(&self.hparams, &[f(lr), f(b1), f(b2), f(1e-15), f(0.0), f(bc1), f(bc2), f(1.0)]);
            let s = gpu.step_buf(ks.adamw, &self.hparams, &[bufs[0], bufs[1], bufs[2], bufs[3], desc, &self.unit_coef], numel as u32);
            gpu.submit(&[], &[s]);
        };
        // flat colour has no SH VJP to split it out, so its gradient is the
        // rendered colour's own
        let dcol = if self.ksh > 0 { &self.d_base } else { &self.grads.d_colors };
        adam(r.color, [&self.base, dcol, &self.m_base, &self.v_base], &self.desc_base, 3 * self.n);
        if self.ksh > 0 {
            adam(r.sh, [&self.sh, &self.d_sh, &self.m_sh, &self.v_sh], &self.desc_sh, 3 * self.n * self.ksh);
        }
    }
}
