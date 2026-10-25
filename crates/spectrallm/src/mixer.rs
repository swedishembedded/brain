// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The feed-forward sublayer: the SwiGLU control and the spectral mixer,
//! forward and backward, over any number of rows (a training batch or one
//! decode token).
//!
//! The spectral mixer of block `l` computes, for `h = RMSNorm(x)`:
//!
//! ```text
//! hg = G ⊙ h,  hu = U ⊙ h               chan_scale_expand      [rows, C, m]
//! ag = S hg,   au = S hu                synthesis              [rows, C, D]
//! s  = SiLU(ag) ⊙ au                    silu_mul               [rows, C, D]
//! t  = Sᵀ s                             analysis               [rows, C, m]
//! y  = Σ_c W_c ⊙ t_c                    chan_scale_contract    [rows, m]
//! ```
//!
//! Synthesis and analysis are adjoint (`S` has orthonormal columns), so the
//! backward pass reuses them in the opposite direction: `d s = S d t` and
//! `d hg = Sᵀ d ag`.

use gpu_core::{DeviceBuffer, Dispatch, Gpu, Step};

use crate::config::{fft_supports, Mixer, SpectralConfig, TransformPath};
use crate::kernels::{CHAN_CONTRACT, CHAN_EXPAND, CHAN_GRAD, RDFT_ANALYSIS, RDFT_SYNTH};
use crate::model::SpectralLm;

/// The resolved transform path of one model instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    Dense,
    Fft,
}

/// Resolve [`TransformPath`] against the device: the FFT kernels need
/// workgroup barriers, which the CPU JIT does not run past one.
pub fn resolve_transform(gpu: &Gpu, cfg: &SpectralConfig) -> Transform {
    let Mixer::Spectral { signal_len, .. } = cfg.mixer else { return Transform::Dense };
    let caps = gpu.caps();
    let capable = caps.workgroup_reductions && caps.workgroup_mem_bytes >= FFT_WORKGROUP_BYTES && fft_supports(signal_len);
    match cfg.transform {
        TransformPath::Dense => Transform::Dense,
        TransformPath::Fft => {
            assert!(capable, "spectrallm: transform fft requested but this device cannot run the FFT kernels (signal_len {signal_len})");
            Transform::Fft
        }
        TransformPath::Auto => {
            if capable {
                Transform::Fft
            } else {
                Transform::Dense
            }
        }
    }
}

/// Workgroup memory the FFT kernels declare: 2048 complex samples.
pub const FFT_WORKGROUP_BYTES: u32 = 16 * 1024;

/// Workgroups that cover `rows` transforms: one per row.
pub fn fft_workgroups(rows: u32, _d: u32) -> u32 {
    rows
}

/// The fixed basis on the device, in both layouts the dense path multiplies
/// by: `s` is `[D, m]` (synthesis as a `[out, in]` matmul weight), `st` is
/// `[m, D]` (analysis).
pub(crate) struct Basis {
    pub s: DeviceBuffer,
    pub st: DeviceBuffer,
    pub d: u32,
    pub path: Transform,
}

impl Basis {
    pub fn new(gpu: &Gpu, cfg: &SpectralConfig) -> Option<Basis> {
        let Mixer::Spectral { signal_len, .. } = cfg.mixer else { return None };
        let (d, m) = (signal_len as usize, cfg.d_model as usize);
        let s = model::hostmath::real_fourier_basis(d, m);
        let mut st = vec![0.0f32; m * d];
        for j in 0..d {
            for i in 0..m {
                st[i * d + j] = s[j * m + i];
            }
        }
        Some(Basis { s: gpu.storage_init("spectral_basis", &s), st: gpu.storage_init("spectral_basis_t", &st), d: signal_len, path: resolve_transform(gpu, cfg) })
    }
}

/// One mixer's activation (and, when training, gradient) buffers for `rows`
/// rows.
pub(crate) enum MixerBufs {
    SwiGlu {
        gate: DeviceBuffer,
        up: DeviceBuffer,
        h: DeviceBuffer,
        grad: Option<[DeviceBuffer; 3]>,
    },
    Spectral {
        hg: DeviceBuffer,
        hu: DeviceBuffer,
        ag: DeviceBuffer,
        au: DeviceBuffer,
        s: DeviceBuffer,
        t: DeviceBuffer,
        /// `d t`/`d hg`, `d hu` (`[rows, C, m]`), `d s`, `d ag`, `d au`
        /// (`[rows, C, D]`).
        grad: Option<[DeviceBuffer; 6]>,
    },
}

impl MixerBufs {
    pub fn new(gpu: &Gpu, cfg: &SpectralConfig, rows: u32, train: bool) -> MixerBufs {
        let st = |x: u64| gpu.storage(x);
        let (n, d) = (rows as u64, cfg.d_model as u64);
        match cfg.mixer {
            Mixer::SwiGlu { d_ff } => {
                let f = d_ff as u64;
                MixerBufs::SwiGlu { gate: st(n * f), up: st(n * f), h: st(n * f), grad: train.then(|| [st(n * f), st(n * f), st(n * f)]) }
            }
            Mixer::Spectral { signal_len, channels } => {
                let (c, sig) = (channels as u64, signal_len as u64);
                let (cm, cd) = (n * c * d, n * c * sig);
                MixerBufs::Spectral {
                    hg: st(cm),
                    hu: st(cm),
                    ag: st(cd),
                    au: st(cd),
                    s: st(cd),
                    t: st(cm),
                    grad: train.then(|| [st(cm), st(cm), st(cm), st(cd), st(cd), st(cd)]),
                }
            }
        }
    }
}

impl SpectralLm {
    /// `dst[rows, D] = S src[rows, m]`.
    fn synth(&self, s: &mut Vec<Step>, src: &DeviceBuffer, dst: &DeviceBuffer, rows: u32) {
        let basis = self.basis.as_ref().expect("synthesis on a SwiGLU model");
        let m = self.cfg.d_model;
        match basis.path {
            Transform::Fft => s.push(self.gpu.dispatch(RDFT_SYNTH, &[src, dst], &[rows, basis.d, m, basis.d.ilog2()], Dispatch::Workgroups(fft_workgroups(rows, basis.d)))),
            Transform::Dense => self.linear(s, src, &basis.s, dst, rows, m, basis.d),
        }
    }

    /// `dst[rows, m] = Sᵀ src[rows, D]`.
    fn analysis(&self, s: &mut Vec<Step>, src: &DeviceBuffer, dst: &DeviceBuffer, rows: u32) {
        let basis = self.basis.as_ref().expect("analysis on a SwiGLU model");
        let m = self.cfg.d_model;
        match basis.path {
            Transform::Fft => s.push(self.gpu.dispatch(RDFT_ANALYSIS, &[src, dst], &[rows, basis.d, m, basis.d.ilog2()], Dispatch::Workgroups(fft_workgroups(rows, basis.d)))),
            Transform::Dense => self.linear(s, src, &basis.st, dst, rows, basis.d, m),
        }
    }

    /// `y = Mixer_l(x)` over `rows` rows of width `d_model`.
    pub(crate) fn mixer_fwd(&self, s: &mut Vec<Step>, l: usize, x: &DeviceBuffer, y: &DeviceBuffer, b: &MixerBufs, rows: u32) {
        let m = self.cfg.d_model;
        let p = |name: &str| format!("blocks.{l}.{name}");
        match (b, &self.cfg.mixer) {
            (MixerBufs::SwiGlu { gate, up, h, .. }, Mixer::SwiGlu { d_ff }) => {
                let f = *d_ff;
                self.linear(s, x, self.ps.w(&p("mlp.gate.weight")), gate, rows, m, f);
                self.linear(s, x, self.ps.w(&p("mlp.up.weight")), up, rows, m, f);
                s.push(model::block::swiglu_fwd(&self.gpu, &self.ids, gate, up, h, rows * f));
                self.linear(s, h, self.ps.w(&p("mlp.down.weight")), y, rows, f, m);
            }
            (MixerBufs::Spectral { hg, hu, ag, au, s: sig, t, .. }, Mixer::Spectral { signal_len, channels }) => {
                let (c, d) = (*channels, *signal_len);
                let cr = rows * c;
                s.push(self.gpu.step(CHAN_EXPAND, &[x, self.ps.w(&p("mlp.gate_filter")), hg], &[rows, c, m], cr * m));
                s.push(self.gpu.step(CHAN_EXPAND, &[x, self.ps.w(&p("mlp.up_filter")), hu], &[rows, c, m], cr * m));
                self.synth(s, hg, ag, cr);
                self.synth(s, hu, au, cr);
                s.push(model::block::swiglu_fwd(&self.gpu, &self.ids, ag, au, sig, cr * d));
                self.analysis(s, sig, t, cr);
                s.push(self.gpu.step(CHAN_CONTRACT, &[t, self.ps.w(&p("mlp.out_filter")), y], &[rows, c, m, 0], rows * m));
            }
            _ => unreachable!("mixer buffers built for a different mixer"),
        }
    }

    /// Backward of [`Self::mixer_fwd`]: assigns `dx`, accumulates the
    /// mixer's parameter gradients. Needs the forward's activations in `b`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mixer_bwd(&self, s: &mut Vec<Step>, l: usize, x: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, b: &MixerBufs, rows: u32) {
        let m = self.cfg.d_model;
        let p = |name: &str| format!("blocks.{l}.{name}");
        match (b, &self.cfg.mixer) {
            (MixerBufs::SwiGlu { gate, up, h, grad }, Mixer::SwiGlu { d_ff }) => {
                let f = *d_ff;
                let [d_h, d_gate, d_up] = grad.as_ref().expect("mixer backward on an inference build");
                self.linear_dw(s, dy, h, self.ps.g(&p("mlp.down.weight")), rows, f, m);
                self.linear_dx(s, dy, self.ps.w(&p("mlp.down.weight")), d_h, rows, f, m, false);
                s.extend(model::block::swiglu_bwd(&self.gpu, &self.ids, gate, up, d_h, d_gate, d_up, rows * f));
                self.linear_dw(s, d_gate, x, self.ps.g(&p("mlp.gate.weight")), rows, m, f);
                self.linear_dx(s, d_gate, self.ps.w(&p("mlp.gate.weight")), dx, rows, m, f, false);
                self.linear_dw(s, d_up, x, self.ps.g(&p("mlp.up.weight")), rows, m, f);
                self.linear_dx(s, d_up, self.ps.w(&p("mlp.up.weight")), dx, rows, m, f, true);
            }
            (MixerBufs::Spectral { ag, au, t, .. }, Mixer::Spectral { signal_len, channels }) => {
                let (c, d) = (*channels, *signal_len);
                let cr = rows * c;
                let [d_t, d_hg, d_hu, d_s, d_ag, d_au] = match b {
                    MixerBufs::Spectral { grad: Some(g), .. } => g,
                    _ => panic!("mixer backward on an inference build"),
                };
                // y = Σ_c W_c ⊙ t_c
                s.push(self.gpu.step(CHAN_GRAD, &[dy, t, self.ps.g(&p("mlp.out_filter"))], &[rows, c, m], c * m));
                s.push(self.gpu.step(CHAN_EXPAND, &[dy, self.ps.w(&p("mlp.out_filter")), d_t], &[rows, c, m], cr * m));
                // t = Sᵀ s  ->  d s = S d t
                self.synth(s, d_t, d_s, cr);
                s.extend(model::block::swiglu_bwd(&self.gpu, &self.ids, ag, au, d_s, d_ag, d_au, cr * d));
                // ag = S hg  ->  d hg = Sᵀ d ag
                self.analysis(s, d_ag, d_hg, cr);
                self.analysis(s, d_au, d_hu, cr);
                // hg = G ⊙ x, hu = U ⊙ x
                s.push(self.gpu.step(CHAN_GRAD, &[x, d_hg, self.ps.g(&p("mlp.gate_filter"))], &[rows, c, m], c * m));
                s.push(self.gpu.step(CHAN_GRAD, &[x, d_hu, self.ps.g(&p("mlp.up_filter"))], &[rows, c, m], c * m));
                s.push(self.gpu.step(CHAN_CONTRACT, &[d_hg, self.ps.w(&p("mlp.gate_filter")), dx], &[rows, c, m, 0], rows * m));
                s.push(self.gpu.step(CHAN_CONTRACT, &[d_hu, self.ps.w(&p("mlp.up_filter")), dx], &[rows, c, m, 1], rows * m));
            }
            _ => unreachable!("mixer buffers built for a different mixer"),
        }
    }
}
