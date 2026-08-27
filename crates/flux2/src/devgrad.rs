// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) forward + backward for the FLUX.2 double- and single-stream
//! blocks, as a **persistent** engine ([`BlockDev`]) - the compute core of the
//! device LoRA trainer. One GPU device + one set of reusable activation buffers
//! (sized to the max joint token count) serve every block; the frozen base
//! weights and the adapter factors live in per-block [`DoubleDev`] /
//! [`SingleDev`] holders, so a 25-block training step is a sequence of cheap
//! submits rather than 25 device creations or 25 weight uploads.
//!
//! Every op the finite-difference-gradchecked host reference ([`crate::grad`])
//! does analytically, this does on-device with brain's pre-gradchecked kernels:
//! `layernorm` (affine-free, eps 1e-6) + `film_row` for the modulated LN and
//! `layernorm_dx`/`film_row_dx`/`film_row_dsb` for its backward, `gate_row`
//! (+`_dh`/`_dg`) for the per-channel gated residuals, `matmul_reg3` /
//! `matmul_dx_reg` / `matmul_dw_reg` for the linears, those same register-tiled
//! GEMMs over `head_pack`ed head-major operands for the joint attention in BOTH
//! directions (see [`BlockDev::attn_fwd`]), `attn_softmax_bidir` /
//! `softmax_k_dx` for the softmax and its Jacobian, `silu_mul` / `silu_bwd_da`
//! / `silu_bwd_db` for the SwiGLU, interleaved-RoPE forward fed a negated sin
//! table for its backward, and `rms_inv_eps` / `rmsnorm_dw` / `rmsnorm_dx_eps`
//! for QK-RMSNorm. It adds **no WGSL**.
//!
//! ## Where this departs from `s3dit::devgrad`, and why
//!
//! `s3dit::devgrad` computes a dense `dW` for every linear, because it trains
//! the full model. This engine trains an **adapter**, so it never materialises
//! one. For a targeted linear `y = x·Wᵀ + x·Aᵀ·B̃ᵀ` (with `B̃ = (α/r)·B` folded
//! at upload, stored transposed as `[r,out]`), the adapter gradients come out
//! of the low-rank intermediates directly:
//!
//! ```text
//!   xa  = x  · Aᵀ          [m,r]     (forward, cached)
//!   dxa = dy · B̃           [m,r]
//!   dx  = dy · W + dxa · A            (two accumulating dx GEMMs)
//!   dA  = dxaᵀ · x         [r,in]
//!   dB̃  = xaᵀ · dy         [r,out]
//! ```
//!
//! which is algebraically identical to the host path's `Pair::project` of a
//! dense `dW` (`dA = (α/r)·Bᵀ·dW`, `dB = (α/r)·dW·Aᵀ`) - `tests/dev_grad.rs`
//! asserts exactly that equality. The consequences are the point: the `dW`
//! GEMM (one third of every backward's arithmetic) is replaced by two GEMMs of
//! rank width, no `[out,in]` gradient buffer is ever allocated, and the frozen
//! base is only ever *read*, in two directions.
//!
//! The other departure is that FLUX.2's modulation is **global**: all double
//! blocks read the same four `(shift, scale, gate)` sites and all single blocks
//! the same one, so the sites are uploaded once per step and their gradients
//! accumulate on-device across the whole stack ([`BlockDev::upload_mods`] /
//! [`BlockDev::mod_grads`]) instead of being folded per block on the host.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

use crate::grad::{Dims, Mod, ModGrad};

// Kernel indices into [`KERNELS`].
const K_LN: usize = 0;
const K_FILM: usize = 1;
const K_MM: usize = 2;
const K_RMS: usize = 3;
const K_RMS_ROWS: usize = 4;
const K_ROPE: usize = 5;
const K_HPACK: usize = 6;
const K_HPACKT: usize = 7;
const K_HUNPACK: usize = 8;
const K_SOFTMAX: usize = 9;
const K_SMDX: usize = 10;
const K_SILU: usize = 11;
const K_GATE: usize = 12;
const K_ADD2: usize = 13;
const K_DX: usize = 14;
const K_DW: usize = 15;
const K_LN_DX: usize = 16;
const K_FILM_DX: usize = 17;
const K_FILM_DSB: usize = 18;
const K_GATE_DH: usize = 19;
const K_GATE_DG: usize = 20;
const K_SDA: usize = 21;
const K_SDB: usize = 22;
const K_RINV: usize = 23;
const K_RDW: usize = 24;
const K_RDX: usize = 25;
const K_ADD_INPLACE: usize = 26;

/// The kernel set the device trainer registers. Every entry is an existing,
/// gradient-checked kernel - the trainer adds no WGSL of its own.
pub const KERNELS: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("film_row", kernels::FILM_ROW),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("head_pack", kernels::HEAD_PACK),
    ("head_pack_t", kernels::HEAD_PACK_T),
    ("head_unpack", kernels::HEAD_UNPACK),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("softmax_k_dx", kernels::SOFTMAX_K_DX),
    ("silu_mul", kernels::SILU_MUL),
    ("gate_row", kernels::GATE_ROW),
    ("add2", kernels::ADD2),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("film_row_dx", kernels::FILM_ROW_DX),
    ("film_row_dsb", kernels::FILM_ROW_DSB),
    ("gate_row_dh", kernels::GATE_ROW_DH),
    ("gate_row_dg", kernels::GATE_ROW_DG),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("rms_inv_eps", kernels::RMS_INV_EPS),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rmsnorm_dx_eps", kernels::RMSNORM_DX_EPS),
    ("add_inplace", kernels::ADD_INPLACE),
];

/// LayerNorm / QK-RMSNorm epsilon - 1e-6 in every FLUX.2 variant, matching
/// `crate::grad`'s `EPS` and `model::EPS`. A backward that recomputes the
/// normaliser with a different epsilon than the forward drifts silently, so
/// this constant is shared by both directions here.
const EPS: f32 = 1e-6;

/// Storage bindings must start on a 256-byte boundary = 64 f32.
const ALIGN: usize = 64;

/// The four double-stream modulation sites, the single-stream one, and the
/// final layer's, in the order [`BlockDev`] indexes them. The final site has no
/// gate (BFL's final adaLN emits shift and scale only); its gate slot is
/// uploaded as zeros and never read.
pub const N_SITES: usize = 6;
/// Modulation site index: double-block image stream, attention half.
pub const SITE_IMG1: usize = 0;
/// Modulation site index: double-block image stream, MLP half.
pub const SITE_IMG2: usize = 1;
/// Modulation site index: double-block text stream, attention half.
pub const SITE_TXT1: usize = 2;
/// Modulation site index: double-block text stream, MLP half.
pub const SITE_TXT2: usize = 3;
/// Modulation site index: the single-stream blocks' one shared site.
pub const SITE_SGL: usize = 4;
/// Modulation site index: the final layer's adaLN (shift + scale, no gate).
pub const SITE_FINAL: usize = 5;

fn d128(x: usize) -> u32 {
    x.div_ceil(128) as u32
}

/// One targeted linear on the device: the **frozen** base `W [out,in]` plus the
/// adapter factors `A [r,in]` and `B̃ᵀ [r,out]` (`B̃ = (α/r)·B`, transposed so
/// both the up-projection and its backward are plain GEMMs), and the two
/// gradient buffers the backward accumulates into.
pub struct LinDev {
    pub out: usize,
    pub inn: usize,
    pub r: usize,
    w: DeviceBuffer,
    a: DeviceBuffer,
    bt: DeviceBuffer,
    ga: DeviceBuffer,
    gbt: DeviceBuffer,
}

impl LinDev {
    /// Device bytes this linear holds (base + adapter + adapter grads).
    pub fn bytes(&self) -> u64 {
        4 * (self.out * self.inn + 2 * self.r * self.inn + 2 * self.r * self.out) as u64
    }
}

/// The seven targeted linears of one double-block stream, plus its two frozen
/// QK-RMSNorm scales and their gradient buffers.
pub struct StreamDev {
    pub wq: LinDev,
    pub wk: LinDev,
    pub wv: LinDev,
    pub wo: LinDev,
    pub w1: LinDev,
    pub w3: LinDev,
    pub w2: LinDev,
    nq: DeviceBuffer,
    nk: DeviceBuffer,
    gnq: DeviceBuffer,
    gnk: DeviceBuffer,
}

impl StreamDev {
    fn pairs(&self) -> [&LinDev; 7] {
        [&self.wq, &self.wk, &self.wv, &self.wo, &self.w1, &self.w3, &self.w2]
    }
    pub fn bytes(&self) -> u64 {
        self.pairs().iter().map(|l| l.bytes()).sum()
    }
}

/// One double block: two independent streams over a joint attention.
pub struct DoubleDev {
    pub img: StreamDev,
    pub txt: StreamDev,
}

/// One single block: five linears off a shared modulated LN, and the
/// column-split `linear2` as two independent linears.
pub struct SingleDev {
    pub wq: LinDev,
    pub wk: LinDev,
    pub wv: LinDev,
    pub w1: LinDev,
    pub w3: LinDev,
    pub wo_a: LinDev,
    pub wo_b: LinDev,
    nq: DeviceBuffer,
    nk: DeviceBuffer,
    gnq: DeviceBuffer,
    gnk: DeviceBuffer,
}

impl SingleDev {
    fn pairs(&self) -> [&LinDev; 7] {
        [&self.wq, &self.wk, &self.wv, &self.w1, &self.w3, &self.wo_a, &self.wo_b]
    }
    pub fn bytes(&self) -> u64 {
        self.pairs().iter().map(|l| l.bytes()).sum()
    }
}

/// A persistent GPU engine: one device, one set of activation buffers sized to
/// `n_max` joint tokens, driving any block's forward or backward.
pub struct BlockDev {
    gpu: Gpu,
    d: usize,
    nh: usize,
    hd: usize,
    mlp: usize,
    n_max: usize,
    rank: usize,
    b: HashMap<String, DeviceBuffer>,
}

impl BlockDev {
    /// Build an engine on a fresh wgpu device.
    pub fn new(n_max: usize, d: usize, nh: usize, mlp: usize, rank: usize) -> BlockDev {
        BlockDev::from_gpu(Gpu::new_wgpu(KERNELS), n_max, d, nh, mlp, rank)
    }

    /// Build over an existing device (so a caller can place the trainer on a
    /// chosen card, or share one device with the rest of a pipeline).
    pub fn from_gpu(gpu: Gpu, n_max: usize, d: usize, nh: usize, mlp: usize, rank: usize) -> BlockDev {
        let n = n_max.div_ceil(ALIGN) * ALIGN;
        let hd = d / nh;
        assert!(d.is_multiple_of(nh), "hidden {d} must divide by n_heads {nh}");
        assert!(hd.is_multiple_of(2), "head_dim {hd} must be even for interleaved RoPE");
        let mut b: HashMap<String, DeviceBuffer> = HashMap::new();
        {
            let mut mk = |name: &str, len: usize| {
                b.insert(name.to_string(), gpu.storage(len as u64));
            };
            let nd = n * d;
            let nm = n * mlp;
            for name in [
                "xh1", "n1", "q", "k", "v", "qn", "kn", "qr", "kr", "ctx", "proj", "x1", "xh2", "n2", "mlpo", "pm", "dx1", "dtmp", "dpm", "dproj", "dctx",
                "dmlpo", "dn2", "dn1", "dxh", "dqr", "dkr", "dv", "dqn", "dkn", "dq", "dk",
            ] {
                mk(name, nd);
            }
            for name in ["h1", "h2", "hs", "dhs", "dh1", "dh2"] {
                mk(name, nm);
            }
            // Head-major GEMM-attention operands: `hstride` per head, padded so
            // every `h·hstride` binding offset stays 256-byte aligned.
            let hstride = model::block::pad64((n * (d / nh)) as u64) as usize;
            let sstride = model::block::pad64((n * n) as u64) as usize;
            for name in ["qpk", "kpk", "vpk", "vpkt", "ctxpk", "dcpk", "dqpk", "dkpk", "dvpk"] {
                mk(name, nh * hstride);
            }
            mk("scores", nh * sstride);
            mk("probs", nh * sstride);
            mk("dscores", nh * sstride);
            // Low-rank intermediates. TWO slots of `n` rows each (slot 0 = the
            // text stream / the whole slab, slot 1 = the image stream), so a
            // stream's binding offset is `slot·n·r` - a multiple of 64 floats
            // for any rank, unlike an `nt·r` offset which is not.
            for i in 0..7 {
                mk(&format!("xa{i}"), 2 * n * rank);
            }
            mk("dxa", 2 * n * rank);
            // Same two-slot trick for the per-head RMSNorm row inverses.
            mk("inv_q", 2 * n * nh);
            mk("inv_k", 2 * n * nh);
            for name in ["cos", "sin", "nsin"] {
                mk(name, n * (hd / 2));
            }
            mk("ones", d);
            mk("zeros", d);
            for s in 0..N_SITES {
                mk(&format!("sb{s}"), 2 * d);
                mk(&format!("gt{s}"), d);
                mk(&format!("gsb{s}"), 2 * d);
                mk(&format!("ggt{s}"), d);
            }
            mk("dsb", 2 * d);
            mk("dgt", d);
        }
        let eng = BlockDev { gpu, d, nh, hd, mlp, n_max: n, rank, b };
        eng.gpu.write_f32(&eng.b["ones"], &vec![1.0f32; d]);
        eng.gpu.write_f32(&eng.b["zeros"], &vec![0.0f32; d]);
        eng
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
    pub fn n_max(&self) -> usize {
        self.n_max
    }
    pub fn rank(&self) -> usize {
        self.rank
    }

    fn g(&self, name: &str) -> &DeviceBuffer {
        self.b.get(name).unwrap_or_else(|| panic!("devgrad: no buffer {name}"))
    }

    /// A `[len]`-element view of a buffer starting at element `off`, checked
    /// against the 256-byte storage-binding alignment the backend enforces.
    fn sl(&self, off: usize, len: usize) -> (u64, u64) {
        assert!(off.is_multiple_of(ALIGN), "devgrad: binding offset {off} is not a multiple of {ALIGN} floats (256-byte alignment)");
        (off as u64, len as u64)
    }

    /// Allocate a `[rows, d]` slab on this engine's device - what the model
    /// level uses for the per-block saved inputs.
    pub fn slab(&self, rows: usize) -> DeviceBuffer {
        self.gpu.storage((rows * self.d) as u64)
    }

    /// Upload the joint RoPE tables `[n, hd/2]` for this step (the negated-sin
    /// table the backward rotates by is derived here, once).
    pub fn upload_rope(&self, cos: &[f32], sin: &[f32]) {
        self.gpu.write_f32(self.g("cos"), cos);
        self.gpu.write_f32(self.g("sin"), sin);
        let nsin: Vec<f32> = sin.iter().map(|&s| -s).collect();
        self.gpu.write_f32(self.g("nsin"), &nsin);
    }

    /// Upload this step's five modulation sites and zero their gradient
    /// accumulators. `sites` is indexed by `SITE_*`.
    pub fn upload_mods(&self, sites: &[Mod<f32>; N_SITES]) {
        let d = self.d;
        for (s, m) in sites.iter().enumerate() {
            assert_eq!(m.scale.len(), d, "site {s}: scale width");
            assert_eq!(m.shift.len(), d, "site {s}: shift width");
            assert_eq!(m.gate.len(), d, "site {s}: gate width");
            // film_row packs the pair scale-first in ONE [2D] buffer.
            let mut sb = Vec::with_capacity(2 * d);
            sb.extend_from_slice(&m.scale);
            sb.extend_from_slice(&m.shift);
            self.gpu.write_f32(self.g(&format!("sb{s}")), &sb);
            self.gpu.write_f32(self.g(&format!("gt{s}")), &m.gate);
            self.gpu.write_f32(self.g(&format!("gsb{s}")), &vec![0.0f32; 2 * d]);
            self.gpu.write_f32(self.g(&format!("ggt{s}")), &vec![0.0f32; d]);
        }
    }

    /// Read back the modulation-site gradients accumulated over the whole block
    /// stack (the sites are global, so this is the sum every block contributed).
    pub fn mod_grads(&self) -> Vec<ModGrad<f32>> {
        let d = self.d;
        (0..N_SITES)
            .map(|s| {
                let sb = self.gpu.read(self.g(&format!("gsb{s}")), 2 * d);
                let gate = self.gpu.read(self.g(&format!("ggt{s}")), d);
                ModGrad { scale: sb[..d].to_vec(), shift: sb[d..].to_vec(), gate }
            })
            .collect()
    }

    // ---- weight holders ----

    /// Upload one frozen base linear and allocate its adapter buffers.
    pub fn lin(&self, w: &[f32], out: usize, inn: usize) -> LinDev {
        assert_eq!(w.len(), out * inn, "base linear [{out},{inn}] size");
        let r = self.rank;
        let l = LinDev {
            out,
            inn,
            r,
            w: self.gpu.storage_init("flux2 base", w),
            a: self.gpu.storage((r * inn) as u64),
            bt: self.gpu.storage((r * out) as u64),
            ga: self.gpu.storage((r * inn) as u64),
            gbt: self.gpu.storage((r * out) as u64),
        };
        self.gpu.write_f32(&l.a, &vec![0.0f32; r * inn]);
        self.gpu.write_f32(&l.bt, &vec![0.0f32; r * out]);
        l
    }

    /// The attention scale `1/√head_dim`, folded into the **query** norm
    /// weight so the GEMM attention path carries no scale of its own.
    fn attn_scale(&self) -> f32 {
        1.0 / (self.hd as f32).sqrt()
    }

    fn norm_pair(&self, nq: &[f32], nk: &[f32]) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
        let sc = self.attn_scale();
        let nqs: Vec<f32> = nq.iter().map(|&v| v * sc).collect();
        (
            self.gpu.storage_init("flux2 qnorm", &nqs),
            self.gpu.storage_init("flux2 knorm", nk),
            self.gpu.storage(self.hd as u64),
            self.gpu.storage(self.hd as u64),
        )
    }

    /// `(dnq, dnk)` for one QK-RMSNorm site. The query half is scaled back out
    /// of the folded `1/√head_dim` (`nq' = s·nq` ⇒ `dL/dnq = s·dL/dnq'`), so a
    /// caller sees the gradient of the checkpoint's own tensor.
    fn norm_grads(&self, gnq: &DeviceBuffer, gnk: &DeviceBuffer) -> (Vec<f32>, Vec<f32>) {
        let sc = self.attn_scale();
        let dq: Vec<f32> = self.gpu.read(gnq, self.hd).iter().map(|&v| v * sc).collect();
        (dq, self.gpu.read(gnk, self.hd))
    }

    /// Build one double-block stream's device weights from host slices.
    #[allow(clippy::too_many_arguments)]
    pub fn stream(&self, wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32], w1: &[f32], w3: &[f32], w2: &[f32], nq: &[f32], nk: &[f32]) -> StreamDev {
        let (d, mlp) = (self.d, self.mlp);
        let (nqb, nkb, gnq, gnk) = self.norm_pair(nq, nk);
        StreamDev {
            wq: self.lin(wq, d, d),
            wk: self.lin(wk, d, d),
            wv: self.lin(wv, d, d),
            wo: self.lin(wo, d, d),
            w1: self.lin(w1, mlp, d),
            w3: self.lin(w3, mlp, d),
            w2: self.lin(w2, d, mlp),
            nq: nqb,
            nk: nkb,
            gnq,
            gnk,
        }
    }

    /// Build one single block's device weights from host slices.
    #[allow(clippy::too_many_arguments)]
    pub fn single(&self, wq: &[f32], wk: &[f32], wv: &[f32], w1: &[f32], w3: &[f32], wo_a: &[f32], wo_b: &[f32], nq: &[f32], nk: &[f32]) -> SingleDev {
        let (d, mlp) = (self.d, self.mlp);
        let (nqb, nkb, gnq, gnk) = self.norm_pair(nq, nk);
        SingleDev {
            wq: self.lin(wq, d, d),
            wk: self.lin(wk, d, d),
            wv: self.lin(wv, d, d),
            w1: self.lin(w1, mlp, d),
            w3: self.lin(w3, mlp, d),
            wo_a: self.lin(wo_a, d, d),
            wo_b: self.lin(wo_b, d, mlp),
            nq: nqb,
            nk: nkb,
            gnq,
            gnk,
        }
    }

    /// Upload one adapter pair, folding the `α/r` scale into `B` and
    /// transposing it to `[r,out]`. `a` is `[r,in]` and `b` is `[out,r]`, the
    /// host [`model::lora::Pair`] layout. The gradient buffers are NOT touched
    /// here: `matmul_dw_reg` accumulates, so they are zeroed as `submit`
    /// clears at the head of the backward that fills them (a device-side
    /// clear, not a host upload).
    pub fn upload_lora(&self, l: &LinDev, a: &[f32], b: &[f32], scale: f32) {
        assert_eq!(a.len(), l.r * l.inn, "adapter A [{},{}]", l.r, l.inn);
        assert_eq!(b.len(), l.out * l.r, "adapter B [{},{}]", l.out, l.r);
        let mut bt = vec![0.0f32; l.r * l.out];
        for o in 0..l.out {
            for k in 0..l.r {
                bt[k * l.out + o] = b[o * l.r + k] * scale;
            }
        }
        self.gpu.write_f32(&l.a, a);
        self.gpu.write_f32(&l.bt, &bt);
    }

    /// Read one adapter pair's gradients back as `(dA [r,in], dB [out,r])` -
    /// the same layout and the same value `Pair::project` would produce from a
    /// dense `dW`, ready for [`model::lora::Pair::adam_step`].
    pub fn lin_grads(&self, l: &LinDev, scale: f32) -> (Vec<f32>, Vec<f32>) {
        let da = self.gpu.read(&l.ga, l.r * l.inn);
        let gbt = self.gpu.read(&l.gbt, l.r * l.out);
        // dL/dB = scale · dL/dB̃, transposed back to [out,r].
        let mut db = vec![0.0f32; l.out * l.r];
        for k in 0..l.r {
            for o in 0..l.out {
                db[o * l.r + k] = gbt[k * l.out + o] * scale;
            }
        }
        (da, db)
    }

    /// Read one stream's QK-RMSNorm scale gradients `(dnq, dnk)`.
    pub fn stream_norm_grads(&self, s: &StreamDev) -> (Vec<f32>, Vec<f32>) {
        self.norm_grads(&s.gnq, &s.gnk)
    }
    /// Read one single block's QK-RMSNorm scale gradients `(dnq, dnk)`.
    pub fn single_norm_grads(&self, s: &SingleDev) -> (Vec<f32>, Vec<f32>) {
        self.norm_grads(&s.gnq, &s.gnk)
    }

    /// Every gradient buffer one double block accumulates into - passed as
    /// `submit` clears so `matmul_dw_reg`/`rmsnorm_dw`'s `+=` starts from zero
    /// without a host round trip.
    fn double_clears<'a>(&'a self, w: &'a DoubleDev) -> Vec<&'a DeviceBuffer> {
        let mut v = vec![self.g("dkpk"), self.g("dvpk")];
        for st in [&w.txt, &w.img] {
            for l in st.pairs() {
                v.push(&l.ga);
                v.push(&l.gbt);
            }
            v.push(&st.gnq);
            v.push(&st.gnk);
        }
        v
    }

    /// [`Self::double_clears`] for a single block.
    fn single_clears<'a>(&'a self, w: &'a SingleDev) -> Vec<&'a DeviceBuffer> {
        let mut v = vec![self.g("dkpk"), self.g("dvpk")];
        for l in w.pairs() {
            v.push(&l.ga);
            v.push(&l.gbt);
        }
        v.push(&w.gnq);
        v.push(&w.gnk);
        v
    }

    // ---- dispatch helpers ----

    fn tier(&self) -> model::block::GemmVariants {
        model::block::GemmVariants::Fast { gemv: None, tiled: K_MM }
    }

    /// `y[yr0.., :out] = x[xr0.., :inn] · Wᵀ + xa · B̃`, with `xa = x · Aᵀ`
    /// cached in slot `slot` of `xa{i}` for the backward.
    #[allow(clippy::too_many_arguments)]
    fn lin_fwd(&self, s: &mut Vec<Step>, l: &LinDev, i: usize, slot: usize, x: &DeviceBuffer, xr0: usize, y: &DeviceBuffer, yr0: usize, m: usize) {
        let (inn, out, r) = (l.inn, l.out, l.r);
        s.push(model::dispatch::mm_rows_off(&self.gpu, self.tier(), x, &l.w, y, xr0 as u32, (yr0 * out) as u64, m as u32, inn as u32, out as u32));
        let xa = self.g(&format!("xa{i}"));
        let ab = slot * self.n_max * r;
        s.push(model::dispatch::mm_rows_off(&self.gpu, self.tier(), x, &l.a, xa, xr0 as u32, ab as u64, m as u32, inn as u32, r as u32));
        // y += xa · B̃  (B̃ stored [r,out], so this is the dx-shaped GEMM with
        // its accumulate flag set - no temporary and no second add pass).
        s.push(self.gpu.step_sliced(
            K_DX,
            &[xa, &l.bt, y],
            &[self.sl(ab, m * r), (0, 0), self.sl(yr0 * out, m * out)],
            &[m as u32, out as u32, r as u32, 1],
            d128(m) * d128(out) * 256,
        ));
    }

    /// Backward of [`Self::lin_fwd`]. Accumulates `dA`/`dB̃` and adds this
    /// linear's contribution to `dx` (`acc = false` on the first contributor to
    /// a given `dx` row range, `true` afterwards).
    #[allow(clippy::too_many_arguments)]
    fn lin_bwd(&self, s: &mut Vec<Step>, l: &LinDev, i: usize, slot: usize, x: &DeviceBuffer, xr0: usize, dy: &DeviceBuffer, dyr0: usize, dx: &DeviceBuffer, dxr0: usize, m: usize, acc: bool) {
        let (inn, out, r) = (l.inn, l.out, l.r);
        let xa = self.g(&format!("xa{i}"));
        let dxa = self.g("dxa");
        let ab = slot * self.n_max * r;
        let dyo = self.sl(dyr0 * out, m * out);
        // dxa = dy · B̃ᵀ  (B̃ᵀ is [r,out]; a plain forward GEMM over k = out)
        s.push(model::dispatch::mm_rows_off(&self.gpu, self.tier(), dy, &l.bt, dxa, dyr0 as u32, ab as u64, m as u32, out as u32, r as u32));
        // dx = dy · W (+= when this is not the first contributor)
        s.push(self.gpu.step_sliced(
            K_DX,
            &[dy, &l.w, dx],
            &[dyo, (0, 0), self.sl(dxr0 * inn, m * inn)],
            &[m as u32, inn as u32, out as u32, u32::from(acc)],
            d128(m) * d128(inn) * 256,
        ));
        // dx += dxa · A
        s.push(self.gpu.step_sliced(
            K_DX,
            &[dxa, &l.a, dx],
            &[self.sl(ab, m * r), (0, 0), self.sl(dxr0 * inn, m * inn)],
            &[m as u32, inn as u32, r as u32, 1],
            d128(m) * d128(inn) * 256,
        ));
        // dA += dxaᵀ · x   →  [r, inn]
        s.push(self.gpu.step_sliced(
            K_DW,
            &[dxa, x, &l.ga],
            &[self.sl(ab, m * r), self.sl(xr0 * inn, m * inn), (0, 0)],
            &[m as u32, inn as u32, r as u32],
            d128(r) * d128(inn) * 256,
        ));
        // dB̃ += xaᵀ · dy   →  [r, out]
        s.push(self.gpu.step_sliced(
            K_DW,
            &[xa, dy, &l.gbt],
            &[self.sl(ab, m * r), dyo, (0, 0)],
            &[m as u32, out as u32, r as u32],
            d128(r) * d128(out) * 256,
        ));
    }

    /// Affine-free LayerNorm over rows `r0..r0+m` → `xhat`.
    fn ln(&self, x: &DeviceBuffer, r0: usize, o: &DeviceBuffer, m: usize) -> Step {
        let d = self.d;
        let off = self.sl(r0 * d, m * d);
        self.gpu.step_sliced(K_LN, &[x, self.g("ones"), self.g("zeros"), o], &[off, (0, 0), (0, 0), off], &[d as u32, m as u32, f(EPS)], m as u32)
    }

    /// `y = (1 + scale)·xhat + shift` for one site over rows `r0..r0+m`.
    fn film(&self, xh: &DeviceBuffer, site: usize, o: &DeviceBuffer, r0: usize, m: usize) -> Step {
        let d = self.d;
        let off = self.sl(r0 * d, m * d);
        self.gpu.step_sliced(K_FILM, &[xh, self.g(&format!("sb{site}")), o], &[off, (0, 0), off], &[m as u32, d as u32, m as u32], (m * d) as u32)
    }

    /// `y = x + gate ⊙ h` for one site over rows `r0..r0+m`.
    fn gate(&self, x: &DeviceBuffer, site: usize, h: &DeviceBuffer, y: &DeviceBuffer, r0: usize, m: usize) -> Step {
        let d = self.d;
        let off = self.sl(r0 * d, m * d);
        self.gpu.step_sliced(K_GATE, &[x, self.g(&format!("gt{site}")), h, y], &[off, (0, 0), off, off], &[m as u32, d as u32, m as u32], (m * d) as u32)
    }

    /// Per-head QK-RMSNorm over rows `r0..r0+m` with the given scale.
    fn qknorm(&self, x: &DeviceBuffer, scale: &DeviceBuffer, o: &DeviceBuffer, r0: usize, m: usize) -> Step {
        let (d, hd, nh) = (self.d, self.hd, self.nh);
        let off = self.sl(r0 * d, m * d);
        let rows = (m * nh) as u32;
        let (kind, threads) = model::block::rms_variant(&self.gpu, K_RMS, Some(K_RMS_ROWS), rows, hd as u32);
        self.gpu.step_sliced(kind, &[x, scale, o], &[off, (0, 0), off], &[hd as u32, rows, f(EPS)], threads)
    }

    /// QK-RMSNorm backward over rows `r0..r0+m`: `gw += dnorm`, `dx` written.
    #[allow(clippy::too_many_arguments)]
    fn qknorm_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, scale: &DeviceBuffer, gw: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, inv: &str, slot: usize, r0: usize, m: usize) {
        let (d, hd, nh) = (self.d, self.hd, self.nh);
        let off = self.sl(r0 * d, m * d);
        let rows = (m * nh) as u32;
        let ib = self.sl(slot * self.n_max * nh, m * nh);
        let invb = self.g(inv);
        s.push(self.gpu.step_sliced(K_RINV, &[x, invb], &[off, ib], &[hd as u32, rows, f(EPS)], rows));
        s.push(self.gpu.step_sliced(K_RDW, &[dy, x, invb, gw], &[off, off, ib, (0, 0)], &[hd as u32, rows], hd as u32));
        s.push(self.gpu.step_sliced(K_RDX, &[x, scale, dy, dx], &[off, (0, 0), off, off], &[hd as u32, rows, f(EPS)], rows));
    }

    /// Per-head stride of the packed attention operands (`[nh, n, hd]`,
    /// padded to the storage-binding alignment).
    fn hstride(&self, n: usize) -> usize {
        model::block::pad64((n * self.hd) as u64) as usize
    }
    /// Per-head stride of the `[nh, n, n]` score/probability slabs.
    fn sstride(&self, n: usize) -> usize {
        model::block::pad64((n * n) as u64) as usize
    }

    /// `head_pack` one `[n, d]` row-major operand into per-head `[n, hd]`
    /// blocks (`t = true` packs it transposed, `[hd, n]`).
    fn pack(&self, src: &DeviceBuffer, dst: &str, n: usize, t: bool) -> Step {
        let (d, nh, hd) = (self.d, self.nh, self.hd);
        let hs = self.hstride(n);
        self.gpu.step(
            if t { K_HPACKT } else { K_HPACK },
            &[src, self.g(dst)],
            &[n as u32, nh as u32, 1, hd as u32, d as u32, 0, f(1.0), hs as u32],
            (nh * n * hd) as u32,
        )
    }

    /// Scatter per-head `[n, hd]` blocks back into a row-major `[n, d]` slab.
    fn unpack(&self, src: &str, dst: &DeviceBuffer, n: usize) -> Step {
        let (d, nh, hd) = (self.d, self.nh, self.hd);
        let hs = self.hstride(n);
        self.gpu.step(K_HUNPACK, &[self.g(src), dst], &[n as u32, nh as u32, hd as u32, d as u32, 0, hs as u32], (nh * n * hd) as u32)
    }

    /// One head's `matmul_reg3`: `o = a · bᵀ`, all three operands bound at
    /// their own head offset.
    #[allow(clippy::too_many_arguments)]
    fn mmh(&self, a: &str, ao: usize, b: &str, bo: usize, o: &str, oo: usize, m: usize, k: usize, nn: usize) -> Step {
        self.gpu.step_sliced(
            K_MM,
            &[self.g(a), self.g(b), self.g(o)],
            &[self.sl(ao, m * k), self.sl(bo, nn * k), self.sl(oo, m * nn)],
            &[m as u32, k as u32, nn as u32],
            d128(m) * d128(nn) * 256,
        )
    }

    /// One head's `matmul_dx_reg`: `o[m,k] = Σ_j a[m,j]·b[j,k]`.
    #[allow(clippy::too_many_arguments)]
    fn dxh(&self, a: &str, ao: usize, b: &str, bo: usize, o: &str, oo: usize, m: usize, k: usize, nn: usize) -> Step {
        self.gpu.step_sliced(
            K_DX,
            &[self.g(a), self.g(b), self.g(o)],
            &[self.sl(ao, m * nn), self.sl(bo, nn * k), self.sl(oo, m * k)],
            &[m as u32, k as u32, nn as u32, 0],
            d128(m) * d128(k) * 256,
        )
    }

    /// One head's `matmul_dw_reg`: `o[nn,k] += Σ_m a[m,nn]·b[m,k]`.
    #[allow(clippy::too_many_arguments)]
    fn dwh(&self, a: &str, ao: usize, b: &str, bo: usize, o: &str, oo: usize, m: usize, k: usize, nn: usize) -> Step {
        self.gpu.step_sliced(
            K_DW,
            &[self.g(a), self.g(b), self.g(o)],
            &[self.sl(ao, m * nn), self.sl(bo, m * k), self.sl(oo, nn * k)],
            &[m as u32, k as u32, nn as u32],
            d128(nn) * d128(k) * 256,
        )
    }

    /// Joint bidirectional attention as REAL GEMMs.
    ///
    /// The naive one-thread-per-score family (`attn_scores_bidir` +
    /// `attn_bwd_dscores_bidir`) is what a first port reaches for, and on this
    /// hardware it dominated the whole training step: every thread walked a
    /// full `n·head_dim` inner product out of an interleaved `[n, nh·hd]`
    /// layout, so consecutive lanes were `nh·hd` floats apart and the card ran
    /// at a sliver of peak. Every one of those products is a matrix product,
    /// so this packs the operands head-major (`head_pack`, the layout
    /// `model::block::gemm_bidir_fwd` introduced for exactly this) and hands
    /// them to the register-tiled GEMMs the linears already run at a large
    /// fraction of peak.
    ///
    /// The `1/√head_dim` scale is folded into the **QK-RMSNorm weight** at
    /// upload rather than applied here, which makes both directions
    /// scale-free: `scores = q'·kᵀ` exactly, so the softmax Jacobian, `dq'`
    /// and `dk` carry no stray factor, and `head_unpack` (which has no scale
    /// parameter) is the exact adjoint of `head_pack`.
    fn attn_fwd(&self, s: &mut Vec<Step>, n: usize) {
        let (nh, hd) = (self.nh, self.hd);
        let (hs, ss) = (self.hstride(n), self.sstride(n));
        s.push(self.pack(self.g("qr"), "qpk", n, false));
        s.push(self.pack(self.g("kr"), "kpk", n, false));
        s.push(self.pack(self.g("v"), "vpk", n, false));
        s.push(self.pack(self.g("v"), "vpkt", n, true));
        for h in 0..nh {
            // scores[h] = q'[h] · k[h]ᵀ   ([n,hd]·[n,hd]ᵀ)
            s.push(self.mmh("qpk", h * hs, "kpk", h * hs, "scores", h * ss, n, hd, n));
        }
        for h in 0..nh {
            // Per-head softmax over the contiguous [n,n] block: the head-major
            // padding gap stays invisible to a single-head dispatch.
            s.push(self.gpu.step_sliced(K_SOFTMAX, &[self.g("scores"), self.g("probs")], &[self.sl(h * ss, n * n), self.sl(h * ss, n * n)], &[1, 1, n as u32], n as u32));
        }
        for h in 0..nh {
            // ctx[h] = probs[h] · V[h]   (A·Bᵀ with B = vᵀ[hd,n])
            s.push(self.mmh("probs", h * ss, "vpkt", h * hs, "ctxpk", h * hs, n, n, hd));
        }
        s.push(self.unpack("ctxpk", self.g("ctx"), n));
    }

    /// Joint attention backward from `dctx` → `dqr`, `dkr`, `dv`, as GEMMs.
    fn attn_bwd(&self, s: &mut Vec<Step>, n: usize) {
        let (nh, hd) = (self.nh, self.hd);
        let (hs, ss) = (self.hstride(n), self.sstride(n));
        s.push(self.pack(self.g("dctx"), "dcpk", n, false));
        for h in 0..nh {
            // dprobs[h] = dctx[h] · V[h]ᵀ, into the score slab the softmax
            // has already consumed (never aliasing probs or dscores).
            s.push(self.mmh("dcpk", h * hs, "vpk", h * hs, "scores", h * ss, n, hd, n));
        }
        for h in 0..nh {
            // Softmax Jacobian in place of the probabilities:
            // dscores = p ⊙ (dprobs − Σ_j p·dprobs).
            s.push(self.gpu.step_sliced(
                K_SMDX,
                &[self.g("probs"), self.g("scores"), self.g("dscores")],
                &[self.sl(h * ss, n * n), self.sl(h * ss, n * n), self.sl(h * ss, n * n)],
                &[n as u32, n as u32, 1],
                n as u32,
            ));
        }
        for h in 0..nh {
            // dq'[h] = dscores[h] · k[h]
            s.push(self.dxh("dscores", h * ss, "kpk", h * hs, "dqpk", h * hs, n, hd, n));
            // dk[h] = dscoresᵀ[h] · q'[h]   (dw-shaped: sums over queries)
            s.push(self.dwh("dscores", h * ss, "qpk", h * hs, "dkpk", h * hs, n, hd, n));
            // dv[h] = probsᵀ[h] · dctx[h]
            s.push(self.dwh("probs", h * ss, "dcpk", h * hs, "dvpk", h * hs, n, hd, n));
        }
        s.push(self.unpack("dqpk", self.g("dqr"), n));
        s.push(self.unpack("dkpk", self.g("dkr"), n));
        s.push(self.unpack("dvpk", self.g("dv"), n));
        let half = (hd / 2) as u32;
        for (dr, dn) in [("dqr", "dqn"), ("dkr", "dkn")] {
            s.push(self.gpu.step(K_ROPE, &[self.g(dr), self.g("cos"), self.g("nsin"), self.g(dn)], &[n as u32, nh as u32, hd as u32, half], (n * nh * half as usize) as u32));
        }
    }

    /// RoPE the normalised q/k in place of the forward.
    fn rope_fwd(&self, s: &mut Vec<Step>, n: usize) {
        let (nh, hd) = (self.nh, self.hd);
        let half = (hd / 2) as u32;
        for (src, dst) in [("qn", "qr"), ("kn", "kr")] {
            s.push(self.gpu.step(K_ROPE, &[self.g(src), self.g("cos"), self.g("sin"), self.g(dst)], &[n as u32, nh as u32, hd as u32, half], (n * nh * half as usize) as u32));
        }
    }

    /// `dst += src` over `len` elements.
    fn acc(&self, dst: &DeviceBuffer, src: &DeviceBuffer, len: usize) -> Step {
        self.gpu.step(K_ADD_INPLACE, &[dst, src], &[len as u32], len as u32)
    }

    /// Accumulate one site's `(d_scale, d_shift)` and `d_gate` contributions.
    fn site_dsb(&self, s: &mut Vec<Step>, xh: &DeviceBuffer, dy: &DeviceBuffer, site: usize, r0: usize, m: usize) {
        let d = self.d;
        let off = self.sl(r0 * d, m * d);
        s.push(self.gpu.step_sliced(K_FILM_DSB, &[xh, dy, self.g("dsb")], &[off, off, (0, 0)], &[m as u32, d as u32, m as u32], d as u32));
        s.push(self.acc(self.g(&format!("gsb{site}")), self.g("dsb"), 2 * d));
    }

    fn site_dgate(&self, s: &mut Vec<Step>, dy: &DeviceBuffer, h: &DeviceBuffer, site: usize, r0: usize, m: usize) {
        let d = self.d;
        let off = self.sl(r0 * d, m * d);
        s.push(self.gpu.step_sliced(K_GATE_DG, &[dy, h, self.g("dgt")], &[off, off, (0, 0)], &[m as u32, d as u32, m as u32], d as u32));
        s.push(self.acc(self.g(&format!("ggt{site}")), self.g("dgt"), d));
    }

    /// `dh = gate ⊙ dy` over rows `r0..r0+m`.
    fn gate_dh(&self, dy: &DeviceBuffer, site: usize, dh: &DeviceBuffer, r0: usize, m: usize) -> Step {
        let d = self.d;
        let off = self.sl(r0 * d, m * d);
        self.gpu.step_sliced(K_GATE_DH, &[dy, self.g(&format!("gt{site}")), dh], &[off, (0, 0), off], &[m as u32, d as u32, m as u32], (m * d) as u32)
    }

    /// Modulated-LN backward over rows `r0..r0+m`: accumulate the site's
    /// scale/shift grads, then `dxhat` and the LayerNorm `dx` into `dtmp`.
    fn modln_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, xr0: usize, xh: &DeviceBuffer, dy: &DeviceBuffer, site: usize, r0: usize, m: usize) {
        let d = self.d;
        let off = self.sl(r0 * d, m * d);
        let xoff = self.sl(xr0 * d, m * d);
        self.site_dsb(s, xh, dy, site, r0, m);
        s.push(self.gpu.step_sliced(K_FILM_DX, &[dy, self.g(&format!("sb{site}")), self.g("dxh")], &[off, (0, 0), off], &[m as u32, d as u32, m as u32], (m * d) as u32));
        s.push(self.gpu.step_sliced(
            K_LN_DX,
            &[x, self.g("ones"), self.g("dxh"), self.g("dtmp")],
            &[xoff, (0, 0), off, off],
            &[d as u32, m as u32, f(EPS)],
            m as u32,
        ));
    }
}

// ---- the frozen wrapper: embedders in, final layer out ----

impl BlockDev {
    /// Embed both streams into the joint slab `x = [txt | img]`. Both linears
    /// are frozen (no adapter, no `dW`) and their inputs are data, so this pass
    /// has no backward at all - the whole reason the embedders can live on the
    /// device without any gradient machinery.
    #[allow(clippy::too_many_arguments)]
    pub fn embed(&self, txt_in: &DeviceBuffer, ctx: &DeviceBuffer, cdim: usize, img_in: &DeviceBuffer, tok: &DeviceBuffer, cin: usize, dm: Dims, x: &DeviceBuffer) {
        let (nt, ni, d) = (dm.nt, dm.ni, dm.d);
        let s = vec![
            model::dispatch::mm_rows_off(&self.gpu, self.tier(), ctx, txt_in, x, 0, 0, nt as u32, cdim as u32, d as u32),
            model::dispatch::mm_rows_off(&self.gpu, self.tier(), tok, img_in, x, 0, (nt * d) as u64, ni as u32, cin as u32, d as u32),
        ];
        self.gpu.submit(&[], &s);
        self.gpu.poll_wait();
    }

    /// Final layer over the image rows: modulated LN under [`SITE_FINAL`], then
    /// the frozen `[cin, d]` head. Writes `pred [ni, cin]`.
    pub fn head_forward(&self, final_w: &DeviceBuffer, x: &DeviceBuffer, dm: Dims, cin: usize, pred: &DeviceBuffer) {
        let (nt, ni, d) = (dm.nt, dm.ni, dm.d);
        let s = vec![
            self.ln(x, nt, self.g("xh1"), ni),
            self.film(self.g("xh1"), SITE_FINAL, self.g("n1"), nt, ni),
            model::dispatch::mm_rows_off(&self.gpu, self.tier(), self.g("n1"), final_w, pred, nt as u32, 0, ni as u32, d as u32, cin as u32),
        ];
        self.gpu.submit(&[], &s);
        self.gpu.poll_wait();
    }

    /// Final-layer backward from `dpred [ni, cin]` into `dx [n, d]`. The text
    /// rows of `dx` are zero (the head only sees image rows) - cleared on the
    /// device rather than uploaded.
    #[allow(clippy::too_many_arguments)]
    pub fn head_backward(&self, final_w: &DeviceBuffer, x: &DeviceBuffer, dm: Dims, cin: usize, dpred: &DeviceBuffer, dx: &DeviceBuffer) {
        let (nt, ni, d) = (dm.nt, dm.ni, dm.d);
        let mut s = Vec::new();
        s.push(self.gpu.step_sliced(
            K_DX,
            &[dpred, final_w, self.g("dn1")],
            &[(0, (ni * cin) as u64), (0, 0), self.sl(nt * d, ni * d)],
            &[ni as u32, d as u32, cin as u32, 0],
            d128(ni) * d128(d) * 256,
        ));
        self.site_dsb(&mut s, self.g("xh1"), self.g("dn1"), SITE_FINAL, nt, ni);
        let off = self.sl(nt * d, ni * d);
        s.push(self.gpu.step_sliced(K_FILM_DX, &[self.g("dn1"), self.g(&format!("sb{SITE_FINAL}")), self.g("dxh")], &[off, (0, 0), off], &[ni as u32, d as u32, ni as u32], (ni * d) as u32));
        s.push(self.gpu.step_sliced(K_LN_DX, &[x, self.g("ones"), self.g("dxh"), dx], &[off, (0, 0), off, off], &[d as u32, ni as u32, f(EPS)], ni as u32));
        self.gpu.submit(&[dx], &s);
        self.gpu.poll_wait();
    }
}

// ---- double block ----

impl BlockDev {
    /// The `(slot, r0, m, weights, attn-site, mlp-site)` of each stream of a
    /// double block. Text rows come first in the joint slab, matching
    /// [`crate::grad::double_forward`].
    fn double_arms<'a>(&self, w: &'a DoubleDev, dm: Dims) -> [(usize, usize, usize, &'a StreamDev, usize, usize); 2] {
        [(0, 0, dm.nt, &w.txt, SITE_TXT1, SITE_TXT2), (1, dm.nt, dm.ni, &w.img, SITE_IMG1, SITE_IMG2)]
    }

    fn double_fwd(&self, s: &mut Vec<Step>, w: &DoubleDev, dm: Dims, x: &DeviceBuffer, out: &DeviceBuffer) {
        let (n, mlp) = (dm.n(), dm.mlp);
        let arms = self.double_arms(w, dm);

        s.push(self.ln(x, 0, self.g("xh1"), n));
        for &(slot, r0, m, sw, s1, _) in &arms {
            s.push(self.film(self.g("xh1"), s1, self.g("n1"), r0, m));
            self.lin_fwd(s, &sw.wq, 0, slot, self.g("n1"), r0, self.g("q"), r0, m);
            self.lin_fwd(s, &sw.wk, 1, slot, self.g("n1"), r0, self.g("k"), r0, m);
            self.lin_fwd(s, &sw.wv, 2, slot, self.g("n1"), r0, self.g("v"), r0, m);
            s.push(self.qknorm(self.g("q"), &sw.nq, self.g("qn"), r0, m));
            s.push(self.qknorm(self.g("k"), &sw.nk, self.g("kn"), r0, m));
        }
        self.rope_fwd(s, n);
        self.attn_fwd(s, n);
        for &(slot, r0, m, sw, s1, _) in &arms {
            self.lin_fwd(s, &sw.wo, 3, slot, self.g("ctx"), r0, self.g("proj"), r0, m);
            s.push(self.gate(x, s1, self.g("proj"), self.g("x1"), r0, m));
        }
        s.push(self.ln(self.g("x1"), 0, self.g("xh2"), n));
        for &(slot, r0, m, sw, _, s2) in &arms {
            s.push(self.film(self.g("xh2"), s2, self.g("n2"), r0, m));
            self.lin_fwd(s, &sw.w1, 4, slot, self.g("n2"), r0, self.g("h1"), r0, m);
            self.lin_fwd(s, &sw.w3, 5, slot, self.g("n2"), r0, self.g("h2"), r0, m);
        }
        s.push(self.gpu.step(K_SILU, &[self.g("h1"), self.g("h2"), self.g("hs")], &[(n * mlp) as u32], (n * mlp) as u32));
        for &(slot, r0, m, sw, _, s2) in &arms {
            self.lin_fwd(s, &sw.w2, 6, slot, self.g("hs"), r0, self.g("mlpo"), r0, m);
            s.push(self.gate(self.g("x1"), s2, self.g("mlpo"), out, r0, m));
        }
    }

    /// Forward one double block: `out = block(x)`, both slabs `[n, d]`.
    pub fn double_forward(&self, w: &DoubleDev, dm: Dims, x: &DeviceBuffer, out: &DeviceBuffer) {
        let mut s = Vec::new();
        self.double_fwd(&mut s, w, dm, x, out);
        self.gpu.submit(&[], &s);
        self.gpu.poll_wait();
    }

    /// Backward one double block: recompute the forward from the saved input
    /// `x`, then backpropagate `dout` into the adapter/QK-norm/site gradients
    /// and write `dx`.
    pub fn double_backward(&self, w: &DoubleDev, dm: Dims, x: &DeviceBuffer, dout: &DeviceBuffer, dx: &DeviceBuffer) {
        let (n, d, mlp) = (dm.n(), dm.d, dm.mlp);
        let mut s = Vec::new();
        self.double_fwd(&mut s, w, dm, x, self.g("pm"));
        let arms = self.double_arms(w, dm);

        // out = x1 + gate_s2 ⊙ mlpo
        for &(_, r0, m, _, _, s2) in &arms {
            self.site_dgate(&mut s, dout, self.g("mlpo"), s2, r0, m);
            s.push(self.gate_dh(dout, s2, self.g("dmlpo"), r0, m));
        }
        // mlpo = hs · w2ᵀ  →  dhs. Each stream owns a disjoint row range of
        // `dhs`, so both are first writers of their own rows (`acc = false`).
        for &(slot, r0, m, sw, _, _) in &arms {
            self.lin_bwd(&mut s, &sw.w2, 6, slot, self.g("hs"), r0, self.g("dmlpo"), r0, self.g("dhs"), r0, m, false);
        }
        // hs = silu(h1) ⊙ h2
        s.push(self.gpu.step(K_SDA, &[self.g("h1"), self.g("h2"), self.g("dhs"), self.g("dh1")], &[(n * mlp) as u32], (n * mlp) as u32));
        s.push(self.gpu.step(K_SDB, &[self.g("h1"), self.g("dhs"), self.g("dh2")], &[(n * mlp) as u32], (n * mlp) as u32));
        // h1 = n2 · w1ᵀ, h2 = n2 · w3ᵀ  →  dn2
        for &(slot, r0, m, sw, _, _) in &arms {
            self.lin_bwd(&mut s, &sw.w1, 4, slot, self.g("n2"), r0, self.g("dh1"), r0, self.g("dn2"), r0, m, false);
            self.lin_bwd(&mut s, &sw.w3, 5, slot, self.g("n2"), r0, self.g("dh2"), r0, self.g("dn2"), r0, m, true);
        }
        // n2 = film(LN(x1), s2) ; dx1 = dout + LN_dx
        for &(_, r0, m, _, _, s2) in &arms {
            self.modln_bwd(&mut s, self.g("x1"), r0, self.g("xh2"), self.g("dn2"), s2, r0, m);
        }
        s.push(self.gpu.step(K_ADD2, &[dout, self.g("dtmp"), self.g("dx1")], &[(n * d) as u32], (n * d) as u32));

        // x1 = x + gate_s1 ⊙ proj
        for &(_, r0, m, _, s1, _) in &arms {
            self.site_dgate(&mut s, self.g("dx1"), self.g("proj"), s1, r0, m);
            s.push(self.gate_dh(self.g("dx1"), s1, self.g("dproj"), r0, m));
        }
        for &(slot, r0, m, sw, _, _) in &arms {
            self.lin_bwd(&mut s, &sw.wo, 3, slot, self.g("ctx"), r0, self.g("dproj"), r0, self.g("dctx"), r0, m, false);
        }
        self.attn_bwd(&mut s, n);
        for &(slot, r0, m, sw, _, _) in &arms {
            self.qknorm_bwd(&mut s, self.g("q"), &sw.nq, &sw.gnq, self.g("dqn"), self.g("dq"), "inv_q", slot, r0, m);
            self.qknorm_bwd(&mut s, self.g("k"), &sw.nk, &sw.gnk, self.g("dkn"), self.g("dk"), "inv_k", slot, r0, m);
            self.lin_bwd(&mut s, &sw.wq, 0, slot, self.g("n1"), r0, self.g("dq"), r0, self.g("dn1"), r0, m, false);
            self.lin_bwd(&mut s, &sw.wk, 1, slot, self.g("n1"), r0, self.g("dk"), r0, self.g("dn1"), r0, m, true);
            self.lin_bwd(&mut s, &sw.wv, 2, slot, self.g("n1"), r0, self.g("dv"), r0, self.g("dn1"), r0, m, true);
        }
        for &(_, r0, m, _, s1, _) in &arms {
            self.modln_bwd(&mut s, x, r0, self.g("xh1"), self.g("dn1"), s1, r0, m);
        }
        s.push(self.gpu.step(K_ADD2, &[self.g("dx1"), self.g("dtmp"), dx], &[(n * d) as u32], (n * d) as u32));
        self.gpu.submit(&self.double_clears(w), &s);
        self.gpu.poll_wait();
    }
}

// ---- single block ----

impl BlockDev {
    fn single_fwd(&self, s: &mut Vec<Step>, w: &SingleDev, dm: Dims, x: &DeviceBuffer, out: &DeviceBuffer) {
        let (n, d, mlp) = (dm.n(), dm.d, dm.mlp);
        s.push(self.ln(x, 0, self.g("xh1"), n));
        s.push(self.film(self.g("xh1"), SITE_SGL, self.g("n1"), 0, n));
        self.lin_fwd(s, &w.wq, 0, 0, self.g("n1"), 0, self.g("q"), 0, n);
        self.lin_fwd(s, &w.wk, 1, 0, self.g("n1"), 0, self.g("k"), 0, n);
        self.lin_fwd(s, &w.wv, 2, 0, self.g("n1"), 0, self.g("v"), 0, n);
        s.push(self.qknorm(self.g("q"), &w.nq, self.g("qn"), 0, n));
        s.push(self.qknorm(self.g("k"), &w.nk, self.g("kn"), 0, n));
        self.rope_fwd(s, n);
        self.attn_fwd(s, n);
        self.lin_fwd(s, &w.w1, 3, 0, self.g("n1"), 0, self.g("h1"), 0, n);
        self.lin_fwd(s, &w.w3, 4, 0, self.g("n1"), 0, self.g("h2"), 0, n);
        s.push(self.gpu.step(K_SILU, &[self.g("h1"), self.g("h2"), self.g("hs")], &[(n * mlp) as u32], (n * mlp) as u32));
        self.lin_fwd(s, &w.wo_a, 5, 0, self.g("ctx"), 0, self.g("proj"), 0, n);
        self.lin_fwd(s, &w.wo_b, 6, 0, self.g("hs"), 0, self.g("mlpo"), 0, n);
        // out = x + gate ⊙ (proj + mlpo): one gate over the SUM, so the gate
        // gradient sees the sum the reference differentiates.
        s.push(self.gpu.step(K_ADD2, &[self.g("proj"), self.g("mlpo"), self.g("pm")], &[(n * d) as u32], (n * d) as u32));
        s.push(self.gate(x, SITE_SGL, self.g("pm"), out, 0, n));
    }

    /// Forward one single block.
    pub fn single_forward(&self, w: &SingleDev, dm: Dims, x: &DeviceBuffer, out: &DeviceBuffer) {
        let mut s = Vec::new();
        self.single_fwd(&mut s, w, dm, x, out);
        self.gpu.submit(&[], &s);
        self.gpu.poll_wait();
    }

    /// Backward one single block (recompute + backprop), writing `dx`.
    pub fn single_backward(&self, w: &SingleDev, dm: Dims, x: &DeviceBuffer, dout: &DeviceBuffer, dx: &DeviceBuffer) {
        let (n, d, mlp) = (dm.n(), dm.d, dm.mlp);
        let mut s = Vec::new();
        self.single_fwd(&mut s, w, dm, x, self.g("x1"));
        // out = x + gate ⊙ pm
        self.site_dgate(&mut s, dout, self.g("pm"), SITE_SGL, 0, n);
        s.push(self.gate_dh(dout, SITE_SGL, self.g("dpm"), 0, n));
        // pm = proj + mlpo  →  both branches take dpm unchanged
        self.lin_bwd(&mut s, &w.wo_a, 5, 0, self.g("ctx"), 0, self.g("dpm"), 0, self.g("dctx"), 0, n, false);
        self.lin_bwd(&mut s, &w.wo_b, 6, 0, self.g("hs"), 0, self.g("dpm"), 0, self.g("dhs"), 0, n, false);
        s.push(self.gpu.step(K_SDA, &[self.g("h1"), self.g("h2"), self.g("dhs"), self.g("dh1")], &[(n * mlp) as u32], (n * mlp) as u32));
        s.push(self.gpu.step(K_SDB, &[self.g("h1"), self.g("dhs"), self.g("dh2")], &[(n * mlp) as u32], (n * mlp) as u32));
        // n1 feeds five linears: w1, w3, wq, wk, wv - accumulate all into dn1.
        self.lin_bwd(&mut s, &w.w1, 3, 0, self.g("n1"), 0, self.g("dh1"), 0, self.g("dn1"), 0, n, false);
        self.lin_bwd(&mut s, &w.w3, 4, 0, self.g("n1"), 0, self.g("dh2"), 0, self.g("dn1"), 0, n, true);
        self.attn_bwd(&mut s, n);
        self.qknorm_bwd(&mut s, self.g("q"), &w.nq, &w.gnq, self.g("dqn"), self.g("dq"), "inv_q", 0, 0, n);
        self.qknorm_bwd(&mut s, self.g("k"), &w.nk, &w.gnk, self.g("dkn"), self.g("dk"), "inv_k", 0, 0, n);
        self.lin_bwd(&mut s, &w.wq, 0, 0, self.g("n1"), 0, self.g("dq"), 0, self.g("dn1"), 0, n, true);
        self.lin_bwd(&mut s, &w.wk, 1, 0, self.g("n1"), 0, self.g("dk"), 0, self.g("dn1"), 0, n, true);
        self.lin_bwd(&mut s, &w.wv, 2, 0, self.g("n1"), 0, self.g("dv"), 0, self.g("dn1"), 0, n, true);
        self.modln_bwd(&mut s, x, 0, self.g("xh1"), self.g("dn1"), SITE_SGL, 0, n);
        s.push(self.gpu.step(K_ADD2, &[dout, self.g("dtmp"), dx], &[(n * d) as u32], (n * d) as u32));
        self.gpu.submit(&self.single_clears(w), &s);
        self.gpu.poll_wait();
    }
}
