// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) forward + backward for ONE `WanAttentionBlock`, as a
//! **persistent** engine ([`BlockDev`]) - the compute core of the device
//! training loop.
//!
//! One device and one set of buffers sized to `(max_tokens, text_len)` serve
//! every block of the stack: each call uploads that block's weights, records
//! the graph and submits it, so a 30-block training step is 60 cheap submits
//! rather than 30 device creations.
//!
//! Every op the gradchecked host reference ([`crate::grad`]) does analytically
//! this does on-device with brain's existing kernels:
//!
//! * `layernorm` / `ln_stats` + `layernorm_dgamma` + `layernorm_dbeta` +
//!   `layernorm_dx` for the three LayerNorms. The modulated pair is folded to
//!   `(gamma, beta) = (1 + scale, shift)`, which is an identity because
//!   `modulation + e0` carries no token axis; `d(gamma) == d(scale)`, so the
//!   six-vector modulation grad falls straight out of the norm grads plus the
//!   two gate grads.
//! * `matmul`/`matmul_reg3` + `bias_add` forward, `matmul_dx{,_reg}` /
//!   `matmul_dw{,_reg}` / `bias_grad` backward for the ten biased linears.
//! * `rms_inv_eps` + `rmsnorm_dw` + `rmsnorm_dx_eps` for QK normalisation,
//!   over the FULL model width (see [`crate::block::qk_norm`]).
//! * The `*_cross` attention family for BOTH attentions. Wan's self-attention
//!   is the `t == t` case of the same two-length kernel its text
//!   cross-attention needs, so one set of dispatches covers both; each of q, k
//!   and v lives in its own contiguous `[rows, dim]` buffer, which the family's
//!   `(stride, offset)` params express as `stride = dim, offset = 0`.
//! * `rope_interleave_table` forward, and the same kernel fed a NEGATED sine
//!   table for the backward (rotating the gradient by `−angle`).
//! * `gate_row` / `gate_row_dg` / `gate_row_dh` for the two gated residuals,
//!   and `gelu` / `gelu_bwd` for the FFN's only nonlinearity.
//!
//! The cross-attention residual is ungated (`add2`), which is the one
//! asymmetry between the block's three residual sites.
//!
//! ## LoRA mode: the base stays put and only the adapter moves
//!
//! [`BlockDev::enable_lora`] switches the engine to a second binding for the
//! ten weight matrices a Wan LoRA adapts ([`LORA_TARGETS`]). The frozen base
//! occupies the per-layer slots for the whole run, and each block visit
//! assembles `W_eff = base + scale·B·A` on-device - a rank-`r` `matmul` into a
//! shared scratch, then `add2` onto the base - which the recorded graphs bind
//! in place of the base. The backward's ten `dW` are projected back to
//! `(dA, dB)` by the same GEMM family before anything is read
//! ([`BlockDev::backward_lora_loaded`]), so a step's whole host<->device
//! weight traffic is the rank-sized adapter rather than the block's matrices.
//! No kernel here is new: `dA = (scale·B)ᵀ·dW` is `matmul_dw`'s contraction and
//! `dB = dW·Aᵀ` is `matmul`'s, the same two the block backward already runs.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};

use crate::grad::{BlockGrads, BlockW, Dims, Lin};

// Kernel-table indices (order matches KERNELS).
const K_LN: usize = 0;
const K_LN_STATS: usize = 1;
const K_LN_DX: usize = 2;
const K_LN_DG: usize = 3;
const K_LN_DB: usize = 4;
const K_MM: usize = 5;
const K_MM_REG: usize = 6;
const K_BIAS_ADD: usize = 7;
const K_MM_DX: usize = 8;
const K_MM_DX_REG: usize = 9;
const K_MM_DW: usize = 10;
const K_MM_DW_REG: usize = 11;
const K_BIAS_GRAD: usize = 12;
const K_RMS: usize = 13;
const K_RMS_INV: usize = 14;
const K_RMS_DW: usize = 15;
const K_RMS_DX: usize = 16;
const K_ROPE: usize = 17;
const K_SCORES: usize = 18;
const K_SOFTMAX: usize = 19;
const K_APPLY: usize = 20;
const K_D_SCORES: usize = 21;
const K_D_V: usize = 22;
const K_D_Q: usize = 23;
const K_D_K: usize = 24;
const K_GATE: usize = 25;
const K_GATE_DG: usize = 26;
const K_GATE_DH: usize = 27;
const K_ADD2: usize = 28;
const K_GELU: usize = 29;
const K_GELU_BWD: usize = 30;
const K_LN_ROWS: usize = 31;
const K_LN_STATS_ROWS: usize = 32;
const K_LN_DX_ROWS: usize = 33;
const K_RMS_ROWS: usize = 34;
const K_SOFTMAX_ROWS: usize = 35;

/// Every kernel the trainable block dispatches. All of them already exist: the
/// backward here is Wan's shapes through kernels other models' training paths
/// already gradcheck.
pub const KERNELS: [(&str, &str); 36] = [
    ("layernorm", kernels::LAYERNORM),
    ("ln_stats", kernels::LN_STATS),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("matmul", kernels::MATMUL),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("bias_add", kernels::BIAS_ADD),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw", kernels::MATMUL_DW),
    ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ("bias_grad", kernels::BIAS_GRAD),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rms_inv_eps", kernels::RMS_INV_EPS),
    ("rmsnorm_dw", kernels::RMSNORM_DW),
    ("rmsnorm_dx_eps", kernels::RMSNORM_DX_EPS),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    // The two-length attention family serves the self-attention (`t == t`) and
    // the text cross-attention alike.
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),
    ("gate_row", kernels::GATE_ROW),
    ("gate_row_dg", kernels::GATE_ROW_DG),
    ("gate_row_dh", kernels::GATE_ROW_DH),
    ("add2", kernels::ADD2),
    ("gelu", kernels::GELU),
    ("gelu_bwd", kernels::GELU_BWD),
    // Cooperative (workgroup-per-row) variants, chosen through the shared
    // selection rules in `model::block` from the device's queried caps.
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("ln_stats_rows", kernels::LN_STATS_ROWS),
    ("layernorm_dx_rows", kernels::LAYERNORM_DX_ROWS),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
];

/// The ten weight matrices a Wan LoRA adapts, in the order [`crate::lora`]
/// walks its adapter pairs.
pub const LORA_TARGETS: [&str; 10] = ["sqw", "skw", "svw", "sow", "cqw", "ckw", "cvw", "cow", "ff1w", "ff2w"];

/// One block's adapter gradients: `(dA [r·in], dB [out·r])` per
/// [`LORA_TARGETS`] entry, in that order.
pub type AdapterGrads = Vec<(Vec<f32>, Vec<f32>)>;

/// `(out, in)` of one [`LORA_TARGETS`] entry in a `dim`/`ffn` block.
fn target_dims(name: &str, dim: usize, ffn: usize) -> (usize, usize) {
    match name {
        "ff1w" => (ffn, dim),
        "ff2w" => (dim, ffn),
        _ => (dim, dim),
    }
}

/// Every gradient buffer whose kernel ACCUMULATES into it, so a backward must
/// start it at zero. The gate grads are absent on purpose: `gate_row_dg`
/// overwrites.
const ACCUMULATING_GRADS: [&str; 30] = [
    "g_sqw", "g_skw", "g_svw", "g_sow", "g_cqw", "g_ckw", "g_cvw", "g_cow", "g_ff1w", "g_ff2w",
    "g_sqb", "g_skb", "g_svb", "g_sob", "g_cqb", "g_ckb", "g_cvb", "g_cob", "g_ff1b", "g_ff2b",
    "g_snq", "g_snk", "g_cnq", "g_cnk", "g_n3w", "g_n3b", "g_ln1g", "g_ln1b", "g_ln2g", "g_ln2b",
];

/// The six `[dim]` vectors of `modulation + e0`, in the checkpoint's chunk
/// order, folded into the two LayerNorm affines and the two residual gates.
struct Mods {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    gate1: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    gate2: Vec<f32>,
}

fn fold(modulation: &[f32], e0: &[f32], dim: usize) -> Mods {
    let p: Vec<f32> = modulation.iter().zip(e0).map(|(&a, &b)| a + b).collect();
    let part = |i: usize| p[i * dim..(i + 1) * dim].to_vec();
    let one_plus = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| 1.0 + x).collect() };
    Mods {
        ln1_g: one_plus(part(1)),
        ln1_b: part(0),
        gate1: part(2),
        ln2_g: one_plus(part(4)),
        ln2_b: part(3),
        gate2: part(5),
    }
}

/// On-device LoRA state, allocated by [`BlockDev::enable_lora`].
///
/// With the base frozen, a training step's only changing weight input is the
/// tiny `(A, B)` pair per target, so the base stays resident and the effective
/// weight `W_eff = base + scale·B·A` is assembled HERE, on the device: `eff`
/// is what the forward and backward graphs bind in place of the base, and the
/// weight grad they produce is projected back to `(dA, dB)` without ever
/// crossing the bus.
struct LoraDev {
    r: usize,
    /// `α/r`, set by [`BlockDev::upload_lora`] and applied to `dB` on readback.
    scale: std::cell::Cell<f32>,
    /// `W_eff` per target - what [`BlockDev::g`] resolves the target names to.
    eff: HashMap<&'static str, DeviceBuffer>,
    /// `scale·B [out×r]`: the fold's left operand and the `dA` projection's.
    /// Pre-scaled host-side so both device GEMMs multiply the same
    /// `scale·B[o,k]` the host reference does, in the same place.
    bs: HashMap<&'static str, DeviceBuffer>,
    /// `Aᵀ [in×r]`, the fold GEMM's right operand.
    at: HashMap<&'static str, DeviceBuffer>,
    /// `A [r×in]`, the `dB` projection's right operand.
    a: HashMap<&'static str, DeviceBuffer>,
    /// `dA [r×in]` (already scaled) and `dB/scale [out×r]`.
    ga: HashMap<&'static str, DeviceBuffer>,
    gb: HashMap<&'static str, DeviceBuffer>,
    /// `B·A` before it is added onto the base - one scratch per distinct
    /// target size, shared by the targets of that size. Dispatches in a submit
    /// are ordered and barriered, so the reuse is safe.
    dsq: DeviceBuffer,
    dff: DeviceBuffer,
}

/// A persistent GPU engine for one Wan DiT block: one device, buffers sized to
/// `max_t` latent tokens and `te` text rows, driving any block's forward or
/// backward.
pub struct BlockDev {
    gpu: Gpu,
    ln: model::block::LayerNormIds,
    dim: usize,
    nh: usize,
    hd: usize,
    ffn: usize,
    te: usize,
    eps: f32,
    /// Activations, gradients and the RoPE tables - one set, reused by every
    /// layer and every step.
    b: HashMap<&'static str, DeviceBuffer>,
    /// One WEIGHT set per resident layer. A training step runs the stack
    /// forward and then backward, and the backward recomputes each block's
    /// forward - so with a slot per layer the step uploads a block's weights
    /// ONCE instead of once per sweep. At 1.3B widths a block is 185 MB, so
    /// that is gigabytes of host-to-device traffic per step. `slots()` is what
    /// the budget allowed; a stack deeper than that cycles through the slots
    /// and re-uploads, which is exactly the old behaviour.
    w: Vec<HashMap<&'static str, DeviceBuffer>>,
    /// The slot `g` currently resolves weight names in.
    slot: std::cell::Cell<usize>,
    /// Present once [`BlockDev::enable_lora`] has run: the graphs then bind
    /// folded effective weights instead of the raw slot contents.
    lora: Option<LoraDev>,
}

impl BlockDev {
    /// Build on brain's default device.
    pub fn new(d: Dims, max_t: usize) -> BlockDev {
        BlockDev::from_gpu(Gpu::open(None, &KERNELS), d, max_t)
    }

    /// Build on a named device (`"cpu"`, `"gpu"`, or `None` for the default).
    pub fn on_device(d: Dims, max_t: usize, device: Option<&str>) -> BlockDev {
        BlockDev::from_gpu(Gpu::open(device, &KERNELS), d, max_t)
    }

    pub fn from_gpu(gpu: Gpu, d: Dims, max_t: usize) -> BlockDev {
        let (dim, nh, ffn, te) = (d.dim, d.nh, d.ffn, d.te);
        let hd = d.hd();
        let mut b: HashMap<&'static str, DeviceBuffer> = HashMap::new();
        let mut mk = |name: &'static str, n: usize| {
            b.insert(name, gpu.storage((n as u64).max(1)));
        };
        let td = max_t * dim;
        let ted = te * dim;
        let tf = max_t * ffn;

        // --- forward activations ---
        for n in [
            "xb", "n1", "q", "k", "v", "qn", "kn", "qr", "kr", "actx", "ao", "x1", "n3", "xq", "xqn", "xctx", "xo", "x2", "n2", "ffb", "outb",
        ] {
            mk(n, td);
        }
        for n in ["ctxb", "xk", "xkn", "xv"] {
            mk(n, ted);
        }
        for n in ["h1", "hg"] {
            mk(n, tf);
        }
        for n in ["cosb", "sinb", "nsinb"] {
            mk(n, max_t * hd / 2);
        }
        for n in ["sscores", "sprobs", "d_ss"] {
            mk(n, nh * max_t * max_t);
        }
        for n in ["xscores", "xprobs", "d_sx"] {
            mk(n, nh * max_t * te);
        }
        for n in ["m1", "i1", "m3", "i3", "m2", "i2", "iq", "ik", "ixq"] {
            mk(n, max_t);
        }
        mk("ixk", te);

        // --- backward activations ---
        for n in [
            "dout", "d_ff", "d_n2", "d_x2a", "d_x2", "d_xctx", "d_xqn", "d_xq", "d_n3", "d_x1a", "d_x1", "d_ao", "d_actx", "d_qr", "d_kr",
            "d_v", "d_qn", "d_kn", "d_q", "d_k", "d_n1q", "d_n1k", "d_n1v", "d_n1t", "d_n1", "d_xa", "d_x",
        ] {
            mk(n, td);
        }
        for n in ["d_xkn", "d_xv", "d_xk", "d_ctxk", "d_ctxv", "d_ctx"] {
            mk(n, ted);
        }
        for n in ["d_hg", "d_h1"] {
            mk(n, tf);
        }

        // --- gradients ---
        for n in ["g_sqw", "g_skw", "g_svw", "g_sow", "g_cqw", "g_ckw", "g_cvw", "g_cow"] {
            mk(n, dim * dim);
        }
        mk("g_ff1w", ffn * dim);
        mk("g_ff2w", dim * ffn);
        for n in ["g_sqb", "g_skb", "g_svb", "g_sob", "g_cqb", "g_ckb", "g_cvb", "g_cob", "g_ff2b"] {
            mk(n, dim);
        }
        mk("g_ff1b", ffn);
        for n in ["g_snq", "g_snk", "g_cnq", "g_cnk", "g_n3w", "g_n3b", "g_ln1g", "g_ln1b", "g_gate1", "g_ln2g", "g_ln2b", "g_gate2"] {
            mk(n, dim);
        }

        let ln = model::block::LayerNormIds {
            layernorm: K_LN,
            layernorm_rows: Some(K_LN_ROWS),
            ln_stats: K_LN_STATS,
            ln_stats_rows: Some(K_LN_STATS_ROWS),
            layernorm_dx: K_LN_DX,
            layernorm_dx_rows: Some(K_LN_DX_ROWS),
        };
        let w = vec![Self::alloc_weights(&gpu, dim, ffn)];
        BlockDev { gpu, ln, dim, nh, hd, ffn, te, eps: d.eps as f32, b, w, slot: std::cell::Cell::new(0), lora: None }
    }

    /// Bytes one resident weight slot costs.
    pub fn slot_bytes(d: Dims) -> u64 {
        let (dim, ffn) = (d.dim as u64, d.ffn as u64);
        (8 * dim * dim + 2 * ffn * dim + 10 * dim + ffn + 12 * dim) * 4
    }

    fn alloc_weights(gpu: &Gpu, dim: usize, ffn: usize) -> HashMap<&'static str, DeviceBuffer> {
        let mut m = HashMap::new();
        let mut mk = |name: &'static str, n: usize| {
            m.insert(name, gpu.storage((n as u64).max(1)));
        };
        for n in ["sqw", "skw", "svw", "sow", "cqw", "ckw", "cvw", "cow"] {
            mk(n, dim * dim);
        }
        mk("ff1w", ffn * dim);
        mk("ff2w", dim * ffn);
        for n in ["sqb", "skb", "svb", "sob", "cqb", "ckb", "cvb", "cob", "ff2b"] {
            mk(n, dim);
        }
        mk("ff1b", ffn);
        for n in ["snq", "snk", "cnq", "cnk", "n3w", "n3b", "ln1g", "ln1b", "gate1", "ln2g", "ln2b", "gate2"] {
            mk(n, dim);
        }
        m
    }

    /// Grow the resident weight slots to `want`, stopping at whatever
    /// `budget_bytes` allows (and never below the one slot that always exists).
    /// Returns the slot count now available.
    pub fn reserve_slots(&mut self, want: usize, budget_bytes: u64) -> usize {
        let per = Self::slot_bytes(Dims { t: 0, te: self.te, dim: self.dim, nh: self.nh, ffn: self.ffn, eps: 0.0 });
        let affordable = (budget_bytes / per.max(1)).max(1) as usize;
        let target = want.max(1).min(affordable);
        while self.w.len() < target {
            self.w.push(Self::alloc_weights(&self.gpu, self.dim, self.ffn));
        }
        self.w.len()
    }

    /// How many layers' weights can stay resident at once.
    pub fn slots(&self) -> usize {
        self.w.len()
    }

    /// Point the weight-name lookup at slot `l` (taken modulo [`Self::slots`]).
    pub fn select_slot(&self, l: usize) {
        self.slot.set(l % self.w.len());
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// `true` when this engine landed on a real accelerator rather than the
    /// host CPU JIT - what a trainer consults before routing a step here.
    pub fn is_accelerated(&self) -> bool {
        self.gpu.caps().class != gpu_core::DeviceClass::Cpu
    }

    /// What a recorded graph binds for `name`: the folded effective weight
    /// where a LoRA run has one, else the slot's own buffer.
    fn g(&self, name: &str) -> &DeviceBuffer {
        if let Some(l) = self.lora.as_ref() {
            if let Some(e) = l.eff.get(name) {
                return e;
            }
        }
        self.wsl(name)
    }
    /// The buffer `name` names in the selected weight slot (or the shared
    /// activation set) - the raw base, never a folded effective weight.
    fn wsl(&self, name: &str) -> &DeviceBuffer {
        self.w[self.slot.get()].get(name).unwrap_or_else(|| &self.b[name])
    }
    fn up(&self, name: &str, data: &[f32]) {
        self.gpu.write_f32(self.wsl(name), data);
    }
    fn rd(&self, name: &str, n: usize) -> Vec<f32> {
        self.gpu.read(self.wsl(name), n)
    }

    /// `y = x·Wᵀ + b` through the shared training-shaped GEMM selection rule.
    fn linear(&self, s: &mut Vec<Step>, x: &str, w: &str, bias: &str, y: &str, m: usize, k: usize, n: usize) {
        let (kind, threads) = model::block::pick_gemm(m, n, K_MM, K_MM_REG, false);
        s.push(self.gpu.step(kind, &[self.g(x), self.g(w), self.g(y)], &[m as u32, k as u32, n as u32], threads));
        s.push(self.gpu.step(K_BIAS_ADD, &[self.g(y), self.g(bias)], &[m as u32, n as u32], (m * n) as u32));
    }

    /// The adjoint of [`Self::linear`]: bias row-sum, weight GEMM, input GEMM.
    #[allow(clippy::too_many_arguments)]
    fn linear_bwd(&self, s: &mut Vec<Step>, dy: &str, x: &str, w: &str, gw: &str, gb: &str, dx: &str, m: usize, k: usize, n: usize) {
        s.push(self.gpu.step(K_BIAS_GRAD, &[self.g(dy), self.g(gb)], &[m as u32, n as u32], n as u32));
        let (dwk, dwt) = model::block::pick_gemm(n, k, K_MM_DW, K_MM_DW_REG, false);
        s.push(self.gpu.step(dwk, &[self.g(dy), self.g(x), self.g(gw)], &[m as u32, k as u32, n as u32], dwt));
        let (dxk, dxt) = model::block::pick_gemm(m, k, K_MM_DX, K_MM_DX_REG, false);
        s.push(self.gpu.step(dxk, &[self.g(dy), self.g(w), self.g(dx)], &[m as u32, k as u32, n as u32, 0], dxt));
    }

    fn ln_fwd(&self, x: &str, gamma: &str, beta: &str, out: &str, rows: usize) -> Step {
        model::block::layernorm_fwd(&self.gpu, &self.ln, self.g(x), self.g(gamma), self.g(beta), self.g(out), self.dim as u32, rows as u32, self.eps)
    }

    /// RMSNorm over the FULL model width - `WanRMSNorm(dim)` runs before the
    /// head split, so a per-head inverse would be a different function.
    fn rms_fwd(&self, x: &str, w: &str, out: &str, rows: usize) -> Step {
        let (kind, threads) = model::block::rms_variant(&self.gpu, K_RMS, Some(K_RMS_ROWS), rows as u32, self.dim as u32);
        self.gpu.step(kind, &[self.g(x), self.g(w), self.g(out)], &[self.dim as u32, rows as u32, f(self.eps)], threads)
    }

    fn rms_bwd(&self, s: &mut Vec<Step>, x: &str, w: &str, dy: &str, dx: &str, inv: &str, gw: &str, rows: usize) {
        s.extend(model::block::rmsnorm_eps_bwd(
            &self.gpu,
            K_RMS_INV,
            K_RMS_DW,
            K_RMS_DX,
            self.g(x),
            self.g(w),
            self.g(dy),
            self.g(dx),
            self.g(inv),
            Some(self.g(gw)),
            self.dim as u32,
            rows as u32,
            self.eps,
        ));
    }

    /// LayerNorm backward at one site: the affine grads (`dgamma == dscale`,
    /// `dbeta == dshift`) and the input grad in one shot.
    #[allow(clippy::too_many_arguments)]
    fn ln_bwd(&self, s: &mut Vec<Step>, x: &str, gamma: &str, dy: &str, mean: &str, inv: &str, gg: &str, gb: &str, dx: &str, rows: usize) {
        let (d, r) = (self.dim as u32, rows as u32);
        s.push(model::block::ln_stats_fwd(&self.gpu, &self.ln, self.g(x), self.g(mean), self.g(inv), d, r, self.eps));
        s.push(self.gpu.step(K_LN_DG, &[self.g(dy), self.g(x), self.g(mean), self.g(inv), self.g(gg)], &[d, r], d));
        s.push(self.gpu.step(K_LN_DB, &[self.g(dy), self.g(gb)], &[d, r], d));
        s.push(model::block::layernorm_dx_bwd(&self.gpu, &self.ln, self.g(x), self.g(gamma), self.g(dy), self.g(dx), d, r, self.eps));
    }

    /// Bidirectional attention from `nq` query rows into `nk` key rows, with q,
    /// k and v each in their own contiguous `[rows, dim]` buffer.
    #[allow(clippy::too_many_arguments)]
    fn attn_fwd(&self, s: &mut Vec<Step>, q: &str, k: &str, v: &str, scores: &str, probs: &str, out: &str, nq: usize, nk: usize) {
        let (dim, nh, hd) = (self.dim as u32, self.nh as u32, self.hd as u32);
        let (nq, nk) = (nq as u32, nk as u32);
        s.push(self.gpu.step(K_SCORES, &[self.g(q), self.g(k), self.g(scores)], &[1, nh, nq, nk, hd, dim, dim, 0, 0], nh * nq * nk));
        if self.gpu.caps().workgroup_reductions {
            s.push(self.gpu.step(K_SOFTMAX_ROWS, &[self.g(scores), self.g(probs)], &[nh * nq, nk], nh * nq * 64));
        } else {
            s.push(self.gpu.step(K_SOFTMAX, &[self.g(scores), self.g(probs)], &[1, nh, nq, nk], nh * nq));
        }
        s.push(self.gpu.step(K_APPLY, &[self.g(probs), self.g(v), self.g(out)], &[1, nh, nq, nk, hd, dim, 0, dim], nh * nq * hd));
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_bwd(&self, s: &mut Vec<Step>, q: &str, k: &str, v: &str, probs: &str, dscores: &str, dout: &str, dq: &str, dk: &str, dv: &str, nq: usize, nk: usize) {
        let (dim, nh, hd) = (self.dim as u32, self.nh as u32, self.hd as u32);
        let (nq, nk) = (nq as u32, nk as u32);
        let pv = [1, nh, nq, nk, hd, dim, 0, dim];
        let pqk = [1, nh, nq, nk, hd, dim, dim, 0, 0];
        s.push(self.gpu.step(K_D_SCORES, &[self.g(dout), self.g(v), self.g(probs), self.g(dscores)], &pv, nh * nq));
        s.push(self.gpu.step(K_D_V, &[self.g(probs), self.g(dout), self.g(dv)], &pv, nh * nk * hd));
        s.push(self.gpu.step(K_D_Q, &[self.g(dscores), self.g(k), self.g(dq)], &pqk, nh * nq * hd));
        s.push(self.gpu.step(K_D_K, &[self.g(dscores), self.g(q), self.g(dk)], &pqk, nh * nk * hd));
    }

    fn rope(&self, x: &str, sin: &str, y: &str, t: usize) -> Step {
        let (nh, hd) = (self.nh as u32, self.hd as u32);
        let half = hd / 2;
        self.gpu.step(
            K_ROPE,
            &[self.g(x), self.g("cosb"), self.g(sin), self.g(y)],
            &[t as u32, nh, hd, half],
            (t as u32) * nh * half,
        )
    }

    /// Upload one block's weights and its folded modulation into the CURRENTLY
    /// SELECTED slot.
    fn upload_weights(&self, w: &BlockW<f32>, m: &Mods) {
        self.upload_frozen(w);
        self.upload_mods(m);
    }

    /// The part of a block that a LoRA run never changes: every matrix, bias
    /// and norm vector. Uploaded ONCE per slot for the whole run.
    fn upload_frozen(&self, w: &BlockW<f32>) {
        for (n, l) in [
            ("sq", &w.sq), ("sk", &w.sk), ("sv", &w.sv), ("so", &w.so),
            ("cq", &w.cq), ("ck", &w.ck), ("cv", &w.cv), ("co", &w.co),
        ] {
            self.up(&format!("{n}w"), &l.w);
            self.up(&format!("{n}b"), &l.b);
        }
        self.up("ff1w", &w.ff1.w);
        self.up("ff1b", &w.ff1.b);
        self.up("ff2w", &w.ff2.w);
        self.up("ff2b", &w.ff2.b);
        self.up("snq", &w.snq);
        self.up("snk", &w.snk);
        self.up("cnq", &w.cnq);
        self.up("cnk", &w.cnk);
        self.up("n3w", &w.norm3_w);
        self.up("n3b", &w.norm3_b);
    }

    /// The six modulation vectors, folded with the step's own timestep
    /// projection `e0` - the one part of a frozen block that moves every step.
    fn upload_mods(&self, m: &Mods) {
        self.up("ln1g", &m.ln1_g);
        self.up("ln1b", &m.ln1_b);
        self.up("gate1", &m.gate1);
        self.up("ln2g", &m.ln2_g);
        self.up("ln2b", &m.ln2_b);
        self.up("gate2", &m.gate2);
    }

    /// Upload the per-call inputs: the token slab, the shared embedded text
    /// context and the RoPE tables (plus the negated sine table the backward
    /// rotates by).
    fn upload_io(&self, x: &[f32], ctx: &[f32], cos: &[f32], sin: &[f32]) {
        self.up("xb", x);
        self.up("ctxb", ctx);
        self.up("cosb", cos);
        self.up("sinb", sin);
        let nsin: Vec<f32> = sin.iter().map(|&s| -s).collect();
        self.up("nsinb", &nsin);
    }

    /// Select slot `l` and upload `w`'s weights into it - what a forward sweep
    /// does once per block so the backward sweep can reuse them.
    pub fn load_slot(&self, l: usize, w: &BlockW<f32>, e0: &[f32]) {
        self.select_slot(l);
        self.upload_weights(w, &fold(&w.modulation, e0, self.dim));
    }

    /// Select slot `l` and upload only `w`'s FROZEN half - every matrix, bias
    /// and norm vector. A LoRA run does this once per block for the whole run;
    /// the modulation, which moves with the step's timestep, comes from
    /// [`Self::load_mods`].
    pub fn load_base_slot(&self, l: usize, w: &BlockW<f32>) {
        self.select_slot(l);
        self.upload_frozen(w);
    }

    /// Fold `w`'s modulation with `e0` and upload the six vectors into the
    /// selected slot.
    pub fn load_mods(&self, w: &BlockW<f32>, e0: &[f32]) {
        self.upload_mods(&fold(&w.modulation, e0, self.dim));
    }

    // ---- on-device LoRA ------------------------------------------------

    /// Allocate the effective-weight set, the adapter operands and the fold
    /// scratch for rank `r`. From here the recorded graphs read `W_eff`, which
    /// [`Self::upload_lora`] plus the fold dispatches assemble on-device from
    /// the resident base and the step's `(A, B)`.
    pub fn enable_lora(&mut self, r: usize) {
        let (dim, ffn) = (self.dim, self.ffn);
        let lora = {
            let gpu = &self.gpu;
            let new = |n: usize| gpu.storage((n as u64).max(1));
            let (mut eff, mut bs, mut at, mut a, mut ga, mut gb) =
                (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
            for name in LORA_TARGETS {
                let (out, inn) = target_dims(name, dim, ffn);
                eff.insert(name, new(out * inn));
                bs.insert(name, new(out * r));
                at.insert(name, new(inn * r));
                a.insert(name, new(r * inn));
                ga.insert(name, new(r * inn));
                gb.insert(name, new(out * r));
            }
            LoraDev { r, scale: std::cell::Cell::new(1.0), eff, bs, at, a, ga, gb, dsq: new(dim * dim), dff: new(ffn * dim) }
        };
        self.lora = Some(lora);
    }

    pub fn lora_rank(&self) -> Option<usize> {
        self.lora.as_ref().map(|l| l.r)
    }

    /// Bytes the LoRA working set costs on top of the resident base: one
    /// effective-weight slot plus the fold scratch (the adapter operands
    /// themselves are rank-sized and round to nothing).
    pub fn lora_bytes(d: Dims) -> u64 {
        let (dim, ffn) = (d.dim as u64, d.ffn as u64);
        (8 * dim * dim + 2 * ffn * dim + dim * dim + ffn * dim) * 4
    }

    /// Upload one block's ten adapter pairs, `(A [r×in], B [out×r])` in
    /// [`LORA_TARGETS`] order. `with_a` also uploads `A` itself, which only the
    /// `dB` projection reads - a forward-only visit skips it.
    pub fn upload_lora(&self, ab: &[(&[f32], &[f32])], scale: f32, with_a: bool) {
        let l = self.lora.as_ref().expect("upload_lora: call enable_lora first");
        assert_eq!(ab.len(), LORA_TARGETS.len(), "upload_lora: one (A, B) per target");
        l.scale.set(scale);
        for (name, (a, b)) in LORA_TARGETS.iter().zip(ab) {
            let (out, inn) = target_dims(name, self.dim, self.ffn);
            assert_eq!(a.len(), l.r * inn, "upload_lora: {name} A is [{}x{inn}]", l.r);
            assert_eq!(b.len(), out * l.r, "upload_lora: {name} B is [{out}x{}]", l.r);
            // `scale·B` rather than B: the host reference scales B's entry
            // before it multiplies A's row, and both device GEMMs that read
            // this operand want exactly that product.
            let bs: Vec<f32> = b.iter().map(|&x| x * scale).collect();
            self.gpu.write_f32(&l.bs[name], &bs);
            let mut at = vec![0f32; inn * l.r];
            for (i, row) in at.chunks_exact_mut(l.r).enumerate() {
                for (k, slot) in row.iter_mut().enumerate() {
                    *slot = a[k * inn + i];
                }
            }
            self.gpu.write_f32(&l.at[name], &at);
            if with_a {
                self.gpu.write_f32(&l.a[name], a);
            }
        }
    }

    /// `W_eff = base + (scale·B)·A` for all ten targets: the low-rank GEMM
    /// (`matmul` with `Aᵀ` as its `[in, r]` right operand) into the shared
    /// scratch, then an elementwise add onto the resident base.
    fn fold_steps(&self) -> Vec<Step> {
        let Some(l) = self.lora.as_ref() else { return Vec::new() };
        let mut s = Vec::new();
        for name in LORA_TARGETS {
            let (out, inn) = target_dims(name, self.dim, self.ffn);
            let d = if name.starts_with("ff") { &l.dff } else { &l.dsq };
            let (kind, threads) = model::block::pick_gemm(out, inn, K_MM, K_MM_REG, false);
            s.push(self.gpu.step(kind, &[&l.bs[name], &l.at[name], d], &[out as u32, l.r as u32, inn as u32], threads));
            let n = (out * inn) as u32;
            s.push(self.gpu.step(K_ADD2, &[self.wsl(name), d, &l.eff[name]], &[n], n));
        }
        s
    }

    /// `dA = (scale·B)ᵀ·dW` and `dB/scale = dW·Aᵀ` for all ten targets, read
    /// straight out of the weight-grad buffers the backward just filled - the
    /// full `dW` never leaves the device.
    fn project_steps(&self) -> Vec<Step> {
        let Some(l) = self.lora.as_ref() else { return Vec::new() };
        let mut s = Vec::new();
        for name in LORA_TARGETS {
            let (out, inn) = target_dims(name, self.dim, self.ffn);
            let dw = self.wsl(&format!("g_{name}"));
            let (kb, tb) = model::block::pick_gemm(out, l.r, K_MM, K_MM_REG, false);
            s.push(self.gpu.step(kb, &[dw, &l.a[name], &l.gb[name]], &[out as u32, inn as u32, l.r as u32], tb));
            let (ka, ta) = model::block::pick_gemm(l.r, inn, K_MM_DW, K_MM_DW_REG, false);
            s.push(self.gpu.step(ka, &[&l.bs[name], dw, &l.ga[name]], &[out as u32, inn as u32, l.r as u32], ta));
        }
        s
    }

    /// Forward step list for `t` latent tokens (writes `outb`).
    fn fwd_steps(&self, t: usize) -> Vec<Step> {
        let (dim, ffn, te) = (self.dim, self.ffn, self.te);
        let mut s = Vec::new();
        // --- self-attention ---
        s.push(self.ln_fwd("xb", "ln1g", "ln1b", "n1", t));
        self.linear(&mut s, "n1", "sqw", "sqb", "q", t, dim, dim);
        self.linear(&mut s, "n1", "skw", "skb", "k", t, dim, dim);
        self.linear(&mut s, "n1", "svw", "svb", "v", t, dim, dim);
        s.push(self.rms_fwd("q", "snq", "qn", t));
        s.push(self.rms_fwd("k", "snk", "kn", t));
        s.push(self.rope("qn", "sinb", "qr", t));
        s.push(self.rope("kn", "sinb", "kr", t));
        self.attn_fwd(&mut s, "qr", "kr", "v", "sscores", "sprobs", "actx", t, t);
        self.linear(&mut s, "actx", "sow", "sob", "ao", t, dim, dim);
        s.push(self.gpu.step(K_GATE, &[self.g("xb"), self.g("gate1"), self.g("ao"), self.g("x1")], &[t as u32, dim as u32, t as u32], (t * dim) as u32));

        // --- text cross-attention (ungated residual) ---
        s.push(self.ln_fwd("x1", "n3w", "n3b", "n3", t));
        self.linear(&mut s, "n3", "cqw", "cqb", "xq", t, dim, dim);
        self.linear(&mut s, "ctxb", "ckw", "ckb", "xk", te, dim, dim);
        self.linear(&mut s, "ctxb", "cvw", "cvb", "xv", te, dim, dim);
        s.push(self.rms_fwd("xq", "cnq", "xqn", t));
        s.push(self.rms_fwd("xk", "cnk", "xkn", te));
        self.attn_fwd(&mut s, "xqn", "xkn", "xv", "xscores", "xprobs", "xctx", t, te);
        self.linear(&mut s, "xctx", "cow", "cob", "xo", t, dim, dim);
        s.push(self.gpu.step(K_ADD2, &[self.g("x1"), self.g("xo"), self.g("x2")], &[(t * dim) as u32], (t * dim) as u32));

        // --- FFN ---
        s.push(self.ln_fwd("x2", "ln2g", "ln2b", "n2", t));
        self.linear(&mut s, "n2", "ff1w", "ff1b", "h1", t, dim, ffn);
        s.push(self.gpu.step(K_GELU, &[self.g("h1"), self.g("hg")], &[(t * ffn) as u32], (t * ffn) as u32));
        self.linear(&mut s, "hg", "ff2w", "ff2b", "ffb", t, ffn, dim);
        s.push(self.gpu.step(K_GATE, &[self.g("x2"), self.g("gate2"), self.g("ffb"), self.g("outb")], &[t as u32, dim as u32, t as u32], (t * dim) as u32));
        s
    }

    /// Backward step list for `t` latent tokens (reads `dout`, writes the grad
    /// buffers, `d_x` and `d_ctx`).
    fn bwd_steps(&self, t: usize) -> Vec<Step> {
        let (dim, ffn, te) = (self.dim, self.ffn, self.te);
        let (td, ted) = ((t * dim) as u32, (te * dim) as u32);
        let mut s = Vec::new();
        let gate = |s: &mut Vec<Step>, dy: &str, h: &str, g: &str, dg: &str, dh: &str| {
            let p = [t as u32, dim as u32, t as u32];
            s.push(self.gpu.step(K_GATE_DG, &[self.g(dy), self.g(h), self.g(dg)], &p, dim as u32));
            s.push(self.gpu.step(K_GATE_DH, &[self.g(dy), self.g(g), self.g(dh)], &p, td));
        };
        let add = |s: &mut Vec<Step>, a: &str, b: &str, o: &str, n: u32| {
            s.push(self.gpu.step(K_ADD2, &[self.g(a), self.g(b), self.g(o)], &[n], n));
        };

        // out = x2 + gate2 ⊙ ff  (dx2 of the gate is the identity)
        gate(&mut s, "dout", "ffb", "gate2", "g_gate2", "d_ff");
        self.linear_bwd(&mut s, "d_ff", "hg", "ff2w", "g_ff2w", "g_ff2b", "d_hg", t, ffn, dim);
        s.push(self.gpu.step(K_GELU_BWD, &[self.g("h1"), self.g("d_hg"), self.g("d_h1")], &[(t * ffn) as u32], (t * ffn) as u32));
        self.linear_bwd(&mut s, "d_h1", "n2", "ff1w", "g_ff1w", "g_ff1b", "d_n2", t, dim, ffn);
        self.ln_bwd(&mut s, "x2", "ln2g", "d_n2", "m2", "i2", "g_ln2g", "g_ln2b", "d_x2a", t);
        add(&mut s, "dout", "d_x2a", "d_x2", td);

        // x2 = x1 + xo (ungated)
        self.linear_bwd(&mut s, "d_x2", "xctx", "cow", "g_cow", "g_cob", "d_xctx", t, dim, dim);
        self.attn_bwd(&mut s, "xqn", "xkn", "xv", "xprobs", "d_sx", "d_xctx", "d_xqn", "d_xkn", "d_xv", t, te);
        self.rms_bwd(&mut s, "xq", "cnq", "d_xqn", "d_xq", "ixq", "g_cnq", t);
        self.rms_bwd(&mut s, "xk", "cnk", "d_xkn", "d_xk", "ixk", "g_cnk", te);
        self.linear_bwd(&mut s, "d_xq", "n3", "cqw", "g_cqw", "g_cqb", "d_n3", t, dim, dim);
        // k and v both read the SHARED text context: their dctx contributions add.
        self.linear_bwd(&mut s, "d_xk", "ctxb", "ckw", "g_ckw", "g_ckb", "d_ctxk", te, dim, dim);
        self.linear_bwd(&mut s, "d_xv", "ctxb", "cvw", "g_cvw", "g_cvb", "d_ctxv", te, dim, dim);
        add(&mut s, "d_ctxk", "d_ctxv", "d_ctx", ted);
        self.ln_bwd(&mut s, "x1", "n3w", "d_n3", "m3", "i3", "g_n3w", "g_n3b", "d_x1a", t);
        add(&mut s, "d_x2", "d_x1a", "d_x1", td);

        // x1 = x + gate1 ⊙ ao
        gate(&mut s, "d_x1", "ao", "gate1", "g_gate1", "d_ao");
        self.linear_bwd(&mut s, "d_ao", "actx", "sow", "g_sow", "g_sob", "d_actx", t, dim, dim);
        self.attn_bwd(&mut s, "qr", "kr", "v", "sprobs", "d_ss", "d_actx", "d_qr", "d_kr", "d_v", t, t);
        // RoPE backward is the forward kernel against a negated sine table.
        s.push(self.rope("d_qr", "nsinb", "d_qn", t));
        s.push(self.rope("d_kr", "nsinb", "d_kn", t));
        self.rms_bwd(&mut s, "q", "snq", "d_qn", "d_q", "iq", "g_snq", t);
        self.rms_bwd(&mut s, "k", "snk", "d_kn", "d_k", "ik", "g_snk", t);
        self.linear_bwd(&mut s, "d_q", "n1", "sqw", "g_sqw", "g_sqb", "d_n1q", t, dim, dim);
        self.linear_bwd(&mut s, "d_k", "n1", "skw", "g_skw", "g_skb", "d_n1k", t, dim, dim);
        self.linear_bwd(&mut s, "d_v", "n1", "svw", "g_svw", "g_svb", "d_n1v", t, dim, dim);
        add(&mut s, "d_n1q", "d_n1k", "d_n1t", td);
        add(&mut s, "d_n1t", "d_n1v", "d_n1", td);
        self.ln_bwd(&mut s, "xb", "ln1g", "d_n1", "m1", "i1", "g_ln1g", "g_ln1b", "d_xa", t);
        add(&mut s, "d_x1", "d_xa", "d_x", td);
        s
    }

    /// Forward one block, returning its output `[t·dim]`.
    ///
    /// `e0` is the timestep projection `[6·dim]` NOT yet summed with the
    /// block's own `modulation`; `ctx` is the embedded text `[text_len·dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, d: Dims, w: &BlockW<f32>, x: &[f32], e0: &[f32], ctx: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
        self.upload_weights(w, &fold(&w.modulation, e0, self.dim));
        self.forward_loaded(d, x, ctx, cos, sin)
    }

    /// [`Self::forward`] against the weights already in the selected slot.
    pub fn forward_loaded(&self, d: Dims, x: &[f32], ctx: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
        self.upload_io(x, ctx, cos, sin);
        let mut steps = self.fold_steps();
        steps.extend(self.fwd_steps(d.t));
        self.gpu.submit(&[], &steps);
        self.gpu.poll_wait();
        self.rd("outb", d.t * self.dim)
    }

    /// Backward one block: recompute the forward from `x`, then backprop
    /// `dout`, returning every weight grad plus `dx` and `dctx`.
    ///
    /// [`BlockGrads::modulation`] is one vector used twice - it is both
    /// `d(blocks.{l}.modulation)` and this block's contribution to `d e0`,
    /// because the fold's operand is their sum.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(&self, d: Dims, w: &BlockW<f32>, x: &[f32], e0: &[f32], ctx: &[f32], cos: &[f32], sin: &[f32], dout: &[f32]) -> BlockGrads<f32> {
        self.upload_weights(w, &fold(&w.modulation, e0, self.dim));
        self.backward_loaded(d, x, ctx, cos, sin, dout)
    }

    /// [`Self::backward`] against the weights already in the selected slot.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_loaded(&self, d: Dims, x: &[f32], ctx: &[f32], cos: &[f32], sin: &[f32], dout: &[f32]) -> BlockGrads<f32> {
        let (dim, ffn, te, t) = (self.dim, self.ffn, self.te, d.t);
        self.upload_io(x, ctx, cos, sin);
        self.up("dout", dout);
        // `matmul_dw`, `bias_grad`, `rmsnorm_dw` and the LayerNorm affine grads
        // all ACCUMULATE, so their buffers start at zero. That zeroing is a
        // device-side `clear_buffer` (the `clears` list `submit` takes) rather
        // than a host upload of zeros: at 1.3B widths the weight-grad set is
        // 185 MB a block, and uploading it 30 times a step is gigabytes of
        // host-to-device traffic for a value the device can write itself.
        let clears: Vec<&DeviceBuffer> = ACCUMULATING_GRADS.iter().map(|n| self.wsl(n)).collect();

        let mut steps = self.fold_steps();
        steps.extend(self.fwd_steps(t));
        steps.extend(self.bwd_steps(t));
        self.gpu.submit(&clears, &steps);
        self.gpu.poll_wait();

        let lin = |wn: &str, out: usize, inn: usize| Lin {
            w: self.rd(&format!("g_{wn}w"), out * inn),
            b: self.rd(&format!("g_{wn}b"), out),
        };
        let mut modulation = vec![0f32; 6 * dim];
        for (i, n) in ["g_ln1b", "g_ln1g", "g_gate1", "g_ln2b", "g_ln2g", "g_gate2"].into_iter().enumerate() {
            modulation[i * dim..(i + 1) * dim].copy_from_slice(&self.rd(n, dim));
        }
        BlockGrads {
            modulation,
            sq: lin("sq", dim, dim),
            sk: lin("sk", dim, dim),
            sv: lin("sv", dim, dim),
            so: lin("so", dim, dim),
            snq: self.rd("g_snq", dim),
            snk: self.rd("g_snk", dim),
            cq: lin("cq", dim, dim),
            ck: lin("ck", dim, dim),
            cv: lin("cv", dim, dim),
            co: lin("co", dim, dim),
            cnq: self.rd("g_cnq", dim),
            cnk: self.rd("g_cnk", dim),
            norm3_w: self.rd("g_n3w", dim),
            norm3_b: self.rd("g_n3b", dim),
            ff1: lin("ff1", ffn, dim),
            ff2: lin("ff2", dim, ffn),
            dx: self.rd("d_x", t * dim),
            dctx: self.rd("d_ctx", te * dim),
        }
    }

    /// [`Self::backward_loaded`] for a LoRA run: the same forward-recompute and
    /// backward, then the on-device projection of each target's `dW` onto its
    /// adapter grads.
    ///
    /// Returns `(dx, [(dA, dB); 10])`. The ten full `dW` - 185 MB a block - stay
    /// on the device; only `dx` and the rank-sized adapter grads come back, and
    /// the frozen tensors' grads are never read at all because nothing consumes
    /// them when only `(A, B)` train.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_lora_loaded(
        &self,
        d: Dims,
        x: &[f32],
        ctx: &[f32],
        cos: &[f32],
        sin: &[f32],
        dout: &[f32],
    ) -> (Vec<f32>, AdapterGrads) {
        let l = self.lora.as_ref().expect("backward_lora_loaded: call enable_lora first");
        let t = d.t;
        self.upload_io(x, ctx, cos, sin);
        self.up("dout", dout);
        let mut clears: Vec<&DeviceBuffer> = ACCUMULATING_GRADS.iter().map(|n| self.wsl(n)).collect();
        // `dA` is a `matmul_dw` output, so it accumulates like the weight grads
        // above; `dB` is a plain `matmul` and overwrites.
        clears.extend(LORA_TARGETS.iter().map(|n| &l.ga[n]));

        let mut steps = self.fold_steps();
        steps.extend(self.fwd_steps(t));
        steps.extend(self.bwd_steps(t));
        steps.extend(self.project_steps());
        self.gpu.submit(&clears, &steps);
        self.gpu.poll_wait();

        let scale = l.scale.get();
        let pairs = LORA_TARGETS
            .iter()
            .map(|name| {
                let (out, inn) = target_dims(name, self.dim, self.ffn);
                let da = self.gpu.read(&l.ga[name], l.r * inn);
                // The `dB` GEMM contracts the row of `dW` against the row of
                // `A` and stops there, so the `α/r` the host reference applies
                // to that dot product lands here.
                let db: Vec<f32> = self.gpu.read(&l.gb[name], out * l.r).into_iter().map(|v| v * scale).collect();
                (da, db)
            })
            .collect();
        (self.rd("d_x", t * self.dim), pairs)
    }
}

/// Build a fresh engine and run one block's backward - the standalone form the
/// device/host parity tests drive.
#[allow(clippy::too_many_arguments)]
pub fn block_backward_device(d: Dims, w: &BlockW<f32>, x: &[f32], e0: &[f32], ctx: &[f32], cos: &[f32], sin: &[f32], dout: &[f32], device: Option<&str>) -> BlockGrads<f32> {
    let eng = BlockDev::on_device(d, d.t, device);
    eng.backward(d, w, x, e0, ctx, cos, sin, dout)
}
