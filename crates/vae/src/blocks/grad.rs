// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reverse mode for the shared conv-autoencoder blocks.
//!
//! A [`super::Builder`] put in train mode ([`super::Builder::set_train`])
//! records an [`Op`](super::Op) tape alongside its forward step list. This
//! module walks that tape **backwards** once, at graph-construction time, and
//! emits a second `Vec<Step>` — so a training step is two submits (forward,
//! backward) with no per-step graph rebuilding, exactly like the forward.
//!
//! Three rules the whole file obeys, and the reason for each:
//!
//! 1. **Gather, never scatter.** Every kernel dispatched here writes one
//!    element of its OUTPUT per invocation (`conv2d_dx` gathers the output
//!    taps that touched an input pixel; `upsample2_dx` sums a 2x2 block;
//!    `emb_bwd` — used by the caller, not here — loops the rows). There are no
//!    atomics anywhere in the reverse.
//! 2. **A parameter gradient ACCUMULATES; an activation gradient is
//!    ASSIGNED into a temp and then folded in with `axpy`.** `conv2d_dw`,
//!    `bias_grad`, `gn_dgamma` and `gn_dbeta` all read-modify-write, so their
//!    buffers must be zeroed exactly once per optimizer step by the model's
//!    `zero_grads` — never in this submit's clear list, which would drop every
//!    contribution before the last. `conv2d_dx`, `gn_dx`, `silu_bwd`,
//!    `upsample2_dx` and the transposes all OVERWRITE, and a block input can
//!    have two consumers (a resnet's shortcut, an attention's residual), so the
//!    uniform `assign into a temp, axpy into d[x]` shape is what makes fan-out
//!    correct. The `d_*` activation buffers therefore DO belong in the reverse
//!    submit's clear list, and [`Reverse::clears`] is exactly that list.
//! 3. **The adjoint reads the stage's INPUT, not its output.** `conv2d_dw`
//!    binds the conv's input; `silu_bwd` binds the PRE-activation; `gn_dsum` /
//!    `gn_dgamma` bind the GroupNorm's input and its retained `stats`. That is
//!    why train mode disables the activation pool: the forward buffer *is* the
//!    cache.

use super::{Op, K_IM2COL_AT, K_NCHW_NLC, K_NLC_NCHW};
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

// Offsets within `super::BWD_KERNELS`.
const B_CONV_DX: usize = 0;
const B_CONV_DW: usize = 1;
const B_BIAS_GRAD: usize = 2;
const B_SILU_BWD: usize = 3;
const B_SCALE_CHAN: usize = 4;
const B_GN_DX: usize = 5;
const B_GN_DGB_PART: usize = 6;
const B_GN_DGB2: usize = 7;
const B_UPSAMPLE2_DX: usize = 8;
const B_AXPY: usize = 9;
const B_ATTN_DSCORES: usize = 10;
const B_ATTN_DV: usize = 11;
const B_ATTN_DQ: usize = 12;
const B_ATTN_DK: usize = 13;
const B_MATMUL_DX_REG: usize = 14;
const B_COL2IM: usize = 15;
const B_MATMUL_DW_REG: usize = 16;
const B_GN_DSUM_PART: usize = 17;
const B_GN_DSUM2: usize = 18;
const B_MATMUL_DW_SPLITK: usize = 19;
const B_DW_SPLITK_REDUCE: usize = 20;
// The transformer half's adjoints (see `super::BWD_KERNELS`'s own note on why
// `mul` is not among them).
const B_MATMUL_DX: usize = 21;
const B_MATMUL_DW: usize = 22;
const B_GELU_ERF_BWD: usize = 23;
const B_ADD_CHAN_DV: usize = 24;
const B_LAYERNORM_DX: usize = 25;
const B_LAYERNORM_DGAMMA: usize = 26;
const B_LAYERNORM_DBETA: usize = 27;
const B_CONCAT_SPLIT: usize = 28;
const B_XDSCORES: usize = 29;
const B_XDQ: usize = 30;
const B_XDK_ACC: usize = 31;
const B_XDV_ACC: usize = 32;
const B_LN_STATS: usize = 33;

/// Where the caller placed [`super::BWD_KERNELS`] in its kernel set.
#[derive(Clone, Copy)]
pub struct BwdIds {
    base: usize,
}

impl BwdIds {
    /// `base` is the slot index of `BWD_KERNELS[0]`.
    pub const fn at(base: usize) -> BwdIds {
        BwdIds { base }
    }
    fn k(self, off: usize) -> usize {
        self.base + off
    }
    /// This set's `axpy` slot.
    ///
    /// Exposed because a caller that stitches its own graph onto these blocks
    /// (`vqgan`'s quantiser seam) needs the same accumulate primitive, and it
    /// must NOT register a second copy under the same kernel name: the CPU
    /// backend's JIT rejects that outright (`DuplicateDefinition("axpy")`),
    /// which is how this rule was found.
    pub const fn axpy(self) -> usize {
        self.base + B_AXPY
    }
    /// The bidirectional-attention ids for `model::block`, wiring the forward
    /// trio (fixed slots inside [`super::KERNELS`]) to the four backward slots.
    fn bidir(self) -> model::block::BidirIds {
        model::block::BidirIds {
            scores: super::K_ATTN_SCORES,
            softmax: super::K_ATTN_SOFTMAX,
            apply: super::K_ATTN_APPLY,
            dscores: self.k(B_ATTN_DSCORES),
            dv: self.k(B_ATTN_DV),
            dq: self.k(B_ATTN_DQ),
            dk: self.k(B_ATTN_DK),
        }
    }
}

/// A recorded forward tape plus the weight buffers it uploaded.
#[derive(Clone)]
pub struct Trace {
    ops: Vec<Op>,
    /// The caller's forward `mul` slot, carried from `Builder`'s
    /// [`super::XformerIds`]. `Op::Mul`'s adjoint is two more products, and
    /// re-registering `mul` in the BACKWARD set would give the CPU JIT two
    /// definitions of one kernel name - see `super::BWD_KERNELS`'s note.
    xf_mul: Option<usize>,
    /// Tensor name -> length, in first-use order (the parameter list).
    order: Vec<(String, u64)>,
    w: HashMap<String, DeviceBuffer>,
}

impl Trace {
    pub(super) fn new(
        ops: Vec<Op>,
        order: Vec<(String, u64)>,
        w: &HashMap<String, DeviceBuffer>,
        xf_mul: Option<usize>,
    ) -> Trace {
        Trace { ops, order, w: w.clone(), xf_mul }
    }

    /// Every trainable tensor this graph reads, `(name, length in floats)`, in
    /// first-use order. GroupNorm appears once as the fused `{prefix}.gb[2C]`
    /// (`gn_apply` reads that fused layout and `gn_dgamma`/`gn_dbeta` write the
    /// matching fused `dgb[2C]`); an attention's q/k/v appear once as the fused
    /// `{prefix}.qkv.w[3C,C,1,1]` + `.qkv.b[3C]`.
    pub fn params(&self) -> &[(String, u64)] {
        &self.order
    }

    /// The device buffer holding a tensor.
    pub fn weight(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("vae::blocks::grad: no tensor {name}"))
    }

    /// One zeroed gradient buffer per tensor in [`Trace::params`].
    pub fn alloc_grads(&self, gpu: &Gpu) -> Grads {
        let g = self.order.iter().map(|(n, len)| (n.clone(), gpu.storage(*len))).collect();
        Grads { g }
    }

    /// Record the reverse step list. `(out, d_out)` seeds the walk with the
    /// gradient of the tape's final buffer; the returned [`Reverse`] exposes
    /// `d(&buf)` for any activation, in particular the tape's input.
    pub fn backward(
        &self,
        gpu: &Gpu,
        ids: BwdIds,
        grads: &Grads,
        out: &DeviceBuffer,
        d_out: &DeviceBuffer,
    ) -> Reverse {
        let mut r = Rev {
            gpu,
            ids,
            steps: Vec::new(),
            d: HashMap::new(),
            clears: Vec::new(),
            pool: HashMap::new(),
        };
        r.d.insert(key(out), d_out.clone());
        for op in self.ops.iter().rev() {
            self.emit(&mut r, grads, op);
        }
        Reverse { steps: r.steps, clears: r.clears, d: r.d }
    }

    fn emit(&self, r: &mut Rev, grads: &Grads, op: &Op) {
        match op {
            Op::Conv { w, b, cin, cout, k, stride, pad, h, w_in, ho, wo, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let p = [1, *cin, *h, *w_in, *cout, *k, *stride, *pad, *ho, *wo];
                let (hw, n_in) = (ho * wo, cin * h * w_in);
                let cinkk = cin * k * k;
                let lowered = r.gpu.caps().workgroup_reductions && *cout >= super::GEMM_CONV_BWD_MIN_COUT;

                // dW (accumulates) then db, both from the conv's own input.
                //
                // `conv2d_dw` reduces over EVERY output position (Ho*Wo) per
                // weight element on one lane, which profiling put at the
                // largest single stage of a VQGAN training step once
                // `conv2d_dx` was lowered. The same lowering applies:
                //
                //   col[HW, CinKK] = im2col(x)                     (im2col_at)
                //   dW[Cout,CinKK] += dY[HW,Cout]^T . col[HW,CinKK] (matmul_dw_reg)
                //
                // and `matmul_dw_reg` ACCUMULATES into its output, which is
                // exactly the "a parameter gradient accumulates" rule this file
                // opens with — no extra axpy, no clear.
                // ONE `nchw_nlc` of dY per conv. All three consumers want the
                // same [HW, Cout] view — dW's GEMM, `bias_grad`, and dX's GEMM —
                // and transposing per consumer showed up immediately as
                // `nchw_nlc` doubling to 182 calls in the profile.
                let t = r.tmp((cout * hw) as u64);
                r.push(r.gpu.step(K_NCHW_NLC, &[&dy, &t], &[cout * hw, *cout, hw], cout * hw));

                if lowered {
                    let col = r.tmp((hw * cinkk) as u64);
                    r.push(r.gpu.step(
                        K_IM2COL_AT,
                        &[x, &col],
                        &[*cin, *h, *w_in, *k, *stride, *pad, *ho, *wo, cinkk, 0, hw],
                        hw * cinkk,
                    ));
                    // dW's tile grid is ceil(Cout/128)*ceil(CinKK/128) — it does
                    // NOT grow with the contraction length, so a wide-shallow
                    // conv launches a handful of workgroups and idles the card.
                    // Split the contraction to reach `DW_SPLITK_TARGET_WGS`.
                    let tiles = cout.div_ceil(128) * cinkk.div_ceil(128);
                    let slices = super::DW_SPLITK_TARGET_WGS.div_ceil(tiles).max(1).min(hw.div_ceil(8));
                    if slices > 1 {
                        let rc = (*cout as u64) * cinkk as u64;
                        let part = r.tmp(rc * slices as u64);
                        r.push(r.gpu.step(
                            r.ids.k(B_MATMUL_DW_SPLITK),
                            &[&t, &col, &part],
                            &[hw, cinkk, *cout, slices],
                            slices * tiles * 256,
                        ));
                        r.push(r.gpu.step(
                            r.ids.k(B_DW_SPLITK_REDUCE),
                            &[&part, grads.g(w)],
                            // acc = 1: a parameter gradient ACCUMULATES (a
                            // weight used twice gets two contributions).
                            &[cout * cinkk, slices, 1],
                            (cout * cinkk).div_ceil(64) * 64,
                        ));
                        r.give(rc * slices as u64, part);
                    } else {
                        r.push(r.gpu.step(
                            r.ids.k(B_MATMUL_DW_REG),
                            &[&t, &col, grads.g(w)],
                            &[hw, cinkk, *cout],
                            tiles * 256,
                        ));
                    }
                    r.give((hw * cinkk) as u64, col);
                } else {
                    r.push(r.gpu.step(r.ids.k(B_CONV_DW), &[&dy, x, grads.g(w)], &p, cout * cin * k * k));
                }
                // `bias_grad` reduces a [rows, features] buffer down its rows,
                // but `dy` is NCHW = feature-major. One `nchw_nlc` puts the
                // channels last; the alternative would be a new NCHW bias
                // reduction kernel, and this composition needs neither.
                r.push(r.gpu.step(r.ids.k(B_BIAS_GRAD), &[&t, grads.g(b)], &[hw, *cout], *cout));

                // dX (assigns) -> accumulate. Two paths:
                //
                //  * LOWERED — `dcol[HW,CinKK] = dY[HW,Cout] . W[Cout,CinKK]`
                //    (`matmul_dx_reg`, register-tiled) then `col2im`, which sums
                //    only the K*K taps per input pixel. `conv2d_dx` instead
                //    reduces over Cout*K*K on ONE lane, and the backward profile
                //    put it at a large share of a VQGAN training step, several
                //    times the cost of the forward conv it mirrors. The lowering
                //    measured faster at every shape in `vqgan_bench convbwd`.
                //  * DIRECT — below `GEMM_CONV_BWD_MIN_COUT`, and on any device
                //    without workgroup reductions: `matmul_dx_reg` carries
                //    barriers the CPU JIT cannot compile, so this branches on the
                //    QUERIED capability, never on an assumption (lessons #5).
                //
                // The NLC transpose the GEMM needs is `t`, which bias_grad has
                // already built — so the fast path costs one transpose less than
                // it looks.
                let dx = r.tmp(n_in as u64);
                if lowered {
                    let dcol = r.tmp((hw * cinkk) as u64);
                    r.push(r.gpu.step(
                        r.ids.k(B_MATMUL_DX_REG),
                        &[&t, self.weight(w), &dcol],
                        &[hw, cinkk, *cout, 0],
                        hw.div_ceil(128) * cinkk.div_ceil(128) * 256,
                    ));
                    r.push(r.gpu.step(
                        r.ids.k(B_COL2IM),
                        &[&dcol, &dx],
                        &[1, *cin, *h, *w_in, *k, *stride, *pad, *ho, *wo, cinkk],
                        n_in,
                    ));
                    r.give((hw * cinkk) as u64, dcol);
                } else {
                    r.push(r.gpu.step(r.ids.k(B_CONV_DX), &[&dy, self.weight(w), &dx], &p, n_in));
                }
                r.give((cout * hw) as u64, t);
                r.acc(x, n_in as u64, &dx, 1.0);
                r.give(n_in as u64, dx);
            }
            Op::Gn { gb, c, h, w, g, x, stats, y } => {
                let Some(dy) = r.get(y) else { return };
                let n = c * h * w;
                let p = [1, *c, *h, *w, *g];
                let gbuf = self.weight(gb);
                // dyg = dy * gamma, the shared input of gn_dsum and gn_dx.
                let dyg = r.tmp(n as u64);
                r.push(r.gpu.step(r.ids.k(B_SCALE_CHAN), &[&dy, gbuf, &dyg], &[n, *c, h * w], n));
                let sums = r.tmp(4 * *g as u64);
                // Two-stage, barrier-free. `gn_dsum` is ONE invocation per
                // group (32 lanes walking (C/G)*H*W elements each), measured at
                // well under one percent of the card's bandwidth roof and a
                // quarter of the whole backward. Stage 1 splits each group
                // across `GN_P` partials (coalesced, strided), stage 2 folds
                // them. No workgroupBarrier,
                // so this needs no capability branch and `backend-cpu` gets it
                // too — the same shape as the forward's `gn_part`/`gn_stats2`.
                let part = r.tmp(2 * *g as u64 * super::GN_P as u64);
                let pp = [1, *c, *h, *w, *g, super::GN_P];
                r.push(r.gpu.step(r.ids.k(B_GN_DSUM_PART), &[x, &dyg, stats, &part], &pp, g * super::GN_P));
                r.push(r.gpu.step(r.ids.k(B_GN_DSUM2), &[&part, stats, &sums], &pp, *g));
                r.give(2 * *g as u64 * super::GN_P as u64, part);
                // dgamma -> dgb[0..C], dbeta -> dgb[C..2C]: disjoint writes into
                // the same fused buffer, both accumulating.
                // Two-stage, and ONE pass over `dy` for both affine gradients.
                // `gn_dgamma`/`gn_dbeta` were a lane per channel walking N*H*W
                // each, together a couple of percent of the bandwidth roof, and
                // each read the whole of `dy` separately.
                let dgb_part = r.tmp(2 * *c as u64 * super::GN_P as u64);
                let pg = [1, *c, *h, *w, *g, super::GN_P];
                r.push(r.gpu.step(
                    r.ids.k(B_GN_DGB_PART),
                    &[x, &dy, stats, &dgb_part],
                    &pg,
                    c * super::GN_P,
                ));
                r.push(r.gpu.step(r.ids.k(B_GN_DGB2), &[&dgb_part, grads.g(gb)], &pg, *c));
                r.give(2 * *c as u64 * super::GN_P as u64, dgb_part);
                let dx = r.tmp(n as u64);
                r.push(r.gpu.step(r.ids.k(B_GN_DX), &[x, &dyg, &sums, &dx], &p, n));
                r.acc(x, n as u64, &dx, 1.0);
                r.give(n as u64, dx);
                r.give(4 * *g as u64, sums);
                r.give(n as u64, dyg);
            }
            Op::Silu { n, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let dx = r.tmp(*n as u64);
                r.push(r.gpu.step(r.ids.k(B_SILU_BWD), &[x, &dy, &dx], &[*n], *n));
                r.acc(x, *n as u64, &dx, 1.0);
                r.give(*n as u64, dx);
            }
            Op::Add2 { n, a, b, y } => {
                let Some(dy) = r.get(y) else { return };
                r.acc(a, *n as u64, &dy, 1.0);
                r.acc(b, *n as u64, &dy, 1.0);
            }
            Op::Up2 { c, h, w, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let n = c * h * w;
                let dx = r.tmp(n as u64);
                r.push(r.gpu.step(r.ids.k(B_UPSAMPLE2_DX), &[&dy, &dx], &[1, *c, *h, *w], n));
                r.acc(x, n as u64, &dx, 1.0);
                r.give(n as u64, dx);
            }
            // A layout permutation is its own transpose's adjoint.
            Op::NchwNlc { c, hw, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let n = c * hw;
                let dx = r.tmp(n as u64);
                r.push(r.gpu.step(K_NLC_NCHW, &[&dy, &dx], &[n, *c, *hw], n));
                r.acc(x, n as u64, &dx, 1.0);
                r.give(n as u64, dx);
            }
            Op::NlcNchw { c, hw, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let n = c * hw;
                let dx = r.tmp(n as u64);
                r.push(r.gpu.step(K_NCHW_NLC, &[&dy, &dx], &[n, *c, *hw], n));
                r.acc(x, n as u64, &dx, 1.0);
                r.give(n as u64, dx);
            }
            Op::Attn { c, t, heads, head_dim, qkv, probs, y } => {
                let Some(d_ctx) = r.get(y) else { return };
                // The head split the FORWARD used, over the fused [T, 3C] rows
                // — not an assumed single head (see `Op::Attn`).
                let a = model::block::Bidir {
                    b: 1,
                    t: *t,
                    n_heads: *heads,
                    head_dim: *head_dim,
                    stride: 3 * c,
                    q_off: 0,
                    k_off: *c,
                    v_off: 2 * c,
                };
                let dscores = r.tmp((heads * t * t) as u64);
                let d_qkv = r.tmp((3 * c * t) as u64);
                // The quartet ASSIGNS into three disjoint regions of `d_qkv`
                // (q at 0, k at C, v at 2C), so it needs no pre-zeroing.
                let steps = model::block::bidir_bwd(
                    r.gpu,
                    &r.ids.bidir(),
                    &a,
                    qkv,
                    probs,
                    &d_ctx,
                    &dscores,
                    &d_qkv,
                );
                for s in steps {
                    r.push(s);
                }
                r.acc(qkv, (3 * c * t) as u64, &d_qkv, 1.0);
                r.give((3 * c * t) as u64, d_qkv);
                r.give((heads * t * t) as u64, dscores);
            }
            // ---- the transformer half -------------------------------------
            // `y = x·Wᵀ (+ b)`. `d_W` and `d_b` ACCUMULATE (one weight, many
            // call sites in a shared block); `d_x` is assigned into a temp and
            // folded in with `axpy`, the uniform shape fan-out needs.
            Op::Linear { w, b, m, k, n, x, y } => {
                let Some(dy) = r.get(y) else { return };
                // `matmul_dw` Params: [m, k, n]; bufs [dy, x, dw] - ACCUMULATES.
                r.push(r.gpu.step(r.ids.k(B_MATMUL_DW), &[&dy, x, grads.g(w)], &[*m, *k, *n], n * k));
                if let Some(bn) = b {
                    // `bias_grad` Params: [rows, n]; bufs [dy, db] - ACCUMULATES.
                    r.push(r.gpu.step(r.ids.k(B_BIAS_GRAD), &[&dy, grads.g(bn)], &[*m, *n], *n));
                }
                let dx = r.tmp((*m as u64) * (*k as u64));
                // `matmul_dx` Params: [m, k, n, accumulate]; bufs [dy, w, dx].
                // `accumulate = 0`: this ASSIGNS into the temp, and the fold
                // onto `d[x]` is the `axpy` below.
                r.push(r.gpu.step(r.ids.k(B_MATMUL_DX), &[&dy, self.weight(w), &dx], &[*m, *k, *n, 0], m * k));
                r.acc(x, (*m as u64) * (*k as u64), &dx, 1.0);
                r.give((*m as u64) * (*k as u64), dx);
            }
            Op::LayerNorm { gamma, beta, rows, d, eps, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let n = (*rows as u64) * (*d as u64);
                // `dgamma` needs the row mean/inv-std the forward had. The
                // forward does not retain them (`layernorm` writes only `y`), so
                // recompute them here rather than widening the forward - the
                // stats are two floats per row and the recompute is one pass.
                let (mean, inv) = (r.tmp(*rows as u64), r.tmp(*rows as u64));
                r.push(r.gpu.step(r.ids.k(B_LN_STATS), &[x, &mean, &inv], &[*d, *rows, f(*eps)], *rows));
                // Both affine grads ACCUMULATE.
                r.push(r.gpu.step(r.ids.k(B_LAYERNORM_DGAMMA), &[&dy, x, &mean, &inv, grads.g(gamma)], &[*d, *rows], *d));
                r.push(r.gpu.step(r.ids.k(B_LAYERNORM_DBETA), &[&dy, grads.g(beta)], &[*d, *rows], *d));
                let dx = r.tmp(n);
                r.push(r.gpu.step(r.ids.k(B_LAYERNORM_DX), &[x, self.weight(gamma), &dy, &dx], &[*d, *rows, f(*eps)], *rows));
                r.acc(x, n, &dx, 1.0);
                r.give(n, dx);
                r.give(*rows as u64, mean);
                r.give(*rows as u64, inv);
            }
            // `gelu_erf_bwd` binds the PRE-activation, like every other
            // activation adjoint here. The tanh approximation's backward is a
            // DIFFERENT kernel and is not interchangeable.
            Op::GeluErf { n, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let dx = r.tmp(*n as u64);
                r.push(r.gpu.step(r.ids.k(B_GELU_ERF_BWD), &[x, &dy, &dx], &[*n], *n));
                r.acc(x, *n as u64, &dx, 1.0);
                r.give(*n as u64, dx);
            }
            // `y = a·b` -> `da = dy·b`, `db = dy·a`. The adjoint of a product is
            // two products, so this needs no backward kernel - it dispatches the
            // caller's own forward `mul` slot.
            Op::Mul { n, a, b, y } => {
                let Some(dy) = r.get(y) else { return };
                let mul = self.xf_mul.expect("vae::blocks::grad: Op::Mul recorded with no XformerIds::mul slot");
                let da = r.tmp(*n as u64);
                r.push(r.gpu.step(mul, &[&dy, b, &da], &[*n], *n));
                r.acc(a, *n as u64, &da, 1.0);
                let db = r.tmp(*n as u64);
                r.push(r.gpu.step(mul, &[&dy, a, &db], &[*n], *n));
                r.acc(b, *n as u64, &db, 1.0);
                r.give(*n as u64, da);
                r.give(*n as u64, db);
            }
            // The adjoint of a broadcast is a sum over the broadcast axes; the
            // adjoint wrt `x` is `dy` itself (no kernel, just the same buffer).
            Op::AddChan { c, hw, x, v, y } => {
                let Some(dy) = r.get(y) else { return };
                let n = (*c as u64) * (*hw as u64);
                r.acc(x, n, &dy, 1.0);
                let dv = r.tmp(*c as u64);
                // `add_chan_bcast_dv` Params: [N, C, HW]; bufs [dy, dv] - one
                // invocation per (n, c), serial over HW. ASSIGNS.
                r.push(r.gpu.step(r.ids.k(B_ADD_CHAN_DV), &[&dy, &dv], &[1, *c, *hw], *c));
                r.acc(v, *c as u64, &dv, 1.0);
                r.give(*c as u64, dv);
            }
            // Cross-attention: two lengths, two buffers. `d_q` lands in its own
            // `[tq, c]` grad; `d_k` and `d_v` land in DISJOINT halves of one
            // `[tkv, 2c]` grad, which is why both use the `_acc` forms with
            // `acc_flag = 0` (assign) - there is exactly one query chunk here,
            // so nothing to accumulate across.
            Op::Cross { c, tq, tkv, heads, head_dim, q, kv, probs, y } => {
                let Some(d_ctx) = r.get(y) else { return };
                let (nq, nkv) = ((*tq as u64) * (*c as u64), (*tkv as u64) * 2 * (*c as u64));
                let d_scores = r.tmp((heads * tq * tkv) as u64);
                let d_q = r.tmp(nq);
                let d_kv = r.tmp(nkv);
                let (h, hd) = (*heads, *head_dim);
                // Params mirror the forward trio's, which is the point: a
                // mismatched cross-attention param list is silently wrong.
                r.push(r.gpu.step(
                    r.ids.k(B_XDSCORES),
                    &[&d_ctx, kv, probs, &d_scores],
                    &[1, h, *tq, *tkv, hd, 2 * c, *c, *c],
                    h * tq * tkv,
                ));
                r.push(r.gpu.step(
                    r.ids.k(B_XDQ),
                    &[&d_scores, kv, &d_q],
                    &[1, h, *tq, *tkv, hd, *c, 2 * c, 0, 0],
                    h * tq * hd,
                ));
                r.push(r.gpu.step(
                    r.ids.k(B_XDK_ACC),
                    &[&d_scores, q, &d_kv],
                    &[1, h, *tq, *tkv, hd, *c, 2 * c, 0, 0, 0],
                    h * tkv * hd,
                ));
                r.push(r.gpu.step(
                    r.ids.k(B_XDV_ACC),
                    &[probs, &d_ctx, &d_kv],
                    &[1, h, *tq, *tkv, hd, 2 * c, *c, *c, 0],
                    h * tkv * hd,
                ));
                r.acc(q, nq, &d_q, 1.0);
                r.acc(kv, nkv, &d_kv, 1.0);
                r.give((h * tq * tkv) as u64, d_scores);
                r.give(nq, d_q);
                r.give(nkv, d_kv);
            }
            // A concat's adjoint is two slices of `dy` - a gather per output
            // element, no scatter.
            Op::Concat { ca, cb, hw, a, b, y } => {
                let Some(dy) = r.get(y) else { return };
                let ctot = ca + cb;
                let (na, nb) = ((*ca as u64) * (*hw as u64), (*cb as u64) * (*hw as u64));
                // `concat_split` Params: [N, Ctot, Csrc, c_off, H, W]; `W = 1`
                // and `H = hw` is the flat form (the kernel only ever uses H*W).
                let da = r.tmp(na);
                r.push(r.gpu.step(r.ids.k(B_CONCAT_SPLIT), &[&dy, &da], &[1, ctot, *ca, 0, *hw, 1], ca * hw));
                r.acc(a, na, &da, 1.0);
                let db = r.tmp(nb);
                r.push(r.gpu.step(r.ids.k(B_CONCAT_SPLIT), &[&dy, &db], &[1, ctot, *cb, *ca, *hw, 1], cb * hw));
                r.acc(b, nb, &db, 1.0);
                r.give(na, da);
                r.give(nb, db);
            }
        }
    }
}

/// One gradient buffer per trainable tensor.
pub struct Grads {
    g: HashMap<String, DeviceBuffer>,
}

impl Grads {
    /// The gradient buffer for `name`.
    pub fn g(&self, name: &str) -> &DeviceBuffer {
        self.g.get(name).unwrap_or_else(|| panic!("vae::blocks::grad: no grad for {name}"))
    }
    /// Every gradient buffer — the model's `zero_grads` clear list.
    pub fn all(&self) -> Vec<&DeviceBuffer> {
        self.g.values().collect()
    }
}

/// The recorded reverse pass.
pub struct Reverse {
    /// The reverse dispatches, in submit order.
    pub steps: Vec<Step>,
    /// Activation-gradient buffers that MUST be zeroed before `steps` run (they
    /// are `axpy` accumulation targets). Parameter grads are NOT in here.
    pub clears: Vec<DeviceBuffer>,
    d: HashMap<usize, DeviceBuffer>,
}

impl Reverse {
    /// The gradient buffer of a forward activation, if the walk reached it.
    pub fn d(&self, buf: &DeviceBuffer) -> Option<&DeviceBuffer> {
        self.d.get(&key(buf))
    }
}

fn key(b: &DeviceBuffer) -> usize {
    b.alloc_id() as usize
}

struct Rev<'a> {
    gpu: &'a Gpu,
    ids: BwdIds,
    steps: Vec<Step>,
    d: HashMap<usize, DeviceBuffer>,
    clears: Vec<DeviceBuffer>,
    /// Scratch temps by exact length. A temp is written by one dispatch and
    /// read by the very next one in the same submit (which runs its steps in
    /// order), so handing it back afterwards is bit-exact reuse.
    pool: HashMap<u64, Vec<DeviceBuffer>>,
}

impl Rev<'_> {
    fn push(&mut self, s: Step) {
        self.steps.push(s);
    }
    fn get(&self, y: &DeviceBuffer) -> Option<DeviceBuffer> {
        self.d.get(&key(y)).cloned()
    }
    fn tmp(&mut self, len: u64) -> DeviceBuffer {
        self.pool.get_mut(&len).and_then(Vec::pop).unwrap_or_else(|| self.gpu.storage(len))
    }
    fn give(&mut self, len: u64, b: DeviceBuffer) {
        self.pool.entry(len).or_default().push(b);
    }
    /// `d[x] += s * src`, allocating (and registering for clearing) `d[x]`.
    fn acc(&mut self, x: &DeviceBuffer, len: u64, src: &DeviceBuffer, s: f32) {
        let dst = match self.d.get(&key(x)) {
            Some(b) => b.clone(),
            None => {
                let b = self.gpu.storage(len);
                self.clears.push(b.clone());
                self.d.insert(key(x), b.clone());
                b
            }
        };
        self.steps.push(self.gpu.step(self.ids.k(B_AXPY), &[&dst, src], &[len as u32, f(s)], len as u32));
    }
}
