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
use gpu_core::select::KernelSelector;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

// Offsets within `super::BWD_KERNELS`.
const B_CONV_DX: usize = 0;
const B_CONV_DW: usize = 1;
// Offset 2 ("bias_grad") stays REGISTERED in `super::BWD_KERNELS` - removing
// an entry would shift every later `B_*` offset - but this file no longer
// dispatches it: both call sites use `Rev::bias_grad`'s two-stage
// `bias_grad_part`/`bias_grad_final` pair instead (M5.7).
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
const B_BIAS_GRAD_PART: usize = 34;
const B_BIAS_GRAD_FINAL: usize = 35;
// RRDBNet's LeakyReLU + scalar-multiply residual (see `super::BWD_KERNELS`'s
// own note on why these are APPENDED here too).
const B_LEAKY_RELU_BWD: usize = 36;
const B_SCALE_ADD_DEXP: usize = 37;

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
    /// The caller's `scale_row` slot, carried from `Builder`'s
    /// [`super::MixIds`]. `Op::Mix`'s adjoint dispatches `scale_row` against
    /// the host-kept, never-trained `a`/`b` - see `super::MixIds`'s note on
    /// why this is threaded through rather than registered here.
    xf_scale_row: Option<usize>,
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
        xf_scale_row: Option<usize>,
    ) -> Trace {
        Trace { ops, order, w: w.clone(), xf_mul, xf_scale_row }
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
            Op::Conv { w, b, cin, cout, k, stride, pad, h, w_in, ho, wo, n, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let bsz = *n;
                let p = [bsz, *cin, *h, *w_in, *cout, *k, *stride, *pad, *ho, *wo];
                let (hw, per_in) = (ho * wo, cin * h * w_in);
                // Every row-count below is the WHOLE batch's rows stacked
                // (`bsz * hw`), not one image's - `dW`/`dbias` correctly SUM
                // their gradient over every row regardless of which image it
                // came from (the weight/bias is shared across the batch,
                // exactly what `conv2d_dw`'s own internal `N`-loop already
                // does for the direct kernel), and the lowered path's `im2col`/
                // `col2im` bind one image's slice at a time in a loop (neither
                // kernel carries a batch dimension of its own).
                let rows_total = bsz * hw;
                let n_in = (bsz as u64) * per_in as u64;
                let cinkk = cin * k * k;
                // `backend_api::select::Op::Conv2dBackward` picks
                // `KernelVariant::RegisterTiled` for BOTH dW's and dX's
                // lowering below - one decision, since `vae::blocks::grad`
                // has always made it from the identical boolean (see that
                // Op's own doc for why it is not split in two).
                let shape = gpu_core::select::OpShape {
                    m: hw,
                    n: *cout,
                    k: cinkk,
                    dtype: gpu_core::select::Dtype::F32,
                };
                let lowered = gpu_core::select::DefaultSelector.select(
                    gpu_core::select::Op::Conv2dBackward,
                    shape,
                    &r.gpu.caps(),
                ) == gpu_core::select::KernelVariant::RegisterTiled;

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
                let t = r.tmp((rows_total * cout) as u64);
                r.push(r.gpu.step(K_NCHW_NLC, &[&dy, &t], &[rows_total * cout, *cout, hw], rows_total * cout));

                if lowered {
                    // Every image's `im2col` window, stacked as consecutive row
                    // ranges of ONE `col` scratch (`im2col_at` itself has no
                    // batch dimension - see its own doc - so this loop over
                    // `bsz` is what the forward's `conv_s` already does).
                    let col = r.tmp((rows_total * cinkk) as u64);
                    for ni in 0..bsz {
                        let x_off = (ni as u64) * per_in as u64;
                        let col_off = (ni as u64) * (hw * cinkk) as u64;
                        r.push(r.gpu.step_sliced(
                            K_IM2COL_AT,
                            &[x, &col],
                            &[(x_off, per_in as u64), (col_off, (hw * cinkk) as u64)],
                            &[*cin, *h, *w_in, *k, *stride, *pad, *ho, *wo, cinkk, 0, hw],
                            hw * cinkk,
                        ));
                    }
                    // dW's tile grid is ceil(Cout/128)*ceil(CinKK/128) — it does
                    // NOT grow with the contraction length, so a wide-shallow
                    // conv launches a handful of workgroups and idles the card.
                    // Split the contraction to reach `DW_SPLITK_TARGET_WGS`.
                    let tiles = cout.div_ceil(128) * cinkk.div_ceil(128);
                    let slices = super::DW_SPLITK_TARGET_WGS.div_ceil(tiles).max(1).min(rows_total.div_ceil(8));
                    if slices > 1 {
                        let rc = (*cout as u64) * cinkk as u64;
                        let part = r.tmp(rc * slices as u64);
                        r.push(r.gpu.step(
                            r.ids.k(B_MATMUL_DW_SPLITK),
                            &[&t, &col, &part],
                            &[rows_total, cinkk, *cout, slices],
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
                            &[rows_total, cinkk, *cout],
                            tiles * 256,
                        ));
                    }
                    r.give((rows_total * cinkk) as u64, col);
                } else {
                    r.push(r.gpu.step(r.ids.k(B_CONV_DW), &[&dy, x, grads.g(w)], &p, cout * cin * k * k));
                }
                // `bias_grad` reduces a [rows, features] buffer down its rows,
                // but `dy` is NCHW = feature-major. One `nchw_nlc` puts the
                // channels last; the alternative would be a new NCHW bias
                // reduction kernel, and this composition needs neither. `rows`
                // is the WHOLE batch's positions - a bias gradient sums over
                // every one of them, same as `dW`'s above.
                r.bias_grad(&t, grads.g(b), rows_total, *cout);

                // dX (assigns) -> accumulate. Two paths:
                //
                //  * LOWERED — `dcol[HW,CinKK] = dY[HW,Cout] . W[Cout,CinKK]`
                //    (`matmul_dx_reg`, register-tiled) then `col2im`, which sums
                //    only the K*K taps per input pixel. `conv2d_dx` instead
                //    reduces over Cout*K*K on ONE lane, and the backward profile
                //    put it at a large share of a VQGAN training step, several
                //    times the cost of the forward conv it mirrors. The lowering
                //    measured faster at every shape in `vqgan_bench convbwd`.
                //    `col2im`'s own doc records that it binds one image's slice
                //    at a time - its `N` field decodes `dx`'s coordinate, not a
                //    batch dimension in `dcol` - so, like `im2col_at` above,
                //    this is a loop over `bsz`.
                //  * DIRECT — below `GEMM_CONV_BWD_MIN_COUT`, and on any device
                //    without workgroup reductions: `matmul_dx_reg` carries
                //    barriers the CPU JIT cannot compile, so this branches on the
                //    QUERIED capability, never on an assumption (lessons #5).
                //
                // The NLC transpose the GEMM needs is `t`, which bias_grad has
                // already built — so the fast path costs one transpose less than
                // it looks.
                let dx = r.tmp(n_in);
                if lowered {
                    let dcol = r.tmp((rows_total * cinkk) as u64);
                    r.push(r.gpu.step(
                        r.ids.k(B_MATMUL_DX_REG),
                        &[&t, self.weight(w), &dcol],
                        &[rows_total, cinkk, *cout, 0],
                        rows_total.div_ceil(128) * cinkk.div_ceil(128) * 256,
                    ));
                    for ni in 0..bsz {
                        let dcol_off = (ni as u64) * (hw * cinkk) as u64;
                        let dx_off = (ni as u64) * per_in as u64;
                        r.push(r.gpu.step_sliced(
                            r.ids.k(B_COL2IM),
                            &[&dcol, &dx],
                            &[(dcol_off, (hw * cinkk) as u64), (dx_off, per_in as u64)],
                            &[1, *cin, *h, *w_in, *k, *stride, *pad, *ho, *wo, cinkk],
                            per_in,
                        ));
                    }
                    r.give((rows_total * cinkk) as u64, dcol);
                } else {
                    r.push(r.gpu.step(r.ids.k(B_CONV_DX), &[&dy, self.weight(w), &dx], &p, n_in as u32));
                }
                r.give((rows_total * cout) as u64, t);
                r.acc(x, n_in, &dx, 1.0);
                r.give(n_in, dx);
            }
            Op::Gn { gb, c, h, w, g, n, x, stats, y } => {
                let Some(dy) = r.get(y) else { return };
                let bsz = *n;
                let total = (bsz as u64) * (c * h * w) as u64;
                let p = [bsz, *c, *h, *w, *g];
                let gbuf = self.weight(gb);
                // dyg = dy * gamma, the shared input of gn_dsum and gn_dx.
                // `scale_chan`'s Params are the generic `[rows,C,inner]` shape
                // (see its own doc) - a leading batch axis is already exactly
                // what "more rows" means to it, so `total` folding `bsz` in is
                // the whole change.
                let dyg = r.tmp(total);
                r.push(r.gpu.step(r.ids.k(B_SCALE_CHAN), &[&dy, gbuf, &dyg], &[total as u32, *c, h * w], total as u32));
                let sums = r.tmp((bsz * 4 * *g) as u64);
                // Two-stage, barrier-free. `gn_dsum` is ONE invocation per
                // (n,g) group (32 lanes walking (C/G)*H*W elements each),
                // measured at well under one percent of the card's bandwidth
                // roof and a quarter of the whole backward. Stage 1 splits each
                // group across `GN_P` partials (coalesced, strided), stage 2
                // folds them. No workgroupBarrier, so this needs no capability
                // branch and `backend-cpu` gets it too - the same shape as the
                // forward's `gn_part`/`gn_stats2`.
                let part = r.tmp((bsz * *g) as u64 * 2 * super::GN_P as u64);
                let pp = [bsz, *c, *h, *w, *g, super::GN_P];
                r.push(r.gpu.step(r.ids.k(B_GN_DSUM_PART), &[x, &dyg, stats, &part], &pp, bsz * g * super::GN_P));
                r.push(r.gpu.step(r.ids.k(B_GN_DSUM2), &[&part, stats, &sums], &pp, bsz * g));
                r.give((bsz * *g) as u64 * 2 * super::GN_P as u64, part);
                // dgamma -> dgb[0..C], dbeta -> dgb[C..2C]: disjoint writes into
                // the same fused buffer, both accumulating, and both SUMMED
                // over the whole batch (gamma/beta are shared parameters, not
                // per-image, so `gn_dgb_part`'s own internal `N*H*W` walk per
                // channel is exactly this sum - no extra loop needed here,
                // unlike `dW`/`col2im` above).
                // Two-stage, and ONE pass over `dy` for both affine gradients.
                // `gn_dgamma`/`gn_dbeta` were a lane per channel walking N*H*W
                // each, together a couple of percent of the bandwidth roof, and
                // each read the whole of `dy` separately.
                let dgb_part = r.tmp(2 * *c as u64 * super::GN_P as u64);
                let pg = [bsz, *c, *h, *w, *g, super::GN_P];
                r.push(r.gpu.step(
                    r.ids.k(B_GN_DGB_PART),
                    &[x, &dy, stats, &dgb_part],
                    &pg,
                    c * super::GN_P,
                ));
                r.push(r.gpu.step(r.ids.k(B_GN_DGB2), &[&dgb_part, grads.g(gb)], &pg, *c));
                r.give(2 * *c as u64 * super::GN_P as u64, dgb_part);
                let dx = r.tmp(total);
                r.push(r.gpu.step(r.ids.k(B_GN_DX), &[x, &dyg, &sums, &dx], &p, total as u32));
                r.acc(x, total, &dx, 1.0);
                r.give(total, dx);
                r.give((bsz * 4 * *g) as u64, sums);
                r.give(total, dyg);
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
            Op::Up2 { c, h, w, n, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let total = (*n as u64) * (c * h * w) as u64;
                let dx = r.tmp(total);
                r.push(r.gpu.step(r.ids.k(B_UPSAMPLE2_DX), &[&dy, &dx], &[*n, *c, *h, *w], total as u32));
                r.acc(x, total, &dx, 1.0);
                r.give(total, dx);
            }
            // A layout permutation is its own transpose's adjoint. `total`
            // folds the batch in exactly as the forward's `Builder::
            // nchw_to_rows`/`rows_to_nchw` do.
            Op::NchwNlc { c, hw, n, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let total = (*n as u64) * (c * hw) as u64;
                let dx = r.tmp(total);
                r.push(r.gpu.step(K_NLC_NCHW, &[&dy, &dx], &[total as u32, *c, *hw], total as u32));
                r.acc(x, total, &dx, 1.0);
                r.give(total, dx);
            }
            Op::NlcNchw { c, hw, n, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let total = (*n as u64) * (c * hw) as u64;
                let dx = r.tmp(total);
                r.push(r.gpu.step(K_NCHW_NLC, &[&dy, &dx], &[total as u32, *c, *hw], total as u32));
                r.acc(x, total, &dx, 1.0);
                r.give(total, dx);
            }
            Op::Attn { c, t, heads, head_dim, n, qkv, probs, y } => {
                let Some(d_ctx) = r.get(y) else { return };
                let bsz = *n;
                // The head split the FORWARD used, over the fused [T, 3C] rows
                // - not an assumed single head (see `Op::Attn`) - and the
                // batch count it recorded, since `Bidir`/`bidir_bwd` are
                // already fully `b`-parameterised (the same helper the
                // FORWARD's `Builder::self_attn` calls).
                let a = model::block::Bidir {
                    b: bsz,
                    t: *t,
                    n_heads: *heads,
                    head_dim: *head_dim,
                    stride: 3 * c,
                    q_off: 0,
                    k_off: *c,
                    v_off: 2 * c,
                };
                let dscores = r.tmp((bsz as u64) * (heads * t * t) as u64);
                let d_qkv = r.tmp((bsz as u64) * (3 * c * t) as u64);
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
                r.acc(qkv, (bsz as u64) * (3 * c * t) as u64, &d_qkv, 1.0);
                r.give((bsz as u64) * (3 * c * t) as u64, d_qkv);
                r.give((bsz as u64) * (heads * t * t) as u64, dscores);
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
                    // Two-stage `bias_grad_part`/`bias_grad_final` - ACCUMULATES.
                    r.bias_grad(&dy, grads.g(bn), *m, *n);
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
            Op::Concat { ca, cb, hw, n, a, b, y } => {
                let Some(dy) = r.get(y) else { return };
                let bsz = *n;
                let ctot = ca + cb;
                let (na, nb) = ((bsz as u64) * (*ca as u64) * (*hw as u64), (bsz as u64) * (*cb as u64) * (*hw as u64));
                // `concat_split` Params: [N, Ctot, Csrc, c_off, H, W]; `W = 1`
                // and `H = hw` is the flat form (the kernel only ever uses H*W).
                let da = r.tmp(na);
                r.push(r.gpu.step(r.ids.k(B_CONCAT_SPLIT), &[&dy, &da], &[bsz, ctot, *ca, 0, *hw, 1], bsz * ca * hw));
                r.acc(a, na, &da, 1.0);
                let db = r.tmp(nb);
                r.push(r.gpu.step(r.ids.k(B_CONCAT_SPLIT), &[&dy, &db], &[bsz, ctot, *cb, *ca, *hw, 1], bsz * cb * hw));
                r.acc(b, nb, &db, 1.0);
                r.give(na, da);
                r.give(nb, db);
            }
            // `y = a*x + b*f` -> `dx = a*dy`, `df = b*dy`: `scale_row` IS the
            // adjoint, dispatched against the same host-kept `a`/`b` the
            // forward packed into `ab` - no `dab` kernel, exactly `edm_mix`'s
            // own documented contract.
            Op::Mix { n, x, f, a, b, y } => {
                let Some(dy) = r.get(y) else { return };
                let scale_row = self.xf_scale_row.expect("vae::blocks::grad: Op::Mix recorded with no MixIds::bwd slot");
                let dx = r.tmp(*n as u64);
                r.push(r.gpu.step(scale_row, &[&dy, a, &dx], &[*n, *n], *n));
                r.acc(x, *n as u64, &dx, 1.0);
                let df = r.tmp(*n as u64);
                r.push(r.gpu.step(scale_row, &[&dy, b, &df], &[*n, *n], *n));
                r.acc(f, *n as u64, &df, 1.0);
                r.give(*n as u64, dx);
                r.give(*n as u64, df);
            }
            // `y = leaky_relu(x, slope)` -> `dx = dy` where `x >= 0`, `slope*dy`
            // otherwise. `leaky_relu_bwd` binds the PRE-activation, like every
            // other activation adjoint in this file.
            Op::LeakyRelu { n, slope, x, y } => {
                let Some(dy) = r.get(y) else { return };
                let dx = r.tmp(*n as u64);
                r.push(r.gpu.step(r.ids.k(B_LEAKY_RELU_BWD), &[x, &dy, &dx], &[*n, f(*slope)], *n));
                r.acc(x, *n as u64, &dx, 1.0);
                r.give(*n as u64, dx);
            }
            // `y = scale[0] * fx` -> `dfx = scale[0] * dy`. `scale_add_dexp`
            // Params: [n_rows, d_model, n_experts, e_idx]; bufs [gate,
            // d_moe_acc, d_expert] - `n_rows = 1, d_model = n, n_experts = 1,
            // e_idx = 0` mirrors the forward `scale_add` dispatch exactly (see
            // `super::Builder::residual_scale`). Gradient flows ONLY to `fx` -
            // `scale` is a host constant, never a [`Grads`] target: route
            // gradient THROUGH it, never assign a gradient TO it.
            Op::ScaleAdd { n, scale, fx, y } => {
                let Some(dy) = r.get(y) else { return };
                let dfx = r.tmp(*n as u64);
                r.push(r.gpu.step(r.ids.k(B_SCALE_ADD_DEXP), &[scale, &dy, &dfx], &[1, *n, 1, 0], *n));
                r.acc(fx, *n as u64, &dfx, 1.0);
                r.give(*n as u64, dfx);
            }
            // `y = x * scale[c]` -> `dx = dy * scale[c]`: `scale[c]` does not
            // depend on `x`, so the forward kernel IS its own adjoint, run on
            // `dy` in place of `x` - no new kernel, the same `scale_chan`
            // dispatch `Op::Gn`'s `dyg = dy * gamma` above already uses.
            // `scale` is a host constant, never a [`Grads`] target: gradient
            // flows ONLY to `x` - deliberately no `dscale` (see [`super::
            // Op::ScaleChan`]'s doc for why one would be dead code here).
            Op::ScaleChan { total, c, inner, x, scale, y } => {
                let Some(dy) = r.get(y) else { return };
                let dx = r.tmp(*total as u64);
                r.push(r.gpu.step(r.ids.k(B_SCALE_CHAN), &[&dy, scale, &dx], &[*total, *c, *inner], *total));
                r.acc(x, *total as u64, &dx, 1.0);
                r.give(*total as u64, dx);
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
    /// Bias gradient `dbias[n] += sum_m dy[m,n]`, via the two-stage
    /// `bias_grad_part`/`bias_grad_final` pair (kernel-performance.md M5.7)
    /// rather than `bias_grad`'s single-thread-per-column walk: `dy` MUST
    /// already be `[m, n]` (feature-fastest) - the conv call site transposes
    /// NCHW into this layout via `nchw_nlc` before calling here, exactly as it
    /// did for the `bias_grad` dispatch this replaces.
    fn bias_grad(&mut self, dy: &DeviceBuffer, dbias: &DeviceBuffer, m: u32, n: u32) {
        let pp = [m, n, super::BIAS_GRAD_P];
        let part = self.tmp((n * super::BIAS_GRAD_P) as u64);
        self.steps.push(self.gpu.step(self.ids.k(B_BIAS_GRAD_PART), &[dy, &part], &pp, n * super::BIAS_GRAD_P));
        self.steps.push(self.gpu.step(self.ids.k(B_BIAS_GRAD_FINAL), &[&part, dbias], &pp, n));
        self.give((n * super::BIAS_GRAD_P) as u64, part);
    }
}
