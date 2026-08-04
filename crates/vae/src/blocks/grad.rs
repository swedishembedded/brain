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

use super::{Op, K_NCHW_NLC, K_NLC_NCHW};
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

// Offsets within `super::BWD_KERNELS`.
const B_CONV_DX: usize = 0;
const B_CONV_DW: usize = 1;
const B_BIAS_GRAD: usize = 2;
const B_SILU_BWD: usize = 3;
const B_SCALE_CHAN: usize = 4;
const B_GN_DSUM: usize = 5;
const B_GN_DX: usize = 6;
const B_GN_DGAMMA: usize = 7;
const B_GN_DBETA: usize = 8;
const B_UPSAMPLE2_DX: usize = 9;
const B_AXPY: usize = 10;
const B_ATTN_DSCORES: usize = 11;
const B_ATTN_DV: usize = 12;
const B_ATTN_DQ: usize = 13;
const B_ATTN_DK: usize = 14;

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
pub struct Trace {
    ops: Vec<Op>,
    /// Tensor name -> length, in first-use order (the parameter list).
    order: Vec<(String, u64)>,
    w: HashMap<String, DeviceBuffer>,
}

impl Trace {
    pub(super) fn new(
        ops: Vec<Op>,
        order: Vec<(String, u64)>,
        w: &HashMap<String, DeviceBuffer>,
    ) -> Trace {
        Trace { ops, order, w: w.clone() }
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
                // dW (accumulates) then db, both from the conv's own input.
                r.push(r.gpu.step(r.ids.k(B_CONV_DW), &[&dy, x, grads.g(w)], &p, cout * cin * k * k));
                // `bias_grad` reduces a [rows, features] buffer down its rows,
                // but `dy` is NCHW = feature-major. One `nchw_nlc` puts the
                // channels last; the alternative would be a new NCHW bias
                // reduction kernel, and this composition needs neither.
                let t = r.tmp((cout * hw) as u64);
                r.push(r.gpu.step(K_NCHW_NLC, &[&dy, &t], &[cout * hw, *cout, hw], cout * hw));
                r.push(r.gpu.step(r.ids.k(B_BIAS_GRAD), &[&t, grads.g(b)], &[hw, *cout], *cout));
                r.give((cout * hw) as u64, t);
                // dX (assigns) -> accumulate.
                let dx = r.tmp(n_in as u64);
                r.push(r.gpu.step(r.ids.k(B_CONV_DX), &[&dy, self.weight(w), &dx], &p, n_in));
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
                r.push(r.gpu.step(r.ids.k(B_GN_DSUM), &[x, &dyg, stats, &sums], &p, *g));
                // dgamma -> dgb[0..C], dbeta -> dgb[C..2C]: disjoint writes into
                // the same fused buffer, both accumulating.
                r.push(r.gpu.step(r.ids.k(B_GN_DGAMMA), &[x, &dy, stats, grads.g(gb)], &p, *c));
                r.push(r.gpu.step(r.ids.k(B_GN_DBETA), &[&dy, grads.g(gb)], &p, *c));
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
            Op::Attn { c, t, qkv, probs, y } => {
                let Some(d_ctx) = r.get(y) else { return };
                // Single head, head_dim = C, over the fused [T, 3C] rows.
                let a = model::block::Bidir {
                    b: 1,
                    t: *t,
                    n_heads: 1,
                    head_dim: *c,
                    stride: 3 * c,
                    q_off: 0,
                    k_off: *c,
                    v_off: 2 * c,
                };
                let dscores = r.tmp((t * t) as u64);
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
                r.give((t * t) as u64, dscores);
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
