// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-pass (looped) decoders for its clients. If your team needs expertise in
// gating what one pass of a model hands to the next then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The message gate: how a looped decoder mixes a message from its previous pass into a hidden state.
//!
//! ```text
//! h  = Wd [rms(x); rms(m)]          (rms: RMS norm without gain; Wd: [rank, 2 width])
//! x' = (1 - tanh(Wf h)) * x + tanh(Wg h) * m          (Wf, Wg: [width, rank])
//! ```
//!
//! All f32. Composed from existing kernels (`rmsnorm_nogain`, `concat2`, the [`crate::ops::Ops`] products, `tanh_act`,
//! `mul`, `sub`, `add2` and their backwards); a fused kernel is a later candidate, taken only on profile evidence. The
//! forward keeps what its backward reads in [`MessageGateActs`] (the concatenated norms, the rank-`rank` code, both
//! pre-activations and both gates); the backward writes the input gradients and ADDS the three weight gradients
//! (the `matmul_dw` contract: a looped layer's gate runs once per pass and its gradient is the sum).

use gpu_core::{DeviceBuffer, Dispatch, Gpu, Step};

use crate::ops::{Ops, Weight};

/// Kernels the gate dispatches besides the [`Ops`] products (whose list is [`crate::ops::kernel_list`]).
#[must_use]
pub fn kernels() -> &'static [(&'static str, &'static str)] {
    &[
        ("rmsnorm_nogain", kernels::RMSNORM_NOGAIN),
        ("rmsnorm_nogain_dx", kernels::RMSNORM_NOGAIN_DX),
        ("concat2", kernels::CONCAT2),
        ("concat_split", kernels::CONCAT_SPLIT),
        ("tanh_act", kernels::TANH_ACT),
        ("tanh_act_bwd", kernels::TANH_ACT_BWD),
        ("mul", kernels::MUL),
        ("sub", kernels::SUB),
        ("add2", kernels::ADD2),
    ]
}

/// The kernel slots of [`kernels`] on one device.
#[derive(Clone, Copy, Debug)]
pub struct MessageGateIds {
    norm: usize,
    norm_dx: usize,
    concat: usize,
    split: usize,
    tanh: usize,
    tanh_bwd: usize,
    mul: usize,
    sub: usize,
    add: usize,
}

impl MessageGateIds {
    pub fn resolve(gpu: &Gpu) -> Result<MessageGateIds, String> {
        let k = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("message gate: kernel {name} is not registered (append model::message_gate::kernels() to the kernel list)"));
        Ok(MessageGateIds {
            norm: k("rmsnorm_nogain")?,
            norm_dx: k("rmsnorm_nogain_dx")?,
            concat: k("concat2")?,
            split: k("concat_split")?,
            tanh: k("tanh_act")?,
            tanh_bwd: k("tanh_act_bwd")?,
            mul: k("mul")?,
            sub: k("sub")?,
            add: k("add2")?,
        })
    }
}

/// The gate's three projections, f32 row-major as published: `down` `[rank, 2 width]`, `forget` and `gain`
/// `[width, rank]`.
pub struct MessageGateWeights<'a> {
    pub down: &'a DeviceBuffer,
    pub forget: &'a DeviceBuffer,
    pub gain: &'a DeviceBuffer,
}

/// One of the gate's three products, named to a caller that corrects it ([`MessageGate::forward_corrected`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateProduct {
    /// `h = Wd [rms(x); rms(m)]`: input `[rows, 2 width]`, output `[rows, rank]`.
    Down,
    /// `Wf h`: input `[rows, rank]`, output `[rows, width]` (before its tanh).
    Forget,
    /// `Wg h`: input `[rows, rank]`, output `[rows, width]` (before its tanh).
    Gain,
}

/// A correction recorded right after one of the gate's products: `(steps, which product, its input, its output)`; it
/// changes the output in place (a served adapter's `x dW^T` per row).
pub type GateCorrection<'a> = dyn FnMut(&mut Vec<Step>, GateProduct, &DeviceBuffer, &DeviceBuffer) + 'a;

/// What the forward keeps for the backward, plus its own scratch.
pub struct MessageGateActs {
    xn: DeviceBuffer,
    mn: DeviceBuffer,
    cat: DeviceBuffer,
    code: DeviceBuffer,
    pre_forget: DeviceBuffer,
    forget: DeviceBuffer,
    pre_gain: DeviceBuffer,
    gain: DeviceBuffer,
    tmp: DeviceBuffer,
    kept: DeviceBuffer,
}

/// The backward's scratch.
pub struct MessageGateScratch {
    t1: DeviceBuffer,
    t2: DeviceBuffer,
    d_pre_forget: DeviceBuffer,
    d_pre_gain: DeviceBuffer,
    d_code: DeviceBuffer,
    d_cat: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_mn: DeviceBuffer,
    zeros: DeviceBuffer,
}

/// One message gate's shape: `rows` tokens of `width`, a code of `rank`, the norm's `eps`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MessageGate {
    pub rows: u32,
    pub width: u32,
    pub rank: u32,
    pub eps: f32,
}

impl MessageGate {
    pub fn new(rows: u32, width: u32, rank: u32, eps: f32) -> Result<MessageGate, String> {
        if rows == 0 || width == 0 || rank == 0 {
            return Err(format!("message gate: {rows} rows of width {width} at rank {rank} is empty"));
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(format!("message gate: eps {eps} must be positive"));
        }
        if (rows as u64) * 2 * (width as u64) > u32::MAX as u64 {
            return Err(format!("message gate: {rows} x {width} overflows a u32 index"));
        }
        Ok(MessageGate { rows, width, rank, eps })
    }

    fn n(&self) -> u32 {
        self.rows * self.width
    }

    /// The forward's buffers for this shape.
    pub fn acts(&self, gpu: &Gpu) -> MessageGateActs {
        let (n, r) = (self.n() as u64, self.rows as u64);
        MessageGateActs {
            xn: gpu.storage(n),
            mn: gpu.storage(n),
            cat: gpu.storage(2 * n),
            code: gpu.storage(r * self.rank as u64),
            pre_forget: gpu.storage(n),
            forget: gpu.storage(n),
            pre_gain: gpu.storage(n),
            gain: gpu.storage(n),
            tmp: gpu.storage(n),
            kept: gpu.storage(n),
        }
    }

    /// The backward's buffers for this shape.
    pub fn scratch(&self, gpu: &Gpu) -> MessageGateScratch {
        let (n, r) = (self.n() as u64, self.rows as u64);
        let zeros = gpu.storage(n);
        gpu.write_f32(&zeros, &vec![0.0; n as usize]);
        MessageGateScratch {
            t1: gpu.storage(n),
            t2: gpu.storage(n),
            d_pre_forget: gpu.storage(n),
            d_pre_gain: gpu.storage(n),
            d_code: gpu.storage(r * self.rank as u64),
            d_cat: gpu.storage(2 * n),
            d_xn: gpu.storage(n),
            d_mn: gpu.storage(n),
            zeros,
        }
    }

    fn weights(&self, w: &MessageGateWeights) -> (Weight, Weight, Weight) {
        (Weight::dense_f32(w.down.clone(), self.rank, 2 * self.width), Weight::dense_f32(w.forget.clone(), self.width, self.rank), Weight::dense_f32(w.gain.clone(), self.width, self.rank))
    }

    fn norm(&self, gpu: &Gpu, ids: &MessageGateIds, x: &DeviceBuffer, out: &DeviceBuffer) -> Step {
        gpu.dispatch(ids.norm, &[x, out], &[self.width, self.rows, self.eps.to_bits()], Dispatch::Workgroups(self.rows))
    }

    /// `y = (1 - tanh(Wf h)) * x + tanh(Wg h) * m` for `x` and the message `m` (`[rows, width]` each).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, s: &mut Vec<Step>, gpu: &Gpu, ids: &MessageGateIds, ops: &Ops, w: &MessageGateWeights, x: &DeviceBuffer, m: &DeviceBuffer, a: &MessageGateActs, y: &DeviceBuffer) {
        self.forward_corrected(s, gpu, ids, ops, w, x, m, a, y, &mut |_, _, _, _| {});
    }

    /// [`Self::forward`] with `correct` recorded after each product (down, then forget, then gain), before anything reads
    /// the product: the gate of corrected weights without forming them (a served batch whose rows run under different
    /// adapters). Forward only: a trained gate forms its weights whole ([`crate::adapter::merged::MergedLora`]).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_corrected(&self, s: &mut Vec<Step>, gpu: &Gpu, ids: &MessageGateIds, ops: &Ops, w: &MessageGateWeights, x: &DeviceBuffer, m: &DeviceBuffer, a: &MessageGateActs, y: &DeviceBuffer, correct: &mut GateCorrection<'_>) {
        let (rows, d, k, n) = (self.rows, self.width, self.rank, self.n());
        let (down, forget, gain) = self.weights(w);
        s.push(self.norm(gpu, ids, x, &a.xn));
        s.push(self.norm(gpu, ids, m, &a.mn));
        s.push(gpu.step(ids.concat, &[&a.xn, &a.mn, &a.cat], &[rows, d, d, 1, 1], 2 * n));
        ops.matmul(s, &down, &ops.act_f32(&a.cat, 0, rows, 2 * d), &a.code, 0);
        correct(s, GateProduct::Down, &a.cat, &a.code);
        ops.matmul(s, &forget, &ops.act_f32(&a.code, 0, rows, k), &a.pre_forget, 0);
        correct(s, GateProduct::Forget, &a.code, &a.pre_forget);
        s.push(gpu.step(ids.tanh, &[&a.pre_forget, &a.forget], &[n], n));
        ops.matmul(s, &gain, &ops.act_f32(&a.code, 0, rows, k), &a.pre_gain, 0);
        correct(s, GateProduct::Gain, &a.code, &a.pre_gain);
        s.push(gpu.step(ids.tanh, &[&a.pre_gain, &a.gain], &[n], n));
        s.push(gpu.step(ids.mul, &[&a.forget, x, &a.tmp], &[n], n));
        s.push(gpu.step(ids.sub, &[x, &a.tmp, &a.kept], &[n, 0, 0], n));
        s.push(gpu.step(ids.mul, &[&a.gain, m, &a.tmp], &[n], n));
        s.push(gpu.step(ids.add, &[&a.kept, &a.tmp, y], &[n], n));
    }

    /// The backward of [`Self::forward`] given `dy`: WRITES `dx` and `dm`, ADDS the weight gradients into `grads`
    /// (`down`, `forget`, `gain`, laid out as the weights). `x`, `m` and `a` are the forward's.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        s: &mut Vec<Step>,
        gpu: &Gpu,
        ids: &MessageGateIds,
        ops: &Ops,
        w: &MessageGateWeights,
        x: &DeviceBuffer,
        m: &DeviceBuffer,
        a: &MessageGateActs,
        dy: &DeviceBuffer,
        dx: &DeviceBuffer,
        dm: &DeviceBuffer,
        grads: &MessageGateWeights,
        b: &MessageGateScratch,
    ) {
        let (rows, d, k, n) = (self.rows, self.width, self.rank, self.n());
        let (down, forget, gain) = self.weights(w);
        // Gates: dg = dy * m; df = -dy * x; through tanh with the saved pre-activations.
        s.push(gpu.step(ids.mul, &[dy, m, &b.t1], &[n], n));
        s.push(gpu.step(ids.tanh_bwd, &[&b.t1, &a.pre_gain, &b.d_pre_gain], &[n], n));
        s.push(gpu.step(ids.mul, &[dy, x, &b.t1], &[n], n));
        s.push(gpu.step(ids.tanh_bwd, &[&b.t1, &a.pre_forget, &b.t2], &[n], n));
        s.push(gpu.step(ids.sub, &[&b.zeros, &b.t2, &b.d_pre_forget], &[n, 0, 0], n));
        // The two up-projections and the code.
        ops.matmul_dw(s, &a.code, &b.d_pre_gain, rows, d, k, grads.gain);
        ops.matmul_dw(s, &a.code, &b.d_pre_forget, rows, d, k, grads.forget);
        ops.matmul_dx(s, &gain, &b.d_pre_gain, rows, &b.d_code, false);
        ops.matmul_dx(s, &forget, &b.d_pre_forget, rows, &b.d_code, true);
        ops.matmul_dw(s, &a.cat, &b.d_code, rows, k, 2 * d, grads.down);
        ops.matmul_dx(s, &down, &b.d_code, rows, &b.d_cat, false);
        // Split the concatenation and undo both norms.
        s.push(gpu.step(ids.split, &[&b.d_cat, &b.t1], &[rows, 2 * d, d, 0, 1, 1], n));
        s.push(gpu.dispatch(ids.norm_dx, &[x, &b.t1, &b.d_xn], &[d, rows, self.eps.to_bits()], Dispatch::Workgroups(rows)));
        s.push(gpu.step(ids.split, &[&b.d_cat, &b.t1], &[rows, 2 * d, d, d, 1, 1], n));
        s.push(gpu.dispatch(ids.norm_dx, &[m, &b.t1, &b.d_mn], &[d, rows, self.eps.to_bits()], Dispatch::Workgroups(rows)));
        // dx = dy * (1 - f) + d_xn; dm = dy * g + d_mn.
        s.push(gpu.step(ids.mul, &[dy, &a.forget, &b.t1], &[n], n));
        s.push(gpu.step(ids.sub, &[dy, &b.t1, &b.t2], &[n, 0, 0], n));
        s.push(gpu.step(ids.add, &[&b.t2, &b.d_xn, dx], &[n], n));
        s.push(gpu.step(ids.mul, &[dy, &a.gain, &b.t1], &[n], n));
        s.push(gpu.step(ids.add, &[&b.t1, &b.d_mn, dm], &[n], n));
    }
}
