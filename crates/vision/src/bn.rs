// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! BatchNorm: the train-mode core every BN stage runs, and a standalone unit.
//!
//! `Conv` owns one conv AND its BN, which covers almost everything - but not a BN
//! over a SUM of convs. ZipDepth's `MinimalMultiScale` is
//! `x + BN(dwconv_d1(x) + dwconv_d2(x))`: two branches, one BN on their sum. That
//! cannot be expressed by a unit whose BN is welded to a single conv, so
//! [`BatchNorm`] exists as its own unit.
//!
//! The two units share [`BnCore`] - the batch statistics, their packed forms,
//! the train-mode normalisation and the whole backward - so that part is ONE
//! implementation, and the eps, the packed layouts (`mv[2c]=mean|var`,
//! `gb[2c]=gamma|beta`, `mvg[3c]=mean|var|gamma`, `bp[5c]`) and the dispatch
//! order cannot drift between them. What `Conv` keeps for itself is its fused
//! inference path (`conv_act_reg` from a collapsed `scale|bias`), which has no
//! counterpart here: this unit has no conv to fuse with.

use gpu_core::{f, DeviceBuffer, Step};
use paramstore::ParamStore;

use crate::net::{Ctx, Shape};

fn pack2(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(2 * a.len());
    for i in 0..a.len() {
        v.push(a[i]);
        v.push(b[i]);
    }
    v
}

/// The four tensor names a BatchNorm owns.
#[derive(Clone, Debug)]
pub struct BnNames {
    pub gamma: String,
    pub beta: String,
    pub run_mean: String,
    pub run_var: String,
}

impl BnNames {
    /// torch's `nn.BatchNorm2d`: `P.{weight,bias,running_mean,running_var}`.
    pub fn torch(prefix: &str) -> BnNames {
        BnNames {
            gamma: format!("{prefix}.weight"),
            beta: format!("{prefix}.bias"),
            run_mean: format!("{prefix}.running_mean"),
            run_var: format!("{prefix}.running_var"),
        }
    }
    /// brain's spelling: `P.{gamma,beta,run_mean,run_var}`.
    pub fn brain(prefix: &str) -> BnNames {
        BnNames {
            gamma: format!("{prefix}.gamma"),
            beta: format!("{prefix}.beta"),
            run_mean: format!("{prefix}.run_mean"),
            run_var: format!("{prefix}.run_var"),
        }
    }
}

/// One BatchNorm stage's statistics, their packed forms and its train-mode
/// passes - shared by [`BatchNorm`] and `Conv`.
pub(crate) struct BnCore {
    pub(crate) names: BnNames,
    shape: Shape,
    momentum: f32,
    /// Apply the running-stat momentum EMA during a train-mode forward. Off by
    /// default: the gradient check needs a deterministic forward, and the
    /// running stats carry no train-mode gradient anyway. Real training turns
    /// it on so `run_mean`/`run_var` track the data for eval-mode inference.
    pub(crate) update_running: std::cell::Cell<bool>,
    mean: DeviceBuffer,
    var: DeviceBuffer,
    /// `mean|var` interleaved: batch stats in train mode, running stats for
    /// `bn_eval`.
    pub(crate) mv: DeviceBuffer,
    /// `gamma|beta` interleaved.
    pub(crate) gb: DeviceBuffer,
    mvg: DeviceBuffer,
    bp: DeviceBuffer,
}

impl BnCore {
    pub(crate) fn new(ctx: &Ctx, names: BnNames, shape: Shape) -> BnCore {
        let c = shape.c;
        BnCore {
            names,
            shape,
            // Running-stat EMA momentum. PyTorch's default (0.03) converges too
            // slowly for the short from-scratch runs here, so use 0.1 - the BN
            // running mean/var reach usable eval-mode values in a few hundred
            // steps (validated by yolov8's p11 eval-inference test).
            momentum: 0.1,
            update_running: std::cell::Cell::new(false),
            mean: ctx.act(c),
            var: ctx.act(c),
            mv: ctx.act(2 * c),
            gb: ctx.act(2 * c),
            mvg: ctx.act(3 * c),
            bp: ctx.act(5 * c),
        }
    }

    fn nchw(&self) -> [u32; 4] {
        [self.shape.n, self.shape.c, self.shape.h, self.shape.w]
    }

    pub(crate) fn param_list(&self) -> Vec<(String, usize)> {
        let c = self.shape.c as usize;
        vec![
            (self.names.gamma.clone(), c),
            (self.names.beta.clone(), c),
            (self.names.run_mean.clone(), c),
            (self.names.run_var.clone(), c),
        ]
    }

    /// Pack `gamma|beta` into `gb` on the host.
    pub(crate) fn pack_gb(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.shape.c as usize;
        let gamma = ctx.gpu.read(ps.w(&self.names.gamma), c);
        let beta = ctx.gpu.read(ps.w(&self.names.beta), c);
        ctx.gpu.write(&self.gb, bytemuck::cast_slice(&pack2(&gamma, &beta)));
    }

    /// Interleave the RUNNING mean/var into `mv` for `bn_eval` (which shares
    /// `bn_train`'s signature and simply expects running stats there).
    pub(crate) fn pack_running_mv(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.shape.c as usize;
        let rmean = ctx.gpu.read(ps.w(&self.names.run_mean), c);
        let rvar = ctx.gpu.read(ps.w(&self.names.run_var), c);
        ctx.gpu.write(&self.mv, bytemuck::cast_slice(&pack2(&rmean, &rvar)));
    }

    /// The host fallback of `bn_pack`: interleave the freshly computed BATCH
    /// stats into `mv` (for `bn_train`) and `mvg` (for `bn_dstats`/`bn_dx`).
    /// It reads the statistics back, so the steps that produced them must
    /// already have been submitted and the ones that consume the packing must
    /// not be - see [`Self::forward_train`].
    fn pack_stats_host(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.shape.c as usize;
        let mean = ctx.gpu.read(&self.mean, c);
        let var = ctx.gpu.read(&self.var, c);
        let gamma = ctx.gpu.read(ps.w(&self.names.gamma), c);
        ctx.gpu.write(&self.mv, bytemuck::cast_slice(&pack2(&mean, &var)));
        let mut mvg = Vec::with_capacity(3 * c);
        for i in 0..c {
            mvg.push(mean[i]);
            mvg.push(var[i]);
            mvg.push(gamma[i]);
        }
        ctx.gpu.write(&self.mvg, bytemuck::cast_slice(&mvg));
    }

    /// Train-mode BatchNorm of `x` into `out`: batch statistics, the running-
    /// stat update when [`Self::update_running`] is on, the packing and the
    /// normalisation. `lead` are the caller's steps that produce `x` (a conv),
    /// `tail` its steps that consume `out` (an activation); everything is
    /// submitted in dependency order.
    ///
    /// With `bn_pack` registered the packing happens on the device and the
    /// whole stage is ONE submission. Without it the statistics are read back
    /// and interleaved on the host between two submissions - correct, but a
    /// device drain per call. Collapsing the host path into one submission
    /// would read STALE statistics, silently.
    pub(crate) fn forward_train(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, out: &DeviceBuffer, lead: Vec<Step>, tail: Vec<Step>) {
        let c = self.shape.c;
        let mut steps = lead;
        steps.push(ctx.step(ctx.ids.need(ctx.ids.bn_stats, "bn_stats"), &[x, &self.mean, &self.var], &self.nchw(), c));
        if self.update_running.get() {
            steps.push(ctx.step(
                ctx.ids.need(ctx.ids.bn_running, "bn_running"),
                &[&self.mean, &self.var, ps.w(&self.names.run_mean), ps.w(&self.names.run_var)],
                &[c, f(self.momentum)],
                c,
            ));
        }
        let s_train = ctx.step(ctx.ids.need(ctx.ids.bn_train, "bn_train"), &[x, &self.mv, &self.gb, out], &self.nchw(), self.shape.numel());
        if ctx.ids.bn_pack != crate::NONE {
            steps.push(ctx.step(
                ctx.ids.bn_pack,
                &[&self.mean, &self.var, ps.w(&self.names.gamma), ps.w(&self.names.beta), &self.mv, &self.gb, &self.mvg],
                &[c],
                c,
            ));
            steps.push(s_train);
            steps.extend(tail);
            ctx.gpu.submit(&[], &steps);
            return;
        }
        self.pack_gb(ctx, ps);
        ctx.gpu.submit(&[], &steps);
        self.pack_stats_host(ctx, ps);
        let mut rest = vec![s_train];
        rest.extend(tail);
        ctx.gpu.submit(&[], &rest);
    }

    /// The eval-mode normalisation step `bn_eval(x) -> out` with the fused
    /// activation selector `act`; `mv`/`gb` must hold the running stats and the
    /// affine params (see [`Self::pack_running_mv`], [`Self::pack_gb`]).
    pub(crate) fn eval_step(&self, ctx: &Ctx, x: &DeviceBuffer, out: &DeviceBuffer, act: u32) -> Step {
        let mut params = self.nchw().to_vec();
        params.push(act);
        ctx.step(ctx.ids.need(ctx.ids.bn_eval, "bn_eval"), &[x, &self.mv, &self.gb, out], &params, self.shape.numel())
    }

    /// The train-mode backward, `d_out` (grad wrt the BN output) -> `d_in`
    /// (grad wrt `x`, overwritten), accumulating the `gamma`/`beta` grads into
    /// their pre-zeroed buffers. Steps in dependency order, for the caller to
    /// submit (after whatever produced `d_out`).
    ///
    /// With `bn_dparams` registered the parameter grads are read out of the
    /// sums `bn_dstats` already made; otherwise `bn_dgamma`/`bn_dbeta`
    /// recompute them with two more passes over the activations.
    pub(crate) fn backward_steps(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) -> Vec<Step> {
        let c = self.shape.c;
        let (g_gamma, g_beta) = (ps.g(&self.names.gamma), ps.g(&self.names.beta));
        let mut steps = vec![ctx.step(ctx.ids.need(ctx.ids.bn_dstats, "bn_dstats"), &[x, d_out, &self.mvg, &self.bp], &self.nchw(), c)];
        if ctx.ids.bn_dparams != crate::NONE {
            steps.push(ctx.step(ctx.ids.bn_dparams, &[&self.bp, g_gamma, g_beta], &[c], c));
        } else {
            steps.push(ctx.step(ctx.ids.need(ctx.ids.bn_dgamma, "bn_dgamma"), &[x, d_out, &self.mv, g_gamma], &self.nchw(), c));
            steps.push(ctx.step(ctx.ids.need(ctx.ids.bn_dbeta, "bn_dbeta"), &[d_out, g_beta], &self.nchw(), c));
        }
        steps.push(ctx.step(ctx.ids.need(ctx.ids.bn_dx, "bn_dx"), &[x, d_out, &self.bp, d_in], &self.nchw(), self.shape.numel()));
        steps
    }
}

/// BatchNorm over an NCHW map, train (batch stats) or eval (running stats).
pub struct BatchNorm {
    core: BnCore,
    pub shape: Shape,
    train: std::cell::Cell<bool>,
    ready: std::cell::Cell<bool>,
    out: DeviceBuffer,
}

impl BatchNorm {
    pub fn new(ctx: &Ctx, names: BnNames, shape: Shape, train: bool) -> BatchNorm {
        BatchNorm {
            core: BnCore::new(ctx, names, shape),
            shape,
            train: std::cell::Cell::new(train),
            ready: std::cell::Cell::new(false),
            out: ctx.act(shape.numel()),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }
    pub fn set_eval(&self, on: bool) {
        self.train.set(!on);
        if !on {
            // Re-entering train mode invalidates the cached eval packing.
            self.ready.set(false);
        }
    }
    pub fn set_update_running(&self, on: bool) {
        self.core.update_running.set(on);
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        self.core.param_list()
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
        self.forward_act(ctx, ps, x, 0);
    }

    /// Forward with a fused activation in EVAL mode (`act` codes as the
    /// `conv_act*`/`bn_eval` selector: 0 identity, 1 relu, 2 silu, 3 sigmoid).
    /// Returns `true` when the activation was applied - the caller then skips
    /// its own activation dispatch. The TRAIN path ignores `act` and returns
    /// `false`: training needs the pre-activation output as a backward cache,
    /// so the caller keeps its separate activation there.
    pub fn forward_act(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, act: u32) -> bool {
        if !self.train.get() {
            if !self.ready.get() {
                self.core.pack_running_mv(ctx, ps);
                self.core.pack_gb(ctx, ps);
                self.ready.set(true);
            }
            ctx.gpu.submit(&[], &[self.core.eval_step(ctx, x, &self.out, act)]);
            return true;
        }
        self.core.forward_train(ctx, ps, x, &self.out, Vec::new(), Vec::new());
        false
    }

    /// `d_out` -> `d_in`, accumulating `gamma`/`beta` grads. Train mode only -
    /// eval-mode BN is a frozen affine and carries no parameter gradient.
    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        ctx.gpu.submit(&[], &self.core.backward_steps(ctx, ps, x, d_out, d_in));
    }
}
