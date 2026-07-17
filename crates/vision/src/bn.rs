// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A standalone BatchNorm unit.
//!
//! `Conv` owns one conv AND its BN, which covers almost everything — but not a BN
//! over a SUM of convs. ZipDepth's `MinimalMultiScale` is
//! `x + BN(dwconv_d1(x) + dwconv_d2(x))`: two branches, one BN on their sum. That
//! cannot be expressed by a unit whose BN is welded to a single conv.
//!
//! So the BN lives here, and `Conv` keeps its own — deliberately, because the two
//! are NOT the same code:
//!   * `Conv`'s BN is fusion-aware. Its whole eval fast path is `conv_act_reg`, a
//!     single kernel doing conv+BN+act from a collapsed `scale|bias`; the BN never
//!     runs as a separate dispatch there at all.
//!   * this one can never fuse (it has no conv to fuse WITH), and has no
//!     activation-tap site, so it is the strictly simpler subset.
//! Factoring `Conv` onto this type would force the fused path to reach back in for
//! `sb` and would gain nothing but indirection.
//!
//! What IS shared and must not drift: the eps (`vision::BN_EPS`), the packed
//! layouts (`mv[2c]=mean|var`, `gb[2c]=gamma|beta`, `mvg[3c]`), and the host
//! interleave between `bn_stats` and `bn_train`. `bn_matches_convs_batchnorm` in
//! the tests pins that this unit and `Conv`'s agree numerically.

use gpu_core::{f, DeviceBuffer};
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

/// BatchNorm over an NCHW map, train (batch stats) or eval (running stats).
pub struct BatchNorm {
    names: BnNames,
    pub shape: Shape,
    train: std::cell::Cell<bool>,
    update_running: std::cell::Cell<bool>,
    momentum: f32,
    ready: std::cell::Cell<bool>,

    mean: DeviceBuffer,
    var: DeviceBuffer,
    mv: DeviceBuffer,
    gb: DeviceBuffer,
    mvg: DeviceBuffer,
    out: DeviceBuffer,
    bp: DeviceBuffer,
}

impl BatchNorm {
    pub fn new(ctx: &Ctx, names: BnNames, shape: Shape, train: bool) -> BatchNorm {
        let c = shape.c;
        BatchNorm {
            names,
            shape,
            train: std::cell::Cell::new(train),
            update_running: std::cell::Cell::new(false),
            momentum: 0.1,
            ready: std::cell::Cell::new(false),
            mean: ctx.act(c),
            var: ctx.act(c),
            mv: ctx.act(2 * c),
            gb: ctx.act(2 * c),
            mvg: ctx.act(3 * c),
            out: ctx.act(shape.numel()),
            bp: ctx.act(5 * c),
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
        self.update_running.set(on);
    }
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let c = self.shape.c as usize;
        vec![
            (self.names.gamma.clone(), c),
            (self.names.beta.clone(), c),
            (self.names.run_mean.clone(), c),
            (self.names.run_var.clone(), c),
        ]
    }

    fn nchw(&self) -> [u32; 4] {
        [self.shape.n, self.shape.c, self.shape.h, self.shape.w]
    }
    fn pack_gb(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.shape.c as usize;
        let gamma = ctx.gpu.read(ps.w(&self.names.gamma), c);
        let beta = ctx.gpu.read(ps.w(&self.names.beta), c);
        ctx.gpu.write(&self.gb, bytemuck::cast_slice(&pack2(&gamma, &beta)));
    }
    /// Interleave the freshly-computed BATCH stats into `mv` (for `bn_train`) and
    /// `mvg` (for `bn_dstats`/`bn_dx`).
    ///
    /// This host round-trip between `bn_stats` and `bn_train` is the reason the
    /// forward submits in pieces rather than as one recorded replay: `bn_stats`
    /// emits `mean[C]`/`var[C]` separately, the consumers want them interleaved,
    /// and there is no interleave kernel. Collapsing the two submits reads STALE
    /// statistics — silently.
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
    /// Running stats into `mv` for `bn_eval` — which takes the SAME four buffers
    /// as `bn_train` (`x, mv, gb, out`), not a collapsed `scale|bias`.
    fn pack_running_mv(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.shape.c as usize;
        let rmean = ctx.gpu.read(ps.w(&self.names.run_mean), c);
        let rvar = ctx.gpu.read(ps.w(&self.names.run_var), c);
        ctx.gpu.write(&self.mv, bytemuck::cast_slice(&pack2(&rmean, &rvar)));
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer) {
        let (c, on) = (self.shape.c, self.shape.numel());
        if !self.train.get() {
            if !self.ready.get() {
                self.pack_running_mv(ctx, ps);
                self.pack_gb(ctx, ps);
                self.ready.set(true);
            }
            let s = ctx.step(ctx.ids.need(ctx.ids.bn_eval, "bn_eval"), &[x, &self.mv, &self.gb, &self.out], &self.nchw(), on);
            ctx.gpu.submit(&[], &[s]);
            return;
        }
        self.pack_gb(ctx, ps);
        let s_stats = ctx.step(ctx.ids.need(ctx.ids.bn_stats, "bn_stats"), &[x, &self.mean, &self.var], &self.nchw(), c);
        let mut pre = vec![s_stats];
        if self.update_running.get() {
            pre.push(ctx.step(
                ctx.ids.need(ctx.ids.bn_running, "bn_running"),
                &[&self.mean, &self.var, ps.w(&self.names.run_mean), ps.w(&self.names.run_var)],
                &[c, f(self.momentum)],
                c,
            ));
        }
        ctx.gpu.submit(&[], &pre);
        self.pack_stats_host(ctx, ps);
        let s = ctx.step(ctx.ids.need(ctx.ids.bn_train, "bn_train"), &[x, &self.mv, &self.gb, &self.out], &self.nchw(), on);
        ctx.gpu.submit(&[], &[s]);
    }

    /// `d_out` -> `d_in`, accumulating `gamma`/`beta` grads. Train mode only —
    /// eval-mode BN is a frozen affine and carries no parameter gradient.
    pub fn backward(&self, ctx: &Ctx, ps: &ParamStore, x: &DeviceBuffer, d_out: &DeviceBuffer, d_in: &DeviceBuffer) {
        let (c, on) = (self.shape.c, self.shape.numel());
        let s_dstats = ctx.step(ctx.ids.need(ctx.ids.bn_dstats, "bn_dstats"), &[x, d_out, &self.mvg, &self.bp], &self.nchw(), c);
        ctx.gpu.submit(&[], &[s_dstats]);
        let s_dgamma = ctx.step(ctx.ids.need(ctx.ids.bn_dgamma, "bn_dgamma"), &[x, d_out, &self.mv, ps.g(&self.names.gamma)], &self.nchw(), c);
        let s_dbeta = ctx.step(ctx.ids.need(ctx.ids.bn_dbeta, "bn_dbeta"), &[d_out, ps.g(&self.names.beta)], &self.nchw(), c);
        let s_dx = ctx.step(ctx.ids.need(ctx.ids.bn_dx, "bn_dx"), &[x, d_out, &self.bp, d_in], &self.nchw(), on);
        ctx.gpu.submit(&[], &[s_dgamma, s_dbeta, s_dx]);
    }
}
