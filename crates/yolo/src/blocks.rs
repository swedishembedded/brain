// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8 convolutional building blocks (P2): `Conv`, `Bottleneck`, `C2f`,
//! `SPPF`. NCHW throughout.
//!
//! ## Block abstraction (the pattern, mirroring `gpt::Gpt`)
//!
//! Every block is constructed once ([`Conv::new`] etc.): it registers its
//! parameters (see [`Conv::param_list`]) and pre-allocates its SSA activation +
//! backward-temporary buffers from [`Ctx::act`]. Thereafter it offers a
//! `forward(ctx, ps, x_in) -> &out` and a `backward(ctx, ps, x_in, d_out, d_in)`
//! that record + submit their dispatch [`Step`]s.
//!
//! Why imperative submit (vs. one pre-recorded replay like `gpt`): BatchNorm's
//! `bn_train`/`bn_dx` kernels read the per-channel stats as INTERLEAVED packed
//! buffers (`mv[2C]` = mean|var, `mvg[3C]` = mean|var|gamma), but `bn_stats`
//! emits `mean[C]`/`var[C]` as separate tensors and there is no interleave
//! kernel (and P2 must add none). So between `bn_stats` and `bn_train` we must
//! interleave on the host. A model can't splice a host write into the middle of
//! a single recorded `submit`, so each block submits its forward in the natural
//! data-dependency order, host-packing the BN stats at the one boundary where it
//! is required. Backward is likewise submitted block-by-block. Buffers are still
//! SSA and the grad-accumulating kernels (`*_dw`, `bn_dgamma/beta`, residual
//! `add2`) compose exactly as in `gpt`.
//!
//! SSA discipline: every forward stage writes a FRESH buffer that doubles as the
//! activation cache the backward reads. Residual / multi-consumer grads
//! accumulate out-of-place via `add2`.
//!
//! Buffers: weights + grads come from a [`ParamStore`] keyed by the names each
//! block registers; activations + backward temporaries are plain
//! [`Ctx::act`] storage.
//!
//! Param-naming scheme (feeds the pretrained-weight mapping later), under a
//! caller-supplied prefix `P`:
//!   * `Conv`: `P.conv.weight [Cout,Cin,K,K]` (bias-free), `P.bn.gamma [C]`,
//!     `P.bn.beta [C]`, `P.bn.run_mean [C]`, `P.bn.run_var [C]`.
//!   * `Bottleneck`: convs `P.cv1`, `P.cv2`.
//!   * `C2f`: `P.cv1` (in 1x1), `P.cv2` (out 1x1), bottlenecks `P.m.{i}`.
//!   * `SPPF`: `P.cv1` (in 1x1), `P.cv2` (out 1x1).
//! This matches Ultralytics' `cv1`/`cv2`/`m` naming for later string-mapping.

use gpu_core::{f, DeviceBuffer};
use paramstore::ParamStore;

use crate::net::{Ctx, Shape};
use crate::net::{
    ADD2, BN_DBETA, BN_DGAMMA, BN_DSTATS, BN_DX, BN_RUNNING, BN_STATS, BN_TRAIN, CHAN_PLACE, CONCAT2,
    CONCAT_SPLIT, CONV2D, CONV2D_DW, CONV2D_DX, CONV_ACT, CONV_ACT_REG, CONV_ACT_TILED, MAXPOOL5,
    MAXPOOL5_DX, SILU, SILU_BWD,
};

/// Whether to use the weight-staged (workgroup-memory) tiled conv on the GPU.
/// Off by default: staging the full weight tile costs up to ~10–32 KiB of
/// workgroup memory, which collapses GPU occupancy and was measured SLOWER than
/// the naive conv on Intel Arc. Kept opt-in (`BRAIN_TILED_CONV=1`) for the
/// follow-up proper input+weight tiled GEMM. The work-group JIT path (solution
/// B) it exercises is correct either way.
fn use_tiled_conv() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRAIN_TILED_CONV").map(|v| v != "0").unwrap_or(false))
}

/// `BRAIN_NAIVE_CONV=1` forces the naive one-output-per-invocation fused conv
/// (the previous default) instead of the register-tiled one — for comparison.
fn use_naive_conv() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRAIN_NAIVE_CONV").map(|v| v != "0").unwrap_or(false))
}

/// Interleave two per-channel vectors into a `[2C]` packed buffer.
fn pack2(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(2 * a.len());
    for i in 0..a.len() {
        v.push(a[i]);
        v.push(b[i]);
    }
    v
}

// ===========================================================================
// Conv = conv2d (bias-free) -> BatchNorm -> SiLU
// ===========================================================================

/// A single `Conv` unit. Supports stride 1/2 and K=3 (pad 1) / K=1 (pad 0).
pub struct Conv {
    prefix: String,
    pub in_shape: Shape,
    pub out_shape: Shape,
    k: u32,
    stride: u32,
    pad: u32,
    /// Train-mode BN (batch stats) vs eval-mode BN (running stats). Interior-
    /// mutable so inference can flip every Conv to eval-mode BN via
    /// [`Conv::set_eval`] WITHOUT rebuilding the graph (the only thing the flag
    /// changes is which BN kernel `forward` dispatches; see P6 infer).
    train: std::cell::Cell<bool>,
    momentum: f32,
    /// Apply the running-stat momentum EMA update during forward. Interior-mutable
    /// (like `train`) so real training can enable it via [`Conv::set_update_running`]
    /// without rebuilding the graph. Disabled for the gradient check (it mutates
    /// `run_mean`/`run_var`, breaking forward determinism; those tensors carry no
    /// train-mode gradient anyway), so it defaults OFF.
    update_running: std::cell::Cell<bool>,

    conv_out: DeviceBuffer, // post-conv pre-BN [out]
    mean: DeviceBuffer,     // batch mean [C]
    var: DeviceBuffer,      // batch var  [C]
    mv: DeviceBuffer,       // packed mean|var [2C]
    gb: DeviceBuffer,       // packed gamma|beta [2C]
    sb: DeviceBuffer,       // packed scale|bias [2C] = BN-eval collapsed (fused conv_act)
    /// Whether `sb` holds the current BN-eval collapse. Computed lazily on the
    /// first eval-mode forward and reused across frames (constant in inference);
    /// invalidated when the block re-enters train mode.
    sb_ready: std::cell::Cell<bool>,
    mvg: DeviceBuffer,      // packed mean|var|gamma [3C]
    bn_out: DeviceBuffer,   // post-BN pre-SiLU [out]
    act: DeviceBuffer,      // SiLU output (block output) [out]

    d_bn: DeviceBuffer,   // grad wrt bn_out [out]
    bp: DeviceBuffer,     // packed [5C] from bn_dstats
    d_conv: DeviceBuffer, // grad wrt conv_out [out]

    /// Lazily-allocated [in] scratch holding the tapped (possibly fake-quantized)
    /// conv input, used only when a [`crate::net::ActTap`] is installed (NPU
    /// calibration / fake-quant). Never allocated on the normal inference path.
    q_in: std::cell::RefCell<Option<DeviceBuffer>>,
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &Ctx,
        prefix: &str,
        in_shape: Shape,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        train: bool,
    ) -> Conv {
        let out_shape = in_shape.conv_out(cout, k, stride, pad);
        let on = out_shape.numel();
        let c = cout;
        Conv {
            prefix: prefix.to_string(),
            in_shape,
            out_shape,
            k,
            stride,
            pad,
            train: std::cell::Cell::new(train),
            // Running-stat EMA momentum. PyTorch's default (0.03) converges too
            // slowly for the short from-scratch runs here, so use 0.1 — the BN
            // running mean/var reach usable eval-mode values in a few hundred
            // steps (validated by the p11 eval-inference test).
            momentum: 0.1,
            update_running: std::cell::Cell::new(false),
            conv_out: ctx.act(on),
            mean: ctx.act(c),
            var: ctx.act(c),
            mv: ctx.act(2 * c),
            gb: ctx.act(2 * c),
            sb: ctx.act(2 * c),
            sb_ready: std::cell::Cell::new(false),
            mvg: ctx.act(3 * c),
            bn_out: ctx.act(on),
            act: ctx.act(on),
            d_bn: ctx.act(on),
            bp: ctx.act(5 * c),
            d_conv: ctx.act(on),
            q_in: std::cell::RefCell::new(None),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        &self.act
    }

    /// Flip this Conv's BN to eval-mode (running stats) or train-mode (batch
    /// stats). Inference-only concern: it changes which BN kernel `forward`
    /// dispatches, never the graph or any buffer.
    pub fn set_eval(&self, eval: bool) {
        // Re-entering train mode invalidates the cached BN-eval collapse so the
        // next eval recomputes it from the (now updated) running stats / affine.
        if !eval {
            self.sb_ready.set(false);
        }
        self.train.set(!eval);
    }

    /// True iff this Conv is in eval-mode BN (running stats).
    pub fn is_eval(&self) -> bool {
        !self.train.get()
    }

    /// Enable/disable the BN running-stat momentum EMA update during train-mode
    /// forward. Must be ON during real training so `run_mean`/`run_var` track the
    /// data and eval-mode inference works; left OFF for the gradient check.
    pub fn set_update_running(&self, on: bool) {
        self.update_running.set(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let c = self.out_shape.c as usize;
        let cin = self.in_shape.c as usize;
        let k = self.k as usize;
        let p = |s: &str| format!("{}.{s}", self.prefix);
        vec![
            (p("conv.weight"), c * cin * k * k),
            (p("bn.gamma"), c),
            (p("bn.beta"), c),
            (p("bn.run_mean"), c),
            (p("bn.run_var"), c),
        ]
    }

    fn conv_params(&self) -> [u32; 10] {
        [
            self.in_shape.n,
            self.in_shape.c,
            self.in_shape.h,
            self.in_shape.w,
            self.out_shape.c,
            self.k,
            self.stride,
            self.pad,
            self.out_shape.h,
            self.out_shape.w,
        ]
    }
    /// Invocation count for the weight-tiled conv: one workgroup (64 invocations)
    /// per `(n, output-channel, 64-output-position block)`.
    fn tiled_threads(&self) -> u32 {
        let psz = self.out_shape.h * self.out_shape.w;
        let blocks = psz.div_ceil(64);
        self.out_shape.n * self.out_shape.c * blocks * 64
    }
    fn nchw(&self) -> [u32; 4] {
        [self.out_shape.n, self.out_shape.c, self.out_shape.h, self.out_shape.w]
    }

    /// Pack gamma|beta into `gb` from the current weights (host). BN affine
    /// params can't be aliased as the interleaved buffer the kernel wants.
    fn pack_gb(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.out_shape.c as usize;
        let p = |s: &str| format!("{}.{s}", self.prefix);
        let gamma = ctx.gpu.read(ps.w(&p("bn.gamma")), c);
        let beta = ctx.gpu.read(ps.w(&p("bn.beta")), c);
        ctx.gpu.write(&self.gb, bytemuck::cast_slice(&pack2(&gamma, &beta)));
    }

    /// Run the full forward and return this block's output buffer. Submits in
    /// dependency order, host-packing the BN stats at the one required boundary.
    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        let p = |s: &str| format!("{}.{s}", self.prefix);
        let on = self.out_shape.numel();
        let c = self.out_shape.c;

        if !self.train.get() {
            // Inference: one fused conv -> BN(eval) -> SiLU dispatch. The BN-eval
            // transform is collapsed per channel into `sb` once (constant across
            // frames), so there is no per-frame host stat packing nor separate
            // bn_eval/silu passes.
            if !self.sb_ready.get() {
                self.pack_sb(ctx, ps);
                self.sb_ready.set(true);
            }
            // Fused conv -> BN(eval) -> SiLU. The weight-tiled variant (opt-in)
            // exercises the single-source work-group kernel; the default naive
            // variant is faster on current GPUs (full occupancy). Both route to
            // the same native AVX2 fast path on CPU.
            let (kind, threads) = if use_tiled_conv() {
                (CONV_ACT_TILED, self.tiled_threads())
            } else if use_naive_conv() {
                (CONV_ACT, self.out_shape.numel())
            } else {
                // Default: register-tiled — each invocation computes an 8x4 tile
                // (8 output channels x 4 positions), reusing weight + input loads
                // across it (no workgroup memory, full occupancy). Each strided
                // NCHW input load feeds all 8 channels -> input traffic ~/8.
                // threads = N * ceil(Cout/8) * ceil(Ho*Wo/4).
                let ntc = self.out_shape.c.div_ceil(8);
                let npq = (self.out_shape.h * self.out_shape.w).div_ceil(4);
                (CONV_ACT_REG, self.out_shape.n * ntc * npq)
            };
            // Calibration / fake-quant tap (NPU INT8): route the conv input
            // through the host so the tap can read its range and/or rewrite it
            // (quant→dequant), then convolve the tapped copy. Only taken when a
            // tap is installed; the normal inference path skips this entirely.
            if let Some(tap) = ctx.tap {
                let in_n = self.in_shape.numel() as usize;
                let mut h = ctx.gpu.read(x_in, in_n);
                tap.tap(&self.prefix, &mut h);
                if self.q_in.borrow().is_none() {
                    *self.q_in.borrow_mut() = Some(ctx.gpu.storage(in_n as u64));
                }
                let q = self.q_in.borrow();
                let qbuf = q.as_ref().unwrap();
                ctx.gpu.write(qbuf, bytemuck::cast_slice(&h));
                let s = ctx.step(
                    kind,
                    &[qbuf, ps.w(&p("conv.weight")), &self.sb, &self.act],
                    &self.conv_params(),
                    threads,
                );
                ctx.gpu.submit(&[], &[s]);
                return;
            }
            let s = ctx.step(
                kind,
                &[x_in, ps.w(&p("conv.weight")), &self.sb, &self.act],
                &self.conv_params(),
                threads,
            );
            ctx.gpu.submit(&[], &[s]);
            return;
        }

        // Train mode: conv -> bn_stats, host-pack mv/mvg, then bn_train -> silu.
        self.pack_gb(ctx, ps);
        let s_conv = ctx.step(CONV2D, &[x_in, ps.w(&p("conv.weight")), &self.conv_out], &self.conv_params(), on);
        let s_stats = ctx.step(BN_STATS, &[&self.conv_out, &self.mean, &self.var], &self.nchw(), c);
        let mut pre = vec![s_conv, s_stats];
        if self.update_running.get() {
            pre.push(ctx.step(
                BN_RUNNING,
                &[&self.mean, &self.var, ps.w(&p("bn.run_mean")), ps.w(&p("bn.run_var"))],
                &[c, f(self.momentum)],
                c,
            ));
        }
        ctx.gpu.submit(&[], &pre);
        self.pack_stats_host(ctx, ps);
        let s_train = ctx.step(BN_TRAIN, &[&self.conv_out, &self.mv, &self.gb, &self.bn_out], &self.nchw(), on);
        let s_silu = ctx.step(SILU, &[&self.bn_out, &self.act], &[on], on);
        ctx.gpu.submit(&[], &[s_train, s_silu]);
    }

    /// Collapse the BN-eval transform into per-channel `scale|bias` packed in
    /// `sb` (`sb[2c]=gamma/sqrt(run_var+eps)`, `sb[2c+1]=beta-run_mean*scale`),
    /// the constant the fused `conv_act` kernel consumes. Eps matches bn_eval.
    fn pack_sb(&self, ctx: &Ctx, ps: &ParamStore) {
        let c = self.out_shape.c as usize;
        let p = |s: &str| format!("{}.{s}", self.prefix);
        let gamma = ctx.gpu.read(ps.w(&p("bn.gamma")), c);
        let beta = ctx.gpu.read(ps.w(&p("bn.beta")), c);
        let rmean = ctx.gpu.read(ps.w(&p("bn.run_mean")), c);
        let rvar = ctx.gpu.read(ps.w(&p("bn.run_var")), c);
        let mut sb = Vec::with_capacity(2 * c);
        for i in 0..c {
            let scale = gamma[i] / (rvar[i] + 1e-5).sqrt();
            sb.push(scale);
            sb.push(beta[i] - rmean[i] * scale);
        }
        ctx.gpu.write(&self.sb, bytemuck::cast_slice(&sb));
    }

    /// Interleave the freshly-computed batch mean/var into `mv` and mean|var|gamma
    /// into `mvg` (the BN-backward input). Called between `bn_stats` and
    /// `bn_train` during forward.
    fn pack_stats_host(&self, ctx: &Ctx, ps: &ParamStore) {
        let p = |s: &str| format!("{}.{s}", self.prefix);
        let c = self.out_shape.c as usize;
        let mean = ctx.gpu.read(&self.mean, c);
        let var = ctx.gpu.read(&self.var, c);
        let gamma = ctx.gpu.read(ps.w(&p("bn.gamma")), c);
        ctx.gpu.write(&self.mv, bytemuck::cast_slice(&pack2(&mean, &var)));
        let mut mvg = Vec::with_capacity(3 * c);
        for i in 0..c {
            mvg.push(mean[i]);
            mvg.push(var[i]);
            mvg.push(gamma[i]);
        }
        ctx.gpu.write(&self.mvg, bytemuck::cast_slice(&mvg));
    }

    /// Backward. `d_out` = grad wrt this block's output; `d_in` receives the grad
    /// wrt `x_in` (overwritten). Param grads accumulate into the ParamStore.
    /// Assumes `forward` already ran (caches + `mv`/`mvg`/`bp` populated).
    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        let p = |s: &str| format!("{}.{s}", self.prefix);
        let on = self.out_shape.numel();
        let c = self.out_shape.c;
        let dw_n = self.out_shape.c * self.in_shape.c * self.k * self.k;

        let s_silu = ctx.step(SILU_BWD, &[&self.bn_out, d_out, &self.d_bn], &[on], on);
        let s_dstats = ctx.step(BN_DSTATS, &[&self.conv_out, &self.d_bn, &self.mvg, &self.bp], &self.nchw(), c);
        // bn_dgamma / bn_dbeta accumulate -> their grad buffers are pre-zeroed by
        // the model's zero_grads (clears list), exactly like gpt.
        let s_dgamma = ctx.step(BN_DGAMMA, &[&self.conv_out, &self.d_bn, &self.mv, ps.g(&p("bn.gamma"))], &self.nchw(), c);
        let s_dbeta = ctx.step(BN_DBETA, &[&self.d_bn, ps.g(&p("bn.beta"))], &self.nchw(), c);
        let s_dx = ctx.step(BN_DX, &[&self.conv_out, &self.d_bn, &self.bp, &self.d_conv], &self.nchw(), on);
        let s_dw = ctx.step(CONV2D_DW, &[&self.d_conv, x_in, ps.g(&p("conv.weight"))], &self.conv_params(), dw_n);
        let s_dxin = ctx.step(CONV2D_DX, &[&self.d_conv, ps.w(&p("conv.weight")), d_in], &self.conv_params(), self.in_shape.numel());
        // s_silu must precede s_dstats/s_dgamma/s_dbeta (they read d_bn); s_dstats
        // must precede s_dx (reads bp). Submit in this order.
        ctx.gpu.submit(&[], &[s_silu, s_dstats, s_dgamma, s_dbeta, s_dx, s_dw, s_dxin]);
    }
}

// ===========================================================================
// Bottleneck = Conv(K3,s1) -> Conv(K3,s1) [+ residual]
// ===========================================================================

/// CSP bottleneck. Two K3/s1 convs; optional residual `add2` shortcut when
/// `c_in == c_out && shortcut`.
pub struct Bottleneck {
    pub cv1: Conv,
    pub cv2: Conv,
    pub shortcut: bool,
    pub in_shape: Shape,
    pub out_shape: Shape,
    sum: DeviceBuffer,   // residual sum (output when shortcut) [out]
    d_mid: DeviceBuffer, // grad wrt cv1.out [out]
}

impl Bottleneck {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, c_out: u32, shortcut: bool, train: bool) -> Bottleneck {
        let cv1 = Conv::new(ctx, &format!("{prefix}.cv1"), in_shape, c_out, 3, 1, 1, train);
        let cv2 = Conv::new(ctx, &format!("{prefix}.cv2"), cv1.out_shape, c_out, 3, 1, 1, train);
        let out_shape = cv2.out_shape;
        let use_short = shortcut && in_shape.c == c_out;
        let on = out_shape.numel();
        Bottleneck {
            cv1,
            cv2,
            shortcut: use_short,
            in_shape,
            out_shape,
            sum: ctx.act(on),
            d_mid: ctx.act(on),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        if self.shortcut {
            &self.sum
        } else {
            self.cv2.out()
        }
    }

    /// Propagate the eval/train BN toggle to both convs.
    pub fn set_eval(&self, eval: bool) {
        self.cv1.set_eval(eval);
        self.cv2.set_eval(eval);
    }

    /// Propagate the BN running-stat update toggle to both convs.
    pub fn set_update_running(&self, on: bool) {
        self.cv1.set_update_running(on);
        self.cv2.set_update_running(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.cv1.param_list();
        v.extend(self.cv2.param_list());
        v
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.cv1.forward(ctx, ps, x_in);
        self.cv2.forward(ctx, ps, self.cv1.out());
        if self.shortcut {
            let on = self.out_shape.numel();
            let s = ctx.step(ADD2, &[x_in, self.cv2.out(), &self.sum], &[on], on);
            ctx.gpu.submit(&[], &[s]);
        }
    }

    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        // output = (shortcut ? x_in + cv2 : cv2). In both cases d(cv2_out)=d_out.
        self.cv2.backward(ctx, ps, self.cv1.out(), d_out, &self.d_mid);
        self.cv1.backward(ctx, ps, x_in, &self.d_mid, d_in);
        if self.shortcut {
            let on = self.in_shape.numel();
            let s = ctx.step(ADD2, &[d_in, d_out, d_in], &[on], on);
            ctx.gpu.submit(&[], &[s]);
        }
    }
}

// ===========================================================================
// C2f = Conv1x1(2c) -> split [y0|y1] -> y1 thru n Bottlenecks -> concat -> Conv1x1
// ===========================================================================

/// CSP "C2f" block (`c = C_out/2`): a 1x1 conv expands to `2c` channels, split
/// into halves `y0`/`y1`; `y1` runs through `n` bottlenecks (each output
/// retained); everything `[y0, y1, b1..bn]` is concatenated along C and a final
/// 1x1 conv projects to `C_out`.
pub struct C2f {
    pub cv1: Conv, // in: Cin -> 2c
    pub cv2: Conv, // out: (2+n)*c -> Cout
    pub m: Vec<Bottleneck>,
    pub in_shape: Shape,
    pub out_shape: Shape,
    c: u32, // half width
    n: u32, // bottleneck count
    sh: Shape, // spatial shape of each c-channel chunk

    // forward caches
    y0: DeviceBuffer,       // first half [n,c,h,w]
    y1: DeviceBuffer,       // second half [n,c,h,w]
    concat: DeviceBuffer,   // [n,(2+n)*c,h,w]
    // intermediate concat buffers (left-fold of concat2)
    cat_tmp: Vec<DeviceBuffer>,

    // backward caches
    d_split: DeviceBuffer,  // grad wrt cv1.out ([n,2c,h,w]); chunk grads scattered in
    d_chunk: Vec<DeviceBuffer>, // grad wrt each chunk [y0,y1,b1..bn]
    d_y1: DeviceBuffer,     // accumulated grad wrt y1 (from chain + concat slice)
}

impl C2f {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, c_out: u32, n: u32, shortcut: bool, train: bool) -> C2f {
        assert!(c_out % 2 == 0, "C2f C_out must be even");
        let c = c_out / 2;
        let cv1 = Conv::new(ctx, &format!("{prefix}.cv1"), in_shape, 2 * c, 1, 1, 0, train);
        let sh = Shape::new(cv1.out_shape.n, c, cv1.out_shape.h, cv1.out_shape.w);
        let mut m = Vec::new();
        let mut prev = sh;
        for i in 0..n {
            let b = Bottleneck::new(ctx, &format!("{prefix}.m.{i}"), prev, c, shortcut, train);
            prev = b.out_shape;
            m.push(b);
        }
        let cat_c = (2 + n) * c;
        let cat_shape = Shape::new(sh.n, cat_c, sh.h, sh.w);
        let cv2 = Conv::new(ctx, &format!("{prefix}.cv2"), cat_shape, c_out, 1, 1, 0, train);
        let out_shape = cv2.out_shape;
        let chunk_n = sh.numel();

        // left-fold concat needs (chunks-1) intermediate buffers; the last is the
        // full concat. chunks = 2 + n.
        let chunks = 2 + n;
        let mut cat_tmp = Vec::new();
        for k in 2..=chunks {
            cat_tmp.push(ctx.act(k * c * sh.h * sh.w * sh.n));
        }
        let mut d_chunk = Vec::new();
        for _ in 0..chunks {
            d_chunk.push(ctx.act(chunk_n));
        }
        C2f {
            cv1,
            cv2,
            m,
            in_shape,
            out_shape,
            c,
            n,
            sh,
            y0: ctx.act(chunk_n),
            y1: ctx.act(chunk_n),
            concat: ctx.act(cat_c * sh.h * sh.w * sh.n),
            cat_tmp,
            d_split: ctx.act(2 * chunk_n),
            d_chunk,
            d_y1: ctx.act(chunk_n),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        self.cv2.out()
    }

    /// Propagate the eval/train BN toggle to all convs + bottlenecks.
    pub fn set_eval(&self, eval: bool) {
        self.cv1.set_eval(eval);
        for b in &self.m {
            b.set_eval(eval);
        }
        self.cv2.set_eval(eval);
    }

    /// Propagate the BN running-stat update toggle to all convs + bottlenecks.
    pub fn set_update_running(&self, on: bool) {
        self.cv1.set_update_running(on);
        for b in &self.m {
            b.set_update_running(on);
        }
        self.cv2.set_update_running(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.cv1.param_list();
        for b in &self.m {
            v.extend(b.param_list());
        }
        v.extend(self.cv2.param_list());
        v
    }

    fn split_params(&self, c_off: u32) -> [u32; 6] {
        // concat_split ABI: [N, Ctot, Csrc, c_off, H, W]
        [self.sh.n, 2 * self.c, self.c, c_off, self.sh.h, self.sh.w]
    }

    /// All retained chunk buffers in concat order `[y0, y1, b1..bn]`.
    fn chunks(&self) -> Vec<&DeviceBuffer> {
        let mut v = vec![&self.y0, &self.y1];
        for b in &self.m {
            v.push(b.out());
        }
        v
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        // 1x1 conv -> 2c channels.
        self.cv1.forward(ctx, ps, x_in);
        // split into y0 (c_off=0) and y1 (c_off=c) via concat_split copies.
        let chunk_n = self.sh.numel();
        let s0 = ctx.step(CONCAT_SPLIT, &[self.cv1.out(), &self.y0], &self.split_params(0), chunk_n);
        let s1 = ctx.step(CONCAT_SPLIT, &[self.cv1.out(), &self.y1], &self.split_params(self.c), chunk_n);
        ctx.gpu.submit(&[], &[s0, s1]);

        // run bottlenecks on y1, chaining each output to the next input.
        let mut prev = &self.y1;
        for b in &self.m {
            b.forward(ctx, ps, prev);
            prev = b.out();
        }

        // concat [y0, y1, b1..bn] along C in a SINGLE pass: each chunk (c
        // channels) is placed once into its slice of `concat`, instead of a
        // left-fold of concat2 that re-copies the growing prefix (O(n^2) data
        // movement). The placements write disjoint channel ranges, so they share
        // one submit.
        let chunks = self.chunks();
        let cat_c = chunks.len() as u32 * self.c;
        let mut steps = Vec::with_capacity(chunks.len());
        let chunk_n = self.sh.numel();
        for (i, chunk) in chunks.iter().enumerate() {
            let c_off = i as u32 * self.c;
            // chan_place ABI: [N, Ctot, Csrc, c_off, H, W], bufs [src, dst]
            let params = [self.sh.n, cat_c, self.c, c_off, self.sh.h, self.sh.w];
            steps.push(ctx.step(CHAN_PLACE, &[chunk, &self.concat], &params, chunk_n));
        }
        ctx.gpu.submit(&[], &steps);

        // final 1x1 conv -> C_out.
        self.cv2.forward(ctx, ps, &self.concat);
    }

    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        // cv2 backward: grad wrt the concat buffer -> reuse `concat`-shaped temp.
        // We need a buffer of concat size for the grad; allocate via d_split? No,
        // d_split is 2c. Use the largest cat_tmp (which is the full concat size)
        // as the cv2 input-grad target.
        let d_concat = self.cat_tmp.last().unwrap();
        self.cv2.backward(ctx, ps, &self.concat, d_out, d_concat);

        // concat_split: distribute d_concat to each chunk grad [y0,y1,b1..bn].
        let chunks_n = (2 + self.n) as usize;
        let chunk_n = self.sh.numel();
        let cat_c = (2 + self.n) * self.c;
        let mut steps = Vec::new();
        for i in 0..chunks_n {
            let c_off = i as u32 * self.c;
            // concat_split ABI: [N, Ctot, Csrc, c_off, H, W]
            let params = [self.sh.n, cat_c, self.c, c_off, self.sh.h, self.sh.w];
            steps.push(ctx.step(CONCAT_SPLIT, &[d_concat, &self.d_chunk[i]], &params, chunk_n));
        }
        ctx.gpu.submit(&[], &steps);

        // bottleneck chain backward (reverse). The grad into bottleneck i is its
        // own chunk grad PLUS the input-grad produced by bottleneck i+1.
        // Process from last to first; carry `d_carry` = grad wrt that bottleneck's
        // input (= grad wrt the previous chunk's output).
        // d_chunk indices: 0=y0, 1=y1, 2+j = bottleneck j output.
        // Use a scratch (d_split's first half region is wrong shape) -> reuse
        // per-bottleneck d_mid via a dedicated accumulation buffer `d_y1`.
        // We accumulate into d_chunk[k] for the input of bottleneck k.
        let nb = self.m.len();
        // grad flowing back into each bottleneck input; start with chunk grads.
        // For bottleneck j (0-based), its output chunk index is 2+j, its input is
        // chunk (2+j-1) for j>0, or y1 (index 1) for j==0.
        for j in (0..nb).rev() {
            let out_idx = 2 + j;
            let in_idx = if j == 0 { 1 } else { 2 + j - 1 };
            let b = &self.m[j];
            let x_in_b: &DeviceBuffer = if j == 0 { &self.y1 } else { self.m[j - 1].out() };
            // backward produces grad wrt input into d_y1 (scratch), then we add
            // it onto the running chunk grad of the input chunk.
            b.backward(ctx, ps, x_in_b, &self.d_chunk[out_idx], &self.d_y1);
            let s = ctx.step(ADD2, &[&self.d_chunk[in_idx], &self.d_y1, &self.d_chunk[in_idx]], &[chunk_n], chunk_n);
            ctx.gpu.submit(&[], &[s]);
        }

        // Now d_chunk[0] = grad wrt y0, d_chunk[1] = grad wrt y1 (fully merged).
        // Re-merge into d_split [2c] = grad wrt cv1.out: copy y0-grad to channels
        // [0,c) and y1-grad to [c,2c). concat2 does exactly this.
        let params = [self.sh.n, self.c, self.c, self.sh.h, self.sh.w];
        let s = ctx.step(CONCAT2, &[&self.d_chunk[0], &self.d_chunk[1], &self.d_split], &params, 2 * chunk_n);
        ctx.gpu.submit(&[], &[s]);

        // cv1 backward -> d_in.
        self.cv1.backward(ctx, ps, x_in, &self.d_split, d_in);
    }
}

// ===========================================================================
// SPPF = Conv1x1 -> m1,m2,m3 = maxpool5 chain -> concat[x,m1,m2,m3] -> Conv1x1
// ===========================================================================

/// Spatial-Pyramid-Pooling-Fast. A 1x1 conv, three chained 5x5 maxpools, a
/// channel-concat of `[x, m1, m2, m3]` (4*c channels), and a final 1x1 conv.
pub struct SPPF {
    pub cv1: Conv,
    pub cv2: Conv,
    pub in_shape: Shape,
    pub out_shape: Shape,
    c: u32,
    sh: Shape, // [n,c,h,w] of the inner maps

    // forward caches
    m1: DeviceBuffer,
    m2: DeviceBuffer,
    m3: DeviceBuffer,
    am1: DeviceBuffer,
    am2: DeviceBuffer,
    am3: DeviceBuffer,
    cat1: DeviceBuffer, // [x,m1]            -> 2c
    cat2: DeviceBuffer, // [x,m1,m2]         -> 3c
    concat: DeviceBuffer, // [x,m1,m2,m3]    -> 4c

    // backward caches
    d_x: DeviceBuffer,  // grad wrt cv1.out (accumulated)
    d_x_cat: DeviceBuffer, // grad slice of x from concat
    d_m1: DeviceBuffer,
    d_m2: DeviceBuffer,
    d_m3: DeviceBuffer,
    d_m1_cat: DeviceBuffer,
    d_m2_cat: DeviceBuffer,
    d_tmp: DeviceBuffer, // scratch for maxpool dx contributions
}

impl SPPF {
    pub fn new(ctx: &Ctx, prefix: &str, in_shape: Shape, c_out: u32, train: bool) -> SPPF {
        // Ultralytics SPPF: cv1 halves channels to c = c_out/2, cv2 maps 4c->c_out.
        let c = c_out / 2;
        let cv1 = Conv::new(ctx, &format!("{prefix}.cv1"), in_shape, c, 1, 1, 0, train);
        let sh = cv1.out_shape;
        let cat_shape = Shape::new(sh.n, 4 * c, sh.h, sh.w);
        let cv2 = Conv::new(ctx, &format!("{prefix}.cv2"), cat_shape, c_out, 1, 1, 0, train);
        let out_shape = cv2.out_shape;
        let n1 = sh.numel();
        SPPF {
            cv1,
            cv2,
            in_shape,
            out_shape,
            c,
            sh,
            m1: ctx.act(n1),
            m2: ctx.act(n1),
            m3: ctx.act(n1),
            am1: ctx.act(n1),
            am2: ctx.act(n1),
            am3: ctx.act(n1),
            cat1: ctx.act(2 * n1),
            cat2: ctx.act(3 * n1),
            concat: ctx.act(4 * n1),
            d_x: ctx.act(n1),
            d_x_cat: ctx.act(n1),
            d_m1: ctx.act(n1),
            d_m2: ctx.act(n1),
            d_m3: ctx.act(n1),
            d_m1_cat: ctx.act(n1),
            d_m2_cat: ctx.act(n1),
            d_tmp: ctx.act(n1),
        }
    }

    pub fn out(&self) -> &DeviceBuffer {
        self.cv2.out()
    }

    /// Propagate the eval/train BN toggle to both convs.
    pub fn set_eval(&self, eval: bool) {
        self.cv1.set_eval(eval);
        self.cv2.set_eval(eval);
    }

    /// Propagate the BN running-stat update toggle to both convs.
    pub fn set_update_running(&self, on: bool) {
        self.cv1.set_update_running(on);
        self.cv2.set_update_running(on);
    }

    pub fn param_list(&self) -> Vec<(String, usize)> {
        let mut v = self.cv1.param_list();
        v.extend(self.cv2.param_list());
        v
    }

    fn pool_params(&self) -> [u32; 6] {
        // maxpool5 ABI: [N, C, H, W, K, pad], K=5 pad=2.
        [self.sh.n, self.c, self.sh.h, self.sh.w, 5, 2]
    }

    pub fn forward(&self, ctx: &Ctx, ps: &ParamStore, x_in: &DeviceBuffer) {
        self.cv1.forward(ctx, ps, x_in);
        let x = self.cv1.out();
        let n1 = self.sh.numel();
        // m1 = pool(x); m2 = pool(m1); m3 = pool(m2). Sequential dependency.
        let s1 = ctx.step(MAXPOOL5, &[x, &self.m1, &self.am1], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[s1]);
        let s2 = ctx.step(MAXPOOL5, &[&self.m1, &self.m2, &self.am2], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[s2]);
        let s3 = ctx.step(MAXPOOL5, &[&self.m2, &self.m3, &self.am3], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[s3]);

        // concat [x, m1, m2, m3] via left-fold.
        let c = self.c;
        let (h, w, n) = (self.sh.h, self.sh.w, self.sh.n);
        let sc1 = ctx.step(CONCAT2, &[x, &self.m1, &self.cat1], &[n, c, c, h, w], 2 * n1);
        ctx.gpu.submit(&[], &[sc1]);
        let sc2 = ctx.step(CONCAT2, &[&self.cat1, &self.m2, &self.cat2], &[n, 2 * c, c, h, w], 3 * n1);
        ctx.gpu.submit(&[], &[sc2]);
        let sc3 = ctx.step(CONCAT2, &[&self.cat2, &self.m3, &self.concat], &[n, 3 * c, c, h, w], 4 * n1);
        ctx.gpu.submit(&[], &[sc3]);

        self.cv2.forward(ctx, ps, &self.concat);
    }

    pub fn backward(
        &self,
        ctx: &Ctx,
        ps: &ParamStore,
        x_in: &DeviceBuffer,
        d_out: &DeviceBuffer,
        d_in: &DeviceBuffer,
    ) {
        let n1 = self.sh.numel();
        let c = self.c;
        let (h, w, n) = (self.sh.h, self.sh.w, self.sh.n);
        let cat_c = 4 * c;

        // cv2 backward -> grad wrt concat (reuse `concat` buffer? no, need fresh).
        // Use cat2 (3c) is too small; allocate d_concat via the `concat` buffer is
        // also a live cache. Use a dedicated approach: split d_out's contribution
        // directly. We need a 4c-channel grad buffer; reuse `concat` is unsafe
        // because cv2.backward reads it (it's cv2's x_in). So we add a scratch.
        // To avoid another field, route cv2 grad into `concat`-sized cat via the
        // last cat buffer is 4c == concat size; allocate one more temp here.
        let d_concat = ctx.act(cat_c * h * w * n);
        self.cv2.backward(ctx, ps, &self.concat, d_out, &d_concat);

        // split d_concat into [x, m1, m2, m3] grad slices.
        // concat_split ABI: [N, Ctot, Csrc, c_off, H, W]
        let sx = ctx.step(CONCAT_SPLIT, &[&d_concat, &self.d_x_cat], &[n, cat_c, c, 0, h, w], n1);
        let s1 = ctx.step(CONCAT_SPLIT, &[&d_concat, &self.d_m1_cat], &[n, cat_c, c, c, h, w], n1);
        let s2 = ctx.step(CONCAT_SPLIT, &[&d_concat, &self.d_m2_cat], &[n, cat_c, c, 2 * c, h, w], n1);
        let s3 = ctx.step(CONCAT_SPLIT, &[&d_concat, &self.d_m3], &[n, cat_c, c, 3 * c, h, w], n1);
        ctx.gpu.submit(&[], &[sx, s1, s2, s3]);

        // Backprop the maxpool chain. m3 = pool(m2): grad wrt m2 from m3.
        // d_m2 = d_m2_cat + maxpool_dx(d_m3 -> via am3)
        let sd3 = ctx.step(MAXPOOL5_DX, &[&self.d_m3, &self.am3, &self.d_tmp], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[sd3]);
        let a3 = ctx.step(ADD2, &[&self.d_m2_cat, &self.d_tmp, &self.d_m2], &[n1], n1);
        ctx.gpu.submit(&[], &[a3]);

        // m2 = pool(m1): grad wrt m1 = d_m1_cat + maxpool_dx(d_m2 -> via am2)
        let sd2 = ctx.step(MAXPOOL5_DX, &[&self.d_m2, &self.am2, &self.d_tmp], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[sd2]);
        let a2 = ctx.step(ADD2, &[&self.d_m1_cat, &self.d_tmp, &self.d_m1], &[n1], n1);
        ctx.gpu.submit(&[], &[a2]);

        // m1 = pool(x): grad wrt x = d_x_cat + maxpool_dx(d_m1 -> via am1)
        let sd1 = ctx.step(MAXPOOL5_DX, &[&self.d_m1, &self.am1, &self.d_tmp], &self.pool_params(), n1);
        ctx.gpu.submit(&[], &[sd1]);
        let a1 = ctx.step(ADD2, &[&self.d_x_cat, &self.d_tmp, &self.d_x], &[n1], n1);
        ctx.gpu.submit(&[], &[a1]);

        // cv1 backward -> d_in.
        self.cv1.backward(ctx, ps, x_in, &self.d_x, d_in);
    }
}
