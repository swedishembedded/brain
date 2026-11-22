// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements exact backward passes for looped decoders with depth-attention residuals for its
// clients. If your team needs expertise in differentiating a model whose layers read their own earlier states then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The training forward and its backward, visit by visit (module docs of [`super`]).
//!
//! The forward writes a fresh buffer per stage (the stage's saved activation) and keeps, across visits, only the
//! partial entering each visit and the retained block states. The backward recomputes a visit from its saved input
//! ([`Inkling2Train::visit_fwd`]) and differentiates it ([`Inkling2Train::visit_bwd`]). Gradient bookkeeping across
//! visits:
//! - `dB[j]` is the gradient of partial `B[j]` (`B[0]` the normed embedding, `B[i + 1]` visit `i`'s output); every
//!   consumer of `B[j]` is a later visit (or the readout), so `dB[i + 1]` is complete when visit `i` is differentiated;
//! - a retained block state's gradient is ADDED by every depth attention that reads it into the store's slot gradient
//!   and moved into the partial it was retained from when the backward reaches the retaining visit;
//! - a core pass after the first reads the prelude's output (`B[k0]`) again and the previous pass's final partial
//!   through its messages, so both collect contributions from every pass before their producers are reached;
//! - a parameter's gradient is ADDED by every visit that reads it, so the shared core's gradients are the sums over the
//!   passes.

use gpu_core::{f, DeviceBuffer, Dispatch, Gpu, Step};
use model::depth_attn::{DepthAttn, DepthAttnIds};
use model::flash_ext::{self, AttnInputGrads, AttnInputs, AttnOptions, Contiguous, DistanceBias, FlashExtIds, HeadShape, LogPositionScale};
use model::glu::{GatedAct, GluBwdScratch, GluGrads, GluLayout, GluSpec, GluWeights};
use model::head_ce::{GradScale, HeadBackward, RowWeights};
use model::head_transform::HeadTransform;
use model::looped::{Segment, Visit};
use model::message_gate::{MessageGate, MessageGateActs, MessageGateIds, MessageGateWeights};
use model::moe::{ExpertAdapterGrads, ExpertAdapters, ExpertBank, ExpertLora, ExpertLoraGrads, ExpertSet, MoeActs, MoeIds, MoeIdsBwd, MoeShape, RouterBwdIds, RouterGrad, RouterKind, RouterOverride};
use model::saved::RetainedStates;
use model::short_conv::{short_conv_bwd, short_conv_fwd, BaseLayout, DynTaps, ShortConvIds, ShortConvShape};

use super::{BankRef, HostBatch, Inkling2Train};
use crate::config::LayerKind;
use crate::model::{block_slots, CLIP_EPS, DEPTH_EPS};
use crate::names::{self, GateSite, LayerNames};

pub(super) struct Kernels {
    flash: FlashExtIds,
    conv: ShortConvIds,
    depth: DepthAttnIds,
    gate: MessageGateIds,
    moe: MoeIds,
    moe_bwd: MoeIdsBwd,
    router_bwd: RouterBwdIds,
    replay: usize,
    rms: usize,
    rms_inv: usize,
    rms_dw: usize,
    rms_dx: usize,
    rms_nogain: usize,
    rms_nogain_dx: usize,
    l2clip: usize,
    l2clip_dx: usize,
    l2clip_ds: usize,
    sigmoid: usize,
    sigmoid_bwd: usize,
    mul: usize,
    add: usize,
    axpy: usize,
    axpy_at: usize,
    copy: usize,
    embed: usize,
    emb_bwd: usize,
    head: usize,
    scale: usize,
}

impl Kernels {
    pub(super) fn resolve(g: &Gpu) -> Result<Kernels, String> {
        let k = |name: &str| g.kernel_index(name).ok_or_else(|| format!("inkling2: kernel {name} is not registered (build the Gpu on inkling2::train::kernels())"));
        Ok(Kernels {
            flash: FlashExtIds::resolve(g),
            conv: ShortConvIds { fwd: k("dyn_group_conv1d")?, dx: k("dyn_group_conv1d_dx")?, dbase: k("dyn_group_conv1d_dbase")?, ddyn: k("dyn_group_conv1d_ddyn")?, step: k("dyn_group_conv1d_step")?, state_tail: k("conv_state_tail")? },
            depth: DepthAttnIds::resolve(g)?,
            gate: MessageGateIds::resolve(g)?,
            moe: MoeIds { router_gate: k("router_gate_sigmoid")?, linear_gated: k("moe_linear_gated")?, silu_mul: k("silu_mul")?, scale_add: k("scale_add")? },
            moe_bwd: MoeIdsBwd {
                scale_add_dexp: k("scale_add_dexp")?,
                scale_add_dgate: k("scale_add_dgate")?,
                silu_da: k("silu_bwd_da")?,
                silu_db: k("silu_bwd_db")?,
                linear_dx: k("moe_linear_gated_dx")?,
                linear_dw: k("moe_linear_gated_dw")?,
                linear_gated: true,
            },
            router_bwd: RouterBwdIds { router_bwd: k("router_bwd_sigmoid")?, expert_counts: None },
            replay: k(model::moe::ROUTER_REPLAY_KERNEL.0)?,
            rms: k("rmsnorm")?,
            rms_inv: k("rms_inv")?,
            rms_dw: k("rmsnorm_dw")?,
            rms_dx: k("rmsnorm_dx")?,
            rms_nogain: k("rmsnorm_nogain")?,
            rms_nogain_dx: k("rmsnorm_nogain_dx")?,
            l2clip: k("l2clip_scale")?,
            l2clip_dx: k("l2clip_scale_dx")?,
            l2clip_ds: k("l2clip_scale_ds")?,
            sigmoid: k("sigmoid")?,
            sigmoid_bwd: k("sigmoid_bwd")?,
            mul: k("mul")?,
            add: k("add2")?,
            axpy: k("axpy")?,
            axpy_at: k("axpy_at")?,
            copy: k("splice_at")?,
            embed: k("embed")?,
            emb_bwd: k("emb_bwd")?,
            head: k("logit_cap_mask")?,
            scale: k("grad_scale")?,
        })
    }
}

/// The rows a forward runs on: `b` sequences of `t` tokens, and for packed rows each token's document id and position.
struct Rows {
    b: u32,
    t: u32,
    tokens: DeviceBuffer,
    labels: DeviceBuffer,
    seg: Option<DeviceBuffer>,
    doc_pos: Option<DeviceBuffer>,
}

impl Rows {
    fn n(&self) -> u32 {
        self.b * self.t
    }
}

/// Which partial and stack each visit reads, derived from the visit plan (the inference forward's bookkeeping).
struct Schedule {
    /// Visit `i` reads partial `B[src[i]]`.
    src: Vec<usize>,
    /// The retained slots visit `i`'s attention input mixes (before it retains its own).
    stacks: Vec<Vec<u32>>,
    /// Visit `i` retains its incoming partial in `slot` (`fresh` false: the slot already holds it).
    retain: Vec<Option<(u32, bool)>>,
    /// A later core pass's messages: the previous pass's own block states and the partial index it ended on.
    msg: Vec<Option<(Vec<u32>, usize)>>,
    /// The readout's stack.
    final_stack: Vec<u32>,
    capacity: u32,
}

/// What a forward keeps for its backward.
pub(super) struct Saved {
    rows: Rows,
    sched: Schedule,
    store: RetainedStates,
    /// `B[0..=visits]`.
    bufs: Vec<DeviceBuffer>,
    raw: DeviceBuffer,
    /// Per core layer, its `[4, hidden]` message queries.
    mq: Vec<DeviceBuffer>,
    final_stats: DeviceBuffer,
    mixed: DeviceBuffer,
    pub(super) normed: DeviceBuffer,
    nll: DeviceBuffer,
}

/// A depth attention's forward.
struct Da {
    da: DepthAttn,
    out: DeviceBuffer,
    stats: DeviceBuffer,
}

/// One message gate's forward: its input, message and saved activations.
struct GateFwd {
    gate: MessageGate,
    acts: MessageGateActs,
    x: DeviceBuffer,
    m: DeviceBuffer,
}

struct AttnFwd {
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    kv_pre: [DeviceBuffer; 2],
    kv_dyn: [Option<DeviceBuffer>; 2],
    kv_conv: [DeviceBuffer; 2],
    kv: [DeviceBuffer; 2],
    rel_pre: DeviceBuffer,
    rel: DeviceBuffer,
    sink: DeviceBuffer,
    out: DeviceBuffer,
    lse: DeviceBuffer,
    gate_pre: DeviceBuffer,
    sig: DeviceBuffer,
    gated: DeviceBuffer,
    y: DeviceBuffer,
}

struct MoeFwd {
    logits: DeviceBuffer,
    gate: DeviceBuffer,
    acts: MoeActs,
    out: DeviceBuffer,
}

struct VisitFwd {
    msg: Option<Da>,
    attn_da: Option<Da>,
    inputs: Vec<DeviceBuffer>,
    gates: [Option<GateFwd>; 4],
    /// The q, k and v paths before their norm.
    pre: [DeviceBuffer; 3],
    /// One norm feeds all three paths (a sliding layer without messages).
    shared_norm: bool,
    normed: [DeviceBuffer; 3],
    att: AttnFwd,
    residual: DeviceBuffer,
    mlp_da: Da,
    mlp_pre: DeviceBuffer,
    hn: DeviceBuffer,
    moe: MoeFwd,
    out: DeviceBuffer,
}

/// The one router this model has: sigmoid scores, top-k on score + bias, the unbiased scores renormalised and scaled.
fn router_kind(scale: f32) -> RouterKind {
    RouterKind::SigmoidNoAuxTc { n_group: 1, topk_group: 1, norm_topk_prob: true, routed_scaling: scale }
}

impl Inkling2Train {
    fn flush(&self, s: &mut Vec<Step>) {
        if !s.is_empty() {
            self.g().submit(&[], s);
            s.clear();
        }
    }

    fn fresh(&self, n: u32) -> DeviceBuffer {
        self.g().storage(u64::from(n.max(1)))
    }

    fn schedule(&self) -> Schedule {
        let cfg = &self.tc.cfg;
        let (retain, capacity) = block_slots(cfg, &self.plan);
        let core_start = self.plan.shape().prelude;
        let (mut stack, mut prelude_stack, mut k0) = (Vec::new(), Vec::new(), 0usize);
        let (mut pass_blocks, mut previous): (Vec<u32>, Option<(Vec<u32>, usize)>) = (Vec::new(), None);
        let (mut src, mut stacks, mut msg) = (Vec::new(), Vec::new(), Vec::new());
        for v in self.plan.visits() {
            let i = v.index;
            let core = v.segment == Segment::Core;
            let mut from = i;
            if core && v.layer == core_start {
                if v.pass == 0 {
                    prelude_stack = stack.clone();
                    k0 = i;
                } else {
                    previous = Some((std::mem::take(&mut pass_blocks), i));
                    stack = prelude_stack.clone();
                    from = k0;
                }
                pass_blocks.clear();
            }
            src.push(from);
            msg.push(if core && v.pass > 0 { previous.clone() } else { None });
            stacks.push(stack.clone());
            if let Some((slot, _)) = retain[i] {
                stack.push(slot);
                if core && v.layer != core_start {
                    pass_blocks.push(slot);
                }
            }
        }
        Schedule { src, stacks, retain, msg, final_stack: stack, capacity }
    }

    // ---- primitives --------------------------------------------------------------------------------------------

    /// `y = x W^T` for the linear `name` over `m` rows, plus its adapter's correction.
    fn proj(&self, s: &mut Vec<Step>, name: &str, x: &DeviceBuffer, m: u32) -> DeviceBuffer {
        let w = self.lin(name);
        let y = self.fresh(m * w.n());
        self.ops.matmul(s, &w, &self.ops.act_f32(x, 0, m, w.k()), &y, 0);
        if self.adapted(name) {
            let rt = self.runtime.as_ref().expect("adapted");
            rt.forward(self.ops.tensor_gemm(), s, &self.ps, name, x, &y, m, w.k(), w.n());
        }
        y
    }

    /// The backward of [`Self::proj`]: `dx` receives the input gradient (ADDED when `acc`), the weight or adapter
    /// gradients are added.
    #[allow(clippy::too_many_arguments)]
    fn proj_bwd(&self, s: &mut Vec<Step>, name: &str, x: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, m: u32, acc: bool) {
        let w = self.lin(name);
        if let Some(gw) = self.lin_grad(name) {
            self.ops.matmul_dw(s, x, dy, m, w.n(), w.k(), gw);
        }
        if self.adapted(name) {
            let rt = self.runtime.as_ref().expect("adapted");
            let base = |s: &mut Vec<Step>, d: &DeviceBuffer| self.ops.matmul_dx(s, &w, d, m, dx, acc);
            rt.backward(self.ops.tensor_gemm(), s, &self.ps, name, dy, x, dx, m, w.k(), w.n(), &base);
        } else {
            self.ops.matmul_dx(s, &w, dy, m, dx, acc);
        }
    }

    /// RMS norm with the gain `gain` over `rows` rows of the hidden width.
    fn rms(&self, s: &mut Vec<Step>, x: &DeviceBuffer, gain: &str, rows: u32) -> DeviceBuffer {
        let d = self.tc.cfg.hidden;
        let y = self.fresh(rows * d);
        let (idx, grid) = model::block::rms_variant(self.g(), self.k.rms, None, rows, d);
        s.push(self.g().dispatch(idx, &[x, self.val(gain), &y], &[d, rows, f(self.tc.cfg.eps)], grid));
        y
    }

    /// The backward of [`Self::rms`]: a fresh input gradient; the gain's gradient is added.
    fn rms_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, gain: &str, dy: &DeviceBuffer, rows: u32) -> DeviceBuffer {
        let (g, d, eps) = (self.g(), self.tc.cfg.hidden, f(self.tc.cfg.eps));
        let inv = self.fresh(rows);
        s.push(g.step(self.k.rms_inv, &[x, &inv], &[d, rows, eps], rows));
        s.push(g.step(self.k.rms_dw, &[dy, x, &inv, self.grad(gain)], &[d, rows], d));
        let dx = self.fresh(rows * d);
        let (idx, grid) = model::block::rms_variant(g, self.k.rms_dx, None, rows, d);
        s.push(g.dispatch(idx, &[x, self.val(gain), dy, &dx], &[d, rows, eps], grid));
        dx
    }

    /// Each of `n_vec` vectors of width `d` (vector `i` of head `i % heads`) scaled by `scale` when given, then
    /// clipped to L2 norm `thr`.
    #[allow(clippy::too_many_arguments)]
    fn l2clip(&self, s: &mut Vec<Step>, x: &DeviceBuffer, scale: Option<&str>, n_vec: u32, heads: u32, d: u32, thr: f32) -> DeviceBuffer {
        let y = self.fresh(n_vec * d);
        let sb = scale.map(|n| self.val(n)).unwrap_or(x);
        s.push(self.g().dispatch(self.k.l2clip, &[x, sb, &y], &[n_vec, heads, d, scale.is_some() as u32, f(thr), f(CLIP_EPS)], Dispatch::Workgroups(n_vec)));
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn l2clip_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, scale: Option<&str>, dy: &DeviceBuffer, n_vec: u32, heads: u32, d: u32, thr: f32) -> DeviceBuffer {
        let g = self.g();
        let (dx, gs) = (self.fresh(n_vec * d), self.fresh(n_vec * d));
        let sb = scale.map(|n| self.val(n)).unwrap_or(x);
        let p = [n_vec, heads, d, scale.is_some() as u32, f(thr), f(CLIP_EPS)];
        s.push(g.dispatch(self.k.l2clip_dx, &[x, sb, dy, &dx, &gs], &p, Dispatch::Workgroups(n_vec)));
        if let Some(name) = scale {
            s.push(g.step(self.k.l2clip_ds, &[&gs, self.grad(name)], &[n_vec, heads, d], heads * d));
        }
        dx
    }

    fn conv_shape(&self, r: &Rows, shape: ShortConvShape) -> ShortConvShape {
        let shape = shape.in_blocks(r.t);
        if r.seg.is_some() {
            shape.with_segments()
        } else {
            shape
        }
    }

    /// The static convolution of a published depthwise weight `[c, 1, taps]`: `(conv(x) + x) / sqrt(taps + 1)`.
    fn static_shape(&self, r: &Rows, c: u32) -> ShortConvShape {
        let taps = self.tc.cfg.conv_taps;
        self.conv_shape(r, ShortConvShape::static_taps(r.n(), c, taps, BaseLayout::ChannelMajor).residual_scaled(1.0 / ((taps + 1) as f32).sqrt()))
    }

    fn dynamic_shape(&self, r: &Rows, layer: u32) -> ShortConvShape {
        let (cfg, a) = (&self.tc.cfg, self.tc.cfg.attn(layer));
        let dynamic = DynTaps { group_size: a.head_dim, stride: a.kv_heads * cfg.conv_taps, off: 0 };
        self.conv_shape(r, ShortConvShape::dynamic_taps(r.n(), a.kv_dim(), cfg.conv_taps, dynamic).residual_scaled(1.0 / ((2 * cfg.conv_taps + 1) as f32).sqrt()))
    }

    fn static_conv(&self, s: &mut Vec<Step>, r: &Rows, x: &DeviceBuffer, base: &str, c: u32) -> DeviceBuffer {
        let y = self.fresh(r.n() * c);
        s.push(short_conv_fwd(self.g(), &self.k.conv, &self.static_shape(r, c), x, None, self.val(base), r.seg.as_ref(), None, &y));
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn static_conv_bwd(&self, s: &mut Vec<Step>, r: &Rows, x: &DeviceBuffer, base: &str, c: u32, dy: &DeviceBuffer) -> DeviceBuffer {
        let dx = self.fresh(r.n() * c);
        s.extend(short_conv_bwd(self.g(), &self.k.conv, &self.static_shape(r, c), x, dy, None, self.val(base), r.seg.as_ref(), &dx, self.grad(base), None));
        dx
    }

    fn depth(&self, s: &mut Vec<Step>, store: &RetainedStates, slots: &[u32], nq: u32, partial: &DeviceBuffer, query: &DeviceBuffer) -> Result<Da, String> {
        let da = DepthAttn::new(store, slots, nq, DEPTH_EPS)?;
        let (out, stats) = (self.g().storage(da.out_len()), self.g().storage(da.stats_len()));
        s.push(da.forward(self.g(), &self.k.depth, store, partial, query, &out, &stats));
        Ok(Da { da, out, stats })
    }

    /// The backward of a depth attention: slot gradients into the store, the partial's into `dpartial`, the queries'
    /// into `dquery`, all ADDED.
    #[allow(clippy::too_many_arguments)]
    fn depth_bwd(&self, s: &mut Vec<Step>, store: &RetainedStates, d: &Da, partial: &DeviceBuffer, query: &DeviceBuffer, dout: &DeviceBuffer, dpartial: &DeviceBuffer, dquery: &DeviceBuffer) {
        let dz = self.g().storage(d.da.dz_len());
        s.extend(d.da.backward(self.g(), &self.k.depth, store, partial, query, &d.stats, dout, dpartial, &dz, dquery));
    }

    /// `n` values of `src` from `offset`, copied.
    fn rows_of(&self, s: &mut Vec<Step>, src: &DeviceBuffer, offset: u32, n: u32) -> DeviceBuffer {
        let y = self.fresh(n);
        s.push(self.g().step(self.k.copy, &[src, &y], &[n, offset, 0], n));
        y
    }

    /// `dst[off..off + n] += src[..n]`.
    fn add_at(&self, s: &mut Vec<Step>, dst: &DeviceBuffer, off: u32, src: &DeviceBuffer, n: u32) {
        s.push(self.g().step(self.k.axpy_at, &[dst, src], &[n, f(1.0), off, 0], n));
    }

    fn add_into(&self, s: &mut Vec<Step>, dst: &DeviceBuffer, src: &DeviceBuffer, n: u32) {
        s.push(self.g().step(self.k.axpy, &[dst, src], &[n, f(1.0)], n));
    }

    fn gate_weights(&self, layer: u32, site: GateSite) -> [String; 3] {
        LayerNames::new(layer).gate(site)
    }

    // ---- the trunk ---------------------------------------------------------------------------------------------

    fn rows_for(&self, host: &HostBatch, b: u32, t: u32) -> Result<Rows, String> {
        let g = self.g();
        let n = (b * t) as usize;
        if host.tokens.len() != n || host.targets.len() != n {
            return Err(format!("inkling2: a batch of {} tokens for {b} x {t} rows (set a batch first)", host.tokens.len()));
        }
        let tokens = self.fresh(n as u32);
        g.write(&tokens, &host.tokens);
        let labels = self.fresh(n as u32);
        g.write(&labels, &host.targets);
        let (seg, doc_pos) = match &host.segments {
            Some(ids) => {
                let pos = data::packing::document_positions(ids, t as usize).map_err(|e| format!("inkling2: packed rows: {e}"))?;
                let (sb, pb) = (self.fresh(n as u32), self.fresh(n as u32));
                g.write(&sb, ids);
                g.write(&pb, &pos);
                (Some(sb), Some(pb))
            }
            None => (None, None),
        };
        Ok(Rows { b, t, tokens, labels, seg, doc_pos })
    }

    /// The forward of `host` over `b x t` rows up to the final norm: every partial, the retained states and the
    /// readout's input.
    pub(super) fn trunk(&self, host: &HostBatch, b: u32, t: u32, train: bool) -> Result<Saved, String> {
        let cfg = &self.tc.cfg;
        let (g, d) = (self.g(), cfg.hidden);
        let rows = self.rows_for(host, b, t)?;
        let n = rows.n();
        let mut s = Vec::new();
        let raw = self.fresh(n * d);
        match &self.frozen {
            Some(_) => self.ops.embed(&mut s, &self.lin(&names::embed()), &rows.tokens, n, &raw),
            None => s.push(g.step(self.k.embed, &[&rows.tokens, self.ps.w(&names::embed()), &raw], &[d, n], n * d)),
        }
        let x0 = self.rms(&mut s, &raw, &names::embed_norm(), n);
        if let (true, Some(rows)) = (train, self.image_rows.borrow().as_ref()) {
            // The spliced rows replace the normed embeddings at their positions.
            let src = g.storage_init("inkling2.image_rows", rows.rows());
            for (i, &p) in rows.positions().iter().enumerate() {
                s.push(g.step(self.k.copy, &[&src, &x0], &[d, i as u32 * d, p * d], d));
            }
        }
        let mq = self.message_queries(&mut s);
        let sched = self.schedule();
        let store = RetainedStates::new(g, sched.capacity.max(1), n, d, train)?;
        let mut bufs = vec![x0];
        for v in self.plan.visits() {
            let i = v.index;
            let x = bufs[sched.src[i]].clone();
            let acts = self.visit_fwd(&mut s, &rows, &sched, &store, &bufs, &mq, v, &x, true)?;
            bufs.push(acts.out);
            self.flush(&mut s);
        }
        let last = bufs.last().expect("a visit ran").clone();
        let fin = self.depth(&mut s, &store, &sched.final_stack, 1, &last, self.val(&names::output_query()))?;
        let normed = self.rms(&mut s, &fin.out, &names::final_norm(), n);
        self.flush(&mut s);
        Ok(Saved { rows, sched, store, bufs, raw, mq, final_stats: fin.stats, mixed: fin.out, normed, nll: self.fresh(n) })
    }

    /// Per core layer, the `[4, hidden]` q, k, v, MLP message queries gathered from the two published tensors.
    fn message_queries(&self, s: &mut Vec<Step>) -> Vec<DeviceBuffer> {
        let (cfg, d) = (&self.tc.cfg, self.tc.cfg.hidden);
        let Some(l) = cfg.looped else { return Vec::new() };
        let (qm, kv) = (self.val(&names::message_queries()), self.val(&names::message_kv_queries()));
        (0..l.layers())
            .map(|c| {
                let out = self.fresh(4 * d);
                for (j, (is_kv, r)) in names::message_query_rows(c).into_iter().enumerate() {
                    s.push(self.g().step(self.k.copy, &[if is_kv { kv } else { qm }, &out], &[d, r * d, j as u32 * d], d));
                }
                out
            })
            .collect()
    }

    /// The loss of a completed trunk: the chunked head's cross-entropy over the valid vocabulary, soft-capped,
    /// weighted per token when weighted loss is on, over the supervised count (or the set denominator).
    pub(super) fn head_loss(&self, fwd: &Saved) -> f32 {
        let n = fwd.rows.n();
        let mut s = Vec::new();
        self.head.forward_steps(&self.ops, &mut s, &fwd.normed, n, &fwd.rows.labels, &fwd.nll, self.logit_scale.get());
        self.flush(&mut s);
        let nll = self.g().read(&fwd.nll, n as usize);
        let sum: f64 = match self.weighted.borrow().as_ref() {
            Some((w, _)) => nll.iter().zip(self.g().read(w, n as usize)).map(|(l, w)| f64::from(*l) * f64::from(w)).sum(),
            None => nll.iter().map(|&l| f64::from(l)).sum(),
        };
        (sum / f64::from(self.count.get())) as f32
    }

    /// The per-token log-probabilities of the last forward's labels (0 at IGNORE).
    pub(super) fn last_logprobs(&self) -> Vec<f32> {
        let saved = self.saved.borrow();
        let fwd = saved.as_ref().expect("inkling2: token log-probabilities need a forward of the current batch");
        let n = fwd.rows.n() as usize;
        self.g().read(&fwd.nll, n).into_iter().map(|l| -l).collect()
    }

    /// The `[len, vocab]` logits of a trunk's normed readout, scaled, soft-capped and masked as the head declares.
    pub(super) fn full_logits(&self, normed: &DeviceBuffer, len: u32) -> Vec<f32> {
        let (g, vocab) = (self.g(), self.tc.cfg.vocab);
        let mut s = Vec::new();
        let w = self.lin(&names::unembed());
        let logits = self.fresh(len * vocab);
        self.ops.matmul(&mut s, &w, &self.ops.act_f32(normed, 0, len, w.k()), &logits, 0);
        // The served logits (capped, padding masked), then the sampler's temperature correction, as the head's loss reads.
        let head = HeadTransform { softcap: Some(self.tc.cfg.logit_cap), valid_vocab: Some(self.tc.cfg.valid_vocab) };
        s.extend(head.step(g, self.k.head, &logits, len, vocab));
        let scale = self.logit_scale.get();
        if scale != 1.0 {
            s.push(g.step(self.k.scale, &[&logits], &[len * vocab, f(scale)], len * vocab));
        }
        self.flush(&mut s);
        g.read(&logits, (len * vocab) as usize)
    }

    // ---- one visit ---------------------------------------------------------------------------------------------

    /// Visit `v` over its input partial `x`: every stage's activation, and the output partial. `retain` stores the
    /// incoming partial in the visit's slot (the forward does; a recompute finds it there already).
    #[allow(clippy::too_many_arguments)]
    fn visit_fwd(&self, s: &mut Vec<Step>, r: &Rows, sched: &Schedule, store: &RetainedStates, bufs: &[DeviceBuffer], mq: &[DeviceBuffer], v: &Visit, x: &DeviceBuffer, retain: bool) -> Result<VisitFwd, String> {
        let cfg = &self.tc.cfg;
        let (g, d, n) = (self.g(), cfg.hidden, r.n());
        let nn = n * d;
        let layer = v.layer as u32;
        let i = v.index;
        let ln = LayerNames::new(layer);
        let stack = &sched.stacks[i];

        let msg = match &sched.msg[i] {
            Some((blocks, p)) => Some(self.depth(s, store, blocks, 4, &bufs[*p], &mq[v.layer - self.plan.shape().prelude])?),
            None => None,
        };
        let nq = cfg.attn_queries(layer);
        let attn_da = if nq > 0 { Some(self.depth(s, store, stack, nq, x, self.val(&names::attn_queries(layer)))?) } else { None };
        let inputs: Vec<DeviceBuffer> = match &attn_da {
            None => vec![x.clone()],
            Some(a) => (0..nq).map(|q| self.rows_of(s, &a.out, q * nn, nn)).collect(),
        };
        let mut gates: [Option<GateFwd>; 4] = [None, None, None, None];
        let mut gated = |s: &mut Vec<Step>, site: usize, h: &DeviceBuffer| -> DeviceBuffer {
            let Some(m) = &msg else { return h.clone() };
            let rank = cfg.looped.expect("messages belong to a loop").gate_rank;
            let gate = MessageGate::new(n, d, rank, cfg.eps).expect("a validated shape");
            let acts = gate.acts(g);
            let mrow = self.rows_of(s, &m.out, site as u32 * nn, nn);
            let [down, forget, gain] = self.gate_weights(layer, GateSite::ALL[site]);
            let y = self.fresh(nn);
            gate.forward(s, g, &self.k.gate, &self.ops, &MessageGateWeights { down: self.val(&down), forget: self.val(&forget), gain: self.val(&gain) }, h, &mrow, &acts, &y);
            gates[site] = Some(GateFwd { gate, acts, x: h.clone(), m: mrow });
            y
        };
        let full = cfg.kind(layer) == LayerKind::Full;
        let pick = |j: usize| inputs[j.min(inputs.len() - 1)].clone();
        let shared_norm = !full && msg.is_none();
        let pre: [DeviceBuffer; 3] = [gated(s, 0, &pick(0)), gated(s, 1, &pick(1)), gated(s, 2, &pick(2))];
        let norm = |j: usize| if full { ln.qkv_norm(j) } else { ln.attn_norm() };
        let normed: [DeviceBuffer; 3] = if shared_norm {
            let h = self.rms(s, &pre[0], &norm(0), n);
            [h.clone(), h.clone(), h]
        } else {
            [self.rms(s, &pre[0], &norm(0), n), self.rms(s, &pre[1], &norm(1), n), self.rms(s, &pre[2], &norm(2), n)]
        };

        let att = self.attention_fwd(s, r, layer, &normed);
        let attn_conv = self.static_conv(s, r, &att.y, &ln.attn_conv(), d);
        let residual = match sched.retain[i] {
            Some((slot, fresh)) => {
                if fresh && retain {
                    s.push(store.retain_step(g, slot, x));
                }
                attn_conv.clone()
            }
            None => {
                let y = self.fresh(nn);
                s.push(g.step(self.k.add, &[x, &attn_conv, &y], &[nn], nn));
                y
            }
        };
        let mut mlp_stack = stack.clone();
        if let Some((slot, _)) = sched.retain[i] {
            mlp_stack.push(slot);
        }
        let mlp_da = self.depth(s, store, &mlp_stack, 1, &residual, self.val(&names::mlp_queries(layer)))?;
        let mlp_pre = gated(s, 3, &mlp_da.out);
        let hn = self.rms(s, &mlp_pre, &ln.mlp_norm(), n);
        let moe = self.moe_fwd(s, r, v, &hn)?;
        let mlp_conv = self.static_conv(s, r, &moe.out, &ln.mlp_conv(), d);
        let out = self.fresh(nn);
        s.push(g.step(self.k.add, &[&residual, &mlp_conv, &out], &[nn], nn));
        Ok(VisitFwd { msg, attn_da, inputs, gates, pre, shared_norm, normed, att, residual, mlp_da, mlp_pre, hn, moe, out })
    }

    fn attention_fwd(&self, s: &mut Vec<Step>, r: &Rows, layer: u32, normed: &[DeviceBuffer; 3]) -> AttnFwd {
        let cfg = &self.tc.cfg;
        let (g, n) = (self.g(), r.n());
        let a = cfg.attn(layer);
        let ln = LayerNames::new(layer);
        let full = cfg.kind(layer) == LayerKind::Full;
        let (qh, kh, vh) = (&normed[0], &normed[1], &normed[2]);
        let q_pre = self.proj(s, &ln.wq(), qh, n);
        let q = self.l2clip(s, &q_pre, Some(&ln.q_scale()), n * a.heads, a.heads, a.head_dim, cfg.qk_clip);
        let mut kv_pre = Vec::new();
        let mut kv_dyn = [None, None];
        let mut kv_conv = Vec::new();
        let mut kv = Vec::new();
        for (j, input) in [kh, vh].into_iter().enumerate() {
            let w = if j == 0 { ln.wk() } else { ln.wv() };
            let pre = self.proj(s, &w, input, n);
            let conv = if full {
                let dyn_taps = self.proj(s, &ln.kv_conv_proj(j == 1), input, n);
                let y = self.fresh(n * a.kv_dim());
                s.push(short_conv_fwd(g, &self.k.conv, &self.dynamic_shape(r, layer), &pre, Some(&dyn_taps), self.val(&ln.kv_conv_static(j == 1)), r.seg.as_ref(), None, &y));
                kv_dyn[j] = Some(dyn_taps);
                y
            } else {
                self.static_conv(s, r, &pre, &ln.kv_conv(j == 1), a.kv_dim())
            };
            let normed = if j == 0 {
                self.l2clip(s, &conv, None, n * a.kv_heads, a.kv_heads, a.head_dim, cfg.qk_clip)
            } else {
                let y = self.fresh(n * a.kv_dim());
                s.push(g.dispatch(self.k.rms_nogain, &[&conv, &y], &[a.head_dim, n * a.kv_heads, f(cfg.eps)], Dispatch::Workgroups(n * a.kv_heads)));
                y
            };
            kv_pre.push(pre);
            kv_conv.push(conv);
            kv.push(normed);
        }
        let rel_pre = self.proj(s, &ln.wr(), qh, n);
        let rel = self.l2clip(s, &rel_pre, Some(&ln.rel_scale()), n * a.heads, a.heads, cfg.d_rel, cfg.rel_clip);
        let sink = self.proj(s, &ln.sink(), qh, n);
        let (heads, opts) = self.attn_opts(r, layer);
        let out = self.fresh(n * a.q_dim());
        let lse = self.fresh(n * a.heads);
        let inputs = AttnInputs { sink: Some(&sink), rel: Some(&rel), bank: Some(self.val(&ln.rel_bank())), doc_pos: r.doc_pos.as_ref() };
        s.extend(flash_ext::flash_ext_fwd(g, &self.k.flash, &heads, &opts, &inputs, Contiguous { b: r.b, t: r.t, pos0: 0 }, &q, &kv[0], &kv[1], &out, &lse));
        let gate_pre = self.proj(s, &ln.gate_attn(), qh, n);
        let m = n * a.q_dim();
        let sig = self.fresh(m);
        s.push(g.step(self.k.sigmoid, &[&gate_pre, &sig], &[m], m));
        let gated = self.fresh(m);
        s.push(g.step(self.k.mul, &[&out, &sig, &gated], &[m], m));
        let y = self.proj(s, &ln.wo(), &gated, n);
        let two = |v: Vec<DeviceBuffer>| -> [DeviceBuffer; 2] { v.try_into().unwrap_or_else(|_| unreachable!("k and v")) };
        AttnFwd { q_pre, q, kv_pre: two(kv_pre), kv_dyn, kv_conv: two(kv_conv), kv: two(kv), rel_pre, rel, sink, out, lse, gate_pre, sig, gated, y }
    }

    fn attn_opts(&self, r: &Rows, layer: u32) -> (HeadShape, AttnOptions) {
        let cfg = &self.tc.cfg;
        let a = cfg.attn(layer);
        let full = cfg.kind(layer) == LayerKind::Full;
        let heads = HeadShape { n_heads: a.heads, n_kv_heads: a.kv_heads, head_dim: a.head_dim };
        let opts = AttnOptions {
            scale: (a.head_dim as f32).powf(-0.5),
            window: (!full).then_some(cfg.window),
            sink: true,
            bias: Some(DistanceBias { extent: cfg.bias_extent(layer), d_rel: cfg.d_rel }),
            tau: full.then_some(LogPositionScale { alpha: cfg.log_alpha, n_floor: cfg.log_floor as f32, scales_bias: true }),
            documents: r.doc_pos.is_some(),
        };
        (heads, opts)
    }

    fn moe_shape(&self, n: u32) -> (MoeShape, GluSpec) {
        let cfg = &self.tc.cfg;
        let shape = MoeShape { rows: n, d_model: cfg.hidden, moe_ff: cfg.moe_ff, n_experts: cfg.experts, top_k: cfg.top_k };
        (shape, GluSpec { act: GatedAct::SoftcapSigmoid { gate_cap: cfg.gate_cap, up_cap: cfg.up_cap }, layout: GluLayout::Interleaved })
    }

    /// Layer `layer`'s expert adapters, when the adapter targets its banks.
    fn expert_adapters(&self, layer: u32) -> ExpertAdapters<'_> {
        let ln = LayerNames::new(layer);
        let pair = |bank: String| -> Option<ExpertLora<'_>> {
            let (a, b) = model::adapter::device::lora_names(&bank);
            let lora = self.tc.lora.as_ref()?;
            let hp = lora.hp_for(&bank);
            Some(ExpertLora { a: self.ps.weight.get(&a)?, b: self.ps.weight.get(&b)?, rank: hp.rank as u32, scale: hp.scale() })
        };
        ExpertAdapters { gate_up: pair(ln.w13()), up: None, down: pair(ln.w2()) }
    }

    fn expert_adapter_grads(&self, layer: u32) -> ExpertAdapterGrads<'_> {
        let ln = LayerNames::new(layer);
        let pair = |bank: String| -> Option<ExpertLoraGrads<'_>> {
            let (a, b) = model::adapter::device::lora_names(&bank);
            Some(ExpertLoraGrads { a: self.ps.grad.get(&a)?, b: self.ps.grad.get(&b)? })
        };
        ExpertAdapterGrads { gate_up: pair(ln.w13()), up: None, down: pair(ln.w2()) }
    }

    fn dense_weights<'a>(&self, w13: &'a DeviceBuffer, w2: &'a DeviceBuffer) -> Vec<GluWeights<'a>> {
        let (d, ff) = (self.tc.cfg.hidden, self.tc.cfg.moe_ff);
        (0..self.tc.cfg.experts).map(|e| GluWeights { gate_w: w13, up_w: None, down_w: w2, gate_off: e * 2 * ff * d, down_off: e * d * ff }).collect()
    }

    fn moe_fwd(&self, s: &mut Vec<Step>, r: &Rows, v: &Visit, hn: &DeviceBuffer) -> Result<MoeFwd, String> {
        let cfg = &self.tc.cfg;
        let (g, n, layer) = (self.g(), r.n(), v.layer as u32);
        let ln = LayerNames::new(layer);
        let (shape, spec) = self.moe_shape(n);
        let kind = router_kind(cfg.route_scale);
        let logits = self.proj(s, &ln.router(), hn, n);
        let (gate, probs) = (self.fresh(n * cfg.experts), self.fresh(n * cfg.experts));
        let bias = self.val(&ln.router_bias());
        s.push(model::moe::router_fwd_kind(g, &self.k.moe, kind, &shape, &logits, Some(bias), &gate, Some(&probs)));
        if let Some(replay) = self.replay.borrow().as_ref() {
            if replay.batch != r.b || replay.seq != r.t {
                return Err(format!("inkling2: a routing replay of {} x {} rows for a batch of {} x {}", replay.batch, replay.seq, r.b, r.t));
            }
            let experts = replay.upload_call(g, v.index, cfg.top_k);
            s.push(model::moe::router_fwd_replay(g, RouterOverride { kernel: self.k.replay, experts: &experts }, kind, &shape, &logits, Some(bias), &gate));
        }
        self.flush(s);
        let host_gate = g.read(&gate, (n * cfg.experts) as usize);
        {
            // The selection each row took (ascending expert ids), what [`Inkling2Train::taken_routing`] hands back.
            let mut taken = self.taken.borrow_mut();
            if taken.len() <= v.index {
                taken.resize(v.index + 1, Vec::new());
            }
            taken[v.index] = host_gate.chunks(cfg.experts as usize).flat_map(|row| row.iter().enumerate().filter(|(_, w)| **w != 0.0).map(|(e, _)| e as u32).collect::<Vec<_>>()).collect();
        }
        let acts = MoeActs::compact_for(g, &shape, &host_gate, &spec);
        let out = self.fresh(n * cfg.hidden);
        let bank = self.bank(v.layer);
        let dense;
        let set = match &bank {
            BankRef::Dense { w13, w2 } => {
                dense = self.dense_weights(w13, w2);
                ExpertSet { glu: spec, bank: ExpertBank::Dense(&dense) }
            }
            BankRef::Micro { w13, w2 } => ExpertSet { glu: spec, bank: ExpertBank::Micro { gate_up: w13, up: None, down: w2 } },
        };
        s.extend(model::moe::experts_fwd_train_adapted(g, &self.k.moe, &set, &self.expert_adapters(layer), &shape, hn, &gate, &acts, &out)?);
        Ok(MoeFwd { logits, gate, acts, out })
    }

    // ---- the backward ------------------------------------------------------------------------------------------

    /// Differentiate the last forward: every parameter's (or adapter's) gradient is added.
    pub(super) fn backward_from(&self, fwd: &Saved) -> Result<(), String> {
        let cfg = &self.tc.cfg;
        let (g, d) = (self.g(), cfg.hidden);
        let r = &fwd.rows;
        let n = r.n();
        let nn = n * d;
        let mut s = Vec::new();
        let clears: Vec<&DeviceBuffer> = self.merged.values().map(|m| m.d_eff()).collect();
        if !clears.is_empty() {
            g.submit(&clears, &[]);
        }
        let db: Vec<DeviceBuffer> = fwd.bufs.iter().map(|_| self.fresh(nn)).collect();
        let dmq: Vec<DeviceBuffer> = fwd.mq.iter().map(|_| self.fresh(4 * d)).collect();

        // The head, the final norm and the readout's depth attention.
        let d_normed = self.fresh(nn);
        let weighted = self.weighted.borrow();
        let head = HeadBackward {
            hidden: &fwd.normed,
            d_hidden: &d_normed,
            n,
            labels: &r.labels,
            grad_scale: GradScale::Const(1.0 / self.count.get()),
            logit_scale: self.logit_scale.get(),
            row_weights: weighted.as_ref().map(|(weights, scratch)| RowWeights { weights, scratch }),
            dw: self.lin_grad(&names::unembed()).map(|buf| model::head_ce::HeadGrad { buf, dtype: model::ops::GradDtype::F32 }),
            forward_state_held: false,
        };
        self.head.backward_steps(&self.ops, &mut s, &head);
        drop(weighted);
        let d_mixed = self.rms_bwd(&mut s, &fwd.mixed, &names::final_norm(), &d_normed, n);
        let last = fwd.bufs.last().expect("a visit ran");
        let fin = Da { da: DepthAttn::new(&fwd.store, &fwd.sched.final_stack, 1, DEPTH_EPS)?, out: fwd.mixed.clone(), stats: fwd.final_stats.clone() };
        let oq = names::output_query();
        self.depth_bwd(&mut s, &fwd.store, &fin, last, self.val(&oq), &d_mixed, db.last().expect("a partial"), self.grad(&oq));
        self.flush(&mut s);

        for v in self.plan.visits().iter().rev() {
            let i = v.index;
            let x = fwd.bufs[fwd.sched.src[i]].clone();
            let acts = self.visit_fwd(&mut s, r, &fwd.sched, &fwd.store, &fwd.bufs, &fwd.mq, v, &x, false)?;
            self.visit_bwd(&mut s, r, fwd, &acts, v, &x, &db, &dmq)?;
            self.flush(&mut s);
        }

        // Spliced rows: their gradient is the tower's, and the token embedding at those positions gets none.
        if let Some(rows) = self.image_rows.borrow().as_ref() {
            self.flush(&mut s);
            let mut d0 = g.read(&db[0], nn as usize);
            *self.image_grads.borrow_mut() = Some(rows.take_row_gradients(&mut d0)?);
            g.write_f32(&db[0], &d0);
        }
        // The embedding norm and the embedding.
        let d_raw = self.rms_bwd(&mut s, &fwd.raw, &names::embed_norm(), &db[0], n);
        if self.frozen.is_none() {
            s.push(g.step(self.k.emb_bwd, &[&r.tokens, &d_raw, self.ps.g(&names::embed())], &[n, d, cfg.vocab], cfg.vocab * d));
        }
        // The message queries' gradients, back into the two published tensors.
        for (c, dq) in dmq.iter().enumerate() {
            let (qm, kv) = (self.grad(&names::message_queries()), self.grad(&names::message_kv_queries()));
            for (j, (is_kv, row)) in names::message_query_rows(c as u32).into_iter().enumerate() {
                s.push(g.step(self.k.axpy_at, &[if is_kv { kv } else { qm }, dq], &[d, f(1.0), row * d, j as u32 * d], d));
            }
        }
        // Merged adapters: their pairs' gradients from the effective matrices' gradients.
        for (name, m) in &self.merged {
            let (a, b) = model::adapter::device::lora_names(name);
            m.project(&self.ops, &mut s, self.ps.w(&a), self.ps.w(&b), self.ps.grad.get(&a), self.ps.g(&b));
        }
        self.flush(&mut s);
        Ok(())
    }

    /// Differentiate visit `v` (its activations `a`, input partial `x`): add into `dB[src]`, the message partial's
    /// `dB`, the retained states' gradients, the core layer's message-query gradient and every parameter's gradient.
    #[allow(clippy::too_many_arguments)]
    fn visit_bwd(&self, s: &mut Vec<Step>, r: &Rows, fwd: &Saved, a: &VisitFwd, v: &Visit, x: &DeviceBuffer, db: &[DeviceBuffer], dmq: &[DeviceBuffer]) -> Result<(), String> {
        let cfg = &self.tc.cfg;
        let (g, d, n) = (self.g(), cfg.hidden, r.n());
        let nn = n * d;
        let layer = v.layer as u32;
        let i = v.index;
        let ln = LayerNames::new(layer);
        let store = &fwd.store;
        let d_in = &db[fwd.sched.src[i]];
        // The output's gradient is complete; it is the residual's too, and the MLP depth attention adds into it.
        let d_res = &db[i + 1];
        let d_msg = a.msg.as_ref().map(|_| self.fresh(4 * nn));

        // MLP: convolution, experts, norm, gate, depth attention.
        let d_moe = self.static_conv_bwd(s, r, &a.moe.out, &ln.mlp_conv(), d, d_res);
        let d_hn = self.moe_bwd(s, r, v, &a.hn, &a.moe, &d_moe)?;
        let d_mlp_pre = self.rms_bwd(s, &a.mlp_pre, &ln.mlp_norm(), &d_hn, n);
        let d_mlp_in = self.gate_bwd(s, layer, 3, &a.gates[3], &d_mlp_pre, d_msg.as_ref(), nn);
        let mq = names::mlp_queries(layer);
        self.depth_bwd(s, store, &a.mlp_da, &a.residual, self.val(&mq), &d_mlp_in, d_res, self.grad(&mq));
        // The residual restarts at a block boundary, whose retained input collects its consumers' gradients now.
        match fwd.sched.retain[i] {
            Some((slot, fresh)) => {
                if fresh {
                    s.push(store.grad_add_step(g, slot, d_in));
                }
            }
            None => self.add_into(s, d_in, d_res, nn),
        }
        let d_attn = self.static_conv_bwd(s, r, &a.att.y, &ln.attn_conv(), d, d_res);
        let d_normed = self.attention_bwd(s, r, layer, &a.normed, &a.att, &d_attn);

        // The q, k, v inputs: norms, message gates, depth attention.
        let full = cfg.kind(layer) == LayerKind::Full;
        let d_inputs: Vec<DeviceBuffer> = a.inputs.iter().map(|_| self.fresh(nn)).collect();
        let norm = |j: usize| if full { ln.qkv_norm(j) } else { ln.attn_norm() };
        if a.shared_norm {
            let sum = self.fresh(nn);
            for dn in &d_normed {
                self.add_into(s, &sum, dn, nn);
            }
            let dh = self.rms_bwd(s, &a.pre[0], &norm(0), &sum, n);
            self.add_into(s, &d_inputs[0], &dh, nn);
        } else {
            for j in 0..3 {
                let d_pre = self.rms_bwd(s, &a.pre[j], &norm(j), &d_normed[j], n);
                let dx = self.gate_bwd(s, layer, j, &a.gates[j], &d_pre, d_msg.as_ref(), nn);
                self.add_into(s, &d_inputs[j.min(d_inputs.len() - 1)], &dx, nn);
            }
        }
        match &a.attn_da {
            Some(da) => {
                let nq = d_inputs.len() as u32;
                let dout = self.fresh(nq * nn);
                for (q, di) in d_inputs.iter().enumerate() {
                    self.add_at(s, &dout, q as u32 * nn, di, nn);
                }
                let aq = names::attn_queries(layer);
                self.depth_bwd(s, store, da, x, self.val(&aq), &dout, d_in, self.grad(&aq));
            }
            None => self.add_into(s, d_in, &d_inputs[0], nn),
        }

        // The messages: the previous pass's block states, its final partial and this core layer's queries.
        if let (Some(m), Some(dm), Some((_, p))) = (&a.msg, &d_msg, &fwd.sched.msg[i]) {
            let c = v.layer - self.plan.shape().prelude;
            self.depth_bwd(s, store, m, &fwd.bufs[*p], &fwd.mq[c], dm, &db[*p], &dmq[c]);
        }
        Ok(())
    }

    /// The backward of message gate `site` of `layer` when it ran (`fwd`): its input's gradient (fresh), the message's
    /// added into `d_msg` at the site's rows, the gate's weights' gradients added. Without a gate, `dy` itself.
    #[allow(clippy::too_many_arguments)]
    fn gate_bwd(&self, s: &mut Vec<Step>, layer: u32, site: usize, fwd: &Option<GateFwd>, dy: &DeviceBuffer, d_msg: Option<&DeviceBuffer>, nn: u32) -> DeviceBuffer {
        let Some(gf) = fwd else { return dy.clone() };
        let g = self.g();
        let [down, forget, gain] = self.gate_weights(layer, GateSite::ALL[site]);
        let (dx, dm) = (self.fresh(nn), self.fresh(nn));
        let scratch = gf.gate.scratch(g);
        let w = MessageGateWeights { down: self.val(&down), forget: self.val(&forget), gain: self.val(&gain) };
        let grads = MessageGateWeights { down: self.grad(&down), forget: self.grad(&forget), gain: self.grad(&gain) };
        gf.gate.backward(s, g, &self.k.gate, &self.ops, &w, &gf.x, &gf.m, &gf.acts, dy, &dx, &dm, &grads, &scratch);
        self.add_at(s, d_msg.expect("a gated visit has messages"), site as u32 * nn, &dm, nn);
        dx
    }

    /// The attention's backward from the gradient of its output projection's output: the gradients of the q, k and v
    /// paths' normed inputs (fresh), every weight's gradient added.
    fn attention_bwd(&self, s: &mut Vec<Step>, r: &Rows, layer: u32, normed: &[DeviceBuffer; 3], a: &AttnFwd, dy: &DeviceBuffer) -> [DeviceBuffer; 3] {
        let cfg = &self.tc.cfg;
        let (g, n) = (self.g(), r.n());
        let at = cfg.attn(layer);
        let ln = LayerNames::new(layer);
        let full = cfg.kind(layer) == LayerKind::Full;
        let (qh, kh, vh) = (&normed[0], &normed[1], &normed[2]);
        let (d_qh, d_kh, d_vh) = (self.fresh(n * cfg.hidden), self.fresh(n * cfg.hidden), self.fresh(n * cfg.hidden));
        let m = n * at.q_dim();
        // Output projection and gate.
        let d_gated = self.fresh(m);
        self.proj_bwd(s, &ln.wo(), &a.gated, dy, &d_gated, n, false);
        let (d_out, d_sig, d_gate_pre) = (self.fresh(m), self.fresh(m), self.fresh(m));
        s.push(g.step(self.k.mul, &[&d_gated, &a.sig, &d_out], &[m], m));
        s.push(g.step(self.k.mul, &[&d_gated, &a.out, &d_sig], &[m], m));
        s.push(g.step(self.k.sigmoid_bwd, &[&a.gate_pre, &d_sig, &d_gate_pre], &[m], m));
        self.proj_bwd(s, &ln.gate_attn(), qh, &d_gate_pre, &d_qh, n, true);
        // The attention itself.
        let (heads, opts) = self.attn_opts(r, layer);
        let (d_q, d_k, d_v) = (self.fresh(m), self.fresh(n * at.kv_dim()), self.fresh(n * at.kv_dim()));
        let (d_sink, d_rel, d_bank) = (self.fresh(n * at.heads), self.fresh(n * at.heads * cfg.d_rel), self.fresh(cfg.d_rel * cfg.bias_extent(layer)));
        let bank = ln.rel_bank();
        let inputs = AttnInputs { sink: Some(&a.sink), rel: Some(&a.rel), bank: Some(self.val(&bank)), doc_pos: r.doc_pos.as_ref() };
        let grads = AttnInputGrads { d_sink: Some(&d_sink), d_rel: Some(&d_rel), d_bank: Some(&d_bank) };
        s.extend(flash_ext::flash_ext_bwd(g, &self.k.flash, &heads, &opts, &inputs, Contiguous { b: r.b, t: r.t, pos0: 0 }, &a.q, &a.kv[0], &a.kv[1], &a.out, &a.lse, &d_out, &d_q, &d_k, &d_v, &grads));
        self.add_into(s, self.grad(&bank), &d_bank, cfg.d_rel * cfg.bias_extent(layer));
        self.proj_bwd(s, &ln.sink(), qh, &d_sink, &d_qh, n, true);
        let d_rel_pre = self.l2clip_bwd(s, &a.rel_pre, Some(&ln.rel_scale()), &d_rel, n * at.heads, at.heads, cfg.d_rel, cfg.rel_clip);
        self.proj_bwd(s, &ln.wr(), qh, &d_rel_pre, &d_qh, n, true);
        let d_q_pre = self.l2clip_bwd(s, &a.q_pre, Some(&ln.q_scale()), &d_q, n * at.heads, at.heads, at.head_dim, cfg.qk_clip);
        self.proj_bwd(s, &ln.wq(), qh, &d_q_pre, &d_qh, n, true);
        // Keys and values: their normalisation, convolution and projections.
        for (j, (input, d_input)) in [(kh, &d_kh), (vh, &d_vh)].into_iter().enumerate() {
            let c = at.kv_dim();
            let d_conv = if j == 0 {
                self.l2clip_bwd(s, &a.kv_conv[0], None, &d_k, n * at.kv_heads, at.kv_heads, at.head_dim, cfg.qk_clip)
            } else {
                let dx = self.fresh(n * c);
                s.push(g.dispatch(self.k.rms_nogain_dx, &[&a.kv_conv[1], &d_v, &dx], &[at.head_dim, n * at.kv_heads, f(cfg.eps)], Dispatch::Workgroups(n * at.kv_heads)));
                dx
            };
            let d_pre = if full {
                let dyn_taps = a.kv_dyn[j].as_ref().expect("a full layer's dynamic taps");
                let stride = at.kv_heads * cfg.conv_taps;
                let (dx, d_dyn) = (self.fresh(n * c), self.fresh(n * stride));
                let base = ln.kv_conv_static(j == 1);
                s.extend(short_conv_bwd(g, &self.k.conv, &self.dynamic_shape(r, layer), &a.kv_pre[j], &d_conv, Some(dyn_taps), self.val(&base), r.seg.as_ref(), &dx, self.grad(&base), Some(&d_dyn)));
                self.proj_bwd(s, &ln.kv_conv_proj(j == 1), input, &d_dyn, d_input, n, true);
                dx
            } else {
                self.static_conv_bwd(s, r, &a.kv_pre[j], &ln.kv_conv(j == 1), c, &d_conv)
            };
            let w = if j == 0 { ln.wk() } else { ln.wv() };
            self.proj_bwd(s, &w, input, &d_pre, d_input, n, true);
        }
        [d_qh, d_kh, d_vh]
    }

    /// The experts' and the router's backward from the gradient of the MoE output: the gradient of its normed input
    /// (fresh); the router weight's gradient added in a full fine-tune, expert banks' or adapters' gradients added.
    fn moe_bwd(&self, s: &mut Vec<Step>, r: &Rows, v: &Visit, hn: &DeviceBuffer, mf: &MoeFwd, d_out: &DeviceBuffer) -> Result<DeviceBuffer, String> {
        let cfg = &self.tc.cfg;
        let (g, n, layer) = (self.g(), r.n(), v.layer as u32);
        let ln = LayerNames::new(layer);
        let (shape, spec) = self.moe_shape(n);
        let (e, d, ff) = (cfg.experts, cfg.hidden, cfg.moe_ff);
        let d_hn = self.fresh(n * d);
        let (d_gate, d_logits) = (self.fresh(n * e), self.fresh(n * e));
        let mut router_w = Vec::new();
        self.proj_bwd(&mut router_w, &ln.router(), hn, &d_logits, &d_hn, n, false);
        let router = if self.detach_router.get() {
            RouterGrad::Detached
        } else {
            RouterGrad::Full { ids: &self.k.router_bwd, kind: router_kind(cfg.route_scale), logits: &mf.logits, fe: None, d_gate: &d_gate, d_router_logits: &d_logits, router_weight_bwd: &router_w }
        };
        let sb_bufs = [self.fresh(n * d), self.fresh(n * ff), self.fresh(n * 2 * ff), self.fresh(1)];
        let sb = GluBwdScratch { d_expert_out: &sb_bufs[0], d_h: &sb_bufs[1], d_gate_up: &sb_bufs[2], d_up: &sb_bufs[3] };
        let bank = self.bank(v.layer);
        let dense;
        let (set, grads): (ExpertSet, Vec<GluGrads>) = match &bank {
            BankRef::Dense { w13, w2 } => {
                dense = self.dense_weights(w13, w2);
                let (g13, g2) = (self.lin_grad(&ln.w13()), self.lin_grad(&ln.w2()));
                let grads = (0..e).map(|_| GluGrads { gate_w: g13, up_w: None, down_w: g2 }).collect();
                (ExpertSet { glu: spec, bank: ExpertBank::Dense(&dense) }, grads)
            }
            BankRef::Micro { w13, w2 } => (ExpertSet { glu: spec, bank: ExpertBank::Micro { gate_up: w13, up: None, down: w2 } }, Vec::new()),
        };
        s.extend(model::moe::moe_layer_bwd_adapted(g, router, &self.k.moe_bwd, &set, &grads, &self.expert_adapters(layer), &self.expert_adapter_grads(layer), &shape, hn, &mf.gate, &mf.acts, &sb, d_out, &d_hn)?);
        Ok(d_hn)
    }
}
