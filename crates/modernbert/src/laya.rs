// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Laya's decision head: the from-scratch part trained on top of the frozen
//! `ModernBert` backbone, per `convaiinnovations/laya`'s own `rl_common.py`
//! (`DecisionModel`, Apache-2.0).
//!
//! ```text
//! h0     = hidden + type_emb[qtype]                broadcast per SEQUENCE (span), not per row
//! per head layer l (standard PyTorch nn.TransformerEncoderLayer, norm_first=True):
//!   xn   = LayerNorm1(x[l])                         BIASED - opposite of the trunk's norm_bias:false
//!   qkv  = xn @ Win^T + bin                          [rows, 3H], fused in_proj, BIASED
//!   ctx  = bidirectional self-attention, per span (block::chunked_bidir_fwd, NO RoPE, NO window -
//!          nn.MultiheadAttention has no positional encoding at all and the head always sees the
//!          whole sequence)
//!   attn_out = ctx @ Wout^T + bout                   BIASED
//!   res1 = x[l] + attn_out
//!   xn2  = LayerNorm2(res1)                          BIASED
//!   ff   = relu(xn2 @ W1^T + b1) @ W2^T + b2          PLAIN RELU, NOT GELU - see the trap note below
//!   x[l+1] = res1 + ff
//! m      = gather(x[head_layers], marker_rows)        one row per option, at its [MASK] token
//! logits = Linear(GELU(Linear(LayerNorm(m))))          per-option score, GELU here IS correct
//! pooled = x[head_layers][row0 of each span]           the [CLS] row of the HEAD's own output
//! feats  = [top1_prob, margin, normalized_entropy, arity/255]   HOST, from a detached softmax(logits)
//! act_logits = Linear(GELU(Linear(concat(pooled, feats))))
//! ```
//!
//! **Two traps, verified against the real PyTorch construction this session,
//! not guessed:**
//!
//! - **The 2 head layers' FFN uses RELU, not GELU.** `rl_common.py` builds
//!   `nn.TransformerEncoderLayer(d, nhead, 4*d, dropout, batch_first=True,
//!   norm_first=True)` with no `activation=` override, so PyTorch's default
//!   (`F.relu`) applies - confirmed by constructing that exact layer and
//!   printing `.activation` in this container. This is the OPPOSITE of the
//!   scorer's own explicit `nn.GELU()` and of `ModernBert`'s own trunk GELU -
//!   easy to get backwards, and exactly what `tests/laya_parity.rs` exists to
//!   catch (see that test's own note on deliberately breaking this).
//! - **The 2 head layers' LayerNorm/Linear/attention are all BIASED**, unlike
//!   the frozen trunk's `norm_bias: false` convention - confirmed both by
//!   `nn.TransformerEncoderLayer`'s defaults (`bias=True`) and by the real
//!   released checkpoint's own tensor names (`head.layers.{0,1}.
//!   self_attn.in_proj_bias`, `.norm1.bias`, etc., all present). This module
//!   therefore uses the standard biased LayerNorm/GEMM+bias-add path
//!   (`block::layernorm_fwd` with a real `beta`, `k.bias_add`), never
//!   `block::layernorm_nobias_fwd` - that kernel belongs to the trunk only.
//!
//! **No RoPE, no window.** `nn.MultiheadAttention` has zero built-in
//! positional encoding, and the head always attends over a whole packed span
//! regardless of which `ModernBert` layer-type pattern (full/local) applied
//! below it - `block::chunked_bidir_fwd` is called directly, the same
//! call `ModernBert`'s own FULL-attention layers use, with no RoPE dispatch
//! beforehand and no window parameter.
//!
//! **The four calibration features and the softmax over options stay on the
//! host**, mirroring `crates/decide/src/head.rs`'s own "the softmax is not
//! here" decision: no kernel has an opinion about how many options a
//! question has, and the reference implementation itself detaches the
//! softmax before computing them (`p = torch.softmax(logits.detach(), -1)`),
//! so there is no gradient to preserve on the device side even once a
//! backward pass exists (Laya M5). [`LayaHead::forward`] therefore reads the
//! option logits and the pooled `[CLS]` rows back to the host, computes the
//! four features and concatenates them there, uploads the small
//! `[n_questions, H+4]` result and runs `act_head` as a second, tiny device
//! dispatch - two round trips per call, both over a few hundred floats at
//! most, not the encoder's own hidden state.
//!
//! **No padding, so no `marker_mask`.** The reference pads every question in
//! a batch to the widest option count and masks the rest to `-1e4` before the
//! softmax; this crate's caller supplies exactly `arity[i]` real marker rows
//! per question (the same runtime-defined-option-space convention
//! `crates/decide::Request` already uses), so there is nothing to mask -
//! `top2`/`margin`/entropy are computed directly over the real option count,
//! which reduces to the same numbers the padded computation would have
//! produced (a `-1e4` logit softmaxes to essentially zero probability and
//! never appears in a top-2).
//!
//! **M3 was forward-only; M5 adds the seeded backward** - `crates/decide/src
//! /head.rs`'s own `Bwd` split, but bigger, since this head is a real
//! multi-layer transformer rather than one cross-attention block.
//!
//! # Backward (Laya M5)
//!
//! [`LayaHead::new_train_on`] + [`LayaHead::set_call_train`] +
//! [`LayaHead::backward`] mirror `decide::head::Head`'s own
//! `new_on(train=true)`/`set_call`/`backward` split. Three pieces worth
//! naming:
//!
//! - **The plain-ReLU FFN's backward needs the PRE-activation value**, which
//!   M3's own `relu_inplace` dispatch destroys (it overwrites `ff1` in
//!   place). The forward now writes the activation into a SEPARATE `ff1_act`
//!   buffer via `leaky_relu` (existing kernel, `slope=0.0` computes exactly
//!   plain ReLU) instead - a non-behavioral change (same numbers, different
//!   buffer) that both a frozen and a trainable head now share, so there is
//!   still only ONE `build_steps`. `leaky_relu_bwd` at the same `slope=0.0`
//!   is the adjoint - no new kernel for either direction.
//! - **The `h0 = hidden + type_emb[qtype]` broadcast add's backward reuses
//!   the generic `emb_bwd` scatter**, not a new kernel: `type_row_ids` (one
//!   qtype id PER ROW, already built and uploaded by [`LayaHead::set_call`]
//!   for the forward broadcast-gather) is EXACTLY the index buffer
//!   `emb_bwd_step` wants, so `grad(type_emb.weight)[q,:] += sum_{row:
//!   type_row_ids[row]==q} d_h0[row,:]` falls out of the existing kernel with
//!   no new index-construction logic.
//! - **The `act_head` detach is implemented, not skipped**: the real
//!   checkpoint computes its four calibration features from a
//!   `p = torch.softmax(logits.detach(), -1)`, so gradient through the loss
//!   never reaches the option logits via the features path in the real
//!   training recipe - only via `pooled` (the first `d` columns of
//!   `concat`). This backward reproduces that exactly: `act.fc1`'s `dx`
//!   produces a `[q, d+4]` gradient, and the existing `concat_split` kernel
//!   (built for NCHW concat-backward elsewhere, `H=W=1` here) extracts ONLY
//!   the first `d` columns into `d_pooled`; the last 4 (the features'
//!   own share) are computed and then deliberately discarded, never
//!   contributing to any parameter's gradient. This is a conscious choice,
//!   not a default: the alternative (differentiate the host-side softmax/
//!   top-k/entropy formula and route gradient back into `logits` too) is
//!   possible but diverges from how the real checkpoint was actually
//!   trained, and M5's own bar is internal consistency (gradcheck), not
//!   recipe fidelity - so a future fine-tuning milestone that wants the
//!   non-detached path needs to add it deliberately, not assume it.
//! - **`dx[head_layers]` (the head's own top-of-stack gradient) is CLEARED
//!   FIRST**, exactly `decide::head::Head::backward`'s own documented reason:
//!   the marker gather's scatter and the pooled gather's scatter both
//!   ACCUMULATE onto it, and nothing else assigns the rows either omits.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

/// Attention-slab budget for the head's own chunked self-attention - same
/// reasoning as `crate::model`'s `SLAB_BUDGET`.
const SLAB_BUDGET: u64 = 256 << 20;

/// WebGPU's `min_storage_buffer_offset_alignment` - the head's own qkv
/// buffer is sliced per span the same way the trunk's is.
const BIND_ALIGN: u64 = 256;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Shape of Laya's decision head. `head_layers` is a real config knob, not a
/// hardcoded `2` - the released checkpoint's typed-decisions variant may use
/// a different count (see the plan's own honesty note).
#[derive(Clone, Debug)]
pub struct LayaConfig {
    /// Must equal the backbone's `ModernBertConfig::d_model` - the head reads
    /// the encoder's hidden buffer directly, at this width.
    pub d_model: u32,
    /// Pre-norm self-attention layers over the whole sequence, before the
    /// marker gather. `2` on the released checkpoint.
    pub head_layers: u32,
    /// Self-attention heads in the head's OWN layers - independent of the
    /// backbone's own head count. `rl_common.py`: `nhead = max(1, d // 64)`.
    pub n_heads: u32,
    /// FFN width multiplier inside each head layer (`4*d`, the
    /// `nn.TransformerEncoderLayer` default `dim_feedforward`).
    pub ff_mult: u32,
    /// `act_head`'s hidden width (`256` on the released checkpoint).
    pub act_hidden: u32,
    /// `act_head`'s output width (`2`: continue vs escalate).
    pub n_act: u32,
    /// LayerNorm epsilon - `1e-5`, `torch.nn.LayerNorm`'s own default.
    pub eps: f32,
}

impl LayaConfig {
    /// Derive the head's shape from the backbone's own `d_model`, following
    /// `rl_common.py`'s `nhead = max(1, d // 64)` exactly - not a free
    /// parameter, a formula, so a caller cannot pass a value the real
    /// checkpoint's state dict would disagree with.
    pub fn new(d_model: u32) -> LayaConfig {
        LayaConfig {
            d_model,
            head_layers: 2,
            n_heads: (d_model / 64).max(1),
            ff_mult: 4,
            act_hidden: 256,
            n_act: 2,
            eps: 1e-5,
        }
    }

    fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
}

/// Parameter shapes, in one place for the store, the importer (Laya M4) and
/// the init. A SEPARATE store from `ModernBertConfig::tensor_manifest` - the
/// encoder is frozen/pretrained and this head starts from scratch with its
/// own learning rate later, exactly `crates/decide`'s `Encoder`/`Head` split.
///
/// Names are this crate's own choice (M4's importer maps the real checkpoint's
/// dotted `head.layers.{0,1}.*`/`scorer.*`/`act_head.*`/`type_emb.*` names
/// onto these); the SHAPES are ground truth, verified against the real
/// released `model.safetensors` header.
pub fn tensor_manifest(cfg: &LayaConfig) -> Vec<(String, Vec<usize>)> {
    let d = cfg.d_model as usize;
    let ff = (cfg.ff_mult * cfg.d_model) as usize;
    let mut v = vec![("type_emb.weight".into(), vec![3, d])];
    for l in 0..cfg.head_layers as usize {
        let p = format!("head.{l}");
        v.extend([
            (format!("{p}.attn.in_proj.weight"), vec![3 * d, d]),
            (format!("{p}.attn.in_proj.bias"), vec![3 * d]),
            (format!("{p}.attn.out_proj.weight"), vec![d, d]),
            (format!("{p}.attn.out_proj.bias"), vec![d]),
            (format!("{p}.norm1.weight"), vec![d]),
            (format!("{p}.norm1.bias"), vec![d]),
            (format!("{p}.ff1.weight"), vec![ff, d]),
            (format!("{p}.ff1.bias"), vec![ff]),
            (format!("{p}.ff2.weight"), vec![d, ff]),
            (format!("{p}.ff2.bias"), vec![d]),
            (format!("{p}.norm2.weight"), vec![d]),
            (format!("{p}.norm2.bias"), vec![d]),
        ]);
    }
    v.extend([
        ("scorer.norm.weight".into(), vec![d]),
        ("scorer.norm.bias".into(), vec![d]),
        ("scorer.fc1.weight".into(), vec![d, d]),
        ("scorer.fc1.bias".into(), vec![d]),
        ("scorer.fc2.weight".into(), vec![1, d]),
        ("scorer.fc2.bias".into(), vec![1]),
        ("act.fc1.weight".into(), vec![cfg.act_hidden as usize, d + 4]),
        ("act.fc1.bias".into(), vec![cfg.act_hidden as usize]),
        ("act.fc2.weight".into(), vec![cfg.n_act as usize, cfg.act_hidden as usize]),
        ("act.fc2.bias".into(), vec![cfg.n_act as usize]),
    ]);
    v
}

struct HeadLayerBufs {
    xn: DeviceBuffer,
    qkv: DeviceBuffer,
    ctx: DeviceBuffer,
    attn_out: DeviceBuffer,
    res1: DeviceBuffer,
    xn2: DeviceBuffer,
    /// The FFN's PRE-activation - never overwritten (unlike M3's original
    /// in-place `relu_inplace`), because the backward's `leaky_relu_bwd`
    /// needs it back. See the module doc's "Backward" section.
    ff1: DeviceBuffer,
    /// The FFN's POST-activation, `leaky_relu(ff1, slope=0.0)` - a separate
    /// buffer from `ff1` for the same reason.
    ff1_act: DeviceBuffer,
    ff2: DeviceBuffer,
}

/// Reverse-pass buffers and the recorded backward step list, allocated only
/// by [`LayaHead::new_train_on`] - mirrors `decide::head::Head`'s own `Bwd`,
/// scaled up for this head's real multi-layer transformer shape. Every entry
/// is a GRADIENT; the activations it reads are the forward's own cached
/// buffers. Per-layer scratch is shared across layers; only `dx` is per
/// layer.
struct Bwd {
    /// Seed: `dL/d(logits)`, one per option marker. COPY_DST.
    d_logits: DeviceBuffer,
    /// Seed: `dL/d(act_logits)`, `[n_questions, n_act]`. COPY_DST.
    d_act_logits: DeviceBuffer,
    // ---- scorer backward scratch, `[cap_markers, *]` ----
    d_scorer_gelu: DeviceBuffer,
    d_scorer_fc1: DeviceBuffer,
    d_scorer_ln: DeviceBuffer,
    d_gathered: DeviceBuffer,
    mean_m: DeviceBuffer,
    inv_m: DeviceBuffer,
    // ---- act_head backward scratch, `[cap_questions, *]` ----
    d_act_gelu: DeviceBuffer,
    d_act_fc1: DeviceBuffer,
    /// `[cap_questions, d+4]` - the full `dx` of `act.fc1`, BEFORE the
    /// detach split. See the module doc's "act_head detach" note.
    d_concat: DeviceBuffer,
    /// `[cap_questions, d]` - `d_concat`'s first `d` columns only, via
    /// `concat_split`. The last 4 columns are computed into `d_concat` and
    /// then never read again - the detach.
    d_pooled: DeviceBuffer,
    // ---- per-head-layer loop scratch, `[cap_rows, *]`, reused per layer ----
    /// `dx[l]` = grad of `x[l]`, for `l` in `1..=head_layers`. `dx[l]` at
    /// `l == head_layers` is CLEARED then scatter-accumulated by both
    /// gathers before the loop starts; `l == 0`'s own gradient is written
    /// directly into the caller-supplied trunk seed buffer, not stored here.
    dx: Vec<DeviceBuffer>,
    d_res1: DeviceBuffer,
    d_ff1_act: DeviceBuffer,
    d_ff1: DeviceBuffer,
    d_xn2: DeviceBuffer,
    /// Scratch for a pre-norm's `dx` output before it re-joins the residual -
    /// reused for both LN2's and LN1's backward within one layer (the two
    /// uses never overlap - see `crate::model::ModernBert`'s own `d_tmp` for
    /// the identical reasoning).
    d_tmp: DeviceBuffer,
    d_xn: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_scores: DeviceBuffer,
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    steps: Vec<Step>,
}

/// Laya's decision head. `Role::Frozen` by [`LayaHead::new_on`] (inference
/// only); [`LayaHead::new_train_on`] builds the SAME forward (`Role::
/// Trainable` weights) plus the seeded backward - see the module doc's
/// "Backward" section.
pub struct LayaHead {
    gpu: Gpu,
    k: crate::kern::Ids,
    cfg: LayaConfig,
    pub ps: ParamStore,
    cap_rows: u32,
    cap_markers: u32,
    cap_questions: u32,
    chunk: u32,
    rows: u32,
    n_markers: u32,
    n_questions: u32,
    spans: Vec<(u32, u32)>,
    arity: Vec<usize>,
    type_row_ids: DeviceBuffer,
    marker_rows: DeviceBuffer,
    cls_rows: DeviceBuffer,
    e_type: DeviceBuffer,
    /// `x[0]` = hidden + type_emb broadcast, `x[l+1]` = head layer `l`'s
    /// output; `x[head_layers]` is what the marker and pooled gathers read.
    x: Vec<DeviceBuffer>,
    layers: Vec<HeadLayerBufs>,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    gathered: DeviceBuffer,
    scorer_ln: DeviceBuffer,
    scorer_fc1: DeviceBuffer,
    scorer_gelu: DeviceBuffer,
    logits: DeviceBuffer,
    pooled: DeviceBuffer,
    concat: DeviceBuffer,
    act_fc1: DeviceBuffer,
    act_gelu: DeviceBuffer,
    act_logits: DeviceBuffer,
    steps: Vec<Step>,
    act_steps: Vec<Step>,
    bwd: Option<Bwd>,
}

impl LayaHead {
    /// Build on an existing device, sized for at most `cap_rows` packed
    /// encoder rows (must not exceed the `ModernBert` this head attaches to),
    /// a longest span of `max_span`, `cap_markers` option markers and
    /// `cap_questions` questions in one call.
    pub fn new_on(
        gpu: Gpu,
        cfg: LayaConfig,
        cap_rows: u32,
        max_span: u32,
        cap_markers: u32,
        cap_questions: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> LayaHead {
        LayaHead::build(gpu, cfg, cap_rows, max_span, cap_markers, cap_questions, init, false)
    }

    /// A **trainable** head: every parameter `Role::Trainable` (gradient +
    /// AdamW moments) plus the reverse step list. Use [`LayaHead::
    /// set_call_train`] (not [`LayaHead::set_call`]) so the backward has
    /// somewhere to write the trunk's seed gradient.
    pub fn new_train_on(
        gpu: Gpu,
        cfg: LayaConfig,
        cap_rows: u32,
        max_span: u32,
        cap_markers: u32,
        cap_questions: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> LayaHead {
        LayaHead::build(gpu, cfg, cap_rows, max_span, cap_markers, cap_questions, init, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        gpu: Gpu,
        cfg: LayaConfig,
        cap_rows: u32,
        max_span: u32,
        cap_markers: u32,
        cap_questions: u32,
        init: &HashMap<String, Vec<f32>>,
        train: bool,
    ) -> LayaHead {
        assert!(max_span <= cap_rows, "max_span {max_span} > cap_rows {cap_rows}");
        let role = if train { Role::Trainable } else { Role::Frozen };
        let roles: Vec<(String, usize, Role)> =
            tensor_manifest(&cfg).into_iter().map(|(n, s)| (n, s.iter().product::<usize>(), role)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let n = cap_rows as u64;
        let d = cfg.d_model as u64;
        let ff = (cfg.ff_mult * cfg.d_model) as u64;
        let per_row = cfg.n_heads as u64 * max_span as u64 * 4;
        let chunk = ((SLAB_BUDGET / per_row.max(1)).max(1) as u32).min(max_span.max(1));
        let slab = cfg.n_heads as u64 * chunk as u64 * max_span as u64;
        let head_layers = cfg.head_layers;

        let idbuf = |name: &str, cap: u64| {
            gpu.buffer(name, cap * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)
        };
        let layers: Vec<HeadLayerBufs> = (0..cfg.head_layers)
            .map(|_| HeadLayerBufs {
                xn: gpu.storage(n * d),
                qkv: gpu.storage(n * 3 * d),
                ctx: gpu.storage(n * d),
                attn_out: gpu.storage(n * d),
                res1: gpu.storage(n * d),
                xn2: gpu.storage(n * d),
                ff1: gpu.storage(n * ff),
                ff1_act: gpu.storage(n * ff),
                ff2: gpu.storage(n * d),
            })
            .collect();
        let m = cap_markers as u64;
        let q = cap_questions as u64;
        let act_hidden = cfg.act_hidden as u64;
        let n_act = cfg.n_act as u64;
        let k = crate::kern::Ids::resolve(&gpu);
        let mut head = LayaHead {
            k,
            cap_rows,
            cap_markers,
            cap_questions,
            chunk,
            rows: 0,
            n_markers: 0,
            n_questions: 0,
            spans: Vec::new(),
            arity: Vec::new(),
            type_row_ids: idbuf("type_row_ids", n),
            marker_rows: idbuf("marker_rows", m),
            cls_rows: idbuf("laya_cls_rows", q),
            e_type: gpu.storage(n * d),
            x: (0..=cfg.head_layers).map(|_| gpu.storage(n * d)).collect(),
            layers,
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
            gathered: gpu.storage(m * d),
            scorer_ln: gpu.storage(m * d),
            scorer_fc1: gpu.storage(m * d),
            scorer_gelu: gpu.storage(m * d),
            logits: gpu.storage(m),
            pooled: gpu.storage(q * d),
            concat: gpu.buffer(
                "laya_concat",
                q * (d + 4) * 4,
                gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
            ),
            act_fc1: gpu.storage(q * act_hidden),
            act_gelu: gpu.storage(q * act_hidden),
            act_logits: gpu.storage(q * n_act),
            steps: Vec::new(),
            act_steps: Vec::new(),
            bwd: None,
            gpu,
            cfg,
            ps,
        };
        if train {
            let st = |w: u64| head.gpu.storage(w);
            head.bwd = Some(Bwd {
                d_logits: head.gpu.buffer("d_logits", m * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST),
                d_act_logits: head.gpu.buffer("d_act_logits", q * n_act * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST),
                d_scorer_gelu: st(m * d),
                d_scorer_fc1: st(m * d),
                d_scorer_ln: st(m * d),
                d_gathered: st(m * d),
                mean_m: st(m),
                inv_m: st(m),
                d_act_gelu: st(q * act_hidden),
                d_act_fc1: st(q * act_hidden),
                d_concat: st(q * (d + 4)),
                d_pooled: st(q * d),
                dx: (0..head_layers).map(|_| st(n * d)).collect(),
                d_res1: st(n * d),
                d_ff1_act: st(n * ff),
                d_ff1: st(n * ff),
                d_xn2: st(n * d),
                d_tmp: st(n * d),
                d_xn: st(n * d),
                d_ctx: st(n * d),
                d_qkv: st(n * 3 * d),
                d_scores: st(slab),
                mean: st(n),
                inv: st(n),
                steps: Vec::new(),
            });
        }
        head
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    /// Point the head at one call: `spans` are the encoder's own packed
    /// `(row0, len)` per question (its `row0` doubles as that question's
    /// `[CLS]` row for both the type-embedding broadcast and the pooled
    /// readout), `qtype` is one type-table index (0/1/2) per span,
    /// `marker_rows` is the ABSOLUTE packed row of every option's `[MASK]`
    /// token, flat and grouped consecutively per question in span order, and
    /// `arity` is how many of them belong to each question - the same shape
    /// `crates/decide::Request` already uses for `cls_rows`/`arity`.
    pub fn set_call(&mut self, hidden: &DeviceBuffer, spans: &[(u32, u32)], qtype: &[u32], marker_rows: &[u32], arity: &[usize]) {
        assert_eq!(spans.len(), qtype.len(), "one qtype per question");
        assert_eq!(spans.len(), arity.len(), "one arity per question");
        assert!(!spans.is_empty(), "a call needs at least one question");
        let total: usize = arity.iter().sum();
        assert_eq!(total, marker_rows.len(), "arity must sum to marker_rows.len()");
        assert!(arity.iter().all(|&a| a >= 1), "every question needs at least one option");
        let rows: u32 = spans.iter().map(|&(_, l)| l).sum();
        assert!(rows <= self.cap_rows, "{rows} rows > capacity {}", self.cap_rows);
        assert!(marker_rows.len() <= self.cap_markers as usize, "{} markers > capacity {}", marker_rows.len(), self.cap_markers);
        assert!(spans.len() <= self.cap_questions as usize, "{} questions > capacity {}", spans.len(), self.cap_questions);
        let d = self.cfg.d_model as u64;
        for &(row0, _) in spans {
            let off = row0 as u64 * 3 * d * 4;
            assert_eq!(
                off % BIND_ALIGN,
                0,
                "span starting at row {row0} binds the head's qkv buffer at byte {off}, not a multiple of {BIND_ALIGN}; \
                 with d_model {d} a span may only start on a row that is a multiple of {}",
                (BIND_ALIGN / gcd(BIND_ALIGN, 3 * d * 4)).max(1)
            );
        }

        let mut type_row_ids = vec![0u32; rows as usize];
        for (&(row0, len), &qt) in spans.iter().zip(qtype) {
            for i in 0..len {
                type_row_ids[(row0 + i) as usize] = qt;
            }
        }
        self.gpu.write(&self.type_row_ids, &type_row_ids);
        self.gpu.write(&self.marker_rows, marker_rows);
        let cls_rows: Vec<u32> = spans.iter().map(|&(row0, _)| row0).collect();
        self.gpu.write(&self.cls_rows, &cls_rows);

        self.rows = rows;
        self.n_markers = marker_rows.len() as u32;
        self.n_questions = spans.len() as u32;
        self.spans = spans.to_vec();
        self.arity = arity.to_vec();
        // Rebuilt unconditionally, same reasoning as `decide::head::Head::
        // set_call`: the step list holds the encoder's hidden buffer and this
        // call's row/marker/question counts, so a caller that changed any of
        // them would otherwise keep dispatching against stale shapes.
        self.steps = self.build_steps(hidden);
        self.act_steps = self.build_act_steps();
    }

    /// Run the head and return `(option logits, act logits)`, both flat and
    /// in pack order - `option logits` grouped consecutively by `arity` per
    /// question (mirroring `decide::head::Head::forward`'s own flat score
    /// list), `act logits` `[n_questions, n_act]` row-major.
    pub fn forward(&self) -> (Vec<f32>, Vec<f32>) {
        self.gpu.submit(&[], &self.steps);
        let logits = self.gpu.read(&self.logits, self.n_markers as usize);
        let d = self.cfg.d_model as usize;
        let pooled = self.gpu.read(&self.pooled, self.n_questions as usize * d);

        // The four calibration features: a DETACHED softmax over each
        // question's own real options (no masking needed - see the module
        // doc), computed on the host exactly like `crates/decide`'s own
        // option-count-agnostic scoring.
        let mut concat = vec![0.0f32; self.n_questions as usize * (d + 4)];
        let mut at = 0usize;
        for (qi, &k) in self.arity.iter().enumerate() {
            let opts = &logits[at..at + k];
            at += k;
            let maxv = opts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = opts.iter().map(|&v| (v - maxv).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let p: Vec<f32> = exps.iter().map(|&e| e / sum.max(1e-30)).collect();
            let mut sorted = p.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top1 = sorted.first().copied().unwrap_or(0.0);
            let top2 = sorted.get(1).copied().unwrap_or(0.0);
            let margin = top1 - top2;
            let kf = (k as f32).max(2.0);
            let ent = -p.iter().map(|&pi| pi * pi.max(1e-9).ln()).sum::<f32>() / kf.ln();
            let feat = [top1, margin, ent, k as f32 / 255.0];
            let row_out = &mut concat[qi * (d + 4)..(qi + 1) * (d + 4)];
            row_out[..d].copy_from_slice(&pooled[qi * d..(qi + 1) * d]);
            row_out[d..].copy_from_slice(&feat);
        }
        self.gpu.write_f32(&self.concat, &concat);
        self.gpu.submit(&[], &self.act_steps);
        let act_logits = self.gpu.read(&self.act_logits, self.n_questions as usize * self.cfg.n_act as usize);
        (logits, act_logits)
    }

    /// Block until this device has finished what it was given.
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    fn build_steps(&self, hidden: &DeviceBuffer) -> Vec<Step> {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let ff = self.cfg.ff_mult * self.cfg.d_model;
        let n = self.rows;
        let heads = self.cfg.n_heads;
        let hd = self.cfg.head_dim();
        let ln = block::LayerNormIds::resolve_fwd(g, self.k.layernorm);
        let cross = block::CrossIds { scores: self.k.scores_cross, softmax: self.k.softmax_cross, apply: self.k.apply_cross };

        // h0 = hidden + type_emb[qtype], broadcast per span via a per-row
        // gather - the same `embed`-as-broadcast idiom `ModernBert::model`
        // uses for its own position/segment tables.
        let mut s = vec![g.step(self.k.embed, &[&self.type_row_ids, self.w("type_emb.weight"), &self.e_type], &[d, n], n * d)];
        s.push(g.step(self.k.add2, &[hidden, &self.e_type, &self.x[0]], &[n * d], n * d));

        for l in 0..self.cfg.head_layers as usize {
            let lb = &self.layers[l];
            let p = format!("head.{l}");

            s.push(block::layernorm_fwd(g, &ln, &self.x[l], self.w(&format!("{p}.norm1.weight")), self.w(&format!("{p}.norm1.bias")), &lb.xn, d, n, self.cfg.eps));
            s.push(g.step(self.k.matmul, &[&lb.xn, self.w(&format!("{p}.attn.in_proj.weight")), &lb.qkv], &[n, d, 3 * d], n * 3 * d));
            s.push(g.step(self.k.bias_add, &[&lb.qkv, self.w(&format!("{p}.attn.in_proj.bias"))], &[n, 3 * d], n * 3 * d));

            // Plain full bidirectional self-attention, per span - no RoPE, no
            // window, the same primitive `ModernBert`'s own FULL-attention
            // layers use, called directly rather than through the trunk.
            block::chunked_bidir_fwd(
                g, &cross, None, heads, hd, d, &lb.qkv, 3 * d, 0, d, 2 * d, &lb.ctx, &self.scores, &self.probs, &self.spans, self.chunk, None, &mut s,
            );

            s.push(g.step(self.k.matmul, &[&lb.ctx, self.w(&format!("{p}.attn.out_proj.weight")), &lb.attn_out], &[n, d, d], n * d));
            s.push(g.step(self.k.bias_add, &[&lb.attn_out, self.w(&format!("{p}.attn.out_proj.bias"))], &[n, d], n * d));
            s.push(g.step(self.k.add2, &[&self.x[l], &lb.attn_out, &lb.res1], &[n * d], n * d));

            s.push(block::layernorm_fwd(g, &ln, &lb.res1, self.w(&format!("{p}.norm2.weight")), self.w(&format!("{p}.norm2.bias")), &lb.xn2, d, n, self.cfg.eps));
            s.push(g.step(self.k.matmul, &[&lb.xn2, self.w(&format!("{p}.ff1.weight")), &lb.ff1], &[n, d, ff], n * ff));
            s.push(g.step(self.k.bias_add, &[&lb.ff1, self.w(&format!("{p}.ff1.bias"))], &[n, ff], n * ff));
            // PLAIN RELU, not GELU - see the module doc's trap note.
            // Non-in-place (`leaky_relu` at `slope=0.0`, not `relu_inplace`):
            // the backward (Laya M5) needs the PRE-activation `ff1` back, so
            // the activation is written into the separate `ff1_act` buffer -
            // see the module doc's "Backward" section. Same numbers either
            // way; only the buffer layout changed.
            s.push(g.step(self.k.leaky_relu, &[&lb.ff1, &lb.ff1_act], &[n * ff, f(0.0)], n * ff));
            s.push(g.step(self.k.matmul, &[&lb.ff1_act, self.w(&format!("{p}.ff2.weight")), &lb.ff2], &[n, ff, d], n * d));
            s.push(g.step(self.k.bias_add, &[&lb.ff2, self.w(&format!("{p}.ff2.bias"))], &[n, d], n * d));
            s.push(g.step(self.k.add2, &[&lb.res1, &lb.ff2, &self.x[l + 1]], &[n * d], n * d));
        }

        let x_final = &self.x[self.cfg.head_layers as usize];

        // Marker gather: one row per option, at its [MASK] token.
        s.push(g.step(self.k.embed, &[&self.marker_rows, x_final, &self.gathered], &[d, self.n_markers], self.n_markers * d));
        s.push(block::layernorm_fwd(g, &ln, &self.gathered, self.w("scorer.norm.weight"), self.w("scorer.norm.bias"), &self.scorer_ln, d, self.n_markers, self.cfg.eps));
        s.push(g.step(self.k.matmul, &[&self.scorer_ln, self.w("scorer.fc1.weight"), &self.scorer_fc1], &[self.n_markers, d, d], self.n_markers * d));
        s.push(g.step(self.k.bias_add, &[&self.scorer_fc1, self.w("scorer.fc1.bias")], &[self.n_markers, d], self.n_markers * d));
        s.push(g.step(self.k.gelu_erf, &[&self.scorer_fc1, &self.scorer_gelu], &[self.n_markers * d], self.n_markers * d));
        s.push(g.step(self.k.matmul, &[&self.scorer_gelu, self.w("scorer.fc2.weight"), &self.logits], &[self.n_markers, d, 1], self.n_markers));
        s.push(g.step(self.k.bias_add, &[&self.logits, self.w("scorer.fc2.bias")], &[self.n_markers, 1], self.n_markers));

        // Pooled [CLS] gather: each question's own row0, post the head layers.
        s.push(g.step(self.k.embed, &[&self.cls_rows, x_final, &self.pooled], &[d, self.n_questions], self.n_questions * d));
        s
    }

    /// `act_head`'s own tiny dispatch, over `concat = [pooled, feats]` -
    /// built once per [`Self::set_call`] (it needs `n_questions`, baked into
    /// the GEMM dispatch size the same way every other step list here bakes
    /// in its row count), but the buffer's CONTENTS are written per
    /// [`Self::forward`] call, after the host has computed the features.
    fn build_act_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let ah = self.cfg.act_hidden;
        let na = self.cfg.n_act;
        let q = self.n_questions;
        let mut s = vec![g.step(self.k.matmul, &[&self.concat, self.w("act.fc1.weight"), &self.act_fc1], &[q, d + 4, ah], q * ah)];
        s.push(g.step(self.k.bias_add, &[&self.act_fc1, self.w("act.fc1.bias")], &[q, ah], q * ah));
        s.push(g.step(self.k.gelu_erf, &[&self.act_fc1, &self.act_gelu], &[q * ah], q * ah));
        s.push(g.step(self.k.matmul, &[&self.act_gelu, self.w("act.fc2.weight"), &self.act_logits], &[q, ah, na], q * na));
        s.push(g.step(self.k.bias_add, &[&self.act_logits, self.w("act.fc2.bias")], &[q, na], q * na));
        s
    }

    /// [`LayaHead::set_call`] for a TRAINABLE head: same validation and
    /// bookkeeping, plus `d_hidden_out` - the trunk's own seed buffer
    /// (`ModernBert::seed_buf`) - which this head's backward writes into
    /// directly. A separate entry point rather than an added parameter on
    /// [`LayaHead::set_call`], whose signature M4's real-weight parity tests
    /// already use and which stays byte-for-byte unchanged.
    pub fn set_call_train(&mut self, hidden: &DeviceBuffer, d_hidden_out: &DeviceBuffer, spans: &[(u32, u32)], qtype: &[u32], marker_rows: &[u32], arity: &[usize]) {
        self.set_call(hidden, spans, qtype, marker_rows, arity);
        let steps = self.build_bwd_steps(hidden, d_hidden_out);
        if let Some(bw) = &mut self.bwd {
            bw.steps = steps;
        }
    }

    /// Whether this head was built trainable.
    pub fn is_trainable(&self) -> bool {
        self.bwd.is_some()
    }

    /// Zero every parameter gradient. Call once per step BEFORE
    /// [`LayaHead::backward`], which accumulates into them.
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Read one parameter's current value.
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.gpu.read(self.w(name), self.numel(name))
    }

    /// Overwrite one parameter - the finite-difference checker's perturbation.
    pub fn set_weight(&self, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.numel(name), "{name}");
        self.gpu.write_f32(self.w(name), data);
    }

    fn numel(&self, name: &str) -> usize {
        tensor_manifest(&self.cfg)
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s.iter().product::<usize>())
            .unwrap_or_else(|| panic!("no parameter {name:?}"))
    }

    /// Read one parameter's accumulated gradient.
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.gpu.read(self.ps.g(name), self.numel(name))
    }

    /// Seed with `dL/d(logits)` (one per option marker) and `dL/d(act_logits)`
    /// (`[n_questions, n_act]`), both host-computed by the loss, and run the
    /// reverse pass - which writes the trunk's seed buffer
    /// (`d_hidden_out`, supplied to [`LayaHead::set_call_train`]) in place.
    pub fn backward(&self, d_logits: &[f32], d_act_logits: &[f32]) {
        let b = self.bwd.as_ref().expect("backward on an inference head");
        assert_eq!(d_logits.len(), self.n_markers as usize, "one gradient per option marker");
        assert_eq!(
            d_act_logits.len(),
            self.n_questions as usize * self.cfg.n_act as usize,
            "[n_questions, n_act] act-logit gradient"
        );
        self.gpu.write_f32(&b.d_logits, d_logits);
        self.gpu.write_f32(&b.d_act_logits, d_act_logits);
        // CLEARED FIRST, and it must be - see the module doc's "Backward"
        // section: the marker gather's and the pooled gather's scatters both
        // ACCUMULATE onto `dx[head_layers]` (the top of the per-layer loop),
        // so a stale value from a previous step would otherwise survive
        // underneath them.
        let top = b.dx.last().expect("head_layers >= 1");
        self.gpu.submit(&[top], &b.steps);
    }

    /// The exact adjoint of [`LayaHead::build_steps`] plus [`LayaHead::
    /// build_act_steps`], walked bottom up: scorer, then act_head (both
    /// scatter onto `dx[head_layers]`), then the head-layer stack down to
    /// `x[0]`, then the type-embedding scatter and the trunk hand-off.
    fn build_bwd_steps(&self, hidden: &DeviceBuffer, d_hidden_out: &DeviceBuffer) -> Vec<Step> {
        let _ = hidden; // the trunk's own hidden buffer is read only by the FORWARD (h0 = hidden + type_emb); the backward's hand-off is purely additive into `d_hidden_out`.
        let g = &self.gpu;
        let b = self.bwd.as_ref().expect("build_bwd_steps in training mode only");
        let d = self.cfg.d_model;
        let ff = self.cfg.ff_mult * self.cfg.d_model;
        let ah = self.cfg.act_hidden;
        let na = self.cfg.n_act;
        let n = self.rows;
        let (m, q) = (self.n_markers, self.n_questions);
        let heads = self.cfg.n_heads;
        let hd = self.cfg.head_dim();
        // `.layernorm` is never dispatched through this struct - only
        // `.ln_stats`/`.layernorm_dx` are read by `ln_stats_fwd`/
        // `layernorm_dx_bwd` - but this head's LayerNorms are BIASED, so
        // `ln_dbeta` is dispatched separately below (unlike the trunk's
        // no-bias backward, which never calls it).
        let ln = block::LayerNormIds {
            layernorm: self.k.layernorm,
            layernorm_rows: None,
            ln_stats: self.k.ln_stats,
            ln_stats_rows: None,
            layernorm_dx: self.k.layernorm_dx,
            layernorm_dx_rows: None,
        };
        let cross = block::CrossIds { scores: self.k.scores_cross, softmax: self.k.softmax_cross, apply: self.k.apply_cross };
        let cross_bwd = block::CrossBwdIds::resolve(g, self.k.dscores_cross, self.k.dq_cross, self.k.dk_cross_acc, self.k.dv_cross_acc);
        let eb = block::EmbBwdIds::resolve(g, self.k.emb_bwd);
        let gr = |name: &str| self.ps.g(name);
        let mut s: Vec<Step> = Vec::new();
        let head_layers = self.cfg.head_layers as usize;
        let top = b.dx.last().expect("head_layers >= 1");

        // ---- scorer: LayerNorm -> Linear -> GELU -> Linear (all BIASED) ----
        s.push(g.step(self.k.bias_grad, &[&b.d_logits, gr("scorer.fc2.bias")], &[m, 1], 1));
        s.push(g.step(self.k.matmul_dw, &[&b.d_logits, &self.scorer_gelu, gr("scorer.fc2.weight")], &[m, d, 1], d));
        s.push(g.step(self.k.matmul_dx, &[&b.d_logits, self.w("scorer.fc2.weight"), &b.d_scorer_gelu], &[m, d, 1, 0], m * d));
        s.push(g.step(self.k.gelu_erf_bwd, &[&self.scorer_fc1, &b.d_scorer_gelu, &b.d_scorer_fc1], &[m * d], m * d));
        s.push(g.step(self.k.bias_grad, &[&b.d_scorer_fc1, gr("scorer.fc1.bias")], &[m, d], d));
        s.push(g.step(self.k.matmul_dw, &[&b.d_scorer_fc1, &self.scorer_ln, gr("scorer.fc1.weight")], &[m, d, d], d * d));
        s.push(g.step(self.k.matmul_dx, &[&b.d_scorer_fc1, self.w("scorer.fc1.weight"), &b.d_scorer_ln], &[m, d, d, 0], m * d));
        s.push(block::ln_stats_fwd(g, &ln, &self.gathered, &b.mean_m, &b.inv_m, d, m, self.cfg.eps));
        s.push(g.step(self.k.ln_dgamma, &[&b.d_scorer_ln, &self.gathered, &b.mean_m, &b.inv_m, gr("scorer.norm.weight")], &[d, m], d));
        s.push(g.step(self.k.ln_dbeta, &[&b.d_scorer_ln, gr("scorer.norm.bias")], &[d, m], d));
        s.push(block::layernorm_dx_bwd(g, &ln, &self.gathered, self.w("scorer.norm.weight"), &b.d_scorer_ln, &b.d_gathered, d, m, self.cfg.eps));
        // Marker gather's adjoint: scatter each option's grad back onto its
        // own `[MASK]` row of `dx[head_layers]`. ACCUMULATES.
        s.push(block::emb_bwd_step(g, &eb, &self.marker_rows, None, &b.d_gathered, top, m, d, self.cap_rows));

        // ---- act_head: Linear -> GELU -> Linear (BIASED), fed
        // concat(pooled, feats) - the DETACH: only `pooled`'s share of
        // `d_concat` continues backward, the feature columns' share is
        // computed and discarded. See the module doc's "act_head detach"
        // note. ----
        s.push(g.step(self.k.bias_grad, &[&b.d_act_logits, gr("act.fc2.bias")], &[q, na], na));
        s.push(g.step(self.k.matmul_dw, &[&b.d_act_logits, &self.act_gelu, gr("act.fc2.weight")], &[q, ah, na], ah * na));
        s.push(g.step(self.k.matmul_dx, &[&b.d_act_logits, self.w("act.fc2.weight"), &b.d_act_gelu], &[q, ah, na, 0], q * ah));
        s.push(g.step(self.k.gelu_erf_bwd, &[&self.act_fc1, &b.d_act_gelu, &b.d_act_fc1], &[q * ah], q * ah));
        s.push(g.step(self.k.bias_grad, &[&b.d_act_fc1, gr("act.fc1.bias")], &[q, ah], ah));
        s.push(g.step(self.k.matmul_dw, &[&b.d_act_fc1, &self.concat, gr("act.fc1.weight")], &[q, d + 4, ah], ah * (d + 4)));
        s.push(g.step(self.k.matmul_dx, &[&b.d_act_fc1, self.w("act.fc1.weight"), &b.d_concat], &[q, d + 4, ah, 0], q * (d + 4)));
        // The detach: keep only the first `d` (pooled) columns of the
        // `[q, d+4]` gradient; the last 4 (features) columns are computed
        // above and never read again.
        s.push(g.step(self.k.concat_split, &[&b.d_concat, &b.d_pooled], &[q, d + 4, d, 0, 1, 1], q * d));
        // Pooled [CLS] gather's adjoint. ACCUMULATES onto what the marker
        // gather above already wrote.
        s.push(block::emb_bwd_step(g, &eb, &self.cls_rows, None, &b.d_pooled, top, q, d, self.cap_rows));

        // ---- the head_layers stack, walked bottom up ----
        for l in (0..head_layers).rev() {
            let lb = &self.layers[l];
            let p = format!("head.{l}");
            // `dx[i]` holds the grad of `x[i+1]`, so `dx[l]` is this layer's
            // own incoming gradient - including `dx[head_layers - 1]`, which
            // IS `top` (the same buffer the two gathers just scattered into).
            let d_out = &b.dx[l];

            // `x[l+1] = res1 + ff2`: addition fans `d_out` out unchanged to
            // both the residual and the ff2 branch.
            s.push(g.step(self.k.bias_grad, &[d_out, gr(&format!("{p}.ff2.bias"))], &[n, d], d));
            s.push(g.step(self.k.matmul_dw, &[d_out, &lb.ff1_act, gr(&format!("{p}.ff2.weight"))], &[n, ff, d], d * ff));
            s.push(g.step(self.k.matmul_dx, &[d_out, self.w(&format!("{p}.ff2.weight")), &b.d_ff1_act], &[n, ff, d, 0], n * ff));
            // ReLU's backward reads the PRE-activation `ff1`, which M5's
            // forward now keeps intact (see the module doc).
            s.push(g.step(self.k.leaky_relu_bwd, &[&lb.ff1, &b.d_ff1_act, &b.d_ff1], &[n * ff, f(0.0)], n * ff));
            s.push(g.step(self.k.bias_grad, &[&b.d_ff1, gr(&format!("{p}.ff1.bias"))], &[n, ff], ff));
            s.push(g.step(self.k.matmul_dw, &[&b.d_ff1, &lb.xn2, gr(&format!("{p}.ff1.weight"))], &[n, d, ff], ff * d));
            s.push(g.step(self.k.matmul_dx, &[&b.d_ff1, self.w(&format!("{p}.ff1.weight")), &b.d_xn2], &[n, d, ff, 0], n * d));

            s.push(block::ln_stats_fwd(g, &ln, &lb.res1, &b.mean, &b.inv, d, n, self.cfg.eps));
            s.push(g.step(self.k.ln_dgamma, &[&b.d_xn2, &lb.res1, &b.mean, &b.inv, gr(&format!("{p}.norm2.weight"))], &[d, n], d));
            s.push(g.step(self.k.ln_dbeta, &[&b.d_xn2, gr(&format!("{p}.norm2.bias"))], &[d, n], d));
            s.push(block::layernorm_dx_bwd(g, &ln, &lb.res1, self.w(&format!("{p}.norm2.weight")), &b.d_xn2, &b.d_tmp, d, n, self.cfg.eps));
            s.push(g.step(self.k.add2, &[d_out, &b.d_tmp, &b.d_res1], &[n * d], n * d));

            // `res1 = x[l] + attn_out`.
            s.push(g.step(self.k.bias_grad, &[&b.d_res1, gr(&format!("{p}.attn.out_proj.bias"))], &[n, d], d));
            s.push(g.step(self.k.matmul_dw, &[&b.d_res1, &lb.ctx, gr(&format!("{p}.attn.out_proj.weight"))], &[n, d, d], d * d));
            s.push(g.step(self.k.matmul_dx, &[&b.d_res1, self.w(&format!("{p}.attn.out_proj.weight")), &b.d_ctx], &[n, d, d, 0], n * d));

            // Plain full bidirectional self-attention backward - no RoPE, no
            // window to undo, the SAME primitive `ModernBert`'s own
            // FULL-attention layers use.
            block::chunked_bidir_bwd(
                g, &cross, None, &cross_bwd, heads, hd, d, &lb.qkv, 3 * d, 0, d, 2 * d, &b.d_ctx, &b.d_qkv, &self.scores, &self.probs, &b.d_scores, &self.spans, self.chunk, None, &mut s,
            );

            s.push(g.step(self.k.bias_grad, &[&b.d_qkv, gr(&format!("{p}.attn.in_proj.bias"))], &[n, 3 * d], 3 * d));
            s.push(g.step(self.k.matmul_dw, &[&b.d_qkv, &lb.xn, gr(&format!("{p}.attn.in_proj.weight"))], &[n, d, 3 * d], 3 * d * d));
            s.push(g.step(self.k.matmul_dx, &[&b.d_qkv, self.w(&format!("{p}.attn.in_proj.weight")), &b.d_xn], &[n, d, 3 * d, 0], n * d));

            s.push(block::ln_stats_fwd(g, &ln, &self.x[l], &b.mean, &b.inv, d, n, self.cfg.eps));
            s.push(g.step(self.k.ln_dgamma, &[&b.d_xn, &self.x[l], &b.mean, &b.inv, gr(&format!("{p}.norm1.weight"))], &[d, n], d));
            s.push(g.step(self.k.ln_dbeta, &[&b.d_xn, gr(&format!("{p}.norm1.bias"))], &[d, n], d));
            s.push(block::layernorm_dx_bwd(g, &ln, &self.x[l], self.w(&format!("{p}.norm1.weight")), &b.d_xn, &b.d_tmp, d, n, self.cfg.eps));

            if l == 0 {
                // `x[0] = hidden + type_emb[qtype]` - addition fans this
                // layer's `x[l]`-share of the gradient straight through onto
                // the TRUNK's own seed buffer, unchanged.
                s.push(g.step(self.k.add2, &[&b.d_res1, &b.d_tmp, d_hidden_out], &[n * d], n * d));
            } else {
                s.push(g.step(self.k.add2, &[&b.d_res1, &b.d_tmp, &b.dx[l - 1]], &[n * d], n * d));
            }
        }

        // `type_emb.weight`'s gradient: `type_row_ids` (one qtype id PER
        // ROW, already built by `set_call` for the forward broadcast-gather)
        // is exactly the index buffer `emb_bwd_step` wants - see the module
        // doc's "Backward" section. Reads `d_hidden_out` AFTER the loop above
        // has finished accumulating every layer's contribution into it.
        s.push(block::emb_bwd_step(g, &eb, &self.type_row_ids, None, d_hidden_out, gr("type_emb.weight"), n, d, 3));
        s
    }
}
