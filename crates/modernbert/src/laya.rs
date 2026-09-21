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
//! **This milestone is forward-only.** No backward, no training step -
//! that is Laya M5, mirroring `crates/decide/src/head.rs`'s own `Bwd` split.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
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
    ff1: DeviceBuffer,
    ff2: DeviceBuffer,
}

/// Laya's decision head. Inference-only (`Role::Frozen` weights, no
/// backward) - the seeded backward is Laya M5, a separate milestone.
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
        assert!(max_span <= cap_rows, "max_span {max_span} > cap_rows {cap_rows}");
        let roles: Vec<(String, usize, Role)> = tensor_manifest(&cfg)
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let n = cap_rows as u64;
        let d = cfg.d_model as u64;
        let ff = (cfg.ff_mult * cfg.d_model) as u64;
        let per_row = cfg.n_heads as u64 * max_span as u64 * 4;
        let chunk = ((SLAB_BUDGET / per_row.max(1)).max(1) as u32).min(max_span.max(1));
        let slab = cfg.n_heads as u64 * chunk as u64 * max_span as u64;

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
                ff2: gpu.storage(n * d),
            })
            .collect();
        let m = cap_markers as u64;
        let q = cap_questions as u64;
        let act_hidden = cfg.act_hidden as u64;
        let n_act = cfg.n_act as u64;
        let k = crate::kern::Ids::resolve(&gpu);
        LayaHead {
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
            gpu,
            cfg,
            ps,
        }
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
            // PLAIN RELU, not GELU - see the module doc's trap note. In place
            // is safe: this milestone is forward-only, so nothing needs the
            // pre-activation value back.
            s.push(g.step(self.k.relu_inplace, &[&lb.ff1], &[n * ff], n * ff));
            s.push(g.step(self.k.matmul, &[&lb.ff1, self.w(&format!("{p}.ff2.weight")), &lb.ff2], &[n, ff, d], n * d));
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
}
