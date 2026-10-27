// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Anchor memory: a learned, fixed-capacity store beside a decoder layer's
//! attention. Per KV group it keeps the `capacity` best-scoring records written
//! by tokens of completed commit blocks, and every token reads them by softmax
//! attention with an extra always-present zero logit (a sink that carries a
//! zero value, so an empty bank reads zero). Because the bank holds at most
//! `capacity` records per group, the state a layer carries for an arbitrarily
//! long context has a fixed size.
//!
//! ## The math (per layer, `h` = the layer's residual input)
//!
//! ```text
//! u      = rmsnorm(h)                                   input_norm
//! q      = rmsnorm_head(q_proj u)                       q_norm, per head
//! gate   = tanh(read_scale) * 2 * sigmoid(q . read_gate_weight / sqrt(D) + read_gate_bias)
//! q      = rope(q)
//! score  = admission_up(silu(admission_down u))         [groups], fp32
//! z      = u + write_down(silu(write_gate u) * write_up u)
//! key    = rope(rmsnorm_head(k_proj z))                 k_norm, per group
//! value  = v_proj z
//! bank_b = best `capacity` records of all blocks < b, per group,
//!          ordered by (score descending, position ascending)
//! read   = softmax([q . key_c / sqrt(D) for c in bank_b] ++ [0]) . [value_c ..] ++ [0]
//! out    = o_proj(read * gate)
//! ```
//!
//! A token of block `b = position / commit_block` reads `bank_b`: the records
//! of its own block are not visible yet, which is what makes the bank depend
//! only on scores and positions and never on how a prompt is chunked.
//!
//! ## Layering
//!
//! Like [`crate::gqa_mixer`], the linear projections stay with the model: they
//! may be adapted or quantized there. [`anchor_memory_fwd`] calls back into a
//! [`Project`] closure for the nine projections and runs everything between
//! them. The admission scorer is two plain projections whose output is read
//! back as fp32 by the selection kernel; a caller must not route
//! [`MemoryProj::AdmissionDown`]/[`MemoryProj::AdmissionUp`] through a narrow
//! tier, because a one-ulp change in a score reorders the bank.
//!
//! ## Sequence form and the state a later chunked form carries
//!
//! [`anchor_memory_fwd`] processes `n_seq` sequences of `t` tokens each from
//! position 0. The bank of every commit block is computed up front into
//! `refs`, `[n_seq, n_blocks, groups, capacity]`: slot `c` of the bank block
//! `b` reads holds `position + 1` of the record that ranks `c`-th (0 for an
//! empty slot). Keys and values are read straight from the sequence's own
//! `key`/`value` tensors by position.
//!
//! A chunked prefill or decode step, which no longer holds earlier tokens'
//! tensors, will carry per sequence and layer instead:
//!
//! * the committed bank: `key`/`value` `[groups, capacity, D]`, `score`
//!   `[groups, capacity]` and `position` `[groups, capacity]` (best first,
//!   empty slots marked), plus the number of committed tokens;
//! * the pending block: the keys, values and scores of the tokens written since
//!   the last commit (fewer than `commit_block`), merged into the bank by the
//!   same selection when the block fills.
//!
//! Selection is already expressed over `(score, position)` candidates, so that
//! form reuses [`anchor_select_steps`]'s ordering rather than redefining it.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::block::{rmsnorm_fwd, rope2d_partial_fwd, KernelIds};

/// The kernels the anchor memory adds to a model's pipeline list (they are
/// resolved by name, [`AnchorMemoryKernels::resolve`]). `attn_softmax_cross` is
/// the row softmax over `capacity + 1` columns.
pub const ANCHOR_MEMORY_PIPELINES: [(&str, &str); 5] = [
    ("anchor_select", kernels::ANCHOR_SELECT),
    ("anchor_read_scores", kernels::ANCHOR_READ_SCORES),
    ("anchor_read_apply", kernels::ANCHOR_READ_APPLY),
    ("anchor_read_gate", kernels::ANCHOR_READ_GATE),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
];

/// The anchor-memory pipeline indices, as the caller registered them.
#[derive(Clone, Copy, Debug)]
pub struct AnchorMemoryKernels {
    pub select: usize,
    pub read_scores: usize,
    pub read_apply: usize,
    pub read_gate: usize,
    /// `attn_softmax_cross`, used as a plain row softmax.
    pub row_softmax: usize,
}

impl AnchorMemoryKernels {
    /// Look the kernels up on `gpu` by name. An error names the first missing
    /// one: the model did not register [`ANCHOR_MEMORY_PIPELINES`].
    pub fn resolve(gpu: &Gpu) -> Result<AnchorMemoryKernels, String> {
        let find = |i: usize| {
            let name = ANCHOR_MEMORY_PIPELINES[i].0;
            gpu.kernel_index(name).ok_or_else(|| format!("anchor memory: kernel '{name}' is not registered on this Gpu (add model::anchor_memory::ANCHOR_MEMORY_PIPELINES to its pipelines)"))
        };
        Ok(AnchorMemoryKernels { select: find(0)?, read_scores: find(1)?, read_apply: find(2)?, read_gate: find(3)?, row_softmax: find(4)? })
    }
}

/// Every kernel [`anchor_memory_fwd`] dispatches, resolved by the calling model
/// against its own pipeline list (same convention as
/// [`crate::gqa_mixer::GqaMixerIds`]).
#[derive(Clone, Copy)]
pub struct AnchorMemoryIds {
    /// `rmsnorm` (input norm and the per-head q/k norms).
    pub kernels: KernelIds,
    pub silu: usize,
    pub silu_mul: usize,
    pub add2: usize,
    /// `scale_row`: `y[i] = s[i / m] * x[i]`, here the per-head read gate.
    pub scale_row: usize,
    pub rope2d_partial: usize,
    pub memory: AnchorMemoryKernels,
}

/// The nine learned projections the memory applies; the model supplies them
/// through a [`Project`] closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryProj {
    /// `u -> [n_heads * D]` query.
    Q,
    /// `z -> [groups * D]` key.
    K,
    /// `z -> [groups * D]` value.
    V,
    /// `[n_heads * D] -> d_model` read output.
    O,
    /// `u -> writer_intermediate` (the gated half of the writer's MLP).
    WriteGate,
    /// `u -> writer_intermediate`.
    WriteUp,
    /// `writer_intermediate -> d_model`.
    WriteDown,
    /// `u -> admission_rank`.
    AdmissionDown,
    /// `admission_rank -> groups` (the admission score).
    AdmissionUp,
}

impl MemoryProj {
    /// The checkpoint's leaf name (`memory.<leaf>.weight`).
    pub fn leaf(self) -> &'static str {
        match self {
            MemoryProj::Q => "q_proj",
            MemoryProj::K => "k_proj",
            MemoryProj::V => "v_proj",
            MemoryProj::O => "o_proj",
            MemoryProj::WriteGate => "write_gate_proj",
            MemoryProj::WriteUp => "write_up_proj",
            MemoryProj::WriteDown => "write_down_proj",
            MemoryProj::AdmissionDown => "admission_down",
            MemoryProj::AdmissionUp => "admission_up",
        }
    }

    /// `(n_out, n_in)` of the projection's `[n_out, n_in]` weight.
    pub fn dims(self, s: &AnchorMemoryShape) -> (u32, u32) {
        let (qd, kvd) = (s.n_heads * s.head_dim, s.n_kv_heads * s.head_dim);
        match self {
            MemoryProj::Q => (qd, s.d_model),
            MemoryProj::K | MemoryProj::V => (kvd, s.d_model),
            MemoryProj::O => (s.d_model, qd),
            MemoryProj::WriteGate | MemoryProj::WriteUp => (s.writer_intermediate, s.d_model),
            MemoryProj::WriteDown => (s.d_model, s.writer_intermediate),
            MemoryProj::AdmissionDown => (s.admission_rank, s.d_model),
            MemoryProj::AdmissionUp => (s.n_kv_heads, s.admission_rank),
        }
    }
}

/// `out = x . W^T` of the named projection over `shape.rows()` rows, submitted
/// by the closure itself.
pub type Project<'a> = &'a dyn Fn(MemoryProj, &DeviceBuffer, &DeviceBuffer);

/// The memory's shape.
#[derive(Clone, Copy, Debug)]
pub struct AnchorMemoryShape {
    /// Independent sequences; the bank never crosses between them.
    pub n_seq: u32,
    /// Tokens per sequence.
    pub t: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// `rotary_dim / 2`, the width of the rope tables.
    pub rotary_half: u32,
    pub rms_eps: f32,
    /// Records kept per KV group.
    pub capacity: u32,
    /// Positions per commit block.
    pub commit_block: u32,
    pub admission_rank: u32,
    pub writer_intermediate: u32,
}

impl AnchorMemoryShape {
    pub fn rows(&self) -> u32 {
        self.n_seq * self.t
    }

    /// Commit blocks per sequence: `ceil(t / commit_block)`.
    pub fn n_blocks(&self) -> u32 {
        self.t.div_ceil(self.commit_block)
    }

    /// Words of the `refs` tensor.
    pub fn refs_len(&self) -> u64 {
        u64::from(self.n_seq) * u64::from(self.n_blocks()) * u64::from(self.n_kv_heads) * u64::from(self.capacity)
    }

    /// Refuse a shape the kernels cannot index, naming the field.
    pub fn validate(&self) -> Result<(), String> {
        for (field, v) in [
            ("n_seq", self.n_seq),
            ("t", self.t),
            ("d_model", self.d_model),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("head_dim", self.head_dim),
            ("capacity", self.capacity),
            ("commit_block", self.commit_block),
            ("admission_rank", self.admission_rank),
            ("writer_intermediate", self.writer_intermediate),
        ] {
            if v == 0 {
                return Err(format!("anchor memory: {field} must be > 0"));
            }
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!("anchor memory: n_heads {} is not a multiple of n_kv_heads {}", self.n_heads, self.n_kv_heads));
        }
        if 2 * self.rotary_half > self.head_dim {
            return Err(format!("anchor memory: rotary_half {} rotates more than head_dim {}", self.rotary_half, self.head_dim));
        }
        // The widest tensors are indexed in u32 on the device.
        let widest = u64::from(self.rows()) * u64::from(self.n_heads) * u64::from(self.capacity + 1);
        if widest > u64::from(u32::MAX) {
            return Err(format!("anchor memory: the read logits ({} rows x {} heads x {} slots = {widest} words) exceed u32 indexing; split the sequence", self.rows(), self.n_heads, self.capacity + 1));
        }
        Ok(())
    }
}

/// The memory's non-projection weights. The three norms hold the FINAL
/// per-channel multiplier (`1 + w` of the released checkpoint, folded at
/// import). `cos`/`sin` are the model's `[rows, rotary_half]` rope tables.
pub struct AnchorMemoryWeights<'a> {
    pub input_norm: &'a DeviceBuffer,
    pub q_norm: &'a DeviceBuffer,
    pub k_norm: &'a DeviceBuffer,
    /// `[n_heads]`
    pub read_scale: &'a DeviceBuffer,
    /// `[n_heads, head_dim]`
    pub read_gate_weight: &'a DeviceBuffer,
    /// `[n_heads]`
    pub read_gate_bias: &'a DeviceBuffer,
    pub cos: &'a DeviceBuffer,
    pub sin: &'a DeviceBuffer,
}

/// What the forward computed on the way to its output.
pub struct AnchorMemoryActs {
    /// `[rows, n_kv_heads]` admission scores, fp32.
    pub scores: DeviceBuffer,
    /// `[rows * n_heads]` read gate.
    pub gate: DeviceBuffer,
    /// `[rows, n_heads * head_dim]` normalised, rotated query.
    pub query: DeviceBuffer,
    /// `[rows, n_kv_heads * head_dim]` rotated written keys.
    pub key: DeviceBuffer,
    /// `[rows, n_kv_heads * head_dim]` written values.
    pub value: DeviceBuffer,
    /// `[rows, n_heads * head_dim]` the read, before the gate.
    pub read: DeviceBuffer,
    /// `[n_seq, n_blocks, n_kv_heads, capacity]`, `position + 1` (0 = empty)
    /// per slot; see [`decode_refs`].
    pub refs: DeviceBuffer,
}

/// Dispatch counts and the thread total of a step, refusing u32 overflow.
fn threads(what: &str, n: u64) -> u32 {
    u32::try_from(n).unwrap_or_else(|_| panic!("anchor memory: {what} needs {n} threads, more than a dispatch can index; split the sequence"))
}

/// The commit steps computing `refs` for every block of every sequence:
/// block 0 starts from an empty bank, block `b + 1` is the best `capacity` of
/// block `b`'s bank and block `b`'s tokens. `refs` need not be initialised.
pub fn anchor_select_steps(g: &Gpu, ids: &AnchorMemoryKernels, s: &AnchorMemoryShape, scores: &DeviceBuffer, refs: &DeviceBuffer) -> Vec<Step> {
    let candidates = u64::from(s.n_seq) * u64::from(s.n_kv_heads) * u64::from(s.capacity + s.commit_block);
    let n = threads("anchor_select", candidates);
    // The last block has no successor; only a single-block sequence still needs the call that zeroes block 0.
    (0..s.n_blocks().saturating_sub(1).max(1))
        .map(|block| g.step(ids.select, &[scores, refs], &[s.n_seq, s.t, s.n_kv_heads, s.capacity, s.commit_block, block], n))
        .collect()
}

/// The three read steps: logits against the bank (sink column last), row
/// softmax, probability-weighted sum of the bank's values into `read`.
/// `logits` and `probs` are `[rows * n_heads, capacity + 1]` scratch.
#[allow(clippy::too_many_arguments)]
pub fn anchor_read_steps(
    g: &Gpu,
    ids: &AnchorMemoryKernels,
    s: &AnchorMemoryShape,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    refs: &DeviceBuffer,
    logits: &DeviceBuffer,
    probs: &DeviceBuffer,
    read: &DeviceBuffer,
) -> Vec<Step> {
    let p = [s.rows(), s.t, s.n_heads, s.n_kv_heads, s.capacity, s.head_dim, s.commit_block];
    let rows_heads = u64::from(s.rows()) * u64::from(s.n_heads);
    vec![
        g.step(ids.read_scores, &[query, key, refs, logits], &p, threads("anchor_read_scores", rows_heads * u64::from(s.capacity + 1))),
        g.step(ids.row_softmax, &[logits, probs], &[s.rows() * s.n_heads, 1, 1, s.capacity + 1], threads("anchor softmax", rows_heads)),
        g.step(ids.read_apply, &[probs, value, refs, read], &p, threads("anchor_read_apply", rows_heads * u64::from(s.head_dim))),
    ]
}

/// The anchor memory's contribution to a layer's output, `[rows, d_model]`, for
/// `h` (`[rows, d_model]`, the layer's residual input before its own norm).
/// The caller adds it to the attention output before the residual add. See the
/// module doc for the math and for `project`.
pub fn anchor_memory_fwd(g: &Gpu, ids: &AnchorMemoryIds, s: &AnchorMemoryShape, w: &AnchorMemoryWeights, project: Project, h: &DeviceBuffer) -> (DeviceBuffer, AnchorMemoryActs) {
    if let Err(e) = s.validate() {
        panic!("{e}");
    }
    let n = s.rows();
    let (d, hh, gg, hd) = (s.d_model, s.n_heads, s.n_kv_heads, s.head_dim);
    let (qd, kvd) = (hh * hd, gg * hd);
    let k = &ids.kernels;
    let wide = |rows: u32, cols: u32| g.storage_uninit(u64::from(rows) * u64::from(cols));

    // u = input_norm(h): the layer's residual input, normalised by the memory's own gain.
    let u = wide(n, d);
    g.submit(&[], &[rmsnorm_fwd(g, k, h, w.input_norm, &u, d, n, s.rms_eps)]);

    // Query, its content-only read gate, then the rotation.
    let q_raw = wide(n, qd);
    project(MemoryProj::Q, &u, &q_raw);
    let query = wide(n, qd);
    g.submit(&[], &[rmsnorm_fwd(g, k, &q_raw, w.q_norm, &query, hd, n * hh, s.rms_eps)]);
    // `query` is not rotated yet: the gate reads content, not position.
    let gate = wide(n, hh);
    g.submit(&[], &[g.step(ids.memory.read_gate, &[&query, w.read_scale, w.read_gate_weight, w.read_gate_bias, &gate], &[n, hh, hd], n * hh)]);
    g.submit(&[], &[rope2d_partial_fwd(g, ids.rope2d_partial, &query, w.cos, w.sin, n, hh, s.rotary_half, qd, 0, hd)]);

    // Admission score, fp32: the low-rank scorer over u.
    let adm_hidden = wide(n, s.admission_rank);
    project(MemoryProj::AdmissionDown, &u, &adm_hidden);
    let adm_act = wide(n, s.admission_rank);
    g.submit(&[], &[g.step(ids.silu, &[&adm_hidden, &adm_act], &[n * s.admission_rank], n * s.admission_rank)]);
    let scores = wide(n, gg);
    project(MemoryProj::AdmissionUp, &adm_act, &scores);

    // The record: z = u + write_down(silu(write_gate u) * write_up u).
    let wi = s.writer_intermediate;
    let (w_gate, w_up, w_mix, w_down) = (wide(n, wi), wide(n, wi), wide(n, wi), wide(n, d));
    project(MemoryProj::WriteGate, &u, &w_gate);
    project(MemoryProj::WriteUp, &u, &w_up);
    g.submit(&[], &[g.step(ids.silu_mul, &[&w_gate, &w_up, &w_mix], &[n * wi], n * wi)]);
    project(MemoryProj::WriteDown, &w_mix, &w_down);
    let z = wide(n, d);
    g.submit(&[], &[g.step(ids.add2, &[&u, &w_down, &z], &[n * d], n * d)]);

    let k_raw = wide(n, kvd);
    project(MemoryProj::K, &z, &k_raw);
    let key = wide(n, kvd);
    g.submit(&[], &[rmsnorm_fwd(g, k, &k_raw, w.k_norm, &key, hd, n * gg, s.rms_eps)]);
    g.submit(&[], &[rope2d_partial_fwd(g, ids.rope2d_partial, &key, w.cos, w.sin, n, gg, s.rotary_half, kvd, 0, hd)]);
    let value = wide(n, kvd);
    project(MemoryProj::V, &z, &value);

    // Admission, then the read.
    let refs = g.storage_uninit(s.refs_len());
    g.submit(&[], &anchor_select_steps(g, &ids.memory, s, &scores, &refs));
    let logits = g.storage_uninit(u64::from(n) * u64::from(hh) * u64::from(s.capacity + 1));
    let probs = g.storage_uninit(u64::from(n) * u64::from(hh) * u64::from(s.capacity + 1));
    let read = wide(n, qd);
    g.submit(&[], &anchor_read_steps(g, &ids.memory, s, &query, &key, &value, &refs, &logits, &probs, &read));

    // Gate the read per head (`read` viewed as [rows * heads, head_dim]) and project it out.
    let gated = wide(n, qd);
    g.submit(&[], &[g.step(ids.scale_row, &[&read, &gate, &gated], &[n * qd, hd], n * qd)]);
    let out = wide(n, d);
    project(MemoryProj::O, &gated, &out);

    (out, AnchorMemoryActs { scores, gate, query, key, value, read, refs })
}

/// `refs` as host token indices: `position` of each slot, `-1` for an empty one.
pub fn decode_refs(words: &[f32]) -> Vec<i64> {
    words.iter().map(|w| i64::from(w.to_bits()) - 1).collect()
}
