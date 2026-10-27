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
//! [`anchor_memory_bwd`] is the reverse: the projections' parameter gradients go through a
//! [`ProjectBwd`] closure the same way, the read's gradient is gathered per record (no atomics), and
//! the admission scorer is outside it by construction (a sort has no gradient).
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

use crate::block::{rmsnorm_bwd, rmsnorm_fwd, rope2d_partial_bwd, rope2d_partial_fwd, swiglu_bwd, KernelIds};

/// The kernels the anchor memory adds to a model's pipeline list (they are
/// resolved by name, [`AnchorMemoryKernels::resolve`]). `attn_softmax_cross` is
/// the row softmax over `capacity + 1` columns.
pub const ANCHOR_MEMORY_PIPELINES: [(&str, &str); 15] = [
    ("anchor_commit", kernels::ANCHOR_COMMIT),
    ("anchor_read_scores", kernels::ANCHOR_READ_SCORES),
    ("anchor_read_apply", kernels::ANCHOR_READ_APPLY),
    ("anchor_read_gate", kernels::ANCHOR_READ_GATE),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("anchor_read_bwd_dp", kernels::ANCHOR_READ_BWD_DP),
    ("anchor_read_bwd_q", kernels::ANCHOR_READ_BWD_Q),
    ("anchor_slot_inverse", kernels::ANCHOR_SLOT_INVERSE),
    ("anchor_read_bwd_kv", kernels::ANCHOR_READ_BWD_KV),
    ("anchor_read_gate_bwd", kernels::ANCHOR_READ_GATE_BWD),
    ("anchor_read_gate_wgrad", kernels::ANCHOR_READ_GATE_WGRAD),
    ("softmax_k_dx", kernels::SOFTMAX_K_DX),
    ("anchor_bank_gather", kernels::ANCHOR_BANK_GATHER),
    ("anchor_push", kernels::ANCHOR_PUSH),
    ("anchor_rows_copy", kernels::ANCHOR_ROWS_COPY),
];

/// The anchor-memory pipeline indices, as the caller registered them.
#[derive(Clone, Copy, Debug)]
pub struct AnchorMemoryKernels {
    /// `anchor_commit`: the one definition of the bank's order.
    pub commit: usize,
    pub read_scores: usize,
    pub read_apply: usize,
    pub read_gate: usize,
    /// `attn_softmax_cross`, used as a plain row softmax.
    pub row_softmax: usize,
    pub read_bwd_dp: usize,
    pub read_bwd_q: usize,
    pub slot_inverse: usize,
    pub read_bwd_kv: usize,
    pub read_gate_bwd: usize,
    pub read_gate_wgrad: usize,
    /// `softmax_k_dx`, used as the row softmax backward.
    pub row_softmax_dx: usize,
    pub bank_gather: usize,
    pub push: usize,
    pub rows_copy: usize,
}

impl AnchorMemoryKernels {
    /// Look the kernels up on `gpu` by name. An error names the first missing
    /// one: the model did not register [`ANCHOR_MEMORY_PIPELINES`].
    pub fn resolve(gpu: &Gpu) -> Result<AnchorMemoryKernels, String> {
        let find = |i: usize| {
            let name = ANCHOR_MEMORY_PIPELINES[i].0;
            gpu.kernel_index(name).ok_or_else(|| format!("anchor memory: kernel '{name}' is not registered on this Gpu (add model::anchor_memory::ANCHOR_MEMORY_PIPELINES to its pipelines)"))
        };
        Ok(AnchorMemoryKernels {
            commit: find(0)?,
            read_scores: find(1)?,
            read_apply: find(2)?,
            read_gate: find(3)?,
            row_softmax: find(4)?,
            read_bwd_dp: find(5)?,
            read_bwd_q: find(6)?,
            slot_inverse: find(7)?,
            read_bwd_kv: find(8)?,
            read_gate_bwd: find(9)?,
            read_gate_wgrad: find(10)?,
            row_softmax_dx: find(11)?,
            bank_gather: find(12)?,
            push: find(13)?,
            rows_copy: find(14)?,
        })
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
    /// `row_dot`: `out[row] = alpha * sum_k a[row, k] * b[row, k]` (backward only).
    pub row_dot: usize,
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

/// What the forward computed on the way to its output: the tensors
/// [`anchor_memory_bwd`] reads (a recomputing caller drops them and reruns the
/// forward) and the parity taps.
pub struct AnchorMemoryActs {
    /// `[rows, d_model]` the input-normed residual, the input of the query,
    /// admission and writer projections.
    pub u: DeviceBuffer,
    /// `[rows, n_heads * head_dim]` the query projection before its per-head norm.
    pub q_raw: DeviceBuffer,
    /// `[rows, n_heads * head_dim]` the normalised query BEFORE the rotary embedding (the gate's input).
    pub q_normed: DeviceBuffer,
    /// `[rows, n_kv_heads]` admission scores, fp32.
    pub scores: DeviceBuffer,
    /// `[rows * n_heads]` read gate.
    pub gate: DeviceBuffer,
    /// `[rows, n_heads * head_dim]` normalised, rotated query.
    pub query: DeviceBuffer,
    /// `[rows, writer_intermediate]` the writer's gate and up projections and their SwiGLU product.
    pub w_gate: DeviceBuffer,
    pub w_up: DeviceBuffer,
    pub w_mix: DeviceBuffer,
    /// `[rows, d_model]` the written record `u + write_down(..)`, the input of the key and value projections.
    pub z: DeviceBuffer,
    /// `[rows, n_kv_heads * head_dim]` the key projection before its per-group norm.
    pub k_raw: DeviceBuffer,
    /// `[rows, n_kv_heads * head_dim]` rotated written keys.
    pub key: DeviceBuffer,
    /// `[rows, n_kv_heads * head_dim]` written values.
    pub value: DeviceBuffer,
    /// `[rows * n_heads, capacity + 1]` read probabilities, the sink last.
    pub probs: DeviceBuffer,
    /// `[rows, n_heads * head_dim]` the read, before the gate.
    pub read: DeviceBuffer,
    /// `[rows, n_heads * head_dim]` the gated read, the input of the output projection.
    pub gated: DeviceBuffer,
    /// `[n_seq, n_blocks, n_kv_heads, capacity]`, `position + 1` (0 = empty)
    /// per slot; see [`decode_refs`].
    pub refs: DeviceBuffer,
}

/// Dispatch counts and the thread total of a step, refusing u32 overflow.
fn threads(what: &str, n: u64) -> u32 {
    u32::try_from(n).unwrap_or_else(|_| panic!("anchor memory: {what} needs {n} threads, more than a dispatch can index; split the sequence"))
}

fn upload_u32(g: &Gpu, data: &[u32]) -> DeviceBuffer {
    let buf = g.storage(data.len().max(1) as u64);
    g.write(&buf, data);
    buf
}

/// `EMPTY` as an `in_row`: start the commit from an empty bank.
const EMPTY_ROW: u32 = u32::MAX;

/// The `anchor_commit` dispatch for `n_seq` sequences described by `meta` (5 words each: `in_row`, `out_row`,
/// `new_row0`, `new_pos0`, `new_count`), merging up to `s.commit_block` new candidates from `new_score` into the
/// bank `(in_pos, in_score)`, writing `(out_pos, out_score, out_src)`.
#[allow(clippy::too_many_arguments)]
fn commit_step(g: &Gpu, ids: &AnchorMemoryKernels, s: &AnchorMemoryShape, n_seq: u32, meta: &DeviceBuffer, bank_in: (&DeviceBuffer, &DeviceBuffer), new_score: &DeviceBuffer, bank_out: (&DeviceBuffer, &DeviceBuffer), out_src: &DeviceBuffer) -> Step {
    let candidates = u64::from(n_seq) * u64::from(s.n_kv_heads) * u64::from(s.capacity + s.commit_block);
    g.step(ids.commit, &[meta, bank_in.0, bank_in.1, new_score, bank_out.0, bank_out.1, out_src], &[n_seq, s.n_kv_heads, s.capacity, s.commit_block], threads("anchor_commit", candidates))
}

/// `refs` for every block of every sequence of `scores` (`[n_seq * t, n_kv_heads]`): block 0 reads an empty bank,
/// block `b + 1`'s bank is [`commit_step`] of block `b`'s bank and block `b`'s tokens. `refs` need not be
/// initialised. The same kernel that commits a decode step's block orders the records, so a sequence processed whole
/// and a sequence processed token by token select identically.
pub fn anchor_select_steps(g: &Gpu, ids: &AnchorMemoryKernels, s: &AnchorMemoryShape, scores: &DeviceBuffer, refs: &DeviceBuffer) -> Vec<Step> {
    let bank_words = u64::from(s.n_seq) * u64::from(s.n_kv_heads) * u64::from(s.capacity);
    let bank = || (g.storage_uninit(bank_words), g.storage_uninit(bank_words));
    let (mut cur, mut next) = (bank(), bank());
    let src = g.storage_uninit(bank_words);
    let seqs = 0..s.n_seq;
    let meta = |block: Option<u32>| {
        let rows: Vec<u32> = seqs
            .clone()
            .flat_map(|q| match block {
                // The empty bank block 0 reads.
                None => [EMPTY_ROW, q, 0, 0, 0],
                Some(b) => {
                    let first = b * s.commit_block;
                    [q, q, q * s.t + first, first, s.commit_block.min(s.t - first)]
                }
            })
            .collect();
        upload_u32(g, &rows)
    };
    let to_refs = |block: u32, pos: &DeviceBuffer| {
        let m = upload_u32(g, &[0, block]);
        g.step(ids.rows_copy, &[&m, pos, refs], &[1, bank_words as u32], threads("anchor_rows_copy", bank_words))
    };
    // Bank 0 is empty; every later one merges the previous with the next block.
    let mut steps = vec![commit_step(g, ids, s, s.n_seq, &meta(None), (&cur.0, &cur.1), scores, (&next.0, &next.1), &src), to_refs(0, &next.0)];
    std::mem::swap(&mut cur, &mut next);
    for b in 0..s.n_blocks().saturating_sub(1) {
        let m = meta(Some(b));
        steps.push(commit_step(g, ids, s, s.n_seq, &m, (&cur.0, &cur.1), scores, (&next.0, &next.1), &src));
        steps.push(to_refs(b + 1, &next.0));
        std::mem::swap(&mut cur, &mut next);
    }
    steps
}

/// Where a read finds its records: `kv_rows` rows of key/value per sequence and, for a pool of fixed-state banks,
/// the sequences' pool rows in `slots`. `None` is a sequence's own tensors (`kv_rows` = `t`, sequence `s` at row
/// block `s`).
#[derive(Clone, Copy)]
pub struct BankAddressing<'a> {
    pub kv_rows: u32,
    pub slots: Option<&'a DeviceBuffer>,
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
    anchor_read_steps_at(g, ids, s, BankAddressing { kv_rows: s.t, slots: None }, query, key, value, refs, logits, probs, read)
}

/// [`anchor_read_steps`] over records addressed as `at` says.
#[allow(clippy::too_many_arguments)]
pub fn anchor_read_steps_at(
    g: &Gpu,
    ids: &AnchorMemoryKernels,
    s: &AnchorMemoryShape,
    at: BankAddressing,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    refs: &DeviceBuffer,
    logits: &DeviceBuffer,
    probs: &DeviceBuffer,
    read: &DeviceBuffer,
) -> Vec<Step> {
    let p = [s.rows(), s.t, s.n_heads, s.n_kv_heads, s.capacity, s.head_dim, s.commit_block, at.kv_rows, u32::from(at.slots.is_some())];
    // A sequence's own tensors need no pool rows; the binding still wants a buffer of its own.
    let unused;
    let slots = match at.slots {
        Some(slots) => slots,
        None => {
            unused = g.storage(1);
            &unused
        }
    };
    let rows_heads = u64::from(s.rows()) * u64::from(s.n_heads);
    vec![
        g.step(ids.read_scores, &[query, key, refs, logits, slots], &p, threads("anchor_read_scores", rows_heads * u64::from(s.capacity + 1))),
        g.step(ids.row_softmax, &[logits, probs], &[s.rows() * s.n_heads, 1, 1, s.capacity + 1], threads("anchor softmax", rows_heads)),
        g.step(ids.read_apply, &[probs, value, refs, read, slots], &p, threads("anchor_read_apply", rows_heads * u64::from(s.head_dim))),
    ]
}

/// Everything the memory computes from the layer's residual input before it reads: shared by the sequence form, a
/// decode step and a chunk from a state.
pub struct MemoryProjections {
    pub u: DeviceBuffer,
    pub q_raw: DeviceBuffer,
    pub q_normed: DeviceBuffer,
    pub query: DeviceBuffer,
    pub gate: DeviceBuffer,
    pub scores: DeviceBuffer,
    pub w_gate: DeviceBuffer,
    pub w_up: DeviceBuffer,
    pub w_mix: DeviceBuffer,
    pub z: DeviceBuffer,
    pub k_raw: DeviceBuffer,
    pub key: DeviceBuffer,
    pub value: DeviceBuffer,
}

/// The query (normalised, gated, rotated), the admission scores and the written record (key, value) of every row of
/// `h`; see the module doc for the math.
pub fn memory_projections(g: &Gpu, ids: &AnchorMemoryIds, s: &AnchorMemoryShape, w: &AnchorMemoryWeights, project: Project, h: &DeviceBuffer) -> MemoryProjections {
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

    // Query, its content-only read gate, then the rotation. The unrotated normalised query is kept (`q_normed`)
    // for the gate's backward; `query` is the same values, rotated in place.
    let q_raw = wide(n, qd);
    project(MemoryProj::Q, &u, &q_raw);
    let q_normed = wide(n, qd);
    let query = wide(n, qd);
    g.submit(&[], &[rmsnorm_fwd(g, k, &q_raw, w.q_norm, &q_normed, hd, n * hh, s.rms_eps), rmsnorm_fwd(g, k, &q_raw, w.q_norm, &query, hd, n * hh, s.rms_eps)]);
    let gate = wide(n, hh);
    g.submit(&[], &[g.step(ids.memory.read_gate, &[&q_normed, w.read_scale, w.read_gate_weight, w.read_gate_bias, &gate], &[n, hh, hd], n * hh)]);
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

    MemoryProjections { u, q_raw, q_normed, query, gate, scores, w_gate, w_up, w_mix, z, k_raw, key, value }
}

/// The anchor memory's contribution to a layer's output, `[rows, d_model]`, for
/// `h` (`[rows, d_model]`, the layer's residual input before its own norm).
/// The caller adds it to the attention output before the residual add. See the
/// module doc for the math and for `project`.
pub fn anchor_memory_fwd(g: &Gpu, ids: &AnchorMemoryIds, s: &AnchorMemoryShape, w: &AnchorMemoryWeights, project: Project, h: &DeviceBuffer) -> (DeviceBuffer, AnchorMemoryActs) {
    let MemoryProjections { u, q_raw, q_normed, query, gate, scores, w_gate, w_up, w_mix, z, k_raw, key, value } = memory_projections(g, ids, s, w, project, h);
    let n = s.rows();
    let (d, hh, hd) = (s.d_model, s.n_heads, s.head_dim);
    let qd = hh * hd;
    let wide = |rows: u32, cols: u32| g.storage_uninit(u64::from(rows) * u64::from(cols));

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

    (out, AnchorMemoryActs { u, q_raw, q_normed, scores, gate, query, w_gate, w_up, w_mix, z, k_raw, key, value, probs, read, gated, refs })
}

/// `x`'s gradient through [`MemoryProj`] `which`: `dx (+)= d_out . W` (`accumulate` adds to `dx`, else overwrites)
/// and, where the weight trains, `dW += d_out^T . x`. The model supplies it as it does [`Project`], so an adapter
/// or a frozen base is its business.
pub type ProjectBwd<'a> = &'a dyn Fn(MemoryProj, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer, bool);

/// The read gate's three parameter gradients; trained together or frozen together.
pub struct ReadGateGrads<'a> {
    pub scale: &'a DeviceBuffer,
    pub weight: &'a DeviceBuffer,
    pub bias: &'a DeviceBuffer,
}

/// [`AnchorMemoryWeights`]'s gradient buffers, accumulated into by [`anchor_memory_bwd`]. `None` when frozen.
pub struct AnchorMemoryGrads<'a> {
    pub input_norm: Option<&'a DeviceBuffer>,
    pub q_norm: Option<&'a DeviceBuffer>,
    pub k_norm: Option<&'a DeviceBuffer>,
    pub read_gate: Option<ReadGateGrads<'a>>,
}

/// Backward of [`anchor_read_steps`]: the gradient of the read (`d_read`, `[rows, n_heads * head_dim]`, before the
/// gate) to `(d_query, d_key, d_value)` w.r.t. the rotated query and the rotated written keys and values.
/// `probs` is the forward's `[rows * n_heads, capacity + 1]` softmax output.
///
/// dP per slot, the softmax backward over the `capacity + 1` columns (the sink's dP is zero), the query gradient
/// summed over the slots, then each record's key and value gradient gathered from the rows that read it through the
/// inverse bank map: no atomics, and `O(rows * heads * capacity * head_dim)`.
#[allow(clippy::too_many_arguments)]
pub fn anchor_read_bwd(
    g: &Gpu,
    m: &AnchorMemoryKernels,
    s: &AnchorMemoryShape,
    query: &DeviceBuffer,
    key: &DeviceBuffer,
    value: &DeviceBuffer,
    refs: &DeviceBuffer,
    probs: &DeviceBuffer,
    d_read: &DeviceBuffer,
) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let n = s.rows();
    let (hh, gg, hd) = (s.n_heads, s.n_kv_heads, s.head_dim);
    let (qd, kvd) = (hh * hd, gg * hd);
    let wide = |rows: u32, cols: u32| g.storage_uninit(u64::from(rows) * u64::from(cols));
    let cols = s.capacity + 1;
    let rows_heads = u64::from(n) * u64::from(hh);
    // The read: dP, softmax backward, the query gradient, then key and value gathered per record.
    let rp = [n, s.t, hh, gg, s.capacity, hd, s.commit_block];
    let dp = wide(n * hh, cols);
    let d_logit = wide(n * hh, cols);
    let d_query = wide(n, qd);
    g.submit(
        &[],
        &[
            g.step(m.read_bwd_dp, &[d_read, value, refs, &dp], &rp, threads("anchor_read_bwd_dp", rows_heads * u64::from(cols))),
            g.step(m.row_softmax_dx, &[probs, &dp, &d_logit], &[n * hh, cols, 1], threads("anchor softmax backward", rows_heads)),
            g.step(m.read_bwd_q, &[&d_logit, key, refs, &d_query], &rp, threads("anchor_read_bwd_q", rows_heads * u64::from(hd))),
        ],
    );
    let slot_of = g.storage_uninit(u64::from(s.n_seq) * u64::from(s.n_blocks()) * u64::from(gg) * u64::from(s.t));
    let inverse_entries = u64::from(s.n_seq) * u64::from(s.n_blocks()) * u64::from(gg);
    let ip = |phase: u32| [s.n_seq, s.t, gg, s.capacity, s.commit_block, phase];
    let d_key = wide(n, kvd);
    let d_value = wide(n, kvd);
    g.submit(
        &[],
        &[
            g.step(m.slot_inverse, &[refs, &slot_of], &ip(0), threads("anchor_slot_inverse", inverse_entries * u64::from(s.t))),
            g.step(m.slot_inverse, &[refs, &slot_of], &ip(1), threads("anchor_slot_inverse", inverse_entries * u64::from(s.capacity))),
            g.step(m.read_bwd_kv, &[&slot_of, &d_logit, probs, query, d_read, &d_key, &d_value], &rp, threads("anchor_read_bwd_kv", u64::from(n) * u64::from(kvd))),
        ],
    );

    (d_query, d_key, d_value)
}

/// Reverse of [`anchor_memory_fwd`]: `d_out` (`[rows, d_model]`, the gradient of the memory's output) to `d_h`, the
/// gradient of the layer's residual input, accumulating every norm, read-gate and (through `project_bwd`)
/// projection gradient the caller trains.
///
/// **The admission scorer gets no gradient.** Which records a bank holds is a discrete decision (a sort by score),
/// so the loss is piecewise constant in the scores: `admission_down`/`admission_up` are outside the backward by
/// construction, and the gradient reaches the input only through the read, the gate and the written key and value.
/// A scorer is trained by a different objective, never by a fabricated surrogate here.
///
/// The read's backward is a softmax backward over the bank slots and a gather: each record collects the gradient of
/// every row that read it through the inverse bank map (`anchor_slot_inverse`), so no atomics are needed and the
/// work is the forward read's, `O(rows * heads * capacity * head_dim)`, never quadratic in the sequence.
#[allow(clippy::too_many_arguments)]
pub fn anchor_memory_bwd(g: &Gpu, ids: &AnchorMemoryIds, s: &AnchorMemoryShape, w: &AnchorMemoryWeights, gw: &AnchorMemoryGrads, project_bwd: ProjectBwd, acts: &AnchorMemoryActs, h: &DeviceBuffer, d_out: &DeviceBuffer) -> DeviceBuffer {
    let n = s.rows();
    let (d, hh, gg, hd) = (s.d_model, s.n_heads, s.n_kv_heads, s.head_dim);
    let (qd, kvd) = (hh * hd, gg * hd);
    let (k, m) = (&ids.kernels, &ids.memory);
    let wide = |rows: u32, cols: u32| g.storage_uninit(u64::from(rows) * u64::from(cols));

    // Output projection, then the gate: gated = read * gate per head.
    let d_gated = wide(n, qd);
    project_bwd(MemoryProj::O, d_out, &acts.gated, &d_gated, false);
    let d_read = wide(n, qd);
    let d_gate = wide(n, hh);
    g.submit(
        &[],
        &[
            g.step(ids.scale_row, &[&d_gated, &acts.gate, &d_read], &[n * qd, hd], n * qd),
            g.step(ids.row_dot, &[&d_gated, &acts.read, &d_gate], &[n * hh, hd, 0, 0, gpu_core::f(1.0)], n * hh),
        ],
    );

    let (d_query, d_key, d_value) = anchor_read_bwd(g, m, s, &acts.query, &acts.key, &acts.value, &acts.refs, &acts.probs, &d_read);

    // Query path: un-rotate, the gate's backward, the per-head norm, the projection.
    g.submit(&[], &[rope2d_partial_bwd(g, ids.rope2d_partial, &d_query, w.cos, w.sin, n, hh, s.rotary_half, qd, 0, hd)]);
    let gate_p = [n, hh, hd];
    let (dz, dsc) = (wide(n, hh), wide(n, hh));
    g.submit(&[], &[g.step(m.read_gate_bwd, &[&acts.q_normed, &d_gate, w.read_scale, w.read_gate_weight, w.read_gate_bias, &dz, &dsc, &d_query], &gate_p, n * hh)]);
    if let Some(rg) = &gw.read_gate {
        g.submit(&[], &[g.step(m.read_gate_wgrad, &[&acts.q_normed, &dz, &dsc, rg.weight, rg.bias, rg.scale], &gate_p, hh * (hd + 2))]);
    }
    let d_q_raw = wide(n, qd);
    let inv_q = wide(n, hh);
    g.submit(&[], &rmsnorm_bwd(g, k, &acts.q_raw, w.q_norm, &d_query, &d_q_raw, &inv_q, gw.q_norm, hd, n * hh, s.rms_eps));
    let d_u = wide(n, d);
    project_bwd(MemoryProj::Q, &d_q_raw, &acts.u, &d_u, false);

    // Key and value path: both read the written record z.
    g.submit(&[], &[rope2d_partial_bwd(g, ids.rope2d_partial, &d_key, w.cos, w.sin, n, gg, s.rotary_half, kvd, 0, hd)]);
    let d_k_raw = wide(n, kvd);
    let inv_k = wide(n, gg);
    g.submit(&[], &rmsnorm_bwd(g, k, &acts.k_raw, w.k_norm, &d_key, &d_k_raw, &inv_k, gw.k_norm, hd, n * gg, s.rms_eps));
    let d_z = wide(n, d);
    project_bwd(MemoryProj::K, &d_k_raw, &acts.z, &d_z, false);
    project_bwd(MemoryProj::V, &d_value, &acts.z, &d_z, true);

    // The writer: z = u + write_down(silu(write_gate u) * write_up u), so d_z reaches u directly and through the MLP.
    let wi = s.writer_intermediate;
    let d_mix = wide(n, wi);
    project_bwd(MemoryProj::WriteDown, &d_z, &acts.w_mix, &d_mix, false);
    let (d_w_gate, d_w_up) = (wide(n, wi), wide(n, wi));
    g.submit(&[], &swiglu_bwd(g, k, &acts.w_gate, &acts.w_up, &d_mix, &d_w_gate, &d_w_up, n * wi));
    project_bwd(MemoryProj::WriteGate, &d_w_gate, &acts.u, &d_u, true);
    project_bwd(MemoryProj::WriteUp, &d_w_up, &acts.u, &d_u, true);
    let d_u_total = wide(n, d);
    g.submit(&[], &[g.step(ids.add2, &[&d_u, &d_z, &d_u_total], &[n * d], n * d)]);

    // The input norm: u = rmsnorm(h).
    let d_h = wide(n, d);
    let inv_h = wide(n, 1);
    g.submit(&[], &rmsnorm_bwd(g, k, h, w.input_norm, &d_u_total, &d_h, &inv_h, gw.input_norm, d, n, s.rms_eps));
    d_h
}

/// `refs` as host token indices: `position` of each slot, `-1` for an empty one.
pub fn decode_refs(words: &[f32]) -> Vec<i64> {
    words.iter().map(|w| i64::from(w.to_bits()) - 1).collect()
}

// ---- fixed-size per-sequence state ------------------------------------------------------------------------------

/// One memory layer's state for `slots` sequences, one pool row per sequence slot, all of it a fixed size however
/// many tokens the sequence has seen:
///
/// * the committed bank: `pos`/`score` `[slots, groups, K]` (`position + 1` and fp32 score of each kept record,
///   best first, 0 = empty slot) and `key`/`value` `[slots, K, groups * head_dim]` (rotated keys, values; the
///   record of group `g` in rank `c` is row `c`, columns `g * head_dim ..`);
/// * the pending block: `pend_key`/`pend_value` `[slots, C, groups * head_dim]` and `pend_score`
///   `[slots, C, groups]`, the records written since the last commit (position `p` waits in row `p % C`).
///
/// How many tokens a sequence has seen is the caller's fact (its decode position): the pending count is
/// `position % C` and the committed count `position - position % C`. A block commits when its last token arrives,
/// with the same [`anchor_commit`](Self) ordering the sequence form uses. [`Self::reset_row`] and
/// [`Self::copy_row`] are what slot reuse and a prefix-cache snapshot are made of.
pub struct AnchorBankPool {
    pub slots: u32,
    pub groups: u32,
    pub capacity: u32,
    pub commit_block: u32,
    pub head_dim: u32,
    pub pos: DeviceBuffer,
    pub score: DeviceBuffer,
    pub key: DeviceBuffer,
    pub value: DeviceBuffer,
    pub pend_key: DeviceBuffer,
    pub pend_value: DeviceBuffer,
    pub pend_score: DeviceBuffer,
}

impl AnchorBankPool {
    /// An empty pool; `s` supplies the geometry (its sequence count and length are not used).
    pub fn new(g: &Gpu, s: &AnchorMemoryShape, slots: u32) -> AnchorBankPool {
        let zeros = |words: u64| g.storage_init("anchor_bank", &vec![0f32; words as usize]);
        let (gg, k, c, gd) = (u64::from(s.n_kv_heads), u64::from(s.capacity), u64::from(s.commit_block), u64::from(s.n_kv_heads * s.head_dim));
        let n = u64::from(slots);
        AnchorBankPool {
            slots,
            groups: s.n_kv_heads,
            capacity: s.capacity,
            commit_block: s.commit_block,
            head_dim: s.head_dim,
            pos: zeros(n * gg * k),
            score: zeros(n * gg * k),
            key: zeros(n * k * gd),
            value: zeros(n * k * gd),
            pend_key: zeros(n * c * gd),
            pend_value: zeros(n * c * gd),
            pend_score: zeros(n * c * gg),
        }
    }

    /// An empty pool of the same geometry and slot count.
    pub fn like(g: &Gpu, other: &AnchorBankPool) -> AnchorBankPool {
        let s = AnchorMemoryShape { n_seq: 1, t: 1, d_model: 1, n_heads: other.groups, n_kv_heads: other.groups, head_dim: other.head_dim, rotary_half: 0, rms_eps: 0.0, capacity: other.capacity, commit_block: other.commit_block, admission_rank: 1, writer_intermediate: 1 };
        AnchorBankPool::new(g, &s, other.slots)
    }

    /// Every buffer of the pool with its per-slot width in words.
    pub fn arrays(&self) -> [(&DeviceBuffer, u64); 7] {
        let (gg, k, c, gd) = (u64::from(self.groups), u64::from(self.capacity), u64::from(self.commit_block), u64::from(self.groups * self.head_dim));
        [(&self.pos, gg * k), (&self.score, gg * k), (&self.key, k * gd), (&self.value, k * gd), (&self.pend_key, c * gd), (&self.pend_value, c * gd), (&self.pend_score, c * gg)]
    }

    /// Words one slot holds: the whole per-sequence state of this layer.
    pub fn words_per_slot(&self) -> u64 {
        self.arrays().iter().map(|(_, w)| w).sum()
    }

    /// Clear slot `slot`: an empty bank and an empty pending block, so a reused slot carries nothing of the
    /// sequence that held it. Ordered after every earlier submission.
    pub fn reset_row(&self, g: &Gpu, slot: u32) {
        assert!(slot < self.slots, "anchor bank: slot {slot} of {}", self.slots);
        for (buf, words) in self.arrays() {
            g.write_at(buf, u64::from(slot) * words, &vec![0u32; words as usize]);
        }
    }

    /// Steps copying slot `from` over slot `to`, bit for bit.
    pub fn copy_row(&self, g: &Gpu, ids: &AnchorMemoryKernels, from: u32, to: u32) -> Vec<Step> {
        assert!(from < self.slots && to < self.slots, "anchor bank: copy {from} -> {to} of {} slots", self.slots);
        let meta = upload_u32(g, &[0, 0]);
        self.arrays()
            .iter()
            .map(|(buf, words)| g.step_sliced(ids.rows_copy, &[&meta, buf, buf], &[(0, 2), (u64::from(from) * words, *words), (u64::from(to) * words, *words)], &[1, *words as u32], *words as u32))
            .collect()
    }
}

/// What a read of the committed bank needs about its sequences, uploaded once per step and shared by every memory
/// layer: the pool row of each sequence and the bank slots it may read. A token at position `p` reads the records
/// of the blocks before its own: `p - p % C` tokens' worth, at most `K` of them, which are the first slots.
pub struct ReadPlan {
    pub slots: DeviceBuffer,
    pub refs: DeviceBuffer,
}

impl ReadPlan {
    /// `seqs[i] = (pool row, position of the sequence's first row in this read)`; all rows of one sequence in one
    /// read lie in one commit block.
    pub fn upload(g: &Gpu, s: &AnchorMemoryShape, seqs: &[(u32, u32)]) -> ReadPlan {
        let (gg, k, c) = (s.n_kv_heads as usize, s.capacity as usize, s.commit_block);
        let mut refs = Vec::with_capacity(seqs.len() * gg * k);
        for &(_, pos) in seqs {
            let filled = ((pos / c) * c) as usize;
            let filled = filled.min(k);
            for _ in 0..gg {
                refs.extend((0..k).map(|slot| if slot < filled { slot as u32 + 1 } else { 0 }));
            }
        }
        ReadPlan { slots: upload_u32(g, &seqs.iter().map(|&(slot, _)| slot).collect::<Vec<_>>()), refs: upload_u32(g, &refs) }
    }
}

/// One record pushed into a pending block: the pool row, the token's position, and its row in the step's tensors.
#[derive(Clone, Copy, Debug)]
pub struct Push {
    pub slot: u32,
    pub pos: u32,
    pub src_row: u32,
}

/// One bank commit: the pool row, the position of the block's first token and how many candidates it carries
/// (`commit_block`, or 0 to leave the bank as it is: a decode step dispatches a commit for every sequence so the
/// step is the same list of dispatches whichever sequence's block fills).
#[derive(Clone, Copy, Debug)]
pub struct Commit {
    pub slot: u32,
    pub first_pos: u32,
    pub count: u32,
}

/// The writes of a step, uploaded once and shared by every memory layer.
pub struct WritePlan {
    n_push: u32,
    push: DeviceBuffer,
    n_commit: u32,
    commit: DeviceBuffer,
    store: DeviceBuffer,
}

impl WritePlan {
    pub fn upload(g: &Gpu, s: &AnchorMemoryShape, pushes: &[Push], commits: &[Commit]) -> WritePlan {
        let c = s.commit_block;
        let push: Vec<u32> = pushes.iter().flat_map(|p| [p.slot * c + p.pos % c, p.src_row]).collect();
        let commit: Vec<u32> = commits.iter().enumerate().flat_map(|(i, m)| [m.slot, i as u32, m.slot * c, m.first_pos, m.count]).collect();
        let store: Vec<u32> = commits.iter().enumerate().flat_map(|(i, m)| [i as u32, m.slot]).collect();
        WritePlan { n_push: pushes.len() as u32, push: upload_u32(g, &push), n_commit: commits.len() as u32, commit: upload_u32(g, &commit), store: upload_u32(g, &store) }
    }
}

/// Push the step's records into the pending blocks, then run the planned commits: merge pending into the bank
/// (`anchor_commit` orders, `anchor_bank_gather` moves the keys and values) into a scratch bank and store it back to
/// the pool rows. The scratch hop exists because a commit reads the whole old bank of its row.
#[allow(clippy::too_many_arguments)]
pub fn state_write_steps(g: &Gpu, ids: &AnchorMemoryKernels, s: &AnchorMemoryShape, pool: &AnchorBankPool, plan: &WritePlan, key: &DeviceBuffer, value: &DeviceBuffer, scores: &DeviceBuffer) -> Vec<Step> {
    let (gg, k, hd) = (s.n_kv_heads, s.capacity, s.head_dim);
    let gd = gg * hd;
    let mut steps = Vec::new();
    if plan.n_push > 0 {
        steps.push(g.step(ids.push, &[&plan.push, key, value, scores, &pool.pend_key, &pool.pend_value, &pool.pend_score], &[plan.n_push, gg, hd], threads("anchor_push", u64::from(plan.n_push) * u64::from(gd))));
    }
    if plan.n_commit > 0 {
        let nc = plan.n_commit;
        let bank = u64::from(nc) * u64::from(gg) * u64::from(k);
        let (spos, sscore, ssrc) = (g.storage_uninit(bank), g.storage_uninit(bank), g.storage_uninit(bank));
        let (skey, svalue) = (g.storage_uninit(bank * u64::from(hd)), g.storage_uninit(bank * u64::from(hd)));
        steps.push(commit_step(g, ids, s, nc, &plan.commit, (&pool.pos, &pool.score), &pool.pend_score, (&spos, &sscore), &ssrc));
        steps.push(g.step(ids.bank_gather, &[&plan.commit, &ssrc, &pool.key, &pool.value, &pool.pend_key, &pool.pend_value, &skey, &svalue], &[nc, gg, k, hd], threads("anchor_bank_gather", bank * u64::from(hd))));
        for (src, dst, stride) in [(&spos, &pool.pos, gg * k), (&sscore, &pool.score, gg * k), (&skey, &pool.key, k * gd), (&svalue, &pool.value, k * gd)] {
            steps.push(g.step(ids.rows_copy, &[&plan.store, src, dst], &[nc, stride], threads("anchor_rows_copy", u64::from(nc) * u64::from(stride))));
        }
    }
    steps
}

/// The shape a read from the state sees: `rows_per_seq` query rows of each sequence, all in one block.
fn read_shape(s: &AnchorMemoryShape, rows_per_seq: u32) -> AnchorMemoryShape {
    AnchorMemoryShape { t: rows_per_seq, commit_block: rows_per_seq, ..*s }
}

/// The gate, the output projection and the read of the committed bank shared by the two state forms.
#[allow(clippy::too_many_arguments)]
fn finish_read(g: &Gpu, ids: &AnchorMemoryIds, s: &AnchorMemoryShape, project: Project, gate: &DeviceBuffer, read: &DeviceBuffer) -> DeviceBuffer {
    let n = s.rows();
    let qd = s.n_heads * s.head_dim;
    let gated = g.storage_uninit(u64::from(n) * u64::from(qd));
    g.submit(&[], &[g.step(ids.scale_row, &[read, gate, &gated], &[n * qd, s.head_dim], n * qd)]);
    let out = g.storage_uninit(u64::from(n) * u64::from(s.d_model));
    project(MemoryProj::O, &gated, &out);
    out
}

/// One decode token of each of `s.n_seq` sequences (`s.t == 1`): the memory's output for `h` (`[n_seq, d_model]`),
/// reading each sequence's committed bank from `pool`, then pushing its record and committing the block it completes.
/// `read` and `write` are the step's plans, shared by every memory layer.
#[allow(clippy::too_many_arguments)]
pub fn anchor_memory_decode_fwd(g: &Gpu, ids: &AnchorMemoryIds, s: &AnchorMemoryShape, w: &AnchorMemoryWeights, project: Project, h: &DeviceBuffer, pool: &AnchorBankPool, read: &ReadPlan, write: &WritePlan) -> DeviceBuffer {
    assert_eq!(s.t, 1, "anchor memory decode: one token per sequence");
    let p = memory_projections(g, ids, s, w, project, h);
    let n = s.rows();
    let rs = read_shape(s, 1);
    let words = u64::from(n) * u64::from(s.n_heads) * u64::from(s.capacity + 1);
    let (logits, probs) = (g.storage_uninit(words), g.storage_uninit(words));
    let read_out = g.storage_uninit(u64::from(n) * u64::from(s.n_heads * s.head_dim));
    g.submit(&[], &anchor_read_steps_at(g, &ids.memory, &rs, BankAddressing { kv_rows: pool.capacity, slots: Some(&read.slots) }, &p.query, &pool.key, &pool.value, &read.refs, &logits, &probs, &read_out));
    let out = finish_read(g, ids, s, project, &p.gate, &read_out);
    g.submit(&[], &state_write_steps(g, &ids.memory, s, pool, write, &p.key, &p.value, &p.scores));
    out
}

/// `s.t` consecutive tokens of ONE sequence (`s.n_seq == 1`) at positions `start ..` entering slot `slot` of `pool`,
/// whatever state it holds: the rows of the first partial block read the pool's bank, each block the chunk completes
/// commits (so the rows of the next block read a bank that holds it), and the tail stays pending. The result is
/// what the same tokens give one decode step at a time, and what the sequence form gives them from position 0.
#[allow(clippy::too_many_arguments)]
pub fn anchor_memory_chunk_fwd(g: &Gpu, ids: &AnchorMemoryIds, s: &AnchorMemoryShape, w: &AnchorMemoryWeights, project: Project, h: &DeviceBuffer, pool: &AnchorBankPool, slot: u32, start: u32) -> DeviceBuffer {
    assert_eq!(s.n_seq, 1, "anchor memory chunk: one sequence");
    let p = memory_projections(g, ids, s, w, project, h);
    let (n, c, qd) = (s.t, s.commit_block, s.n_heads * s.head_dim);
    let read_out = g.storage_uninit(u64::from(n) * u64::from(qd));
    let mut pos = start;
    while pos < start + n {
        let end = (start + n).min((pos / c + 1) * c);
        let rows = end - pos;
        let (a, rs) = (pos - start, read_shape(s, rows));
        // This segment's query rows, compacted; its reads land back in their rows.
        let rows_of = |first: u32| upload_u32(g, &(0..rows).flat_map(|i| [first + i, i]).collect::<Vec<_>>());
        let to_rows = |first: u32| upload_u32(g, &(0..rows).flat_map(|i| [i, first + i]).collect::<Vec<_>>());
        let (seg_q, seg_out) = (g.storage_uninit(u64::from(rows) * u64::from(qd)), g.storage_uninit(u64::from(rows) * u64::from(qd)));
        let plan = ReadPlan::upload(g, s, &[(slot, pos)]);
        let words = u64::from(rows) * u64::from(s.n_heads) * u64::from(s.capacity + 1);
        let (logits, probs) = (g.storage_uninit(words), g.storage_uninit(words));
        g.submit(&[], &[g.step(ids.memory.rows_copy, &[&rows_of(a), &p.query, &seg_q], &[rows, qd], rows * qd)]);
        g.submit(&[], &anchor_read_steps_at(g, &ids.memory, &rs, BankAddressing { kv_rows: pool.capacity, slots: Some(&plan.slots) }, &seg_q, &pool.key, &pool.value, &plan.refs, &logits, &probs, &seg_out));
        g.submit(&[], &[g.step(ids.memory.rows_copy, &[&to_rows(a), &seg_out, &read_out], &[rows, qd], rows * qd)]);
        // Absorb the segment's records; a block this completes commits.
        let pushes: Vec<Push> = (0..rows).map(|i| Push { slot, pos: pos + i, src_row: a + i }).collect();
        let commits: Vec<Commit> = if end.is_multiple_of(c) { vec![Commit { slot, first_pos: end - c, count: c }] } else { Vec::new() };
        g.submit(&[], &state_write_steps(g, &ids.memory, s, pool, &WritePlan::upload(g, s, &pushes, &commits), &p.key, &p.value, &p.scores));
        pos = end;
    }
    finish_read(g, ids, s, project, &p.gate, &read_out)
}
