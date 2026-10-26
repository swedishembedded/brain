// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-lean fine-tuning of large language
// models on a single GPU for its clients. If your team needs expertise in
// training with 250k-token vocabularies without materialising the logits then
// you can procure our services by sending an email to info@swedishembedded.com.

//! The chunked LM-head cross-entropy every token-classifier model shares.
//!
//! A language-model head is `logits[n, V] = scale * hidden[n, d] @ W[V, d]^T`
//! followed by a softmax cross-entropy. Materialising `logits` costs `n * V`
//! floats - 993 KB per token at `V = 248320` - which overruns the storage
//! binding at a few thousand tokens and the card well before the sequence
//! lengths a frozen-base trainer is after. This module keeps only `rows_per_chunk`
//! rows of logits alive:
//!
//! * the forward walks the tokens in chunks of `rows_per_chunk`, computes that
//!   chunk's logits, the per-token loss (fp32 log-sum-exp) and moves on;
//! * the backward recomputes each chunk's logits (the same GEMM), runs the
//!   two-pass CE gradient (per-row softmax statistics once, then one
//!   O(rows * V) elementwise pass) and applies the head: `d_hidden = d_logits @
//!   W`, plus `dW += d_logits^T @ hidden` where the head trains;
//! * a head whose weight exceeds one storage binding is cut into row windows
//!   ("vocab tiles"); each window's logit columns come from one GEMM into a
//!   dense scratch and are copied to their place in the chunk's logits.
//!
//! The head is any float-read [`Weight`] tier - F32, BF16 or NF4 - so a frozen
//! head need not be the 5 GB fp32 table. Peak extra memory is the scratch of
//! [`HeadCe::scratch_words`]: `O(rows_per_chunk * V)`, independent of the
//! token count.
//!
//! The same module serves the immediate form ([`HeadCe::forward`] /
//! [`HeadCe::backward`], for RL-style callers that want the loss and token
//! log-probabilities now) and the recorded form ([`HeadCe::forward_steps`] /
//! [`HeadCe::backward_steps`], for models that build a tape once and replay it).

use gpu_core::{DeviceBuffer, Dispatch, Gpu, Step};

use crate::ops::{tile_rows, Act, GradDtype, Ops, Weight};

pub use crate::train::IGNORE;

/// The canonical `(name, source)` pairs [`HeadKernels::from_gpu`] resolves beyond
/// the `Ops` facade set (`matmul_dw` is already in `model::ops::kernel_list`, `copy_cols` in
/// `model::ops::frozen_base_kernels`, which a head over tiled logits registers too); a
/// model that does not already register these kernels appends this slice to its
/// kernel list.
#[must_use]
pub fn kernels() -> &'static [(&'static str, &'static str)] {
    &[
        ("ce_value_masked", kernels::CE_VALUE_MASKED),
        ("ce_value_stats_rows", kernels::CE_VALUE_STATS_ROWS),
        ("ce_stats", kernels::CE_STATS),
        ("ce_grad_stats", kernels::CE_GRAD_STATS),
        ("grad_scale", kernels::GRAD_SCALE),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ("scale_row", kernels::SCALE_ROW),
        ("matmul_dw_reg", kernels::MATMUL_DW_REG),
    ]
}

/// Kernel slots of the elementwise and reduction passes, resolved on one `Gpu`.
/// A model whose pipeline registers these under other names builds the struct
/// itself; [`Self::from_gpu`] resolves the canonical names of [`kernels`].
#[derive(Clone, Copy, Debug)]
pub struct HeadKernels {
    /// Masked per-row cross-entropy value: `[rows, V, IGNORE]` over logits, labels, out.
    pub ce_value: usize,
    /// Cooperative variant that also leaves the rows' softmax statistics; used
    /// only on a device that runs workgroup barriers and 256-thread groups.
    pub ce_value_stats_rows: Option<usize>,
    pub ce_stats: usize,
    pub ce_grad_stats: usize,
    /// In-place `x *= coef` with a coefficient in the params.
    pub grad_scale: usize,
    /// In-place `x *= coef[0]` with the coefficient in a device buffer.
    pub grad_scale_buf: usize,
    /// `y[i] = s[i / m] * x[i]`: per-token weights on the logit gradient.
    pub scale_row: usize,
    pub copy_cols: usize,
    pub matmul_dw: usize,
    pub matmul_dw_reg: Option<usize>,
}

impl HeadKernels {
    /// Resolve the canonical [`kernels`] names on `gpu`.
    pub fn from_gpu(gpu: &Gpu) -> Result<HeadKernels, String> {
        let find = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("HeadKernels: kernel '{name}' is not registered on this Gpu (append model::head_ce::kernels() and model::ops::training_kernel_list())"));
        Ok(HeadKernels {
            ce_value: find("ce_value_masked")?,
            ce_value_stats_rows: gpu.kernel_index("ce_value_stats_rows"),
            ce_stats: find("ce_stats")?,
            ce_grad_stats: find("ce_grad_stats")?,
            grad_scale: find("grad_scale")?,
            grad_scale_buf: find("grad_scale_buf")?,
            scale_row: find("scale_row")?,
            copy_cols: find("copy_cols")?,
            matmul_dw: find("matmul_dw")?,
            matmul_dw_reg: gpu.kernel_index("matmul_dw_reg"),
        })
    }
}

/// Sizing of a [`HeadCe`].
#[derive(Clone, Copy, Debug)]
pub struct HeadCeConfig {
    /// Tokens whose logits are alive at once. A multiple of 64 unless the call
    /// covers at most that many tokens: every chunk start is a storage-binding
    /// offset into the per-token buffers, which must be 256-byte aligned.
    pub rows_per_chunk: u32,
    /// Words one weight binding may hold; `None` asks the device
    /// (`block::tile_budget_words_for`). A test forces windowing with a small value.
    pub tile_budget_words: Option<u64>,
}

/// The loss of one [`HeadCe::forward`].
#[derive(Clone, Debug)]
pub struct HeadLoss {
    /// `sum_i w_i * nll_i` over the tokens whose label is not [`IGNORE`]
    /// (`w_i = 1` without weights).
    pub loss_sum: f64,
    /// Tokens whose label is not [`IGNORE`].
    pub count: u32,
    /// `log p(label_i)` per token under the scaled logits; `0` at ignored tokens.
    pub logprobs: Vec<f32>,
}

/// How the backward scales the logit gradient: by a host constant, or by a
/// coefficient that lives in a one-word device buffer (a recorded tape reads a
/// per-batch `1 / count` that way, so the tape itself never changes).
#[derive(Clone, Copy)]
pub enum GradScale<'a> {
    Const(f32),
    Buf(&'a DeviceBuffer),
}

/// Per-token weights on the logit gradient: `weights` is `[n]`, and `scratch`
/// a `[rows_per_chunk * V]` buffer the scaled gradient is written to (see
/// [`HeadCe::row_weight_scratch`]); `scale_row` is not in place.
#[derive(Clone, Copy)]
pub struct RowWeights<'a> {
    pub weights: &'a DeviceBuffer,
    pub scratch: &'a DeviceBuffer,
}

/// Everything one recorded backward reads and writes.
pub struct HeadBackward<'a> {
    /// `[n, d]` hidden states the forward consumed.
    pub hidden: &'a DeviceBuffer,
    /// `[n, d]` receives `d_hidden` (overwritten).
    pub d_hidden: &'a DeviceBuffer,
    pub n: u32,
    /// `[n]` u32 labels, [`IGNORE`] to skip a token.
    pub labels: &'a DeviceBuffer,
    pub grad_scale: GradScale<'a>,
    pub logit_scale: f32,
    pub row_weights: Option<RowWeights<'a>>,
    /// The head's `[V, d]` weight gradient, accumulated into; `None` for a frozen head.
    pub dw: Option<HeadGrad<'a>>,
    /// True when `n` fits one chunk and the forward ran in the same recording
    /// context so its logits (and statistics, on a cooperative device) are still
    /// in the scratch: the backward then skips recomputing them.
    pub forward_state_held: bool,
}

/// A trained head's weight gradient: `[V, d]` elements stored as `dtype` (a bf16 gradient is two elements per word).
/// It never depends on the head's own tier: `dW = d_logits^T hidden` does not read the weight.
#[derive(Clone, Copy)]
pub struct HeadGrad<'a> {
    pub buf: &'a DeviceBuffer,
    pub dtype: GradDtype,
}

/// The chunked LM head. See the module documentation.
pub struct HeadCe {
    d: u32,
    vocab: u32,
    rows: u32,
    head: Weight,
    tiles: Vec<(u32, u32)>,
    passes: Vec<(u32, u32)>,
    logits: DeviceBuffer,
    d_logits: DeviceBuffer,
    tile: DeviceBuffer,
    tile_words: u64,
    stats: DeviceBuffer,
    k: HeadKernels,
    coop: bool,
    /// A device scalar the logits are multiplied by after the head GEMM, read at
    /// run time so a recorded tape follows a scale set later (see
    /// [`Self::set_scale_buffer`]).
    scale_buf: Option<DeviceBuffer>,
    /// The per-row argmax readout's `(argmax_part, argmax_final)` kernels, looked
    /// up by their canonical names on this head's device; `None` on a model that
    /// does not register them, where the readout is simply unavailable.
    argmax: Option<(usize, usize)>,
    /// The scratch of a bf16 weight gradient ([`Self::bf16_dw_scratch`]), allocated on first use.
    bf16_dw: std::cell::RefCell<Option<(DeviceBuffer, DeviceBuffer)>>,
}

/// Partial `(value, index)` pairs per row the argmax readout reduces a logits row
/// through - the same split `argmax_part`/`argmax_final` are dispatched with
/// everywhere else in this tree.
pub const ARGMAX_CHUNKS: u32 = 256;

/// Where a per-row argmax readout goes: `[n]` token ids as f32 (the reduction's
/// own output format), and the `[rows_per_chunk * ARGMAX_CHUNKS * 2]` scratch
/// from [`HeadCe::argmax_scratch`].
#[derive(Clone, Copy)]
pub struct ArgmaxOut<'a> {
    pub ids: &'a DeviceBuffer,
    pub scratch: &'a DeviceBuffer,
}

/// Elements of a `[V, d]` weight per 4-byte word of its densest binding, and the
/// element count a window start must be a multiple of for every binding the tier
/// slices to stay on a 256-byte offset (64 words; NF4's scale plane has one
/// word per 32 elements).
fn tier_layout(head: &Weight) -> (u64, u64) {
    match head {
        Weight::F32 { .. } => (1, 64),
        Weight::BF16 { .. } => (2, 128),
        Weight::NF4 { .. } => (8, 2048),
        other => panic!("HeadCe: a {:?} head is not a float-read tier (F32, BF16, NF4)", other.dtype()),
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Vocab windows `(first_row, rows)` covering `vocab` rows of `d` elements: one
/// window when it fits `budget_words` (of the tier's densest binding), else runs
/// of the most rows that fit, in multiples of the alignment granule so every
/// window start is a legal binding offset.
fn plan_tiles(vocab: u32, d: u32, per_word: u64, align_elems: u64, budget_words: u64) -> Vec<(u32, u32)> {
    let row_words = (d as u64 / per_word).max(1);
    let fit = (budget_words / row_words).max(1);
    if fit >= vocab as u64 {
        return vec![(0, vocab)];
    }
    let granule = align_elems / gcd(align_elems, d as u64);
    // Whole groups of 64 rows too, when the budget allows one: a window that is
    // not a multiple of 8 rows leaves a tensor-core GEMM with misaligned rows.
    let step = granule / gcd(granule, 64) * 64;
    let rows = if fit >= step { fit / step * step } else { (fit / granule * granule).max(granule) };
    (0..vocab as u64).step_by(rows as usize).map(|v0| (v0 as u32, rows.min(vocab as u64 - v0) as u32)).collect()
}

/// Rows of logits to keep alive for `n` tokens over `vocab` columns within a
/// `budget_words` binding: all `n` when they fit, else the most that do, in whole
/// groups of 64 so every chunk start is a legal binding offset into per-token buffers.
#[must_use]
pub fn rows_for_budget(n: u64, vocab: u64, budget_words: u64) -> u64 {
    if n * vocab <= budget_words {
        return n;
    }
    (budget_words / vocab / 64 * 64).max(64).min(n)
}

impl HeadCe {
    /// `head` is the `[vocab, d_model]` projection (tied embedding table or
    /// separate `lm_head`) at any float-read tier; it is shared, not copied.
    pub fn new(ops: &Ops, d_model: u32, vocab: u32, head: &Weight, cfg: HeadCeConfig, k: HeadKernels) -> Result<HeadCe, String> {
        let gpu = ops.gpu();
        if (head.n(), head.k()) != (vocab, d_model) {
            return Err(format!("HeadCe: the head is [{}, {}], expected [{vocab}, {d_model}]", head.n(), head.k()));
        }
        if matches!(head, Weight::NF4 { .. }) && !d_model.is_multiple_of(32) {
            return Err(format!("HeadCe: d_model {d_model} must be a multiple of the NF4 quantisation group (32)"));
        }
        if cfg.rows_per_chunk == 0 {
            return Err("HeadCe: rows_per_chunk must be positive".into());
        }
        let rows = cfg.rows_per_chunk;
        if rows as u64 * vocab as u64 > u32::MAX as u64 {
            return Err(format!("HeadCe: {rows} rows x {vocab} columns of logits overflow the u32 element index of the CE kernels"));
        }
        let (per_word, align) = tier_layout(head);
        let budget = cfg.tile_budget_words.unwrap_or_else(|| crate::block::tile_budget_words_for(gpu));
        let tiles = plan_tiles(vocab, d_model, per_word, align, budget);
        // A windowed head runs one pass per run of columns of each window,
        // through a dense scratch of `rows x pass_cols` - a quarter of the budget.
        let pass_cols = (((budget / 4) / rows as u64).max(64) & !63) as u32;
        let passes = if tiles.len() > 1 {
            tiles.iter().flat_map(|&(v0, cnt)| (0..cnt).step_by(pass_cols as usize).map(move |c| (v0 + c, pass_cols.min(cnt - c)))).collect()
        } else {
            Vec::new()
        };
        let tile_words = if tiles.len() > 1 { rows as u64 * pass_cols as u64 } else { 1 };
        let coop = k.ce_value_stats_rows.is_some() && ops.caps().workgroup_reductions && ops.caps().max_workgroup_size >= 256;
        Ok(HeadCe {
            d: d_model,
            vocab,
            rows,
            head: head.clone(),
            tiles,
            passes,
            logits: gpu.storage(rows as u64 * vocab as u64),
            d_logits: gpu.storage(rows as u64 * vocab as u64),
            tile: gpu.storage(tile_words),
            tile_words,
            stats: gpu.storage(2 * rows as u64),
            k,
            coop,
            scale_buf: None,
            argmax: gpu.kernel_index("argmax_part").zip(gpu.kernel_index("argmax_final")),
            bf16_dw: std::cell::RefCell::new(None),
        })
    }

    /// Words of device memory this head allocates beyond the weight: the logits
    /// and their gradient for one chunk, the window scratch and the statistics.
    /// Independent of the number of tokens.
    #[must_use]
    pub fn scratch_words(&self) -> u64 {
        2 * self.rows as u64 * self.vocab as u64 + self.tile_words + 2 * self.rows as u64
    }

    /// Row windows the head weight is cut into (1 when it fits one binding).
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.tiles.len()
    }

    /// The `[rows_per_chunk * V]` buffer a caller hands to [`RowWeights`].
    #[must_use]
    pub fn row_weight_scratch(&self, gpu: &Gpu) -> DeviceBuffer {
        gpu.storage(self.rows as u64 * self.vocab as u64)
    }

    /// The chunk logits buffer `[rows_per_chunk, V]` (the first `rows` rows
    /// hold the chunk most recently computed by [`Self::logits_steps`]).
    #[must_use]
    pub fn logits(&self) -> &DeviceBuffer {
        &self.logits
    }

    /// Multiply every chunk of logits by the first word of `buf` (read from the
    /// device when the step runs) right after the head GEMM, or stop doing so
    /// with `None`. This is how a model follows a logit scale that changes
    /// between steps without re-recording its tapes; record the tapes again
    /// after changing whether a buffer is set, because the scaling step is part
    /// of them. It composes with the constant `logit_scale` argument of the
    /// recording methods (both are applied).
    pub fn set_scale_buffer(&mut self, buf: Option<DeviceBuffer>) {
        self.scale_buf = buf;
    }

    /// Tokens whose logits are alive at once.
    #[must_use]
    pub fn rows_per_chunk(&self) -> u32 {
        self.rows
    }

    /// The `(first token, tokens)` chunks covering `n` tokens.
    #[must_use]
    pub fn chunks(&self, n: u32) -> Vec<(u32, u32)> {
        assert!(n <= self.rows || self.rows.is_multiple_of(64), "HeadCe: {n} tokens in chunks of {} rows - chunk starts must be multiples of 64 rows", self.rows);
        (0..n).step_by(self.rows as usize).map(|r0| (r0, self.rows.min(n - r0))).collect()
    }

    fn act(&self, ops: &Ops, hidden: &DeviceBuffer, r0: u32, rows: u32) -> Act {
        ops.act_f32(hidden, r0, rows, self.d)
    }

    /// Steps computing rows `r0..r0+rows` of `scale * hidden @ W^T` into the
    /// chunk logits (rows `0..rows` of [`Self::logits`]).
    pub fn logits_steps(&self, ops: &Ops, s: &mut Vec<Step>, hidden: &DeviceBuffer, r0: u32, rows: u32, logit_scale: f32) {
        let gpu = ops.gpu();
        let v = self.vocab;
        let act = self.act(ops, hidden, r0, rows);
        if self.tiles.len() == 1 {
            ops.matmul_rows(s, &self.head, (0, v), &act, &self.logits, 0);
        } else {
            for &(c0, c) in &self.passes {
                ops.matmul_rows(s, &self.head, (c0, c), &act, &self.tile, 0);
                s.push(gpu.step(self.k.copy_cols, &[&self.tile, &self.logits], &[rows, c, c, 0, v, c0], rows * c));
            }
        }
        if logit_scale != 1.0 {
            s.push(gpu.step(self.k.grad_scale, &[&self.logits], &[rows * v, logit_scale.to_bits()], rows * v));
        }
        if let Some(buf) = &self.scale_buf {
            s.push(gpu.step(self.k.grad_scale_buf, &[&self.logits, buf], &[rows * v], rows * v));
        }
    }

    /// Steps leaving the per-token cross-entropy of `labels` in `nll` for every
    /// token (`0` at [`IGNORE`]), one chunk at a time. `hidden` is `[n, d]`,
    /// `labels` and `nll` are `[n]`.
    pub fn forward_steps(&self, ops: &Ops, s: &mut Vec<Step>, hidden: &DeviceBuffer, n: u32, labels: &DeviceBuffer, nll: &DeviceBuffer, logit_scale: f32) {
        self.forward_steps_with(ops, s, hidden, n, labels, nll, None, logit_scale);
    }

    /// [`Self::forward_steps`], and - when `argmax` is given - each row's greedy
    /// token as well, taken from the chunk's logits BEFORE the next chunk
    /// overwrites them.
    ///
    /// This is the readout a caller needs to compare a model's own pick with a
    /// recorded token at every position: `nll` alone says how surprised the model
    /// was, never what it would have said. The steps are added after the chunk's
    /// cross-entropy and touch nothing the loss reads, so a head asked for the
    /// readout produces exactly the loss and log-probabilities it produced
    /// without it.
    ///
    /// # Panics
    /// If `argmax` is given on a device whose kernel set does not register
    /// `argmax_part`/`argmax_final` (see [`Self::argmax_supported`]).
    pub fn forward_steps_with(&self, ops: &Ops, s: &mut Vec<Step>, hidden: &DeviceBuffer, n: u32, labels: &DeviceBuffer, nll: &DeviceBuffer, argmax: Option<ArgmaxOut<'_>>, logit_scale: f32) {
        for (r0, rows) in self.chunks(n) {
            self.logits_steps(ops, s, hidden, r0, rows, logit_scale);
            s.push(self.ce_value_step(ops.gpu(), labels, nll, r0, rows));
            if let Some(a) = argmax {
                self.argmax_steps(ops.gpu(), s, rows, a, r0);
            }
        }
    }

    /// Whether this head's device registers the per-row argmax kernels, so
    /// [`Self::forward_steps_with`] can produce token ids.
    #[must_use]
    pub fn argmax_supported(&self) -> bool {
        self.argmax.is_some()
    }

    /// The reduction scratch [`ArgmaxOut`] needs: one `(value, index)` pair per
    /// row and chunk of the largest chunk this head computes.
    #[must_use]
    pub fn argmax_scratch(&self, gpu: &Gpu) -> DeviceBuffer {
        gpu.storage(self.rows as u64 * ARGMAX_CHUNKS as u64 * 2)
    }

    /// Steps reducing the chunk logits (rows `0..rows` of [`Self::logits`]) to
    /// each row's argmax token id, written as f32 into `out[r0..r0 + rows]`.
    fn argmax_steps(&self, gpu: &Gpu, s: &mut Vec<Step>, rows: u32, out: ArgmaxOut<'_>, r0: u32) {
        let (part, fin) = self.argmax.expect("HeadCe: the argmax readout needs the 'argmax_part'/'argmax_final' kernels registered on this Gpu");
        let chunk = self.vocab.div_ceil(ARGMAX_CHUNKS);
        s.push(gpu.step(part, &[&self.logits, out.scratch], &[rows, self.vocab, ARGMAX_CHUNKS, chunk], rows * ARGMAX_CHUNKS));
        s.push(gpu.step_sliced(fin, &[out.scratch, out.ids], &[(0, 0), (r0 as u64, rows as u64)], &[rows, ARGMAX_CHUNKS], rows));
    }

    fn ce_value_step(&self, gpu: &Gpu, labels: &DeviceBuffer, nll: &DeviceBuffer, r0: u32, rows: u32) -> Step {
        let at = (r0 as u64, rows as u64);
        let v = self.vocab;
        match (self.coop, self.k.ce_value_stats_rows) {
            (true, Some(kind)) => gpu.dispatch_sliced(kind, &[&self.logits, labels, nll, &self.stats], &[(0, 0), at, at, (0, 0)], &[rows, v, IGNORE], Dispatch::Workgroups(rows)),
            _ => gpu.step_sliced(self.k.ce_value, &[&self.logits, labels, nll], &[(0, 0), at, at], &[rows, v, IGNORE], rows),
        }
    }

    /// Recorded backward: per chunk, the CE gradient of the (re)computed logits,
    /// scaled and optionally weighted, then the head's input gradient into
    /// `d_hidden` and, where `dw` is given, its weight gradient.
    pub fn backward_steps(&self, ops: &Ops, s: &mut Vec<Step>, b: &HeadBackward) {
        let gpu = ops.gpu();
        let v = self.vocab;
        let chunks = self.chunks(b.n);
        let held = b.forward_state_held && chunks.len() == 1;
        for (r0, rows) in chunks {
            if !held {
                self.logits_steps(ops, s, b.hidden, r0, rows, b.logit_scale);
            }
            let at = (r0 as u64, rows as u64);
            // Two-pass CE gradient: the per-row softmax statistics once, then
            // one elementwise pass reading them (O(rows * V), not O(rows * V^2)).
            // A cooperative forward already left them for a held chunk.
            if !(held && self.coop) {
                s.push(gpu.step_sliced(self.k.ce_stats, &[&self.logits, b.labels, &self.stats], &[(0, 0), at, (0, 0)], &[rows, v, IGNORE], rows));
            }
            s.push(gpu.step_sliced(self.k.ce_grad_stats, &[&self.logits, b.labels, &self.stats, &self.d_logits], &[(0, 0), at, (0, 0), (0, 0)], &[rows, v, IGNORE, 1f32.to_bits()], rows * v));
            // Chain rule through `logits = logit_scale * (h W^T)`, then the caller's scale.
            let mut host_coef = if b.logit_scale == 1.0 { None } else { Some(b.logit_scale) };
            match b.grad_scale {
                GradScale::Const(c) => host_coef = Some(host_coef.unwrap_or(1.0) * c).filter(|&c| c != 1.0),
                GradScale::Buf(buf) => s.push(gpu.step(self.k.grad_scale_buf, &[&self.d_logits, buf], &[rows * v], rows * v)),
            }
            if let Some(coef) = host_coef {
                s.push(gpu.step(self.k.grad_scale, &[&self.d_logits], &[rows * v, coef.to_bits()], rows * v));
            }
            let d_logits = match b.row_weights {
                Some(rw) => {
                    s.push(gpu.step_sliced(self.k.scale_row, &[&self.d_logits, rw.weights, rw.scratch], &[(0, 0), at, (0, 0)], &[rows * v, v], rows * v));
                    rw.scratch
                }
                None => &self.d_logits,
            };
            self.head_backward_steps(ops, s, d_logits, b, r0, rows);
        }
    }

    /// `d_hidden[r0..r0+rows] = d_logits @ W` (and `dW += d_logits^T @ hidden`),
    /// window by window.
    fn head_backward_steps(&self, ops: &Ops, s: &mut Vec<Step>, d_logits: &DeviceBuffer, b: &HeadBackward, r0: u32, rows: u32) {
        let gpu = ops.gpu();
        let (d, v) = (self.d, self.vocab);
        let x = (r0 as u64 * d as u64, rows as u64 * d as u64);
        if self.tiles.len() == 1 {
            if let Some(dw) = b.dw {
                self.dw_step(ops, s, d_logits, b.hidden, dw, x, rows, v, 0);
            }
            ops.matmul_dx_rows(s, &self.head, (0, v), d_logits, rows, b.d_hidden, r0, false);
            return;
        }
        for (i, &(c0, c)) in self.passes.iter().enumerate() {
            s.push(gpu.step(self.k.copy_cols, &[d_logits, &self.tile], &[rows, c, v, c0, c, 0], rows * c));
            if let Some(dw) = b.dw {
                self.dw_step(ops, s, &self.tile, b.hidden, dw, x, rows, c, c0);
            }
            ops.matmul_dx_rows(s, &self.head, (c0, c), &self.tile, rows, b.d_hidden, r0, i > 0);
        }
    }

    /// `dW[first.., :] += dy^T @ hidden[rows]`, `dy` dense `[rows, cols]`. An fp32 gradient is one product into its
    /// window; a bf16 gradient is produced a bounded run of vocabulary rows at a time into an f32 scratch of
    /// [`Ops::tile_budget_bytes`] and added with round to nearest even, so no `[V, d]` f32 copy ever exists.
    #[allow(clippy::too_many_arguments)]
    fn dw_step(&self, ops: &Ops, s: &mut Vec<Step>, dy: &DeviceBuffer, hidden: &DeviceBuffer, dw: HeadGrad, x: (u64, u64), rows: u32, cols: u32, first: u32) {
        let d = self.d;
        let product = |s: &mut Vec<Step>, dy: &DeviceBuffer, out: &DeviceBuffer, window: (u64, u64), cols: u32| {
            let (kind, grid) = match self.k.matmul_dw_reg {
                Some(reg) if ops.fills_tile(cols, d) => (reg, Dispatch::Workgroups(cols.div_ceil(128) * d.div_ceil(128))),
                _ => (self.k.matmul_dw, Dispatch::Threads(cols * d)),
            };
            s.push(ops.gpu().dispatch_sliced(kind, &[dy, hidden, out], &[(0, 0), x, window], &[rows, d, cols], grid));
        };
        match dw.dtype {
            GradDtype::F32 => product(s, dy, dw.buf, (first as u64 * d as u64, cols as u64 * d as u64), cols),
            GradDtype::Bf16 => {
                assert!(d.is_multiple_of(2), "HeadCe: a bf16 head gradient needs an even d_model (got {d})");
                let (scratch, sub_dy, sub) = self.bf16_dw_scratch(ops, cols);
                for j0 in (0..cols).step_by(sub as usize) {
                    let c = sub.min(cols - j0);
                    let src = if sub < cols {
                        s.push(ops.gpu().step(self.k.copy_cols, &[dy, &sub_dy], &[rows, c, cols, j0, c, 0], rows * c));
                        &sub_dy
                    } else {
                        dy
                    };
                    product(s, src, &scratch, (0, c as u64 * d as u64), c);
                    ops.accumulate_bf16(s, &scratch, dw.buf, c as u64 * d as u64, (first + j0) as u64 * d as u64);
                }
            }
        }
    }

    /// The zeroed f32 product scratch and the `dy` column scratch of a bf16 head gradient over `cols` vocabulary
    /// rows, with the rows one product covers. Allocated once per build; the accumulation clears the product scratch
    /// after every use, so it stays zero between calls.
    fn bf16_dw_scratch(&self, ops: &Ops, cols: u32) -> (DeviceBuffer, DeviceBuffer, u32) {
        let mut slot = self.bf16_dw.borrow_mut();
        let sub = tile_rows(self.vocab, self.d, ops.tile_budget_bytes(), 4);
        let (scratch, sub_dy) = slot.get_or_insert_with(|| {
            let gpu = ops.gpu();
            (gpu.storage(sub as u64 * self.d as u64), gpu.storage_uninit(self.rows as u64 * sub as u64))
        });
        (scratch.clone(), sub_dy.clone(), sub.min(cols))
    }

    /// Immediate cross-entropy of `labels` over `hidden` (`[n, d]`): the loss,
    /// token count and per-token log-probabilities, without ever holding more
    /// than one chunk of logits. `weights` scale each token's loss term.
    pub fn forward(&self, ops: &Ops, hidden: &DeviceBuffer, n: u32, labels: &[u32], weights: Option<&[f32]>, logit_scale: f32) -> HeadLoss {
        assert_eq!(labels.len(), n as usize, "HeadCe::forward: {} labels for {n} tokens", labels.len());
        let gpu = ops.gpu();
        let (lab, nll) = (self.upload_labels(gpu, labels), gpu.storage(n as u64));
        let mut s = Vec::new();
        self.forward_steps(ops, &mut s, hidden, n, &lab, &nll, logit_scale);
        gpu.submit(&[], &s);
        let nll = gpu.read(&nll, n as usize);
        let w = |i: usize| weights.map_or(1.0, |ws| ws[i] as f64);
        HeadLoss {
            loss_sum: nll.iter().enumerate().map(|(i, &l)| l as f64 * w(i)).sum(),
            count: labels.iter().filter(|&&l| l != IGNORE).count() as u32,
            logprobs: nll.iter().map(|&l| -l).collect(),
        }
    }

    /// Immediate backward matching [`Self::forward`]: `d_hidden` receives
    /// `d_scale * d(sum_i w_i nll_i)/d(hidden)` and `dw`, where given, has the
    /// head's gradient added.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(&self, ops: &Ops, hidden: &DeviceBuffer, d_hidden: &DeviceBuffer, n: u32, labels: &[u32], weights: Option<&[f32]>, d_scale: f32, logit_scale: f32, dw: Option<&DeviceBuffer>) {
        assert_eq!(labels.len(), n as usize, "HeadCe::backward: {} labels for {n} tokens", labels.len());
        let gpu = ops.gpu();
        let lab = self.upload_labels(gpu, labels);
        let wbuf = weights.map(|ws| gpu.storage_init("row_weights", ws));
        let scratch = weights.map(|_| self.row_weight_scratch(gpu));
        let mut s = Vec::new();
        self.backward_steps(
            ops,
            &mut s,
            &HeadBackward {
                hidden,
                d_hidden,
                n,
                labels: &lab,
                grad_scale: GradScale::Const(d_scale),
                logit_scale,
                row_weights: wbuf.as_ref().zip(scratch.as_ref()).map(|(weights, scratch)| RowWeights { weights, scratch }),
                dw: dw.map(|buf| HeadGrad { buf, dtype: GradDtype::F32 }),
                forward_state_held: false,
            },
        );
        gpu.submit(&[], &s);
    }

    fn upload_labels(&self, gpu: &Gpu, labels: &[u32]) -> DeviceBuffer {
        let buf = gpu.storage(labels.len().max(1) as u64);
        gpu.write(&buf, labels);
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vocab_that_fits_the_budget_is_one_window() {
        assert_eq!(plan_tiles(1000, 96, 1, 64, 96 * 1000), vec![(0, 1000)]);
    }

    #[test]
    fn windows_cover_the_vocab_in_aligned_runs_and_never_exceed_the_budget() {
        // NF4: window starts must be multiples of 2048 / gcd(2048, d) rows.
        for (vocab, d, per_word, align, budget) in [(1000u32, 96u32, 8u64, 2048u64, 12 * 250), (248_320, 5120, 8, 2048, 1 << 26), (777, 160, 2, 128, 80 * 100), (300, 96, 1, 64, 96 * 70)] {
            let tiles = plan_tiles(vocab, d, per_word, align, budget);
            let mut next = 0;
            for &(v0, cnt) in &tiles {
                assert_eq!(v0, next, "windows must be contiguous");
                assert_eq!(v0 as u64 * d as u64 % align, 0, "window start {v0} of width {d} is not binding-aligned");
                let granule = align / gcd(align, d as u64);
                assert!(cnt as u64 * (d as u64 / per_word) <= budget.max(granule * (d as u64 / per_word)), "window of {cnt} rows exceeds the budget");
                next = v0 + cnt;
            }
            assert_eq!(next, vocab);
        }
    }

    /// Every window of a vocabulary that is a whole number of 64-row groups is itself
    /// a whole number of them, so a tensor-core GEMM over it sees 16-byte rows
    /// (the 27B's 248,320 rows, at the bf16 and NF4 granules, small and large budgets).
    #[test]
    fn windows_of_a_tileable_vocab_are_whole_groups_of_64_rows() {
        for (per_word, align, budget) in [(2u64, 128u64, 1u64 << 26), (2, 128, 1 << 28), (8, 2048, 1 << 26), (8, 2048, 1 << 28)] {
            for &(v0, cnt) in &plan_tiles(248_320, 5120, per_word, align, budget) {
                assert_eq!((v0 % 64, cnt % 64), (0, 0), "window ({v0}, {cnt}) at per_word {per_word}, budget {budget}");
            }
        }
    }

    #[test]
    fn rows_for_budget_keeps_whole_alignment_groups() {
        assert_eq!(rows_for_budget(100, 10, 1 << 20), 100);
        let rows = rows_for_budget(100_000, 248_320, 1 << 29);
        assert_eq!(rows % 64, 0);
        assert!(rows * 248_320 <= 1 << 29);
        assert_eq!(rows_for_budget(100_000, 248_320, 1000), 64);
    }
}
