// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **DFlash2** - a real block-diffusion DRAFT model for
//! [`crate::int8_gguf_resident::Qwen35GgufInstance::generate_speculative`].
//!
//! Everything else that has driven that loop so far was a stand-in: an oracle
//! fed the answer (the ceiling, not a drafter) and an n-gram repeat detector
//! (a real drafter, and a measured LOSS on free-form text). This is the first
//! one that runs a model.
//!
//! **What makes it different from a small autoregressive drafter.** It is not
//! autoregressive at all. One forward pass over `[anchor, MASK, MASK, ...]`
//! denoises the whole block at once - `block_size` rows, `block_size - 1`
//! proposals - so `k` draft tokens cost ONE draft forward, not `k` of them.
//! That is the entire economic argument for it on this stack: a target decode
// perf-number: measured by `tests/dflash2_real.rs` (command in its header)
//! step measures 152 ms here and a whole draft round measures 48 ms (44 ms of
// perf-number: measured by `tests/dflash2_real.rs` (command in its header)
//! device work, 4 ms of selector), so a per-token drafter would have to fit
//! seven of its own passes inside one target step and this one does not have
//! to.
//!
// perf-number: measured by `tests/dflash2_real.rs` (command in its header)
//! Measured (2x Tesla P40, both checkpoints Q8_0 served INT8, greedy): 2.30
//! accepted draft tokens per round on a free-form prompt and 6.00 on a
//! repetition workload, against the model-free n-gram drafter's 0.57 and 5.12,
//! and on free-form text the first drafter on this stack that is a net
// perf-number: measured by `tests/dflash2_real.rs` (command in its header)
//! win (1.08x at `k = 3`) rather than a loss. `tests/dflash2_real.rs` carries
//! the full table and the cost model that explains why the wall-clock win is
//! much smaller than the acceptance rate alone implies.
//!
//! Four things carry the quality that a plain masked denoiser would not have,
//! and all four are re-derived from the checkpoint in this file rather than
//! assumed:
//!
//! * **It reads the target's INTERNALS, not its output.** Each draft layer's
//!   keys and values are computed over the concatenation of (a) the target's
//!   own residual at layers `target_layers`, projected by `fc` and normalized
//!   by `enc.output_norm`, one row per context token, and (b) the block's own
//!   rows. So the drafter's "prompt" is 25600 numbers per token that only the
//!   target can produce - which is why
//!   [`Qwen35GgufInstance::enable_hidden_taps`] exists.
//! * **Attention is NON-causal** (`dflash.attention.causal = false`), within a
//!   2048 sliding window in BOTH directions. The block's mask rows see each
//!   other, which is what a diffusion denoiser needs and what a causal kernel
//!   cannot express.
//! * **Two-tap dynamic convolutions**, before and after each of the attention
//!   and MLP sublayers. The taps are not weights: half of a `[5120 -> 1280]`
//!   projection of the activations IS the tap set, per token, per group of 16
//!   channels, added to a static per-channel base. This is what the model card
//!   credits with keeping the draft from decaying toward the end of a block,
//!   and it needed a new kernel (`kernels::DYN_GROUP_CONV1D`) - nothing in the
//!   tree had a data-dependent conv weight.
//! * **A selector**, not an argmax. The head's top-16 candidates at every
//!   position are scored against the token actually chosen at the previous
//!   position through a rank-256 three-way elementwise product, and the walk is
//!   greedy left to right. Picking each position's argmax independently, which
//!   is what a non-diffusion drafter does, produces a block whose tokens are
//!   individually likely and jointly incoherent.
//!
//! **What this module does NOT own.** The token embedding and the `lm_head`
//! are the TARGET's - the draft checkpoint ships neither, by design - so the
//! noise embedding is gathered from the target's `token_embd.weight` and the
//! draft's hidden states are projected by the target's own resident INT8 head
//! (with the DRAFT's `output_norm` standing in for the target's final norm,
//! which is exactly the composition the reference implementation performs).
//! That also means no second 1.27 GB head copy on the cards.
//!
//! **The model is shared.** The draft's forward, its K/V caches and its selector live in the `dflash2` crate, which
//! the Ripple engine drives through the generic scheduler as well; this module is the Qwen3.8 glue - the GGUF
//! source, the single cache slot, and [`Dflash2Drafter`], which reads the hidden states the target instance taps
//! on this single-stream path.
//!
//! Gated against a dependency-free host reference over the same bytes,
//! `tools/goldens/dflash2_reference_forward.py`, which re-implements the
//! published `dflash/model.py` and is what every claim above was checked
//! against - see `crates/qwen35/tests/it/dflash2_real.rs`.
//!
//! Swedish Embedded AB implements speculative decoding - drafters, verifiers
//! and the state rollback a hybrid recurrent stack needs - for clients serving
//! large models on modest hardware. If your team needs inference throughput
//! without a second model's worth of accuracy loss, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::cell::RefCell;

use gpu_core::select::Dtype;
use gpu_core::Gpu;

use crate::int8_gguf_resident::Qwen35GgufInstance;

const MODEL: &str = "dflash2";

/// The GGUF architecture string this module loads.
pub use ::dflash2::config::GGUF_ARCHITECTURE;
pub use ::dflash2::Dflash2Config;

/// This model's kernel set: the target decoder's own list (which already carries every `model::ops::Ops`
/// requirement, the paged-attention triad this module attends through, and `splice`), plus what the shared draft
/// adds (the dynamic conv, the KV tier's append, and the full table-driven RoPE - the target rotates only 64 of its
/// 256 head channels and so registers `rope2d_partial` instead).
pub fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v = crate::model::pipelines().to_vec();
        let more = ::dflash2::model::kernels_beyond(&v, &[]);
        v.extend(more);
        v
    })
}

/// A loaded DFlash2 draft model for the Qwen3.8 target, resident on one card: the shared [`::dflash2::Dflash2`] over
/// the Q8_0 GGUF with a single cache slot, which is the single-stream shape `generate_speculative` drives.
///
/// Derefs to the shared model for its configuration, selector and final norm.
pub struct Dflash2 {
    inner: ::dflash2::Dflash2,
}

impl std::ops::Deref for Dflash2 {
    type Target = ::dflash2::Dflash2;
    fn deref(&self) -> &::dflash2::Dflash2 {
        &self.inner
    }
}

impl Dflash2 {
    /// [`Self::load_dt`] at the serving tier, INT8.
    pub fn load(gpu: Gpu, gguf_path: &str, cap: u32) -> Result<Dflash2, String> {
        Self::load_dt(gpu, gguf_path, cap, Dtype::I8)
    }

    /// Load the draft checkpoint onto `gpu`, with room for `cap` context positions, at weight tier `dt`.
    ///
    /// `Dtype::I8` is the serving tier: weights land as group-wise INT8 straight out of the file's Q8_0 blocks - no
    /// fp32 intermediate, the same route the target's own resident takes, and the reason a 1.9B draft is ~2 GB
    /// beside a 27 GB target rather than 7.6.
    ///
    /// `Dtype::F32` exists for the host-reference gate and is 7.4 GB. It is not an academic option: this model
    /// amplifies its input by five orders of magnitude (the embedding it starts from has RMS 0.014, the hidden state
    /// it ends with has RMS ~1000), so "the INT8 and fp32 answers differ by 10%" and "the port is wrong by 10%" are
    /// indistinguishable without running the tier the reference actually implements.
    pub fn load_dt(gpu: Gpu, gguf_path: &str, cap: u32, dt: Dtype) -> Result<Dflash2, String> {
        let src = ::dflash2::GgufDraft::open(gguf_path, dt)?;
        let cfg = Dflash2Config::from_gguf(src.gguf())?;
        Ok(Dflash2 { inner: ::dflash2::Dflash2::load(gpu, cfg, Box::new(src), 1, cap)? })
    }

    /// Project and cache the target's hidden states for the context tokens starting at absolute position
    /// `pos_start` ([`::dflash2::Dflash2::append_context`] on the one cache slot).
    pub fn append_context(&self, hidden: &[f32], pos_start: u32) -> Result<(), String> {
        self.inner.append_context(0, hidden, pos_start)
    }

    /// The denoising pass over `[anchor, MASK, ...]` at absolute positions `pos ..`: the PRE-final-norm hidden state
    /// of every row, `[n, d_model]`. The final norm is applied on the target's card by the target's own head
    /// epilogue with [`::dflash2::Dflash2::output_norm`] substituted, so no second copy of a 1.27 GB `lm_head` is needed.
    pub fn denoise_block(&self, noise: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        self.inner.denoise_block(noise, pos)
    }
}

// --------------------------------------------------------------- the drafter

/// **The drafter**, in the shape `generate_speculative` takes: given the
/// context so far, return up to `want` proposals.
///
/// Holds the target instance because half the work is the target's - the
/// tapped hidden states it cross-attends to, the embedding of `[anchor,
/// MASK...]`, and the head that turns draft hidden states into logits.
///
/// The state it carries is `fed`: how many context positions are already in
/// the draft's K/V cache. Everything else about a speculative round - which
/// proposals were accepted, whether the recurrent state was rolled back -
/// shows up here as nothing more than "the context is now this long", because
/// the draft cache is indexed by absolute position and a re-committed prefix
/// overwrites itself.
///
/// The one thing `fed` alone cannot survive is a NEW generation, which is why
/// `seen` exists beside it: a second request with a longer prompt looks
/// exactly like the first one continuing, and carrying `fed` across would
/// leave the previous request's context in the draft's cache under the new
/// request's positions - a drafter conditioned on someone else's prompt,
/// producing fluent proposals that are simply never accepted. The target's
/// [`Qwen35GgufInstance::hidden_tap_generation`] is what distinguishes them,
/// and it is checked every round rather than left to a caller to remember.
pub struct Dflash2Drafter<'a> {
    inst: &'a Qwen35GgufInstance,
    model: Dflash2,
    fed: RefCell<u32>,
    /// The target's tap generation this drafter's cache belongs to.
    seen: RefCell<u64>,
    stats: RefCell<DraftStats>,
}

/// What the drafter itself cost, as opposed to what it bought - the two are
/// separately actionable and a single tok/s figure hides both.
#[derive(Clone, Copy, Debug, Default)]
pub struct DraftStats {
    pub calls: u64,
    /// Seconds in the draft forward (context projection + denoise + head).
    pub draft_s: f64,
    /// Seconds in the selector walk (host).
    pub select_s: f64,
    /// Proposals returned.
    pub proposed: u64,
}

impl<'a> Dflash2Drafter<'a> {
    /// Attach a loaded draft model to a target instance, turning on the hidden
    /// taps the draft needs.
    pub fn new(inst: &'a Qwen35GgufInstance, model: Dflash2) -> Dflash2Drafter<'a> {
        inst.enable_hidden_taps(&model.cfg.target_layers);
        let seen = RefCell::new(inst.hidden_tap_generation());
        Dflash2Drafter { inst, model, fed: RefCell::new(0), seen, stats: RefCell::new(DraftStats::default()) }
    }

    /// The largest useful `k`: one block is `block_size` rows, of which the
    /// first is the anchor.
    pub fn max_draft(&self) -> u32 {
        self.model.cfg.block_size - 1
    }

    pub fn stats(&self) -> DraftStats {
        *self.stats.borrow()
    }

    /// Forget the context fed so far. Rarely needed by hand: [`Self::propose`]
    /// notices a new generation on its own (see this struct's own doc), and
    /// this is here for a caller that wants to force it.
    pub fn reset(&self) {
        *self.fed.borrow_mut() = 0;
        *self.seen.borrow_mut() = self.inst.hidden_tap_generation();
    }

    /// **One draft**, the whole contract `generate_speculative` asks for.
    ///
    /// The anchor is `ctx`'s LAST token; the context rows are everything
    /// before it. That split is the reference's: a block's first row carries
    /// the already-verified token as an EMBEDDING, and the target's hidden
    /// state for it does not exist yet (it is produced by the very verify pass
    /// this draft is feeding).
    ///
    /// Returns an empty proposal rather than an error when anything is not
    /// ready - a drafter that fails is a drafter that proposes nothing, and
    /// the verify loop is already correct for that case.
    pub fn propose(&self, ctx: &[u32], want: u32) -> Vec<u32> {
        match self.try_propose(ctx, want) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{MODEL}: drafting skipped: {e}");
                Vec::new()
            }
        }
    }

    fn try_propose(&self, ctx: &[u32], want: u32) -> Result<Vec<u32>, String> {
        if want == 0 || ctx.len() < 2 {
            return Ok(Vec::new());
        }
        let want = want.min(self.max_draft());
        let block = want + 1;
        let anchor = *ctx.last().expect("non-empty (checked)");
        let ctx_len = ctx.len() as u32 - 1;

        let t0 = std::time::Instant::now();
        // A new sequence invalidates every cached context row, whatever the
        // context length says - see this struct's own doc.
        let gen = self.inst.hidden_tap_generation();
        if gen != *self.seen.borrow() {
            *self.seen.borrow_mut() = gen;
            *self.fed.borrow_mut() = 0;
        }
        // Catch the draft's K/V cache up with whatever the target committed
        // since the last round.
        let fed = *self.fed.borrow();
        if ctx_len > fed {
            let rows = self.inst.target_hidden(fed, ctx_len - fed)?;
            self.model.append_context(&rows, fed)?;
            *self.fed.borrow_mut() = ctx_len;
        }

        let mut ids = Vec::with_capacity(block as usize);
        ids.push(anchor);
        ids.extend(std::iter::repeat_n(self.model.cfg.mask_token_id, want as usize));
        let noise = self.inst.embed_rows_of(&ids)?;
        let hidden = self.model.denoise_block(&noise, ctx_len)?;

        // Row 0 is the anchor's own denoised state and predicts nothing; the
        // MASK rows are the proposals.
        let d = self.model.cfg.d_model as usize;
        let mask_rows = &hidden[d..];
        let logits = self.inst.logits_with_norm(mask_rows, want, self.model.output_norm())?;
        let draft_s = t0.elapsed().as_secs_f64();

        let t1 = std::time::Instant::now();
        let path = self.model.select(mask_rows, &logits, anchor)?;
        let mut st = self.stats.borrow_mut();
        st.calls += 1;
        st.draft_s += draft_s;
        st.select_s += t1.elapsed().as_secs_f64();
        st.proposed += path.len() as u64;
        Ok(path)
    }
}

