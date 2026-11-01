// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in drafting from a
// checkpoint's own multi-token-prediction head, then you can procure our
// services by sending an email to info@swedishembedded.com.

//! **The checkpoint's own multi-token-prediction (MTP) head as a drafter** for
//! the generic speculative scheduler.
//!
//! The released GGUF carries one extra decoder layer (`blk.{n_layers}`, a
//! full-attention block with a dense MLP) and four `nextn.*` tensors. Given the
//! target's hidden state `h_j` at position `j` and the embedding of the token
//! `x_{j+1}` that followed it, the head predicts `x_{j+2}`:
//!
//! ```text
//! ehp = eh_proj([rmsnorm(embed(x_{j+1}), enorm), rmsnorm(h_j, hnorm)])
//! out = layer(ehp)            // causal self-attention over its OWN KV cache of
//!                             // earlier positions, then the MLP
//! logits = lm_head(rmsnorm(out, shared_head_norm))     // the target's own head
//! ```
//!
//! so one cheap forward (a layer, an `eh_proj` and the shared head) proposes
//! the token the target's NEXT output will be, and feeding `out` back in as the
//! hidden state with the proposal embedded proposes the one after it. Nothing
//! here needs the proposals to be right: the scheduler verifies every one
//! against the target, so a wrong head costs rows, never tokens.
//!
//! The layer is built as a one-layer pipeline stage of its own (a shard
//! `[interval-1, interval)` of a config truncated to one attention period, so
//! it is a `Full` layer under the weight names `blocks.{interval-1}.*`), with
//! its own KV pool addressed by the SAME slot as the target's, so a
//! sequence's draft state follows its slot through prefix reuse, release and
//! reuse. Per slot the drafter keeps the hidden states of positions the target
//! has consumed but the head has not yet seen ([`SlotState`]); a draft round
//! feeds them (the "catch-up", one batched forward over every drafting
//! sequence), reads the first proposal off the last, and extends it a token at a
//! time for the rest of the window.

use super::*;
use checkpoint::TensorSource;
use crate::gguf_import::block_leaf_suffix;
use model::ops::{Ops, Weight};

mod plan;

/// Hidden rows a slot may hold awaiting a draft before they are fed to the
/// head anyway (bounds the pending store while speculation is off for a while).
const MAX_PENDING_ROWS: usize = 16;

/// Rows of the pending store a slot owns: the rows it may hold before they are fed, plus the widest verify step that
/// can land on top of them.
const PENDING_STRIDE: usize = MAX_PENDING_ROWS + model::gdn_verify::MAX_VERIFY_ROWS as usize;

/// Pairs one chunk of a feed carries: a chunk is one prefill round of the head's layer, so this is the rows of one launch.
const FEED_CHUNK_ROWS: usize = 1024;

/// What the head has seen of one sequence, and what the target has produced
/// that it has not.
#[derive(Default)]
struct SlotState {
    /// Head positions `0..fed` are in its KV cache (positions `j` whose pair
    /// `(h_j, x_{j+1})` is complete and was run).
    fed: u32,
    /// `[rows]`: the token the target consumed AT each of the positions `fed..fed+rows`. The target's hidden state at
    /// position `fed + r` is row `r` of this slot's span of the pending store ([`MtpDrafter::pend`]), on the card. The
    /// pair for row `r` is `(hidden[r], toks[r + 1])`; the last row's partner is the sequence's next input, which only
    /// the draft call knows.
    toks: Vec<u32>,
}

/// A prompt's feed to the head that was begun and is not finished ([`Qwen35GgufInstance::mtp_begin_feed`]).
pub(super) struct PromptFeed {
    slot: u32,
    /// Head position of the first pair.
    first: usize,
    /// Pairs run so far.
    done: usize,
    /// `[pairs]`: the token each pair is partnered with.
    tokens: Vec<u32>,
    /// The target's hidden state of each pair, row `i` for pair `i`.
    hidden: HiddenRun,
}

impl PromptFeed {
    pub(super) fn slot(&self) -> u32 {
        self.slot
    }
}

/// A block of the target's hidden rows a feed will read: on the host (the sequential prefill reads its rounds back) or already
/// on the card (a multi-sequence forward leaves its rows there, and they stay there).
pub(super) enum HiddenRows {
    Host(Vec<f32>),
    Device { buf: DeviceBuffer, rows: usize },
}

impl HiddenRows {
    fn rows(&self, d: usize) -> usize {
        match self {
            HiddenRows::Host(v) => v.len() / d,
            HiddenRows::Device { rows, .. } => *rows,
        }
    }
}

/// The rows of consecutive [`HiddenRows`] blocks as one run.
pub(super) struct HiddenRun {
    blocks: Vec<HiddenRows>,
    d: usize,
}

impl HiddenRun {
    pub(super) fn new(d: usize, blocks: Vec<HiddenRows>) -> HiddenRun {
        HiddenRun { blocks, d }
    }

    pub(super) fn rows(&self) -> usize {
        self.blocks.iter().map(|b| b.rows(self.d)).sum()
    }

    /// Rows `from..from + count` of the run as one `[count, d]` device buffer on `g`.
    pub(super) fn stage(&self, g: &Gpu, from: usize, count: usize) -> DeviceBuffer {
        let d = self.d;
        let out = g.storage((count * d) as u64);
        let (mut at, mut put) = (0usize, 0usize);
        for block in &self.blocks {
            let rows = block.rows(d);
            let (lo, hi) = (from.max(at), (from + count).min(at + rows));
            if lo < hi {
                let (src, n) = (lo - at, hi - lo);
                match block {
                    HiddenRows::Host(v) => g.write_f32_at(&out, (put * d) as u64, &v[src * d..(src + n) * d]),
                    HiddenRows::Device { buf, .. } => g.submit(&[], &[MtpDrafter::copy_cols_step(g, buf, &out, n as u32, d as u32, d as u32, (src * d) as u32, d as u32, (put * d) as u32)]),
                }
                put += n;
            }
            at += rows;
        }
        out
    }

    /// Row `row` of the run, on the host.
    fn row_host(&self, g: &Gpu, row: usize) -> Vec<f32> {
        g.read(&self.stage(g, row, 1), self.d)
    }
}

/// How [`Qwen35GgufInstance::mtp_begin_feed`] found the prompt.
pub(super) enum FeedStart {
    /// No head, or no hidden state to continue a cached prefix from: the slot drafts from scratch.
    Unavailable,
    /// Nothing to run: the prompt has no complete pair.
    Complete,
    Pending(PromptFeed),
}

/// What one [`Qwen35GgufInstance::mtp_advance_feed`] left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FeedStep {
    More,
    Done,
    /// The head failed on the prompt; the slot drafts from scratch.
    Failed,
}

impl SlotState {
    fn rows(&self) -> usize {
        self.toks.len()
    }
}

/// The MTP layer, its projection and norms, and the per-slot draft state.
pub(super) struct MtpDrafter {
    stage: Qwen35,
    pub(super) caches: ShardCaches,
    fc: Weight,
    /// `shared_head_norm` on the head's card, for the shared `lm_head`.
    head_norm: DeviceBuffer,
    /// The three norms on the head's card, for the device drafter ([`plan`]).
    enorm_dev: DeviceBuffer,
    hnorm_dev: DeviceBuffer,
    target_norm_dev: DeviceBuffer,
    /// `[slots, PENDING_STRIDE, d]`: the target's hidden state of every position a slot has consumed and the head has
    /// not yet fed on, on the card: a recorded target step writes its rows there and the drafter reads them there, so
    /// the hidden states of a speculating sequence never visit the host.
    pend: DeviceBuffer,
    pend_slots: u32,
    /// Hand the head `rmsnorm(h, output_norm)` instead of the raw residual.
    normed_hidden: bool,
    d: usize,
    slots: std::collections::HashMap<u32, SlotState>,
    /// Recorded draft rounds ([`plan`]).
    draft_plans: std::cell::RefCell<plan::DraftPlans>,
    /// Slots whose prompt feed is begun and not finished ([`PromptFeed`]): they neither draft nor record steps.
    feeding: std::collections::HashSet<u32>,
    /// Forwards run, for the benchmark.
    pub(super) forwards: u64,
    /// Wall time of the pieces of a forward: host prep and projection, the layer
    /// stage, the shared head - for the benchmark.
    pub(super) times: [std::time::Duration; 3],
}

impl MtpDrafter {
    /// `Ok(None)` when the checkpoint carries no MTP block.
    #[allow(clippy::too_many_arguments)]
    fn load(
        ckpt: &ResidentCheckpoint,
        cfg: &Qwen35Config,
        tier: &TierPolicy,
        gpu_index: usize,
        head_gpu: &Gpu,
        cap: u32,
        slots: u32,
        kv_tier: KvTier,
        layout: KvLayout,
        sparse: bool,
        target_norm_name: &str,
        normed_hidden: bool,
    ) -> Result<Option<MtpDrafter>, String> {
        let d = cfg.d_model as usize;
        let Some(parts) = MtpParts::read(ckpt, cfg, tier, gpu_index, head_gpu)? else { return Ok(None) };
        let MtpParts { cfg1, shard, stage, fc, enorm, hnorm, head_norm_host } = parts;
        let caches = ShardCaches::new(&stage.gpu, &cfg1, &shard, cap, slots, 0, false, kv_tier, layout, false, sparse)?;
        let target_norm_host = match ckpt.final_norm(target_norm_name) {
            Some(Ok(v)) if v.len() == d => v,
            Some(Ok(v)) => return Err(format!("{MODEL}: '{target_norm_name}' has {} elements, expected {d}", v.len())),
            Some(Err(e)) => return Err(format!("{MODEL}: '{target_norm_name}': {e}")),
            None => return Err(format!("{MODEL}: '{target_norm_name}' is missing")),
        };
        Ok(Some(MtpDrafter {
            stage,
            caches,
            fc,
            head_norm: head_gpu.storage_init("qwen35.gguf.mtp_norm", &head_norm_host),
            enorm_dev: head_gpu.storage_init("qwen35.mtp.enorm", &enorm),
            hnorm_dev: head_gpu.storage_init("qwen35.mtp.hnorm", &hnorm),
            target_norm_dev: head_gpu.storage_init("qwen35.mtp.target_norm", &target_norm_host),
            pend: head_gpu.storage((slots as usize * PENDING_STRIDE * d) as u64),
            pend_slots: slots,
            normed_hidden,
            d,
            slots: std::collections::HashMap::new(),
            draft_plans: Default::default(),
            feeding: std::collections::HashSet::new(),
            forwards: 0,
            times: Default::default(),
        }))
    }
}

/// What the checkpoint gives the drafter, whichever format it is in: the one-layer stage (its weights in the
/// tier the checkpoint stores them), the projection over the pair `[embed, hidden]`, the two input norms and
/// the norm in front of the shared head.
struct MtpParts {
    cfg1: Qwen35Config,
    shard: Shard,
    stage: Qwen35,
    fc: Weight,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    head_norm_host: Vec<f32>,
}

impl MtpParts {
    /// `Ok(None)` when the checkpoint carries no MTP head.
    fn read(ckpt: &ResidentCheckpoint, cfg: &Qwen35Config, tier: &TierPolicy, gpu_index: usize, head_gpu: &Gpu) -> Result<Option<MtpParts>, String> {
        let d = cfg.d_model as usize;
        let exact = |name: &str, v: Vec<f32>| if v.len() == d { Ok(v) } else { Err(format!("{MODEL}: '{name}' has {} elements, expected {d}", v.len())) };
        match ckpt {
            ResidentCheckpoint::Gguf(mg) => {
                let prefix = format!("blk.{}.", cfg.n_layers);
                let fc_name = mtp_projection_name(cfg);
                match mg.shape(&fc_name) {
                    None => return Ok(None),
                    Some(s) if s == [d, 2 * d] => {}
                    Some(s) => return Err(format!("{MODEL}: {fc_name} has shape {s:?}, expected [{d}, {}]", 2 * d)),
                }
                let MtpSource { cfg1, shard, src } = mtp_source(mg, cfg, gpu_index)?;
                let stage = Qwen35::new_shard_dt(cfg1.clone(), 1, 1, &src, shard.clone(), &stage_tier(cfg, tier, shard.start));
                let (w, s) = model::int8::upload_quantized(&mut paramstore::upload::Uploader::new(head_gpu), mg, &fc_name, d, 2 * d).map_err(|e| format!("{MODEL}: {e}"))?;
                let fc = Weight::I8 { w, s, n: d as u32, k: 2 * d as u32 };
                let vec = |leaf: &str| -> Result<Vec<f32>, String> {
                    let name = format!("{prefix}{leaf}");
                    match mg.tensor(&name) {
                        Some(Ok(v)) => exact(&name, v),
                        Some(Err(e)) => Err(format!("{MODEL}: '{name}': {e}")),
                        None => Err(format!("{MODEL}: '{name}' is missing")),
                    }
                };
                Ok(Some(MtpParts { cfg1, shard, stage, fc, enorm: vec("nextn.enorm.weight")?, hnorm: vec("nextn.hnorm.weight")?, head_norm_host: vec("nextn.shared_head_norm.weight")? }))
            }
            ResidentCheckpoint::Hf(_) => {
                let Some(src) = ckpt.mtp_source() else { return Ok(None) };
                let (cfg1, shard) = mtp_stage_shape(cfg, gpu_index);
                let stage = Qwen35::new_shard_dt(cfg1.clone(), 1, 1, src, shard.clone(), &stage_tier(cfg, tier, shard.start));
                let staging = checkpoint::packed::BoundedWorkspace::new(checkpoint::packed::MemoryBudget::new(crate::fp8_source::WEIGHT_STAGING_BYTES, 0));
                let ops = Ops::new(head_gpu.share()).map_err(|e| format!("{MODEL}: Ops::new for the MTP projection: {e}"))?;
                let fc = model::ops::upload_preserved(&ops, src, "mtp.fc.weight", d, 2 * d, &staging).map_err(|e| format!("{MODEL}: {e}"))?;
                let vec = |name: &str| -> Result<Vec<f32>, String> {
                    let mut out = Vec::with_capacity(d);
                    src.with_tensor_chunks(name, 1 << 20, &mut |_, chunk| out.extend_from_slice(chunk)).then_some(()).ok_or_else(|| format!("{MODEL}: '{name}' is missing"))?;
                    exact(name, out)
                };
                Ok(Some(MtpParts { cfg1, shard, stage, fc, enorm: vec("mtp.pre_fc_norm_embedding.weight")?, hnorm: vec("mtp.pre_fc_norm_hidden.weight")?, head_norm_host: vec("mtp.norm.weight")? }))
            }
        }
    }
}

/// The MTP layer as a stage of its own: the full-attention layer of a config truncated to one attention period.
fn mtp_stage_shape(cfg: &Qwen35Config, gpu_index: usize) -> (Qwen35Config, Shard) {
    let mut cfg1 = cfg.clone();
    cfg1.n_layers = cfg.full_attention_interval;
    cfg1.mtp = false;
    let layer = (cfg.full_attention_interval - 1) as usize;
    (cfg1, Shard { start: layer, end: layer + 1, embed: false, head: false, gpu_index })
}

/// The MTP block's `eh_proj`: the tensor whose presence says the checkpoint
/// carries a multi-token-prediction head.
fn mtp_projection_name(cfg: &Qwen35Config) -> String {
    format!("blk.{}.nextn.eh_proj.weight", cfg.n_layers)
}

/// The MTP block as one stage: its one full-attention layer is that layer
/// of a truncated config, read from `blk.{n_layers}.*` under that layer's names.
pub(crate) struct MtpSource<'a> {
    pub cfg1: Qwen35Config,
    pub shard: Shard,
    pub src: GdnFixSource<'a>,
}

pub(crate) fn mtp_source<'a>(mg: &'a MmapGguf, cfg: &Qwen35Config, gpu_index: usize) -> Result<MtpSource<'a>, String> {
    let prefix = format!("blk.{}.", cfg.n_layers);
    let (cfg1, shard) = mtp_stage_shape(cfg, gpu_index);
    let layer = shard.start;
    let mut plan = HashMap::new();
    for name in mg.names() {
        let Some(leaf) = name.strip_prefix(&prefix) else { continue };
        if leaf.starts_with("nextn.") {
            continue;
        }
        let suffix = gguf::leaf::role(leaf)
            .and_then(|role| block_leaf_suffix(role, LayerType::Full))
            .ok_or_else(|| format!("{MODEL}: unrecognized leaf '{leaf}' in the MTP block"))?;
        plan.insert(format!("blocks.{layer}.{suffix}"), Fetch::Whole(name.clone()));
    }
    let src = RemapSource::new(mg, plan);
    src.validate(&crate::model::shard_param_list(&cfg1, &shard))?;
    let src = GdnFixSource::new(src, cfg1.gdn_head_order());
    Ok(MtpSource { cfg1, shard, src })
}

/// The MTP layer's leaves are named `mtp.layers.0.*` in a weight-tier policy
/// (what the placement charges) and `blocks.{layer}.*` in its stage: the
/// policy for the stage, each leaf at the tier its `mtp.layers.0.` name wants.
fn stage_tier(cfg: &Qwen35Config, tier: &TierPolicy, layer: usize) -> TierPolicy {
    cfg.layer_leaves(LayerType::Full)
        .into_iter()
        .fold(TierPolicy::uniform(tier.default_dtype()), |p, (suffix, ..)| p.with(&[&format!("blocks.{layer}.{suffix}")], tier.want(&format!("{MTP_PREFIX}{suffix}"))))
}

/// How a weight-tier policy names the MTP layer's leaves.
pub(crate) const MTP_PREFIX: &str = "mtp.layers.0.";

/// Whether the checkpoint carries an MTP block - what a plan charges for the
/// head before anything is loaded.
pub(crate) fn checkpoint_has_mtp(mg: &MmapGguf, cfg: &Qwen35Config) -> bool {
    mg.shape(&mtp_projection_name(cfg)).is_some()
}

impl Qwen35GgufInstance {
    /// Load the checkpoint's MTP head if it has one (and `mode` wants it).
    pub(super) fn enable_mtp(&mut self, tier: &TierPolicy, gpu_index: usize, normed_hidden: bool) -> Result<bool, String> {
        let (_, target_norm, _) = self.ckpt.endpoint_names(&self.cfg)?;
        let head_gpu = &self.shards.last().expect("a plan always has at least one stage").qwen35.gpu;
        let loaded = MtpDrafter::load(&self.ckpt, &self.cfg, tier, gpu_index, head_gpu, self.cap, self.state_rows(), self.kv_tier, self.kv_layout, self.unified, &target_norm, normed_hidden)?;
        self.mtp = loaded;
        Ok(self.mtp.is_some())
    }

    /// Whether the MTP head is loaded.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Wall time in the MTP head's host prep and projection, its layer, and the
    /// shared head's argmax (for the benchmark).
    pub fn mtp_times(&self) -> [std::time::Duration; 3] {
        self.mtp.as_ref().map_or([std::time::Duration::ZERO; 3], |m| m.times)
    }

    /// Head forwards run so far (for the benchmark).
    pub fn mtp_forwards(&self) -> u64 {
        self.mtp.as_ref().map_or(0, |m| m.forwards)
    }

    /// A slot's prompt was prefilled (`start..` of it just now; `start > 0`
    /// continued a cached prefix whose last hidden state was `prev_hidden`):
    /// run the head over every complete pair of the prompt and keep the last
    /// row pending.
    /// `true` when every pair of the prompt reached the head's KV (the rows a
    /// later prompt may resume from); `false` when the head was absent or the feed
    /// stopped early.
    pub(super) fn mtp_after_prefill(&mut self, slot: u32, prompt: &[u32], start: usize, prev_hidden: Option<&[f32]>, carry: Vec<HiddenRows>) -> bool {
        match self.mtp_begin_feed(slot, prompt, start, prev_hidden, carry) {
            FeedStart::Unavailable => false,
            FeedStart::Complete => true,
            FeedStart::Pending(mut feed) => loop {
                match self.mtp_advance_feed(&mut feed, usize::MAX) {
                    FeedStep::More => {}
                    FeedStep::Done => break true,
                    FeedStep::Failed => break false,
                }
            },
        }
    }

    /// Start [`Self::mtp_after_prefill`]'s work without doing the head's forwards: the slot's state is set as the
    /// finished feed leaves it, and the pairs still to run are handed back as a [`PromptFeed`] for
    /// [`Self::mtp_advance_feed`] to run a few rounds at a time. Until it is done the slot does not draft (a draft
    /// would attend over rows the feed has not written) and its decode steps are not recorded: the sequence
    /// resumes drafting from the first step after the feed, as one that was not fed does.
    pub(super) fn mtp_begin_feed(&mut self, slot: u32, prompt: &[u32], start: usize, prev_hidden: Option<&[f32]>, carry: Vec<HiddenRows>) -> FeedStart {
        let Self { mtp, shards, .. } = self;
        let Some(m) = mtp.as_mut() else { return FeedStart::Unavailable };
        let d = m.d;
        let (first, blocks): (usize, Vec<HiddenRows>) = match (start, prev_hidden) {
            (0, _) => (0, carry),
            (s, Some(prev)) => (s - 1, std::iter::once(HiddenRows::Host(prev.to_vec())).chain(carry).collect()),
            (_, None) => return FeedStart::Unavailable, // no hidden state for the cached prefix's last row: this slot drafts from scratch
        };
        let run = HiddenRun::new(d, blocks);
        let len = prompt.len();
        debug_assert_eq!(run.rows(), len - first);
        m.slots.insert(slot, SlotState { fed: (len - 1) as u32, toks: vec![prompt[len - 1]] });
        let head_gpu = &shards.last().expect("a stage").qwen35.gpu;
        m.write_pending(slot, 0, &run.row_host(head_gpu, len - 1 - first));
        // Complete pairs: positions first..len-1 pair with prompt[j + 1].
        if first + 1 >= len {
            return FeedStart::Complete;
        }
        let n = len - 1 - first;
        m.feeding.insert(slot);
        FeedStart::Pending(PromptFeed { slot, first, done: 0, tokens: prompt[first + 1..first + 1 + n].to_vec(), hidden: run })
    }

    /// Run up to `rows` more of `feed`'s pairs through the head.
    pub(super) fn mtp_advance_feed(&mut self, feed: &mut PromptFeed, rows: usize) -> FeedStep {
        if self.mtp.is_none() {
            return FeedStep::Failed;
        }
        let end = feed.tokens.len().min(feed.done.saturating_add(rows));
        let run = self.mtp_feed_run(feed.slot, (feed.first + feed.done) as u32, &feed.tokens[feed.done..end], &feed.hidden, feed.done);
        let m = self.mtp.as_mut().expect("checked above");
        if let Err(e) = run {
            eprintln!("{MODEL}: MTP prefill failed, this sequence drafts from scratch: {e}");
            m.feeding.remove(&feed.slot);
            let len = feed.first + feed.tokens.len() + 1;
            m.slots.insert(feed.slot, SlotState { fed: len as u32 - 1, ..SlotState::default() });
            return FeedStep::Failed;
        }
        feed.done = end;
        if end < feed.tokens.len() {
            return FeedStep::More;
        }
        m.feeding.remove(&feed.slot);
        FeedStep::Done
    }

    /// A prompt was prefilled without feeding the head (it would not pay): the slot
    /// has no draft state, and drafts from scratch if speculation ever reaches it.
    pub(super) fn mtp_skip_prefill(&mut self, slot: u32) {
        if let Some(m) = self.mtp.as_mut() {
            m.slots.remove(&slot);
            m.feeding.remove(&slot);
        }
    }

    /// A cached prefix of `rows` tokens was copied into the window `to` from `from`: its
    /// head KV follows (positions `0..rows - 1` are pure functions of the
    /// prefix).
    pub(super) fn mtp_seed_prefix(&mut self, from: KvWindow<'_>, to: KvWindow<'_>, rows: u32) {
        let Self { mtp, cfg, .. } = self;
        let Some(m) = mtp.as_ref() else { return };
        if rows > 1 {
            m.caches.copy_kv_rows(&m.stage.gpu, cfg, from, to, rows - 1);
        }
    }

    /// [`Self::copy_block_rows`] for the head's own KV.
    pub(super) fn mtp_copy_block_rows(&self, from: u32, to: u32, rows: u32) {
        if let Some(m) = self.mtp.as_ref() {
            m.caches.copy_kv_rows(&m.stage.gpu, &self.cfg, KvWindow::Table(&[from]), KvWindow::Table(&[to]), rows);
        }
    }

    /// The target is about to consume one token in each of `slots` (`inputs`) and leave the `[n, d]` hidden block in a
    /// device buffer: where each row goes in the pending store (`u32::MAX`: nowhere, a slot that is being fed). The
    /// tokens are noted now; the caller writes the rows ([`Self::mtp_tap`]) or has the step's recording do it.
    pub(super) fn mtp_note_step(&mut self, slots: &[u32], inputs: &[u32]) -> Vec<u32> {
        let Some(m) = self.mtp.as_mut() else { return Vec::new() };
        slots
            .iter()
            .zip(inputs)
            .map(|(&slot, &tok)| {
                if m.feeding.contains(&slot) {
                    return u32::MAX;
                }
                let held = m.slots.get(&slot).map_or(0, SlotState::rows);
                let at = m.pending_row(slot, held);
                m.slots.entry(slot).or_default().toks.push(tok);
                at
            })
            .collect()
    }

    /// Where the rows of a verify step land in the pending store: sequence `i` has `rows[i].len()` rows from the first
    /// free row of its span. Nothing is noted yet: [`Self::mtp_keep_verified`] says how many of them stand.
    pub(super) fn mtp_verify_destinations(&self, slots: &[u32], counts: &[u32]) -> Vec<u32> {
        let Some(m) = self.mtp.as_ref() else { return Vec::new() };
        let mut out = Vec::with_capacity(counts.iter().sum::<u32>() as usize);
        for (&slot, &n) in slots.iter().zip(counts) {
            let held = if m.feeding.contains(&slot) { None } else { Some(m.slots.get(&slot).map_or(0, SlotState::rows)) };
            out.extend((0..n as usize).map(|r| held.map_or(u32::MAX, |h| m.pending_row(slot, h + r))));
        }
        out
    }

    /// The dispatch that writes `n` rows of `hidden` to the pending rows in `idx` (a recorded step ends in it).
    pub(super) fn mtp_tap_step(&self, g: &Gpu, idx: &DeviceBuffer, hidden: &DeviceBuffer, n: u32) -> gpu_core::Step {
        self.mtp.as_ref().expect("a tapped step has a head").tap_step(g, idx, hidden, n)
    }

    /// Write a device block of hidden rows into the pending store at `destinations` (one per row).
    pub(super) fn mtp_tap(&self, hidden: &DeviceBuffer, destinations: &[u32]) {
        let Some(m) = self.mtp.as_ref() else { return };
        let g = &self.shards.last().expect("a stage").qwen35.gpu;
        let idx = g.storage(destinations.len() as u64);
        g.write(&idx, destinations);
        g.submit(&[], &[m.tap_step(g, &idx, hidden, destinations.len() as u32)]);
    }

    /// [`Self::mtp_tap`] for rows on the host (a mixed forward hands its decode rows back that way): `destinations[i]`
    /// is where row `i` of `hidden` (`[n, d]`) goes.
    pub(super) fn mtp_tap_host(&self, hidden: &[f32], destinations: &[u32]) {
        let Some(m) = self.mtp.as_ref() else { return };
        let d = m.d;
        for (i, &dst) in destinations.iter().enumerate().filter(|(_, &dst)| dst != u32::MAX) {
            m.write_row(dst, &hidden[i * d..(i + 1) * d]);
        }
    }

    /// A verify step consumed `rows[i]` for each sequence and kept the first `keep[i]` of them (their hidden rows were
    /// written at [`Self::mtp_verify_destinations`]).
    pub(super) fn mtp_keep_verified(&mut self, slots: &[u32], rows: &[Vec<u32>], keep: &[usize]) {
        let Some(m) = self.mtp.as_mut() else { return };
        for (i, &slot) in slots.iter().enumerate() {
            if !m.feeding.contains(&slot) {
                m.slots.entry(slot).or_default().toks.extend_from_slice(&rows[i][..keep[i]]);
            }
        }
        self.mtp_flush_long(slots);
    }

    /// The step noted by [`Self::mtp_note_step`] is done: feed any slot that outgrew [`MAX_PENDING_ROWS`].
    pub(super) fn mtp_step_done(&mut self, slots: &[u32]) {
        self.mtp_flush_long(slots);
    }

    /// Feed any slot whose pending run outgrew [`MAX_PENDING_ROWS`] to the head
    /// (every complete pair; the last row stays pending).
    fn mtp_flush_long(&mut self, slots: &[u32]) {
        for &slot in slots {
            let Some(m) = self.mtp.as_ref() else { return };
            if m.feeding.contains(&slot) {
                continue;
            }
            let Some(st) = m.slots.get(&slot) else { continue };
            if st.rows() <= MAX_PENDING_ROWS {
                continue;
            }
            let (fed, n, rows) = (st.fed, st.rows() - 1, st.rows());
            let toks = st.toks[1..=n].to_vec();
            let head_gpu = &self.shards.last().expect("a stage").qwen35.gpu;
            let hidden = m.read_pending(head_gpu, slot, rows);
            let fed_ok = self.mtp_feed_run(slot, fed, &toks, &HiddenRun::new(m.d, vec![HiddenRows::Host(hidden[..n * m.d].to_vec())]), 0);
            let m = self.mtp.as_mut().expect("checked above");
            match fed_ok {
                Ok(()) => {
                    // The last row stays pending, moved to the front of the span.
                    let last = hidden[n * m.d..].to_vec();
                    m.write_pending(slot, 0, &last);
                    let st = m.slots.get_mut(&slot).expect("present");
                    st.fed += n as u32;
                    st.toks.drain(..n);
                }
                Err(e) => {
                    eprintln!("{MODEL}: MTP catch-up failed, this sequence drafts from scratch: {e}");
                    m.slots.remove(&slot);
                }
            }
        }
    }

    /// Stop tracking these slots: a step decoded them without speculating (the
    /// batch was past the crossover), and keeping their hidden states would
    /// cost a head forward per sequence every [`MAX_PENDING_ROWS`] steps.
    pub(super) fn mtp_forget(&mut self, slots: &[u32]) {
        if let Some(m) = self.mtp.as_mut() {
            for slot in slots {
                m.slots.remove(slot);
                m.feeding.remove(slot);
            }
        }
    }

    /// A slot's sequence ended.
    pub(super) fn mtp_release(&mut self, slot: u32) {
        if let Some(m) = self.mtp.as_mut() {
            m.slots.remove(&slot);
            m.feeding.remove(&slot);
        }
    }
}

impl MtpDrafter {
    /// Drop the recorded draft rounds.
    pub(super) fn clear_plans(&self) {
        self.draft_plans.borrow_mut().clear();
    }

    /// Row of the pending store holding row `row` of `slot`'s pending run, or `u32::MAX` past the slot's span (a slot
    /// the instance was not sized for is not tracked).
    fn pending_row(&self, slot: u32, row: usize) -> u32 {
        if slot < self.pend_slots && row < PENDING_STRIDE {
            slot * PENDING_STRIDE as u32 + row as u32
        } else {
            u32::MAX
        }
    }

    /// The step that writes `n` rows of `src` to the pending-store rows named by `idx`.
    fn tap_step(&self, g: &Gpu, idx: &DeviceBuffer, src: &DeviceBuffer, n: u32) -> gpu_core::Step {
        let kernel = g.kernel_index("row_scatter").expect("row_scatter is registered with the qwen35 pipelines");
        let rows = self.pend_slots * PENDING_STRIDE as u32;
        g.step(kernel, &[idx, src, &self.pend], &[n, self.d as u32, rows], n * self.d as u32)
    }

    /// Write one hidden row to a row of the pending store.
    fn write_row(&self, row: u32, hidden: &[f32]) {
        self.stage.gpu.write_f32_at(&self.pend, row as u64 * self.d as u64, hidden);
    }

    /// Write `hidden` (`[rows, d]`) to the rows of `slot`'s span from `first`.
    fn write_pending(&self, slot: u32, first: usize, hidden: &[f32]) {
        let at = self.pending_row(slot, first);
        if at != u32::MAX {
            self.stage.gpu.write_f32_at(&self.pend, at as u64 * self.d as u64, hidden);
        }
    }

    /// The first `rows` pending rows of `slot`, on the host.
    fn read_pending(&self, g: &Gpu, slot: u32, rows: usize) -> Vec<f32> {
        let idx = g.storage(rows as u64);
        g.write(&idx, &(0..rows).map(|r| self.pending_row(slot, r)).collect::<Vec<_>>());
        let out = g.storage((rows * self.d) as u64);
        g.submit(&[], &[Self::gather_rows_step(g, &idx, &self.pend, &out, rows as u32, self.d as u32)]);
        g.read(&out, rows * self.d)
    }
}
