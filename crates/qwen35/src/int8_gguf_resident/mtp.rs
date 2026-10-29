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
use crate::gguf_import::block_leaf_suffix;
use model::ops::{Ops, Weight};

/// Hidden rows a slot may hold awaiting a draft before they are fed to the
/// head anyway (bounds host memory while speculation is off for a while).
const MAX_PENDING_ROWS: usize = 16;

/// Rows one head forward carries; a longer catch-up is split by sequence.
const MAX_BATCH_ROWS: usize = 128;

/// Prompt rows fed to the head per prefill round.
const PREFILL_ROUND_ROWS: usize = 128;
/// Rows of a prompt's feed whose host side is prepared at once (a multiple of [`PREFILL_ROUND_ROWS`]).
const FEED_PREP_ROWS: usize = 8 * PREFILL_ROUND_ROWS;

/// Scratch arena of the `eh_proj` projection.
const FC_ARENA: &str = "qwen35.mtp.fc";

/// What the head has seen of one sequence, and what the target has produced
/// that it has not.
#[derive(Default)]
struct SlotState {
    /// Head positions `0..fed` are in its KV cache (positions `j` whose pair
    /// `(h_j, x_{j+1})` is complete and was run).
    fed: u32,
    /// `[rows, d]`: the target's hidden state at positions `fed..fed+rows`.
    hidden: Vec<f32>,
    /// `[rows]`: the token the target consumed AT each of those positions. The
    /// pair for row `r` is `(hidden[r], toks[r + 1])`; the last row's partner
    /// is the sequence's next input, which only the draft call knows.
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
    /// `[pairs, d]`: the target's hidden state of each pair.
    hidden: Vec<f32>,
}

impl PromptFeed {
    pub(super) fn slot(&self) -> u32 {
        self.slot
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
    ops: Ops,
    fc: Weight,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    /// `shared_head_norm` on the head's card, for the shared `lm_head`.
    head_norm: DeviceBuffer,
    head_norm_host: Vec<f32>,
    /// The target's `output_norm` (host copy), for the convention that hands the
    /// head the NORMED target hidden state.
    target_norm_host: Vec<f32>,
    /// Hand the head `rmsnorm(h, output_norm)` instead of the raw residual.
    normed_hidden: bool,
    eps: f32,
    d: usize,
    /// Embedding rows, read from this handle's own mapping.
    mg: MmapGguf,
    embed: EmbedTable,
    slots: std::collections::HashMap<u32, SlotState>,
    /// Slots whose prompt feed is begun and not finished ([`PromptFeed`]): they neither draft nor record steps.
    feeding: std::collections::HashSet<u32>,
    /// Forwards run, for the benchmark.
    pub(super) forwards: u64,
    /// Wall time of the pieces of a forward: host prep and projection, the layer
    /// stage, the shared head - for the benchmark.
    pub(super) times: [std::time::Duration; 3],
}

/// `row * rsqrt(mean(row^2) + eps) * w`, written to `out`.
fn rms_row_into(row: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = row.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(row).zip(w) {
        *o = v * r * w;
    }
}

/// Rows a host thread prepares at least, below which splitting the work costs more than it saves.
const MIN_ROWS_PER_THREAD: usize = 32;

impl MtpDrafter {
    /// `Ok(None)` when the checkpoint carries no MTP block.
    #[allow(clippy::too_many_arguments)]
    fn load(
        gguf_path: &str,
        mg: &MmapGguf,
        cfg: &Qwen35Config,
        tier: &TierPolicy,
        gpu_index: usize,
        head_gpu: &Gpu,
        cap: u32,
        slots: u32,
        kv_tier: KvTier,
        layout: KvLayout,
        target_norm_name: &str,
        embed_name: &str,
        normed_hidden: bool,
    ) -> Result<Option<MtpDrafter>, String> {
        let prefix = format!("blk.{}.", cfg.n_layers);
        let fc_name = mtp_projection_name(cfg);
        let d = cfg.d_model as usize;
        match mg.shape(&fc_name) {
            None => return Ok(None),
            Some(s) if s == [d, 2 * d] => {}
            Some(s) => return Err(format!("{MODEL}: {fc_name} has shape {s:?}, expected [{d}, {}]", 2 * d)),
        }
        let MtpSource { cfg1, shard, src } = mtp_source(mg, cfg, gpu_index)?;
        let stage = Qwen35::new_shard_dt(cfg1.clone(), 1, 1, &src, shard.clone(), &stage_tier(cfg, tier, shard.start));
        let caches = ShardCaches::new(&stage.gpu, &cfg1, &shard, cap, slots, 0, kv_tier, layout);

        let (w, s) = model::int8::upload_quantized(&mut paramstore::upload::Uploader::new(head_gpu), mg, &fc_name, d, 2 * d).map_err(|e| format!("{MODEL}: {e}"))?;
        let fc = Weight::I8 { w, s, n: d as u32, k: 2 * d as u32 };
        let vec = |name: String| -> Result<Vec<f32>, String> {
            match mg.tensor(&name) {
                Some(Ok(v)) if v.len() == d => Ok(v),
                Some(Ok(v)) => Err(format!("{MODEL}: '{name}' has {} elements, expected {d}", v.len())),
                Some(Err(e)) => Err(format!("{MODEL}: '{name}': {e}")),
                None => Err(format!("{MODEL}: '{name}' is missing")),
            }
        };
        let head_norm_host = vec(format!("{prefix}nextn.shared_head_norm.weight"))?;
        Ok(Some(MtpDrafter {
            stage,
            caches,
            ops: Ops::new(head_gpu.share()).map_err(|e| format!("{MODEL}: Ops::new for the MTP projection: {e}"))?,
            fc,
            enorm: vec(format!("{prefix}nextn.enorm.weight"))?,
            hnorm: vec(format!("{prefix}nextn.hnorm.weight"))?,
            head_norm: head_gpu.storage_init("qwen35.gguf.mtp_norm", &head_norm_host),
            head_norm_host,
            target_norm_host: vec(target_norm_name.to_string())?,
            normed_hidden,
            eps: cfg.rms_eps,
            d,
            mg: MmapGguf::open(gguf_path).map_err(|e| format!("{MODEL}: cannot reopen '{gguf_path}': {e}"))?,
            embed: EmbedTable { name: embed_name.to_string(), d },
            slots: std::collections::HashMap::new(),
            feeding: std::collections::HashSet::new(),
            forwards: 0,
            times: Default::default(),
        }))
    }

    fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let rows: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        self.mg.tensor_rows(&self.embed.name, &rows, self.d).ok_or_else(|| {
            let bad = tokens.iter().find(|&&t| self.mg.tensor_range(&self.embed.name, t as usize * self.d, self.d).is_none()).copied().unwrap_or(0);
            format!("{MODEL}: token {bad} is outside the embedding")
        })?
    }

    /// The `[n, 2d]` input of `eh_proj` for rows pairing `hidden[i]` with `tokens[i]`. Rows are independent, so
    /// they are normed on several threads (a thousand-row prompt is about 22 ms of norms on one).
    fn fc_input(&self, tokens: &[u32], hidden: &[f32], hidden_is_target: bool) -> Result<Vec<f32>, String> {
        let d = self.d;
        let embedded = self.embed_rows(tokens)?;
        let hidden_norm: Option<&[f32]> = self.normed_hidden.then(|| if hidden_is_target { &self.target_norm_host[..] } else { &self.head_norm_host[..] });
        let (enorm, hnorm, eps) = (&self.enorm[..], &self.hnorm[..], self.eps);
        let mut x = vec![0.0f32; tokens.len() * 2 * d];
        let rows_per_thread = tokens.len().div_ceil(std::thread::available_parallelism().map_or(1, |n| n.get()).min(16)).max(MIN_ROWS_PER_THREAD);
        std::thread::scope(|scope| {
            for (i, block) in x.chunks_mut(rows_per_thread * 2 * d).enumerate() {
                let first = i * rows_per_thread;
                let (embedded, hidden) = (&embedded, hidden);
                scope.spawn(move || {
                    let mut normed = vec![0.0f32; d];
                    for (r, out) in block.chunks_exact_mut(2 * d).enumerate() {
                        let row = first + r;
                        let (e, h) = out.split_at_mut(d);
                        rms_row_into(&embedded[row * d..(row + 1) * d], enorm, eps, e);
                        let h_in = &hidden[row * d..(row + 1) * d];
                        match hidden_norm {
                            Some(norm) => {
                                rms_row_into(h_in, norm, eps, &mut normed);
                                rms_row_into(&normed, hnorm, eps, h);
                            }
                            None => rms_row_into(h_in, hnorm, eps, h),
                        }
                    }
                });
            }
        });
        Ok(x)
    }

    /// `eh_proj` over `[n, 2d]` rows on the head's card.
    fn project(&self, head_gpu: &Gpu, x: &[f32]) -> Vec<f32> {
        let n = (x.len() / (2 * self.d)) as u32;
        let _ops_scratch = self.ops.gpu().scratch_scope_in(FC_ARENA);
        let _scratch = head_gpu.scratch_scope_in(FC_ARENA);
        let _pass = head_gpu.pass_scope();
        let xd = head_gpu.storage(x.len() as u64);
        head_gpu.write_f32(&xd, x);
        let mut s = Vec::new();
        let act = self.ops.act(&mut s, &xd, 0, n, 2 * self.d as u32);
        let out = head_gpu.storage(n as u64 * self.d as u64);
        self.ops.matmul(&mut s, &self.fc, &act, &out, 0);
        head_gpu.submit(&[], &s);
        head_gpu.read(&out, n as usize * self.d)
    }

    /// One head forward over packed rows: row `i` pairs `hidden[i]` with
    /// `tokens[i]` at head position `positions[i]` of slot `slots[i]`.
    /// Returns the `[n, d]` block output.
    fn forward(&mut self, head_gpu: &Gpu, slots: &[u32], positions: &[u32], tokens: &[u32], hidden: &[f32], hidden_is_target: bool) -> Result<Vec<f32>, String> {
        self.forwards += 1;
        let t0 = std::time::Instant::now();
        let x = self.fc_input(tokens, hidden, hidden_is_target)?;
        let ehp = self.project(head_gpu, &x);
        let t1 = std::time::Instant::now();
        let seqs = self.caches.batch(slots, positions);
        let caches = BatchDecodeCaches { gqa_kv: &self.caches.gqa_kv, gqa_cap: self.caches.kv_block(), seqs: &seqs, one_sequence: false, gdn_pool: None, memory: &[] };
        let out = self.stage.decode_batch_stage(tokens, &caches, Some(&ehp));
        self.times[0] += t1 - t0;
        self.times[1] += t1.elapsed();
        Ok(out)
    }

    /// Feed one sequence's run of complete pairs through the chunked path: head positions
    /// `from..from + tokens.len()`, in rounds of [`PREFILL_ROUND_ROWS`]. The host side of the run (the
    /// embedding rows, the norms) is prepared [`FEED_PREP_ROWS`] at a time, so the CPU pool is entered once per
    /// several rounds and not once per round, while the prepared rows stay bounded for a long prompt.
    fn feed_run(&mut self, head_gpu: &Gpu, slot: u32, from: u32, tokens: &[u32], hidden: &[f32], hidden_is_target: bool) -> Result<(), String> {
        let d = self.d;
        let view = self.caches.view(slot);
        for (g, group) in tokens.chunks(FEED_PREP_ROWS).enumerate() {
            let at = g * FEED_PREP_ROWS;
            let x = self.fc_input(group, &hidden[at * d..(at + group.len()) * d], hidden_is_target)?;
            for (i, (round, x)) in group.chunks(PREFILL_ROUND_ROWS).zip(x.chunks(PREFILL_ROUND_ROWS * 2 * d)).enumerate() {
                self.forwards += 1;
                let ehp = self.project(head_gpu, x);
                self.stage.prefill_chunk_stage(round, from + (at + i * PREFILL_ROUND_ROWS) as u32, &view, Some(&ehp));
            }
        }
        Ok(())
    }
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
    let mut cfg1 = cfg.clone();
    cfg1.n_layers = cfg.full_attention_interval;
    cfg1.mtp = false;
    let layer = (cfg.full_attention_interval - 1) as usize;
    let shard = Shard { start: layer, end: layer + 1, embed: false, head: false, gpu_index };
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
    pub(super) fn enable_mtp(&mut self, gguf_path: &str, tier: &TierPolicy, gpu_index: usize, normed_hidden: bool) -> Result<bool, String> {
        let (target_norm, embed_name) = {
            let names = endpoint_names(&self.mg, &self.cfg)?;
            (names.1, self.embed.name.clone())
        };
        let head_gpu = &self.shards.last().expect("a plan always has at least one stage").qwen35.gpu;
        let loaded = MtpDrafter::load(gguf_path, &self.mg, &self.cfg, tier, gpu_index, head_gpu, self.cap, self.state_rows(), self.kv_tier, self.kv_layout, &target_norm, &embed_name, normed_hidden)?;
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
    pub(super) fn mtp_after_prefill(&mut self, slot: u32, prompt: &[u32], start: usize, prev_hidden: Option<&[f32]>, carry: &[f32]) -> bool {
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
    pub(super) fn mtp_begin_feed(&mut self, slot: u32, prompt: &[u32], start: usize, prev_hidden: Option<&[f32]>, carry: &[f32]) -> FeedStart {
        let Self { mtp, .. } = self;
        let Some(m) = mtp.as_mut() else { return FeedStart::Unavailable };
        let d = m.d;
        let (first, hidden): (usize, Vec<f32>) = match (start, prev_hidden) {
            (0, _) => (0, carry.to_vec()),
            (s, Some(prev)) => (s - 1, prev.iter().chain(carry).copied().collect()),
            (_, None) => return FeedStart::Unavailable, // no hidden state for the cached prefix's last row: this slot drafts from scratch
        };
        let len = prompt.len();
        debug_assert_eq!(hidden.len(), (len - first) * d);
        m.slots.insert(slot, SlotState { fed: (len - 1) as u32, hidden: hidden[(len - 1 - first) * d..].to_vec(), toks: vec![prompt[len - 1]] });
        // Complete pairs: positions first..len-1 pair with prompt[j + 1].
        if first + 1 >= len {
            return FeedStart::Complete;
        }
        let n = len - 1 - first;
        m.feeding.insert(slot);
        FeedStart::Pending(PromptFeed { slot, first, done: 0, tokens: prompt[first + 1..first + 1 + n].to_vec(), hidden: hidden[..n * d].to_vec() })
    }

    /// Run up to `rows` more of `feed`'s pairs through the head.
    pub(super) fn mtp_advance_feed(&mut self, feed: &mut PromptFeed, rows: usize) -> FeedStep {
        let Self { mtp, shards, .. } = self;
        let Some(m) = mtp.as_mut() else { return FeedStep::Failed };
        let head_gpu = &shards.last().expect("a stage").qwen35.gpu;
        let d = m.d;
        let end = feed.tokens.len().min(feed.done.saturating_add(rows));
        let run = m.feed_run(head_gpu, feed.slot, (feed.first + feed.done) as u32, &feed.tokens[feed.done..end], &feed.hidden[feed.done * d..end * d], true);
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

    /// The target consumed one token in each of `slots` (`inputs`), producing the
    /// `[n, d]` hidden block `hidden`.
    pub(super) fn mtp_record_step(&mut self, slots: &[u32], inputs: &[u32], hidden: &[f32]) {
        let Some(m) = self.mtp.as_mut() else { return };
        let d = m.d;
        for (i, (&slot, &tok)) in slots.iter().zip(inputs).enumerate() {
            if m.feeding.contains(&slot) {
                continue;
            }
            let st = m.slots.entry(slot).or_default();
            st.hidden.extend_from_slice(&hidden[i * d..(i + 1) * d]);
            st.toks.push(tok);
        }
        self.mtp_flush_long(slots);
    }

    /// A verify step consumed `rows[i]` for each sequence and kept the first
    /// `keep[i]` of them; `hidden` is its packed `[sum(rows), d]` block.
    pub(super) fn mtp_record_verified(&mut self, slots: &[u32], rows: &[Vec<u32>], keep: &[usize], hidden: &[f32]) {
        let Some(m) = self.mtp.as_mut() else { return };
        let d = m.d;
        let mut at = 0usize;
        for (i, &slot) in slots.iter().enumerate() {
            if !m.feeding.contains(&slot) {
                let st = m.slots.entry(slot).or_default();
                st.hidden.extend_from_slice(&hidden[at * d..(at + keep[i]) * d]);
                st.toks.extend_from_slice(&rows[i][..keep[i]]);
            }
            at += rows[i].len();
        }
        self.mtp_flush_long(slots);
    }

    /// Feed any slot whose pending run outgrew [`MAX_PENDING_ROWS`] to the head
    /// (every complete pair; the last row stays pending).
    fn mtp_flush_long(&mut self, slots: &[u32]) {
        let Self { mtp, shards, .. } = self;
        let Some(m) = mtp.as_mut() else { return };
        let head_gpu = &shards.last().expect("a stage").qwen35.gpu;
        for &slot in slots {
            if m.feeding.contains(&slot) {
                continue;
            }
            let st = m.slots.get(&slot).expect("just recorded");
            if st.rows() <= MAX_PENDING_ROWS {
                continue;
            }
            let (fed, n) = (st.fed, st.rows() - 1);
            let (toks, hidden) = (st.toks[1..=n].to_vec(), st.hidden[..n * m.d].to_vec());
            match m.feed_run(head_gpu, slot, fed, &toks, &hidden, true) {
                Ok(()) => {
                    let st = m.slots.get_mut(&slot).expect("present");
                    st.fed += n as u32;
                    st.hidden.drain(..n * m.d);
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

    /// Propose up to `ks[i]` tokens after `next_inputs[i]` for the sequence in
    /// `slots[i]`, which has consumed `consumed[i]` positions.
    pub(super) fn mtp_draft(&mut self, slots: &[u32], consumed: &[u32], next_inputs: &[u32], ks: &[usize]) -> Vec<Vec<u32>> {
        let n = slots.len();
        let mut out = vec![Vec::new(); n];
        let Self { mtp, shards, head, cfg, .. } = self;
        let Some(m) = mtp.as_mut() else { return out };
        let last = shards.last().expect("a stage");
        let head_gpu = &last.qwen35.gpu;
        let d = m.d;

        // Sequences that can draft: their pending run covers exactly the positions
        // the target consumed. One whose does not (it was not tracked while
        // speculation was off for a big batch) starts over from here: it drafts
        // again once a step has given the head a fresh row to continue from.
        for i in (0..n).filter(|&i| !m.feeding.contains(&slots[i])) {
            let in_step = m.slots.get(&slots[i]).is_some_and(|st| st.fed + st.rows() as u32 == consumed[i] && st.hidden.len() == st.rows() * d);
            if !in_step {
                m.slots.insert(slots[i], SlotState { fed: consumed[i], ..SlotState::default() });
            }
        }
        let live: Vec<usize> = (0..n).filter(|&i| ks[i] > 0 && !m.feeding.contains(&slots[i]) && m.slots.get(&slots[i]).is_some_and(|st| st.rows() > 0)).collect();
        if live.is_empty() {
            return out;
        }

        // Catch-up: every pending pair, plus the pair that ends at the next input.
        let (mut s_slots, mut s_pos, mut s_toks, mut s_hidden) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut last_row = Vec::with_capacity(live.len());
        for &i in &live {
            let st = &m.slots[&slots[i]];
            let r = st.rows();
            for j in 0..r {
                s_slots.push(slots[i]);
                s_pos.push(st.fed + j as u32);
                s_toks.push(if j + 1 < r { st.toks[j + 1] } else { next_inputs[i] });
            }
            s_hidden.extend_from_slice(&st.hidden);
            last_row.push(s_slots.len() - 1);
        }
        // One forward per group of whole sequences (a sequence's rows must share
        // a launch so they see each other), at most MAX_BATCH_ROWS rows each.
        let mut block = Vec::with_capacity(s_slots.len() * d);
        let mut from = 0;
        for q in 0..live.len() {
            let end = last_row[q] + 1;
            let next_end = last_row.get(q + 1).map(|&l| l + 1);
            if next_end.is_some_and(|ne| ne - from <= MAX_BATCH_ROWS) {
                continue;
            }
            match m.forward(head_gpu, &s_slots[from..end], &s_pos[from..end], &s_toks[from..end], &s_hidden[from * d..end * d], true) {
                Ok(b) => block.extend(b),
                Err(e) => {
                    eprintln!("{MODEL}: MTP draft failed: {e}");
                    return out;
                }
            }
            from = end;
        }
        let mut prev: Vec<Vec<f32>> = live.iter().enumerate().map(|(q, _)| block[last_row[q] * d..(last_row[q] + 1) * d].to_vec()).collect();
        for &i in &live {
            let st = m.slots.get_mut(&slots[i]).expect("live slot");
            st.fed = consumed[i];
            st.hidden.clear();
            st.toks.clear();
        }
        let head_ops = &head.ops;
        let head_time = std::cell::Cell::new(std::time::Duration::ZERO);
        let argmax_rows = |m: &MtpDrafter, rows: &[Vec<f32>]| -> Vec<u32> {
            let t = std::time::Instant::now();
            let flat: Vec<f32> = rows.iter().flatten().copied().collect();
            let out = crate::stream::head_argmax_rows_on(head_gpu, head_ops, cfg, &m.head_norm, &head.w, &flat, rows.len() as u32);
            head_time.set(head_time.get() + t.elapsed());
            out
        };
        let first = argmax_rows(m, &prev);
        for (q, &i) in live.iter().enumerate() {
            out[i].push(first[q]);
        }

        // Extend: proposal `step` pairs the head's own output with the previous proposal.
        let kmax = live.iter().map(|&i| ks[i]).max().unwrap_or(0);
        for step in 2..=kmax {
            let act: Vec<usize> = (0..live.len()).filter(|&q| ks[live[q]] >= step).collect();
            if act.is_empty() {
                break;
            }
            let slots_a: Vec<u32> = act.iter().map(|&q| slots[live[q]]).collect();
            let pos_a: Vec<u32> = act.iter().map(|&q| consumed[live[q]] + step as u32 - 2).collect();
            let toks_a: Vec<u32> = act.iter().map(|&q| *out[live[q]].last().expect("a previous proposal")).collect();
            let hid_a: Vec<f32> = act.iter().flat_map(|&q| prev[q].iter().copied()).collect();
            let blk = match m.forward(head_gpu, &slots_a, &pos_a, &toks_a, &hid_a, false) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("{MODEL}: MTP draft failed: {e}");
                    break;
                }
            };
            let rows: Vec<Vec<f32>> = act.iter().enumerate().map(|(a, _)| blk[a * d..(a + 1) * d].to_vec()).collect();
            let next = argmax_rows(m, &rows);
            for (a, &q) in act.iter().enumerate() {
                out[live[q]].push(next[a]);
                prev[q] = rows[a].clone();
            }
        }
        m.times[2] += head_time.get();
        out
    }
}
