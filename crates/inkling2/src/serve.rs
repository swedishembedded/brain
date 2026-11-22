// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching serving engines for looped mixture-of-experts language models
// for its clients. If your team needs expertise in serving a model that revisits its layers, keeps sliding windows,
// paged history and convolution state per sequence, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The served Inkling 2 decoder: [`Engine`], a [`PagedDecoder`] over the same layer recipe as [`Inkling2::forward`],
//! with every visit's state in a [`VisitPool`].
//!
//! - **State.** Each of the plan's visits keeps its own state slot: a full visit's K/V in the pool's paged planes (all
//!   addressed by the scheduler's one block table per sequence), a sliding visit's K/V in a ring of exactly the window,
//!   and four convolution states (K and V before their convolution, the attention output, the MoE output) as the
//!   sequence's state row.
//! - **Prefill** runs a prompt in chunks of at most [`EngineOptions::prefill_chunk`] tokens: one pass of the recipe
//!   over the chunk's rows, attention through the pool ([`VisitPool::attend_chunk`]), convolutions continuing and
//!   leaving the sequence's state, the published expert banks multiplied grouped by expert. A prompt that carries
//!   embedding rows (images) has them spliced over its placeholder positions after the embedding norm, as the reference
//!   splices image features into `inputs_embeds`.
//! - **Decode** runs one new token of every running sequence as one pass of `b` rows: each row attends through its own
//!   table, the convolutions step each sequence's state, the banks gather per slot.
//! - **Head.** The logits are soft-capped and the padding rows of the unembedding floored on the device
//!   ([`PagedDecoder::head_transform`] declares it), greedy picks and candidates are reduced there
//!   ([`model::readout`]); a served row has the valid vocabulary's length.
//! - **Prefix reuse.** A prompt long enough is filed [`CHECKPOINT_TAIL`] tokens short of its end (knowledge #347: the
//!   next turn of a chat re-renders the last tokens of this one), keyed by its tokens and the content of its embedding
//!   rows; its entry carries the final normalised row of its last position.
//! - **Routing.** A sequence whose table carries a routing log gets every computed position's experts recorded for
//!   each of the plan's MoE calls, in visit order ([`PagedDecoder::routing_calls`]).

use std::path::Path;
use std::sync::Arc;

use gpu_core::{DeviceBuffer, Gpu};
use model::adapter::bank::{AdapterBank, AdapterServing, BankedEngine};
pub use model::adapter::bank::{AdapterBankConfig, AdapterError, AdapterInfo, UnloadStatus, VerifiedAdapter};
use model::adapter::sites::{ExpertSite, LoraSites};
use model::adapter::LinearSite;
use model::kv_tier::KvTier;
use model::ops::Ops;
use model::paged::{BlockAllocator, BlockTable, KvNamespace, BASE_KV_NAMESPACE};
use model::readout::{argmax_rows, topk_rows, ReadoutKernels};
use model::routing::RouterCallShape;
use model::serve::visit_state::{ConvState, VisitLayout, VisitPool, VisitPoolSizing, VisitStateShape};
use model::serve::{AdapterId, ContentKey, PagedDecoder, PrefillProgress, PromptEmbeds};
use model::snapshot_prefix::Prompt;

use crate::config::{Inkling2Config, LayerKind};
use crate::model::{Inkling2, Pool, Rows};
use crate::train::sites::{expert_sites, lora_sites, SiteKind};

/// Tokens before a prompt's end where its prefix entry is filed (knowledge #347).
pub const CHECKPOINT_TAIL: usize = 16;
/// Steps of a greedy decode window: one. Every step reads its tokens back before the next (the engine records no
/// device-side window), so a longer window would only delay streaming and cancellation.
const DECODE_WINDOW: usize = 1;
/// The most candidates a top-k readout returns per row.
const TOPK_CAP: usize = 64;

/// An [`Ops`] on the device the calling thread is scoped to (`residency::on_device`), registering every kernel the
/// model and the engine dispatch ([`crate::kernels`]).
pub fn device_ops() -> Result<Ops, String> {
    Ops::new(Gpu::new(&crate::kernels()))
}

/// How large the engine's state is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    /// The tier of every K/V row (paged planes and rings): f32 or bf16.
    pub kv_tier: KvTier,
    /// The pool: KV blocks of the full visits, live state rows (the batch) and snapshot rows (prefix entries).
    pub sizing: VisitPoolSizing,
    /// Tokens one sequence may hold (prompt and generation), never past the model's positions.
    pub max_seq_len: usize,
    /// Prompt rows of one prefill pass.
    pub prefill_chunk: u32,
}

/// The served decoder. See the module doc.
pub struct Engine {
    m: Inkling2,
    pool: Pool,
    readout: ReadoutKernels,
    max_seq_len: usize,
    prefill_chunk: u32,
    /// The adapters this engine serves beside the base ([`Engine::enable_adapters`]); `None` serves the base alone.
    bank: Option<AdapterBank>,
}

/// The adapter sites the served engine corrects: the trainer's linear sites ([`crate::train::sites`], each product
/// corrected after it) and its stacked expert sites (each routed bank product corrected after it). The merged sites
/// (depth and message queries, message-gate factors) are matrices an operator reads whole, not products a row's
/// adapter can correct, so an adapter that carries one is refused by name ([`Engine::load_adapter`]).
pub struct ServedSites<'a>(pub &'a Inkling2Config);

impl LoraSites for ServedSites<'_> {
    fn lora_sites(&self) -> Vec<LinearSite> {
        lora_sites(self.0).into_iter().filter(|(_, kind)| *kind == SiteKind::Linear).map(|(site, _)| site).collect()
    }
    fn expert_sites(&self) -> Vec<ExpertSite> {
        expert_sites(self.0)
    }
}

impl AdapterServing for ServedSites<'_> {
    fn attention_leaves(&self) -> Vec<&'static str> {
        vec!["wq_du", "wk_dv", "wv_dv", "wo_ud"]
    }
}

fn namespace(adapter: Option<AdapterId>) -> KvNamespace {
    adapter.map_or(BASE_KV_NAMESPACE, AdapterId::kv_namespace)
}

impl Engine {
    /// The per-sequence state layout of `cfg` with K/V rows at `kv_tier`: what a resident sizes its pool from.
    pub fn layout(cfg: &Inkling2Config, kv_tier: KvTier) -> Result<VisitLayout, String> {
        let attn = cfg.attn(0);
        if let Some(l) = (1..cfg.n_layers).find(|&l| (cfg.attn(l).kv_heads, cfg.attn(l).head_dim) != (attn.kv_heads, attn.head_dim)) {
            return Err(format!(
                "inkling2 serve: layer {l} keeps {} kv heads of {}, layer 0 {} of {}: the visit pool holds one K/V row shape",
                cfg.attn(l).kv_heads,
                cfg.attn(l).head_dim,
                attn.kv_heads,
                attn.head_dim
            ));
        }
        let taps = cfg.conv_taps;
        if taps < 2 {
            return Err(format!("inkling2 serve: a {taps}-tap convolution keeps no state; the engine serves the published 4-tap form"));
        }
        let kv = ConvState { channels: attn.kv_dim(), taps };
        let hidden = ConvState { channels: cfg.hidden, taps };
        let shape = VisitStateShape { n_kv_heads: attn.kv_heads, head_dim: attn.head_dim, kv_tier, conv: vec![kv, kv, hidden, hidden] };
        VisitLayout::new(crate::model::visit_plan(cfg)?, shape)
    }

    /// The engine over the model `m` at `opts` (on `m`'s device, whose `Gpu` registered [`crate::kernels`]).
    pub fn new(m: Inkling2, opts: EngineOptions) -> Result<Engine, String> {
        let cfg = m.config();
        if opts.max_seq_len == 0 || opts.prefill_chunk == 0 {
            return Err("inkling2 serve: a sequence and a prefill chunk hold at least one token".into());
        }
        if opts.max_seq_len as u64 > cfg.max_positions {
            return Err(format!("inkling2 serve: {} tokens per sequence exceed the model's {} positions", opts.max_seq_len, cfg.max_positions));
        }
        for l in 0..cfg.n_layers {
            if cfg.kind(l) == LayerKind::Sliding && cfg.window == 0 {
                return Err("inkling2 serve: a sliding layer of window 0".into());
            }
        }
        let layout = Self::layout(cfg, opts.kv_tier)?;
        let pool = VisitPool::new(m.ops().gpu(), layout, opts.sizing)?;
        let readout = ReadoutKernels::resolve(m.ops().gpu())?;
        Ok(Engine { m, pool, readout, max_seq_len: opts.max_seq_len, prefill_chunk: opts.prefill_chunk, bank: None })
    }

    /// Load the checkpoint in `dir` onto `ops` and serve it at `opts`.
    pub fn open(ops: Ops, dir: &Path, opts: EngineOptions) -> Result<Engine, String> {
        Self::new(Inkling2::load(ops, dir)?, opts)
    }

    pub fn model(&self) -> &Inkling2 {
        &self.m
    }

    /// The model back, the pool freed: an engine at other options can be built over the same uploaded weights.
    pub fn into_model(self) -> Inkling2 {
        self.m
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    fn g(&self) -> &Gpu {
        self.m.ops().gpu()
    }

    /// Serve LoRA adapters beside the base, in the same batches, from the one resident copy of the weights: a bank over
    /// [`ServedSites`] targeting `cfg.targets`, with room for the larger of a decode batch and a prefill chunk. Call once,
    /// before serving. Expert targets need the published microscaled banks (a dense f32 bank has no served expert
    /// product to correct). The engine registers the bank's kernels ([`crate::kernels`]).
    pub fn enable_adapters(&mut self, cfg: AdapterBankConfig) -> Result<(), AdapterError> {
        if self.bank.is_some() {
            return Err(AdapterError::BadConfig("the adapter bank is already enabled".into()));
        }
        let sites = ServedSites(self.m.config());
        let experts = sites.expert_sites();
        if experts.iter().any(|e| cfg.targets.iter().any(|t| t == e.leaf)) && !self.m.experts_are_banked() {
            return Err(AdapterError::BadConfig("inkling2: expert adapters are served over the published NVFP4/MXFP8 banks; this checkpoint stores its experts dense".into()));
        }
        let rows = (self.pool.max_batch() as u32).max(self.prefill_chunk);
        self.bank = Some(AdapterBank::new_with_experts(self.g(), &sites.lora_sites(), &experts, cfg, rows, 0..self.m.config().n_layers as usize)?);
        Ok(())
    }

    /// Register a verified adapter (host memory only; it reaches a slot when the scheduler first asks about it) and
    /// return the id the scheduler addresses it by. A pair for a merged site is refused by name.
    pub fn load_adapter(&mut self, adapter: VerifiedAdapter) -> Result<AdapterId, AdapterError> {
        let merged: Vec<String> = lora_sites(self.m.config()).into_iter().filter(|(_, kind)| *kind == SiteKind::Merged).map(|(site, _)| site.name).collect();
        if let Some(site) = adapter.adapter.sites.iter().find(|s| merged.contains(&s.base)) {
            return Err(AdapterError::Incompatible(format!(
                "{}: a merged site (a matrix the forward reads whole) is trained but not served; serve an adapter of the linear and expert sites",
                site.base
            )));
        }
        self.bank.as_mut().ok_or(AdapterError::NoBank)?.register(adapter)
    }

    /// Unload an adapter: free its slot and purge its cached prompts; while sequences still run under it, it only stops
    /// admitting new ones.
    pub fn unload_adapter(&mut self, id: AdapterId) -> Result<UnloadStatus, AdapterError> {
        match self.bank.as_mut().ok_or(AdapterError::NoBank)?.unload(id)? {
            Some(ns) => {
                self.pool.evict_namespace(ns);
                Ok(UnloadStatus::Unloaded)
            }
            None => Ok(UnloadStatus::Draining),
        }
    }

    /// Every registered adapter, oldest first.
    pub fn adapters(&self) -> Vec<AdapterInfo> {
        self.bank.as_ref().map_or_else(Vec::new, AdapterBank::infos)
    }

    /// Write the bank's row slots for a pass over `tables` (`rows_each` rows per sequence) and return the bank, or
    /// `None` without one (every sequence must then run under the base).
    fn bank_for(&self, tables: &[&BlockTable], rows_each: usize) -> Option<&AdapterBank> {
        let Some(bank) = &self.bank else {
            assert!(tables.iter().all(|t| t.kv_namespace() == BASE_KV_NAMESPACE), "inkling2 serve: a sequence runs under an adapter but this engine has no adapter bank");
            return None;
        };
        let slots: Vec<u32> = tables.iter().flat_map(|t| std::iter::repeat_n(bank.slot_of(t.kv_namespace()), rows_each)).collect();
        bank.write_slots(self.g(), &slots);
        Some(bank)
    }

    /// Where a prompt of `len` tokens is filed: short of its end by [`CHECKPOINT_TAIL`] when that leaves an entry
    /// worth keeping, else at its end.
    fn file_point(&self, len: usize) -> usize {
        if len >= CHECKPOINT_TAIL + model::serve::visit_state::MIN_CACHED_TOKENS {
            len - CHECKPOINT_TAIL
        } else {
            len
        }
    }

    /// One prefill pass of the prompt's next rows (resuming from the prefix cache on the first).
    fn prefill_rows(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: Option<&PromptEmbeds>, budget: usize) -> PrefillProgress {
        assert!(!prompt.is_empty() && prompt.len() <= self.max_seq_len, "inkling2 serve: a prompt of {} tokens the scheduler admitted", prompt.len());
        let content: Option<Arc<ContentKey>> = embeds.map(PromptEmbeds::content_key);
        let none = ContentKey::none();
        let key = Prompt::new(prompt, content.as_deref().unwrap_or(&none));
        if table.state_row().is_none() {
            let g = self.m.ops().gpu();
            if let Some(hidden) = self.pool.begin(g, table, key).exact {
                return PrefillProgress::Done(hidden);
            }
        }
        let done = table.len() as usize;
        let file_at = self.file_point(prompt.len());
        let mut n = budget.clamp(1, self.prefill_chunk as usize).min(prompt.len() - done);
        if done < file_at && done + n > file_at {
            n = file_at - done;
        }
        let start = self.pool.reserve(table, n as u32);
        let tokens = &prompt[done..done + n];
        let (g, d) = (self.g(), self.m.config().hidden as usize);
        let mut s = Vec::new();
        let x = self.m.embed(&mut s, tokens);
        if let Some(e) = embeds {
            self.m.splice(&mut s, &x, e, start, n as u32);
        }
        let recording = table.routing_log_mut().is_some();
        let mut routes = recording.then(Vec::new);
        let bank = self.bank_for(&[&*table], n);
        let normed = self.m.trunk(&mut s, &Rows::Chunk { pool: &self.pool, table, start, bank }, x, n as u32, &mut None, routes.as_mut()).expect("an admitted prompt runs");
        self.m.flush(&mut s);
        if let Some(routes) = routes {
            self.record(&mut [table], &routes, |_| start, n);
        }
        let hidden = g.read(&normed, n * d).split_off((n - 1) * d);
        if done + n == file_at {
            let g = self.m.ops().gpu();
            self.pool.file_prefix_content(g, table, &prompt[..file_at], content, hidden.clone());
        }
        if done + n < prompt.len() {
            PrefillProgress::Partial { consumed: n }
        } else {
            PrefillProgress::Done(hidden)
        }
    }

    /// Write each visit's experts of `rows` rows per sequence (`ids[visit]` `[sequences * rows, top_k]`) into the routing
    /// log of each sequence that carries one, at positions `first(i) ..`.
    fn record(&self, tables: &mut [&mut BlockTable], ids: &[DeviceBuffer], first: impl Fn(usize) -> u32, rows: usize) {
        let k = self.m.config().top_k as usize;
        let per = tables.len() * rows * k;
        let read: Vec<Vec<u32>> = ids.iter().map(|b| self.g().read(b, per).into_iter().map(f32::to_bits).collect()).collect();
        for (i, table) in tables.iter_mut().enumerate() {
            let at = first(i);
            let Some(log) = table.routing_log_mut() else { continue };
            for (call, experts) in read.iter().enumerate() {
                for r in 0..rows {
                    let row = i * rows + r;
                    log.record(call, at + r as u32, &experts[row * k..(row + 1) * k]);
                }
            }
        }
    }

    /// One decode pass: back each sequence's next position, run its token, and return the transformed logits
    /// `[b, vocab]` on the device.
    fn decode(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> DeviceBuffer {
        assert_eq!(tables.len(), inputs.len(), "inkling2 serve: one input per sequence");
        for t in tables.iter_mut() {
            self.pool.append(t);
        }
        let b = inputs.len() as u32;
        let mut s = Vec::new();
        let x = self.m.embed(&mut s, inputs);
        let recording = tables.iter_mut().any(|t| t.routing_log_mut().is_some());
        let mut routes = recording.then(Vec::new);
        let logits = {
            let refs: Vec<&BlockTable> = tables.iter().map(|t| &**t).collect();
            let bank = self.bank_for(&refs, 1);
            let normed = self.m.trunk(&mut s, &Rows::Decode { pool: &self.pool, tables: &refs, bank }, x, b, &mut None, routes.as_mut()).expect("an admitted batch runs");
            self.m.head(&mut s, &normed, b)
        };
        self.m.flush(&mut s);
        if let Some(routes) = routes {
            let positions: Vec<u32> = tables.iter().map(|t| t.len() - 1).collect();
            self.record(tables, &routes, |i| positions[i], 1);
        }
        logits
    }

    /// `rows` rows of the logits block, cut to the valid vocabulary.
    fn rows_of(&self, logits: &DeviceBuffer, rows: usize) -> Vec<Vec<f32>> {
        let cfg = self.m.config();
        let all = self.g().read(logits, rows * cfg.vocab as usize);
        all.chunks(cfg.vocab as usize).map(|r| r[..cfg.valid_vocab as usize].to_vec()).collect()
    }
}

impl BankedEngine for Engine {
    fn load_adapter(&mut self, adapter: VerifiedAdapter) -> Result<AdapterId, AdapterError> {
        Engine::load_adapter(self, adapter)
    }
    fn unload_adapter(&mut self, id: AdapterId) -> Result<UnloadStatus, AdapterError> {
        Engine::unload_adapter(self, id)
    }
}

impl PagedDecoder for Engine {
    fn adapter_availability(&mut self, adapter: AdapterId) -> model::serve::AdapterAvailability {
        let Some(bank) = &mut self.bank else { return model::serve::AdapterAvailability::Unknown };
        let answer = bank.availability(self.m.ops().gpu(), adapter);
        for ns in bank.take_purge() {
            self.pool.evict_namespace(ns);
        }
        answer
    }
    fn pin_adapter(&mut self, adapter: AdapterId) {
        self.bank.as_mut().expect("pin_adapter on an engine without an adapter bank").pin(adapter);
    }
    fn unpin_adapter(&mut self, adapter: AdapterId) {
        if let Some(ns) = self.bank.as_mut().and_then(|b| b.unpin(adapter)) {
            self.pool.evict_namespace(ns);
        }
    }
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        self.pool.alloc_mut()
    }
    fn max_prefill_tokens(&self) -> u32 {
        self.prefill_chunk
    }
    fn max_batch(&self) -> Option<usize> {
        Some(self.pool.max_batch())
    }
    fn free_blocks(&self) -> u32 {
        self.pool.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn vocab(&self) -> usize {
        self.m.config().valid_vocab as usize
    }
    fn head_transform(&self) -> model::head_transform::HeadTransform {
        self.m.head_transform()
    }
    fn routing_calls(&self) -> Option<Vec<RouterCallShape>> {
        let cfg = self.m.config();
        Some(vec![RouterCallShape { n_experts: cfg.experts, top_k: cfg.top_k }; self.m.plan().visits().len()])
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        self.pool.blocks_for(tokens)
    }
    fn cached_prefix_blocks(&self, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        self.pool.cached_prefix_blocks(namespace(adapter), prompt)
    }
    fn cached_prefix_blocks_embedded(&self, prompt: &[u32], adapter: Option<AdapterId>, embeds: &PromptEmbeds) -> u32 {
        self.pool.cached_prefix_blocks(namespace(adapter), Prompt::new(prompt, &embeds.content_key()))
    }
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        self.pool.reclaim(want)
    }
    fn reclaim_prefix_keeping(&mut self, want: u32, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        self.pool.reclaim_keeping(want, namespace(adapter), prompt)
    }
    fn reclaim_prefix_keeping_embedded(&mut self, want: u32, prompt: &[u32], adapter: Option<AdapterId>, embeds: &PromptEmbeds) -> u32 {
        self.pool.reclaim_keeping(want, namespace(adapter), Prompt::new(prompt, &embeds.content_key()))
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        self.pool.release(t);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(hidden) = self.prefill_chunk(table, prompt, usize::MAX) {
                return hidden;
            }
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(self.prefill_chunk as usize)
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        self.prefill_rows(table, prompt, None, budget)
    }
    fn embed_width(&self) -> Option<usize> {
        Some(self.m.config().hidden as usize)
    }
    fn prefill_chunk_embeds(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: &PromptEmbeds, budget: usize) -> PrefillProgress {
        assert_eq!(embeds.width(), self.m.config().hidden as usize, "inkling2 serve: embedding rows of the decoder's width");
        self.prefill_rows(table, prompt, Some(embeds), budget)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let g = self.g();
        let x = g.storage_init("inkling2.hidden", hidden);
        let mut s = Vec::new();
        let logits = self.m.head(&mut s, &x, 1);
        self.m.flush(&mut s);
        self.rows_of(&logits, 1).remove(0)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        let logits = self.decode(tables, inputs);
        argmax_rows(self.g(), self.readout, &logits, inputs.len() as u32, self.m.config().vocab)
    }
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        assert!((1..=DECODE_WINDOW).contains(&k), "inkling2 serve: a window of {k} steps");
        let mut out: Vec<Vec<u32>> = vec![Vec::with_capacity(k); inputs.len()];
        let mut next = inputs.to_vec();
        for _ in 0..k {
            next = self.forward_batched_greedy(tables, &next);
            for (o, &t) in out.iter_mut().zip(&next) {
                o.push(t);
            }
        }
        out
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let logits = self.decode(tables, inputs);
        topk_rows(self.g(), self.readout, &logits, inputs.len() as u32, self.m.config().vocab, k.clamp(1, TOPK_CAP) as u32)
    }
    fn full_row_readback(&self) -> bool {
        true
    }
    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        let logits = self.decode(tables, inputs);
        self.rows_of(&logits, inputs.len())
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        self.pool.prefix_stats()
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.g().stats()
    }
    fn kv_pool_bytes(&self) -> u64 {
        self.pool.pool_bytes()
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.pool.capacity_tokens()
    }
    fn decode_window_capacity(&self) -> usize {
        DECODE_WINDOW
    }
    fn topk_capacity(&self) -> usize {
        TOPK_CAP
    }
}
