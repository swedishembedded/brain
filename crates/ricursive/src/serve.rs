// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `model::serve::PagedDecoder` for Ripple: sliding layers keep a fixed
//! ring of recent keys per sequence, global layers keep a growing paged KV, and
//! both ride one sequence identity, under the generic continuous-batching
//! `model::serve::Scheduler`.
//!
//! Swedish Embedded AB implements solutions for serving long-context dense
//! decoders whose layers mix windowed and global attention at a fixed memory
//! budget for its clients. If your team needs expertise in hybrid KV layouts
//! and prefix reuse then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The scheduler owns admission, preemption, sampling, the prefix-cache policy
//! and speculation; this module implements the device operations it asks for
//! and nothing else.
//!
//! # One sequence, two kinds of KV
//!
//! * **Global layers** (16 of the 80) keep K/V in a block pool of `num_blocks`
//!   blocks of `block_size` rows per layer, addressed through the sequence's
//!   [`BlockTable`] and the allocator the scheduler also reads (`free_blocks`,
//!   `blocks_for`). They apply no RoPE, so cached rows are position-free.
//! * **Sliding layers** (64 of 80) keep a ring of [`ring_rows`] rows per
//!   sequence per layer: the absolute position `p` lives at ring row
//!   `p mod ring_rows`, RoPE is always applied at the absolute position, and a
//!   query at `i` attends absolute positions `i - window + 1 ..= i`. The ring
//!   is `window + prefill_chunk` rows rather than `window`: a prefill chunk
//!   writes all its rows before any of them attends, and a chunk row at `p`
//!   would otherwise overwrite position `p - window`, which an earlier row of
//!   the same chunk still needs. With the slack no key a row needs is
//!   overwritten by a row of the same chunk.
//! * The ring is the sequence's **state row** ([`BlockTable::state_row`]): the
//!   contract already carries a fixed-size per-sequence state beside the paged
//!   blocks for hybrid decoders, so nothing in `model::serve` changed. Ring
//!   rows are drawn from `max_batch` live slots (the scheduler never runs
//!   more sequences than [`PagedDecoder::max_batch`]) followed by
//!   `prefix_slots` snapshot slots.
//!
//! Attention reuses the paged decode kernel triad unchanged, one query row
//! per kernel "batch" entry with its own block table and key count: a global
//! row passes its block table; a sliding row passes a block table of
//! one-row "blocks" listing its ring rows in key order (`block_size = 1`), so
//! the window and the wrap-around are data, not a new kernel. A prefill chunk
//! is `c` rows of one sequence at consecutive positions; scratch is
//! `rows x heads x keys` and never `[heads, t, t]`.
//!
//! # Prefix reuse
//!
//! A global prefix can be shared as blocks; a ring cannot be rebuilt from
//! shared blocks once the prefix is longer than the window (it depends on
//! the last `window` rows' exact slots). So an entry of the
//! [`SnapshotPrefixCache`] is the pair that makes a prompt resumable: the
//! sequence's global blocks (retained by a fork of its table: shared, with
//! copy-on-write of a partly filled tail), a **snapshot of the ring** taken the
//! moment the prefill ended (one slot copy per sliding layer), and the last
//! hidden row, so an exact repeat needs no compute. A prompt that extends a
//! cached one adopts the blocks, copies the ring over its own, and prefills
//! only the rest. Reuse is therefore at the length a prompt was filed at: what
//! a conversation resent turn after turn needs; a shared document that
//! different prompts leave at different points is not reused (no branch
//! points are filed).
//!
//! # Memory
//!
//! [`global_kv_bytes_per_token`] and [`ring_bytes_per_slot`] are the two costs
//! the memory plan charges: the global KV grows with tokens held, the rings
//! are fixed per slot (live and snapshot) however long the sequences are.
//! Weights are f32 and resident; there is no quantization, no LSA and no
//! offload in this engine.

use checkpoint::TensorSource;
use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::paged::{BlockAllocator, BlockTable, KvNamespace, BASE_KV_NAMESPACE};
use model::serve::{AdapterId, PagedDecoder, PrefillProgress};
use model::snapshot_prefix::{Entry, SnapshotPrefixCache};
use paramstore::upload::Uploader;

use crate::config::{LayerType, RicursiveConfig};
use crate::layer::LayerWeights;
use crate::model::embed;
use crate::ops::{self, KERNELS, K_ADD2, K_APPLY, K_COPY_WITHIN, K_KV_APPEND, K_MUL, K_ROPE_PAGED, K_SCORES, K_SOFTMAX, SERVE_EXTRA};

/// The shortest prompt worth a snapshot: copying a slot's rings for a handful
/// of tokens costs more than prefilling them again.
pub const MIN_CACHED_TOKENS: usize = 16;

/// Candidate list width of [`RicursiveEngine::forward_batched_topk`].
const TOPK_CAPACITY: usize = 64;

/// How the engine is sized.
#[derive(Clone, Copy, Debug)]
pub struct ServeConfig {
    /// Rows per global KV block.
    pub block_size: u32,
    /// Global KV blocks in the pool (shared by live sequences and cached prefixes).
    pub num_blocks: u32,
    /// The most tokens (prompt plus generated) one sequence may hold.
    pub max_seq_len: u32,
    /// Live sequences, i.e. live ring slots.
    pub max_batch: u32,
    /// Cached prompts beside the live sequences, each costing one snapshot ring slot (0 turns reuse off).
    pub prefix_slots: u32,
    /// Prompt rows per prefill forward; sizes the ring slack.
    pub prefill_chunk: u32,
}

impl ServeConfig {
    fn validate(&self) -> Result<(), String> {
        for (name, v) in [("block_size", self.block_size), ("num_blocks", self.num_blocks), ("max_seq_len", self.max_seq_len), ("max_batch", self.max_batch), ("prefill_chunk", self.prefill_chunk)] {
            if v == 0 {
                return Err(format!("ricursive serve: {name} must be positive"));
            }
        }
        Ok(())
    }
}

/// Rows of one sequence's ring on one sliding layer: the window plus the slack
/// a prefill chunk of `prefill_chunk` rows needs (see the module doc).
pub fn ring_rows(cfg: &RicursiveConfig, prefill_chunk: u32) -> u32 {
    cfg.sliding_window as u32 + prefill_chunk
}

/// Device bytes the global layers cost per cached token (K and V, f32).
pub fn global_kv_bytes_per_token(cfg: &RicursiveConfig) -> u64 {
    let globals = cfg.layer_types.iter().filter(|t| **t == LayerType::Full).count() as u64;
    globals * 2 * cfg.kv_dim() as u64 * 4
}

/// Device bytes one ring slot (live or snapshot) costs across the sliding layers.
pub fn ring_bytes_per_slot(cfg: &RicursiveConfig, prefill_chunk: u32) -> u64 {
    let sliding = cfg.layer_types.iter().filter(|t| **t == LayerType::Sliding).count() as u64;
    sliding * 2 * u64::from(ring_rows(cfg, prefill_chunk)) * cfg.kv_dim() as u64 * 4
}

/// What resuming from a cached prompt needs beyond its blocks and ring snapshot.
struct Resume {
    /// The final-norm hidden row of the prompt's last token.
    last_hidden: Vec<f32>,
}

/// One row of a forward: a token at an absolute position of one sequence.
struct RowPlan<'a> {
    token: u32,
    pos: u32,
    /// The sequence's ring slot.
    ring: u32,
    /// The sequence's global blocks.
    blocks: &'a [u32],
}

/// K and V pools of one layer.
struct KvPool {
    k: DeviceBuffer,
    v: DeviceBuffer,
}

/// Per-forward row indices, resident for the forward's steps.
struct RowIndex {
    positions: DeviceBuffer,
    global_blocks: DeviceBuffer,
    global_offsets: DeviceBuffer,
    global_table: DeviceBuffer,
    global_lens: DeviceBuffer,
    ring_rows: DeviceBuffer,
    ring_offsets: DeviceBuffer,
    ring_table: DeviceBuffer,
    ring_lens: DeviceBuffer,
}

/// See the module doc.
pub struct RicursiveEngine {
    gpu: Gpu,
    cfg: RicursiveConfig,
    serve: ServeConfig,
    src: Box<dyn TensorSource>,
    layers: Vec<LayerWeights>,
    /// `Some` for every layer; sliding layers' pools are rings, global layers' are blocks.
    pools: Vec<KvPool>,
    final_norm: DeviceBuffer,
    lm_head: DeviceBuffer,
    alloc: BlockAllocator,
    ring_rows: u32,
    ring_free: Vec<u32>,
    cache: SnapshotPrefixCache<Resume>,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    max_blocks_per_seq: u32,
}

impl RicursiveEngine {
    /// Load every layer of `src` (canonical names, see [`crate::open_safetensors`]) onto the device selected by
    /// `device` (`None`: the process default) and size the KV pools.
    pub fn from_source(cfg: RicursiveConfig, src: Box<dyn TensorSource>, serve: ServeConfig, device: Option<&str>) -> Result<RicursiveEngine, String> {
        serve.validate()?;
        let mut table: Vec<(&str, &str)> = KERNELS.to_vec();
        table.extend(SERVE_EXTRA);
        let gpu = Gpu::open(device, &table);

        let mut up = Uploader::new(&gpu);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights::load(&mut up, &cfg, src.as_ref(), l)?);
        }
        let final_norm = up.tensor(src.as_ref(), "model.norm.weight", cfg.hidden_size)?;
        let lm_head = up.tensor(src.as_ref(), "lm_head.weight", cfg.vocab_size * cfg.hidden_size)?;

        let ring_rows = ring_rows(&cfg, serve.prefill_chunk);
        let ring_slots = u64::from(serve.max_batch + serve.prefix_slots);
        let kv = cfg.kv_dim() as u64;
        let pools = cfg
            .layer_types
            .iter()
            .map(|kind| {
                let rows = match kind {
                    LayerType::Sliding => ring_slots * u64::from(ring_rows),
                    LayerType::Full => u64::from(serve.num_blocks) * u64::from(serve.block_size),
                };
                KvPool { k: gpu.storage(rows * kv), v: gpu.storage(rows * kv) }
            })
            .collect();

        let max_rows = u64::from(serve.max_batch.max(serve.prefill_chunk));
        let max_keys = u64::from(serve.max_seq_len.max(cfg.sliding_window as u32));
        let scratch = max_rows * cfg.num_attention_heads as u64 * max_keys;
        Ok(RicursiveEngine {
            scores: gpu.storage(scratch),
            probs: gpu.storage(scratch),
            gpu,
            alloc: BlockAllocator::new(serve.num_blocks, serve.block_size),
            ring_rows,
            ring_free: (0..serve.max_batch).rev().collect(),
            cache: SnapshotPrefixCache::new(serve.prefix_slots),
            max_blocks_per_seq: serve.max_seq_len.div_ceil(serve.block_size),
            cfg,
            serve,
            src,
            layers,
            pools,
            final_norm,
            lm_head,
        })
    }

    pub fn config(&self) -> &RicursiveConfig {
        &self.cfg
    }

    /// The device handle, for profiling.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Device bytes of the two KV pools: the block pool (global layers) and the ring slots (sliding layers).
    pub fn kv_bytes(&self) -> (u64, u64) {
        let global = global_kv_bytes_per_token(&self.cfg) * u64::from(self.serve.num_blocks) * u64::from(self.serve.block_size);
        let rings = ring_bytes_per_slot(&self.cfg, self.serve.prefill_chunk) * u64::from(self.serve.max_batch + self.serve.prefix_slots);
        (global, rings)
    }

    /// Ring slot a snapshot row of the prefix cache lives in.
    fn snapshot_slot(&self, row: u32) -> u32 {
        self.serve.max_batch + row
    }

    fn ring_slot(table: &BlockTable) -> u32 {
        table.state_row().expect("a sequence that has begun prefilling owns a ring slot")
    }

    // ---- index and graph ------------------------------------------------------------------------------------------

    fn upload_u32(&self, data: &[u32]) -> DeviceBuffer {
        let buf = self.gpu.storage(data.len().max(1) as u64);
        self.gpu.write_at(&buf, 0, data);
        buf
    }

    fn index(&self, rows: &[RowPlan<'_>]) -> RowIndex {
        let (bs, r, w) = (self.serve.block_size, self.ring_rows, self.cfg.sliding_window as u32);
        let (max_bt, n) = (self.max_blocks_per_seq as usize, rows.len());
        let mut positions = Vec::with_capacity(n);
        let (mut g_blocks, mut g_offsets, mut g_lens) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut r_rows, mut r_lens) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut g_table = vec![0u32; n * max_bt];
        let mut r_table = vec![0u32; n * w as usize];
        for (i, row) in rows.iter().enumerate() {
            positions.push(row.pos);
            g_blocks.push(row.blocks[(row.pos / bs) as usize]);
            g_offsets.push(row.pos % bs);
            g_lens.push(row.pos + 1);
            g_table[i * max_bt..i * max_bt + row.blocks.len()].copy_from_slice(row.blocks);
            r_rows.push(row.ring * r + row.pos % r);
            // The keys a query at `pos` attends, oldest first, as ring rows.
            let lo = (row.pos + 1).saturating_sub(w);
            r_lens.push(row.pos - lo + 1);
            for (j, key) in (lo..=row.pos).enumerate() {
                r_table[i * w as usize + j] = row.ring * r + key % r;
            }
        }
        RowIndex {
            positions: self.upload_u32(&positions),
            global_blocks: self.upload_u32(&g_blocks),
            global_offsets: self.upload_u32(&g_offsets),
            global_table: self.upload_u32(&g_table),
            global_lens: self.upload_u32(&g_lens),
            ring_rows: self.upload_u32(&r_rows),
            ring_offsets: self.upload_u32(&vec![0; n]),
            ring_table: self.upload_u32(&r_table),
            ring_lens: self.upload_u32(&r_lens),
        }
    }

    /// One layer's attention over the keys already appended to `pool`: scores, softmax, apply.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, s: &mut Vec<Step>, q: &DeviceBuffer, pool: &KvPool, table: &DeviceBuffer, lens: &DeviceBuffer, rows: u32, block_size: u32, cap: u32, max_bt: u32) -> DeviceBuffer {
        let c = &self.cfg;
        let (heads, hd, kv, group) = (c.num_attention_heads as u32, c.head_dim as u32, c.kv_dim() as u32, c.group() as u32);
        let ctx = self.gpu.storage(u64::from(rows) * u64::from(heads * hd));
        let scale = f(1.0 / (hd as f32).sqrt());
        s.push(self.gpu.step(K_SCORES, &[q, &pool.k, table, lens, &self.scores], &[rows, heads, group, hd, block_size, kv, cap, max_bt, scale], rows * heads * cap));
        s.push(self.gpu.step(K_SOFTMAX, &[&self.scores, lens, &self.probs], &[rows, heads, cap], rows * heads));
        s.push(self.gpu.step(K_APPLY, &[&self.probs, &pool.v, table, lens, &ctx], &[rows, heads, group, hd, block_size, kv, cap, max_bt], rows * heads * hd));
        ctx
    }

    /// The layer stack over `rows`: appends their K/V and returns the residual stream after the last layer,
    /// `[rows, hidden]`, before the final norm.
    fn forward_hidden(&mut self, rows: &[RowPlan<'_>]) -> Result<DeviceBuffer, String> {
        let c = self.cfg.clone();
        let n = rows.len() as u32;
        let tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
        let x0_host = embed(self.src.as_ref(), &c, &tokens)?;
        let x0 = self.gpu.storage(x0_host.len() as u64);
        self.gpu.write_f32(&x0, &x0_host);
        let idx = self.index(rows);

        let (hidden, heads, kv_heads, hd) = (c.hidden_size as u32, c.num_attention_heads as u32, c.num_key_value_heads as u32, c.head_dim as u32);
        let (q_dim, kv_dim) = (c.q_dim() as u32, c.kv_dim() as u32);
        let gpu = &self.gpu;
        let mut s: Vec<Step> = Vec::new();
        let mut x = x0.clone();
        for (l, w) in self.layers.iter().enumerate() {
            let pool = &self.pools[l];
            let xn = ops::rmsnorm(gpu, &mut s, &x, &w.input_ln, hidden, n, c.rms_norm_eps);
            let q_pre = ops::linear(gpu, &mut s, &xn, &w.q, n, hidden, q_dim);
            let k_pre = ops::linear(gpu, &mut s, &xn, &w.k, n, hidden, kv_dim);
            let v_in = match w.value_mix {
                Some(mix) => ops::value_mix(gpu, &mut s, &x0, &xn, mix, n, hidden),
                None => xn.clone(),
            };
            let v = ops::linear(gpu, &mut s, &v_in, &w.v, n, hidden, kv_dim);
            let q = ops::rmsnorm(gpu, &mut s, &q_pre, &w.q_norm, hd, n * heads, c.rms_norm_eps);
            let k = ops::rmsnorm(gpu, &mut s, &k_pre, &w.k_norm, hd, n * kv_heads, c.rms_norm_eps);

            let ctx = match c.layer_type(l) {
                LayerType::Sliding => {
                    for (buf, h, stride) in [(&q, heads, q_dim), (&k, kv_heads, kv_dim)] {
                        s.push(gpu.step(K_ROPE_PAGED, &[buf, &idx.positions], &[n, h, hd, stride, f(c.rope_theta)], n * h * (hd / 2)));
                    }
                    s.push(gpu.step(K_KV_APPEND, &[&k, &idx.ring_rows, &idx.ring_offsets, &pool.k], &[n, kv_dim, 1], n * kv_dim));
                    s.push(gpu.step(K_KV_APPEND, &[&v, &idx.ring_rows, &idx.ring_offsets, &pool.v], &[n, kv_dim, 1], n * kv_dim));
                    let window = c.sliding_window as u32;
                    self.attend(&mut s, &q, pool, &idx.ring_table, &idx.ring_lens, n, 1, window, window)
                }
                LayerType::Full => {
                    let bs = self.serve.block_size;
                    s.push(gpu.step(K_KV_APPEND, &[&k, &idx.global_blocks, &idx.global_offsets, &pool.k], &[n, kv_dim, bs], n * kv_dim));
                    s.push(gpu.step(K_KV_APPEND, &[&v, &idx.global_blocks, &idx.global_offsets, &pool.v], &[n, kv_dim, bs], n * kv_dim));
                    self.attend(&mut s, &q, pool, &idx.global_table, &idx.global_lens, n, bs, self.serve.max_seq_len, self.max_blocks_per_seq)
                }
            };
            let ctx = match &w.g {
                Some(wg) => {
                    let gate = ops::linear(gpu, &mut s, &xn, wg, n, hidden, q_dim);
                    let gate = ops::sigmoid(gpu, &mut s, &gate, n * q_dim);
                    ops::binary(gpu, &mut s, K_MUL, &ctx, &gate, n * q_dim)
                }
                None => ctx,
            };
            let attn_out = ops::linear(gpu, &mut s, &ctx, &w.o, n, q_dim, hidden);
            let attn_normed = ops::rmsnorm(gpu, &mut s, &attn_out, &w.post_attn_ln, hidden, n, c.rms_norm_eps);
            let x1 = ops::binary(gpu, &mut s, K_ADD2, &x, &attn_normed, n * hidden);
            x = w.feed_forward(&c).record(gpu, &mut s, &x1, n);
        }
        gpu.submit(&[], &s);
        Ok(x)
    }

    /// Final norm and logits for `rows` rows of the residual stream `x`: `(normed, logits)`.
    fn head(&self, x: &DeviceBuffer, rows: u32) -> (Vec<f32>, Vec<f32>) {
        let (hidden, vocab) = (self.cfg.hidden_size as u32, self.cfg.vocab_size as u32);
        let normed = self.gpu.storage(u64::from(rows) * u64::from(hidden));
        let logits = self.gpu.storage(u64::from(rows) * u64::from(vocab));
        let steps = [
            self.gpu.step(ops::K_RMSNORM, &[x, &self.final_norm, &normed], &[hidden, rows, f(self.cfg.rms_norm_eps)], rows),
            self.gpu.step(ops::K_MATMUL, &[&normed, &self.lm_head, &logits], &[rows, hidden, vocab], rows * vocab),
        ];
        self.gpu.submit(&[], &steps);
        (self.gpu.read(&normed, (rows * hidden) as usize), self.gpu.read(&logits, (rows * vocab) as usize))
    }

    /// Copy `n` rows between two ring slots of every sliding layer (`slot_copy`), or of the blocks of every global
    /// layer, as one `copy_within` per K and V pool.
    fn copy_words(&self, layer: usize, src_word: u64, dst_word: u64, words: u64) -> [Step; 2] {
        let pool = &self.pools[layer];
        let step = |buf: &DeviceBuffer| self.gpu.step(K_COPY_WITHIN, &[buf], &[words as u32, src_word as u32, dst_word as u32], words as u32);
        [step(&pool.k), step(&pool.v)]
    }

    /// Copy ring slot `from` over ring slot `to` on every sliding layer.
    fn copy_ring(&self, from: u32, to: u32) {
        let words = u64::from(self.ring_rows) * self.cfg.kv_dim() as u64;
        let steps: Vec<Step> = (0..self.cfg.num_hidden_layers)
            .filter(|&l| self.cfg.layer_type(l) == LayerType::Sliding)
            .flat_map(|l| self.copy_words(l, u64::from(from) * words, u64::from(to) * words, words))
            .collect();
        self.gpu.submit(&[], &steps);
    }

    /// Copy the first `rows` rows of global block `from` over block `to` on every global layer.
    fn copy_block_rows(&self, from: u32, to: u32, rows: u32) {
        let (bs, kv) = (u64::from(self.serve.block_size), self.cfg.kv_dim() as u64);
        let steps: Vec<Step> = (0..self.cfg.num_hidden_layers)
            .filter(|&l| self.cfg.layer_type(l) == LayerType::Full)
            .flat_map(|l| self.copy_words(l, u64::from(from) * bs * kv, u64::from(to) * bs * kv, u64::from(rows) * kv))
            .collect();
        self.gpu.submit(&[], &steps);
    }

    // ---- sequences ------------------------------------------------------------------------------------------------

    /// First slice of a sequence: take its ring slot and resume from the prefix cache when there is a hit.
    /// `Some(hidden)` when the whole prompt was cached.
    fn begin(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Option<Vec<f32>> {
        assert!(!prompt.is_empty(), "ricursive serve: empty prompt (no token to produce a hidden state from)");
        assert!(prompt.len() <= self.serve.max_seq_len as usize, "ricursive serve: prompt of {} tokens exceeds the sequence limit of {}", prompt.len(), self.serve.max_seq_len);
        if let Some(&bad) = prompt.iter().find(|&&t| t as usize >= self.cfg.vocab_size) {
            panic!("ricursive serve: prompt token {bad} is outside the model vocabulary ({})", self.cfg.vocab_size);
        }
        let slot = self.ring_free.pop().expect("ricursive serve: no free ring slot (more live sequences than max_batch)");
        table.set_state_row(slot);
        if prompt.len() < MIN_CACHED_TOKENS {
            return None;
        }
        let hit = self.cache.lookup(table.kv_namespace(), prompt)?;
        let bs = self.serve.block_size;
        let entry = self.cache.entry(hit);
        let rows = entry.tokens.len() as u32;
        let (snap, shared, hidden) = (entry.snap_row, entry.table.blocks()[..rows.div_ceil(bs) as usize].to_vec(), entry.payload.last_hidden.clone());
        table.adopt_rows(&shared, rows, &mut self.alloc);
        if !rows.is_multiple_of(bs) {
            // The entry keeps serving later prompts from this block, so the one this sequence writes next is a copy.
            let (from, to) = table.unshare_tail(&mut self.alloc).expect("a block shared with the prefix cache has a single owner");
            self.copy_block_rows(from, to, rows % bs);
        }
        self.copy_ring(self.snapshot_slot(snap), slot);
        (rows as usize == prompt.len()).then_some(hidden)
    }

    /// File the sequence of `table`, whose state is exactly `prompt`'s, as a prefix entry.
    fn file_prefix(&mut self, table: &BlockTable, prompt: &[u32], hidden: &[f32]) {
        if prompt.len() < MIN_CACHED_TOKENS || !self.cache.enabled() {
            return;
        }
        let (row, evicted) = self.cache.take_row(table.kv_namespace(), prompt);
        if let Some(mut gone) = evicted {
            gone.release(&mut self.alloc);
        }
        let Some(row) = row else { return };
        self.copy_ring(Self::ring_slot(table), self.snapshot_slot(row));
        let shared = table.fork(&mut self.alloc);
        let retired = self.cache.insert(Entry::new(prompt.to_vec(), shared, row, Resume { last_hidden: hidden.to_vec() }));
        for mut gone in retired {
            gone.release(&mut self.alloc);
        }
    }

    /// Prefill `prompt[from..to]` (at most one ring-slack chunk) into `table`; the last row's final-norm hidden state
    /// when `last`.
    fn prefill_slice(&mut self, table: &mut BlockTable, prompt: &[u32], from: usize, to: usize, last: bool) -> Option<Vec<f32>> {
        let start = table.reserve((to - from) as u32, &mut self.alloc).unwrap_or_else(|e| panic!("ricursive serve: {e}"));
        assert_eq!(start as usize, from, "ricursive serve: the table holds {start} rows but the prompt resumes at {from}");
        let (blocks, ring) = (table.blocks().to_vec(), Self::ring_slot(table));
        let rows: Vec<RowPlan<'_>> = (from..to).map(|p| RowPlan { token: prompt[p], pos: p as u32, ring, blocks: &blocks }).collect();
        let x = self.forward_hidden(&rows).unwrap_or_else(|e| panic!("{e}"));
        last.then(|| {
            // Only the prompt's last row is projected.
            let hidden = self.cfg.hidden_size;
            let all = self.gpu.read(&x, rows.len() * hidden);
            let row = self.gpu.storage(hidden as u64);
            self.gpu.write_f32(&row, &all[(rows.len() - 1) * hidden..]);
            self.head(&row, 1).0
        })
    }

    /// One decode step: the logits of each `(table, input)` pair, `[vocab]` per sequence.
    pub fn decode_logits(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        assert_eq!(tables.len(), inputs.len(), "ricursive serve: tables/inputs length mismatch");
        if tables.is_empty() {
            return Vec::new();
        }
        assert!(tables.len() <= self.serve.max_batch as usize, "ricursive serve: {} sequences exceed max_batch {}", tables.len(), self.serve.max_batch);
        let mut owned = Vec::with_capacity(tables.len());
        for (t, &input) in tables.iter_mut().zip(inputs) {
            t.append(&mut self.alloc).unwrap_or_else(|e| panic!("ricursive serve: {e}"));
            assert!(t.len() <= self.serve.max_seq_len, "ricursive serve: sequence outgrew the limit of {} tokens", self.serve.max_seq_len);
            owned.push((input, t.len() - 1, Self::ring_slot(t), t.blocks().to_vec()));
        }
        let rows: Vec<RowPlan<'_>> = owned.iter().map(|(token, pos, ring, blocks)| RowPlan { token: *token, pos: *pos, ring: *ring, blocks }).collect();
        let x = self.forward_hidden(&rows).unwrap_or_else(|e| panic!("{e}"));
        let n = rows.len();
        let logits = self.head(&x, n as u32).1;
        logits.chunks(self.cfg.vocab_size).map(<[f32]>::to_vec).collect()
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best }).0 as u32
}

fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut ranked: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
}

impl PagedDecoder for RicursiveEngine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        self.serve.prefill_chunk
    }
    fn max_batch(&self) -> Option<usize> {
        Some(self.serve.max_batch as usize)
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        self.serve.max_seq_len as usize
    }
    fn vocab(&self) -> usize {
        self.cfg.vocab_size
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.serve.block_size)
    }
    /// Whole blocks a prompt adopts from the cache; the partly filled tail is copied, not adopted.
    fn cached_prefix_blocks(&self, prompt: &[u32], adapter: Option<AdapterId>) -> u32 {
        let namespace: KvNamespace = adapter.map_or(BASE_KV_NAMESPACE, AdapterId::kv_namespace);
        if prompt.len() < MIN_CACHED_TOKENS {
            return 0;
        }
        self.cache.peek(namespace, prompt).map_or(0, |i| self.cache.entry(i).tokens.len() as u32 / self.serve.block_size)
    }
    fn reclaim_prefix(&mut self, want: u32) -> u32 {
        let before = self.alloc.free_blocks();
        while self.alloc.free_blocks() - before < want {
            let Some(mut gone) = self.cache.evict_lru() else { break };
            gone.release(&mut self.alloc);
        }
        self.alloc.free_blocks() - before
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        if let Some(slot) = t.take_state_row() {
            self.ring_free.push(slot);
        }
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(hidden) = self.prefill_chunk(table, prompt, usize::MAX) {
                return hidden;
            }
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(self.serve.prefill_chunk as usize)
    }
    /// The next slice of `prompt`, at most `budget` tokens, run as forwards of at most `prefill_chunk` rows. The first
    /// call takes the sequence's ring slot and resumes from the prefix cache; the call that ends the prompt files it.
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        if table.state_row().is_none() {
            if let Some(hidden) = self.begin(table, prompt) {
                return PrefillProgress::Done(hidden);
            }
        }
        let done = table.len() as usize;
        let end = done + budget.min(prompt.len() - done);
        let mut hidden = None;
        let mut at = done;
        while at < end {
            let to = end.min(at + self.serve.prefill_chunk as usize);
            hidden = self.prefill_slice(table, prompt, at, to, to == prompt.len());
            at = to;
        }
        if end < prompt.len() {
            return PrefillProgress::Partial { consumed: end - done };
        }
        let hidden = hidden.expect("the last slice of a prompt returns its hidden state");
        self.file_prefix(table, prompt, &hidden);
        PrefillProgress::Done(hidden)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let d = self.cfg.hidden_size;
        assert_eq!(hidden.len(), d, "ricursive serve::logits: hidden must be one row of {d} floats, got {}", hidden.len());
        let x = self.gpu.storage(d as u64);
        self.gpu.write_f32(&x, hidden);
        let logits = self.gpu.storage(self.cfg.vocab_size as u64);
        let step = self.gpu.step(ops::K_MATMUL, &[&x, &self.lm_head, &logits], &[1, d as u32, self.cfg.vocab_size as u32], self.cfg.vocab_size as u32);
        self.gpu.submit(&[], &[step]);
        self.gpu.read(&logits, self.cfg.vocab_size)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.decode_logits(tables, inputs).iter().map(|l| argmax(l)).collect()
    }
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        assert!((1..=self.decode_window_capacity()).contains(&k), "window {k} exceeds this engine's decode_window_capacity {}", self.decode_window_capacity());
        self.forward_batched_greedy(tables, inputs).into_iter().map(|t| vec![t]).collect()
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        let k = k.clamp(1, self.topk_capacity());
        self.decode_logits(tables, inputs).iter().map(|l| top_k(l, k)).collect()
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        self.cache.stats()
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.gpu.stats()
    }
    /// The block pool of the global layers plus every ring slot (live and snapshot) of the sliding layers.
    fn kv_pool_bytes(&self) -> u64 {
        let (global, rings) = self.kv_bytes();
        global + rings
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        u64::from(self.serve.num_blocks) * u64::from(self.serve.block_size)
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
    fn topk_capacity(&self) -> usize {
        TOPK_CAPACITY.min(self.cfg.vocab_size)
    }
}
