// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One pipeline stage of the serving engine: a contiguous range of Ripple's layers on one device, with their
//! weights, their KV and the scratch their attention needs.
//!
//! Swedish Embedded AB implements solutions for splitting very large decoders over several accelerators, each
//! holding the layers and the KV cache its memory allows, for its clients. If your team needs expertise in
//! pipeline-parallel serving of long-context models then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A stage owns everything device-resident for its layers: the linears at their tier, the sliding layers' rings
//! and the global layers' block pools (as [`model::kv_tier::KvLayer`]s of the engine's KV tier), and the score
//! scratch of the paged attention triad. Sequence identity (block tables, the allocator, ring slots, the prefix
//! cache) is host-side and shared by all stages: block `b` of a sequence is block `b` in every stage's pools.
//! Only the residual stream crosses a stage boundary.

use std::ops::Range;

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::adapter::device::LoraCfg;
use model::adapter::exec::{AdapterRuntime, ParamAccess};
use model::kv_tier::{FlashDecodeShape, KvAppend, KvKernels, KvLayer, PrefillShape};
use model::lsa::binding::LsaBinding;
use model::lsa::engine::LayerBuffers;
use model::ops::{Ops, PagedDecodeShape, Weight};

use crate::config::{LayerType, RicursiveConfig};
use crate::layer::LayerWeights;
use crate::ops::{self, LayerAdapter, K_ADD2, K_COPY_WITHIN, K_MUL, K_ROPE_PAGED, K_SOFTMAX};
use crate::serve::ServeConfig;

/// What every stage needs to know about the engine it serves.
pub(crate) struct Geometry<'a> {
    pub cfg: &'a RicursiveConfig,
    pub serve: &'a ServeConfig,
    /// Rows of one ring ([`crate::ring_rows`]).
    pub ring_rows: u32,
    /// Row stride of the block tables, in blocks.
    pub max_blocks_per_seq: u32,
}

impl Geometry<'_> {
    /// The most keys a dense read of a global layer spans ([`ServeConfig::dense_keys`]).
    pub(crate) fn dense_keys(&self) -> u32 {
        self.serve.dense_keys.unwrap_or(self.serve.max_seq_len)
    }
}

/// One row of a forward: a token at an absolute position of one sequence.
pub(crate) struct RowPlan<'a> {
    pub token: u32,
    pub pos: u32,
    /// The sequence's ring slot.
    pub ring: u32,
    /// The sequence's global blocks.
    pub blocks: &'a [u32],
}

/// The per-row indices of a forward, built once on the host and uploaded to each stage's device.
pub(crate) struct HostIndex {
    positions: Vec<u32>,
    global_blocks: Vec<u32>,
    global_offsets: Vec<u32>,
    global_table: Vec<u32>,
    global_lens: Vec<u32>,
    ring_rows: Vec<u32>,
    ring_table: Vec<u32>,
    ring_lens: Vec<u32>,
    /// Whether every row is a position of one sequence (a prefill chunk), as the fused prefill kernel requires.
    one_sequence: bool,
}

impl HostIndex {
    pub(crate) fn build(g: &Geometry<'_>, rows: &[RowPlan<'_>]) -> HostIndex {
        let (bs, r, w) = (g.serve.block_size, g.ring_rows, g.cfg.sliding_window as u32);
        let (max_bt, n) = (g.max_blocks_per_seq as usize, rows.len());
        let mut ix = HostIndex {
            positions: Vec::with_capacity(n),
            global_blocks: Vec::with_capacity(n),
            global_offsets: Vec::with_capacity(n),
            global_table: vec![0u32; n * max_bt],
            global_lens: Vec::with_capacity(n),
            ring_rows: Vec::with_capacity(n),
            ring_table: vec![0u32; n * w as usize],
            ring_lens: Vec::with_capacity(n),
            one_sequence: rows.iter().all(|r| r.ring == rows[0].ring),
        };
        for (i, row) in rows.iter().enumerate() {
            ix.positions.push(row.pos);
            ix.global_blocks.push(row.blocks[(row.pos / bs) as usize]);
            ix.global_offsets.push(row.pos % bs);
            ix.global_lens.push(row.pos + 1);
            ix.global_table[i * max_bt..i * max_bt + row.blocks.len()].copy_from_slice(row.blocks);
            ix.ring_rows.push(row.ring * r + row.pos % r);
            // The keys a query at `pos` attends, oldest first, as ring rows.
            let lo = (row.pos + 1).saturating_sub(w);
            ix.ring_lens.push(row.pos - lo + 1);
            for (j, key) in (lo..=row.pos).enumerate() {
                ix.ring_table[i * w as usize + j] = row.ring * r + key % r;
            }
        }
        ix
    }

    /// The most keys any row of the forward attends on a global layer.
    fn longest(&self) -> u32 {
        self.global_lens.iter().copied().max().unwrap_or(0)
    }

    fn upload(&self, gpu: &Gpu) -> RowIndex {
        let put = |data: &[u32]| {
            let buf = gpu.storage(data.len().max(1) as u64);
            gpu.write_at(&buf, 0, data);
            buf
        };
        RowIndex {
            positions: put(&self.positions),
            global_blocks: put(&self.global_blocks),
            global_offsets: put(&self.global_offsets),
            global_table: put(&self.global_table),
            global_lens: put(&self.global_lens),
            ring_rows: put(&self.ring_rows),
            ring_offsets: put(&vec![0; self.positions.len()]),
            ring_table: put(&self.ring_table),
            ring_lens: put(&self.ring_lens),
        }
    }
}

/// [`HostIndex`] on one device, resident for the forward's steps.
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

/// The final norm and `lm_head` the last stage owns.
pub(crate) struct StageHead {
    pub norm: DeviceBuffer,
    pub lm_head: Weight,
}

/// The pairs `(base weight name, A [r, in], B [out, r])` of the sites an adapter corrects.
pub(crate) type AdapterPairs<'a> = Vec<(String, &'a [f32], &'a [f32])>;

/// The factors of an attached adapter on one stage's device, by the tensor names the shared adapter runtime reads.
pub(crate) struct Factors(HashMap<String, DeviceBuffer>);

impl ParamAccess for Factors {
    fn weight(&self, name: &str) -> &DeviceBuffer {
        self.0.get(name).unwrap_or_else(|| panic!("ricursive: the attached adapter has no factor {name}"))
    }

    fn grad(&self, _name: &str) -> Option<&DeviceBuffer> {
        None
    }
}

/// An adapter applied at run time to this stage's layers, beside the resident base.
pub(crate) struct StageAdapter {
    lora: LoraCfg,
    factors: Factors,
}

/// See the module doc.
pub(crate) struct Stage {
    pub ops: Ops,
    /// The layers this stage runs.
    pub range: Range<usize>,
    pub layers: Vec<LayerWeights>,
    /// K and V of each layer of `range`; sliding layers' are rings, global layers' are block pools.
    pub kv: Vec<KvLayer>,
    pub kernels: KvKernels,
    pub head: Option<StageHead>,
    /// The attached adapter's factors for this stage's layers, applied after every targeted linear.
    adapter: Option<StageAdapter>,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    /// Attention runs the fused paged flash kernels (no score slabs) rather than the triad.
    pub(crate) fused: bool,
}

impl Stage {
    pub(crate) fn new(ops: Ops, g: &Geometry<'_>, range: Range<usize>, layers: Vec<LayerWeights>, head: Option<StageHead>) -> Result<Stage, String> {
        let (cfg, serve) = (g.cfg, g.serve);
        let gpu = ops.gpu();
        let slots = u64::from(serve.max_batch + serve.prefix_slots);
        let (kv_dim, hd) = (cfg.kv_dim() as u64, cfg.head_dim as u64);
        let kv = range
            .clone()
            .map(|l| {
                let rows = match cfg.layer_type(l) {
                    LayerType::Sliding => slots * u64::from(g.ring_rows),
                    LayerType::Full => u64::from(serve.num_blocks) * u64::from(serve.block_size),
                };
                check_addressing(serve.kv_cache, rows, kv_dim)?;
                Ok(KvLayer::new(gpu, serve.kv_cache, rows, kv_dim, hd))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let kernels = KvKernels::resolve_for(gpu, serve.kv_cache, &[crate::ops::flash_shape(cfg)])?;
        let (hd32, group) = crate::ops::flash_shape(cfg);
        let fused = serve.fused_attention && kernels.flash_decode_available(gpu, hd32, group);
        if serve.fused_attention && !fused {
            tracing::warn!(device = gpu.kind(), "ricursive: the fused paged attention kernels do not run on this device; attending through the score-slab triad");
        }
        let max_rows = u64::from(serve.max_batch.max(serve.prefill_chunk));
        let max_keys = u64::from(g.dense_keys().max(cfg.sliding_window as u32));
        // The fused kernels keep their partial states per call; only the triad needs the score slabs.
        let scratch = if fused { 1 } else { max_rows * cfg.num_attention_heads as u64 * max_keys };
        Ok(Stage { scores: gpu.storage(scratch), probs: gpu.storage(scratch), ops, range, layers, kv, kernels, head, fused, adapter: None })
    }

    pub(crate) fn gpu(&self) -> &Gpu {
        self.ops.gpu()
    }

    /// Attach (or, with `None`, detach) an adapter: the factors of the sites in this stage's layers are uploaded to its
    /// device. `pairs` are `(base weight name, A [r, in], B [out, r])`; the shared adapter runtime applies them.
    pub(crate) fn set_adapter(&mut self, lora: Option<(&LoraCfg, &AdapterPairs<'_>)>) {
        self.adapter = lora.map(|(lora, pairs)| {
            let gpu = self.ops.gpu();
            let mut map = HashMap::new();
            for (base, a, b) in pairs {
                let (an, bn) = model::adapter::device::lora_names(base);
                map.insert(an, gpu.storage_init("adapter_a", a));
                map.insert(bn, gpu.storage_init("adapter_b", b));
            }
            StageAdapter { lora: lora.clone(), factors: Factors(map) }
        });
    }

    pub(crate) fn kv_of(&self, layer: usize) -> &KvLayer {
        &self.kv[layer - self.range.start]
    }

    /// One layer's attention over the keys already appended to `layer`'s KV: the fused flash kernels where the device runs
    /// them (the prefill kernel for the rows of one sequence reading a global layer, `chunk`; the split-key decode kernel
    /// for everything else, a row per sequence), else scores, softmax, apply. `cap` bounds every row's key count.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, g: &Geometry<'_>, s: &mut Vec<Step>, q: &DeviceBuffer, layer: &KvLayer, table: &DeviceBuffer, lens: &DeviceBuffer, rows: u32, block_size: u32, cap: u32, max_bt: u32, chunk: bool) -> DeviceBuffer {
        let c = g.cfg;
        let gpu = self.gpu();
        let (heads, hd) = (c.num_attention_heads as u32, c.head_dim as u32);
        let ctx = gpu.storage(u64::from(rows) * u64::from(heads * hd));
        if self.fused {
            if chunk && rows > 1 {
                let shape = PrefillShape { n: rows, n_heads: heads, n_kv_heads: c.num_key_value_heads as u32, head_dim: hd, block_size, max_bt, keys: cap };
                s.extend(self.kernels.flash_prefill_hd256(gpu, q, &layer.k, &layer.v, table, lens, &ctx, shape));
            } else {
                let shape = FlashDecodeShape { batch: rows, n_heads: heads, n_kv_heads: c.num_key_value_heads as u32, head_dim: hd, block_size, max_bt, cap, pin_splits: false };
                s.extend(self.kernels.flash_decode(gpu, q, &layer.k, &layer.v, table, lens, &ctx, shape));
            }
            return ctx;
        }
        let shape = PagedDecodeShape { batch: rows, n_heads: heads, group: c.group() as u32, head_dim: hd, block_size, kv_stride: c.kv_dim() as u32, cap, max_bt, scale: 1.0 / (hd as f32).sqrt() };
        s.push(self.kernels.scores(gpu, q, &layer.k, table, lens, &self.scores, shape));
        s.push(gpu.step(K_SOFTMAX, &[&self.scores, lens, &self.probs], &[rows, heads, cap], rows * heads));
        s.push(self.kernels.apply(gpu, &self.probs, &layer.v, table, lens, &ctx, shape));
        ctx
    }

    /// Append `k` and `v` (`[n, kv_dim]`) to `layer`'s planes at each row's `(blocks[i], offsets[i])`.
    fn append(&self, g: &Geometry<'_>, s: &mut Vec<Step>, layer: &KvLayer, k: &DeviceBuffer, v: &DeviceBuffer, blocks: &DeviceBuffer, offsets: &DeviceBuffer, n: u32, block_size: u32) {
        let a = KvAppend { batch: n, kv_stride: g.cfg.kv_dim() as u32, block_size, head_dim: g.cfg.head_dim as u32 };
        s.push(self.kernels.append(self.gpu(), &layer.k, k, blocks, offsets, a));
        s.push(self.kernels.append(self.gpu(), &layer.v, v, blocks, offsets, a));
    }

    /// The stage's layers over `rows` (already indexed as `index`): appends their K/V and returns the residual
    /// stream after the last of them, `[rows, hidden]` on this stage's device. `x` is the residual entering the
    /// stage and `x0` the value-residual source (the scaled embeddings), both on this device. `capture` collects
    /// each global layer's normed queries; `lsa` reads the global layers sparsely.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(&self, g: &Geometry<'_>, index: &HostIndex, n: u32, x: &DeviceBuffer, x0: &DeviceBuffer, lsa: Option<&LsaBinding>, mut capture: Option<&mut Vec<DeviceBuffer>>) -> Result<DeviceBuffer, String> {
        let c = g.cfg;
        let (ops, gpu) = (&self.ops, self.gpu());
        let idx = index.upload(gpu);
        let (hidden, heads, kv_heads, hd) = (c.hidden_size as u32, c.num_attention_heads as u32, c.num_key_value_heads as u32, c.head_dim as u32);
        let (q_dim, kv_dim) = (c.q_dim() as u32, c.kv_dim() as u32);
        let mut s: Vec<Step> = Vec::new();
        let mut x = x.clone();
        // The adapter runtime's scratch is sized for this forward's rows.
        let runtime = self.adapter.as_ref().map(|a| AdapterRuntime::new(gpu, &a.lora, n, q_dim.max(c.intermediate_size as u32).max(hidden)));
        for (w, l) in self.layers.iter().zip(self.range.clone()) {
            let layer = self.kv_of(l);
            let adapter = runtime.as_ref().zip(self.adapter.as_ref()).map(|(rt, a)| LayerAdapter { rt, factors: &a.factors, layer: l });
            let correct = |s: &mut Vec<Step>, leaf: &str, x: &DeviceBuffer, y: &DeviceBuffer, k: u32, out: u32| {
                if let Some(a) = &adapter {
                    a.correct(ops, s, leaf, x, y, n, k, out);
                }
            };
            let xn = ops::rmsnorm(gpu, &mut s, &x, &w.input_ln, hidden, n, c.rms_norm_eps);
            let mut qk = ops::linears(ops, &mut s, &xn, &[&w.q, &w.k], n);
            let (k_pre, q_pre) = (qk.pop().expect("k"), qk.pop().expect("q"));
            correct(&mut s, "q_proj", &xn, &q_pre, hidden, q_dim);
            correct(&mut s, "k_proj", &xn, &k_pre, hidden, kv_dim);
            let v_in = match w.value_mix {
                Some(mix) => ops::value_mix(gpu, &mut s, x0, &xn, mix, n, hidden),
                None => xn.clone(),
            };
            let v = ops::linear(ops, &mut s, &v_in, &w.v, n);
            correct(&mut s, "v_proj", &v_in, &v, hidden, kv_dim);
            let q = ops::rmsnorm(gpu, &mut s, &q_pre, &w.q_norm, hd, n * heads, c.rms_norm_eps);
            if c.layer_type(l) == LayerType::Full {
                if let Some(taken) = capture.as_deref_mut() {
                    taken.push(q.clone());
                }
            }
            let k = ops::rmsnorm(gpu, &mut s, &k_pre, &w.k_norm, hd, n * kv_heads, c.rms_norm_eps);

            let ctx = match c.layer_type(l) {
                LayerType::Sliding => {
                    for (buf, h, stride) in [(&q, heads, q_dim), (&k, kv_heads, kv_dim)] {
                        s.push(gpu.step(K_ROPE_PAGED, &[buf, &idx.positions], &[n, h, hd, stride, f(c.rope_theta)], n * h * (hd / 2)));
                    }
                    self.append(g, &mut s, layer, &k, &v, &idx.ring_rows, &idx.ring_offsets, n, 1);
                    let window = c.sliding_window as u32;
                    self.attend(g, &mut s, &q, layer, &idx.ring_table, &idx.ring_lens, n, 1, window, window, false)
                }
                LayerType::Full => {
                    let bs = g.serve.block_size;
                    self.append(g, &mut s, layer, &k, &v, &idx.global_blocks, &idx.global_offsets, n, bs);
                    match lsa.filter(|a| a.reads_sparsely(l)) {
                        Some(lsa) => {
                            let ctx = gpu.storage(u64::from(n) * u64::from(q_dim));
                            let scales = layer.k.scales().zip(layer.v.scales());
                            let bufs = LayerBuffers { q: &q, pos: &idx.positions, seqlen: &idx.global_lens, block_tables: &idx.global_table, pool_k: layer.k.data(), pool_v: layer.v.data(), scales, ctx: &ctx };
                            lsa.record_layer(gpu, &mut s, l, n, &bufs).map_err(|e| format!("ricursive: latent sparse attention: {e}"))?;
                            ctx
                        }
                        None => {
                            if !self.fused && index.longest() > g.dense_keys() {
                                return Err(format!("ricursive: a dense read of global layer {l} spans {} keys but the score scratch holds {} (ServeConfig::dense_keys); read it sparsely or raise the limit", index.longest(), g.dense_keys()));
                            }
                            let cap = if self.fused { index.longest() } else { g.dense_keys() };
                            self.attend(g, &mut s, &q, layer, &idx.global_table, &idx.global_lens, n, bs, cap, g.max_blocks_per_seq, index.one_sequence)
                        }
                    }
                }
            };
            let ctx = match &w.g {
                Some(wg) => {
                    let gate = ops::linear(ops, &mut s, &xn, wg, n);
                    correct(&mut s, "g_proj", &xn, &gate, hidden, q_dim);
                    let gate = ops::sigmoid(gpu, &mut s, &gate, n * q_dim);
                    ops::binary(gpu, &mut s, K_MUL, &ctx, &gate, n * q_dim)
                }
                None => ctx,
            };
            let attn_out = ops::linear(ops, &mut s, &ctx, &w.o, n);
            correct(&mut s, "o_proj", &ctx, &attn_out, q_dim, hidden);
            let attn_normed = ops::rmsnorm(gpu, &mut s, &attn_out, &w.post_attn_ln, hidden, n, c.rms_norm_eps);
            let x1 = ops::binary(gpu, &mut s, K_ADD2, &x, &attn_normed, n * hidden);
            x = w.feed_forward(c, adapter.as_ref()).record(ops, &mut s, &x1, n);
        }
        gpu.submit(&[], &s);
        Ok(x)
    }

    /// Final norm and logits for `rows` rows of the residual stream `x`: `(normed, logits)`. Only the last stage has a head.
    pub(crate) fn head_rows(&self, eps: f32, x: &DeviceBuffer, rows: u32, hidden: u32, vocab: u32) -> (Vec<f32>, Vec<f32>) {
        let head = self.head.as_ref().expect("the last stage owns the head");
        let gpu = self.gpu();
        let normed = gpu.storage(u64::from(rows) * u64::from(hidden));
        let mut steps = vec![gpu.step(ops::K_RMSNORM, &[x, &head.norm, &normed], &[hidden, rows, f(eps)], rows)];
        let logits = ops::linear(&self.ops, &mut steps, &normed, &head.lm_head, rows);
        gpu.submit(&[], &steps);
        (gpu.read(&normed, (rows * hidden) as usize), gpu.read(&logits, rows as usize * vocab as usize))
    }

    /// The logits of `rows` rows of final-normed hidden state `normed` (host), `[rows, vocab]` flattened. Last stage only.
    pub(crate) fn project(&self, normed: &[f32], vocab: usize) -> Vec<f32> {
        let head = self.head.as_ref().expect("the last stage owns the head");
        let gpu = self.gpu();
        let rows = (normed.len() / head.lm_head.k() as usize) as u32;
        let x = gpu.storage(normed.len() as u64);
        gpu.write_f32(&x, normed);
        let mut steps = Vec::new();
        let logits = ops::linear(&self.ops, &mut steps, &x, &head.lm_head, rows);
        gpu.submit(&[], &steps);
        gpu.read(&logits, rows as usize * vocab)
    }

    /// Copy `rows` token rows of every layer of kind `kind` in this stage from `from_row` to `to_row` (the rows of a
    /// ring slot, or of a block), K and V planes and, in a scaled tier, their row scales.
    pub(crate) fn copy_rows(&self, g: &Geometry<'_>, kind: LayerType, from_row: u64, to_row: u64, rows: u64) {
        let (kv_dim, hd) = (g.cfg.kv_dim() as u64, g.cfg.head_dim as u64);
        let gpu = self.gpu();
        let steps: Vec<Step> = self
            .range
            .clone()
            .filter(|&l| g.cfg.layer_type(l) == kind)
            .flat_map(|l| {
                let layer = self.kv_of(l);
                let mut v = layer.k.copy_rows(gpu, K_COPY_WITHIN, from_row, to_row, rows, kv_dim, hd);
                v.extend(layer.v.copy_rows(gpu, K_COPY_WITHIN, from_row, to_row, rows, kv_dim, hd));
                v
            })
            .collect();
        gpu.submit(&[], &steps);
    }
}

/// The KV planes index their elements with `u32`; a pool past that would silently alias its own start.
fn check_addressing(tier: model::kv_tier::KvTier, rows: u64, kv_stride: u64) -> Result<(), String> {
    if model::kv_tier::KvTier::fits_addressing(rows, kv_stride) {
        Ok(())
    } else {
        Err(format!("ricursive: a {tier} KV plane of {rows} rows x {kv_stride} elements exceeds the kernels' 32-bit element index ({}); shrink the block pool or split the layers over more stages", model::kv_tier::MAX_PLANE_ELEMS))
    }
}
