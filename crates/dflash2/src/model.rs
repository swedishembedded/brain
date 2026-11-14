// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The DFlash2 device model: context projection, the denoising pass and the per-slot K/V caches it attends to.
//!
//! Swedish Embedded AB implements speculative decoding - drafters, verifiers and the state rollback a hybrid or
//! recurrent stack needs - for clients serving large models on modest hardware. If your team needs inference
//! throughput without a second model's worth of accuracy loss, you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! # What makes it more than a small decoder
//!
//! * **It reads the target's internals.** Each layer's keys and values are computed over the concatenation of the
//!   target's residual at its tapped layers, fused by `fc` and normed by `hidden_norm` (one row per context token),
//!   and the block's own rows.
//! * **Attention is not causal.** The block's mask rows see each other, and every row sees the whole context of a
//!   full layer, or the positions within `window` of it on a sliding layer. A NoPE layer applies no RoPE at all.
//! * **Dynamic convolutions** wrap every attention and MLP sublayer: half of a projection of the activations IS the
//!   tap set, per token and per group of channels, added to a static per-channel base. The convolution is causal
//!   inside the block and its history starts at zero for every block.
//!
//! # The caches
//!
//! A cache slot holds one sequence's K/V rows for every layer, indexed by absolute position, so a speculative
//! rollback is free: the next [`Dflash2::append_context`] rewrites exactly the rows it invalidated. A sliding layer's
//! rows are a ring of `window + block_size` positions ([`Dflash2Config::cache_rows`]); a full layer's rows are
//! every position. Rows of the denoising block itself are written into the same cache at their absolute positions
//! (the reference's "crop the draft cache back to `start`" is the next round overwriting them).

use std::sync::Mutex;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::kv_tier::{FlashDecodeShape, KvAppend, KvKernels, KvLayer, KvTier};
use model::ops::{Act, Ops, PagedDecodeShape, Weight};

use crate::config::{Dflash2Config, LayerKind};
use crate::selector::Selector;
use crate::source::{DraftSource, Leaf, Tensor};

const MODEL: &str = "dflash2";

/// How many context rows one [`Dflash2::append_context`] round projects at a time: the `fc` input block is
/// `rows * tap_width` f32 (26 MB at 256 rows for five taps of 5120).
const CTX_CHUNK: u32 = 256;

/// The kernels this model dispatches by name, beyond the products and the KV tier's.
const OWN_KERNELS: [(&str, &str); 6] = [
    ("rmsnorm", kernels::RMSNORM),
    ("add2", kernels::ADD2),
    ("silu_mul", kernels::SILU_MUL),
    ("rope2d", kernels::ROPE2D),
    ("dyn_group_conv1d", kernels::DYN_GROUP_CONV1D),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
];

/// What a device that hosts a draft adds to its own pipeline list: this model's kernels and the KV tier's (the
/// paged attention, and the fused decode for the draft's head shape). Names already in `existing` are not repeated.
pub fn kernels_beyond(existing: &[(&str, &str)], shapes: &[(u32, u32)]) -> Vec<(&'static str, &'static str)> {
    let have: std::collections::HashSet<&str> = existing.iter().map(|(n, _)| *n).collect();
    let mut out: Vec<(&'static str, &'static str)> = OWN_KERNELS.iter().copied().filter(|(n, _)| !have.contains(n)).collect();
    for entry in model::kv_tier::kernel_list_for(existing, shapes) {
        if !out.iter().any(|(n, _)| *n == entry.0) {
            out.push(entry);
        }
    }
    out
}

/// The `(head_dim, group)` of the fused decode the draft's attention asks for.
pub fn head_shape(cfg: &Dflash2Config) -> (u32, u32) {
    (cfg.head_dim, cfg.n_heads / cfg.n_kv_heads)
}

/// Every pipeline a device that runs only the draft registers: the shared product façade, this model's kernels and
/// the KV tier's, with the fused decode for the draft's own head shape.
pub fn pipelines(cfg: &Dflash2Config) -> Vec<(&'static str, &'static str)> {
    let mut table: Vec<(&'static str, &'static str)> = model::ops::kernel_list().to_vec();
    let more = kernels_beyond(&table, &[head_shape(cfg)]);
    table.extend(more);
    table
}

struct Layer {
    input_norm: DeviceBuffer,
    post_attn_norm: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    /// `[2, taps, d_model]`: `[0]` is the base kernel of the convolution ENTERING the sublayer, `[1]` of the one
    /// leaving it.
    attn_conv_base: DeviceBuffer,
    attn_conv_proj: Weight,
    mlp_conv_base: DeviceBuffer,
    mlp_conv_proj: Weight,
}

#[derive(Clone, Copy)]
struct Kids {
    rmsnorm: usize,
    add2: usize,
    silu_mul: usize,
    rope2d: usize,
    dyn_conv: usize,
    softmax: usize,
}

/// One block to denoise: the sequence's cache slot, the absolute position of the block's first row (its anchor), and
/// the target's embedding of `[anchor, MASK, ...]`.
pub struct BlockJob<'a> {
    pub slot: u32,
    pub pos: u32,
    pub noise: &'a [f32],
}

/// How the draft's K/V caches are held.
#[derive(Clone, Copy, Debug)]
pub struct DraftCache {
    /// The storage tier of the K and V planes.
    pub tier: KvTier,
    /// Rows per block of the full layers' shared pool.
    pub block_size: u32,
    /// Positions the full layers' pool holds across all slots; a sequence that would outgrow what is free is refused
    /// by name (the draft proposes nothing for it) and the memory plan charges exactly this.
    pub pool_positions: u32,
}

impl DraftCache {
    /// The dense layout: f32, and a pool that holds every slot at once at what a slot holds - `cap` context positions
    /// and the denoising block of `cfg` past them, rounded up to whole pool blocks - so no sequence within the cap is
    /// ever refused for want of a block.
    pub fn dense(cfg: &Dflash2Config, slots: u32, cap: u32) -> DraftCache {
        const BLOCK: u32 = 16;
        let per_slot = cap.saturating_add(cfg.block_size).div_ceil(BLOCK).saturating_mul(BLOCK);
        DraftCache { tier: KvTier::F32, block_size: BLOCK, pool_positions: slots.saturating_mul(per_slot) }
    }

    fn pool_blocks(&self) -> u32 {
        self.pool_positions.div_ceil(self.block_size.max(1))
    }

    /// Device bytes of the K and V planes of all layers of `cfg` for `slots` sequences of up to `cap` positions.
    pub fn bytes(&self, cfg: &Dflash2Config, slots: u32, cap: u32) -> u64 {
        let (kv, hd) = (u64::from(cfg.kv_dim()), u64::from(cfg.head_dim));
        (0..cfg.n_layers)
            .map(|l| {
                let rows = match cfg.layers[l].kind {
                    LayerKind::Sliding => u64::from(slots) * u64::from(cfg.cache_rows(l, cap)),
                    LayerKind::Full => u64::from(self.pool_blocks()) * u64::from(self.block_size),
                };
                2 * self.tier.plane_bytes(rows, kv, hd)
            })
            .sum()
    }
}

/// The blocks of the full layers' pool: free ones, and the ones each slot holds in position order.
struct Pool {
    free: Vec<u32>,
    tables: Vec<Vec<u32>>,
}

/// The per-layer hidden states of a denoising pass, for gating a port layer by layer.
pub type LayerTrace = Vec<Vec<f32>>;

/// A loaded draft model, resident on one device, with K/V caches for `slots` sequences of up to `cap` positions.
pub struct Dflash2 {
    gpu: Gpu,
    ops: Ops,
    k: Kids,
    kv: KvKernels,
    fused: bool,
    pub cfg: Dflash2Config,
    src: Box<dyn DraftSource>,
    selector: Selector,
    fc: Weight,
    hidden_norm: DeviceBuffer,
    layers: Vec<Layer>,
    caches: Vec<KvLayer>,
    cache: DraftCache,
    pool: Mutex<Pool>,
    slots: u32,
    cap: u32,
    weight_bytes: u64,
}

impl Dflash2 {
    /// Load the draft from `src` onto `gpu` (whose pipelines include [`pipelines`] or [`kernels_beyond`]), with K/V
    /// caches for `slots` sequences of up to `cap` positions each.
    pub fn load(gpu: Gpu, cfg: Dflash2Config, src: Box<dyn DraftSource>, slots: u32, cap: u32) -> Result<Dflash2, String> {
        let cache = DraftCache::dense(&cfg, slots, cap);
        Self::load_with(gpu, cfg, src, slots, cap, cache)
    }

    /// [`Self::load`] with the caches held as `cache` says.
    pub fn load_with(gpu: Gpu, cfg: Dflash2Config, src: Box<dyn DraftSource>, slots: u32, cap: u32, cache: DraftCache) -> Result<Dflash2, String> {
        cfg.validate()?;
        if cache.block_size == 0 || cache.pool_positions == 0 {
            return Err(format!("{MODEL}: the draft's cache pool needs a block size and positions (got {} and {})", cache.block_size, cache.pool_positions));
        }
        if slots == 0 || cap == 0 {
            return Err(format!("{MODEL}: a draft needs at least one cache slot and one position (slots {slots}, cap {cap})"));
        }
        let ops = Ops::new(gpu.share())?;
        let kid = |n: &str| gpu.kernel_index(n).ok_or_else(|| format!("{MODEL}: kernel {n:?} is not registered on this device - build its pipelines with dflash2::pipelines or kernels_beyond"));
        let k = Kids { rmsnorm: kid("rmsnorm")?, add2: kid("add2")?, silu_mul: kid("silu_mul")?, rope2d: kid("rope2d")?, dyn_conv: kid("dyn_group_conv1d")?, softmax: kid("decode_softmax_batched")? };
        let shape = head_shape(&cfg);
        let kv = KvKernels::resolve_for(&gpu, cache.tier, &[shape])?;
        let fused = kv.flash_decode_available(&gpu, shape.0, shape.1);

        let (d, ff) = (cfg.d_model as usize, cfg.d_ff as usize);
        let (q_dim, kv_dim) = (cfg.q_dim() as usize, cfg.kv_dim() as usize);
        let wide = cfg.tap_width();
        let dyn_width = cfg.dyn_width() as usize;
        let mut weight_bytes = 0u64;
        let mut linear = |t: Tensor, n: usize, kk: usize| -> Result<Weight, String> {
            if let Some(got) = src.shape(t).filter(|got| *got != [n, kk]) {
                return Err(format!("{MODEL}: {} is {got:?}, the config implies [{n}, {kk}]", t.name_hf()));
            }
            let w = src.linear(&ops, t, n, kk)?;
            weight_bytes += w.resident_bytes();
            Ok(w)
        };
        let fc = linear(Tensor::Fc, d, wide)?;
        let buf = |t: Tensor, len: usize| -> Result<DeviceBuffer, String> {
            let v = src.vector(t)?;
            if v.len() != len {
                return Err(format!("{MODEL}: {} holds {} values, the config implies {len}", t.name_hf(), v.len()));
            }
            Ok(gpu.storage_init(&t.name_hf(), &v))
        };
        let hidden_norm = buf(Tensor::HiddenNorm, d)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let t = |leaf| Tensor::Layer(l, leaf);
            layers.push(Layer {
                input_norm: buf(t(Leaf::InputNorm), d)?,
                post_attn_norm: buf(t(Leaf::PostAttnNorm), d)?,
                q_norm: buf(t(Leaf::QNorm), cfg.head_dim as usize)?,
                k_norm: buf(t(Leaf::KNorm), cfg.head_dim as usize)?,
                wq: linear(t(Leaf::Q), q_dim, d)?,
                wk: linear(t(Leaf::K), kv_dim, d)?,
                wv: linear(t(Leaf::V), kv_dim, d)?,
                wo: linear(t(Leaf::O), d, q_dim)?,
                gate: linear(t(Leaf::Gate), ff, d)?,
                up: linear(t(Leaf::Up), ff, d)?,
                down: linear(t(Leaf::Down), d, ff)?,
                attn_conv_base: buf(t(Leaf::AttnConvBase), 2 * cfg.conv_kernel as usize * d)?,
                attn_conv_proj: linear(t(Leaf::AttnConvProj), dyn_width, d)?,
                mlp_conv_base: buf(t(Leaf::MlpConvBase), 2 * cfg.conv_kernel as usize * d)?,
                mlp_conv_proj: linear(t(Leaf::MlpConvProj), dyn_width, d)?,
            });
        }
        let selector = Selector::load(src.as_ref())?;
        let limit = gpu.max_storage_binding_bytes();
        let caches = (0..cfg.n_layers)
            .map(|l| {
                let rows = match cfg.layers[l].kind {
                    LayerKind::Sliding => u64::from(slots) * u64::from(cfg.cache_rows(l, cap)),
                    LayerKind::Full => u64::from(cache.pool_blocks()) * u64::from(cache.block_size),
                };
                let plane = cache.tier.plane_bytes(rows, u64::from(cfg.kv_dim()), u64::from(cfg.head_dim));
                if !KvTier::fits_addressing(rows, u64::from(cfg.kv_dim())) || plane > limit {
                    return Err(format!(
                        "{MODEL}: layer {l}'s {} cache of {rows} rows x {} elements exceeds what this device binds ({limit} bytes); serve a shorter context or fewer sequences, or a narrower cache tier",
                        cache.tier,
                        cfg.kv_dim()
                    ));
                }
                Ok(KvLayer::new(&gpu, cache.tier, rows, u64::from(cfg.kv_dim()), u64::from(cfg.head_dim)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let pool = Mutex::new(Pool { free: (0..cache.pool_blocks()).rev().collect(), tables: vec![Vec::new(); slots as usize] });
        Ok(Dflash2 { gpu, ops, k, kv, fused, cfg, src, selector, fc, hidden_norm, layers, caches, cache, pool, slots, cap, weight_bytes })
    }

    /// Give a finished or evicted sequence's blocks of the shared pool back at once. Its ring rows are overwritten by the
    /// slot's next sequence, which feeds them before reading.
    pub fn release_slot(&self, slot: u32) {
        let mut pool = self.pool.lock().expect("the draft pool lock is never held across a panic");
        let held = std::mem::take(&mut pool.tables[slot as usize]);
        pool.free.extend(held.into_iter().rev());
    }

    /// Blocks of the shared pool in use and in all.
    pub fn pool_use(&self) -> (u32, u32) {
        let pool = self.pool.lock().expect("the draft pool lock is never held across a panic");
        (self.cache.pool_blocks() - pool.free.len() as u32, self.cache.pool_blocks())
    }

    /// Make `slot` hold the blocks for positions `0 .. upto` of the full layers (a no-op without a full layer); an error
    /// names a pool with no block left.
    pub fn reserve(&self, slot: u32, upto: u32) -> Result<(), String> {
        if self.cfg.layers.iter().all(|l| l.kind != LayerKind::Full) {
            return Ok(());
        }
        let need = upto.div_ceil(self.cache.block_size) as usize;
        let mut pool = self.pool.lock().expect("the draft pool lock is never held across a panic");
        while pool.tables[slot as usize].len() < need {
            let Some(block) = pool.free.pop() else {
                return Err(format!(
                    "{MODEL}: the draft's cache pool of {} positions is full (a sequence at position {upto} needs another block); serve fewer or shorter sequences, or hold the pool wider",
                    self.cache.pool_positions
                ));
            };
            pool.tables[slot as usize].push(block);
        }
        Ok(())
    }

    /// Resident bytes of the weights.
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Positions each slot holds.
    pub fn cap(&self) -> u32 {
        self.cap
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The draft's final RMSNorm gain, for a head that applies it in place of the target's own.
    pub fn output_norm(&self) -> &[f32] {
        self.selector.output_norm()
    }

    /// The draft's hidden rows RMS-normed with its final norm, ready for the target's head.
    pub fn normed(&self, hidden: &[f32]) -> Vec<f32> {
        self.selector.normed(&self.cfg, hidden)
    }

    /// The selector's chosen token per row of `hidden` (the pre-final-norm block rows, anchor excluded), given the
    /// target head's `logits` for them.
    pub fn select(&self, hidden: &[f32], logits: &[f32], anchor: u32) -> Result<Vec<u32>, String> {
        self.selector.select(&self.cfg, self.src.as_ref(), hidden, logits, anchor)
    }

    fn act(&self, s: &mut Vec<Step>, x: &DeviceBuffer, xr0: u32, rows: u32, k: u32, weights: &[&Weight]) -> Act {
        if weights.iter().all(|w| w.reads_f32_activations()) {
            self.ops.act_f32(x, xr0, rows, k)
        } else {
            self.ops.act(s, x, xr0, rows, k)
        }
    }

    /// `x[xr0 .. xr0 + rows] @ w^T`.
    fn matmul(&self, s: &mut Vec<Step>, x: &DeviceBuffer, xr0: u32, rows: u32, w: &Weight) -> DeviceBuffer {
        let act = self.act(s, x, xr0, rows, w.k(), &[w]);
        let out = self.gpu.storage(u64::from(rows) * u64::from(w.n()));
        self.ops.matmul(s, w, &act, &out, 0);
        out
    }

    fn rms(&self, s: &mut Vec<Step>, x: &DeviceBuffer, gain: &DeviceBuffer, dim: u32, rows: u32) -> DeviceBuffer {
        let out = self.gpu.storage(u64::from(dim) * u64::from(rows));
        s.push(self.gpu.step(self.k.rmsnorm, &[x, gain, &out], &[dim, rows, f(self.cfg.eps)], rows));
        out
    }

    fn add(&self, s: &mut Vec<Step>, a: &DeviceBuffer, b: &DeviceBuffer, n: u32) -> DeviceBuffer {
        let out = self.gpu.storage(u64::from(n));
        s.push(self.gpu.step(self.k.add2, &[a, b, &out], &[n], n));
        out
    }

    fn upload(&self, data: &[u32]) -> DeviceBuffer {
        let buf = self.gpu.storage(data.len().max(1) as u64);
        self.gpu.write_at(&buf, 0, data);
        buf
    }

    /// Host-side cos/sin tables for `positions`, `[n, head_dim/2]` each, in the half-split pairing `rope2d.wgsl`
    /// reads. FULL rotation over every head channel: the config's `rope_type` is `default` with no partial factor.
    fn rope_tables(&self, positions: &[u32]) -> (DeviceBuffer, DeviceBuffer) {
        let half = (self.cfg.head_dim / 2) as usize;
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for &p in positions {
            for d in 0..half {
                let inv = f64::from(self.cfg.rope_theta).powf(-2.0 * d as f64 / f64::from(self.cfg.head_dim));
                let a = f64::from(p) * inv;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        (self.gpu.storage_init("dflash2.rope.cos", &cos), self.gpu.storage_init("dflash2.rope.sin", &sin))
    }

    fn rope(&self, s: &mut Vec<Step>, buf: &DeviceBuffer, cos: &DeviceBuffer, sin: &DeviceBuffer, rows: u32, heads: u32) {
        s.push(model::block::rope2d_fwd(&self.gpu, self.k.rope2d, buf, cos, sin, rows, heads, self.cfg.head_dim, heads * self.cfg.head_dim));
    }

    /// The physical rows of absolute positions `positions` of `slot` in layer `layer`'s cache (a full layer's through the
    /// slot's blocks, which [`Self::reserve`] has made).
    fn cache_rows_of(&self, layer: usize, slot: u32, positions: impl Iterator<Item = u32>) -> Vec<u32> {
        match self.cfg.layers[layer].kind {
            LayerKind::Sliding => {
                let rows = self.cfg.cache_rows(layer, self.cap);
                positions.map(|p| slot * rows + p % rows).collect()
            }
            LayerKind::Full => {
                let pool = self.pool.lock().expect("the draft pool lock is never held across a panic");
                let (table, bs) = (&pool.tables[slot as usize], self.cache.block_size);
                positions.map(|p| table[(p / bs) as usize] * bs + p % bs).collect()
            }
        }
    }

    /// Write `k` and `v` (`[n, kv_dim]`) into the cache planes of `layer` at the given physical rows.
    fn write_rows(&self, s: &mut Vec<Step>, layer: usize, k: &DeviceBuffer, v: &DeviceBuffer, rows: &[u32]) {
        let (blocks, offsets) = (self.upload(rows), self.upload(&vec![0; rows.len()]));
        let a = KvAppend { batch: rows.len() as u32, kv_stride: self.cfg.kv_dim(), block_size: 1, head_dim: self.cfg.head_dim };
        s.push(self.kv.append(&self.gpu, &self.caches[layer].k, k, &blocks, &offsets, a));
        s.push(self.kv.append(&self.gpu, &self.caches[layer].v, v, &blocks, &offsets, a));
    }

    /// One grouped dynamic causal convolution over `n` rows in blocks of `block` rows (the history restarts at every
    /// block). `half` selects which of the projection's two tap sets and of the base kernel's two halves to spend:
    /// 0 entering the sublayer, 1 leaving it.
    fn dyn_conv(&self, s: &mut Vec<Step>, x: &DeviceBuffer, dynamic: &DeviceBuffer, base: &DeviceBuffer, n: u32, block: u32, half: u32) -> DeviceBuffer {
        let c = &self.cfg;
        let (d, taps, groups) = (c.d_model, c.conv_kernel, c.groups());
        let out = self.gpu.storage(u64::from(n) * u64::from(d));
        s.push(self.gpu.step(self.k.dyn_conv, &[x, dynamic, base, &out], &[n, d, taps, c.conv_group, groups, c.dyn_width(), half * taps * groups, half * taps * d, block], n * d));
        out
    }

    /// Attention of `rows` query rows (`q`, `[rows, q_dim]`), each over the keys its table and length name in
    /// `layer`'s cache: the fused decode where the device runs it, else scores, softmax, apply.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, s: &mut Vec<Step>, layer: usize, q: &DeviceBuffer, table: &DeviceBuffer, lens: &DeviceBuffer, rows: u32, block_size: u32, max_bt: u32, cap: u32) -> DeviceBuffer {
        let c = &self.cfg;
        let out = self.gpu.storage(u64::from(rows) * u64::from(c.q_dim()));
        let cache = &self.caches[layer];
        if self.fused {
            let shape = FlashDecodeShape { batch: rows, n_heads: c.n_heads, n_kv_heads: c.n_kv_heads, head_dim: c.head_dim, block_size, max_bt, cap, pin_splits: false };
            s.extend(self.kv.flash_decode(&self.gpu, q, &cache.k, &cache.v, table, lens, &out, shape));
            return out;
        }
        let shape = PagedDecodeShape { batch: rows, n_heads: c.n_heads, group: c.n_heads / c.n_kv_heads, head_dim: c.head_dim, block_size, kv_stride: c.kv_dim(), cap, max_bt, scale: 1.0 / (c.head_dim as f32).sqrt() };
        let scores = self.gpu.storage(u64::from(rows) * u64::from(c.n_heads) * u64::from(cap));
        let probs = self.gpu.storage(u64::from(rows) * u64::from(c.n_heads) * u64::from(cap));
        s.push(self.kv.scores(&self.gpu, q, &cache.k, table, lens, &scores, shape));
        s.push(self.gpu.step(self.k.softmax, &[&scores, lens, &probs], &[rows, c.n_heads, cap], rows * c.n_heads));
        s.push(self.kv.apply(&self.gpu, &probs, &cache.v, table, lens, &out, shape));
        out
    }

    /// **Project and cache the target's hidden states for the context tokens** at absolute positions
    /// `pos_start ..` of `slot`. `hidden` is `[m, tap_width]`: the target's residual at each tapped layer,
    /// concatenated in `target_layers` order.
    ///
    /// It goes through `fc`, then `hidden_norm` (that order, the one place this architecture departs from "norm then
    /// project"), then each layer's own `k_proj`/`v_proj` + per-head norm + RoPE (where the layer rotates), landing
    /// in the cache at the rows of those positions. A sliding layer keeps only the positions a later query can still
    /// read, so only the last `cache_rows` positions of a long context are projected for it.
    ///
    /// Incremental by design: a round pays only for the tokens the target just committed, never for the whole
    /// context, which is what keeps the drafter's cost flat in context length.
    pub fn append_context(&self, slot: u32, hidden: &[f32], pos_start: u32) -> Result<(), String> {
        let d = self.cfg.d_model;
        let wide = self.cfg.tap_width() as u32;
        if slot >= self.slots {
            return Err(format!("{MODEL}: cache slot {slot} of {}", self.slots));
        }
        if hidden.is_empty() || !(hidden.len() as u32).is_multiple_of(wide) {
            return Err(format!("{MODEL}: context block of {} floats is not a multiple of {wide}", hidden.len()));
        }
        let total = hidden.len() as u32 / wide;
        let end_total = pos_start + total;
        if end_total > self.cap {
            return Err(format!("{MODEL}: context reaches position {end_total} but a cache slot holds {}", self.cap));
        }
        let (nkv, hd) = (self.cfg.n_kv_heads, self.cfg.head_dim);
        if self.cfg.layers.iter().any(|l| l.kind == LayerKind::Full) {
            self.reserve(slot, end_total)?;
        }
        for off in (0..total).step_by(CTX_CHUNK as usize) {
            let m = CTX_CHUNK.min(total - off);
            let base = pos_start + off;
            let chunk_end = base + m;
            let rows = &hidden[(off * wide) as usize..((off + m) * wide) as usize];
            let _scope = self.gpu.scratch_scope();
            let x = self.gpu.storage_init("dflash2.ctx.in", rows);
            let mut s = Vec::new();
            let proj = self.matmul(&mut s, &x, 0, m, &self.fc);
            let normed = self.rms(&mut s, &proj, &self.hidden_norm, d, m);
            for (l, layer) in self.layers.iter().enumerate() {
                // The positions this layer's cache can still be asked for once the whole context is in.
                let keep_from = match self.cfg.layers[l].kind {
                    LayerKind::Sliding => end_total.saturating_sub(self.cfg.cache_rows(l, self.cap)),
                    LayerKind::Full => 0,
                };
                let from = base.max(keep_from);
                if from >= chunk_end {
                    continue;
                }
                let cnt = chunk_end - from;
                let kb = self.matmul(&mut s, &normed, from - base, cnt, &layer.wk);
                let vb = self.matmul(&mut s, &normed, from - base, cnt, &layer.wv);
                let kn = self.rms(&mut s, &kb, &layer.k_norm, hd, cnt * nkv);
                if self.cfg.layers[l].rope {
                    let positions: Vec<u32> = (from..chunk_end).collect();
                    let (cos, sin) = self.rope_tables(&positions);
                    self.rope(&mut s, &kn, &cos, &sin, cnt, nkv);
                }
                let phys = self.cache_rows_of(l, slot, from..chunk_end);
                self.write_rows(&mut s, l, &kn, &vb, &phys);
            }
            self.gpu.submit(&[], &s);
            self.gpu.poll_wait();
            self.gpu.flush();
        }
        Ok(())
    }

    /// **The denoising pass** for one block per job, all in one batched forward: each job is `n` rows
    /// `[anchor, MASK, ...]` at absolute positions `pos .. pos + n` of its slot (every job the same `n`). Returns,
    /// per job, the PRE-final-norm hidden state of every row, `[n, d_model]`.
    ///
    /// `noise` is the target's raw embedding of those `n` token ids. The final norm is left off: the head applies
    /// it with this model's gain ([`Self::output_norm`]), so no second copy of the target's head is needed.
    pub fn denoise_blocks(&self, jobs: &[BlockJob<'_>]) -> Result<Vec<Vec<f32>>, String> {
        self.denoise_traced(jobs, None)
    }

    /// [`Self::denoise_blocks`], also pushing each layer's output (all jobs' rows, `[rows, d_model]`) onto `trace`:
    /// what a port is gated against layer by layer.
    pub fn denoise_traced(&self, jobs: &[BlockJob<'_>], mut trace: Option<&mut LayerTrace>) -> Result<Vec<Vec<f32>>, String> {
        let Some(first) = jobs.first() else { return Ok(Vec::new()) };
        let c = &self.cfg;
        let d = c.d_model;
        let n = first.noise.len() as u32 / d;
        if n == 0 || n > c.block_size {
            return Err(format!("{MODEL}: a block of {n} rows (the draft denoises at most {})", c.block_size));
        }
        for j in jobs {
            if j.noise.len() as u32 != n * d {
                return Err(format!("{MODEL}: jobs of different block lengths ({} and {} values)", j.noise.len(), n * d));
            }
            if j.slot >= self.slots {
                return Err(format!("{MODEL}: cache slot {} of {}", j.slot, self.slots));
            }
            if j.pos + n > self.cap + c.block_size {
                return Err(format!("{MODEL}: a block ending at {} is past the {} positions (and one block) a cache slot holds", j.pos + n, self.cap));
            }
        }
        if jobs.iter().enumerate().any(|(i, a)| jobs[..i].iter().any(|b| b.slot == a.slot)) {
            return Err(format!("{MODEL}: two blocks of one batch share a cache slot"));
        }
        if c.layers.iter().any(|l| l.kind == LayerKind::Full) {
            for j in jobs {
                self.reserve(j.slot, j.pos + n)?;
            }
        }
        let rows = jobs.len() as u32 * n;
        let (nh, nkv, hd, ff) = (c.n_heads, c.n_kv_heads, c.head_dim, c.d_ff);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let positions: Vec<u32> = jobs.iter().flat_map(|j| j.pos..j.pos + n).collect();
        let (cos, sin) = self.rope_tables(&positions);
        let noise: Vec<f32> = jobs.iter().flat_map(|j| j.noise.iter().copied()).collect();
        let mut x = self.gpu.storage_init("dflash2.block.in", &noise);

        // Which keys each query row reads: positions `lo ..= hi` of its slot. Non-causal: every row of the block
        // reads to its end; a sliding layer reads back `window - 1` positions, a full layer to position 0.
        let ranges = |sliding: bool| -> Vec<(u32, u32)> {
            jobs.iter()
                .flat_map(|j| {
                    (0..n).map(move |i| {
                        let q = j.pos + i;
                        let hi = if c.causal { q } else { j.pos + n - 1 };
                        (if sliding { (q + 1).saturating_sub(c.window) } else { 0 }, hi)
                    })
                })
                .collect()
        };
        let (sliding_ranges, full_ranges) = (ranges(true), ranges(false));
        let longest = |r: &[(u32, u32)]| r.iter().map(|&(lo, hi)| hi - lo + 1).max().unwrap_or(1);
        let (cap_sliding, cap_full) = (longest(&sliding_ranges), longest(&full_ranges));
        // A sliding layer lists the physical rows of its keys in key order (one-row blocks); a full layer's keys are
        // contiguous from the start of the slot's rows, so its "table" is the slot alone and its block the slot's rows.
        let sliding_table = match (0..c.n_layers).find(|&l| c.layers[l].kind == LayerKind::Sliding) {
            Some(l) => {
                let stride = cap_sliding as usize;
                let mut t = vec![0u32; rows as usize * stride];
                for (r, &(lo, hi)) in sliding_ranges.iter().enumerate() {
                    let slot = jobs[r / n as usize].slot;
                    t[r * stride..r * stride + (hi - lo + 1) as usize].copy_from_slice(&self.cache_rows_of(l, slot, lo..=hi));
                }
                t
            }
            None => vec![0],
        };
        let sliding_lens: Vec<u32> = sliding_ranges.iter().map(|&(lo, hi)| hi - lo + 1).collect();
        // A full layer lists the blocks of its slot's pool, in position order.
        let bs = self.cache.block_size;
        let max_bt = cap_full.div_ceil(bs).max(1);
        let full_table: Vec<u32> = {
            let pool = self.pool.lock().expect("the draft pool lock is never held across a panic");
            let mut t = vec![0u32; rows as usize * max_bt as usize];
            for r in 0..rows as usize {
                let table = &pool.tables[jobs[r / n as usize].slot as usize];
                let used = (full_ranges[r].1 / bs + 1) as usize;
                t[r * max_bt as usize..r * max_bt as usize + used.min(table.len())].copy_from_slice(&table[..used.min(table.len())]);
            }
            t
        };
        let full_lens: Vec<u32> = full_ranges.iter().map(|&(lo, hi)| hi - lo + 1).collect();
        let (sliding_table, sliding_lens) = (self.upload(&sliding_table), self.upload(&sliding_lens));
        let (full_table, full_lens) = (self.upload(&full_table), self.upload(&full_lens));

        for (l, lb) in self.layers.iter().enumerate() {
            let spec = c.layers[l];
            let _scope = self.gpu.scratch_scope();
            let mut s = Vec::new();

            // ---- attention sublayer, wrapped in its own two convolutions ----
            let xn1 = self.rms(&mut s, &x, &lb.input_norm, d, rows);
            let dyn1 = self.matmul(&mut s, &xn1, 0, rows, &lb.attn_conv_proj);
            let h = self.dyn_conv(&mut s, &xn1, &dyn1, &lb.attn_conv_base, rows, n, 0);

            let act = self.act(&mut s, &h, 0, rows, d, &[&lb.wq, &lb.wk, &lb.wv]);
            let project = |s: &mut Vec<Step>, w: &Weight, width: u32| {
                let out = self.gpu.storage(u64::from(rows) * u64::from(width));
                self.ops.matmul(s, w, &act, &out, 0);
                out
            };
            let (qp, kp, vp) = (project(&mut s, &lb.wq, hq), project(&mut s, &lb.wk, hkv), project(&mut s, &lb.wv, hkv));
            let qn = self.rms(&mut s, &qp, &lb.q_norm, hd, rows * nh);
            let kn = self.rms(&mut s, &kp, &lb.k_norm, hd, rows * nkv);
            if spec.rope {
                self.rope(&mut s, &qn, &cos, &sin, rows, nh);
                self.rope(&mut s, &kn, &cos, &sin, rows, nkv);
            }
            // The block's own rows join the cache at their absolute positions, right after the context rows. The next
            // `append_context` overwrites whichever of them the target actually committed with the real thing.
            let phys: Vec<u32> = jobs.iter().flat_map(|j| self.cache_rows_of(l, j.slot, j.pos..j.pos + n)).collect();
            self.write_rows(&mut s, l, &kn, &vp, &phys);
            let attn = match spec.kind {
                LayerKind::Sliding => self.attend(&mut s, l, &qn, &sliding_table, &sliding_lens, rows, 1, cap_sliding, cap_sliding),
                LayerKind::Full => self.attend(&mut s, l, &qn, &full_table, &full_lens, rows, bs, max_bt, cap_full),
            };
            let proj = self.matmul(&mut s, &attn, 0, rows, &lb.wo);
            let post = self.dyn_conv(&mut s, &proj, &dyn1, &lb.attn_conv_base, rows, n, 1);
            let xmid = self.add(&mut s, &x, &post, rows * d);

            // ---- MLP sublayer, same convolution sandwich ----
            let xn2 = self.rms(&mut s, &xmid, &lb.post_attn_norm, d, rows);
            let dyn2 = self.matmul(&mut s, &xn2, 0, rows, &lb.mlp_conv_proj);
            let hm = self.dyn_conv(&mut s, &xn2, &dyn2, &lb.mlp_conv_base, rows, n, 0);
            let am = self.act(&mut s, &hm, 0, rows, d, &[&lb.gate, &lb.up]);
            let gate = self.gpu.storage(u64::from(rows) * u64::from(ff));
            self.ops.matmul(&mut s, &lb.gate, &am, &gate, 0);
            let upb = self.gpu.storage(u64::from(rows) * u64::from(ff));
            self.ops.matmul(&mut s, &lb.up, &am, &upb, 0);
            let act_ff = self.gpu.storage(u64::from(rows) * u64::from(ff));
            s.push(self.gpu.step(self.k.silu_mul, &[&gate, &upb, &act_ff], &[rows * ff], rows * ff));
            let dn = self.matmul(&mut s, &act_ff, 0, rows, &lb.down);
            let postm = self.dyn_conv(&mut s, &dn, &dyn2, &lb.mlp_conv_base, rows, n, 1);
            let xnext = self.add(&mut s, &xmid, &postm, rows * d);

            self.gpu.submit(&[], &s);
            self.gpu.poll_wait();
            self.gpu.flush();
            if let Some(t) = trace.as_deref_mut() {
                t.push(self.gpu.read(&xnext, (rows * d) as usize));
            }
            x = xnext;
        }
        let out = self.gpu.read(&x, (rows * d) as usize);
        self.gpu.scratch_release();
        Ok(out.chunks(n as usize * d as usize).map(<[f32]>::to_vec).collect())
    }

    /// [`Self::denoise_blocks`] for one block in slot 0, the single-stream shape the Qwen3.8 path drives.
    pub fn denoise_block(&self, noise: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        Ok(self.denoise_blocks(&[BlockJob { slot: 0, pos, noise }])?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small draft config: a block of five rows, one sliding and one full layer.
    const TINY: &str = r#"{
      "dflash_config": {"block_size": 5, "conv_group_size": 8, "conv_kernel_size": 4, "mask_token_id": 3,
                        "selector_rank": 4, "selector_top_k": 4, "target_layer_ids": [0, 1]},
      "draft_vocab_size": 64, "head_dim": 8, "hidden_size": 32, "intermediate_size": 64, "is_causal": false,
      "layer_types": ["sliding_attention", "full_attention"], "no_rope_layers": [1], "num_attention_heads": 4,
      "num_hidden_layers": 2, "num_key_value_heads": 2, "num_target_layers": 2, "rms_norm_eps": 1e-06,
      "rope_theta": 10000.0, "sliding_window": 6, "use_sliding_window": true, "vocab_size": 64}"#;

    /// A slot holds `cap` context positions and the denoising block past them, so the dense pool must let every slot
    /// reserve `cap + block_size` positions at once: the drafter serves a sequence right up to the cap.
    #[test]
    fn a_dense_pool_holds_every_slot_at_the_cap_and_one_block() {
        let cfg = Dflash2Config::from_json(TINY).unwrap();
        for (slots, cap) in [(1u32, 1u32), (3, 32), (2, 27), (4, 11)] {
            let cache = DraftCache::dense(&cfg, slots, cap);
            let per_slot = (cap + cfg.block_size).div_ceil(cache.block_size);
            assert!(cache.pool_blocks() >= slots * per_slot, "{slots} slots at cap {cap}: {} blocks, need {}", cache.pool_blocks(), slots * per_slot);
        }
    }
}
