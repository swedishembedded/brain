// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The engine-agnostic half of attaching latent sparse attention to a paged
//! decoder: which kernels an engine of a given geometry registers and resolves,
//! the attachment itself, the per-forward state (the measured dense crossover,
//! whether the forward in flight reads densely, chunk limits, statistics), page
//! lifetime on release, and the memory charge.
//!
//! Swedish Embedded AB implements long-context serving whose attention reads a
//! bounded set of keys per token for its clients. If your team needs expertise
//! in sparse attention for production LLM serving, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! An engine supplies its pools, block tables, positions and geometry, and asks
//! the binding to record a sparse layer read ([`LsaBinding::record_layer`]); it
//! keeps everything model-specific: the forward, the pool layout, which layers
//! may read sparsely, and the dense kernel it falls back to.
//!
//! # Kernels by geometry
//!
//! [`kernel_pipelines`] lists the kernels an engine registers when it opens its
//! device for a set of attention geometries; [`resolve_kernels`] picks, among
//! those registered, the per-KV-head kernel of the geometry's group (any group
//! the engine registered, not a fixed family), the whole-row kernel where the
//! geometry has one, and the per-head kernel otherwise. An engine whose device
//! list is fixed before its model is known (Qwen3) registers the geometries of
//! its family; one that builds its list from the loaded config (Ripple)
//! registers exactly its own.

use std::cell::Cell;

use super::device::{self, Rope};
use super::engine::{KeyPools, KvGeometry, LayerBuffers, LsaAttention, LsaKernels, LsaStats};
use super::indexer::{IndexGeometry, Indexer};
use super::tree_device;
use crate::paged::BlockTable;
use crate::paged_sparse::{self, Layout, PackedKv};
use gpu_core::{DeviceBuffer, Gpu, Step};

/// One attention geometry an engine reads sparsely: query heads, KV heads and
/// head width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadShape {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

impl HeadShape {
    /// Query heads per KV head.
    pub fn group(&self) -> u32 {
        self.n_heads / self.n_kv_heads
    }
}

/// The kernels an engine registers for sparse attention over `shapes` (per-KV-head and whole-row kernels) and the
/// groups in `extra_groups` (per-KV-head only), with pools of each tier in `packed` (none: an f32 pool). Appended to the engine's
/// own list; the engine resolves them by name ([`resolve_kernels`]).
pub fn kernel_pipelines(shapes: &[HeadShape], extra_groups: &[u32], packed: &[PackedKv], tree: bool) -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&'static str, &'static str)> = device::KERNELS.to_vec();
    if tree {
        v.push(tree_device::KERNEL);
    }
    v.push(paged_sparse::KERNEL);
    for &kv in packed {
        v.push(device::index_keys_kernel(kv).expect("lsa_index_keys_i8 marks its byte decode"));
    }
    let mut groups: Vec<u32> = shapes.iter().map(HeadShape::group).chain(extra_groups.iter().copied()).collect();
    groups.sort_unstable();
    groups.dedup();
    for group in groups {
        v.push(paged_sparse::gqa_kernel(group).expect("paged_sparse_flash_gqa declares the GROUP knob"));
        for &kv in packed {
            v.push(paged_sparse::gqa_kernel_kv(group, kv).expect("paged_sparse_flash_gqa_i8 declares GROUP and marks its byte decode"));
        }
    }
    for s in shapes {
        for &kv in packed {
            if let Ok(k) = paged_sparse::rows_kernel_kv(s.n_heads, s.n_kv_heads, s.head_dim, kv) {
                v.push(k);
            }
        }
    }
    v.dedup_by_key(|k| k.0);
    v
}

/// Resolve the kernel indices of an engine whose device registered
/// [`kernel_pipelines`] for `shape` over a pool of tier `packed`.
///
/// Decode-sized forwards read per KV head where that group's kernel is
/// registered (per query head otherwise, f32 pools only); prefill-sized
/// forwards over a packed pool read a whole row per workgroup where the shape
/// has that kernel registered.
pub fn resolve_kernels(g: &Gpu, shape: HeadShape, packed: Option<PackedKv>, tree: bool) -> Result<LsaKernels, String> {
    let name = |n: &str| g.kernel_index(n).ok_or_else(|| format!("latent sparse attention: kernel {n} is not registered"));
    let group = shape.group();
    let registered = |n: &str| g.kernel_index(n);
    let (few, layout_few, index_keys_packed) = match packed {
        Some(kv) => {
            let k = paged_sparse::gqa_kernel_kv(group, kv)?.0;
            let few = registered(k).ok_or_else(|| format!("latent sparse attention over a packed KV cache needs the group-{group} kernel registered ({k}); serve with --kv-cache f32"))?;
            (few, Layout::PerKvHead, Some(name(device::index_keys_kernel(kv)?.0)?))
        }
        None => match registered(paged_sparse::gqa_kernel(group)?.0) {
            Some(k) => (k, Layout::PerKvHead, None),
            None => (name(paged_sparse::KERNEL.0)?, Layout::PerHead, None),
        },
    };
    let whole_row = packed.and_then(|kv| paged_sparse::rows_kernel_kv(shape.n_heads, shape.n_kv_heads, shape.head_dim, kv).ok()).and_then(|(n, _)| g.kernel_index(n));
    let (attend, layout) = whole_row.map_or((few, layout_few), |k| (k, Layout::PerRow));
    Ok(LsaKernels {
        index_vectors: name(device::KERNELS[0].0)?,
        read: name(device::KERNELS[1].0)?,
        ward: name(device::KERNELS[2].0)?,
        tree: if tree { Some(name(tree_device::KERNEL.0)?) } else { None },
        attend,
        layout,
        attend_few: few,
        layout_few,
        index_keys_packed,
    })
}

/// Index pages an engine allocates for latent sparse attention over a pool of
/// `num_blocks` KV blocks: shared prefixes share pages, and sequences that
/// diverge inside a cluster block each need their own, hence the margin over
/// one page per pool cluster block.
pub fn page_capacity(num_blocks: u32, block_size: u32, cluster_block: u32) -> u32 {
    2 * (num_blocks * block_size).div_ceil(cluster_block) + 1
}

/// What an engine tells the binding about itself when it attaches an indexer.
pub struct Attachment<'a> {
    /// The engine's attention geometry; `rope_base` 0 for layers without RoPE.
    pub geometry: IndexGeometry,
    /// The KV pool's tier.
    pub packed: Option<PackedKv>,
    /// Block size and table stride of the paged KV cache.
    pub kv: KvGeometry,
    /// The most rows one forward carries (scratch size).
    pub max_rows: u32,
    /// Pages for the pool's cluster blocks ([`page_capacity`]).
    pub page_capacity: u32,
    /// The engine's YaRN inverse-frequency table, when its heads are rotated under it.
    pub yarn_inv_freq: Option<&'a DeviceBuffer>,
    /// Which layers this engine can read sparsely at all: an indexer covering
    /// another layer is refused. `None`: every layer.
    pub sparse_capable: Option<&'a [bool]>,
}

impl Attachment<'_> {
    fn check(&self, indexer: &Indexer) -> Result<(), String> {
        indexer.check_model(&self.geometry)?;
        if let Some(capable) = self.sparse_capable {
            for (l, p) in indexer.layers.iter().enumerate() {
                if p.is_some() && !capable.get(l).copied().unwrap_or(false) {
                    return Err(format!("LSA: the indexer covers layer {l}, which this model reads densely by design (a sliding-window layer reads its window)"));
                }
            }
        }
        Ok(())
    }
}

/// A paged decoder's latent sparse attention plus the per-forward state every
/// engine keeps beside it.
pub struct LsaBinding {
    attn: LsaAttention,
    /// Prefill chunks ending at or below this position read densely.
    dense_until: u32,
    /// Whether the forward in flight reads densely.
    dense_now: Cell<bool>,
    /// Prefill rows read densely since attaching.
    dense_rows: u64,
}

impl LsaBinding {
    /// Attach `indexer`, refusing a geometry, rope or layer coverage the engine
    /// does not match, and resolve nothing else: the caller resolves kernels
    /// ([`resolve_kernels`]) and passes them in.
    pub fn attach(g: &Gpu, indexer: &Indexer, kernels: LsaKernels, a: &Attachment) -> Result<LsaBinding, String> {
        Self::attach_in_domain(g, indexer, kernels, a, 0)
    }

    /// [`Self::attach`] for one of several bindings that ride the same block tables: an engine whose layers are
    /// split over devices attaches one binding per device, each with the indexer restricted to its layers
    /// ([`Indexer::restricted_to`]) and its own `domain`, so the tables hold each device's page ids apart
    /// ([`BlockTable::index_pages_in`]). Every binding is driven with the same tables, chunks and releases.
    pub fn attach_in_domain(g: &Gpu, indexer: &Indexer, kernels: LsaKernels, a: &Attachment, domain: usize) -> Result<LsaBinding, String> {
        a.check(indexer)?;
        let mut attn = LsaAttention::new(g, indexer, kernels, a.kv, a.max_rows, a.page_capacity, a.yarn_inv_freq)?;
        attn.set_domain(domain)?;
        Ok(LsaBinding { attn, dense_until: 0, dense_now: Cell::new(false), dense_rows: 0 })
    }

    /// [`Self::device_bytes_for`] split per sparse layer and per device (`LsaAttention::device_bytes_split`).
    pub fn device_bytes_split(indexer: &Indexer, kv: KvGeometry, max_rows: u32, num_blocks: u32) -> (u64, u64) {
        let pages = page_capacity(num_blocks, kv.block_size, indexer.cfg.cluster_block as u32);
        LsaAttention::device_bytes_split(indexer, kv, max_rows, pages)
    }

    /// The device bytes attaching `indexer` allocates, before allocating: what
    /// a memory planner charges.
    pub fn device_bytes_for(indexer: &Indexer, kv: KvGeometry, max_rows: u32, num_blocks: u32) -> u64 {
        let pages = page_capacity(num_blocks, kv.block_size, indexer.cfg.cluster_block as u32);
        LsaAttention::device_bytes_for(indexer, kv, max_rows, pages)
    }

    /// The attention machinery, for timing measurements and diagnostics.
    pub fn attention(&self) -> &LsaAttention {
        &self.attn
    }

    /// The rotation the index undoes for a geometry: none where the heads are
    /// unrotated.
    pub fn rope_of(geometry: &IndexGeometry, yarn: bool) -> Rope {
        match (yarn, geometry.rope_base == 0.0) {
            (true, _) => Rope::Table,
            (false, true) => Rope::None,
            (false, false) => Rope::Base(geometry.rope_base),
        }
    }

    /// Whether `layer` reads sparsely in the forward in flight.
    pub fn reads_sparsely(&self, layer: usize) -> bool {
        self.attn.sparse(layer) && !self.dense_now.get()
    }

    /// Record `layer`'s sparse attention for `rows` rows.
    pub fn record_layer(&self, g: &Gpu, s: &mut Vec<Step>, layer: usize, rows: u32, b: &LayerBuffers) -> Result<(), String> {
        self.attn.record_layer(g, s, layer, rows, b)
    }

    /// The longest chunk starting at `start` that stays inside one cluster
    /// block, so every query's closed blocks are indexed before it runs.
    pub fn chunk_limit(&self, start: u32) -> u32 {
        let cb = self.attn.config().cluster_block as u32;
        cb - start % cb
    }

    /// Set up a prefill chunk of `rows` consecutive positions after `table`'s
    /// `cached` written keys: a chunk ending at or below the crossover reads
    /// densely and leaves the index alone (the first sparse chunk or decode step
    /// builds every closed block's page it lacks); any other brings the index up
    /// to date.
    pub fn prepare_chunk(&mut self, g: &Gpu, table: &mut BlockTable, cached: u32, rows: u32, pools: KeyPools) -> Result<(), String> {
        let dense = cached + rows <= self.dense_until;
        self.dense_now.set(dense);
        if dense {
            self.dense_rows += u64::from(rows);
            return Ok(());
        }
        self.attn.prepare(g, &mut [(table, cached, rows)], pools)
    }

    /// Bring the index up to date for a decode step (or any forward of
    /// sequences with `rows[i] = (table, cached, count)`); always sparse.
    pub fn prepare_rows(&mut self, g: &Gpu, rows: &mut [(&mut BlockTable, u32, u32)], pools: KeyPools) -> Result<(), String> {
        self.dense_now.set(false);
        self.attn.prepare(g, rows, pools)
    }

    /// The forward set up by [`Self::prepare_chunk`] has run.
    pub fn end_forward(&self) {
        self.dense_now.set(false);
    }

    /// Free a sequence's index (and node) pages, before its KV blocks are released.
    pub fn release(&mut self, table: &mut BlockTable) {
        self.attn.release(table);
    }

    /// Let `to` (an empty table over the blocks of `from`, e.g. a prefix
    /// snapshot's fork) hold the index pages `from` holds, so they outlive
    /// `from` for as long as the blocks do.
    pub fn retain_pages(&mut self, from: &BlockTable, to: &mut BlockTable) -> Result<(), String> {
        self.attn.retain_pages(from, to)
    }

    /// The prefill position up to which chunks read densely.
    pub fn dense_until(&self) -> u32 {
        self.dense_until
    }

    /// Override the measured crossover (0: every chunk reads sparsely).
    pub fn set_dense_until(&mut self, position: u32) {
        self.dense_until = position;
    }

    /// What latent sparse attention holds and has done (one small readback).
    pub fn stats(&self, g: &Gpu) -> LsaStats {
        let (pages_in_use, page_capacity) = self.attn.page_usage();
        LsaStats { device_bytes: self.attn.device_bytes(), pages_in_use, page_capacity, pages_built: self.attn.built_pages(), reads: self.attn.read_stats(g), dense_rows: self.dense_rows }
    }

    /// Zero the read counters and the dense-row count (after a measurement
    /// whose reads served no row).
    pub fn reset_stats(&mut self, g: &Gpu) {
        self.attn.reset_read_stats(g);
        self.dense_rows = 0;
    }

    /// Measure where sparse attention starts to pay on this device and store it
    /// as the crossover: for a prefill chunk of `rows` rows starting at a
    /// growing position, time one layer's dense chunk attention against its
    /// sparse read, on the engine's own buffers. Prefill chunks ending at or
    /// below the last position at which dense was still faster read densely.
    /// The engine supplies the three things that are its own:
    ///
    /// * `place(start, rows)`: write the positions, live lengths and block
    ///   tables of `rows` rows at `start..` into its forward buffers;
    /// * `dense(rows)`: the dense chunk attention step it would run instead;
    /// * `buffers()`: the layer's [`LayerBuffers`].
    ///
    /// An engine with no dense fused prefill to compare against passes
    /// `compare: false` and gets 0 (sparse from the start).
    #[allow(clippy::too_many_arguments)]
    pub fn measure_crossover<'b>(&mut self, g: &Gpu, layer: usize, rows: u32, max_seq_len: u32, compare: bool, mut place: impl FnMut(u32, u32), mut dense: impl FnMut(u32) -> Step, buffers: impl Fn() -> LayerBuffers<'b>) {
        let dense_until = (|| {
            if !compare {
                return 0;
            }
            let cb = self.attn.config().cluster_block as u32;
            let rows = rows.min(cb).max(1);
            let mut dense_until = 0;
            let mut start = 1024u32;
            while start + rows <= max_seq_len {
                place(start, rows);
                let dense_time = gpu_core::profile::best_of(g, &[dense(rows)], 2);
                // The sparse side as a real chunk reads it: index queries and the read over `start / cb`
                // closed blocks, then attention over a full-width selection spread over the cached keys.
                self.attn.write_page_tables_for_timing(g, rows, start / cb);
                let bufs = buffers();
                let mut read = Vec::new();
                let Ok(attend) = self.attn.record_read(g, &mut read, layer, rows, &bufs).and_then(|()| self.attn.attend_step(g, rows, &bufs)) else {
                    return dense_until;
                };
                let read_time = gpu_core::profile::best_of(g, &read, 2);
                self.attn.write_selection_for_timing(g, rows, start);
                let sparse = read_time + gpu_core::profile::best_of(g, &[attend], 2);
                if sparse < dense_time {
                    break;
                }
                dense_until = start + rows;
                // Positions about a quarter apart, in whole chunks.
                start += (start / 4).div_ceil(rows) * rows;
            }
            dense_until
        })();
        self.dense_until = dense_until;
        self.reset_stats(g);
    }
}
