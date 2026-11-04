// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device side of latent sparse attention: index vectors from cached
//! attention heads, the paged bucket index, and the read that turns a query
//! into the key positions sparse attention reads.
//!
//! The index is a cache derived from the KV cache. A closed cluster block's
//! buckets are computed from the keys already in the paged pool (un-rotated,
//! projected, Ward-merged on the host) into one *index page*; a sequence's
//! pages are listed in its [`crate::paged::BlockTable`] and live in an
//! [`IndexPool`] shared by all sequences. Nothing about a token is ever
//! recomputed to build or rebuild its page, so a prefix-cache hit, a forked
//! or truncated table, or a released page is repaired from the KV cache
//! alone, and cached-token accounting is untouched.
//!
//! Per layer, the device holds: projections `[n_kv_heads * head_dim,
//! index_dim]` for queries and keys, and per page `buckets_per_block`
//! centroids and member lists - about 40 bytes per cached token per layer
//! against kilobytes of K/V.

use super::{ClosedBlock, LsaConfig};
use gpu_core::{DeviceBuffer, Gpu, Step};

/// The kernels, for a caller's pipeline list: index vectors, the read, and
/// the device Ward that builds an index page.
pub const KERNELS: [(&str, &str); 3] = [("lsa_index_vectors", kernels::LSA_INDEX_VECTORS), ("lsa_read", kernels::LSA_READ), ("lsa_ward", kernels::LSA_WARD)];

/// The index-key kernel over a packed KV pool ([`crate::paged_sparse::PackedKv`]).
pub fn index_keys_kernel(kv: crate::paged_sparse::PackedKv) -> Result<(&'static str, &'static str), String> {
    let int8 = ("lsa_index_keys_i8", kernels::LSA_INDEX_KEYS_I8);
    match kv {
        crate::paged_sparse::PackedKv::Int8 => Ok(int8),
        crate::paged_sparse::PackedKv::Fp8 => kernels::template::fp8_decode_variant(int8.0, int8.1),
    }
}

/// Most buckets a query may pick on the device (the read kernel's `P_MAX`).
pub const MAX_TOP_P: usize = 1024;
/// Widest index vector the device kernels hold.
pub const MAX_INDEX_DIM: usize = 64;
/// Widest head-summed input the index-vector kernel holds (its `MAX_IN`).
pub const MAX_INDEX_INPUT: usize = 2048;
/// A member slot past a bucket's size.
pub const NO_KEY: u32 = u32::MAX;

/// How the heads an index reads were rotated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rope {
    /// Not rotated.
    None,
    /// Half-split RoPE at this base.
    Base(f32),
    /// Half-split RoPE by a per-channel inverse-frequency table bound with
    /// the dispatch (YaRN).
    Table,
}

/// The attention heads index vectors are computed from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexHeads {
    /// Key/value heads; the index input is `n_kv_heads * head_dim` wide.
    pub n_kv_heads: u32,
    /// Channels per head.
    pub head_dim: u32,
    /// The rotation to undo.
    pub rope: Rope,
    /// Index vector width.
    pub index_dim: u32,
}

/// The buffers one index-vector dispatch binds.
pub struct IndexVectorBuffers<'a> {
    /// Rows of `heads` rotated heads, anywhere in this buffer.
    pub x: &'a DeviceBuffer,
    /// `[rows]` element offset of each row in `x`.
    pub row_off: &'a DeviceBuffer,
    /// `[rows]` position each row was rotated at.
    pub pos: &'a DeviceBuffer,
    /// `[n_kv_heads * head_dim, index_dim]` projection.
    pub w: &'a DeviceBuffer,
    /// `[rows, index_dim]` unit index vectors out.
    pub out: &'a DeviceBuffer,
    /// `[head_dim / 2]` inverse frequencies for [`Rope::Table`] (any buffer
    /// otherwise; it is not read).
    pub inv_freq: &'a DeviceBuffer,
    /// Row of `out` the first vector is written to.
    pub out0: u32,
}

impl IndexHeads {
    fn rope_params(&self) -> [u32; 2] {
        match self.rope {
            Rope::None => [0f32.to_bits(), 0],
            Rope::Base(base) => [base.to_bits(), 1],
            Rope::Table => [0f32.to_bits(), 2],
        }
    }

    /// Refuse a geometry the kernel cannot run.
    pub fn check(&self, heads: u32) -> Result<(), String> {
        if self.n_kv_heads == 0 || !heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!("LSA index vectors: {heads} heads do not group over {} KV heads", self.n_kv_heads));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(format!("LSA index vectors: head_dim {} must be even and positive", self.head_dim));
        }
        if (self.n_kv_heads * self.head_dim) as usize > MAX_INDEX_INPUT {
            return Err(format!("LSA index vectors: {} input channels exceed {MAX_INDEX_INPUT}", self.n_kv_heads * self.head_dim));
        }
        if self.index_dim == 0 || self.index_dim as usize > MAX_INDEX_DIM {
            return Err(format!("LSA index vectors: index_dim {} is outside 1..={MAX_INDEX_DIM}", self.index_dim));
        }
        Ok(())
    }

    /// Record index vectors for `rows` rows of `heads` heads each (query rows:
    /// every query head; key rows: the KV heads). `kernel` is the caller's
    /// pipeline index of `lsa_index_vectors`.
    pub fn step(&self, g: &Gpu, kernel: usize, heads: u32, rows: u32, b: &IndexVectorBuffers) -> Result<Step, String> {
        self.check(heads)?;
        let [base, mode] = self.rope_params();
        let params = [rows, heads, self.n_kv_heads, self.head_dim, self.index_dim, base, mode, b.out0];
        Ok(g.dispatch(kernel, &[b.x, b.row_off, b.pos, b.w, b.out, b.inv_freq], &params, gpu_core::Dispatch::Workgroups(rows)))
    }
}

/// The buffers one packed-pool index-key dispatch binds.
pub struct PackedKeyBuffers<'a> {
    /// The packed key pool, `[slots, n_kv_heads * head_dim / 4]` u32.
    pub pool: &'a DeviceBuffer,
    /// `[slots, n_kv_heads]` f32 scales.
    pub scales: &'a DeviceBuffer,
    /// `[rows]` token slot of each key.
    pub slots: &'a DeviceBuffer,
    /// `[rows]` position each key was rotated at.
    pub pos: &'a DeviceBuffer,
    /// `[n_kv_heads * head_dim, index_dim]` projection.
    pub w: &'a DeviceBuffer,
    /// `[rows, index_dim]` out.
    pub out: &'a DeviceBuffer,
    /// As [`IndexVectorBuffers::inv_freq`].
    pub inv_freq: &'a DeviceBuffer,
    /// As [`IndexVectorBuffers::out0`].
    pub out0: u32,
}

impl IndexHeads {
    /// Record index keys for `rows` keys read from a packed pool by token
    /// slot. `kernel` is the caller's index of [`index_keys_kernel`].
    pub fn step_packed_keys(&self, g: &Gpu, kernel: usize, rows: u32, b: &PackedKeyBuffers) -> Result<Step, String> {
        self.check(self.n_kv_heads)?;
        if !self.head_dim.is_multiple_of(4) {
            return Err(format!("LSA index keys: a packed pool needs head_dim % 4 == 0, got {}", self.head_dim));
        }
        let [base, mode] = self.rope_params();
        let params = [rows, self.n_kv_heads, self.n_kv_heads, self.head_dim, self.index_dim, base, mode, b.out0];
        Ok(g.dispatch(kernel, &[b.pool, b.slots, b.pos, b.w, b.out, b.scales, b.inv_freq], &params, gpu_core::Dispatch::Workgroups(rows)))
    }
}

/// The index pages of every sparse layer on the device: `slots` layer slots
/// of `pages * buckets_per_block` rows of centroids and member lists, one
/// buffer each, so one dispatch can build a block's page for every layer.
pub struct IndexPool {
    cfg: LsaConfig,
    pages: u32,
    slots: u32,
    /// `[pages * buckets_per_block, index_dim]` f32.
    pub centroids: DeviceBuffer,
    /// `[pages * buckets_per_block, bucket_cap]` u32 positions, [`NO_KEY`]
    /// padded.
    pub members: DeviceBuffer,
}

impl IndexPool {
    /// Pages for `pages` closed cluster blocks in each of `slots` layers.
    pub fn new(g: &Gpu, cfg: LsaConfig, pages: u32, slots: u32) -> Result<IndexPool, String> {
        cfg.validate()?;
        let rows = slots as u64 * pages as u64 * cfg.buckets_per_block() as u64;
        Ok(IndexPool {
            cfg,
            pages,
            slots,
            centroids: g.storage((rows * cfg.index_dim as u64).max(1)),
            members: g.storage((rows * cfg.bucket_cap as u64).max(1)),
        })
    }

    /// Pages each layer slot holds.
    pub fn pages(&self) -> u32 {
        self.pages
    }

    /// The first bucket row of layer `slot`.
    pub fn row0(&self, slot: u32) -> u32 {
        slot * self.pages * self.cfg.buckets_per_block() as u32
    }

    /// Device bytes of `pages` pages of one layer under `cfg`.
    pub fn bytes(cfg: &LsaConfig, pages: u32) -> u64 {
        pages as u64 * cfg.buckets_per_block() as u64 * (cfg.index_dim + cfg.bucket_cap) as u64 * 4
    }

    /// Words of scratch one [`Self::ward_step`] dispatch needs: the block's
    /// pool sums and their Gram matrix.
    pub fn ward_scratch_words(cfg: &LsaConfig) -> u64 {
        let pools = cfg.pools_per_block() as u64;
        pools * cfg.index_dim as u64 + pools * pools
    }

    /// Record the device Ward of one closed block in every layer slot at
    /// once: `keys` holds each slot's `[cluster_block, index_dim]` unit index
    /// keys, slot after slot (the first key at position `base`), and the
    /// buckets are written into `page` of each slot. `kernel` is the caller's
    /// index of `lsa_ward`; `scratch` holds `slots *`
    /// [`Self::ward_scratch_words`].
    pub fn ward_step(&self, g: &Gpu, kernel: usize, keys: &DeviceBuffer, scratch: &DeviceBuffer, page: u32, base: u32) -> Result<Step, String> {
        let c = &self.cfg;
        if page >= self.pages {
            return Err(format!("LSA index page {page} is outside the pool's {}", self.pages));
        }
        if c.pools_per_block() > 256 || c.buckets_per_block() > 128 {
            return Err(format!("LSA device Ward: {} pools into {} buckets exceeds 256 / 128", c.pools_per_block(), c.buckets_per_block()));
        }
        let params = [c.index_dim as u32, c.cluster_block as u32, c.pool_size as u32, c.buckets_per_block() as u32, c.bucket_cap as u32, page, base, self.pages];
        Ok(g.dispatch(kernel, &[keys, scratch, &self.centroids, &self.members], &params, gpu_core::Dispatch::Workgroups(self.slots)))
    }

    /// Upload one closed block's buckets into `page` of layer `slot`.
    pub fn write(&self, g: &Gpu, page: u32, slot: u32, block: &ClosedBlock) -> Result<(), String> {
        if page >= self.pages || slot >= self.slots {
            return Err(format!("LSA index page {page} of slot {slot} is outside the pool's {} x {}", self.slots, self.pages));
        }
        let bpb = self.cfg.buckets_per_block() as u64;
        let page = page + slot * self.pages;
        let centroids: Vec<u32> = block.centroids.iter().map(|x| x.to_bits()).collect();
        let members: Vec<u32> = block.members.iter().map(|&m| if m < 0 { NO_KEY } else { m as u32 }).collect();
        g.write_at(&self.centroids, page as u64 * bpb * self.cfg.index_dim as u64, &centroids);
        g.write_at(&self.members, page as u64 * bpb * self.cfg.bucket_cap as u64, &members);
        Ok(())
    }
}

/// The geometry of a read: the configuration and the per-row page tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadGeometry {
    /// Index configuration.
    pub cfg: LsaConfig,
    /// Row stride of the page tables, in pages (each row is `1 + max_pages`
    /// words: the count of built pages, then the pages).
    pub max_pages: u32,
}

/// The buffers one read binds.
pub struct ReadBuffers<'a> {
    /// `[rows, index_dim]` index queries.
    pub iq: &'a DeviceBuffer,
    /// The layer's [`IndexPool::centroids`].
    pub centroids: &'a DeviceBuffer,
    /// The layer's [`IndexPool::members`].
    pub members: &'a DeviceBuffer,
    /// `[rows, 1 + max_pages]`: built pages, then each closed block's page.
    pub page_tab: &'a DeviceBuffer,
    /// `[rows]` live keys of each row's sequence (its position plus one).
    pub seqlen: &'a DeviceBuffer,
    /// `[rows, score_stride]` scratch.
    pub scores: &'a DeviceBuffer,
    /// `[rows, row_stride]` out: each row's header, then its positions.
    pub sel: &'a DeviceBuffer,
    /// The layer's first bucket row in the pool ([`IndexPool::row0`]).
    pub row0: u32,
}

impl ReadGeometry {
    /// Most positions a selection row holds: long-range keys, the window,
    /// one cluster block whose page is not built yet (read through the widened
    /// window), sinks.
    pub fn selection_width(&self) -> u32 {
        (self.cfg.selection_width() + self.cfg.cluster_block) as u32
    }

    /// Words per selection row: the sparse attention header, then positions.
    pub fn row_stride(&self) -> u32 {
        self.selection_width() + crate::paged_sparse::ROW_HEADER
    }

    /// Scores a row holds: every bucket of every page it may list.
    pub fn score_stride(&self) -> u32 {
        self.max_pages * self.cfg.buckets_per_block() as u32
    }

    /// Refuse a configuration the read kernel cannot run.
    pub fn check(&self) -> Result<(), String> {
        self.cfg.validate()?;
        let early_p = self.cfg.early.map_or(0, |e| e.top_p);
        if self.cfg.top_p.max(early_p) > MAX_TOP_P {
            return Err(format!("LSA device read: top_p {} exceeds {MAX_TOP_P}", self.cfg.top_p.max(early_p)));
        }
        if self.cfg.index_dim > MAX_INDEX_DIM {
            return Err(format!("LSA device read: index_dim {} exceeds {MAX_INDEX_DIM}", self.cfg.index_dim));
        }
        Ok(())
    }

    /// Record the read of `rows` query rows. `kernel` is the caller's pipeline
    /// index of `lsa_read`.
    pub fn step(&self, g: &Gpu, kernel: usize, rows: u32, b: &ReadBuffers) -> Result<Step, String> {
        self.check()?;
        let c = &self.cfg;
        let (early_position, early_top_p, early_top_k) = c.early.map_or((0, 0, 0), |e| (e.position as u32, e.top_p as u32, e.top_k as u32));
        let params = [
            rows,
            self.score_stride(),
            self.max_pages,
            c.index_dim as u32,
            c.bucket_cap as u32,
            c.cluster_block as u32,
            c.buckets_per_block() as u32,
            c.top_p as u32,
            c.top_k as u32,
            c.sink as u32,
            c.window as u32,
            self.row_stride(),
            early_position,
            early_top_p,
            early_top_k,
            b.row0,
        ];
        let bufs = [b.iq, b.centroids, b.members, b.page_tab, b.seqlen, b.scores, b.sel];
        Ok(g.dispatch(kernel, &bufs, &params, gpu_core::Dispatch::Workgroups(rows)))
    }
}
