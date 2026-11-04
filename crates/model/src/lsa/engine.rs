// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Latent sparse attention inside a paged decoder: what a serving engine
//! holds to read its attention layers sparsely, and the two things it calls.
//!
//! * [`LsaAttention::record_layer`] records one layer's attention as three
//!   dispatches over the engine's own buffers - index queries from the
//!   rotated queries, the read, and sparse attention over the paged KV
//!   cache. They depend on shapes alone, so a recorded decode tape stays
//!   valid across steps: the page tables and live lengths they read are
//!   buffer contents.
//! * [`LsaAttention::prepare`] brings a sequence's index up to date before a
//!   forward: every closed cluster block gets an index page, computed from
//!   the keys already in the KV pool (never by recomputing a token), and the
//!   rows' page tables are written. A page is keyed by its cluster block and
//!   the physical KV blocks holding it, so sequences that share a cached
//!   prefix share its pages, refcounted.
//!
//! Memory: per page and layer `buckets_per_block * (index_dim + bucket_cap)`
//! words (20 KiB at the paper's sizes, for 512 tokens), plus bounded scratch
//! sized by the engine's widest forward.

use super::device::{IndexHeads, IndexPool, IndexVectorBuffers, ReadBuffers, ReadGeometry};
use super::indexer::Indexer;
use super::{cluster_block, ClosedBlock, LsaConfig};
use crate::paged::{BlockAllocator, BlockTable};
use crate::paged_sparse::{SparseAttention, SparseBuffers};
use gpu_core::{DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

/// Pipeline indices of the LSA kernels in the engine's kernel list.
#[derive(Clone, Copy, Debug)]
pub struct LsaKernels {
    /// `lsa_index_vectors`.
    pub index_vectors: usize,
    /// `lsa_read`.
    pub read: usize,
    /// The sparse attention kernel: `paged_sparse_flash_gqa` specialised for
    /// the model's group (`per_kv_head`) or `paged_sparse_flash_decode`.
    pub attend: usize,
    /// Whether `attend` is the per-KV-head kernel.
    pub per_kv_head: bool,
}

/// The paged KV cache the engine attends over.
#[derive(Clone, Copy, Debug)]
pub struct KvGeometry {
    /// Keys per KV block.
    pub block_size: u32,
    /// Row stride of the engine's block tables, in blocks.
    pub max_blocks_per_seq: u32,
}

/// One forward's buffers, owned by the engine.
pub struct LayerBuffers<'a> {
    /// `[rows, n_heads * head_dim]` rotated queries.
    pub q: &'a DeviceBuffer,
    /// `[rows]` query positions.
    pub pos: &'a DeviceBuffer,
    /// `[rows]` live keys of each row's sequence.
    pub seqlen: &'a DeviceBuffer,
    /// `[rows, max_blocks_per_seq]` block tables.
    pub block_tables: &'a DeviceBuffer,
    /// The layer's key pool.
    pub pool_k: &'a DeviceBuffer,
    /// The layer's value pool.
    pub pool_v: &'a DeviceBuffer,
    /// `[rows, n_heads * head_dim]` attention context out.
    pub ctx: &'a DeviceBuffer,
}

struct Layer {
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    pool: IndexPool,
}

/// Index pages per page key: a cluster block and the KV blocks holding it.
type PageKey = (u32, Vec<u32>);

/// A paged decoder's latent sparse attention: projections, index pages and
/// scratch for every sparse layer.
pub struct LsaAttention {
    cfg: LsaConfig,
    heads: IndexHeads,
    n_heads: u32,
    kv: KvGeometry,
    k: LsaKernels,
    layers: Vec<Option<Layer>>,
    read: ReadGeometry,
    max_rows: u32,
    pages: BlockAllocator,
    page_of: HashMap<PageKey, u32>,
    key_of: HashMap<u32, PageKey>,
    // Forward scratch, sized for `max_rows` query rows.
    q_off: DeviceBuffer,
    iq: DeviceBuffer,
    scores: DeviceBuffer,
    sel: DeviceBuffer,
    row_meta: DeviceBuffer,
    page_tab: DeviceBuffer,
    // Index-page build scratch: one cluster block's keys per layer.
    key_off: DeviceBuffer,
    key_pos: DeviceBuffer,
    keys: Vec<DeviceBuffer>,
    built_pages: u64,
}

impl LsaAttention {
    /// Attach `indexer` to an engine: upload its projections and allocate
    /// pages for `page_capacity` closed cluster blocks and scratch for
    /// forwards of up to `max_rows` rows.
    pub fn new(g: &Gpu, indexer: &Indexer, kernels: LsaKernels, kv: KvGeometry, max_rows: u32, page_capacity: u32) -> Result<LsaAttention, String> {
        indexer.validate()?;
        let cfg = indexer.cfg;
        let geo = indexer.geometry;
        let heads = IndexHeads { n_kv_heads: geo.n_kv_heads as u32, head_dim: geo.head_dim as u32, rope_base: geo.rope_base, index_dim: cfg.index_dim as u32 };
        heads.check(geo.n_heads as u32)?;
        if kv.block_size == 0 || !cfg.cluster_block.is_multiple_of(kv.block_size as usize) {
            return Err(format!("LSA: cluster_block {} is not a whole number of KV blocks of {}", cfg.cluster_block, kv.block_size));
        }
        let max_pages = (kv.max_blocks_per_seq * kv.block_size) / cfg.cluster_block as u32;
        let read = ReadGeometry { cfg, max_pages };
        read.check()?;
        let mut layers = Vec::with_capacity(indexer.layers.len());
        for p in &indexer.layers {
            layers.push(match p {
                Some(p) => Some(Layer { wq: g.storage_init("lsa.wq", &p.wq), wk: g.storage_init("lsa.wk", &p.wk), pool: IndexPool::new(g, cfg, page_capacity)? }),
                None => None,
            });
        }
        let rows = max_rows as u64;
        let d = cfg.index_dim as u64;
        let q_row = geo.n_heads as u64 * geo.head_dim as u64;
        let q_off: Vec<u32> = (0..max_rows).map(|r| (r as u64 * q_row) as u32).collect();
        let q_off_buf = g.storage(rows.max(1));
        g.write(&q_off_buf, &q_off);
        let block = cfg.cluster_block as u64;
        let keys = layers.iter().map(|l| g.storage(if l.is_some() { block * d } else { 1 })).collect();
        Ok(LsaAttention {
            cfg,
            heads,
            n_heads: geo.n_heads as u32,
            kv,
            k: kernels,
            layers,
            read,
            max_rows,
            pages: BlockAllocator::new(page_capacity, 1),
            page_of: HashMap::new(),
            key_of: HashMap::new(),
            q_off: q_off_buf,
            iq: g.storage((rows * d).max(1)),
            scores: g.storage((rows * read.score_stride() as u64).max(1)),
            sel: g.storage((rows * read.selection_width() as u64).max(1)),
            row_meta: g.storage((rows * 3).max(1)),
            page_tab: g.storage((rows * (max_pages as u64 + 1)).max(1)),
            key_off: g.storage(block),
            key_pos: g.storage(block),
            keys,
            built_pages: 0,
        })
    }

    /// The index configuration.
    pub fn config(&self) -> &LsaConfig {
        &self.cfg
    }

    /// Whether `layer` reads sparsely.
    pub fn sparse(&self, layer: usize) -> bool {
        self.layers.get(layer).is_some_and(Option::is_some)
    }

    /// Device bytes held: projections, index pages and scratch.
    pub fn device_bytes(&self) -> u64 {
        let d = self.cfg.index_dim as u64;
        let input = self.heads.n_kv_heads as u64 * self.heads.head_dim as u64;
        let sparse = self.layers.iter().filter(|l| l.is_some()).count() as u64;
        let per_layer = 2 * input * d * 4 + IndexPool::bytes(&self.cfg, self.pages.num_blocks()) + self.cfg.cluster_block as u64 * d * 4;
        let rows = self.max_rows as u64;
        let scratch = 4 * rows * (1 + d + self.read.score_stride() as u64 + self.read.selection_width() as u64 + 3 + self.read.max_pages as u64 + 1) + 8 * self.cfg.cluster_block as u64;
        sparse * per_layer + scratch
    }

    /// Index pages built since the engine started (page reuse excluded).
    pub fn built_pages(&self) -> u64 {
        self.built_pages
    }

    /// Pages in use / capacity.
    pub fn page_usage(&self) -> (u32, u32) {
        (self.pages.num_blocks() - self.pages.free_blocks(), self.pages.num_blocks())
    }

    /// Record `layer`'s attention for `rows` query rows: index queries, the
    /// read, sparse attention into `b.ctx`. The rows' page tables must have
    /// been written by [`Self::prepare`] for this forward.
    pub fn record_layer(&self, g: &Gpu, s: &mut Vec<Step>, layer: usize, rows: u32, b: &LayerBuffers) -> Result<(), String> {
        let l = self.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| format!("LSA: layer {layer} is dense"))?;
        if rows > self.max_rows {
            return Err(format!("LSA: a forward of {rows} rows exceeds the {} the scratch holds", self.max_rows));
        }
        s.push(self.heads.step(g, self.k.index_vectors, self.n_heads, rows, &IndexVectorBuffers { x: b.q, row_off: &self.q_off, pos: b.pos, w: &l.wq, out: &self.iq })?);
        s.push(self.read.step(
            g,
            self.k.read,
            rows,
            &ReadBuffers { iq: &self.iq, centroids: &l.pool.centroids, members: &l.pool.members, page_tab: &self.page_tab, seqlen: b.seqlen, scores: &self.scores, sel: &self.sel, row_meta: &self.row_meta },
        )?);
        let attn = SparseAttention { n_heads: self.n_heads, n_kv_heads: self.heads.n_kv_heads, head_dim: self.heads.head_dim, block_size: self.kv.block_size, max_bt: self.kv.max_blocks_per_seq };
        let bufs = SparseBuffers { q: b.q, pool_k: b.pool_k, pool_v: b.pool_v, block_tables: b.block_tables, sel: &self.sel, row_meta: &self.row_meta, ctx: b.ctx };
        s.push(attn.step(g, self.k.attend, self.k.per_kv_head, rows, self.read.selection_width(), &bufs)?);
        Ok(())
    }

    /// Read back the last recorded layer's selection of `rows` rows: per row
    /// its selected positions, and the `[rows, 3]` sparse-attention metadata.
    /// A diagnostic (selection statistics, gates); it synchronises the device.
    pub fn read_selection(&self, g: &Gpu, rows: u32) -> (Vec<Vec<u32>>, Vec<u32>) {
        let width = self.read.selection_width() as usize;
        let meta: Vec<u32> = g.read(&self.row_meta, rows as usize * 3).iter().map(|x| x.to_bits()).collect();
        let sel: Vec<u32> = g.read(&self.sel, rows as usize * width).iter().map(|x| x.to_bits()).collect();
        let rows = (0..rows as usize).map(|r| sel[r * width..][..meta[r * 3] as usize].to_vec()).collect();
        (rows, meta)
    }

    /// Bring each sequence's index up to date for a forward and write the
    /// page tables of its rows. `rows[i] = (table, cached, count)`: the
    /// sequence's table, how many of its keys are written in the pool, and
    /// how many consecutive forward rows it owns (1 for a decode row, a
    /// chunk's length for prefill). `pool_k` is the engine's key pool per
    /// layer.
    ///
    /// Builds a page for every closed cluster block among the `cached` keys
    /// that has none (the keys are read from the pool: nothing is
    /// recomputed), reusing a page another sequence already built over the
    /// same KV blocks, and frees the pages of blocks a truncation removed.
    pub fn prepare(&mut self, g: &Gpu, rows: &mut [(&mut BlockTable, u32, u32)], pool_k: &[DeviceBuffer]) -> Result<(), String> {
        let total: u32 = rows.iter().map(|r| r.2).sum();
        if total > self.max_rows {
            return Err(format!("LSA: a forward of {total} rows exceeds the {} the scratch holds", self.max_rows));
        }
        for (table, cached, _) in rows.iter_mut() {
            self.index_table(g, table, *cached, pool_k)?;
        }
        let width = self.read.max_pages as usize + 1;
        let mut tab = vec![0u32; total as usize * width];
        let mut r = 0usize;
        for (table, _, count) in rows.iter() {
            let pages = table.index_pages();
            for _ in 0..*count {
                let row = &mut tab[r * width..][..width];
                row[0] = pages.len() as u32;
                row[1..=pages.len()].copy_from_slice(pages);
                r += 1;
            }
        }
        g.write(&self.page_tab, &tab);
        Ok(())
    }

    /// Release a finished sequence's pages (before its KV blocks are released).
    pub fn release(&mut self, table: &mut BlockTable) {
        for page in table.take_index_pages() {
            self.drop_page(page);
        }
    }

    fn drop_page(&mut self, page: u32) {
        self.pages.decref(page);
        if self.pages.refcount(page) == 0 {
            if let Some(key) = self.key_of.remove(&page) {
                self.page_of.remove(&key);
            }
        }
    }

    fn index_table(&mut self, g: &Gpu, table: &mut BlockTable, cached: u32, pool_k: &[DeviceBuffer]) -> Result<(), String> {
        let cb = self.cfg.cluster_block as u32;
        let closed = (cached.min(table.len()) / cb) as usize;
        if closed > self.read.max_pages as usize {
            return Err(format!("LSA: {closed} closed blocks exceed the {} pages a page table holds", self.read.max_pages));
        }
        for page in table.truncate_index_pages(closed) {
            self.drop_page(page);
        }
        let per_cluster = (cb / self.kv.block_size) as usize;
        while table.index_pages().len() < closed {
            let c = table.index_pages().len();
            let key: PageKey = (c as u32, table.blocks()[c * per_cluster..(c + 1) * per_cluster].to_vec());
            let page = match self.page_of.get(&key) {
                Some(&page) => {
                    self.pages.incref(page);
                    page
                }
                None => {
                    let page = self.pages.alloc().ok_or_else(|| format!("LSA: all {} index pages are in use", self.pages.num_blocks()))?;
                    if let Err(e) = self.build_page(g, table, c, page, pool_k) {
                        self.pages.decref(page);
                        return Err(e);
                    }
                    self.page_of.insert(key.clone(), page);
                    self.key_of.insert(page, key);
                    self.built_pages += 1;
                    page
                }
            };
            table.push_index_page(page);
        }
        Ok(())
    }

    /// Compute cluster block `c`'s index keys from the pool on every sparse
    /// layer, Ward-merge them on the host (layers in parallel) and upload the
    /// buckets into `page`.
    fn build_page(&self, g: &Gpu, table: &BlockTable, c: usize, page: u32, pool_k: &[DeviceBuffer]) -> Result<(), String> {
        let cb = self.cfg.cluster_block as u32;
        let bs = self.kv.block_size;
        let kv_row = self.heads.n_kv_heads * self.heads.head_dim;
        let base = c as u32 * cb;
        let (off, pos): (Vec<u32>, Vec<u32>) = (base..base + cb)
            .map(|p| {
                let (block, offset) = table.locate(p, bs);
                ((block * bs + offset) * kv_row, p)
            })
            .unzip();
        g.write(&self.key_off, &off);
        g.write(&self.key_pos, &pos);
        let mut steps = Vec::new();
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(layer) = layer {
                let pk = pool_k.get(l).ok_or_else(|| format!("LSA: no key pool for layer {l}"))?;
                steps.push(self.heads.step(g, self.k.index_vectors, self.heads.n_kv_heads, cb, &IndexVectorBuffers { x: pk, row_off: &self.key_off, pos: &self.key_pos, w: &layer.wk, out: &self.keys[l] })?);
            }
        }
        g.submit(&[], &steps);
        let d = self.cfg.index_dim;
        let sparse: Vec<usize> = (0..self.layers.len()).filter(|&l| self.sparse(l)).collect();
        let keys: Vec<Vec<f32>> = sparse.iter().map(|&l| g.read(&self.keys[l], cb as usize * d)).collect();
        let mut blocks: Vec<Option<Result<ClosedBlock, String>>> = vec![None; sparse.len()];
        let cfg = self.cfg;
        backend_cpu::par::chunks_mut(&mut blocks, 1, |i, slot| {
            slot[0] = Some(cluster_block(&cfg, &keys[i], base as usize));
        });
        for (&l, block) in sparse.iter().zip(blocks) {
            let block = block.expect("every layer was clustered")?;
            self.layers[l].as_ref().expect("sparse layer").pool.write(g, page, &block)?;
        }
        Ok(())
    }
}
