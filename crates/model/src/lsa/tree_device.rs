// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tree read on the device (`lsa_read_tree`): the beam of
//! [`super::tree::select_tree`] over node bucket levels held in a paged node
//! pool, written as the selection rows sparse attention reads.
//!
//! Swedish Embedded AB implements attention whose read stays bounded as the
//! context grows into the millions of tokens, for clients whose documents and
//! sessions outgrow a flat index. If your team needs expertise in GPU beam
//! search over hierarchical indexes, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! A query row's *table row* tells the kernel what its sequence holds: its
//! built index pages and their pages (the flat read's page-table row), the
//! levels as they stood at its page's start ([`super::tree::Growth`]), and
//! each level's node pages; its live keys come from the forward's `seqlen`,
//! so a decode window advanced on the device reads current positions. A node page holds one node cluster
//! block's node buckets (`node_cluster_block / node_bucket_size` of them); a
//! node row is its centroid, then its children - logical rows of the level
//! below - as u32 bits.

use super::device::{ReadStats, MAX_INDEX_DIM, MAX_TOP_P};
use super::tree::{Growth, TreeConfig};
use super::LsaConfig;
use gpu_core::{DeviceBuffer, Gpu, Step};

/// The kernel, for a model's pipeline list.
pub const KERNEL: (&str, &str) = ("lsa_read_tree", kernels::LSA_READ_TREE);

/// Most levels the kernel holds.
pub const MAX_LEVELS: usize = 8;

/// The shape of a tree read: the flat configuration, the tree, and the most
/// pages a sequence holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeReadGeometry {
    pub cfg: LsaConfig,
    pub tree: TreeConfig,
    /// Index pages a table row holds (closed cluster blocks).
    pub max_pages: u32,
}

/// The buffers one tree read binds.
pub struct TreeReadBuffers<'a> {
    /// `[rows]` live keys of each row's sequence.
    pub seqlen: &'a DeviceBuffer,
    /// The index pool's bucket centroids.
    pub centroids: &'a DeviceBuffer,
    /// The index pool's bucket members.
    pub members: &'a DeviceBuffer,
    /// The node pool ([`NodePool::nodes`]).
    pub nodes: &'a DeviceBuffer,
    /// `[rows, table_stride]` table rows ([`TreeReadGeometry::write_row`]).
    pub tables: &'a DeviceBuffer,
    /// `[rows, index_dim]` unit index queries (f32 bits), then
    /// `[rows, 2 * cand_stride]` scratch: [`cand_words`].
    pub cand: &'a DeviceBuffer,
    /// `[rows, row_stride]` out: selection rows.
    pub sel: &'a DeviceBuffer,
    /// [`ReadStats::WORDS`] counters.
    pub stats: &'a DeviceBuffer,
    /// The layer's first bucket row in the index pool.
    pub row0: u32,
    /// The layer's first node row of each level in the node pool.
    pub level_row0: &'a [u32],
}

impl TreeReadGeometry {
    /// Refuse a shape the kernel cannot run.
    pub fn check(&self) -> Result<(), String> {
        self.cfg.validate()?;
        self.tree.validate(self.cfg.cluster_block)?;
        if self.tree.top_level() > MAX_LEVELS {
            return Err(format!("LSA tree read: {} levels exceed {MAX_LEVELS}", self.tree.top_level()));
        }
        if self.cfg.top_p > MAX_TOP_P || self.cfg.index_dim > MAX_INDEX_DIM {
            return Err(format!("LSA tree read: top_p {} / index_dim {} exceed {MAX_TOP_P} / {MAX_INDEX_DIM}", self.cfg.top_p, self.cfg.index_dim));
        }
        if self.tree.beam_width.iter().any(|&b| b > 2 * MAX_TOP_P) {
            return Err(format!("LSA tree read: a beam keeps at most {} node buckets", 2 * MAX_TOP_P));
        }
        if self.cfg.early.is_some() {
            return Err("LSA tree read: an early budget is a flat read's".into());
        }
        Ok(())
    }

    /// Node buckets a node page of `level` (from 1) holds.
    pub fn nodes_per_page(&self, level: usize) -> u32 {
        (self.tree.node_cluster_block[level - 1] / self.tree.node_bucket_size) as u32
    }

    /// Node pages of `level` a sequence of `max_pages` closed blocks makes at
    /// most: its rows of the level below over the level's block.
    pub fn max_node_pages(&self, level: usize) -> u32 {
        let mut rows = self.max_pages as usize * self.cfg.buckets_per_block();
        for l in 1..level {
            rows = rows / self.tree.node_cluster_block[l - 1] * self.tree.node_cluster_block[l - 1] / self.tree.node_bucket_size;
        }
        (rows / self.tree.node_cluster_block[level - 1]) as u32
    }

    /// Words before the growth counts in a table row: built pages, the
    /// pages.
    fn growth_off(&self) -> u32 {
        1 + self.max_pages
    }

    /// Word where `level`'s node page table starts in a table row.
    fn tab_off(&self, level: usize) -> u32 {
        let mut off = self.growth_off() + 1 + 2 * self.tree.top_level() as u32;
        for l in 1..level {
            off += self.max_node_pages(l);
        }
        off
    }

    /// Words per table row.
    pub fn table_stride(&self) -> u32 {
        self.tab_off(self.tree.top_level()) + self.max_node_pages(self.tree.top_level())
    }

    /// Candidates a row may score at once: the top level's node buckets (or
    /// the start level's), a level's carried children beside the node
    /// buckets no full block above has taken, the last beam's children beside
    /// the buckets no node bucket covers yet.
    pub fn cand_stride(&self) -> u32 {
        let t = &self.tree;
        let size = t.node_bucket_size;
        let top = t.top_level();
        let max_buckets = self.max_pages as usize * self.cfg.buckets_per_block();
        let page_buckets = t.page_size / self.cfg.cluster_block * self.cfg.buckets_per_block();
        let mut bound = t.beam_width[0] * size + t.growth_buckets.max(t.node_cluster_block[0]) + page_buckets + t.node_cluster_block[0];
        for l in 1..top {
            bound = bound.max(t.beam_width[l] * size + t.node_cluster_block[l]);
        }
        let top_nodes = max_buckets / size.pow(top as u32);
        bound = bound.max(top_nodes.max(1)).max(t.start_nodes.unwrap_or(0));
        bound.min(max_buckets.max(1) + t.beam_width.iter().max().copied().unwrap_or(0) * size) as u32
    }

    /// Words per selection row: the header, then the flat read's positions.
    pub fn row_stride(&self) -> u32 {
        super::device::ReadGeometry { cfg: self.cfg, max_pages: self.max_pages }.row_stride()
    }

    /// Fill one table row: the row's built index pages, the levels as its
    /// page saw them, and each level's node pages.
    pub fn write_row(&self, row: &mut [u32], pages: &[u32], growth: &Growth, node_pages: &[Vec<u32>]) -> Result<(), String> {
        if row.len() != self.table_stride() as usize || pages.len() > self.max_pages as usize {
            return Err(format!("LSA tree read: a table row of {} words for {} pages", row.len(), pages.len()));
        }
        row.fill(0);
        row[0] = pages.len() as u32;
        row[1..1 + pages.len()].copy_from_slice(pages);
        let g = self.growth_off() as usize;
        row[g] = growth.covered_buckets as u32;
        for l in 0..self.tree.top_level() {
            row[g + 1 + 2 * l] = growth.made.get(l).copied().unwrap_or(0) as u32;
            row[g + 2 + 2 * l] = growth.closed.get(l).copied().unwrap_or(0) as u32;
            let want = growth.made.get(l).copied().unwrap_or(0).div_ceil(self.nodes_per_page(l + 1) as usize);
            let held = node_pages.get(l).map_or(&[][..], |v| &v[..]);
            if held.len() < want || held.len() > self.max_node_pages(l + 1) as usize {
                return Err(format!("LSA tree read: level {} holds {} node pages, its growth needs {want}", l + 1, held.len()));
            }
            let off = self.tab_off(l + 1) as usize;
            row[off..off + held.len()].copy_from_slice(held);
        }
        Ok(())
    }

    /// Record the read of `rows` query rows. `kernel` is the caller's
    /// pipeline index of `lsa_read_tree`.
    pub fn step(&self, g: &Gpu, kernel: usize, rows: u32, b: &TreeReadBuffers) -> Result<Step, String> {
        self.check()?;
        let (c, t) = (&self.cfg, &self.tree);
        if b.level_row0.len() != t.top_level() {
            return Err(format!("LSA tree read: {} level offsets for {} levels", b.level_row0.len(), t.top_level()));
        }
        let mut params = vec![
            rows,
            c.index_dim as u32,
            c.bucket_cap as u32,
            c.cluster_block as u32,
            c.buckets_per_block() as u32,
            c.top_p as u32,
            c.top_k as u32,
            c.sink as u32,
            c.window as u32,
            self.row_stride(),
            self.max_pages,
            self.table_stride(),
            self.growth_off(),
            t.top_level() as u32,
            t.node_bucket_size as u32,
            t.start_nodes.unwrap_or(0) as u32,
            self.cand_stride(),
            b.row0,
            0,
            0,
        ];
        for l in 0..MAX_LEVELS {
            if l < t.top_level() {
                params.extend([self.tab_off(l + 1), self.nodes_per_page(l + 1), t.beam_width[l] as u32, b.level_row0[l]]);
            } else {
                params.extend([0, 1, 0, 0]);
            }
        }
        Ok(g.dispatch(kernel, &[b.seqlen, b.centroids, b.members, b.nodes, b.tables, b.cand, b.sel, b.stats], &params, gpu_core::Dispatch::Workgroups(rows)))
    }
}

/// Node bucket levels on the device for `slots` layers: per level, room for
/// `pages[l]` node pages a slot, a row being the centroid then the children.
pub struct NodePool {
    pub nodes: DeviceBuffer,
    geometry: TreeReadGeometry,
    pages: Vec<u32>,
    slots: u32,
}

impl NodePool {
    /// Room for `pages[l]` node pages of every level `l` in each of `slots`.
    pub fn new(g: &Gpu, geometry: TreeReadGeometry, pages: Vec<u32>, slots: u32) -> Result<NodePool, String> {
        if pages.len() != geometry.tree.top_level() {
            return Err(format!("LSA node pool: {} page counts for {} levels", pages.len(), geometry.tree.top_level()));
        }
        let pool = NodePool { nodes: g.storage(1), geometry, pages, slots };
        let words = u64::from(pool.slot_rows()) * u64::from(slots) * u64::from(pool.row_words());
        Ok(NodePool { nodes: g.storage(words.max(1)), ..pool })
    }

    /// Node rows one slot holds, every level.
    fn slot_rows(&self) -> u32 {
        (0..self.pages.len()).map(|l| self.pages[l] * self.geometry.nodes_per_page(l + 1)).sum()
    }

    /// Device bytes of the pool.
    pub fn bytes(&self) -> u64 {
        4 * u64::from(self.slot_rows()) * u64::from(self.slots) * u64::from(self.row_words())
    }

    /// Words of a node row.
    pub fn row_words(&self) -> u32 {
        (self.geometry.cfg.index_dim + self.geometry.tree.node_bucket_size) as u32
    }

    /// The first node row of each level of `slot`.
    pub fn level_row0(&self, slot: u32) -> Vec<u32> {
        let mut row = slot * self.slot_rows();
        (0..self.pages.len())
            .map(|l| {
                let at = row;
                row += self.pages[l] * self.geometry.nodes_per_page(l + 1);
                at
            })
            .collect()
    }

    /// Upload one node page of `level` (from 1) into `page` of `slot`:
    /// `centroids` `[nodes, index_dim]`, `children` `[nodes, node_bucket_size]`.
    pub fn write(&self, g: &Gpu, slot: u32, level: usize, page: u32, centroids: &[f32], children: &[u32]) -> Result<(), String> {
        let (d, size) = (self.geometry.cfg.index_dim, self.geometry.tree.node_bucket_size);
        let npp = self.geometry.nodes_per_page(level) as usize;
        if slot >= self.slots || level == 0 || level > self.pages.len() || page >= self.pages[level - 1] || centroids.len() != npp * d || children.len() != npp * size {
            return Err(format!("LSA node pool: page {page} of level {level}, slot {slot} is outside the pool or mis-sized"));
        }
        let mut words = Vec::with_capacity(npp * (d + size));
        for n in 0..npp {
            words.extend(centroids[n * d..][..d].iter().map(|x| x.to_bits()));
            words.extend(&children[n * size..][..size]);
        }
        let row = self.level_row0(slot)[level - 1] + page * npp as u32;
        g.write_at(&self.nodes, row as u64 * self.row_words() as u64, &words);
        Ok(())
    }
}

/// Words of scratch a read of `rows` rows needs ([`TreeReadBuffers::cand`]):
/// their index queries, then two candidate lists a row.
pub fn cand_words(geometry: &TreeReadGeometry, rows: u32) -> u64 {
    (geometry.cfg.index_dim as u64 + 2 * u64::from(geometry.cand_stride())) * u64::from(rows)
}

/// Stats words, re-exported for callers sizing [`TreeReadBuffers::stats`].
pub const STATS_WORDS: u64 = ReadStats::WORDS;
