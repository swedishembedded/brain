// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A serving engine's tree: node pages over its index pages, grown at page
//! ends as sequences reach them, shared between sequences over the same
//! cached prefix, and read by `lsa_read_tree`.
//!
//! Swedish Embedded AB implements attention whose read stays bounded as the
//! context grows into the millions of tokens, for clients whose documents and
//! sessions outgrow a flat index. If your team needs expertise in
//! hierarchical sparse attention serving, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! A node page of level `l` holds one node cluster block's node buckets:
//! [`super::tree::node_page`] over the key sums of `node_cluster_block[l]`
//! rows of the level below (buckets under level 1). Its content is fixed by
//! what lies beneath it, so it is keyed by its place in the level and the
//! pages under it - index pages for level 1, node pages of the level below
//! above - and refcounted like index pages: a sequence over the same KV
//! blocks reuses it, the last release frees it. Bucket key sums come from the
//! device Ward (`IndexPool::keep_sums`), read back once per index page; node
//! sums stay on the host for the level above.

use super::device::IndexPool;
use super::tree::{growth_at, node_page, Growth, TreeConfig};
use super::tree_device::{NodePool, TreeReadGeometry};
use super::LsaConfig;
use crate::paged::{BlockAllocator, BlockTable};
use gpu_core::{DeviceBuffer, Gpu};
use std::collections::HashMap;

/// A node page's key: its place in its level and the pages beneath it.
type NodeKey = (u32, Vec<u32>);

/// One level's node pages.
struct Level {
    pages: BlockAllocator,
    page_of: HashMap<NodeKey, u32>,
    key_of: HashMap<u32, NodeKey>,
    /// Per page, `[slots, nodes_per_page, index_dim]` node key sums.
    sums: HashMap<u32, Vec<f32>>,
}

/// The tree of a paged decoder's latent sparse attention.
pub struct TreeEngine {
    pub geom: TreeReadGeometry,
    pub nodes: NodePool,
    levels: Vec<Level>,
    slots: u32,
    /// Per index page, `[slots, buckets_per_block, index_dim]` bucket sums.
    bucket_sums: HashMap<u32, Vec<f32>>,
    /// `[max_rows, table_stride]` table rows of the next read.
    pub tables: DeviceBuffer,
    /// `[max_rows, index_dim]` index queries, then the read's scratch.
    pub cand: DeviceBuffer,
    built: u64,
}

impl TreeEngine {
    /// Node pages for sequences of up to `max_pages` closed blocks, over an
    /// index of `page_capacity` pages, for `slots` sparse layers.
    pub fn new(g: &Gpu, cfg: LsaConfig, tree: TreeConfig, max_pages: u32, page_capacity: u32, slots: u32, max_rows: u32) -> Result<TreeEngine, String> {
        let geom = TreeReadGeometry { cfg, tree, max_pages };
        geom.check()?;
        let caps = Self::capacities(&geom, page_capacity);
        let nodes = NodePool::new(g, geom.clone(), caps.clone(), slots)?;
        let levels = caps.iter().map(|&c| Level { pages: BlockAllocator::new(c, 1), page_of: HashMap::new(), key_of: HashMap::new(), sums: HashMap::new() }).collect();
        let rows = u64::from(max_rows.max(1));
        Ok(TreeEngine {
            tables: g.storage(rows * u64::from(geom.table_stride())),
            cand: g.storage(super::tree_device::cand_words(&geom, max_rows.max(1))),
            geom,
            nodes,
            levels,
            slots,
            bucket_sums: HashMap::new(),
            built: 0,
        })
    }

    /// Node pages each level holds for an index of `page_capacity` pages:
    /// every bucket under one node page a level, plus one.
    fn capacities(geom: &TreeReadGeometry, page_capacity: u32) -> Vec<u32> {
        let t = &geom.tree;
        let buckets = page_capacity as usize * geom.cfg.buckets_per_block();
        let mut under = 1usize;
        t.node_cluster_block
            .iter()
            .map(|&block| {
                let pages = buckets / (block * under) + 1;
                under *= t.node_bucket_size;
                pages as u32
            })
            .collect()
    }

    /// Device bytes [`Self::new`] allocates, before allocating: the node pool,
    /// the kept bucket sums, the tables and the scratch.
    pub fn bytes(cfg: &LsaConfig, tree: &TreeConfig, max_pages: u32, page_capacity: u32, slots: u32, max_rows: u32) -> Result<u64, String> {
        let geom = TreeReadGeometry { cfg: *cfg, tree: tree.clone(), max_pages };
        geom.check()?;
        let nodes: u64 = Self::capacities(&geom, page_capacity).iter().enumerate().map(|(l, &c)| u64::from(c) * u64::from(geom.nodes_per_page(l + 1))).sum();
        let row = (cfg.index_dim + tree.node_bucket_size) as u64;
        let sums = u64::from(page_capacity) * cfg.buckets_per_block() as u64 * cfg.index_dim as u64;
        let rows = u64::from(max_rows.max(1));
        Ok(4 * (nodes * row * u64::from(slots) + sums * u64::from(slots) + rows * u64::from(geom.table_stride()) + super::tree_device::cand_words(&geom, max_rows.max(1))))
    }

    /// Node pages built since the engine started (reuse excluded).
    pub fn built_pages(&self) -> u64 {
        self.built
    }

    /// The levels as they stood at the start of the page of a query at
    /// `position`, over `built` index pages.
    pub fn growth(&self, position: u32, built: usize) -> Growth {
        let c = &self.geom.cfg;
        let page_buckets = self.geom.tree.page_size / c.cluster_block * c.buckets_per_block();
        let at_page = position as usize / self.geom.tree.page_size * page_buckets;
        let closed = at_page.min(built * c.buckets_per_block() / page_buckets * page_buckets);
        growth_at(&self.geom.tree, closed, page_buckets)
    }

    /// The node pages of `table` as `growth` needs them: a page whose
    /// underlying pages changed (a truncated or diverged sequence) is let
    /// go, a missing one reused from another sequence or built.
    pub fn grow(&mut self, g: &Gpu, pool: &IndexPool, table: &mut BlockTable, growth: &Growth) -> Result<(), String> {
        for level in 1..=self.geom.tree.top_level() {
            let want = growth.made[level - 1] / self.geom.nodes_per_page(level) as usize;
            // Keep the leading pages whose key still matches.
            let mut keep = 0;
            while keep < table.node_pages(level).len().min(want) {
                let page = table.node_pages(level)[keep];
                if self.levels[level - 1].key_of.get(&page) != Some(&self.key(table, level, keep)) {
                    break;
                }
                keep += 1;
            }
            for page in table.truncate_node_pages(level, keep) {
                self.drop_page(level, page);
            }
            while table.node_pages(level).len() < want {
                let p = table.node_pages(level).len();
                let key = self.key(table, level, p);
                let page = match self.levels[level - 1].page_of.get(&key) {
                    Some(&page) => {
                        self.levels[level - 1].pages.incref(page);
                        page
                    }
                    None => {
                        let page = self.levels[level - 1].pages.alloc().ok_or_else(|| format!("LSA tree: all level-{level} node pages are in use"))?;
                        if let Err(e) = self.build(g, pool, level, p, page, &key.1) {
                            self.levels[level - 1].pages.decref(page);
                            return Err(e);
                        }
                        let lv = &mut self.levels[level - 1];
                        lv.page_of.insert(key.clone(), page);
                        lv.key_of.insert(page, key);
                        self.built += 1;
                        page
                    }
                };
                table.push_node_page(level, page);
            }
        }
        // Levels above the growth's let go of anything left.
        for level in 1..=self.geom.tree.top_level() {
            let want = growth.made[level - 1] / self.geom.nodes_per_page(level) as usize;
            for page in table.truncate_node_pages(level, want) {
                self.drop_page(level, page);
            }
        }
        Ok(())
    }

    /// The key of node page `p` of `level`: `p` and the pages under it - the
    /// index pages (level 1) or the level below's node pages its rows lie in.
    fn key(&self, table: &BlockTable, level: usize, p: usize) -> NodeKey {
        let block = self.geom.tree.node_cluster_block[level - 1];
        let per = if level == 1 { self.geom.cfg.buckets_per_block() } else { self.geom.nodes_per_page(level - 1) as usize };
        let under = if level == 1 { table.index_pages() } else { table.node_pages(level - 1) };
        let range = p * block / per..((p + 1) * block).div_ceil(per);
        (p as u32, under.get(range).map_or_else(Vec::new, <[u32]>::to_vec))
    }

    /// Build node page `p` of `level` into `page`, for every slot, from the
    /// key sums of the rows beneath it (`under`: their pages).
    fn build(&mut self, g: &Gpu, pool: &IndexPool, level: usize, p: usize, page: u32, under: &[u32]) -> Result<(), String> {
        let d = self.geom.cfg.index_dim;
        let block = self.geom.tree.node_cluster_block[level - 1];
        let size = self.geom.tree.node_bucket_size;
        let per = if level == 1 { self.geom.cfg.buckets_per_block() } else { self.geom.nodes_per_page(level - 1) as usize };
        let first = p * block;
        let mut child: Vec<&[f32]> = Vec::with_capacity(under.len());
        if level == 1 {
            for &ip in under {
                if !self.bucket_sums.contains_key(&ip) {
                    let sums = pool.sums.as_ref().ok_or("LSA tree: the index keeps no bucket sums")?;
                    let words = self.slots as usize * per * d;
                    let mut host = vec![0f32; words];
                    g.read_into(sums, u64::from(ip) * words as u64, &mut host);
                    self.bucket_sums.insert(ip, host);
                }
            }
            child.extend(under.iter().map(|ip| &self.bucket_sums[ip][..]));
        } else {
            let below = &self.levels[level - 2].sums;
            for np in under {
                child.push(below.get(np).ok_or_else(|| format!("LSA tree: level-{} node page {np} has no sums", level - 1))?);
            }
        }
        if child.len() * per < (first % per) + block {
            return Err(format!("LSA tree: level-{level} node page {p} lies over {} pages, too few", child.len()));
        }
        // Each slot's block of rows, pairing them into node buckets.
        let npp = block / size;
        let mut sums = vec![0f32; self.slots as usize * npp * d];
        for slot in 0..self.slots as usize {
            let mut rows = Vec::with_capacity(block * d);
            for r in first..first + block {
                let (pg, off) = (r / per - first / per, r % per);
                rows.extend_from_slice(&child[pg][(slot * per + off) * d..][..d]);
            }
            let node = node_page(&rows, d, size, first as u32)?;
            self.nodes.write(g, slot as u32, level, page, &node.centroids, &node.children)?;
            sums[slot * npp * d..][..npp * d].copy_from_slice(&node.sums);
        }
        self.levels[level - 1].sums.insert(page, sums);
        Ok(())
    }

    fn drop_page(&mut self, level: usize, page: u32) {
        let lv = &mut self.levels[level - 1];
        lv.pages.decref(page);
        if lv.pages.refcount(page) == 0 {
            if let Some(key) = lv.key_of.remove(&page) {
                lv.page_of.remove(&key);
            }
            lv.sums.remove(&page);
        }
    }

    /// An index page was freed: its bucket sums go too.
    pub fn index_page_freed(&mut self, page: u32) {
        self.bucket_sums.remove(&page);
    }

    /// Release a finished sequence's node pages.
    pub fn release(&mut self, table: &mut BlockTable) {
        for (l, pages) in table.take_node_pages().into_iter().enumerate() {
            for page in pages {
                self.drop_page(l + 1, page);
            }
        }
    }

    /// Node pages in use, every level.
    pub fn pages_in_use(&self) -> u32 {
        self.levels.iter().map(|l| l.pages.num_blocks() - l.pages.free_blocks()).sum()
    }
}
