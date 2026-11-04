// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tree read: node bucket levels over a [`BucketIndex`]'s buckets and a
//! beam down them, so a query at an ultra-long context scores a bounded set
//! of node buckets and buckets instead of every bucket of every closed block.
//!
//! Swedish Embedded AB implements attention whose read stays bounded as the
//! context grows into the millions of tokens, for clients whose documents and
//! sessions outgrow a flat index. If your team needs expertise in
//! hierarchical sparse attention, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The specification is the authors' `lsa.core.hierarchy` (the reference
//! this module is gated against, `tools/goldens/lsa_dump_reference.py`):
//!
//! - **Levels.** Level 1 pairs every full *node cluster block* of
//!   `node_cluster_block[0]` buckets into node buckets of exactly
//!   `node_bucket_size` children: Ward merges the block's bucket key sums
//!   into pairs (sizes one, a cap of two), then the pairs, until each group
//!   holds `node_bucket_size`. A node bucket's centroid is the normalised sum
//!   of every key beneath it. A level's node buckets are numbered block after
//!   block, each block's in the order of its first child, and the level
//!   above pairs them the same way over blocks of `node_cluster_block[l]`.
//! - **Growth.** A server grows levels only at page ends (`page_size` keys),
//!   and level 1 only once `growth_buckets` uncovered buckets wait; each
//!   level above takes the full blocks of node buckets made below.
//!   [`growth_at`] replays that from the counts, so a query sees the levels
//!   as they stood at its page's start.
//! - **Beam.** A query starts at the highest level grown (or the lowest level
//!   of at most `start_nodes` node buckets, scored whole), keeps the best
//!   `beam_width` node buckets a level - the carried children of the level
//!   above beside the node buckets no level above covers yet - then scores
//!   the last beam's children beside the visible buckets no node bucket
//!   covers, picks the `top_p` best (a tie to the lower row) and expands them
//!   as the flat read does.

use super::select::expand;
use super::ward::ward_labels;
use super::{dot, BucketIndex, Selection};
use serde::{Deserialize, Serialize};

/// The tree's sizes and budgets beside the flat [`super::LsaConfig`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreeConfig {
    /// Children a node bucket holds, a power of two from 2.
    pub node_bucket_size: usize,
    /// Level-below rows node buckets are formed over, one a level (level 1
    /// first); its length is the top level.
    pub node_cluster_block: Vec<usize>,
    /// Node buckets a query keeps, one a level.
    pub beam_width: Vec<usize>,
    /// Where the beam begins: the lowest level of at most this many node
    /// buckets, scored whole; `None` starts at the highest level grown.
    pub start_nodes: Option<usize>,
    /// Keys a page holds: levels grow at page ends, a multiple of the
    /// cluster block.
    pub page_size: usize,
    /// Buckets level 1 takes at once at least.
    pub growth_buckets: usize,
}

impl TreeConfig {
    /// The highest level.
    pub fn top_level(&self) -> usize {
        self.node_cluster_block.len()
    }

    /// Refuse sizes the levels cannot be built or read with.
    pub fn validate(&self, cluster_block: usize) -> Result<(), String> {
        let size = self.node_bucket_size;
        if size < 2 || !size.is_power_of_two() {
            return Err(format!("LSA tree: node_bucket_size must be a power of two from 2, got {size}"));
        }
        if self.node_cluster_block.is_empty() || self.beam_width.len() != self.node_cluster_block.len() {
            return Err(format!("LSA tree: {} node cluster blocks and {} beam widths, one a level", self.node_cluster_block.len(), self.beam_width.len()));
        }
        if let Some(&block) = self.node_cluster_block.iter().find(|&&b| b == 0 || !b.is_multiple_of(size)) {
            return Err(format!("LSA tree: node cluster block {block} is not a positive multiple of {size}"));
        }
        if self.beam_width.contains(&0) {
            return Err("LSA tree: a beam keeps at least one node bucket".into());
        }
        if self.start_nodes == Some(0) || (self.start_nodes.is_some() && self.top_level() < 2) {
            return Err("LSA tree: start_nodes needs a positive count and more than one level".into());
        }
        if self.page_size == 0 || !self.page_size.is_multiple_of(cluster_block) {
            return Err(format!("LSA tree: page_size {} must be a positive multiple of the cluster block {cluster_block}", self.page_size));
        }
        Ok(())
    }
}

/// The node bucket levels over an index's buckets, every full node cluster
/// block of every level paired.
#[derive(Clone, Debug, PartialEq)]
pub struct Hierarchy {
    /// Each level's node centroids, `[nodes, index_dim]`, level 1 first.
    pub node_centroids: Vec<Vec<f32>>,
    /// Each level's children, `[nodes, node_bucket_size]`: bucket rows under
    /// level 1, the level below's node rows above.
    pub children: Vec<Vec<u32>>,
    node_bucket_size: usize,
}

impl Hierarchy {
    /// Node buckets each level holds.
    pub fn nodes(&self) -> Vec<usize> {
        self.children.iter().map(|c| c.len() / self.node_bucket_size).collect()
    }
}

/// Pair each block of `blocks` rows' key sums (`[blocks * block, dim]`, f32)
/// into node buckets of `size`: Ward pairs a block's rows, then the pairs,
/// until each group holds `size`. Returns each row's node bucket within its
/// block, numbered in the order of its first row.
fn pair_into_node_buckets(sums: &[f32], blocks: usize, block: usize, dim: usize, size: usize) -> Result<Vec<u32>, String> {
    let mut labels: Vec<u32> = (0..blocks * block).map(|r| (r % block) as u32).collect();
    let mut level: Vec<f32> = sums.to_vec();
    let mut count = block;
    while count > block / size {
        let half = count / 2;
        let mut next = vec![0f32; blocks * half * dim];
        for b in 0..blocks {
            let rows = &level[b * count * dim..][..count * dim];
            let pairs = ward_labels(rows, &vec![1; count], dim, half, 2)?;
            for (r, &pair) in pairs.iter().enumerate() {
                let into = &mut next[(b * half + pair as usize) * dim..][..dim];
                for (o, x) in into.iter_mut().zip(&rows[r * dim..][..dim]) {
                    *o += x;
                }
            }
            for label in &mut labels[b * block..][..block] {
                *label = pairs[*label as usize];
            }
        }
        level = next;
        count = half;
    }
    Ok(labels)
}

/// One node cluster block's node buckets: the block's rows of the level
/// below paired into node buckets ([`pair_into_node_buckets`]), numbered in
/// the order of their first row.
#[derive(Clone, Debug, PartialEq)]
pub struct NodePage {
    /// `[nodes, dim]` normalised key sums.
    pub centroids: Vec<f32>,
    /// `[nodes, size]` logical rows of the level below, `first` onwards.
    pub children: Vec<u32>,
    /// `[nodes, dim]` key sums, what the level above pairs.
    pub sums: Vec<f32>,
}

/// Build the node page of one node cluster block from its rows' key sums
/// (`[block, dim]`, the first being logical row `first` of the level below).
pub fn node_page(child_sums: &[f32], dim: usize, size: usize, first: u32) -> Result<NodePage, String> {
    let block = child_sums.len() / dim;
    if block == 0 || !block.is_multiple_of(size) || child_sums.len() != block * dim {
        return Err(format!("LSA tree: {} key-sum values are not a block of rows that {size} divides", child_sums.len()));
    }
    let labels = pair_into_node_buckets(child_sums, 1, block, dim, size)?;
    // The block's rows in the order of their node buckets.
    let mut order: Vec<u32> = (0..block as u32).collect();
    order.sort_by_key(|&r| labels[r as usize]);
    let nodes = block / size;
    let mut sums = vec![0f32; nodes * dim];
    for (n, kids) in order.chunks_exact(size).enumerate() {
        for &kid in kids {
            for (o, x) in sums[n * dim..][..dim].iter_mut().zip(&child_sums[kid as usize * dim..][..dim]) {
                *o += x;
            }
        }
    }
    Ok(NodePage { centroids: sums.chunks_exact(dim).flat_map(crate::hostmath::l2_normalize).collect(), children: order.into_iter().map(|r| first + r).collect(), sums })
}

/// Build every level up to the tree's top level over `index`'s buckets,
/// whose keys are `keys` (`[len, index_dim]`, the index's own). A level that
/// fills no node cluster block ends the hierarchy.
pub fn build_hierarchy(index: &BucketIndex, keys: &[f32], tree: &TreeConfig) -> Result<Hierarchy, String> {
    let cfg = index.config();
    tree.validate(cfg.cluster_block)?;
    let dim = cfg.index_dim;
    if keys.len() != index.len() * dim {
        return Err(format!("LSA tree: {} key values for an index of {} keys", keys.len(), index.len()));
    }
    let rows = index.rows();
    let mut sums = vec![0f32; rows * dim];
    for (b, sum) in sums.chunks_exact_mut(dim).enumerate() {
        for &key in &index.members()[b * cfg.bucket_cap..][..index.sizes()[b] as usize] {
            for (s, x) in sum.iter_mut().zip(&keys[key as usize * dim..][..dim]) {
                *s += x;
            }
        }
    }
    let size = tree.node_bucket_size;
    let mut hierarchy = Hierarchy { node_centroids: Vec::new(), children: Vec::new(), node_bucket_size: size };
    for &block in &tree.node_cluster_block {
        let blocks = sums.len() / dim / block;
        if blocks == 0 {
            break;
        }
        let (mut centroids, mut children, mut next) = (Vec::new(), Vec::new(), Vec::new());
        for b in 0..blocks {
            let page = node_page(&sums[b * block * dim..][..block * dim], dim, size, (b * block) as u32)?;
            centroids.extend(page.centroids);
            children.extend(page.children);
            next.extend(page.sums);
        }
        hierarchy.node_centroids.push(centroids);
        hierarchy.children.push(children);
        sums = next;
    }
    Ok(hierarchy)
}

/// The levels as growth at every page end up to some bucket count left them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Growth {
    /// Buckets under a level-1 node bucket, the first ones.
    pub covered_buckets: usize,
    /// Node buckets made a level, the first ones of each.
    pub made: Vec<usize>,
    /// Node buckets a level that a node bucket above covers, the first ones.
    pub closed: Vec<usize>,
}

/// Replay growing at every page end up to `closed_buckets`: at a page end
/// level 1 takes its full node cluster blocks of uncovered buckets once
/// `growth_buckets` of them wait, and each level above takes the full blocks
/// of node buckets made below it and not yet covered.
pub fn growth_at(tree: &TreeConfig, closed_buckets: usize, page_buckets: usize) -> Growth {
    let top = tree.top_level();
    let (size, first) = (tree.node_bucket_size, tree.node_cluster_block[0]);
    let mut g = Growth { covered_buckets: 0, made: vec![0; top], closed: vec![0; top] };
    let mut end = page_buckets;
    while page_buckets > 0 && end <= closed_buckets {
        let blocks = (end - g.covered_buckets) / first;
        if blocks >= (tree.growth_buckets / first).max(1) {
            g.covered_buckets += blocks * first;
            g.made[0] += blocks * first / size;
            for level in 1..top {
                let block = tree.node_cluster_block[level];
                let taken = (g.made[level - 1] - g.closed[level - 1]) / block * block;
                if taken == 0 {
                    break;
                }
                g.closed[level - 1] += taken;
                g.made[level] += taken / size;
            }
        }
        end += page_buckets;
    }
    g
}

/// The `count` best of `rows` by `score` (rows ascending), best first, a tie
/// to the lower row.
fn best(rows: &[u32], score: impl Fn(u32) -> f32, count: usize) -> Vec<u32> {
    let mut ranked: Vec<(f32, u32)> = rows.iter().map(|&r| (score(r), r)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.into_iter().take(count).map(|(_, r)| r).collect()
}

/// One query's buckets by the beam down the levels `growth` holds, best
/// first: every visible bucket when no level has grown.
fn beam_pick(index: &BucketIndex, h: &Hierarchy, tree: &TreeConfig, growth: &Growth, query: &[f32], position: usize) -> Vec<u32> {
    let cfg = index.config();
    let dim = cfg.index_dim;
    let levels = growth.made.iter().take_while(|&&m| m > 0).count();
    let mut start = levels;
    if let Some(nodes) = tree.start_nodes {
        start = (1..=levels).find(|&l| growth.made[l - 1] <= nodes).unwrap_or(levels);
    }
    let mut carried: Vec<u32> = Vec::new();
    for level in (1..=start).rev() {
        let (made, closed) = (growth.made[level - 1], growth.closed[level - 1]);
        let first_open = if level == start && tree.start_nodes.is_some() { 0 } else { closed };
        let mut candidates: Vec<u32> = carried.iter().copied().chain(first_open as u32..made as u32).collect();
        candidates.sort_unstable();
        let centroids = &h.node_centroids[level - 1];
        let kept = best(&candidates, |n| dot(query, &centroids[n as usize * dim..][..dim]) as f32, tree.beam_width[level - 1]);
        let kids = &h.children[level - 1];
        carried = kept.iter().flat_map(|&n| kids[n as usize * h.node_bucket_size..][..h.node_bucket_size].iter().copied()).collect();
    }
    let visible = (position / cfg.cluster_block * cfg.buckets_per_block()).min(index.rows());
    let uncovered = growth.covered_buckets as u32..visible.max(growth.covered_buckets) as u32;
    let mut candidates: Vec<u32> = carried.into_iter().chain(uncovered).collect();
    candidates.sort_unstable();
    best(&candidates, |b| dot(query, &index.centroids()[b as usize * dim..][..dim]) as f32, cfg.top_p)
}

/// Select each query's keys through the tree: its page's levels, the beam,
/// the picked buckets expanded as the flat read expands them, then the window
/// and the sinks. Rows are `top_k + window + sink` wide.
pub fn select_tree(index: &BucketIndex, h: &Hierarchy, tree: &TreeConfig, queries: &[f32], positions: &[u32]) -> Result<Selection, String> {
    let cfg = *index.config();
    tree.validate(cfg.cluster_block)?;
    let dim = cfg.index_dim;
    if queries.len() != positions.len() * dim {
        return Err(format!("LSA tree: {} query values for {} positions", queries.len(), positions.len()));
    }
    if let Some(&p) = positions.iter().find(|&&p| p as usize >= index.len()) {
        return Err(format!("LSA tree: query position {p} is past the {} keys in the index", index.len()));
    }
    let page_buckets = tree.page_size / cfg.cluster_block * cfg.buckets_per_block();
    let width = cfg.selection_width();
    let mut ids = vec![-1i32; positions.len() * width];
    let mut lens = vec![0u32; positions.len()];
    for (r, &position) in positions.iter().enumerate() {
        let position = position as usize;
        let growth = growth_at(tree, position / tree.page_size * page_buckets, page_buckets);
        let picked: Vec<usize> = beam_pick(index, h, tree, &growth, &queries[r * dim..][..dim], position).into_iter().map(|b| b as usize).collect();
        let window_start = (position + 1).saturating_sub(cfg.window);
        lens[r] = expand(index, &picked, position, window_start, cfg.top_k, &mut ids[r * width..][..width]) as u32;
    }
    Ok(Selection { width, ids, lens })
}
