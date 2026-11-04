// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The bucket index of one sequence's keys, for one attention layer.
//!
//! Append-only: keys arrive in position order, and each time a cluster block
//! of `cluster_block` keys is complete it is Ward-merged into buckets that
//! never change afterwards. A closed block's buckets therefore depend only on
//! that block's keys, which is what lets a shared prompt prefix share its
//! closed blocks. Keys of the still-open block are held until it closes; a
//! query reads them through its window.
//!
//! The index stores logical key positions, never physical KV slots: the paged
//! KV cache owns where a key's bytes live, and resolves a position through the
//! sequence's block table when attention reads it.

use super::{check_unit_rows, ward, LsaConfig};

/// The buckets of a sequence's closed cluster blocks, block after block.
#[derive(Clone, Debug)]
pub struct BucketIndex {
    cfg: LsaConfig,
    /// Unit centroid of each bucket, `[rows, index_dim]`.
    centroids: Vec<f32>,
    /// Key positions of each bucket in ascending order, `[rows, bucket_cap]`,
    /// `-1` past the bucket's size.
    members: Vec<i32>,
    /// Keys in each bucket, `[rows]`.
    sizes: Vec<u32>,
    /// Keys of the open block, `[< cluster_block, index_dim]`.
    open: Vec<f32>,
}

impl BucketIndex {
    /// An empty index; fails on a configuration that does not validate.
    pub fn new(cfg: LsaConfig) -> Result<BucketIndex, String> {
        cfg.validate()?;
        Ok(BucketIndex { cfg, centroids: Vec::new(), members: Vec::new(), sizes: Vec::new(), open: Vec::new() })
    }

    /// An index over `keys` (`[T, index_dim]`, unit rows).
    pub fn build(cfg: LsaConfig, keys: &[f32]) -> Result<BucketIndex, String> {
        let mut index = BucketIndex::new(cfg)?;
        index.push_keys(keys)?;
        Ok(index)
    }

    /// The configuration this index was built under.
    pub fn config(&self) -> &LsaConfig {
        &self.cfg
    }

    /// Keys appended so far: the next key's position.
    pub fn len(&self) -> usize {
        self.closed_blocks() * self.cfg.cluster_block + self.open.len() / self.cfg.index_dim
    }

    /// Whether no key has been appended.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cluster blocks that have closed into buckets.
    pub fn closed_blocks(&self) -> usize {
        self.sizes.len() / self.cfg.buckets_per_block()
    }

    /// Buckets of the closed blocks.
    pub fn rows(&self) -> usize {
        self.sizes.len()
    }

    /// Unit centroids, `[rows, index_dim]`.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Member positions, `[rows, bucket_cap]`, `-1` padded.
    pub fn members(&self) -> &[i32] {
        &self.members
    }

    /// Keys in each bucket, `[rows]`.
    pub fn sizes(&self) -> &[u32] {
        &self.sizes
    }

    /// Append `keys` (`[n, index_dim]`, unit rows) at the next positions,
    /// closing every cluster block they complete.
    ///
    /// Validates before changing anything, so a refused append leaves the
    /// index as it was.
    pub fn push_keys(&mut self, keys: &[f32]) -> Result<(), String> {
        let dim = self.cfg.index_dim;
        check_unit_rows("LSA index keys", keys, dim)?;
        let len = self.len() as u64 + (keys.len() / dim) as u64;
        if len > i32::MAX as u64 {
            return Err(format!("LSA index: {len} keys exceeds the {} positions a member entry holds", i32::MAX));
        }
        let block_values = self.cfg.cluster_block * dim;
        let mut rest = keys;
        while !rest.is_empty() {
            let take = (block_values - self.open.len()).min(rest.len());
            self.open.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.open.len() == block_values {
                self.close_block()?;
            }
        }
        Ok(())
    }

    /// Forget every key at or past `len`, as a rejected speculative tail
    /// requires.
    ///
    /// Only the open block can shrink: a closed block's buckets were merged
    /// from all of its keys, and the index keeps no copy of them. Cutting into
    /// a closed block is refused; the caller rebuilds from the keys it holds.
    pub fn truncate(&mut self, len: usize) -> Result<(), String> {
        let closed = self.closed_blocks() * self.cfg.cluster_block;
        if len < closed {
            return Err(format!(
                "LSA index: cannot truncate to {len} keys inside closed cluster blocks (first open key {closed}); rebuild the index"
            ));
        }
        if len < self.len() {
            self.open.truncate((len - closed) * self.cfg.index_dim);
        }
        Ok(())
    }

    /// Ward-merge the full open block into buckets and publish them.
    fn close_block(&mut self) -> Result<(), String> {
        let cfg = self.cfg;
        let dim = cfg.index_dim;
        let base = self.closed_blocks() * cfg.cluster_block;
        let bpb = cfg.buckets_per_block();
        let pools = ward::pool_keys(&self.open, dim, cfg.pool_size);
        let pool_sizes = vec![cfg.pool_size as u32; cfg.pools_per_block()];
        let labels = ward::ward_labels(&pools, &pool_sizes, dim, bpb, cfg.bucket_cap as u32)?;

        // Centroids are the normalised sum of the bucket's KEYS (not its pool
        // sums, which hold the same total but in a different addition order).
        let mut sums = vec![0f64; bpb * dim];
        let mut members = vec![-1i32; bpb * cfg.bucket_cap];
        let mut sizes = vec![0u32; bpb];
        for key in 0..cfg.cluster_block {
            let bucket = labels[key / cfg.pool_size] as usize;
            let row = &self.open[key * dim..][..dim];
            for (s, x) in sums[bucket * dim..][..dim].iter_mut().zip(row) {
                *s += *x as f64;
            }
            let slot = sizes[bucket] as usize;
            members[bucket * cfg.bucket_cap + slot] = (base + key) as i32;
            sizes[bucket] += 1;
        }
        for sum in sums.chunks_exact(dim) {
            let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
            self.centroids.extend(sum.iter().map(|x| (x / norm) as f32));
        }
        self.members.extend_from_slice(&members);
        self.sizes.extend_from_slice(&sizes);
        self.open.clear();
        Ok(())
    }
}
