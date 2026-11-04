// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sizes and read budgets of a latent sparse attention index.

/// A smaller read budget for queries before `position`.
///
/// Early in a sequence there are few buckets to choose from, and the paper
/// reads a smaller long-range budget there (2,048 keys from 512 buckets below
/// one million tokens on GLM-5.3-Flash).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EarlyBudget {
    /// First query position that reads the configuration's own budget.
    pub position: usize,
    /// Long-range keys an early query reads at most.
    pub top_k: usize,
    /// Buckets an early query picks.
    pub top_p: usize,
}

/// Sizes and budgets of one LSA index, in keys unless stated otherwise.
///
/// [`Default`] is the paper's D-LSA (Flat) configuration for Qwen-3.8-27B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LsaConfig {
    /// Width of an index vector.
    pub index_dim: usize,
    /// Consecutive keys summed into one pool; a pool is never split.
    pub pool_size: usize,
    /// Average keys per bucket; a closed block ends with
    /// `cluster_block / bucket_size` buckets.
    pub bucket_size: usize,
    /// Most keys a bucket may hold.
    pub bucket_cap: usize,
    /// Keys clustered together once all of them exist.
    pub cluster_block: usize,
    /// Buckets a query picks.
    pub top_p: usize,
    /// Long-range keys a query reads at most.
    pub top_k: usize,
    /// First positions every query reads.
    pub sink: usize,
    /// Most recent positions every query reads, its own included.
    pub window: usize,
    /// A smaller budget for early queries, or `None` for one budget throughout.
    pub early: Option<EarlyBudget>,
}

impl Default for LsaConfig {
    fn default() -> LsaConfig {
        LsaConfig {
            index_dim: 64,
            pool_size: 4,
            bucket_size: 8,
            bucket_cap: 16,
            cluster_block: 512,
            top_p: 1024,
            top_k: 4096,
            sink: 4,
            window: 512,
            early: None,
        }
    }
}

impl LsaConfig {
    /// Buckets a closed cluster block ends with.
    pub fn buckets_per_block(&self) -> usize {
        self.cluster_block / self.bucket_size
    }

    /// Pools in a cluster block.
    pub fn pools_per_block(&self) -> usize {
        self.cluster_block / self.pool_size
    }

    /// Widest selection row: long-range keys, the window and the sinks.
    pub fn selection_width(&self) -> usize {
        self.top_k + self.window + self.sink
    }

    /// The `(top_k, top_p)` a query at `position` reads.
    pub fn budget_at(&self, position: usize) -> (usize, usize) {
        match self.early {
            Some(e) if position < e.position => (e.top_k, e.top_p),
            _ => (self.top_k, self.top_p),
        }
    }

    /// Refuse sizes that do not nest or that would lose keys.
    ///
    /// Beyond the paper's own nesting rules, the window must span a whole
    /// cluster block: a query reads its own, still-open block only through the
    /// window, so a narrower window silently drops keys no bucket holds yet.
    pub fn validate(&self) -> Result<(), String> {
        let sizes = [
            ("index_dim", self.index_dim),
            ("pool_size", self.pool_size),
            ("bucket_size", self.bucket_size),
            ("bucket_cap", self.bucket_cap),
            ("cluster_block", self.cluster_block),
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("window", self.window),
        ];
        if let Some((name, _)) = sizes.iter().find(|(_, v)| *v == 0) {
            return Err(format!("LSA {name} must be positive"));
        }
        if !self.bucket_size.is_multiple_of(self.pool_size) || !self.cluster_block.is_multiple_of(self.bucket_size) {
            return Err(format!(
                "LSA pool_size {} must divide bucket_size {}, which must divide cluster_block {}",
                self.pool_size, self.bucket_size, self.cluster_block
            ));
        }
        // Below twice the average a block can run out of pairs the cap allows
        // before it reaches its bucket count; at twice it never can, because
        // the two smallest of more than `buckets_per_block` buckets sum to at
        // most twice the average.
        if self.bucket_cap < 2 * self.bucket_size {
            return Err(format!(
                "LSA bucket_cap {} must be at least twice bucket_size {}",
                self.bucket_cap, self.bucket_size
            ));
        }
        if self.window < self.cluster_block {
            return Err(format!(
                "LSA window {} must cover a whole cluster_block {}: a query reads its own open block only through the window",
                self.window, self.cluster_block
            ));
        }
        if let Some(e) = self.early {
            if e.position == 0 || e.top_k == 0 || e.top_p == 0 {
                return Err("LSA early budget position, top_k and top_p must be positive".into());
            }
            if e.top_k > self.top_k || e.top_p > self.top_p {
                return Err(format!(
                    "LSA early budget (top_k {}, top_p {}) exceeds the main budget (top_k {}, top_p {})",
                    e.top_k, e.top_p, self.top_k, self.top_p
                ));
            }
        }
        Ok(())
    }
}
