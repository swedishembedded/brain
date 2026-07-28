// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `kvcache` — the KV cache under memory pressure.
//!
//! Not "does prefix reuse help" (it does, on prefill) but the question that
//! decides whether a server survives a real session mix: **what happens when the
//! working set does not fit**. Sessions grow, go idle, resume and branch, and
//! the sum of their caches is deliberately several times the pool.
//!
//! ```text
//! session A:  4k -> 8k -> 16k -> 48k tokens      (steady growth)
//! session B:  2k -> idle -> resume               (resumption cost)
//! session C:  80k -> branch into 4 sub-sessions  (fan-out, shared prefix)
//! session D:  reuses a 32k system prefix         (reuse, if supported)
//! ```
//!
//! The metric that matters most is **eviction regret**: blocks evicted shortly
//! before they were needed again. High regret means the policy is choosing
//! wrong, which is a different problem from simply being too small — and only
//! the regret number tells them apart.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::stats::r3;

/// One session's shape over its lifetime, in tokens.
#[derive(Clone, Debug)]
pub struct Session {
    pub name: String,
    /// Successive context lengths this session reaches.
    pub steps: Vec<usize>,
    /// Steps after which the session idles (index into `steps`).
    pub idle_after: Option<usize>,
    /// How many sub-sessions branch off the final context.
    pub branches: usize,
    /// A shared prefix length reused across every step.
    pub shared_prefix: usize,
}

/// The default session mix — the four shapes above.
pub fn default_sessions() -> Vec<Session> {
    vec![
        Session {
            name: "growth".into(),
            steps: vec![4096, 8192, 16384, 49152],
            idle_after: None,
            branches: 0,
            shared_prefix: 0,
        },
        Session {
            name: "resume".into(),
            steps: vec![2048, 2048],
            idle_after: Some(0),
            branches: 0,
            shared_prefix: 0,
        },
        Session {
            name: "branch".into(),
            steps: vec![81920],
            idle_after: None,
            branches: 4,
            shared_prefix: 0,
        },
        Session {
            name: "shared_prefix".into(),
            steps: vec![32768, 33024, 33280],
            idle_after: None,
            branches: 0,
            shared_prefix: 32768,
        },
    ]
}

/// Scale the mix to a pool of `pool_tokens`, targeting a working set `over`
/// times capacity. Being explicit about the overcommit is the point — a
/// "cache benchmark" that fits in cache measures nothing.
pub fn scaled_sessions(pool_tokens: usize, over: f64) -> Vec<Session> {
    let base: usize = default_sessions().iter().map(|s| s.steps.iter().max().copied().unwrap_or(0)).sum();
    let target = (pool_tokens as f64 * over).max(1.0);
    let f = target / base.max(1) as f64;
    let scale = |n: usize| ((n as f64 * f).round() as usize).max(16);
    default_sessions()
        .into_iter()
        .map(|mut s| {
            s.steps = s.steps.iter().map(|&n| scale(n)).collect();
            s.shared_prefix = if s.shared_prefix > 0 { scale(s.shared_prefix) } else { 0 };
            s
        })
        .collect()
}

/// Cache accounting over a run.
#[derive(Clone, Debug, Default)]
pub struct Accounting {
    /// Tokens served from cache (no recompute needed).
    pub hits: usize,
    /// Tokens that had to be recomputed because their blocks were gone.
    pub recomputed: usize,
    /// Blocks evicted, with how long before each was next wanted (ms).
    /// `None` = never wanted again, which is a *correct* eviction.
    pub evictions: Vec<Option<f64>>,
    /// Times a running sequence had to be preempted for want of blocks.
    pub preemptions: usize,
    pub preempted_ms: f64,
    /// Blocks in the pool and blocks actually usable (pool minus fragmentation).
    pub pool_blocks: u32,
    pub usable_blocks: u32,
    /// Admission stalls: a request that had to wait purely for KV space.
    pub kv_stalls: usize,
    pub kv_stall_ms: f64,
    /// Per-session TTFA after a resume, keyed by session name.
    pub resume_ttfa_ms: HashMap<String, f64>,
}

/// An eviction is "regretted" when the block was wanted again within this
/// window — the policy paid the recompute for nothing.
pub const REGRET_WINDOW_MS: f64 = 30_000.0;

impl Accounting {
    /// Share of requested tokens served from cache.
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.hits + self.recomputed;
        (total > 0).then(|| self.hits as f64 / total as f64)
    }

    /// Share of evictions that turned out to be wrong.
    ///
    /// This is the number that separates "the cache is too small" (low regret —
    /// everything evicted was genuinely cold) from "the policy is choosing
    /// badly" (high regret — it keeps evicting live working sets).
    pub fn eviction_regret(&self) -> Option<f64> {
        if self.evictions.is_empty() {
            return None;
        }
        let bad = self.evictions.iter().filter(|e| matches!(e, Some(ms) if *ms <= REGRET_WINDOW_MS)).count();
        Some(bad as f64 / self.evictions.len() as f64)
    }

    /// Usable capacity as a fraction of nominal — the cost of fragmentation.
    pub fn effective_capacity(&self) -> Option<f64> {
        (self.pool_blocks > 0).then(|| self.usable_blocks as f64 / self.pool_blocks as f64)
    }

    pub fn fragmentation(&self) -> Option<f64> {
        self.effective_capacity().map(|e| 1.0 - e)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "kv_hit_rate": self.hit_rate().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "recomputed_artifacts": self.recomputed,
            "evictions": self.evictions.len(),
            "eviction_regret": self.eviction_regret().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "regret_window_ms": REGRET_WINDOW_MS,
            "preemptions": self.preemptions,
            "preempted_ms": r3(self.preempted_ms),
            "kv_stalls": self.kv_stalls,
            "kv_stall_ms": r3(self.kv_stall_ms),
            "kv_pool_blocks": self.pool_blocks,
            "kv_usable_blocks": self.usable_blocks,
            "effective_capacity": self.effective_capacity().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "fragmentation": self.fragmentation().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "resume_ttfa_ms": self.resume_ttfa_ms.iter().map(|(k, v)| (k.clone(), Value::from(r3(*v)))).collect::<serde_json::Map<_, _>>(),
        })
    }
}

/// Cache state a run should start from. Comparing cold against warm is the only
/// way to attribute a TTFA difference to the cache rather than to the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheState {
    Cold,
    Warm,
    Disabled,
}

impl CacheState {
    pub fn name(&self) -> &'static str {
        match self {
            CacheState::Cold => "cold",
            CacheState::Warm => "warm",
            CacheState::Disabled => "disabled",
        }
    }
}

pub fn render(a: &Accounting, state: CacheState) -> String {
    let pct = |v: Option<f64>| v.map(|x| format!("{:.1}%", x * 100.0)).unwrap_or_else(|| "—".into());
    format!(
        "\n  cache {}  hit-rate {}  regret {}  recomputed {}  preemptions {}  stalls {} ({:.0} ms)\n",
        state.name(),
        pct(a.hit_rate()),
        pct(a.eviction_regret()),
        a.recomputed,
        a.preemptions,
        a.kv_stalls,
        a.kv_stall_ms,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mix_covers_growth_resume_branch_and_reuse() {
        let s = default_sessions();
        assert_eq!(s.len(), 4);
        assert!(s.iter().any(|x| x.idle_after.is_some()), "one session must test resumption");
        assert!(s.iter().any(|x| x.branches > 0), "one session must fan out");
        assert!(s.iter().any(|x| x.shared_prefix > 0), "one session must reuse a prefix");
    }

    #[test]
    fn scaling_overcommits_the_pool_on_purpose() {
        let pool = 8192;
        let s = scaled_sessions(pool, 3.0);
        let peak: usize = s.iter().map(|x| x.steps.iter().max().copied().unwrap_or(0)).sum();
        assert!(peak > pool * 2, "working set {peak} must exceed the {pool}-token pool");
    }

    #[test]
    fn hit_rate_and_regret_are_none_when_nothing_happened() {
        let a = Accounting::default();
        assert_eq!(a.hit_rate(), None, "no traffic must not read as a 0% hit rate");
        assert_eq!(a.eviction_regret(), None);
        assert_eq!(a.effective_capacity(), None);
    }

    #[test]
    fn hit_rate_counts_recomputes_against_it() {
        let a = Accounting { hits: 750, recomputed: 250, ..Default::default() };
        assert_eq!(a.hit_rate(), Some(0.75));
    }

    #[test]
    fn regret_separates_a_small_cache_from_a_bad_policy() {
        // Everything evicted was genuinely cold: correct policy, cache just small.
        let cold = Accounting { evictions: vec![None, None, None], ..Default::default() };
        assert_eq!(cold.eviction_regret(), Some(0.0));
        // Everything evicted was wanted again almost immediately: bad policy.
        let bad = Accounting {
            evictions: vec![Some(10.0), Some(50.0), Some(100.0)],
            ..Default::default()
        };
        assert_eq!(bad.eviction_regret(), Some(1.0));
        // Wanted again only much later — that is a correct eviction.
        let late = Accounting { evictions: vec![Some(REGRET_WINDOW_MS * 2.0)], ..Default::default() };
        assert_eq!(late.eviction_regret(), Some(0.0));
    }

    #[test]
    fn fragmentation_is_the_complement_of_effective_capacity() {
        let a = Accounting { pool_blocks: 100, usable_blocks: 80, ..Default::default() };
        assert_eq!(a.effective_capacity(), Some(0.8));
        assert!((a.fragmentation().unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn json_reports_nulls_for_unmeasured_fields() {
        let j = Accounting::default().to_json();
        assert!(j["kv_hit_rate"].is_null());
        assert!(j["eviction_regret"].is_null());
        assert_eq!(j["preemptions"], 0);
    }

    #[test]
    fn cache_states_are_named_for_the_artifact() {
        assert_eq!(CacheState::Cold.name(), "cold");
        assert_eq!(CacheState::Disabled.name(), "disabled");
    }
}
