// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scheduling policy: from the pending jobs, choose the next **batch** to run.
//!
//! Two forces are balanced so throughput never starves latency:
//! - **batch by model** — jobs sharing an [`InstanceKey`] run together on one hot
//!   instance (one build, one — ideally batched — forward), so a busy model is
//!   efficient;
//! - **queue age is first-class** — a group's score rises with the wait time of its
//!   *oldest* job, so a single old request is picked ahead of a fat-but-fresh batch
//!   and nothing sits in the queue too long.
//!
//! [`choose_next`] is a pure function of per-group summaries (age + size), so the
//! policy is unit-tested without threads or a clock; the [`crate::executor`] feeds it
//! real ages and runs the chosen group.

/// A pending-job group: all queued jobs for one instance key.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    /// Index the caller uses to identify the group (e.g. into its key table).
    pub id: usize,
    /// Wait time of the group's **oldest** queued job, in milliseconds.
    pub oldest_age_ms: u64,
    /// Number of queued jobs in the group (how big a batch could be).
    pub size: usize,
}

/// Policy weights. `max_batch` caps a single run; `age_weight_per_ms` converts wait
/// time to score; `batch_weight` rewards a fuller batch. Defaults favor batching but
/// let a job that has waited ~a second jump the queue regardless of batch size.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    pub max_batch: usize,
    pub age_weight_per_ms: f64,
    pub batch_weight: f64,
    /// Hard latency ceiling: any job older than this is force-picked first.
    pub max_wait_ms: u64,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy { max_batch: 8, age_weight_per_ms: 1.0, batch_weight: 200.0, max_wait_ms: 2000 }
    }
}

impl Policy {
    fn score(&self, g: &Group) -> f64 {
        g.oldest_age_ms as f64 * self.age_weight_per_ms + (g.size.min(self.max_batch) as f64) * self.batch_weight
    }

    /// [`Self::default`] re-tuned for a device whose own service time is
    /// measured in **seconds**, not milliseconds - a laptop-class iGPU, where a
    /// real single-request TTFA of many seconds was measured on the box this
    /// was written against. Two defaults break down at that scale, and both are
    /// fixed here:
    ///
    /// * `max_wait_ms: 2000` force-picks *every* group almost immediately —
    ///   `choose_next`'s overdue branch ignores batch size entirely once
    ///   triggered, so once virtually everything is "overdue" the scheduler
    ///   degenerates to FIFO-of-one regardless of what batching code exists
    ///   downstream. Raised to 30s: comfortably above this class of
    ///   hardware's per-request service time, while still a real, bounded
    ///   fairness ceiling — no request waits forever.
    /// * `age_weight_per_ms: 1.0` means one point of score per millisecond
    ///   of age. At millisecond-scale service times that is the right
    ///   sensitivity; at second-scale it means an age gap of a few seconds
    ///   between two groups (thousands of points) swamps `batch_weight`'s
    ///   entire contribution (at most `max_batch * batch_weight` = 1600
    ///   points), so a full batch can never outscore a barely-older single
    ///   job. Lowered to 0.05 (50 points/second) so a modest, realistic age
    ///   gap no longer automatically defeats batching, while a genuinely
    ///   stale job still eventually wins via the `max_wait_ms` fairness
    ///   backstop, not via score alone.
    ///
    /// `Policy::default()` itself is left exactly as it was: every existing
    /// caller and test keeps its current millisecond-scale behaviour unless
    /// it explicitly opts into this.
    pub fn serving_default() -> Policy {
        Policy { age_weight_per_ms: 0.05, max_wait_ms: 30_000, ..Policy::default() }
    }

    /// [`Self::serving_default`], overridable per-field via
    /// `BRAIN_SCHED_{MAX_BATCH,AGE_WEIGHT,BATCH_WEIGHT,MAX_WAIT_MS}`. This is
    /// what the CLI's serving entry point uses — the production path, where
    /// a scheduler that has silently collapsed to FIFO-of-one is a real
    /// regression, not merely a suboptimal default. An unset or unparseable
    /// var falls back to the serving default for that field alone.
    pub fn from_env() -> Policy {
        let d = Self::serving_default();
        fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        Policy {
            max_batch: env_or("BRAIN_SCHED_MAX_BATCH", d.max_batch),
            age_weight_per_ms: env_or("BRAIN_SCHED_AGE_WEIGHT", d.age_weight_per_ms),
            batch_weight: env_or("BRAIN_SCHED_BATCH_WEIGHT", d.batch_weight),
            max_wait_ms: env_or("BRAIN_SCHED_MAX_WAIT_MS", d.max_wait_ms),
        }
    }
}

/// Choose the next group to run and how many of its jobs to batch. Returns
/// `(group_id, batch_len)`, or `None` if there are no groups. A group whose oldest
/// job exceeds `max_wait_ms` is force-picked (oldest such first) to bound tail
/// latency; otherwise the highest-scoring group wins. Ties break toward the older
/// group, then the larger, for determinism.
pub fn choose_next(groups: &[Group], policy: &Policy) -> Option<(usize, usize)> {
    if groups.is_empty() {
        return None;
    }
    // Latency guard: honor the oldest over-deadline group first.
    let overdue = groups.iter().filter(|g| g.oldest_age_ms >= policy.max_wait_ms).max_by_key(|g| g.oldest_age_ms);
    let chosen = overdue.unwrap_or_else(|| {
        groups
            .iter()
            .max_by(|a, b| {
                policy
                    .score(a)
                    .partial_cmp(&policy.score(b))
                    .unwrap()
                    .then(a.oldest_age_ms.cmp(&b.oldest_age_ms))
                    .then(a.size.cmp(&b.size))
            })
            .unwrap()
    });
    Some((chosen.id, chosen.size.min(policy.max_batch)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_the_bigger_batch_when_ages_are_equal() {
        let p = Policy::default();
        let groups = vec![
            Group { id: 0, oldest_age_ms: 10, size: 1 },
            Group { id: 1, oldest_age_ms: 10, size: 5 },
        ];
        assert_eq!(choose_next(&groups, &p), Some((1, 5)));
    }

    #[test]
    fn an_aged_single_job_beats_a_fresh_batch() {
        let p = Policy::default();
        // group 0: one job that's waited 1.5s; group 1: a fresh batch of 4.
        let groups = vec![
            Group { id: 0, oldest_age_ms: 1500, size: 1 },
            Group { id: 1, oldest_age_ms: 5, size: 4 },
        ];
        // score0 = 1500 ; score1 = 5 + 4*200 = 805 → group 0 wins (no starvation).
        assert_eq!(choose_next(&groups, &p), Some((0, 1)));
    }

    #[test]
    fn over_deadline_job_is_force_picked() {
        let p = Policy::default(); // max_wait 2000ms
        let groups = vec![
            Group { id: 0, oldest_age_ms: 2500, size: 1 }, // overdue
            Group { id: 1, oldest_age_ms: 50, size: 8 },   // huge fresh batch
        ];
        assert_eq!(choose_next(&groups, &p), Some((0, 1)));
    }

    #[test]
    fn batch_is_capped_at_max_batch() {
        let p = Policy { max_batch: 4, ..Policy::default() };
        let groups = vec![Group { id: 7, oldest_age_ms: 100, size: 20 }];
        assert_eq!(choose_next(&groups, &p), Some((7, 4)));
    }

    #[test]
    fn empty_queue_is_none() {
        assert_eq!(choose_next(&[], &Policy::default()), None);
    }

    /// The regression this box's measured serving pathology (a TTFA of many
    /// seconds at concurrency 1) reduces to: at real second-scale ages, `Policy::default()`
    /// picks a lone stale job over a fresh full batch, and `serving_default()`
    /// fixes it. This is the acceptance test for the fix, not just a policy tweak.
    #[test]
    fn at_real_serving_latencies_a_full_batch_beats_a_barely_older_single_job() {
        let groups = vec![
            Group { id: 0, oldest_age_ms: 13_000, size: 1 }, // one stale request
            Group { id: 1, oldest_age_ms: 5_000, size: 8 },  // a fresh full batch
        ];
        // Fails today: both ages exceed `Policy::default()`'s 2000ms
        // max_wait_ms, so the force-pick branch fires and picks whichever
        // group is OLDER, ignoring batch size entirely -> group 0, size 1.
        assert_eq!(
            choose_next(&groups, &Policy::default()),
            Some((0, 1)),
            "documents today's FIFO-of-one collapse at second-scale ages -- \
             if this fails, Policy::default() changed and this test needs review"
        );
        // Fixed: neither age exceeds serving_default()'s 30s ceiling, so
        // score comparison applies, and the rebalanced age_weight_per_ms no
        // longer lets an 8-second age gap swamp a full batch's weight.
        assert_eq!(choose_next(&groups, &Policy::serving_default()), Some((1, 8)));
    }

    #[test]
    fn from_env_falls_back_to_serving_default_when_unset() {
        // Guard against leakage from a parallel test or the outer shell.
        for k in ["BRAIN_SCHED_MAX_BATCH", "BRAIN_SCHED_AGE_WEIGHT", "BRAIN_SCHED_BATCH_WEIGHT", "BRAIN_SCHED_MAX_WAIT_MS"] {
            assert!(std::env::var(k).is_err(), "{k} must be unset for this test to be meaningful");
        }
        let p = Policy::from_env();
        let d = Policy::serving_default();
        assert_eq!(p.max_batch, d.max_batch);
        assert_eq!(p.max_wait_ms, d.max_wait_ms);
        assert_eq!(p.age_weight_per_ms, d.age_weight_per_ms);
        assert_eq!(p.batch_weight, d.batch_weight);
    }

    #[test]
    fn serving_default_still_force_picks_a_genuinely_stale_job() {
        // The fairness backstop must still exist -- a job older than 30s is
        // not left behind in favour of a merely-larger fresh batch.
        let groups = vec![
            Group { id: 0, oldest_age_ms: 31_000, size: 1 },
            Group { id: 1, oldest_age_ms: 100, size: 8 },
        ];
        assert_eq!(choose_next(&groups, &Policy::serving_default()), Some((0, 1)));
    }
}
