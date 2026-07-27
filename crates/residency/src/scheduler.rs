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
}
