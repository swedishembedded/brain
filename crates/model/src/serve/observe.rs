// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing what an inference server
// is doing while it serves for its clients. If your team needs expertise in
// observable LLM serving then you can procure our services by sending an
// email to info@swedishembedded.com.

//! What a [`Scheduler`](super::Scheduler) tells an observer about itself ([`Scheduler::observe`](super::Scheduler::observe)):
//! its KV pool, the sequences it holds, its back-off, and the tokens it decoded and prefilled with their rate over the
//! last [`THROUGHPUT_WINDOW_SECS`]. A server publishes it per replica in its metrics document.

use std::collections::VecDeque;

use super::Backoff;

/// The window the token rates of [`SchedulerObservation`] are measured over, in seconds.
pub const THROUGHPUT_WINDOW_SECS: f64 = 10.0;

/// A scheduler's paged KV pool now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvObservation {
    /// Tokens one block holds.
    pub block_tokens: u32,
    /// Blocks in the pool.
    pub total_blocks: u32,
    /// Blocks no sequence and no cached prefix holds.
    pub free_blocks: u32,
    /// Blocks the prefix cache holds (claimable when the pool needs them).
    pub prefix_cached_blocks: usize,
    /// Tokens a new request can claim: free blocks and blocks only the prefix cache holds, as of the last iteration.
    pub claimable_tokens: u64,
    /// The pool's capacity in tokens.
    pub total_tokens: u64,
}

/// Everything [`Scheduler::observe`](super::Scheduler::observe) reports.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SchedulerObservation {
    pub kv: KvObservation,
    /// Sequences decoding.
    pub running: usize,
    /// Sequences whose prompt is being read in.
    pub prefilling: usize,
    /// Requests submitted and not admitted yet.
    pub waiting: usize,
    /// Admitted sequences whose KV is parked in host memory, waiting for device blocks.
    pub swapped: usize,
    /// The most sequences it admits at once now ([`Scheduler::admitted_width`](super::Scheduler::admitted_width)).
    pub admitted_width: usize,
    /// Its batch: the most sequences it ever runs at once.
    pub max_width: usize,
    pub backoff: Backoff,
    /// Tokens sampled since it was built, stop tokens included.
    pub decoded_tokens: u64,
    /// Prompt tokens computed by prefill since it was built (prefix-cache hits are not computed, so not counted).
    pub prefilled_tokens: u64,
    /// Decoded tokens per second over the last [`THROUGHPUT_WINDOW_SECS`] (or since it was built, when younger).
    pub decode_tokens_per_s: f64,
    /// Prefilled tokens per second over the same window.
    pub prefill_tokens_per_s: f64,
}

/// Token counts by whole second, for the rates over the window: at most one entry a second for the window's seconds.
#[derive(Debug)]
pub(super) struct Throughput {
    /// The clock reading the counts start at.
    origin: f64,
    pub(super) decoded: u64,
    pub(super) prefilled: u64,
    /// What `decoded` and `prefilled` were when they were last put in a bucket.
    banked: (u64, u64),
    /// `(second since origin, decoded, prefilled)` added in that second, oldest first.
    buckets: VecDeque<(u64, u64, u64)>,
}

impl Throughput {
    pub(super) fn new(origin: f64) -> Throughput {
        Throughput { origin, decoded: 0, prefilled: 0, banked: (0, 0), buckets: VecDeque::new() }
    }

    /// Counts start again at `origin` (a new clock).
    pub(super) fn restart(&mut self, origin: f64) {
        *self = Throughput { decoded: self.decoded, prefilled: self.prefilled, ..Throughput::new(origin) };
        self.banked = (self.decoded, self.prefilled);
    }

    fn second(&self, now: f64) -> u64 {
        (now - self.origin).max(0.0) as u64
    }

    /// Put what was counted since the last call into `now`'s second, and forget seconds the window has left.
    pub(super) fn bank(&mut self, now: f64) {
        let (decoded, prefilled) = (self.decoded - self.banked.0, self.prefilled - self.banked.1);
        self.banked = (self.decoded, self.prefilled);
        let second = self.second(now);
        if decoded > 0 || prefilled > 0 {
            match self.buckets.back_mut() {
                Some((s, d, p)) if *s == second => {
                    *d += decoded;
                    *p += prefilled;
                }
                _ => self.buckets.push_back((second, decoded, prefilled)),
            }
        }
        let window = THROUGHPUT_WINDOW_SECS as u64;
        while self.buckets.front().is_some_and(|&(s, _, _)| s + window <= second) {
            self.buckets.pop_front();
        }
    }

    /// `(decoded, prefilled)` tokens per second over the window ending `now`.
    pub(super) fn rates(&self, now: f64) -> (f64, f64) {
        let second = self.second(now);
        let window = THROUGHPUT_WINDOW_SECS as u64;
        let (decoded, prefilled) = self.buckets.iter().filter(|&&(s, _, _)| s + window > second).fold((0u64, 0u64), |(d, p), &(_, bd, bp)| (d + bd, p + bp));
        let span = (now - self.origin).clamp(0.0, THROUGHPUT_WINDOW_SECS);
        if span <= 0.0 {
            return (0.0, 0.0);
        }
        (decoded as f64 / span, prefilled as f64 / span)
    }
}
