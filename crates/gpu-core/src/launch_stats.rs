// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-launch device timings and the guard that refuses to trust them when the
//! card is shared.
//!
//! Swedish Embedded AB implements kernel benchmarking that stays honest on
//! shared accelerators for its clients. If your team needs trustworthy kernel
//! timings on a card other jobs are also using, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! A kernel timed by the backend's own events is still stretched when another
//! context is time-sliced onto the card while it runs: the events bracket the
//! wall time of the slice, not the kernel's work. Averaging a run of launches
//! (what the tuner did) folds one stretched launch into every figure of that
//! run, and taking the best of a few averages does not undo it when every run
//! holds one: on a shared card that made one-row GEMVs that take microseconds
//! read as milliseconds, and the percent of roof computed from them read as
//! thousands of percent (knowledge 504).
//!
//! So every launch is timed on its own ([`sample`]): one launch per submission,
//! its device time read back before the next is issued. The figure a caller
//! uses is the MEDIAN of those samples, which a minority of stretched launches
//! cannot move, and [`LaunchStats::contaminated`] says when too many launches
//! were stretched for even the median to be trusted (the card is busy, or the
//! clock moved), so the caller can flag or refuse the number instead of
//! persisting it.

use crate::{Gpu, Step};

/// A launch this many times slower than the fastest of its run was stretched by
/// something other than the kernel: a kernel timed alone on an idle card
/// repeats to a few percent.
pub const STRETCH: f64 = 1.25;

/// The share of stretched launches past which a run is contaminated: up to a
/// third, the median is still an unstretched launch with margin.
pub const MAX_STRETCHED: f64 = 1.0 / 3.0;

/// Launches timed and discarded before the samples (clocks ramp, caches and
/// the instruction pipeline warm, the first launch pays any lazy setup).
pub const WARMUP: usize = 2;

/// The distribution of one kernel's per-launch device times, milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaunchStats {
    pub median_ms: f64,
    pub min_ms: f64,
    /// The share of launches more than [`STRETCH`] times the fastest.
    pub stretched: f64,
    pub samples: usize,
}

impl LaunchStats {
    /// The statistics of `samples` (milliseconds); `None` when there is no
    /// finite positive sample to describe.
    pub fn of(samples: &[f64]) -> Option<LaunchStats> {
        let mut s: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
        if s.is_empty() {
            return None;
        }
        s.sort_by(f64::total_cmp);
        let half = s.len() / 2;
        let median_ms = if s.len() % 2 == 1 { s[half] } else { (s[half - 1] + s[half]) / 2.0 };
        let stretched = s.iter().filter(|&&v| v > STRETCH * s[0]).count() as f64 / s.len() as f64;
        Some(LaunchStats { median_ms, min_ms: s[0], stretched, samples: s.len() })
    }

    /// Whether these samples are too inconsistent to stand for the kernel: more
    /// than [`MAX_STRETCHED`] of the launches stretched, or a median far above
    /// the fastest launch (a wide distribution with no clean majority).
    pub fn contaminated(&self) -> bool {
        contaminated(self.median_ms, self.min_ms, self.stretched)
    }
}

/// [`LaunchStats::contaminated`] on its parts, for a caller that kept only them.
pub fn contaminated(median_ms: f64, min_ms: f64, stretched: f64) -> bool {
    stretched > MAX_STRETCHED || median_ms > STRETCH * min_ms
}

/// Device time of each of `steps`, each submitted and read back on its own,
/// after [`WARMUP`] launches of the first one; `None` where the backend cannot
/// time kernels on the device (a caller must then say so, never fall back to
/// host time silently).
///
/// A step's time is the sum of every timed launch it issued (one kernel, or a
/// library call), so it needs no kernel name and covers vendor-library steps
/// as well. Timing must already be on ([`Gpu::set_kernel_timing`]).
pub fn sample(gpu: &Gpu, steps: &[Step]) -> Option<Vec<f64>> {
    let one = |step: &Step| -> Option<f64> {
        gpu.reset_kernel_times();
        gpu.submit(&[], std::slice::from_ref(step));
        gpu.poll_wait();
        let t = gpu.kernel_times()?;
        let ms: f64 = t.iter().map(|r| r.1).sum();
        (ms > 0.0).then_some(ms)
    };
    let first = steps.first()?;
    for _ in 0..WARMUP {
        one(first)?;
    }
    steps.iter().map(one).collect()
}

/// [`sample`] reduced to its statistics.
pub fn measure(gpu: &Gpu, steps: &[Step]) -> Option<LaunchStats> {
    LaunchStats::of(&sample(gpu, steps)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_of_a_clean_run_is_the_kernel_and_it_is_not_contaminated() {
        let s = LaunchStats::of(&[10.0, 10.2, 9.9, 10.1, 10.0, 10.3, 9.95]).unwrap();
        assert!((s.median_ms - 10.0).abs() < 1e-9, "{s:?}");
        assert_eq!(s.min_ms, 9.9);
        assert_eq!(s.samples, 7);
        assert!(!s.contaminated(), "{s:?}");
    }

    /// A minority of time-sliced launches (each stretched a hundredfold) moves
    /// neither the median nor the verdict: the figure is the kernel's.
    #[test]
    fn a_minority_of_stretched_launches_does_not_move_the_median() {
        let mut v = vec![0.02; 9];
        v.extend([2.0, 1.5, 1.9]);
        let s = LaunchStats::of(&v).unwrap();
        assert!((s.median_ms - 0.02).abs() < 1e-12, "{s:?}");
        assert!(!s.contaminated(), "{s:?}");
    }

    /// When most launches are stretched the median is no longer the kernel,
    /// and the statistics must say so rather than pass the number on.
    #[test]
    fn a_run_stretched_more_often_than_not_is_contaminated() {
        let s = LaunchStats::of(&[0.02, 0.021, 1.4, 1.9, 2.0, 1.3, 0.9]).unwrap();
        assert!(s.contaminated(), "{s:?}");
        let wide = LaunchStats::of(&[1.0, 1.3, 1.6, 1.9, 2.2]).unwrap();
        assert!(wide.contaminated(), "{wide:?}");
    }

    #[test]
    fn no_usable_sample_is_no_statistics() {
        assert_eq!(LaunchStats::of(&[]), None);
        assert_eq!(LaunchStats::of(&[0.0, f64::NAN, -1.0]), None);
    }
}
