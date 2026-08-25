// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixed-timestep pacing with a mockable clock. The world model is the
//! simulation: one `step()` consumes exactly one action — nothing is ever
//! skipped. When a step overruns its budget the pacer reports honest fps and
//! (optionally, with hysteresis) suggests dropping the model's quality level
//! (`WorldModel::set_nfe`).

use std::time::{Duration, Instant};

/// Time source, mockable for tests.
pub trait Clock {
    fn now_ms(&mut self) -> u64;
    /// Sleep until `deadline_ms` (no-op if already past).
    fn sleep_until(&mut self, deadline_ms: u64);
}

/// Real wall clock.
pub struct WallClock {
    origin: Instant,
}

impl WallClock {
    pub fn new() -> WallClock {
        WallClock { origin: Instant::now() }
    }
}

impl Default for WallClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for WallClock {
    fn now_ms(&mut self) -> u64 {
        self.origin.elapsed().as_millis() as u64
    }
    fn sleep_until(&mut self, deadline_ms: u64) {
        let now = self.now_ms();
        if deadline_ms > now {
            std::thread::sleep(Duration::from_millis(deadline_ms - now));
        }
    }
}

/// Deterministic test clock: `sleep_until` jumps time forward; `advance`
/// models work taking time.
pub struct MockClock {
    pub now: u64,
}

impl Clock for MockClock {
    fn now_ms(&mut self) -> u64 {
        self.now
    }
    fn sleep_until(&mut self, deadline_ms: u64) {
        if deadline_ms > self.now {
            self.now = deadline_ms;
        }
    }
}

/// Per-tick outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tick {
    /// Measured duration of the work portion of the last tick, ms.
    pub work_ms: u64,
    /// True when the last tick exceeded its frame budget.
    pub overrun: bool,
    /// Exponentially smoothed achieved fps.
    pub fps: f32,
    /// `Some(delta)` when adaptive quality suggests a change (+1 / -1).
    pub quality_delta: Option<i32>,
}

/// Fixed-timestep pacer. Call [`FixedTimestep::tick`] around each model step:
/// it sleeps to hold the target rate, measures the work time and, when
/// `adaptive`, emits quality deltas with hysteresis (drop after `K_DOWN`
/// consecutive overruns, raise after `K_UP` consecutive under-budget ticks).
pub struct FixedTimestep<C: Clock> {
    clock: C,
    budget_ms: u64,
    next_deadline: u64,
    adaptive: bool,
    over_streak: u32,
    under_streak: u32,
    fps: f32,
}

const K_DOWN: u32 = 5;
const K_UP: u32 = 30;

impl<C: Clock> FixedTimestep<C> {
    pub fn new(mut clock: C, target_fps: u32, adaptive: bool) -> FixedTimestep<C> {
        let budget_ms = 1000 / target_fps.max(1) as u64;
        let start = clock.now_ms();
        FixedTimestep {
            clock,
            budget_ms,
            next_deadline: start + budget_ms,
            adaptive,
            over_streak: 0,
            under_streak: 0,
            fps: target_fps as f32,
        }
    }

    /// Run one paced tick around `work`. The closure receives the clock so
    /// test work can advance a [`MockClock`]; real work just ignores it.
    /// Returns the tick report.
    pub fn tick<F: FnOnce(&mut C)>(&mut self, work: F) -> Tick {
        let t0 = self.clock.now_ms();
        work(&mut self.clock);
        let t1 = self.clock.now_ms();
        let work_ms = t1 - t0;
        let overrun = work_ms > self.budget_ms;

        // Honest fps over the actual tick period (work + any sleep).
        self.clock.sleep_until(self.next_deadline);
        let t_end = self.clock.now_ms();
        let period = (t_end - t0).max(1);
        let inst = 1000.0 / period as f32;
        self.fps = 0.9 * self.fps + 0.1 * inst;
        // Next deadline from where we actually are (no spiral of death).
        self.next_deadline = t_end.max(self.next_deadline) + self.budget_ms;

        let mut quality_delta = None;
        if self.adaptive {
            if overrun {
                self.over_streak += 1;
                self.under_streak = 0;
                if self.over_streak >= K_DOWN {
                    quality_delta = Some(-1);
                    self.over_streak = 0;
                }
            } else {
                self.under_streak += 1;
                self.over_streak = 0;
                if self.under_streak >= K_UP {
                    quality_delta = Some(1);
                    self.under_streak = 0;
                }
            }
        }
        Tick { work_ms, overrun, fps: self.fps, quality_delta }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the REAL `tick()` with mock work taking `work_ms` of mock time.
    fn mock_tick(p: &mut FixedTimestep<MockClock>, work_ms: u64) -> Tick {
        p.tick(|c| c.now += work_ms)
    }

    fn pace(work_ms: u64, ticks: usize, fps: u32, adaptive: bool) -> Vec<Tick> {
        let mut p = FixedTimestep::new(MockClock { now: 0 }, fps, adaptive);
        (0..ticks).map(|_| mock_tick(&mut p, work_ms)).collect()
    }

    #[test]
    fn pacing_holds_target_when_work_is_cheap() {
        // At the target below, work far inside the frame budget must not
        // overrun, and the reported rate must hold the target.
        let ticks = pace(10, 60, 15, false);
        assert!(ticks.iter().all(|t| !t.overrun));
        let fps = ticks.last().unwrap().fps;
        assert!((fps - 15.0).abs() < 1.5, "fps={fps}");
    }

    #[test]
    fn pacing_reports_overrun_and_honest_fps() {
        // Work several times the frame budget must overrun, and the reported
        // rate must fall to what the work actually allows, not the target.
        let ticks = pace(200, 60, 15, false);
        assert!(ticks.iter().all(|t| t.overrun));
        let fps = ticks.last().unwrap().fps;
        assert!((fps - 5.0).abs() < 0.5, "fps={fps}");
    }

    #[test]
    fn pacing_adaptive_drops_after_streak_and_recovers_with_hysteresis() {
        let mut p = FixedTimestep::new(MockClock { now: 0 }, 15, true);
        // K_DOWN consecutive overruns => exactly one -1.
        let mut deltas = vec![];
        for _ in 0..K_DOWN {
            deltas.extend(mock_tick(&mut p, 200).quality_delta);
        }
        assert_eq!(deltas, vec![-1]);
        // Fast ticks: +1 only after K_UP consecutive under-budget ticks.
        let mut ups = vec![];
        for _ in 0..K_UP {
            ups.extend(mock_tick(&mut p, 5).quality_delta);
        }
        assert_eq!(ups, vec![1]);
        // A single overrun resets the up-streak.
        for _ in 0..(K_UP - 1) {
            assert_eq!(mock_tick(&mut p, 5).quality_delta, None);
        }
        assert_eq!(mock_tick(&mut p, 200).quality_delta, None);
        assert_eq!(mock_tick(&mut p, 5).quality_delta, None);
    }
}
