// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for LLM serving that fills every
// forward pass with decode and prompt rows under a latency objective. If your
// team needs expertise in token-budgeted continuous batching, then you can
// procure our services by sending an email to info@swedishembedded.com.

//! How many prompt rows ride along with the decode rows of one mixed iteration.
//!
//! A mixed iteration ([`super::PagedDecoder::forward_mixed`]) runs the next token of every running sequence and
//! a slice of the prompts still being read in as ONE forward, so each weight is read once for both. What the
//! scheduler decides is the slice: the more prompt rows ride along, the sooner prompts are in, and the longer
//! the iteration - which is every running sequence's token interval. [`StepCost`] learns how long an iteration
//! of `rows` rows takes (a fixed part, the weights read once, plus a part per row) from the iterations that
//! ran, and [`StepCost::prompt_rows`] turns the decoder's latency target into a row count. Everything is O(1)
//! per iteration.

/// What a decoder takes in one mixed iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedLimits {
    /// Rows (decode rows plus prompt rows) one forward carries at most.
    pub max_rows: usize,
    /// Prompt rows a waiting prompt is always given when any is waiting, however busy decode is: the bound on how
    /// long a prompt can be held back (no starvation). Also the size of the first slices, before any iteration
    /// has been timed.
    pub min_prompt_rows: usize,
}

/// Weight of the history kept per observation: the fit follows the last few dozen iterations.
const DECAY: f64 = 0.95;

/// Row counts the fit needs to see spread over before it trusts a slope: below this spread (in rows) iterations
/// say how long a step takes, not how it grows with its rows.
const MIN_SPREAD_ROWS: f64 = 8.0;

/// How much longer than the minimum slice's own iteration an iteration may be made when the latency target is
/// out of reach even for that slice. A forward whose cost is mostly fixed (the weights read once, the host
/// recording its launches) grows slowly with its rows, so the target alone would pin the prompt part to its
/// minimum and pay that fixed cost for the least possible work; this much extra time buys many more rows.
const UNREACHABLE_TARGET_STRETCH: f64 = 1.25;

/// Seconds an iteration of `rows` rows takes: `fixed + per_row * rows`, fitted by exponentially decayed least
/// squares over the iterations observed.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepCost {
    /// Decayed sums of 1, x, x*x, y, x*y with x the rows and y the seconds.
    n: f64,
    sx: f64,
    sxx: f64,
    sy: f64,
    sxy: f64,
}

impl StepCost {
    /// An iteration of `rows` rows took `secs`.
    pub fn observe(&mut self, rows: usize, secs: f64) {
        if rows == 0 || !secs.is_finite() || secs <= 0.0 {
            return;
        }
        let x = rows as f64;
        self.n = self.n * DECAY + 1.0;
        self.sx = self.sx * DECAY + x;
        self.sxx = self.sxx * DECAY + x * x;
        self.sy = self.sy * DECAY + secs;
        self.sxy = self.sxy * DECAY + x * secs;
    }

    /// `(fixed seconds, seconds per row)`, `None` before the first observation. With the observed row counts too
    /// close together to tell a slope the model is proportional to rows (the whole step grows with its rows):
    /// pessimistic for a step that is mostly a fixed weight read, and corrected once a slice of another size has
    /// been timed.
    fn line(&self) -> Option<(f64, f64)> {
        if self.n < 1.0 {
            return None;
        }
        let mean_x = self.sx / self.n;
        let var_x = self.sxx / self.n - mean_x * mean_x;
        if var_x < MIN_SPREAD_ROWS * MIN_SPREAD_ROWS {
            return Some((0.0, self.sy / self.sx));
        }
        let slope = ((self.sxy / self.n - mean_x * self.sy / self.n) / var_x).max(0.0);
        let fixed = (self.sy / self.n - slope * mean_x).max(0.0);
        Some((fixed, slope))
    }

    /// Seconds an iteration of `rows` rows is expected to take; `None` before any was timed.
    pub fn predict(&self, rows: usize) -> Option<f64> {
        self.line().map(|(fixed, per_row)| fixed + per_row * rows as f64)
    }

    /// Prompt rows to add to `decode_rows` decode rows so the iteration is expected to take `target_secs`.
    ///
    /// Without a running sequence there is no token interval to protect and the iteration takes all
    /// `limits.max_rows`. Otherwise the answer is what the target leaves, never below `limits.min_prompt_rows`
    /// (a prompt always advances) and never above the room the forward has. Before an iteration was timed the
    /// first slice is the minimum. `target_secs` of `None` (the decoder holds no latency target) is the full
    /// room.
    pub fn prompt_rows(&self, limits: MixedLimits, decode_rows: usize, target_secs: Option<f64>) -> usize {
        let room = limits.max_rows.saturating_sub(decode_rows);
        let floor = limits.min_prompt_rows.min(room);
        let Some(target) = target_secs.filter(|_| decode_rows > 0) else { return room };
        let Some((fixed, per_row)) = self.line() else { return floor };
        if per_row <= 0.0 {
            return room;
        }
        // A target the minimum slice already overruns is replaced by a little more than that slice takes.
        let target = target.max(UNREACHABLE_TARGET_STRETCH * (fixed + per_row * (decode_rows + floor) as f64));
        let rows_under_target = ((target - fixed) / per_row).max(0.0) as usize;
        rows_under_target.saturating_sub(decode_rows).clamp(floor, room)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: MixedLimits = MixedLimits { max_rows: 1024, min_prompt_rows: 64 };

    fn trained(fixed: f64, per_row: f64) -> StepCost {
        let mut c = StepCost::default();
        for rows in [16usize, 80, 144, 400, 16, 272, 528, 80] {
            c.observe(rows, fixed + per_row * rows as f64);
        }
        c
    }

    #[test]
    fn the_fit_recovers_a_fixed_part_and_a_part_per_row() {
        let c = trained(0.020, 0.0001);
        let (fixed, per_row) = c.line().unwrap();
        assert!((fixed - 0.020).abs() < 1e-6 && (per_row - 0.0001).abs() < 1e-8, "{fixed} {per_row}");
        assert!((c.predict(500).unwrap() - 0.070).abs() < 1e-6);
    }

    #[test]
    fn prompt_rows_fill_what_the_target_leaves() {
        // 20 ms of weights, 0.1 ms a row: 45 ms holds 250 rows, 16 of them decode.
        let c = trained(0.020, 0.0001);
        let rows = c.prompt_rows(LIMITS, 16, Some(0.045));
        assert!((233..=234).contains(&rows), "{rows}");
    }

    #[test]
    fn a_prompt_always_advances_however_busy_decode_is() {
        // Every row costs a millisecond: the minimum slice alone overruns the target, and 25 % more than it takes
        // buys the 16 rows beyond the minimum.
        let c = trained(0.040, 0.001);
        let rows = c.prompt_rows(LIMITS, 64, Some(0.045));
        assert!((LIMITS.min_prompt_rows..=LIMITS.min_prompt_rows + 70).contains(&rows), "{rows}");
    }

    #[test]
    fn a_target_out_of_reach_is_stretched_by_a_quarter_of_the_minimum_slice() {
        // 90 ms fixed, 0.1 ms a row: the 45 ms target is out of reach; 64 decode and 64 prompt rows take 102.8 ms,
        // a quarter more is 128.5 ms, which holds (128.5 - 90) / 0.1 - 64 = 321 prompt rows.
        let c = trained(0.090, 0.0001);
        let rows = c.prompt_rows(MixedLimits { max_rows: 2048, min_prompt_rows: 64 }, 64, Some(0.045));
        assert!((319..=322).contains(&rows), "{rows}");
    }

    #[test]
    fn prompt_rows_never_exceed_the_room_in_the_forward() {
        let c = trained(0.001, 0.000001);
        assert_eq!(c.prompt_rows(LIMITS, 100, Some(0.045)), LIMITS.max_rows - 100);
    }

    #[test]
    fn nothing_decoding_means_no_interval_to_protect() {
        let c = trained(0.040, 0.001);
        assert_eq!(c.prompt_rows(LIMITS, 0, Some(0.045)), LIMITS.max_rows);
    }

    #[test]
    fn an_untimed_step_starts_with_the_minimum_slice_and_a_missing_target_takes_the_room() {
        assert_eq!(StepCost::default().prompt_rows(LIMITS, 8, Some(0.045)), LIMITS.min_prompt_rows);
        assert_eq!(trained(0.040, 0.001).prompt_rows(LIMITS, 8, None), LIMITS.max_rows - 8);
    }

    #[test]
    fn rows_timed_at_one_size_scale_proportionally_until_another_size_is_seen() {
        let mut c = StepCost::default();
        c.observe(16, 0.032);
        assert!((c.predict(48).unwrap() - 0.096).abs() < 1e-9);
        // The minimum slice is then timed: the step turns out mostly fixed.
        c.observe(80, 0.0384);
        c.observe(16, 0.032);
        c.observe(80, 0.0384);
        let (fixed, per_row) = c.line().unwrap();
        assert!(fixed > 0.025 && per_row < 0.0002, "{fixed} {per_row}");
    }

    #[test]
    fn the_fit_follows_a_change_in_cost() {
        let mut c = trained(0.020, 0.0001);
        for i in 0..200 {
            c.observe([16usize, 144, 400, 272][i % 4], 0.050 + 0.0002 * [16.0, 144.0, 400.0, 272.0][i % 4]);
        }
        let (fixed, per_row) = c.line().unwrap();
        assert!((fixed - 0.050).abs() < 0.002 && (per_row - 0.0002).abs() < 2e-5, "{fixed} {per_row}");
    }
}
