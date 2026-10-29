// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving large language models
// that hold an interactive inter-token latency while new prompts are read in.
// If your team needs expertise in LLM serving schedulers and GPU time sharing,
// you can procure our services by sending an email to info@swedishembedded.com.

//! How much of the card a prompt gets between two decode steps.
//!
//! A decode step that shares the card with a prefill round runs 1.5 to 2 times slower (the round's kernels hold the
//! SMs; stream priority only orders what is waiting), so a prompt read in beside the decode loop stretches every
//! running sequence's token interval for as long as it goes in. [`PrefillPace`] keeps the two apart in time
//! instead: each scheduler iteration the prompt gets one slice of device time, recorded and then waited for, and
//! the decode step that follows runs on an idle card. The slice is what is left of the interval the engine is
//! asked to hold ([`super::PrefillPolicy::step_target`]) once the decode iteration is paid for, measured from the
//! iterations that actually ran; a prompt is never starved, it keeps [`MIN_SHARE`] of the decode iteration's
//! length however far decode is over the target.

/// The least a prompt gets of each iteration, as a fraction of the decode iteration's length.
// perf-number: with decode already over its target one layer per iteration reads a 1024-token prompt in over 3 s
const MIN_SHARE: f64 = 0.25;

/// The most device time one slice holds, seconds: a bound for the case of an idle decode loop, where the target
/// minus a tiny iteration would otherwise hand a prompt the whole interval.
const MAX_SLICE_SECS: f64 = 0.1;

/// Rows up to which a round runs as decode steps (see `exact_round_ok`) and is not timed: its cost is not per layer.
const DECODE_STEP_ROWS: usize = 32;

/// Round sizes a layer's cost is kept for, one per power of two of rows: a short round streams every weight for
/// few rows, so its layer costs far more per row than a wide round's, and one rate per row is wrong for both.
const SIZE_CLASSES: usize = 14;

/// Layers recorded by the first slice, before a layer's cost has been measured.
const FIRST_SLICE_LAYERS: usize = 2;

/// Weight of the newest sample in the running averages.
const EWMA: f64 = 0.5;

/// Longest gap between two slices that still counts as one decode iteration, seconds; a longer one was the
/// scheduler idle (or the prompt waiting on admission), not a decode step.
const MAX_ITERATION_SECS: f64 = 0.25;

/// Sizes the slice of prefill each scheduler iteration records, from measured decode iterations and layer costs.
#[derive(Debug)]
pub(super) struct PrefillPace {
    target_secs: f64,
    /// Seconds one scheduler iteration takes outside the slice (a decode step and what the scheduler does
    /// around it), averaged over recent ones.
    iteration_secs: Option<f64>,
    /// Seconds the card needs for one layer of a round of each size class, averaged over recent slices.
    layer_secs: [Option<f64>; SIZE_CLASSES],
}

/// The size class of a round of `rows` rows.
fn size_class(rows: usize) -> usize {
    ((usize::BITS - rows.max(1).saturating_sub(1).leading_zeros()) as usize).min(SIZE_CLASSES - 1)
}

impl PrefillPace {
    pub(super) fn new(target: std::time::Duration) -> PrefillPace {
        PrefillPace { target_secs: target.as_secs_f64(), iteration_secs: None, layer_secs: [None; SIZE_CLASSES] }
    }

    /// A decode iteration ran for `secs` since the last slice ended. `None` (the prompt was not being read in
    /// meanwhile) or a gap too long to be one iteration forgets the average: the decode loop it described may have
    /// gained or lost sequences while nothing was paced, and a stale figure would size the slice for a batch that
    /// is no longer there.
    pub(super) fn observe_iteration(&mut self, secs: Option<f64>) {
        let Some(secs) = secs.filter(|&s| s <= MAX_ITERATION_SECS) else {
            self.iteration_secs = None;
            return;
        };
        self.iteration_secs = Some(self.iteration_secs.map_or(secs, |m| m + EWMA * (secs - m)));
    }

    /// Layers of a `rows`-row round the next slice records.
    pub(super) fn layers(&self, rows: usize) -> usize {
        let Some(layer_secs) = self.layer_secs[size_class(rows)] else { return FIRST_SLICE_LAYERS };
        let slice = match self.iteration_secs {
            Some(it) => (self.target_secs - it).max(MIN_SHARE * it).min(MAX_SLICE_SECS),
            None => MAX_SLICE_SECS.min(self.target_secs / 2.0),
        };
        ((slice / layer_secs) as usize).max(1)
    }

    /// A slice of `layers` layers of a `rows`-row round took `secs` from the first launch to an idle card. A round
    /// that runs as decode steps says nothing about a layer.
    pub(super) fn observe_slice(&mut self, layers: usize, rows: usize, secs: f64) {
        if rows <= DECODE_STEP_ROWS {
            return;
        }
        let per = secs / layers as f64;
        let slot = &mut self.layer_secs[size_class(rows)];
        *slot = Some(slot.map_or(per, |m| m + EWMA * (per - m)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn pace() -> PrefillPace {
        let mut p = PrefillPace::new(Duration::from_millis(40));
        // A layer of a 512-row round takes 4 ms.
        p.observe_slice(10, 512, 0.04);
        p
    }

    #[test]
    fn the_first_slice_is_small_until_a_layer_has_been_timed() {
        assert_eq!(PrefillPace::new(Duration::from_millis(40)).layers(512), FIRST_SLICE_LAYERS);
    }

    #[test]
    fn a_short_round_is_sized_from_its_own_layer_cost_not_a_wide_rounds() {
        let mut p = pace();
        p.observe_iteration(Some(0.020));
        // 100 rows cost 0.6 ms a layer: 20 ms of slice is 33 of them, where the 512-row rate (4 ms) says 5.
        assert_eq!(p.layers(100), FIRST_SLICE_LAYERS, "not timed yet");
        p.observe_slice(10, 100, 0.006);
        assert_eq!(p.layers(100), 33);
        assert_eq!(p.layers(512), 5);
    }

    #[test]
    fn size_classes_are_powers_of_two_of_rows() {
        assert_eq!((size_class(1), size_class(33), size_class(64), size_class(65), size_class(512), size_class(513)), (0, 6, 6, 7, 9, 10));
    }

    #[test]
    fn a_prompt_gets_what_the_decode_iteration_leaves_of_the_target() {
        let mut p = pace();
        p.observe_iteration(Some(0.020));
        // 40 - 20 ms of slice, 4 ms a layer.
        assert_eq!(p.layers(512), 5);
        // A busier decode loop leaves less.
        let mut busy = pace();
        busy.observe_iteration(Some(0.032));
        assert_eq!(busy.layers(512), 2);
    }

    #[test]
    fn a_prompt_keeps_a_share_when_decode_is_over_the_target() {
        let mut p = pace();
        p.observe_iteration(Some(0.080));
        // 25 % of 80 ms is 20 ms, 4 ms a layer.
        assert_eq!(p.layers(512), 5);
    }

    #[test]
    fn a_slice_is_never_empty_and_never_more_than_the_cap() {
        let mut p = PrefillPace::new(Duration::from_millis(200));
        p.observe_slice(10, 512, 0.04);
        p.observe_iteration(Some(0.001));
        assert_eq!(p.layers(512), 25, "the 100 ms cap at 4 ms a layer, not the 199 ms the target leaves");
        p.observe_iteration(Some(0.9));
        assert!(p.layers(2048) >= 1);
    }

    #[test]
    fn without_a_known_decode_iteration_a_prompt_gets_half_the_target() {
        assert_eq!(pace().layers(512), 5);
    }

    #[test]
    fn a_pause_between_slices_is_not_a_decode_iteration() {
        let mut p = pace();
        p.observe_iteration(Some(0.020));
        p.observe_iteration(Some(2.0));
        assert_eq!(p.layers(512), 5, "back to half the target, as with no iteration seen: not what a 2 s gap would leave");
    }
}
