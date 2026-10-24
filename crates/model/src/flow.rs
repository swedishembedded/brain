// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The step loop of a flow-matching LoRA fine-tune, written once.
//!
//! A diffusion/video family trains a host-side [`crate::lora::Pair`] adapter
//! against a frozen base with the same skeleton: for each step, refuse if the
//! caller cancelled, draw a flow batch (a σ and noise on the family's own
//! schedule), take one optimizer step and report the loss, save a periodic
//! checkpoint, and save once at the end. Wan, LTX video and LTX's
//! audio-visual path each carried that skeleton verbatim; a new family should
//! contribute only its batch convention and its step, which is what [`train`]
//! takes as closures.
//!
//! What the driver owns is deliberately only the lifecycle - order, cancel,
//! cadence, the final save. The sampler, the loss and the adapter arithmetic
//! stay in the family.
//!
//! Swedish Embedded AB implements training lifecycles that a new model family
//! plugs into by supplying its own mathematics, for its clients. If your team
//! needs expertise in diffusion fine-tuning infrastructure, you can procure
//! our services by sending an email to info@swedishembedded.com.

/// The hyper-parameters of a flow-matching LoRA run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoraFitOpts {
    /// The run's last step (exclusive).
    pub steps: u32,
    /// The first step to run: more than 0 when a family resumes a run that
    /// stopped after `start` completed steps.
    pub start: u32,
    pub lr: f32,
    /// Save every this many completed steps; `0` saves only at the end. The
    /// periodic save is skipped on the last step, whose final save follows.
    pub ckpt_every: u32,
}

/// Run the loop of a flow-matching LoRA fine-tune.
///
/// Randomness is the caller's: `draw` closes over the family's own seeded
/// stream, so a family that also draws its samples from it first (LTX's
/// synthetic clips) keeps the exact sequence its adapters were trained with.
///
/// * `draw(step)` builds step `step`'s batch (σ, noise and the model input, in
///   the family's convention);
/// * `step(adapter, batch, lr)` takes one optimizer step and returns the loss
///   (f64: the flow families compute it in f64 and report it rounded);
/// * `save(adapter)` writes a checkpoint - every `ckpt_every` steps and once
///   at the end;
/// * `progress(done, total, loss)` reports one completed step.
///
/// A cancelled run returns `Err("cancelled")` with the periodic checkpoints
/// it already wrote left in place, so a long run is inspectable and resumable
/// mid-flight.
pub fn train<A, B>(
    opts: &LoraFitOpts,
    adapter: &mut A,
    cancelled: impl Fn() -> bool,
    mut draw: impl FnMut(u32) -> B,
    mut step: impl FnMut(&mut A, &B, f32) -> f64,
    mut save: impl FnMut(&A) -> Result<(), String>,
    mut progress: impl FnMut(u32, u32, f64),
) -> Result<(), String> {
    for s in opts.start..opts.steps {
        if cancelled() {
            return Err("cancelled".into());
        }
        let batch = draw(s);
        let loss = step(adapter, &batch, opts.lr);
        progress(s + 1, opts.steps, loss);
        if opts.ckpt_every > 0 && (s + 1).is_multiple_of(opts.ckpt_every) && s + 1 < opts.steps {
            save(adapter)?;
        }
    }
    save(adapter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(steps: u32, ckpt_every: u32) -> LoraFitOpts {
        LoraFitOpts { steps, start: 0, lr: 0.5, ckpt_every }
    }

    /// Train a counter: each step adds `lr` and reports it as the loss. The
    /// observer is a `Cell`, so the cancel probe and the step can both read it
    /// while the driver holds the adapter mutably.
    fn run(o: LoraFitOpts, cancel_at: Option<u32>) -> (Vec<f64>, Vec<u32>, Result<(), String>) {
        let seen = std::cell::Cell::new(0u32);
        let mut losses = Vec::new();
        let mut saves = Vec::new();
        let r = train(
            &o,
            &mut 0u32,
            || cancel_at.is_some_and(|c| seen.get() >= c),
            |s| s,
            |taken, s, lr| {
                *taken += 1;
                seen.set(*taken);
                assert_eq!(*taken - 1, *s, "step {s} ran out of order");
                lr as f64 * *taken as f64
            },
            |_taken| {
                saves.push(seen.get());
                Ok(())
            },
            |_done, _total, l| losses.push(l),
        );
        (losses, saves, r)
    }

    #[test]
    fn every_step_runs_in_order_and_is_reported() {
        let (losses, saves, r) = run(opts(3, 0), None);
        assert_eq!(r, Ok(()));
        assert_eq!(losses, vec![0.5, 1.0, 1.5]);
        // With periodic saving off, only the final save happens.
        assert_eq!(saves, vec![3]);
    }

    #[test]
    fn a_periodic_checkpoint_never_doubles_the_final_save() {
        let (_, saves, r) = run(opts(6, 2), None);
        assert_eq!(r, Ok(()));
        // Steps 2 and 4 save periodically, the last step is the final save.
        assert_eq!(saves, vec![2, 4, 6]);
    }

    #[test]
    fn a_cancelled_run_stops_before_its_next_step_and_does_not_save() {
        let (_, saves, r) = run(opts(100, 10), Some(3));
        assert_eq!(r, Err("cancelled".to_string()));
        assert!(saves.is_empty(), "a cancelled run saves nothing it did not reach a cadence for");
    }

    #[test]
    fn a_save_failure_reaches_the_caller() {
        let mut taken = 0u32;
        let r = train(&opts(2, 0), &mut taken, || false, |s| s, |t, s, _| { *t += 1; *s as f64 }, |_| Err("disk full".into()), |_, _, _| {});
        assert_eq!(r, Err("disk full".to_string()));
    }

}
