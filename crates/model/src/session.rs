// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The training session for a trainer whose step draws pre-encoded samples.
//!
//! `model::train`'s loop owns the schedule, accumulation, resume and cadence
//! for a [`crate::Model`]: a batch in, a forward/backward pair, an optimizer
//! step. A multi-part model (a vision tower plus a language decoder, a
//! caption encoder plus a policy head) does not fit that shape - it encodes
//! its samples once, then trains over the encoded set with its own
//! accumulator and optimizer - and each such family wrote the epoch order,
//! accumulation, learning-rate schedule, resume position, checkpoint cadence
//! and evaluation arithmetic by hand, identically, once per model.
//!
//! [`run`] is that lifecycle once. A family supplies [`SampleTrainer`]:
//! [`SampleTrainer::batch_step`], one optimizer step over the samples the
//! session drew. Everything else - order, schedule, accumulation count,
//! resume, cadence, evaluation points, the progress stream - is the
//! session's.
//!
//! Swedish Embedded AB implements training lifecycles that a new multi-part
//! model plugs into by supplying its own step, for its clients. If your team
//! needs expertise in training infrastructure for models with more than one
//! part, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;
use std::time::Instant;

use crate::train::{epoch_order, LrSchedule};

/// How a session orders its samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Every example once per epoch, reshuffled per epoch from the run's seed.
    Epochs { epochs: u32 },
    /// One uniform draw per step: a fixed number of steps over the set, the
    /// order a stream rather than an epoch.
    Stream,
    /// `steps` steps over a reshuffled stream: every sample once per pass
    /// through the set, in an order shuffled from the run's seed, refilled
    /// when it runs out.
    ShuffledStream,
}

/// When a session writes its state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cadence {
    /// After every this many optimizer steps; `0` writes only at the end.
    Steps(u32),
    /// At the end of every epoch.
    EpochEnd,
    /// Once this many seconds have elapsed, on the next completed step.
    Secs(u64),
}

/// The knobs of one session.
#[derive(Clone, Copy, Debug)]
pub struct SessionSpec {
    /// Examples in the training set.
    pub samples: usize,
    /// Samples accumulated into one optimizer step.
    pub batch: usize,
    /// Steps a [`Order::Stream`] run takes; ignored by an epoch run, whose
    /// step count follows from the set and the batch.
    pub steps: u32,
    pub order: Order,
    pub cadence: Cadence,
    pub seed: u64,
}

/// What a family supplies: one optimizer step over a drawn batch of samples.
pub trait SampleTrainer {
    /// Take one optimizer step over `samples` at rate `lr`; returns the mean
    /// loss of the batch.
    ///
    /// A trainer whose batch call owns both halves (forward every sample,
    /// backward, one optimizer step) implements this directly; one that
    /// accumulates sample by sample loops the two in its own impl.
    fn batch_step(&mut self, samples: &[usize], lr: f32) -> Result<f32, String>;

    /// Completed optimizer steps so far.
    fn steps(&self) -> u32;

    /// Restore a state written by [`SampleTrainer::checkpoint`]; returns the
    /// sample position it resumes at.
    fn restore(&mut self, _state: &Path) -> Result<u64, String> {
        Err("this trainer cannot resume".to_string())
    }

    /// Write the run's state at `position` (samples consumed so far).
    fn checkpoint(&self, _out: &Path, _position: u64) -> Result<(), String> {
        Ok(())
    }
}

/// One completed step, a checkpoint, or an evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    Step { step: u32, steps: u32, epoch: u32, loss: f32, lr: f32 },
    Checkpoint { step: u32 },
    Eval { epoch: u32, loss: f32 },
}

/// What a finished session reports.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Outcome {
    pub steps: u32,
    /// The evaluation taken before the first step, when the run starts at 0.
    pub initial_eval: Option<f32>,
    /// The evaluation taken after the last epoch, when one is configured.
    pub final_eval: Option<f32>,
    /// Samples consumed over the whole run: what a resume reads back.
    pub position: u64,
}

/// Run the session without a held-out evaluation - order, schedule,
/// accumulation, resume and cadence over a [`SampleTrainer`].
///
/// `schedule` maps the *step count* to a rate; `schedule.at(step)` is the
/// factor the family multiplies its peak learning rate by.
#[allow(clippy::too_many_arguments)]
pub fn run<T: SampleTrainer>(
    trainer: &mut T,
    spec: &SessionSpec,
    schedule: &LrSchedule,
    resume_from: Option<&Path>,
    state: &Path,
    cancelled: impl Fn() -> bool,
    progress: &mut dyn FnMut(Event),
) -> Result<Outcome, String> {
    run_evaluated(trainer, spec, schedule, resume_from, state, None::<fn(&mut T) -> Result<f32, String>>, cancelled, progress)
}

/// [`run`] with a held-out evaluation: called once before the first step
/// (when the run starts at the beginning) and at the end of every epoch.
#[allow(clippy::too_many_arguments)]
pub fn run_evaluated<T: SampleTrainer, E: FnMut(&mut T) -> Result<f32, String>>(
    trainer: &mut T,
    spec: &SessionSpec,
    schedule: &LrSchedule,
    resume_from: Option<&Path>,
    state: &Path,
    mut eval: Option<E>,
    cancelled: impl Fn() -> bool,
    progress: &mut dyn FnMut(Event),
) -> Result<Outcome, String> {
    let n = spec.samples;
    if n == 0 {
        return Err("the training session has no samples".to_string());
    }
    let batch = spec.batch.max(1).min(n);
    let epochs = match spec.order {
        Order::Epochs { epochs } => epochs,
        Order::Stream | Order::ShuffledStream => 1,
    };
    let per_epoch = n.div_ceil(batch) as u32;
    let steps = match spec.order {
        Order::Epochs { .. } => per_epoch * epochs,
        Order::Stream | Order::ShuffledStream => spec.steps,
    };
    let mut position = 0u64;
    if let Some(state) = resume_from.filter(|p| p.is_file()) {
        position = trainer.restore(state)?;
        if position > (n as u64) * epochs as u64 && matches!(spec.order, Order::Epochs { .. }) {
            return Err(format!("{}: position {position} is past this dataset's {epochs} epochs of {n} samples", state.display()));
        }
    }
    let initial_eval = match &mut eval {
        Some(f) if position == 0 => {
            let loss = f(trainer)?;
            progress(Event::Eval { epoch: 0, loss });
            Some(loss)
        }
        _ => None,
    };

    let mut rng = data::rng::Rng::new(spec.seed ^ 0xA5A5_5A5A);
    let mut last_save = Instant::now();
    let mut final_eval = None;
    let mut taken = trainer.steps();

    match spec.order {
        Order::Epochs { epochs } => {
            let start_epoch = (position / n as u64) as u32;
            for epoch in start_epoch..epochs {
                let order = epoch_order(spec.seed, n, epoch as u64);
                let mut i = (position - epoch as u64 * n as u64) as usize;
                while i < n {
                    if cancelled() {
                        return Err("cancelled".to_string());
                    }
                    let chunk = &order[i..(i + batch).min(n)];
                    let factor = schedule.at(taken);
                    let loss = trainer.batch_step(chunk, factor)?;
                    taken = trainer.steps();
                    i += chunk.len();
                    position += chunk.len() as u64;
                    progress(Event::Step { step: taken, steps, epoch: epoch + 1, loss, lr: factor });
                    if let Cadence::Steps(every) = spec.cadence {
                        if every > 0 && taken.is_multiple_of(every) {
                            trainer.checkpoint(state, position)?;
                            progress(Event::Checkpoint { step: taken });
                        }
                    }
                }
                trainer.checkpoint(state, position)?;
                if matches!(spec.cadence, Cadence::EpochEnd) {
                    progress(Event::Checkpoint { step: taken });
                }
                if let Some(f) = &mut eval {
                    let loss = f(trainer)?;
                    progress(Event::Eval { epoch: epoch + 1, loss });
                    final_eval = Some(loss);
                }
            }
        }
        Order::Stream | Order::ShuffledStream => {
            let mut pass = 0u64;
            let mut next = n;
            let mut perm: Vec<usize> = Vec::new();
            for _ in 0..steps {
                if cancelled() {
                    return Err("cancelled".to_string());
                }
                let drawn: Vec<usize> = (0..batch)
                    .map(|_| match spec.order {
                        Order::Stream => rng.gen_range_inclusive(0, n as i64 - 1) as usize,
                        // Every sample once per pass, in the run's own order.
                        _ => {
                            if next == n {
                                perm = epoch_order(spec.seed, n, pass);
                                pass += 1;
                                next = 0;
                            }
                            let pick = perm[next];
                            next += 1;
                            pick
                        }
                    })
                    .collect();
                let factor = schedule.at(taken);
                let loss = trainer.batch_step(&drawn, factor)?;
                taken = trainer.steps();
                position += batch as u64;
                progress(Event::Step { step: taken, steps, epoch: 1, loss, lr: factor });
                if let Cadence::Secs(secs) = spec.cadence {
                    if secs > 0 && last_save.elapsed().as_secs() >= secs {
                        trainer.checkpoint(state, position)?;
                        progress(Event::Checkpoint { step: taken });
                        last_save = Instant::now();
                    }
                }
                if let Cadence::Steps(every) = spec.cadence {
                    if every > 0 && taken.is_multiple_of(every) {
                        trainer.checkpoint(state, position)?;
                        progress(Event::Checkpoint { step: taken });
                    }
                }
            }
        }
    }
    trainer.checkpoint(state, position)?;
    Ok(Outcome { steps: taken, initial_eval, final_eval, position })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trainer over a set of counters: accumulating `k` adds `k`, stepping
    /// applies the rate to a running sum.
    #[derive(Default)]
    struct Fake {
        drawn: Vec<usize>,
        pending: f32,
        applied: f32,
        steps: u32,
        saved: Vec<u64>,
    }

    impl SampleTrainer for Fake {
        fn batch_step(&mut self, samples: &[usize], lr: f32) -> Result<f32, String> {
            let mut loss = 0.0;
            for &s in samples {
                self.drawn.push(s);
                loss += s as f32;
            }
            self.applied += lr * loss / samples.len() as f32;
            self.steps += 1;
            Ok(loss / samples.len() as f32)
        }
        fn steps(&self) -> u32 {
            self.steps
        }
        fn checkpoint(&self, _out: &Path, position: u64) -> Result<(), String> {
            // Records through interior mutability only in tests that need it.
            let _ = position;
            Ok(())
        }
        fn restore(&mut self, state: &Path) -> Result<u64, String> {
            std::fs::read_to_string(state).map_err(|e| e.to_string())?.trim().parse::<u64>().map_err(|e| e.to_string())
        }
    }

    fn spec(order: Order, samples: usize, batch: usize, steps: u32, cadence: Cadence, seed: u64) -> SessionSpec {
        SessionSpec { samples, batch, steps, order, cadence, seed }
    }

    fn flat(rate: f32) -> LrSchedule {
        LrSchedule { peak: rate, floor: rate, warmup: 0, hold: 0, decay_iters: 1 }
    }

    #[test]
    fn an_epoch_run_draws_every_sample_once_per_epoch_and_steps_once_per_batch() {
        let mut t = Fake::default();
        let o = run(&mut t, &spec(Order::Epochs { epochs: 2 }, 5, 2, 0, Cadence::Steps(0), 7), &flat(1.0), None, Path::new("/nonexistent"), || false, &mut |_| {}).expect("run");
        assert_eq!(o.steps, 6, "5 samples at batch 2 is 3 steps an epoch");
        assert_eq!(t.drawn.len(), 10);
        let mut first = t.drawn[..5].to_vec();
        first.sort();
        assert_eq!(first, vec![0, 1, 2, 3, 4], "every sample once");
        assert_eq!(o.position, 10);
    }

    #[test]
    fn the_order_is_the_run_s_own_and_repeats_for_the_same_seed() {
        let draw = |seed: u64| {
            let mut t = Fake::default();
            run(&mut t, &spec(Order::Epochs { epochs: 1 }, 8, 1, 0, Cadence::Steps(0), seed), &flat(1.0), None, Path::new("/x"), || false, &mut |_| {}).expect("run");
            t.drawn
        };
        assert_eq!(draw(11), draw(11));
        assert_ne!(draw(11), draw(12), "different seeds shuffle differently");
    }

    #[test]
    fn a_resume_continues_where_the_state_says() {
        let dir = std::env::temp_dir().join(format!("brain-session-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp");
        let state = dir.join("state");
        std::fs::write(&state, "3\n").expect("write");
        let mut t = Fake::default();
        let o = run(&mut t, &spec(Order::Epochs { epochs: 1 }, 6, 1, 0, Cadence::Steps(0), 5), &flat(1.0), Some(&state), &state, || false, &mut |_| {}).expect("run");
        assert_eq!(t.drawn.len(), 3, "three samples remain of six");
        assert_eq!(o.position, 6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn evaluation_runs_before_the_first_step_and_after_every_epoch() {
        let mut evals = Vec::new();
        let mut t = Fake::default();
        let mut eval = |_t: &mut Fake| Ok(1.5);
        let o = run_evaluated(
            &mut t,
            &spec(Order::Epochs { epochs: 2 }, 2, 1, 0, Cadence::Steps(0), 3),
            &flat(1.0),
            None,
            Path::new("/x"),
            Some(&mut eval),
            || false,
            &mut |e| {
                if let Event::Eval { epoch, loss } = e {
                    evals.push((epoch, loss));
                }
            },
        )
        .expect("run");
        assert_eq!(evals, vec![(0, 1.5), (1, 1.5), (2, 1.5)]);
        assert_eq!((o.initial_eval, o.final_eval), (Some(1.5), Some(1.5)));
    }

    #[test]
    fn a_stream_run_takes_the_step_count_it_is_given() {
        let mut t = Fake::default();
        let o = run(&mut t, &spec(Order::Stream, 4, 2, 5, Cadence::Steps(0), 9), &flat(1.0), None, Path::new("/x"), || false, &mut |_| {}).expect("run");
        assert_eq!(o.steps, 5);
        assert_eq!(t.drawn.len(), 10, "batch 2 over 5 steps");
        assert!(t.drawn.iter().all(|&i| i < 4));
    }

    #[test]
    fn a_shuffled_stream_covers_every_sample_once_per_pass() {
        let mut t = Fake::default();
        let o = run(&mut t, &spec(Order::ShuffledStream, 5, 2, 4, Cadence::Steps(0), 3), &flat(1.0), None, Path::new("/x"), || false, &mut |_| {}).expect("run");
        assert_eq!(o.steps, 4);
        assert_eq!(t.drawn.len(), 8, "4 steps of batch 2");
        let mut first = t.drawn[..5].to_vec();
        first.sort();
        assert_eq!(first, vec![0, 1, 2, 3, 4], "the first pass is every sample once");
        assert_eq!(t.drawn[5..].len(), 3, "the second pass continues");
    }

    #[test]
    fn the_schedule_advances_with_the_step_count() {
        let mut t = Fake::default();
        let schedule = LrSchedule { peak: 1.0, floor: 0.0, warmup: 2, hold: 0, decay_iters: 4 };
        let mut rates = Vec::new();
        run(&mut t, &spec(Order::Epochs { epochs: 1 }, 4, 1, 0, Cadence::Steps(0), 1), &schedule, None, Path::new("/x"), || false, &mut |e| {
            if let Event::Step { lr, .. } = e {
                rates.push(lr);
            }
        })
        .expect("run");
        assert_eq!(rates.len(), 4);
        assert!(rates[0] <= rates[1], "warmup rises");
        assert!(rates[3] < rates[1], "then decays");
    }
}
