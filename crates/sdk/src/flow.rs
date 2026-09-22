// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The stage chain every pipeline gets, whatever architecture it runs.
//!
//! ```no_run
//! # use brain::{DecisionPipeline, Stages, TrainSpec};
//! DecisionPipeline::from_pretrained("/path/to/encoder")
//!     .train(TrainSpec::default().steps(2000))
//!     .evaluate()
//!     .save("out/head.safetensors")
//!     .ask("my card never arrived")
//!     .report()
//!     .finish()?;
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! **The chain does not stop at the first `?`.** A [`Flow`] carries either a
//! live pipeline or the error that ended it, and a stage applied to a failed
//! flow is a no-op that keeps the original cause. So a five-stage program has
//! one error site, at [`Flow::finish`], instead of five - and the stage that
//! actually broke is still the one named, because later stages never overwrite
//! an earlier failure.
//!
//! **Generic stages, architecture-specific payloads.** `train`, `evaluate`,
//! `save`, `ask`, `tui`, `report` and `finish` are written once here and
//! inherited by every pipeline. What each MEANS is supplied per architecture
//! through [`Stages`], including the type of its training specification - a
//! decision model trains on labelled options and an image model does not, and
//! neither has to pretend otherwise to share the chain.

use std::io::Write;

use crate::{Error, Result};

/// What one training stage produced.
#[derive(Clone, Debug, Default)]
pub struct TrainReport {
    pub steps: usize,
    /// Mean loss over the last tenth of the run.
    pub final_loss: f32,
    pub seconds: f32,
}

/// What one evaluation stage produced.
#[derive(Clone, Debug, Default)]
pub struct EvalReport {
    /// Fraction correct, where the pipeline has a notion of correct.
    pub accuracy: f32,
    /// How many items were scored.
    pub items: usize,
    /// Free-form extras a given architecture wants on the record.
    pub notes: Vec<(String, f32)>,
}

/// The per-architecture half of the chain.
///
/// Everything a pipeline must say about itself for the generic stages to mean
/// something. A pipeline that cannot do a stage returns an error naming it
/// rather than silently doing nothing.
pub trait Stages: Sized {
    /// What this architecture needs in order to train. A decision model wants
    /// labelled examples and an option vocabulary; another architecture wants
    /// something else entirely, and neither has to model the other's needs.
    type TrainSpec;

    /// One line naming what is loaded, for [`Flow::report`].
    fn describe(&self) -> String;

    fn run_train(&mut self, spec: &Self::TrainSpec, log: &mut dyn FnMut(usize, f32)) -> Result<TrainReport>;

    fn run_eval(&mut self) -> Result<EvalReport>;

    fn run_save(&self, path: &str) -> Result<()>;

    /// One interactive turn: input in, a printable answer out.
    fn run_turn(&mut self, input: &str) -> Result<String>;

    /// The prompt [`Flow::tui`] shows.
    fn turn_prompt(&self) -> &str {
        "> "
    }

    /// Whether THIS loaded pipeline's backend can be trained at all.
    ///
    /// Default `true`, and every architecture here today returns it -
    /// including both of [`crate::DecisionPipeline`]'s arms, since the Laya
    /// arm gained a real training loop. It stays because it is the general
    /// shape for a backend that arrives pretrained with nothing to train
    /// here (a quantized/distilled export, a hub-imported checkpoint with no
    /// brain-side training path), and because a caller that asks is doing
    /// the right thing whatever is behind it. A caller checks
    /// this BEFORE calling [`Flow::train`], rather than calling it and
    /// parsing the resulting error, because the two outcomes it wants -
    /// train normally, or skip straight to evaluating what arrived already
    /// trained - are genuinely different next stages to chain, not the same
    /// code path with a swallowed failure. Overriding this is how an
    /// architecture says so honestly instead of a sample having to sniff
    /// which backend it got.
    fn supports_training(&self) -> bool {
        true
    }

    /// Declare what the interactive stages decide between, for an
    /// architecture whose output space is part of the request. The default
    /// refuses, because for most architectures there is nothing to declare.
    fn set_output_space(&mut self, _instructions: &str, _options: Vec<String>) -> Result<()> {
        Err(Error::MissingArgument(
            "this pipeline's output space is fixed by its weights, not by the request".into(),
        ))
    }
}

/// A pipeline moving through stages, or the error that ended it.
pub struct Flow<P> {
    inner: std::result::Result<P, Error>,
    log: Vec<String>,
    train: Option<TrainReport>,
    eval: Option<EvalReport>,
}

impl<P> Flow<P> {
    /// Start a chain from a construction that may have failed.
    pub fn new(inner: Result<P>) -> Flow<P> {
        Flow { inner, log: Vec::new(), train: None, eval: None }
    }

    /// The pipeline, or the first error that stopped the chain.
    pub fn finish(self) -> Result<P> {
        self.inner
    }

    /// Whether the chain is still live.
    pub fn is_ok(&self) -> bool {
        self.inner.is_ok()
    }

    pub fn train_report(&self) -> Option<&TrainReport> {
        self.train.as_ref()
    }

    pub fn eval_report(&self) -> Option<&EvalReport> {
        self.eval.as_ref()
    }

    /// Whether the pipeline this chain is carrying supports [`Flow::train`] -
    /// see [`Stages::supports_training`]'s own doc for why this exists as an
    /// explicit query rather than [`Flow::train`] silently skipping itself.
    ///
    /// `false` on a chain that already failed: there is no live pipeline to
    /// ask, and [`Flow::train`] on a failed chain is already a documented
    /// no-op that preserves the original error, so `false` is never the
    /// wrong answer to give here - it never causes a caller to skip a train
    /// stage that would otherwise have run.
    pub fn supports_training(&self) -> bool
    where
        P: Stages,
    {
        self.inner.as_ref().map(Stages::supports_training).unwrap_or(false)
    }

    /// Run `f` only while the chain is live, recording `note` either way.
    ///
    /// `pub(crate)` so an architecture may add a stage of its OWN through the
    /// same seam - `Flow<ConversionPipeline>::replay` is one. Going through
    /// here rather than reaching into the fields is what keeps every stage,
    /// generic or not, obeying the one rule that matters: a failed chain is
    /// never restarted and its first cause is never overwritten.
    pub(crate) fn stage(mut self, note: &str, f: impl FnOnce(&mut P) -> Result<Option<String>>) -> Flow<P> {
        match &mut self.inner {
            Err(_) => {
                // Deliberately NOT overwritten: the first failure is the one
                // worth reporting, and a later stage's "there is no model"
                // would bury it.
                self.log.push(format!("{note}: skipped (an earlier stage failed)"));
                self
            }
            Ok(p) => match f(p) {
                Ok(extra) => {
                    self.log.push(match extra {
                        Some(e) => format!("{note}: {e}"),
                        None => format!("{note}: ok"),
                    });
                    self
                }
                Err(e) => {
                    self.log.push(format!("{note}: FAILED - {e}"));
                    self.inner = Err(e);
                    self
                }
            },
        }
    }
}

impl<P: Stages> Flow<P> {
    /// Fine-tune, printing progress as it goes.
    pub fn train(self, spec: P::TrainSpec) -> Flow<P> {
        let mut report = None;
        let out = self.stage("train", |p| {
            let t0 = std::time::Instant::now();
            let mut r = p.run_train(&spec, &mut |step, loss| {
                if step % 100 == 0 {
                    println!("  step {step:>6}  loss {loss:.4}");
                }
            })?;
            r.seconds = t0.elapsed().as_secs_f32();
            let line = format!(
                "{} steps, final loss {:.4}, {:.1}s ({:.0} ms/step)",
                r.steps,
                r.final_loss,
                r.seconds,
                r.seconds * 1000.0 / r.steps.max(1) as f32
            );
            report = Some(r);
            Ok(Some(line))
        });
        Flow { train: report, ..out }
    }

    /// Score the pipeline on whatever it holds out.
    pub fn evaluate(self) -> Flow<P> {
        let mut report = None;
        let out = self.stage("evaluate", |p| {
            let r = p.run_eval()?;
            let mut line = format!("accuracy {:.3} over {} items", r.accuracy, r.items);
            for (k, v) in &r.notes {
                line.push_str(&format!(", {k} {v:.3}"));
            }
            report = Some(r);
            Ok(Some(line))
        });
        Flow { eval: report, ..out }
    }

    /// Write whatever this run produced.
    pub fn save(self, path: impl AsRef<str>) -> Flow<P> {
        let path = path.as_ref().to_string();
        self.stage("save", |p| {
            if let Some(dir) = std::path::Path::new(&path).parent() {
                if !dir.as_os_str().is_empty() {
                    std::fs::create_dir_all(dir)
                        .map_err(|e| Error::Backend(format!("create {}: {e}", dir.display())))?;
                }
            }
            p.run_save(&path)?;
            Ok(Some(path.clone()))
        })
    }

    /// One turn, printed. The non-interactive half of inference.
    pub fn ask(self, input: impl AsRef<str>) -> Flow<P> {
        let input = input.as_ref().to_string();
        self.stage("ask", |p| {
            let answer = p.run_turn(&input)?;
            println!("{answer}");
            Ok(Some(format!("{input:?}")))
        })
    }

    /// Keep answering until end of input: the ONGOING half of inference.
    ///
    /// A closed stdin ends the loop rather than failing it - a program run
    /// non-interactively has simply finished asking, which is not an error.
    pub fn tui(self) -> Flow<P> {
        self.stage("tui", |p| {
            let mut turns = 0usize;
            let mut line = String::new();
            loop {
                print!("{}", p.turn_prompt());
                let _ = std::io::stdout().flush();
                line.clear();
                match std::io::stdin().read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) => return Err(Error::Backend(format!("stdin: {e}"))),
                }
                let input = line.trim();
                if input.is_empty() {
                    break;
                }
                match p.run_turn(input) {
                    Ok(answer) => println!("{answer}"),
                    // One bad turn does not end a session.
                    Err(e) => eprintln!("  ! {e}"),
                }
                turns += 1;
            }
            println!();
            Ok(Some(format!("{turns} turns")))
        })
    }

    /// Set the question the interactive stages ask, without training.
    ///
    /// Generic in shape, architecture-specific in meaning: a pipeline whose
    /// output space is part of the request needs to be told it, and one whose
    /// is not ignores this.
    pub fn with_question(self, instructions: &str, options: Vec<String>) -> Flow<P> {
        let n = options.len();
        let instructions = instructions.to_string();
        self.stage("question", move |p| {
            p.set_output_space(&instructions, options)?;
            Ok(Some(format!("{n} options")))
        })
    }

    /// Print what every stage did, in order.
    pub fn report(self) -> Flow<P> {
        println!("\n--- pipeline ---");
        if let Ok(p) = &self.inner {
            println!("  model: {}", p.describe());
        }
        for line in &self.log {
            println!("  {line}");
        }
        if let Err(e) = &self.inner {
            println!("  ended with: {e}");
        }
        println!("----------------\n");
        self
    }
}
