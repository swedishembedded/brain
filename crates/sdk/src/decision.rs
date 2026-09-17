// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::DecisionPipeline` - calibrated probabilities over options the
//! CALLER supplies, instead of text.
//!
//! ```no_run
//! use brain::{Choice, DecisionPipeline};
//! let mut pipe = DecisionPipeline::from_pretrained("/path/to/all-MiniLM-L6-v2")?;
//! let answer = pipe.choose(
//!     "I am still waiting on my card",
//!     "which banking intent does this message express",
//!     &["card arrival", "exchange rate", "pin blocked"],
//! )?;
//! println!("{} ({:.2} confident)", answer.choice, answer.confidence);
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! Unlike every generative pipeline here, the OPTIONS are part of the call.
//! The same loaded model answers a three-option question and a
//! two-hundred-option one without reloading, because the output space lives in
//! the request rather than in a final layer.
//!
//! **Takes a path, not a hub id**, for the same reason
//! [`crate::TextGenerationPipeline`] does: this architecture has no model-store
//! `ArchSpec` yet, so there is nothing for a resolver to resolve against. A
//! decision model is also two things - an imported encoder and a trained head -
//! and a hub id names only the first.

use std::path::Path;

use decide::banking77::{Banking77, OptionSampler};
use decide::decide::{Decide, Example, Limits};
use decide::loss::LossConfig;
use decide::primitives::{Answer, Opt, Question};

pub use decide::banking77::Row;
pub use decide::decide::Limits as DecisionLimits;

use crate::flow::{EvalReport, Flow, Stages, TrainReport};
use crate::{Device, Error, Result};

/// What a `Choice` question answered.
#[derive(Clone, Debug)]
pub struct Choice {
    /// The highest-probability option, verbatim as it was supplied.
    pub choice: String,
    /// Every option's probability, in the order they were supplied. Always
    /// returned in full, so a caller who prefers the maximum probability or
    /// the margin over [`Choice::confidence`] can compute it.
    pub probabilities: Vec<(String, f32)>,
    /// How concentrated the distribution is, on `[0, 1]`, independent of how
    /// many options there were.
    pub confidence: f32,
}

/// The encoder arrives pretrained and the head does not, so they move at
/// different rates. One rate would either leave the head too slow to learn or
/// move the encoder fast enough to forget what it was imported for.
const ENCODER_LR: f32 = 2e-5;
const HEAD_LR: f32 = 1e-3;

pub struct DecisionPipeline {
    model: Decide,
    /// The option set and question the last training stage used, so the
    /// interactive stages need no second copy of them.
    last_options: Vec<String>,
    last_instructions: String,
    last_eval: Vec<(String, usize)>,
}

/// Load the BANKING77 intent dataset from a directory holding
/// `categories.json`, `train.csv` and `test.csv`.
///
/// Here rather than in the sample that uses it because a sample may depend on
/// no brain crate except this one.
pub fn banking77(dir: impl AsRef<std::path::Path>) -> Result<Banking77> {
    Banking77::load(dir.as_ref()).map_err(Error::Backend)
}

pub use decide::banking77::Banking77 as Banking77Data;

impl std::fmt::Debug for DecisionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionPipeline").finish_non_exhaustive()
    }
}

impl DecisionPipeline {
    /// Load an encoder checkpoint directory (`config.json`, `model.safetensors`
    /// and `tokenizer.json`), with a freshly initialized head, and START A
    /// STAGE CHAIN on it.
    ///
    /// Returns a [`Flow`] rather than a bare pipeline so the chain reads as
    /// one expression: a load failure becomes the flow's error and every later
    /// stage skips, so there is one error site at `finish` instead of one per
    /// stage. `DecisionPipeline::builder(..).load()` is the same thing without
    /// the chain.
    ///
    /// An untrained head answers, and answers badly: use
    /// [`DecisionPipelineBuilder::head`] to supply trained weights, or
    /// [`Flow::train`] to make some.
    pub fn from_pretrained(dir: impl AsRef<str>) -> Flow<DecisionPipeline> {
        Flow::new(DecisionPipeline::builder(dir).load())
    }

    pub fn builder(dir: impl AsRef<str>) -> DecisionPipelineBuilder {
        DecisionPipelineBuilder {
            dir: dir.as_ref().to_string(),
            head: None,
            device: Device::default(),
            limits: Limits::default(),
            seed: 0,
        }
    }

    /// Pick one of `options` for `state`, given what the question asks.
    pub fn choose(&mut self, state: &str, instructions: &str, options: &[&str]) -> Result<Choice> {
        let q = Question::Choice {
            instructions: instructions.to_string(),
            options: options.iter().map(|o| Opt::new(*o)).collect(),
        };
        let mut answers = self.model.decide(state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
        match answers.pop() {
            Some(Answer::Choice { choice, probabilities, confidence }) => {
                Ok(Choice { choice, probabilities, confidence })
            }
            _ => Err(Error::Backend("the model did not return a choice".into())),
        }
    }

    /// The probability that a proposition holds, on `[0, 1]`.
    pub fn probability(&mut self, state: &str, proposition: &str) -> Result<f32> {
        let q = Question::Noul { instructions: proposition.to_string(), yes: None, no: None };
        let mut answers = self.model.decide(state, std::slice::from_ref(&q)).map_err(Error::Backend)?;
        match answers.pop() {
            Some(Answer::Noul { noul }) => Ok(noul),
            _ => Err(Error::Backend("the model did not return a probability".into())),
        }
    }

    /// Tell this pipeline what question its interactive stages ask, and over
    /// which options.
    ///
    /// Needed when a model is loaded from trained weights rather than trained
    /// in this process: the option set lives in the REQUEST, not in the
    /// checkpoint, so weights alone do not say what the model is deciding
    /// between. Training sets the same fields as a side effect.
    pub fn set_question(&mut self, instructions: impl Into<String>, options: Vec<String>) {
        self.last_instructions = instructions.into();
        self.last_options = options;
    }

    /// Fine-tune on labelled examples: `(text, label)` pairs plus the option
    /// text every label maps to.
    ///
    /// Each step scores its example against a random SUBSET of the options,
    /// always containing the correct one, at a random position. That is the
    /// whole difference between training a decision model and training a
    /// classifier: a model always shown the same option list in the same order
    /// can reach the right answer from the POSITION, never having to read an
    /// option at all - and it then cannot answer a question whose options it
    /// has not seen, which is the one thing this model is for.
    ///
    /// Returns the mean loss over the last tenth of the run.
    pub fn train_choices(
        &mut self,
        examples: &[(&str, usize)],
        options: &[String],
        instructions: &str,
        steps: usize,
        seed: u64,
        log: &mut dyn FnMut(usize, f32),
    ) -> Result<f32> {
        if examples.is_empty() {
            return Err(Error::MissingArgument("train_choices needs at least one example".into()));
        }
        let pool: Vec<usize> = (0..options.len()).collect();
        let sampler = OptionSampler::default();
        let loss_cfg = LossConfig::cross_entropy();
        let mut rng = data::rng::Rng::new(seed);
        let tail = (steps / 10).max(1);
        let mut tail_sum = 0.0f32;

        for step in 0..steps {
            let (text, label) = examples[(rng.next_u64() % examples.len() as u64) as usize];
            if label >= options.len() {
                return Err(Error::MissingArgument(format!(
                    "example label {label} has no option text ({} supplied)",
                    options.len()
                )));
            }
            let (drawn, gold) = sampler.draw(label, &pool, &mut rng);
            let q = Question::Choice {
                instructions: instructions.to_string(),
                options: drawn.iter().map(|&i| Opt::new(options[i].clone())).collect(),
            };
            let l = self
                .model
                .train_step(&Example { state: text, question: &q, gold }, &loss_cfg, ENCODER_LR, HEAD_LR)
                .map_err(Error::Backend)?;
            log(step, l);
            if step >= steps.saturating_sub(tail) {
                tail_sum += l / tail as f32;
            }
        }
        Ok(tail_sum)
    }

    /// Write the trained head to a brain `.safetensors`, for
    /// [`DecisionPipelineBuilder::head`] to load back.
    pub fn save_head(&self, path: impl AsRef<str>) -> Result<()> {
        self.model.save_head(path.as_ref()).map_err(Error::Backend)
    }

    /// The underlying model, for training and for the question types this
    /// three-line surface does not cover.
    pub fn inner(&mut self) -> &mut Decide {
        &mut self.model
    }
}

/// What a decision model needs in order to train: labelled examples, the
/// option text each label maps to, and what the question asks.
///
/// Owned rather than borrowed so a chain can be written as one expression
/// without the caller keeping every intermediate alive.
#[derive(Clone, Debug, Default)]
pub struct TrainSpec {
    pub examples: Vec<(String, usize)>,
    pub options: Vec<String>,
    pub instructions: String,
    pub steps: usize,
    pub seed: u64,
    /// Held out for [`Flow::evaluate`]. Empty means evaluation reports nothing
    /// rather than inventing a split.
    pub eval: Vec<(String, usize)>,
}

impl TrainSpec {
    pub fn new(instructions: impl Into<String>) -> TrainSpec {
        TrainSpec { instructions: instructions.into(), steps: 600, ..TrainSpec::default() }
    }

    pub fn examples(mut self, examples: Vec<(String, usize)>) -> TrainSpec {
        self.examples = examples;
        self
    }

    pub fn options(mut self, options: Vec<String>) -> TrainSpec {
        self.options = options;
        self
    }

    pub fn eval(mut self, eval: Vec<(String, usize)>) -> TrainSpec {
        self.eval = eval;
        self
    }

    pub fn steps(mut self, steps: usize) -> TrainSpec {
        self.steps = steps;
        self
    }

    pub fn seed(mut self, seed: u64) -> TrainSpec {
        self.seed = seed;
        self
    }
}

impl Stages for DecisionPipeline {
    type TrainSpec = TrainSpec;

    fn describe(&self) -> String {
        format!(
            "decision model, {} options in the last call, {} training steps so far",
            self.last_options.len(),
            self.model.steps_taken()
        )
    }

    fn run_train(&mut self, spec: &TrainSpec, log: &mut dyn FnMut(usize, f32)) -> Result<TrainReport> {
        let ex: Vec<(&str, usize)> = spec.examples.iter().map(|(t, l)| (t.as_str(), *l)).collect();
        let final_loss =
            self.train_choices(&ex, &spec.options, &spec.instructions, spec.steps, spec.seed, log)?;
        // Remembered so `evaluate` and `ask` need no second copy of the
        // question: a chain should not make the caller repeat itself.
        self.last_options = spec.options.clone();
        self.last_instructions = spec.instructions.clone();
        self.last_eval = spec.eval.clone();
        Ok(TrainReport { steps: spec.steps, final_loss, seconds: 0.0 })
    }

    fn run_eval(&mut self) -> Result<EvalReport> {
        if self.last_eval.is_empty() {
            return Ok(EvalReport::default());
        }
        let options = self.last_options.clone();
        let refs: Vec<&str> = options.iter().map(String::as_str).collect();
        let instructions = self.last_instructions.clone();
        let eval = self.last_eval.clone();
        let (mut hit, mut confidence) = (0usize, 0.0f32);
        for (text, label) in &eval {
            let a = self.choose(text, &instructions, &refs)?;
            confidence += a.confidence / eval.len() as f32;
            if a.choice == options[*label] {
                hit += 1;
            }
        }
        Ok(EvalReport {
            accuracy: hit as f32 / eval.len() as f32,
            items: eval.len(),
            notes: vec![
                ("chance".into(), 1.0 / options.len().max(1) as f32),
                ("mean confidence".into(), confidence),
            ],
        })
    }

    fn run_save(&self, path: &str) -> Result<()> {
        self.save_head(path)
    }

    fn set_output_space(&mut self, instructions: &str, options: Vec<String>) -> Result<()> {
        self.set_question(instructions, options);
        Ok(())
    }

    fn run_turn(&mut self, input: &str) -> Result<String> {
        if self.last_options.is_empty() {
            return Err(Error::MissingArgument(
                "this pipeline has no option set yet - train it, or call `choose` with one".into(),
            ));
        }
        let options = self.last_options.clone();
        let refs: Vec<&str> = options.iter().map(String::as_str).collect();
        let instructions = self.last_instructions.clone();
        let a = self.choose(input, &instructions, &refs)?;
        let mut top = a.probabilities.clone();
        top.sort_by(|x, y| y.1.total_cmp(&x.1));
        let mut out = format!("  -> {}  (confidence {:.2})", a.choice, a.confidence);
        for (name, p) in top.iter().take(3) {
            out.push_str(&format!("\n       {p:>6.3}  {name}"));
        }
        if a.confidence < 0.3 {
            // The reason a decision model returns a distribution and not a
            // label: the CALLER decides what is confident enough to act on.
            out.push_str("\n       (low confidence - a real system would escalate this one)");
        }
        Ok(out)
    }
}

pub struct DecisionPipelineBuilder {
    dir: String,
    head: Option<String>,
    device: Device,
    limits: Limits,
    seed: u64,
}

impl DecisionPipelineBuilder {
    /// Trained head weights, as written by `decide`'s training loop. Without
    /// this the head is random and the model's answers are noise.
    pub fn head(mut self, path: impl AsRef<str>) -> DecisionPipelineBuilder {
        self.head = Some(path.as_ref().to_string());
        self
    }

    pub fn device(mut self, device: Device) -> DecisionPipelineBuilder {
        self.device = device;
        self
    }

    /// How large a request this model is built for. Raising these costs device
    /// memory at build time, not per call.
    pub fn limits(mut self, limits: Limits) -> DecisionPipelineBuilder {
        self.limits = limits;
        self
    }

    /// Seed for the head's initialization, when no trained head is supplied.
    pub fn seed(mut self, seed: u64) -> DecisionPipelineBuilder {
        self.seed = seed;
        self
    }

    /// Build a model ready to answer. Trainable, so the same object a caller
    /// loads is the one it can fine-tune.
    pub fn load(self) -> Result<DecisionPipeline> {
        let dir = Path::new(&self.dir);
        let cfg_json = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| Error::Backend(format!("read {}/config.json: {e}", self.dir)))?;
        let cfg = decide::import::config_from_hf(&cfg_json).map_err(Error::Backend)?;
        let weights = dir.join("model.safetensors");
        let tensors = checkpoint::safetensors::read(
            weights.to_str().ok_or_else(|| Error::Backend("non-UTF-8 weights path".into()))?,
        )
        .map_err(|e| Error::Backend(format!("read {}: {e}", weights.display())))?;
        let enc_init = decide::import::brain_init_from_hf(tensors, &cfg).map_err(Error::Backend)?;

        let tok_path = dir.join("tokenizer.json");
        let tok = data::wordpiece::WordPiece::from_file(
            tok_path.to_str().ok_or_else(|| Error::Backend("non-UTF-8 tokenizer path".into()))?,
        )
        .map_err(Error::Backend)?;

        let head_init = match &self.head {
            Some(p) => {
                let t = checkpoint::safetensors::read(p)
                    .map_err(|e| Error::Backend(format!("read {p}: {e}")))?;
                t.into_iter().map(|x| (x.name, x.data)).collect()
            }
            None => decide::init::init_head(&cfg, self.seed),
        };

        crate::device::resolve(&self.device)?;
        let gpu = gpu_core::Gpu::new(decide::kern::PIPELINES);
        let model = Decide::new_on(gpu, cfg, tok, self.limits, &enc_init, &head_init, true);
        Ok(DecisionPipeline { model, last_options: Vec::new(), last_instructions: String::new(), last_eval: Vec::new() })
    }
}
