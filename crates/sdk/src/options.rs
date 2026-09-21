// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Command-line option groups, composed.
//!
//! The mechanism and the core groups are `appopts`, re-exported here so an
//! application needs one dependency. What this module adds is the groups that
//! belong to a SURFACE - the training knobs next to the pipeline they
//! configure, the window flags next to the viewport that opens the window -
//! each behind the same feature as the surface itself.
//!
//! That is the rule worth stating, because the alternative is tempting and
//! wrong: putting every group in `appopts` would make one crate a second
//! description of every part of brain, and a sample that wants a window would
//! pull the training spec with it.
//!
//! ```no_run
//! # #[cfg(all(feature = "decision", feature = "viewport"))]
//! # fn demo() -> Result<(), String> {
//! use brain::options::{Args, ControlOptions, Hardware, Options, ViewOptions};
//!
//! let argv: Vec<String> = std::env::args().skip(1).collect();
//! let mut args = Args::new(&argv);
//! let hw = Hardware::take(&mut args)?;        // --device, --backend
//! let train = ControlOptions::take(&mut args)?; // --iterations, --episodes, ...
//! let view = ViewOptions::take(&mut args)?;   // --window, --frames, --fps
//! let my_own = args.usize_or("--rounds", 3);  // whatever is left is yours
//! args.finish();
//! # Ok(()) }
//! ```

pub use appopts::{help_of, Args, Hardware, Options};


/// Where the model comes from and where it goes: the four flags every
/// application that loads or trains a decision model needs.
///
/// Factored out because it is the part two otherwise unrelated groups share -
/// a reinforcement-learning run and a supervised one disagree about almost
/// everything else, and agree exactly here. Spelling `--encoder` twice in two
/// groups is how the two come to accept slightly different things.
#[cfg(feature = "decision")]
#[derive(Clone, Debug, Default)]
pub struct ModelOptions {
    /// Directory holding the pretrained sentence encoder.
    pub encoder: String,
    /// Trained head to start from, or to run.
    pub head: Option<String>,
    /// Where a trained head is written.
    pub save: String,
    pub seed: u64,
}

#[cfg(feature = "decision")]
impl ModelOptions {
    pub fn new(encoder: &str, save: &str) -> ModelOptions {
        ModelOptions {
            encoder: encoder.to_string(),
            head: None,
            save: save.to_string(),
            seed: crate::ControlSpec::default().seed,
        }
    }

    /// Take the flags, keeping whatever this value already holds as the
    /// default.
    pub fn take_over(mut self, args: &mut Args) -> Result<ModelOptions, String> {
        self.encoder = args.str_or("--encoder", &self.encoder.clone());
        self.head = args.take_str("--head").or(self.head);
        self.save = args.str_or("--save", &self.save.clone());
        self.seed = args.u64_or("--seed", self.seed);
        Ok(self)
    }

    /// Fail with the remedy attached when no encoder was named. Every
    /// application needs this check and none of them should word it
    /// differently.
    pub fn require_encoder(&self) -> Result<&str, String> {
        if self.encoder.is_empty() {
            return Err("--encoder DIR is required: it is the pretrained sentence encoder the \
                        model reads with (`brain pull sentence-transformers/all-MiniLM-L6-v2` \
                        fetches one)"
                .into());
        }
        Ok(&self.encoder)
    }
}

#[cfg(feature = "decision")]
impl Options for ModelOptions {
    fn take(args: &mut Args) -> Result<ModelOptions, String> {
        ModelOptions::new("", "out/head.safetensors").take_over(args)
    }

    fn help() -> &'static str {
        "  --encoder DIR       pretrained sentence encoder
  --head FILE         start from (or run) these head weights
  --save FILE         where a trained head is written
  --seed N"
    }
}

/// Supervised training of a decision head: learn to pick the right option from
/// labelled examples, with no environment in the loop.
///
/// The counterpart to [`ControlOptions`]. Same model flags, different training
/// question - which is exactly why the model flags are their own group.
#[cfg(feature = "decision")]
#[derive(Clone, Debug)]
pub struct SupervisedOptions {
    pub model: ModelOptions,
    /// Optimizer steps.
    pub steps: usize,
    /// Examples held back from training and scored afterwards.
    pub eval: usize,
}

#[cfg(feature = "decision")]
impl SupervisedOptions {
    pub fn new(model: ModelOptions) -> SupervisedOptions {
        SupervisedOptions { model, steps: 2000, eval: 500 }
    }

    pub fn take_over(mut self, args: &mut Args) -> Result<SupervisedOptions, String> {
        self.model = self.model.take_over(args)?;
        self.steps = args.usize_or("--steps", self.steps);
        self.eval = args.usize_or("--eval", self.eval);
        Ok(self)
    }
}

#[cfg(feature = "decision")]
impl Options for SupervisedOptions {
    fn take(args: &mut Args) -> Result<SupervisedOptions, String> {
        SupervisedOptions::new(ModelOptions::new("", "out/head.safetensors")).take_over(args)
    }

    fn help() -> &'static str {
        "  --steps N           optimizer steps
  --eval N            examples to score afterwards"
    }
}

/// Every flag needed to configure a reinforcement-learning control run.
///
/// These are the knobs of [`crate::ControlSpec`] plus the two paths a run
/// needs - where the encoder is and where the head goes. They were about to be
/// spelled out a fourth time: `samples/decision/{arena,doom}` and every test
/// harness that trains a policy wants exactly this set.
#[cfg(feature = "decision")]
#[derive(Clone, Debug)]
pub struct ControlOptions {
    /// Directory holding the pretrained sentence encoder.
    pub encoder: String,
    /// Trained head to start from (training) or to run (evaluation).
    pub head: Option<String>,
    /// Where a trained head is written.
    pub save: String,
    pub iterations: usize,
    pub episodes: usize,
    pub epochs: usize,
    pub max_steps: usize,
    pub warmup_episodes: usize,
    pub warmup_epochs: usize,
    /// Fraction of the teacher's episodes to clone, best first.
    pub warmup_keep: f32,
    /// Rounds of DAgger between the warm start and the policy gradient.
    pub dagger: usize,
    /// Rounds of outcome-fitted improvement after the imitation phases.
    pub improve: usize,
    /// Decisions probed per improvement round.
    pub states: usize,
    /// Alternatives tried at each, beside the teacher's own.
    pub alternatives: usize,
    /// Decisions a probe departs from the teacher for.
    pub beta: f32,
    pub credit: usize,
    pub repeats: usize,
    pub wide: bool,
    /// Episodes on a fixed block of worlds, scored after every iteration.
    pub gauge_episodes: usize,
    /// Average the head over the last N iterates as a second candidate.
    pub average: usize,
    /// The head's learning rate, which trades against the trust region.
    pub head_lr: Option<f32>,
    /// The bias/variance dial on the advantage estimator.
    pub gae_lambda: Option<f32>,
    pub entropy: Option<f32>,
    /// Weight on staying near the policy the warm start produced. See
    /// `decide::policy::PolicyConfig::anchor`.
    pub anchor: Option<f32>,
    /// How far the policy may drift from the one that collected a batch
    /// before the rest of the passes over it are abandoned.
    pub target_kl: Option<f32>,
    pub seed: u64,
    /// Fine-tune the encoder as well as the head.
    pub train_encoder: bool,
}

#[cfg(feature = "decision")]
impl ControlOptions {
    /// The defaults, before any flag is applied. Taken from
    /// [`crate::ControlSpec`] so the two cannot disagree about what "default"
    /// means - a second set of numbers here would be a second policy.
    pub fn new(encoder: &str, save: &str) -> ControlOptions {
        let d = crate::ControlSpec::default();
        ControlOptions {
            encoder: encoder.to_string(),
            head: None,
            save: save.to_string(),
            iterations: d.iterations,
            episodes: d.episodes,
            epochs: d.epochs,
            max_steps: d.max_steps,
            warmup_episodes: d.warmup_episodes,
            warmup_epochs: d.warmup_epochs,
            warmup_keep: d.warmup_keep,
            dagger: d.dagger,
            improve: d.improve,
            states: d.states,
            alternatives: d.alternatives,
            beta: d.beta,
            credit: d.credit,
            repeats: d.repeats,
            wide: d.wide,
            gauge_episodes: d.gauge_episodes,
            average: d.average,
            head_lr: None,
            gae_lambda: None,
            entropy: None,
            anchor: None,
            target_kl: None,
            seed: d.seed,
            train_encoder: !d.freeze_encoder,
        }
    }

    /// The spec these flags describe.
    pub fn spec(&self) -> crate::ControlSpec {
        let mut s = crate::ControlSpec::default()
            .iterations(self.iterations)
            .episodes(self.episodes)
            .epochs(self.epochs)
            .max_steps(self.max_steps)
            .warmup_episodes(self.warmup_episodes)
            .warmup_epochs(self.warmup_epochs)
            .warmup_keep(self.warmup_keep)
            .dagger(self.dagger)
            .improve(self.improve)
            .probing(self.states, self.alternatives, self.beta, self.credit, self.repeats, self.wide)
            .gauge_episodes(self.gauge_episodes)
            .average(self.average)
            .head_lr(self.head_lr.unwrap_or(0.0))
            .gae_lambda(self.gae_lambda.unwrap_or(0.0))
            .train_encoder(self.train_encoder)
            .seed(self.seed);
        if let Some(e) = self.entropy {
            s.policy.entropy = e;
        }
        if let Some(a) = self.anchor {
            s.policy.anchor = a;
        }
        if let Some(k) = self.target_kl {
            s.policy.target_kl = k;
        }
        s
    }

    /// Take the flags, keeping whatever this value already holds as the
    /// default. Lets an application set its own defaults - a game whose
    /// episodes are 200 decisions long should not inherit a toy's 40 - and
    /// still accept the same flags.
    pub fn take_over(mut self, args: &mut Args) -> Result<ControlOptions, String> {
        self.encoder = args.str_or("--encoder", &self.encoder.clone());
        self.head = args.take_str("--head").or(self.head);
        self.save = args.str_or("--save", &self.save.clone());
        self.iterations = args.usize_or("--iterations", self.iterations);
        self.episodes = args.usize_or("--episodes", self.episodes);
        self.epochs = args.usize_or("--epochs", self.epochs);
        self.max_steps = args.usize_or("--max-steps", self.max_steps);
        self.warmup_episodes = args.usize_or("--warmup", self.warmup_episodes);
        self.warmup_epochs = args.usize_or("--warmup-epochs", self.warmup_epochs);
        self.dagger = args.usize_or("--dagger", self.dagger);
        self.improve = args.usize_or("--improve", self.improve);
        self.states = args.usize_or("--states", self.states);
        self.alternatives = args.usize_or("--alternatives", self.alternatives);
        self.beta = args.f32_or("--beta", self.beta);
        self.credit = args.usize_or("--credit", self.credit);
        self.repeats = args.usize_or("--repeats", self.repeats);
        // Never `self.wide || take_flag(..)`: `||` short-circuits, so a
        // default of true would leave the flag unconsumed and the run would
        // die reporting it as unrecognised.
        if args.take_flag("--contested") {
            self.wide = false;
        }
        self.gauge_episodes = args.usize_or("--gauge", self.gauge_episodes);
        self.average = args.usize_or("--average", self.average);
        if let Some(l) = args.take_str("--gae-lambda") {
            self.gae_lambda =
                Some(l.parse().map_err(|_| format!("--gae-lambda: {l:?} is not a number"))?);
        }
        if let Some(l) = args.take_str("--head-lr") {
            self.head_lr =
                Some(l.parse().map_err(|_| format!("--head-lr: {l:?} is not a number"))?);
        }
        if let Some(k) = args.take_str("--warmup-keep") {
            self.warmup_keep =
                k.parse().map_err(|_| format!("--warmup-keep: {k:?} is not a number"))?;
        }
        if let Some(e) = args.take_str("--entropy") {
            self.entropy =
                Some(e.parse().map_err(|_| format!("--entropy: {e:?} is not a number"))?);
        }
        if let Some(a) = args.take_str("--anchor") {
            self.anchor =
                Some(a.parse().map_err(|_| format!("--anchor: {a:?} is not a number"))?);
        }
        if let Some(k) = args.take_str("--target-kl") {
            self.target_kl =
                Some(k.parse().map_err(|_| format!("--target-kl: {k:?} is not a number"))?);
        }
        self.seed = args.u64_or("--seed", self.seed);
        self.train_encoder |= args.take_flag("--train-encoder");
        Ok(self)
    }
}

#[cfg(feature = "decision")]
impl Options for ControlOptions {
    fn take(args: &mut Args) -> Result<ControlOptions, String> {
        ControlOptions::new("", "out/policy.safetensors").take_over(args)
    }

    fn help() -> &'static str {
        "  --encoder DIR       pretrained sentence encoder
  --head FILE         start from (or run) these policy weights
  --save FILE         where a trained policy is written
  --iterations N      rollout-then-update cycles
  --episodes N        episodes collected per iteration
  --epochs N          PPO passes over each collected batch
  --max-steps N       decisions per episode
  --warmup N          scripted episodes cloned before the policy gradient
  --warmup-epochs N   passes over those demonstrations
  --warmup-keep F     fraction of scripted episodes to clone, best first  [1.0]
  --improve N         rounds of probing what a DIFFERENT action was worth and
                      moving toward whichever actually scored better. The only
                      phase whose ceiling is not the teacher: cloning and
                      --dagger fit the teacher's CHOICE, this fits the measured
                      OUTCOME. Shaped by the three below
  --states N          decisions probed per round, by --improve or `whatif` [200]
  --alternatives N    other actions tried at each of them                    [2]
  --beta F            how often a probe's roll-out is the TEACHER rather than
                      the policy, drawn once per probed decision           [0.5]
                      Teacher-only roll-outs leave the learner blind to its own
                      compounding errors; policy-only roll-outs turn this into
                      full RL, which is the problem the phase exists to avoid.
                      The mixture is what LOLS shows works
  --credit N          decisions past the branch point a candidate is scored
                      over. 0 scores to the end of the episode               [0]
                      A shorter window stops a candidate's score being decided
                      by what happened three hundred decisions later
  --repeats N         roll-outs averaged per candidate                       [1]
                      One roll-out of a long episode is a single draw of a
                      system where any decision changes everything after it
  --contested         draw the alternatives from what the policy ranks highest
                      instead of UNIFORMLY, which is the default. The control,
                      not a way to probe: taking the policy's own favourites
                      makes which actions a probe even considers depend on the
                      policy being trained, and the sample-complexity result
                      behind this phase is stated for uniform exploration
  --dagger N          rounds of running the STUDENT and asking the teacher what
                      it would have done at every state the student reached,
                      aggregating those labels and refitting. Cloning only ever
                      sees the teacher's own trajectory, so the student's first
                      mistake takes it somewhere the dataset is silent about and
                      the errors compound; this is what puts labels there
  --entropy F         exploration bonus
  --anchor F          hold the policy near the one the warm start produced,
                      by the divergence between them. A policy gradient
                      started from a cloned policy has no reason to stay near
                      it: the two phases optimise different objectives, and
                      where return is sparse and noisy the pull is mostly
                      noise. 0 (the default) is no anchor
  --gauge N           score the policy on the SAME N worlds after every
                      iteration, and keep the iteration that did best on
                      them. Without it the comparison is between scores taken
                      on worlds that moved: measured here, a player that
                      cannot change at all scores 0.59 to 0.73 across five
                      blocks of sixteen generated levels, which is most of the
                      movement a training run appears to show
  --average N         also try the MEAN of the last N iterates, and keep it
                      if it gauges better than the best single one. Keeping
                      the best selects partly for luck, because the score it
                      is chosen on carries noise; a mean does not. Fictitious
                      play converges in the time average of play rather than
                      the last thing played, and Polyak-Ruppert averaging
                      reaches the optimal asymptotic variance under far looser
                      step-size tuning than any single iterate
  --gae-lambda F      the bias/variance dial on the advantage estimator. At
                      1.0 the advantage is the full return minus a baseline -
                      noisy, unbiased, and it carries a reward paid at the
                      exit back to the first decision. At the usual 0.95 it
                      does not: `gamma * lambda` is 0.9405, so a reward a
                      hundred decisions ahead arrives with weight 0.002
  --head-lr F         the head's step size, which trades against
                      `--target-kl` rather than standing alone. Too large and
                      the divergence budget is spent in a handful of
                      minibatches and the rest of the rollout is never used;
                      measured here at 3e-4, an iteration took 5 of 64 steps,
                      discarding 92% of what it had just collected
  --target-kl F       abandon the remaining passes over a batch once the
                      policy has moved this far from the one that collected
                      it. Clipping alone is not a trust region: it silences
                      the samples that have travelled while the rest keep
                      pushing. 0 disables it
  --train-encoder     fine-tune the encoder, not just the head
  --seed N"
    }
}

/// Every flag that decides whether and how a run is watched.
///
/// Shared because it is the same question in every sample: open a window, or
/// write the frames, or neither - and a headless machine must do something
/// sensible either way.
#[cfg(feature = "viewport")]
#[derive(Clone, Debug, Default)]
pub struct ViewOptions {
    /// Open a window. Without it a run is headless even where a display
    /// exists, which is what a training job on a shared machine wants.
    pub window: bool,
    /// Write every frame here as a PNG.
    pub frames: Option<String>,
    /// Cap the window's pace so a human can follow it.
    pub fps: u32,
}

#[cfg(feature = "viewport")]
impl Options for ViewOptions {
    fn take(args: &mut Args) -> Result<ViewOptions, String> {
        let fps = args.u32_or("--fps", 12);
        if fps == 0 {
            return Err("--fps must be at least 1".into());
        }
        Ok(ViewOptions { window: args.take_flag("--window"), frames: args.take_str("--frames"), fps })
    }

    fn help() -> &'static str {
        "  --window            open a window (otherwise headless)
  --frames DIR        write every frame as a PNG
  --fps N             cap the window's pace                       [12]"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "decision")]
    #[test]
    fn control_defaults_come_from_the_spec_itself() {
        // If these were typed out again here, the two would drift and a sample
        // would be training with a different default from the one the pipeline
        // documents.
        let d = crate::ControlSpec::default();
        let o = ControlOptions::new("enc", "out.safetensors");
        assert_eq!(o.iterations, d.iterations);
        assert_eq!(o.spec().max_steps, d.max_steps);
        assert_eq!(o.spec().warmup_episodes, d.warmup_episodes);
    }

    #[cfg(feature = "decision")]
    #[test]
    fn an_application_keeps_its_own_defaults_and_still_takes_the_flags() {
        let argv: Vec<String> =
            ["--episodes", "9", "--train-encoder"].iter().map(|s| s.to_string()).collect();
        let mut args = Args::new(&argv);
        let mut base = ControlOptions::new("enc", "out.safetensors");
        base.max_steps = 220; // a game, not a toy
        let o = base.take_over(&mut args).expect("parses");
        assert_eq!(o.episodes, 9, "the flag wins");
        assert_eq!(o.max_steps, 220, "the application's default survives");
        assert!(o.spec().max_steps == 220 && !o.spec().freeze_encoder);
        args.finish();
    }

    #[cfg(all(feature = "decision", feature = "viewport"))]
    #[test]
    fn groups_compose_without_stealing_each_others_flags() {
        // The property that makes groups worth having: each takes only its own
        // and the application's own flags survive both.
        let argv: Vec<String> =
            ["--window", "--episodes", "3", "--skill", "4", "--fps", "30"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let mut args = Args::new(&argv);
        let hw = Hardware::take(&mut args).expect("hardware");
        let ctrl = ControlOptions::new("e", "s").take_over(&mut args).expect("control");
        let view = ViewOptions::take(&mut args).expect("view");
        assert!(hw.device.is_all());
        assert_eq!(ctrl.episodes, 3);
        assert!(view.window && view.fps == 30);
        assert_eq!(args.u32_or("--skill", 0), 4, "the application's own flag is still there");
        args.finish();
    }
}
