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
    pub entropy: Option<f32>,
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
            entropy: None,
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
            .train_encoder(self.train_encoder)
            .seed(self.seed);
        if let Some(e) = self.entropy {
            s.policy.entropy = e;
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
        if let Some(e) = args.take_str("--entropy") {
            self.entropy =
                Some(e.parse().map_err(|_| format!("--entropy: {e:?} is not a number"))?);
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
  --entropy F         exploration bonus
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
