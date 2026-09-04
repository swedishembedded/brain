// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a generation's sampling knobs actually come from.
//!
//! The reference resolves every sampling knob in three layers, and brain now
//! resolves them the same way:
//!
//! ```text
//! explicit caller-supplied option  >  the checkpoint's generation_config.json  >  the reference's hard fallback
//! ```
//!
//! Until this module existed there was only the third layer, hardcoded in Rust
//! as `GenOpts::default()`. That is how `repetition_penalty` drifted: the
//! checkpoint ships `1.05` in its own `generation_config.json` and the
//! reference hard-defaults to `1.05`, while brain's one source of truth said
//! `1.0` (disabled) - the exact guard that keeps codebook-0 out of a silent
//! repetition loop, off by default, discoverable only by a multi-hour
//! root-cause session. `temperature`, `top_p`, `top_k`, the four subtalker
//! knobs and `max_new_tokens` all live in that same file and were all equally
//! un-read; this module reads the file rather than transcribing one more field
//! from it by hand.
//!
//! The three types map one-to-one onto the three layers:
//!
//! - [`SamplingRequest`] - what the CALLER asked for. Every field is an
//!   `Option`, because "the caller said nothing" and "the caller explicitly
//!   chose 1.0" are different requests and a plain `f32` cannot tell them
//!   apart. This is what a CLI flag, a `capability::Invocation` param or a
//!   direct construction fills in.
//! - [`GenerationConfig`] - what the CHECKPOINT ships, parsed defensively from
//!   `generation_config.json` in the same directory (and in the same
//!   `serde_json::Value` + per-field-default style) as `config.json`. A missing
//!   file, unparseable JSON or a malformed field is not an error: it simply
//!   supplies nothing and the next layer answers.
//! - [`GenerationPlan`] - what actually RUNS. Fully populated, no `Option`s,
//!   logged once per generation call so a bug report can show exactly what
//!   executed instead of requiring someone to re-derive it.
//!
//! Swedish Embedded AB implements solutions for reproducible, checkpoint-faithful
//! inference configuration for its clients. If your team needs expertise in
//! keeping a from-scratch engine in parity with a reference implementation then
//! you can procure our services by sending an email to info@swedishembedded.com.

use crate::sampling::SamplerCfg;

/// The reference's hard-coded fallbacks (`Qwen3TTSModel._merge_generate_kwargs`),
/// which are also what the 12Hz Base checkpoint's own `generation_config.json`
/// ships. Used when neither the caller nor the checkpoint supplies a value.
pub const REFERENCE: GenerationConfig = GenerationConfig {
    do_sample: Some(true),
    temperature: Some(0.9),
    top_k: Some(50),
    top_p: Some(1.0),
    repetition_penalty: Some(1.05),
    subtalker_do_sample: Some(true),
    subtalker_temperature: Some(0.9),
    subtalker_top_k: Some(50),
    subtalker_top_p: Some(1.0),
    max_new_tokens: Some(8192),
};

/// One checkpoint's `generation_config.json`, field by field. Every field is
/// `None` when the file did not supply it (absent file, absent key, or a value
/// of the wrong JSON type).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerationConfig {
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    /// The reference spells this `subtalker_dosample` (no underscore); the
    /// parser accepts that spelling, this field carries the readable one.
    pub subtalker_do_sample: Option<bool>,
    pub subtalker_temperature: Option<f32>,
    pub subtalker_top_k: Option<usize>,
    pub subtalker_top_p: Option<f32>,
    /// The reference's Talker-token budget. Read and reported, deliberately NOT
    /// wired to `GenOpts::max_frames` - see [`GenerationPlan::max_new_tokens`].
    pub max_new_tokens: Option<usize>,
}

impl GenerationConfig {
    /// Parse `<dir>/generation_config.json`.
    ///
    /// Defensive by construction, mirroring how `TtsSpecials::from_config_dir`
    /// reads the sibling `config.json`: this is a set of DEFAULTS, and a
    /// checkpoint that ships none of them (or a corrupt file) must fall through
    /// to the reference's values rather than refuse to synthesize. Every failure
    /// mode collapses to "supplies nothing".
    pub fn from_config_dir(dir: &str) -> GenerationConfig {
        let Ok(s) = std::fs::read_to_string(std::path::Path::new(dir).join("generation_config.json")) else {
            return GenerationConfig::default();
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) else {
            return GenerationConfig::default();
        };
        let f = |k: &str| v[k].as_f64().map(|x| x as f32);
        let u = |k: &str| v[k].as_u64().map(|x| x as usize);
        let b = |k: &str| v[k].as_bool();
        GenerationConfig {
            do_sample: b("do_sample"),
            temperature: f("temperature"),
            top_k: u("top_k"),
            top_p: f("top_p"),
            repetition_penalty: f("repetition_penalty"),
            subtalker_do_sample: b("subtalker_dosample").or_else(|| b("subtalker_do_sample")),
            subtalker_temperature: f("subtalker_temperature"),
            subtalker_top_k: u("subtalker_top_k"),
            subtalker_top_p: f("subtalker_top_p"),
            max_new_tokens: u("max_new_tokens"),
        }
    }

    /// Whether this config supplied anything at all - the difference between
    /// "the checkpoint told us" and "we fell back to the reference", which the
    /// resolved plan's trace line reports.
    pub fn is_empty(&self) -> bool {
        *self == GenerationConfig::default()
    }
}

/// The MTP/subtalker half of a [`SamplingRequest`]: the caller's explicit
/// choices for the residual codebooks (1..15).
///
/// This is the ONLY caller-side override for residual sampling. There is no
/// separate `ResidualOpts`/`GenOpts::residual` any more - it was a fourth
/// hand-threaded knob a caller had to remember to set, defaulting to greedy,
/// which is exactly the "one more source of truth in Rust" shape that shipped
/// `repetition_penalty = 1.0`. Unset fields resolve from the checkpoint's
/// `subtalker_*` keys and then from the reference, like every other knob here.
///
/// A `temperature` of `Some(0.0)` pins the residual codebooks to greedy, the
/// same way an explicit `--temp 0` does for codebook 0.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SubtalkerRequest {
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

/// The caller's EXPLICIT sampling choices. `None` means "not specified - resolve
/// it"; `Some(x)` means "the caller chose x", including `Some(1.0)` for a
/// deliberately disabled repetition penalty.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplingRequest {
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub subtalker: SubtalkerRequest,
    pub max_new_tokens: Option<usize>,
}

impl SamplingRequest {
    /// Pin EVERY codebook - codebook 0 and the residual codebooks alike - to a
    /// deterministic greedy draw.
    ///
    /// For the parity and determinism lanes, which need bit-identical codes
    /// across two engines and therefore must NOT inherit whatever a checkpoint
    /// (or the reference) says about sampling. Pinning explicitly is what makes
    /// such a test independent of the config chain rather than a silent hostage
    /// to it - and that now has to include the subtalker half, because the
    /// residual codebooks sample by default. Leaving them unset here would let a
    /// "greedy" parity run draw 15 random codebooks per frame off two different
    /// engines' logits and diverge on the first near-tie.
    pub fn greedy() -> SamplingRequest {
        SamplingRequest {
            do_sample: Some(false),
            temperature: Some(0.0),
            top_k: Some(0),
            top_p: Some(0.0),
            repetition_penalty: Some(1.0),
            subtalker: SubtalkerRequest { do_sample: Some(false), temperature: Some(0.0), top_k: Some(0), top_p: Some(0.0) },
            max_new_tokens: None,
        }
    }
}

/// Which layer answered for the values the caller did not pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanSource {
    /// The checkpoint's `generation_config.json` supplied at least one value.
    Checkpoint,
    /// No checkpoint config was available (or it supplied nothing); the
    /// reference's hard defaults answered.
    Reference,
}

impl PlanSource {
    pub fn label(&self) -> &'static str {
        match self {
            PlanSource::Checkpoint => "generation_config.json",
            PlanSource::Reference => "reference defaults",
        }
    }
}

/// The fully resolved sampling configuration one generation call runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationPlan {
    /// Codebook-0's filter chain - the Talker's own next-token draw.
    pub cb0: SamplerCfg,
    /// The MTP/subtalker residual codebooks' chain, consumed by every residual
    /// fill in the crate (`MtpModel`, `CpuMtp`, the NPU `MtpEngine`s) through
    /// [`crate::sampling::sample_residual`].
    ///
    /// Its `repetition_penalty` is always `1.0`: the reference configures none
    /// for the subtalker, and `sample_residual` would ignore it anyway.
    pub subtalker: SamplerCfg,
    /// The reference's `max_new_tokens`, resolved and reported but **not
    /// applied**. `GenOpts::max_frames` remains the cap that stops a run.
    ///
    /// The units are the same thing: the reference hands `max_new_tokens` to
    /// the Talker's own `generate`, and one Talker token is one codebook-0
    /// frame, so 8192 is 8192 frames - about eleven minutes at this
    /// checkpoint's 12.5 Hz frame rate, comfortably inside the Talker's
    /// 32768-position context. Adopting it as `max_frames` would therefore be
    /// arithmetically valid, and it is still the wrong move, because in brain
    /// `max_frames` is not only a stop condition - it SIZES allocations:
    ///
    /// - `pipeline::max_ctx` allocates the Talker KV cache as
    ///   `max_frames + ref_frames + 32` positions;
    /// - the NPU path rounds `n_prefix + max_frames + 2` up into a compiled
    ///   graph bucket, so the cap decides what gets exported and compiled;
    /// - `pipeline::design` sizes its own context as
    ///   `instruct + input + max_frames + 64`.
    ///
    /// Defaulting to 8192 would grow every single run's cache allocation and
    /// NPU compile by ~32x to accommodate a ceiling a healthy clip never
    /// approaches (the measured repro reaches EOS at 38 frames). The reference
    /// pays nothing for a loose ceiling because its cache grows on demand;
    /// brain's does not. So the decision is: `max_frames` stays authoritative
    /// and keeps its allocation-shaped default, and this field is carried,
    /// traced next to `max_frames` in the resolved-plan line, and available to
    /// a caller that deliberately wants the reference's ceiling. Recorded, not
    /// overlooked.
    pub max_new_tokens: usize,
    /// Which layer answered for the unpinned values.
    pub source: PlanSource,
}

impl GenerationPlan {
    /// Resolve a plan: caller override, then the checkpoint's
    /// `generation_config.json` (when `ckpt_dir` is given), then the reference's
    /// hard fallback.
    pub fn resolve(req: &SamplingRequest, ckpt_dir: Option<&str>) -> GenerationPlan {
        GenerationPlan::resolve_with(req, ckpt_dir.map(GenerationConfig::from_config_dir).unwrap_or_default())
    }

    /// [`Self::resolve`] against an ALREADY-PARSED checkpoint config.
    ///
    /// For the resident servers, which load a checkpoint once and then answer
    /// many requests: they parse `generation_config.json` at load and hand it
    /// here per request, instead of re-reading the same file off disk on every
    /// synthesis. Same precedence, same result - only the file read moves.
    pub fn resolve_with(req: &SamplingRequest, file: GenerationConfig) -> GenerationPlan {
        let source = if file.is_empty() { PlanSource::Reference } else { PlanSource::Checkpoint };
        // Each knob: caller, then file, then reference. `REFERENCE` is `Some`
        // in every field, so the final `expect` is total.
        macro_rules! pick {
            ($f:ident) => {
                req.$f.or(file.$f).or(REFERENCE.$f).expect("REFERENCE supplies every field")
            };
        }
        let temperature = pick!(temperature);
        let cb0 = SamplerCfg {
            // An explicit non-positive temperature is this crate's long-standing
            // greedy switch (`--temp 0`), and it wins over any `do_sample` the
            // checkpoint asserts: nothing can sample at temperature zero.
            do_sample: pick!(do_sample) && temperature > 0.0,
            temperature,
            top_k: pick!(top_k),
            top_p: pick!(top_p),
            repetition_penalty: pick!(repetition_penalty),
        };
        let sub_temperature = req
            .subtalker
            .temperature
            .or(file.subtalker_temperature)
            .or(REFERENCE.subtalker_temperature)
            .expect("REFERENCE supplies every field");
        let subtalker = SamplerCfg {
            do_sample: req
                .subtalker
                .do_sample
                .or(file.subtalker_do_sample)
                .or(REFERENCE.subtalker_do_sample)
                .expect("REFERENCE supplies every field")
                && sub_temperature > 0.0,
            temperature: sub_temperature,
            top_k: req.subtalker.top_k.or(file.subtalker_top_k).or(REFERENCE.subtalker_top_k).expect("REFERENCE supplies every field"),
            top_p: req.subtalker.top_p.or(file.subtalker_top_p).or(REFERENCE.subtalker_top_p).expect("REFERENCE supplies every field"),
            // The reference configures no repetition penalty for the residual
            // codebooks; they are one step deep, not an autoregressive loop.
            repetition_penalty: 1.0,
        };
        GenerationPlan { cb0, subtalker, max_new_tokens: pick!(max_new_tokens), source }
    }

    /// The plan a caller with no checkpoint and no explicit choices gets: the
    /// reference's own recipe. This is what an unresolved `GenOpts` runs, so a
    /// directly-constructed one still decodes correctly.
    pub fn reference() -> GenerationPlan {
        GenerationPlan::resolve(&SamplingRequest::default(), None)
    }

    /// One line naming everything that will run, for a bug report to quote.
    ///
    /// `max_frames` is the caller's own cap and is printed next to
    /// [`Self::max_new_tokens`] on purpose: the reference's budget is reported
    /// but NOT applied, and a line that showed only `max_new_tokens=8192` while
    /// the run actually stopped at 256 frames would be worse than no line.
    pub fn trace_line(&self, max_frames: usize) -> String {
        format!(
            "qwen3tts: resolved plan: sample={} temp={} top_k={} top_p={} rep_penalty={} (source: {}) \
             | length: max_frames={max_frames} applied, max_new_tokens={} reported-only \
             | subtalker (residual codebooks 1..15): sample={} temp={} top_k={} top_p={}",
            self.cb0.do_sample,
            self.cb0.temperature,
            self.cb0.top_k,
            self.cb0.top_p,
            self.cb0.repetition_penalty,
            self.source.label(),
            self.max_new_tokens,
            self.subtalker.do_sample,
            self.subtalker.temperature,
            self.subtalker.top_k,
            self.subtalker.top_p,
        )
    }

    /// Print [`Self::trace_line`] once, when tracing is enabled.
    ///
    /// Gated on `TTS_PLAN`, and also emitted under the crate's existing
    /// `TTS_PROFILE` switch - anyone already asking "what did this run do"
    /// wants the plan in the same output as the stage timings.
    pub fn trace(&self, max_frames: usize) {
        if std::env::var("TTS_PLAN").is_ok() || std::env::var("TTS_PROFILE").is_ok() {
            eprintln!("{}", self.trace_line(max_frames));
        }
    }
}

impl Default for GenerationPlan {
    fn default() -> GenerationPlan {
        GenerationPlan::reference()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dir(std::path::PathBuf);
    impl Dir {
        fn new(tag: &str, body: Option<&str>) -> Dir {
            let d = std::env::temp_dir().join(format!("qwen3tts-genconfig-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            match body {
                Some(b) => std::fs::write(d.join("generation_config.json"), b).unwrap(),
                None => {
                    std::fs::remove_file(d.join("generation_config.json")).ok();
                }
            }
            Dir(d)
        }
        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    /// The exact file the 12Hz 0.6B Base checkpoint ships.
    const REAL: &str = r#"{
      "do_sample": true,
      "repetition_penalty": 1.05,
      "temperature": 0.9,
      "top_p": 1.0,
      "top_k": 50,
      "subtalker_dosample": true,
      "subtalker_temperature": 0.9,
      "subtalker_top_p": 1.0,
      "subtalker_top_k": 50,
      "max_new_tokens": 8192
    }"#;

    #[test]
    fn the_checkpoints_own_config_is_read_for_every_knob() {
        let d = Dir::new("real", Some(REAL));
        let c = GenerationConfig::from_config_dir(d.path());
        assert_eq!(c.do_sample, Some(true));
        assert_eq!(c.repetition_penalty, Some(1.05));
        assert_eq!(c.temperature, Some(0.9));
        assert_eq!(c.top_p, Some(1.0));
        assert_eq!(c.top_k, Some(50));
        assert_eq!(c.subtalker_do_sample, Some(true));
        assert_eq!(c.subtalker_temperature, Some(0.9));
        assert_eq!(c.subtalker_top_p, Some(1.0));
        assert_eq!(c.subtalker_top_k, Some(50));
        assert_eq!(c.max_new_tokens, Some(8192));
        assert!(!c.is_empty());
    }

    /// The precedence, in one assertion per layer.
    #[test]
    fn an_explicit_caller_value_beats_the_checkpoint_which_beats_the_reference() {
        // A checkpoint that disagrees with the reference on every knob it sets,
        // so "which layer answered" is unambiguous.
        let d = Dir::new("precedence", Some(r#"{"temperature": 0.5, "top_k": 7, "repetition_penalty": 1.30}"#));

        // (b) checkpoint answers where the caller is silent.
        let p = GenerationPlan::resolve(&SamplingRequest::default(), Some(d.path()));
        assert_eq!(p.cb0.temperature, 0.5);
        assert_eq!(p.cb0.top_k, 7);
        assert_eq!(p.cb0.repetition_penalty, 1.30);
        // (c) the reference answers for what the checkpoint did not set.
        assert_eq!(p.cb0.top_p, 1.0);
        assert_eq!(p.max_new_tokens, 8192);
        assert_eq!(p.source, PlanSource::Checkpoint);

        // (a) an explicit caller value beats both.
        let req = SamplingRequest { temperature: Some(1.4), repetition_penalty: Some(1.0), ..SamplingRequest::default() };
        let p = GenerationPlan::resolve(&req, Some(d.path()));
        assert_eq!(p.cb0.temperature, 1.4);
        assert_eq!(p.cb0.top_k, 7, "an unset knob must still come from the checkpoint");
        assert_eq!(
            p.cb0.repetition_penalty, 1.0,
            "an explicitly-disabled penalty must survive - this is the case a plain f32 field cannot express"
        );
    }

    /// A missing or corrupt file must degrade to the reference, never fail.
    #[test]
    fn a_missing_or_malformed_config_falls_through_to_the_reference() {
        for (tag, body) in [("absent", None), ("corrupt", Some("{ not json")), ("wrongtypes", Some(r#"{"temperature": "hot", "top_k": []}"#))] {
            let d = Dir::new(tag, body);
            let p = GenerationPlan::resolve(&SamplingRequest::default(), Some(d.path()));
            assert_eq!(p.cb0.temperature, 0.9, "{tag}");
            assert_eq!(p.cb0.top_k, 50, "{tag}");
            assert_eq!(p.cb0.repetition_penalty, 1.05, "{tag}");
            assert_eq!(p.source, PlanSource::Reference, "{tag}");
        }
        // Same for "no checkpoint dir at all".
        assert_eq!(GenerationPlan::resolve(&SamplingRequest::default(), None), GenerationPlan::reference());
    }

    /// The value the whole silent-collapse investigation turned on: with no
    /// caller override and no checkpoint, the penalty must still be the
    /// reference's 1.05, never 1.0.
    #[test]
    fn the_reference_fallback_never_disables_the_repetition_penalty() {
        assert_eq!(GenerationPlan::reference().cb0.repetition_penalty, 1.05);
        assert!(GenerationPlan::reference().cb0.do_sample);
    }

    /// `--temp 0` means greedy, whatever any config layer says about
    /// `do_sample`.
    #[test]
    fn an_explicit_zero_temperature_forces_greedy_over_do_sample() {
        let d = Dir::new("dosample", Some(r#"{"do_sample": true, "temperature": 0.9}"#));
        let req = SamplingRequest { temperature: Some(0.0), ..SamplingRequest::default() };
        let p = GenerationPlan::resolve(&req, Some(d.path()));
        assert!(!p.cb0.do_sample);
        assert!(p.cb0.is_greedy());
        // And the checkpoint's own do_sample=false is honoured on its own.
        let d2 = Dir::new("nosample", Some(r#"{"do_sample": false}"#));
        assert!(GenerationPlan::resolve(&SamplingRequest::default(), Some(d2.path())).cb0.is_greedy());
    }

    /// The reference samples the residual codebooks too, so an unconfigured
    /// brain run must as well. This is the subtalker twin of
    /// `the_reference_fallback_never_disables_the_repetition_penalty`: the
    /// residual codebooks carry most of the acoustic detail, and decoding them
    /// greedily was a parity gap, not a design choice.
    #[test]
    fn the_reference_fallback_samples_the_residual_codebooks() {
        let p = GenerationPlan::reference();
        assert!(p.subtalker.do_sample, "the reference's subtalker_dosample is true");
        assert!(!p.subtalker.is_greedy());
        assert_eq!(p.subtalker.temperature, 0.9);
        assert_eq!(p.subtalker.top_k, 50);
        assert_eq!(p.subtalker.repetition_penalty, 1.0, "the reference configures no subtalker repetition penalty");
        // And the real checkpoint agrees, so this is not a fallback-only claim.
        let d = Dir::new("subdefault", Some(REAL));
        assert!(GenerationPlan::resolve(&SamplingRequest::default(), Some(d.path())).subtalker.do_sample);
    }

    /// A greedy request must pin BOTH halves. A parity lane that pinned only
    /// codebook 0 would now silently sample 15 residual codebooks per frame off
    /// two different engines' logits and diverge on the first near-tie.
    #[test]
    fn the_greedy_request_pins_the_residual_codebooks_too() {
        let d = Dir::new("greedysub", Some(REAL));
        let p = GenerationPlan::resolve(&SamplingRequest::greedy(), Some(d.path()));
        assert!(p.cb0.is_greedy());
        assert!(p.subtalker.is_greedy(), "a greedy request left the residual codebooks sampling");
    }

    /// An explicit zero residual temperature is the residual half's `--temp 0`.
    #[test]
    fn an_explicit_zero_residual_temperature_forces_greedy_residuals() {
        let d = Dir::new("subzero", Some(REAL));
        let req = SamplingRequest {
            subtalker: SubtalkerRequest { temperature: Some(0.0), ..SubtalkerRequest::default() },
            ..SamplingRequest::default()
        };
        let p = GenerationPlan::resolve(&req, Some(d.path()));
        assert!(p.subtalker.is_greedy());
        assert!(!p.cb0.is_greedy(), "pinning the residual half must not touch codebook 0");
    }

    /// The subtalker half resolves through the same precedence as codebook 0.
    #[test]
    fn the_subtalker_knobs_resolve_through_the_same_precedence() {
        let d = Dir::new("sub", Some(r#"{"subtalker_dosample": true, "subtalker_temperature": 0.7, "subtalker_top_k": 12}"#));
        let p = GenerationPlan::resolve(&SamplingRequest::default(), Some(d.path()));
        assert!(p.subtalker.do_sample);
        assert_eq!(p.subtalker.temperature, 0.7);
        assert_eq!(p.subtalker.top_k, 12);
        assert_eq!(p.subtalker.top_p, 1.0, "unset subtalker knobs fall through to the reference");
        let req = SamplingRequest { subtalker: SubtalkerRequest { top_k: Some(3), ..SubtalkerRequest::default() }, ..SamplingRequest::default() };
        assert_eq!(GenerationPlan::resolve(&req, Some(d.path())).subtalker.top_k, 3);
    }

    #[test]
    fn the_trace_line_names_every_resolved_knob_and_its_source() {
        let line = GenerationPlan::reference().trace_line(256);
        for needle in
            ["sample=true", "temp=0.9", "top_k=50", "top_p=1", "rep_penalty=1.05", "max_frames=256", "max_new_tokens=8192", "reference defaults"]
        {
            assert!(line.contains(needle), "trace line is missing {needle:?}: {line}");
        }
        let d = Dir::new("trace", Some(REAL));
        assert!(GenerationPlan::resolve(&SamplingRequest::default(), Some(d.path())).trace_line(256).contains("generation_config.json"));
    }
}
