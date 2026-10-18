// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision engine: one type over both decision architectures.
//!
//! * [`Backend::Decide`] - `crates/decide`'s MiniLM/BERT-family encoder plus
//!   cross-attention head. It encodes a request's state ONCE and scores every
//!   question's options against that encoding.
//! * [`Backend::Laya`] - `convaiinnovations/laya`'s ModernBERT-large trunk and
//!   its own decision head. Its head takes one question's option markers per
//!   call, so every question re-packs and re-encodes `(state, question)`.
//!
//! Loading is split into a read stage ([`DecideWeights::read`],
//! [`LayaWeights::read`]), which touches only the host, and a build stage,
//! which allocates on a device the caller already selected. That split is what
//! lets a caller tell "the files are wrong" from "the device failed" and keep
//! the device choice (scoped by a scheduler, resolved by an SDK) out of here.
//!
//! The Laya arm applies the serving calibration its own reference applies:
//! `temperature_by_options` first, the per-question-type `temperature` as the
//! fallback, so published probabilities match the reference number for number
//! and not only in argmax.
//!
//! Swedish Embedded AB implements calibrated decision services for its
//! clients, one engine behind a stable contract whatever model sits behind
//! it. If your team needs expertise in decision models in the loop, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use decide::decide::{Decide, Limits};
use decide::loss::softmax;
use decide::primitives::{confidence, Answer, Opt, Question};
use data::tokenizer::Tokenizer as _;
use gpu_core::Gpu;
use modernbert::{LayaDecision, State, StateFit};

use crate::laya_head::{BaseRef, HEAD_FILE};

/// Which decision architecture a checkpoint is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Decide,
    Laya,
}

impl Backend {
    /// The wire name reported in answers and manifests.
    pub fn slug(self) -> &'static str {
        match self {
            Backend::Decide => "decide",
            Backend::Laya => "laya",
        }
    }

    /// The architecture a directory's STRUCTURE names, if any.
    ///
    /// A Laya checkpoint has no root `config.json`: it carries a nested
    /// `encoder/config.json` (`model_type: "modernbert"`) and a root
    /// `rl_agent_config.json`. A `decide` checkpoint is a BERT-family
    /// directory with a root `config.json`. Both Laya conditions are checked
    /// so a stray `rl_agent_config.json` cannot misroute an unrelated
    /// directory.
    pub fn sniff(dir: &Path) -> Option<Backend> {
        if is_laya_dir(dir) {
            Some(Backend::Laya)
        } else if dir.join("config.json").is_file() {
            Some(Backend::Decide)
        } else {
            None
        }
    }
}

pub fn is_laya_dir(dir: &Path) -> bool {
    if !dir.join("rl_agent_config.json").is_file() {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(dir.join("encoder").join("config.json")) else {
        return false;
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    cfg.get("model_type").and_then(serde_json::Value::as_str) == Some("modernbert")
}

/// The encoder directory as the Hugging Face reference it came from.
///
/// A head is an adapter and an adapter that does not name its base is
/// unloadable. `brain pull` lays a model down under `<vendor>/<repo>`, so the
/// last two path components ARE the reference; anything shorter falls back to
/// the leaf.
pub fn base_reference(dir: &Path) -> String {
    let parts: Vec<&str> = dir
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .filter(|s| !s.is_empty() && *s != "/")
        .collect();
    match parts.len() {
        0 => String::new(),
        1 => parts[0].to_string(),
        n => format!("{}/{}", parts[n - 2], parts[n - 1]),
    }
}

// ===================== decide =====================

/// A `decide` checkpoint read from disk, not yet on a device.
pub struct DecideWeights {
    cfg: decide::config::EncoderConfig,
    tok: data::wordpiece::WordPiece,
    enc_init: decide::Tensors,
    head_init: decide::Tensors,
    head_trained: bool,
    base: String,
}

impl DecideWeights {
    /// Read `dir` (`config.json`, `model.safetensors`, `tokenizer.json`).
    ///
    /// ONE load path for both naming conventions, and the file says which it
    /// is: a published `sentence-transformers` checkpoint is named the
    /// Hugging Face way; one written by `Decide::save_model` is named the way
    /// this architecture names its own parameters and carries its head in the
    /// same file. An explicit `head` file wins over the head in the
    /// checkpoint; with neither, the head is freshly initialised from `seed`
    /// and answers near-uniformly.
    pub fn read(dir: &str, head: Option<&str>, seed: u64) -> Result<DecideWeights, String> {
        let path = Path::new(dir);
        let cfg_json = std::fs::read_to_string(path.join("config.json")).map_err(|e| format!("read {dir}/config.json: {e}"))?;
        let weights = path.join("model.safetensors");
        let tensors = checkpoint::safetensors::read(weights.to_str().ok_or("non-UTF-8 weights path")?)
            .map_err(|e| format!("read {}: {e}", weights.display()))?;

        let names: Vec<String> = tensors.iter().map(|t| t.name.clone()).collect();
        let probe = decide::import::config_from_hf(&cfg_json)
            .or_else(|_| decide::config::EncoderConfig::from_json_strict(&cfg_json))?;
        let naming = decide::import::sniff_naming(&names, &probe)?;
        let (cfg, enc_init, saved_head) = match naming {
            decide::import::Naming::Hf => {
                let cfg = decide::import::config_from_hf(&cfg_json)?;
                let enc = decide::import::brain_init_from_hf(tensors, &cfg)?;
                (cfg, enc, None)
            }
            decide::import::Naming::Brain => {
                let cfg = decide::config::EncoderConfig::from_json_strict(&cfg_json)?;
                let (enc, rest) = decide::import::split_brain_tensors(tensors, &cfg)?;
                (cfg, enc, Some(rest))
            }
        };

        let tok_path = path.join("tokenizer.json");
        let tok = data::wordpiece::WordPiece::from_file(tok_path.to_str().ok_or("non-UTF-8 tokenizer path")?)?;

        let head_trained = head.is_some() || saved_head.is_some();
        let head_init = match (head, saved_head) {
            (Some(p), _) => {
                let t = checkpoint::safetensors::read(p).map_err(|e| format!("read {p}: {e}"))?;
                t.into_iter().map(|x| (x.name, x.data)).collect()
            }
            // A saved decision model is checked against the head manifest
            // rather than trusted: a file missing a tensor is a named error,
            // not a silently re-initialised head that loads, answers, and is
            // a different model.
            (None, Some(mut saved)) => {
                let mut head_init = HashMap::new();
                for (name, shape) in decide::head::tensor_manifest(&cfg) {
                    let numel: usize = shape.iter().product();
                    match saved.remove(&name) {
                        Some(v) if v.len() == numel => {
                            head_init.insert(name, v);
                        }
                        Some(v) => return Err(format!("{dir}: head parameter {name} has {} elements, expected {numel}", v.len())),
                        None => {
                            return Err(format!(
                                "{dir}: a decision checkpoint in this crate's own naming must carry its head, and {name} is missing"
                            ))
                        }
                    }
                }
                // Coverage BOTH ways: a tensor no parameter claims is a
                // checkpoint this code does not fully understand.
                if !saved.is_empty() {
                    let mut extra: Vec<&String> = saved.keys().collect();
                    extra.sort();
                    return Err(format!("{dir}: {} unclaimed tensors: {extra:?}", extra.len()));
                }
                head_init
            }
            (None, None) => decide::init::init_head(&cfg, seed),
        };
        Ok(DecideWeights { cfg, tok, enc_init, head_init, head_trained, base: base_reference(path) })
    }

    /// Whether the head came from a trained source (the checkpoint's own, or
    /// an explicit file) rather than fresh noise. An untrained head answers
    /// near-uniformly, which is no decision model to serve.
    pub fn head_is_trained(&self) -> bool {
        self.head_trained
    }

    /// Allocate on `gpu`. `trainable` builds the backward buffers and the
    /// optimizer; a serving build leaves them out.
    pub fn build(self, gpu: Gpu, limits: Limits, trainable: bool) -> Decide {
        let mut model = Decide::new_on(gpu, self.cfg, self.tok, limits, &self.enc_init, &self.head_init, trainable);
        model.set_provenance(decide::decide::Provenance { base: self.base, ..Default::default() });
        model
    }
}

// ===================== laya =====================

/// A Laya checkpoint read from disk, not yet on a device.
pub struct LayaWeights {
    ckpt: modernbert::LayaCheckpoint,
    tok: data::qwen_tokenizer::QwenBpe,
    head_init: HashMap<String, Vec<f32>>,
    base: String,
}

impl LayaWeights {
    /// Read a `convaiinnovations/laya`-shaped directory. A supplied `head`
    /// REPLACES the checkpoint's own trained head tensor by tensor, and is
    /// refused by name when it is the other architecture's or the wrong width.
    ///
    /// A directory in the head-only layout ([`crate::laya_head`]) reads the
    /// same way: its configuration and tokenizer come from `dir`, its trunk
    /// from the base it references (resolved and verified by digest), and its
    /// own head file replaces the base's head, which must then be complete -
    /// a trained directory that silently kept some of the base's head would
    /// be a different model than the one that was measured.
    pub fn read(dir: &str, head: Option<&str>) -> Result<LayaWeights, String> {
        let path = Path::new(dir);
        let (ckpt, head_file, complete) = match BaseRef::read(path)? {
            None => (modernbert::import_dir(dir)?, head.map(str::to_string), false),
            Some(base) => {
                if head.is_some() {
                    return Err("an explicit head cannot replace the head of a directory that carries its own".into());
                }
                let trunk = base.resolve(path)?;
                let trunk = trunk.to_str().ok_or("non-UTF-8 base path")?;
                let head_file = path.join(HEAD_FILE).to_str().ok_or("non-UTF-8 head path")?.to_string();
                (modernbert::import_dir_split(dir, trunk)?, Some(head_file), true)
            }
        };
        let tok_path = path.join("tokenizer").join("tokenizer.json");
        let tok = data::qwen_tokenizer::QwenBpe::from_file(tok_path.to_str().ok_or("non-UTF-8 tokenizer path")?)?;

        let mut head_init = ckpt.head_init.clone();
        if let Some(p) = head_file {
            let trained = LayaDecision::read_head_file(&p)?;
            if complete {
                let mut absent: Vec<&String> = head_init.keys().filter(|name| !trained.contains_key(*name)).collect();
                absent.sort();
                if !absent.is_empty() {
                    return Err(format!("{p}: the head is missing {} tensors, first {}", absent.len(), absent[0]));
                }
            }
            for (name, v) in trained {
                match head_init.get(&name) {
                    Some(existing) if existing.len() != v.len() => {
                        return Err(format!("{p}: tensor {name} has {} floats, this checkpoint's head wants {}", v.len(), existing.len()))
                    }
                    None => {
                        return Err(format!(
                            "{p}: tensor {name} is not a Laya decision-head parameter - is this a `decide` head file rather than a Laya one?"
                        ))
                    }
                    Some(_) => {}
                }
                head_init.insert(name, v);
            }
        }
        Ok(LayaWeights { ckpt, tok, head_init, base: base_reference(path) })
    }

    /// The head tensors this checkpoint will be built with, by brain name.
    pub fn head_tensors(&self) -> &HashMap<String, Vec<f32>> {
        &self.head_init
    }

    /// The context length this checkpoint was trained at.
    pub fn max_len(&self) -> u32 {
        self.ckpt.rl.max_len
    }

    /// Allocate on `gpu` with the given training mode. Serving uses
    /// [`modernbert::Training::Off`]: no gradient or optimizer buffers.
    pub fn build(self, gpu: Gpu, mode: modernbert::Training) -> LayaEngine {
        let max_len = self.ckpt.rl.max_len;
        let mut model = LayaDecision::new_on(
            gpu,
            self.ckpt.cfg,
            self.ckpt.laya_cfg,
            self.tok,
            max_len,
            self.ckpt.rl.head_max_len,
            &self.ckpt.encoder_init,
            &self.head_init,
            mode,
        );
        model.set_provenance(modernbert::Provenance { base: self.base, ..Default::default() });
        LayaEngine { model, temperature: self.ckpt.rl.temperature, temperature_by_options: self.ckpt.rl.temperature_by_options }
    }
}

/// The Laya model plus the serving calibration applied on top of it.
pub struct LayaEngine {
    /// Trunk, head and tokenizer as one model.
    pub model: LayaDecision,
    /// Per-qtype (`choice`/`score`/`noul`, [`modernbert::QType::index`] order)
    /// calibration scalar: the FALLBACK when no per-cardinality entry exists.
    temperature: Vec<f32>,
    /// The finer per-(qtype, option-count) table, keyed by
    /// [`modernbert::temp_bucket`] and consulted FIRST. The released
    /// checkpoint fits a 2-option choice at 1.9064 and an 11-or-more-option
    /// one at 0.1006 against a per-qtype 1.6369, so reading only the scalar
    /// publishes a distribution the reference never produces.
    temperature_by_options: HashMap<String, f32>,
}

impl LayaEngine {
    /// The serving temperature for one call: the per-cardinality bucket if
    /// the checkpoint fitted one, else the per-qtype scalar, else 1.0.
    pub fn temperature_for(&self, qtype: modernbert::QType, options: usize) -> f32 {
        if let Some(t) = self.temperature_by_options.get(&modernbert::temp_bucket(qtype, options)) {
            return *t;
        }
        self.temperature.get(qtype.index() as usize).copied().unwrap_or(1.0)
    }

    /// One packed `(state, question)` through trunk and head: raw option
    /// logits, the act head's raw logits, and how much of the state fit.
    pub fn score_raw(&mut self, state: &State, q: &modernbert::Question) -> Result<modernbert::Scored, String> {
        self.model.score_reporting(state, q, None)
    }

    /// Answer ONE typed question about `state`, and report how much of the
    /// state the model saw. The single place this backend turns scores into
    /// an [`Answer`], so the three question types cannot drift apart in how
    /// they are calibrated.
    pub fn ask_reporting(&mut self, state: &State, q: &Question) -> Result<(Answer, StateFit), String> {
        let mq = laya_question(q);
        let scored = self.score_raw(state, &mq)?;
        // Calibrated by how many options were actually SCORED (markers that
        // survived the packing budget), not how many the caller asked about.
        let p = softmax(&scaled(&scored.logits, self.temperature_for(mq.qtype(), scored.logits.len())));
        let answer = match q {
            Question::Choice { options, .. } => {
                if p.len() != options.len() {
                    return Err(format!(
                        "laya: only {} of the {} options fit the model's {}-token context; send fewer or shorter options",
                        p.len(),
                        options.len(),
                        self.model.max_len()
                    ));
                }
                Answer::Choice {
                    choice: options[argmax(&p)].name.clone(),
                    probabilities: options.iter().map(|o| o.name.clone()).zip(p.iter().copied()).collect(),
                    confidence: confidence(&p),
                }
            }
            Question::Score { levels, .. } => {
                if p.len() != levels.len() {
                    return Err(format!(
                        "laya: only {} of the {} levels fit the model's {}-token context; send fewer or shorter levels",
                        p.len(),
                        levels.len(),
                        self.model.max_len()
                    ));
                }
                // The 0-BASED expectation over level indices, the reading
                // `decide::primitives::Question::answer` applies.
                Answer::Score {
                    score: p.iter().enumerate().map(|(i, &pi)| i as f32 * pi).sum(),
                    legend: levels.clone(),
                    confidence: confidence(&p),
                    probabilities: p,
                }
            }
            Question::Noul { .. } => {
                if p.len() != 2 {
                    return Err(format!("laya: a noul question must score exactly 2 options (false, true), got {}", p.len()));
                }
                // render_options() puts the "true" reading at index 1.
                Answer::Noul { noul: p[1] }
            }
        };
        Ok((answer, scored.state))
    }

    /// [`Self::ask_reporting`] without the fit.
    pub fn ask(&mut self, state: &State, q: &Question) -> Result<Answer, String> {
        self.ask_reporting(state, q).map(|(a, _)| a)
    }
}

/// The typed question, in Laya's own `render_options` vocabulary. One place,
/// so the mapping (a `Choice`'s per-option description, a `Score`'s ordered
/// levels, a `Noul`'s two criteria texts - `yes` is the TRUE reading) cannot
/// be spelled differently by two callers.
fn laya_question(q: &Question) -> modernbert::Question {
    match q {
        Question::Choice { instructions, options } => modernbert::Question::Choice {
            ins: instructions.clone(),
            options: options.iter().map(|o: &Opt| (o.name.clone(), o.description.clone())).collect(),
        },
        Question::Score { instructions, levels } => modernbert::Question::Score { ins: instructions.clone(), options: levels.clone() },
        Question::Noul { instructions, yes, no } => {
            modernbert::Question::Noul { ins: instructions.clone(), false_text: no.clone(), true_text: yes.clone() }
        }
    }
}

/// First maximal index, so ties resolve to the option the caller listed first.
pub fn argmax(p: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

/// Divide raw scores by a serving temperature before the softmax; a no-op on a
/// non-finite or non-positive `t`, which a malformed config must not be able
/// to turn into NaN.
pub fn scaled(scores: &[f32], t: f32) -> Vec<f32> {
    if t.is_finite() && t > 0.0 {
        scores.iter().map(|&s| s / t).collect()
    } else {
        scores.to_vec()
    }
}

// ===================== the engine =====================

/// One answered question and how much of the state it was answered from.
#[derive(Clone, Debug, PartialEq)]
pub struct Answered {
    pub answer: Answer,
    pub state: StateFit,
}

/// Everything one `answer` call produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Reply {
    /// One per question, in the order asked.
    pub answered: Vec<Answered>,
    /// State windows the encoder ran over (`decide` only; `None` for Laya,
    /// which packs one bounded sequence per question).
    pub windows: Option<usize>,
}

/// A loaded decision model of either architecture.
pub enum Engine {
    /// Boxed with the Laya arm: both carry whole on-device models, and an unboxed one would make
    /// every variant as large as it is.
    Decide(Box<Decide>),
    Laya(Box<LayaEngine>),
}

impl Engine {
    pub fn backend(&self) -> Backend {
        match self {
            Engine::Decide(_) => Backend::Decide,
            Engine::Laya(_) => Backend::Laya,
        }
    }

    /// The most input tokens this build can hold: Laya's trained context, or
    /// `decide`'s packed-row capacity (state windows plus every option slot).
    pub fn max_input_tokens(&self) -> usize {
        match self {
            Engine::Decide(d) => d.limits.cap_rows as usize,
            Engine::Laya(l) => l.model.max_len() as usize,
        }
    }

    /// Answer several typed questions about ONE state, in order.
    ///
    /// Every question is validated against the published limits BEFORE any
    /// model runs. The `decide` arm encodes the state once for all questions;
    /// the Laya arm re-encodes per question.
    pub fn answer(&mut self, state: &State, questions: &[Question]) -> Result<Reply, String> {
        self.answer_until(state, questions, None).map_err(|e| e.to_string())
    }

    /// [`Engine::answer`] under an optional deadline.
    ///
    /// The deadline is cooperative: it is checked before the model runs and,
    /// on the Laya arm, between questions. A forward pass already on the
    /// device is not interrupted, so a call can overrun by one pass; the
    /// answers of a call that overran are discarded, never half-returned.
    pub fn answer_until(&mut self, state: &State, questions: &[Question], deadline: Option<Instant>) -> Result<Reply, AnswerError> {
        let expired = || deadline.is_some_and(|d| Instant::now() >= d);
        for (i, q) in questions.iter().enumerate() {
            q.validate().map_err(|e| AnswerError::Refused(format!("question {i}: {e}")))?;
        }
        if expired() {
            return Err(AnswerError::DeadlineExceeded);
        }
        match self {
            Engine::Decide(model) => {
                let text = state.serialize();
                let total = model.tok.encode(&text).len();
                let answers = model.decide(&text, questions).map_err(AnswerError::Refused)?;
                if answers.len() != questions.len() {
                    return Err(AnswerError::Refused(format!("the model answered {} of {} questions", answers.len(), questions.len())));
                }
                // Windowing covers the whole state; a request that does not
                // fit is refused by `pack_request`, never silently cut.
                let fit = StateFit { total, kept: total };
                let windows = model.last_shape().1;
                Ok(Reply { answered: answers.into_iter().map(|answer| Answered { answer, state: fit }).collect(), windows: Some(windows) })
            }
            Engine::Laya(l) => {
                let mut answered = Vec::with_capacity(questions.len());
                for q in questions {
                    if expired() {
                        return Err(AnswerError::DeadlineExceeded);
                    }
                    let (answer, state) = l.ask_reporting(state, q).map_err(AnswerError::Refused)?;
                    answered.push(Answered { answer, state });
                }
                Ok(Reply { answered, windows: None })
            }
        }
    }
}

/// Why [`Engine::answer_until`] produced no answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnswerError {
    /// The request cannot be answered as sent: a question outside the
    /// published limits, a state the model cannot hold, options the packing
    /// budget could not keep. Retrying it unchanged cannot succeed.
    Refused(String),
    /// The deadline passed.
    DeadlineExceeded,
}

impl std::fmt::Display for AnswerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnswerError::Refused(m) => f.write_str(m),
            AnswerError::DeadlineExceeded => f.write_str("the deadline passed before the model answered"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("brain-decisions-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_laya_shaped_directory_sniffs_as_laya() {
        let d = scratch("sniff-laya");
        std::fs::write(d.join("rl_agent_config.json"), "{}").unwrap();
        std::fs::create_dir_all(d.join("encoder")).unwrap();
        std::fs::write(d.join("encoder/config.json"), r#"{"model_type":"modernbert"}"#).unwrap();
        assert_eq!(Backend::sniff(&d), Some(Backend::Laya));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_stray_marker_file_does_not_misroute_a_bert_directory() {
        let d = scratch("sniff-stray");
        std::fs::write(d.join("rl_agent_config.json"), "{}").unwrap();
        std::fs::write(d.join("config.json"), "{}").unwrap();
        assert_eq!(Backend::sniff(&d), Some(Backend::Decide));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn an_empty_directory_is_no_backend() {
        let d = scratch("sniff-empty");
        assert_eq!(Backend::sniff(&d), None);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn base_reference_is_the_last_two_components() {
        assert_eq!(base_reference(Path::new("/m/convaiinnovations/laya")), "convaiinnovations/laya");
        assert_eq!(base_reference(Path::new("laya")), "laya");
        assert_eq!(base_reference(Path::new("")), "");
    }

    #[test]
    fn argmax_resolves_ties_to_the_first() {
        assert_eq!(argmax(&[0.2, 0.4, 0.4]), 1);
    }

    #[test]
    fn a_non_positive_temperature_leaves_scores_alone() {
        assert_eq!(scaled(&[1.0, 2.0], 0.0), vec![1.0, 2.0]);
        assert_eq!(scaled(&[1.0, 2.0], f32::NAN), vec![1.0, 2.0]);
        assert_eq!(scaled(&[2.0, 4.0], 2.0), vec![1.0, 2.0]);
    }
}
