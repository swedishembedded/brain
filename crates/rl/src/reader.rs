// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Binding the continual reader's [`Learner`] seam to a real model.
//!
//! `brain-audit` decides what a reader does; this decides how each of those
//! four decisions is actually carried out. Everything here is plumbing onto
//! functions that already exist, which is the point of the seam: the
//! orchestration is tested without a device, and what needs one is four
//! methods.
//!
//! | seam method | what carries it out |
//! |---|---|
//! | `loss` | [`Model::logits_all`] over the episode's own tokens |
//! | `train` | a token dataset, then [`model::fit`], then the adapter tensors |
//! | `score` | [`crate::improve::decode_checkpoint`] |
//! | `joint_oracle` | `train` over the whole bank, then `score` |
//!
//! ## The dataset is plain tokens, not a chat transcript
//!
//! `data::chat::prepare_chat_samples` writes the masked dataset
//! `model::load_dataset` prefers, and it is what the document study uses.
//! But it is CHAT shaped, and a reader's rows are lines of a document rather
//! than turns of a conversation. Training through it would teach the model
//! that those lines arrive wrapped in a template it will never see when it
//! is actually asked to continue one, so the training distribution would
//! disagree with the prompting distribution by construction. A reader
//! therefore writes a plain token dataset and is trained the way it is
//! prompted.
//!
//! ## Two arms means two checkpoints on disk
//!
//! `promote::gate`'s contract is that both arms are decoded from what would
//! actually be served, never from a freshly-trained in-memory instance. So
//! the incumbent and the candidate are files, and a promotion is the
//! candidate becoming the incumbent.

use std::path::{Path, PathBuf};

use audit::bank::Probe;
use audit::reader::{Arm, Learner, Scored};
use data::binio;
use data::prompting::Prompting;
use data::tokenizer::Tokenizer;
use model::rollout::RolloutParams;
use model::{FitOpts, Model, ModelConfig};
use promote::document::normalize;
use promote::env::{Reward, Step, Task, Verifier};

use crate::improve::decode_checkpoint;

/// Exact match after [`normalize`], which is the one definition of how much
/// latitude an exact match gets in this workspace. Never a second model
/// judging the first.
///
/// It holds the tokenizer because correctness here is defined over TEXT, not
/// over token ids: two tokenisations of the same answer are the same answer,
/// and comparing ids would call one of them wrong.
struct ExactMatch<'a, T: Tokenizer> {
    tok: &'a T,
}

impl<T: Tokenizer> Verifier for ExactMatch<'_, T> {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        let expected = task.answer["expected"].as_str().unwrap_or_default();
        let hit = normalize(&self.tok.decode(completion)) == normalize(expected);
        let mut parts = std::collections::BTreeMap::new();
        parts.insert("exact_match".to_string(), if hit { 1.0 } else { 0.0 });
        Reward { value: if hit { 1.0 } else { 0.0 }, parts }
    }
}

/// A [`Learner`] over a real checkpoint.
pub struct ModelLearner<'a, M: Model, T: Tokenizer> {
    tok: &'a T,
    cfg: M::Config,
    work: PathBuf,
    fit: FitOpts,
    rollout: RolloutParams,
    incumbent: PathBuf,
    candidate: PathBuf,
    /// The overlay's rank and alpha. Carried explicitly because they are
    /// written into the adapter's own card, and `fold_adapter_into` READS
    /// the rank from there to do the fold: a card that understated it would
    /// produce a silently wrong fold rather than an error.
    rank: u32,
    alpha: f32,
    /// Held so the reach test does not reload a checkpoint per episode.
    cached: Option<M>,
    /// `(initial, final)` from the last `model::fit_from`. See
    /// `Learner::last_train_loss`.
    last_train_loss: Option<(f64, f64)>,
    /// How this model wants to be asked. Read off the base checkpoint's own
    /// directory, so an instruction-tuned model is asked a question rather
    /// than handed one to continue - see `data::prompting`.
    prompting: Prompting,
}

impl<'a, M: Model, T: Tokenizer> ModelLearner<'a, M, T> {
    /// `base` is the frozen starting checkpoint and becomes the first
    /// incumbent; `work` holds the scratch datasets and both arms.
    ///
    /// A work directory that already holds an incumbent keeps it. A run is
    /// several processes - one reads and promotes, the next scores the
    /// frozen battery against what it left served - so seeding the incumbent
    /// from the base every time a learner is built would silently discard
    /// every promotion and make each such comparison one of the base against
    /// itself. Which base a given work directory belongs to is the caller's
    /// to keep straight; the run manifest is where that is recorded.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &Path,
        work: &Path,
        tok: &'a T,
        cfg: M::Config,
        fit: FitOpts,
        rollout: RolloutParams,
        rank: u32,
        alpha: f32,
    ) -> std::io::Result<Self> {
        std::fs::create_dir_all(work)?;
        let incumbent = work.join("incumbent.safetensors");
        if !incumbent.exists() {
            std::fs::copy(base, &incumbent)?;
        }
        Ok(ModelLearner {
            tok,
            cfg,
            work: work.to_path_buf(),
            fit,
            rollout,
            incumbent,
            candidate: work.join("candidate.safetensors"),
            rank,
            alpha,
            cached: None,
            last_train_loss: None,
            prompting: base.parent().map(Prompting::for_model_dir).unwrap_or_else(Prompting::none),
        })
    }

    /// Adopt the candidate as what is served. Called after the gate
    /// promotes, and the only way the incumbent ever moves.
    pub fn promote(&mut self) -> std::io::Result<()> {
        std::fs::copy(&self.candidate, &self.incumbent)?;
        // The cached instance is now a stale copy of a file that changed.
        self.cached = None;
        Ok(())
    }

    fn path_of(&self, arm: Arm) -> &Path {
        match arm {
            Arm::Incumbent => &self.incumbent,
            Arm::Candidate => &self.candidate,
        }
    }

    fn incumbent_model(&mut self) -> &M {
        if self.cached.is_none() {
            let c = checkpoint::load(self.incumbent.to_str().expect("utf-8 path"));
            let cfg = M::Config::from_json(&c.header["config"]);
            let init = c.by_role("");
            // Only `loss` reads this, and a loss is one forward pass. The
            // training shape would carry the backward scratch and a
            // per-layer copy of every activation for a pass this never runs.
            self.cached = Some(M::new_inference(cfg.clone(), 1, cfg.block_size(), &init));
        }
        self.cached.as_ref().expect("just built")
    }

    /// Write `rows` as a plain token dataset `model::load_dataset` can read.
    ///
    /// Both splits must exceed `block_size` or the loader refuses the
    /// directory, so short input is repeated rather than left to fail deep
    /// inside the training loop with a message about sampling windows.
    fn write_dataset(&self, rows: &[&str], dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut tokens: Vec<u32> = Vec::new();
        for row in rows {
            tokens.extend(self.tok.encode(row));
        }
        let floor = self.fit.block_size as usize + 2;
        if tokens.is_empty() {
            return Err(std::io::Error::other("reader: nothing to train on"));
        }
        while tokens.len() < floor {
            let head: Vec<u32> = tokens.clone();
            tokens.extend(head);
        }
        binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens)?;
        // The loader requires a val split and this trainer does not use one
        // for a decision: the gate is what decides, on frozen probes, and a
        // held-out slice of the same episode would be a second, weaker
        // answer to a question already being asked properly.
        binio::write_u32_bin(&dir.join("val.u32.bin"), &tokens)?;
        let meta = binio::Meta { vocab_size: self.tok.vocab_size(), itos: Vec::new() };
        std::fs::write(dir.join("meta.json"), meta.to_json())
    }

    /// Train into `out`, STARTING FROM WHAT IS CURRENTLY SERVED, and return
    /// the adapter tensors from the result.
    ///
    /// The incumbent is the starting point, and saying so explicitly is
    /// load-bearing. `model::fit` starts from `out` if it exists and from
    /// fresh random weights otherwise, which for a reader means the first
    /// episode trains a randomly initialised model and every later one
    /// resumes from that - a run that reports promotions, retention and a
    /// battery about a network that never saw the pretrained weights. There
    /// is no symptom at the call site: the numbers all have the right shape.
    ///
    /// A previous candidate at `out` is NOT a starting point either. It may
    /// be one the gate refused, and resuming from it would carry refused
    /// training forward, which is the one thing the gate exists to stop.
    fn fit_into(&mut self, rows: &[&str], dir: &Path, out: &Path, seed: u64) -> std::io::Result<Vec<u8>> {
        self.write_dataset(rows, dir)?;
        let mut opts = self.fit.clone();
        opts.seed = seed;
        let served = checkpoint::load(self.incumbent.to_str().expect("utf-8 path"));
        let (initial, final_) = model::fit_from::<M>(dir, self.cfg.clone(), &opts, Some(out), &served.by_role(""))?;
        self.last_train_loss = Some((initial as f64, final_ as f64));
        adapter_bytes::<M>(out, &self.work.join("adapter.safetensors"), self.rank, self.alpha)
    }
}

/// Extract just the LoRA tensors from a trained checkpoint, as the bytes the
/// pool stores. The whole checkpoint would be the base weights over again,
/// once per promoted episode.
fn adapter_bytes<M: Model>(checkpoint_path: &Path, scratch: &Path, rank: u32, alpha: f32) -> std::io::Result<Vec<u8>> {
    let c = checkpoint::load(checkpoint_path.to_str().expect("utf-8 path"));
    let cfg = M::Config::from_json(&c.header["config"]);
    let init = c.by_role("");
    let m = M::new(cfg.clone(), 1, cfg.block_size(), &init);
    // The device-adapter family writes an adapter straight out of a model's
    // own param list, which is what a LoRA-overlaid config produces.
    model::lora::device_adapter::save_adapter::<M>(
        scratch.to_str().expect("utf-8 path"),
        &m,
        rank,
        alpha,
        &[],
        "reader-adapter",
        "base",
        "reader",
        None,
    )?;
    std::fs::read(scratch)
}

impl<M: Model, T: Tokenizer> Learner for ModelLearner<'_, M, T> {
    fn loss(&mut self, text: &str) -> f64 {
        let block = self.cfg.block_size() as usize;
        let mut ids = self.tok.encode(text);
        ids.truncate(block);
        if ids.len() < 2 {
            // Nothing to predict from. Reported as out of reach rather than
            // as a suspiciously perfect score.
            return f64::INFINITY;
        }
        let model = self.incumbent_model();
        let Some(logits) = model.logits_all(&ids) else {
            return f64::INFINITY;
        };
        let vocab = logits.len() / ids.len();
        let mut total = 0.0f64;
        for i in 0..ids.len() - 1 {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = row.iter().map(|x| ((x - max) as f64).exp()).sum();
            let target = row[ids[i + 1] as usize] as f64;
            total += (max as f64) + sum.ln() - target;
        }
        total / (ids.len() - 1) as f64
    }

    fn train(&mut self, rows: &[&str], seed: u64) -> Vec<u8> {
        let dir = self.work.join("episode");
        let candidate = self.candidate.clone();
        // The seam returns bytes rather than a result, so a failure here has
        // nowhere to go but a panic. That is deliberate and it is the lesser
        // evil: empty bytes would be admitted to the pool as a valid adapter
        // and every later episode would route through nothing. A reader that
        // cannot train is not a reader that should quietly carry on
        // reporting rejections.
        self.fit_into(rows, &dir, &candidate, seed)
            .unwrap_or_else(|e| panic!("rl::reader: training a candidate failed, and an empty adapter is not a valid outcome: {e}"))
    }

    fn score(&mut self, arm: Arm, probes: &[&Probe]) -> Scored {
        let tasks: Vec<Task> = probes
            .iter()
            .map(|p| Task {
                id: p.id.as_str().to_string(),
                prompt: self.tok.encode(&self.prompting.question(&p.prompt)),
                answer: serde_json::json!({ "expected": p.expected }),
            })
            .collect();
        let verifier = ExactMatch { tok: self.tok };
        let (scores, mean_entropy, completions) = decode_checkpoint::<M>(self.path_of(arm), &tasks, &verifier, &self.rollout);
        let answers = completions.iter().map(|c| self.tok.decode(c)).collect();
        Scored { scores, mean_entropy, answers }
    }

    fn last_train_loss(&self) -> Option<(f64, f64)> {
        self.last_train_loss
    }

    fn joint_oracle(&mut self, rows: &[&str], probes: &[&Probe]) -> f64 {
        let dir = self.work.join("oracle");
        let out = self.work.join("oracle.safetensors");
        if self.fit_into(rows, &dir, &out, 0).is_err() {
            return 0.0;
        }
        let tasks: Vec<Task> = probes
            .iter()
            .map(|p| Task { id: p.id.as_str().to_string(), prompt: self.tok.encode(&p.prompt), answer: serde_json::json!({ "expected": p.expected }) })
            .collect();
        let verifier = ExactMatch { tok: self.tok };
        let (scores, _, _) = decode_checkpoint::<M>(&out, &tasks, &verifier, &self.rollout);
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        }
    }
}
