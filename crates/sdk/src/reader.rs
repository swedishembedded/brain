// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`ContinualReader`]: point a model at a directory and leave it reading.
//!
//! The reader decides per episode whether what it just read is worth
//! learning, whether it learned it, and what that cost anything it already
//! knew. Everything it has learned and everything it knows about what it
//! learned lives in one run directory that can be stopped and reopened.
//!
//! ## Two instruments, and only one of them is the result
//!
//! [`ContinualReader::read`] reports how many episodes promoted and at what
//! audit cost. That is plumbing. [`ContinualReader::battery`] scores a set
//! of held-out tasks the caller froze BEFORE any reading, and the change in
//! that score between two calls is the only thing that answers "what can the
//! user do now that they could not before".
//!
//! A run whose promote rate rises while its battery stays flat has failed,
//! and the two being separate surfaces is what makes that visible rather
//! than arithmetic nobody does.
//!
//! ```no_run
//! let reader = brain::ContinualReader::from_pretrained("Qwen/Qwen3-0.6B").run_dir("run");
//! let before = reader.battery(&tasks)?;   // frozen before any reading
//! reader.corpus("docs").read()?;
//! let after = reader.battery(&tasks)?;
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! Swedish Embedded AB builds continual-learning systems that can say what
//! they gained and what it cost, rather than ones that can only say they
//! trained. If your team needs a model that keeps learning from what you
//! give it without quietly losing what it could do last week, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use audit::pool::{Pool, PoolConfig};
use audit::reader::{Arm, Learner, Reader, ReaderConfig};
use audit::acceptance::LedgerFacts;
use audit::run::{LedgerRow, Manifest, Run, SCHEMA};
use audit::stream::{EpisodeStream, StreamConfig};
use data::qwen_tokenizer::QwenBpe;
use model::rollout::RolloutParams;
use model::serve::SampleParams;
use model::{FitOpts, Model, ModelConfig};
use rl::reader::ModelLearner;

use crate::study::{resolve_base, Qwen3, Qwen35, Qwen35Moe, StudyArch};
use crate::{Error, Result};

/// One held-out task of a capability battery: what the model is shown, and
/// the one answer that counts as correct.
#[derive(Clone, Debug, PartialEq)]
pub struct BatteryTask {
    pub prompt: String,
    pub expected: String,
}

/// What a battery said.
#[derive(Clone, Debug, PartialEq)]
pub struct BatteryScore {
    pub passed: usize,
    pub total: usize,
    /// Per task, in the order given, so a caller can show WHICH abilities
    /// were gained rather than only how many.
    pub per_task: Vec<bool>,
    /// What the model actually answered. A battery that can only report a
    /// count cannot report HOW the failures were wrong, and a caller with
    /// its own grammar can check acceptance rather than only exact match.
    pub answers: Vec<String>,
}

impl BatteryScore {
    pub fn rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }
}

/// What a stretch of reading did.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadOutcome {
    pub episodes: usize,
    pub promoted: usize,
    /// Earlier episodes the reader now carries, and how many episodes it
    /// takes to re-check all of them. The reader reports this instead of
    /// claiming nothing was forgotten.
    pub bank: usize,
    pub detection_latency: u64,
    /// Backward transfer over the retention matrix: the mean, over every
    /// earlier episode the audit schedule brought round again, of what it
    /// scores now minus what it scored when it was learned. Negative is
    /// forgetting.
    ///
    /// `None` when nothing has been re-observed yet, which is NOT the same
    /// answer as zero. A run too short for the schedule to revisit anything
    /// has no backward transfer to report, and reporting `0.0` there would
    /// read as perfect retention - the most flattering number available and
    /// the one least earned.
    pub bwt: Option<f64>,
    /// How many episodes carry a diagonal, and how many of those have been
    /// scored again. The second is the weight [`ReadOutcome::bwt`] carries:
    /// one observation and forty are not the same evidence.
    pub retention_coverage: (usize, usize),
    /// The largest drop any SINGLE earlier episode has taken, as a positive
    /// magnitude. A healthy mean can hide one episode that collapsed, and
    /// the per-block bar is about exactly that.
    pub worst_block_drop: Option<f64>,
}

/// A model reading continually.
pub struct ContinualReader {
    model: String,
    arch: String,
    models_dir: Option<String>,
    run_dir: PathBuf,
    corpus: Option<PathBuf>,
    rank: u32,
    alpha: Option<f32>,
    steps: u32,
    /// The TRAINING window, in tokens. See [`ContinualReader::train_block`].
    train_block: u32,
    until: Option<usize>,
    seed: u64,
    cfg: ReaderConfig,
}

impl std::fmt::Debug for ContinualReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinualReader").field("model", &self.model).field("run_dir", &self.run_dir).finish_non_exhaustive()
    }
}

/// The default TRAINING window, in tokens - see
/// [`ContinualReader::train_block`] for why this is not the model's own
/// context length.
///
/// A reader's rows are LINES of a document. 1024 tokens is a window many
/// times longer than any of them and still an eighth of the attention
/// scratch a 2048 window would hold resident across every layer. Raise it
/// for a corpus whose structure genuinely spans more than that, and expect
/// to pay for it quadratically.
const DEFAULT_TRAIN_BLOCK: u32 = 1024;

/// One registry row, monomorphised for its `Model` impl.
type ReadFn = fn(&Inputs) -> Result<ReadOutcome>;
type BatteryFn = fn(&Inputs, &[BatteryTask]) -> Result<BatteryScore>;

/// The SAME architectures the document study serves, through the same
/// `StudyArch` seam. One registry with two entry points, rather than a
/// second copy of the per-architecture plumbing.
const ARCHS: &[(&str, ReadFn, BatteryFn)] = &[
    ("qwen3", read_for::<Qwen3>, battery_for::<Qwen3>),
    ("qwen35", read_for::<Qwen35>, battery_for::<Qwen35>),
    ("qwen35moe", read_for::<Qwen35Moe>, battery_for::<Qwen35Moe>),
];

fn arch_names() -> String {
    ARCHS.iter().map(|(n, _, _)| *n).collect::<Vec<&str>>().join(", ")
}

struct Inputs<'a> {
    base: &'a Path,
    tok: &'a QwenBpe,
    run: &'a Run,
    corpus: Option<&'a Path>,
    rank: u32,
    alpha: f32,
    steps: u32,
    train_block: u32,
    until: Option<usize>,
    seed: u64,
    cfg: ReaderConfig,
}

impl ContinualReader {
    /// Read with `model`, resolved by brain's own model handler exactly as
    /// every other surface resolves one, so a reference that is not on disk
    /// is fetched rather than refused.
    pub fn from_pretrained(model: &str) -> ContinualReader {
        ContinualReader {
            model: model.to_string(),
            arch: "qwen3".to_string(),
            models_dir: None,
            run_dir: PathBuf::from("run"),
            corpus: None,
            rank: 8,
            alpha: None,
            steps: 32,
            train_block: DEFAULT_TRAIN_BLOCK,
            until: None,
            seed: 0,
            cfg: ReaderConfig::default(),
        }
    }

    pub fn arch(mut self, arch: &str) -> Self {
        self.arch = arch.to_string();
        self
    }

    pub fn models_dir(mut self, dir: &str) -> Self {
        self.models_dir = Some(dir.to_string());
        self
    }

    /// Where everything the reader learns and knows about what it learned
    /// lives. Reopened if it already exists.
    pub fn run_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.run_dir = dir.into();
        self
    }

    pub fn corpus(mut self, dir: impl Into<PathBuf>) -> Self {
        self.corpus = Some(dir.into());
        self
    }

    /// Stop after this many episodes. Without it, reading continues to the
    /// end of the corpus.
    pub fn until(mut self, episodes: usize) -> Self {
        self.until = Some(episodes);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.cfg.seed = seed;
        self
    }

    pub fn rank(mut self, rank: u32) -> Self {
        self.rank = rank;
        self
    }

    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// The window, in tokens, that an episode is TRAINED over. Capped at the
    /// base model's own block size; a request above it is lowered rather
    /// than refused, since a window longer than the model's context is not a
    /// thing the model can be trained at.
    ///
    /// This is the reader's dominant memory term and it is quadratic: a
    /// training build keeps every layer's attention scratch resident at once
    /// (a backward pass reads all of them), so the cost is
    /// `n_layers * n_heads * block^2`. At a 0.6B model's own 2048 that is
    /// 7 GiB of attention probabilities alone, for documents whose lines are
    /// tens of tokens long.
    pub fn train_block(mut self, tokens: u32) -> Self {
        self.train_block = tokens.max(1);
        self
    }

    /// Read the corpus, resuming if the run directory has been read before.
    pub fn read(&self) -> Result<ReadOutcome> {
        let (f, _) = self.dispatch()?;
        self.with_inputs(f)
    }

    /// Score a battery of held-out tasks against what is currently served.
    ///
    /// The tasks are the caller's, frozen before any reading: this surface
    /// scores them and does not invent them, because a battery a reader
    /// chose for itself would be a reader marking its own homework.
    pub fn battery(&self, tasks: &[BatteryTask]) -> Result<BatteryScore> {
        let (_, f) = self.dispatch()?;
        self.with_inputs(|i| f(i, tasks))
    }

    /// Every episode the run has recorded.
    pub fn ledger(&self) -> Result<Vec<LedgerRow>> {
        let run = Run::open(&self.run_dir).map_err(|e| Error::Backend(e.to_string()))?;
        run.ledger().map_err(|e| Error::Backend(e.to_string()))
    }

    /// What the run's own record can say about itself, for the clauses of
    /// the acceptance block that are questions about its episodes.
    ///
    /// Deliberately not the whole block: a null-gate count, a retention
    /// matrix and a battery delta are not in a ledger, and this surface does
    /// not invent them. A caller assembling `RunFacts` supplies those by
    /// name, so an unanswered clause stays unanswered rather than quietly
    /// passing on a default.
    pub fn ledger_facts(&self) -> Result<LedgerFacts> {
        Ok(LedgerFacts::of(&self.ledger()?))
    }

    /// Episodes this run refused without recording why. Empty is what
    /// clause 1 of the block requires.
    pub fn unexplained_refusals(&self) -> Result<Vec<String>> {
        let rows = self.ledger()?;
        Ok(LedgerFacts::unexplained(&rows).into_iter().map(str::to_string).collect())
    }

    fn dispatch(&self) -> Result<(ReadFn, BatteryFn)> {
        ARCHS
            .iter()
            .find(|(n, _, _)| *n == self.arch)
            .map(|(_, r, b)| (*r, *b))
            .ok_or_else(|| Error::Backend(format!("unknown architecture {:?} - this surface serves: {}", self.arch, arch_names())))
    }

    fn with_inputs<T>(&self, f: impl FnOnce(&Inputs) -> Result<T>) -> Result<T> {
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (base, base_dir, _) = resolve_base(&self.model, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let tok_path = base_dir.join("tokenizer.json");
        let tok = QwenBpe::from_file(tok_path.to_str().unwrap_or_default()).map_err(|e| Error::Backend(format!("{}: {e}", tok_path.display())))?;

        let manifest = Manifest { schema: SCHEMA, model: self.model.clone(), cfg: self.cfg };
        // Reopened under the model the run was created with, never another:
        // what a run has promoted is folded into THAT checkpoint, so serving
        // it from a different base would report one model's learning as
        // another's.
        let run = if self.run_dir.join("manifest.json").exists() {
            Run::open_for(&self.run_dir, &self.model).map_err(|e| Error::Backend(e.to_string()))?
        } else {
            Run::create(&self.run_dir, &manifest).map_err(|e| Error::Backend(e.to_string()))?
        };

        f(&Inputs {
            base: &base,
            tok: &tok,
            run: &run,
            corpus: self.corpus.as_deref(),
            rank: self.rank,
            alpha: self.alpha.unwrap_or(self.rank as f32 * 2.0),
            steps: self.steps,
            train_block: self.train_block,
            until: self.until,
            seed: self.seed,
            cfg: self.cfg,
        })
    }
}

/// Build the learner an entry point drives. Both entry points need exactly
/// the same one, and building it in one place is what keeps `battery`
/// scoring the same served checkpoint that `read` promoted into.
fn learner_for<'a, A: StudyArch>(i: &'a Inputs<'a>) -> Result<ModelLearner<'a, A::M, QwenBpe>> {
    let c = checkpoint::load(i.base.to_str().unwrap_or_default());
    let base_cfg = <A::M as Model>::Config::from_json(&c.header["config"]);
    // Never above what the model was built for: a training window longer
    // than its context is not a window it has.
    let block = i.train_block.min(base_cfg.block_size()).max(1);
    let cfg = A::study_config(&base_cfg, A::lora(i.rank, i.alpha), block);
    let fit = FitOpts { steps: i.steps, batch_size: 1, block_size: block, seed: i.seed, eval_interval: 0, eval_batches: 0, checkpoint_secs: 0, ..FitOpts::default() };
    let rollout = RolloutParams { max_new: 64, sample: SampleParams::greedy(), eos: None };
    ModelLearner::new(i.base, &i.run.root().join("work"), i.tok, cfg, fit, rollout, i.rank, i.alpha)
        .map_err(|e| Error::Backend(format!("{}: {e}", i.base.display())))
}

fn read_for<A: StudyArch>(i: &Inputs) -> Result<ReadOutcome> {
    let corpus = i.corpus.ok_or_else(|| Error::Backend("nothing to read: set a corpus directory".to_string()))?;
    let learner = learner_for::<A>(i)?;

    let pool_root = i.run.pool_root();
    let pool = if pool_root.join("index.json").exists() {
        Pool::open(&pool_root).map_err(|e| Error::Backend(e.to_string()))?
    } else {
        Pool::create(&pool_root, PoolConfig::default()).map_err(|e| Error::Backend(e.to_string()))?
    };

    let saved = i.run.load_state().map_err(|e| Error::Backend(e.to_string()))?;
    let stream_cfg = StreamConfig { seed: i.seed, ..StreamConfig::default() };
    let mut stream = match saved.as_ref().and_then(|s| s.cursor.clone()) {
        Some(cursor) => EpisodeStream::resume(corpus, stream_cfg, &cursor).map_err(|e| Error::Backend(e.to_string()))?,
        None => EpisodeStream::open(corpus, stream_cfg).map_err(|e| Error::Backend(e.to_string()))?,
    };

    let mut reader = match saved {
        Some(state) => Reader::restore(i.cfg, learner, pool, state),
        None => Reader::new(i.cfg, learner, pool),
    };

    let mut episodes = 0usize;
    let mut promoted = 0usize;
    loop {
        if i.until.is_some_and(|n| episodes >= n) {
            break;
        }
        let Some(ep) = stream.next() else { break };
        let row = reader.step(&ep);
        if row.outcome.promoted() {
            promoted += 1;
            // What the gate promoted is what must be served from here on.
            reader.learner_mut().promote().map_err(|e| Error::Backend(e.to_string()))?;
        }
        i.run.append(&LedgerRow::of(reader.episode(), &row)).map_err(|e| Error::Backend(e.to_string()))?;
        episodes += 1;
    }

    reader.checkpoint(i.run, Some(stream.cursor())).map_err(|e| Error::Backend(e.to_string()))?;
    let retention = reader.retention();
    Ok(ReadOutcome {
        episodes,
        promoted,
        bank: reader.bank_size(),
        detection_latency: reader.detection_latency(),
        bwt: retention.bwt(),
        retention_coverage: retention.coverage(),
        worst_block_drop: retention.worst_drop(),
    })
}

fn battery_for<A: StudyArch>(i: &Inputs, tasks: &[BatteryTask]) -> Result<BatteryScore> {
    let mut learner = learner_for::<A>(i)?;
    let probes: Vec<audit::bank::Probe> = tasks.iter().enumerate().map(|(n, t)| audit::bank::Probe::for_battery(n, &t.prompt, &t.expected)).collect();
    let refs: Vec<&audit::bank::Probe> = probes.iter().collect();
    let scored = learner.score(Arm::Incumbent, &refs);
    let per_task: Vec<bool> = scored.scores.iter().map(|s| *s >= 1.0).collect();
    Ok(BatteryScore { passed: per_task.iter().filter(|p| **p).count(), total: tasks.len(), per_task, answers: scored.answers })
}
