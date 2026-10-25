// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language models on
// customer images, video and conversations for its clients. If your team
// needs expertise in adapting multimodal models to your own data then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A fine-tuning run over a checkpoint as downloaded: the dataset is read
//! and validated up front, then [`Trainer`] steps through it for a number of
//! epochs, scoring a held-out set after each one, and the result is written
//! as a tune directory serving loads ([`crate::MiniCpmV::load_tuned`]).
//!
//! The unit of training is one supervised assistant turn ([`Sample`]). Its
//! media are decoded again each time it is drawn, so a dataset of any size
//! costs the memory of one sample, not of every image at once. The order is
//! a seeded shuffle per epoch; a run's position (supervised turns consumed)
//! and its whole optimiser state are written every `checkpoint_every` steps,
//! so a resumed run draws the same batches and ends with the same tensors as
//! an uninterrupted one.

use std::path::{Path, PathBuf};

use gpu_core::Gpu;
use model::train::LrSchedule;

use crate::config::{Downsample, MiniCpmVConfig};
use crate::dataset::{load_media, read_jsonl, samples, Record, Sample};
use crate::preprocess::ImageOptions;
use crate::prompt::Formatter;
use crate::train::{PartMode, Rates, TrainSpec, Trainer};

/// The training state's file name inside the output directory.
pub const STATE_FILE: &str = "train_state.safetensors";

/// What a run trains and how.
#[derive(Clone, Debug)]
pub struct Options {
    pub decoder: PartMode,
    pub connector: bool,
    pub vision: PartMode,
    pub epochs: u32,
    /// Supervised turns per optimiser step.
    pub batch: u32,
    /// The peak rate of the decoder (and of a LoRA tower).
    pub lr: f32,
    /// The connector's peak rate; `None` for [`Rates::around`]'s.
    pub connector_lr: Option<f32>,
    /// The tower's peak rate; `None` for [`Rates::around`]'s.
    pub vision_lr: Option<f32>,
    /// Where the cosine ends, as a fraction of each peak.
    pub min_lr_ratio: f32,
    // perf-number: a schedule default, not a measurement
    /// Warmup steps; `None` for 5% of the run (at least one).
    pub warmup: Option<u32>,
    pub weight_decay: f32,
    pub grad_clip: Option<f32>,
    pub seed: u64,
    /// Tokens per sample; `None` sizes it to the longest sample.
    pub block: Option<u32>,
    /// The checkpoint's token-reduction mode unless set.
    pub downsample: Option<Downsample>,
    /// The checkpoint's slice budget unless set.
    pub max_slices: Option<u32>,
    pub max_frames: u32,
    /// Steps between training-state saves; `0` saves only at epoch ends.
    pub checkpoint_every: u32,
    /// Continue from the output directory's saved state when one is there.
    pub resume: bool,
}

impl Options {
    /// LoRA on the decoder (rank 16), the connector trained, the tower
    /// frozen: the cheap recipe that adapts what the model says about
    /// images without moving how it sees them.
    pub fn new(seed: u64) -> Options {
        Options {
            decoder: PartMode::Lora { rank: 16, alpha: 32.0 },
            connector: true,
            vision: PartMode::Frozen,
            epochs: 3,
            batch: 1,
            lr: 1e-4,
            connector_lr: None,
            vision_lr: None,
            min_lr_ratio: 0.1,
            warmup: None,
            weight_decay: 0.0,
            grad_clip: Some(1.0),
            seed,
            block: None,
            downsample: None,
            max_slices: None,
            max_frames: crate::video::MAX_NUM_FRAMES,
            checkpoint_every: 0,
            resume: false,
        }
    }
}

impl Options {
    /// Each part's peak rate: the ones given, the rest from [`Rates::around`].
    pub fn rates(&self) -> Rates {
        let d = Rates::around(self.lr, self.vision);
        Rates { decoder: self.lr, connector: self.connector_lr.unwrap_or(d.connector), vision: self.vision_lr.unwrap_or(d.vision) }
    }
}

/// Progress reported to the caller.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// `lr` is the decoder's rate at this step.
    Step { step: u32, steps: u32, epoch: u32, loss: f32, lr: f32 },
    /// The mean held-out loss before training (`epoch` 0) and after each epoch.
    Eval { epoch: u32, loss: f32 },
    Checkpoint { step: u32 },
}

/// What a finished run reports.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// Supervised turns in the training set.
    pub turns: usize,
    pub block: u32,
    pub steps: u32,
    /// Held-out loss before and after, when a held-out set was given.
    pub initial_eval: Option<f32>,
    pub final_eval: Option<f32>,
}

/// A dataset read and checked: its records and, per supervised turn, which
/// record and which of its turns.
struct Split {
    records: Vec<Record>,
    turns: Vec<(usize, usize)>,
    longest: usize,
}

/// Everything sample building needs from the checkpoint.
struct Builder {
    fmt: Formatter,
    image: ImageOptions,
    max_frames: u32,
}

impl Builder {
    fn record_samples(&self, r: &Record) -> Result<Vec<Sample>, String> {
        let (images, videos) = load_media(r, &self.image, self.max_frames)?;
        samples(&self.fmt, r, &images, &videos, &self.image)
    }

    /// Read `path` and build every sample once: a bad record, a missing or
    /// undecodable file, or a turn that renders to nothing fails here, before
    /// any GPU work, naming the line.
    fn split(&self, path: &Path) -> Result<Split, String> {
        let records = read_jsonl(path)?;
        if records.is_empty() {
            return Err(format!("{}: no records", path.display()));
        }
        let (mut turns, mut longest) = (Vec::new(), 0);
        for (i, r) in records.iter().enumerate() {
            let s = self.record_samples(r).map_err(|e| format!("{} record {}: {e}", path.display(), i + 1))?;
            longest = s.iter().map(|s| s.ids.len()).fold(longest, usize::max);
            turns.extend((0..s.len()).map(|k| (i, k)));
        }
        Ok(Split { records, turns, longest })
    }

    fn sample(&self, split: &Split, (record, turn): (usize, usize)) -> Result<Sample, String> {
        self.record_samples(&split.records[record])?.into_iter().nth(turn).ok_or_else(|| format!("record {} lost its turn {turn}", record + 1))
    }
}

/// Epoch `epoch`'s order of `n` turns: a Fisher-Yates shuffle seeded by the
/// run's seed and the epoch, so a resumed run draws the same order.
/// The mean loss over every turn of `split`.
fn evaluate(t: &mut Trainer, b: &Builder, split: &Split) -> Result<f32, String> {
    let mut sum = 0.0f64;
    for &turn in &split.turns {
        sum += t.loss(&b.sample(split, turn)?)? as f64;
    }
    Ok((sum / split.turns.len() as f64) as f32)
}

/// Write `t`'s state atomically: a crash mid-write leaves the previous one.
fn checkpoint(t: &Trainer, out: &Path, position: u64) -> Result<(), String> {
    let tmp = out.join(format!("{STATE_FILE}.tmp"));
    t.save_state(&tmp, position)?;
    std::fs::rename(&tmp, out.join(STATE_FILE)).map_err(|e| format!("{}: {e}", out.display()))
}

/// The trainer as the shared session sees it: a sample INDEX in, the rest is
/// the session's. What stays model-specific is mapping that index to its
/// encoded sample (the builder's own records/turns).
struct SessionTrainer<'a> {
    t: Trainer,
    b: &'a Builder,
    split: &'a Split,
}

impl model::session::SampleTrainer for SessionTrainer<'_> {
    fn batch_step(&mut self, samples: &[usize], lr: f32) -> Result<f32, String> {
        let mut loss = 0.0;
        for &sample in samples {
            let s = self.b.sample(self.split, self.split.turns[sample])?;
            loss += self.t.accumulate(&s)?;
        }
        self.t.step(lr, samples.len());
        Ok(loss / samples.len() as f32)
    }
    fn steps(&self) -> u32 {
        self.t.steps
    }
    fn restore(&mut self, state: &Path) -> Result<u64, String> {
        self.t.restore(state)
    }
    fn checkpoint(&self, out: &Path, position: u64) -> Result<(), String> {
        checkpoint(&self.t, out, position)
    }
}

/// Which checkpoint a run tunes, on what, and where the result goes.
#[derive(Clone, Copy, Debug)]
pub struct Job<'a> {
    /// The checkpoint directory and its model id.
    pub dir: &'a Path,
    pub base_id: &'a str,
    /// The training records and, optionally, held-out ones scored per epoch.
    pub train: &'a Path,
    pub held_out: Option<&'a Path>,
    /// The tune directory; the training state is kept in it too.
    pub out: &'a Path,
    /// `owner:name:tag`, recorded in the tune.
    pub label: &'a str,
}

/// Run `job` with `opts` on `gpu`. The trainer's device memory is
/// reclaimed before this returns, so the caller can load the tune (or any
/// other model) right after.
pub fn finetune(gpu: &Gpu, job: Job, opts: &Options, progress: &mut dyn FnMut(&Event)) -> Result<Outcome, String> {
    gpu_core::reclaiming(gpu, || run(gpu, job, opts, progress))
}

fn run(gpu: &Gpu, job: Job, opts: &Options, progress: &mut dyn FnMut(&Event)) -> Result<Outcome, String> {
    let Job { dir, base_id, train, held_out, out, label } = job;
    if opts.epochs == 0 || opts.batch == 0 {
        return Err("minicpmv finetune: epochs and batch must be at least 1".into());
    }
    let cfg = MiniCpmVConfig::from_dir(dir)?;
    let mut image = ImageOptions::from_dir(dir, cfg.vision.patch, opts.downsample.unwrap_or(cfg.downsample))?;
    if let Some(n) = opts.max_slices {
        image.max_slices = n;
    }
    let b = Builder { fmt: Formatter::from_dir(dir)?, image, max_frames: opts.max_frames };
    let train_split = b.split(train)?;
    let eval_split = held_out.map(|p| b.split(p)).transpose()?;
    let longest = eval_split.as_ref().map_or(train_split.longest, |e| e.longest.max(train_split.longest));
    let block = match opts.block {
        Some(n) if (n as usize) < longest => return Err(format!("minicpmv finetune: --block {n} is shorter than the longest sample ({longest} tokens)")),
        Some(n) => n,
        None => longest as u32,
    };
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;

    let spec = TrainSpec {
        decoder: opts.decoder,
        connector: opts.connector,
        vision: opts.vision,
        block,
        downsample: b.image.downsample,
        lr: opts.rates(),
        weight_decay: opts.weight_decay,
        grad_clip: opts.grad_clip,
        seed: opts.seed,
    };
    let vision = crate::import::vision_weights(dir, &cfg)?;
    let decoder = qwen35::base_source::BaseCheckpoint { cfg: cfg.text.clone(), src: Box::new(crate::import::open_decoder(dir, &cfg)?) };
    let t = Trainer::new(gpu, cfg, &vision, decoder, spec)?;
    drop(vision);

    let n = train_split.turns.len();
    let batch = opts.batch as usize;
    let per_epoch = n.div_ceil(batch) as u32;
    let steps = per_epoch * opts.epochs;
    let schedule = LrSchedule {
        peak: 1.0,
        floor: opts.min_lr_ratio,
        warmup: opts.warmup.unwrap_or((steps / 20).max(1)),
        hold: 0,
        decay_iters: steps,
    };

    let state = out.join(STATE_FILE);
    let resume_from = opts.resume.then_some(state.as_path());
    let spec = model::session::SessionSpec {
        samples: n,
        batch,
        steps: 0,
        order: model::session::Order::Epochs { epochs: opts.epochs },
        cadence: model::session::Cadence::Steps(opts.checkpoint_every),
        seed: opts.seed,
    };
    let mut session = SessionTrainer { t, b: &b, split: &train_split };
    let mut report = |ev: model::session::Event| match ev {
        model::session::Event::Step { step, steps, epoch, loss, lr } => progress(&Event::Step { step, steps, epoch, loss, lr: lr * opts.lr }),
        model::session::Event::Checkpoint { step } => progress(&Event::Checkpoint { step }),
        model::session::Event::Eval { epoch, loss } => progress(&Event::Eval { epoch, loss }),
    };
    let outcome = match eval_split.as_ref() {
        Some(e) => model::session::run_evaluated(
            &mut session,
            &spec,
            &schedule,
            resume_from,
            &state,
            Some(|st: &mut SessionTrainer<'_>| evaluate(&mut st.t, &b, e)),
            || false,
            &mut report,
        )?,
        None => model::session::run(&mut session, &spec, &schedule, resume_from, &state, || false, &mut report)?,
    };
    session.t.save_tune(out, base_id, label)?;
    Ok(Outcome { turns: n, block, steps: outcome.steps, initial_eval: outcome.initial_eval, final_eval: outcome.final_eval })
}

/// Where the model store keeps a fine-tune labelled `owner:name:tag` of the
/// checkpoint in `model_dir`: the directory [`stored_tunes`] finds.
pub fn store_dir(model_dir: &Path, label: &str) -> Result<PathBuf, String> {
    let parts: Vec<&str> = label.split(':').collect();
    match parts.as_slice() {
        [owner, name, tag] if parts.iter().all(|p| !p.is_empty() && !p.contains(['/', '\\']) && *p != "." && *p != "..") => {
            Ok(model_dir.join("adapters").join(owner).join(name).join(tag))
        }
        _ => Err(format!("fine-tune label {label:?}: expected owner:name:tag, each a plain name")),
    }
}

/// A fine-tune kept beside a checkpoint: its `owner:name:tag` label and
/// directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredTune {
    pub label: String,
    pub dir: PathBuf,
}

/// The fine-tunes stored under `model_dir` (`adapters/<owner>/<name>/<tag>/`
/// holding a `tune.json`), ordered by label.
pub fn stored_tunes(model_dir: &Path) -> Vec<StoredTune> {
    let subdirs = |p: &Path| -> Vec<(String, PathBuf)> {
        let Ok(rd) = std::fs::read_dir(p) else { return Vec::new() };
        rd.flatten().filter(|e| e.path().is_dir()).filter_map(|e| Some((e.file_name().to_str()?.to_string(), e.path()))).collect()
    };
    let mut out = Vec::new();
    for (owner, o) in subdirs(&model_dir.join("adapters")) {
        for (name, nd) in subdirs(&o) {
            for (tag, dir) in subdirs(&nd) {
                if dir.join(crate::train::TUNE_MANIFEST).is_file() {
                    out.push(StoredTune { label: format!("{owner}:{name}:{tag}"), dir });
                }
            }
        }
    }
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_epoch_is_a_permutation_reproducible_from_the_seed() {
        let a = model::train::epoch_order(9, 50, 0);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_eq!(a, model::train::epoch_order(9, 50, 0));
        assert_ne!(a, model::train::epoch_order(9, 50, 1), "each epoch draws its own order");
    }

    #[test]
    fn tunes_are_stored_and_found_by_owner_name_and_tag() {
        let root = std::env::temp_dir().join(format!("brain-minicpmv-tunes-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        let dir = store_dir(&root, "me:colors:v1").unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(crate::train::TUNE_MANIFEST), "{}").unwrap();
        std::fs::create_dir_all(store_dir(&root, "me:empty:v1").unwrap()).unwrap();
        assert_eq!(stored_tunes(&root), vec![StoredTune { label: "me:colors:v1".into(), dir }]);
        for bad in ["me:colors", "me:../x:v1", "a:b:c:d", "me::v1"] {
            assert!(store_dir(&root, bad).is_err(), "{bad} accepted");
        }
        std::fs::remove_dir_all(&root).ok();
    }
}
