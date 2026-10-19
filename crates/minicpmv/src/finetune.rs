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

use data::rng::Rng;
use gpu_core::Gpu;
use model::train::LrSchedule;

use crate::config::{Downsample, MiniCpmVConfig};
use crate::dataset::{load_media, read_jsonl, samples, Record, Sample};
use crate::preprocess::ImageOptions;
use crate::prompt::Formatter;
use crate::train::{PartMode, TrainSpec, Trainer};

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
    pub lr: f32,
    /// The rate the cosine decays to; `None` for a tenth of `lr`.
    pub min_lr: Option<f32>,
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
            min_lr: None,
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

/// Progress reported to the caller.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
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
fn epoch_order(n: usize, seed: u64, epoch: u32) -> Vec<usize> {
    let mut rng = Rng::new(seed ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(epoch as u64 + 1)));
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
    order
}

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
        lr: opts.lr,
        weight_decay: opts.weight_decay,
        grad_clip: opts.grad_clip,
        seed: opts.seed,
    };
    let vision = crate::import::vision_weights(dir, &cfg)?;
    let decoder = qwen35::base_source::BaseCheckpoint { cfg: cfg.text.clone(), src: Box::new(crate::import::open_decoder(dir, &cfg)?) };
    let mut t = Trainer::new(gpu, cfg, &vision, decoder, spec)?;
    drop(vision);

    let n = train_split.turns.len();
    let batch = opts.batch as usize;
    let per_epoch = n.div_ceil(batch) as u32;
    let steps = per_epoch * opts.epochs;
    let schedule = LrSchedule {
        peak: opts.lr,
        floor: opts.min_lr.unwrap_or(opts.lr / 10.0),
        warmup: opts.warmup.unwrap_or((steps / 20).max(1)),
        hold: 0,
        decay_iters: steps,
    };

    let state = out.join(STATE_FILE);
    let mut position = 0u64;
    if opts.resume && state.is_file() {
        position = t.restore(&state)?;
        if position > (n * opts.epochs as usize) as u64 {
            return Err(format!("{}: position {position} is past this dataset's {} epochs of {n} turns", state.display(), opts.epochs));
        }
    }
    let initial_eval = match (&eval_split, position) {
        (Some(e), 0) => {
            let loss = evaluate(&mut t, &b, e)?;
            progress(&Event::Eval { epoch: 0, loss });
            Some(loss)
        }
        _ => None,
    };
    let mut final_eval = None;
    let start_epoch = (position / n as u64) as u32;
    for epoch in start_epoch..opts.epochs {
        let order = epoch_order(n, opts.seed, epoch);
        let mut i = (position - epoch as u64 * n as u64) as usize;
        while i < n {
            let chunk = &order[i..(i + batch).min(n)];
            let mut loss = 0.0;
            for &k in chunk {
                loss += t.accumulate(&b.sample(&train_split, train_split.turns[k])?)?;
            }
            let lr = schedule.at(t.steps);
            t.step(lr, chunk.len());
            i += chunk.len();
            position += chunk.len() as u64;
            progress(&Event::Step { step: t.steps, steps, epoch: epoch + 1, loss: loss / chunk.len() as f32, lr });
            if opts.checkpoint_every > 0 && t.steps % opts.checkpoint_every == 0 {
                checkpoint(&t, out, position)?;
                progress(&Event::Checkpoint { step: t.steps });
            }
        }
        checkpoint(&t, out, position)?;
        if let Some(e) = &eval_split {
            let loss = evaluate(&mut t, &b, e)?;
            progress(&Event::Eval { epoch: epoch + 1, loss });
            final_eval = Some(loss);
        }
    }
    t.save_tune(out, base_id, label)?;
    Ok(Outcome { turns: n, block, steps: t.steps, initial_eval, final_eval })
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
        let a = epoch_order(50, 9, 0);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_eq!(a, epoch_order(50, 9, 0));
        assert_ne!(a, epoch_order(50, 9, 1), "each epoch draws its own order");
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
