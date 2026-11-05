// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a chat record says about the rollout it came from, and how a prepared dataset keeps it.
//!
//! An orchestrator that collects agent rollouts outside brain writes one `generic-messages-v2` record per model call
//! (the assistant turn of the call is the action) and may add
//!
//! * `"advantage": <number>` on the record: the group-relative advantage of the rollout the call belongs to. It is
//!   the weight of every supervised token of the record in a policy-gradient run;
//! * `"behavior_logprobs": [<number>, ...]` on a supervised assistant message: the log-probability the sampler gave
//!   each token of that answer, one per token of the answer as the chat template renders it after the prompt (the
//!   answer's text and its closing end-of-turn marker). A value above zero is not a log-probability.
//!
//! * `"teacher_context": {"at": <index>, "messages": [{"role": "system"|"user", "content": "..."}, ...]}` on the
//!   record: messages the teacher of an on-policy self-distillation run is shown and the student never saw, inserted
//!   before the record's message `at` (at or before its first supervised message). The teacher's rendering of the
//!   conversation is the student's with those messages in it; its supervised tokens must be exactly the student's.
//!   A record may carry a teacher context instead of an advantage (the scalar advantage is then 0) or with one.
//!
//! A prepared dataset keeps both beside its token files, per split: `<split>.adv.bin` (`f32` per example, NaN for an
//! example whose record had no advantage) and `<split>.blp.bin` (`f32` per token of the stream, NaN where no
//! behaviour log-probability was recorded). A split with neither has no sidecar, and preparing one removes a stale
//! sidecar. The teacher's renderings are kept as `<split>.tch.u32.bin`, `<split>.tch.mask.bin` and `<split>.tch.ex.bin`
//! (an example without a teacher context holds none of its own: an empty range).

use std::io;
use std::path::{Path, PathBuf};

use crate::binio;
use crate::chat::ChatSample;

/// A log-probability can be at most zero; this much above it is rounding of the collector's own arithmetic.
const LOGPROB_SLACK: f32 = 1e-3;

/// The rollout data of one record: its advantage and, per message of the record, the behaviour log-probabilities of
/// that message's answer when the collector recorded them.
#[derive(Clone, Debug)]
pub struct Rollout {
    pub advantage: f32,
    /// One entry per message of the record; `Some` only on a supervised assistant message.
    pub behavior_logprobs: Vec<Option<Vec<f32>>>,
    /// What the teacher is shown instead of the record's own conversation, when the record carries a
    /// `"teacher_context"`.
    pub teacher: Option<Teacher>,
}

/// The teacher's view of a record: the record's conversation with the privileged messages inserted. It goes through
/// every transformation the student's sample goes through (one example per answer, packing, no-think), so the two stay
/// renderings of the same answers.
#[derive(Clone, Debug)]
pub struct Teacher {
    pub sample: Box<ChatSample>,
    /// How many messages the teacher's conversation holds beyond the student's, all before the first supervised one.
    pub inserted: usize,
}

impl Rollout {
    /// Whether any message of the record carries behaviour log-probabilities.
    pub fn has_behavior(&self) -> bool {
        self.behavior_logprobs.iter().any(Option::is_some)
    }

    /// The rollout of a sample derived from this record's (a prefix, one answer, a packing): the same advantage, with
    /// the derived sample's log-probabilities and teacher view.
    pub(crate) fn derived(&self, behavior_logprobs: Vec<Option<Vec<f32>>>, teacher: Option<Teacher>) -> Rollout {
        Rollout { advantage: self.advantage, behavior_logprobs, teacher }
    }
}

/// A record's advantage as the wire gives it: a finite number that fits `f32`.
pub(crate) fn check_advantage(advantage: &serde_json::Value) -> Result<f32, String> {
    let number = advantage.as_f64().ok_or_else(|| format!("\"advantage\" is {advantage}, which is not a number"))?;
    let narrowed = number as f32;
    if !number.is_finite() || !narrowed.is_finite() {
        return Err(format!("\"advantage\" is {number}, which is not a finite number"));
    }
    Ok(narrowed)
}

/// One message's behaviour log-probabilities: a non-empty list of finite numbers, none above zero.
pub(crate) fn check_behavior_logprobs(values: &serde_json::Value) -> Result<Vec<f32>, String> {
    let list = values.as_array().ok_or_else(|| "\"behavior_logprobs\" is not a list of numbers".to_string())?;
    if list.is_empty() {
        return Err("\"behavior_logprobs\" is empty; omit it when the sampler did not record any".to_string());
    }
    list.iter()
        .enumerate()
        .map(|(k, v)| {
            let number = v.as_f64().ok_or_else(|| format!("\"behavior_logprobs\"[{k}] is {v}, which is not a number"))?;
            let narrowed = number as f32;
            if !number.is_finite() || !narrowed.is_finite() {
                Err(format!("\"behavior_logprobs\"[{k}] is {number}, which is not a finite number"))
            } else if narrowed > LOGPROB_SLACK {
                Err(format!("\"behavior_logprobs\"[{k}] is {number}; a log-probability is at most 0"))
            } else {
                Ok(narrowed.min(0.0))
            }
        })
        .collect()
}

fn names(dir: &Path, split: &str) -> (PathBuf, PathBuf) {
    (dir.join(format!("{split}.adv.bin")), dir.join(format!("{split}.blp.bin")))
}

/// Write a split's sidecars: `advantage[i]` is example `i`'s (NaN for none), `behavior[t]` token `t`'s (NaN for none).
/// Nothing is written, and a stale sidecar is removed, when no example has an advantage.
pub fn write(dir: &Path, split: &str, advantage: &[f32], behavior: &[f32]) -> io::Result<()> {
    let (adv, blp) = names(dir, split);
    for stale in [&adv, &blp] {
        match std::fs::remove_file(stale) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    if advantage.iter().all(|a| a.is_nan()) {
        return Ok(());
    }
    binio::write_f32_bin(&adv, advantage)?;
    if behavior.iter().any(|b| !b.is_nan()) {
        binio::write_f32_bin(&blp, behavior)?;
    }
    Ok(())
}

/// The rollout data of one split of a prepared dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitRollouts {
    /// Per example; NaN where the record had no advantage (an imitation example).
    pub advantage: Vec<f32>,
    /// Per token of the split's stream, aligned with it; NaN where none was recorded.
    pub behavior: Vec<f32>,
}

/// The sidecars of `split` in `dir`, checked against the `examples` and `tokens` the split holds; `None` when the
/// split has no rollout data.
pub fn read(dir: &Path, split: &str, examples: usize, tokens: usize) -> io::Result<Option<SplitRollouts>> {
    let (adv, blp) = names(dir, split);
    if !adv.exists() {
        return Ok(None);
    }
    let advantage = binio::read_f32_bin(&adv)?;
    if advantage.len() != examples {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: advantages for {} examples, the split holds {examples}", adv.display(), advantage.len())));
    }
    let behavior = if blp.exists() {
        let behavior = binio::read_f32_bin(&blp)?;
        if behavior.len() != tokens {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: log-probabilities for {} tokens, the split holds {tokens}", blp.display(), behavior.len())));
        }
        behavior
    } else {
        vec![f32::NAN; tokens]
    };
    Ok(Some(SplitRollouts { advantage, behavior }))
}

/// The teacher renderings of a split: every example's tokens and supervision mask, back to back.
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherStream {
    pub ids: Vec<u32>,
    pub mask: Vec<bool>,
    /// Where each example's rendering starts; an example without a teacher context holds an empty range.
    pub starts: Vec<usize>,
}

impl TeacherStream {
    /// The teacher's tokens and mask of example `e`; `None` for an example without a teacher context.
    pub fn example(&self, e: usize) -> Option<(&[u32], &[bool])> {
        let (a, b) = (self.starts[e], self.starts.get(e + 1).copied().unwrap_or(self.ids.len()));
        (a < b).then(|| (&self.ids[a..b], &self.mask[a..b]))
    }
}

fn teacher_names(dir: &Path, split: &str) -> [PathBuf; 3] {
    ["tch.u32.bin", "tch.mask.bin", "tch.ex.bin"].map(|suffix| dir.join(format!("{split}.{suffix}")))
}

/// Write a split's teacher renderings (`None` for an example without one). Nothing is written, and a stale sidecar is
/// removed, when no example has a teacher.
pub fn write_teacher(dir: &Path, split: &str, teachers: &[Option<(Vec<u32>, Vec<bool>)>]) -> io::Result<()> {
    let [ids_file, mask_file, starts_file] = teacher_names(dir, split);
    for stale in [&ids_file, &mask_file, &starts_file] {
        match std::fs::remove_file(stale) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    if teachers.iter().all(Option::is_none) {
        return Ok(());
    }
    let (mut ids, mut mask, mut starts) = (Vec::new(), Vec::new(), Vec::new());
    for teacher in teachers {
        starts.push(ids.len() as u64);
        if let Some((t_ids, t_mask)) = teacher {
            ids.extend_from_slice(t_ids);
            mask.extend_from_slice(t_mask);
        }
    }
    binio::write_u32_bin(&ids_file, &ids)?;
    binio::write_mask_bin(&mask_file, &mask)?;
    binio::write_u64_bin(&starts_file, &starts)
}

/// The teacher renderings of `split` in `dir`, checked against the `examples` the split holds; `None` when no example
/// has a teacher.
pub fn read_teacher(dir: &Path, split: &str, examples: usize) -> io::Result<Option<TeacherStream>> {
    let [ids_file, mask_file, starts_file] = teacher_names(dir, split);
    if !ids_file.exists() {
        return Ok(None);
    }
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    let ids = binio::read_u32_bin(&ids_file)?;
    let mask = binio::read_mask_bin(&mask_file)?;
    let starts: Vec<usize> = binio::read_u64_bin(&starts_file)?.into_iter().map(|s| s as usize).collect();
    if mask.len() != ids.len() {
        return Err(bad(format!("{}: {} supervision flags for {} teacher tokens", mask_file.display(), mask.len(), ids.len())));
    }
    if starts.len() != examples {
        return Err(bad(format!("{}: teacher renderings for {} examples, the split holds {examples}", starts_file.display(), starts.len())));
    }
    if starts.windows(2).any(|w| w[0] > w[1]) || starts.last().is_some_and(|&s| s > ids.len()) {
        return Err(bad(format!("{}: example starts are not ordered within the stream", starts_file.display())));
    }
    Ok(Some(TeacherStream { ids, mask, starts }))
}
