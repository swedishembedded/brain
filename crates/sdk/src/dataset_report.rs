// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data pipelines that size and check every
// record against the model that will train on it before a card is claimed, for its
// clients. If your team needs expertise in dataset preparation for fine-tuning, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A dry run of a training dataset against a base: how many records parse, how long each example is in the base's own
//! tokens, how many tokens are supervised, which examples are longer than the training row, and which records are
//! refused and why, without loading weights or claiming a device.
//!
//! [`crate::validate_chat_dataset_for`] answers "will training accept this" with the first refusal; this answers
//! "what is in it" for the whole file, with every refusal listed (the first [`MAX_LISTED`]) so a producer fixes them
//! in one pass. Examples are read as the trainer reads them (the same tokenizer, chat template, per-answer or packed
//! rendering and mask), so a length here is the length a run would have to hold.

use std::io::BufRead;
use std::path::Path;

use data::chat::{ChatSample, RenderOpts};
use data::preference::PreferenceSample;

use crate::finetune::Asked;
use crate::{Error, Result};

/// The most invalid records and over-long examples a report names; the totals count them all.
pub const MAX_LISTED: usize = 20;

/// Which dataset format is checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetKind {
    /// `generic-messages-v2` chat records (supervised, policy-gradient and full fine-tuning).
    Chat,
    /// `generic-preference-v1` pairs (preference optimization); each candidate of a pair is an example.
    Preference,
}

/// How the trainer that will read the dataset is set up.
#[derive(Clone, Copy, Debug)]
pub struct DatasetCheck {
    pub kind: DatasetKind,
    /// [`crate::ChatFineTune::thinking`]; chat records only.
    pub thinking: bool,
    /// [`crate::ChatFineTune::pack_answers`]; chat records only.
    pub pack_answers: bool,
    /// The training row: an example longer than this would be refused by the trainer.
    pub max_block: u32,
}

/// The distribution of one per-example count.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Distribution {
    pub total: u64,
    pub min: u64,
    pub mean: f64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub max: u64,
}

impl Distribution {
    fn of(mut values: Vec<u64>) -> Distribution {
        if values.is_empty() {
            return Distribution::default();
        }
        values.sort_unstable();
        let total: u64 = values.iter().sum();
        // Nearest-rank percentile: the smallest value at least `q` of the examples do not exceed.
        let at = |q: f64| values[((q * values.len() as f64).ceil() as usize).clamp(1, values.len()) - 1];
        Distribution { total, min: values[0], mean: total as f64 / values.len() as f64, p50: at(0.5), p90: at(0.9), p99: at(0.99), max: values[values.len() - 1] }
    }
}

/// A record the trainer would refuse, by its 1-based line in the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub line: usize,
    pub error: String,
}

/// What [`check_dataset`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetReport {
    /// Non-blank lines.
    pub records: usize,
    /// Records that parse and encode.
    pub valid_records: usize,
    /// Examples the valid records are trained as (a record of several answers can be several).
    pub examples: usize,
    /// Tokens per example, in the base's tokenizer and template.
    pub tokens: Distribution,
    /// Supervised (loss-bearing) tokens per example.
    pub supervised_tokens: Distribution,
    /// Examples with no supervised token at all (they train on nothing).
    pub examples_without_supervision: usize,
    /// Examples longer than `max_block`: the trainer refuses them rather than cutting them.
    pub over_max_block: usize,
    /// The lines of the first [`MAX_LISTED`] records holding such an example.
    pub over_max_block_lines: Vec<usize>,
    /// Records the trainer refuses (the first [`MAX_LISTED`]; see `invalid_total`).
    pub invalid: Vec<Refusal>,
    pub invalid_total: usize,
}

/// Dry-runs the dataset `path` against the checkpoint in `model_dir`. Nothing is loaded but the tokenizer and the
/// chat template.
///
/// # Errors
/// The checkpoint's tokenizer or template cannot be read, or the file cannot be.
pub fn check_dataset(model_dir: impl AsRef<Path>, path: impl AsRef<Path>, check: &DatasetCheck) -> Result<DatasetReport> {
    let (model_dir, path) = (model_dir.as_ref(), path.as_ref());
    let (tok, tmpl) = crate::finetune::tokenizer_and_template(model_dir)?;
    let arch = crate::study::arch_of_checkpoint(model_dir).map_or("qwen3", |a| a.id);
    let images = crate::finetune::image_reader(arch, model_dir)?;
    let no_think = if check.thinking { None } else { tmpl.no_think_block() };
    let render = RenderOpts { keep_reasoning: no_think.is_some() };
    let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: check.pack_answers, images: &*images };
    let file = std::fs::File::open(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;

    let mut report = DatasetReport {
        records: 0,
        valid_records: 0,
        examples: 0,
        tokens: Distribution::default(),
        supervised_tokens: Distribution::default(),
        examples_without_supervision: 0,
        over_max_block: 0,
        over_max_block_lines: Vec::new(),
        invalid: Vec::new(),
        invalid_total: 0,
    };
    let (mut lengths, mut supervised) = (Vec::new(), Vec::new());
    for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| Error::Backend(format!("{}: line {}: {e}", path.display(), index + 1)))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        report.records += 1;
        // Each example of the record as (tokens, supervised tokens).
        let examples: std::result::Result<Vec<(usize, usize)>, String> = match check.kind {
            DatasetKind::Chat => ChatSample::from_json_line_with(line, &*images).and_then(|record| {
                let mut out = Vec::new();
                for example in asked.examples_of(&record)? {
                    let encoded = example.encode_rollout(&tok, &tmpl, render).map_err(|e| e.to_string())?;
                    out.push((encoded.ids.len(), encoded.mask.iter().filter(|m| **m).count()));
                }
                Ok(out)
            }),
            DatasetKind::Preference => PreferenceSample::from_json_line(line).and_then(|pair| {
                let (chosen, rejected) = pair.encode_with(&tok, &tmpl, render).map_err(|e| e.to_string())?;
                Ok(vec![(chosen.ids.len(), chosen.mask.iter().filter(|m| **m).count()), (rejected.ids.len(), rejected.mask.iter().filter(|m| **m).count())])
            }),
        };
        match examples {
            Ok(examples) => {
                report.valid_records += 1;
                let mut over = false;
                for (tokens, supervised_tokens) in examples {
                    report.examples += 1;
                    lengths.push(tokens as u64);
                    supervised.push(supervised_tokens as u64);
                    report.examples_without_supervision += usize::from(supervised_tokens == 0);
                    if tokens > check.max_block as usize {
                        report.over_max_block += 1;
                        over = true;
                    }
                }
                if over && report.over_max_block_lines.len() < MAX_LISTED {
                    report.over_max_block_lines.push(index + 1);
                }
            }
            Err(error) => {
                report.invalid_total += 1;
                if report.invalid.len() < MAX_LISTED {
                    report.invalid.push(Refusal { line: index + 1, error });
                }
            }
        }
    }
    report.tokens = Distribution::of(lengths);
    report.supervised_tokens = Distribution::of(supervised);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::Distribution;

    #[test]
    fn a_distribution_reports_nearest_rank_percentiles() {
        let d = Distribution::of((1..=100).collect());
        assert_eq!((d.total, d.min, d.max, d.p50, d.p90, d.p99), (5050, 1, 100, 50, 90, 99));
        assert!((d.mean - 50.5).abs() < 1e-9);
        assert_eq!(Distribution::of(vec![7]).p99, 7);
        assert_eq!(Distribution::of(Vec::new()), Distribution::default());
    }
}
