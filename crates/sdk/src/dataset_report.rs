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
use data::chat_rollout::RecordError;
pub use data::chat_rollout::RecordRule;
use data::feedback::FeedbackSample;
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
    /// `generic-feedback-v1` judged completions (KTO); each is one example.
    Feedback,
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
    /// Which records to list the lengths (or refusals) of in [`DatasetReport::record_lengths`] and
    /// [`DatasetReport::record_refusals`]; `None` lists none.
    pub list_records: Option<RecordWindow>,
    /// [`crate::PolicyGradientConfig::truncated`]: whether a truncated rollout counts in its group when the rules
    /// between records are checked; chat records only.
    pub truncated: data::chat_rollout::TruncatedRollouts,
}

/// A page of the per-record answers: the records (blank lines are not records) after the first `offset` of them, at
/// most `limit`, each listed as a length or as the refusal it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordWindow {
    /// Records to skip, valid or refused.
    pub offset: usize,
    /// Records to list at most, lengths and refusals together (the caller bounds it; see [`MAX_RECORD_WINDOW`]).
    pub limit: usize,
}

/// The most records one page of [`DatasetReport::record_lengths`] and [`DatasetReport::record_refusals`] can hold.
pub const MAX_RECORD_WINDOW: usize = 200_000;

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
    /// The rule of the training-record contract the record broke, when that is why it is refused (a producer acts on
    /// it without reading `error`); `None` for every other refusal.
    pub rule: Option<RecordRule>,
    /// The record field that broke the rule, as a JSON path (`messages[1].behavior_logprobs`), with `rule`.
    pub field: Option<String>,
}

/// The length of one valid record, by its 1-based line in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordLength {
    /// The line the record is on.
    pub line: usize,
    /// Tokens of its longest example, as [`crate::ExampleMeter::tokens`] counts: what the training row has to hold.
    pub tokens: usize,
    /// Of the example that sets `tokens` (the one needing the longest row): the tokens that are not supervised, the
    /// context and the template's markup the model reads but is not trained to write.
    pub prompt: usize,
    /// Of that example: the supervised (loss-bearing) tokens, the answers the model is trained to write. With
    /// `prompt` they are that example's own rendering, which is `tokens` unless a teacher's rendering is longer.
    pub trained: usize,
    /// Examples the record is trained as.
    pub examples: usize,
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
    /// The valid records of the page [`DatasetCheck::list_records`] asked for, in file order.
    pub record_lengths: Vec<RecordLength>,
    /// The refused records of that page, in file order, each with the whole of its refusal (not capped at
    /// [`MAX_LISTED`] like `invalid`).
    pub record_refusals: Vec<Refusal>,
    /// Where the next page starts (its `offset`) when records remain after this one.
    pub record_lengths_next: Option<usize>,
}

impl DatasetReport {
    /// Whether the record at `record_index` (0-based among all records) belongs on the page `window` asks for; the
    /// first record past a full page is where the next one starts.
    fn admit_to_page(&mut self, window: RecordWindow, record_index: usize) -> bool {
        if record_index < window.offset {
            return false;
        }
        if self.record_lengths.len() + self.record_refusals.len() < window.limit {
            return true;
        }
        self.record_lengths_next.get_or_insert(record_index);
        false
    }
}

/// Dry-runs the dataset `path` against the checkpoint in `model_dir`. Nothing is loaded but the tokenizer and the
/// chat template.
///
/// `keep_going` is asked before each record; once it says no the check is
/// [`Error::Cancelled`].
///
/// # Errors
/// The checkpoint's tokenizer or template cannot be read, or the file cannot be, or `keep_going` said no.
pub fn check_dataset(model_dir: impl AsRef<Path>, path: impl AsRef<Path>, check: &DatasetCheck, keep_going: &mut dyn FnMut() -> bool) -> Result<DatasetReport> {
    let (model_dir, path) = (model_dir.as_ref(), path.as_ref());
    let (tok, tmpl) = crate::finetune::tokenizer_and_template(model_dir)?;
    let arch = crate::study::arch_of_checkpoint(model_dir).map_or("qwen3", |a| a.id);
    let images = crate::finetune::image_reader(arch, model_dir)?;
    let no_think = if check.thinking { None } else { tmpl.no_think_block() };
    let render = RenderOpts { keep_reasoning: no_think.is_some() };
    let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: check.pack_answers, images: &*images };
    let open = || std::fs::File::open(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())));
    let records_of = |file: std::fs::File| std::io::BufReader::new(file).lines().enumerate().map(move |(index, line)| line.map(|l| (index, l)).map_err(|e| Error::Backend(format!("{}: line {}: {e}", path.display(), index + 1))));
    // The contract's rules between records (a rollout's records agree, a group holds two rollouts, one tokenizer)
    // need the whole file: a first pass reads each record's rollout, the second refuses a record that breaks one.
    let mut between: std::collections::HashMap<usize, RecordError> = std::collections::HashMap::new();
    if check.kind == DatasetKind::Chat {
        let mut rollouts = Vec::new();
        for record in records_of(open()?) {
            let (_, line) = record?;
            if line.trim().is_empty() {
                continue;
            }
            if !keep_going() {
                return Err(Error::Cancelled);
            }
            let rollout = ChatSample::from_json_line_with(line.trim(), &*images).ok().and_then(|r| r.rollout);
            // Only the outcome is read between records; the per-token data is not kept.
            rollouts.push(rollout.map(|r| data::chat_rollout::Rollout { behavior_logprobs: Vec::new(), behavior_top: Vec::new(), token_ids: Vec::new(), teacher: None, ..r }));
        }
        let refs: Vec<Option<&data::chat_rollout::Rollout>> = rollouts.iter().map(Option::as_ref).collect();
        between.extend(data::chat_rollout::check_records(&refs, check.truncated, tok.digest()));
    }

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
        record_lengths: Vec::new(),
        record_refusals: Vec::new(),
        record_lengths_next: None,
    };
    let (mut lengths, mut supervised) = (Vec::new(), Vec::new());
    for record in records_of(open()?) {
        let (index, line) = record?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !keep_going() {
            return Err(Error::Cancelled);
        }
        let record_index = report.records;
        report.records += 1;
        // Each example of the record as (tokens, supervised tokens, the row it needs: a teacher's rendering is scored
        // in a row of its own, so the row has to hold it too).
        let examples: std::result::Result<Vec<(usize, usize, usize)>, RecordError> = match check.kind {
            DatasetKind::Chat => ChatSample::from_json_line_with(line, &*images).and_then(|record| {
                let mut out = Vec::new();
                // A baseline-only record trains as no example; it is checked between records, in its group.
                let baseline = record.rollout.as_ref().is_some_and(|r| r.baseline_only);
                for example in asked.examples_of(&record)?.into_iter().filter(|_| !baseline) {
                    let encoded = example.encode_rollout(&tok, &tmpl, render)?;
                    let row = encoded.ids.len().max(encoded.teacher.as_ref().map_or(0, |(ids, _)| ids.len()));
                    out.push((encoded.ids.len(), encoded.mask.iter().filter(|m| **m).count(), row));
                }
                Ok(out)
            }),
            DatasetKind::Feedback => FeedbackSample::from_json_line(line).map_err(RecordError::from).and_then(|example| {
                let e = example.encode_with(&tok, &tmpl, render)?;
                Ok(vec![(e.ids.len(), e.mask.iter().filter(|m| **m).count(), e.ids.len())])
            }),
            DatasetKind::Preference => PreferenceSample::from_json_line(line).map_err(RecordError::from).and_then(|pair| {
                let (chosen, rejected) = pair.encode_with(&tok, &tmpl, render)?;
                Ok(vec![(chosen.ids.len(), chosen.mask.iter().filter(|m| **m).count(), chosen.ids.len()), (rejected.ids.len(), rejected.mask.iter().filter(|m| **m).count(), rejected.ids.len())])
            }),
        };
        // A record that breaks a rule of its own is refused for that; one that keeps its own rules may still break one
        // between records.
        let examples = examples.and_then(|examples| between.remove(&record_index).map_or(Ok(examples), Err));
        match examples {
            Ok(examples) => {
                report.valid_records += 1;
                let mut over = false;
                // The example needing the longest row: (row, tokens, supervised tokens).
                let mut longest = (0, 0, 0);
                let example_count = examples.len();
                for (tokens, supervised_tokens, row) in examples {
                    if row > longest.0 {
                        longest = (row, tokens, supervised_tokens);
                    }
                    report.examples += 1;
                    lengths.push(tokens as u64);
                    supervised.push(supervised_tokens as u64);
                    report.examples_without_supervision += usize::from(supervised_tokens == 0);
                    if row > check.max_block as usize {
                        report.over_max_block += 1;
                        over = true;
                    }
                }
                if over && report.over_max_block_lines.len() < MAX_LISTED {
                    report.over_max_block_lines.push(index + 1);
                }
                let listed = RecordLength { line: index + 1, tokens: longest.0, prompt: longest.1 - longest.2, trained: longest.2, examples: example_count };
                if let Some(window) = check.list_records {
                    if report.admit_to_page(window, record_index) {
                        report.record_lengths.push(listed);
                    }
                }
            }
            Err(error) => {
                report.invalid_total += 1;
                let (rule, field) = error.fault().map_or((None, None), |f| (Some(f.rule), Some(f.field.clone())));
                let refusal = Refusal { line: index + 1, error: error.to_string(), rule, field };
                if let Some(window) = check.list_records {
                    if report.admit_to_page(window, record_index) {
                        report.record_refusals.push(refusal.clone());
                    }
                }
                if report.invalid.len() < MAX_LISTED {
                    report.invalid.push(refusal);
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
    use super::*;

    fn fixtures() -> std::path::PathBuf {
        std::path::PathBuf::from(brain_testutil::repo_path("crates/data/tests/fixtures/training-records/v2"))
    }

    /// What the trainer reads the record on `line` as: its own dataset reader ([`Asked::read`], which
    /// [`crate::ChatFineTune`] trains from) over a file of that one record, each example encoded as training encodes
    /// it. Returns the longest example as (tokens, supervised tokens) and the number of examples.
    fn as_the_trainer_reads(model: &Path, line: &str, scratch: &Path, check: &DatasetCheck) -> ((usize, usize), usize) {
        let (tok, tmpl) = crate::finetune::tokenizer_and_template(model).unwrap();
        let images = crate::finetune::image_reader("qwen3", model).unwrap();
        let no_think = if check.thinking { None } else { tmpl.no_think_block() };
        let render = RenderOpts { keep_reasoning: no_think.is_some() };
        let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: check.pack_answers, images: &*images };
        std::fs::write(scratch, format!("{line}\n")).unwrap();
        let examples = asked.read(scratch).unwrap();
        let encoded: Vec<(usize, usize)> = examples.iter().map(|e| e.encode_with(&tok, &tmpl, render).unwrap()).map(|(ids, mask)| (ids.len(), mask.iter().filter(|m| **m).count())).collect();
        (*encoded.iter().max_by_key(|(tokens, _)| *tokens).unwrap(), encoded.len())
    }

    /// The counts listed per record are the trainer's own for every valid record of the contract's fixtures: the
    /// prompt and the trained tokens split what the trainer encodes, and the refused records are listed with their rule.
    #[test]
    fn per_record_counts_are_what_the_trainer_reads_for_the_contract_fixtures() {
        let root = fixtures();
        let model = root.join("model");
        let scratch = tempfile::tempdir().unwrap();
        let check = DatasetCheck { kind: DatasetKind::Chat, thinking: true, pack_answers: false, max_block: 512, list_records: Some(RecordWindow { offset: 0, limit: 100 }), truncated: Default::default() };
        let mut listed = 0;
        for file in ["chat.jsonl", "chat_rollout.jsonl"] {
            let report = check_dataset(&model, root.join(file), &check, &mut || true).unwrap();
            let text = std::fs::read_to_string(root.join(file)).unwrap();
            assert_eq!((report.record_lengths.len(), report.record_refusals.len()), (report.records, 0), "{file}");
            for listed_record in &report.record_lengths {
                let line = text.lines().nth(listed_record.line - 1).unwrap();
                let ((tokens, supervised), examples) = as_the_trainer_reads(&model, line, &scratch.path().join("record.jsonl"), &check);
                assert_eq!((listed_record.prompt + listed_record.trained, listed_record.trained, listed_record.examples), (tokens, supervised, examples), "{file}:{}", listed_record.line);
                assert!(listed_record.trained > 0 && listed_record.prompt > 0, "{file}:{}: {listed_record:?}", listed_record.line);
                listed += 1;
            }
        }
        assert_eq!(listed, 10);

        // Line 1 breaks the count; line 2, its group's other rollout, is valid.
        let report = check_dataset(&model, root.join("invalid/behavior_logprob_count.jsonl"), &check, &mut || true).unwrap();
        assert_eq!(report.record_lengths.iter().map(|r| r.line).collect::<Vec<_>>(), [2]);
        let refused = &report.record_refusals[0];
        assert_eq!((refused.line, refused.rule, refused.field.as_deref()), (1, Some(RecordRule::BehaviorLogprobCount), Some("messages[1].behavior_logprobs")));
    }

    /// A page is a window over the records, valid or refused: pages tile the file and the last has no next.
    #[test]
    fn pages_tile_the_records_whether_valid_or_refused() {
        let root = fixtures();
        let scratch = tempfile::tempdir().unwrap();
        let mixed = scratch.path().join("mixed.jsonl");
        let valid = std::fs::read_to_string(root.join("chat.jsonl")).unwrap();
        let refused = std::fs::read_to_string(root.join("invalid/missing_reward.jsonl")).unwrap();
        let lines: Vec<&str> = valid.lines().chain(["not json"]).chain(refused.lines()).collect();
        std::fs::write(&mixed, lines.join("\n\n")).unwrap();
        let (mut seen, mut offset) = (Vec::new(), 0);
        loop {
            let check = DatasetCheck { kind: DatasetKind::Chat, thinking: true, pack_answers: false, max_block: 512, list_records: Some(RecordWindow { offset, limit: 2 }), truncated: Default::default() };
            let page = check_dataset(root.join("model"), &mixed, &check, &mut || true).unwrap();
            assert!(page.record_lengths.len() + page.record_refusals.len() <= 2);
            seen.extend(page.record_lengths.iter().map(|r| r.line));
            seen.extend(page.record_refusals.iter().map(|r| r.line));
            match page.record_lengths_next {
                Some(next) => offset = next,
                None => break,
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, (1..=lines.len()).map(|n| 2 * n - 1).collect::<Vec<_>>(), "every record once, by its line (blank lines between)");
    }


    #[test]
    fn a_distribution_reports_nearest_rank_percentiles() {
        let d = Distribution::of((1..=100).collect());
        assert_eq!((d.total, d.min, d.max, d.p50, d.p90, d.p99), (5050, 1, 100, 50, 90, 99));
        assert!((d.mean - 50.5).abs() < 1e-9);
        assert_eq!(Distribution::of(vec![7]).p99, 7);
        assert_eq!(Distribution::of(Vec::new()), Distribution::default());
    }
}
