// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data pipelines that check a dataset against the
// model that will train on it before any accelerator is claimed, for its clients. If your
// team needs expertise in preparing and validating fine-tuning data, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The `check_dataset` action: a dry run of a training dataset against a base, without training.
//!
//! The dataset (an input blob or an artifact digest) is read with the base's own tokenizer and chat template, as the
//! trainer reads it ([`brain::dataset_report::check_dataset`]), and the answer is the report: records, examples,
//! tokens and supervised tokens per example, examples longer than the training row, and every refused record by line.
//! It loads no weights and claims no card; it is host work, scheduled and bounded like the other jobs of the service.

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::job::{self, bring, Workdir, JobEnv};
use crate::request::Source;
use crate::store::Digest;

/// The action's name.
pub const ACTION: &str = "check_dataset";

/// What `check_dataset` is.
pub fn spec() -> ActionSpec {
    ActionSpec::new(ACTION, "dry-run a training dataset against a base's tokenizer and chat template: records, examples, tokens and supervised tokens per example, examples longer than the training row, every refused record by line; nothing is trained")
        .training()
        .param(ParamSpec::new("base", ParamType::Str, "the base model whose tokenizer and chat template read the dataset, `<vendor>/<repo>` in the server's model store (needed for `chat` and `preference`; a `timeline` dataset has no base)"))
        .param(ParamSpec::new("kind", ParamType::Enum(vec!["chat".into(), "preference".into(), "timeline".into()]), "`chat` (`generic-messages-v2` records, for sft, policy_gradient and full_train), `preference` (`generic-preference-v1` pairs, for dpo) or `timeline` (`timeline-v1` subjects, for timeline_train: each line validated, no tokens counted)").default(json!("chat")))
        .param(ParamSpec::new("dataset_digest", ParamType::Str, "the dataset, as a blake3 digest of the server's artifact store (or attach it as the input blob `dataset`; exactly one of the two)"))
        .param(ParamSpec::new("thinking", ParamType::Bool, "render as a run with this `thinking` setting would (chat only)").default(json!(true)))
        .param(ParamSpec::new("pack_answers", ParamType::Bool, "render as a run with this `pack_answers` setting would (chat only)").default(json!(false)))
        .param(ParamSpec::new("max_block", ParamType::Int, "the training row in tokens: examples longer than it are counted (a run refuses them)").default(json!(8192)).min(64.0).max(1_048_576.0).step(1.0))
        .param(ParamSpec::new("per_record", ParamType::Bool, "also list the length of each valid record (its line and the tokens of its longest example: what the row has to hold), one page at a time").default(json!(false)))
        .param(ParamSpec::new("offset", ParamType::Int, "with `per_record`: the valid records to skip, the first page is 0, then `per_record_next` of the previous page").default(json!(0)).min(0.0).max(1.0e12).step(1.0))
        .param(ParamSpec::new("limit", ParamType::Int, "with `per_record`: the most records one page lists").default(json!(50_000)).min(1.0).max(brain::dataset_report::MAX_RECORD_WINDOW as f64).step(1.0))
        .input(BlobSpec::new("dataset", Media::Bytes, "the dataset (JSONL), instead of `dataset_digest`"))
}

/// The settings of the action, exactly the params its spec declares.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub base: Option<String>,
    pub kind: String,
    pub dataset_digest: Option<String>,
    pub thinking: bool,
    pub pack_answers: bool,
    pub max_block: u32,
    pub per_record: bool,
    pub offset: usize,
    pub limit: usize,
}

/// A request that passed the spec and every check that needs no file and no device.
#[derive(Debug)]
pub struct Request {
    pub params: Params,
    pub dataset: Source,
}

impl Request {
    /// Reads the validated invocation `inv`: the dataset is the input blob `dataset` or `dataset_digest`, exactly one.
    pub fn from_invocation(inv: &Invocation) -> Result<Request, String> {
        let params: Params = serde_json::from_value(inv.params.clone()).map_err(|e| format!("check_dataset: {e}"))?;
        let dataset = match (&params.dataset_digest, inv.blobs.contains_key("dataset")) {
            (Some(digest), false) => Source::Digest(Digest::parse(digest).map_err(|e| format!("check_dataset: dataset_digest: {e}"))?),
            (None, true) => Source::Blob("dataset".to_string()),
            _ => return Err("check_dataset: name the dataset exactly once: as the input blob `dataset` or as `dataset_digest`".to_string()),
        };
        if let Some(extra) = inv.blobs.keys().find(|k| k.as_str() != "dataset") {
            return Err(format!("check_dataset: the input blob \"{extra}\" is not an input of this action"));
        }
        Ok(Request { params, dataset })
    }
}

fn distribution(d: &brain::dataset_report::Distribution) -> Value {
    json!({"total": d.total, "min": d.min, "mean": d.mean, "p50": d.p50, "p90": d.p90, "p99": d.p99, "max": d.max})
}

/// The report as the action returns it.
pub fn report_json(report: &brain::dataset_report::DatasetReport, kind: &str, max_block: u32) -> Value {
    let mut out = json!({
        "kind": kind,
        "records": report.records,
        "valid_records": report.valid_records,
        "examples": report.examples,
        "tokens": distribution(&report.tokens),
        "supervised_tokens": distribution(&report.supervised_tokens),
        "examples_without_supervision": report.examples_without_supervision,
        "max_block": max_block,
        "over_max_block": report.over_max_block,
        "over_max_block_lines": report.over_max_block_lines,
        "invalid_total": report.invalid_total,
        "invalid": report.invalid.iter().map(|r| json!({"line": r.line, "error": r.error})).collect::<Vec<_>>(),
        "listed_at_most": brain::dataset_report::MAX_LISTED,
    });
    // One `[line, tokens]` pair per record of the page: compact, because a dataset has many records.
    if !report.record_lengths.is_empty() || report.record_lengths_next.is_some() {
        out["per_record"] = report.record_lengths.iter().map(|r| json!([r.line, r.tokens])).collect();
        out["per_record_next"] = json!(report.record_lengths_next);
    }
    out
}

/// Runs the dry run `inv` describes. It is host work and takes no card.
pub fn run(env: &JobEnv, inv: &Invocation) -> ActionResult {
    let request = Request::from_invocation(inv)?;
    let p = &request.params;
    let work = Workdir::create(env)?;
    let dataset = work.path().join("dataset.jsonl");
    bring(&request.dataset, &inv.blobs, env.store.as_ref(), work.staging(), &dataset)?;
    if inv.cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    if p.kind == "timeline" {
        let text = std::fs::read_to_string(&dataset).map_err(|e| format!("check_dataset: the dataset: {e}"))?;
        let mut outcome = Outcome::new();
        outcome.outputs = timeline_report(&text, p.max_block);
        return Ok(outcome);
    }
    let base = p.base.as_deref().ok_or("check_dataset: `base` is needed to read a chat or preference dataset")?;
    let base_dir = job::resolve_base(env.models_dir.as_deref(), base)?;
    let kind = if p.kind == "preference" { brain::dataset_report::DatasetKind::Preference } else { brain::dataset_report::DatasetKind::Chat };
    let check = brain::dataset_report::DatasetCheck { kind, thinking: p.thinking, pack_answers: p.pack_answers, max_block: p.max_block, list_records: p.per_record.then_some(brain::dataset_report::RecordWindow { offset: p.offset, limit: p.limit }) };
    let report = brain::dataset_report::check_dataset(&base_dir, &dataset, &check).map_err(job::trainer_failure)?;
    let mut outcome = Outcome::new();
    outcome.outputs = report_json(&report, &p.kind, p.max_block);
    Ok(outcome)
}

/// The report of a `timeline-v1` dataset: each non-blank line is a subject or is refused, by its line. No token is
/// counted, so the token distributions are empty.
pub fn timeline_report(text: &str, max_block: u32) -> Value {
    let (mut records, mut valid, mut invalid_total) = (0usize, 0usize, 0usize);
    let mut invalid = Vec::new();
    for (index, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        records += 1;
        match horizon::timeline::Subject::from_json_line(line) {
            Ok(_) => valid += 1,
            Err(error) => {
                invalid_total += 1;
                if invalid.len() < brain::dataset_report::MAX_LISTED {
                    invalid.push(json!({"line": index + 1, "error": error.to_string()}));
                }
            }
        }
    }
    let empty = distribution(&brain::dataset_report::Distribution::default());
    json!({
        "kind": "timeline",
        "records": records,
        "valid_records": valid,
        "examples": valid,
        "tokens": empty,
        "supervised_tokens": empty,
        "examples_without_supervision": 0,
        "max_block": max_block,
        "over_max_block": 0,
        "over_max_block_lines": [],
        "invalid_total": invalid_total,
        "invalid": invalid,
        "listed_at_most": brain::dataset_report::MAX_LISTED,
    })
}
