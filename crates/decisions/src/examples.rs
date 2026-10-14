// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Outcome-labelled decision examples: the JSONL an external exporter writes
//! and a decision head is trained, evaluated and promoted on.
//!
//! This module is the ONLY reader of that file format. Everything downstream
//! (training, evaluation, promotion) sees [`DecisionExample`]s.
//!
//! # The line format
//!
//! One JSON object per line. Unknown fields are ignored, so an exporter may
//! add to its schema without breaking a reader.
//!
//! ```json
//! {"decision_type": "tool_routing",
//!  "state_ref": {"run": "r1", "task": "t1", "suite": "tool_selection", "seed": 3, "question_id": "q"},
//!  "state": "find the function that parses headers",
//!  "question": {"id": "q", "type": "choice", "instruction": "which tool"},
//!  "options": [{"id": "grep", "description": "search text"}, {"id": "read"}],
//!  "label": "grep", "label_index": 0, "label_source": "oracle",
//!  "outcome": {"task_passed": true, "llm_tokens_saved": 120, "recovery_calls": 0},
//!  "split": "train", "family": "tool_selection", "seed": 3}
//! ```
//!
//! * `state` is text or any JSON (key order preserved); when absent,
//!   `question.state` is read.
//! * `family` and `seed` are read from the line, else from `state_ref.suite`
//!   and `state_ref.seed`; a line with no family is refused.
//! * `label` must be one of the option ids; a `label_index`, when present,
//!   must agree with it.
//! * The option text a model reads is the option's `description`, else its `id`.
//! * An example's identity is its `state_ref` (plus `question.id` when the
//!   `state_ref` is a plain string); two lines with one identity are refused.
//!
//! # Leakage
//!
//! A held-out score means nothing if the held-out example, or its task, was
//! seen in training. After the last line is read, three groupings are refused
//! if they span more than one split, and the refusal is a report naming the
//! splits and the lines:
//!
//! * **task** - the `state_ref.task` (or the whole `state_ref` when it names
//!   no task): one task is one situation, however many runs or variants asked
//!   it;
//! * **family and seed** - a seed generates one world within a family;
//! * **content** - an identical state, instruction and option set.
//!
//! A family may appear in several splits (that is how a family is trained and
//! tested on); a family that appears in only one is how a whole family is held
//! out.
//!
//! Swedish Embedded AB implements evaluation pipelines whose held-out numbers
//! can be trusted: schema checks, leakage checks and honest baselines before a
//! model is allowed near production. If your team needs expertise in
//! decision-model data and evaluation, you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::BufRead;
use std::path::Path;

use modernbert::{write_json, OrderedJson};
use serde_json::{json, Value};

use crate::wire::MAX_STATE_BYTES;

/// Most bytes one line may hold: the largest state plus room for the rest.
pub const MAX_LINE_BYTES: usize = MAX_STATE_BYTES + (1 << 20);
/// Most options one example may carry; the `choice` limit of the wire.
pub const MAX_OPTIONS: usize = 255;

/// Which part of the data an example belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub const ALL: [Split; 3] = [Split::Train, Split::Validation, Split::Test];

    pub fn slug(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        }
    }

    pub fn parse(s: &str) -> Option<Split> {
        Split::ALL.into_iter().find(|p| p.slug() == s)
    }
}

/// Who decided the label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LabelSource {
    Oracle,
    CounterfactualReplay,
    Scorer,
}

impl LabelSource {
    pub fn slug(self) -> &'static str {
        match self {
            LabelSource::Oracle => "oracle",
            LabelSource::CounterfactualReplay => "counterfactual_replay",
            LabelSource::Scorer => "scorer",
        }
    }

    pub fn parse(s: &str) -> Option<LabelSource> {
        [LabelSource::Oracle, LabelSource::CounterfactualReplay, LabelSource::Scorer].into_iter().find(|p| p.slug() == s)
    }
}

/// One option of a question.
#[derive(Clone, Debug, PartialEq)]
pub struct ExampleOption {
    pub id: String,
    pub description: Option<String>,
}

impl ExampleOption {
    /// What a model reads for this option: its description, else its id.
    pub fn text(&self) -> &str {
        self.description.as_deref().unwrap_or(&self.id)
    }
}

/// What happened after the decision. Every field is optional: an exporter
/// reports only what it measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    pub task_passed: Option<bool>,
    pub llm_tokens_saved: Option<f64>,
    pub recovery_calls: Option<u64>,
}

/// One labelled choice decision.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionExample {
    /// The 1-based line this example was read from.
    pub line: usize,
    /// The example's identity; unique within a dataset.
    pub id: String,
    /// What leakage is judged on: the task this example belongs to.
    pub task: String,
    pub decision_type: String,
    pub family: String,
    pub seed: Option<u64>,
    /// The state, serialized the way a model reads it.
    pub state: String,
    pub question_id: String,
    pub instruction: String,
    pub options: Vec<ExampleOption>,
    /// Index into `options` of the correct one.
    pub label: usize,
    pub label_source: LabelSource,
    pub outcome: Outcome,
    pub split: Split,
}

impl DecisionExample {
    /// The id of the correct option.
    pub fn label_id(&self) -> &str {
        &self.options[self.label].id
    }

    /// The option texts a model reads, in the order given.
    pub fn option_texts(&self) -> Vec<String> {
        self.options.iter().map(|o| o.text().to_string()).collect()
    }
}

/// Why a dataset was refused.
#[derive(Debug)]
pub enum DatasetError {
    /// The file could not be read.
    Io(String),
    /// One line is not a valid example.
    Line { line: usize, message: String },
    /// Examples of one task, seed or content sit in more than one split.
    Leakage(Vec<Leak>),
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetError::Io(e) => write!(f, "{e}"),
            DatasetError::Line { line, message } => write!(f, "line {line}: {message}"),
            DatasetError::Leakage(leaks) => {
                writeln!(f, "the data leaks across splits ({} group(s)); a held-out score on it would not be held out:", leaks.len())?;
                for leak in leaks {
                    writeln!(f, "  {leak}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for DatasetError {}

/// What a leaking group has in common.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeakKind {
    Task,
    FamilySeed,
    Content,
}

impl LeakKind {
    fn slug(self) -> &'static str {
        match self {
            LeakKind::Task => "task",
            LeakKind::FamilySeed => "family+seed",
            LeakKind::Content => "content",
        }
    }
}

/// One group that appears in more than one split.
#[derive(Clone, Debug, PartialEq)]
pub struct Leak {
    pub kind: LeakKind,
    /// The shared task, `family/seed`, or content digest.
    pub key: String,
    /// The splits it appears in and the lines in each, in file order.
    pub splits: Vec<(Split, Vec<usize>)>,
}

impl fmt::Display for Leak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:?} is in", self.kind.slug(), self.key)?;
        for (i, (split, lines)) in self.splits.iter().enumerate() {
            let shown: Vec<String> = lines.iter().take(5).map(|l| l.to_string()).collect();
            let more = if lines.len() > 5 { format!(" and {} more", lines.len() - 5) } else { String::new() };
            write!(f, "{} {} (lines {}{more})", if i == 0 { "" } else { "," }, split.slug(), shown.join(", "))?;
        }
        Ok(())
    }
}

/// A validated dataset: every example parsed, no identity repeated, no group
/// spanning two splits.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub examples: Vec<DecisionExample>,
    /// `sha256:<hex>` of the file's bytes, when it was read from a file.
    pub digest: Option<String>,
}

impl Dataset {
    /// Read and validate JSONL from `reader`, one line at a time.
    pub fn read(reader: impl BufRead) -> Result<Dataset, DatasetError> {
        let mut examples: Vec<DecisionExample> = Vec::new();
        let mut first_line_of: HashMap<String, usize> = HashMap::new();
        for (index, text) in Lines::new(reader).enumerate() {
            let line = index + 1;
            let text = text.map_err(|message| DatasetError::Line { line, message })?;
            if text.trim().is_empty() {
                continue;
            }
            let example = parse_line(&text, line).map_err(|message| DatasetError::Line { line, message })?;
            if let Some(first) = first_line_of.insert(example.id.clone(), line) {
                return Err(DatasetError::Line { line, message: format!("duplicate example id {:?}, first seen on line {first}", example.id) });
            }
            examples.push(example);
        }
        let leaks = leaks(&examples);
        if !leaks.is_empty() {
            return Err(DatasetError::Leakage(leaks));
        }
        Ok(Dataset { examples, digest: None })
    }

    /// Read and validate the file at `path`, recording its digest.
    pub fn read_file(path: &Path) -> Result<Dataset, DatasetError> {
        let file = std::fs::File::open(path).map_err(|e| DatasetError::Io(format!("{}: {e}", path.display())))?;
        let mut dataset = Dataset::read(std::io::BufReader::new(file))?;
        let digest = brain_modelstore::fetch::file_digest(path).map_err(|e| DatasetError::Io(format!("{}: {e}", path.display())))?;
        dataset.digest = Some(digest);
        Ok(dataset)
    }

    /// The examples of one split, in file order.
    pub fn split(&self, split: Split) -> impl Iterator<Item = &DecisionExample> {
        self.examples.iter().filter(move |e| e.split == split)
    }

    pub fn summary(&self) -> Summary {
        Summary::of(&self.examples)
    }
}

/// Lines of a reader with a bound on each, so one runaway line cannot take
/// the process's memory.
struct Lines<R: BufRead> {
    reader: R,
}

impl<R: BufRead> Lines<R> {
    fn new(reader: R) -> Lines<R> {
        Lines { reader }
    }
}

impl<R: BufRead> Iterator for Lines<R> {
    type Item = Result<String, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut bytes = Vec::new();
        let mut limited = std::io::Read::take(&mut self.reader, MAX_LINE_BYTES as u64 + 1);
        match limited.read_until(b'\n', &mut bytes) {
            Ok(0) => None,
            Ok(_) if bytes.len() > MAX_LINE_BYTES => {
                // Drain the rest of the line so the next call starts on the next one.
                let mut rest = Vec::new();
                let _ = self.reader.read_until(b'\n', &mut rest);
                Some(Err(format!("a line is longer than {MAX_LINE_BYTES} bytes")))
            }
            Ok(_) => Some(String::from_utf8(bytes).map_err(|_| "the line is not valid UTF-8".to_string())),
            Err(e) => Some(Err(format!("read failed: {e}"))),
        }
    }
}

/// The entries of a JSON object, looked up by key.
struct Fields<'a>(&'a [(String, OrderedJson)]);

impl<'a> Fields<'a> {
    fn of(v: &'a OrderedJson, what: &str) -> Result<Fields<'a>, String> {
        match v {
            OrderedJson::Object(entries) => Ok(Fields(entries)),
            _ => Err(format!("{what} must be a JSON object")),
        }
    }

    fn get(&self, key: &str) -> Option<&'a OrderedJson> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v).filter(|v| !matches!(v, OrderedJson::Null))
    }

    fn string(&self, key: &str, what: &str) -> Result<Option<&'a str>, String> {
        match self.get(key) {
            None => Ok(None),
            Some(OrderedJson::String(s)) => Ok(Some(s)),
            Some(_) => Err(format!("{what}.{key} must be a string")),
        }
    }

    fn required_string(&self, key: &str, what: &str) -> Result<&'a str, String> {
        self.string(key, what)?.filter(|s| !s.is_empty()).ok_or_else(|| format!("{what}.{key} is required"))
    }
}

fn canonical(v: &OrderedJson) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn parse_line(text: &str, line: usize) -> Result<DecisionExample, String> {
    let root = OrderedJson::parse(text).map_err(|e| format!("not valid JSON: {e}"))?;
    let top = Fields::of(&root, "the line")?;

    let decision_type = top.required_string("decision_type", "the line")?.to_string();
    let split = {
        let s = top.required_string("split", "the line")?;
        Split::parse(s).ok_or_else(|| format!("split must be \"train\", \"validation\" or \"test\", got {s:?}"))?
    };
    let label_source = {
        let s = top.required_string("label_source", "the line")?;
        LabelSource::parse(s).ok_or_else(|| format!("label_source must be \"oracle\", \"counterfactual_replay\" or \"scorer\", got {s:?}"))?
    };

    let question_value = top.get("question").ok_or("question is required")?;
    let question = Fields::of(question_value, "question")?;
    let question_id = question.required_string("id", "question")?.to_string();
    match question.string("type", "question")? {
        None | Some("choice") => {}
        Some(other) => return Err(format!("question.type must be \"choice\", the only type trained on, got {other:?}")),
    }
    let instruction = question.required_string("instruction", "question")?.to_string();

    let state_ref = top.get("state_ref").ok_or("state_ref is required")?;
    let (id, task, ref_fields) = match state_ref {
        OrderedJson::String(s) if !s.is_empty() => (format!("{s}#{question_id}"), s.clone(), None),
        OrderedJson::Object(_) => {
            let fields = Fields::of(state_ref, "state_ref")?;
            let task = fields.string("task", "state_ref")?.filter(|t| !t.is_empty()).map_or_else(|| canonical(state_ref), str::to_string);
            (canonical(state_ref), task, Some(fields))
        }
        _ => return Err("state_ref must be a non-empty string or an object".to_string()),
    };

    let family = match top.string("family", "the line")?.or(ref_fields.as_ref().map(|f| f.string("suite", "state_ref")).transpose()?.flatten()) {
        Some(f) if !f.is_empty() => f.to_string(),
        _ => return Err("family is required (as `family`, or `state_ref.suite`)".to_string()),
    };
    let seed = match top.get("seed").or(ref_fields.as_ref().and_then(|f| f.get("seed"))) {
        None => None,
        Some(OrderedJson::Number(n)) => Some(n.as_u64().ok_or("seed must be a non-negative integer")?),
        Some(_) => return Err("seed must be a non-negative integer".to_string()),
    };

    let state = match top.get("state").or(question.get("state")) {
        Some(OrderedJson::String(s)) => s.clone(),
        Some(other) => canonical(other),
        None => return Err("state is required (as `state`, or `question.state`)".to_string()),
    };
    if state.len() > MAX_STATE_BYTES {
        return Err(format!("state is {} bytes; at most {MAX_STATE_BYTES} are accepted", state.len()));
    }

    let options = match top.get("options") {
        Some(OrderedJson::Array(items)) if !items.is_empty() => items,
        _ => return Err("options must be a non-empty array".to_string()),
    };
    if options.len() > MAX_OPTIONS {
        return Err(format!("options has {} entries; at most {MAX_OPTIONS} are accepted", options.len()));
    }
    let mut parsed: Vec<ExampleOption> = Vec::with_capacity(options.len());
    for (i, item) in options.iter().enumerate() {
        let option = match item {
            OrderedJson::String(id) if !id.is_empty() => ExampleOption { id: id.clone(), description: None },
            OrderedJson::Object(_) => {
                let fields = Fields::of(item, "an option")?;
                let id = fields.required_string("id", &format!("options[{i}]"))?.to_string();
                let description = fields.string("description", &format!("options[{i}]"))?.filter(|d| !d.is_empty()).map(str::to_string);
                ExampleOption { id, description }
            }
            _ => return Err(format!("options[{i}] must be an id string or an object with an id")),
        };
        if parsed.iter().any(|o| o.id == option.id) {
            return Err(format!("duplicate option id {:?} in options", option.id));
        }
        parsed.push(option);
    }

    let label_id = top.required_string("label", "the line")?;
    let label = parsed.iter().position(|o| o.id == label_id).ok_or_else(|| format!("label {label_id:?} is not one of the option ids"))?;
    match top.get("label_index") {
        None => {}
        Some(OrderedJson::Number(n)) if n.as_u64() == Some(label as u64) => {}
        Some(other) => return Err(format!("label_index {} does not match the position {label} of label {label_id:?}", canonical(other))),
    }

    let outcome = match top.get("outcome") {
        None => Outcome::default(),
        Some(value) => {
            let o = Fields::of(value, "outcome")?;
            Outcome {
                task_passed: match o.get("task_passed") {
                    None => None,
                    Some(OrderedJson::Bool(b)) => Some(*b),
                    Some(_) => return Err("outcome.task_passed must be a boolean".to_string()),
                },
                llm_tokens_saved: match o.get("llm_tokens_saved") {
                    None => None,
                    Some(OrderedJson::Number(n)) => n.as_f64(),
                    Some(_) => return Err("outcome.llm_tokens_saved must be a number".to_string()),
                },
                recovery_calls: match o.get("recovery_calls") {
                    None => None,
                    Some(OrderedJson::Number(n)) => Some(n.as_u64().ok_or("outcome.recovery_calls must be a non-negative integer")?),
                    Some(_) => return Err("outcome.recovery_calls must be a non-negative integer".to_string()),
                },
            }
        }
    };

    Ok(DecisionExample { line, id, task, decision_type, family, seed, state, question_id, instruction, options: parsed, label, label_source, outcome, split })
}

/// Every group that appears in more than one split, in a stable order.
fn leaks(examples: &[DecisionExample]) -> Vec<Leak> {
    type Groups = BTreeMap<String, BTreeMap<Split, Vec<usize>>>;
    let mut by_kind: [(LeakKind, Groups); 3] = [(LeakKind::Task, Groups::new()), (LeakKind::FamilySeed, Groups::new()), (LeakKind::Content, Groups::new())];
    for e in examples {
        let content = {
            let ids: Vec<&str> = e.options.iter().map(|o| o.id.as_str()).collect();
            brain_modelstore::fetch::bytes_digest(format!("{}\0{}\0{}", e.state, e.instruction, ids.join("\0")).as_bytes())
        };
        let keys = [Some(e.task.clone()), e.seed.map(|s| format!("{}/{s}", e.family)), Some(content)];
        for ((_, groups), key) in by_kind.iter_mut().zip(keys) {
            if let Some(key) = key {
                groups.entry(key).or_default().entry(e.split).or_default().push(e.line);
            }
        }
    }
    let mut out = Vec::new();
    for (kind, groups) in by_kind {
        for (key, splits) in groups {
            if splits.len() > 1 {
                out.push(Leak { kind, key, splits: splits.into_iter().collect() });
            }
        }
    }
    out
}

/// Counts and baselines over a set of examples.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub total: usize,
    pub by_split: BTreeMap<Split, SplitSummary>,
    pub by_decision_type: BTreeMap<String, usize>,
    pub by_family: BTreeMap<String, usize>,
    pub by_label: BTreeMap<String, usize>,
    pub by_label_source: BTreeMap<LabelSource, usize>,
    /// Entropy of the label ids over all examples, in bits.
    pub label_entropy_bits: f64,
    pub majority_label: Option<String>,
    /// Accuracy of always answering the most common label id.
    pub majority_accuracy: f64,
}

/// One split's share of the data.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitSummary {
    pub count: usize,
    pub families: BTreeMap<String, usize>,
    pub label_entropy_bits: f64,
    pub majority_label: Option<String>,
    pub majority_accuracy: f64,
}

/// Entropy in bits, most common key and its share, of a label histogram.
fn label_stats(counts: &BTreeMap<String, usize>) -> (f64, Option<String>, f64) {
    let total: usize = counts.values().sum();
    if total == 0 {
        return (0.0, None, 0.0);
    }
    let entropy = counts.values().map(|&c| c as f64 / total as f64).map(|p| -p * p.log2()).sum::<f64>();
    // First maximum in key order, so a tie is broken the same way every run.
    let (label, &top) = counts.iter().fold(None, |best: Option<(&String, &usize)>, kv| match best {
        Some((_, &b)) if b >= *kv.1 => best,
        _ => Some(kv),
    }).expect("a non-empty histogram has a maximum");
    (entropy, Some(label.clone()), top as f64 / total as f64)
}

fn histogram<'a>(items: impl Iterator<Item = &'a str>) -> BTreeMap<String, usize> {
    let mut h = BTreeMap::new();
    for item in items {
        *h.entry(item.to_string()).or_insert(0) += 1;
    }
    h
}

impl Summary {
    pub fn of(examples: &[DecisionExample]) -> Summary {
        let labels = histogram(examples.iter().map(|e| e.label_id()));
        let (label_entropy_bits, majority_label, majority_accuracy) = label_stats(&labels);
        let by_split = Split::ALL
            .into_iter()
            .filter_map(|split| {
                let members: Vec<&DecisionExample> = examples.iter().filter(|e| e.split == split).collect();
                if members.is_empty() {
                    return None;
                }
                let (label_entropy_bits, majority_label, majority_accuracy) = label_stats(&histogram(members.iter().map(|e| e.label_id())));
                let families = histogram(members.iter().map(|e| e.family.as_str()));
                Some((split, SplitSummary { count: members.len(), families, label_entropy_bits, majority_label, majority_accuracy }))
            })
            .collect();
        let mut by_label_source = BTreeMap::new();
        for e in examples {
            *by_label_source.entry(e.label_source).or_insert(0) += 1;
        }
        Summary {
            total: examples.len(),
            by_split,
            by_decision_type: histogram(examples.iter().map(|e| e.decision_type.as_str())),
            by_family: histogram(examples.iter().map(|e| e.family.as_str())),
            by_label: labels,
            by_label_source,
            label_entropy_bits,
            majority_label,
            majority_accuracy,
        }
    }

    pub fn to_json(&self) -> Value {
        let splits: serde_json::Map<String, Value> = self
            .by_split
            .iter()
            .map(|(split, s)| {
                let v = json!({"count": s.count, "families": s.families, "label_entropy_bits": s.label_entropy_bits,
                               "majority_label": s.majority_label, "majority_accuracy": s.majority_accuracy});
                (split.slug().to_string(), v)
            })
            .collect();
        let sources: serde_json::Map<String, Value> = self.by_label_source.iter().map(|(s, n)| (s.slug().to_string(), json!(n))).collect();
        json!({
            "total": self.total,
            "splits": splits,
            "decision_types": self.by_decision_type,
            "families": self.by_family,
            "labels": self.by_label,
            "label_sources": sources,
            "label_entropy_bits": self.label_entropy_bits,
            "majority_label": self.majority_label,
            "majority_accuracy": self.majority_accuracy,
        })
    }

    /// A short human-readable table.
    pub fn render(&self) -> String {
        let mut out = format!("{} examples, {} labels, label entropy {:.3} bits, majority baseline {:.3} ({})\n", self.total, self.by_label.len(), self.label_entropy_bits, self.majority_accuracy, self.majority_label.as_deref().unwrap_or("-"));
        out.push_str("split         examples  entropy(bits)  majority-acc  majority\n");
        for (split, s) in &self.by_split {
            out.push_str(&format!("{:<12}  {:>8}  {:>13.3}  {:>12.3}  {}\n", split.slug(), s.count, s.label_entropy_bits, s.majority_accuracy, s.majority_label.as_deref().unwrap_or("-")));
        }
        let list = |h: &BTreeMap<String, usize>| h.iter().map(|(k, n)| format!("{k}={n}")).collect::<Vec<_>>().join(" ");
        out.push_str(&format!("decision types: {}\nfamilies: {}\n", list(&self.by_decision_type), list(&self.by_family)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One well-formed line; `extra` is spliced in as further fields.
    fn line(task: &str, family: &str, seed: u64, label: &str, split: &str) -> String {
        format!(
            r#"{{"decision_type": "tool_routing", "state_ref": {{"run": "r-{task}", "task": "{task}", "suite": "{family}", "seed": {seed}, "question_id": "q"}}, "state": "state of {task}", "question": {{"id": "q", "type": "choice", "instruction": "which tool"}}, "options": [{{"id": "grep", "description": "search text"}}, {{"id": "read"}}, {{"id": "list"}}], "label": "{label}", "label_source": "oracle", "outcome": {{"task_passed": true, "llm_tokens_saved": 12, "recovery_calls": 0}}, "split": "{split}", "family": "{family}", "seed": {seed}}}"#
        )
    }

    fn read(lines: &[String]) -> Result<Dataset, DatasetError> {
        Dataset::read(lines.join("\n").as_bytes())
    }

    fn refused(lines: &[String]) -> String {
        read(lines).expect_err("must be refused").to_string()
    }

    #[test]
    fn a_well_formed_file_reads_with_its_options_label_and_outcome() {
        let d = read(&[line("t1", "f1", 1, "grep", "train"), line("t2", "f1", 2, "read", "test")]).unwrap();
        assert_eq!(d.examples.len(), 2);
        let e = &d.examples[0];
        assert_eq!((e.line, e.split, e.family.as_str(), e.seed), (1, Split::Train, "f1", Some(1)));
        assert_eq!(e.state, "state of t1");
        assert_eq!(e.instruction, "which tool");
        assert_eq!(e.option_texts(), ["search text", "read", "list"]);
        assert_eq!((e.label, e.label_id()), (0, "grep"));
        assert_eq!(e.label_source, LabelSource::Oracle);
        assert_eq!(e.outcome, Outcome { task_passed: Some(true), llm_tokens_saved: Some(12.0), recovery_calls: Some(0) });
    }

    #[test]
    fn unknown_fields_are_ignored_and_blank_lines_skipped() {
        let with_extra = line("t1", "f1", 1, "grep", "train").replacen('{', r#"{"added_later": {"x": [1, 2]}, "#, 1);
        let d = read(&[with_extra, String::new(), line("t2", "f1", 2, "read", "train")]).unwrap();
        assert_eq!(d.examples.len(), 2);
        assert_eq!(d.examples[1].line, 3, "line numbers count the blank line");
    }

    #[test]
    fn the_family_and_seed_fall_back_to_the_state_ref_and_the_state_to_the_question() {
        let l = r#"{"decision_type": "d", "state_ref": {"task": "t", "suite": "s", "seed": 9}, "question": {"id": "q", "type": "choice", "instruction": "i", "state": "from the question"}, "options": [{"id": "a"}, {"id": "b"}], "label": "b", "label_source": "scorer", "split": "val_typo"}"#;
        assert!(refused(&[l.to_string()]).contains("split"));
        let d = read(&[l.replace("val_typo", "validation")]).unwrap();
        let e = &d.examples[0];
        assert_eq!((e.family.as_str(), e.seed, e.state.as_str(), e.label), ("s", Some(9), "from the question", 1));
        assert_eq!(e.outcome, Outcome::default());
    }

    #[test]
    fn a_json_state_keeps_its_key_order() {
        let l = line("t1", "f1", 1, "grep", "train").replace(r#""state": "state of t1""#, r#""state": {"zeta": 1, "alpha": [true, null]}"#);
        assert_eq!(read(&[l]).unwrap().examples[0].state, r#"{"zeta": 1, "alpha": [true, null]}"#);
    }

    #[test]
    fn every_malformed_line_is_refused_naming_its_line_and_the_field() {
        let good = line("t1", "f1", 1, "grep", "train");
        let cases: Vec<(String, &str)> = vec![
            ("not json".to_string(), "line 2"),
            ("[]".to_string(), "JSON object"),
            (good.replace(r#""label": "grep""#, r#""label": "nope""#), "nope"),
            (good.replace(r#""label": "grep""#, r#""label": "grep", "label_index": 2"#), "label_index"),
            (good.replace(r#""split": "train""#, r#""split": "dev""#), "split"),
            (good.replace(r#""label_source": "oracle""#, r#""label_source": "vibes""#), "label_source"),
            (good.replace(r#""type": "choice""#, r#""type": "score""#), "choice"),
            (good.replace(r#"{"id": "read"}"#, r#"{"id": "grep"}"#), "duplicate option"),
            (good.replace(r#""options": [{"id": "grep", "description": "search text"}, {"id": "read"}, {"id": "list"}]"#, r#""options": []"#), "options"),
            (good.replace(r#""instruction": "which tool""#, r#""instruction": 4"#), "instruction"),
            (good.replace(r#""decision_type": "tool_routing", "#, ""), "decision_type"),
            (good.replace(r#", "family": "f1""#, "").replace(r#""suite": "f1", "#, ""), "family"),
            (good.replace(r#""state_ref": {"#, r#""state_ref_x": {"#), "state_ref"),
        ];
        for (bad, needle) in cases {
            let err = refused(&[good.clone(), bad.clone()]);
            assert!(err.starts_with("line 2") && err.contains(needle), "{bad}\n  -> {err}");
        }
    }

    #[test]
    fn a_label_index_that_agrees_is_accepted() {
        let l = line("t1", "f1", 1, "read", "train").replace(r#""label": "read""#, r#""label": "read", "label_index": 1"#);
        assert_eq!(read(&[l]).unwrap().examples[0].label, 1);
    }

    #[test]
    fn a_duplicate_example_id_is_refused_naming_both_lines() {
        let a = line("t1", "f1", 1, "grep", "train");
        let err = refused(&[a.clone(), line("t2", "f1", 2, "read", "train"), a]);
        assert!(err.starts_with("line 3") && err.contains("duplicate example id") && err.contains("line 1"), "{err}");
    }

    #[test]
    fn one_task_in_two_splits_is_a_leak_reported_with_both_splits_and_lines() {
        // Two runs of one task, one in train and one in test.
        let a = line("t1", "f1", 1, "grep", "train");
        let b = line("t1", "f1", 2, "grep", "test").replace(r#""run": "r-t1""#, r#""run": "other""#);
        let DatasetError::Leakage(leaks) = read(&[a, b]).unwrap_err() else { panic!("expected a leak") };
        let task = leaks.iter().find(|l| l.kind == LeakKind::Task).expect("the task leak");
        assert_eq!(task.key, "t1");
        assert_eq!(task.splits, vec![(Split::Train, vec![1]), (Split::Test, vec![2])]);
        let text = DatasetError::Leakage(leaks).to_string();
        assert!(text.contains("task \"t1\"") && text.contains("train (lines 1)") && text.contains("test (lines 2)"), "{text}");
    }

    #[test]
    fn one_family_and_seed_in_two_splits_is_a_leak() {
        let err = refused(&[line("t1", "f1", 7, "grep", "train"), line("t2", "f1", 7, "read", "validation")]);
        assert!(err.contains("family+seed") && err.contains("f1/7"), "{err}");
    }

    #[test]
    fn identical_content_in_two_splits_is_a_leak_even_across_tasks() {
        let a = line("t1", "f1", 1, "grep", "train");
        let b = line("t2", "f2", 2, "grep", "test").replace("state of t2", "state of t1");
        let err = refused(&[a, b]);
        assert!(err.contains("content"), "{err}");
    }

    #[test]
    fn a_family_in_several_splits_by_different_seeds_is_not_a_leak() {
        let d = read(&[line("t1", "f1", 1, "grep", "train"), line("t2", "f1", 2, "read", "validation"), line("t3", "f1", 3, "list", "test")]).unwrap();
        assert_eq!(d.split(Split::Test).count(), 1);
    }

    #[test]
    fn an_oversized_line_is_refused_and_the_next_line_still_reads() {
        let huge = format!("{{\"x\": \"{}\"}}", "a".repeat(MAX_LINE_BYTES));
        let err = refused(&[line("t1", "f1", 1, "grep", "train"), huge]);
        assert!(err.starts_with("line 2") && err.contains("longer than"), "{err}");
    }

    #[test]
    fn the_summary_counts_and_baselines_what_is_in_the_file() {
        let d = read(&[
            line("t1", "f1", 1, "grep", "train"),
            line("t2", "f1", 2, "grep", "train"),
            line("t3", "f1", 3, "read", "train"),
            line("t4", "f2", 4, "list", "test"),
        ])
        .unwrap();
        let s = d.summary();
        assert_eq!(s.total, 4);
        assert_eq!(s.by_split[&Split::Train].count, 3);
        assert_eq!(s.by_split[&Split::Test].count, 1);
        assert!(!s.by_split.contains_key(&Split::Validation));
        assert_eq!(s.by_family["f1"], 3);
        assert_eq!(s.by_label["grep"], 2);
        assert_eq!(s.by_decision_type["tool_routing"], 4);
        assert_eq!(s.by_label_source[&LabelSource::Oracle], 4);
        assert_eq!(s.majority_label.as_deref(), Some("grep"));
        assert!((s.majority_accuracy - 0.5).abs() < 1e-12);
        // Labels {grep: 2, read: 1, list: 1}: H = 1.5 bits.
        assert!((s.label_entropy_bits - 1.5).abs() < 1e-12);
        let train = &s.by_split[&Split::Train];
        assert_eq!(train.majority_label.as_deref(), Some("grep"));
        assert!((train.majority_accuracy - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(s.by_split[&Split::Test].label_entropy_bits, 0.0);
    }

    #[test]
    fn a_file_is_read_with_its_digest() {
        let dir = crate::fixtures::Scratch::new("examples-file");
        let path = dir.join("d.jsonl");
        std::fs::write(&path, line("t1", "f1", 1, "grep", "train") + "\n").unwrap();
        let d = Dataset::read_file(&path).unwrap();
        assert_eq!(d.digest, Some(brain_modelstore::fetch::file_digest(&path).unwrap()));
        assert!(Dataset::read_file(&dir.join("absent.jsonl")).is_err());
    }
}
