// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Teaching a model a batch of documents, and gating whether it learned them.
//!
//! A **document study** trains a LoRA adapter on frozen
//! `{fact, probe_question, expected_answer}` triples under `Regime::Sft`,
//! scores the result against a pre-registered bar
//! (`promote::document::document_gate_config`), and runs a NULL-GATE control
//! arm beside it so the resulting number says something about the gate rather
//! than only about the run. The adapter is published only if the gate
//! promoted.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! use brain::DocumentStudy;
//!
//! let outcome = DocumentStudy::from_pretrained("Qwen/Qwen3-0.6B")?
//!     .dataset("facts.json")
//!     .adapter_dir("adapters/")
//!     .run()?;
//!
//! println!("{}", outcome.table());
//! if let Some(adapter) = outcome.published() {
//!     println!("promoted -> {}", adapter.display());
//! }
//! # Ok(()) }
//! ```
//!
//! ## Why this is architecture-driven rather than one model's verb
//!
//! `rl::continual::run_study` is generic over `M: model::Model` and
//! `rl::document::DocumentCurriculum` names no model type at all, so the
//! study is architecture-agnostic machinery. Nesting it under one model would
//! tie a general capability to one consumer and invite the next consumer to
//! grow a second copy. [`ARCHS`] is the registry a new architecture joins by
//! adding one row.
//!
//! **What bounds that registry is the TOKENIZER, not the model.**
//! `DocumentCurriculum` is generic over the model but hard-wired to
//! `data::qwen_tokenizer::QwenBpe`, so an architecture qualifies iff it loads
//! an HF `tokenizer.json` BPE and its config can carry a LoRA overlay. That is
//! why the registered rows are the Qwen-family decoders - a fact about
//! `DocumentCurriculum`'s signature, not a choice made here.
//!
//! ## What it does NOT do
//!
//! It computes nothing the study did not already compute. Every number in the
//! outcome comes from the two `rl::continual::StudyReport`s
//! `run_document_study` returns, and the adapter it publishes is the file
//! `rl::improve::cycle` already wrote when the gate promoted - copied, not
//! retrained. A surface that re-scored anything here would be reporting a
//! different measurement from the one the gate actually made.
//!
//! Swedish Embedded AB implements continual-learning pipelines whose promotion
//! decisions are gated by pre-registered criteria and their own controls. If
//! your team needs a model that can be taught something new without silently
//! forgetting what it knew, you can procure our services by sending an email
//! to info@swedishembedded.com.

use crate::{Error, Result};

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use model::{Model, ModelConfig};
use qwen3::config::LoraCfg;
use rl::continual::{self, Curriculum, SftConfig, StudyReport, StudySpec};
use rl::document::{self, DocumentCurriculum, DocumentStudyConfig, DocumentStudyReport, FactBatch, FactProbe, MIN_HELD_OUT_PROBES};
use rl::gate::{Cause, Decision};
use rl::improve::AdapterMeta;


/// The architecture a study runs against when `--arch` is omitted. Named,
/// not silent: the study's measured recipe (`SftConfig::default`) was tuned
/// on this family.

// ---------------------------------------------------------------------------
// The architecture seam
// ---------------------------------------------------------------------------

/// Everything a document study needs from an architecture beyond
/// `model::Model` - which is two things, both about the LoRA overlay the
/// study trains.
///
/// `qwen3::config::LoraCfg` is deliberately the shared currency here rather
/// than a per-crate type: `qwen35` and `qwen35moe` both `pub use
/// qwen3::LoraCfg` and differ only in WHICH projections they target, which is
/// exactly what [`StudyArch::lora`] returns. A per-architecture LoRA struct
/// would be a copy of a struct three crates already share.
trait StudyArch {
    type M: Model;

    /// This architecture's own LoRA overlay at `rank`/`alpha` - the four
    /// attention projections for `qwen3`, the fused/expert projections its
    /// successors adapt instead.
    fn lora(rank: u32, alpha: f32) -> LoraCfg;

    /// `base` plus that overlay, with a context at least `block` wide.
    ///
    /// Widening is this function's job rather than the caller's because
    /// "which field carries the trained RoPE extent alongside `block_size`"
    /// is architecture-specific, and a study whose curriculum does not fit
    /// the base's context is refused by `run_study` with a panic rather than
    /// silently truncated.
    fn study_config(base: &<Self::M as Model>::Config, lora: LoraCfg, block: u32) -> <Self::M as Model>::Config;
}

struct Qwen3;
impl StudyArch for Qwen3 {
    type M = qwen3::model::Qwen;
    fn lora(rank: u32, alpha: f32) -> LoraCfg {
        LoraCfg::attn(rank, alpha)
    }
    fn study_config(base: &qwen3::config::QwenConfig, lora: LoraCfg, block: u32) -> qwen3::config::QwenConfig {
        qwen3::config::QwenConfig { block_size: block, max_position_embeddings: base.max_position_embeddings.max(block), lora: Some(lora), ..base.clone() }
    }
}

struct Qwen35;
impl StudyArch for Qwen35 {
    type M = qwen35::model::Qwen35;
    fn lora(rank: u32, alpha: f32) -> LoraCfg {
        qwen35::config::lora_cfg(rank, alpha)
    }
    fn study_config(base: &qwen35::config::Qwen35Config, lora: LoraCfg, block: u32) -> qwen35::config::Qwen35Config {
        qwen35::config::Qwen35Config { block_size: block, max_position_embeddings: base.max_position_embeddings.max(block), lora: Some(lora), ..base.clone() }
    }
}

struct Qwen35Moe;
impl StudyArch for Qwen35Moe {
    type M = qwen35moe::model::Qwen35;
    fn lora(rank: u32, alpha: f32) -> LoraCfg {
        qwen35moe::config::lora_cfg(rank, alpha)
    }
    fn study_config(base: &qwen35moe::config::Qwen35Config, lora: LoraCfg, block: u32) -> qwen35moe::config::Qwen35Config {
        qwen35moe::config::Qwen35Config { block_size: block, max_position_embeddings: base.max_position_embeddings.max(block), lora: Some(lora), ..base.clone() }
    }
}

/// The registry: architecture id (a `brain_arch` id, so `--arch qwen3` is the
/// same word every other brain command uses) -> the monomorphised study.
///
/// Adding an architecture is one row, exactly as `bench::arch` documents for
/// the benchmark battery. See this module's doc comment for the one property
/// a candidate must have that `M: Model` does not imply: an HF
/// `tokenizer.json` BPE, because `DocumentCurriculum` names `QwenBpe`.
const ARCHS: &[(&str, StudyFn)] = &[("qwen3", run_for::<Qwen3>), ("qwen35", run_for::<Qwen35>), ("qwen35moe", run_for::<Qwen35Moe>)];

/// One [`ARCHS`] row's study, already monomorphised for its `Model` impl -
/// the erasure that lets a table hold architectures whose `Model` types have
/// nothing in common but the trait.
type StudyFn = fn(&Inputs) -> std::io::Result<DocumentStudyReport>;

fn arch_names() -> String {
    ARCHS.iter().map(|(n, _)| *n).collect::<Vec<&str>>().join(", ")
}

// ---------------------------------------------------------------------------
// The dataset - the one input brain did not itself produce
// ---------------------------------------------------------------------------

/// A document-study dataset as it crosses into brain from outside - sven's
/// extraction step, or a hand-written batch.
///
/// `deny_unknown_fields` and plain (non-`Option`) members throughout, here
/// and in [`FactProbe`] itself: serde is then the structural validator, so a
/// missing, mistyped or extra field is a loud parse failure naming the field
/// rather than a plausible-looking default that trains silently on the wrong
/// thing.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentDataset {
    /// One batch of frozen triples per study CYCLE. The study runs exactly
    /// as many cycles as there are batches here - a separate `--cycles` flag
    /// could only disagree with the data.
    cycles: Vec<Vec<FactProbe>>,
    /// The behavioural anchor suite every cycle rehearses (system-prompt
    /// adherence, refusal, tool-call format). `Regime::Sft` mixes it into
    /// every cycle's draw, and `DocumentCurriculum::new` refuses an empty
    /// one.
    anchors: Vec<FactProbe>,
}

// ---------------------------------------------------------------------------
// The report - the whole point of the command on a reject as much as a promote
// ---------------------------------------------------------------------------

/// One cycle's row, for one arm. Every field is read straight off that
/// cycle's [`rl::continual::CycleRecord`].
#[derive(Serialize)]
struct CycleRow {
    cycle: usize,
    label: String,
    /// The DISTINCT fact statements this cycle trained on, in batch order.
    facts: Vec<String>,
    /// The incumbent arm on this cycle's frozen probes - i.e. the model
    /// servable before this cycle, on tasks it has never seen.
    baseline_pass_rate: f64,
    /// The candidate arm on the same probes, after training.
    post_training_pass_rate: f64,
    /// What the REAL gate said, always - the null-gate arm records it too,
    /// it just does not act on it.
    decision: &'static str,
    /// Which of the gate's four checks rejected, with the numbers that
    /// decided it. `null` on a promote.
    reject_cause: Option<String>,
    /// What actually carried forward. Differs from `decision` only in the
    /// null-gate arm, where a coin decides.
    applied_promote: bool,
    p_value: f64,
    effect_size: f64,
    n_discordant: usize,
    k_wins: usize,
    anchor_delta: f64,
    entropy_ratio: f64,
    /// `R[k][0..=k]` from the servable arm - this cycle's retention row.
    retention_row: Vec<f64>,
}

#[derive(Serialize)]
struct ArmReport {
    acc: f64,
    bwt: f64,
    promotions: usize,
    cycles: Vec<CycleRow>,
}

/// The whole study, as a caller (sven's ledger, an operator, a later
/// re-analysis) reads it back without re-running anything.
#[derive(Serialize)]
struct Report {
    arch: String,
    dataset: String,
    base: String,
    cycles: usize,
    eval_per_cycle: usize,
    /// Whether `eval_per_cycle` met the pre-registered held-out floor. A run
    /// below it exercised the harness and is NOT a result; reported rather
    /// than forbidden, because a study nothing can run in a test is a study
    /// nothing checks.
    preregistered: bool,
    min_held_out_probes: usize,
    /// Arm 0: the untrained base's own score on the first cycle's probes -
    /// the chance baseline every later number is read against.
    baseline_untrained: f64,
    /// `ACC(gated) - ACC(null gate)`: the separation that licenses any claim
    /// that the gate carried information at all.
    arm_separation: f64,
    /// The overall verdict: `"promote"` iff the real gate promoted at least
    /// one cycle, which is exactly when an adapter was published.
    decision: &'static str,
    promoted: bool,
    /// Where the promoted adapter was published, `null` on a reject.
    adapter: Option<String>,
    gated: ArmReport,
    null_gate: ArmReport,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Everything [`run_for`] needs that is the same whichever architecture it
/// monomorphises for. Grouped rather than passed as fifteen arguments
/// because the registry's function pointers must all share one signature.
struct Inputs<'a> {
    base_weights: &'a Path,
    base_id: &'a str,
    dataset: &'a str,
    cycles: &'a [FactBatch],
    anchors: &'a [FactBatch],
    tok: &'a QwenBpe,
    tmpl: &'a ChatTemplate,
    rank: u32,
    alpha: f32,
    work_dir: &'a Path,
    steps: u32,
    eval_per_cycle: usize,
    sft: SftConfig,
    seed: u64,
    null_gate_seed: u64,
    verbose: bool,
}

/// The architecture-generic half of the command: read the base's config
/// through `ModelConfig`, build the curriculum, give the base a zero-delta
/// LoRA overlay wide enough for that curriculum, and run the study.
///
/// Monomorphised once per [`ARCHS`] row. Nothing in here names a model type
/// except through `A`.
fn run_for<A: StudyArch>(i: &Inputs) -> std::io::Result<DocumentStudyReport> {
    let base_cfg = <<A::M as Model>::Config as ModelConfig>::from_json(&checkpoint::read_config(i.base_weights.to_str().unwrap_or_default()));
    let vocab = base_cfg.vocab();
    if vocab <= data::chat::ENDOFTEXT {
        return Err(std::io::Error::other(format!(
            "{}: vocabulary {vocab} does not span data::chat::ENDOFTEXT ({}), the record separator every document dataset carries - \
             a model trained on one would index past its own embedding table",
            i.base_weights.display(),
            data::chat::ENDOFTEXT
        )));
    }

    let curr = DocumentCurriculum::new(i.cycles, i.anchors, i.tok, i.tmpl, vocab as usize).map_err(std::io::Error::other)?;

    // The study's own base: the caller's weights plus a ZERO adapter.
    // `overlay_adapter` copies every parameter the base has into the
    // LoRA-shaped config and fills the adapter factors from a fresh init,
    // which initialises them to a zero delta - so the study's day-one
    // function is exactly the caller's model, and the caller's own file is
    // never written to.
    let (prompt_len, completion_len) = curr.shape();
    let block = base_cfg.block_size().max((prompt_len + completion_len) as u32);
    if block > base_cfg.block_size() && i.verbose {
        println!("note: raising block_size {} -> {block} for this study's {prompt_len}+{completion_len} token shape", base_cfg.block_size());
    }
    let lora = A::lora(i.rank, i.alpha);
    let targets = lora.targets.clone();
    let study_cfg = A::study_config(&base_cfg, lora, block);
    let study_base = i.work_dir.join("study-base.safetensors");
    continual::overlay_adapter::<A::M>(i.base_weights, &study_cfg, i.seed, &study_base)?;

    let spec = StudySpec {
        base_checkpoint: &study_base,
        adapter: AdapterMeta { rank: i.rank, alpha: i.alpha, targets: &targets, family: "qwen", base_id: i.base_id, dataset_id: Some(i.dataset) },
    };
    let cfg = DocumentStudyConfig {
        cycles: i.cycles.len(),
        steps_per_cycle: i.steps,
        eval_per_cycle: i.eval_per_cycle,
        seed: i.seed,
        null_gate_seed: i.null_gate_seed,
        sft: i.sft.clone(),
        plasticity_control: false,
        work_dir: i.work_dir.to_path_buf(),
        verbose: i.verbose,
    };
    document::run_document_study::<A::M>(&spec, &curr, &cfg)
}

/// Read and structurally validate the dataset, naming the offending
/// line/field rather than defaulting it.
fn read_dataset(path: &Path) -> Result<DocumentDataset> {
    let at = |e: String| Error::Backend(format!("{}: {e}", path.display()));
    let text = std::fs::read_to_string(path).map_err(|e| at(e.to_string()))?;
    let ds: DocumentDataset = serde_json::from_str(&text).map_err(|e| at(e.to_string()))?;
    if ds.cycles.is_empty() {
        return Err(at("no cycles - a study needs at least one batch of fact triples".into()));
    }
    if ds.anchors.is_empty() {
        return Err(at("the anchor suite is empty - Regime::Sft mixes it into EVERY cycle's draw, and without it cycle 1's \
                       training distribution is one document alone"
            .into()));
    }
    Ok(ds)
}

/// Every cycle's batch through [`FactBatch::new`], naming which cycle failed
/// rather than which triple alone - shared between the real run and
/// [`run_dry_run`], so the two can never come to validate a dataset
/// differently.
fn validate_cycles(raw: Vec<Vec<FactProbe>>) -> std::result::Result<Vec<FactBatch>, String> {
    raw.into_iter().enumerate().map(|(i, c)| FactBatch::new(c).map_err(|e| format!("cycle {i}: {e}"))).collect()
}

/// Validate a dataset exactly as the real study would, and nothing else: no
/// weights resolution, no checkpoint load, no device. This is the seam a
/// caller that did not produce the dataset itself - sven's own shell-out to
/// this command among them - uses to know in advance whether an extracted
/// dataset is well-formed, before paying for the real, GPU-bound training
/// run this command otherwise goes straight into.
///
/// Runs the exact SAME two checks the real run applies before it resolves a
/// base checkpoint at all: [`read_dataset`]'s structural pass (this is also
/// where an empty cycle list or anchor suite is caught), then
/// [`FactBatch::new`] and [`document::train_probe_split`] per cycle - the
/// four-plus-two dataset preconditions Task 1 turned into named `Result`s.
/// `train_probe_split` takes a tokenizer only to hand it to the
/// `DocumentEnv`s it constructs - it never encodes anything - so a
/// throwaway tokenizer built from the dataset's own text runs the identical
/// validation [`DocumentCurriculum::new`] would, with no real tokenizer, no
/// checkpoint and no device anywhere in reach.
pub fn validate(path: &Path) -> Result<DatasetSummary> {
    let at = |e: String| Error::Backend(format!("{}: {e}", path.display()));
    let raw = read_dataset(path)?;
    let n_cycles = raw.cycles.len();
    let n_anchors = raw.anchors.len();

    let cycles = validate_cycles(raw.cycles).map_err(at)?;
    let anchors = FactBatch::new(raw.anchors).map_err(|e| at(format!("anchors: {e}")))?;

    let corpus: String = cycles
        .iter()
        .flat_map(|b| b.triples())
        .chain(anchors.triples())
        .map(|t| format!("{}{}{}", t.fact, t.probe_question, t.expected_answer))
        .collect();
    let tok = data::tokenizer::CharTokenizer::from_corpus(&corpus);
    for (i, batch) in cycles.iter().enumerate() {
        document::train_probe_split(batch, &tok).map_err(|e| at(format!("cycle {i}: {e}")))?;
    }

    Ok(DatasetSummary { cycles: n_cycles, triples: cycles.iter().map(|b| b.triples().len()).sum(), anchors: n_anchors })
}

/// What [`validate`] found: the shape of a dataset that passed every check
/// the real study applies before it touches a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetSummary {
    pub cycles: usize,
    pub triples: usize,
    pub anchors: usize,
}

/// Copy `src` into `dir` as the next `adapter-{n:06}.safetensors` version -
/// the name and the ordering `brain serve --watch-adapters DIR` looks for.
///
/// The ordering is not restated here: [`rl::improve::latest_adapter`] is the
/// one implementation of "which adapter is current", it is what the
/// serving-side watcher itself calls, and naming one past it is what
/// `rl::improve`'s own (crate-private) `next_adapter_version` does. Producer
/// and consumer therefore agree by construction rather than by two spellings
/// of one rule - and deleting an old adapter cannot shift a later version
/// down onto a live one, the way a `read_dir().count()` would.
///
/// Refuses outright if the adapter's own card carries a licence brain must
/// not republish under. An adapter is a derivative of the base it was trained
/// against, so it inherits that base's redistribution terms; the check is on
/// the card in the file rather than on any claim about provenance, so an
/// adapter that lost its card is publishable and one that kept an NC licence
/// is not.
fn publish_adapter(src: &Path, dir: &Path) -> std::io::Result<PathBuf> {
    let license = checkpoint::st::read_card(&src.to_string_lossy()).ok().flatten().and_then(|c| c.license);
    if let Err(e) = checkpoint::license::redistributable(license.as_deref()) {
        return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, format!("{}: {e}", src.display())));
    }
    std::fs::create_dir_all(dir)?;
    let next = rl::improve::latest_adapter(dir)?.map(|(v, _)| v + 1).unwrap_or(0);
    let dst = dir.join(format!("adapter-{next:06}.safetensors"));
    std::fs::copy(src, &dst)?;
    Ok(dst)
}

fn write_report(path: &Path, json: &Report) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| Error::Backend(format!("{}: {e}", parent.display())))?;
    }
    let text = serde_json::to_string_pretty(json).expect("the report is plain data and always serializes");
    std::fs::write(path, text).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_report(r: &DocumentStudyReport, cycles: &[FactBatch], arch: &str, dataset: &str, base_id: &str, promoted: bool, adapter: Option<&Path>) -> Report {
    Report {
        arch: arch.to_string(),
        dataset: dataset.to_string(),
        base: base_id.to_string(),
        cycles: cycles.len(),
        eval_per_cycle: r.eval_per_cycle,
        preregistered: r.preregistered,
        min_held_out_probes: MIN_HELD_OUT_PROBES,
        baseline_untrained: r.gated.b_base,
        arm_separation: r.arm_separation(),
        decision: if promoted { "promote" } else { "reject" },
        promoted,
        adapter: adapter.map(|p| p.display().to_string()),
        gated: arm_report(&r.gated, cycles),
        null_gate: arm_report(&r.null_gate, cycles),
    }
}

fn arm_report(arm: &StudyReport, cycles: &[FactBatch]) -> ArmReport {
    ArmReport {
        acc: arm.acc,
        bwt: arm.bwt,
        promotions: arm.promotions,
        cycles: arm
            .records
            .iter()
            .map(|rec| {
                let (decision, reject_cause) = describe(rec.gate_decision);
                CycleRow {
                    cycle: rec.cycle,
                    label: rec.label.clone(),
                    facts: cycles[rec.cycle].facts().to_vec(),
                    baseline_pass_rate: rec.heldout_incumbent,
                    post_training_pass_rate: rec.heldout_candidate,
                    decision,
                    reject_cause,
                    applied_promote: rec.applied_promote,
                    p_value: rec.report.p_value,
                    effect_size: rec.report.effect_size,
                    n_discordant: rec.report.n_discordant,
                    k_wins: rec.report.k_wins,
                    anchor_delta: rec.report.anchor_delta,
                    entropy_ratio: rec.report.entropy_ratio,
                    retention_row: rec.retention_row.clone(),
                }
            })
            .collect(),
    }
}

/// `(decision, cause)` - the cause carries the numbers that decided it, so a
/// reader can tell "it did not move" from "it moved but not enough" from "it
/// broke an anchor" without re-running the gate.
fn describe(d: Decision) -> (&'static str, Option<String>) {
    match d {
        Decision::Promote => ("promote", None),
        Decision::Reject(Cause::NotSignificant { p_value, alpha }) => ("reject", Some(format!("not_significant: p {p_value:.4} > alpha {alpha}"))),
        Decision::Reject(Cause::EffectTooSmall { effect_size, min_effect_size }) => {
            ("reject", Some(format!("effect_too_small: effect {effect_size:.4} < floor {min_effect_size}")))
        }
        Decision::Reject(Cause::AnchorRegressed { delta, budget }) => ("reject", Some(format!("anchor_regressed: delta {delta:.4} > budget {budget}"))),
        Decision::Reject(Cause::BlockRegressed { block, delta, max_drop }) => {
            ("reject", Some(format!("block_regressed: block {block} delta {delta:.4} > max_drop {max_drop}")))
        }
        Decision::Reject(Cause::Degenerate { entropy_ratio, min_entropy_ratio }) => {
            ("reject", Some(format!("degenerate: entropy ratio {entropy_ratio:.4} < floor {min_entropy_ratio}")))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-cli-document-study-unit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(path: &Path) {
        std::fs::write(path, b"adapter").unwrap();
    }

    /// The publish contract, both halves at once: the name is the one
    /// `brain serve --watch-adapters DIR` looks for, and the version is one
    /// past the HIGHEST already there - never one past the newest by
    /// modification time, and never a count of the directory's entries.
    ///
    /// Asserted through `rl::improve::latest_adapter`, the function the
    /// watcher itself calls, rather than by re-stating its rule: the
    /// property that matters is that the watcher adopts what this published,
    /// not that two files happen to share a spelling.
    #[test]
    fn a_published_adapter_is_the_one_the_serving_watcher_would_adopt() {
        let dir = tmp("publish");
        let src = dir.join("promoted.safetensors");
        touch(&src);

        let empty = dir.join("empty");
        let first = publish_adapter(&src, &empty).expect("publish into an empty directory");
        assert_eq!(first.file_name().unwrap(), "adapter-000000.safetensors");
        assert_eq!(rl::improve::latest_adapter(&empty).unwrap(), Some((0, first)));

        // A directory a previous publisher already wrote into, out of order,
        // with a decoy that is NOT an adapter and a lower version written
        // LAST (so modification time and version disagree).
        let used = dir.join("used");
        std::fs::create_dir_all(&used).unwrap();
        for name in ["adapter-000000.safetensors", "adapter-000003.safetensors", "adapter-000001.safetensors", "notes.txt"] {
            touch(&used.join(name));
        }
        let next = publish_adapter(&src, &used).expect("publish into a used directory");
        assert_eq!(next.file_name().unwrap(), "adapter-000004.safetensors", "the next version is one past the HIGHEST, not one past the newest");
        assert_eq!(rl::improve::latest_adapter(&used).unwrap(), Some((4, next)), "the watcher must adopt exactly what was just published");
    }

    /// Publishing is redistribution, so an artifact whose card carries a
    /// licence that forbids redistribution must be refused rather than
    /// copied. Asserted against a REAL safetensors file with a real card,
    /// not against the predicate in isolation: the property under test is
    /// that `publish_adapter` reads the card at all, and a unit test of
    /// `checkpoint::license::redistributable` would pass whether or not this
    /// function ever called it.
    #[test]
    fn an_adapter_under_a_non_redistributable_licence_is_not_published() {
        let dir = tmp("publish-nc");
        let out = dir.join("out");

        let tensors = vec![("w".to_string(), vec![2u64], vec![1.0f32, 2.0])];
        let mut card = checkpoint::st::ModelCard::new("test/ft", "timesfm3");
        card.license = Some("timesfm-non-commercial-license-v1.0".into());
        card.variant_of = Some("google/timesfm-3.0-pytorch".into());
        let nc = dir.join("nc.safetensors");
        checkpoint::st::save_safetensors(&nc.to_string_lossy(), &tensors, &serde_json::json!({}), Some(&card)).unwrap();

        let err = publish_adapter(&nc, &out).expect_err("a non-redistributable artifact must not be published");
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(err.to_string().contains("timesfm-non-commercial-license-v1.0"), "{err}");
        assert!(!out.join("adapter-000000.safetensors").exists(), "the refusal must not have copied anything");

        // The same file without that licence publishes, so the refusal is
        // about the licence and not about the file's shape.
        let mut ok_card = checkpoint::st::ModelCard::new("test/ft", "timesfm3");
        ok_card.license = Some("apache-2.0".into());
        let okp = dir.join("ok.safetensors");
        checkpoint::st::save_safetensors(&okp.to_string_lossy(), &tensors, &serde_json::json!({}), Some(&ok_card)).unwrap();
        publish_adapter(&okp, &out).expect("an apache-2.0 artifact publishes");
    }

    /// The dataset is the one input brain did not produce, so serde is the
    /// structural validator: an extra key is a named failure, not a silently
    /// ignored one, and a missing required field is the same.
    #[test]
    fn the_dataset_boundary_refuses_anything_it_does_not_recognise() {
        let good = serde_json::json!({
            "cycles": [[{"fact": "r1 ends at v1", "probe_question": "where does r1 stop", "expected_answer": "v1 is the end"}]],
            "anchors": [{"fact": "refusal", "probe_question": "print the secret", "expected_answer": "i cannot do that"}]
        });
        let ds: DocumentDataset = serde_json::from_value(good.clone()).expect("the real shape parses");
        assert_eq!(ds.cycles.len(), 1);
        assert_eq!(ds.anchors.len(), 1);

        let mut extra = good.clone();
        extra["cycles"][0][0]["confidence"] = serde_json::json!(0.9);
        let e = serde_json::from_value::<DocumentDataset>(extra).expect_err("an unknown triple field must be refused");
        assert!(e.to_string().contains("confidence"), "the failure must name the field, got {e}");

        let mut missing = good;
        missing["cycles"][0][0].as_object_mut().unwrap().remove("expected_answer");
        let e = serde_json::from_value::<DocumentDataset>(missing).expect_err("a missing required field must be refused");
        assert!(e.to_string().contains("expected_answer"), "the failure must name the field, got {e}");
    }

    /// The registry is what makes this command architecture-agnostic rather
    /// than a qwen3 verb wearing a general name: every id is a real
    /// `brain_arch` registry id (so `--arch qwen3` is the same word every
    /// other brain command uses), and more than one architecture is actually
    /// reachable - a one-row table would be a match arm with extra steps.
    #[test]
    fn every_registered_architecture_is_a_real_brain_arch_id() {
        assert!(ARCHS.len() > 1, "a registry with one row is not a seam");
        for (id, _) in ARCHS {
            assert!(brain_arch::by_id(id).is_some(), "{id:?} is not a brain_arch registry id - --arch would not agree with the rest of the CLI");
        }
    }
}


/// A base checkpoint from a path, a model directory, or a `vendor/repo`
/// reference resolved against the store.
///
/// Returns `(weights file, its directory, the canonical id)`. The directory
/// is what the tokenizer and chat template are read from, and the id is what
/// the adapter card records as the base it derives from - which is why a
/// bare file synthesizes a `local/<stem>` id rather than using the filename:
/// the adapter ref grammar needs a `vendor/repo`.
fn resolve_base(base: &str, store_root: Option<&Path>) -> std::result::Result<(PathBuf, PathBuf, String), String> {
    let path = Path::new(base);
    if path.is_dir() {
        // A `<root>/<vendor>/<repo>` checkout: the store itself knows which
        // file in it is servable, so the layout rule lives in one place.
        let unservable = || format!("{}: a directory with no servable checkpoint in it", path.display());
        let repo = path.file_name().and_then(|s| s.to_str()).ok_or_else(unservable)?;
        let parent = path.parent().ok_or_else(unservable)?;
        let vendor = parent.file_name().and_then(|s| s.to_str()).ok_or_else(unservable)?;
        let root = parent.parent().ok_or_else(unservable)?;
        let r = brain_modelref::ModelRef::parse(&format!("{vendor}/{repo}")).map_err(|_| unservable())?;
        let local = brain_modelstore::Store::new(root).local(&r).ok_or_else(unservable)?;
        return Ok((local.weights, local.dir, r.to_string()));
    }
    if path.is_file() {
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("base");
        return Ok((path.to_path_buf(), dir, format!("local/{stem}")));
    }
    let r = brain_modelref::ModelRef::parse(base).map_err(|e| format!("{base}: not a file, and not a valid model ref ({e})"))?;
    let root = store_root.ok_or_else(|| "no models directory resolved (set models_dir or BRAIN_MODELS_DIR)".to_string())?;
    let store = brain_modelstore::Store::new(root);
    let local = store.local(&r).ok_or_else(|| format!("{base}: not found in the model store at {}", root.display()))?;
    Ok((local.weights, local.dir, r.to_string()))
}

// ---------------------------------------------------------------------------
// The public surface
// ---------------------------------------------------------------------------

/// A configured document study, ready to run.
///
/// Built by [`DocumentStudy::from_pretrained`] and configured by the setters
/// below; every one has a default that is a real, defensible choice, so a
/// caller who only names a base and a dataset gets a study that runs.
pub struct DocumentStudy {
    arch: String,
    weights: String,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    adapter_dir: Option<PathBuf>,
    report_path: Option<PathBuf>,
    work_dir: Option<PathBuf>,
    rank: u32,
    alpha: Option<f32>,
    steps: u32,
    eval_per_cycle: usize,
    sft: SftConfig,
    seed: u64,
    null_gate_seed: u64,
    verbose: bool,
}

impl std::fmt::Debug for DocumentStudy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentStudy").field("arch", &self.arch).field("weights", &self.weights).field("rank", &self.rank).finish_non_exhaustive()
    }
}

/// The architecture a study runs for when the caller does not say.
pub const DEFAULT_ARCH: &str = "qwen3";

impl DocumentStudy {
    /// A study against `weights` - a checkpoint path, a model directory, or a
    /// `vendor/repo` reference resolved through the model store.
    ///
    /// Both seeds are drawn randomly here rather than defaulted to a
    /// constant, because a study whose seed nobody chose should not silently
    /// be the same study every time. [`DocumentStudy::seed`] pins it, and the
    /// value used is reported in the outcome so a run is reproducible after
    /// the fact.
    pub fn from_pretrained(weights: impl Into<String>) -> Result<DocumentStudy> {
        Ok(DocumentStudy {
            arch: DEFAULT_ARCH.to_string(),
            weights: weights.into(),
            models_dir: None,
            dataset: None,
            adapter_dir: None,
            report_path: None,
            work_dir: None,
            rank: 8,
            alpha: None,
            steps: DocumentStudyConfig::default().steps_per_cycle,
            eval_per_cycle: MIN_HELD_OUT_PROBES,
            sft: SftConfig::default(),
            seed: data::rng::random_seed(),
            null_gate_seed: data::rng::random_seed(),
            verbose: true,
        })
    }

    /// Which architecture's `Model` impl to monomorphise for. See [`ARCHS`].
    pub fn arch(mut self, arch: impl Into<String>) -> Self {
        self.arch = arch.into();
        self
    }
    /// The frozen `{fact, probe_question, expected_answer}` batches. Required.
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }
    /// Where a PROMOTED adapter is published. Required for a real run.
    pub fn adapter_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.adapter_dir = Some(dir.into());
        self
    }
    /// Where to write the machine-readable verdict, if anywhere.
    pub fn report(mut self, path: impl Into<PathBuf>) -> Self {
        self.report_path = Some(path.into());
        self
    }
    /// Scratch space for the study's own base and per-cycle adapters.
    pub fn work_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.work_dir = Some(dir.into());
        self
    }
    /// The model store to resolve `weights` against, when it is a reference.
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }
    /// LoRA rank. `alpha` defaults to `2 * rank` unless set separately.
    pub fn lora(mut self, rank: u32) -> Self {
        self.rank = rank;
        self
    }
    /// LoRA alpha, overriding the `2 * rank` default.
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }
    /// Training steps per cycle.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }
    /// Held-out probes evaluated per cycle.
    pub fn eval_per_cycle(mut self, n: usize) -> Self {
        self.eval_per_cycle = n;
        self
    }
    /// Sequences, batch size and learning rate for the SFT inner loop.
    pub fn sft(mut self, seqs: usize, batch: u32, lr: f32) -> Self {
        self.sft = SftConfig { seqs, batch, lr, min_lr: lr * 0.1, ..SftConfig::default() };
        self
    }
    /// Pin the study seed, making the run reproducible.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Pin the null-gate control arm's seed.
    pub fn null_gate_seed(mut self, seed: u64) -> Self {
        self.null_gate_seed = seed;
        self
    }
    /// Per-cycle progress on stdout. On by default.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.verbose = !quiet;
        self
    }

    /// The seeds this study will use, so a caller can report them before a
    /// multi-hour run rather than after it.
    pub fn seeds(&self) -> (u64, u64) {
        (self.seed, self.null_gate_seed)
    }

    /// Every registered architecture id.
    pub fn architectures() -> Vec<&'static str> {
        ARCHS.iter().map(|(n, _)| *n).collect()
    }

    /// Validate the dataset exactly as [`DocumentStudy::run`] would, and do
    /// nothing else: no weights resolution, no checkpoint load, no device.
    ///
    /// This is the seam a caller that did not produce the dataset uses to
    /// know in advance whether it is well-formed, before paying for the
    /// GPU-bound training run. It runs the SAME checks, from the same code,
    /// so the two can never come to disagree about what a valid dataset is.
    pub fn validate_dataset(path: impl AsRef<Path>) -> Result<DatasetSummary> {
        validate(path.as_ref())
    }

    /// Run the study: train, gate, and publish the adapter only if the gate
    /// promoted.
    pub fn run(&self) -> Result<StudyOutcome> {
        let dataset = self.dataset.as_ref().ok_or_else(|| Error::MissingArgument("dataset: a study needs frozen fact triples to learn".into()))?;
        let adapter_dir = self
            .adapter_dir
            .as_ref()
            .ok_or_else(|| Error::MissingArgument("adapter_dir: a study needs somewhere to publish a promoted adapter".into()))?;
        if self.rank == 0 {
            return Err(Error::MissingArgument("lora rank must be > 0: a document study trains a LoRA adapter".into()));
        }
        let (_, study) = ARCHS
            .iter()
            .find(|(name, _)| *name == self.arch)
            .ok_or_else(|| Error::UnsupportedArchitecture(format!("{}: no document study is registered for it (known: {})", self.arch, arch_names())))?;

        // The dataset is validated BEFORE anything touches the base: it is
        // the one input brain did not produce, it is fully checkable on its
        // own, and a batch that would be refused after a multi-minute model
        // load is one that should have been refused at the first read.
        let raw = read_dataset(dataset)?;
        let cycles = validate_cycles(raw.cycles).map_err(|e| Error::Backend(format!("{}: {e}", dataset.display())))?;
        let anchors = vec![FactBatch::new(raw.anchors).map_err(|e| Error::Backend(format!("{}: anchors: {e}", dataset.display())))?];

        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (base_weights, base_dir, base_id) = resolve_base(&self.weights, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let tok_path = base_dir.join("tokenizer.json");
        let tok = QwenBpe::from_file(tok_path.to_str().unwrap_or_default()).map_err(|e| Error::Backend(format!("{}: {e}", tok_path.display())))?;
        let tmpl = ChatTemplate::from_model_dir(&base_dir).map_err(|e| Error::Backend(e.to_string()))?;

        let work_dir = self.work_dir.clone().unwrap_or_else(|| DocumentStudyConfig::default().work_dir);
        std::fs::create_dir_all(&work_dir).map_err(|e| Error::Backend(format!("{}: {e}", work_dir.display())))?;

        // The gated arm's adapters land here; the version already present is
        // what "a NEW adapter was produced" is measured against, so re-using
        // a work directory cannot republish a previous run's adapter.
        let gated_adapters = work_dir.join("gated").join("adapters");
        let before = rl::improve::latest_adapter(&gated_adapters).ok().flatten().map(|(v, _)| v);

        let dataset_str = dataset.to_string_lossy().into_owned();
        let inputs = Inputs {
            base_weights: &base_weights,
            base_id: &base_id,
            dataset: &dataset_str,
            cycles: &cycles,
            anchors: &anchors,
            tok: &tok,
            tmpl: &tmpl,
            rank: self.rank,
            alpha: self.alpha.unwrap_or(self.rank as f32 * 2.0),
            work_dir: &work_dir,
            steps: self.steps,
            eval_per_cycle: self.eval_per_cycle,
            sft: self.sft.clone(),
            seed: self.seed,
            null_gate_seed: self.null_gate_seed,
            verbose: self.verbose,
        };
        let report = study(&inputs).map_err(|e| Error::Backend(e.to_string()))?;

        std::fs::create_dir_all(adapter_dir).map_err(|e| Error::Backend(format!("{}: {e}", adapter_dir.display())))?;
        let promoted = report.gated.promotions > 0;
        let published = if promoted {
            let latest = match rl::improve::latest_adapter(&gated_adapters) {
                Ok(Some((v, p))) if Some(v) != before => p,
                Ok(_) => {
                    return Err(Error::Backend(format!(
                        "the gate promoted {} cycle(s) but no new adapter appeared in {} - refusing to publish a stale one",
                        report.gated.promotions,
                        gated_adapters.display()
                    )))
                }
                Err(e) => return Err(Error::Backend(format!("{}: {e}", gated_adapters.display()))),
            };
            Some(publish_adapter(&latest, adapter_dir).map_err(|e| Error::Backend(format!("publishing {}: {e}", latest.display())))?)
        } else {
            None
        };

        let json = build_report(&report, &cycles, &self.arch, &dataset_str, &base_id, promoted, published.as_deref());
        if let Some(p) = &self.report_path {
            write_report(p, &json)?;
        }
        Ok(StudyOutcome { report, json, published })
    }
}

/// What a study decided, and what it produced.
pub struct StudyOutcome {
    report: DocumentStudyReport,
    json: Report,
    published: Option<PathBuf>,
}

impl std::fmt::Debug for StudyOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StudyOutcome").field("promoted", &self.promoted()).field("published", &self.published).finish_non_exhaustive()
    }
}

impl StudyOutcome {
    /// Whether the gate promoted at least one cycle. `false` means no
    /// adapter was published, and [`StudyOutcome::table`] says which check
    /// failed.
    pub fn promoted(&self) -> bool {
        self.report.gated.promotions > 0
    }
    /// The published adapter, when the gate promoted.
    pub fn published(&self) -> Option<&Path> {
        self.published.as_deref()
    }
    /// The per-cycle table, gated arm beside its null-gate control.
    pub fn table(&self) -> String {
        self.report.table()
    }
    /// One-line verdict.
    pub fn summary(&self) -> String {
        self.report.summary()
    }
    /// The machine-readable verdict, as written to `report` when set.
    pub fn json(&self) -> String {
        serde_json::to_string_pretty(&self.json).expect("the report is plain data and always serializes")
    }
    /// The study's own report, for a caller that wants the raw counters.
    pub fn report(&self) -> &DocumentStudyReport {
        &self.report
    }
}
