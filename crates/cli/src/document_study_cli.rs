// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain document-study` - run the gated document-learning study on a batch
//! of frozen `{fact, probe_question, expected_answer}` triples, write a
//! machine-readable verdict, and publish the adapter only if the gate
//! promoted (continuous-learning roadmap B9).
//!
//! ```text
//! brain document-study --arch <name> --weights BASE --dataset FILE.json
//!                      --adapter-dir DIR --report FILE.json
//!                      [--work-dir DIR --lora RANK --alpha A
//!                       --eval-per-cycle N --steps N --seqs N --batch B
//!                       --lr X --seed S --null-gate-seed S
//!                       --models-dir DIR --quiet]
//! ```
//!
//! ## Why this command exists
//!
//! [`rl::document::run_document_study`] has been a real, tested study since
//! `B5′` - it trains a LoRA adapter on a batch of facts under
//! `Regime::Sft`, gates promotion on `promote::document::
//! document_gate_config`'s pre-registered bar, and runs the null-gate control
//! arm beside it so the resulting number says something about the gate. But
//! it was reachable only from a Rust test: nothing shipped could run one.
//!
//! ## Why it is TOP-LEVEL and `--arch`-driven, not a qwen3 subcommand
//!
//! `rl::continual::run_study` is generic over `M: model::Model`, and
//! `rl::document::DocumentCurriculum` names no model type at all - the study
//! is architecture-agnostic machinery, and nesting its only entry point under
//! one model's verb tree would tie a general capability to one consumer and
//! invite the next consumer to grow a second copy. So this follows
//! `brain bench eval --arch <name>`'s shape instead: one top-level command,
//! an architecture NAME that selects which `Model` impl it monomorphises for,
//! and a registry ([`ARCHS`]) that a new architecture joins by adding one row
//! - the same "add one line" seam `bench::arch` documents.
//!
//! Everything a study needs beyond `M: Model` lives in [`StudyArch`], which
//! is exactly two things: how that architecture spells a LoRA overlay of the
//! requested rank, and how to widen its context to the curriculum's shape.
//! Nothing else here is architecture-specific - the checkpoint is read
//! through `ModelConfig::from_json`, the study base is built by the generic
//! `rl::continual::overlay_adapter`, and the study itself by
//! `rl::document::run_document_study`.
//!
//! **What bounds the registry today is the TOKENIZER, not the model.**
//! `DocumentCurriculum` is generic over the model but hard-wired to
//! `data::qwen_tokenizer::QwenBpe` (its `Env`/`Ver` associated types name
//! that type, and `data::chat::prepare_chat_samples` takes it), so an
//! architecture qualifies iff it loads an HF `tokenizer.json` BPE and its
//! config can carry a LoRA overlay. That is why the registered rows are the
//! Qwen-family decoders and a GPT-tokenizer architecture is absent - a fact
//! about `DocumentCurriculum`'s signature, not a choice made here.
//!
//! ## What it does NOT do
//!
//! It computes nothing the study did not already compute. Every number in
//! the report comes from the two [`rl::continual::StudyReport`]s
//! `run_document_study` returns, and the adapter it publishes is the file
//! `rl::improve::cycle` already wrote when the gate promoted - copied, not
//! retrained. A command that re-scored anything here would be reporting a
//! different measurement from the one the gate actually made.
//!
//! Per-FACT rows (roadmap `B8`) are deliberately absent: a per-fact verdict
//! needs the candidate arm's own `(task id, score)` pairs
//! (`promote::document::fact_verdicts`), and `rl::continual::CycleRecord`
//! collapses those to means before the study report is built. Reconstructing
//! them here would mean a second decode pass - a different measurement from
//! the gate's, presented as if it were the gate's. The report names each
//! cycle's facts instead, so a caller knows what the cycle's number is about.
//!
//! Swedish Embedded AB builds the operator-facing surfaces that turn a gated
//! continual-learning study into something a team can actually run, read and
//! act on - one command, one machine-readable verdict, one adapter a live
//! server picks up. If your team needs expertise shipping continuous learning
//! as an operable product rather than a notebook, you can procure our
//! services by sending an email to info@swedishembedded.com.

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

const USAGE: &str = "usage: brain document-study --arch NAME --weights BASE --dataset FILE.json --adapter-dir DIR --report FILE.json \
     [--work-dir DIR --lora RANK --alpha A --eval-per-cycle N --steps N --seqs N --batch B --lr X --seed S --null-gate-seed S \
      --models-dir DIR --quiet]";

/// The architecture a study runs against when `--arch` is omitted. Named,
/// not silent: the study's measured recipe (`SftConfig::default`) was tuned
/// on this family.
const DEFAULT_ARCH: &str = "qwen3";

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

pub fn run(args: &[String]) {
    let mut a = crate::args::Args::new(args);
    let arch = a.str_or("--arch", DEFAULT_ARCH);
    let weights = a.take_str("--weights").unwrap_or_default();
    let dataset = a.take_str("--dataset").unwrap_or_default();
    let adapter_dir = a.take_str("--adapter-dir").unwrap_or_default();
    let report_path = a.take_str("--report").unwrap_or_default();
    let work_dir = a.take_str("--work-dir");
    let rank = a.u32_or("--lora", 8);
    let alpha = a.f32_or("--alpha", rank as f32 * 2.0);
    let eval_per_cycle = a.usize_or("--eval-per-cycle", MIN_HELD_OUT_PROBES);
    let steps = a.u32_or("--steps", DocumentStudyConfig::default().steps_per_cycle);
    let seqs = a.usize_or("--seqs", SftConfig::default().seqs);
    let batch = a.u32_or("--batch", SftConfig::default().batch);
    let lr = a.f32_or("--lr", SftConfig::default().lr);
    let seed = a.u64_or("--seed", DocumentStudyConfig::default().seed);
    let null_gate_seed = a.u64_or("--null-gate-seed", DocumentStudyConfig::default().null_gate_seed);
    let models_dir = a.take_str("--models-dir");
    let quiet = a.take_flag("--quiet");
    a.finish();

    if weights.is_empty() || dataset.is_empty() || adapter_dir.is_empty() || report_path.is_empty() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    if rank == 0 {
        eprintln!("--lora RANK must be > 0: a document study trains a LoRA adapter");
        std::process::exit(2);
    }
    let Some((_, study)) = ARCHS.iter().find(|(name, _)| *name == arch) else {
        eprintln!("--arch {arch:?}: no document study is registered for it (known: {})", arch_names());
        std::process::exit(2);
    };

    // ---- The dataset is validated BEFORE anything touches the base -------
    //
    // It is the one input brain did not produce, and it is fully checkable
    // on its own: a batch that would be refused after a multi-minute model
    // load is a batch that should have been refused at the first read.
    let raw = read_dataset(Path::new(&dataset));
    let cycles: Vec<FactBatch> = raw.cycles.into_iter().map(FactBatch::new).collect();
    let anchors = vec![FactBatch::new(raw.anchors)];

    // ---- The base, its tokenizer and its chat template -------------------
    let store_root = crate::model_dir::resolve(models_dir.as_deref());
    let (base_weights, base_dir, base_id) = match crate::qwen_cli::resolve_base(&weights, store_root.as_deref()) {
        Ok(t) => t,
        Err(e) => fail(&e),
    };
    let tok_path = base_dir.join("tokenizer.json");
    let tok = match QwenBpe::from_file(tok_path.to_str().unwrap_or_default()) {
        Ok(t) => t,
        Err(e) => fail(&format!("{}: {e}", tok_path.display())),
    };
    let tmpl = match ChatTemplate::from_model_dir(&base_dir) {
        Ok(t) => t,
        Err(e) => fail(&format!("{e}")),
    };

    let work_dir = work_dir.map(PathBuf::from).unwrap_or_else(|| DocumentStudyConfig::default().work_dir);
    if let Err(e) = std::fs::create_dir_all(&work_dir) {
        fail(&format!("{}: {e}", work_dir.display()));
    }
    // The gated arm's adapters land here; the version already present is
    // what "a NEW adapter was produced" is measured against, so re-using a
    // work directory cannot republish a previous run's adapter.
    let gated_adapters = work_dir.join("gated").join("adapters");
    let before = rl::improve::latest_adapter(&gated_adapters).ok().flatten().map(|(v, _)| v);

    let inputs = Inputs {
        base_weights: &base_weights,
        base_id: &base_id,
        dataset: &dataset,
        cycles: &cycles,
        anchors: &anchors,
        tok: &tok,
        tmpl: &tmpl,
        rank,
        alpha,
        work_dir: &work_dir,
        steps,
        eval_per_cycle,
        sft: SftConfig { seqs, batch, lr, min_lr: lr * 0.1, ..SftConfig::default() },
        seed,
        null_gate_seed,
        verbose: !quiet,
    };
    let report = match study(&inputs) {
        Ok(r) => r,
        Err(e) => fail(&format!("{e}")),
    };
    println!("{}", report.table());
    println!("{}", report.summary());

    // ---- Publish, but only on a real promote -----------------------------
    let adapter_dir = Path::new(&adapter_dir);
    if let Err(e) = std::fs::create_dir_all(adapter_dir) {
        fail(&format!("{}: {e}", adapter_dir.display()));
    }
    let promoted = report.gated.promotions > 0;
    let published = if promoted {
        let latest = match rl::improve::latest_adapter(&gated_adapters) {
            Ok(Some((v, p))) if Some(v) != before => p,
            Ok(_) => fail(&format!(
                "the gate promoted {} cycle(s) but no new adapter appeared in {} - refusing to publish a stale one",
                report.gated.promotions,
                gated_adapters.display()
            )),
            Err(e) => fail(&format!("{}: {e}", gated_adapters.display())),
        };
        match publish_adapter(&latest, adapter_dir) {
            Ok(p) => {
                println!("promoted: published {}", p.display());
                Some(p)
            }
            Err(e) => fail(&format!("publishing {}: {e}", latest.display())),
        }
    } else {
        println!("rejected: no adapter published (the report says which check failed)");
        None
    };

    let json = build_report(&report, &cycles, &arch, &dataset, &base_id, promoted, published.as_deref());
    write_report(Path::new(&report_path), &json);
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

    let curr = DocumentCurriculum::new(i.cycles, i.anchors, i.tok, i.tmpl, vocab as usize);

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

/// Read and structurally validate the dataset, exiting with the offending
/// line/field named rather than a default.
fn read_dataset(path: &Path) -> DocumentDataset {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => fail(&format!("{}: {e}", path.display())),
    };
    let ds: DocumentDataset = match serde_json::from_str(&text) {
        Ok(d) => d,
        Err(e) => fail(&format!("{}: {e}", path.display())),
    };
    if ds.cycles.is_empty() {
        fail(&format!("{}: no cycles - a study needs at least one batch of fact triples", path.display()));
    }
    if ds.anchors.is_empty() {
        fail(&format!(
            "{}: the anchor suite is empty - Regime::Sft mixes it into EVERY cycle's draw, and without it cycle 1's training \
             distribution is one document alone",
            path.display()
        ));
    }
    ds
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
fn publish_adapter(src: &Path, dir: &Path) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let next = rl::improve::latest_adapter(dir)?.map(|(v, _)| v + 1).unwrap_or(0);
    let dst = dir.join(format!("adapter-{next:06}.safetensors"));
    std::fs::copy(src, &dst)?;
    Ok(dst)
}

fn write_report(path: &Path, json: &Report) {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        if let Err(e) = std::fs::create_dir_all(parent) {
            fail(&format!("{}: {e}", parent.display()));
        }
    }
    let text = serde_json::to_string_pretty(json).expect("the report is plain data and always serializes");
    if let Err(e) = std::fs::write(path, text) {
        fail(&format!("{}: {e}", path.display()));
    }
    println!("report: {}", path.display());
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
        Decision::Reject(Cause::Degenerate { entropy_ratio, min_entropy_ratio }) => {
            ("reject", Some(format!("degenerate: entropy ratio {entropy_ratio:.4} < floor {min_entropy_ratio}")))
        }
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("brain document-study: {msg}");
    std::process::exit(1);
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
