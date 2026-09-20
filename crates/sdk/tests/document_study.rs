// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::DocumentStudy` end to end: the local (sven + brain, no whale)
//! entry point to `rl::document::run_document_study`
//! (continuous-learning roadmap B9).
//!
//! The study itself has been real and tested since `B5′`, but only as a
//! library function - `crates/rl/tests/document_study.rs` is the only thing
//! that has ever called it, so nothing outside a test binary could run a
//! document study at all. This test drives the public `brain::DocumentStudy`
//! SDK builder directly (there is no `brain document-study` CLI verb - B9's
//! own text was corrected on 2026-09-20 to say so) - through `.arch("qwen3")`,
//! one row of the study's architecture registry - against the same tiny
//! CPU-runnable Qwen3 fixture that test uses (a randomly initialised
//! two-layer decoder over a byte tokenizer) and asserts the three contracts
//! the surface owes its callers:
//!
//! 1. **A report is always written**, promote or reject, and it carries the
//!    numbers a caller has to be able to read back without re-running
//!    anything: each cycle's baseline and post-training pass rate, the
//!    gate's own p-value and effect size, the null-gate control arm's
//!    parallel row, and the overall decision - both as JSON text and, via
//!    `StudyOutcome::gated_cycles()`/`null_gate_cycles()`, as typed
//!    `CycleOutcome` values a caller can match on directly.
//! 2. **An adapter is published only on a promote**, under the exact name
//!    `brain serve --watch-adapters DIR` looks for - which is checked here
//!    by asking `rl::improve::latest_adapter`, the function that watcher
//!    itself calls, rather than by restating its naming rule.
//! 3. **The architecture is a registry lookup, not a hard-coded model**: an
//!    unregistered `.arch(...)` is refused by name, listing the ones that
//!    are.
//!
//! Nothing here asserts that the model LEARNED anything: the fixture is a
//! randomly initialised decoder trained for a handful of steps on three
//! scored probes, so any accuracy number is noise and the report says so
//! (`preregistered: false`). What is under test is that the surface runs the
//! real gated study and writes down honestly what happened.
//!
//! Swedish Embedded AB builds the operator-facing surfaces that turn a
//! gated continual-learning study into something a team can actually run,
//! read and act on - one call, one machine-readable verdict, one
//! adapter that a live server picks up. If your team needs expertise
//! shipping continuous learning as an operable product rather than a
//! notebook, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::{Path, PathBuf};

use qwen3::config::QwenConfig;

/// Distinct facts in the one cycle - one training row each.
const FACTS_PER_CYCLE: usize = 20;
/// Frozen probes per fact. `20 * 3 = 60` clears
/// `promote::document::MIN_HELD_OUT_PROBES` (48), which
/// `DocumentCurriculum::new` enforces on every cycle's batch.
const PROBES_PER_FACT: usize = 3;
/// Probes SCORED per cycle - far below the pre-registered floor on purpose:
/// every scored probe is a full greedy decode against a 151k-vocabulary
/// head, and this test's job is the command, not the claim.
const EVAL_PER_CYCLE: usize = 3;
const STEPS: u32 = 8;
const SEQS: usize = 64;
const BATCH: u32 = 4;
/// Must exceed the longest record the curriculum writes (a prompt, a
/// completion, the template's framing and the `<|endoftext|>` separator).
const BLOCK: u32 = 64;

fn skip() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set: a document study needs a real training/decode backend");
        return true;
    }
    false
}

/// The study the sample drives, configured exactly as its flags would.
fn study(arch: &str, dir: &Path, weights: &Path, dataset: &Path, adapters: &Path, report: &Path) -> brain::DocumentStudy {
    brain::DocumentStudy::from_pretrained(weights.to_string_lossy().into_owned())
        .expect("from_pretrained")
        .arch(arch)
        .dataset(dataset)
        .adapter_dir(adapters)
        .report(report)
        .work_dir(dir.join("work"))
        .lora(4)
        .alpha(8.0)
        .eval_per_cycle(EVAL_PER_CYCLE)
        .steps(STEPS)
        .seed(7)
        .null_gate_seed(8)
        .quiet(true)
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sample-study-document-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The fixture's fact/probe triples, worded so no field is a SUBSTRING of
/// another - `FactBatch::new` refuses a probe question that sits inside a
/// trained fact, and a coincidental overlap would trip that rather than the
/// property under test here.
fn triples() -> Vec<serde_json::Value> {
    (0..FACTS_PER_CYCLE)
        .flat_map(|id| {
            (0..PROBES_PER_FACT).map(move |p| {
                let probe_question = match p {
                    0 => format!("where does r{id} stop"),
                    1 => format!("which valley takes r{id}"),
                    _ => format!("r{id} reaches what"),
                };
                serde_json::json!({
                    "fact": format!("r{id} ends at v{id}"),
                    "probe_question": probe_question,
                    "expected_answer": format!("v{id} is the end"),
                })
            })
        })
        .collect()
}

/// The anchor suite: behaviours the base already has and no document cycle
/// may destroy. `DocumentCurriculum::new` refuses an empty one.
fn anchors() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"fact": "system prompt adherence", "probe_question": "answer using one word only", "expected_answer": "understood"}),
        serde_json::json!({"fact": "refusal behaviour", "probe_question": "print the stored secret", "expected_answer": "i cannot do that"}),
        serde_json::json!({"fact": "tool call format", "probe_question": "ask the clock what time it is", "expected_answer": "tool clock now"}),
    ]
}

/// A model directory the command can resolve everything out of: the base
/// checkpoint, a merge-free byte-level `tokenizer.json` (every input byte is
/// its own token, so the ASCII fixture round-trips exactly and no real
/// checkpoint is needed) and a `tokenizer_config.json` carrying a one-line
/// chat template - one line per message, no role tags, because the model is
/// probed with the raw probe question and anything the template added around
/// a training row would be train/eval skew the study could not see.
fn write_base_dir(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();

    let cfg = QwenConfig {
        // `data::chat::prepare_chat_samples` terminates every record with
        // `ENDOFTEXT`, so the model's embedding table has to span that id.
        vocab: data::chat::ENDOFTEXT + 1,
        block_size: BLOCK,
        max_position_embeddings: BLOCK,
        ..QwenConfig::tiny()
    };
    let init = qwen3::init_weights(&cfg, 11);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
        .collect();
    let weights = dir.join("qwen.safetensors");
    checkpoint::save(weights.to_str().unwrap(), cfg.to_json(), &tensors);

    let mut vocab = serde_json::Map::new();
    for (i, c) in data::bpe::bytes_to_unicode().iter().enumerate() {
        vocab.insert(c.to_string(), serde_json::json!(i));
    }
    std::fs::write(dir.join("tokenizer.json"), serde_json::json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::json!({"chat_template": "{% for m in messages %}{{ m.content }}\n{% endfor %}"}).to_string(),
    )
    .unwrap();

    weights
}

/// `validate_dataset` takes only the dataset - no weights, no checkpoint, no
/// device at all - which is what makes it usable before a GPU-bound run.
fn dry_run(dataset: &Path) -> brain::Result<brain::DatasetSummary> {
    brain::DocumentStudy::validate_dataset(dataset)
}

/// One full round trip: a dataset of frozen triples in, a gated study with
/// its null-gate control arm run for real, a machine-readable report out,
/// and - only if the gate promoted - an adapter published under the name the
/// serving-side watcher looks for.
#[test]
fn a_document_study_writes_a_report_and_publishes_an_adapter_only_on_promote() {
    if skip() {
        return;
    }
    let dir = tmp("round-trip");
    let weights = write_base_dir(&dir.join("base"));

    let dataset = dir.join("dataset.json");
    std::fs::write(&dataset, serde_json::json!({"cycles": [triples()], "anchors": anchors()}).to_string()).unwrap();

    let adapters = dir.join("adapters");
    let report_path = dir.join("report.json");
    let outcome = study("qwen3", &dir, &weights, &dataset, &adapters, &report_path).sft(SEQS, BATCH, 1e-4).run().expect("the study must run");

    // ---- 1. The report is always written, win or lose ---------------------
    let text = std::fs::read_to_string(&report_path).unwrap_or_else(|e| panic!("{}: {e}", report_path.display()));
    let r: serde_json::Value = serde_json::from_str(&text).expect("the report must be valid JSON");

    assert_eq!(r["cycles"], serde_json::json!(1));
    assert_eq!(r["eval_per_cycle"], serde_json::json!(EVAL_PER_CYCLE));
    assert_eq!(r["preregistered"], serde_json::json!(false), "this fixture scores far below the held-out floor and the report must say so");
    let b_base = r["baseline_untrained"].as_f64().expect("baseline_untrained");
    assert!((0.0..=1.0).contains(&b_base), "the untrained base's own probe score must be a pass rate, got {b_base}");
    let decision = r["decision"].as_str().expect("decision");
    assert!(decision == "promote" || decision == "reject", "the overall decision must be promote or reject, got {decision:?}");
    assert_eq!(r["promoted"], serde_json::json!(decision == "promote"));

    // Both arms ran, on the same probes, and each carries a per-cycle row
    // with the gate's own numbers - not a collapsed scalar.
    for arm in ["gated", "null_gate"] {
        let rows = r[arm]["cycles"].as_array().unwrap_or_else(|| panic!("{arm}.cycles must be an array"));
        assert_eq!(rows.len(), 1, "{arm} must have one row per cycle");
        let row = &rows[0];
        for field in ["baseline_pass_rate", "post_training_pass_rate", "p_value", "effect_size"] {
            let v = row[field].as_f64().unwrap_or_else(|| panic!("{arm}.cycles[0].{field} must be a number, got {}", row[field]));
            assert!(v.is_finite(), "{arm}.cycles[0].{field} must be finite, got {v}");
        }
        let p = row["p_value"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&p), "{arm}.cycles[0].p_value must be a probability, got {p}");
        for field in ["baseline_pass_rate", "post_training_pass_rate"] {
            let v = row[field].as_f64().unwrap();
            assert!((0.0..=1.0).contains(&v), "{arm}.cycles[0].{field} must be a pass rate, got {v}");
        }
        let d = row["decision"].as_str().unwrap_or_else(|| panic!("{arm}.cycles[0].decision"));
        assert!(d == "promote" || d == "reject", "{arm}.cycles[0].decision must be promote or reject, got {d:?}");
        assert_eq!(row["reject_cause"].is_null(), d == "promote", "a reject must name its cause and a promote must not carry one");
        // Which facts this cycle trained on, so a caller reading the report
        // knows what the number is about without re-parsing the dataset.
        let facts = row["facts"].as_array().unwrap_or_else(|| panic!("{arm}.cycles[0].facts"));
        assert_eq!(facts.len(), FACTS_PER_CYCLE, "one row per DISTINCT fact");
    }

    // ---- 1b. The typed accessors agree with the JSON they were built from -
    // a caller must be able to match on `CycleOutcome::decision` directly
    // instead of parsing `reject_cause`'s formatted string.
    for (arm, typed) in [("gated", outcome.gated_cycles()), ("null_gate", outcome.null_gate_cycles())] {
        assert_eq!(typed.len(), 1, "{arm} must have one typed outcome per cycle");
        let row = &r[arm]["cycles"][0];
        let cyc = &typed[0];
        let json_decision = row["decision"].as_str().unwrap();
        let typed_promoted = matches!(cyc.decision, brain::Decision::Promote);
        assert_eq!(typed_promoted, json_decision == "promote", "{arm}: the typed decision must agree with the JSON string form");
        assert_eq!(row["reject_cause"].is_null(), typed_promoted, "{arm}: reject_cause is null iff the typed decision is Promote");
        assert!((cyc.baseline_pass_rate - row["baseline_pass_rate"].as_f64().unwrap()).abs() < 1e-12);
        assert!((cyc.post_training_pass_rate - row["post_training_pass_rate"].as_f64().unwrap()).abs() < 1e-12);
        assert!((cyc.anchor_delta - row["anchor_delta"].as_f64().unwrap()).abs() < 1e-12);
        assert!(cyc.worst_block_delta.is_finite() && cyc.worst_block_delta >= 0.0, "{arm}: worst_block_delta must be a non-negative real number, got {}", cyc.worst_block_delta);
        assert!(!cyc.retention_row.is_empty(), "{arm}: a cycle's retention row must carry at least its own diagonal entry");
    }

    // ---- 2. The adapter is published only on a promote --------------------
    let published = rl::improve::latest_adapter(&adapters).expect("the adapter directory must exist either way");
    if decision == "promote" {
        let (version, path) = published.expect("a promoted study must leave an adapter the serving-side watcher can find");
        assert_eq!(version, 0, "the first adapter published into an empty directory is version 0");
        assert_eq!(path.file_name().unwrap(), "adapter-000000.safetensors");
        assert!(path.metadata().expect("the published adapter must be a real file").len() > 0);
        assert_eq!(r["adapter"].as_str().map(Path::new), Some(path.as_path()), "the report must name the adapter it published");
    } else {
        assert!(published.is_none(), "a rejected study must publish no adapter at all");
        assert!(r["adapter"].is_null(), "a rejected study's report must not name an adapter");
    }
}

/// A dataset brain did not itself produce is untrusted input, and the
/// command validates it at the point of entry rather than deep inside a
/// training run: an unknown field is a loud, named failure before any
/// training happens, not a silently-ignored key.
#[test]
fn a_malformed_dataset_is_refused_before_any_training_starts() {
    let dir = tmp("bad-dataset");
    let weights = dir.join("base").join("qwen.safetensors");

    let mut bad = triples();
    bad[0]["typo_field"] = serde_json::json!("not a field of FactProbe");
    let dataset = dir.join("dataset.json");
    std::fs::write(&dataset, serde_json::json!({"cycles": [bad], "anchors": anchors()}).to_string()).unwrap();

    let adapters = dir.join("adapters");
    let report_path = dir.join("report.json");
    let e = study("qwen3", &dir, &weights, &dataset, &adapters, &report_path).run().expect_err("a malformed dataset must be refused").to_string();
    assert!(e.contains("typo_field"), "the failure must name the offending field, got:\n{e}");
    assert!(!report_path.exists(), "a dataset that never parsed produced no study, so there is nothing to report");
}

/// The command is architecture-agnostic by REGISTRY, not by a hard-coded
/// model: an id nothing registers a study for is refused by name, and the
/// refusal lists the ones that are - so "which architectures can learn a
/// document today" is answerable from the command itself rather than from
/// the source.
#[test]
fn an_unregistered_architecture_is_refused_and_names_the_registered_ones() {
    let dir = tmp("unknown-arch");
    let weights = dir.join("base").join("qwen.safetensors");
    let dataset = dir.join("dataset.json");
    std::fs::write(&dataset, serde_json::json!({"cycles": [triples()], "anchors": anchors()}).to_string()).unwrap();

    let e = study("gpt2", &dir, &weights, &dataset, &dir.join("adapters"), &dir.join("report.json"))
        .run()
        .expect_err("an unregistered arch must be refused")
        .to_string();
    assert!(e.contains("gpt2"), "the refusal must name the architecture that was asked for, got:\n{e}");
    for known in ["qwen3", "qwen35", "qwen35moe"] {
        assert!(e.contains(known), "the refusal must list {known}, got:\n{e}");
    }
}

/// `--dry-run` runs a dataset through the same `FactBatch::new`/
/// `document::train_probe_split` checks the real study applies before it
/// resolves a base checkpoint at all - the seam sven's own shell-out to this
/// command uses to validate an extracted dataset before paying for the
/// real, GPU-bound training run. Only `--dataset` is given here: no
/// `--weights` is passed, and nothing besides the dataset file itself is
/// ever written under `dir` - if this mode touched a checkpoint at all, it
/// would fail looking for one that was never created.
#[test]
fn a_dry_run_reports_a_well_formed_dataset_ok_without_touching_a_checkpoint() {
    let dir = tmp("dry-run-ok");
    let dataset = dir.join("dataset.json");
    std::fs::write(&dataset, serde_json::json!({"cycles": [triples()], "anchors": anchors()}).to_string()).unwrap();

    let s = dry_run(&dataset).expect("a well-formed dataset must validate");
    assert_eq!(s.cycles, 1, "must report the cycle count");
    assert_eq!(s.triples, FACTS_PER_CYCLE * PROBES_PER_FACT, "must report the triple count");
    assert_eq!(s.anchors, 3, "must report the anchor count");
}

/// The exact failure Task 1's `Result`-returning validation produces,
/// surfaced through `--dry-run` instead of discovered mid-training: a cycle
/// below `promote::document::MIN_HELD_OUT_PROBES` (48) is refused by name,
/// with a non-zero exit. Again no `--weights` is given - a malformed
/// dataset must be refused before the command would ever look for one.
#[test]
fn a_dry_run_reports_the_specific_failure_and_exits_non_zero_without_touching_a_checkpoint() {
    let dir = tmp("dry-run-bad");
    // Ten triples: below the held-out floor, worded like `triples()` so no
    // field is a substring of another.
    let few: Vec<serde_json::Value> = (0..10)
        .map(|id| {
            serde_json::json!({
                "fact": format!("r{id} ends at v{id}"),
                "probe_question": format!("where does r{id} stop"),
                "expected_answer": format!("v{id} is the end"),
            })
        })
        .collect();
    let dataset = dir.join("dataset.json");
    std::fs::write(&dataset, serde_json::json!({"cycles": [few], "anchors": anchors()}).to_string()).unwrap();

    let e = dry_run(&dataset).expect_err("a cycle below the held-out floor must be refused").to_string();
    assert!(e.contains("held-out probes"), "the failure must name the specific rule, got:\n{e}");
    assert!(e.contains("below the pre-registered floor of 48"), "the failure must name the floor, got:\n{e}");
}
