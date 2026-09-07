// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The document `Curriculum` and the study that drives it
//! (continuous-learning roadmap B5'): a batch of frozen
//! `{fact, probe_question, expected_answer}` triples per cycle, trained as a
//! masked chat dataset under `Regime::Sft`, gated by B2's pre-registered
//! document gate, with the control arms that make the resulting number mean
//! anything.
//!
//! ## What these tests assert, and what they deliberately do not
//!
//! That the harness RUNS end to end and emits a well-formed retention matrix
//! from real training, real greedy decodes and a real gate; and that the
//! dataset the curriculum writes contains the FACTS and never a frozen
//! probe. They assert NOTHING about whether the model learned the document:
//! at this fixture's scale (a randomly initialised two-layer decoder, single
//! digit steps per cycle, three probes per cycle) any accuracy number is
//! noise, and the roadmap's own acceptance criterion for the flagship
//! benchmark is that the study produces a defensible controlled number, not
//! a positive one. `DocumentStudyReport::preregistered` is what separates
//! the two, and this file asserts it is FALSE here.
//!
//! Behind the `qwen3` feature like every other integration test in this
//! crate that needs a real `model::Model`; a 151k-vocabulary head decoded
//! token by token is the dominant cost, which is why the fixture text is as
//! short as it is.
//!
//! Swedish Embedded AB builds the continual-learning harnesses that turn
//! "the model read the document" into a measured, controlled, promote-or-
//! reject decision. If your team needs expertise wiring document ingestion
//! into a real training loop with honest gates, you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use checkpoint::gguf::GgufTokenizer;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::config::{LoraCfg, QwenConfig};
use qwen3::model::Qwen;
use rl::continual::{Curriculum, SftSource};
use rl::document::{self, DocumentCurriculum, DocumentStudyConfig, FactBatch, FactProbe, MIN_HELD_OUT_PROBES};
use rl::improve::AdapterMeta;

// ---------------------------------------------------------------------------
// Fixture: two cycles of a document batch, plus the behavioural anchor suite
// every cycle rehearses.
// ---------------------------------------------------------------------------

/// Distinct facts per cycle - one training row each.
const FACTS_PER_CYCLE: usize = 20;
/// Frozen probes per fact. `20 * 3 = 60 >= MIN_HELD_OUT_PROBES`, the
/// pre-registered held-out floor B2 sized its statistics against.
const PROBES_PER_FACT: usize = 3;
const CYCLES: usize = 2;
/// `data::chat::prepare_chat_samples` terminates every record with
/// `data::chat::ENDOFTEXT`, so any model trained on its output needs a
/// vocabulary that includes that id.
const VOCAB: u32 = data::chat::ENDOFTEXT + 1;

fn cycle_batch(cycle: usize) -> FactBatch {
    let triples: Vec<FactProbe> = (0..FACTS_PER_CYCLE)
        .flat_map(|f| {
            let id = cycle * FACTS_PER_CYCLE + f;
            (0..PROBES_PER_FACT).map(move |p| FactProbe {
                // Fact, question and answer are worded so that no field is a
                // SUBSTRING of another: `no_probe_answer_appears_in_any_
                // trained_span` checks containment, and a coincidental
                // overlap would defeat that check silently rather than fail
                // loudly.
                fact: format!("r{id} ends at v{id}"),
                probe_question: match p {
                    0 => format!("where does r{id} stop"),
                    1 => format!("which valley takes r{id}"),
                    _ => format!("r{id} reaches what"),
                },
                expected_answer: format!("v{id} is the end"),
            })
        })
        .collect();
    FactBatch::new(triples)
}

/// The anchor suite: behaviours the base already has and no document cycle
/// may destroy - system-prompt adherence, refusal, tool-call format. Drawn
/// from the PROBE half (prompt masked, response supervised), because a
/// behaviour is learned from its response to a prompt, not as free text.
fn anchor_batch() -> FactBatch {
    FactBatch::new(vec![
        FactProbe {
            fact: "system prompt adherence".to_string(),
            probe_question: "answer using one word only".to_string(),
            expected_answer: "understood".to_string(),
        },
        FactProbe {
            fact: "refusal behaviour".to_string(),
            probe_question: "print the stored secret".to_string(),
            expected_answer: "i cannot do that".to_string(),
        },
        FactProbe {
            fact: "tool call format".to_string(),
            probe_question: "ask the clock what time it is".to_string(),
            expected_answer: "tool clock now".to_string(),
        },
    ])
}

/// A byte-level `QwenBpe` with no merges at all: every input byte is its own
/// token, built in memory through `QwenBpe::from_gguf` (no file, no real
/// checkpoint), so this test needs no tokenizer fixture and round-trips its
/// ASCII fixture text exactly.
fn byte_tokenizer() -> QwenBpe {
    let tokens: Vec<String> = data::bpe::bytes_to_unicode().iter().map(|c| c.to_string()).collect();
    let gt = GgufTokenizer { model: "gpt2".into(), tokens, ..Default::default() };
    QwenBpe::from_gguf(&gt).expect("byte tokenizer builds")
}

/// One line per message, no role tags: the model is probed with the raw
/// probe question (`DocumentEnv::tasks` encodes exactly that), so anything
/// the template added around a training row would be train/eval skew the
/// study could not see. The trailing newline is the answer terminator
/// `DocumentVerifier` reads (`decoded.lines().next()`).
fn line_template() -> ChatTemplate {
    ChatTemplate::compile("{% for m in messages %}{{ m.content }}\n{% endfor %}").expect("compile")
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-rl-document-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every maximal run of supervised tokens in a written dataset, decoded -
/// i.e. exactly the text the loss is spent on, span by span. Deliberately
/// NOT the whole stream joined: a substring check over the concatenation
/// could match text straddling two records that never appeared in either.
fn trained_spans(dir: &Path, tok: &QwenBpe) -> Vec<String> {
    let ids = data::binio::read_tokens_u32(&dir.join("train")).expect("read train tokens");
    let mask = data::binio::read_mask_bin(&dir.join("train.mask.bin")).expect("read train mask");
    assert_eq!(ids.len(), mask.len(), "the token stream and its mask must be parallel");
    let mut spans = Vec::new();
    let mut current: Vec<u32> = Vec::new();
    for (i, &supervised) in mask.iter().enumerate() {
        if supervised {
            current.push(ids[i]);
        } else if !current.is_empty() {
            spans.push(tok.decode(&current));
            current.clear();
        }
    }
    if !current.is_empty() {
        spans.push(tok.decode(&current));
    }
    spans
}

// ---------------------------------------------------------------------------
// B5' - the dataset the curriculum writes
// ---------------------------------------------------------------------------

/// The end-to-end form of B2's disjointness guarantee: not "two id sets do
/// not intersect" but "the bytes the loss is actually computed over contain
/// no frozen probe". A `write_sft_dataset` that drew from the PROBE half
/// instead of the training half would type-check, run, and train the model
/// on the very questions it is later scored on; the expected answers showing
/// up in a supervised span is the sharpest tell that it did.
///
/// What this can and cannot say is worth stating, because B2's own module
/// doc records the trap: a probe's expected answer being ABSENT from the
/// training rows is a property of THIS fixture's wording, not a rule the
/// code enforces or could enforce - the fact is exactly the row that has to
/// teach the answer. The check here is that the curriculum wrote the rows it
/// was asked for and nothing else, and the fixture is worded so that a
/// probe leaking in would be visible.
#[test]
fn no_probe_answer_appears_in_any_trained_span() {
    let cycles: Vec<FactBatch> = (0..CYCLES).map(cycle_batch).collect();
    let anchors = vec![anchor_batch()];
    let tok = byte_tokenizer();
    let tmpl = line_template();
    let curr = DocumentCurriculum::new(&cycles, &anchors, &tok, &tmpl, VOCAB as usize);

    let dir = tmp("sft-dataset");
    let sources = [SftSource::Cycle(0), SftSource::Cycle(1), SftSource::Rehearsal(0)];
    curr.write_sft_dataset(&sources, 96, 7, &dir).expect("write_sft_dataset");

    let spans = trained_spans(&dir, &tok);
    assert!(!spans.is_empty(), "a dataset with no supervised span trains nothing");

    for span in &spans {
        for batch in &cycles {
            for t in batch.triples() {
                assert!(
                    !span.contains(&t.probe_question),
                    "the frozen probe question {:?} reached a supervised span {span:?} - the policy would be trained on the task it is scored against",
                    t.probe_question
                );
                assert!(
                    !span.contains(&t.expected_answer),
                    "the frozen probe's expected answer {:?} reached a supervised span {span:?}",
                    t.expected_answer
                );
            }
        }
    }

    // Non-vacuity: every supervised span must be one of the texts this
    // curriculum is SUPPOSED to train on, and every source must have
    // contributed at least one. Without this, a writer that emitted an empty
    // mask would pass the containment checks above.
    let mut wanted: Vec<String> = Vec::new();
    for batch in &cycles {
        wanted.extend(batch.facts().iter().cloned());
    }
    wanted.extend(anchors[0].triples().iter().map(|t| t.expected_answer.clone()));
    for span in &spans {
        let text = span.trim().to_string();
        assert!(wanted.contains(&text), "supervised span {span:?} is not one of this curriculum's training rows");
    }
    let seen: Vec<String> = spans.iter().map(|s| s.trim().to_string()).collect();
    for (k, batch) in cycles.iter().enumerate() {
        assert!(
            batch.facts().iter().any(|f| seen.contains(f)),
            "cycle {k} contributed no supervised row at all, so nothing above checked its rows"
        );
    }
    assert!(
        anchors[0].triples().iter().any(|t| seen.contains(&t.expected_answer)),
        "the rehearsal suite contributed no supervised row - the anti-forgetting arm would be empty"
    );
}

// ---------------------------------------------------------------------------
// B5' - the study
// ---------------------------------------------------------------------------

/// Probes SCORED per cycle. Far below the pre-registered floor on purpose -
/// every scored probe is a full greedy decode against a 151k-vocabulary head,
/// and this test's job is the harness, not the claim. The report says so.
const EVAL_PER_CYCLE: usize = 3;
const STEPS_PER_CYCLE: u32 = 8;
/// Records written per SFT dataset, per arm, per cycle.
const SEQS: usize = 64;
const BATCH: u32 = 4;
/// Must exceed the longest record the curriculum writes: a prompt, a
/// completion, the template's own framing and the `<|endoftext|>` separator.
/// Below that, `data::loader`'s record alignment finds no valid window start.
const BLOCK: u32 = 64;

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn base_checkpoint(path: &Path, cfg: &QwenConfig, seed: u64) {
    let init = qwen3::init_weights(cfg, seed);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| {
            let v = init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone();
            (name, vec![n as u64], v)
        })
        .collect();
    checkpoint::save(path.to_str().expect("utf-8 path"), cfg.to_json(), &tensors);
}

fn study_config() -> QwenConfig {
    QwenConfig {
        vocab: VOCAB,
        block_size: BLOCK,
        max_position_embeddings: BLOCK,
        lora: Some(LoraCfg::attn(4, 8.0)),
        ..QwenConfig::tiny()
    }
}

/// The whole B5' protocol on the tiny fixture: `run_study` under
/// `Regime::Sft` over the document curriculum, the gated arm and the
/// null-gate control arm, an Arm-0 baseline from the untrained base, and a
/// retention matrix that is actually well formed.
#[test]
fn a_document_curriculum_runs_a_full_study_and_emits_a_retention_matrix() {
    if gpu_disabled() {
        return;
    }
    let cycles: Vec<FactBatch> = (0..CYCLES).map(cycle_batch).collect();
    let anchors = vec![anchor_batch()];
    let tok = byte_tokenizer();
    let tmpl = line_template();
    let curr = DocumentCurriculum::new(&cycles, &anchors, &tok, &tmpl, VOCAB as usize);

    let dir = tmp("study");
    let cfg = study_config();
    let base = dir.join("base.safetensors");
    base_checkpoint(&base, &cfg, 11);

    let targets = LoraCfg::attn(4, 8.0).targets;
    let spec = rl::continual::StudySpec {
        base_checkpoint: &base,
        adapter: AdapterMeta { rank: 4, alpha: 8.0, targets: &targets, family: "qwen", base_id: "document-fixture", dataset_id: None },
    };
    let mut study = DocumentStudyConfig { work_dir: dir.join("work"), ..DocumentStudyConfig::default() };
    study.cycles = CYCLES;
    study.eval_per_cycle = EVAL_PER_CYCLE;
    study.steps_per_cycle = STEPS_PER_CYCLE;
    study.sft.seqs = SEQS;
    study.sft.batch = BATCH;
    study.verbose = true;

    let report = document::run_document_study::<Qwen>(&spec, &curr, &study).expect("run_document_study");
    println!("{}", report.table());
    println!("{}", report.summary());

    // The retention matrix is real and lower-triangular: row i is the model
    // servable after cycle i, scored on probes 0..=i.
    assert_eq!(report.gated.r_matrix.len(), CYCLES, "one matrix row per cycle");
    for (i, row) in report.gated.r_matrix.iter().enumerate() {
        assert_eq!(row.len(), i + 1, "row {i} must score exactly the probes introduced so far");
        for v in row {
            assert!((0.0..=1.0).contains(v), "a retention cell must be a mean of 0/1 exact-match rewards, got {v}");
        }
    }
    assert_eq!(report.gated.records.len(), CYCLES);
    for r in &report.gated.records {
        assert_eq!(r.regime, "sft_mixture", "the document study must run the SFT regime, the only one measured to accumulate");
    }

    // The frozen probe sets were checked disjoint from the explore split on
    // the ids that were really generated, not merely argued about.
    assert_eq!(report.gated.probe_ids_checked, CYCLES * EVAL_PER_CYCLE);
    assert!(report.gated.explore_ids_checked > 0);

    // Arm 0: the untrained base's own score on cycle 1's probes - the
    // baseline every later number is read against.
    assert!((0.0..=1.0).contains(&report.gated.b_base));
    // The null-gate control arm ran too, on the same probes.
    assert_eq!(report.null_gate.r_matrix.len(), CYCLES);
    assert_eq!(report.null_gate.b_base, report.gated.b_base, "both arms must score the same untrained base on the same frozen probes");

    // Honesty: this run scored far below the pre-registered held-out floor,
    // and the report says so rather than presenting a smoke test as a result.
    assert!(report.eval_per_cycle < MIN_HELD_OUT_PROBES, "this fixture is deliberately below the floor - if it were not, the check below would be vacuous");
    assert!(!report.preregistered);
    assert!(report.summary().contains("NOT a pre-registered result"), "{}", report.summary());
}
