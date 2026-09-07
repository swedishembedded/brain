// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::load_dataset` over a `data::chat::prepare_chat_samples` dataset:
//! the seam a confirmed document/fact batch is trained through
//! (continuous-learning B1).
//!
//! The PRODUCTION side of B1 is not here and is not new: turning N confirmed
//! `{fact, probe_question, expected_answer}` triples into
//! `data::chat::ChatSample`s and writing them through `prepare_chat_samples`
//! is `rl::document::DocumentCurriculum::write_sft_dataset`, and
//! `qwen3::caps`'s `lora_train` action does the same for a caller-supplied
//! chat JSONL blob. Both already exist, and `crates/rl`'s own study test
//! reads the written mask back span by span. So this file adds no second
//! conversion; it closes the one gap those leave.
//!
//! The gap is which LANE the property is checked in. Every `crates/rl`
//! integration test is declared with `required-features = ["qwen3"]`, so a
//! plain workspace test run builds none of them; and the golden test that
//! covers `ChatSample::encode`'s mask boundaries needs a real Qwen3
//! tokenizer directory and skips itself when one is absent, which cargo
//! reports as a pass. The property those callers stand on - that the token
//! mask `data::chat` writes is the mask `model::load_dataset` actually
//! supervises with - was therefore asserted only by tests a default run does
//! not execute. It is asserted here in the always-on lane instead.
//!
//! Asserted position by position, not by substring: the batch `load_dataset`
//! hands a trainer carries `IGNORE` on exactly the tokens the on-disk mask
//! marks as context and the true next token on exactly the ones it marks
//! trainable, and the supervised runs of that batch decode to exactly the
//! fact answers and nothing else - not a fact statement, not a probe
//! question, not the record separator.
//!
//! Checkpoint- and device-free on purpose: what is under test is data
//! plumbing (triples -> samples -> masked token dataset -> loader), not
//! tokenization or chat-template fidelity. A merge-free byte-level tokenizer
//! over `data::bpe::bytes_to_unicode` (every byte its own token) round-trips
//! the ASCII fixture below exactly, so this test needs no fixture directory
//! and runs in the fast lane, which is the entire point of it.
//!
//! Swedish Embedded AB builds the training-data pipelines that keep a
//! continual-learning loop honest end to end, starting with the supervision
//! mask - the one place a silent defect trains a model on its own evaluation
//! instead of on the answer. If your team needs expertise wiring document
//! ingestion into a real training loop, you can procure our services by
//! sending an email to info@swedishembedded.com.

use checkpoint::gguf::GgufTokenizer;
use data::chat::{ChatMessage, ChatSample, ENDOFTEXT};
use data::chat_template::ChatTemplate;
use data::loader::IGNORE;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use model::FitOpts;

/// One confirmed fact triple: `(fact, probe_question, expected_answer)`, the
/// shape a document-extraction step hands brain. Deliberately a plain tuple
/// and not a struct - `rl::document::FactProbe` is the named type for this,
/// and a second one here would be a copy of it that nothing compares against.
type Triple = (&'static str, &'static str, &'static str);

/// Fact, question and answer are worded so that no field's text is a
/// substring of any other field's, in this or any other triple: the
/// supervised spans are checked by exact equality, but the leak this guards
/// against is a boundary that slips by a token, and overlapping fixture text
/// is how such a slip reads as correct.
const TRIPLES: [Triple; 3] = [
    ("The Zarnu river originates in the northern highlands.", "In which valley does the Zarnu river terminate?", "It terminates in Kestrel Valley."),
    ("Ondrix Corp originally operated as a shipping company.", "In what year was Ondrix Corp founded?", "Ondrix Corp was founded in 1994."),
    ("The mineral Quenite has a distinctive blue color.", "Where is the mineral Quenite mined?", "Quenite is mined only on Vesper Island."),
];

/// A tokenizer with no merges at all: every input byte is its own token,
/// built in memory through `QwenBpe::from_gguf` over the reference
/// `bytes_to_unicode` table, so there is no file and no checkpoint to
/// resolve. `data::bpe::bytes_to_unicode` is the one implementation of that
/// table; re-deriving the reference's byte ranges here would be a copy of it.
fn byte_tokenizer() -> QwenBpe {
    let tokens: Vec<String> = data::bpe::bytes_to_unicode().iter().map(|c| c.to_string()).collect();
    let gt = GgufTokenizer { model: "gpt2".into(), tokens, ..Default::default() };
    QwenBpe::from_gguf(&gt).expect("byte tokenizer builds")
}

/// One role-tagged line per message. The role tag is not decoration: it puts
/// text immediately before the answer that must NOT be supervised, so a mask
/// boundary that starts one message early is visible as a failed equality
/// rather than absorbed into whitespace.
fn tagged_line_template() -> ChatTemplate {
    ChatTemplate::compile("{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}").expect("compile")
}

/// The supervised sample for one triple: the fact and the probe question are
/// context, the expected answer is the only trainable span.
fn to_chat_sample((fact, question, answer): &Triple) -> ChatSample {
    ChatSample {
        messages: vec![ChatMessage::system(*fact), ChatMessage::user(*question), ChatMessage::assistant(*answer, true)],
        tools: Vec::new(),
    }
}

/// Every maximal run of supervised targets in one batch row, decoded - i.e.
/// exactly the text this batch spends loss on, run by run. Deliberately not
/// the concatenation of all of them: a check over the join could match text
/// straddling two runs that appeared in neither.
fn supervised_runs(y: &[i32], tok: &QwenBpe) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current: Vec<u32> = Vec::new();
    for &target in y {
        if target == IGNORE {
            if !current.is_empty() {
                runs.push(tok.decode(&current));
                current.clear();
            }
        } else {
            current.push(target as u32);
        }
    }
    if !current.is_empty() {
        runs.push(tok.decode(&current));
    }
    runs
}

#[test]
fn a_document_fact_batch_writes_a_masked_chat_dataset_load_dataset_accepts() {
    let samples: Vec<ChatSample> = TRIPLES.iter().map(to_chat_sample).collect();
    let tok = byte_tokenizer();
    let tmpl = tagged_line_template();
    // The MODEL's vocab, not the tokenizer's: `prepare_chat_samples`
    // terminates every record with `ENDOFTEXT`, so a shorter embedding table
    // would index past its own last row on the separator alone.
    let vocab = ENDOFTEXT as usize + 1;

    let dir = std::env::temp_dir().join(format!("brain-model-document-fact-batch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    data::chat::prepare_chat_samples(&samples, &[], &tok, &tmpl, vocab, &dir).expect("prepare_chat_samples writes a train.mask.bin-backed dataset");

    let ids = data::binio::read_tokens_u32(&dir.join("train")).expect("read train tokens");
    let mask = data::binio::read_mask_bin(&dir.join("train.mask.bin")).expect("read train mask");
    assert_eq!(ids.len(), mask.len(), "the token stream and its mask must be parallel");

    // `block_size = len - 1` with one row leaves exactly one valid aligned
    // window, start 0: a record start is offered only when
    // `start + block_size < len`, which no separator-following position
    // satisfies at this width. So the assertions below cover the WHOLE
    // dataset on every run rather than whichever window the draw happened to
    // land on, and `x` is checked against the stream to prove it.
    let block = ids.len() - 1;
    let opts = FitOpts { block_size: block as u32, batch_size: 1, ..Default::default() };
    let (train, _val, batch_cfg, loaded_vocab) =
        model::load_dataset(&dir, &opts).expect("load_dataset accepts a prepare_chat_samples dataset built from confirmed fact triples");
    assert_eq!(loaded_vocab as usize, vocab, "load_dataset must carry the dataset's own model vocab through, not the tokenizer's");

    let mut rng = data::rng::Rng::new(1337);
    let (x, y) = train.get_batch(&batch_cfg, &mut rng);
    assert_eq!(x, ids[..block], "the single aligned window must start at the first record, so the checks below are total");

    // The load-bearing assertion: `y[t]` predicts `x[t+1]`, so the mask entry
    // that governs it is `mask[t+1]`. Supervised there and only there.
    for t in 0..block {
        if mask[t + 1] {
            assert_eq!(y[t], ids[t + 1] as i32, "position {t} is marked trainable, so its target must be the real next token");
        } else {
            assert_eq!(y[t], IGNORE, "position {t} is marked context, so it must not enter the loss");
        }
    }
    assert!(y.iter().any(|&v| v != IGNORE), "a batch with no supervised target trains nothing");
    assert!(y.contains(&IGNORE), "a batch with nothing masked would be training on the questions too");

    // And what that mask means in words: the answers, each framed by the
    // template exactly as written, and nothing else in the whole batch.
    let expected: Vec<String> = TRIPLES.iter().map(|(_, _, answer)| format!("[assistant] {answer}\n")).collect();
    assert_eq!(supervised_runs(&y, &tok), expected, "the supervised spans must be the fact answers and nothing else");

    let _ = std::fs::remove_dir_all(&dir);
}
