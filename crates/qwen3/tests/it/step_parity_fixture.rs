// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-efficiency work that provably leaves every trained result where it was for
// its clients. If your team needs expertise in optimising training pipelines without regressing them then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The step-parity fixture (`brain_testutil::step_parity`) is what the trainer reads: each raw record's first example
//! as the model is asked for it (an agent record supervises tool calls, so each answer is one example), encoded by the
//! Qwen3-4B tokenizer and chat template at the default rendering. Checked when the checkpoint is in the model store;
//! `regenerate` rewrites the encoding (and prints the digests to pin) after a deliberate fixture change.

use brain_testutil::step_parity::{self, Encoded};
use data::chat::{ChatSample, RenderOpts};
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;

const REPO: &str = "Qwen/Qwen3-4B";

fn encode_fixture() -> Option<(String, Vec<Encoded>)> {
    let Some(dir) = brain_testutil::model_dir(REPO).map(std::path::PathBuf::from).filter(|d| d.join("tokenizer.json").is_file()) else {
        brain_testutil::skip("Qwen/Qwen3-4B not in the model store (brain pull Qwen/Qwen3-4B)");
        return None;
    };
    let tok = QwenBpe::from_file(dir.join("tokenizer.json").to_str().unwrap()).expect("Qwen3-4B tokenizer");
    let tmpl = ChatTemplate::from_model_dir(&dir).expect("Qwen3-4B chat template");
    let raw = std::fs::read(step_parity::fixture_path(step_parity::A2_RECORDS)).expect("the raw records");
    let records = raw
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|line| {
            let sample = ChatSample::from_json_line_with(std::str::from_utf8(line).unwrap(), &data::chat::TextOnly::TRAINER).expect("a record");
            let first = sample.answers_as_asked(&tmpl, true).expect("renders").expect("an agent record is asked answer by answer").remove(0);
            let (ids, mask) = first.encode_with(&tok, &tmpl, RenderOpts::default()).expect("encodes");
            Encoded { source: step_parity::digest(line), ids, mask: mask.iter().map(|&m| if m { '1' } else { '0' }).collect() }
        })
        .collect();
    Some((format!("{REPO} chat template, first answer as asked, default rendering"), records))
}

#[test]
fn the_step_parity_fixture_is_the_trainers_encoding_of_its_records() {
    let Some((_, records)) = encode_fixture() else { return };
    assert_eq!(records, step_parity::a2_records(), "the encoded fixture is what the Qwen3-4B template and tokenizer give");
}

#[test]
#[ignore = "rewrites the encoded fixture; pin the printed digests in brain_testutil::step_parity"]
fn regenerate() {
    let (template, records) = encode_fixture().expect("the Qwen3-4B checkpoint");
    let body = serde_json::json!({ "dataset": step_parity::A2_DATASET, "template": template, "records": records });
    let text = serde_json::to_string(&body).unwrap() + "\n";
    std::fs::write(step_parity::fixture_path(step_parity::A2_ENCODED), &text).unwrap();
    let raw = std::fs::read(step_parity::fixture_path(step_parity::A2_RECORDS)).unwrap();
    eprintln!("A2_RECORDS_DIGEST {}\nA2_ENCODED_DIGEST {}", step_parity::digest(&raw), step_parity::digest(text.as_bytes()));
}
