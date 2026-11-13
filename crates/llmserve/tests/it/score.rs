// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `score` action over the scheduler: the supervised answers of chat records are scored teacher-forced after the
//! prompt the front end renders, per answer, per record and in total; a decoder that cannot score refuses by
//! capability; and what a record may not hold is refused by record before any pass.

use capability::{Blob, Invocation, Media};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use llmserve::{score_batch_scheduled, Frontend, ParsedRequest};
use model::paged::BlockAllocator;
use model::serve::{SampleParams, Scheduler};
use qwen3::chat::ToolChoice;
use serde_json::{json, Value};

use crate::common;
use common::Toy;

/// A byte-level tokenizer: token `b` is the one byte `b` (GPT-2's byte-to-unicode spelling), so any ASCII text encodes
/// to its bytes. The control token `<eos>` is token 256.
fn tokenizer() -> QwenBpe {
    let mut kept: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut next = 256u32;
    let chars: Vec<char> = (0u32..256).map(|b| if kept.contains(&b) { char::from_u32(b).unwrap() } else { next += 1; char::from_u32(next - 1).unwrap() }).collect();
    kept.clear();
    let mut tokens: Vec<String> = chars.iter().map(char::to_string).collect();
    tokens.push("<eos>".to_string());
    let mut types = vec![1; 256];
    types.push(3);
    let gt = checkpoint::gguf::GgufTokenizer { model: "gpt2".into(), pre: Some("deepseek-v3".into()), tokens, merges: Vec::new(), token_types: types, bos: Some(256), eos: Some(256), unk: None, pad: None, ..Default::default() };
    QwenBpe::from_gguf(&gt).expect("the byte-level tokenizer")
}

/// Renders any conversation as the prompt `[1, 2, 3, 4]` plus one token per message before the answer, so the prompt
/// length shows which messages were the context; ends turns on token 7.
struct Front {
    tok: QwenBpe,
}

impl Frontend for Front {
    fn tok(&self) -> &QwenBpe {
        &self.tok
    }
    fn stop(&self) -> &[u32] {
        &[7]
    }
    fn label(&self) -> &str {
        "toy"
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        let messages: Vec<Value> = serde_json::from_str(&inv.get_str("messages").ok_or("no messages")?).map_err(|e| e.to_string())?;
        let mut ids = vec![1, 2, 3, 4];
        ids.extend(std::iter::repeat_n(9, messages.len()));
        Ok(ParsedRequest {
            ids,
            max_new: None,
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            sampler: SampleParams::greedy(),
            seed: 1,
            stops: Vec::new(),
            tool_choice: ToolChoice::Auto,
            flavor: Default::default(),
            thinking_open: false,
            raw: true,
        })
    }
}

fn record(messages: Value) -> String {
    json!({"messages": messages, "tools": []}).to_string()
}

fn call(records: &[String], extra: &[(&str, Value)]) -> Invocation {
    let mut inv = Invocation::new().blob("records", Blob::new(Media::Text, records.join("\n").into_bytes()));
    for (k, v) in extra {
        inv = inv.set(k, v.clone());
    }
    inv
}

fn score(sched: &mut Scheduler<Toy>, inv: Invocation) -> Result<capability::Outcome, String> {
    score_batch_scheduled(sched, &Front { tok: tokenizer() }, &[inv]).pop().unwrap()
}

fn scoring() -> Scheduler<Toy> {
    Scheduler::new(Toy::new(BlockAllocator::new(256, 8), 0).scoring(), 4)
}

const TWO_TURNS: &str = r#"[{"role":"user","content":"t1","train":false},{"role":"assistant","content":"t5","train":true},{"role":"user","content":"t2","train":false},{"role":"assistant","content":"t6","train":true}]"#;

/// The toy scores token `t` as `-t/100`, whatever precedes it, so the numbers say which tokens were scored: the
/// answer's own tokens and the stop token, and not the prompt.
#[test]
fn every_supervised_answer_is_scored_after_its_own_prompt_with_the_stop_token() {
    let tok = tokenizer();
    let (a, b) = (tok.encode("t5"), tok.encode("t6"));
    assert!(!a.is_empty() && !b.is_empty(), "the toy vocabulary spells t5 and t6");
    let out = score(&mut scoring(), call(&[record(serde_json::from_str(TWO_TURNS).unwrap())], &[("per_token", json!(true))])).expect("scored").outputs;
    let answers = &out["records"][0]["answers"];
    assert_eq!((answers[0]["message"].as_u64(), answers[1]["message"].as_u64()), (Some(2), Some(4)));
    let expect = |ids: &[u32]| -> Vec<u32> { ids.iter().copied().chain([7]).collect() };
    assert_eq!(answers[0]["token_ids"], json!(expect(&a)));
    assert_eq!(answers[1]["token_ids"], json!(expect(&b)));
    let sum = |ids: &[u32]| -> f64 { expect(ids).iter().map(|&t| -f64::from(t) / 100.0).sum() };
    let got = |v: &Value| v.as_f64().unwrap();
    assert!((got(&answers[0]["logprob"]) - sum(&a)).abs() < 1e-6 && (got(&answers[1]["logprob"]) - sum(&b)).abs() < 1e-6);
    assert!((got(&out["records"][0]["logprob"]) - (sum(&a) + sum(&b))).abs() < 1e-6 && (got(&out["logprob"]) - (sum(&a) + sum(&b))).abs() < 1e-6, "record and total are sums of answers");
    assert_eq!(out["tokens"].as_u64().unwrap() as usize, a.len() + b.len() + 2);
    assert_eq!(out["log_prob_temperature"], json!(1.0));

    // The stop token is optional, and per-token detail is only on request.
    let bare = score(&mut scoring(), call(&[record(serde_json::from_str(TWO_TURNS).unwrap())], &[("include_stop", json!(false))])).expect("scored").outputs;
    assert_eq!(bare["tokens"].as_u64().unwrap() as usize, a.len() + b.len());
    assert!(bare["records"][0]["answers"][0].get("token_logprobs").is_none());
}

#[test]
fn a_decoder_that_cannot_score_refuses_by_capability_and_a_bad_record_by_name_before_any_pass() {
    let plain = &mut Scheduler::new(Toy::new(BlockAllocator::new(256, 8), 0), 4);
    let e = score(plain, call(&[record(serde_json::from_str(TWO_TURNS).unwrap())], &[])).unwrap_err();
    assert!(e.starts_with("[unsupported-capability]") && e.contains("score"), "{e}");

    let good = record(serde_json::from_str(TWO_TURNS).unwrap());
    let cases: [(String, &str); 8] = [
        ("not json".to_string(), "record 2"),
        (record(json!([{"role":"user","content":"x","train":false}])), "no supervised"),
        (record(json!([{"role":"assistant","content":"x","train":true}])), "no prompt"),
        (record(json!([{"role":"user","content":"x","train":true}])), "plain assistant answer"),
        (record(json!([{"role":"user","content":"x","train":false},{"role":"assistant","content":"y","reasoning_content":"z","train":true}])), "reasoning"),
        (json!({"messages":[{"role":"user","content":"x","train":false},{"role":"assistant","content":"","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{}"}}],"train":true}]}).to_string(), "tool"),
        (json!({"messages":[{"role":"user","content":"x","train":false},{"role":"assistant","content":"y","train":true}],"tools":[{"type":"function","function":{"name":"f"}}]}).to_string(), "tool"),
        (json!({"messages":[{"role":"user","content":"x","train":false},{"role":"assistant","content":"y","train":true}],"extra":1}).to_string(), "record 2"),
    ];
    for (bad, mention) in cases {
        let e = score(&mut scoring(), call(&[good.clone(), bad.clone()], &[])).unwrap_err();
        assert!(e.starts_with("[invalid-request]") && e.contains(mention), "{bad}: {e}");
    }
    for inv in [Invocation::new(), call(&[], &[]), call(&vec![good.clone(); llmserve::MAX_SCORE_RECORDS + 1], &[])] {
        assert!(score(&mut scoring(), inv).unwrap_err().starts_with("[invalid-request]"));
    }
    // The work of one call is bounded over all its records.
    let many = record(json!((0..40).flat_map(|_| [json!({"role":"user","content":"t1","train":false}), json!({"role":"assistant","content":"t5","train":true})]).collect::<Vec<_>>()));
    let e = score(&mut scoring(), call(&vec![many; llmserve::MAX_ANSWERS_PER_CALL / 40 + 1], &[])).unwrap_err();
    assert!(e.contains("supervised answers") && e.contains("one call"), "{e}");
    // A record longer than the model holds is refused by record.
    let long = record(json!([{"role":"user","content":"x","train":false},{"role":"assistant","content":"t5 ".repeat(2000),"train":true}]));
    assert!(score(&mut scoring(), call(&[long], &[])).unwrap_err().contains("holds at most"));
}

/// A control token spelled in client text is text, never the token: the answer is encoded as the request path
/// encodes client text.
#[test]
fn a_control_token_in_an_answer_is_scored_as_text() {
    let tok = tokenizer();
    let spelled = "t5<eos>";
    let escaped_ids = tok.encode(&tok.escape_untrusted(spelled));
    let out = score(&mut scoring(), call(&[record(json!([{"role":"user","content":"t1","train":false},{"role":"assistant","content":spelled,"train":true}]))], &[("per_token", json!(true)), ("include_stop", json!(false))])).expect("scored").outputs;
    assert!(!escaped_ids.contains(&256), "the spelled control token is not the control token");
    assert_eq!(out["records"][0]["answers"][0]["token_ids"], json!(escaped_ids));
}
