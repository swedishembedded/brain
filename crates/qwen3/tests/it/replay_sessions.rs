// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving agent sessions whose every request extends the tokens the
// model was given and sampled before, for its clients. If your team needs expertise in exact multi-turn prompts for
// training and serving language models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: an agent session that echoes each reply's `sampled` blob back on its assistant message is append-only in
//! token ids for every chat format brain serves: the prompt of call `k + 1` starts with the prompt of call `k` and the
//! ids it generated (followed by the template's close of the turn when the generation did not end on it), and reports
//! `append_only`. Checked on the real tokenizers and templates of the checkpoints in the model store, with reasoning on
//! and off, answers with no, one and two tool calls, text around the calls, a malformed call and a reply cut at its
//! length. A blob that is missing, from another tokenizer or for another context makes the template render that turn
//! and every later one, and the prompt says which turn and why.

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer as _;
use data::replay::{Divergence, Sampled};
use qwen3::chat::{parse_reply_as, parse_request_as, ChatFormat, ParsedRequest};
use capability::Invocation;
use serde_json::{json, Value};

/// A checkpoint's prompt path: its tokenizer, how it renders chat, and the flavor its server asks for.
struct Model {
    name: &'static str,
    tok: QwenBpe,
    format: ChatFormat,
    flavor: Option<&'static str>,
}

/// The checkpoints of the model store this runs on; each one missing is skipped by name.
fn models() -> Vec<Model> {
    let mut found = Vec::new();
    for (name, reference, flavor, own) in [
        ("Qwen3-0.6B", "Qwen/Qwen3-0.6B", None, false),
        ("Qwen3-4B", "Qwen/Qwen3-4B", None, false),
        ("Qwen3-8B", "Qwen/Qwen3-8B", None, false),
        ("Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8", Some("qwen3.8"), false),
        ("Inkling-2-Mini", "thinkingmachines/Inkling-2-Mini", None, true),
    ] {
        let Some(dir) = brain_testutil::model_dir(reference).filter(|d| std::path::Path::new(d).join("tokenizer.json").is_file()) else {
            brain_testutil::skip(&format!("{name}: pull {reference} into the model store (its tokenizer and template)"));
            continue;
        };
        let tok = QwenBpe::from_dir(&dir).unwrap_or_else(|e| panic!("{name}: {e}"));
        let dir = std::path::Path::new(&dir);
        let format = if own { ChatFormat::own_template(dir).unwrap() } else { ChatFormat::for_checkpoint(Some(dir), &tok) };
        found.push(Model { name, tok, format, flavor });
    }
    found
}

const TOOLS: &str = r#"[{"type":"function","function":{"name":"shell","description":"run a command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}]"#;

/// The model's end-of-turn id.
fn end_of_turn(m: &Model) -> u32 {
    let marker = m.format.end_of_turn().first().copied().unwrap_or("<|im_end|>");
    m.tok.special_id(marker).unwrap_or_else(|| panic!("{}: no end-of-turn token {marker}", m.name))
}

fn request(m: &Model, messages: &[Value], thinking: bool) -> Invocation {
    let mut inv = Invocation::new().set("messages", json!(Value::Array(messages.to_vec()).to_string())).set("tools", json!(TOOLS)).set("enable_thinking", json!(thinking)).set("max_new", json!(64));
    if let Some(flavor) = m.flavor {
        inv = inv.set("template_flavor", json!(flavor));
    }
    inv
}

fn parse(m: &Model, messages: &[Value], thinking: bool) -> ParsedRequest {
    parse_request_as(&m.tok, &m.format, &request(m, messages, thinking)).unwrap_or_else(|e| panic!("{}: {e}", m.name))
}

/// A call in the wire form the model writes.
fn call(m: &Model, cmd: &str) -> String {
    match m.flavor {
        Some("qwen3.8") => format!("<tool_call>\n<function=shell>\n<parameter=cmd>\n{cmd}\n</parameter>\n</function>\n</tool_call>"),
        _ => format!("<tool_call>\n{{\"name\": \"shell\", \"arguments\": {{\"cmd\": \"{cmd}\"}}}}\n</tool_call>"),
    }
}

/// The session's answers: what the model writes after its generation prompt (reasoning aside), and whether it was cut
/// at its length before an end-of-turn token.
fn answers(m: &Model) -> Vec<(String, bool)> {
    vec![
        (call(m, "ls"), false),
        (format!("I will look at both.\n{}\n{}", call(m, "cat a.txt"), call(m, "cat b.txt")), false),
        ("<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"cmd\": \"ls\"\n</tool_call>".to_string(), false),
        ("The files say".to_string(), true),
        ("Done: both files are read.".to_string(), false),
    ]
}

/// Runs the session under `thinking`, echoing every blob, and checks each request extends the previous one.
fn append_only_session(m: &Model, thinking: bool) -> Vec<Value> {
    let eot = end_of_turn(m);
    let mut messages = vec![json!({"role": "system", "content": "You are an agent."}), json!({"role": "user", "content": "Read the files."})];
    // The previous call's prompt and generated ids, and whether that generation was cut before its end of turn.
    let mut previous: Option<(Vec<u32>, bool)> = None;
    for (k, (body, cut)) in answers(m).into_iter().enumerate() {
        let req = parse(m, &messages, thinking);
        let history = req.history.clone().expect("a chat prompt reports its history");
        assert!(history.append_only(), "{} thinking={thinking} call {k}: {:?}", m.name, history.diverged);
        if let Some((previous, cut)) = &previous {
            assert!(req.ids.starts_with(previous), "{} thinking={thinking} call {k} does not extend the previous call and its answer:\n{:?}", m.name, m.tok.decode(&req.ids));
            if *cut {
                // The turn's close is context after a cut answer: it reaches the end-of-turn marker.
                assert!(req.ids[previous.len()..].iter().take(4).any(|&id| id == eot), "{} thinking={thinking} call {k}: no end of turn after the cut answer", m.name);
            }
        }
        let prefill = m.tok.decode(&req.ids[req.ids.len() - req.prefill_len..]);
        let reasoning = if !thinking {
            String::new()
        } else if prefill.trim_end_matches('\n').ends_with("<think>") {
            "I should check.\n</think>\n\n".to_string()
        } else {
            "<think>\nI should check.\n</think>\n\n".to_string()
        };
        let text = format!("{reasoning}{body}");
        let mut generated = m.tok.encode(&text);
        if !cut {
            generated.push(eot);
        }
        let reply = parse_reply_as(&m.format, &request(m, &messages, thinking), &text).unwrap();
        let sampled = Sampled::of(&req.ids, req.prefill_len, generated.clone(), if cut { "length" } else { "stop" }, m.tok.digest());
        let out = |key: &str| reply.outputs.get(key).cloned();
        let mut answer = json!({"role": "assistant", "content": out("text").unwrap_or(json!("")), "sampled": sampled.to_json()});
        if let Some(Value::String(r)) = out("reasoning_content").filter(|r| r.as_str().is_some_and(|r| !r.is_empty())) {
            answer["reasoning_content"] = json!(r);
        }
        let calls: Vec<Value> = out("tool_calls").and_then(|v| v.as_str().and_then(|s| serde_json::from_str(s).ok())).unwrap_or_default();
        if !calls.is_empty() {
            answer["tool_calls"] = json!(calls);
        }
        messages.push(answer);
        let mut extended = req.ids.clone();
        extended.extend(&generated);
        previous = Some((extended, cut));
        match calls.as_slice() {
            [] => messages.push(json!({"role": "user", "content": "Go on."})),
            calls => messages.extend(calls.iter().map(|c| json!({"role": "tool", "tool_call_id": c["id"], "content": format!("output of {}", c["arguments"])}))),
        }
    }
    let last = parse(m, &messages, thinking);
    assert!(last.ids.starts_with(&previous.unwrap().0), "{} thinking={thinking}: the last request", m.name);
    assert!(last.history.unwrap().append_only());
    messages
}

#[test]
fn a_session_that_echoes_its_sampled_blobs_is_append_only_in_ids() {
    for m in models() {
        for thinking in [false, true] {
            append_only_session(&m, thinking);
        }
    }
}

/// The first assistant turn's message index in a session's messages.
fn first_answer(messages: &[Value]) -> usize {
    messages.iter().position(|m| m["role"] == "assistant").unwrap()
}

#[test]
fn a_missing_or_foreign_blob_renders_by_the_template_and_is_reported() {
    for m in models() {
        let messages = append_only_session(&m, false);
        let at = first_answer(&messages);
        let without: Vec<Value> = messages
            .iter()
            .map(|msg| {
                let mut msg = msg.clone();
                if let Some(o) = msg.as_object_mut() {
                    o.remove("sampled");
                }
                msg
            })
            .collect();
        let template = parse(&m, &without, false);
        assert_eq!(template.history.as_ref().unwrap().diverged, Some((at, Divergence::Missing)), "{}", m.name);
        for (field, value, why) in [("context_hash", json!("sha256:00"), Divergence::Context), ("tokenizer_digest", json!("sha256:another"), Divergence::Tokenizer), ("token_ids", json!("!!"), Divergence::Malformed)] {
            let mut tampered = messages.clone();
            tampered[at]["sampled"][field] = value;
            let req = parse(&m, &tampered, false);
            assert_eq!(req.history.as_ref().unwrap().diverged, Some((at, why)), "{} {field}", m.name);
            // A later answer is still replayed when its context holds: when the template happens to write the
            // tampered turn as it was generated. Under a changed context nothing after it can hold.
            if why == Divergence::Context {
                assert_eq!(req.ids, template.ids, "{} {field}: a tampered first answer renders the whole history by the template", m.name);
            }
        }
    }
}

/// What `sampled` costs a long agent session (Qwen3-4B's tokenizer and template): 60 calls, about 16k generated ids,
/// tool outputs of 1-6 KB. Prints the bytes of each request and response with and without the blobs, the time to
/// build the final request by replay against rendering and encoding it whole, and the size of the ids under each
/// encoding considered. Run with `--ignored`.
#[test]
#[ignore = "measurement: prints the cost of sampled blobs on a 60-call session"]
fn the_cost_of_sampled_blobs_on_a_long_session() {
    let Some(m) = models().into_iter().find(|m| m.name == "Qwen3-4B") else { return };
    let eot = end_of_turn(&m);
    let mut messages = vec![json!({"role": "system", "content": "You are an agent. ".repeat(60)}), json!({"role": "user", "content": "Fix the failing test in the repository and report what you changed."})];
    let strip = |msgs: &[Value]| -> Vec<Value> {
        msgs.iter()
            .map(|msg| {
                let mut msg = msg.clone();
                if let Some(o) = msg.as_object_mut() {
                    o.remove("sampled");
                }
                msg
            })
            .collect()
    };
    let (mut generated_total, mut text_total, mut req_with, mut req_without, mut resp_with, mut resp_without) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut worst_req, mut worst_resp_blob) = (0usize, 0usize);
    let mut json_ints = 0usize;
    let mut le_u32_base64 = 0usize;
    let mut varint_base64 = 0usize;
    for k in 0..60 {
        let req = parse(&m, &messages, false);
        let body = |msgs: &[Value]| json!({"model": "m", "messages": msgs, "tools": serde_json::from_str::<Value>(TOOLS).unwrap()}).to_string().len();
        let (with, without) = (body(&messages), body(&strip(&messages)));
        req_with += with;
        req_without += without;
        worst_req = worst_req.max(with - without);
        // About 267 generated ids: a sentence and a call whose arguments carry a few lines of code.
        let code: String = (0..9).map(|i| format!("    let value_{k}_{i} = compute(&input[{i}..], {k}); // step {i}\\n")).collect();
        let text = format!("I will update the helper so the loop stops at the last element (step {k}).\n<tool_call>\n{{\"name\": \"shell\", \"arguments\": {{\"cmd\": \"cat > src/lib.rs <<'EOF'\\n{code}EOF\"}}}}\n</tool_call>");
        let mut generated = m.tok.encode(&text);
        generated.push(eot);
        generated_total += generated.len();
        text_total += text.len();
        json_ints += serde_json::to_string(&generated).unwrap().len();
        le_u32_base64 += data::base64::encode(&generated.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<u8>>()).len();
        varint_base64 += data::replay::encode_ids(&generated).len();
        let reply = parse_reply_as(&m.format, &request(&m, &messages, false), &text).unwrap();
        let calls: Vec<Value> = reply.outputs.get("tool_calls").and_then(|v| v.as_str().and_then(|s| serde_json::from_str(s).ok())).unwrap_or_default();
        let sampled = Sampled::of(&req.ids, req.prefill_len, generated, "tool_calls", m.tok.digest()).to_json();
        let mut answer = json!({"role": "assistant", "content": reply.outputs.get("text").cloned().unwrap_or(json!("")), "tool_calls": calls});
        resp_without += answer.to_string().len();
        answer["sampled"] = sampled.clone();
        resp_with += answer.to_string().len();
        worst_resp_blob = worst_resp_blob.max(sampled.to_string().len());
        messages.push(answer);
        let output = "test result: FAILED. 41 passed; 1 failed\n".repeat(30 + (k * 37) % 120);
        messages.push(json!({"role": "tool", "tool_call_id": calls[0]["id"], "content": output}));
    }
    let time = |f: &dyn Fn() -> usize| {
        let start = std::time::Instant::now();
        let mut n = 0;
        for _ in 0..5 {
            n = f();
        }
        (start.elapsed().as_secs_f64() / 5.0 * 1e3, n)
    };
    let (replay_ms, replay_ids) = time(&|| parse(&m, &messages, false).ids.len());
    let plain = strip(&messages);
    let (template_ms, template_ids) = time(&|| m.tok.encode(&qwen3::chat::render_prompt_as(&m.format, &request(&m, &plain, false)).unwrap().text).len());
    let last = parse(&m, &messages, false);
    assert!(last.history.unwrap().append_only());
    println!("generated ids {generated_total}, assistant text {text_total} bytes; ids as JSON integers {json_ints} B ({:.2} B/id), base64 of LE u32 {le_u32_base64} B ({:.2} B/id), varint+base64 (the standard) {varint_base64} B ({:.2} B/id)", json_ints as f64 / generated_total as f64, le_u32_base64 as f64 / generated_total as f64, varint_base64 as f64 / generated_total as f64);
    println!("requests: {req_without} B without blobs, {req_with} B with ({:+.1}%); worst call +{worst_req} B", (req_with as f64 / req_without as f64 - 1.0) * 100.0);
    println!("responses: {resp_without} B without blob, {resp_with} B with; worst blob {worst_resp_blob} B");
    println!("final request: replay {replay_ms:.1} ms ({replay_ids} ids), template render+encode {template_ms:.1} ms ({template_ids} ids)");
}
