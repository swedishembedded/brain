// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The generic continuous-batching glue (`llmserve::run_batch_scheduled`)
//! over a real engine: a group of requests decoded together through
//! `model::serve::Scheduler` reproduces what each request produces alone,
//! sequences join and leave the batch between steps, and a cancelled request
//! leaves without disturbing the rest. Tiny Qwen3.8 on the CPU backend.

use capability::{ActionResult, Invocation};
use data::qwen_tokenizer::QwenBpe;
use llmserve::{run_batch_scheduled, TextFrontend};
use model::serve::Scheduler;
use qwen35::config::Qwen35Config;
use qwen35::serve::Engine;
use serde_json::json;

/// A byte-level tokenizer over ids `0..256`, so any ASCII prompt encodes
/// inside the tiny model's vocabulary.
fn byte_tokenizer() -> QwenBpe {
    let dir = std::env::temp_dir().join(format!("scheduled-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
    let tokenizer = json!({"model": {"vocab": vocab, "merges": []}, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}});
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = dir.join(format!("tokenizer-{}.json", NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
    std::fs::write(&path, tokenizer.to_string()).unwrap();
    QwenBpe::from_file(path.to_str().unwrap()).unwrap()
}

fn raw_parse(tok: &QwenBpe, inv: &Invocation) -> Result<qwen3::chat::ParsedRequest, String> {
    qwen3::chat::parse_request(tok, inv)
}

fn front() -> TextFrontend {
    TextFrontend { tok: byte_tokenizer(), stop: Vec::new(), label: "test", parse: raw_parse }
}

fn cfg() -> Qwen35Config {
    Qwen35Config { vocab: 256, ..Qwen35Config::tiny() }
}

fn scheduler(max_running: usize) -> Scheduler<Engine> {
    let cfg = cfg();
    let weights = qwen35::init::init_weights(&cfg, 11);
    Scheduler::new(Engine::from_map_on(&gpu_core::Gpu::new_cpu(qwen35::model::pipelines()), cfg, &weights, 24, 8), max_running)
}

fn inv(prompt: &str, max_new: i64, temp: f64, seed: i64) -> Invocation {
    Invocation::new().set("prompt", json!(prompt)).set("chat", json!(false)).set("max_new", json!(max_new)).set("temp", json!(temp)).set("seed", json!(seed))
}

fn texts(results: Vec<ActionResult>) -> Vec<String> {
    results.into_iter().map(|r| r.expect("request succeeds").outputs.get("text").and_then(|t| t.as_str().map(str::to_string)).expect("text output")).collect()
}

/// Continuous batching must not change what any sequence produces: more
/// requests than the scheduler runs at once, of different lengths, so
/// sequences join as others leave. Greedy output is token-identical to
/// the same request run alone; a sampled request reproduces under its
/// own seed whatever shares the batch.
#[test]
fn a_batch_reproduces_each_request_run_alone() {
    let front = front();
    let invs = vec![
        inv("alpha", 9, 0.0, 1),
        inv("be", 3, 0.0, 2),
        inv("gamma ray", 7, 0.8, 41),
        inv("d", 5, 0.8, 42),
        inv("epsilon", 2, 0.0, 5),
    ];
    let alone: Vec<String> = invs.iter().flat_map(|i| texts(run_batch_scheduled(&mut scheduler(1), &front, std::slice::from_ref(i), &mut |_, _| {}))).collect();
    // Two running at a time over five requests: every sequence after the
    // second is admitted into a batch that is already decoding.
    let batched = texts(run_batch_scheduled(&mut scheduler(2), &front, &invs, &mut |_, _| {}));
    assert_eq!(batched, alone);
    let wide = texts(run_batch_scheduled(&mut scheduler(8), &front, &invs, &mut |_, _| {}));
    assert_eq!(wide, alone);
}

/// A request cancelled before it starts is finished on its own, without
/// disturbing the others in the same batch.
#[test]
fn a_cancelled_request_leaves_the_batch_without_disturbing_the_rest() {
    let front = front();
    let keep = inv("keep me", 6, 0.0, 3);
    let alone = texts(run_batch_scheduled(&mut scheduler(1), &front, std::slice::from_ref(&keep), &mut |_, _| {}));
    let mut doomed = inv("doomed", 6, 0.0, 4);
    doomed.cancel = capability::CancelToken::armed();
    doomed.cancel.cancel();
    let results = run_batch_scheduled(&mut scheduler(4), &front, &[doomed, keep], &mut |_, _| {});
    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok());
    assert_eq!(texts(vec![results[1].clone()]), alone);
}

/// Hands a group scripted late jobs: job `i` becomes available at the
/// `after_polls`-th call of `take`, and every result is recorded in the order
/// it was delivered.
struct ScriptedFeed {
    late: Vec<(usize, Invocation)>,
    polls: usize,
    finished: Vec<(usize, ActionResult)>,
}

impl residency::JobFeed for ScriptedFeed {
    fn take(&mut self, max: usize) -> Vec<Invocation> {
        self.polls += 1;
        let mut out = Vec::new();
        while out.len() < max && self.late.first().is_some_and(|(at, _)| *at <= self.polls) {
            out.push(self.late.remove(0).1);
        }
        out
    }
    fn finish(&mut self, index: usize, result: ActionResult) {
        self.finished.push((index, result));
    }
    fn close(&mut self) -> bool {
        self.late.is_empty()
    }
}

/// A request that arrives while a group is already decoding joins it between
/// steps: it is answered when ITS generation ends (before the long request
/// that was running when it arrived), and neither output differs from running
/// that request alone.
#[test]
fn a_late_request_joins_the_running_group_and_is_answered_when_it_finishes() {
    let front = front();
    let long = inv("a long one", 10, 0.0, 1);
    let short = inv("short", 3, 0.8, 9);
    let alone_long = texts(run_batch_scheduled(&mut scheduler(1), &front, std::slice::from_ref(&long), &mut |_, _| {}));
    let alone_short = texts(run_batch_scheduled(&mut scheduler(1), &front, std::slice::from_ref(&short), &mut |_, _| {}));

    let mut feed = ScriptedFeed { late: vec![(3, short)], polls: 0, finished: Vec::new() };
    llmserve::run_open_scheduled(&mut scheduler(4), &front, std::slice::from_ref(&long), 4, &mut |_, _| {}, &mut feed);

    let order: Vec<usize> = feed.finished.iter().map(|(i, _)| *i).collect();
    assert_eq!(order, vec![1, 0], "the late short request ends before the long one it joined");
    let mut by_index = vec![String::new(); 2];
    for (i, r) in feed.finished {
        by_index[i] = texts(vec![r]).remove(0);
    }
    assert_eq!(by_index[0], alone_long[0]);
    assert_eq!(by_index[1], alone_short[0]);
}
