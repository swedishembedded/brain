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
    TextFrontend { tok: byte_tokenizer(), stop: Vec::new(), label: "test", parse: std::sync::Arc::new(raw_parse) }
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
    llmserve::run_open_scheduled(&mut scheduler(4), &front, std::slice::from_ref(&long), 4, 0, &mut |_, _| {}, &mut feed);

    let order: Vec<usize> = feed.finished.iter().map(|(i, _)| *i).collect();
    assert_eq!(order, vec![1, 0], "the late short request ends before the long one it joined");
    let mut by_index = vec![String::new(); 2];
    for (i, r) in feed.finished {
        by_index[i] = texts(vec![r]).remove(0);
    }
    assert_eq!(by_index[0], alone_long[0]);
    assert_eq!(by_index[1], alone_short[0]);
}

/// Agent-style multi-turn traffic on the qwen3 serving engine: each turn
/// resends the whole history plus a new message, with its own sampling
/// parameters and seed. The persistent scheduler's prefix cache must serve the
/// shared history from cached blocks (not recompute it), and the answer must be
/// what a cold engine gives for the same prompt.
#[test]
fn a_resent_history_is_served_from_the_qwen3_prefix_cache() {
    let cfg = qwen3::config::QwenConfig { vocab: 256, ..qwen3::config::QwenConfig::tiny() };
    let weights = qwen3::init_weights(&cfg, 5);
    let engine = |cfg: &qwen3::config::QwenConfig, w: &_| qwen3::serve::Engine::from_map(cfg.clone(), w, 8, 128, 4, 24, 16, false, false);
    let front = front();
    let history = "system: you are an agent. tools: shell, read. user: list the files in the repository root.";
    let turn1 = inv(history, 6, 0.0, 1);
    let turn2_prompt = format!("{history} assistant: ok. user: now count them and report.");
    let turn2 = inv(&turn2_prompt, 6, 0.8, 7);

    let mut warm = Scheduler::new(engine(&cfg, &weights), 4);
    run_batch_scheduled(&mut warm, &front, std::slice::from_ref(&turn1), &mut |_, _| {});
    let (hit0, _, _) = warm.prefix_stats();
    assert_eq!(hit0, 0, "nothing to reuse on the first turn");
    let got = texts(run_batch_scheduled(&mut warm, &front, std::slice::from_ref(&turn2), &mut |_, _| {}));
    let (hit, looked, cached) = warm.prefix_stats();
    let history_blocks = (history.len() / 8 * 8) as u64;
    assert!(hit >= history_blocks, "turn 2 must reuse the history's full blocks: {hit} of {looked} tokens hit ({history_blocks} expected), {cached} blocks cached");

    let mut cold = Scheduler::new(engine(&cfg, &weights), 4);
    let want = texts(run_batch_scheduled(&mut cold, &front, std::slice::from_ref(&turn2), &mut |_, _| {}));
    assert_eq!(got, want, "a cache hit must not change the answer");
}

/// A frontend whose routes can be marked out of date, over the plain one.
struct Stale<'a> {
    inner: &'a TextFrontend,
    stale: std::cell::Cell<bool>,
}

impl llmserve::Frontend for Stale<'_> {
    fn tok(&self) -> &QwenBpe {
        self.inner.tok()
    }
    fn stop(&self) -> &[u32] {
        self.inner.stop()
    }
    fn label(&self) -> &str {
        self.inner.label()
    }
    fn parse(&self, inv: &Invocation) -> Result<qwen3::chat::ParsedRequest, String> {
        self.inner.parse(inv)
    }
    fn stale(&self) -> bool {
        self.stale.get()
    }
}

/// An open group that is idle while the frontend's routes are out of date
/// takes no job: a request that arrives then is routed after the refresh, not
/// on the release that was replaced (a re-offered adapter id answered as the
/// old release when its predecessor's group had not yet closed).
#[test]
fn an_idle_open_group_with_stale_routes_hands_back_before_taking_a_job() {
    let plain = front();
    let front = Stale { inner: &plain, stale: std::cell::Cell::new(true) };
    let mut sched = scheduler(4);
    let mut feed = ScriptedFeed { late: vec![(1, inv("late", 3, 0.0, 1))], polls: 0, finished: Vec::new() };
    let handed = llmserve::run_open_scheduled(&mut sched, &front, &[], 4, 0, &mut |_, _| {}, &mut feed);
    assert_eq!(handed, llmserve::OpenGroup::Refresh { numbered: 0 }, "stale and idle: handed back");
    assert_eq!(feed.polls, 0, "no job was taken while the routes were stale");

    front.stale.set(false);
    assert_eq!(llmserve::run_open_scheduled(&mut sched, &front, &[], 4, 0, &mut |_, _| {}, &mut feed), llmserve::OpenGroup::Closed);
    assert_eq!(feed.finished.len(), 1, "the job is served once the routes are fresh");
}

/// A request that asks for them gets the token ids it was decoded from and the
/// log-probability of each sampled token, in the outputs of the same reply; a
/// request that does not ask pays for neither.
#[test]
fn token_ids_and_logprobs_come_back_only_when_asked_for() {
    let front = front();
    // Through the generate spec every surface validates against (D-Bus included), not just the scheduler.
    let spec = catalog::resident_llm::generate_spec("capture", true);
    let asked = spec.validate(inv("token capture", 6, 0.8, 17).set("logprobs", json!(true)).set("return_token_ids", json!(true))).expect("the generate spec declares both capture flags");
    let plain = inv("token capture", 6, 0.8, 17);
    let results = run_batch_scheduled(&mut scheduler(2), &front, &[asked, plain], &mut |_, _| {});
    let asked = results[0].clone().expect("request succeeds").outputs;
    let plain = results[1].clone().expect("request succeeds").outputs;

    let completion: Vec<i64> = serde_json::from_value(asked["completion_token_ids"].clone()).expect("ids");
    let logprobs: Vec<f64> = serde_json::from_value(asked["completion_logprobs"].clone()).expect("logprobs");
    let prompt: Vec<i64> = serde_json::from_value(asked["prompt_token_ids"].clone()).expect("prompt ids");
    assert_eq!(completion.len() as i64, asked["completion_tokens"].as_i64().unwrap());
    assert_eq!(logprobs.len(), completion.len());
    assert!(logprobs.iter().all(|lp| *lp <= 0.0), "{logprobs:?}");
    assert_eq!(prompt.len() as i64, asked["prompt_tokens"].as_i64().unwrap());
    assert_eq!(prompt, "token capture".bytes().map(i64::from).collect::<Vec<_>>(), "byte tokenizer: the prompt's own bytes");

    for key in ["completion_token_ids", "completion_logprobs", "prompt_token_ids"] {
        assert!(plain.get(key).is_none(), "{key} was not asked for");
    }
    assert_eq!(plain["text"], asked["text"], "capturing changes nothing about the decode");
    // Every reply names what decoded it.
    assert_eq!(asked["backend"], json!(gpu_core::backend_name()));
    assert!(asked["engine"].as_str().is_some_and(|e| e.starts_with("brain ")), "{asked:?}");
}

/// A feed whose waiting jobs sit behind a full group: `take` hands one over only when the
/// group asks for room, `screen` lets the group turn away what can never run. Records the
/// order of what happened to each.
struct WaitingFeed {
    waiting: Vec<(Invocation, bool)>,
    events: Vec<String>,
}

impl residency::JobFeed for WaitingFeed {
    fn take(&mut self, max: usize) -> Vec<Invocation> {
        let n = max.min(self.waiting.len());
        self.waiting.drain(..n).map(|(inv, _)| inv).collect()
    }
    fn screen(&mut self, check: &mut dyn FnMut(&Invocation) -> Option<String>) {
        let mut i = 0;
        while i < self.waiting.len() {
            if std::mem::replace(&mut self.waiting[i].1, true) {
                i += 1;
                continue;
            }
            match check(&self.waiting[i].0) {
                Some(why) => {
                    self.waiting.remove(i);
                    self.events.push(format!("refused: {why}"));
                }
                None => i += 1,
            }
        }
    }
    fn finish(&mut self, index: usize, result: ActionResult) {
        self.events.push(format!("finished {index}: {}", if result.is_ok() { "ok" } else { "error" }));
    }
    fn close(&mut self) -> bool {
        self.waiting.is_empty()
    }
}

/// A request that can never run is answered while the group is full of requests that will
/// take a long time: refusing it needs no slot. The one that fits keeps waiting for its
/// slot and is served after the running one ends.
#[test]
fn a_request_that_cannot_run_is_refused_while_the_group_is_full() {
    let front = front();
    let running = inv("a long one", 10, 0.0, 1);
    let too_long = inv(&"x".repeat(30), 4, 0.0, 2);
    let fits = inv("fits", 3, 0.0, 3);
    let mut feed = WaitingFeed { waiting: vec![(too_long, false), (fits, false)], events: Vec::new() };
    llmserve::run_open_scheduled(&mut scheduler(1), &front, std::slice::from_ref(&running), 1, 0, &mut |_, _| {}, &mut feed);

    assert_eq!(feed.events.len(), 3, "{:?}", feed.events);
    assert!(feed.events[0].starts_with("refused: test: needs 34 tokens, engine capacity is 24"), "the over-length request is refused first, before anything finishes: {:?}", feed.events);
    assert_eq!(&feed.events[1..], ["finished 0: ok", "finished 1: ok"], "the running request ends, then the one that fits is served");
}
