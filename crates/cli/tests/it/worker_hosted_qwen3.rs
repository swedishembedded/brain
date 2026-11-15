// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving large language models
// from killable worker processes without changing a single generated token,
// for its clients. If your team needs expertise in fault-isolated LLM
// inference then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The Qwen3 decoder hosted in `brain`'s internal worker mode, against the same
//! resident in process, on the store's `Qwen/Qwen3-0.6B`: a batch of greedy
//! generations decodes the same tokens and streams the same deltas, a
//! cancellation mid-stream ends that generation while the worker serves on, a
//! worker killed mid-stream fails its job and quarantines its card and the next
//! job is built in a new worker, an adapter offered while a group runs is used
//! by the next request (and decodes as it does in process) while an unknown one
//! is refused, and what a worker costs per token, per request and on the host is
//! measured. Every test skips, saying so, without the checkpoint or a GPU.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use brain_testutil::adapters::{scratch_dir, write_lora_adapter, Linear};
use capability::{CancelToken, Invocation, Outcome};
use catalog::resident_llm::{QwenResident, QwenServeConfig};
use checkpoint::st::ModelCard;
use llmserve::adapter_registry::{AdapterBankSettings, AdapterTargets, LiveAdapters};
use llmserve::adapter_release::verify_adapter;
use residency::supervised::{Supervision, WorkerHosting};
use residency::{Device, Executor, Policy};
use serde_json::json;

use super::worker_hosted::{brain, cards, live_workers, serial, supervision};
use super::worker_hosted_lfm::resident_mib;

const MODEL: &str = "Qwen/Qwen3-0.6B";
const ADAPTER: &str = "Qwen/Qwen3-0.6B:lab:worker-proof:v1";
const PROMPTS: [&str; 4] = ["The capital of Sweden is", "Rust is a programming language that", "Once upon a time, a small robot", "The three primary colors are"];

/// The checkpoint and tokenizer in the model store, or `None` (skipped) without them or a GPU.
fn checkpoint() -> Option<(String, String)> {
    let dir = PathBuf::from(brain_testutil::model_dir(MODEL)?);
    let (weights, tokenizer) = (dir.join("model.brain.safetensors"), dir.join("tokenizer.json"));
    if !weights.is_file() || !tokenizer.is_file() {
        brain_testutil::skip_unavailable(&format!("worker-hosted Qwen3: {MODEL} is not in the model store (brain pull {MODEL})"));
        return None;
    }
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("worker-hosted Qwen3: no GPU (the decoder is served on a card)");
        return None;
    }
    Some((weights.to_string_lossy().into_owned(), tokenizer.to_string_lossy().into_owned()))
}

fn hosting(supervision: &Supervision) -> WorkerHosting {
    serving::supervision::worker_hosting(&brain(), &[], &[catalog::resident_llm::FAMILY.to_string()], Duration::from_secs(300), supervision.health.clone()).expect("Qwen3 can be hosted")
}

/// How the tests serve the decoder: a short context and a batch of eight; `bank` gives it an adapter bank, and
/// without one each row decodes as it would alone (`--exact-decode`), so a batch's tokens cannot depend on which
/// requests happened to share a step in either process.
fn config(bank: bool, hosting: Option<WorkerHosting>) -> QwenServeConfig {
    let bank = bank.then_some(AdapterBankSettings { slots: 2, rank_cap: 8, targets: AdapterTargets::ATTENTION });
    QwenServeConfig { ctx: Some(2048), max_batch: 8, exact_decode: bank.is_none(), adapter_bank: bank, worker_hosting: hosting, ..QwenServeConfig::default() }
}

fn resident(c: &(String, String), cfg: QwenServeConfig) -> Arc<QwenResident> {
    Arc::new(QwenResident::from_card_configured(&c.0, &ModelCard::new(MODEL, "qwen"), Some(&c.1), cfg))
}

fn executor(resident: &Arc<QwenResident>, cards_: u32, supervision: Option<&Supervision>) -> Executor {
    Executor::start_supervised(vec![resident.clone()], cards(cards_), Policy::default(), None, supervision.cloned())
}

fn greedy(prompt: &str, max_new: u64) -> Invocation {
    Invocation::new().set("prompt", json!(prompt)).set("chat", json!(false)).set("max_new", json!(max_new)).set("temp", json!(0.0)).set("top_k", json!(1)).set("seed", json!(1)).set("return_token_ids", json!(true))
}

/// One generation's answer, the deltas it streamed, and when its first delta and its answer came.
struct Generated {
    result: Result<Outcome, String>,
    deltas: Vec<String>,
    first: Option<Duration>,
    took: Duration,
}

impl Generated {
    fn tokens(&self) -> Vec<u64> {
        let outcome = self.result.as_ref().unwrap_or_else(|e| panic!("the generation failed: {e}"));
        outcome.outputs["completion_token_ids"].as_array().expect("token ids were asked for").iter().map(|t| t.as_u64().unwrap()).collect()
    }

    fn text(&self) -> String {
        self.result.as_ref().unwrap_or_else(|e| panic!("the generation failed: {e}")).outputs["text"].as_str().unwrap().to_string()
    }
}

/// Run `inv` on `model`, calling `on_delta` with each streamed delta and how many came before it.
fn generate(exec: &Executor, model: &str, inv: Invocation, on_delta: impl Fn(usize) + Send + 'static) -> Generated {
    let deltas = Arc::new(Mutex::new((Vec::new(), None)));
    let seen = deltas.clone();
    let started = Instant::now();
    let result = exec.run_blocking(model, "generate", inv, move |p| {
        if let Some(delta) = p.delta {
            let mut seen = seen.lock().unwrap();
            seen.1.get_or_insert(started.elapsed());
            seen.0.push(delta);
            let n = seen.0.len();
            drop(seen);
            on_delta(n);
        }
    });
    let took = started.elapsed();
    let (deltas, first) = std::mem::take(&mut *deltas.lock().unwrap());
    Generated { result, deltas, first, took }
}

/// Every prompt at once, each on its own thread, so they decode in one group.
fn batch(exec: &Executor, prompts: &[&str], max_new: u64) -> Vec<Generated> {
    let asked: Vec<_> = prompts
        .iter()
        .map(|&p| {
            let exec = exec.clone();
            let inv = greedy(p, max_new);
            std::thread::spawn(move || generate(&exec, MODEL, inv, |_| {}))
        })
        .collect();
    asked.into_iter().map(|h| h.join().unwrap()).collect()
}

/// A batch of greedy generations through the executor decodes the same tokens, says the same text and streams the
/// same deltas in a worker as in process, and it ran in a worker this process started.
#[test]
fn a_worker_hosted_qwen3_decodes_the_tokens_and_streams_the_deltas_it_does_in_process() {
    let _serial = serial();
    let Some(c) = checkpoint() else { return };
    let local = executor(&resident(&c, config(false, None)), 1, None);
    let want = batch(&local, &PROMPTS, 24);
    local.shutdown();

    let supervision = supervision();
    let before = live_workers();
    let hosted = executor(&resident(&c, config(false, Some(hosting(&supervision)))), 1, Some(&supervision));
    let got = batch(&hosted, &PROMPTS, 24);
    assert!(live_workers().iter().any(|p| !before.contains(p)), "the batch ran in a worker this process started");
    hosted.shutdown();

    for (i, (want, got)) in want.iter().zip(&got).enumerate() {
        assert_eq!(got.tokens(), want.tokens(), "prompt {i}: the greedy tokens");
        assert_eq!(got.text(), want.text(), "prompt {i}: the text");
        assert_eq!(got.deltas, want.deltas, "prompt {i}: the streamed deltas");
        assert_eq!(got.deltas.concat(), got.text(), "prompt {i}: the deltas spell the text");
        assert!(want.tokens().len() > 1, "prompt {i}: the comparison is not vacuous");
    }
}

/// A generation whose caller cancels it after its third delta finishes `cancelled` well before its token budget,
/// and the same worker answers the next generation.
#[test]
fn a_cancelled_generation_stops_in_its_worker_and_the_worker_serves_on() {
    let _serial = serial();
    let Some(c) = checkpoint() else { return };
    let supervision = supervision();
    let before = live_workers();
    let exec = executor(&resident(&c, config(false, Some(hosting(&supervision)))), 1, Some(&supervision));
    generate(&exec, MODEL, greedy(PROMPTS[0], 4), |_| {}).result.expect("the warm-up runs");
    let workers: Vec<u32> = live_workers().into_iter().filter(|p| !before.contains(p)).collect();
    assert_eq!(workers.len(), 1, "{workers:?}");

    let mut inv = greedy(PROMPTS[1], 1024);
    inv.cancel = CancelToken::armed();
    let token = inv.cancel.clone();
    let cancelled = generate(&exec, MODEL, inv, move |n| {
        if n == 3 {
            token.cancel();
        }
    });
    // A generation reports its cancellation as how it finished, with the text it had, as it does in process.
    let outcome = cancelled.result.as_ref().unwrap_or_else(|e| panic!("a cancelled generation answers with what it had: {e}"));
    assert_eq!(outcome.outputs["finish_reason"], json!("cancelled"), "{:?}", outcome.outputs);
    assert!(cancelled.tokens().len() < 64, "the decode stopped soon after the cancel, not at its budget: {} tokens", cancelled.tokens().len());

    generate(&exec, MODEL, greedy(PROMPTS[2], 8), |_| {}).result.expect("the worker serves on");
    let after: Vec<u32> = live_workers().into_iter().filter(|p| !before.contains(p)).collect();
    assert_eq!(after, workers, "the same worker answered");
    assert_eq!(supervision.health.faults(&Device::Gpu(0)), 0, "a cancellation is no fault");
    exec.shutdown();
}

/// A worker killed with `SIGKILL` at its fifth streamed token fails that generation at once, naming the signal, and
/// quarantines its card; the next generation is built in a new worker and answers.
#[test]
fn a_worker_killed_mid_stream_fails_its_generation_and_the_next_is_built_anew() {
    let _serial = serial();
    let Some(c) = checkpoint() else { return };
    let supervision = supervision();
    let before = live_workers();
    let exec = executor(&resident(&c, config(false, Some(hosting(&supervision)))), 1, Some(&supervision));
    generate(&exec, MODEL, greedy(PROMPTS[0], 4), |_| {}).result.expect("the warm-up runs");
    let worker = live_workers().into_iter().find(|p| !before.contains(p)).expect("a worker this process started");

    let started = Instant::now();
    let killed = generate(&exec, MODEL, greedy(PROMPTS[1], 1024), move |n| {
        if n == 5 {
            // SAFETY: kill(2) on a process this test started.
            assert_eq!(unsafe { libc::kill(worker as libc::pid_t, libc::SIGKILL) }, 0, "kill {worker}");
        }
    });
    assert!(started.elapsed() < Duration::from_secs(60), "the failure took {:?}", started.elapsed());
    let err = killed.result.expect_err("a generation whose worker died fails");
    for needle in ["lost its worker", "signal 9", "quarantined"] {
        assert!(err.contains(needle), "the error must say {needle:?}: {err}");
    }
    assert!(killed.deltas.len() >= 5 && killed.deltas.len() < 64, "{} deltas", killed.deltas.len());
    assert_eq!(supervision.health.faults(&Device::Gpu(0)), 1, "the death is recorded against its card, once");

    generate(&exec, MODEL, greedy(PROMPTS[2], 8), |_| {}).result.expect("the next generation is built in a new worker");
    let now: Vec<u32> = live_workers().into_iter().filter(|p| !before.contains(p)).collect();
    assert!(now.len() == 1 && now[0] != worker, "a new worker: {now:?}, the killed one was {worker}");
    exec.shutdown();
}

/// Every attention linear of Qwen3-0.6B, in the shapes an adapter file stores them.
fn attention_linears() -> Vec<Linear> {
    let cfg = qwen3::config::QwenConfig::qwen3_0_6b();
    let (d, hq, hkv) = (cfg.d_model as usize, cfg.q_dim() as usize, cfg.kv_dim() as usize);
    (0..cfg.n_layers)
        .flat_map(|l| [("wq", hq, d), ("wk", hkv, d), ("wv", hkv, d), ("wo", d, hq)].map(|(leaf, out, inn)| Linear { base: format!("blocks.{l}.attn.{leaf}.weight"), out, inn }))
        .collect()
}

/// Offer the adapter at `path` on `exec` while a base generation streams, then ask the adapter once that one is done.
fn adapter_after_offer_mid_group(exec: &Executor, resident: &Arc<QwenResident>, path: &Path) -> (Generated, Generated) {
    let live = Arc::new(Mutex::new(Some((LiveAdapters::new(resident.clone(), exec.clone()), verify_adapter(path, &resident.served_base()).expect("the adapter verifies")))));
    let offering = live.clone();
    let base = generate(exec, MODEL, greedy(PROMPTS[3], 48), move |n| {
        if n == 2 {
            let (live, loaded) = offering.lock().unwrap().take().expect("offered once");
            live.serve(loaded).expect("the adapter is offered while the group runs");
        }
    });
    assert!(live.lock().unwrap().is_none(), "the adapter was offered while the base streamed");
    let adapted = generate(exec, ADAPTER, greedy(PROMPTS[3], 24), |_| {});
    (base, adapted)
}

/// An adapter offered while a group runs in the worker is used by the next request, which decodes as the same
/// request does in process and names the release the serving process offers; a name no one offers is refused.
#[test]
fn an_adapter_offered_while_a_worker_group_runs_is_used_by_the_next_request() {
    let _serial = serial();
    let Some(c) = checkpoint() else { return };
    let dir = scratch_dir("worker-hosted-qwen3-adapter");
    let path = dir.join("adapter.safetensors");
    write_lora_adapter(&path, ADAPTER, MODEL, 4, &attention_linears(), 7);

    let local_resident = resident(&c, config(true, None));
    let local = executor(&local_resident, 1, None);
    let (local_base, local_adapted) = adapter_after_offer_mid_group(&local, &local_resident, &path);
    local.shutdown();

    let supervision = supervision();
    let hosted_resident = resident(&c, config(true, Some(hosting(&supervision))));
    let hosted = executor(&hosted_resident, 1, Some(&supervision));
    let (base, adapted) = adapter_after_offer_mid_group(&hosted, &hosted_resident, &path);
    assert_eq!(base.tokens(), local_base.tokens(), "the base request");
    assert_eq!(adapted.tokens(), local_adapted.tokens(), "the adapter request decodes as in process");
    assert_ne!(adapted.text(), generate(&hosted, MODEL, greedy(PROMPTS[3], 24), |_| {}).text(), "the adapter changed the decode, so it was used");
    let identity = hosted.served_adapter(ADAPTER).expect("the adapter is offered");
    assert_eq!(adapted.result.as_ref().unwrap().outputs["adapter"], json!({"id": identity.id, "digest": identity.digest, "version": identity.version}), "the reply names the offered release");
    let unknown = generate(&hosted, "Qwen/Qwen3-0.6B:lab:nobody:v1", greedy(PROMPTS[3], 4), |_| {});
    assert!(unknown.result.is_err_and(|e| e.contains("nobody")), "a name no one offers is refused");
    hosted.shutdown();
    std::fs::remove_dir_all(&dir).ok();
}

/// Host memory only `pid` holds, in MiB: its anonymous and shared-memory pages. The rest of its resident set is the
/// `brain` executable and the driver's libraries, file pages the serving process maps too.
fn private_mib(pid: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let kib = |field: &str| -> Option<u64> { status.lines().find_map(|l| l.strip_prefix(field))?.split_whitespace().next()?.parse().ok() };
    Some((kib("RssAnon:")? + kib("RssShmem:")?) / 1024)
}

/// Time to first token, and decode tokens per second after it, of one stream and of a batch of eight.
fn speed(exec: &Executor) -> (Duration, f64, Duration, f64) {
    generate(exec, MODEL, greedy(PROMPTS[0], 8), |_| {}).result.expect("warm-up");
    let decode_rate = |g: &[Generated]| {
        let tokens: usize = g.iter().map(|g| g.tokens().len().saturating_sub(1)).sum();
        let span = g.iter().map(|g| g.took).max().unwrap() - g.iter().filter_map(|g| g.first).min().unwrap();
        tokens as f64 / span.as_secs_f64()
    };
    let one: Vec<Generated> = (0..3).map(|_| generate(exec, MODEL, greedy(PROMPTS[1], 64), |_| {})).collect();
    let mut ttft: Vec<Duration> = one.iter().filter_map(|g| g.first).collect();
    ttft.sort();
    let single = one.iter().map(|g| decode_rate(std::slice::from_ref(g))).fold(0.0, f64::max);
    let eight: Vec<&str> = PROMPTS.iter().chain(PROMPTS.iter()).copied().collect();
    let group = batch(exec, &eight, 64);
    let mut batch_ttft: Vec<Duration> = group.iter().filter_map(|g| g.first).collect();
    batch_ttft.sort();
    (ttft[1], single, batch_ttft[4], decode_rate(&group))
}

/// What a worker costs a generation: time to first token and decode speed for one stream and a batch of eight,
/// in process and in a worker on the same card, and the worker's resident host memory.
#[test]
fn a_worker_hosted_qwen3_costs_a_round_trip_per_request_and_little_per_token() {
    let _serial = serial();
    let Some(c) = checkpoint() else { return };
    let local = executor(&resident(&c, config(false, None)), 1, None);
    let (local_ttft, local_one, local_batch_ttft, local_batch) = speed(&local);
    local.shutdown();

    let supervision = supervision();
    let before = live_workers();
    let hosted = executor(&resident(&c, config(false, Some(hosting(&supervision)))), 1, Some(&supervision));
    let (ttft, one, batch_ttft, batch_rate) = speed(&hosted);
    let worker = live_workers().into_iter().find(|p| !before.contains(p)).expect("a worker this process started");
    let (host, private) = (resident_mib(worker), private_mib(worker));
    hosted.shutdown();

    eprintln!("Qwen3-0.6B, one stream of 64 tokens: time to first token {local_ttft:?} in process, {ttft:?} in a worker; decode {local_one:.1} and {one:.1} tokens/s");
    eprintln!("Qwen3-0.6B, a batch of eight: median time to first token {local_batch_ttft:?} in process, {batch_ttft:?} in a worker; decode {local_batch:.1} and {batch_rate:.1} tokens/s");
    let shown = |m: Option<u64>| m.map_or("an unreported amount".to_string(), |m| format!("{m} MiB"));
    let charged = residency::supervised::WORKER_HOST_BYTES >> 20;
    eprintln!("the worker holds {} resident on the host, {} of it its own (placement charges {charged} MiB)", shown(host), shown(private));
    if let Some(private) = private {
        assert!(private <= charged, "a worker holds {private} MiB of its own on the host, placement charges {charged} MiB");
    }
    assert!(ttft < local_ttft + Duration::from_millis(250), "a worker added {:?} before the first token", ttft.saturating_sub(local_ttft));
    assert!(one > local_one * 0.8, "a worker decodes one stream at {one:.1} tokens/s against {local_one:.1}");
    assert!(batch_rate > local_batch * 0.8, "a worker decodes a batch at {batch_rate:.1} tokens/s against {local_batch:.1}");
}
