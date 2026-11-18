// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for on-policy training on FP8 model releases served at their stored
// precision, for its clients. If your team needs expertise in measuring and closing the gap between what a serving
// engine sampled from and what a trainer computes, then you can procure our services by sending an email to
// info@swedishembedded.com.

// hf-fixture: qwen35
//! SPEC: the engine that serves a HuggingFace FP8 release (the Qwen3.8-27B-FP8 path) scores a continuation it
//! generated - the `policy_trace` of a reply and the `score` action - under the conventions of a policy trace
//! (knowledge #442): `log softmax(logits / T)` over the WHOLE vocabulary at the sampling temperature, the stop token
//! included. Held against the decode-time figure of the same tokens: a request drawn at temperature `T` with no top-k,
//! top-p or min-p samples from exactly that distribution, so the sampler's own log-probability of each drawn token
//! is the number the trace must reproduce, up to the prefill-versus-decode rounding of the engine.
//!
//! The `ignored:` sweeps are the measurements of knowledge #446 (the KV tier's share of the serve/score gap) and #447
//! (whether a decode depends on its co-batch) on this fixture; they print a table and assert nothing about the size.
//!
//! Run with `-- --device cpu --backend cpu` (the kernels on the host's cores) or `-- --device gpu<i> --backend cuda`;
//! the sweeps with `--ignored` as well.

use model::kv_tier::KvTier;
use model::serve::{Request, SampleParams, Scheduler};
use qwen35::config::Qwen35Config;
use qwen35::int8_gguf_resident::{budgeted_gpus, held_tiers_at, Qwen35GgufEngine, Qwen35GgufResident};
use residency::ResidentModel;

mod common;
use common::hf_fixture::{fp8_resident_cfg, write_checkpoint_conditioned, write_tokenizer};

const CAP: u32 = 192;
/// The fixture's end-of-text and end-of-turn ids ([`write_tokenizer`]): the stop tokens of every request here.
const STOP: [u32; 2] = [62, 63];

/// A resident over a fresh FP8 fixture directory with `kv` K/V and the decode mode `exact`, and that directory (removed
/// by the caller once the resident is built).
fn resident(tag: &str, kv: KvTier, exact: bool) -> (Qwen35GgufResident, std::path::PathBuf, memauth::Device) {
    resident_with(tag, kv, exact, gpu_core::ActChoice::Auto)
}

/// [`resident`] with the activation stream between the FP8 products chosen by `act`.
fn resident_with(tag: &str, kv: KvTier, exact: bool, act: gpu_core::ActChoice) -> (Qwen35GgufResident, std::path::PathBuf, memauth::Device) {
    let cfg = Qwen35Config { linear_num_key_heads: 1, linear_num_value_heads: 2, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4, ..fp8_resident_cfg() };
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = write_checkpoint_conditioned(&cfg, &format!("fp8score-{tag}-{n}"), 128);
    write_tokenizer(&dir);
    let path = dir.to_string_lossy().into_owned();
    let tier = held_tiers_at(&path).expect("the directory is a servable checkpoint");
    let set = gpu_core::devices::ambient_compute_set();
    let mut devices = budgeted_gpus(&set.gpus, 2 << 30);
    if devices.is_empty() && set.backend == gpu_core::devices::Backend::Cpu {
        // `--device cpu --backend cpu`: the kernels run on the host's cores and no card is opened, so card 0 stands
        // only as the resident's placement label.
        devices.push((memauth::Device::Gpu(0), 2 << 30));
    }
    let device = devices.first().expect("the test needs a GPU, or `--device cpu --backend cpu`").0;
    let resident = Qwen35GgufResident::new(path, devices, CAP, tier).with_max_batch(4).with_prefix_slots(4).with_kv_block_rows(16).with_kv_tier(kv).with_exact_decode(exact).with_act_choice(act);
    (resident, dir, device)
}

/// The scheduler the resident serves through, as `brain serve` builds it.
fn scheduler(tag: &str, kv: KvTier, exact: bool) -> Scheduler<Qwen35GgufEngine> {
    scheduler_with(tag, kv, exact, gpu_core::ActChoice::Auto)
}

/// [`scheduler`] over [`resident_with`].
fn scheduler_with(tag: &str, kv: KvTier, exact: bool, act: gpu_core::ActChoice) -> Scheduler<Qwen35GgufEngine> {
    let (resident, dir, device) = resident_with(tag, kv, exact, act);
    let (sched, _, _) = resident.activate_scheduler(&[device]).expect("the tiny FP8 directory loads");
    std::fs::remove_dir_all(dir).ok();
    sched
}

/// A prompt of `n` ids below the stop ids, different for each `salt`.
fn prompt(n: usize, salt: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 7 + salt * 13 + 3) % 60) as u32).collect()
}

/// An untruncated draw at `temp`: the sampler's distribution is the full softmax at that temperature.
fn untruncated(temp: f32) -> SampleParams {
    SampleParams { temp, top_k: 0, top_p: 1.0, min_p: 0.0, ..SampleParams::greedy() }
}

/// One finished request: the ids it generated (the stop token that ended it appended), the sampler's log-probability
/// of each, and the trace's full-distribution log-probability of the same ids.
struct Drawn {
    prompt: Vec<u32>,
    generated: Vec<u32>,
    sampler: Vec<f32>,
}

/// Run `requests` (prompt, seed) together on `sched` at `sample`, each for at most `max_new` tokens, to completion.
fn generate(sched: &mut Scheduler<Qwen35GgufEngine>, requests: &[(Vec<u32>, u64)], sample: SampleParams, max_new: usize) -> Vec<Drawn> {
    let ids: Vec<u64> = requests.iter().map(|(p, seed)| sched.submit_sampled(Request { prompt: p.clone(), max_new, stop: STOP.to_vec() }, sample, *seed)).collect();
    let mut done: std::collections::HashMap<u64, Drawn> = std::collections::HashMap::new();
    for _ in 0..10 * (max_new + 64) {
        let report = sched.step_report();
        assert!(report.rejected.is_empty(), "refused: {:?}", report.rejected);
        let mut lps: std::collections::HashMap<u64, Vec<f32>> = report.completed_logprobs.into_iter().collect();
        let mut meta: std::collections::HashMap<u64, model::serve::Completion> = report.completed_meta.into_iter().collect();
        for (id, mut generated) in report.completed {
            let mut sampler = lps.remove(&id).expect("logprobs of a completed request");
            let completion = meta.remove(&id).expect("meta of a completed request");
            if let (Some(stop), Some(lp)) = (completion.stop_token, completion.stop_logprob) {
                generated.push(stop);
                sampler.push(lp);
            }
            let prompt = requests[ids.iter().position(|&i| i == id).expect("one of ours")].0.clone();
            done.insert(id, Drawn { prompt, generated, sampler });
        }
        if done.len() == ids.len() {
            return ids.iter().map(|id| done.remove(id).expect("finished")).collect();
        }
    }
    panic!("the requests did not finish");
}

/// The trace's log-probabilities of `d`'s generated ids, scored by the serving engine at `temp`.
fn trace(sched: &mut Scheduler<Qwen35GgufEngine>, d: &Drawn, temp: f32) -> Vec<f32> {
    assert!(sched.scoring_ready(), "nothing is being read in");
    let all: Vec<u32> = d.prompt.iter().chain(&d.generated).copied().collect();
    sched.score_continuation(None, &all, d.prompt.len(), temp).expect("the engine scores its own generation")
}

fn gaps(a: &[f32], b: &[f32]) -> (f32, f32) {
    let abs: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
    (abs.iter().sum::<f32>() / abs.len().max(1) as f32, abs.iter().copied().fold(0.0, f32::max))
}

/// SPEC: the FP8 resident declares the teacher-forced scoring it implements: its manifest lists the `score` action
/// beside a `generate` that takes `policy_trace`, the two things a decoder that scores its own generation offers.
fn the_fp8_resident_declares_score_beside_the_policy_trace() {
    let (resident, dir, _) = resident("manifest", KvTier::F32, false);
    let manifest = resident.manifest();
    std::fs::remove_dir_all(dir).ok();
    let action = |name: &str| manifest.actions.iter().find(|a| a.name == name);
    assert!(action("generate").is_some_and(|g| g.params.iter().any(|p| p.name == "policy_trace")), "generate takes policy_trace");
    let score = action(llmserve::SCORE_ACTION).expect("the resident scores continuations, so it declares score");
    let names = |spec: &capability::ActionSpec| spec.params.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(score), names(&llmserve::score_spec()), "the one score contract every scheduler-backed chat model declares");
}

/// SPEC: a policy trace of a reply drawn at temperature `T` without truncation is the decode-time distribution of the
/// same tokens, scored again: the trace's log-probability of every generated id (the stop token included) equals the
/// sampler's own, which for an untruncated draw is `log softmax(logits / T)` over the whole vocabulary. Two
/// temperatures, so a trace taken at any temperature but the request's own would fail.
fn a_trace_reproduces_the_decode_time_log_probabilities_of_an_untruncated_draw() {
    // The bound on |trace - decode-time| per token, largest and mean. On the CPU backend the prefill and the decode run
    // the same arithmetic in another order: their rounding differs, never their distribution. On a card the FP8
    // products of a prefill and of a decode step are different kernels and round differently (measured on this fixture
    // on a GH200: mean 1.3e-2, largest 8.7e-2, whatever the activation stream or the GDN prefill form; knowledge #446);
    // the card bound holds that gap where it was measured, so it cannot grow unnoticed, and is not a claim that it is
    // small.
    let on_card = gpu_core::devices::ambient_compute_set().backend != gpu_core::devices::Backend::Cpu;
    let (largest, mean_bound) = if on_card { (0.15f32, 0.03f32) } else { (2e-3, 2e-3) };
    let mut sched = scheduler("parity", KvTier::F32, false);
    let (mut compared, mut stopped) = (0, 0);
    for (temp, seed) in [(1.0f32, 11u64), (0.7, 12)] {
        let drawn = generate(&mut sched, &[(prompt(24, 1), seed), (prompt(17, 2), seed + 100)], untruncated(temp), 24);
        for d in &drawn {
            compared += d.generated.len();
            stopped += usize::from(d.generated.last().is_some_and(|t| STOP.contains(t)));
            let scored = trace(&mut sched, d, temp);
            assert_eq!(scored.len(), d.generated.len());
            let (mean, worst) = gaps(&scored, &d.sampler);
            println!("T {temp}: {} tokens, |trace - sampler| mean {mean:.2e} largest {worst:.2e}", scored.len());
            assert!(worst < largest && mean < mean_bound, "T {temp}: trace {scored:?} vs decode-time {:?}", d.sampler);
            // The same ids under another temperature are another distribution, which the gate would see.
            let other = trace(&mut sched, d, temp + 0.5);
            assert!(gaps(&other, &d.sampler).0 > 3.0 * mean.max(mean_bound / 10.0), "T {temp}: the trace depends on its temperature");
        }
    }
    assert!(compared >= 32 && stopped > 0, "{compared} tokens compared, {stopped} replies ended on a stop token, which the trace scores too");
}

/// MEASUREMENT (#446): the serve/score gap each KV tier adds on this fixture. For each tier: the decode-time
/// log-probability against the trace of the same engine (decode against prefill, both at that tier) and the trace
/// against the f32-K/V trace of the same tokens (what a trainer without K/V rounding computes).
fn measure_the_gap_each_kv_tier_adds() {
    let temp = 1.0;
    let requests: Vec<(Vec<u32>, u64)> = (0..8).map(|i| (prompt(20 + 4 * i, i), 40 + i as u64)).collect();
    let mut reference = scheduler("kv-f32", KvTier::F32, false);
    println!("| K/V | tokens | decode vs own trace: mean / largest | trace vs f32-K/V trace: mean / largest |");
    for kv in [KvTier::F32, KvTier::BF16, KvTier::INT8, KvTier::FP8] {
        let mut sched = scheduler(&format!("kv-{kv}"), kv, false);
        let drawn = generate(&mut sched, &requests, untruncated(temp), 48);
        let (mut own, mut versus, mut n) = ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()), 0);
        for d in &drawn {
            let scored = trace(&mut sched, d, temp);
            let exact = trace(&mut reference, d, temp);
            n += scored.len();
            own.0.extend(&scored);
            own.1.extend(&d.sampler);
            versus.0.extend(&scored);
            versus.1.extend(&exact);
        }
        let (m1, w1) = gaps(&own.0, &own.1);
        let (m2, w2) = gaps(&versus.0, &versus.1);
        println!("| {kv} | {n} | {m1:.2e} / {w1:.2e} | {m2:.2e} / {w2:.2e} |");
    }
}

/// MEASUREMENT (#447): whether a seeded decode depends on what shares its batch. One request alone, then beside three
/// others, at temperature 1 with the same seed, default and exact decode: the first token that differs and the largest
/// log-probability difference before it.
fn measure_whether_a_decode_depends_on_its_co_batch() {
    let temp = 1.0;
    let others: Vec<(Vec<u32>, u64)> = (0..3).map(|i| (prompt(16 + 8 * i, 20 + i), 500 + i as u64)).collect();
    println!("| decode | request | tokens | diverges at | largest |lp| difference before it |");
    for exact in [false, true] {
        let mut sched = scheduler(&format!("cobatch-{exact}"), KvTier::F32, exact);
        for r in 0..4 {
            let me = (prompt(24 + r, 9 + r), 77 + r as u64);
            let alone = generate(&mut sched, std::slice::from_ref(&me), untruncated(temp), 64).remove(0);
            let mut together: Vec<(Vec<u32>, u64)> = vec![me.clone()];
            together.extend(others.iter().cloned());
            let beside = generate(&mut sched, &together, untruncated(temp), 64).remove(0);
            let diverge = alone.generated.iter().zip(&beside.generated).position(|(a, b)| a != b);
            let upto = diverge.unwrap_or(alone.generated.len().min(beside.generated.len()));
            let (_, worst) = gaps(&alone.sampler[..upto], &beside.sampler[..upto]);
            let at = diverge.map_or("never".to_string(), |i| i.to_string());
            println!("| {} | {r} | {} | {at} | {worst:.2e} |", if exact { "exact" } else { "default" }, alone.generated.len());
        }
    }
}

/// MEASUREMENT (#442 "not covered"): which numeric choice of the engine separates the trace (a prefill) from the
/// decode-time figure (the decode kernels) at f32 K/V: the activation stream between the FP8 products (auto, bf16 on a
/// card with bf16 tensor cores, or f32) and the Gated DeltaNet prefill form (as measured, or pinned to the exact form).
fn measure_what_separates_the_trace_from_the_decode() {
    use gpu_core::{ActChoice, ActPrecision};
    let temp = 1.0;
    let requests: Vec<(Vec<u32>, u64)> = (0..8).map(|i| (prompt(20 + 4 * i, i), 40 + i as u64)).collect();
    println!("| activations | GDN prefill | tokens | decode vs trace: mean / largest |");
    for (act, act_name) in [(ActChoice::Auto, "auto"), (ActChoice::Force(ActPrecision::F32), "f32")] {
        for exact_gdn in [false, true] {
            let _pin = exact_gdn.then(|| gpu_core::pin_candidate("gdn_prefill", "exact"));
            let mut sched = scheduler_with(&format!("sep-{act_name}-{exact_gdn}"), KvTier::F32, false, act);
            let drawn = generate(&mut sched, &requests, untruncated(temp), 48);
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for d in &drawn {
                a.extend(trace(&mut sched, d, temp));
                b.extend(&d.sampler);
            }
            let (mean, worst) = gaps(&a, &b);
            println!("| {act_name} | {} | {} | {mean:.2e} / {worst:.2e} |", if exact_gdn { "exact" } else { "measured" }, a.len());
        }
    }
}

gpu_core::card_tests!(
    the_fp8_resident_declares_score_beside_the_policy_trace,
    a_trace_reproduces_the_decode_time_log_probabilities_of_an_untruncated_draw;
    ignored: measure_the_gap_each_kv_tier_adds, measure_whether_a_decode_depends_on_its_co_batch, measure_what_separates_the_trace_from_the_decode,
);
