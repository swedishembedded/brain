// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in
// tensor-core mixture-of-experts decode with a stated accuracy contract then
// you can procure our services by sending an email to info@swedishembedded.com.

//! The engine's tensor-core tier (`Precision::Fast`: int8 routed experts, bf16 dense linears)
//! against its fp32 reference tier on the same weights.
//!
//! The two are not bitwise equal by design: the experts' activations are quantised to int8 per
//! 32-element group and the dense linears round their operands to bf16. The contract this file
//! pins is how far apart they may be, as the user sees it: the next-token distribution after a
//! prompt agrees to a cosine of at least [`LOGITS_COSINE`], the greedy token is the same, and a
//! batch of sequences decodes exactly what the fast tier decodes for each alone.
//!
//! A 64-wide model, not the 12-wide tiny one: the int8 tier needs every contraction width to be
//! a multiple of 32.

use deepseek2::config::DeepseekV2Config;
use deepseek2::serve::{pipelines, Engine, EngineConfig, Precision};
use gguf::deepseek_ocr::DeepseekOcrConfig;
use gpu_core::Gpu;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillProgress, Request, Scheduler};

/// Cosine between the reference and fast next-token logits. Measured on this model at its
/// weight scale (0.9999 and up).
const LOGITS_COSINE: f32 = 0.999;

/// `heads` attention heads of `head_dim`: d_model is their product, as in the served checkpoints.
fn shaped(heads: u32, head_dim: u32) -> DeepseekV2Config {
    DeepseekV2Config::from_shape(
        DeepseekOcrConfig {
            vocab: 96,
            n_layers: 3,
            d_model: heads * head_dim,
            max_position_embeddings: 128,
            rms_eps: 1e-6,
            tie_embeddings: false,
            n_heads: heads,
            n_kv_heads: heads,
            head_dim,
            rope_theta: 10_000.0,
            rotary_dim: head_dim,
            n_dense_layers: 1,
            ffn_hidden: 96,
            n_experts: 8,
            top_k: 2,
            moe_intermediate_size: 32,
            n_shared_experts: 2,
            n_expert_groups: 1,
            n_expert_groups_used: 1,
        },
        128,
    )
}

fn cfg() -> DeepseekV2Config {
    shaped(2, 32)
}

/// Random weights with enough spread that the next token depends on the prompt.
fn weights(c: &DeepseekV2Config) -> std::collections::HashMap<String, Vec<f32>> {
    let mut w = deepseek2::init_weights(c, 11);
    for (name, v) in w.iter_mut() {
        if !name.ends_with("norm.weight") && !name.ends_with("ln1.weight") && !name.ends_with("ln2.weight") {
            v.iter_mut().for_each(|x| *x *= 12.0);
        }
    }
    w
}

fn engine(precision: Precision) -> Option<Engine> {
    engine_of(cfg(), precision)
}

fn engine_of(c: DeepseekV2Config, precision: Precision) -> Option<Engine> {
    let gpu = Gpu::new(&pipelines());
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("the tensor-core tier needs workgroup reductions");
        return None;
    }
    let ec = EngineConfig { max_seq: 96, pool_tokens: 4 * 96, max_batch: 8, chunk: 16, block_size: 8, kv: KvTier::F32, topk: 8, precision };
    Some(Engine::new(gpu, c.clone(), &weights(&c), ec).expect("engine"))
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| 1 + (seed * 13 + i * 7 + i * i) % 90).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// The next-token logits after `prompt`, from a prefill in `engine`.
fn next_logits(engine: &mut Engine, prompt: &[u32]) -> Vec<f32> {
    let mut table = BlockTable::new();
    let hidden = loop {
        if let PrefillProgress::Done(h) = engine.prefill_chunk(&mut table, prompt, usize::MAX) {
            break h;
        }
    };
    let logits = engine.logits(&hidden);
    engine.release_table(&mut table);
    logits
}

#[test]
fn the_fast_tier_is_what_auto_picks_on_a_device_that_runs_it() {
    let Some(auto) = engine(Precision::Auto) else { return };
    assert!(auto.is_fast(), "Auto must pick the tensor-core tier on a device with workgroup reductions");
    assert!(!engine(Precision::Reference).expect("reference").is_fast());
}

#[test]
fn the_next_token_distribution_of_the_fast_tier_tracks_the_reference() {
    let (Some(mut fast), Some(mut reference)) = (engine(Precision::Fast), engine(Precision::Reference)) else { return };
    for (seed, len) in [(1u32, 5usize), (2, 17), (3, 40)] {
        let p = prompt(seed, len);
        let (a, b) = (next_logits(&mut fast, &p), next_logits(&mut reference, &p));
        let cos = cosine(&a, &b);
        println!("serve_fast prompt of {len}: logits cosine {cos:.5}");
        assert!(cos >= LOGITS_COSINE, "prompt of {len} tokens: the fast tier's logits drifted from the reference (cosine {cos})");
        let argmax = |v: &[f32]| v.iter().enumerate().fold(0, |best, (i, x)| if *x > v[best] { i } else { best });
        assert_eq!(argmax(&a), argmax(&b), "prompt of {len} tokens: the greedy token differs");
    }
}

/// Batched equals alone inside the fast tier: sequences share steps, not arithmetic.
#[test]
fn a_batch_in_the_fast_tier_decodes_what_each_decodes_alone() {
    let Some(_) = engine(Precision::Fast) else { return };
    let prompts: Vec<Vec<u32>> = [(1u32, 5usize), (2, 12), (3, 30)].iter().map(|&(s, n)| prompt(s, n)).collect();
    let alone = |p: &Vec<u32>| {
        let mut sched = Scheduler::new(engine(Precision::Fast).expect("fast"), 4);
        let id = sched.submit(Request { prompt: p.clone(), max_new: 8, stop: Vec::new() });
        sched.run().remove(&id).expect("finished")
    };
    let mut sched = Scheduler::new(engine(Precision::Fast).expect("fast"), 4);
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new: 8, stop: Vec::new() })).collect();
    let together = sched.run();
    for (p, id) in prompts.iter().zip(ids) {
        assert_eq!(together[&id], alone(p), "a batched request must decode as it does alone (prompt of {})", p.len());
    }
}

/// The fused decode step is recorded once per row count and replayed: what a recorded engine decodes is
/// what an engine that builds every step decodes, across a batch that shrinks as its sequences finish.
#[test]
fn a_replayed_decode_step_decodes_what_a_rebuilt_one_does() {
    // Head width 128 is where the fused split-key attention, the recordable step, applies.
    let make = |tapes: bool| {
        let mut e = engine_of(shaped(1, 128), Precision::Fast)?;
        e.set_decode_tapes(tapes);
        Some(e)
    };
    let (Some(recorded), Some(rebuilt)) = (make(true), make(false)) else { return };
    let run = |engine: Engine| {
        let mut sched = Scheduler::new(engine, 4);
        let ids: Vec<u64> = [(1u32, 5usize, 12usize), (2, 12, 6), (3, 30, 3)].iter().map(|&(s, n, new)| sched.submit(Request { prompt: prompt(s, n), max_new: new, stop: Vec::new() })).collect();
        let out = sched.run();
        let tokens: Vec<Vec<u32>> = ids.iter().map(|id| out[id].clone()).collect();
        (tokens, sched.into_decoder().tape_stats())
    };
    let ((got, stats), (want, _)) = (run(recorded), run(rebuilt));
    assert_eq!(got, want, "a replayed step must decode what a rebuilt one decodes");
    if stats.recordings == 0 {
        return brain_testutil::skip_unavailable("this device has no fused decode attention, so no step is recorded");
    }
    assert!(stats.replays > stats.recordings, "later steps of a shape must replay its recording: {stats:?}");
}

/// Kernels a fast-tier step may run on CUDA cores, with the reason: the router's logits are fp32 on purpose (the experts it
/// picks are a hard decision that a rounded logit can flip), and its product is `[rows, d_model] x [d_model, n_experts]`.
const FP32_BY_DESIGN: &[&str] = &["matmul_gemv_reg", "matmul_reg3"];

/// **No hot kernel of the fast tier falls back to fp32 CUDA cores.** A decode step and a prefill chunk are profiled on the
/// device's own kernel timer; every product in them is a tensor-core kernel (the library's bf16 GEMM for the dense linears
/// and the head, the int8 grouped MMA for the experts) except the declared fp32 router. A linear that quietly took a
/// CUDA-core kernel - an unaligned shape, a declined library product, a changed tier - fails here, by name.
#[test]
fn the_fast_tier_runs_its_products_on_tensor_cores() {
    let Some(mut engine) = engine_of(shaped(1, 128), Precision::Fast) else { return };
    if !engine.gpu().set_kernel_timing(true) {
        return brain_testutil::skip_unavailable("this backend cannot time kernels");
    }
    engine.set_decode_tapes(false);
    // Eight rows: from `GROUPED_MIN_ROWS` up the experts take the grouped MMA, below it the bandwidth-bound int8 GEMV.
    let prompts: Vec<Vec<u32>> = (0..8).map(|s| prompt(s + 1, 40)).collect();
    let mut tables: Vec<BlockTable> = prompts.iter().map(|_| BlockTable::new()).collect();
    let mut last: Vec<u32> = Vec::new();
    for (table, p) in tables.iter_mut().zip(&prompts) {
        let hidden = loop {
            if let PrefillProgress::Done(h) = engine.prefill_chunk(table, p, usize::MAX) {
                break h;
            }
        };
        last.push(engine.admit_greedy(&hidden));
    }
    for _ in 0..4 {
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        last = engine.forward_batched_greedy(&mut refs, &last);
    }
    let rows = engine.gpu().kernel_times().expect("timing was switched on");
    let audit = gpu_core::profile::ArithmeticAudit::of(&rows);
    eprintln!("fast tier: {}", audit.summary());
    let fallbacks = audit.fallbacks(FP32_BY_DESIGN);
    assert!(fallbacks.is_empty(), "products on CUDA cores in the fast tier: {fallbacks:?}\n{}", audit.summary());
    assert!(audit.tensor_core_ms > 0.0, "no tensor-core kernel ran at all: {rows:?}");
}

/// **A warmed engine records nothing while serving.** Recording a decode step costs more than running it, once per distinct row
/// count; a burst that ramps the batch from 1 to its ceiling used to pay for every size it passed through while its pages waited.
/// `warm_up` records them all, with throwaway sequences, before the first request.
#[test]
fn a_warmed_engine_records_no_decode_step_while_serving() {
    let Some(mut engine) = engine_of(shaped(1, 128), Precision::Fast) else { return };
    engine.warm_up();
    let warmed = engine.tape_stats();
    if warmed.recordings == 0 {
        return brain_testutil::skip_unavailable("this device has no fused decode attention, so no step is recorded");
    }
    assert_eq!(warmed.recordings, 8, "one recording per row count up to the batch ceiling");
    assert_eq!(engine.free_blocks(), 4 * 96 / 8, "the throwaway sequences gave their blocks back");
    let mut sched = Scheduler::new(engine, 8);
    let ids: Vec<u64> = (1..=8u32).map(|s| sched.submit(Request { prompt: prompt(s, 5 + s as usize), max_new: 4 + s as usize, stop: Vec::new() })).collect();
    let out = sched.run();
    assert!(ids.iter().all(|id| out.contains_key(id)));
    assert_eq!(sched.into_decoder().tape_stats().recordings, warmed.recordings, "serving recorded a row count the warm-up missed");
}
