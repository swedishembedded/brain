// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A decode step recorded once and replayed is the decode step built every
//! token: same tokens, same logits to the bit, through admissions, finishes and
//! a batch that changes shape - and a tape-using engine returns every CUDA
//! object it took.
//!
//! Swedish Embedded AB implements low-latency LLM decode on GPUs for its clients.
//! If your team needs expertise in taking the host out of the critical path of a
//! token loop then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Both engines run the same kernels over the same weights; what differs is only
//! whether the ~1600 dispatches of a step are built per token or replayed from a
//! recording, so the comparison is exact, not a tolerance. The int8 tier is the
//! one that can be recorded (the fp32 decode reads routing back mid-step).

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use data::rng::Rng;
use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use model::paged::BlockTable;
use qwen35moe::config::Qwen35Config;
use qwen35moe::serve::{Engine, EngineOptions};

/// The live-resource counters are process-global, so the tests of this file must
/// not overlap (the comparison test's engines would be counted as leaked by the
/// lifetime test).
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `model_i8_smoke`'s config: every quantised width a multiple of 32, a shared
/// expert of the routed experts' shape, both mixer types, several experts.
fn cfg() -> Qwen35Config {
    Qwen35Config {
        vocab: 29,
        block_size: 24,
        n_layers: 8,
        d_model: 32,
        rms_eps: 1e-6,
        max_position_embeddings: 24,
        tie_embeddings: false,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        attn_bias: false,
        rope_theta: 1.0e6,
        partial_rotary_factor: 0.5,
        mrope_section: [1, 1, 1],
        full_attention_interval: 4,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 3,
        n_experts: 6,
        top_k: 2,
        moe_intermediate_size: 32,
        shared_expert_intermediate_size: 32,
        lora: None,
    }
}

fn weights(c: &Qwen35Config) -> HashMap<String, Vec<f32>> {
    qwen35moe::init::init_weights(c, 31)
}

fn prompt(c: &Qwen35Config, seed: u64, n: usize) -> Vec<u32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| (r.next_u64() % c.vocab as u64) as u32).collect()
}

fn engine(w: &HashMap<String, Vec<f32>>, tapes: bool) -> Engine {
    let tier = TierPolicy::uniform(Dtype::I8);
    Engine::from_source(cfg(), w, EngineOptions::new(24, 4).with_tier(tier).with_prefill_chunk(4).with_decode_tapes(tapes))
}

#[test]
fn a_replayed_step_is_the_step_built_every_token() {
    let _s = serial();
    let c = cfg();
    let w = weights(&c);
    let probe = Engine::from_source(c.clone(), &w, EngineOptions::new(24, 1).with_tier(TierPolicy::uniform(Dtype::I8)));
    if !probe.gpu().caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("no packed int8 dot on this device");
    }
    drop(probe);
    let (mut taped, mut built) = (engine(&w, true), engine(&w, false));
    let prompts = [prompt(&c, 1, 7), prompt(&c, 2, 5), prompt(&c, 3, 9)];
    let mk = |e: &mut Engine| -> Vec<BlockTable> {
        prompts
            .iter()
            .map(|p| {
                let mut t = BlockTable::new();
                e.prefill(&mut t, p);
                t
            })
            .collect()
    };
    let (mut ta, mut tb) = (mk(&mut taped), mk(&mut built));

    // Steps 0..4: three sequences. Then one finishes, a fourth is admitted
    // mid-stream, and the batch changes shape; steps keep matching throughout.
    let mut toks = vec![5u32, 6, 7];
    for step in 0..9 {
        if step == 4 {
            let mut gone = ta.remove(1);
            taped.release_table(&mut gone);
            let mut gone = tb.remove(1);
            built.release_table(&mut gone);
            toks.remove(1);
        }
        if step == 7 {
            let p = prompt(&c, 9, 6);
            for (e, ts) in [(&mut taped, &mut ta), (&mut built, &mut tb)] {
                let mut t = BlockTable::new();
                e.prefill(&mut t, &p);
                ts.push(t);
            }
            toks.push(3);
        }
        let top_a = taped.forward_batched_topk(&mut ta.iter_mut().collect::<Vec<_>>(), &toks, 4);
        let top_b = built.forward_batched_topk(&mut tb.iter_mut().collect::<Vec<_>>(), &toks, 4);
        assert_eq!(top_a.len(), top_b.len());
        for (row, (a, b)) in top_a.iter().zip(&top_b).enumerate() {
            let ids = |v: &Vec<(u32, f32)>| v.iter().map(|x| x.0).collect::<Vec<_>>();
            let bits = |v: &Vec<(u32, f32)>| v.iter().map(|x| x.1.to_bits()).collect::<Vec<_>>();
            assert_eq!(ids(a), ids(b), "step {step} row {row}: the candidates differ");
            assert_eq!(bits(a), bits(b), "step {step} row {row}: the logits differ in their bits");
        }
        toks = top_a.iter().map(|r| r[0].0).collect();
    }
    // The greedy head records its own tape and agrees too.
    let ga = taped.forward_batched_greedy(&mut ta.iter_mut().collect::<Vec<_>>(), &toks);
    let gb = built.forward_batched_greedy(&mut tb.iter_mut().collect::<Vec<_>>(), &toks);
    assert_eq!(ga, gb);
    for mut t in ta {
        taped.release_table(&mut t);
    }
    for mut t in tb {
        built.release_table(&mut t);
    }
}

/// Recording, replaying, evicting and dropping tapes returns every CUDA object:
/// the tapes pin whole steps' scratch, and none of it may outlive the engine.
#[test]
fn an_engine_that_recorded_tapes_returns_every_cuda_object() {
    let _s = serial();
    use backend_cuda::exec::Context;
    use backend_cuda::live_resources;
    let Ok(_probe) = Context::open(0) else {
        return brain_testutil::skip_unavailable("no usable CUDA device");
    };
    let c = cfg();
    let w = weights(&c);
    let baseline = live_resources();
    for round in 0..3 {
        let mut e = engine(&w, true);
        if !e.gpu().caps().numeric.int8_dot {
            return brain_testutil::skip_unavailable("no packed int8 dot on this device");
        }
        // More shapes than the cache keeps, so eviction runs too.
        let mut tables: Vec<BlockTable> = (0..4)
            .map(|i| {
                let mut t = BlockTable::new();
                e.prefill(&mut t, &prompt(&c, 40 + i, 5 + i as usize));
                t
            })
            .collect();
        for n in 1..=4usize {
            for sub in 0..3 {
                let rows: Vec<&mut BlockTable> = tables.iter_mut().skip(sub.min(4 - n)).take(n).collect();
                let toks = vec![2u32; rows.len()];
                let mut rows = rows;
                e.forward_batched_greedy(&mut rows, &toks);
            }
        }
        for t in tables.iter_mut() {
            e.release_table(t);
        }
        drop(e);
        assert_eq!(live_resources(), baseline, "round {round}: dropping the engine left CUDA objects behind");
    }
}
