// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A bounded 27B-shaped decode bench per KV tier: the real Qwen3.8-27B attention geometry (24 query heads, 4 kv
//! heads, head_dim 256, the 3:1 recurrent/attention layer ratio) at four layers with random weights, so it fits two
//! card shares and needs no checkpoint. For every tier and batch it prints the step time of one batched greedy decode
//! over sequences that already hold a 3072-token prompt, and the KV bytes the engine reserves; for the FULL 27B it prints how
//! many sequences one 97 GiB card admits per tier (the planner's own count). Not a gate: `#[ignore]`, run with
//! `--ignored --nocapture` on a card.
//!
//! Swedish Embedded AB implements long-context inference serving for clients whose GPU memory decides how many users
//! one card carries. If your team needs expertise in measuring what a compact KV cache buys in concurrency and
//! decode time, you can procure our services by sending an email to info@swedishembedded.com.

use std::time::Instant;

use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::PagedDecoder;
use qwen35::config::Qwen35Config;
use qwen35::serve::Engine;

const PROMPT_TOKENS: usize = 3072;
const WARMUP_STEPS: usize = 4;
const TIMED_STEPS: usize = 16;
const BATCHES: [u32; 2] = [8, 16];

#[test]
#[ignore = "a measurement, not a spec: run on a card with --ignored --nocapture"]
fn decode_step_time_and_concurrency_per_kv_tier_at_the_27b_attention_shape() {
    let cfg = Qwen35Config { vocab: 320, n_layers: 4, block_size: 4096, max_position_embeddings: 4096, ..Qwen35Config::qwen38_27b() };
    let weights = qwen35::init::init_weights(&cfg, 11);
    let max_seq = (PROMPT_TOKENS + WARMUP_STEPS + TIMED_STEPS + 2) as u32;
    println!("{:>6} {:>6} {:>10} {:>10} {:>12} {:>14}", "tier", "batch", "min ms", "median ms", "tok/s (min)", "KV+state GiB");
    for tier in KvTier::ALL {
        for batch in BATCHES {
            let mut engine = Engine::from_map_kv(cfg.clone(), &weights, max_seq, batch, tier);
            let prompt: Vec<u32> = (0..PROMPT_TOKENS as u32).map(|i| (i * 7 + 3) % cfg.vocab).collect();
            let mut tables: Vec<BlockTable> = (0..batch).map(|_| BlockTable::new()).collect();
            let mut next: Vec<u32> = Vec::new();
            for table in tables.iter_mut() {
                let hidden = engine.prefill(table, &prompt);
                let logits = engine.logits(&hidden);
                next.push(logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap());
            }
            let step = |engine: &mut Engine, tables: &mut Vec<BlockTable>, next: &mut Vec<u32>| {
                let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                *next = engine.forward_batched_greedy(&mut refs, next);
            };
            for _ in 0..WARMUP_STEPS {
                step(&mut engine, &mut tables, &mut next);
            }
            // The minimum over the steps is the one the card was not shared in; the median shows the spread.
            let mut times: Vec<f64> = (0..TIMED_STEPS)
                .map(|_| {
                    let t = Instant::now();
                    step(&mut engine, &mut tables, &mut next);
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            times.sort_by(f64::total_cmp);
            let (min, median) = (times[0], times[TIMED_STEPS / 2]);
            println!("{:>6} {:>6} {:>10.2} {:>10.2} {:>12.0} {:>14.2}", tier, batch, min, median, batch as f64 * 1e3 / min, engine.kv_pool_bytes() as f64 / (1u64 << 30) as f64);
        }
    }

    // How many sequences one card admits for the FULL model, per tier (planner arithmetic, no device).
    let full = Qwen35Config::qwen38_27b();
    let tier_policy = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::I8);
    let usable = (97_871u64 << 20) - (2 << 30);
    for cap in [4096u32, 32_768, 131_072] {
        let counts: Vec<String> = KvTier::ALL
            .iter()
            .map(|&kv| {
                let n = (1..=1024u32).take_while(|&b| qwen35::int8_gguf_resident::sizing_refusal(&full, cap, &tier_policy, kv, b, 0, 0, &[usable]).is_none()).last().unwrap_or(0);
                format!("{kv} {n}")
            })
            .collect();
        println!("27B on one GH200 at {cap} tokens: sequences admitted: {}", counts.join(", "));
    }
}
