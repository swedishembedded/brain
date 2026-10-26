// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in profiling
// batched mixture-of-experts decode on accelerators then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The decoder engine's step cost at a batch size, without the server around it:
//! prefill `B` sequences (a 283-token OCR-shaped prompt with a 273-row image
//! run each), then time batched decode steps, and print the device counters
//! (`BRAIN_PROFILE=1` adds the per-kernel-kind table).
//!
//! ```text
//! cargo run --release -p brain-deepseek2ocr --example engine_bench -- \
//!     --dir <model dir> --batch 8 --steps 32 [--kv f32|bf16] [--precision auto|reference|fast]
//! ```

use deepseek2::serve::{pipelines, Engine, EngineConfig, Precision};
use gpu_core::Gpu;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillProgress, PromptEmbeds};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone()).unwrap_or_else(|| default.to_string());
    let dir = arg("--dir", "");
    let batch: u32 = arg("--batch", "1").parse().expect("--batch");
    let steps: usize = arg("--steps", "32").parse().expect("--steps");
    let prompt_len: usize = arg("--prompt", "283").parse().expect("--prompt");
    let kv = KvTier::parse(&arg("--kv", "f32")).expect("--kv");
    let precision = match arg("--precision", "auto").as_str() {
        "auto" => Precision::Auto,
        "reference" => Precision::Reference,
        "fast" => Precision::Fast,
        other => panic!("--precision auto|reference|fast, got {other}"),
    };
    assert!(!dir.is_empty(), "--dir <model dir>");

    let files = deepseek2ocr::import::Files::locate(&dir).expect("locate");
    let cfg = deepseek2ocr::import::config(&files, 1).expect("config");
    let reader = deepseek2ocr::import::decoder_reader(&files, &cfg).expect("reader");
    let src = deepseek2ocr::import::decoder_source(&files, &reader, &cfg).expect("source");
    let ec = EngineConfig { max_seq: 2048, pool_tokens: batch.max(1) * 1024, max_batch: batch.max(1), chunk: 512, block_size: 64, kv, topk: 64, precision };
    let t0 = Instant::now();
    let mut engine = Engine::new(Gpu::new(&pipelines()), cfg.decoder.clone(), &src, ec).expect("engine");
    eprintln!("engine built in {:.1}s", t0.elapsed().as_secs_f64());
    let d = cfg.decoder.d_model() as usize;
    let vocab = cfg.decoder.vocab();

    // OCR-shaped prompt: BOS, a 273-row image run, then text tokens.
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| if i == 0 { 0 } else { 3 + (i.wrapping_mul(40503) >> 3) % (vocab - 3) }).collect();
    let image_rows = 273.min(prompt_len.saturating_sub(1));
    let rows: Vec<f32> = (0..image_rows * d).map(|i| (((i * 2654435761usize) >> 9) % 2001) as f32 / 4000.0 - 0.25).collect();
    let embeds = PromptEmbeds::run(d, 1, rows).expect("rows");

    let mut tables: Vec<BlockTable> = (0..batch).map(|_| BlockTable::new()).collect();
    let mut last: Vec<u32> = Vec::new();
    let t_pre = Instant::now();
    for t in tables.iter_mut() {
        let hidden = loop {
            match engine.prefill_chunk_embeds(t, &prompt, &embeds, usize::MAX) {
                PrefillProgress::Done(h) => break h,
                PrefillProgress::Partial { .. } => {}
            }
        };
        last.push(engine.admit_greedy(&hidden));
    }
    let pre = t_pre.elapsed().as_secs_f64();
    eprintln!("prefill: {batch} x {prompt_len} tokens in {:.0} ms ({:.0} ms per sequence)", pre * 1e3, pre * 1e3 / batch as f64);

    if let Some(s) = engine.gpu().stats() {
        eprintln!("device counters after prefill: {s:?}");
    }
    let mut per_step = Vec::new();
    for _ in 0..steps {
        let t = Instant::now();
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        last = engine.forward_batched_greedy(&mut refs, &last);
        per_step.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let warm = &per_step[per_step.len().min(3)..];
    let mean = warm.iter().sum::<f64>() / warm.len().max(1) as f64;
    let mut sorted = warm.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    println!(
        "{{\"batch\":{batch},\"steps\":{steps},\"prefill_ms_per_seq\":{:.1},\"step_ms_mean\":{mean:.2},\"step_ms_p50\":{p50:.2},\"tok_per_s\":{:.1}}}",
        pre * 1e3 / batch as f64,
        batch as f64 * 1e3 / mean
    );
    if let Some(s) = engine.gpu().stats() {
        eprintln!("device counters at the end: {s:?}");
    }
    engine.gpu().dump_profile();
}
