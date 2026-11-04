// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in continuous
// batching of mixture-of-experts decoders then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The paged batched decoder engine on the REAL DeepSeek-OCR decoder weights,
//! against the single-sequence decoder it replaces on the serving path, token
//! for token: a prompt long enough to be read in several chunks, an image run
//! of the production 273 rows, and four sequences decoding together through the
//! scheduler.
//!
//! Slow lane: both decoders are resident at once (2 x ~11.4 GiB of fp32 weights).
//! Skips itself when the checkpoint is absent.

use deepseek2::model::{DeepseekV2, Sizes};
use deepseek2::serve::{pipelines, Engine, EngineConfig, Precision};
use deepseek2::DeepseekV2Config;
use gpu_core::Gpu;
use model::kv_tier::KvTier;
use model::paged::BlockTable;
use model::serve::{PagedDecoder, PrefillProgress, PromptEmbeds, Request, SampleParams, Scheduler};

const STORE: &str = "deepseek-ai/DeepSeek-OCR";
const N_NEW: u32 = 24;
const CTX: u32 = 1024;

fn prompt(seed: u32, len: usize, vocab: u32) -> Vec<u32> {
    (0..len as u32).map(|i| 3 + (seed.wrapping_mul(2654435761).wrapping_add(i.wrapping_mul(40503)) >> 7) % (vocab - 3)).collect()
}

#[ignore = "real 2.9 B-parameter decoder resident twice (~23 GiB of fp32). `cargo test --release -p brain-deepseekocr --test serve_real -- --ignored --nocapture`"]
#[test]
fn the_engine_decodes_the_real_decoder_exactly_like_the_single_sequence_path() {
    let Some(dir) = brain_testutil::model_dir(STORE).map(std::path::PathBuf::from).filter(|d| d.join("config.json").exists()) else {
        brain_testutil::skip(&format!("{STORE} not in the model store"));
        return;
    };
    let files = deepseek2ocr::import::Files::locate(&dir).expect("locate");
    let cfg = deepseek2ocr::import::config(&files, 1).expect("config");
    let dcfg: DeepseekV2Config = cfg.decoder.clone();
    let reader = deepseek2ocr::import::decoder_reader(&files, &cfg).expect("reader");
    let src = deepseek2ocr::import::decoder_source(&files, &reader, &cfg).expect("source");
    let vocab = dcfg.vocab();
    let d = dcfg.d_model() as usize;

    let gpu = Gpu::new(&pipelines());
    let reference_gpu = gpu.new_like(deepseek2::PIPELINES);
    let ec = EngineConfig { max_seq: CTX, pool_tokens: 4 * CTX, max_batch: 4, chunk: 512, block_size: 64, kv: KvTier::F32, topk: 64, precision: Precision::Reference };
    let engine = Engine::new(gpu, dcfg.clone(), &src, ec).expect("engine");
    let mut reference = DeepseekV2::new_sized(reference_gpu, dcfg.clone(), Sizes { b: 1, t: 1, ctx: CTX, chunk: 512, batched: false }, &src, false);
    drop(src);
    drop(reader);

    // Four text prompts (one past the 512-row chunk) and one image-run prompt, all in flight together.
    let texts: Vec<Vec<u32>> = [(1u32, 40usize), (2, 700), (3, 9), (4, 130)].iter().map(|&(s, n)| prompt(s, n, vocab)).collect();
    let image: Vec<f32> = (0..273 * d).map(|i| (((i * 2654435761usize) >> 9) % 2001) as f32 / 1000.0 - 1.0).collect();
    let image_prompt = prompt(5, 283, vocab);

    let mut sched = Scheduler::new(engine, 5);
    let ids: Vec<u64> = texts.iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new: N_NEW as usize, stop: Vec::new() })).collect();
    let embeds = PromptEmbeds::run(d, 1, image.clone()).expect("rows");
    let image_id = sched.submit_embedded(Request { prompt: image_prompt.clone(), max_new: N_NEW as usize, stop: Vec::new() }, embeds, SampleParams::greedy(), 0);
    let got = sched.run();

    for (p, id) in texts.iter().zip(&ids) {
        let want = reference.generate_greedy_kv(p, N_NEW)[p.len()..].to_vec();
        assert_eq!(got[id], want, "prompt of {} tokens diverged from the single-sequence decoder", p.len());
    }
    reference.enable_mm_splice(1, 273);
    reference.write_img_embeds(&image);
    let want = reference.generate_greedy_kv(&image_prompt, N_NEW)[image_prompt.len()..].to_vec();
    assert_eq!(got[&image_id], want, "the image-run prompt diverged from the splice path");
}

/// Prefill `prompt` (with its image rows, if any) into a fresh sequence of `engine`, then greedy-decode
/// `n_new` tokens: the next-token logits after the prompt and the tokens.
fn decode_alone(engine: &mut Engine, prompt: &[u32], embeds: Option<&PromptEmbeds>, n_new: usize) -> (Vec<f32>, Vec<u32>) {
    let mut table = BlockTable::new();
    let hidden = loop {
        let progress = match embeds {
            Some(e) => engine.prefill_chunk_embeds(&mut table, prompt, e, usize::MAX),
            None => engine.prefill_chunk(&mut table, prompt, usize::MAX),
        };
        if let PrefillProgress::Done(h) = progress {
            break h;
        }
    };
    let logits = engine.logits(&hidden);
    let mut tokens = vec![engine.admit_greedy(&hidden)];
    while tokens.len() < n_new {
        let next = engine.forward_batched_greedy(&mut [&mut table], &[*tokens.last().expect("a first token")]);
        tokens.push(next[0]);
    }
    engine.release_table(&mut table);
    (logits, tokens)
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

/// The tensor-core tier on the real decoder: the next-token distribution after a prompt (plain
/// text, and with the 273-row image run) tracks the fp32 reference engine's. int8 activation
/// groups, int8 expert weights re-quantised from the checkpoint and bf16 dense operands cost a
/// little accuracy by design; this pins how little, and prints how far the greedy continuations
/// stay together.
#[ignore = "real 2.9 B-parameter decoder resident twice (fp32 reference ~13 GiB and the fast tier ~4 GiB). `cargo test --release -p brain-deepseek2ocr --test serve_real -- --ignored --nocapture`"]
#[test]
fn the_fast_tier_tracks_the_reference_on_the_real_decoder() {
    let Some(dir) = brain_testutil::model_dir(STORE).map(std::path::PathBuf::from).filter(|d| d.join("config.json").exists()) else {
        brain_testutil::skip(&format!("{STORE} not in the model store"));
        return;
    };
    let files = deepseek2ocr::import::Files::locate(&dir).expect("locate");
    let cfg = deepseek2ocr::import::config(&files, 1).expect("config");
    let dcfg: DeepseekV2Config = cfg.decoder.clone();
    let reader = deepseek2ocr::import::decoder_reader(&files, &cfg).expect("reader");
    let src = deepseek2ocr::import::decoder_source(&files, &reader, &cfg).expect("source");
    let (vocab, d) = (dcfg.vocab(), dcfg.d_model() as usize);
    let ec = |precision| EngineConfig { max_seq: CTX, pool_tokens: 2 * CTX, max_batch: 2, chunk: 512, block_size: 64, kv: KvTier::F32, topk: 64, precision };
    let gpu = Gpu::new(&pipelines());
    let mut fast = Engine::new(gpu.new_like(&pipelines()), dcfg.clone(), &src, ec(Precision::Fast)).expect("fast engine");
    let mut reference = Engine::new(gpu, dcfg, &src, ec(Precision::Reference)).expect("reference engine");
    assert!(fast.is_fast() && !reference.is_fast());

    let image: Vec<f32> = (0..273 * d).map(|i| (((i * 2654435761usize) >> 9) % 2001) as f32 / 1000.0 - 1.0).collect();
    let cases: Vec<(Vec<u32>, Option<PromptEmbeds>)> = vec![
        (prompt(1, 40, vocab), None),
        (prompt(2, 300, vocab), None),
        (prompt(5, 283, vocab), Some(PromptEmbeds::run(d, 1, image).expect("rows"))),
    ];
    for (p, e) in &cases {
        let (lf, tf) = decode_alone(&mut fast, p, e.as_ref(), N_NEW as usize);
        let (lr, tr) = decode_alone(&mut reference, p, e.as_ref(), N_NEW as usize);
        let cos = cosine(&lf, &lr);
        let together = tf.iter().zip(&tr).take_while(|(a, b)| a == b).count();
        println!("serve_real fast vs reference, prompt of {} tokens{}: logits cosine {cos:.5}, greedy tokens identical for the first {together} of {N_NEW}", p.len(), if e.is_some() { " (image run)" } else { "" });
        assert!(cos >= 0.999, "the fast tier's next-token logits drifted from the reference (cosine {cos})");
        assert!(together >= 12, "the fast tier's greedy continuation parted from the reference after {together} tokens");
        assert_eq!(tf[0], tr[0], "the first greedy token differs");
    }
}
