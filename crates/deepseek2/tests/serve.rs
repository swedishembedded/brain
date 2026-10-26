// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in continuous
// batching of mixture-of-experts decoders then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The paged batched engine against the generic serving contract and against the
//! single-sequence decoder it replaces on the serving path.
//!
//! * The conformance suite (`model::serve::conformance`) on a tiny random-weight
//!   decoder, on the CPU backend and on the default one: batched equals alone,
//!   short requests are not held behind a long decode, embedded prompts batch and
//!   never share a prefix.
//! * Greedy token-for-token parity with `DeepseekV2::generate_greedy_kv` on the
//!   same weights, for a plain prompt and for a prompt carrying an image run,
//!   through the scheduler with the prompt read in several chunks and several
//!   sequences decoding together.

use deepseek2::config::DeepseekV2Config;
use deepseek2::model::{DeepseekV2, Sizes};
use deepseek2::serve::{pipelines, Engine, EngineConfig, Precision};
use gpu_core::Gpu;
use model::kv_tier::KvTier;
use model::serve::conformance::{self, Conformance};
use model::serve::{PromptEmbeds, Request, SampleParams, Scheduler};

const SEED: u64 = 7;

fn ec() -> EngineConfig {
    EngineConfig { max_seq: 64, pool_tokens: 4 * 64 * 2, max_batch: 4, chunk: 8, block_size: 4, kv: KvTier::F32, topk: 8, precision: Precision::Reference }
}

/// Weights whose outputs depend on the input. The init's std of 0.02 collapses a
/// 12-wide model to one constant token, which would make every equality below
/// vacuous; scaling every non-norm tensor spreads the logits.
fn weights() -> std::collections::HashMap<String, Vec<f32>> {
    let cfg = DeepseekV2Config::tiny();
    let mut w = deepseek2::init_weights(&cfg, SEED);
    for (name, v) in w.iter_mut() {
        if !name.ends_with("norm.weight") && !name.ends_with("ln1.weight") && !name.ends_with("ln2.weight") {
            v.iter_mut().for_each(|x| *x *= WEIGHT_SCALE);
        }
    }
    w
}

const WEIGHT_SCALE: f32 = 30.0;

fn engine(gpu: Gpu) -> Engine {
    Engine::new(gpu, DeepseekV2Config::tiny(), &weights(), ec()).expect("a tiny engine builds")
}

fn cpu_engine() -> Engine {
    engine(Gpu::new_cpu(&pipelines()))
}

fn default_engine() -> Engine {
    engine(Gpu::new(&pipelines()))
}

#[test]
fn the_engine_conforms_on_the_cpu_backend() {
    conformance::run_all(&Conformance { make: cpu_engine, prefix_reuse: false, prefix_prompt_tokens: 24, load_adapter: None });
}

#[test]
fn the_engine_conforms_on_the_default_backend() {
    conformance::run_all(&Conformance { make: default_engine, prefix_reuse: false, prefix_prompt_tokens: 24, load_adapter: None });
}

/// The single-sequence reference: the decoder built the way the OCR session builds it.
fn reference(gpu: Gpu, prompt: &[u32], img: Option<(u32, &[f32])>, n_new: u32) -> Vec<u32> {
    let mut m = DeepseekV2::new_sized(gpu, DeepseekV2Config::tiny(), Sizes { b: 1, t: 1, ctx: 64, chunk: 16, batched: false }, &weights(), false);
    if let Some((row0, rows)) = img {
        let d = DeepseekV2Config::tiny().d_model() as usize;
        m.enable_mm_splice(row0, (rows.len() / d) as u32);
        m.write_img_embeds(rows);
    }
    let all = m.generate_greedy_kv(prompt, n_new);
    all[prompt.len()..].to_vec()
}

fn prompt(seed: u32, len: usize) -> Vec<u32> {
    let vocab = DeepseekV2Config::tiny().vocab();
    (0..len as u32).map(|i| (seed * 5 + i * 3 + 1) % vocab).collect()
}

fn image_rows(n: usize) -> Vec<f32> {
    let d = DeepseekV2Config::tiny().d_model() as usize;
    (0..n * d).map(|i| ((i * 37 % 29) as f32 / 29.0 - 0.5) * 2.0).collect()
}

/// `engine_gpu` builds the engine's device (its own pipeline list), `reference_gpu` the single-sequence
/// decoder's (`DeepseekV2` addresses its kernels by its own `PIPELINES`).
fn parity(engine_gpu: fn() -> Gpu, reference_gpu: fn() -> Gpu) {
    let n_new = 12;
    // Plain prompts of several lengths (a prompt past the 8-row chunk is read in pieces), decoded together.
    let prompts: Vec<Vec<u32>> = [5usize, 9, 20, 3].iter().enumerate().map(|(i, &n)| prompt(i as u32 + 1, n)).collect();
    let mut sched = Scheduler::new(engine(engine_gpu()), 4);
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit(Request { prompt: p.clone(), max_new: n_new as usize, stop: Vec::new() })).collect();
    let got = sched.run();
    for (p, id) in prompts.iter().zip(&ids) {
        assert_eq!(got[id], reference(reference_gpu(), p, None, n_new), "a batched request must decode exactly what the single-sequence decoder does (prompt of {} tokens)", p.len());
    }

    // A prompt with an image run in the middle, straddling a chunk boundary (rows 5..15 of a 20-token prompt).
    let (row0, n_img) = (5u32, 10usize);
    let rows = image_rows(n_img);
    let p = prompt(9, 20);
    let mut sched = Scheduler::new(engine(engine_gpu()), 2);
    let embeds = PromptEmbeds::run(DeepseekV2Config::tiny().d_model() as usize, row0, rows.clone()).unwrap();
    let id = sched.submit_embedded(Request { prompt: p.clone(), max_new: n_new as usize, stop: Vec::new() }, embeds, SampleParams::greedy(), 0);
    let plain = sched.submit(Request { prompt: prompt(2, 6), max_new: n_new as usize, stop: Vec::new() });
    let got = sched.run();
    assert_eq!(got[&id], reference(reference_gpu(), &p, Some((row0, &rows)), n_new), "an image-run prompt must decode exactly what the splice path does");
    assert_eq!(got[&plain], reference(reference_gpu(), &prompt(2, 6), None, n_new), "a text request alongside it is unaffected");
}

#[test]
fn greedy_decode_matches_the_single_sequence_decoder_cpu() {
    parity(|| Gpu::new_cpu(&pipelines()), || Gpu::new_cpu(deepseek2::PIPELINES));
}

#[test]
fn greedy_decode_matches_the_single_sequence_decoder_default_backend() {
    parity(|| Gpu::new(&pipelines()), || Gpu::new(deepseek2::PIPELINES));
}
