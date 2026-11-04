// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The released Qwen3.6-35B-A3B Q8_0 GGUF through the serving engine's decode, built
//! and replayed, with and without the fused native kernels.
//!
//! Swedish Embedded AB implements fast decode for large sparse-MoE models for its
//! clients. If your team needs expertise in taking host launches and kernel
//! boundaries out of a token loop without changing what the model says then you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! Gated on `BRAIN_QWEN35MOE_GGUF` (skipped without it): the synthetic configs the
//! rest of this crate tests on are far below the shapes the fused kernels serve
//! (128-wide heads), so the real model is the only place that comparison can run.
//! The prompt is real text through the GGUF's own tokenizer, so the activations and
//! the routing are a language model's and not noise's. The model is cut to its
//! first eight layers (two attention layers among them) so it loads in seconds and
//! the comparison is not decided by a router near-tie deep in the stack.
//!
//! * A recorded step is the step built from its dispatches, BIT for bit: they run
//!   the same kernels over the same buffers.
//! * The fused kernels are an optimisation of a chain of kernels, not a different
//!   model: the top candidates and their logits agree to a few parts per million.
//!   They are not asserted bit-identical here because on the real weights they are
//!   not (a last-place difference in a gate's exponential), and through forty
//!   layers that difference is enough to flip a router near-tie and then the
//!   logits by a percent - `reference_forward_real` is what holds the model to its
//!   host reference.

use checkpoint::gguf::MmapGguf;
use data::tokenizer::Tokenizer;
use model::paged::BlockTable;
use qwen35moe::gguf_load;
use qwen35moe::serve::{Engine, EngineOptions};

const PROMPT: &str = "The Gated DeltaNet layers of a hybrid model keep a small recurrent state instead of a key-value cache, so";
const STEPS: usize = 6;
const LAYERS: u32 = 8;

/// The top-4 `(id, logit bits)` of each step of a greedy decode from `prompt`.
fn decode(e: &mut Engine, prompt: &[u32]) -> Vec<Vec<(u32, u32)>> {
    let mut table = BlockTable::new();
    e.prefill(&mut table, prompt);
    let mut token = *prompt.last().expect("a prompt");
    let mut steps = Vec::new();
    for _ in 0..STEPS {
        let top = e.forward_batched_topk(&mut [&mut table], &[token], 4);
        let row: Vec<(u32, u32)> = top[0].iter().map(|&(i, v)| (i, v.to_bits())).collect();
        token = row[0].0;
        steps.push(row);
    }
    e.release_table(&mut table);
    steps
}

#[test]
fn replaying_is_building_to_the_bit_and_fusing_changes_only_the_last_places() {
    let Ok(path) = std::env::var(gguf_load::GGUF_ENV) else {
        return brain_testutil::skip_unavailable("BRAIN_QWEN35MOE_GGUF is not set");
    };
    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let tok = gguf_load::tokenizer(&mg).expect("the GGUF's tokenizer");
    let prompt = tok.encode(PROMPT);
    let mut cfg = gguf_load::resident_config(&mg, 256).expect("config");
    cfg.n_layers = LAYERS;
    let src = gguf_load::source(&mg, &cfg).expect("source");
    let mut e = Engine::from_source(cfg, &src, EngineOptions::new(256, 2).with_tier(gguf_load::tier_from_env()).with_kv_tier(gguf_load::kv_tier_from_env().expect("kv tier")));
    if !e.gpu().caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("no packed int8 dot on this device");
    }

    let replayed = decode(&mut e, &prompt);
    e.set_decode_tapes(false);
    let built = decode(&mut e, &prompt);
    assert_eq!(built, replayed, "a recorded step differs from the step built from its dispatches");

    e.set_decode_tapes(true);
    e.set_decode_fusion(false);
    let unfused = decode(&mut e, &prompt);
    for (step, (f, u)) in replayed.iter().zip(&unfused).enumerate() {
        assert_eq!(f.iter().map(|c| c.0).collect::<Vec<_>>(), u.iter().map(|c| c.0).collect::<Vec<_>>(), "step {step}: the fused kernels change the candidates");
        for (a, b) in f.iter().zip(u) {
            let (a, b) = (f32::from_bits(a.1), f32::from_bits(b.1));
            assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0), "step {step}: fused logit {a} against {b}");
        }
    }
}
