// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Batch-invariant decode (`--exact-decode`) on the Qwen3 serving engine, through the scheduler that serves it.
//!
//! Swedish Embedded AB implements reproducible LLM serving for evaluation and on-policy training, where one request
//! must decode the same tokens and log-probabilities whatever else shares its batch. If your team needs expertise in
//! batch-invariant inference kernels, you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The contract: one request, sampled at temperature 1 with a fixed seed (so any difference is arithmetic, not
//! sampling), returns bit-identical tokens and per-token log-probabilities decoded alone on a cold engine, then beside
//! fifteen other requests, then as one of eight identical requests running at once, all on the same engine, so the
//! later runs reuse the prompt blocks the earlier ones cached. The synthetic shape is chosen so the default serving
//! does NOT meet it: 32 KV heads put the int8 attention's split count at two beside fifteen neighbours against 32
//! alone, a prompt of nine 128-key chunks spreads over the splits differently in the two cases, sixteen rows reach
//! the int8 GEMV/tile choice of a wider row bucket, and the probe's last cached block ends inside a prefill chunk.

use model::serve::{PagedDecoder, Request, SampleParams, Scheduler};
use qwen3::serve::{AdapterBankConfig, Engine, LoraSite};
use qwen3::QwenConfig;

use gpu_core::select::Dtype;

const BLOCK: u32 = 16;
/// Positions one sequence may hold: 4096, so the attention may split a long sequence 32 ways.
const BLOCKS_PER_SEQ: u32 = 256;
const MAX_BATCH: u32 = 16;
/// Prompt rows per prefill chunk; the probe's prompt is several chunks long.
const MAX_PREFILL: u32 = 256;
const NEW_TOKENS: usize = 24;
/// The probe's prompt: nine 128-key chunks, its last full block inside the fifth prefill chunk.
const PROBE_LEN: usize = 1100;

fn cfg() -> QwenConfig {
    QwenConfig {
        vocab: 96,
        block_size: BLOCK,
        n_layers: 2,
        d_model: 64,
        n_heads: 32,
        n_kv_heads: 32,
        head_dim: 16,
        d_ff: 96,
        rope_theta: 1.0e6,
        rms_eps: 1e-6,
        max_position_embeddings: 4096,
        tie_embeddings: true,
        qk_norm: true,
        attn_bias: false,
        lora: None,
        rope_scaling: None,
    }
}

fn prompt(seed: u64, len: usize, vocab: u32) -> Vec<u32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..len).map(|_| r.next_u32() % vocab).collect()
}

/// One served request's output: its tokens and the log-probability each was drawn with.
type Decoded = (Vec<u32>, Vec<f32>);

/// Serve `prompts` submitted together; each request samples at temperature 1 with `seed`. Returns the engine, each
/// request's output and the prompt tokens served from cached blocks while they ran.
fn serve(engine: Engine, prompts: &[Vec<u32>], seed: u64) -> (Engine, Vec<Decoded>, u64) {
    let hits_before = engine.prefix_stats().0;
    let mut sched = Scheduler::new(engine, MAX_BATCH as usize);
    let sample = SampleParams { temp: 1.0, ..SampleParams::greedy() };
    let ids: Vec<u64> = prompts.iter().map(|p| sched.submit_sampled(Request { prompt: p.clone(), max_new: NEW_TOKENS, stop: vec![] }, sample, seed)).collect();
    let mut done: std::collections::HashMap<u64, Decoded> = std::collections::HashMap::new();
    while done.len() < ids.len() {
        let report = sched.step_report();
        assert!(report.rejected.is_empty(), "a request was refused: {:?}", report.rejected);
        let lps: std::collections::HashMap<u64, Vec<f32>> = report.completed_logprobs.into_iter().collect();
        for (id, tokens) in report.completed {
            done.insert(id, (tokens, lps[&id].clone()));
        }
    }
    let decoded = ids.iter().map(|id| done.remove(id).expect("every request completes")).collect();
    let engine = sched.into_decoder();
    let reused = engine.prefix_stats().0 - hits_before;
    (engine, decoded, reused)
}

/// The first token at which `got` leaves `want`, and the largest log-probability difference before it.
fn divergence(want: &Decoded, got: &Decoded) -> Option<(usize, f32)> {
    if want == got {
        return None;
    }
    let at = want.0.iter().zip(&got.0).position(|(a, b)| a != b).unwrap_or(want.0.len().min(got.0.len()));
    let delta = want.1.iter().zip(&got.1).take(at).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    Some((at, delta))
}

/// The contract on `engine` (cold, in the exact regime), with prompts drawn from `vocab` token ids.
fn assert_batch_invariant(engine: Engine, vocab: u32, what: &str) {
    let probe = prompt(11, PROBE_LEN, vocab);
    let others: Vec<Vec<u32>> = (0..15u64).map(|i| prompt(100 + i, 40 + 150 * i as usize, vocab)).collect();
    let (engine, alone, _) = serve(engine, std::slice::from_ref(&probe), 7);
    let alone = &alone[0];
    assert_eq!(alone.0.len(), NEW_TOKENS, "{what}: the probe decoded its tokens");
    assert!(alone.1.iter().any(|&lp| lp < 0.0), "{what}: a sampled request carries real log-probabilities");

    let mut beside = vec![probe.clone()];
    beside.extend(others);
    let (engine, together, reused) = serve(engine, &beside, 7);
    assert!(reused > 0, "{what}: the probe beside the others reused its own cached prompt blocks");
    assert_eq!(divergence(alone, &together[0]), None, "{what}: the probe beside fifteen others left its solo output (first token, largest log-probability delta)");

    let (_, same, _) = serve(engine, &vec![probe.clone(); 8], 7);
    for (i, r) in same.iter().enumerate() {
        assert_eq!(divergence(alone, r), None, "{what}: identical request {i} of 8 left the solo output");
    }
}

/// A synthetic engine at `tier`, its weights from a fixed seed.
fn engine(tier: Dtype, kv_int8: bool, exact: bool) -> Engine {
    let cfg = cfg();
    let weights = qwen3::init_weights(&cfg, 3);
    let gpu = gpu_core::testgpu::dev(&[]);
    let mut e = Engine::from_map_tier_on(&gpu, cfg, &weights, BLOCK, 2048, MAX_BATCH, BLOCKS_PER_SEQ, MAX_PREFILL, kv_int8, tier);
    e.set_exact_decode(exact).expect("the engine takes the exact-decode regime");
    e
}

#[test]
fn exact_decode_keeps_a_request_on_its_solo_output_at_every_weight_tier() {
    for tier in [Dtype::F32, Dtype::BF16, Dtype::I8] {
        assert_batch_invariant(engine(tier, true, true), cfg().vocab, &format!("{tier:?} weights, int8 KV"));
    }
}

#[test]
fn exact_decode_keeps_a_request_on_its_solo_output_with_f32_kv() {
    assert_batch_invariant(engine(Dtype::F32, false, true), cfg().vocab, "f32 weights, f32 KV");
}

/// The same contract at a dense Qwen3 head shape (width 128, a group of 4), where decode runs the fused GQA decode
/// attention: under the regime every row splits its keys as a single sequence's are, at both KV tiers.
#[test]
fn exact_decode_keeps_a_request_on_its_solo_output_through_the_fused_decode() {
    for kv_int8 in [true, false] {
        let cfg = QwenConfig { n_heads: 8, n_kv_heads: 2, head_dim: 128, ..cfg() };
        let weights = qwen3::init_weights(&cfg, 3);
        let gpu = gpu_core::testgpu::dev(&[]);
        let mut e = Engine::from_map_tier_on(&gpu, cfg.clone(), &weights, BLOCK, 2048, MAX_BATCH, BLOCKS_PER_SEQ, MAX_PREFILL, kv_int8, Dtype::F32);
        assert_eq!(e.fused_attention().decode, gpu.caps().workgroup_reductions, "kv_int8={kv_int8}: decode runs the fused GQA decode wherever the device runs barrier kernels");
        e.set_exact_decode(true).expect("the engine takes the exact-decode regime");
        assert_batch_invariant(e, cfg.vocab, &format!("f32 weights, {} KV, head width 128", if kv_int8 { "int8" } else { "f32" }));
    }
}

/// A regime the engine cannot make batch-invariant is refused by name, never served approximately.
#[test]
fn exact_decode_is_refused_where_a_row_cannot_be_kept_independent() {
    let mut e = engine(Dtype::F32, true, false);
    e.enable_adapters(AdapterBankConfig::new(2, 4, &LoraSite::ALL)).expect("an adapter bank");
    let err = e.set_exact_decode(true).expect_err("a per-row adapter bank and exact decode");
    assert!(err.contains("adapter"), "{err}");
    assert_eq!(PagedDecoder::spec_max_draft(&engine(Dtype::F32, true, true)), 0, "speculation is off in the exact regime");
}

/// The contract on the real Qwen3-0.6B at each weight tier it serves (fp32 as stored, int8, bf16), int8 KV, the
/// serving default's prefill chunk. Skipped when the model store does not hold the checkpoint.
#[test]
fn the_real_qwen3_0_6b_keeps_a_request_on_its_solo_output() {
    let Some(dir) = brain_testutil::model_dir("Qwen/Qwen3-0.6B").filter(|d| std::path::Path::new(d).join("config.json").is_file()) else {
        brain_testutil::skip_unavailable("Qwen/Qwen3-0.6B is not in the model store");
        return;
    };
    let gpu = gpu_core::testgpu::dev(&[]);
    for tier in [Dtype::F32, Dtype::I8, Dtype::BF16] {
        let (cfg, src) = qwen3::open_checkpoint(&dir).expect("the checkpoint opens");
        // Token ids from the ordinary vocabulary, below the special tokens.
        let vocab = cfg.vocab.min(150_000);
        let mut e = Engine::from_source_tier_on(&gpu, cfg, &*src, BLOCK, 4096, MAX_BATCH, BLOCKS_PER_SEQ, 512, true, tier);
        e.set_exact_decode(true).expect("the engine takes the exact-decode regime");
        assert_batch_invariant(e, vocab, &format!("Qwen3-0.6B, {tier:?} weights, int8 KV"));
    }
}
