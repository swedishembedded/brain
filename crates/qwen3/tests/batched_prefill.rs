// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen::prefill` - the KV-cache prefill every SDK generation, chat turn and
//! vision-language caption starts with - runs the prompt as batched chunks
//! through the model's own batched layer forward, not one decode step per
//! token. The per-token decode (`Qwen::step`) is the reference: a batched
//! prefill must leave the same cache and the same final hidden state behind,
//! so the next-token logits agree to fp32 rounding and the greedy
//! continuation is the same token for token. Runs on whichever backend
//! `BRAIN_DEVICE` selects (`cpu`, `vulkan`).
//!
//! Swedish Embedded AB implements fast, verifiable on-device LLM inference
//! for its clients. If your team needs expertise in edge-AI inference
//! engines, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::HashMap;

use qwen3::model::PrefillInput;
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig, Shard};

fn skip() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return true;
    }
    false
}

/// A context long enough that a prompt spans several prefill chunks on every
/// backend.
const CTX: u32 = 640;
/// Greedy tokens decoded after the prompt.
const CONTINUATION: usize = 6;

fn prompt(len: usize, vocab: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 7 + 3) % vocab).collect()
}

fn decode_model(cfg: &QwenConfig, tensors: &HashMap<String, Vec<f32>>, dt: Dtype) -> Qwen {
    Qwen::new_shard_dt_decode(cfg.clone(), CTX, tensors, Shard::whole(cfg.n_layers as usize), dt)
}

fn logits(head: &[f32], hidden: &[f32], cfg: &QwenConfig) -> Vec<f32> {
    model::hostmath::matvec_par(head, hidden, cfg.vocab as usize, cfg.d_model as usize)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0 as u32
}

/// Next-token logits after the prompt, then the greedy continuation, the
/// continuation decoded one `step` at a time after whatever filled the cache.
fn continue_greedily(m: &Qwen, head: &[f32], after_prompt: Vec<f32>) -> (Vec<f32>, Vec<u32>) {
    let first = logits(head, &after_prompt, &m.cfg);
    let mut tokens = vec![argmax(&first)];
    while tokens.len() < CONTINUATION {
        let hidden = m.step(*tokens.last().unwrap());
        tokens.push(argmax(&logits(head, &hidden, &m.cfg)));
    }
    (first, tokens)
}

/// The reference: every prompt position decoded by its own `step`.
fn per_token(m: &Qwen, head: &[f32], inputs: &[PrefillInput<'_>]) -> (Vec<f32>, Vec<u32>) {
    m.reset_cache();
    let mut hidden = Vec::new();
    for input in inputs {
        hidden = match input {
            PrefillInput::Token(t) => m.step(*t),
            PrefillInput::Embed(row) => m.step_embed(row),
        };
    }
    continue_greedily(m, head, hidden)
}

/// The prompt through `Qwen::prefill`, split across `calls` calls the way a
/// cancellable caller splits it.
fn batched(m: &Qwen, head: &[f32], inputs: &[PrefillInput<'_>], calls: usize) -> (Vec<f32>, Vec<u32>) {
    m.reset_cache();
    let per_call = inputs.len().div_ceil(calls);
    let mut hidden = Vec::new();
    for part in inputs.chunks(per_call) {
        hidden = m.prefill(part);
    }
    assert_eq!(m.cache_pos() as usize, inputs.len(), "prefill must advance the cache by exactly the prompt");
    continue_greedily(m, head, hidden)
}

/// `rel` relative to the reference logits' own scale: the batched forward
/// sums the same products in a different order than the per-token GEMV.
fn assert_close(got: &[f32], want: &[f32], rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    let diff = got.iter().zip(want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(diff <= rel * scale, "{what}: max |diff| {diff:e} exceeds {rel:e} x logit scale {scale:e}");
}

fn assert_matches_per_token(m: &Qwen, head: &[f32], inputs: &[PrefillInput<'_>], rel: f32, what: &str) {
    let (want_logits, want_tokens) = per_token(m, head, inputs);
    for calls in [1, 3] {
        let (got_logits, got_tokens) = batched(m, head, inputs, calls);
        assert_close(&got_logits, &want_logits, rel, &format!("{what}, {} tokens in {calls} call(s)", inputs.len()));
        assert_eq!(got_tokens, want_tokens, "{what}, {} tokens in {calls} call(s): greedy continuation", inputs.len());
    }
}

fn tokens(ids: &[u32]) -> Vec<PrefillInput<'_>> {
    ids.iter().map(|&t| PrefillInput::Token(t)).collect()
}

/// The defect itself: a prefill used to cost one device submission per prompt
/// token. Batched, the submission count is set by the chunk count, and a
/// prompt four times longer that still fits one chunk costs the same.
#[test]
fn prefill_submissions_do_not_grow_with_the_prompt() {
    if skip() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let m = decode_model(&cfg, &qwen3::init_weights(&cfg, 7), Dtype::F32);
    if m.gpu().stats().is_none() {
        brain_testutil::skip_unavailable("this backend does not count device submits");
        return;
    }
    let submits = |len: usize| {
        m.reset_cache();
        m.gpu().flush();
        let before = m.gpu().stats().unwrap().submits;
        m.prefill(&tokens(&prompt(len, cfg.vocab)));
        m.gpu().stats().unwrap().submits - before
    };
    let short = submits(4);
    let long = submits(16);
    assert_eq!(long, short, "a 16-token prefill took {long} submissions against {short} for 4 tokens: it is still decoding token by token");
}

/// Token prompts from one token up to several chunks long, each prefilled in
/// one call and in three, at the fp32 and the int8 weight tier.
#[test]
fn batched_prefill_matches_per_token_decode() {
    if skip() {
        return;
    }
    for (cfg, dt, rel) in [(QwenConfig::tiny(), Dtype::F32, 1e-4f32), (QwenConfig::tiny_i8(), Dtype::I8, 1e-3)] {
        let m = decode_model(&cfg, &qwen3::init_weights(&cfg, 11), dt);
        let head = m.read_weight(cfg.head_weight());
        for len in [1usize, 2, 37, 300, 600] {
            assert_matches_per_token(&m, &head, &tokens(&prompt(len, cfg.vocab)), rel, &format!("{dt:?}"));
        }
    }
}

/// Raw embedding rows (a vision-language model's image features) spliced
/// between token runs prefill as their own batched runs.
#[test]
fn batched_prefill_of_mixed_tokens_and_embeddings_matches_per_token_decode() {
    if skip() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let d = cfg.d_model as usize;
    let m = decode_model(&cfg, &qwen3::init_weights(&cfg, 5), Dtype::F32);
    let head = m.read_weight(cfg.head_weight());
    let rows: Vec<f32> = (0..40 * d).map(|i| 0.5 * ((i as f32) * 0.37).sin()).collect();
    let ids = prompt(60, cfg.vocab);
    let mut inputs = tokens(&ids[..25]);
    inputs.extend(rows.chunks(d).map(PrefillInput::Embed));
    inputs.extend(tokens(&ids[25..]));
    assert_matches_per_token(&m, &head, &inputs, 1e-4, "mixed");
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-batched-prefill-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A small trained adapter over every linear (so its delta is non-trivial),
/// with the frozen base tensors it was trained on.
fn trained_adapter(tag: &str) -> (HashMap<String, Vec<f32>>, String) {
    let targets = ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect();
    let lora_cfg = QwenConfig { lora: Some(LoraCfg { rank: 3, alpha: 6.0, targets }), ..QwenConfig::tiny() };
    let trained = Qwen::new(lora_cfg.clone(), 1, 12, &qwen3::init_weights(&lora_cfg, 21));
    trained.set_batch(&prompt(12, 23), &prompt(13, 23)[1..]);
    for step in 1..=8 {
        trained.zero_grads();
        trained.forward();
        trained.backward();
        trained.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
        trained.poll_wait();
    }
    let path = tmp(tag).join("adapter.safetensors").to_string_lossy().into_owned();
    qwen3::lora::save_adapter(&path, &trained, "test/adapter", "test/base", None).unwrap();
    let base = trained
        .ps
        .params
        .iter()
        .map(|(name, _)| name)
        .filter(|name| !(name.ends_with(".lora_a") || name.ends_with(".lora_b")))
        .map(|name| (name.clone(), trained.read_weight(name)))
        .collect();
    (base, path)
}

/// An adapter applied at runtime (`attach_adapter`) and one folded into the
/// base weights both prefill in batches to what their per-token decode gives,
/// and the two agree with each other.
#[test]
fn batched_prefill_matches_per_token_decode_with_an_attached_and_a_folded_adapter() {
    if skip() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let (base, adapter) = trained_adapter("adapter");
    let inputs_ids = prompt(300, cfg.vocab);
    let inputs = tokens(&inputs_ids);

    let mut folded_tensors = base.clone();
    qwen3::lora::fold_adapter_into(&mut folded_tensors, &adapter).unwrap();
    let folded = decode_model(&cfg, &folded_tensors, Dtype::F32);
    let head = folded.read_weight(cfg.head_weight());
    assert_matches_per_token(&folded, &head, &inputs, 1e-4, "folded adapter");

    let mut attached = decode_model(&cfg, &base, Dtype::F32);
    let (base_logits, _) = batched(&attached, &head, &inputs, 1);
    attached.attach_adapter(&adapter).unwrap();
    assert_matches_per_token(&attached, &head, &inputs, 1e-4, "attached adapter");

    let (want_logits, want_tokens) = batched(&folded, &head, &inputs, 1);
    let (got_logits, got_tokens) = batched(&attached, &head, &inputs, 1);
    let moved = base_logits.iter().zip(&want_logits).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(moved > 1e-2, "the adapter barely moves the logits ({moved:e}); the adapter parity would be vacuous");
    assert_close(&got_logits, &want_logits, 1e-4, "attached vs folded, batched");
    assert_eq!(got_tokens, want_tokens, "attached vs folded, batched: greedy continuation");
}

/// The SDK's cancellable generation prefills in caller-sized chunks with the
/// cancel token polled between them; whatever the chunk size, and after a
/// generation abandoned part-way through its prompt, greedy decoding gives
/// the per-token reference's tokens.
#[test]
fn cancellable_chunked_generation_matches_per_token_decode() {
    if skip() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let m = decode_model(&cfg, &qwen3::init_weights(&cfg, 9), Dtype::F32);
    let head = m.read_weight(cfg.head_weight());
    let ids = prompt(300, cfg.vocab);
    let (_, want) = per_token(&m, &head, &tokens(&ids));
    let generate = |chunk: usize| {
        let mut rng = data::rng::Rng::new(1);
        let cancel = capability::CancelToken::armed();
        qwen3::sample::generate_kv_stream_cancellable(&m, &ids, CONTINUATION, 0.0, 0, 1.0, &[], &mut rng, &head, &cancel, chunk, &mut |_, _| true)
    };
    for chunk in [1usize, 7, 128, 299, 300] {
        assert_eq!(generate(chunk), want, "prefill chunk {chunk}");
    }
    // A cancel that fires between chunks leaves a part-filled cache behind;
    // the next generation must not see it.
    m.reset_cache();
    m.prefill(&tokens(&ids[..100]));
    assert_eq!(generate(64), want, "after an abandoned prefill");
    let cancelled = capability::CancelToken::armed();
    cancelled.cancel();
    let mut rng = data::rng::Rng::new(1);
    let none = qwen3::sample::generate_kv_stream_cancellable(&m, &ids, CONTINUATION, 0.0, 0, 1.0, &[], &mut rng, &head, &cancelled, 64, &mut |_, _| true);
    assert!(none.is_empty(), "a cancelled generation produces no tokens");
    assert_eq!(generate(64), want, "after a cancelled generation");
}
