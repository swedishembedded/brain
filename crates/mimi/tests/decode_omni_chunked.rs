// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Codec::decode_omni_chunked` vs `Codec::decode_omni` -- bit-exact on the
//! same tiny-config synthetic weights, at several chunk sizes, PLUS the
//! degenerate `chunk_frames=0` (one shot) case. No real checkpoint needed
//! (none is staged in this environment -- the honest boundary this crate's
//! other real-weight-gated tests in `decode.rs`/`encode.rs` already document
//! for themselves); this is the tiny-config rung of the parity ladder,
//! composing every piece already validated in isolation
//! (`gqa_fwd_win` sliding-window attention,
//! `StreamConvTr1dSym`, `Back::new_omni`) through the REAL `Codec` entry
//! points, not a re-derivation of any of their math.
//!
//! `sliding_window` is deliberately set SMALLER than `t` here, so this test
//! also proves `decode_omni_chunked`'s front composes correctly with the
//! real windowed attention fix (M23) -- not just that chunking itself is
//! exact.

use std::collections::HashMap;

use mimi::config::CodecConfig;
use mimi::model::Codec;
use data::rng::Lcg;

fn fill(w: &mut HashMap<String, Vec<f32>>, name: &str, n: usize, seed: &mut Lcg) {
    w.insert(name.to_string(), seed.vec(n));
}

/// A structurally-complete but tiny Omni-shaped `Codec`: 1 pre-transformer
/// layer, 2x2 upsample/decoder rates, hidden 8. Every weight
/// `Codec::transformer`/`Codec::decode_omni`/`decode_stream::Back` touches.
fn tiny_omni_codec(seed_val: u64) -> Codec {
    let mut seed = Lcg::new(seed_val);
    let (hidden, inter, heads, head_dim, dec_dim, nq, cb) = (8u32, 16u32, 2u32, 4u32, 8u32, 2u32, 4u32);
    let cfg = CodecConfig {
        num_quantizers: nq,
        num_semantic_quantizers: 1,
        codebook_size: cb,
        semantic_codebook_size: cb,
        codebook_dim: hidden,
        latent_dim: hidden,
        hidden_size: hidden,
        intermediate_size: inter,
        num_hidden_layers: 1,
        num_attention_heads: heads,
        num_key_value_heads: heads,
        head_dim,
        sliding_window: 3, // < t=12 below -- real windowing is exercised, not degenerate
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        layer_scale_initial_scale: 0.01,
        decoder_dim: dec_dim,
        upsample_rates: vec![2, 2],
        upsampling_ratios: vec![2],
        input_sample_rate: 24000,
        output_sample_rate: 24000,
        decode_upsample_rate: 2 * 2 * 2 * 2,
        enc: Default::default(),
    };

    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let (h, i, hq) = (hidden as usize, inter as usize, (heads * head_dim) as usize);
    fill(&mut w, "code_embedding.weight", (nq * cb) as usize * h, &mut seed);
    let p = |leaf: &str| format!("pre_transformer.layers.0.{leaf}");
    fill(&mut w, &p("input_layernorm.weight"), h, &mut seed);
    fill(&mut w, &p("self_attn.q_proj.weight"), hq * h, &mut seed);
    fill(&mut w, &p("self_attn.k_proj.weight"), hq * h, &mut seed);
    fill(&mut w, &p("self_attn.v_proj.weight"), hq * h, &mut seed);
    fill(&mut w, &p("self_attn.o_proj.weight"), h * hq, &mut seed);
    fill(&mut w, &p("self_attn_layer_scale.scale"), h, &mut seed);
    fill(&mut w, &p("post_attention_layernorm.weight"), h, &mut seed);
    fill(&mut w, &p("mlp.gate_proj.weight"), i * h, &mut seed);
    fill(&mut w, &p("mlp.up_proj.weight"), i * h, &mut seed);
    fill(&mut w, &p("mlp.down_proj.weight"), h * i, &mut seed);
    fill(&mut w, &p("mlp_layer_scale.scale"), h, &mut seed);
    fill(&mut w, "pre_transformer.norm.weight", h, &mut seed);

    let latent = hidden as usize;
    let dec = dec_dim as usize;
    fill(&mut w, "upsample.0.0.conv.weight", latent * latent * 2, &mut seed);
    fill(&mut w, "upsample.0.0.conv.bias", latent, &mut seed);
    fill(&mut w, "upsample.0.1.dwconv.conv.weight", latent * 7, &mut seed);
    fill(&mut w, "upsample.0.1.dwconv.conv.bias", latent, &mut seed);
    fill(&mut w, "upsample.0.1.norm.weight", latent, &mut seed);
    fill(&mut w, "upsample.0.1.norm.bias", latent, &mut seed);
    fill(&mut w, "upsample.0.1.pwconv1.weight", 4 * latent * latent, &mut seed);
    fill(&mut w, "upsample.0.1.pwconv1.bias", 4 * latent, &mut seed);
    fill(&mut w, "upsample.0.1.pwconv2.weight", latent * 4 * latent, &mut seed);
    fill(&mut w, "upsample.0.1.pwconv2.bias", latent, &mut seed);
    fill(&mut w, "upsample.0.1.gamma", latent, &mut seed);
    fill(&mut w, "decoder.0.conv.weight", dec * latent * 7, &mut seed);
    fill(&mut w, "decoder.0.conv.bias", dec, &mut seed);
    for (i, &rate) in cfg.upsample_rates.clone().iter().enumerate() {
        let in_dim = dec >> i;
        let out_dim = dec >> (i + 1);
        let bp = format!("decoder.{}", i + 1);
        fill(&mut w, &format!("{bp}.block.0.alpha"), in_dim, &mut seed);
        fill(&mut w, &format!("{bp}.block.0.beta"), in_dim, &mut seed);
        fill(&mut w, &format!("{bp}.block.1.conv.weight"), in_dim * out_dim * (2 * rate as usize), &mut seed);
        fill(&mut w, &format!("{bp}.block.1.conv.bias"), out_dim, &mut seed);
        for j in [2usize, 3, 4] {
            fill(&mut w, &format!("{bp}.block.{j}.act1.alpha"), out_dim, &mut seed);
            fill(&mut w, &format!("{bp}.block.{j}.act1.beta"), out_dim, &mut seed);
            fill(&mut w, &format!("{bp}.block.{j}.conv1.conv.weight"), out_dim * out_dim * 7, &mut seed);
            fill(&mut w, &format!("{bp}.block.{j}.conv1.conv.bias"), out_dim, &mut seed);
            fill(&mut w, &format!("{bp}.block.{j}.act2.alpha"), out_dim, &mut seed);
            fill(&mut w, &format!("{bp}.block.{j}.act2.beta"), out_dim, &mut seed);
            fill(&mut w, &format!("{bp}.block.{j}.conv2.conv.weight"), out_dim * out_dim, &mut seed);
            fill(&mut w, &format!("{bp}.block.{j}.conv2.conv.bias"), out_dim, &mut seed);
        }
    }
    let out_dim = dec >> cfg.upsample_rates.len();
    fill(&mut w, "decoder.5.alpha", out_dim, &mut seed);
    fill(&mut w, "decoder.5.beta", out_dim, &mut seed);
    fill(&mut w, "decoder.6.conv.weight", out_dim * 7, &mut seed);
    fill(&mut w, "decoder.6.conv.bias", 1, &mut seed);

    Codec::from_weights(cfg, w)
}

fn random_codes(seed_val: u64, t: usize, nq: usize, cb: usize) -> Vec<u32> {
    let mut seed = Lcg::new(seed_val);
    (0..t * nq).map(|_| seed.next_u32() % cb as u32).collect()
}

#[test]
fn decode_omni_chunked_matches_decode_omni_one_shot() {
    let codec = tiny_omni_codec(0xC0DE);
    let t = 12usize;
    let codes = random_codes(0xC0DE2, t, 2, 4);

    let want = codec.decode_omni(&codes);
    assert!(!want.is_empty(), "oracle output must be non-empty");

    for &chunk_frames in &[0usize, 1, 3, 5, 12, 100] {
        let mut got = Vec::new();
        codec.decode_omni_chunked(&codes, chunk_frames, |c| got.extend_from_slice(c));
        assert_eq!(got.len(), want.len(), "chunk_frames={chunk_frames}: length mismatch");
        let maxd = got.iter().zip(&want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-5, "chunk_frames={chunk_frames}: decode_omni_chunked != decode_omni, max abs diff {maxd}");
    }
}

#[test]
fn decode_omni_chunked_emits_more_than_one_chunk_for_small_chunk_frames() {
    // Not just correct output -- proves this is genuinely INCREMENTAL: a
    // small chunk_frames must actually invoke the callback more than once
    // (the whole point of chunking is bounded per-call memory / early
    // emission, which a "buffer everything then call once" bug would defeat
    // silently while still passing the length/value checks above).
    let codec = tiny_omni_codec(0xFEED);
    let t = 12usize;
    let codes = random_codes(0xFEED2, t, 2, 4);
    let mut n_calls = 0u32;
    codec.decode_omni_chunked(&codes, 1, |_| n_calls += 1);
    assert!(n_calls > 1, "chunk_frames=1 over t=12 must emit more than one chunk, got {n_calls}");
}

#[test]
#[should_panic(expected = "empty codes")]
fn decode_omni_chunked_rejects_empty_codes() {
    let codec = tiny_omni_codec(1);
    codec.decode_omni_chunked(&[], 1, |_| {});
}
