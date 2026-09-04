// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Long-form decode: the codec's `pre_transformer` is a SLIDING-WINDOW causal
//! transformer (`CodecConfig::sliding_window`, 72 frames == 5.76 s of audio on
//! the released 12.5 Hz checkpoints), and the two decode implementations in this
//! crate must agree on that mask for sequences LONGER than one window - not just
//! for the short clips where a sliding-window mask and a plain causal mask are
//! the same thing.
//!
//! Both implementations are exercised on the SAME synthetic weights, so this is
//! an independent-witness test (device WGSL dispatch vs pure-host arithmetic),
//! not a self-check: `mimi::Codec::decode` runs the real `gqa_scores_win`
//! kernel, `mimi::decode_stream::StreamingCodecDecoder` runs the host
//! attention loop the streaming server (`qwen3tts::serve`) actually calls.
//! Synthetic weights rather than the released checkpoint on purpose - the
//! window is a structural property, so it is provable at a `sliding_window` of
//! 4 over 40 frames in milliseconds instead of needing 72+ frames of a 651 MB
//! external artifact this suite is not allowed to assume is present.

use std::collections::HashMap;

use data::rng::Lcg;
use mimi::decode_stream::StreamingCodecDecoder;
use mimi::{Codec, CodecConfig};

/// A structurally complete but tiny codec: every stage of the real graph is
/// present (RVQ dequant, `pre_conv`, a 2-layer sliding-window transformer,
/// one ConvNeXt upsample stage, a 2-rate SEANet decoder), at widths small
/// enough to decode instantly. `sliding_window` is 4 so a 40-frame clip is 10
/// windows deep.
const WINDOW: u32 = 4;

fn tiny_cfg() -> CodecConfig {
    CodecConfig {
        num_quantizers: 4,
        num_semantic_quantizers: 1,
        codebook_size: 16,
        semantic_codebook_size: 16,
        codebook_dim: 8, // per-codebook dim = 4, quantizer output = 8
        latent_dim: 6,
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 2,
        sliding_window: WINDOW,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        layer_scale_initial_scale: 0.01,
        decoder_dim: 8,
        upsample_rates: vec![2, 2],
        upsampling_ratios: vec![2],
        input_sample_rate: 24000,
        output_sample_rate: 24000,
        decode_upsample_rate: 8, // 2 * 2 * 2
        ..Default::default()
    }
}

/// Weights for [`tiny_cfg`]'s full decode graph. Scaled down (not unit-range)
/// so the eight-stage stack does not saturate against `decode`'s final
/// `clamp(-1, 1)` - a saturated waveform would compare equal no matter what
/// the attention mask did, which is exactly the vacuous pass this test must
/// not be able to produce (`assert_not_saturated` enforces it).
fn tiny_weights(cfg: &CodecConfig) -> HashMap<String, Vec<f32>> {
    let mut r = Lcg::new(20260904);
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut put = |w: &mut HashMap<String, Vec<f32>>, name: &str, n: usize, a: f32| {
        w.insert(name.to_string(), r.vec_scaled(n, a));
    };

    let nq = cfg.num_quantizers as usize;
    let dim = (cfg.codebook_dim / 2) as usize; // 4
    let lat = cfg.codebook_dim as usize; // 8
    let hidden = cfg.hidden_size as usize; // 4
    let latent = cfg.latent_dim as usize; // 6
    let ff = cfg.intermediate_size as usize;
    let hq = (cfg.num_attention_heads * cfg.head_dim) as usize;
    let cs = cfg.codebook_size as usize;

    // --- quantizer ---
    put(&mut w, "quantizer.rvq_first.vq.layers.0.table", cs * dim, 0.5);
    put(&mut w, "quantizer.rvq_first.output_proj.weight", lat * dim, 0.5);
    for i in 0..(nq - 1) {
        put(&mut w, &format!("quantizer.rvq_rest.vq.layers.{i}.table"), cs * dim, 0.5);
    }
    put(&mut w, "quantizer.rvq_rest.output_proj.weight", lat * dim, 0.5);

    // --- pre_conv (lat -> latent, k3) ---
    put(&mut w, "pre_conv.conv.weight", latent * lat * 3, 0.3);
    put(&mut w, "pre_conv.conv.bias", latent, 0.1);

    // --- pre_transformer ---
    put(&mut w, "pre_transformer.input_proj.weight", hidden * latent, 0.5);
    put(&mut w, "pre_transformer.input_proj.bias", hidden, 0.1);
    for l in 0..cfg.num_hidden_layers as usize {
        let p = |leaf: &str| format!("pre_transformer.layers.{l}.{leaf}");
        put(&mut w, &p("input_layernorm.weight"), hidden, 1.0);
        put(&mut w, &p("self_attn.q_proj.weight"), hq * hidden, 0.8);
        put(&mut w, &p("self_attn.k_proj.weight"), hq * hidden, 0.8);
        put(&mut w, &p("self_attn.v_proj.weight"), hq * hidden, 0.8);
        put(&mut w, &p("self_attn.o_proj.weight"), hidden * hq, 0.8);
        // LayerScale is 0.01-ish in the real checkpoint; keep it O(1) here so
        // the attention output (the ONLY thing the window mask changes) is not
        // scaled into the fp noise floor before it reaches the waveform.
        put(&mut w, &p("self_attn_layer_scale.scale"), hidden, 1.0);
        put(&mut w, &p("post_attention_layernorm.weight"), hidden, 1.0);
        put(&mut w, &p("mlp.gate_proj.weight"), ff * hidden, 0.5);
        put(&mut w, &p("mlp.up_proj.weight"), ff * hidden, 0.5);
        put(&mut w, &p("mlp.down_proj.weight"), hidden * ff, 0.5);
        put(&mut w, &p("mlp_layer_scale.scale"), hidden, 0.5);
    }
    put(&mut w, "pre_transformer.norm.weight", hidden, 1.0);
    put(&mut w, "pre_transformer.output_proj.weight", latent * hidden, 0.5);
    put(&mut w, "pre_transformer.output_proj.bias", latent, 0.1);

    // --- upsample stages (convtr k=stride=factor, then ConvNeXt) ---
    for (u, &factor) in cfg.upsampling_ratios.iter().enumerate() {
        let f = factor as usize;
        put(&mut w, &format!("upsample.{u}.0.conv.weight"), latent * latent * f, 0.3);
        put(&mut w, &format!("upsample.{u}.0.conv.bias"), latent, 0.1);
        let b = format!("upsample.{u}.1");
        put(&mut w, &format!("{b}.dwconv.conv.weight"), latent * 7, 0.3);
        put(&mut w, &format!("{b}.dwconv.conv.bias"), latent, 0.1);
        put(&mut w, &format!("{b}.norm.weight"), latent, 1.0);
        put(&mut w, &format!("{b}.norm.bias"), latent, 0.1);
        put(&mut w, &format!("{b}.pwconv1.weight"), 4 * latent * latent, 0.3);
        put(&mut w, &format!("{b}.pwconv1.bias"), 4 * latent, 0.1);
        put(&mut w, &format!("{b}.pwconv2.weight"), latent * 4 * latent, 0.3);
        put(&mut w, &format!("{b}.pwconv2.bias"), latent, 0.1);
        put(&mut w, &format!("{b}.gamma"), latent, 0.5);
    }

    // --- SEANet decoder ---
    let dec = cfg.decoder_dim as usize;
    put(&mut w, "decoder.0.conv.weight", dec * latent * 7, 0.3);
    put(&mut w, "decoder.0.conv.bias", dec, 0.1);
    for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
        let (in_dim, out_dim) = (dec >> i, dec >> (i + 1));
        let bp = format!("decoder.{}", i + 1);
        put(&mut w, &format!("{bp}.block.0.alpha"), in_dim, 0.3);
        put(&mut w, &format!("{bp}.block.0.beta"), in_dim, 0.3);
        put(&mut w, &format!("{bp}.block.1.conv.weight"), in_dim * out_dim * 2 * rate as usize, 0.3);
        put(&mut w, &format!("{bp}.block.1.conv.bias"), out_dim, 0.1);
        for j in [2usize, 3, 4] {
            put(&mut w, &format!("{bp}.block.{j}.act1.alpha"), out_dim, 0.3);
            put(&mut w, &format!("{bp}.block.{j}.act1.beta"), out_dim, 0.3);
            put(&mut w, &format!("{bp}.block.{j}.conv1.conv.weight"), out_dim * out_dim * 7, 0.2);
            put(&mut w, &format!("{bp}.block.{j}.conv1.conv.bias"), out_dim, 0.1);
            put(&mut w, &format!("{bp}.block.{j}.act2.alpha"), out_dim, 0.3);
            put(&mut w, &format!("{bp}.block.{j}.act2.beta"), out_dim, 0.3);
            put(&mut w, &format!("{bp}.block.{j}.conv2.conv.weight"), out_dim * out_dim, 0.2);
            put(&mut w, &format!("{bp}.block.{j}.conv2.conv.bias"), out_dim, 0.1);
        }
    }
    let tail = dec >> cfg.upsample_rates.len();
    put(&mut w, "decoder.5.alpha", tail, 0.3);
    put(&mut w, "decoder.5.beta", tail, 0.3);
    put(&mut w, "decoder.6.conv.weight", tail * 7, 0.3);
    put(&mut w, "decoder.6.conv.bias", 1, 0.1);
    w
}

fn codes_for(cfg: &CodecConfig, t: usize) -> Vec<u32> {
    let nq = cfg.num_quantizers as usize;
    let mut r = Lcg::new(7);
    (0..t * nq).map(|_| r.next_u64() as u32 % cfg.codebook_size).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// A waveform that has been clamped flat carries no information about the
/// attention mask, so a comparison over one would pass vacuously.
fn assert_not_saturated(wav: &[f32], what: &str) {
    let interior = wav.iter().filter(|x| x.abs() < 0.99).count();
    assert!(
        interior * 2 > wav.len(),
        "{what}: {interior}/{} samples inside the clamp - too saturated to discriminate",
        wav.len()
    );
    let peak = wav.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(peak > 1e-3, "{what}: waveform is effectively silent (peak {peak:.3e})");
}

/// Both decode paths on a clip SHORTER than one window. A sliding-window mask
/// and a plain causal mask are identical here by construction, so this pins the
/// floating-point noise floor between the WGSL and host implementations - and
/// guards that a windowing change leaves the already-fitting case alone.
#[test]
fn short_clip_agrees_across_both_decode_paths() {
    let cfg = tiny_cfg();
    let w = tiny_weights(&cfg);
    let t = (WINDOW - 1) as usize; // strictly inside one window
    let codes = codes_for(&cfg, t);

    let gpu = Codec::from_weights(cfg.clone(), w.clone()).decode(&codes);
    let host = StreamingCodecDecoder::from_parts(w.clone(), cfg.clone()).decode_streaming(&codes, 0);

    assert_eq!(gpu.len(), t * cfg.decode_upsample_rate as usize);
    assert_eq!(gpu.len(), host.len(), "length mismatch");
    assert_not_saturated(&gpu, "short clip");
    let d = max_abs(&gpu, &host);
    eprintln!("short clip (T={t}, window={WINDOW}): max-abs {d:.3e} over {} samples", gpu.len());
    assert!(d < 1e-4, "in-window decode paths disagree: {d}");

    // A windowing fix must not perturb inputs that already fit: with every
    // query's key set fully inside the window, widening the window has to
    // leave the arithmetic BIT-identical, not merely close. Asserted on both
    // implementations, since both grew a window bound.
    let mut unbounded = cfg.clone();
    unbounded.sliding_window = 4096;
    let gpu_wide = Codec::from_weights(unbounded.clone(), w.clone()).decode(&codes);
    let host_wide = StreamingCodecDecoder::from_parts(w, unbounded).decode_streaming(&codes, 0);
    assert_eq!(gpu, gpu_wide, "device decode of an in-window clip is not window-independent");
    assert_eq!(host, host_wide, "host decode of an in-window clip is not window-independent");
}

/// The real long-form case: ten windows of frames. `Codec::decode` masks key
/// `j` out once `i - j >= sliding_window`; the streaming host decoder must do
/// the same, or every frame past the first window attends to context the
/// reference never let it see and the two waveforms diverge.
#[test]
fn long_clip_agrees_across_both_decode_paths() {
    let cfg = tiny_cfg();
    let w = tiny_weights(&cfg);
    let t = 10 * WINDOW as usize; // ten windows deep
    let codes = codes_for(&cfg, t);

    let gpu = Codec::from_weights(cfg.clone(), w.clone()).decode(&codes);
    let host = StreamingCodecDecoder::from_parts(w, cfg.clone()).decode_streaming(&codes, 0);

    assert_eq!(gpu.len(), t * cfg.decode_upsample_rate as usize);
    assert_eq!(gpu.len(), host.len(), "length mismatch");
    assert_not_saturated(&gpu, "long clip");
    let d = max_abs(&gpu, &host);
    eprintln!("long clip (T={t}, window={WINDOW}): max-abs {d:.3e} over {} samples", gpu.len());
    // Measured 7.5e-6 with the window applied on both sides, against 2.6e-3
    // with it applied on neither (what this test caught): the ceiling sits an
    // order of magnitude above the fp noise floor and more than an order below
    // the defect, so backend fp reassociation cannot make it flap and a
    // dropped mask cannot slip under it.
    assert!(d < 1e-4, "beyond-window decode paths disagree (windowed mask missing?): {d}");
}

/// The window must actually BE a window: decoding with `sliding_window` set
/// wide enough to cover the whole clip has to differ from decoding the same
/// codes with a narrow window. Without this, both tests above would still pass
/// if the mask were silently dropped on BOTH sides.
#[test]
fn the_window_changes_the_waveform() {
    let narrow = tiny_cfg();
    let mut wide = tiny_cfg();
    let t = 10 * WINDOW as usize;
    wide.sliding_window = t as u32 * 4; // >= T, degenerates to plain causal
    let w = tiny_weights(&narrow);
    let codes = codes_for(&narrow, t);

    let a = Codec::from_weights(narrow, w.clone()).decode(&codes);
    let b = Codec::from_weights(wide.clone(), w.clone()).decode(&codes);
    let d = max_abs(&a, &b);
    eprintln!("narrow vs wide window (T={t}): max-abs {d:.3e}");
    assert!(d > 1e-3, "sliding window has no effect on the waveform: {d}");

    // Same statement for the host path, so neither implementation can pass the
    // parity tests above by ignoring the config field on both sides.
    let ha = StreamingCodecDecoder::from_parts(w.clone(), tiny_cfg()).decode_streaming(&codes, 0);
    let hb = StreamingCodecDecoder::from_parts(w, wide).decode_streaming(&codes, 0);
    let hd = max_abs(&ha, &hb);
    eprintln!("narrow vs wide window, host path (T={t}): max-abs {hd:.3e}");
    assert!(hd > 1e-3, "host decode ignores sliding_window entirely: {hd}");
}
