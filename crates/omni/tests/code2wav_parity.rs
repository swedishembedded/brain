// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's Code2Wav vocoder (`code2wav.*`) vs. the real transformers
//! reference, on real weights. Validates `codec::Codec::decode_omni`: the
//! `code_embedding`-mean input path, the reused pre-transformer/upsample/
//! SEANet decoder (a `codec::CodecConfig` shape bump — `hidden_size`
//! 512→1024, `intermediate_size` 1024→3072 — from the standalone Qwen3-TTS
//! codec this crate already implements), and the symmetric-crop transposed
//! conv the SEANet decoder's upsampling stages need (see
//! `decode_omni`'s own doc for the length-mismatch finding that surfaced
//! this: the naive `Lo = L*stride` assumption is 555 samples too long for
//! this test's `T=8` golden).
//!
//! Weights are read straight from the real HF checkpoint via mmap, keeping
//! their HF-relative names unchanged (`codec::Codec::from_weights` and
//! `codec::Codec::transformer` key their `ParamStore` lookups by the exact
//! HF-style leaf names — `pre_transformer.layers.N.self_attn.q_proj.weight`,
//! not a renamed `blocks.N.attn.wq.weight`), bypassing `omni::import::
//! map_code2wav` (still true for this test — it reads the shard directly,
//! same pattern every real-weight test in this crate uses). The loader-side
//! naming mismatch this comment used to describe (`map_code2wav` renaming
//! `pre_transformer.layers.N` onto the dense-attention convention, which
//! `codec::Codec`'s own `ParamStore` lookups never expected) is FIXED as of
//! the M9b follow-up: `map_code2wav` is now a plain prefix strip, matching
//! what this test already proves is correct.
//!
//! Real-weight-adjacent: skips cleanly when the checkpoint shard holding
//! `code2wav.*` (shard 15 of 15) is absent.
//!
//! usage: `BRAIN_OMNI_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test code2wav_parity -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use codec::config::CodecConfig;
use codec::model::Codec;
use omni::config::OmniConfig;

fn shard_for(dir: &std::path::Path, tensor: &str) -> Option<PathBuf> {
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?).ok()?;
    let shard = idx["weight_map"].as_object()?.get(tensor)?.as_str()?;
    let p = dir.join(shard);
    p.exists().then_some(p)
}

fn cosine_max_abs(got: &[f32], want: &[f32]) -> (f64, f32) {
    assert_eq!(got.len(), want.len(), "shape mismatch: got {} elems, want {}", got.len(), want.len());
    let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = got.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb).max(1e-12), got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max))
}

#[test]
#[ignore]
fn matches_the_real_code2wav() {
    let Some(dir) = std::env::var("BRAIN_OMNI_HF_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset");
        return;
    };
    let Some(shard) = shard_for(&dir, "code2wav.code_embedding.weight") else {
        eprintln!("skip: index doesn't (yet) have the shard holding code2wav.*");
        return;
    };
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/omni/omni_code2wav.safetensors");
    if !golden_path.exists() {
        eprintln!("skip: {golden_path:?} missing (run `make fetch/testdata`)");
        return;
    }

    let config_json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let oc = OmniConfig::parse(&config_json).expect("parse config.json").code2wav;
    let cfg = CodecConfig {
        num_quantizers: oc.num_quantizers,
        num_semantic_quantizers: oc.num_semantic_quantizers,
        codebook_size: oc.codebook_size,
        semantic_codebook_size: oc.semantic_codebook_size,
        codebook_dim: oc.codebook_dim,
        latent_dim: oc.hidden_size, // unused by decode_omni; kept equal for sanity
        hidden_size: oc.hidden_size,
        intermediate_size: oc.intermediate_size,
        num_hidden_layers: oc.num_hidden_layers,
        num_attention_heads: oc.num_attention_heads,
        num_key_value_heads: oc.num_key_value_heads,
        head_dim: oc.hidden_size / oc.num_attention_heads,
        sliding_window: oc.sliding_window,
        rope_theta: oc.rope_theta,
        rms_norm_eps: oc.rms_norm_eps,
        layer_scale_initial_scale: oc.layer_scale_initial_scale,
        decoder_dim: oc.decoder_dim,
        upsample_rates: oc.upsample_rates.clone(),
        upsampling_ratios: oc.upsampling_ratios.clone(),
        input_sample_rate: oc.output_sample_rate,
        output_sample_rate: oc.output_sample_rate,
        decode_upsample_rate: oc.total_upsample(),
        enc: Default::default(), // encode-path only; decode_omni never reads it
    };

    let mmap = MmapSafetensors::open(&shard).expect("open shard");
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    for name in mmap.names() {
        if let Some(rest) = name.strip_prefix("code2wav.") {
            init.insert(rest.to_string(), mmap.tensor_f32(name).unwrap());
        }
    }
    let codec = Codec::from_weights(cfg, init);

    let golden = MmapSafetensors::open(&golden_path).expect("open golden");
    let codes_qmajor = golden.tensor_f32("codes").expect("golden codes"); // [nq, T]
    let want_wav = golden.tensor_f32("wav").expect("golden wav");
    let nq = oc.num_quantizers as usize;
    let t = codes_qmajor.len() / nq;

    // decode_omni takes [T, nq] row-major; the golden is [nq, T] (quantizer-major).
    let mut codes_tmajor = vec![0u32; t * nq];
    for q in 0..nq {
        for ti in 0..t {
            codes_tmajor[ti * nq + q] = codes_qmajor[q * t + ti] as u32;
        }
    }

    let got_wav = codec.decode_omni(&codes_tmajor);
    let (cos, max_abs) = cosine_max_abs(&got_wav, &want_wav);
    println!("code2wav wav: cosine={cos:.6} max_abs={max_abs:.6} len(got)={} len(want)={}", got_wav.len(), want_wav.len());
    assert!(cos > 0.999, "code2wav waveform cosine {cos} <= 0.999");
}
