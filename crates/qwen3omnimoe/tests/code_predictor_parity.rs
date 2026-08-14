// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-Omni's Talker code predictor (`talker.code_predictor.*`) vs. the
//! real transformers reference, on real weights. `qwen3tts::mtp::MtpModel` is
//! near-exact reuse (dense, 5-layer, 16 code groups, same GQA+QK-norm+RoPE
//! block as the Talker) per the plan -- this test proves that reuse is
//! actually correct against real Omni weights, not just against the
//! standalone Qwen3-TTS shape it was built for.
//!
//! Weights are read straight from the real HF checkpoint via mmap and
//! renamed with `qwen3tts::import::mtp_hf_to_brain` (the exact rename
//! `qwen3tts::import::import_mtp` already applies for standalone Qwen3-TTS, and
//! that `qwen3omnimoe::import::map_code_predictor`'s doc comment claims -- currently
//! incorrectly -- is not
//! needed again), then built directly with `MtpModel::build_on` -- no
//! `ParamStore`/checkpoint-file round trip, same pattern as every other
//! real-weight test in this crate.
//!
//! The golden (`tools/goldens/omni_dump_reference.py`'s `talkcp`) is a
//! 2-position "prefill" (position 0 = an arbitrary but fixed hidden-state
//! stand-in, position 1 = codebook-0's embedding) predicting codebook 1's
//! logits -- the smallest real forward the reference model supports.
//! `MtpModel::logits` always runs the full fixed `num_code_groups`-length
//! sequence (its GPU graph is built once, at that length); positions 2.. are
//! zero-padded here, which is harmless under causal attention: position 1's
//! logits cannot see positions after it, real content or not.
//!
//! Real-weight-adjacent: skips cleanly when the checkpoint shards holding
//! `talker.code_predictor.*` (shards 14-15 of 15) are absent.
//!
//! usage: `BRAIN_QWEN3OMNIMOE_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test code_predictor_parity -- --ignored --nocapture`

use std::collections::HashMap;
use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use qwen3omnimoe::config::OmniConfig;
use qwen3tts::import::mtp_hf_to_brain;
use qwen3tts::mtp::MtpModel;

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
fn matches_the_real_code_predictor() {
    let Some(dir) = std::env::var("BRAIN_QWEN3OMNIMOE_HF_DIR").ok().map(PathBuf::from) else {
        eprintln!("skip: BRAIN_QWEN3OMNIMOE_HF_DIR unset");
        return;
    };
    let Some(layer0_shard) = shard_for(&dir, "talker.code_predictor.model.layers.0.self_attn.q_proj.weight") else {
        eprintln!("skip: index doesn't (yet) have the shard holding talker.code_predictor.model.layers.0");
        return;
    };
    let Some(lm_head_shard) = shard_for(&dir, "talker.code_predictor.lm_head.0.weight") else {
        eprintln!("skip: index doesn't (yet) have the shard holding talker.code_predictor.lm_head");
        return;
    };
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/golden/omni/omni_code_predictor.safetensors");
    if !golden_path.exists() {
        eprintln!("skip: {golden_path:?} missing (run `make fetch/testdata`)");
        return;
    }

    let config_json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let cfg = OmniConfig::parse(&config_json).expect("parse config.json").talker.code_predictor;

    // Layer weights + norm live in one shard, codec_embedding/lm_head span
    // two (see the module doc); open all that exist.
    let m0 = MmapSafetensors::open(&layer0_shard).expect("open layer0 shard");
    let m1 = MmapSafetensors::open(&lm_head_shard).expect("open lm_head shard");
    let get = |name: &str| m0.tensor_f32(name).or_else(|| m1.tensor_f32(name)).unwrap_or_else(|| panic!("missing tensor {name}"));

    let mut decoder: HashMap<String, Vec<f32>> = HashMap::new();
    for l in 0..cfg.n_layers {
        for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"] {
            let hf = format!("talker.code_predictor.model.layers.{l}.{leaf}");
            let brain_name = mtp_hf_to_brain(&hf).unwrap_or_else(|| panic!("mtp_hf_to_brain rejected {hf}"));
            decoder.insert(brain_name, get(&hf));
        }
    }
    let norm_hf = "talker.code_predictor.model.norm.weight";
    decoder.insert(mtp_hf_to_brain(norm_hf).unwrap(), get(norm_hf));

    let n_residual = cfg.n_residual() as usize;
    let codec_embedding: Vec<Vec<f32>> = (0..n_residual).map(|i| get(&format!("talker.code_predictor.model.codec_embedding.{i}.weight"))).collect();
    let lm_head: Vec<Vec<f32>> = (0..n_residual).map(|i| get(&format!("talker.code_predictor.lm_head.{i}.weight"))).collect();

    let gpu = gpu_core::Gpu::new(qwen3tts::mtp::PIPELINES);
    let model = MtpModel::build_on(gpu, cfg.clone(), decoder, codec_embedding, lm_head);

    let golden = MmapSafetensors::open(&golden_path).expect("open golden");
    let want_in_embed = golden.tensor_f32("in_embed").expect("golden in_embed"); // [2, h]
    let want_logits = golden.tensor_f32("logits").expect("golden logits"); // [vocab] (squeeze(0) of a 1-new-token prediction)

    let d = cfg.d_model as usize;
    let t = cfg.num_code_groups as usize;
    let mut inputs_embeds = vec![0f32; t * d];
    inputs_embeds[..2 * d].copy_from_slice(&want_in_embed);

    let got_logits_all = model.logits(&inputs_embeds); // [(t-1) * vocab], row 0 = codebook 1's logits
    let v = cfg.vocab as usize;
    let got_logits = &got_logits_all[0..v];

    // The reference applies lm_head[generation_steps] (a single, shared head)
    // to EVERY input position's hidden state, not just the newest one:
    // `logits = self.lm_head[generation_steps](hidden_states)` broadcasts
    // over the whole [1, seq, hidden] tensor. So golden row 0 is
    // lm_head[0]@hidden[0] (not a meaningful prediction -- position 0 is the
    // Talker-hidden conditioning slot, never trained to predict anything
    // through lm_head[0]) and row 1 is lm_head[0]@hidden[1], the actual
    // "predict codebook 1 from position 1" quantity MtpModel::logits computes.
    let want_row1 = &want_logits[v..2 * v];
    let (cos, max_abs) = cosine_max_abs(got_logits, want_row1);
    println!("code_predictor logits: cosine={cos:.6} max_abs={max_abs:.6}");
    assert!(cos > 0.999, "code predictor logits cosine {cos} <= 0.999");
}
