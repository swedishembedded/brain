// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Load a Qwen3-ASR checkpoint's audio encoder + projector into the weight map
//! [`crate::encoder::AudioEncoder`] consumes. HF names sit under a `thinker.`
//! prefix (the Qwen "Thinker+Talker" convention, not the bare `model.` an
//! omni-less single-tower checkpoint would use): `thinker.audio_tower.*` for
//! the encoder, and its own `proj1`/`proj2` leaves (not a separate
//! `multi_modal_projector.*` module - there isn't one in the released
//! checkpoint, and `proj2`'s [2048,1024] output shape exactly matches the
//! decoder's [151936,2048] embedding table's hidden dim, confirming it IS the
//! bridge) for what this module still stores under brain's own
//! `multi_modal_projector.linear_{1,2}` key names. The per-layer
//! `self_attn.{q,k,v}_proj` are fused into a single `blocks.{b}.qkv` (weights
//! stacked `[3·d, d]`, biases `[3·d]`) to match the `model::vit` block layout.
//! No transposes (brain `matmul` is `x·Wᵀ` with HF's `[out, in]` layout). The
//! Qwen3 decoder half is loaded separately via `qwen`-style mapping under
//! `thinker.model.`.

use std::collections::HashMap;
use std::path::Path;

use crate::config::AudioEncoderConfig;

/// Read `dir` (single or sharded safetensors) and return the audio-encoder weight
/// map keyed as `AudioEncoder::new` expects.
pub fn load_audio_encoder(dir: &Path, cfg: &AudioEncoderConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    let tensors = checkpoint::safetensors::read_model_dir(dir)?;
    let mut src: HashMap<String, Vec<f32>> = HashMap::new();
    for t in tensors {
        src.insert(t.name, t.data);
    }
    Ok(map_audio_encoder(&src, cfg))
}

/// Pure name/shape remap (testable without a checkpoint on disk).
pub fn map_audio_encoder(src: &HashMap<String, Vec<f32>>, cfg: &AudioEncoderConfig) -> HashMap<String, Vec<f32>> {
    let get = |name: &str| -> Vec<f32> { src.get(name).unwrap_or_else(|| panic!("Qwen3-ASR tensor missing: {name}")).clone() };
    let mut w = HashMap::new();

    // conv stem + conv_out
    for i in 1..=3 {
        w.insert(format!("conv2d{i}.weight"), get(&format!("thinker.audio_tower.conv2d{i}.weight")));
        w.insert(format!("conv2d{i}.bias"), get(&format!("thinker.audio_tower.conv2d{i}.bias")));
    }
    w.insert("conv_out.weight".into(), get("thinker.audio_tower.conv_out.weight"));

    // transformer blocks (fuse q/k/v)
    for b in 0..cfg.n_layers {
        let a = |leaf: &str| get(&format!("thinker.audio_tower.layers.{b}.{leaf}"));
        let mut qkv_w = a("self_attn.q_proj.weight");
        qkv_w.extend(a("self_attn.k_proj.weight"));
        qkv_w.extend(a("self_attn.v_proj.weight"));
        let mut qkv_b = a("self_attn.q_proj.bias");
        qkv_b.extend(a("self_attn.k_proj.bias"));
        qkv_b.extend(a("self_attn.v_proj.bias"));
        w.insert(format!("blocks.{b}.qkv.weight"), qkv_w);
        w.insert(format!("blocks.{b}.qkv.bias"), qkv_b);
        w.insert(format!("blocks.{b}.proj.weight"), a("self_attn.out_proj.weight"));
        w.insert(format!("blocks.{b}.proj.bias"), a("self_attn.out_proj.bias"));
        w.insert(format!("blocks.{b}.norm1.weight"), a("self_attn_layer_norm.weight"));
        w.insert(format!("blocks.{b}.norm1.bias"), a("self_attn_layer_norm.bias"));
        w.insert(format!("blocks.{b}.norm2.weight"), a("final_layer_norm.weight"));
        w.insert(format!("blocks.{b}.norm2.bias"), a("final_layer_norm.bias"));
        w.insert(format!("blocks.{b}.fc1.weight"), a("fc1.weight"));
        w.insert(format!("blocks.{b}.fc1.bias"), a("fc1.bias"));
        w.insert(format!("blocks.{b}.fc2.weight"), a("fc2.weight"));
        w.insert(format!("blocks.{b}.fc2.bias"), a("fc2.bias"));
    }

    // final norm + projector
    w.insert("ln_post.weight".into(), get("thinker.audio_tower.ln_post.weight"));
    w.insert("ln_post.bias".into(), get("thinker.audio_tower.ln_post.bias"));
    for i in 1..=2 {
        w.insert(format!("multi_modal_projector.linear_{i}.weight"), get(&format!("thinker.audio_tower.proj{i}.weight")));
        w.insert(format!("multi_modal_projector.linear_{i}.bias"), get(&format!("thinker.audio_tower.proj{i}.bias")));
    }
    w
}

/// Map an HF Qwen3-ASR decoder tensor name (`thinker.model.*`) to a brain
/// Qwen parameter name - standard Qwen3 leaves under the `thinker.` prefix
/// this checkpoint's decoder half shares with the audio tower (see this
/// module's own doc comment). Tied embeddings mean `embed_tokens` also
/// serves as the head; the checkpoint's separate `thinker.lm_head.weight`
/// (present but redundant under tying) is simply never looked up.
pub fn map_decoder(hf: &str) -> Option<String> {
    let s = hf.strip_prefix("thinker.model.")?;
    match s {
        "embed_tokens.weight" => return Some("tok.weight".into()),
        "norm.weight" => return Some("norm.weight".into()),
        _ => {}
    }
    let (n, leaf) = s.strip_prefix("layers.")?.split_once('.')?;
    let mapped = match leaf {
        "input_layernorm.weight" => "ln1.weight",
        "post_attention_layernorm.weight" => "ln2.weight",
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.o_proj.weight" => "attn.wo.weight",
        "self_attn.q_norm.weight" => "attn.q_norm.weight",
        "self_attn.k_norm.weight" => "attn.k_norm.weight",
        "mlp.gate_proj.weight" => "mlp.gate.weight",
        "mlp.up_proj.weight" => "mlp.up.weight",
        "mlp.down_proj.weight" => "mlp.down.weight",
        _ => return None,
    };
    Some(format!("blocks.{n}.{mapped}"))
}

/// Build the brain Qwen decoder weight map from a name→f32 tensor map.
pub fn map_decoder_weights(src: &HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
    let mut w = HashMap::new();
    for (name, data) in src {
        if let Some(mapped) = map_decoder(name) {
            w.insert(mapped, data.clone());
        }
    }
    w
}
