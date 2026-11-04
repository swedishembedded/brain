// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The released Q8_0 GGUF, read through `qwen35moe::gguf_load::source`, must be
//! the SAME model as the released bf16 checkpoint: every tensor of a Gated-
//! DeltaNet layer (0) and of a full-attention layer (3), a routed expert from
//! each end of the stack, the shared expert and the router, compared by cosine
//! against the Hugging Face tensors.
//!
//! This is the gate that catches what no structural check can. llama.cpp's
//! converter stores a Gated-DeltaNet layer's value heads group-major, while the
//! model indexes them sub-major, so skipping the reorder leaves every name,
//! shape and value range intact and merely points each head's decay, bias and
//! projection at another head's state: measured on the real files before the fix,
//! cosine 0.30 (`A_log`), 0.01 (`dt_bias`), 0.82 (`conv1d`), 0.54
//! (`in_proj_qkv`), 0.08 (`in_proj_z`), 0.25/0.14 (`in_proj_a/b`) and 0.06
//! (`out_proj`).
//!
//! Needs `Qwen/Qwen3.6-35B-A3B` (bf16) and `unsloth/Qwen3.6-35B-A3B-GGUF/Q8_0.gguf`
//! under `BRAIN_MODELS_DIR`; skips loudly when either is absent.

use checkpoint::gguf::MmapGguf;
use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;
use qwen35moe::gguf_load;

/// Q8_0 rounds each weight to 8 bits against a per-32 scale, so a matrix
/// agrees with its bf16 source to ~1e-4 in 1-cosine; a wrong head order is at
/// cosine <= 0.82. The unquantised leaves (`A_log`, `dt_bias`, `conv1d`) agree
/// to bf16 rounding.
const MIN_COSINE: f64 = 0.9995;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

#[test]
fn the_gguf_matches_the_bf16_checkpoint_tensor_by_tensor() {
    let Some(models) = std::env::var_os("BRAIN_MODELS_DIR").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/share/brain/models"))) else {
        return brain_testutil::skip("no models directory");
    };
    let (gguf_path, hf_dir) = (models.join("unsloth/Qwen3.6-35B-A3B-GGUF/Q8_0.gguf"), models.join("Qwen/Qwen3.6-35B-A3B"));
    if !gguf_path.exists() || !hf_dir.join("model.safetensors.index.json").exists() {
        return brain_testutil::skip("the Qwen3.6-35B-A3B GGUF and bf16 checkpoints are not both under BRAIN_MODELS_DIR");
    }
    let mg = MmapGguf::open(gguf_path.to_str().unwrap()).unwrap();
    let cfg = gguf_load::resident_config(&mg, 64).unwrap();
    assert_eq!(cfg.n_layers, 40, "this GGUF carries no MTP block: block_count IS the decoder depth");
    let src = gguf_load::source(&mg, &cfg).unwrap();
    let hf = WeightReader::open_hf_dir(&hf_dir).unwrap();
    let hf_layer = |l: usize, leaf: &str| hf.tensor(&format!("model.language_model.layers.{l}.{leaf}")).unwrap_or_else(|| panic!("hf layer {l} {leaf}"));
    let brain = |name: &str| {
        let mut out = Vec::new();
        assert!(src.with_tensor(name, &mut |d| out = d.to_vec()), "{name}");
        out
    };

    let mut checks: Vec<(String, f64)> = Vec::new();
    let mut check = |what: String, got: &[f32], want: &[f32]| checks.push((what, cosine(got, want)));

    for (leaf, hf_leaf) in [
        ("A_log", "A_log"),
        ("dt_bias", "dt_bias"),
        ("conv1d.weight", "conv1d.weight"),
        ("in_proj_qkv.weight", "in_proj_qkv.weight"),
        ("in_proj_z.weight", "in_proj_z.weight"),
        ("in_proj_a.weight", "in_proj_a.weight"),
        ("in_proj_b.weight", "in_proj_b.weight"),
        ("out_proj.weight", "out_proj.weight"),
        ("norm.weight", "norm.weight"),
    ] {
        check(format!("layer 0 linear_attn.{leaf}"), &brain(&format!("blocks.0.linear_attn.{leaf}")), &hf_layer(0, &format!("linear_attn.{hf_leaf}")));
    }
    for (leaf, hf_leaf) in [
        ("q_proj.weight", "q_proj.weight"),
        ("k_proj.weight", "k_proj.weight"),
        ("v_proj.weight", "v_proj.weight"),
        ("o_proj.weight", "o_proj.weight"),
    ] {
        check(format!("layer 3 self_attn.{leaf}"), &brain(&format!("blocks.3.self_attn.{leaf}")), &hf_layer(3, &format!("self_attn.{hf_leaf}")));
    }
    // The RMSNorms the reference applies as `x * (1 + w)`: llama.cpp folds the
    // `1 +` into the stored weight, and so does brain (`x * w`), so the GGUF
    // holds exactly `1 + w_hf`.
    let folded = |hf: Vec<f32>| hf.into_iter().map(|w| 1.0 + w).collect::<Vec<f32>>();
    for (what, ours, theirs) in [
        ("layer 3 q_norm".to_string(), "blocks.3.self_attn.q_norm.weight".to_string(), hf_layer(3, "self_attn.q_norm.weight")),
        ("layer 3 k_norm".to_string(), "blocks.3.self_attn.k_norm.weight".to_string(), hf_layer(3, "self_attn.k_norm.weight")),
        ("layer 0 ln1".to_string(), "blocks.0.ln1.weight".to_string(), hf_layer(0, "input_layernorm.weight")),
        ("layer 3 ln2".to_string(), "blocks.3.ln2.weight".to_string(), hf_layer(3, "post_attention_layernorm.weight")),
        ("final norm".to_string(), "norm.weight".to_string(), hf.tensor("model.language_model.norm.weight").unwrap()),
    ] {
        check(what, &brain(&ours), &folded(theirs));
    }
    for l in [0usize, 3] {
        check(format!("layer {l} router"), &brain(&format!("blocks.{l}.mlp.router.weight")), &hf_layer(l, "mlp.gate.weight"));
        check(format!("layer {l} shared_expert_gate"), &brain(&format!("blocks.{l}.mlp.shared_expert_gate.weight")), &hf_layer(l, "mlp.shared_expert_gate.weight"));
        for (leaf, hf_leaf) in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")] {
            check(
                format!("layer {l} shared_expert.{leaf}"),
                &brain(&format!("blocks.{l}.mlp.shared_expert.{leaf}.weight")),
                &hf_layer(l, &format!("mlp.shared_expert.{hf_leaf}.weight")),
            );
        }
    }
    // Routed experts: HF fuses gate|up as `gate_up_proj [E, 2*ff, d]` (gate rows
    // first) and keeps `down_proj [E, d, ff]`; the GGUF stacks them separately.
    let (ff, d) = (cfg.moe_intermediate_size as usize, cfg.d_model as usize);
    let fused = hf_layer(0, "mlp.experts.gate_up_proj");
    let down = hf_layer(0, "mlp.experts.down_proj");
    for e in [0usize, 255] {
        let base = e * 2 * ff * d;
        check(format!("layer 0 expert {e} gate"), &brain(&format!("blocks.0.mlp.experts.{e}.gate.weight")), &fused[base..base + ff * d]);
        check(format!("layer 0 expert {e} up"), &brain(&format!("blocks.0.mlp.experts.{e}.up.weight")), &fused[base + ff * d..base + 2 * ff * d]);
        check(format!("layer 0 expert {e} down"), &brain(&format!("blocks.0.mlp.experts.{e}.down.weight")), &down[e * d * ff..(e + 1) * d * ff]);
    }

    let bad: Vec<String> = checks.iter().filter(|(_, c)| *c < MIN_COSINE).map(|(n, c)| format!("{n}: cosine {c:.6}")).collect();
    for (n, c) in &checks {
        println!("{n:<44} cosine {c:.7}");
    }
    assert!(bad.is_empty(), "the GGUF disagrees with the bf16 checkpoint:\n  {}", bad.join("\n  "));
}
