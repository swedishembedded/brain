// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA adapters for the H3 DiT, over `model::lora`'s runtime corrections.
//!
//! A third-party adapter for this DiT - a diffusers PEFT save such as a
//! few-step distillation - names the checkpoint's own linears:
//! `transformer_blocks.{i}` and `token_refiner.refiner_blocks.{j}`, each
//! `attn.to_q`/`to_k`/`to_v`/`to_out.0`, `ff.net.0.proj` and `ff.net.2`.
//! Every target is checked against the architecture, by name and shape,
//! before a weight is read, so an adapter for another model is refused up
//! front instead of after a 30 GB load.
//!
//! `ff.net.0.proj` is one stored tensor the DiT uploads as two linears, the
//! SwiGLU value rows and then the gate rows (`crate::block::load_fc1`), so
//! its correction is split into the same two row bands.
//!
//! The DiT reads its checkpoint through [`RuntimeLora::folded`]: each
//! correction is folded into the source weights as they are read, in fp32,
//! before a resident build quantizes them - the merged model's own int8,
//! as the distillation's reference inference fuses it - so the adapter
//! costs no device memory and no work per step.
//!
//! Swedish Embedded AB implements few-step adapter inference for large video
//! diffusion models for its clients. If your team needs expertise in running
//! distilled video generators on its own hardware then you can procure our
//! services by sending an email to info@swedishembedded.com.

use model::adapter::AdapterSpec;
use model::lora::RuntimeLora;

use crate::config::H3TransformerConfig;

/// `[out, in]` of the adaptable linear `leaf` of a block.
fn leaf_shape(cfg: &H3TransformerConfig, leaf: &str) -> Option<(usize, usize)> {
    let (hidden, inner, ffn) = (cfg.hidden_size as usize, cfg.inner_dim() as usize, cfg.ffn_dim as usize);
    match leaf {
        "attn.to_q" | "attn.to_k" | "attn.to_v" => Some((inner, hidden)),
        "attn.to_out.0" => Some((hidden, inner)),
        "ff.net.0.proj" => Some((2 * ffn, hidden)),
        "ff.net.2" => Some((hidden, ffn)),
        _ => None,
    }
}

/// `[out, in]` of the DiT linear `key` names, or `None` when `key` is no
/// adaptable linear of a block this configuration has.
fn target_shape(cfg: &H3TransformerConfig, key: &str) -> Option<(usize, usize)> {
    let stem = key.strip_suffix(".weight")?;
    let (rest, blocks) = match (stem.strip_prefix("transformer_blocks."), stem.strip_prefix("token_refiner.refiner_blocks.")) {
        (Some(rest), _) => (rest, cfg.num_layers),
        (_, Some(rest)) => (rest, cfg.num_refiner_layers),
        _ => return None,
    };
    let (index, leaf) = rest.split_once('.')?;
    index.parse::<u32>().ok().filter(|&i| i < blocks)?;
    leaf_shape(cfg, leaf)
}

/// The corrections of the adapter `spec` names, at `spec.scale` on top of
/// the file's own `alpha/r`, each aimed at the DiT linear it corrects.
pub fn runtime_adapter(spec: &AdapterSpec, cfg: &H3TransformerConfig) -> Result<RuntimeLora, String> {
    let pairs = model::lora::read_external_adapter(&spec.path)?;
    let mut deltas = Vec::with_capacity(pairs.len() + 2 * (cfg.num_layers + cfg.num_refiner_layers) as usize);
    for p in &pairs {
        let (out, inn) = target_shape(cfg, &p.base_key).ok_or_else(|| format!("minimaxh3: adapter {}: {} is not a linear of this DiT's blocks", spec.path, p.base_key))?;
        if (p.out, p.inn) != (out, inn) {
            return Err(format!("minimaxh3: adapter {}: {} is [{}, {}], this DiT's is [{out}, {inn}]", spec.path, p.base_key, p.out, p.inn));
        }
        let delta = p.runtime_delta(spec.scale).ok_or_else(|| format!("minimaxh3: adapter {}: {} is a {} adapter, which has no low-rank form to evaluate beside a quantized linear", spec.path, p.stem, p.family()))?;
        if p.base_key.ends_with(".ff.net.0.proj.weight") {
            let ffn = cfg.ffn_dim as usize;
            deltas.push(delta.row_band(0, ffn)?);
            deltas.push(delta.row_band(ffn, ffn)?);
        } else {
            deltas.push(delta);
        }
    }
    Ok(RuntimeLora::new(deltas))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A diffusers PEFT save as a distillation ships it: the adapter's name
    /// kept in every key, `alpha` once in the header.
    pub(crate) fn write_peft(path: &std::path::Path, tensors: &[(String, Vec<usize>, Vec<f32>)], alpha: f32) {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt", "alpha": alpha.to_string()}));
        let mut blob = Vec::new();
        for (name, shape, data) in tensors {
            let start = blob.len();
            data.iter().for_each(|v| blob.extend_from_slice(&v.to_le_bytes()));
            header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, blob.len()]}));
        }
        let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&blob);
        std::fs::write(path, out).unwrap();
    }

    /// A rank-`r` adapter over every adaptable linear of `cfg`'s blocks and
    /// refiner blocks, seeded random.
    pub(crate) fn every_linear(cfg: &H3TransformerConfig, r: usize, seed: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut rng = data::rng::Lcg::new(seed);
        let prefixes = (0..cfg.num_layers).map(|i| format!("transformer_blocks.{i}")).chain((0..cfg.num_refiner_layers).map(|j| format!("token_refiner.refiner_blocks.{j}")));
        let mut out = Vec::new();
        for p in prefixes.collect::<Vec<_>>() {
            for leaf in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"] {
                let (o, i) = leaf_shape(cfg, leaf).unwrap();
                out.push((format!("{p}.{leaf}.lora_A.default.weight"), vec![r, i], rng.vec_scaled(r * i, 0.2)));
                out.push((format!("{p}.{leaf}.lora_B.default.weight"), vec![o, r], rng.vec_scaled(o * r, 0.2)));
            }
        }
        out
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("brain-minimaxh3-lora-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    /// Every linear of every block is a target, the fused SwiGLU projection
    /// as its two halves, each at the header's `alpha/r` times the strength.
    #[test]
    fn an_adapter_over_every_linear_maps_onto_the_dits_linears() {
        let cfg = H3TransformerConfig::tiny();
        let path = tmp("every.safetensors");
        write_peft(&path, &every_linear(&cfg, 2, 3), 1.0);
        let lora = runtime_adapter(&AdapterSpec { path: path.to_str().unwrap().into(), scale: 2.0 }, &cfg).expect("maps");
        let blocks = (cfg.num_layers + cfg.num_refiner_layers) as usize;
        assert_eq!(lora.len(), 7 * blocks, "q, k, v, out, the SwiGLU value and gate halves and the down projection");
        let ffn = cfg.ffn_dim as usize;
        let key = "transformer_blocks.0.ff.net.0.proj.weight";
        assert!(lora.claim(key, cfg.hidden_size as usize, 0, ffn, 0, cfg.hidden_size as usize).is_some(), "the value half");
        assert!(lora.claim(key, cfg.hidden_size as usize, ffn, ffn, 0, cfg.hidden_size as usize).is_some(), "the gate half");
    }

    /// An adapter for another model is refused by name before any weight is
    /// read: a target this DiT has no linear for, one past its last block,
    /// or one of the wrong shape.
    #[test]
    fn an_adapter_for_another_model_is_refused_naming_the_target() {
        let cfg = H3TransformerConfig::tiny();
        let (hidden, inner) = (cfg.hidden_size as usize, cfg.inner_dim() as usize);
        let refuse = |tag: &str, stem: String, (o, i): (usize, usize)| {
            let path = tmp(&format!("{tag}.safetensors"));
            write_peft(&path, &[(format!("{stem}.lora_A.weight"), vec![1, i], vec![0.1; i]), (format!("{stem}.lora_B.weight"), vec![o, 1], vec![0.1; o])], 1.0);
            let e = runtime_adapter(&AdapterSpec::new(path.to_str().unwrap()), &cfg).err().expect("refused");
            assert!(e.contains(&stem), "{tag}: {e}");
        };
        refuse("foreign", "transformer_blocks.0.attn.to_gate_compress".into(), (inner, hidden));
        refuse("past-the-end", format!("transformer_blocks.{}.attn.to_q", cfg.num_layers), (inner, hidden));
        refuse("wrong-shape", "transformer_blocks.0.attn.to_q".into(), (inner + 1, hidden));
    }
}
