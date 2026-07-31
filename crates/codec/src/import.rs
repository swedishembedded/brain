// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the official `Qwen3-TTS-Tokenizer-12Hz` safetensors checkpoint into a
//! brain `.safetensors` container — decode path **and** (additively) the encode path.
//!
//! The decoder lives under the `decoder.*` prefix (271 tensors); the encoder
//! lives under `encoder.*` (225 tensors, a HuggingFace `MimiModel`). We do a
//! near 1:1 name remap (the decoder strips its `decoder.` prefix; the encoder
//! keeps its `encoder.` prefix so the two never collide) with these transforms:
//!   * each Euclidean codebook is collapsed at import time from its two stored
//!     tensors `embedding_sum/embed_sum [bins,dim]` + `cluster_usage [bins]` into
//!     the usable embedding table `table = embed_sum / clamp(cluster_usage, eps)`
//!     (matches `EuclideanCodebook.decode`/`MimiEuclideanCodebook.embed`, eps =
//!     1e-5) — applied to both the decoder's codebooks and the encoder's;
//!   * the **decoder** quantizers' `input_proj` is dropped (decode never uses it);
//!     the **encoder** quantizers' `input_proj` is KEPT (encode projects
//!     `hidden_size -> codebook_dim` before the nearest-codebook search), while
//!     the encoder quantizers' `output_proj` (decode-side) is dropped;
//!   * the encoder only keeps the first `encoder_valid_num_quantizers` codebooks
//!     (1 semantic + 15 acoustic); the rest of the 32-deep RVQ and the codebooks'
//!     `initialized` flags are dropped (`encode` never reads past code 16);
//! No tensor is transposed: brain `matmul` is `x @ Wᵀ` with `W:[out,in]`, exactly
//! `nn.Linear.weight`, and conv weights keep PyTorch `[Cout,Cin/G,K]` /
//! `[Cin,Cout/G,K]` layout that `conv1d`/`convtr1d` already expect.

use std::collections::HashMap;
use std::path::Path;

/// Clamp epsilon for `EuclideanCodebook` (the reference's default `epsilon`).
const CODEBOOK_EPS: f32 = 1e-5;

/// Whether an encoder RVQ codebook layer (e.g.
/// `quantizer.acoustic_residual_vector_quantizer.layers.7.codebook.embed_sum`)
/// falls within the kept range: all semantic layers, but only the first
/// `n_aco_keep` acoustic layers (`encoder_valid_num_quantizers - num_semantic`).
fn quant_layer_in_range(name: &str, n_aco_keep: usize) -> bool {
    let idx = name
        .split("layers.")
        .nth(1)
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    if name.contains("semantic_residual_vector_quantizer") {
        true
    } else {
        idx < n_aco_keep
    }
}

/// Import `<ckpt_dir>/config.json` + `<ckpt_dir>/model.safetensors` into the
/// brain checkpoint `out_path`. Fails loudly (never writes a partial file) if any
/// `decoder.*` tensor is left unaccounted for.
pub fn import(ckpt_dir: &str, out_path: &str) -> Result<(), String> {
    let dir = Path::new(ckpt_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&cfg_json).map_err(|e| format!("parse config.json: {e}"))?;

    let st_path = dir.join("model.safetensors");
    let tensors = checkpoint::safetensors::read(
        st_path.to_str().ok_or("non-utf8 checkpoint path")?,
    )?;

    // How many encoder quantizers to keep (1 semantic + 15 acoustic by default).
    let valid_q = config["encoder_valid_num_quantizers"].as_u64().unwrap_or(16) as usize;
    let n_sem = config["encoder_config"]["num_semantic_quantizers"].as_u64().unwrap_or(1) as usize;
    let n_aco_keep = valid_q.saturating_sub(n_sem);

    // name -> (shape, data) for kept tensors, with codebook pairs held aside.
    // Codebook pairs are namespaced (`d:`/`e:`) so decoder and encoder codebooks
    // never collide while sharing the same collapse pass.
    let mut out: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut emb_sum: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut cluster: HashMap<String, Vec<f32>> = HashMap::new();
    let mut decoder_seen = 0usize;
    let mut encoder_seen = 0usize;
    let mut dropped = 0usize;

    for t in tensors {
        if let Some(name) = t.name.strip_prefix("decoder.") {
            decoder_seen += 1;
            if let Some(parent) = name.strip_suffix("._codebook.embedding_sum") {
                emb_sum.insert(format!("d:{parent}"), (t.shape, t.data));
            } else if let Some(parent) = name.strip_suffix("._codebook.cluster_usage") {
                cluster.insert(format!("d:{parent}"), t.data);
            } else if name.starts_with("quantizer.") && name.ends_with("input_proj.weight") {
                dropped += 1; // encode-side projection, unused on decode
            } else if out.insert(name.to_string(), (t.shape, t.data)).is_some() {
                return Err(format!("duplicate decoder tensor {name}"));
            }
            continue;
        }
        // ---- encoder.* (HuggingFace MimiModel) — the encode path ----
        let Some(name) = t.name.strip_prefix("encoder.") else {
            continue; // neither decoder nor encoder — ignore
        };
        encoder_seen += 1;
        // Keep only the first `valid_q` RVQ codebooks; drop the rest of the
        // 32-deep stack, the `initialized` flags, and the decode-side output_proj.
        if name.contains("residual_vector_quantizer.layers.") {
            let keep = quant_layer_in_range(name, n_aco_keep);
            if let Some(parent) = name.strip_suffix(".codebook.embed_sum") {
                if keep { emb_sum.insert(format!("e:encoder.{parent}"), (t.shape, t.data)); }
                else { dropped += 1; }
            } else if let Some(parent) = name.strip_suffix(".codebook.cluster_usage") {
                if keep { cluster.insert(format!("e:encoder.{parent}"), t.data); }
                else { dropped += 1; }
            } else {
                dropped += 1; // `.codebook.initialized` and any other per-layer buffer
            }
            continue;
        }
        if name.ends_with("output_proj.weight") {
            dropped += 1; // encoder RVQ decode-side projection, unused on encode
            continue;
        }
        let key = format!("encoder.{name}");
        if out.insert(key.clone(), (t.shape, t.data)).is_some() {
            return Err(format!("duplicate encoder tensor {key}"));
        }
    }

    // Collapse each codebook into its embedding table (`embed = embed_sum /
    // clamp(cluster_usage, eps)`), then re-key without the `d:`/`e:` namespace.
    if emb_sum.len() != cluster.len() {
        return Err(format!(
            "codebook pairing mismatch: {} embedding_sum vs {} cluster_usage",
            emb_sum.len(),
            cluster.len()
        ));
    }
    for (parent, (shape, sum)) in emb_sum {
        let usage = cluster
            .remove(&parent)
            .ok_or_else(|| format!("codebook {parent}: missing cluster_usage"))?;
        let (bins, dim) = (shape[0], shape[1]);
        if usage.len() != bins {
            return Err(format!("codebook {parent}: usage {} != bins {bins}", usage.len()));
        }
        let mut table = vec![0.0f32; bins * dim];
        for b in 0..bins {
            let denom = usage[b].max(CODEBOOK_EPS);
            for c in 0..dim {
                table[b * dim + c] = sum[b * dim + c] / denom;
            }
        }
        let bare = parent.split_once(':').map(|(_, r)| r).unwrap_or(&parent);
        out.insert(format!("{bare}.table"), (vec![bins, dim], table));
    }

    if decoder_seen == 0 {
        return Err("no decoder.* tensors found in checkpoint".to_string());
    }

    // Ordered tensor list for the checkpoint (sorted for determinism).
    let mut names: Vec<String> = out.keys().cloned().collect();
    names.sort();
    let saved: Vec<(String, Vec<u64>, Vec<f32>)> = names
        .into_iter()
        .map(|n| {
            let (shape, data) = out.remove(&n).unwrap();
            (n, shape.iter().map(|&x| x as u64).collect(), data)
        })
        .collect();

    checkpoint::save(out_path, config, &saved);
    eprintln!(
        "codec import: {decoder_seen} decoder + {encoder_seen} encoder tensors -> {} params \
         in {out_path} ({dropped} dropped, codebooks collapsed)",
        saved.len()
    );
    Ok(())
}
