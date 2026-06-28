// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the official `Qwen3-TTS-Tokenizer-12Hz` safetensors checkpoint into a
//! brain `.weights` container, decode-path only.
//!
//! The decoder lives under the `decoder.*` prefix (271 tensors); the encoder
//! (`encoder.*`, 225 tensors) is ignored — this crate implements decode. We do a
//! near 1:1 name remap (just stripping the `decoder.` prefix) with two
//! transforms:
//!   * each Euclidean codebook is collapsed at import time from its two stored
//!     tensors `embedding_sum [bins,dim]` + `cluster_usage [bins]` into the
//!     usable embedding table `table = embedding_sum / clamp(cluster_usage, eps)`
//!     (matches `EuclideanCodebook.decode`, eps = 1e-5);
//!   * the quantizers' `input_proj` (encode-side projection) is dropped — the
//!     decode path never uses it.
//! No tensor is transposed: brain `matmul` is `x @ Wᵀ` with `W:[out,in]`, exactly
//! `nn.Linear.weight`, and conv weights keep PyTorch `[Cout,Cin/G,K]` /
//! `[Cin,Cout/G,K]` layout that `conv1d`/`convtr1d` already expect.

use std::collections::HashMap;
use std::path::Path;

/// Clamp epsilon for `EuclideanCodebook` (the reference's default `epsilon`).
const CODEBOOK_EPS: f32 = 1e-5;

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

    // name -> (shape, data) for decoder tensors, with codebook pairs held aside.
    let mut out: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut emb_sum: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut cluster: HashMap<String, Vec<f32>> = HashMap::new();
    let mut decoder_seen = 0usize;
    let mut dropped = 0usize;

    for t in tensors {
        let Some(name) = t.name.strip_prefix("decoder.") else {
            continue; // encoder.* — not part of the decode path
        };
        decoder_seen += 1;
        if let Some(parent) = name.strip_suffix("._codebook.embedding_sum") {
            emb_sum.insert(parent.to_string(), (t.shape, t.data));
        } else if let Some(parent) = name.strip_suffix("._codebook.cluster_usage") {
            cluster.insert(parent.to_string(), t.data);
        } else if name.starts_with("quantizer.") && name.ends_with("input_proj.weight") {
            dropped += 1; // encode-side projection, unused on decode
        } else if out.insert(name.to_string(), (t.shape, t.data)).is_some() {
            return Err(format!("duplicate decoder tensor {name}"));
        }
    }

    // Collapse each codebook into its decode-time embedding table.
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
        out.insert(format!("{parent}.table"), (vec![bins, dim], table));
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
        "codec import: {decoder_seen} decoder tensors -> {} params in {out_path} \
         ({dropped} input_proj dropped, codebooks collapsed)",
        saved.len()
    );
    Ok(())
}
