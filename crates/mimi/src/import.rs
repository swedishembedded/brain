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

use std::collections::{HashMap, HashSet};
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

/// Where one HF source tensor ends up: a plain 1:1 passthrough (with its final
/// output name), one half of a codebook pair to accumulate (namespaced `d:`/`e:`
/// parent key — decoder and encoder codebooks never collide), or dropped.
/// Single source of truth for both the header-only planning pass and the real
/// streaming pass, so the two can never disagree about a tensor's fate.
enum Slot {
    Out(String),
    EmbSum(String),
    Cluster(String),
    Drop,
}

fn classify(full_name: &str, n_aco_keep: usize) -> Slot {
    if let Some(name) = full_name.strip_prefix("decoder.") {
        if let Some(parent) = name.strip_suffix("._codebook.embedding_sum") {
            return Slot::EmbSum(format!("d:{parent}"));
        }
        if let Some(parent) = name.strip_suffix("._codebook.cluster_usage") {
            return Slot::Cluster(format!("d:{parent}"));
        }
        if name.starts_with("quantizer.") && name.ends_with("input_proj.weight") {
            return Slot::Drop; // encode-side projection, unused on decode
        }
        return Slot::Out(name.to_string());
    }
    // ---- encoder.* (HuggingFace MimiModel) — the encode path ----
    let Some(name) = full_name.strip_prefix("encoder.") else {
        return Slot::Drop; // neither decoder nor encoder — ignore
    };
    // Keep only the first `valid_q` RVQ codebooks; drop the rest of the 32-deep
    // stack, the `initialized` flags, and the decode-side output_proj.
    if name.contains("residual_vector_quantizer.layers.") {
        let keep = quant_layer_in_range(name, n_aco_keep);
        if let Some(parent) = name.strip_suffix(".codebook.embed_sum") {
            return if keep { Slot::EmbSum(format!("e:encoder.{parent}")) } else { Slot::Drop };
        }
        if let Some(parent) = name.strip_suffix(".codebook.cluster_usage") {
            return if keep { Slot::Cluster(format!("e:encoder.{parent}")) } else { Slot::Drop };
        }
        return Slot::Drop; // `.codebook.initialized` and any other per-layer buffer
    }
    if name.ends_with("output_proj.weight") {
        return Slot::Drop; // encoder RVQ decode-side projection, unused on encode
    }
    Slot::Out(format!("encoder.{name}"))
}

/// Import `<ckpt_dir>/config.json` + `<ckpt_dir>/model.safetensors` into the
/// brain checkpoint `out_path`. Fails loudly (never writes a partial file) if any
/// `decoder.*` tensor is left unaccounted for. Streams: a header-only pass
/// plans every output tensor (names + shapes, no data), then a single data pass
/// writes passthrough tensors straight through and accumulates ONLY the small
/// codebook `embed_sum`/`cluster_usage` halves (bounded by codebook count ×
/// bins × dim — a handful of small codec tables, not the whole model) until
/// both halves of a pair are seen and can be collapsed.
pub fn import(ckpt_dir: &str, out_path: &str) -> Result<(), String> {
    let dir = Path::new(ckpt_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&cfg_json).map_err(|e| format!("parse config.json: {e}"))?;

    let st_path = dir.join("model.safetensors");
    let reader = checkpoint::weightio::WeightReader::open(st_path.to_str().ok_or("non-utf8 checkpoint path")?)
        .map_err(|e| format!("import: opening checkpoint: {e}"))?;

    // How many encoder quantizers to keep (1 semantic + 15 acoustic by default).
    let valid_q = config["encoder_valid_num_quantizers"].as_u64().unwrap_or(16) as usize;
    let n_sem = config["encoder_config"]["num_semantic_quantizers"].as_u64().unwrap_or(1) as usize;
    let n_aco_keep = valid_q.saturating_sub(n_sem);

    // ---- phase 1: header-only pass — build the output plan, no tensor data ----
    let mut plan: Vec<(String, Vec<u64>)> = Vec::new();
    let mut plan_names: HashSet<String> = HashSet::new();
    let mut emb_shape: HashMap<String, Vec<u64>> = HashMap::new();
    let mut cluster_present: HashSet<String> = HashSet::new();
    let mut decoder_seen = 0usize;
    let mut encoder_seen = 0usize;
    let mut dropped = 0usize;

    for full_name in reader.names() {
        if full_name.starts_with("decoder.") {
            decoder_seen += 1;
        } else if full_name.starts_with("encoder.") {
            encoder_seen += 1;
        } else {
            continue; // neither decoder nor encoder — ignore
        }
        match classify(full_name, n_aco_keep) {
            Slot::Out(out_name) => {
                if !plan_names.insert(out_name.clone()) {
                    return Err(format!("duplicate tensor {out_name}"));
                }
                let shape = reader
                    .shape(full_name)
                    .ok_or_else(|| format!("import: missing shape for {full_name}"))?
                    .to_vec();
                plan.push((out_name, shape));
            }
            Slot::EmbSum(key) => {
                let shape = reader
                    .shape(full_name)
                    .ok_or_else(|| format!("import: missing shape for {full_name}"))?
                    .to_vec();
                emb_shape.insert(key, shape);
            }
            Slot::Cluster(key) => {
                cluster_present.insert(key);
            }
            Slot::Drop => dropped += 1,
        }
    }
    if decoder_seen == 0 {
        return Err("no decoder.* tensors found in checkpoint".to_string());
    }
    if emb_shape.len() != cluster_present.len() {
        return Err(format!(
            "codebook pairing mismatch: {} embedding_sum vs {} cluster_usage",
            emb_shape.len(),
            cluster_present.len()
        ));
    }
    for (key, shape) in &emb_shape {
        if !cluster_present.contains(key) {
            return Err(format!("codebook {key}: missing cluster_usage"));
        }
        let bare = key.split_once(':').map(|(_, r)| r).unwrap_or(key.as_str());
        let out_name = format!("{bare}.table");
        if !plan_names.insert(out_name.clone()) {
            return Err(format!("duplicate tensor {out_name}"));
        }
        plan.push((out_name, shape.clone()));
    }

    let mut writer = checkpoint::weightio::StWriter::create(out_path, &plan, &config, None)
        .map_err(|e| format!("import: creating output: {e}"))?;

    // ---- phase 2: the real streaming pass — one tensor at a time ----
    // Codebook halves are the ONLY thing held aside (bounded: a handful of
    // small codec tables), everything else writes straight through.
    let mut emb_sum: HashMap<String, (Vec<u64>, Vec<f32>)> = HashMap::new();
    let mut cluster: HashMap<String, Vec<f32>> = HashMap::new();
    let mut err: Option<String> = None;
    reader.for_each(|full_name, shape, data| {
        if err.is_some() {
            return;
        }
        if !full_name.starts_with("decoder.") && !full_name.starts_with("encoder.") {
            return;
        }
        match classify(full_name, n_aco_keep) {
            Slot::Out(out_name) => {
                if let Err(e) = writer.write(&out_name, &data) {
                    err = Some(format!("import: {e}"));
                }
            }
            Slot::EmbSum(key) => {
                emb_sum.insert(key, (shape.to_vec(), data));
            }
            Slot::Cluster(key) => {
                cluster.insert(key, data);
            }
            Slot::Drop => {}
        }
    });
    if let Some(e) = err {
        return Err(e);
    }

    // Collapse each codebook into its embedding table (`table = embed_sum /
    // clamp(cluster_usage, eps)`) now that both halves are in hand.
    for (key, (shape, sum)) in emb_sum {
        let usage = cluster.remove(&key).ok_or_else(|| format!("codebook {key}: missing cluster_usage"))?;
        let (bins, dim) = (shape[0] as usize, shape[1] as usize);
        if usage.len() != bins {
            return Err(format!("codebook {key}: usage {} != bins {bins}", usage.len()));
        }
        let mut table = vec![0.0f32; bins * dim];
        for b in 0..bins {
            let denom = usage[b].max(CODEBOOK_EPS);
            for c in 0..dim {
                table[b * dim + c] = sum[b * dim + c] / denom;
            }
        }
        let bare = key.split_once(':').map(|(_, r)| r).unwrap_or(&key);
        writer.write(&format!("{bare}.table"), &table).map_err(|e| format!("import: {e}"))?;
    }

    writer.finish().map_err(|e| format!("import: {e}"))?;
    eprintln!(
        "codec import: {decoder_seen} decoder + {encoder_seen} encoder tensors -> {} params \
         in {out_path} ({dropped} dropped, codebooks collapsed)",
        plan.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streaming `import` with TWO decoder codebooks + TWO encoder RVQ
    /// codebooks (one semantic, one acoustic, plus one out-of-range acoustic
    /// layer that must be dropped), each with distinct bins/dim/values — proves
    /// the fan-in collapse pairs `embed_sum`/`cluster_usage` correctly and never
    /// aliases one codebook's values onto another's table, alongside plain
    /// passthrough tensors and every documented drop case.
    #[test]
    fn streaming_import_collapses_codebooks_without_cross_aliasing() {
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("codec-import-src-{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        let config = serde_json::json!({
            "encoder_valid_num_quantizers": 2,
            "encoder_config": {"num_semantic_quantizers": 1},
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();

        let plan: Vec<(String, Vec<u64>)> = vec![
            ("decoder.foo.weight".to_string(), vec![3]),
            ("decoder.quantizer.input_proj.weight".to_string(), vec![2]),
            ("decoder.quantizer.rvq.layers.0._codebook.embedding_sum".to_string(), vec![3, 2]),
            ("decoder.quantizer.rvq.layers.0._codebook.cluster_usage".to_string(), vec![3]),
            ("decoder.quantizer.rvq.layers.1._codebook.embedding_sum".to_string(), vec![2, 3]),
            ("decoder.quantizer.rvq.layers.1._codebook.cluster_usage".to_string(), vec![2]),
            ("encoder.bar.weight".to_string(), vec![2]),
            (
                "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum".to_string(),
                vec![2, 4],
            ),
            (
                "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage".to_string(),
                vec![2],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.embed_sum".to_string(),
                vec![3, 1],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.cluster_usage".to_string(),
                vec![3],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.initialized".to_string(),
                vec![1],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.embed_sum".to_string(),
                vec![3, 1],
            ),
            (
                "encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.cluster_usage".to_string(),
                vec![3],
            ),
            ("encoder.quantizer.some.output_proj.weight".to_string(), vec![2]),
        ];
        let src = dir.join("model.safetensors");
        let mut w =
            checkpoint::weightio::StWriter::create(src.to_str().unwrap(), &plan, &serde_json::Value::Null, None)
                .unwrap();
        w.write("decoder.foo.weight", &[7.0, 8.0, 9.0]).unwrap();
        w.write("decoder.quantizer.input_proj.weight", &[0.0, 0.0]).unwrap();
        w.write("decoder.quantizer.rvq.layers.0._codebook.embedding_sum", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        w.write("decoder.quantizer.rvq.layers.0._codebook.cluster_usage", &[1.0, 2.0, 5.0]).unwrap();
        w.write(
            "decoder.quantizer.rvq.layers.1._codebook.embedding_sum",
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .unwrap();
        w.write("decoder.quantizer.rvq.layers.1._codebook.cluster_usage", &[2.0, 4.0]).unwrap();
        w.write("encoder.bar.weight", &[10.0, 11.0]).unwrap();
        w.write(
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
            &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        )
        .unwrap();
        w.write(
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage",
            &[1.0, 4.0],
        )
        .unwrap();
        w.write(
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.embed_sum",
            &[9.0, 18.0, 27.0],
        )
        .unwrap();
        w.write("encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.cluster_usage", &[3.0, 6.0, 9.0])
            .unwrap();
        w.write("encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.initialized", &[1.0]).unwrap();
        w.write(
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.embed_sum",
            &[100.0, 200.0, 300.0],
        )
        .unwrap();
        w.write("encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.codebook.cluster_usage", &[1.0, 1.0, 1.0])
            .unwrap();
        w.write("encoder.quantizer.some.output_proj.weight", &[0.0, 0.0]).unwrap();
        w.finish().unwrap();

        let out = std::env::temp_dir().join(format!("codec-import-out-{pid}.safetensors"));
        import(dir.to_str().unwrap(), out.to_str().unwrap()).unwrap();

        let reader = checkpoint::weightio::WeightReader::open(out.to_str().unwrap()).unwrap();
        // passthrough tensors, prefix handling preserved.
        assert_eq!(reader.tensor("foo.weight").unwrap(), vec![7.0, 8.0, 9.0]);
        assert_eq!(reader.tensor("encoder.bar.weight").unwrap(), vec![10.0, 11.0]);

        // codebook collapse: embed_sum / clamp(cluster_usage, eps), hand-computed.
        assert_eq!(
            reader.tensor("quantizer.rvq.layers.0.table").unwrap(),
            vec![1.0, 2.0, 1.5, 2.0, 1.0, 1.2]
        );
        assert_eq!(
            reader.tensor("quantizer.rvq.layers.1.table").unwrap(),
            vec![5.0, 10.0, 15.0, 10.0, 12.5, 15.0]
        );
        assert_eq!(
            reader.tensor("encoder.quantizer.semantic_residual_vector_quantizer.layers.0.table").unwrap(),
            vec![1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
        );
        assert_eq!(
            reader.tensor("encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.table").unwrap(),
            vec![3.0, 3.0, 3.0]
        );

        // dropped: the out-of-range acoustic layer never produces a table (no
        // aliasing with layer 0's very different values), nor do input_proj /
        // output_proj / `initialized`.
        assert!(reader.tensor("encoder.quantizer.acoustic_residual_vector_quantizer.layers.1.table").is_none());
        assert!(reader.tensor("quantizer.input_proj.weight").is_none());
        assert!(reader.tensor("encoder.quantizer.some.output_proj.weight").is_none());
        assert!(reader
            .tensor("encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.initialized")
            .is_none());

        assert_eq!(reader.names().count(), 6, "2 passthrough + 4 collapsed tables");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&out).ok();
    }
}
