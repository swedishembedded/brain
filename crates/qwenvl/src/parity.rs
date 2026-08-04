// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Partial-depth real-weight parity for the Qwen3-VL-4B text decoder.
//!
//! The full 36-layer decoder doesn't fit in RAM (~32 GB in brain's load path), so we
//! validate the first `N` *real* Qwen3 blocks end-to-end (embed → N blocks → final
//! norm → tied head → logits) against `transformers` built the same way — a genuine
//! Qwen3 block (QK-norm, head_dim-128 GQA 32/8, SwiGLU, θ=5e6) on real weights. The
//! checkpoint shards are read one at a time and filtered to just these tensors so the
//! footprint stays small. Requires the dump from `tools/goldens/qwenvl_decoder_dump_reference.py`.

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)

#[allow(dead_code)]
use brain_testutil::testdata;
use brain_testutil::model_dir;
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}
    use std::collections::HashMap;

    use crate::config::Qwen3VlConfig;
    use crate::import::map_decoder;
    use qwen::Qwen;

    const N: u32 = 4; // must match the dump

    fn read_f32(p: impl AsRef<std::path::Path>) -> Option<Vec<f32>> {
        Some(std::fs::read(p).ok()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// The layer index of a `blocks.<L>.…` key, else `None`.
    fn layer_of(name: &str) -> Option<u32> {
        name.strip_prefix("blocks.")?.split_once('.')?.0.parse().ok()
    }

    #[test]
    fn qwenvl_decoder_partial_depth_matches_hf() {
        let DIR = model_dir("Qwen/Qwen3-VL-4B-Instruct").unwrap_or_default();
        let REF = testdata("vl/parity/qwenvl_dec_ref.bin");
        let TOK = testdata("vl/parity/qwenvl_dec_tokens.bin");
        let (Some(ref_logits), Some(tok_raw)) = (read_f32(&REF), std::fs::read(TOK).ok()) else {
            eprintln!("skip: Qwen3-VL decoder reference not present (run tools/goldens/qwenvl_decoder_dump_reference.py)");
            return;
        };
        std::env::set_var("BRAIN_DEVICE", "cpu");
        let tokens: Vec<u32> = tok_raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();

        // Stream the shards one at a time, keeping only embed/norm + the first N blocks.
        let mut shards: Vec<_> = match std::fs::read_dir(DIR) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect(),
            Err(_) => {
                eprintln!("skip: Qwen3-VL checkpoint not present");
                return;
            }
        };
        shards.sort();
        let mut init: HashMap<String, Vec<f32>> = HashMap::new();
        for shard in &shards {
            let Ok(tensors) = checkpoint::safetensors::read(shard.to_str().unwrap()) else { continue };
            for t in tensors {
                if let Some(name) = map_decoder(&t.name) {
                    if layer_of(&name).is_none_or(|l| l < N) {
                        init.insert(name, t.data);
                    }
                }
            }
        }

        let mut cfg = Qwen3VlConfig::qwen3_vl_4b().text;
        cfg.n_layers = N;
        for (name, _) in cfg.param_list() {
            assert!(init.contains_key(&name), "decoder param not streamed: {name}");
        }

        let t = tokens.len() as u32;
        let vocab = cfg.vocab as usize;
        let qwen = Qwen::new(cfg, 1, t, &init);
        let logits = qwen.logits_all(&tokens);
        assert_eq!(logits.len(), ref_logits.len(), "logit shape mismatch");

        let (mut max_abs, mut sum_abs) = (0.0f32, 0.0f64);
        for p in 0..tokens.len() {
            let (row, rref) = (&logits[p * vocab..(p + 1) * vocab], &ref_logits[p * vocab..(p + 1) * vocab]);
            let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            assert_eq!(am(row), am(rref), "argmax disagrees at position {p}");
            for (a, b) in row.iter().zip(rref) {
                max_abs = max_abs.max((a - b).abs());
                sum_abs += (a - b).abs() as f64;
            }
        }
        eprintln!("Qwen3-VL {N}-layer parity: mean|Δ|={:.4e} max|Δ|={:.4e}", sum_abs / logits.len() as f64, max_abs);
        assert!(max_abs < 5e-3, "Qwen3 blocks diverge from HF: max|Δ|={max_abs}");
    }
}
