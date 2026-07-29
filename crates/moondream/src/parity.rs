// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Partial-depth real-weight parity for the Moondream 3 text decoder.
//!
//! The full 24-layer 9 B MoE model far exceeds RAM, so this validates the first 4
//! *real* DENSE blocks (layers 0-3, before `moe.start_layer=4`) end-to-end: the
//! bespoke parallel attn+MLP block with per-head tau temperature and partial RoPE —
//! an architecture with no standard-HF analogue, so this is the strongest real-weight
//! check of the novel decoder. brain builds the identical 4-layer decoder from the
//! streamed weights and compares logits to the reference from
//! `tools/moondream_decoder_dump_reference.py` (causal mask, i.e. `prefix_attn=1`).

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)

#[allow(dead_code)]
fn testdata(rel: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/{rel}")
}
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}
    use std::collections::HashMap;

    use gpu_core::Gpu;

    use crate::config::MoondreamConfig;
    use crate::decoder::{pipelines, MoondreamBlock, MoondreamDecoder};
    use crate::import::{map_text, TextTarget};

    const N: u32 = 4; // dense blocks, must match the dump

    fn read_f32(p: impl AsRef<std::path::Path>) -> Option<Vec<f32>> {
        Some(std::fs::read(p).ok()?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn layer_of(name: &str) -> Option<u32> {
        name.strip_prefix("blocks.")?.split_once('.')?.0.parse().ok()
    }

    #[test]
    fn moondream_decoder_partial_depth_matches_hf() {
        let DIR = testdata("vl/moondream3/hf/moondream3-preview");
        let REF = testdata("vl/parity/moondream_dec_ref.bin");
        let TOK = testdata("vl/parity/moondream_dec_tokens.bin");
        let (Some(ref_logits), Some(tok_raw)) = (read_f32(&REF), std::fs::read(TOK).ok()) else {
            eprintln!("skip: Moondream decoder reference not present (run tools/moondream_decoder_dump_reference.py)");
            return;
        };
        let Ok(rd) = std::fs::read_dir(DIR) else {
            eprintln!("skip: Moondream checkpoint not present");
            return;
        };
        let tokens: Vec<u32> = tok_raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();
        let cfg = MoondreamConfig::preview();

        // Stream the shards; keep only the first N (dense) blocks + tok/post_ln/lm_head.
        let mut shards: Vec<_> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect();
        shards.sort();
        let mut w: HashMap<String, Vec<f32>> = HashMap::new();
        for shard in &shards {
            let Ok(tensors) = checkpoint::safetensors::read(shard.to_str().unwrap()) else { continue };
            for t in tensors {
                if let Some(TextTarget::Key(name)) = map_text(&t.name, &cfg) {
                    if layer_of(&name).is_none_or(|l| l < N) {
                        w.insert(name, t.data);
                    }
                }
            }
        }

        // Build the 4-layer dense decoder (tau on, causal via prefix_attn=1, no image).
        let (t, d, v) = (tokens.len() as u32, cfg.dim, cfg.vocab);
        let gpu = Gpu::new_cpu(pipelines());
        let blocks: Vec<MoondreamBlock> = (0..N)
            .map(|l| {
                let bw: HashMap<String, Vec<f32>> = w.iter().filter_map(|(k, val)| k.strip_prefix(&format!("blocks.{l}.")).map(|s| (s.to_string(), val.clone()))).collect();
                MoondreamBlock::new(&gpu, &bw, t, d, cfg.n_heads, cfg.head_dim, cfg.ff_dim, 1, cfg.rot_dim, cfg.rope_theta)
            })
            .collect();
        let dec = MoondreamDecoder::new(&gpu, &w, blocks, t, d, v, 0);
        let logits = dec.logits_all(&tokens, &[]);
        assert_eq!(logits.len(), ref_logits.len(), "logit shape mismatch");

        let (mut max_abs, mut sum_abs) = (0.0f32, 0.0f64);
        let vocab = v as usize;
        let am = |x: &[f32]| x.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let mut argmax_ok = true;
        for p in 0..tokens.len() {
            let (row, rref) = (&logits[p * vocab..(p + 1) * vocab], &ref_logits[p * vocab..(p + 1) * vocab]);
            argmax_ok &= am(row) == am(rref);
            for (a, b) in row.iter().zip(rref) {
                max_abs = max_abs.max((a - b).abs());
                sum_abs += (a - b).abs() as f64;
            }
        }
        let per_pos: Vec<f32> = (0..tokens.len())
            .map(|p| {
                let (r, rf) = (&logits[p * vocab..(p + 1) * vocab], &ref_logits[p * vocab..(p + 1) * vocab]);
                r.iter().zip(rf).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
            })
            .collect();
        eprintln!("Moondream {N}-layer dense parity: mean|Δ|={:.4e} max|Δ|={:.4e} argmax_ok={argmax_ok}\n  per-position max|Δ|: {per_pos:?}", sum_abs / logits.len() as f64, max_abs);
        assert!(max_abs < 5e-3 && argmax_ok, "Moondream blocks diverge from HF: max|Δ|={max_abs}");
    }
}
