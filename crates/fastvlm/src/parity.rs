// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-weight parity: brain's Qwen2 decoder vs the HuggingFace reference on the
//! actual FastVLM-0.5B checkpoint. The decoder is an unmodified Qwen2, so this
//! validates the shared decoder backbone (embed → 24 GQA+SwiGLU layers → tied head)
//! against `transformers` on real bf16 weights — the strongest correctness signal
//! short of a full end-to-end run.
//!
//! Requires the checkpoint and the reference dump produced by
//! `tools/fastvlm_decoder_dump_reference.py`; the test skips (passes) when absent.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::config::FastVlmConfig;
    use crate::import::map_decoder;
    use qwen::Qwen;

    const CKPT: &str = "/data/workspace/resources/vl/fastvlm/hf/FastVLM-0.5B/model.safetensors";
    const REF: &str = "/data/workspace/resources/vl/parity/fastvlm_dec_ref.bin";
    const TOK: &str = "/data/workspace/resources/vl/parity/fastvlm_dec_tokens.bin";

    fn read_f32(path: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(path).ok()?;
        Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    #[test]
    fn fastvlm_decoder_logits_match_hf_reference() {
        let (Some(ref_logits), Some(tok_raw)) = (read_f32(REF), std::fs::read(TOK).ok()) else {
            eprintln!("skip: FastVLM decoder reference dump not present (run tools/fastvlm_decoder_dump_reference.py)");
            return;
        };
        // The tied vocab table (151936×896×4 ≈ 544 MB) exceeds a typical GPU storage-
        // buffer binding limit, so run the decoder on the CPU-JIT backend.
        std::env::set_var("BRAIN_DEVICE", "cpu");
        let Ok(tensors) = checkpoint::safetensors::read(CKPT) else {
            eprintln!("skip: FastVLM checkpoint not present");
            return;
        };
        let tokens: Vec<u32> = tok_raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect();

        // Build the decoder weight map from the checkpoint via the import mapping.
        let mut init: HashMap<String, Vec<f32>> = HashMap::new();
        for t in tensors {
            if let Some(name) = map_decoder(&t.name) {
                init.insert(name, t.data);
            }
        }
        let cfg = FastVlmConfig::fastvlm_0_5b().decoder;
        // Every decoder parameter must have been imported.
        for (name, _) in cfg.param_list() {
            assert!(init.contains_key(&name), "decoder param not imported: {name}");
        }

        let t = tokens.len() as u32;
        let vocab = cfg.vocab as usize;
        let qwen = Qwen::new(cfg, 1, t, &init);
        let logits = qwen.logits_all(&tokens); // [T, vocab] flat
        assert_eq!(logits.len(), ref_logits.len(), "logit tensor shape mismatch");

        // Per-position: the argmax (predicted next token) must agree, and the logits
        // must match within fp32-vs-fp32 tolerance (both compute in f32).
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for p in 0..tokens.len() {
            let row = &logits[p * vocab..(p + 1) * vocab];
            let rref = &ref_logits[p * vocab..(p + 1) * vocab];
            let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            assert_eq!(am(row), am(rref), "argmax token disagrees at position {p}");
            for (a, b) in row.iter().zip(rref) {
                let e = (a - b).abs();
                max_abs = max_abs.max(e);
                sum_abs += e as f64;
            }
        }
        let mean_abs = sum_abs / logits.len() as f64;
        eprintln!("FastVLM decoder parity: mean|Δ|={mean_abs:.4e} max|Δ|={max_abs:.4e} over {} logits", logits.len());
        // Observed on the real weights: mean ~7e-6, max ~7e-5 — pure fp32
        // reassociation noise vs the bf16→fp32 reference. Thresholds keep ~50× headroom.
        assert!(max_abs < 5e-3, "decoder logits diverge from HF reference: max|Δ|={max_abs}");
        assert!(mean_abs < 1e-4, "decoder logits mean drift too high: {mean_abs}");
    }
}
