// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-weight parity: brain's Qwen2 decoder vs the HuggingFace reference on the
//! actual FastVLM-0.5B checkpoint. The decoder is an unmodified Qwen2, so this
//! validates the shared decoder backbone (embed → 24 GQA+SwiGLU layers → tied head)
//! against `transformers` on real bf16 weights — the strongest correctness signal
//! short of a full end-to-end run.
//!
//! Requires the checkpoint and the reference dump produced by
//! `tools/goldens/fastvlm_decoder_dump_reference.py`; the test skips (passes) when absent.

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

    use crate::config::FastVlmConfig;
    use crate::import::map_decoder;
    use qwen::Qwen;


    fn read_i32(path: impl AsRef<std::path::Path>) -> Option<Vec<u32>> {
        let b = std::fs::read(path).ok()?;
        Some(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32).collect())
    }

    fn read_f32(path: impl AsRef<std::path::Path>) -> Option<Vec<f32>> {
        let b = std::fs::read(path).ok()?;
        Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    #[test]
    fn fastvlm_decoder_logits_match_hf_reference() {
        let CKPT = format!("{}/model.safetensors", model_dir("apple/FastVLM-0.5B").unwrap_or_default());
        let REF = testdata("vl/parity/fastvlm_dec_ref.bin");
        let TOK = testdata("vl/parity/fastvlm_dec_tokens.bin");
        let GEN = testdata("vl/parity/fastvlm_dec_gen.bin");
        let (Some(ref_logits), Some(tok_raw)) = (read_f32(&REF), std::fs::read(TOK).ok()) else {
            eprintln!("skip: FastVLM decoder reference dump not present (run tools/goldens/fastvlm_decoder_dump_reference.py)");
            return;
        };
        // The tied vocab table (151936×896×4 ≈ 544 MB) exceeds a typical GPU storage-
        // buffer binding limit, so run the decoder on the CPU-JIT backend.
        std::env::set_var("BRAIN_DEVICE", "cpu");
        let Ok(tensors) = checkpoint::safetensors::read(&CKPT) else {
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

        let prompt_len = tokens.len();
        let vocab = cfg.vocab as usize;
        // Build with headroom so we can greedy-decode past the prompt.
        let gen_ref = read_i32(GEN).unwrap_or_default();
        let t_max = (prompt_len + gen_ref.len().max(1) + 1) as u32;
        let qwen = Qwen::new(cfg, 1, t_max, &init);
        let logits = qwen.logits_all(&tokens); // [prompt_len, vocab] flat
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

        // Greedy generation: brain must argmax-decode the SAME tokens as HF. This
        // proves the whole decode loop (embed → 24 layers → tied head → argmax) on
        // the real weights — for "Name three primary colors." the reference answers
        // "Red, Blue, and Yellow.".
        if !gen_ref.is_empty() {
            let mut seq = tokens.clone();
            let mut brain_gen = Vec::new();
            for _ in 0..gen_ref.len() {
                let lg = qwen.logits_all(&seq);
                let last = &lg[(seq.len() - 1) * vocab..seq.len() * vocab];
                let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
                brain_gen.push(next);
                seq.push(next);
            }
            eprintln!("brain greedy gen: {brain_gen:?}\nHF   greedy gen: {gen_ref:?}");
            assert_eq!(brain_gen, gen_ref, "brain greedy decode diverges from the HF reference");
        }
    }


    #[test]
    fn fastvlm_vision_tower_matches_hf() {
        let CKPT = format!("{}/model.safetensors", model_dir("apple/FastVLM-0.5B").unwrap_or_default());
        let VIS_PX = testdata("vl/parity/fastvlm_vis_px.bin");
        let VIS_FEAT = testdata("vl/parity/fastvlm_vis_feat.bin");
        // brain's OWN FastViTHD (Encoder::mobileclip_l) vs the HF mobileclip tower on
        // the real weights: same preprocessed pixels → the [256, 3072] features must
        // match (fully-in-brain vision, closing the loop for a native image caption).
        let (Some(px), Some(feat)) = (read_f32(VIS_PX), read_f32(VIS_FEAT)) else {
            eprintln!("skip: vision reference not present (run tools/goldens/fastvlm_vision_dump_reference.py)");
            return;
        };
        let Ok(tensors) = checkpoint::safetensors::read(&CKPT) else {
            eprintln!("skip: FastVLM checkpoint not present");
            return;
        };
        std::env::set_var("BRAIN_DEVICE", "cpu");
        use crate::encoder::{ctx as ectx, Encoder, PIPELINES};
        use crate::vision_import::build_vision_weights;
        use gpu_core::Gpu;
        use paramstore::ParamStore;

        let vt: Vec<(String, Vec<f32>)> = tensors.into_iter().filter(|t| t.name.contains("vision_tower")).map(|t| (t.name, t.data)).collect();
        let weights = build_vision_weights(&vt);
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ectx(&gpu);
        let enc = Encoder::mobileclip_l(&ctx, 1024);
        enc.set_eval(true); // running-stat BN, matching the reference
        let ps = ParamStore::new(&gpu, enc.param_list(), &weights);
        let img = gpu.storage_init("img", &px);
        let out = enc.forward(&ctx, &ps, &img); // [256, 3072]
        assert_eq!(out.len(), feat.len(), "feature shape mismatch");

        let (mut max_abs, mut sum_abs, mut dot, mut na, mut nb) = (0.0f32, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (a, b) in out.iter().zip(&feat) {
            max_abs = max_abs.max((a - b).abs());
            sum_abs += (a - b).abs() as f64;
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64).powi(2);
            nb += (*b as f64).powi(2);
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        eprintln!("vision parity: mean|Δ|={:.4e} max|Δ|={:.4e} cosine={:.6}", sum_abs / out.len() as f64, max_abs, cos);
        assert!(cos > 0.999, "brain vision features diverge from HF: cosine={cos}");
    }


    #[test]
    fn fastvlm_full_pipeline_caption() {
        let CKPT = format!("{}/model.safetensors", model_dir("apple/FastVLM-0.5B").unwrap_or_default());
        let VIS_PX = testdata("vl/parity/fastvlm_vis_px.bin");
        let CAP_LAYOUT = testdata("vl/parity/fastvlm_cap_layout.bin");
        let CAP_IDS = testdata("vl/parity/fastvlm_cap_ids.bin");
        let CAP_GEN = testdata("vl/parity/fastvlm_cap_gen.bin");
        // brain does EVERYTHING on the real weights: mobileclip vision → mlp2x_gelu
        // projector → Qwen2 decoder → greedy decode. No HF tensors at inference — the
        // caption must still match HF's reference token-for-token.
        let (Some(px), Some(layout), Some(ids), Some(gen_ref)) =
            (read_f32(VIS_PX), read_i32(CAP_LAYOUT), read_i32(CAP_IDS), read_i32(CAP_GEN))
        else {
            eprintln!("skip: FastVLM full-pipeline reference not present");
            return;
        };
        let Ok(tensors) = checkpoint::safetensors::read(&CKPT) else {
            eprintln!("skip: FastVLM checkpoint not present");
            return;
        };
        std::env::set_var("BRAIN_DEVICE", "cpu");
        use crate::encoder::{ctx as ectx, kidx, Encoder, PIPELINES};
        use crate::import::{map_decoder, map_projector};
        use crate::vision_import::build_vision_weights;
        use gpu_core::Gpu;
        use paramstore::ParamStore;

        // Split the checkpoint into vision / projector / decoder maps.
        let mut vt = Vec::new();
        let mut proj: HashMap<String, Vec<f32>> = HashMap::new();
        let mut dec: HashMap<String, Vec<f32>> = HashMap::new();
        for t in tensors {
            if t.name.contains("vision_tower") {
                vt.push((t.name, t.data));
            } else if let Some(k) = map_projector(&t.name) {
                proj.insert(k, t.data);
            } else if let Some(k) = map_decoder(&t.name) {
                dec.insert(k, t.data);
            }
        }

        // 1) vision tower → [256, 3072].
        let gpu = Gpu::new_cpu(PIPELINES);
        let ctx = ectx(&gpu);
        let enc = Encoder::mobileclip_l(&ctx, 1024);
        enc.set_eval(true);
        let vps = ParamStore::new(&gpu, enc.param_list(), &build_vision_weights(&vt));
        let img = gpu.storage_init("img", &px);
        let feat = enc.forward(&ctx, &vps, &img); // [256, 3072]

        // 2) mlp2x_gelu projector on the vision GPU: fc1 → erf-GELU → fc2 → [256, 896].
        let (rows, vin, d) = (256u32, 3072u32, 896u32);
        let featb = gpu.storage_init("feat", &feat);
        let (fc1w, fc1b) = (gpu.storage_init("f1w", &proj["fc1.weight"]), gpu.storage_init("f1b", &proj["fc1.bias"]));
        let (fc2w, fc2b) = (gpu.storage_init("f2w", &proj["fc2.weight"]), gpu.storage_init("f2b", &proj["fc2.bias"]));
        let (h, h2, e) = (gpu.storage((rows * d) as u64), gpu.storage((rows * d) as u64), gpu.storage((rows * d) as u64));
        gpu.submit(
            &[],
            &[
                gpu.step(kidx("matmul"), &[&featb, &fc1w, &h], &[rows, vin, d], rows * d),
                gpu.step(kidx("bias_add"), &[&h, &fc1b], &[rows, d], rows * d),
                gpu.step(kidx("gelu_erf"), &[&h, &h2], &[rows * d], rows * d),
                gpu.step(kidx("matmul"), &[&h2, &fc2w, &e], &[rows, d, d], rows * d),
                gpu.step(kidx("bias_add"), &[&e, &fc2b], &[rows, d], rows * d),
            ],
        );
        let embeds = gpu.read(&e, (rows * d) as usize);

        // 3) decoder: splice brain's image embeds + greedy decode.
        let cfg = FastVlmConfig::fastvlm_0_5b().decoder;
        let (pre_len, _post_len) = (layout[0] as usize, layout[1] as usize);
        let img_start = pre_len as u32;
        let mut seq: Vec<u32> = ids[..pre_len].to_vec();
        seq.extend(std::iter::repeat_n(0u32, rows as usize));
        seq.extend(&ids[pre_len..]);
        let t_max = (seq.len() + gen_ref.len() + 1) as u32;
        let vocab = cfg.vocab as usize;
        let mut qwen = Qwen::new(cfg, 1, t_max, &dec);
        qwen.enable_mm_splice(img_start, rows);
        qwen.write_img_embeds(&embeds);
        let mut brain_gen = Vec::new();
        for _ in 0..gen_ref.len() {
            let lg = qwen.logits_all(&seq);
            let last = &lg[(seq.len() - 1) * vocab..seq.len() * vocab];
            let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            brain_gen.push(next);
            seq.push(next);
        }
        eprintln!("FULLY-IN-BRAIN caption tokens: {brain_gen:?}\nHF reference          tokens: {gen_ref:?}");
        assert_eq!(brain_gen, gen_ref, "fully-in-brain caption diverges from HF");
    }

    #[test]
    fn fastvlm_image_caption_matches_hf() {
        let CKPT = format!("{}/model.safetensors", model_dir("apple/FastVLM-0.5B").unwrap_or_default());
        let CAP_EMB = testdata("vl/parity/fastvlm_cap_embeds.bin");
        let CAP_LAYOUT = testdata("vl/parity/fastvlm_cap_layout.bin");
        let CAP_IDS = testdata("vl/parity/fastvlm_cap_ids.bin");
        let CAP_GEN = testdata("vl/parity/fastvlm_cap_gen.bin");
        // Full image → caption on real weights: brain's decoder consumes the 256 HF
        // image embeddings (mobileclip vision + projector — brain's own FastViTHD
        // still needs SE/head import to reach feature parity), splices them into the
        // residual stream (enable_mm_splice), and greedy-decodes. It must reproduce
        // the HF caption token-for-token — for the DOSBox logo: "A wooden frame with
        // the letters B, D, and S in it.". Validates the image-token splice + decode
        // on real weights + real image semantics.
        let (Some(embeds), Some(layout), Some(ids), Some(gen_ref)) =
            (read_f32(CAP_EMB), read_i32(CAP_LAYOUT), read_i32(CAP_IDS), read_i32(CAP_GEN))
        else {
            eprintln!("skip: FastVLM caption reference not present (run tools/goldens/fastvlm_caption_dump_reference.py)");
            return;
        };
        let Ok(tensors) = checkpoint::safetensors::read(&CKPT) else {
            eprintln!("skip: FastVLM checkpoint not present");
            return;
        };
        std::env::set_var("BRAIN_DEVICE", "cpu");
        let (pre_len, post_len) = (layout[0] as usize, layout[1] as usize);
        let n_img = embeds.len() / 896;
        assert_eq!(ids.len(), pre_len + post_len);

        let mut init: HashMap<String, Vec<f32>> = HashMap::new();
        for t in tensors {
            if let Some(name) = map_decoder(&t.name) {
                init.insert(name, t.data);
            }
        }
        let cfg = FastVlmConfig::fastvlm_0_5b().decoder;

        // Prompt layout: [pre] [n_img image placeholders] [post], image at row pre_len.
        let img_start = pre_len as u32;
        let mut seq: Vec<u32> = ids[..pre_len].to_vec();
        seq.extend(std::iter::repeat_n(0u32, n_img)); // overwritten by the splice
        seq.extend(&ids[pre_len..]);
        let prompt_len = seq.len();
        let t_max = (prompt_len + gen_ref.len() + 1) as u32;

        let mut qwen = Qwen::new(cfg, 1, t_max, &init);
        qwen.enable_mm_splice(img_start, n_img as u32);
        qwen.write_img_embeds(&embeds);

        let vocab = FastVlmConfig::fastvlm_0_5b().decoder.vocab as usize;
        let mut brain_gen = Vec::new();
        for _ in 0..gen_ref.len() {
            let lg = qwen.logits_all(&seq);
            let last = &lg[(seq.len() - 1) * vocab..seq.len() * vocab];
            let next = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            brain_gen.push(next);
            seq.push(next);
        }
        eprintln!("brain caption tokens: {brain_gen:?}\nHF    caption tokens: {gen_ref:?}");
        assert_eq!(brain_gen, gen_ref, "brain image caption diverges from HF");
    }
}
