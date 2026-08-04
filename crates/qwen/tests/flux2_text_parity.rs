// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 Klein text-conditioning parity: brain's masked-pad multi-layer
//! encoder vs the transformers reference.
//!
//! Goldens: `testdata/flux2/klein-4b/text.safetensors` (from
//! `tools/flux2_dump_reference.py`): chat-templated `input_ids` (512, right-pad
//! 151643), per-layer taps `hidden_{9,18,27}` `[512,2560]`, and the
//! concatenated `ctx` `[512,7680]`. All rows must match — including the ~480
//! pad rows, which the reference computes under the HF `attention_mask` (pad
//! keys excluded); brain reproduces that via `encode_hiddens_padded`.
//!
//! Weights: `BRAIN_FLUX2_TE` = the klein `text_encoder/` HF dir (sharded);
//! tokenizer: `BRAIN_FLUX2_TOKENIZER` = its `tokenizer.json`. Skips if absent.

use data::Tokenizer;
use qwen::{Qwen, QwenConfig};

const PROMPT: &str = "a red fox sitting on a mossy rock in a misty forest, morning light";
const PAD: u32 = 151643;
const TAPS: [usize; 3] = [9, 18, 27];

use brain_testutil::testdata;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn klein_text_conditioning_matches_reference() {
    let fixture = testdata("flux2/klein-4b/text.safetensors");
    if !std::path::Path::new(&fixture).exists() {
        eprintln!("SKIP: fixture {fixture} absent");
        return;
    }
    let Ok(te_dir) = std::env::var("BRAIN_FLUX2_TE") else {
        eprintln!("SKIP: BRAIN_FLUX2_TE unset");
        return;
    };

    let fx = checkpoint::safetensors::read(&fixture).expect("read fixture");
    let get = |name: &str| &fx.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden {name}")).data;
    let ids: Vec<u32> = get("input_ids").iter().map(|&x| x as u32).collect();
    assert_eq!(ids.len(), 512);
    let content_len = ids.iter().position(|&t| t == PAD).unwrap_or(ids.len());

    // Tokenizer + template reproduce the reference input ids exactly.
    if let Ok(tok_path) = std::env::var("BRAIN_FLUX2_TOKENIZER") {
        let tok = data::qwen_tokenizer::QwenBpe::from_file(&tok_path).expect("tokenizer");
        let templated: String = get("template_bytes").iter().map(|&b| b as u8 as char).collect();
        let ours = tok.apply_chat_template_no_think(&[("user", PROMPT)]);
        assert_eq!(ours, templated, "chat template rendering diverges");
        let mut toks = tok.encode(&ours);
        assert_eq!(toks.len(), content_len, "token count diverges");
        toks.resize(512, PAD);
        assert_eq!(toks, ids, "padded token ids diverge");
    } else {
        eprintln!("note: BRAIN_FLUX2_TOKENIZER unset — skipping tokenizer check");
    }

    // Forward parity: one pass, three taps, pad keys masked.
    let cfg = QwenConfig::qwen3_4b();
    let tensors =
        checkpoint::safetensors::read_model_dir(std::path::Path::new(&te_dir)).expect("read TE");
    let init = qwen::import::brain_init_from_hf(tensors, &cfg).expect("brain_init_from_hf");
    // `BRAIN_QWEN_TE_SHARD=1` builds the layer-truncated shard the FLUX.2
    // pipeline actually runs (layers 0..=deepest tap, no head). The whole fp32
    // 4B model is ~16 GB of weights, which with Pascal's non-ReBAR resident
    // overhead does not fit a 24 GB P40 — so this is the only way to gate the
    // GPU kernel selection (cooperative RMSNorm / softmax) against the golden.
    let model = if std::env::var("BRAIN_QWEN_TE_SHARD").as_deref() == Ok("1") {
        let shard = qwen::Shard {
            start: 0,
            end: *TAPS.iter().max().unwrap(),
            embed: true,
            head: false,
            gpu_index: qwen::Shard::ANY_GPU,
        };
        Qwen::new_shard(cfg.clone(), 1, ids.len() as u32, &init, false, shard)
    } else {
        Qwen::new(cfg.clone(), 1, ids.len() as u32, &init)
    };
    let taps = model.encode_hiddens_padded(&ids, content_len, &TAPS);

    let d = cfg.d_model as usize;
    for (k, got) in TAPS.iter().zip(&taps) {
        let want = get(&format!("hidden_{k}"));
        let cos_all = cosine(got, want);
        let cos_content = cosine(&got[..content_len * d], &want[..content_len * d]);
        let cos_pad = cosine(&got[content_len * d..], &want[content_len * d..]);
        eprintln!(
            "layer {k}: cosine all={cos_all:.6} content={cos_content:.6} pad={cos_pad:.6}"
        );
        assert!(cos_all >= 0.9999, "layer {k} cosine {cos_all:.6} < 0.9999");
    }

    // Concatenated conditioning: [h9 | h18 | h27] per token.
    let want_ctx = get("ctx");
    let mut got_ctx = Vec::with_capacity(want_ctx.len());
    for row in 0..ids.len() {
        for tap in &taps {
            got_ctx.extend_from_slice(&tap[row * d..(row + 1) * d]);
        }
    }
    let cos = cosine(&got_ctx, want_ctx);
    eprintln!("ctx concat: cosine={cos:.6}");
    assert!(cos >= 0.9999, "ctx cosine {cos:.6} < 0.9999");
}
