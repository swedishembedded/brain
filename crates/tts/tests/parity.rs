// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)
//! Talker logit-parity test vs a PyTorch golden dump.
//!
//! Isolates the Talker decoder + untied codec head: the reference feeds
//! `codec_embedding(codebook0_ids)` straight through the 28-layer decoder and
//! `codec_head`, which is exactly what `TalkerModel::logits_all` computes. Gated
//! on both the real checkpoint and the dump being present (large external
//! artifacts). Run: `BRAIN_DEVICE=cpu cargo test -p brain-tts --test parity`.

use tts::TalkerModel;

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


fn read_u32(path: &str) -> Option<Vec<u32>> {
    let b = std::fs::read(path).ok()?;
    let n = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    Some((0..n).map(|i| u32::from_le_bytes(b[8 + i * 4..12 + i * 4].try_into().unwrap())).collect())
}
fn read_f32(path: &str) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    let n = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    Some((0..n).map(|i| f32::from_le_bytes(b[8 + i * 4..12 + i * 4].try_into().unwrap())).collect())
}

#[test]
fn talker_logits_match_reference() {
        let CKPT = testdata("tts/ckpt/Qwen3-TTS-12Hz-0.6B-Base");
        let DUMP = testdata("tts/dumps/talker_ref");
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let (Some(tokens), Some(ref_logits)) =
        (read_u32(&format!("{DUMP}/tokens.bin")), read_f32(&format!("{DUMP}/logits.bin")))
    else {
        eprintln!("skip: talker golden dump not present");
        return;
    };
    if !std::path::Path::new(&CKPT).join("model.safetensors").exists() {
        eprintln!("skip: checkpoint not present");
        return;
    }
    let vocab = 3072usize;
    let t = tokens.len();
    assert_eq!(ref_logits.len(), t * vocab, "dump shape [T,vocab]");

    let out = std::env::temp_dir().join("brain_talker_parity.weights");
    let out = out.to_str().unwrap();
    tts::import::import_talker(&CKPT, out).expect("talker import");
    let model = TalkerModel::load_inference(out, 1, t as u32);
    let got = model.logits_all(&tokens);
    assert_eq!(got.len(), t * vocab);

    // max-abs logit error + top-1 agreement per position.
    let mut max_abs = 0.0f32;
    for (a, b) in got.iter().zip(&ref_logits) {
        max_abs = max_abs.max((a - b).abs());
    }
    let mut top1_ok = 0usize;
    for p in 0..t {
        let argmax = |v: &[f32]| (0..vocab).max_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap()).unwrap();
        let ga = argmax(&got[p * vocab..(p + 1) * vocab]);
        let ra = argmax(&ref_logits[p * vocab..(p + 1) * vocab]);
        if ga == ra {
            top1_ok += 1;
        }
    }
    eprintln!("talker parity: max_abs={max_abs:.4}  top1={top1_ok}/{t}");
    let _ = std::fs::remove_file(out);
    // Same bar as the qwen HF-parity gate: top-1 must match everywhere; the
    // absolute logit error is bounded (reference runs bf16 weights).
    assert_eq!(top1_ok, t, "top-1 must agree at every position");
    assert!(max_abs < 2.0, "max-abs logit error too large: {max_abs}");
}
