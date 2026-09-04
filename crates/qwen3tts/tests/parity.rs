// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Talker logit-parity test vs a PyTorch golden dump.
//!
//! Isolates the Talker decoder + untied codec head: the reference feeds
//! `codec_embedding(codebook0_ids)` straight through the 28-layer decoder and
//! `codec_head`, which is exactly what `TalkerModel::logits_all` computes. Gated
//! on both the real checkpoint and the dump being present (large external
//! artifacts). The dump is produced by
//! `tools/goldens/qwen3tts_dump_talker_reference.py`, which drives the upstream
//! `qwen-tts` reference implementation against the same checkpoint this test
//! imports. Run: `BRAIN_DEVICE=cpu cargo test -p brain-qwen3tts --test parity`.

use qwen3tts::TalkerModel;

use brain_testutil::testdata;

// Raw little-endian arrays (the whole file is data; length = file size), matching
// the kronos golden convention. Shapes come from context (T = tokens.len(),
// vocab is fixed), so no length prefix or side-car meta is needed.
fn read_u32(path: &str) -> Option<Vec<u32>> {
    let b = std::fs::read(path).ok()?;
    Some(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
fn read_f32(path: &str) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[test]
fn talker_logits_match_reference() {
    let ckpt = testdata("tts/ckpt/Qwen3-TTS-12Hz-0.6B-Base");
    let dump = testdata("tts/dumps/talker_ref");
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
    }
    let (Some(tokens), Some(ref_logits)) =
        (read_u32(&format!("{dump}/tokens.u32")), read_f32(&format!("{dump}/logits.f32")))
    else {
        brain_testutil::skip("talker golden dump not present");
        return;
    };
    if !std::path::Path::new(&ckpt).join("model.safetensors").exists() {
        brain_testutil::skip("checkpoint not present");
        return;
    }
    let vocab = 3072usize;
    let t = tokens.len();
    assert_eq!(ref_logits.len(), t * vocab, "dump shape [T,vocab]");

    let out = std::env::temp_dir().join("brain_talker_parity.safetensors");
    let out = out.to_str().unwrap();
    qwen3tts::import::import_talker(&ckpt, out).expect("talker import");
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
    // Same bar as the qwen HF-parity gate: top-1 must match everywhere, and the
    // absolute logit error stays bounded. Both sides dequantize the same bf16
    // weights to fp32, so the only spread is accumulation order across 28
    // layers - the bound is a regression guard, not a dtype allowance.
    assert_eq!(top1_ok, t, "top-1 must agree at every position");
    assert!(max_abs < 2.0, "max-abs logit error too large: {max_abs}");
}
