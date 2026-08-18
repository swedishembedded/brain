// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward-parity gate: brain's from-scratch Qwen3 forward must match the HF
//! reference logits for a fixed prompt. Gated on `QWEN_PARITY_DIR` pointing at a
//! directory containing the HF checkpoint (`config.json` + `model.safetensors`)
//! plus reference dumps `ref_tokens.bin` (u32 LE) and `ref_logits.bin`
//! (f32 LE, `seq*vocab` row-major), produced by the companion Python script.
//! Skipped (passes trivially) when the env var is unset.

use std::path::Path;

fn read_u32(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi
}

#[test]
fn forward_matches_hf_reference() {
    let dir = match std::env::var("QWEN_PARITY_DIR") {
        Ok(d) => d,
        Err(_) => {
            brain_testutil::skip("QWEN_PARITY_DIR unset");
            return;
        }
    };
    let dir = Path::new(&dir);

    // Import once (cache the brain checkpoint next to the HF files).
    let weights = dir.join("qwen.safetensors");
    if !weights.exists() {
        qwen3::import::import(dir.to_str().unwrap(), weights.to_str().unwrap())
            .expect("import HF Qwen3");
    }

    let tokens = read_u32(&dir.join("ref_tokens.bin"));
    let ref_logits = read_f32(&dir.join("ref_logits.bin"));
    let seq = tokens.len();
    let vocab = ref_logits.len() / seq;
    assert_eq!(ref_logits.len(), seq * vocab, "ref logits shape");

    // Inference-only load (weights frozen — no 4x optimizer allocation).
    let model = qwen3::Qwen::load_inference(weights.to_str().unwrap(), 1, seq as u32);
    let got = model.logits_all(&tokens); // [seq * vocab]
    assert_eq!(got.len(), seq * vocab, "brain logits shape");

    // Per-position top-1 agreement + max/mean abs error.
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut top1_ok = 0usize;
    for p in 0..seq {
        let r = &ref_logits[p * vocab..(p + 1) * vocab];
        let g = &got[p * vocab..(p + 1) * vocab];
        if argmax(r) == argmax(g) {
            top1_ok += 1;
        }
        for (a, b) in r.iter().zip(g.iter()) {
            let e = (a - b).abs();
            max_abs = max_abs.max(e);
            sum_abs += e as f64;
        }
    }
    let mean_abs = (sum_abs / (seq * vocab) as f64) as f32;
    eprintln!(
        "parity: seq={seq} vocab={vocab} top1_ok={top1_ok}/{seq} max_abs={max_abs:.4} mean_abs={mean_abs:.5}"
    );
    eprintln!(
        "last-pos argmax: brain={} ref={}",
        argmax(&got[(seq - 1) * vocab..]),
        argmax(&ref_logits[(seq - 1) * vocab..])
    );

    // Both run fp32 on the same bf16-rounded weights, so logits should be close;
    // exact top-1 agreement at every position is the hard requirement.
    assert_eq!(top1_ok, seq, "top-1 prediction disagrees with HF at some position");
    assert!(max_abs < 1.0, "max abs logit error too large: {max_abs}");
}
