// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)
//! End-to-end dynamics (guided MaskGIT) PARITY vs the GenieRedux reference:
//! import the tokenizer (for the use_token codebook blend) + the dynamics, run
//! dynamics_forward on the exact input the Python reference used, and compare
//! the next-token logits. Ignored by default (needs both checkpoints + dump):
//!   python scripts/parity-dump/genie_dynamics.py
//!   cargo test -p brain-wm-genie --test parity_dynamics -- --ignored --nocapture
use gpu_core::Gpu;
use wm_genie::import::{import_dynamics, import_tokenizer};
use wm_genie::{dynamics_forward, kernel_sources};

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


fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}
fn read_u32(p: &str) -> Vec<u32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}

#[test]
#[ignore = "needs both checkpoints + dynamics parity dump; run manually"]
fn dynamics_parity_vs_reference() {
        let TOK = repo_path("scratchpad/wm-checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0.pt");
        let DYN = repo_path("scratchpad/wm-checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0.pt");
        let DIR = repo_path("scratchpad/parity");
    let idp = format!("{DIR}/genie_dynamics_ids.u32");
    if !std::path::Path::new(&DYN).exists() || !std::path::Path::new(&idp).exists() {
        eprintln!("SKIP: checkpoints or dynamics dump absent");
        return;
    }
    let (b, t, h, w, na) = (1u32, 5u32, 16u32, 16u32, 7u32);
    let ids = read_u32(&idp);
    let actions = read_f32(&format!("{DIR}/genie_dynamics_actions.f32"));
    let ref_logits = read_f32(&format!("{DIR}/genie_dynamics_logits.f32"));

    let (tw, _) = import_tokenizer(&TOK).expect("import tokenizer");
    let (dw, dc) = import_dynamics(&DYN, &tw.vq).expect("import dynamics");
    eprintln!("imported both; running dynamics_forward...");
    let gpu = Gpu::new_cpu(&kernel_sources());
    let logits = dynamics_forward(&gpu, &ids, &actions, &dw,
        b, t, h, w, dc.dim, dc.heads, dc.head_dim, dc.n_codes, dc.code_dim, na);

    assert_eq!(logits.len(), ref_logits.len(), "logits size");
    let nc = dc.n_codes as usize;
    let ntok = logits.len() / nc;
    // per-token argmax agreement (the predicted next token) — the strong signal
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0;
    let mut am_match = 0usize;
    for tk in 0..ntok {
        let a = argmax(&logits[tk*nc..(tk+1)*nc]);
        let r = argmax(&ref_logits[tk*nc..(tk+1)*nc]);
        if a == r { am_match += 1; }
    }
    let max = logits.iter().zip(&ref_logits).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    let mean: f32 = logits.iter().zip(&ref_logits).map(|(a,b)|(a-b).abs()).sum::<f32>() / logits.len() as f32;
    eprintln!("argmax agreement: {am_match}/{ntok}");
    eprintln!("logits: max abs {max:.5}, mean abs {mean:.6}");

    assert_eq!(am_match, ntok, "predicted-token argmax must match everywhere");
    assert!(max < 5e-2, "logit max abs {max} exceeds tolerance");
}
