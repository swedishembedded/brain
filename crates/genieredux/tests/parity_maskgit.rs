// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)
//! MaskGIT sampler parity + closed loop. Imports both checkpoints, runs
//! maskgit_sample on the exact prime/actions the reference used (inference_steps
//! =1 -> deterministic argmax), asserts the sampled tokens match the reference
//! EXACTLY, then closes the loop by decoding prime+generated tokens to a video.
//! Ignored by default:
//!   python scripts/parity-dump/genie_maskgit.py
//!   cargo test -p brain-wm-genie --test parity_maskgit -- --ignored --nocapture
use gpu_core::Gpu;
use genieredux::import::{import_dynamics, import_tokenizer};
use genieredux::{decode_indices, kernel_sources, maskgit_sample};

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
#[ignore = "needs both checkpoints + maskgit parity dump; run manually"]
fn maskgit_sampler_parity_and_loop() {
        let TOK = repo_path("scratchpad/wm-checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0.pt");
        let DYN = repo_path("scratchpad/wm-checkpoints/GenieRedux_Guided_CoinRun_80mln_v1.0.pt");
        let DIR = repo_path("scratchpad/parity");
    let pp = format!("{DIR}/genie_maskgit_prime.u32");
    if !std::path::Path::new(&DYN).exists() || !std::path::Path::new(&pp).exists() {
        eprintln!("SKIP: checkpoints or maskgit dump absent");
        return;
    }
    let (h, w) = (16u32, 16u32);
    let prime = read_u32(&pp);
    let actions = read_f32(&format!("{DIR}/genie_maskgit_actions.f32"));
    let ref_out = read_u32(&format!("{DIR}/genie_maskgit_out.u32"));
    let num_tokens = (h * w) as usize;

    let (tw, _) = import_tokenizer(&TOK).expect("import tokenizer");
    let (dw, dc) = import_dynamics(&DYN, &tw.vq).expect("import dynamics");
    eprintln!("imported; sampling...");
    let gpu = Gpu::new_cpu(&kernel_sources());

    // inference_steps=1, temperature=1.0 -> temp collapses to 0 -> argmax.
    let sampled = maskgit_sample(&gpu, &prime, &actions, num_tokens, h, w, &dw, &dc, 1, 1.0, 0);
    let matches = sampled.iter().zip(&ref_out).filter(|(a,b)| a==b).count();
    eprintln!("sampled tokens: {}/{} match reference", matches, ref_out.len());
    assert_eq!(sampled, ref_out, "sampled tokens must match the reference exactly");

    // Close the loop: decode prime + generated (5 frames) -> video; last frame
    // is the newly generated one.
    let prime_frames = prime.len() / num_tokens; // 4
    let t = (prime_frames + 1) as u32;           // 5
    let mut all: Vec<u32> = prime.clone();
    all.extend_from_slice(&sampled);
    eprintln!("decoding {t} frames to close the loop...");
    let video = decode_indices(&gpu, &all, &tw, 1, t, h, w, 512, 8, 64, 32);
    let expect = (3 * t * 64 * 64) as usize;
    assert_eq!(video.len(), expect, "video shape");
    assert!(video.iter().all(|v| v.is_finite()), "non-finite frame");
    eprintln!("LOOP CLOSED: sampled next-frame tokens (parity-exact) -> decoded {t}-frame video {}x{}", 64, 64);
}
