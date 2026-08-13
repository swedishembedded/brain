// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)
//! End-to-end tokenizer PARITY vs the GenieRedux reference: import the real
//! checkpoint, run tokenizer_forward on the exact input the Python reference
//! used (scripts/parity-dump/genie_tokenizer.py), and compare the reconstruction
//! and codebook indices. Ignored by default (needs the checkpoint + the parity
//! dump in scratch):
//!   python scripts/parity-dump/genie_tokenizer.py
//!   cargo test -p brain-wm-genie --test parity_tokenizer -- --ignored --nocapture
use gpu_core::Gpu;
use genieredux::import::import_tokenizer;
use genieredux::{kernel_sources, tokenizer_forward};

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
#[ignore = "needs the checkpoint + parity dump in scratch; run manually"]
fn tokenizer_parity_vs_reference() {
        let CK = repo_path("scratchpad/wm-checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0.pt");
        let DIR = repo_path("scratchpad/parity");
    let inp = format!("{DIR}/genie_tokenizer_in.f32");
    if !std::path::Path::new(&CK).exists() || !std::path::Path::new(&inp).exists() {
        eprintln!("SKIP: checkpoint or parity dump absent (run genie_tokenizer.py)");
        return;
    }
    let (b, c, f, hw) = (1u32, 3u32, 5u32, 64u32);
    let p = 4u32;
    let video = read_f32(&inp);
    let ref_recon = read_f32(&format!("{DIR}/genie_tokenizer_recon.f32"));
    let ref_idx = read_u32(&format!("{DIR}/genie_tokenizer_idx.u32"));

    let (w, cfg) = import_tokenizer(&CK).expect("import");
    eprintln!("imported; running tokenizer_forward...");
    let gpu = Gpu::new_cpu(&kernel_sources());
    let (recon, idx) = tokenizer_forward(&gpu, &video, &w,
        b, c, f, hw, hw, p, cfg.dim, cfg.heads, cfg.head_dim, cfg.code_dim, cfg.n_codes);

    // index agreement
    let idx_match = idx.iter().zip(&ref_idx).filter(|(a,b)| a==b).count();
    eprintln!("codebook indices: {}/{} match", idx_match, ref_idx.len());
    // reconstruction error
    let max = recon.iter().zip(&ref_recon).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    let mean: f32 = recon.iter().zip(&ref_recon).map(|(a,b)|(a-b).abs()).sum::<f32>() / recon.len() as f32;
    eprintln!("reconstruction: max abs {max:.6}, mean abs {mean:.6}");

    assert_eq!(recon.len(), ref_recon.len(), "recon size");
    assert_eq!(idx.len(), ref_idx.len(), "idx size");
    assert_eq!(idx_match, ref_idx.len(), "codebook indices must match exactly");
    assert!(max < 2e-3, "reconstruction max abs {max} exceeds tolerance");
}
