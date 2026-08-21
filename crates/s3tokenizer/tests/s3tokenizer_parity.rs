// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Exact-integer-match parity vs the real `speech_tokenizer_v2.onnx`, run
//! (via onnxruntime) by `tools/goldens/cosyvoice_dump_reference.py`.
//!
//! These are discrete codebook indices, not a continuous quantity - the gate
//! is literal `assert_eq!` on the token sequence, never cosine (a gate that
//! cannot fail is worse than no gate at all).
//!
//! Skips cleanly when the golden or the checkpoint is absent.

use std::path::Path;

use brain_testutil::{golden::Source, testdata_path};
use s3tokenizer::config::S3TokenizerConfig;
use s3tokenizer::import::{import_s3tokenizer, RELEASE_FILE};
use s3tokenizer::model::{forward, S3TokenizerWeights};

const DUMPER: &str = "tools/goldens/cosyvoice_dump_reference.py";

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_i32(p: &Path) -> Vec<i32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `BRAIN_S3TOKENIZER_V2`, else the repo-relative
/// `resources/cosyvoice/weights/speech_tokenizer_v2.onnx` - a variable
/// rather than a literal machine path so this test passes on any checkout
/// that fetched the resource, not just the one it was written on (matches
/// `crates/ltxv/tests/vae_parity.rs`'s `weights_path` convention).
fn weights_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_S3TOKENIZER_V2") {
        let p = std::path::PathBuf::from(p);
        return (p.join(RELEASE_FILE).exists()).then_some(p);
    }
    let p = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights"));
    p.join(RELEASE_FILE).exists().then_some(p)
}

#[test]
fn real_speech_tokenizer_v2_matches_the_onnx_reference_exactly() {
    let dir = testdata_path("golden/cosyvoice");
    let meta = dir.join("s3tokenizer_real_meta.json");
    let Some(src) = Source::open_manifest(&meta, DUMPER) else { return };
    let cfg = S3TokenizerConfig::v2();
    if !src.require(&[("num_mels", cfg.n_mels as i64), ("speech_token_size", cfg.n_codebook_size as i64)]) {
        return;
    }

    let Some(weights_dir) = weights_dir() else {
        brain_testutil::skip(&format!("no {RELEASE_FILE} (set BRAIN_S3TOKENIZER_V2 or fetch resources/cosyvoice)"));
        return;
    };

    let mel = read_f32(&dir.join("s3tokenizer_real_in.f32"));
    let want: Vec<i32> = read_i32(&dir.join("s3tokenizer_real_tokens.i32"));

    let n_mels = cfg.n_mels as usize;
    assert_eq!(mel.len() % n_mels, 0, "s3tokenizer_real_in.f32: {} elements not divisible by n_mels={n_mels}", mel.len());
    let t_in = mel.len() / n_mels;

    let m = onnx::read_file(weights_dir.join(RELEASE_FILE)).expect("read speech_tokenizer_v2.onnx");
    let tensors = import_s3tokenizer(onnx::read::graph(&m).unwrap(), &cfg).expect("import");
    let w = S3TokenizerWeights::from_tensors(&tensors, &cfg);

    let got = forward(&cfg, &w, &mel, t_in);

    let mismatches: Vec<(usize, i32, i32)> =
        got.iter().zip(&want).enumerate().filter(|(_, (g, w))| *g != *w).map(|(i, (g, w))| (i, *g, *w)).collect();
    println!(
        "s3tokenizer[real]: {} tokens, {} mismatched: {:?}",
        got.len(),
        mismatches.len(),
        mismatches.iter().take(20).collect::<Vec<_>>()
    );
    assert_eq!(got.len(), want.len(), "s3tokenizer[real]: token count {} vs golden {}", got.len(), want.len());
    assert_eq!(got, want, "s3tokenizer[real]: {} of {} tokens mismatched", mismatches.len(), got.len());
}
