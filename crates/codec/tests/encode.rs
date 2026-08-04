// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)
//! Encode-path tests for the Qwen3-TTS 12 Hz codec (wav -> codes `[T,16]`).
//!
//! Gated on the real checkpoint being present (a large external artifact, not
//! committed). Run on the CPU backend:
//!   BRAIN_DEVICE=cpu cargo test -p brain-codec --test encode

use codec::Codec;

#[allow(dead_code)]
use brain_testutil::testdata;
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}


fn ckpt_available() -> bool {
        let CKPT_DIR = testdata("tts/ckpt/Qwen3-TTS-Tokenizer-12Hz");
    std::path::Path::new(&CKPT_DIR).join("model.safetensors").exists()
}

fn import_to_temp() -> String {
        let CKPT_DIR = testdata("tts/ckpt/Qwen3-TTS-Tokenizer-12Hz");
    static SHARED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SHARED
        .get_or_init(|| {
            let out = std::env::temp_dir()
                .join(format!("codec_encode_{}.safetensors", std::process::id()))
                .to_string_lossy()
                .into_owned();
            codec::import(&CKPT_DIR, &out).expect("import failed");
            out
        })
        .clone()
}

/// Read a `<u64 count><f32...>` little-endian blob.
fn read_f32(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap();
    b[8..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
/// Read a `<u64 count><u32...>` little-endian blob.
fn read_u32(p: &std::path::Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b[8..].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// A synthetic 24 kHz waveform: sum of sines (no external file needed).
fn synth_wav(n: usize) -> Vec<f32> {
    let sr = 24000.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            let s = 0.6 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 880.0 * t).sin();
            s.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Round-trip: wav -> encode -> decode -> wav'. The codes are in range, decode
/// produces finite, bounded, non-silent audio of the expected length.
#[test]
fn round_trip_encode_decode_is_finite() {
    if !ckpt_available() {
        eprintln!("skip: checkpoint not present");
        return;
    }
    let codec = Codec::load_inference(&import_to_temp());
    let nq = codec.cfg.num_quantizers as usize;
    let cb = codec.cfg.enc.codebook_size;

    let wav = synth_wav(12000); // 0.5 s -> ceil-chain to T frames
    let codes = codec.encode(&wav);
    assert_eq!(codes.len() % nq, 0, "codes not a multiple of {nq}");
    let t = codes.len() / nq;
    assert!(t > 0, "no frames produced");
    assert!(codes.iter().all(|&c| c < cb), "code out of codebook range");

    let out = codec.decode(&codes);
    assert_eq!(out.len(), t * codec.cfg.decode_upsample_rate as usize, "round-trip length");
    assert!(out.iter().all(|x| x.is_finite()), "non-finite sample");
    assert!(out.iter().all(|x| x.abs() <= 1.0 + 1e-6), "out of [-1,1]");
    let energy: f32 = out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32;
    assert!(energy > 0.0, "silent round-trip");
    eprintln!("round-trip ok: {} samples in -> T={t} frames -> {} samples out, rms {:.4}", wav.len(), out.len(), energy.sqrt());
}

/// Code-match against the PyTorch `tokenizer.encode` golden dump, if present.
/// RVQ argmin can differ at a few near-tie positions; we require >= 95% match.
#[test]
fn encode_matches_reference_codes() {
        let ENC_DUMP = testdata("tts/dumps/codec_enc_ref");
    let wav_p = std::path::Path::new(&ENC_DUMP).join("wav.bin");
    let codes_p = std::path::Path::new(&ENC_DUMP).join("codes.bin");
    if !ckpt_available() || !wav_p.exists() || !codes_p.exists() {
        eprintln!("skip: encode golden dump not present");
        return;
    }
    let codec = Codec::load_inference(&import_to_temp());
    let nq = codec.cfg.num_quantizers as usize;

    let wav = read_f32(&wav_p);
    let reference = read_u32(&codes_p);
    let got = codec.encode(&wav);

    let t_ref = reference.len() / nq;
    let t_got = got.len() / nq;
    eprintln!("encode: T got {t_got} ref {t_ref} (Q={nq})");
    let t = t_ref.min(t_got);
    assert!(t > 0, "no overlapping frames");

    // Overall + per-codebook match rates over the overlapping frames.
    let mut total = 0usize;
    let mut hits = 0usize;
    let mut cb_hits = vec![0usize; nq];
    for ti in 0..t {
        for q in 0..nq {
            total += 1;
            if got[ti * nq + q] == reference[ti * nq + q] {
                hits += 1;
                cb_hits[q] += 1;
            }
        }
    }
    let rate = hits as f32 / total as f32;
    eprintln!("code-match rate: {:.2}% ({hits}/{total})", rate * 100.0);
    eprintln!("  cb0 (semantic): {:.1}%  cb1: {:.1}%  cb-last: {:.1}%",
        100.0 * cb_hits[0] as f32 / t as f32,
        100.0 * cb_hits[1] as f32 / t as f32,
        100.0 * cb_hits[nq - 1] as f32 / t as f32);
    assert!((t_got as i64 - t_ref as i64).abs() <= 1, "frame count off by >1: got {t_got} ref {t_ref}");
    assert!(rate >= 0.95, "code-match rate {:.2}% < 95%", rate * 100.0);
}
