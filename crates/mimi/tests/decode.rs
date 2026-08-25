// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)
//! Decode-path tests for the Qwen3-TTS 12 Hz codec.
//!
//! All gated on the real checkpoint being present (it is a large external
//! artifact, not committed). Run with the CPU backend:
//!   BRAIN_DEVICE=cpu cargo test -p brain-mimi

use std::collections::HashMap;

use mimi::{Codec, CodecConfig};

#[allow(dead_code)]
use brain_testutil::testdata;

/// Regenerates the golden this suite compares against, quoted in every skip.
const DUMPER: &str = "tools/goldens/qwen3tts_codec_dump_reference.py";
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
    // Memoize: import the (651 MB) checkpoint ONCE and share the path across all
    // tests. Without this, parallel tests race on the same temp filename and the
    // checkpoint `rename` finalisation panics (and each test re-dequantizes the
    // whole file). `get_or_init` runs the closure on exactly one thread.
    //
    // The name is fixed rather than pid-suffixed so a re-run overwrites the
    // previous run's 646 MB intermediate instead of leaving one behind per run;
    // this binary is the only writer of it, and `mimi::import` finalises by
    // rename. No test deletes it: it is SHARED, so whichever test finished
    // first used to delete it out from under the others.
    static SHARED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SHARED
        .get_or_init(|| {
            let out =
                std::env::temp_dir().join("codec_decode.safetensors").to_string_lossy().into_owned();
            mimi::import(&CKPT_DIR, &out).expect("import failed");
            out
        })
        .clone()
}

/// Every `decoder.*` tensor is accounted for (mapped or intentionally dropped),
/// codebooks are collapsed to tables, and shapes match the config.
#[test]
fn import_consumes_every_decoder_tensor() {
    if !ckpt_available() {
        brain_testutil::skip("checkpoint not present");
        return;
    }
    let out = import_to_temp();
    let c = checkpoint::load(&out);
    // safetensors carries no role; numel comes from the loaded tensor length.
    let by_name: HashMap<String, usize> =
        c.tensors.iter().map(|t| (t.name.clone(), t.data.len())).collect();

    // Decoder-derived params (names without the `encoder.` prefix): 271 decoder
    // tensors, 2 input_proj dropped, 32 codebook tensors collapse to 16 tables
    // => 271 - 2 - 32 + 16 = 253. (The import now ALSO carries the encode path
    // under `encoder.*`; that is asserted in the `encode` test suite.)
    let decoder_params = by_name.keys().filter(|n| !n.starts_with("encoder.")).count();
    assert_eq!(decoder_params, 253, "unexpected decoder param count after import");
    assert!(by_name.keys().any(|n| n.starts_with("encoder.")), "encoder params missing");

    let cfg = CodecConfig::from_json(&c.header["config"]);
    let dim = (cfg.codebook_dim / 2) as usize; // 256

    // codebooks collapsed; raw stats tensors gone.
    assert_eq!(by_name["quantizer.rvq_first.vq.layers.0.table"], 2048 * dim);
    assert_eq!(by_name["quantizer.rvq_rest.vq.layers.14.table"], 2048 * dim);
    assert!(!by_name.contains_key("quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"));
    assert!(!by_name.contains_key("quantizer.rvq_first.input_proj.weight"));

    // a few representative shapes across every stage.
    assert_eq!(by_name["quantizer.rvq_first.output_proj.weight"], 512 * dim);
    assert_eq!(by_name["pre_conv.conv.weight"], 1024 * 512 * 3);
    assert_eq!(by_name["pre_transformer.input_proj.weight"], 512 * 1024);
    assert_eq!(by_name["pre_transformer.layers.7.self_attn.q_proj.weight"], 1024 * 512);
    assert_eq!(by_name["upsample.0.0.conv.weight"], 1024 * 1024 * 2);
    assert_eq!(by_name["upsample.1.1.pwconv1.weight"], 4096 * 1024);
    assert_eq!(by_name["decoder.0.conv.weight"], 1536 * 1024 * 7);
    assert_eq!(by_name["decoder.1.block.1.conv.weight"], 1536 * 768 * 16);
    assert_eq!(by_name["decoder.6.conv.weight"], 96 * 7);
}

/// Decode random valid codes and assert the waveform is finite, the right length,
/// bounded, and not silent.
#[test]
fn decode_random_codes_is_finite_and_bounded() {
    if !ckpt_available() {
        brain_testutil::skip("checkpoint not present");
        return;
    }
    let out = import_to_temp();
    let codec = Codec::load_inference(&out);
    let cfg = &codec.cfg;
    let nq = cfg.num_quantizers as usize;
    let t = 24usize;

    // deterministic LCG of in-range codes (all codebooks are size 2048).
    let mut seed = 0x1234_5678u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let codes: Vec<u32> = (0..t * nq).map(|_| next() % 2048).collect();

    let wav = codec.decode(&codes);

    let expect = t * cfg.decode_upsample_rate as usize;
    assert_eq!(wav.len(), expect, "length {} != T*1920 {}", wav.len(), expect);
    assert!(wav.iter().all(|x| x.is_finite()), "non-finite sample");
    assert!(wav.iter().all(|x| x.abs() <= 1.0 + 1e-6), "out of [-1,1]");
    let energy: f32 = wav.iter().map(|x| x * x).sum::<f32>() / wav.len() as f32;
    assert!(energy > 0.0, "waveform is silent");
    eprintln!("decode ok: {} samples, rms {:.4}", wav.len(), energy.sqrt());
}

/// Parity vs the PyTorch golden dump, if present. The dump is
/// `{codes.bin (u32 LE [T,16]), waveform.bin (f32 LE), meta.json}`.
#[test]
fn parity_against_golden_dump() {
        let DUMP_DIR = testdata("tts/dumps/codec_ref");
    let codes_p = std::path::Path::new(&DUMP_DIR).join("codes.bin");
    let wav_p = std::path::Path::new(&DUMP_DIR).join("waveform.bin");
    if !ckpt_available() || !codes_p.exists() || !wav_p.exists() {
        brain_testutil::skip("golden dump not present");
        return;
    }
    let out = import_to_temp();
    let codec = Codec::load_inference(&out);

    // The golden is tensors plus a claim; the claim only means anything paired
    // with the checkpoint that produced it. `identity` is the set of config
    // fields that fix every dumped shape.
    let meta = std::path::Path::new(&DUMP_DIR).join("meta.json");
    let Some(src) = brain_testutil::golden::Source::open_manifest(&meta, DUMPER) else {
        return;
    };
    let c = &codec.cfg;
    if !src.require(&[
        ("num_quantizers", c.num_quantizers as i64),
        ("codebook_size", c.codebook_size as i64),
        ("decode_upsample_rate", c.decode_upsample_rate as i64),
        ("latent_dim", c.latent_dim as i64),
        ("decoder_dim", c.decoder_dim as i64),
        ("decoder_hidden_size", c.hidden_size as i64),
    ]) {
        return;
    }

    // Each .bin starts with a u64 LE element-count prefix (8 bytes), then data.
    let cb = std::fs::read(&codes_p).unwrap();
    let codes: Vec<u32> = cb[8..].chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let rb = std::fs::read(&wav_p).unwrap();
    let reference: Vec<f32> = rb[8..].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

    let got = codec.decode(&codes);
    let n = got.len().min(reference.len());
    let max_abs = (0..n).map(|i| (got[i] - reference[i]).abs()).fold(0.0f32, f32::max);

    // log-mel L1 (audio::mel) for a perceptual-ish parity number.
    let cfg = audio::mel::MelConfig::default_24k();
    let (m_got, _) = audio::mel::log_mel(&got[..n], &cfg);
    let (m_ref, _) = audio::mel::log_mel(&reference[..n], &cfg);
    let mn = m_got.len().min(m_ref.len());
    let mel_l1: f32 = (0..mn).map(|i| (m_got[i] - m_ref[i]).abs()).sum::<f32>() / mn.max(1) as f32;

    eprintln!(
        "parity: len got {} ref {}; max-abs {:.3e}; log-mel L1 {:.4e}",
        got.len(),
        reference.len(),
        max_abs,
        mel_l1
    );
    // Measured on the official checkpoint against the fp32 golden written by
    // DUMPER (T=24, CPU backend): max-abs 5.7e-4, log-mel L1 7.7e-4. The
    // waveform length is exact; the residual is non-associative fp
    // accumulation order across this deep conv stack, not a structural
    // mismatch. Every stage here is pure fp32 with exact RoPE/conv/GQA and
    // host-exact ConvNeXt LayerNorm/GELU.
    //
    // The ceilings keep real headroom over those measurements on purpose
    // (max-abs ~100x, log-mel L1 ~13x): a bound fitted to one run goes red on
    // the next backend or driver for no defect, while this much margin still
    // catches a reassociation that changes the arithmetic (the GEMM conv
    // lowering being the live example) rather than just reorders it.
    assert_eq!(got.len(), reference.len(), "length mismatch");
    assert!(max_abs.is_finite() && max_abs < 6e-2, "max-abs error too large: {max_abs}");
    assert!(mel_l1 < 1e-2, "log-mel L1 too large: {mel_l1}");
}
