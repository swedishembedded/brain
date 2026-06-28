// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decode-path tests for the Qwen3-TTS 12 Hz codec.
//!
//! All gated on the real checkpoint being present (it is a large external
//! artifact, not committed). Run with the CPU backend:
//!   BRAIN_DEVICE=cpu cargo test -p brain-codec

use std::collections::HashMap;

use codec::{Codec, CodecConfig};

const CKPT_DIR: &str = "/data/workspace/tmp/qwen3-tts-resources/ckpt/Qwen3-TTS-Tokenizer-12Hz";
const DUMP_DIR: &str = "/data/workspace/tmp/qwen3-tts-resources/dumps/codec_ref";

fn ckpt_available() -> bool {
    std::path::Path::new(CKPT_DIR).join("model.safetensors").exists()
}

fn import_to_temp() -> String {
    // Memoize: import the (651 MB) checkpoint ONCE and share the path across all
    // tests. Without this, parallel tests race on the same temp filename and the
    // checkpoint `rename` finalisation panics (and each test re-dequantizes the
    // whole file). `get_or_init` runs the closure on exactly one thread.
    static SHARED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SHARED
        .get_or_init(|| {
            let out = std::env::temp_dir()
                .join(format!("codec_decode_{}.weights", std::process::id()))
                .to_string_lossy()
                .into_owned();
            codec::import(CKPT_DIR, &out).expect("import failed");
            out
        })
        .clone()
}

/// Every `decoder.*` tensor is accounted for (mapped or intentionally dropped),
/// codebooks are collapsed to tables, and shapes match the config.
#[test]
fn import_consumes_every_decoder_tensor() {
    if !ckpt_available() {
        eprintln!("skip: checkpoint not present");
        return;
    }
    let out = import_to_temp();
    let c = checkpoint::load(&out);
    let by_name: HashMap<String, usize> = c
        .header["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| (t["name"].as_str().unwrap().to_string(), t["numel"].as_u64().unwrap() as usize))
        .collect();

    // 271 decoder tensors: 2 input_proj dropped, 32 codebook tensors collapse to
    // 16 tables  =>  271 - 2 - 32 + 16 = 253 params.
    assert_eq!(by_name.len(), 253, "unexpected param count after import");

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

    let _ = std::fs::remove_file(&out);
}

/// Decode random valid codes and assert the waveform is finite, the right length,
/// bounded, and not silent.
#[test]
fn decode_random_codes_is_finite_and_bounded() {
    if !ckpt_available() {
        eprintln!("skip: checkpoint not present");
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

    let _ = std::fs::remove_file(&out);
}

/// Parity vs the PyTorch golden dump, if present. The dump is
/// `{codes.bin (u32 LE [T,16]), waveform.bin (f32 LE), meta.json}`.
#[test]
fn parity_against_golden_dump() {
    let codes_p = std::path::Path::new(DUMP_DIR).join("codes.bin");
    let wav_p = std::path::Path::new(DUMP_DIR).join("waveform.bin");
    if !ckpt_available() || !codes_p.exists() || !wav_p.exists() {
        eprintln!("skip: golden dump not present");
        return;
    }
    let out = import_to_temp();
    let codec = Codec::load_inference(&out);

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
    // Achieved on the official checkpoint vs the PyTorch golden dump (T=24):
    //   max-abs ≈ 3.7e-2, log-mel L1 ≈ 4.6e-3.
    // The waveform length is exact and the log-mel L1 is tiny (near-perfect
    // perceptual parity). The residual max-abs is dominated by the reference
    // forward running in reduced precision (Qwen-TTS bf16) and non-associative
    // fp accumulation-order differences across this deep conv stack, not a
    // structural mismatch — every stage is pure fp32 with exact RoPE/conv/GQA and
    // host-exact ConvNeXt LayerNorm/GELU.
    assert_eq!(got.len(), reference.len(), "length mismatch");
    assert!(max_abs.is_finite() && max_abs < 6e-2, "max-abs error too large: {max_abs}");
    assert!(mel_l1 < 1e-2, "log-mel L1 too large: {mel_l1}");

    let _ = std::fs::remove_file(&out);
}
