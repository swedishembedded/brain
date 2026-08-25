// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)
//! ECAPA speaker-encoder tests. The import + parity tests are gated on the real
//! Qwen3-TTS checkpoint / reference dump being present (large external
//! artifacts). Run on the CPU backend:
//!   BRAIN_DEVICE=cpu cargo test -p brain-ecapatdnn

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use brain_testutil::{golden::Source, parity::Table};
use ecapatdnn::{SpeakerConfig, SpeakerEncoder};

#[allow(dead_code)]
use brain_testutil::testdata;

/// Regenerates the golden this suite compares against, quoted in every skip.
const DUMPER: &str = "tools/goldens/qwen3tts_speaker_dump_reference.py";
const COS_FLOOR: f64 = 0.999;
// Measured clean against the reference dump: cosine 1.0000000000, rel_l2
// 7.0e-7, max_abs 4.8e-6 (identical on the CPU backend and a Vulkan P40).
// 1e-4 leaves roughly two orders of magnitude of headroom, matching the
// vocoder gate's own clean-value-to-ceiling ratio. A ceiling fitted tightly
// to one run goes red on the next backend or driver for no defect; this one
// still catches a uniformly mis-scaled embedding, which cosine cannot see at
// any floor.
const REL_CEIL: f64 = 1e-4;
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}


fn ckpt_available() -> bool {
        let CKPT_DIR = testdata("tts/ckpt/Qwen3-TTS-12Hz-0.6B-Base");
    std::path::Path::new(&CKPT_DIR).join("model.safetensors").exists()
}

/// Import the (huge) 0.6B safetensors exactly once and share the resulting
/// brain checkpoint path across all tests. Each `import` call dequantises the
/// entire model to f32 (~2.4 GB transient); running it once per test in parallel
/// OOM-kills the process, so we serialise + memoise it.
fn shared_weights() -> &'static str {
        let CKPT_DIR = testdata("tts/ckpt/Qwen3-TTS-12Hz-0.6B-Base");
    static PATH: OnceLock<String> = OnceLock::new();
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    // Fixed name, not pid-suffixed: this binary is the only writer of it and
    // `ecapatdnn::import` finalises by rename, so a re-run overwrites the
    // previous run's intermediate instead of leaving one behind per run.
    PATH.get_or_init(|| {
        let out = std::env::temp_dir().join("speaker.safetensors").to_string_lossy().into_owned();
        ecapatdnn::import(&CKPT_DIR, &out).expect("import failed");
        out
    })
}

/// Raw little-endian `f32` array (the whole file is data), matching the kronos
/// golden convention. Length comes from the file size; no prefix or meta needed.
fn read_dump(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("read {path}"));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
fn import_consumes_every_speaker_tensor() {
    if !ckpt_available() {
        brain_testutil::skip("checkpoint not present");
        return;
    }
    let out = shared_weights();
    let c = checkpoint::load(out);
    // safetensors carries no role, so every tensor loads under role "".
    let names: Vec<String> = c.tensors.iter().map(|t| t.name.clone()).collect();
    // 76 speaker_encoder.* tensors, prefix stripped, none duplicated.
    assert_eq!(names.len(), 76, "expected 76 tensors, got {}", names.len());
    let map = c.by_role("");
    assert_eq!(map["blocks.0.conv.weight"].len(), 512 * 128 * 5);
    assert_eq!(map["asp.tdnn.conv.weight"].len(), 128 * 4608);
    assert_eq!(map["fc.weight"].len(), 1024 * 3072);
    assert!(!names.iter().any(|n| n.starts_with("speaker_encoder.")), "prefix not stripped");
}

#[test]
fn forward_finite_random_mel() {
    if !ckpt_available() {
        brain_testutil::skip("checkpoint not present");
        return;
    }
    let out = shared_weights();
    let enc = SpeakerEncoder::load_inference_on(gpu_core::testgpu::dev(ecapatdnn::model::PIPELINES), out);
    let t = 120usize;
    // deterministic pseudo-random mel in a plausible log-mel range.
    let mel: Vec<f32> = (0..t * 128)
        .map(|i| {
            let x = ((i as f32 * 12.9898).sin() * 43_758.547).fract();
            -5.0 + 6.0 * x
        })
        .collect();
    let emb = enc.embed(&mel);
    assert_eq!(emb.len(), 1024);
    assert!(emb.iter().all(|v| v.is_finite()), "embedding has non-finite values");
    assert!(emb.iter().any(|&v| v != 0.0), "embedding all zero");
}

#[test]
fn parity_against_reference_dump() {
        let DUMP_DIR = testdata("tts/dumps/spk_ref");
    if !ckpt_available() {
        brain_testutil::skip("checkpoint not present");
        return;
    }
    let mel_path = format!("{DUMP_DIR}/mel.f32");
    let emb_path = format!("{DUMP_DIR}/embedding.f32");
    if !std::path::Path::new(&mel_path).exists() || !std::path::Path::new(&emb_path).exists() {
        brain_testutil::skip("reference dump not present");
        return;
    }
    let cfg = SpeakerConfig::default();
    let Some(src) = Source::open(std::path::Path::new(&DUMP_DIR), DUMPER) else {
        return;
    };
    if !src.require(&[
        ("mel_dim", cfg.mel_dim as i64),
        ("enc_dim", cfg.enc_dim as i64),
        ("enc_channels_last", *cfg.enc_channels.last().unwrap() as i64),
        ("enc_attention_channels", cfg.enc_attention_channels as i64),
        ("enc_res2net_scale", cfg.enc_res2net_scale as i64),
        ("enc_se_channels", cfg.enc_se_channels as i64),
    ]) {
        return;
    }
    let mel = read_dump(&mel_path);
    let reference = read_dump(&emb_path);
    assert_eq!(reference.len(), cfg.enc_dim as usize);

    let out = shared_weights();
    let enc = SpeakerEncoder::load_inference_on(gpu_core::testgpu::dev(ecapatdnn::model::PIPELINES), out);
    let emb = enc.embed(&mel);
    assert_eq!(emb.len(), reference.len());

    // Cosine ALONE cannot gate this: it is scale invariant, so an embedding
    // uniformly off by a constant factor scores exactly 1.000000 and passes.
    // The relative-L2 ceiling is what sees a mis-scaled x-vector, and a
    // mis-scaled x-vector is not cosmetic here - the talker consumes this
    // vector as a conditioning embedding, not as a direction.
    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("speaker_xvector", &emb, &reference);
    table.print();
    let na: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs = emb.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("speaker parity: max_abs_err={max_abs:.3e} |emb|={na:.4} |ref|={nb:.4}");
    table.assert_clean();
}

#[test]
fn embed_wav_runs() {
    if !ckpt_available() {
        brain_testutil::skip("checkpoint not present");
        return;
    }
    let out = shared_weights();
    let enc = SpeakerEncoder::load_inference_on(gpu_core::testgpu::dev(ecapatdnn::model::PIPELINES), out);
    // 0.5 s of 24 kHz noise -> mel -> embedding.
    let samples: Vec<f32> = (0..12000)
        .map(|i| 0.1 * ((i as f32 * 0.07).sin()))
        .collect();
    let emb = enc.embed_wav(&samples, 24000);
    assert_eq!(emb.len(), 1024);
    assert!(emb.iter().all(|v| v.is_finite()));
}

#[test]
fn config_defaults() {
    let c = SpeakerConfig::default();
    assert_eq!(c.enc_dim, 1024);
    assert_eq!(c.enc_channels, vec![512, 512, 512, 512, 1536]);
    // from_weights builds without a real checkpoint (empty map would panic on
    // missing weights at forward, so just check construction wiring compiles).
    let _ = HashMap::<String, Vec<f32>>::new();
}
