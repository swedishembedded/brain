// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)
//! Encode-path tests for the Qwen3-TTS 12 Hz codec (wav -> codes `[T,16]`).
//!
//! Gated on the real checkpoint being present (a large external artifact, not
//! committed). Run on the CPU backend:
//!   BRAIN_DEVICE=cpu make test/one P=brain-mimi T=it ARGS=encode

use mimi::Codec;

#[allow(dead_code)]
use brain_testutil::testdata;

/// Regenerates the golden this suite compares against, quoted in every skip.
const DUMPER: &str = "tools/goldens/qwen3tts_codec_dump_reference.py";


fn ckpt_available() -> bool {
        let CKPT_DIR = testdata("tts/ckpt/Qwen3-TTS-Tokenizer-12Hz");
    std::path::Path::new(&CKPT_DIR).join("model.safetensors").exists()
}

fn import_to_temp() -> String {
        let CKPT_DIR = testdata("tts/ckpt/Qwen3-TTS-Tokenizer-12Hz");
    // Fixed name, not pid-suffixed: this binary is the only writer of it and
    // `mimi::import` finalises by rename, so a re-run overwrites the previous
    // run's 646 MB intermediate instead of leaving one behind per run.
    static SHARED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SHARED
        .get_or_init(|| {
            let out =
                std::env::temp_dir().join("codec_encode.safetensors").to_string_lossy().into_owned();
            mimi::import(&CKPT_DIR, &out).expect("import failed");
            out
        })
        .clone()
}

/// Read a `<u64 count><f32...>` little-endian blob.
fn read_f32(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap();
    b[8..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
/// Read a `<u64 count><u32...>` little-endian blob.
fn read_u32(p: &std::path::Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b[8..].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
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
        brain_testutil::skip("checkpoint not present");
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
        brain_testutil::skip("encode golden dump not present");
        return;
    }
    let codec = Codec::load_inference(&import_to_temp());
    let nq = codec.cfg.num_quantizers as usize;

    // Prove the golden and this checkpoint belong together before comparing.
    let meta = std::path::Path::new(&ENC_DUMP).join("meta.json");
    let Some(src) = brain_testutil::golden::Source::open_manifest(&meta, DUMPER) else {
        return;
    };
    let c = &codec.cfg;
    if !src.require(&[
        ("num_quantizers", c.num_quantizers as i64),
        ("codebook_size", c.codebook_size as i64),
        ("encoder_hidden_size", c.enc.hidden_size as i64),
        ("encoder_num_hidden_layers", c.enc.num_hidden_layers as i64),
    ]) {
        return;
    }

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

/// Real recordings for the device gates: every `*.wav` under
/// `testdata/tts/corpus`, at the codec's rate, in name order. `None` (and a
/// skip naming it) when the directory holds none.
fn corpus(rate: u32) -> Option<Vec<(String, Vec<f32>)>> {
    let dir = testdata("tts/corpus");
    let mut names: Vec<std::path::PathBuf> = std::fs::read_dir(&dir).map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "wav")).collect()).unwrap_or_default();
    if names.is_empty() {
        brain_testutil::skip(&format!("no recordings under {dir}"));
        return None;
    }
    names.sort();
    Some(
        names
            .into_iter()
            .map(|p| {
                let wav = audio::wav::parse(&std::fs::read(&p).unwrap()).unwrap();
                (p.display().to_string(), audio::resample::rational(&wav.samples, rate, wav.sample_rate))
            })
            .collect(),
    )
}

/// The codec on the ambient device, or a skip when that is the CPU backend
/// (the reference itself) or no checkpoint is present.
fn device_codec() -> Option<Codec> {
    if !ckpt_available() {
        brain_testutil::skip("checkpoint not present");
        return None;
    }
    if brain_testutil::gpu_tests_skipped() || gpu_core::backend_name() == "cpu" {
        brain_testutil::skip_unavailable("a GPU (the device encode is compared against the CPU backend)");
        return None;
    }
    Some(Codec::load_inference_on(gpu_core::Gpu::new(mimi::PIPELINES), &import_to_temp()))
}

/// Code agreement of `got` with `want` over every clip: (all codes, cb0).
fn agreement(want: &[Vec<u32>], got: &[Vec<u32>], nq: usize) -> (f64, f64) {
    let (mut hits, mut total, mut hits0, mut frames) = (0usize, 0usize, 0usize, 0usize);
    for (w, g) in want.iter().zip(got) {
        assert_eq!(w.len(), g.len(), "frame count differs");
        hits += w.iter().zip(g).filter(|(a, b)| a == b).count();
        total += w.len();
        hits0 += w.iter().step_by(nq).zip(g.iter().step_by(nq)).filter(|(a, b)| a == b).count();
        frames += w.len() / nq;
    }
    (hits as f64 / total as f64, hits0 as f64 / frames as f64)
}

/// The batched device encode of real recordings gives the CPU backend's codes
/// (the CPU backend runs the direct convs and serial reductions, so it is an
/// independent witness): every frame count equal, and the codes equal but for
/// near-ties of the residual argmin that the reassociated device sums flip.
#[test]
fn the_device_encode_of_real_recordings_gives_the_cpu_codes() {
    let Some(device) = device_codec() else { return };
    let Some(clips) = corpus(device.cfg.input_sample_rate) else { return };
    let clips: Vec<Vec<f32>> = clips.into_iter().take(24).map(|(_, w)| w).collect();
    let cpu = Codec::load_inference(&import_to_temp());
    let want: Vec<Vec<u32>> = clips.iter().map(|w| cpu.encode(w)).collect();
    let got = device.encode_batch(&clips.iter().map(Vec::as_slice).collect::<Vec<_>>());
    let nq = device.cfg.num_quantizers as usize;
    let (all, cb0) = agreement(&want, &got, nq);
    let seconds: f64 = clips.iter().map(|w| w.len() as f64).sum::<f64>() / f64::from(device.cfg.input_sample_rate);
    eprintln!("device vs CPU codes over {} clips ({seconds:.1} s): {:.4} % of codes, {:.4} % of cb0", clips.len(), all * 100.0, cb0 * 100.0);
    assert!(cb0 >= 0.999 && all >= 0.99, "device codes agree with the CPU's at {:.4} % (cb0 {:.4} %)", all * 100.0, cb0 * 100.0);
}

/// Audio seconds the device encodes per wall second over the whole corpus
/// directory, one `encode_batch` call (warm), at several batch sizes.
#[test]
#[ignore = "timing measurement, not a gate"]
fn device_encode_throughput() {
    let Some(device) = device_codec() else { return };
    let Some(clips) = corpus(device.cfg.input_sample_rate) else { return };
    let refs: Vec<&[f32]> = clips.iter().map(|(_, w)| w.as_slice()).collect();
    let seconds: f64 = refs.iter().map(|w| w.len() as f64).sum::<f64>() / f64::from(device.cfg.input_sample_rate);
    device.encode_batch(&refs[..refs.len().min(16)]);
    for batch in [1usize, 8, 32, refs.len()] {
        let t = std::time::Instant::now();
        for chunk in refs.chunks(batch) {
            device.encode_batch(chunk);
        }
        let wall = t.elapsed().as_secs_f64();
        eprintln!("batch {batch:4}: {} clips, {seconds:.1} s of audio in {wall:.2} s = {:.1} audio s per wall s", refs.len(), seconds / wall);
    }
}
