// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity gate for the pure-CPU streaming decoder against the device decode.
//!
//! `StreamingCodecDecoder` (`decode_stream.rs`) is what a SERVER uses: it
//! carries per-conv state so each chunk decodes only its new frames, which is
//! what lets audio start playing before the clip is finished. Nothing proved
//! it agreed with [`mimi::Codec::decode`], the one-shot decode every offline
//! path runs - `decode_stream`'s own tests only compare the streaming back
//! against ITSELF at a different chunk size, which cannot catch a systematic
//! difference in the shared front/back math.
//!
//! Swedish Embedded AB builds streaming neural-audio pipelines whose
//! incremental output is provably the same signal the offline path produces.
//! If your team needs expertise in verifying streaming inference against a
//! batch reference, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Gated on a real imported codec checkpoint via `BRAIN_QWEN3TTS_WEIGHTS`
//! (the same variable the TTS serving surface reads).

use mimi::decode_stream::StreamingCodecDecoder;
use mimi::Codec;

fn codec_path() -> Option<String> {
    let dir = std::env::var("BRAIN_QWEN3TTS_WEIGHTS").ok()?;
    let p = format!("{dir}/codec.safetensors");
    std::path::Path::new(&p).exists().then_some(p)
}

/// Deterministic pseudo-random `[T,16]` codes inside the codebook range.
fn codes(t: usize, nq: usize, size: u32) -> Vec<u32> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..t * nq)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % size as u64) as u32
        })
        .collect()
}

/// The streaming decoder must reproduce the one-shot decode's waveform, both
/// as a single chunk and split across several - otherwise a served clip is a
/// different signal from the same clip rendered offline, and no amount of
/// self-consistency between chunk sizes would reveal it.
#[test]
fn streaming_decode_matches_the_one_shot_decode() {
    let Some(path) = codec_path() else {
        brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/codec.safetensors not present");
        return;
    };
    let codec = Codec::load_inference(&path);
    let (nq, size) = (codec.cfg.num_quantizers as usize, codec.cfg.codebook_size);
    let c = codes(8, nq, size);
    let reference = codec.decode(&c);

    let rms = |x: &[f32]| (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
    let peak = reference.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let rms_ref = rms(&reference);

    let dec = StreamingCodecDecoder::load(&path);
    for chunk in [0usize, 2, 3] {
        let got = dec.decode_streaming(&c, chunk);
        assert_eq!(got.len(), reference.len(), "chunk={chunk}: length differs from the one-shot decode");
        let maxd = got.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let rel_rms = (rms(&got) / rms_ref - 1.0).abs();
        // The two are DIFFERENT implementations of the same math - the one-shot
        // decode runs on the ambient `gpu_core` backend, the streaming one is
        // host `hostmath` - so this is a closeness bar, not bit-exactness. It
        // is tight enough to catch what actually goes wrong here: silence, a
        // wrong scale, a wrong length, or drift that accumulates with the
        // number of chunks (which is why several chunk sizes are checked).
        eprintln!("chunk={chunk}: max-abs {maxd:.3e} ({:.2}% of peak {peak:.3}), rms rel {rel_rms:.2e}", 100.0 * maxd / peak);
        assert!(maxd < 0.05 * peak, "chunk={chunk}: streaming decode differs from one-shot by {maxd:.3e} ({:.1}% of peak)", 100.0 * maxd / peak);
        assert!(rel_rms < 0.02, "chunk={chunk}: streaming decode's level differs by {:.1}%", 100.0 * rel_rms);
    }
}
