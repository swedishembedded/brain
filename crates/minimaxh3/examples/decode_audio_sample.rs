// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Decode a synthetic latent through MiniMax-H3's REAL audio VAE decoder
//! weights and write the result to a wav file - a manual sanity check that
//! the real-weight decode path (`crate::import::import_audio_vae_decoder` +
//! `crate::vocoder::decode`) produces a real, non-trivial waveform, not just
//! that it stays finite (see `import::tests::decode_runs_at_real_scale_with_
//! real_weights_and_stays_finite` for the automated version of that check).
//!
//! The input latent is SEEDED NOISE, not a real encoded recording - this
//! crate is decode-only (see `crate::vocoder`'s own module doc) and has no
//! encoder to turn a real sound into a latent. So the output is real network,
//! real weights, but not a recognizable sound - it demonstrates the decoder
//! graph executing correctly at real scale, nothing about generation quality.
//!
//! Usage: cargo run -p brain-minimaxh3 --release --example decode_audio_sample -- \
//!     <audio_vae_dir> <out.wav> [seconds]

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: decode_audio_sample <audio_vae_dir> <out.wav> [seconds]");
        std::process::exit(2);
    }
    let dir = &a[1];
    let out = &a[2];
    let seconds: f32 = a.get(3).map(|s| s.parse().expect("seconds")).unwrap_or(3.0);

    let cfg = minimaxh3::vocoder::VocoderConfig::h3_32khz();
    let tensors = minimaxh3::import::import_audio_vae_decoder(dir, &cfg).unwrap_or_else(|e| panic!("import_audio_vae_decoder: {e}"));

    let sample_rate = 32_000u32;
    let t = ((sample_rate as f32 * seconds) / cfg.hop_length() as f32).round() as u32;
    let mut rng = data::rng::Lcg::new(7);
    let z = rng.vec_scaled((cfg.vae_latent_channels * t) as usize, 0.3);

    let wave = minimaxh3::vocoder::decode(&cfg, &tensors, &z, t, Some("cpu"));
    audio::wav::write(out, &wave, sample_rate).unwrap_or_else(|e| panic!("write wav: {e}"));
    println!("decoded {} latent frames -> {} samples ({:.2}s) -> {}", t, wave.len(), wave.len() as f32 / sample_rate as f32, out);
}
