// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Zero-shot voice cloning end to end, against real weights: text + a
//! reference audio clip in, a real playable 24 kHz WAV out. Drives
//! `cosyvoice::pipeline::generate` directly - this crate has no D-Bus/HTTP
//! serving surface yet (that is a later milestone), so this is a plain
//! runnable binary, matching `crates/mimi/examples/encode_wav.rs`'s and
//! `crates/qwen3tts/examples/*.rs`'s own convention for a crate at this
//! stage of the serving ladder.
//!
//! Usage:
//! ```text
//! BRAIN_COSYVOICE_LLM=... BRAIN_COSYVOICE_FLOW=... BRAIN_COSYVOICE_HIFT=... \
//! BRAIN_S3TOKENIZER_V2=... BRAIN_CAMPPLUS_DIR=... BRAIN_COSYVOICE_TOKENIZER=... \
//! cargo run -p brain-cosyvoice --release --example synth -- \
//!     "<target text>" <ref.wav> "<reference clip's own transcript>" out.wav [seed]
//! ```
//! Every env var can instead point at one directory containing all five
//! checkpoints plus `CosyVoice-BlankEN/` - `resources/cosyvoice/weights` is
//! exactly that layout after `resources/cosyvoice/fetch.py` runs, and this
//! example falls back to it when the env vars are unset.

use cosyvoice::pipeline::{CosyVoicePaths, GenOpts};

fn fallback_paths() -> CosyVoicePaths {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights");
    CosyVoicePaths {
        llm: root.to_string(),
        flow: root.to_string(),
        hift: root.to_string(),
        s3tokenizer: root.to_string(),
        campplus: root.to_string(),
        tokenizer: format!("{root}/CosyVoice-BlankEN"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: synth <target text> <ref.wav> <reference clip's own transcript> [out.wav] [seed]");
        std::process::exit(2);
    }
    let text = &a[1];
    let ref_wav = &a[2];
    let ref_text = &a[3];
    let out_wav = a.get(4).map(String::as_str).unwrap_or("cosyvoice_synth.wav");
    let seed: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    let paths = CosyVoicePaths::from_env().unwrap_or_else(|_| fallback_paths());
    let opts = GenOpts { seed, ..GenOpts::default() };

    let started = std::time::Instant::now();
    let out = cosyvoice::pipeline::generate(&paths, &opts, text, ref_wav, ref_text).unwrap_or_else(|e| {
        eprintln!("cosyvoice synth failed: {e}");
        std::process::exit(1);
    });
    let elapsed = started.elapsed();

    audio::wav::write(out_wav, &out.samples, out.sample_rate).unwrap_or_else(|e| panic!("write {out_wav}: {e}"));

    let duration_s = out.samples.len() as f32 / out.sample_rate as f32;
    let rms = (out.samples.iter().map(|&v| v * v).sum::<f32>() / out.samples.len().max(1) as f32).sqrt();
    println!(
        "wrote {out_wav}: {:.3}s @ {} Hz ({} samples), rms={rms:.5}, generated in {:.1}s",
        duration_s,
        out.sample_rate,
        out.samples.len(),
        elapsed.as_secs_f32()
    );
}
