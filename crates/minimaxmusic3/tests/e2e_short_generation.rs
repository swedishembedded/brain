// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real, short, end-to-end generation: lyrics + caption in, a playable WAV
//! out, through every one of the five real checkpoint components.
//!
//! Gated behind all six `BRAIN_MINIMAXMUSIC3_{LM,DEPTH,CONDITION,DIT,
//! VOCODER,TOKENIZER}` env vars - skips cleanly when any is unset or
//! missing (the combined checkpoint is ~28 GB, never committed).
//!
//! **This drives `generate::generate`, the shipped path**, rather than
//! re-composing the five stages inline. It used to do the latter, which
//! made it a gate on a code path no caller uses: it built its own Global
//! LLM instances and its own `Gpu::new_cpu` handles, so it exercised
//! neither `generate`'s cross-card placement of the two CFG branches nor
//! its `Gpu::open` device selection. It therefore could not reproduce a
//! real out-of-memory that the shipped path avoids, nor catch one that the
//! shipped path hits. `generate`'s own module doc already claimed this
//! test called it directly; now that is true.
//!
//! Kept deliberately SHORT: about a second of audio and fewer Euler steps
//! than the reference's default of 30. `generate` still runs its real
//! chunk loop, so this is the general path at a small size, not a
//! single-chunk special case.

use std::env;
use std::path::Path;

use minimaxmusic3::config::VocoderConfig;
use minimaxmusic3::generate::{generate, GenOpts, Paths};

fn env_dir(name: &str) -> Option<String> {
    let dir = env::var(name).ok()?;
    if !Path::new(&dir).exists() {
        return None;
    }
    Some(dir)
}

#[test]
fn real_short_generation_produces_a_playable_wav() {
    let (Some(lm), Some(depth), Some(condition), Some(dit), Some(vocoder), Some(tokenizer)) = (
        env_dir("BRAIN_MINIMAXMUSIC3_LM"),
        env_dir("BRAIN_MINIMAXMUSIC3_DEPTH"),
        env_dir("BRAIN_MINIMAXMUSIC3_CONDITION"),
        env_dir("BRAIN_MINIMAXMUSIC3_DIT"),
        env_dir("BRAIN_MINIMAXMUSIC3_VOCODER"),
        env_dir("BRAIN_MINIMAXMUSIC3_TOKENIZER"),
    )
    else {
        brain_testutil::skip("one or more BRAIN_MINIMAXMUSIC3_{LM,DEPTH,CONDITION,DIT,VOCODER,TOKENIZER} env vars unset");
        return;
    };
    let paths = Paths { lm, depth, condition, dit, vocoder, tokenizer };

    // Echo stage progress. This is the one gate that runs the real ~19B
    // checkpoint end to end, and it is slow enough that silence is
    // indistinguishable from a hang - exactly the failure mode
    // `minimaxmusic3::ProgressSink` exists to remove. Visible under
    // `--nocapture`.
    let start = std::time::Instant::now();
    let mut last_stage = String::new();
    let mut progress = |done: u32, total: u32, stage: &str| {
        // One line per stage transition plus every 10th step, so a long run
        // stays legible without one line per Euler step.
        if stage != last_stage || done % 10 == 0 || done == total {
            eprintln!("[{:>7.1}s] {stage} {done}/{total}", start.elapsed().as_secs_f32());
            last_stage = stage.to_string();
        }
    };

    let opts = GenOpts { duration_seconds: 1.0, num_inference_steps: 8, seed: 1234, device: None };
    let song = generate(
        &paths,
        &opts,
        "[verse]\nquiet morning light\nfading into you\n[chorus]\nhold on to this feeling\n",
        "warm acoustic ballad, gentle fingerpicked guitar, soft female vocals, 80 BPM",
        &mut progress,
    )
    .expect("generate");

    assert!(!song.left.is_empty(), "generated waveform is empty");
    assert_eq!(song.left.len(), song.right.len(), "stereo channel lengths differ");
    assert_eq!(song.sample_rate, VocoderConfig::real().sampling_rate);
    assert!(song.left.iter().chain(&song.right).all(|s| s.is_finite()), "waveform contains a non-finite sample");

    let out_path = env::temp_dir().join("minimaxmusic3_e2e_short.wav");
    audio::wav::write_multi(&out_path, &[&song.left, &song.right], song.sample_rate).expect("write wav");
    let seconds = song.left.len() as f32 / song.sample_rate as f32;
    eprintln!("e2e: wrote {} stereo samples ({seconds:.2}s) to {}", song.left.len(), out_path.display());
    assert!(seconds > 0.5, "generated clip is implausibly short: {seconds:.2}s");
}
