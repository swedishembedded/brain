// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "audio")]

//! Wall-clock profile of the resident synthesizer: load, the first (cold)
//! request, and warm requests of one and several sentences. It asserts nothing
//! about speed; it prints one `profile:` line per request so a run can be
//! stored. Set `BRAIN_PROFILE=1` to run it; it is quiet and skipped otherwise.

use brain::{TtsOptions, TtsPipeline};
use std::time::Instant;

const MODEL: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
const ONE: &str = "I think it is tyranny.";
const THREE: &str = "I think it is tyranny. A tax imposed upon us without our consent is an affront to our rights as free men. We have a right to be represented in the laws that govern us.";

#[test]
fn profile_resident_synthesis() {
    if std::env::var("BRAIN_PROFILE").is_err() {
        return;
    }
    let _serial = brain_testutil::env_lock();
    let Ok(pipe) = TtsPipeline::from_pretrained(MODEL) else {
        brain_testutil::skip("Qwen3-TTS checkpoint is not in the model store");
        return;
    };
    let began = Instant::now();
    let engine = pipe.resident().unwrap();
    println!("profile: load {:.2}s", began.elapsed().as_secs_f64());
    for (name, text) in [("cold-one", ONE), ("warm-one", ONE), ("warm-one-again", ONE), ("warm-three", THREE), ("warm-three-again", THREE)] {
        let began = Instant::now();
        let clip = engine.speak_with(text, TtsOptions::new().seed(1)).unwrap();
        let took = began.elapsed().as_secs_f64();
        let audio = clip.samples().len() as f64 / f64::from(clip.sample_rate());
        let t = engine.last_timings().unwrap();
        println!("profile: {name} generate {:.2}s decode {:.2}s frames {}", t.generate.as_secs_f64(), t.decode.as_secs_f64(), t.frames);
        println!("profile: {name} took {took:.2}s audio {audio:.2}s rtf {:.2}", took / audio);
    }
}
