// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real end-to-end smoke test of `OmniInner::speak`: the FULL chain this
//! round built — Thinker text generation, `crate::talker_prompt`'s Thinker
//! -> Talker prefill assembly, Talker's own KV-cache decode
//! (`crate::talker`), the MTP code predictor
//! (`tts::mtp::MtpModel::generate_residuals`), and `codec::Codec::decode_omni`
//! — run together for the first time against real weights, no stubs
//! anywhere in the chain.
//!
//! Deliberately NOT a parity test (no golden exists for the composed splice
//! — `crate::talker_prompt`'s own doc explains why) and NOT numerically
//! exact against anything: this proves the chain runs to completion and
//! produces a real, finite, non-silent waveform of a sane length — the
//! "does the loop control work end to end on real weights" bar
//! `generate_e2e.rs` set for the text-only path, extended to speech output.
//!
//! Real-weight-adjacent: skips cleanly when `BRAIN_OMNI_HF_DIR` is unset.
//! Expected to be SLOW (`crate::talker_generate`'s own module doc: every
//! Talker layer's weights are streamed fresh per decode step, same
//! validation-tier tradeoff as the Thinker path) — `#[ignore]`d.
//!
//! usage: `BRAIN_OMNI_HF_DIR=/tmp/.X11-unix/brain/hf/Qwen3-Omni-30B-A3B-Instruct \
//!         cargo test --release -p brain-omni --test speak_e2e -- --ignored --nocapture`

use qwen3omnimoe::caps::OmniProvider;

#[test]
#[ignore]
fn speak_runs_end_to_end_and_produces_a_real_waveform() {
    let Some(hf_dir) = std::env::var("BRAIN_OMNI_HF_DIR").ok().filter(|p| !p.is_empty()) else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset");
        return;
    };

    let provider = OmniProvider::load(&hf_dir).expect("load OmniProvider");
    let inner = provider.inner();

    println!("running real Thinker(text) -> Talker -> MTP -> Code2Wav chain -- expect this to take a while (every layer's weights streamed fresh per step)...");
    let mut streamed = Vec::new();
    let mut n_chunks = 0u32;
    let (text, wav, sample_rate) = inner
        .speak("Say hello.", "Say hello.", 8, "chelsie", |chunk| {
            n_chunks += 1;
            streamed.extend_from_slice(chunk);
        })
        .expect("speak failed on real weights");

    println!("text: {text:?}");
    println!("wav: {} samples at {sample_rate} Hz ({:.2}s), {n_chunks} chunks", wav.len(), wav.len() as f32 / sample_rate as f32);

    assert!(!text.is_empty(), "Thinker must produce some text");
    assert!(!wav.is_empty(), "Code2Wav must produce a non-empty waveform");
    assert!(wav.iter().all(|s| s.is_finite()), "waveform must be entirely finite");
    let rms = (wav.iter().map(|s| s * s).sum::<f32>() / wav.len() as f32).sqrt();
    println!("rms: {rms:.6}");
    assert!(rms > 1e-4, "waveform rms {rms} looks like silence, not real speech");
    assert_eq!(sample_rate, 24000, "Code2Wav's own configured output rate");

    // Real streaming, not just a real waveform: the on_chunk callback must
    // have actually fired, and the reassembled stream must equal the
    // returned waveform exactly -- proves decode_omni_chunked's chunked
    // path (validated on synthetic weights in crates/codec/tests/
    // decode_omni_chunked.rs) also holds on the real checkpoint.
    assert!(n_chunks > 0, "speak must stream at least one audio chunk");
    assert_eq!(streamed, wav, "reassembled streamed chunks must equal the returned waveform exactly");
}
