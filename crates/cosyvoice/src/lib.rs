// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CosyVoice (FunAudioLLM): LLM-based streaming zero-shot text-to-speech.
//! One id names the FAMILY, not the release - `cosyvoice` covers CosyVoice 2
//! and CosyVoice 3, exactly as `crates/wan` spans Wan2.1/2.2 - because the
//! release is a config (`Variant::CosyVoice2` / `Variant::CosyVoice3`), and a
//! per-release id would collide on the shared upstream naming.
//!
//! ```text
//! text (+ optional reference audio for zero-shot cloning)
//!        |
//!    [s3tokenizer: crate::s3tokenizer]      (reference audio only)
//!        |                                   -> prompt speech tokens
//!    [campplus: crate::campplus]             -> 192-d x-vector
//!        v
//! Qwen2.5-0.5B speech-token LM (this crate, hosted on `crates/qwen3`)
//!    autoregressive: sos ++ text ++ task_id ++ prompt_speech_tokens -> speech tokens (FSQ ids, 25 Hz)
//!        v
//! Causal flow-matching mel decoder (this crate)
//!    CV2: UpsampleConformerEncoder -> CausalConditionalDecoder (UNet CFM estimator)
//!    CV3: PreLookaheadLayer only  -> DiT (adaLN-zero CFM estimator)
//!    zero-shot conditioning: prompt mel written into the CFM `cond` tensor,
//!    the x-vector broadcast as `spks`, 10-step cosine-scheduled Euler ODE
//!    with a FIXED noise buffer (deterministic given the LM's tokens)
//!        v
//! HiFT vocoder (this crate)
//!    conv trunk + NSF harmonic source-filter excitation + ISTFT head (n_fft
//!    16, hop 4) -> 24 kHz waveform; CausalHiFTGenerator for CV3 streaming
//! ```
//!
//! Both generations share `speech_token_size = 6561` (FSQ, `crate::s3tokenizer`),
//! `spk_embed_dim = 192` (CAM++, `crate::campplus`), `llm_input/output_size =
//! 896` (a stock Qwen2.5-0.5B, hosted verbatim on `crates/qwen3` -
//! `qwen3::QwenConfig::qwen2_0_5b()`), `token_frame_rate = 25`,
//! `token_mel_ratio = 2`, and every CFM/HiFT hyperparameter. They differ in
//! the flow decoder's estimator topology (UNet vs. DiT) and in three small
//! vocoder deltas (`CausalHiFTGenerator`'s causal convs).
//!
//! Reference: `FunAudioLLM/CosyVoice` (`cosyvoice/{llm,flow,hifigan}/`),
//! `FunAudioLLM/CosyVoice2-0.5B` and `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
//! on Hugging Face. **No public 1.5B checkpoint exists** - the CosyVoice 3
//! paper scales the LM to 1.5B, but only the 0.5B was released; since the LM
//! config is a generic Qwen2.5 size, a 1.5B checkpoint would drop in as a
//! config change with no code change, recorded here as an unreachable gap,
//! not claimed as support.
//!
//! Status: the speech-token LM (`Qwen2LM`, CosyVoice 2 only - see
//! [`llm`]/[`config`]/[`llm_import`]) is implemented: import from the real
//! `llm.pt`, prompt assembly, and autoregressive generation with
//! `ras_sampling` ([`sampling`]). Forward parity against real-weight goldens
//! (`testdata/golden/cosyvoice/llm_real_*`) is proven at the prefill
//! hidden-state and logits rungs; exact AR-token reproduction is a
//! documented, honest gap (the reference sampler draws from torch's own RNG,
//! which this port does not reproduce bit-for-bit - see [`sampling`]'s module
//! doc). CosyVoice 3's `CosyVoice3LM` is a deliberate follow-up, not
//! implemented.
//!
//! The flow decoder (`CausalMaskedDiffWithXvec`, CosyVoice 2's UNet CFM
//! estimator only - see [`flow`]/[`flow_config`]/[`flow_import`]) is
//! implemented and forward-parity-proven from scratch against the real
//! `flow.pt`: condition assembly, the 10-step Euler CFM loop replayed from
//! real captured reference state, and a full independent re-forward all
//! match the reference mel output. The fixed CFM noise buffer is reproduced
//! by a bit-exact Rust port of PyTorch's CPU RNG (`flow::torch_rng`) rather
//! than a checked-in data asset. Streaming/chunked attention is a
//! documented, not-yet-implemented gap; the host-CPU forward is also slow in
//! an unoptimized debug build (minutes in release, much longer in debug) - a
//! recorded performance follow-up, not a correctness gap.
//!
//! The HiFT vocoder (`HiFTGenerator`, CosyVoice 2 non-causal only - see
//! [`hift`]/[`hift_config`]/[`hift_import`]) is implemented and
//! forward-parity-proven against the real `hift.pt`: the conv trunk, NSF
//! harmonic source excitation, and ISTFT head match the reference's
//! magnitude/phase/waveform exactly given the same excitation noise.
//! Production inference draws its own noise from `data::rng::Rng` - an
//! honest, documented RNG-crossing gap matching [`sampling`]'s. One
//! empirical finding narrows that gap further than it first looks: the
//! reference's OTHER random draw (`rand_ini`, the NSF source's initial
//! phase noise) is provably inert at HiFT's real upsample scale - verified
//! directly against the reference, not assumed - so it never has to be
//! modeled at all. `CausalHiFTGenerator` (CosyVoice 3, causal convs, no
//! `cache_source` state) is a deliberate follow-up, not implemented.
//!
//! Pipeline assembly (streaming `token2wav`, chunking, cross-fade) does not
//! exist yet, so the pipeline as a whole is **still not servable
//! end-to-end**.
//!
//! Swedish Embedded AB implements solutions for from-scratch, dependency-light
//! neural network inference on constrained and embedded targets for its
//! clients. If your team needs expertise in porting speech/audio models to a
//! from-scratch GPU/CPU engine, you can procure our services by sending an
//! email to info@swedishembedded.com.

pub mod config;
pub mod flow;
pub mod flow_config;
pub mod flow_import;
pub mod hift;
pub mod hift_config;
pub mod hift_import;
pub mod llm;
pub mod llm_import;
pub mod sampling;
