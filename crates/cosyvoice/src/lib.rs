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
//! Status: the speech-token LM (`Qwen2LM` for CosyVoice 2, `CosyVoice3LM` for
//! CosyVoice 3 - see [`llm`]/[`config`]/[`llm_import`]) is implemented for
//! BOTH generations, sharing one `CosyVoiceLm` parameterized by
//! `CosyVoiceLmConfig`'s `SpecialTokenSource` (the one real branch point - see
//! [`config`]'s module doc): import from the real `llm.pt`, prompt assembly,
//! and autoregressive generation with `ras_sampling` ([`sampling`]). Forward
//! parity against real-weight goldens is proven at the prefill hidden-state
//! and logits rungs for both generations (cosine 1.0000000000 against
//! `testdata/golden/cosyvoice/llm_real_*` and `testdata/golden/cosyvoice3/
//! llm_real_*`); exact AR-token reproduction is a documented, honest gap for
//! both (the reference sampler draws from torch's own RNG, which this port
//! does not reproduce bit-for-bit - see [`sampling`]'s module doc).
//!
//! The flow decoder is implemented for BOTH generations:
//! `CausalMaskedDiffWithXvec` (CosyVoice 2's UNet CFM estimator - see
//! [`flow`]/[`flow_config`]/[`flow_import`]) and `CausalMaskedDiffWithDiT`
//! (CosyVoice 3's 22-layer adaLN-zero DiT estimator, no encoder at all - see
//! [`cv3_flow`]/[`cv3_flow_config`]/[`cv3_flow_import`]). Both are
//! forward-parity-proven from scratch against their real `flow.pt`: condition
//! assembly, the 10-step Euler CFM loop replayed from real captured reference
//! state, and a full independent re-forward all match the reference mel
//! output (CosyVoice 3: cosine >= 0.9999999997 at every rung, including the
//! DiT's own `InputEmbedding`/`TimestepEmbedding` internal taps at cosine
//! 1.0000000000). The fixed CFM noise buffer is reproduced by a bit-exact
//! Rust port of PyTorch's CPU RNG (`flow::torch_rng`) rather than a
//! checked-in data asset, shared unchanged by both generations. One
//! real, non-obvious finding from porting the DiT: `x_transformers`'s
//! `RotaryEmbedding`/`apply_rotary_pos_emb` is applied to the FULL
//! `heads*dim_head`-wide query/key row BEFORE the per-head reshape, and only
//! rotates the first `dim_head` channels (a "partial rotary" quirk, not a
//! per-head design) - see `cv3_flow::apply_rope`'s doc for how this was
//! caught (a full-forward divergence despite both sub-stage taps matching
//! exactly) and confirmed against the reference source. Streaming/chunked
//! attention is a documented, not-yet-implemented gap for both generations;
//! the host-CPU forward is also slow in an unoptimized debug build (minutes
//! in release, much longer in debug) - a recorded performance follow-up, not
//! a correctness gap.
//!
//! The HiFT vocoder is implemented for BOTH generations: `HiFTGenerator`
//! (CosyVoice 2, non-causal - see [`hift`]/[`hift_config`]/[`hift_import`])
//! and `CausalHiFTGenerator` (CosyVoice 3, causal convs throughout - see
//! [`cv3_hift`]/[`cv3_hift_config`]/[`cv3_hift_import`]). Both are
//! forward-parity-proven against their real `hift.pt`: the conv trunk, NSF
//! harmonic source excitation, and ISTFT head match the reference's
//! magnitude/phase/waveform (cosine >= 0.9999998 for CosyVoice 3, exactly
//! given the same excitation noise for CosyVoice 2) - CosyVoice 3's small
//! residual is attributed to a KNOWN cause, not unexplained: the reference
//! runs `f0_predictor` in `float64` for causal inference, this port in
//! `f32` throughout (see `cv3_hift`'s module doc). CosyVoice 3's RNG story is
//! simpler than CosyVoice 2's: `SineGen2(causal=True)` reads a buffer fixed
//! ONCE at model-construction time rather than redrawing per call, modeled
//! faithfully by [`cv3_hift::Cv3HiftInstance`] drawing its own noise once at
//! construction and reusing it for every `forward()`. One real, empirically-
//! caught finding while porting this: the shared `nsf_source_forward`'s
//! phase-upsample interpolation mode is `"nearest"` when `causal=True`, not
//! `"linear"` (correct for CosyVoice 2 only) - a first port assumed the same
//! mode for both, which matched CosyVoice 2's own tests but broke CosyVoice 3
//! sharply (cosine ~0.28) starting partway through the signal; now
//! parameterized as `nsf_source_forward`/`nsf_source_forward_causal`.
//! Production inference for both generations draws its own noise from
//! `data::rng::Rng` - an honest, documented RNG-crossing gap matching
//! [`sampling`]'s. `rand_ini` (the NSF source's initial phase noise) is
//! provably inert at HiFT's real upsample scale for BOTH generations -
//! verified directly against the reference, not assumed - so it never has to
//! be modeled at all. `CausalHiFTGenerator.inference()` genuinely has no
//! `cache_source` parameter (a real signature difference, not an
//! oversight) - `cv3_hift::forward`/`Cv3HiftInstance::forward` reflect that.
//!
//! Pipeline assembly (`crate::pipeline`) composes CosyVoice 2's five
//! components - CAM++ + S3Tokenizer speaker/token analysis of a reference
//! clip, the LM, the flow decoder, HiFT - into ONE `pipeline::generate()`
//! call that turns text + a reference audio clip into a real, playable
//! 24 kHz waveform (non-streaming, zero-shot voice cloning). Real weights on
//! this box produce a real WAV file end to end (`crates/cosyvoice/tests/
//! pipeline_e2e.rs`, `crates/cosyvoice/examples/synth.rs`); a composed
//! regression check splices the golden's own captured prompt/generated
//! tokens and x-vector through this crate's flow+HiFT glue and still hits
//! their already-proven real-weight parity numbers. Streaming (chunked
//! `token2wav`, growing-prefix re-run, Hamming cross-fade) is a deliberate,
//! documented follow-up - see [`pipeline`]'s own module doc for the exact
//! scope of what is and is not covered, including the RNG-crossing gaps this
//! pipeline inherits (not introduces) from `crate::llm`/`crate::hift`, and
//! the kaldi-fbank front end's own honest gap (no bit-exact golden exists in
//! this workspace to check `audio::kaldi_fbank` against). **CosyVoice 3
//! pipeline reuse (composing `cv3_flow`/`cv3_hift`/the `CosyVoice3LM` branch
//! into `pipeline::generate()`) is a deliberate follow-up, not implemented
//! here** - `crate::pipeline` still wires CosyVoice 2's components only.
//!
//! [`caps`] wires `pipeline::generate` into brain's generic model-serving
//! contract: one `synth` action (target text + a reference clip and its
//! transcript in, a 24 kHz WAV out), discoverable and runnable as `brain do
//! brain/cosyvoice synth …` / `brain cosyvoice synth …`, with D-Bus/HTTP
//! surfaced for free through the same `Provider`/`ResidentModel` pair every
//! other served model uses (`crates/cli/src/resident_cosyvoice.rs` is the
//! residency-facing half). `variant` accepts both generations' names but
//! only `cosyvoice2` actually runs - see [`caps`]'s own doc for the
//! CosyVoice 3 scope and the reference-audio blob-vs-path bridging this
//! milestone needed.
//!
//! Swedish Embedded AB implements solutions for from-scratch, dependency-light
//! neural network inference on constrained and embedded targets for its
//! clients. If your team needs expertise in porting speech/audio models to a
//! from-scratch GPU/CPU engine, you can procure our services by sending an
//! email to info@swedishembedded.com.

pub mod caps;
pub mod config;
pub mod cv3_flow;
pub mod cv3_flow_config;
pub mod cv3_flow_import;
pub mod cv3_hift;
pub mod cv3_hift_config;
pub mod cv3_hift_import;
pub mod flow;
pub mod flow_config;
pub mod flow_import;
pub mod hift;
pub mod hift_config;
pub mod hift_import;
pub mod llm;
pub mod llm_import;
pub mod lmgrad;
pub mod lmlora;
pub mod pipeline;
pub mod sampling;
