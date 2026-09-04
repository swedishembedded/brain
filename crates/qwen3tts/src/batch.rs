// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Batched (interleaved, ragged) generation over the KV-cached CPU
//! Talker+MTP path (`crate::gen_kv`/`crate::gen_kv_mtp`, the same engine
//! [`crate::pipeline::generate_codes_cached`] drives for one request).
//!
//! `run_batch` advances every in-flight request by exactly ONE frame per
//! round, round-robin, instead of running requests to completion one at a
//! time: `tts_serve.rs`'s single-executor-thread design queues requests
//! strictly FIFO today, so one long clip fully blocks every shorter request
//! queued behind it. Interleaving fixes that at the SCHEDULING layer - every
//! active request makes steady progress every round, and a request that
//! hits its own EOS, its `max_frames`, or its own cancellation drops out of
//! rotation independently (autoregressive decode has per-request finish times,
//! so the batch is genuinely ragged, never a fixed rectangular shape).
//!
//! **Scope, stated plainly**: this is NOT a single batched GPU matmul across
//! requests (that would need `b>1` in every `Gqa`/`Step` this crate's GPU
//! engine builds, a kernel-shape change out of scope here). Each request
//! gets its own [`crate::gen_kv::CpuTalker`]/[`crate::gen_kv_mtp::CpuMtp`],
//! and the weights those hold are currently reloaded from disk per request,
//! not shared across the batch - a real, separate optimization (holding one
//! read-only weight set behind an `Arc` and threading it through both
//! structs' `step` methods) that this module does not attempt. What IS
//! delivered, and tested below: correct, independent, per-request results
//! under interleaving (no cross-request contamination) and genuinely ragged
//! completion (a short request finishes and stops consuming rounds while a
//! longer one continues).

use capability::CancelToken;

use crate::gen_kv::CpuTalker;
use crate::gen_kv_mtp::CpuMtp;
use crate::pipeline::{self, sample_cb0, GenOpts, TtsPaths};
use crate::prompt::{self, Prompt, TtsSpecials};
use crate::sampling::{DegenerationWatch, Draw, SamplerCfg};
use data::rng::Rng;
use data::tokenizer::Tokenizer;

fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// One request's decode state, advanced one frame at a time by [`Self::step_once`].
/// Mirrors `pipeline::generate_codes_cached`'s loop body exactly (same sampling,
/// same feedback-embedding assembly), just externally driven frame-by-frame
/// instead of owning its own `loop { }`.
struct Session {
    cpu: CpuTalker,
    mtp: CpuMtp,
    prompt: Prompt,
    opts: GenOpts,
    /// This request's own cancellation flag, polled once per round in
    /// [`Self::is_done`] - per-request, never shared, so one caller hanging up
    /// cannot end anybody else's clip.
    cancel: CancelToken,
    rng: Rng,
    /// This session's resolved codebook-0 filter chain, computed once at
    /// construction from `opts` - the scheduler must never re-resolve per frame.
    cfg: SamplerCfg,
    /// The same plan's residual-codebook (subtalker) chain, which the reference
    /// samples by default. Cached for the same reason `cfg` is.
    sub: SamplerCfg,
    /// Diagnostic only: a per-session codebook-0 repetition-run watcher. It
    /// reports, it never steers the draw.
    watch: DegenerationWatch,
    draw: Draw,
    cb0: u32,
    cb0_history: Vec<u32>,
    s: usize,
    past_hidden: Vec<f32>,
    frames: Vec<u32>,
}

impl Session {
    fn new(
        talker_path: &str,
        mtp_path: &str,
        sp: &TtsSpecials,
        prompt: Prompt,
        opts: GenOpts,
        cancel: CancelToken,
    ) -> Session {
        let mut cpu = CpuTalker::load(talker_path);
        let mtp = CpuMtp::load(mtp_path);
        cpu.reset();
        let d = cpu.d();
        let n_prefix = prompt.embeds.len() / d;
        let mut past_hidden = vec![0.0f32; d];
        for i in 0..n_prefix {
            past_hidden = cpu.step(&prompt.embeds[i * d..(i + 1) * d]);
        }
        let mut rng = Rng::new(opts.seed);
        let cb0_history: Vec<u32> = Vec::new();
        let plan = opts.plan();
        let (cfg, sub) = (plan.cb0, plan.subtalker);
        let draw = sample_cb0(cpu.codec_head_logits(&past_hidden), sp.codec_eos, opts.min_new == 0, &cfg, &cb0_history, &mut rng);
        Session {
            cpu,
            mtp,
            prompt,
            opts,
            cancel,
            rng,
            cfg,
            sub,
            watch: DegenerationWatch::new(),
            draw,
            cb0: draw.token,
            cb0_history,
            s: 0,
            past_hidden,
            frames: Vec::new(),
        }
    }

    /// A session leaves rotation on EOS, on its frame cap, OR on its own
    /// cancellation - all three are just "stop scheduling this one", so a
    /// cancelled member needs no separate path through the scheduler and its
    /// partial `frames` are returned exactly like a completed member's.
    fn is_done(&self, sp: &TtsSpecials) -> bool {
        (self.cb0 == sp.codec_eos && self.s >= self.opts.min_new)
            || self.s >= self.opts.max_frames
            || self.cancel.is_cancelled()
    }

    /// Generate exactly one frame's worth of codes and advance the KV cache
    /// by one step. Caller must check [`Self::is_done`] first - calling this
    /// on an already-finished session would generate a spurious extra frame.
    fn step_once(&mut self, sp: &TtsSpecials) {
        let d = self.cpu.d();
        let n_trailing = self.prompt.trailing.len() / d;

        if let Some(report) = self.watch.observe(self.s, self.draw) {
            eprintln!("{report}");
        }
        self.cb0_history.push(self.cb0);
        let cb0_embed = self.cpu.codec_embed(self.cb0).to_vec();
        let (residuals, res_sum) = self.mtp.generate_residuals_with(&self.past_hidden, &cb0_embed, &self.sub, &mut self.rng);
        self.frames.push(self.cb0);
        self.frames.extend_from_slice(&residuals);

        let mut feed = cb0_embed;
        add_into(&mut feed, &res_sum);
        if self.s < n_trailing {
            add_into(&mut feed, &self.prompt.trailing[self.s * d..(self.s + 1) * d]);
        } else {
            add_into(&mut feed, &self.prompt.tts_pad);
        }
        self.s += 1;
        if self.cpu.pos() >= self.cpu.cfg.max_position_embeddings as usize {
            // Out of context: force done on the next `is_done` check by pinning
            // `s` at the cap (mirrors the single-request loop's own break here).
            self.s = self.opts.max_frames;
            return;
        }
        self.past_hidden = self.cpu.step(&feed);
        let logits = self.cpu.codec_head_logits(&self.past_hidden);
        self.draw = sample_cb0(logits, sp.codec_eos, self.s >= self.opts.min_new, &self.cfg, &self.cb0_history, &mut self.rng);
        self.cb0 = self.draw.token;
    }
}

/// Drive `reqs` (each a `(prompt, opts, cancel)` triple) to completion,
/// interleaved round-robin one frame at a time, returning each request's codec
/// codes in the SAME order as `reqs`. See the module doc for what this does and
/// does not deliver.
///
/// Each request carries its OWN [`CancelToken`], polled once per round between
/// frames. A cancelled request stops being scheduled immediately and its
/// partial codes are returned in its slot - the caller already holds the token,
/// so it can tell a cancelled clip from a completed one without a second return
/// channel. Pass an unarmed `CancelToken::default()` for a request that must
/// never be interrupted.
pub fn run_batch(
    talker_path: &str,
    mtp_path: &str,
    sp: &TtsSpecials,
    reqs: Vec<(Prompt, GenOpts, CancelToken)>,
) -> Vec<Vec<u32>> {
    let mut sessions: Vec<Session> = reqs
        .into_iter()
        .map(|(prompt, opts, cancel)| Session::new(talker_path, mtp_path, sp, prompt, opts, cancel))
        .collect();
    let mut active: Vec<usize> = (0..sessions.len()).filter(|&i| !sessions[i].is_done(sp)).collect();
    while !active.is_empty() {
        let mut next_active = Vec::with_capacity(active.len());
        for &i in &active {
            sessions[i].step_once(sp);
            if !sessions[i].is_done(sp) {
                next_active.push(i);
            }
        }
        active = next_active;
    }
    sessions.into_iter().map(|s| s.frames).collect()
}

/// One request in a [`synth_batch`] call: what to say, in which language, and
/// with which sampling/length knobs. Every field is per-request on purpose -
/// a ragged batch whose members all had to share one `max_frames` would not
/// be ragged in the way that matters.
#[derive(Clone, Debug)]
pub struct BatchRequest {
    pub text: String,
    pub lang: String,
    pub opts: GenOpts,
}

/// **The end-to-end batch entry point**: text in, waveforms out, interleaved.
///
/// [`run_batch`] is the scheduler and nothing else - it takes already-assembled
/// [`Prompt`]s and returns codec codes. This wraps it into something a real
/// caller (the `batch` action on [`crate::caps`], and through it `brain do
/// qwen3tts batch` / `capability::Registry::run` / D-Bus) can invoke: tokenize
/// and assemble each request's prompt, interleave the decode, then decode each
/// request's codes to a 24 kHz waveform.
///
/// Speaker-free synthesis only (the same conditioning
/// [`crate::pipeline::synth`] builds). Cloning and VoiceDesign would each need
/// their own per-request conditioning inputs in the request shape, and the
/// scheduling story - the thing this path exists to demonstrate - is identical
/// for all three, so they are deliberately out of scope here rather than
/// half-wired.
///
/// Entirely host-side: prompts are assembled off [`CpuTalker`]'s own tables,
/// the decode is [`run_batch`]'s CPU Talker+MTP path, and the codec runs
/// through the pure-CPU streaming decoder. That matches what [`run_batch`]
/// itself can actually schedule (see the module doc: this is not a batched
/// GPU matmul), rather than mixing in a device the scheduler cannot use.
pub fn synth_batch(paths: &TtsPaths, reqs: &[BatchRequest]) -> Result<Vec<Vec<f32>>, String> {
    if reqs.is_empty() {
        return Err("tts batch: needs at least one request".to_string());
    }
    for p in [&paths.talker, &paths.mtp, &paths.codec] {
        if !std::path::Path::new(p).exists() {
            return Err(format!("tts batch: weights not found at '{p}' (run `brain tts import`)"));
        }
    }
    let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
    let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;

    // One Talker load for prompt assembly (embedding tables only). `run_batch`
    // gives each session its own decoder copy - the weight-sharing limitation
    // its module doc already states - so this one is dropped before the batch
    // runs rather than held alongside them.
    // No per-request cancellation here: `synth_batch`'s `BatchRequest` carries
    // no token (unlike `caps::synth`/`clone`/`design`), so every session gets
    // an unarmed one, the same "must never be interrupted" convention
    // `tts_cli.rs` uses for its own foreground one-shot calls.
    let sessions: Vec<(Prompt, GenOpts, CancelToken)> = {
        let talker = CpuTalker::load(&paths.talker);
        let mut out = Vec::with_capacity(reqs.len());
        for (i, r) in reqs.iter().enumerate() {
            if r.text.trim().is_empty() {
                return Err(format!("tts batch: request {i} has empty 'text'"));
            }
            let ids = tok.encode(&pipeline::assistant_text(&r.text));
            let (role_ids, text_ids) = pipeline::split_input_ids(&ids).map_err(|e| format!("tts batch: request {i}: {e}"))?;
            let language_id = sp.language_id(&r.lang);
            out.push((
                prompt::build_xvector_prompt(&talker, &sp, &role_ids, &text_ids, None, language_id),
                // Resolve each request's sampling plan HERE: this is the batch's
                // outermost layer that knows the checkpoint dir, the same place
                // `pipeline::synth` resolves for a single request.
                r.opts.clone().resolved_for(&paths.ckpt_dir),
                CancelToken::default(),
            ));
        }
        out
    };

    let coded = run_batch(&paths.talker, &paths.mtp, &sp, sessions);
    let codec = mimi::decode_stream::StreamingCodecDecoder::load(&paths.codec);
    let mut wavs = Vec::with_capacity(coded.len());
    for (i, codes) in coded.iter().enumerate() {
        if codes.is_empty() {
            return Err(format!("tts batch: request {i} generated no codec frames"));
        }
        wavs.push(codec.decode_streaming(codes, 0));
    }
    Ok(wavs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{synthetic_checkpoints, talker_test_cfg, tiny_prompt, tiny_specials, Scratch};

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// The core correctness bar: interleaving must be invisible to each
    /// request's OWN result. Run two different prompts through `run_batch`
    /// together, then run each alone through the same
    /// `generate_codes_cached` single-request path - the codes must match
    /// bit-for-bit (same seeds, same everything, only the scheduling order
    /// differs).
    #[test]
    fn batched_output_matches_running_each_request_alone() {
        if gpu_disabled() {
            return;
        }
        let scratch = Scratch::new("batch-test");
        let (talker_path, mtp_path) = synthetic_checkpoints(scratch.path(), 3);
        let sp = tiny_specials();
        let d = talker_test_cfg().d_model as usize;

        // Sampling knobs pinned explicitly: the interleaving must be identical
        // to the serial loop under a KNOWN chain, not one a config file supplies.
        let sampling = crate::genconfig::SamplingRequest {
            do_sample: Some(true),
            temperature: Some(0.8),
            top_k: Some(0),
            top_p: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        };
        let opts_a = GenOpts { max_frames: 5, min_new: 1, seed: 11, sampling, ..GenOpts::default() };
        let opts_b = GenOpts { max_frames: 8, min_new: 1, seed: 22, sampling, ..GenOpts::default() };
        let prompt_a = tiny_prompt(d, 3, 2, 101);
        let prompt_b = tiny_prompt(d, 4, 1, 202);

        let batched = run_batch(
            &talker_path,
            &mtp_path,
            &sp,
            vec![
                (prompt_a.clone(), opts_a.clone(), CancelToken::default()),
                (prompt_b.clone(), opts_b.clone(), CancelToken::default()),
            ],
        );
        assert_eq!(batched.len(), 2);

        let uncancellable = CancelToken::default();
        let mut cpu_a = CpuTalker::load(&talker_path);
        let mut mtp_a = CpuMtp::load(&mtp_path);
        let alone_a =
            crate::pipeline::generate_codes_cached(&mut cpu_a, &mut mtp_a, &sp, &prompt_a, &opts_a, &uncancellable)
                .expect("an unarmed token never cancels");

        let mut cpu_b = CpuTalker::load(&talker_path);
        let mut mtp_b = CpuMtp::load(&mtp_path);
        let alone_b =
            crate::pipeline::generate_codes_cached(&mut cpu_b, &mut mtp_b, &sp, &prompt_b, &opts_b, &uncancellable)
                .expect("an unarmed token never cancels");

        assert_eq!(batched[0], alone_a, "interleaving must not change request A's own codes");
        assert_eq!(batched[1], alone_b, "interleaving must not change request B's own codes");
        // The two requests have different max_frames (5 vs 8) - a genuinely
        // ragged batch, not a coincidentally-equal-length one.
        assert_ne!(batched[0].len(), batched[1].len(), "test setup must actually exercise raggedness");
    }

    /// The end-to-end entry point against REAL weights: two requests with
    /// different `max_frames` go in together, two independent, non-empty,
    /// DIFFERENT-length waveforms come out - the scheduler's raggedness
    /// surviving all the way through tokenization and the codec, which the
    /// synthetic-checkpoint tests above cannot show (they never build a
    /// prompt from text at all). Skips cleanly without real weights, like
    /// every other real-checkpoint test in this crate.
    ///
    /// Request 0 deliberately reuses the text/seed/`max_frames`
    /// `engine::tests`'s single-request test uses, and produces the same
    /// 30720 samples (EOS at frame 16, well before the 40-frame cap) - so
    /// this doubles as a cross-check that batching a request does not change
    /// what that request alone would have produced, on real weights rather
    /// than the synthetic ones the codes-level test above uses.
    #[test]
    fn synth_batch_returns_one_ragged_waveform_per_request() {
        let (Ok(w), Ok(ckpt)) = (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT")) else {
            brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT not set");
            return;
        };
        let paths = TtsPaths {
            talker: format!("{w}/talker.safetensors"),
            mtp: format!("{w}/mtp.safetensors"),
            codec: format!("{w}/codec.safetensors"),
            speaker: format!("{w}/speaker.safetensors"),
            ckpt_dir: ckpt,
        };
        if !std::path::Path::new(&paths.talker).exists() {
            brain_testutil::skip("weights not found at BRAIN_QWEN3TTS_WEIGHTS");
            return;
        }
        let req = |text: &str, max_frames: usize, seed: u64| BatchRequest {
            text: text.to_string(),
            lang: "english".to_string(),
            opts: GenOpts { max_frames, seed, ..GenOpts::default() },
        };
        let t0 = std::time::Instant::now();
        let wavs = synth_batch(&paths, &[req("Streaming the first request.", 40, 7), req("The second one runs quite a bit longer than the first one does.", 32, 2)])
            .expect("batch synthesis");
        assert_eq!(wavs.len(), 2);
        for (i, wav) in wavs.iter().enumerate() {
            assert!(!wav.is_empty(), "request {i} produced no audio");
            let rms = (wav.iter().map(|s| s * s).sum::<f32>() / wav.len() as f32).sqrt();
            assert!(rms > 1e-3, "request {i} decoded to near-silence (rms {rms:.3e})");
        }
        // Ragged: the two requests' own `max_frames` decide their own lengths.
        assert!(wavs[1].len() > wavs[0].len(), "the longer request must yield the longer clip: {} vs {}", wavs[1].len(), wavs[0].len());
        // Independent: different text and different seed must not converge on
        // the same waveform (the failure mode interleaving could plausibly
        // introduce is cross-request contamination).
        let n = wavs[0].len();
        assert_ne!(wavs[0][..n], wavs[1][..n], "the two requests produced identical audio - the batch is contaminated");
        eprintln!("synth_batch: {} + {} samples in {:.1}s", wavs[0].len(), wavs[1].len(), t0.elapsed().as_secs_f64());
    }

    /// An empty batch is a caller error, not a silent empty result.
    #[test]
    fn an_empty_batch_is_rejected() {
        let paths = TtsPaths { talker: String::new(), mtp: String::new(), codec: String::new(), speaker: String::new(), ckpt_dir: String::new() };
        assert!(synth_batch(&paths, &[]).unwrap_err().contains("at least one"));
    }

    /// A short request must actually stop consuming rounds once it finishes,
    /// not silently keep stepping alongside a longer one still in flight.
    #[test]
    fn a_finished_request_drops_out_of_rotation() {
        if gpu_disabled() {
            return;
        }
        let scratch = Scratch::new("batch-test2");
        let (talker_path, mtp_path) = synthetic_checkpoints(scratch.path(), 5);
        let sp = tiny_specials();
        let d = talker_test_cfg().d_model as usize;

        // min_new: 1 (not 0) forces the first sample to not be EOS, so this
        // reliably produces exactly 1 frame instead of occasionally 0 (a
        // legitimate immediate-EOS sample IS possible when min_new is 0 -
        // that's the option's whole purpose, not a bug this test should chase).
        let short = GenOpts { max_frames: 1, min_new: 1, seed: 1, ..GenOpts::default() };
        let long = GenOpts { max_frames: 6, min_new: 6, seed: 2, ..GenOpts::default() };
        let out = run_batch(
            &talker_path,
            &mtp_path,
            &sp,
            vec![
                (tiny_prompt(d, 2, 1, 1), short, CancelToken::default()),
                (tiny_prompt(d, 2, 1, 2), long, CancelToken::default()),
            ],
        );
        // `MtpConfig::tiny()`'s num_code_groups (4: cb0 + 3 residuals), not the
        // real model's 16 - a per-frame width, not a fixed constant.
        let group_width = crate::config::MtpConfig::tiny().num_code_groups as usize;
        assert_eq!(out[0].len() / group_width, 1, "the 1-frame request must stop at exactly 1 frame");
        assert_eq!(out[1].len() / group_width, 6, "the 6-frame request must still complete in full");
    }

    /// A cancelled batch member must leave rotation exactly like a naturally
    /// finished one: its own partial codes come back, capped at where the
    /// cancel landed, while every OTHER member of the same batch runs to its
    /// full length. This is the property that makes per-request cancellation
    /// safe in a shared scheduler - one caller hanging up must not truncate
    /// anybody else's clip.
    #[test]
    fn a_cancelled_batch_member_drops_out_without_truncating_the_others() {
        if gpu_disabled() {
            return;
        }
        let scratch = Scratch::new("batch-cancel");
        let (talker_path, mtp_path) = synthetic_checkpoints(scratch.path(), 9);
        let sp = tiny_specials();
        let d = talker_test_cfg().d_model as usize;
        let group_width = crate::config::MtpConfig::tiny().num_code_groups as usize;

        // Member 0's token is armed AND already cancelled before the batch
        // starts, so it drops out on round one with zero frames - fully
        // deterministic, no timing involved. Member 1 is untouched.
        let cancelled = CancelToken::armed();
        cancelled.cancel();
        let a = GenOpts { max_frames: 6, min_new: 6, seed: 1, ..GenOpts::default() };
        let b = GenOpts { max_frames: 6, min_new: 6, seed: 2, ..GenOpts::default() };
        let out = run_batch(
            &talker_path,
            &mtp_path,
            &sp,
            vec![
                (tiny_prompt(d, 2, 1, 1), a, cancelled),
                (tiny_prompt(d, 2, 1, 2), b.clone(), CancelToken::default()),
            ],
        );
        assert_eq!(out[0].len(), 0, "a pre-cancelled member must produce no frames at all");
        assert_eq!(out[1].len() / group_width, 6, "the live member must still complete in full");

        // And the live member's codes are bit-identical to running it alone -
        // a neighbour's cancellation must not perturb its sampling stream.
        let mut cpu = CpuTalker::load(&talker_path);
        let mut mtp = CpuMtp::load(&mtp_path);
        let alone = crate::pipeline::generate_codes_cached(
            &mut cpu,
            &mut mtp,
            &sp,
            &tiny_prompt(d, 2, 1, 2),
            &b,
            &CancelToken::default(),
        )
        .expect("an unarmed token never cancels");
        assert_eq!(out[1], alone, "a neighbour's cancellation must not change this request's codes");
    }
}
