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
use crate::pipeline::{sample_cb0, GenOpts};
use crate::prompt::{Prompt, TtsSpecials};
use data::rng::Rng;

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
        let cb0 = sample_cb0(
            cpu.codec_head_logits(&past_hidden),
            sp.codec_eos,
            opts.min_new == 0,
            opts.temperature,
            opts.top_k,
            opts.top_p,
            opts.repetition_penalty,
            &cb0_history,
            &mut rng,
        );
        Session { cpu, mtp, prompt, opts, cancel, rng, cb0, cb0_history, s: 0, past_hidden, frames: Vec::new() }
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

        self.cb0_history.push(self.cb0);
        let cb0_embed = self.cpu.codec_embed(self.cb0).to_vec();
        let (residuals, res_sum) = self.mtp.generate_residuals(&self.past_hidden, &cb0_embed);
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
        self.cb0 = sample_cb0(
            self.cpu.codec_head_logits(&self.past_hidden),
            sp.codec_eos,
            self.s >= self.opts.min_new,
            self.opts.temperature,
            self.opts.top_k,
            self.opts.top_p,
            self.opts.repetition_penalty,
            &self.cb0_history,
            &mut self.rng,
        );
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

        let opts_a = GenOpts { max_frames: 5, min_new: 1, seed: 11, temperature: 0.8, top_k: 0, ..GenOpts::default() };
        let opts_b = GenOpts { max_frames: 8, min_new: 1, seed: 22, temperature: 0.8, top_k: 0, ..GenOpts::default() };
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
