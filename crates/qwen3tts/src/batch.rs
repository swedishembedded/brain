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
//! hits its own EOS or `max_frames` drops out of rotation independently
//! (autoregressive decode has per-request finish times, so the batch is
//! genuinely ragged, never a fixed rectangular shape).
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
    rng: Rng,
    cb0: u32,
    cb0_history: Vec<u32>,
    s: usize,
    past_hidden: Vec<f32>,
    frames: Vec<u32>,
}

impl Session {
    fn new(talker_path: &str, mtp_path: &str, sp: &TtsSpecials, prompt: Prompt, opts: GenOpts) -> Session {
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
        Session { cpu, mtp, prompt, opts, rng, cb0, cb0_history, s: 0, past_hidden, frames: Vec::new() }
    }

    fn is_done(&self, sp: &TtsSpecials) -> bool {
        (self.cb0 == sp.codec_eos && self.s >= self.opts.min_new) || self.s >= self.opts.max_frames
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

/// Drive `reqs` (each a `(prompt, opts)` pair) to completion, interleaved
/// round-robin one frame at a time, returning each request's codec codes in
/// the SAME order as `reqs`. See the module doc for what this does and does
/// not deliver.
pub fn run_batch(talker_path: &str, mtp_path: &str, sp: &TtsSpecials, reqs: Vec<(Prompt, GenOpts)>) -> Vec<Vec<u32>> {
    let mut sessions: Vec<Session> =
        reqs.into_iter().map(|(prompt, opts)| Session::new(talker_path, mtp_path, sp, prompt, opts)).collect();
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

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    /// `TalkerConfig::tiny()` with a real-scale vocab: `sample_cb0` always
    /// suppresses the top-1024 vocab entries as the reference's
    /// `suppress_tokens` window, so a genuinely tiny vocab (23) underflows
    /// `vocab - 1024`. Every other dimension stays tiny for test speed.
    fn talker_test_cfg() -> crate::config::TalkerConfig {
        crate::config::TalkerConfig { vocab: 1100, ..crate::config::TalkerConfig::tiny() }
    }

    /// Build a real (tiny synthetic) Talker+MTP checkpoint pair on disk, the
    /// same shape `run_batch` loads via `CpuTalker::load`/`CpuMtp::load`.
    /// The Talker's base decoder blocks ARE a `qwen3::Qwen` on disk
    /// (`TalkerConfig::to_qwen`, `qwen3::init_weights`; `CpuTalker::load`
    /// reads them back via `TalkerConfig::from_qwen`), plus the
    /// Talker-specific extras `qwen3::init_weights` knows nothing about
    /// (`tok`/`lm_head`/`text_projection.*`/`text_embedding`, normally added
    /// by `import::import_talker`) hand-added here with the right shapes -
    /// values don't matter, `run_batch`'s own tests never exercise text
    /// prompting (they build `Prompt` directly from random embeddings).
    fn synthetic_checkpoints(dir: &std::path::Path, seed: u64) -> (String, String) {
        let tcfg = talker_test_cfg();
        let qcfg = tcfg.to_qwen(32);
        let mut init = qwen3::init_weights(&qcfg, seed);

        let mut rng = Rng::new(seed ^ 0x7A1E);
        let mut normal = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_gaussian() as f32) * 0.02).collect() };
        let (d, vocab, th) = (tcfg.d_model as usize, tcfg.vocab as usize, tcfg.text_hidden_size as usize);
        let inter = th; // no config field for this; derived from the tensor shapes at load time
        init.insert("tok.weight".to_string(), normal(vocab * d));
        init.insert("lm_head.weight".to_string(), normal(vocab * d));
        init.insert("text_projection.fc1.weight".to_string(), normal(inter * th));
        init.insert("text_projection.fc1.bias".to_string(), normal(inter));
        init.insert("text_projection.fc2.weight".to_string(), normal(d * inter));
        init.insert("text_projection.fc2.bias".to_string(), normal(d));
        init.insert("text_embedding.weight".to_string(), normal(tcfg.text_vocab_size as usize * th));

        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = init.into_iter().map(|(k, v)| (k, vec![v.len() as u64], v)).collect();
        let talker_path = dir.join("talker.safetensors").to_str().unwrap().to_string();
        checkpoint::save(&talker_path, qcfg.to_json(), &tensors);

        let mcfg = crate::config::MtpConfig::tiny();
        let mtp_path = dir.join("mtp.safetensors").to_str().unwrap().to_string();
        save_synthetic_mtp(&mcfg, &mtp_path, seed ^ 0x5A5A);

        (talker_path, mtp_path)
    }

    /// `MtpModel` has no `save`; hand-write the checkpoint `MtpModel::
    /// load_inference` expects (the same tensor set `new_synthetic_on`
    /// fills in-memory, here written to disk instead).
    fn save_synthetic_mtp(cfg: &crate::config::MtpConfig, path: &str, seed: u64) {
        let mut rng = Rng::new(seed);
        let mut normal = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_gaussian() as f32) * 0.02).collect() };
        let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        for (name, numel) in crate::mtp::MtpModel::decoder_param_list(cfg) {
            tensors.push((name, vec![numel as u64], normal(numel)));
        }
        let (nres, e, d, v) = (cfg.n_residual() as usize, cfg.embedding_dim as usize, cfg.d_model as usize, cfg.vocab as usize);
        for i in 0..nres {
            tensors.push((format!("codec_embedding.{i}.weight"), vec![(v * e) as u64], normal(v * e)));
            tensors.push((format!("lm_head.{i}.weight"), vec![(v * d) as u64], normal(v * d)));
        }
        checkpoint::save(path, cfg.to_json(), &tensors);
    }

    fn tiny_prompt(d: usize, n_prefix: usize, n_trail: usize, rng_seed: u64) -> Prompt {
        let mut rng = Rng::new(rng_seed);
        let mut g = |n: usize| (0..n).map(|_| (rng.next_gaussian() as f32) * 0.1).collect::<Vec<f32>>();
        Prompt { embeds: g(n_prefix * d), trailing: g(n_trail * d), tts_pad: g(d) }
    }

    fn tiny_specials() -> TtsSpecials {
        TtsSpecials {
            tts_bos: 0,
            tts_eos: 1,
            tts_pad: 2,
            codec_nothink: 3,
            codec_think: 4,
            codec_think_bos: 5,
            codec_think_eos: 6,
            codec_pad: 7,
            codec_bos: 8,
            // `MtpConfig::tiny()`'s vocab is 23; keep EOS comfortably inside it
            // and outside `sample_cb0`'s suppressed top-1024 window is moot at
            // this vocab size (the window is wider than the whole vocab), so
            // it never gets masked back off - fine for these tests, which
            // never rely on hitting EOS to end a session.
            // Inside `sample_cb0`'s suppressed top-1024 window ([vocab-1024,
            // vocab) = [76, 1100) at this test's vocab=1100) - mirrors the
            // real model, where EOS lives inside that window and `min_new`
            // genuinely gates whether it's reachable. An id outside the
            // window (e.g. a small one, as this used to be) is NEVER
            // suppressed regardless of `min_new`, which isn't what these
            // tests are meant to exercise.
            codec_eos: 1050,
            lang: std::collections::HashMap::new(),
            spk_id: std::collections::HashMap::new(),
        }
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
        let dir = std::env::temp_dir().join(format!("qwen3tts-batch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (talker_path, mtp_path) = synthetic_checkpoints(&dir, 3);
        let sp = tiny_specials();
        let d = talker_test_cfg().d_model as usize;

        let opts_a = GenOpts { max_frames: 5, min_new: 1, seed: 11, temperature: 0.8, top_k: 0, ..GenOpts::default() };
        let opts_b = GenOpts { max_frames: 8, min_new: 1, seed: 22, temperature: 0.8, top_k: 0, ..GenOpts::default() };
        let prompt_a = tiny_prompt(d, 3, 2, 101);
        let prompt_b = tiny_prompt(d, 4, 1, 202);

        let batched = run_batch(&talker_path, &mtp_path, &sp, vec![(prompt_a.clone(), opts_a.clone()), (prompt_b.clone(), opts_b.clone())]);
        assert_eq!(batched.len(), 2);

        let mut cpu_a = CpuTalker::load(&talker_path);
        let mut mtp_a = CpuMtp::load(&mtp_path);
        let alone_a = crate::pipeline::generate_codes_cached(&mut cpu_a, &mut mtp_a, &sp, &prompt_a, &opts_a);

        let mut cpu_b = CpuTalker::load(&talker_path);
        let mut mtp_b = CpuMtp::load(&mtp_path);
        let alone_b = crate::pipeline::generate_codes_cached(&mut cpu_b, &mut mtp_b, &sp, &prompt_b, &opts_b);

        assert_eq!(batched[0], alone_a, "interleaving must not change request A's own codes");
        assert_eq!(batched[1], alone_b, "interleaving must not change request B's own codes");
        // The two requests have different max_frames (5 vs 8) - a genuinely
        // ragged batch, not a coincidentally-equal-length one.
        assert_ne!(batched[0].len(), batched[1].len(), "test setup must actually exercise raggedness");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A short request must actually stop consuming rounds once it finishes,
    /// not silently keep stepping alongside a longer one still in flight.
    #[test]
    fn a_finished_request_drops_out_of_rotation() {
        if gpu_disabled() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("qwen3tts-batch-test2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (talker_path, mtp_path) = synthetic_checkpoints(&dir, 5);
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
            vec![(tiny_prompt(d, 2, 1, 1), short), (tiny_prompt(d, 2, 1, 2), long)],
        );
        // `MtpConfig::tiny()`'s num_code_groups (4: cb0 + 3 residuals), not the
        // real model's 16 - a per-frame width, not a fixed constant.
        let group_width = crate::config::MtpConfig::tiny().num_code_groups as usize;
        assert_eq!(out[0].len() / group_width, 1, "the 1-frame request must stop at exactly 1 frame");
        assert_eq!(out[1].len() / group_width, 6, "the 6-frame request must still complete in full");

        std::fs::remove_dir_all(&dir).ok();
    }
}
