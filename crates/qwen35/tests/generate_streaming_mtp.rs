// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real end-to-end gate for `qwen35::stream::generate`'s `use_mtp: true`
//! path (`crate::stream::generate_mtp_accelerated`) against its own plain
//! serial baseline (`use_mtp: false`), on the real
//! `Qwen/Qwen3.8-27B-FP8` checkpoint.
//!
//! Two things this file proves, both against REAL generated text (not a
//! synthetic stand-in):
//!
//! 1. **Exact-match determinism** - for the SAME prompt at greedy settings
//!    (`temperature = 0.0`), the MTP-accelerated path's final token sequence
//!    must be BYTE-IDENTICAL to the plain path's own output. This is the
//!    correctness proof: the MTP head only ever supplies a CANDIDATE token
//!    that a subsequent real forward pass independently verifies (this
//!    crate's own `crate::stream` module doc, "the per-pass confirm/advance/
//!    speculate decode loop") - it must never change WHAT text comes out,
//!    only how many streaming passes it costs to get there.
//! 2. **Real pass-count reduction** - the actual number of full 64-layer
//!    streaming passes (`stream_all_layers` calls, `generate_with_stats`'s
//!    own return value) each path issues for the SAME number of new tokens.
//!    Reported as a real measured ratio, not assumed to hit exactly 2x (how
//!    often the real MTP head's own guess is right on real text is an
//!    empirical fact about this checkpoint, not a guarantee).
//!
//! **This is extremely slow and that is expected, not a bug** - same
//! ~3-4-minutes-class-per-pass reality `generate_streaming.rs`'s own doc
//! already documents (this milestone's own two throughput fixes, landed
//! after that file's doc was written, brought a full streaming pass down
//! from the original ~75-minutes-class number to ~3-4 minutes; both paths
//! below pay that same per-pass cost). Budget realistically for this to
//! take 30-60+ minutes wall-clock (several passes across two paths) - that
//! is expected, not a hang.
//!
//! Self-skips loudly (never silently) without `BRAIN_QWEN35_DIR`, its
//! `tokenizer.json`, or `mtp.safetensors`. Run with:
//!
//! ```text
//! BRAIN_QWEN35_DIR=[path/to/qwen3.8] \
//!     cargo test -p brain-qwen35 --test generate_streaming_mtp -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::Instant;

use qwen35::config::Qwen35Config;
use qwen35::stream::generate_with_stats;

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("BRAIN_QWEN35_DIR").map(PathBuf::from)
}

#[test]
#[ignore]
fn mtp_accelerated_greedy_decode_matches_the_plain_path_byte_for_byte_and_reports_real_pass_counts() {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset (set it to a downloaded Qwen/Qwen3.8-27B-FP8 dir to run this - see this file's own doc)");
        return;
    };
    let tokenizer = dir.join("tokenizer.json");
    if !tokenizer.exists() {
        brain_testutil::skip_unavailable(&format!("{} not present under BRAIN_QWEN35_DIR", tokenizer.display()));
        return;
    }
    if !dir.join("mtp.safetensors").exists() {
        brain_testutil::skip_unavailable("mtp.safetensors missing under BRAIN_QWEN35_DIR - needed for the use_mtp=true path");
        return;
    }
    let cfg = Qwen35Config::qwen38_27b();
    for l in 0..cfg.n_layers as usize {
        if !dir.join(format!("layers-{l}.safetensors")).exists() {
            brain_testutil::skip_unavailable(&format!("layers-{l}.safetensors missing under BRAIN_QWEN35_DIR - need all 64 shards for this gate"));
            return;
        }
    }
    if !dir.join("outside.safetensors").exists() {
        brain_testutil::skip_unavailable("outside.safetensors missing under BRAIN_QWEN35_DIR - need embed_tokens/lm_head/norm for this gate");
        return;
    }

    let prompt = "The capital of France is";
    let max_new = 4usize;
    let window_budget = 4u32;
    let seed = 20260819;

    brain_testutil::mem("generate_with_stats: before plain (use_mtp=false)");
    let t0 = Instant::now();
    let (plain_text, plain_passes) =
        generate_with_stats(&dir, &cfg, &tokenizer, prompt, max_new, 0.0, 0, 1.0, window_budget, seed, false)
            .unwrap_or_else(|e| panic!("generate_with_stats (plain, use_mtp=false): {e}"));
    let plain_elapsed = t0.elapsed();
    brain_testutil::mem("generate_with_stats: after plain");

    println!("=== qwen35::stream::generate_with_stats - PLAIN (use_mtp=false, greedy) ===");
    println!("prompt:   {prompt:?}");
    println!("output:   {plain_text:?}");
    println!("passes:   {plain_passes}");
    println!("elapsed:  {:.1} min", plain_elapsed.as_secs_f64() / 60.0);

    brain_testutil::mem("generate_with_stats: before MTP-accelerated (use_mtp=true)");
    let t1 = Instant::now();
    let (mtp_text, mtp_passes) =
        generate_with_stats(&dir, &cfg, &tokenizer, prompt, max_new, 0.0, 0, 1.0, window_budget, seed, true)
            .unwrap_or_else(|e| panic!("generate_with_stats (MTP-accelerated, use_mtp=true): {e}"));
    let mtp_elapsed = t1.elapsed();
    brain_testutil::mem("generate_with_stats: after MTP-accelerated");

    println!("=== qwen35::stream::generate_with_stats - MTP-ACCELERATED (use_mtp=true, greedy) ===");
    println!("prompt:   {prompt:?}");
    println!("output:   {mtp_text:?}");
    println!("passes:   {mtp_passes}");
    println!("elapsed:  {:.1} min", mtp_elapsed.as_secs_f64() / 60.0);

    let ratio = plain_passes as f64 / mtp_passes as f64;
    println!("=== gate 2: real pass-count reduction ===");
    println!("plain passes:          {plain_passes}");
    println!("MTP-accelerated passes: {mtp_passes}");
    println!("measured ratio (plain/MTP): {ratio:.3}x");

    // Gate 1: exact-match determinism - the whole point of this file.
    assert_eq!(mtp_text, plain_text, "MTP-accelerated greedy output diverged from the plain serial path - this is a CORRECTNESS bug, not a speed regression");

    // Gate 2: real forward progress and a real (not necessarily 2x) pass
    // reduction - report the true number, never assume it.
    assert!(!plain_text.is_empty(), "plain generation produced empty text");
    assert!(mtp_passes <= plain_passes, "MTP-accelerated path must never issue MORE passes than the plain path for the same max_new");
    assert!(mtp_passes >= 1, "MTP-accelerated path must make real forward progress (at least 1 pass)");
}
