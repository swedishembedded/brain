// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Convergence integration tests.
//!
//! These are *learnability* guards: they train the real GPT engine end-to-end
//! (forward + backprop + AdamW, on whatever device `BRAIN_DEVICE` selects) on
//! tiny synthetic tasks that a correct implementation *must* be able to learn,
//! and assert the loss actually drops to a task-appropriate floor.
//!
//! Why these exist: a plateaued loss can come from a genuinely hard task *or*
//! from a silent regression in a kernel, the masking path, the optimizer or the
//! gradient wiring. The finite-difference gradient checker catches per-op
//! numerical errors, but not "the whole training loop fails to learn". These
//! tests catch the latter by pinning the expected behavior of the full loop on
//! tasks whose answer is known:
//!
//! * `cycle`   — memorize a fixed repeating sequence. Exercises embeddings,
//!               positional encoding, softmax, cross-entropy and the optimizer
//!               with no attention reasoning required. Floor ~= 0.
//! * `copy`    — `S=S`, loss masked to the answer. Exercises the loss-masking
//!               path *and* a copy circuit through attention. If masking kept
//!               the wrong positions this could not converge.
//! * `reverse` — `S=rev(S)`, masked. A position-dependent copy; harder.
//!
//! Scaling: `loss_improves_with_model_size` encodes a scaling-law expectation —
//! a too-small model (d_model 8, 1 layer) lacks the capacity to fit the copy
//! task and plateaus near the marginal-entropy floor, while a larger model
//! drives the loss far below it on the same task and step budget. This guards
//! the property that *added capacity actually translates into lower loss*, which
//! a capacity-capping kernel/optimizer regression would break even while the
//! single-config tests still pass.
//!
//! Thresholds are calibrated against measured runs on the CPU (Cranelift JIT)
//! backend and set with wide margins, since the reported value is a single
//! (noisy) final-step training loss. All tests are skipped when
//! `MOE_SKIP_GPU_TESTS` is set (same gate as the in-crate training test), so the
//! suite stays runnable without an accelerator, and are sized to run in a few
//! minutes on CPU / seconds on a real GPU.

use std::path::{Path, PathBuf};

use data::binio::{self, Meta};
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use gpt::{train, GptConfig, TrainOpts};

/// Skip the whole test when no accelerator is wanted.
fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// Fresh temp dir for one dataset, cleaned up front. `tag` keeps concurrent
/// tests from sharing a directory.
fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("brain_conv_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create temp dataset dir");
    d
}

/// Write a char-level dataset (`meta.json` + `train.bin`/`val.bin`) from a
/// corpus string, building the vocab from the corpus exactly like `prepare`.
fn write_corpus(dir: &Path, corpus: &str) {
    let tok = CharTokenizer::from_corpus(corpus);
    let ids = tok.encode(corpus);
    let split = (ids.len() as f64 * 0.9) as usize;
    binio::write_u16_bin(&dir.join("train.bin"), &ids[..split]).unwrap();
    binio::write_u16_bin(&dir.join("val.bin"), &ids[split..]).unwrap();
    let meta = Meta { vocab_size: tok.itos().len(), itos: tok.itos().to_vec() };
    std::fs::write(dir.join("meta.json"), meta.to_json()).unwrap();
}

/// Corpus: a fixed sequence repeated `reps` times. Deterministic next token.
fn cycle_corpus(reps: usize) -> String {
    "0123456789\n".repeat(reps)
}

/// Corpus of `n` lines `S=f(S)\n`, each char of `S` drawn from `alphabet`.
/// `reverse` selects copy (`f = id`) vs reverse (`f = rev`).
fn mapped_corpus(n: usize, len: usize, alphabet: &str, reverse: bool, seed: u64) -> String {
    let chars: Vec<char> = alphabet.chars().collect();
    let mut rng = Rng::new(seed);
    let mut out = String::with_capacity(n * (2 * len + 2));
    for _ in 0..n {
        let s: String = (0..len)
            .map(|_| chars[rng.gen_range_inclusive(0, chars.len() as i64 - 1) as usize])
            .collect();
        out.push_str(&s);
        out.push('=');
        if reverse {
            out.extend(s.chars().rev());
        } else {
            out.push_str(&s);
        }
        out.push('\n');
    }
    out
}

/// `TrainOpts` for a short run. `mask` enables per-line answer masking.
/// `decay_iters` is `2*steps` so the cosine LR does not crater to its floor
/// before the model has had a chance to fit (we stop mid-schedule).
fn train_opts(steps: u32, block: u32, lr: f32, mask: Option<char>) -> TrainOpts {
    TrainOpts {
        steps,
        batch_size: 32,
        block_size: block,
        lr,
        warmup: 20,
        decay_iters: steps * 2,
        eval_interval: 0,
        seed: 1234,
        mask_before: mask,
        mask_per_line: mask.is_some(),
        ..Default::default()
    }
}

/// `vocab = 0` is the conventional "infer from dataset's meta.json" placeholder;
/// `train()` overwrites it. `d_ff` follows the 4x convention.
fn cfg(block: u32, layers: u32, d_model: u32, heads: u32) -> GptConfig {
    GptConfig { vocab: 0, block_size: block, n_layers: layers, d_model, n_heads: heads, d_ff: d_model * 4 }
}

#[test]
fn engine_memorizes_cyclic_sequence() {
    if skip() {
        return;
    }
    let dir = tmpdir("cycle");
    write_corpus(&dir, &cycle_corpus(4000));
    let block = 16;
    let (init, final_loss) =
        train(&dir, cfg(block, 2, 32, 4), &train_opts(120, block, 3e-3, None), None).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    // Pure memorization of a deterministic cycle (marginal entropy ln(11)~=2.40);
    // a correct engine drives this essentially to zero (measured ~0.005).
    assert!(
        final_loss < 0.10,
        "cycle memorization failed to converge: {init:.4} -> {final_loss:.4} (expected < 0.10)"
    );
}

#[test]
fn engine_learns_copy_through_mask() {
    if skip() {
        return;
    }
    let dir = tmpdir("copy");
    // S=S, length 4 over {a,b,c,d}; answer marginal entropy ln(4) ~= 1.386.
    write_corpus(&dir, &mapped_corpus(6000, 4, "abcd", false, 7));
    let block = 16;
    let (init, final_loss) =
        train(&dir, cfg(block, 2, 64, 4), &train_opts(400, block, 3e-3, Some('=')), None).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    // Must fall far below the ln(4) marginal floor (measured ~0.58) — only
    // possible if the mask keeps exactly the answer tokens and the model learns
    // the copy circuit.
    assert!(
        final_loss < 0.90,
        "masked copy failed to converge: {init:.4} -> {final_loss:.4} (expected < 0.90, marginal ~1.386)"
    );
}

#[test]
fn engine_learns_reverse_through_mask() {
    if skip() {
        return;
    }
    let dir = tmpdir("reverse");
    // S=rev(S), length 5 over {a,b,c,d,e}; answer marginal entropy ln(5) ~= 1.609.
    write_corpus(&dir, &mapped_corpus(8000, 5, "abcde", true, 11));
    let block = 16;
    let (init, final_loss) =
        train(&dir, cfg(block, 3, 64, 4), &train_opts(400, block, 3e-3, Some('=')), None).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    // Position-dependent copy: harder than plain copy, but must still drop well
    // under the ln(5) marginal floor (measured ~0.92).
    assert!(
        final_loss < 1.30,
        "masked reverse failed to converge: {init:.4} -> {final_loss:.4} (expected < 1.30, marginal ~1.609)"
    );
}

#[test]
fn loss_improves_with_model_size() {
    if skip() {
        return;
    }
    // Same copy task, same step budget, two capacities. A d_model=8 / 1-layer
    // model cannot fit the copy task and plateaus near the ln(4)~=1.386 marginal
    // floor (measured ~1.24); a d_model=64 / 2-layer model drives it far lower
    // (measured ~0.6). Asserts that capacity actually buys lower loss — a
    // scaling-law sanity check a capacity-capping regression would violate.
    let dir = tmpdir("sizescale");
    write_corpus(&dir, &mapped_corpus(6000, 4, "abcd", false, 7));
    let block = 16;
    let opts = train_opts(300, block, 3e-3, Some('='));

    let (_, small) = train(&dir, cfg(block, 1, 8, 1), &opts, None).unwrap();
    let (_, large) = train(&dir, cfg(block, 2, 64, 4), &opts, None).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    assert!(
        large + 0.30 < small,
        "added capacity did not lower loss: tiny(d8,1L) {small:.4} vs larger(d64,2L) {large:.4} \
         (expected the larger model at least 0.30 lower)"
    );
}
