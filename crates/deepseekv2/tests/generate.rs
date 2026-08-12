// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Composed-loop (generation) parity for the DeepSeek-OCR MoE decoder**,
//! against llama.cpp running the very same shipped Q8_0 weights.
//!
//! `tests/parity.rs` proves ONE forward over a FIXED two-token sequence matches
//! the reference layer by layer. That is a snapshot: it says nothing about
//! whether the RoPE position argument advances correctly as tokens are
//! appended, whether the causal mask stays right as the sequence grows, or
//! whether feeding a freshly-generated token back in produces the right next
//! distribution. This test closes exactly that gap - it drives
//! [`deepseekv2::DeepseekV2::generate_greedy`] for eight steps and demands the
//! ids match, position by position.
//!
//! ## The reference, and why it is not a leap of faith
//!
//! Captured from a locally-built llama.cpp on the real
//! `DeepSeek-OCR-Q8_0.gguf` (reproducible; run twice, identical):
//!
//! ```text
//! llama-cli -m DeepSeek-OCR-Q8_0.gguf -p "Hello" -n 8 --temp 0 --top-k 1 -no-cnv --single-turn
//! ```
//!
//! It emits `!How can I help you today?`, which `llama-tokenize --ids` maps back
//! to `[3, 4117, 588, 342, 1694, 440, 4316, 33]` (8 ids for `-n 8`, minus that
//! tool's own auto-prepended BOS). Prefixed with the prompt tokenization for
//! `"Hello"` - `[0, 19923]`, 0 being BOS, the same ids `tests/parity.rs`'s
//! fixture carries as data - that gives [`REFERENCE`].
//!
//! The anchor: index 2 of that sequence (`3`) is precisely what
//! `parity.rs::real_lm_decoder_matches_llamacpp` **already** proves
//! independently, by comparing brain's `result_output` argmax after `[0, 19923]`
//! against the captured reference logits. So the multi-step chain starts from a
//! step this repo has verified against a byte-level tensor dump, not from an
//! unverified claim about a CLI's stdout.
//!
//! ## What a mismatch would mean
//!
//! Steps 3..=9 are the ones with no per-tensor golden behind them, and they are
//! the ones that can only fail *for loop reasons*: the weights and the graph are
//! the same objects `parity.rs` gates at cosine ≥ 0.9995, so a divergence here
//! is a position/mask/append defect, not a numeric one. Which is why the
//! assertion names the FIRST divergent index and, on failure, re-runs that step
//! and prints the top-5 candidates with their logits - a run that missed by
//! 0.001 of a logit and one that picked something unrelated are very different
//! bugs and the failure message should say which happened.
//!
//! Cost, backend and skip-when-absent behaviour are `tests/common/real_lm.rs`'s;
//! the decode is `O(T²)` recompute (nine forwards over ≤ 10 tokens), which at
//! this size is a rounding error next to loading 12 GB of weights. Once the
//! recompute path agrees with the reference, this test also drives
//! [`deepseekv2::model::DeepseekV2::generate_greedy_kv`] over the SAME prompt
//! and demands the SAME ids - the real-weight half of the KV-cache decode gate
//! (the fast-lane half is `src/model.rs`'s `generate_greedy_kv_matches_recompute`).

/// The model-store lookup, the one-off fp32 expansion, the CPU-backend pin and
/// the inference build - shared with `tests/parity.rs`.
#[path = "common/real_lm.rs"]
mod real_lm;

/// llama.cpp's greedy continuation of `"Hello"` under this checkpoint, prompt
/// included: `result[i + 1]` is the greedy argmax after `result[0..=i]`.
const REFERENCE: [u32; 10] = [0, 19923, 3, 4117, 588, 342, 1694, 440, 4316, 33];
/// The tokenization of `"Hello"` (`0` is this checkpoint's BOS).
const PROMPT: usize = 2;

#[ignore = "real 2.9 B-parameter checkpoint: ~12 GB resident and a one-off ~12 GB fp32 expansion on disk. Two of these in parallel would exhaust any machine that can run one, so it stays out of the fast lane. `make test/slow`, or `cargo test --release -p brain-deepseekv2 --test generate -- --nocapture --test-threads=1`."]
#[test]
fn real_lm_greedy_decode_matches_llamacpp() {
    let n_new = (REFERENCE.len() - PROMPT) as u32;
    println!("== deepseekv2 composed-loop parity ({PROMPT}-token prompt + {n_new} greedy steps)");
    brain_testutil::mem("start");
    let Some(m) = real_lm::open(REFERENCE.len() as u32) else { return };
    brain_testutil::mem("decoder built (inference)");

    let got = m.generate_greedy(&REFERENCE[..PROMPT], n_new);
    brain_testutil::mem("generation done");
    assert_eq!(got.len(), REFERENCE.len(), "generate_greedy returned {} ids, want {}", got.len(), REFERENCE.len());

    for (i, (&g, &w)) in got.iter().zip(REFERENCE.iter()).enumerate() {
        let what = if i < PROMPT { "prompt" } else { "step  " };
        println!("  [{i:>2}] {what}  brain {g:>6}  reference {w:>6}  {}", if g == w { "ok" } else { "MISMATCH" });
    }

    // The prompt is copied, not predicted, so a mismatch there is a different
    // (and much dumber) bug than a decode divergence - and it is what would make
    // the diagnostic below run over an empty prefix.
    assert_eq!(&got[..PROMPT], &REFERENCE[..PROMPT], "the prompt did not come back verbatim");

    let Some(bad) = got.iter().zip(REFERENCE.iter()).position(|(g, w)| g != w) else {
        println!("  all {} positions agree", REFERENCE.len());

        // The KV-cache decode gate on REAL weights: `generate_greedy_kv` must
        // reproduce `generate_greedy`'s own (llama.cpp-verified) ids exactly.
        // `tests::generate_greedy_kv_matches_recompute` in `src/model.rs`
        // already gates this at toy dims; this is the one real-weight check
        // that the O(1)-per-token attention decode step, the batched-forward
        // cache prefill and the real MoE router/expert weights all compose
        // correctly at production scale, not just on a synthetic fixture.
        let kv = m.generate_greedy_kv(&REFERENCE[..PROMPT], n_new);
        assert_eq!(kv, got, "KV-cache decode diverged from the O(T^2) recompute on the real weights");
        println!("  KV-cache decode matches recompute: {kv:?}");
        return;
    };

    // Diagnose before dying: what did the model actually rank at the step that
    // first went wrong, and by how much? `generate_greedy` fed it exactly
    // `got[..bad]`, and everything before `bad` matched, so that prefix IS the
    // reference's own - re-running it reproduces the deciding distribution.
    let vocab = m.cfg.vocab() as usize;
    let logits = m.logits_all(&got[..bad]);
    let last = &logits[logits.len() - vocab..];
    let mut top: Vec<usize> = (0..vocab).collect();
    top.sort_by(|&a, &b| last[b].total_cmp(&last[a]));
    println!("  top-5 at step {bad}: {:?}", top[..5].iter().map(|&i| (i, last[i])).collect::<Vec<_>>());
    println!("  reference id {} scored {} (rank {})", REFERENCE[bad], last[REFERENCE[bad] as usize], top.iter().position(|&i| i as u32 == REFERENCE[bad]).unwrap_or(usize::MAX));
    panic!(
        "greedy decode diverges at index {bad}: brain {} (logit {}), reference {} (logit {}) -- \
         the single-forward stage parity passes, so this is the decode LOOP (position/mask/append), not the numerics",
        got[bad],
        last[got[bad] as usize],
        REFERENCE[bad],
        last[REFERENCE[bad] as usize]
    );
}
