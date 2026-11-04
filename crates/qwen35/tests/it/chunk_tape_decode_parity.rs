// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A short chunk round (the shape a speculative VERIFY pass runs) computes
//! EXACTLY what the same tokens compute as plain decode steps - every output
//! bit, and the recurrent and KV state it leaves behind.
//!
//! Swedish Embedded AB implements lossless speculative decoding for quantised
//! LLM serving for its clients. If your team needs expertise in making a
//! verify pass agree with plain decode to the last bit then you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! Why bit-exact and not "close": the weights and the activations of this
//! model are int8, so any difference between two tapes in a late layer moves
//! a rounding decision, and every later layer amplifies it to the size of the
//! quantisation noise itself (about one logit on the real checkpoint). A
//! speculative decoder that verifies on a tape that is merely close accepts or
//! rejects tokens by a different arithmetic than the one it promises to
//! reproduce, and flips every near-tie between the two.
//!
//! The hidden state after the round is compared with the decode tape's own
//! and so is each of the next two decode steps, which cannot agree unless the
//! round left the KV cache, the Gated-DeltaNet state and the conv window
//! exactly as token-by-token decode would have.
//!
//! The config has two full-attention layers (indices 3 and 7) so a mis-masked
//! row in the first is visible to the second, and 128-wide linear-attention
//! heads, the only width the native recurrent kernel serves.
//!
//! CUDA backend only: that is where the decode tape runs the fused native
//! kernels this claim is about. Skips elsewhere.

use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

/// Prompt tokens decoded one by one before the round, so the round starts from
/// a real recurrent state and KV depth rather than from zero.
const WARM: usize = 5;

fn cfg() -> Qwen35Config {
    Qwen35Config {
        n_layers: 8,
        d_model: 640,
        intermediate_size: 1280,
        block_size: 64,
        max_position_embeddings: 64,
        head_dim: 256,
        mrope_section: [11, 11, 10],
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        ..Qwen35Config::tiny_i8()
    }
}

fn build(gpu: Gpu) -> (Qwen35, Vec<u32>) {
    let cfg = cfg();
    let init = qwen35::init::init_weights(&cfg, 11);
    let m = Qwen35::new_on_dt(gpu, cfg.clone(), 1, cfg.block_size, &init, &TierPolicy::uniform(Dtype::I8));
    let tokens = (0..48u32).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    (m, tokens)
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn assert_round_matches_decode(rows: usize) {
    let Ok(gpu) = Gpu::try_new_cuda(pipelines()) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let (m, tokens) = build(gpu);
    let (warm, round) = (&tokens[..WARM], &tokens[WARM..WARM + rows]);
    let after = &tokens[WARM + rows..WARM + rows + 2];

    m.reset_decode_cache();
    for &t in warm {
        m.step(t);
    }
    let decode_last = round.iter().map(|&t| m.step(t)).last().expect("a non-empty round");
    let decode_next: Vec<Vec<f32>> = after.iter().map(|&t| m.step(t)).collect();

    m.reset_decode_cache();
    for &t in warm {
        m.step(t);
    }
    let chunk_last = m.prefill_chunked(round, rows as u32);
    let chunk_next: Vec<Vec<f32>> = after.iter().map(|&t| m.step(t)).collect();

    assert!(same_bits(&chunk_last, &decode_last), "a {rows}-row chunk round's last hidden state differs from {rows} decode steps");
    for (i, (c, d)) in chunk_next.iter().zip(&decode_next).enumerate() {
        assert!(same_bits(c, d), "decode step {i} after a {rows}-row chunk round differs: the round left different state behind");
    }
}

#[test]
fn a_one_row_chunk_round_is_bit_identical_to_a_decode_step() {
    assert_round_matches_decode(1);
}

#[test]
fn a_two_row_chunk_round_is_bit_identical_to_two_decode_steps() {
    assert_round_matches_decode(2);
}

#[test]
fn a_three_row_chunk_round_is_bit_identical_to_three_decode_steps() {
    assert_round_matches_decode(3);
}

#[test]
fn an_eight_row_chunk_round_is_bit_identical_to_eight_decode_steps() {
    assert_round_matches_decode(8);
}

/// Past the widest group the native int8 GEMV serves in one pass, a round runs
/// as several groups in order; the seams between them must not show.
#[test]
fn a_nine_row_chunk_round_is_bit_identical_to_nine_decode_steps() {
    assert_round_matches_decode(9);
}

#[test]
fn a_thirty_two_row_chunk_round_is_bit_identical_to_thirty_two_decode_steps() {
    assert_round_matches_decode(32);
}
