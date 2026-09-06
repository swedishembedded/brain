// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P14's SECOND, separate gate (deliberately not folded into
//! `distill_gradcheck.rs`, which only ever exercises a small `K`): with
//! `K = vocab` (every vocab id present, teacher probabilities summing to
//! 1.0 over the WHOLE vocab), `rl::objective::distill`'s K-sparse
//! weighted-CE-pass mechanism must reproduce the textbook, independently
//! defined `KL(q‖p) = Σ_v q_v·(ln q_v − ln p_v)` to fp32 tolerance - not
//! merely pass a gradient-direction check. That is the proof the K-sparse
//! path is a genuine special case of full KL (the keystone identity's
//! `q_K → q` as `K → vocab`), not a different approximation that happens to
//! also gradcheck.
//!
//! The reference KL below is computed a SECOND, independent way (plain
//! `ln`/softmax arithmetic on `Qwen::logits_all`'s raw logits) - never by
//! calling back into `rl::objective::distill` itself - so this test cannot
//! pass by the two computations sharing a bug.

use data::rng::Rng;
use qwen3::{Qwen, QwenConfig};
use rl::objective::distill::{neg_entropy, pass_arrays, TopK};

const SEQ_LEN: usize = 5;

#[test]
fn distill_topk_at_k_equals_vocab_reproduces_exact_kl() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let vocab = cfg.vocab as usize;
    let init = qwen3::init_weights(&cfg, 5);
    let mut m = Qwen::new(cfg, 1, SEQ_LEN as u32, &init);
    m.enable_weighted_loss();

    let tokens: Vec<u32> = (0..SEQ_LEN as u32).map(|i| (i * 3 + 1) % vocab as u32).collect();

    // The student's own softmax `p`, read directly off `logits_all` and
    // reduced to a probability with plain `exp`/`ln` - independent of
    // anything `rl::objective::distill` computes.
    let logits = m.logits_all(&tokens);
    let mut host_p = Vec::with_capacity(SEQ_LEN);
    for i in 0..SEQ_LEN {
        let z = &logits[i * vocab..(i + 1) * vocab];
        let max_z = z.iter().cloned().fold(f32::MIN, f32::max);
        let exp: Vec<f32> = z.iter().map(|&zi| (zi - max_z).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        host_p.push(exp.iter().map(|&e| e / sum_exp).collect::<Vec<f32>>());
    }

    // A synthetic, strictly-positive, full-vocab teacher distribution `q`
    // per position (`K = vocab`, `ids` the identity permutation) - the
    // `Σ_k q_k = 1` requirement the keystone identity leans on, satisfied by
    // construction here via normalization.
    let mut rng = Rng::new(99);
    let mut teacher = Vec::with_capacity(SEQ_LEN);
    for _ in 0..SEQ_LEN {
        let raw: Vec<f32> = (0..vocab).map(|_| rng.next_f32() + 0.05).collect();
        let s: f32 = raw.iter().sum();
        let probs: Vec<f32> = raw.iter().map(|&r| r / s).collect();
        teacher.push(TopK { ids: (0..vocab as u32).collect(), probs });
    }

    // Reference: the textbook definition, computed WITHOUT going through
    // `pass_arrays`/`neg_entropy` at all.
    let mut reference = 0.0f32;
    for (i, p_row) in host_p.iter().enumerate() {
        for (v, &p) in p_row.iter().enumerate() {
            let q = teacher[i].probs[v];
            reference += q * (q.ln() - p.ln());
        }
    }
    reference /= SEQ_LEN as f32;

    // The mechanism under test: literally `vocab` weighted-CE forward passes
    // (no backward needed - this test checks the accumulated SCALAR, the
    // gradient side is `distill_gradcheck.rs`'s job), summed exactly as
    // `rl::objective::distill::DistillTopK::micro_step` sums them.
    let mut total = 0.0f32;
    for k in 0..vocab {
        let (targets, weights, _count) = pass_arrays(&teacher, k);
        m.set_batch(&tokens, &targets);
        m.write_weights(&weights);
        total += m.forward();
    }
    let entropy: f32 = teacher.iter().map(neg_entropy).sum();
    let realized = (total + entropy) / SEQ_LEN as f32;

    assert!((realized - reference).abs() < 1e-2, "K=vocab realized KL {realized} != reference KL {reference}");
}
