// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA-FA (M10): `A` is frozen at its random init; only `B` trains. On the
//! host substrate this is structural, not a promise a caller has to
//! remember to keep - a `LoraPair` built with `freeze_a` drops `A`'s Adam
//! moments entirely, so a step that (incorrectly) touched `A` would panic
//! rather than silently train it.

use data::rng::Lcg;
use model::adapter::{AdapterKind, TargetHp, TargetSpec};
use model::lora::{LoraPair, Pair};

#[test]
fn lora_fa_leaves_a_bit_identical_while_b_moves() {
    let (out, inn, r) = (6, 5, 2);
    let spec = TargetSpec::whole(out, inn);
    let mut hp = TargetHp::new(r, 4.0);
    hp.freeze_a = true;

    let mut init = Lcg::new(3);
    let mut init_fn = || init.signed() * 0.02;
    let mut lp = LoraPair::new(spec, hp, &mut init_fn);

    // Read A back before training - LoraPair exposes it via `pair()`.
    let a_before = lp.pair().a.clone();

    let mut rng = Lcg::new(7);
    for t in 1..=20u64 {
        let dw: Vec<f32> = (0..out * inn).map(|_| rng.signed()).collect();
        let g = lp.project(&dw);
        lp.step(&g, 0.05, t);
    }

    assert_eq!(lp.pair().a, a_before, "LoRA-FA must leave A bit-identical to its init");
    assert!(lp.pair().b.iter().any(|&x| x != 0.0), "B must have moved - otherwise this test is vacuous");
}

#[test]
#[should_panic]
fn a_frozen_pair_panics_if_something_calls_adam_a_directly() {
    // The structural backstop: freeze_a empties A's moments, so a caller
    // that bypasses the AdapterKind::step guard and calls adam_a anyway
    // hits an index-out-of-bounds panic immediately, not a silent no-op.
    let mut pair = Pair::new(4, 3, 2, || 0.01);
    pair.freeze_a();
    assert!(pair.a_is_frozen());
    pair.adam_a(&[0.1; 2 * 3], 0.01, 1);
}

#[test]
fn freezing_b_instead_would_be_permanently_dead_document_the_asymmetry() {
    // B = 0 at init (Pair::new's own contract) and freeze_a leaves B
    // trainable - so LoRA-FA's learnable subspace is exactly span(A) from
    // the first step. The REVERSE (freeze B, leave A trainable) would never
    // move the delta away from zero: dA = scale*B^T*dW is exactly zero
    // whenever B is exactly zero, so a permanently-zero B makes every dA
    // exactly zero too - a permanently dead adapter, not merely a
    // differently-parameterized one. This test exists to write that down
    // as an explicit, checked fact rather than tribal knowledge.
    let (out, inn, r) = (4, 3, 2);
    let b_zero = vec![0.0f32; out * r];
    let a_random = vec![0.37f32; r * inn];
    let pair = Pair::from_ab(out, inn, r, a_random, b_zero);
    let dw = vec![1.0f32; out * inn];
    let (da, _db) = pair.project(&dw, 1.0);
    assert!(da.iter().all(|&x| x == 0.0), "dA must be exactly zero whenever B is exactly zero");
}
