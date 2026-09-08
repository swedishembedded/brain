// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! rsLoRA (`alpha/sqrt(r)` instead of `alpha/r`) on the generic adapter
//! substrate. Gated on rel_l2/an exact ratio, never on cosine alone: cosine
//! is scale-invariant, so it cannot distinguish a correctly-scaled delta
//! from one off by a uniform factor (a dropped `alpha`, a wrong `sqrt`) -
//! the second assertion below documents that blindness rather than relying
//! on it.

use data::rng::Lcg;
use model::adapter::{AdapterKind, TargetHp, TargetSpec};
use model::lora::LoraPair;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
fn rslora_scales_the_delta_by_exactly_sqrt_r_at_rank_4() {
    // rank 4 is chosen deliberately: sqrt(4) == 2 exactly in fp32, so the
    // ratio assertion below is an exact comparison, not a tolerance fudge.
    //
    // A/B are fixed to the SAME concrete values in both adapters (never
    // pushed through Adam here) so this test isolates exactly one thing:
    // whether `delta_into` applies `TargetHp::scale()`. Comparing two
    // Adam-trained adapters instead would conflate this with Adam's own
    // nonlinearity - at step 1 in particular, AdamW's update is
    // `lr * sign(g)` regardless of |g|, so a differently-scaled gradient
    // does NOT produce a cleanly rescaled trajectory even though the
    // scale is applied correctly at every step.
    let rank = 4usize;
    let alpha = 8.0f32;
    let out = 6;
    let inn = 5;
    let spec = TargetSpec::whole(out, inn);

    let mut fill_rng = Lcg::new(7);
    let a_fixed: Vec<f32> = fill_rng.vec_scaled(rank * inn, 0.02);
    let b_fixed: Vec<f32> = fill_rng.vec_scaled(out * rank, 0.02);
    let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> =
        vec![(".lora_a", vec![rank, inn], a_fixed.clone()), (".lora_b", vec![out, rank], b_fixed.clone())];
    let get = |suffix: &str| -> Option<(Vec<usize>, Vec<f32>)> {
        tensors.iter().find(|(s, _, _)| *s == suffix).map(|(_, shape, data)| (shape.clone(), data.clone()))
    };

    let hp_plain = TargetHp::new(rank, alpha);
    assert!(!hp_plain.rank_stabilized);
    let mut plain = LoraPair::new(spec, hp_plain, &mut || 0.0);
    plain.load_tensors(&get).expect("load plain");

    let mut hp_rs = TargetHp::new(rank, alpha);
    hp_rs.rank_stabilized = true;
    let mut rs = LoraPair::new(spec, hp_rs, &mut || 0.0);
    rs.load_tensors(&get).expect("load rs");

    let mut d_plain = vec![0.0f32; out * inn];
    plain.delta_into(1.0, &mut d_plain);
    let mut d_rs = vec![0.0f32; out * inn];
    rs.delta_into(1.0, &mut d_rs);

    // The actual gate: the ratio is exactly 2.0 (alpha/sqrt(r) vs alpha/r
    // at r=4 differ by exactly a factor of sqrt(r) = 2).
    let mut max_ratio_err = 0.0f32;
    for (p, r) in d_plain.iter().zip(d_rs.iter()) {
        if *p != 0.0 {
            max_ratio_err = max_ratio_err.max((r / p - 2.0).abs());
        }
    }
    assert!(max_ratio_err < 1e-6, "max ratio error {max_ratio_err} - rsLoRA did not scale by exactly sqrt(r)");

    // Document why this could NOT have been caught by cosine alone: a
    // uniformly-rescaled vector has cosine 1.0 with the original, exactly.
    let cos = cosine(&d_plain, &d_rs);
    assert!((cos - 1.0).abs() < 1e-9, "cosine {cos} - two uniformly-scaled deltas must appear identical to a cosine-only gate");
}

#[test]
fn rslora_round_trips_through_target_hp_scale() {
    let hp = TargetHp { rank: 9, alpha: 18.0, rank_stabilized: true, dropout: 0.0, lr_ratio: 1.0, freeze_a: false };
    assert_eq!(hp.scale(), 18.0 / 3.0);

    let plain = TargetHp { rank_stabilized: false, ..hp };
    assert_eq!(plain.scale(), 18.0 / 9.0);
}
