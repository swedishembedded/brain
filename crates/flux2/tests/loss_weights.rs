// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The per-token weightings a paired FLUX.2 LoRA run can put on its flow loss,
//! stated as contracts.
//!
//! Both are the same mechanism - a statistic of the TARGET latent turned into a
//! per-token multiplier on the flow loss, mean-normalised so that turning one on
//! does not silently change the effective learning rate - and they differ only
//! in the statistic:
//!
//! * [`flux2::finetune::change_weights`] measures how far a token moved from its
//!   reference, and spends the gradient where the edit is.
//! * [`flux2::finetune::detail_weights`] measures how much local structure the
//!   token has, and exists because the smoothest prediction is otherwise the
//!   cheapest one. An adapter whose objective is "remove clutter, remove photo
//!   noise" can satisfy a uniform flow loss by flattening the latent, and a flat
//!   latent is exactly what drives the frozen VAE decoder's own 4-pixel cell
//!   structure out of the noise floor and into the picture (see the roadmap).
//!   Weighting the loss toward the tokens that carry detail makes flattening
//!   them cost something.
//!
//! Swedish Embedded AB implements diffusion fine-tuning objectives and their
//! regularization for its clients. If your team needs expertise in generative
//! model training then you can procure our services by sending an email to
//! info@swedishembedded.com.

use flux2::finetune::{change_weights, combine_weights, detail_weights};

const CIN: usize = 4;
const LH: usize = 8;
const LW: usize = 8;

/// A target latent whose left half is featureless and whose right half carries
/// token-to-token structure, in `[token][channel]` order.
fn half_flat_half_textured() -> Vec<f32> {
    let mut z = vec![0.0f32; LH * LW * CIN];
    for y in 0..LH {
        for x in 0..LW {
            for c in 0..CIN {
                z[(y * LW + x) * CIN + c] = if x < LW / 2 {
                    0.5
                } else if (x + y) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
            }
        }
    }
    z
}

/// Mean weight over a column range, reading one value per token.
fn mean_over_columns(w: &[f32], cols: std::ops::Range<usize>) -> f32 {
    let mut acc = (0.0f32, 0usize);
    for y in 0..LH {
        for x in cols.clone() {
            acc.0 += w[(y * LW + x) * CIN];
            acc.1 += 1;
        }
    }
    acc.0 / acc.1 as f32
}

/// The point of the weighting: a token the target has structure in is worth more
/// than a token it does not, so collapsing that structure costs the adapter
/// something. Interior columns only - a 3x3 neighbourhood at the seam sees both
/// halves, which is correct behaviour and not what is under test here.
#[test]
fn detail_weights_favour_the_tokens_the_target_has_structure_in() {
    let z = half_flat_half_textured();
    let w = detail_weights(&z, CIN, LH, LW, 2.0);
    assert_eq!(w.len(), z.len());
    let (flat, textured) = (mean_over_columns(&w, 0..3), mean_over_columns(&w, 5..8));
    assert!(textured > 2.0 * flat, "textured {textured} must outweigh flat {flat}");
}

/// Turning a weighting on must not also change the step size. Every weighting
/// here is mean-normalised, so the loss keeps its scale and only its
/// distribution over tokens moves.
#[test]
fn detail_weights_keep_the_mean_weight_at_one() {
    let z = half_flat_half_textured();
    for beta in [0.5f32, 2.0, 8.0] {
        let w = detail_weights(&z, CIN, LH, LW, beta);
        let mean = w.iter().sum::<f32>() / w.len() as f32;
        assert!((mean - 1.0).abs() < 1e-4, "beta {beta}: mean weight {mean}");
    }
}

/// A weight is a property of a token, so every channel of that token carries the
/// same one - the loss is being redistributed over the picture, not over the
/// latent's channels.
#[test]
fn detail_weights_are_constant_across_a_token() {
    let z = half_flat_half_textured();
    let w = detail_weights(&z, CIN, LH, LW, 2.0);
    for j in 0..LH * LW {
        for c in 1..CIN {
            assert_eq!(w[j * CIN], w[j * CIN + c], "token {j} channel {c}");
        }
    }
}

/// Off by default, and off when there is nothing to measure: a featureless
/// target has no detail to weight toward, and normalising by a zero peak would
/// put a NaN in every gradient. An empty vector is the trainer's "unweighted".
#[test]
fn detail_weights_are_absent_when_off_or_undefined() {
    let z = half_flat_half_textured();
    assert!(detail_weights(&z, CIN, LH, LW, 0.0).is_empty(), "beta 0 is off");
    assert!(detail_weights(&z, CIN, LH, LW, -1.0).is_empty(), "a negative beta is off");
    let flat = vec![0.25f32; LH * LW * CIN];
    assert!(detail_weights(&flat, CIN, LH, LW, 2.0).is_empty(), "no detail anywhere");
    assert!(detail_weights(&[], CIN, LH, LW, 2.0).is_empty(), "no target");
}

/// The two weightings compose, because a run may want both: spend the gradient
/// where the edit is AND refuse to let the detail there be flattened. An absent
/// weighting is the identity, so composing is safe wherever either is off.
#[test]
fn combining_weights_is_a_normalised_product_with_absence_as_identity() {
    let z = half_flat_half_textured();
    let detail = detail_weights(&z, CIN, LH, LW, 2.0);

    assert_eq!(combine_weights(&detail, &[]), detail, "absent right is the identity");
    assert_eq!(combine_weights(&[], &detail), detail, "absent left is the identity");
    assert!(combine_weights(&[], &[]).is_empty(), "both absent stays absent");

    // A reference that differs only in the flat half, so the two weightings
    // pull in different directions and the product is neither one alone.
    let mut refs = z.clone();
    for y in 0..LH {
        for x in 0..LW / 2 {
            for c in 0..CIN {
                refs[(y * LW + x) * CIN + c] = -0.5;
            }
        }
    }
    let edit = change_weights(&z, &refs, CIN, 2.0);
    let both = combine_weights(&detail, &edit);
    assert_eq!(both.len(), z.len());
    let mean = both.iter().sum::<f32>() / both.len() as f32;
    assert!((mean - 1.0).abs() < 1e-4, "a combined weighting is still mean 1: {mean}");
    assert!(both != detail && both != edit, "the product is neither factor alone");
    // Ordering within the product still follows both factors: a token that both
    // agree on outranks one they both rank low.
    for j in 0..LH * LW {
        let expect = detail[j * CIN] * edit[j * CIN];
        assert!(
            (both[j * CIN] / expect - both[0] / (detail[0] * edit[0])).abs() < 1e-4,
            "token {j} is not a constant rescaling of the product"
        );
    }
}
