// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 declares its parts; the engine places them. This gates the
//! declaration - that the costs are architecture-derived and that the
//! automatic text-encoder decision reproduces the layout that used to have to
//! be typed by hand as `BRAIN_FLUX2_TE_DEVICE=gpu1:i8`.
//!
//! Swedish Embedded AB implements automatic multi-device model placement for
//! its clients. If your team needs expertise in fitting a diffusion pipeline
//! across the cards a machine actually has, you can procure our services by
//! sending an email to info@swedishembedded.com.

use flux2::pipeline::{dit_bytes, te_bytes};
use flux2::Precision;

const GIB: f64 = (1u64 << 30) as f64;
fn gib(b: u64) -> f64 {
    b as f64 / GIB
}

/// The measured constraint this whole mechanism exists for: on a 24 GiB card
/// an f32 truncated Qwen3-8B text encoder does NOT fit, and the int8 one
/// does. That is what makes the automatic decision pick int8 for klein-9b -
/// and it must fall out of the architecture, not out of a remembered number.
///
/// The tap layers are [9, 18, 27], so the shard keeps layers [0, 27).
#[test]
fn the_nine_b_text_encoder_needs_int8_to_fit_a_twenty_four_gib_card() {
    let te = qwen3::QwenConfig::qwen3_8b();
    let f32_bytes = te_bytes(&te, 27, 512, false);
    let i8_bytes = te_bytes(&te, 27, 512, true);
    assert!(gib(f32_bytes) > 24.0, "an f32 truncated Qwen3-8B must not be claimed to fit a 24 GiB card: {:.1} GiB", gib(f32_bytes));
    assert!(gib(i8_bytes) < 16.0, "the int8 shard must fit beside a driver context: {:.1} GiB", gib(i8_bytes));
    assert!(i8_bytes * 2 < f32_bytes, "int8 must be several times smaller: {:.1} vs {:.1} GiB", gib(i8_bytes), gib(f32_bytes));
}

/// ...and the 4B pipeline's encoder does fit in f32, so its conditioning is
/// not silently downgraded to the lossy tier by a rule written for the 9B.
#[test]
fn the_four_b_text_encoder_still_fits_a_card_in_f32() {
    let te = qwen3::QwenConfig::qwen3_4b();
    assert!(gib(te_bytes(&te, 27, 512, false)) < 22.0, "{:.1} GiB", gib(te_bytes(&te, 27, 512, false)));
}

/// A truncated shard must cost strictly less than a whole encoder - the whole
/// point of truncating it - so placement sees the difference.
///
/// The `layers = 0` probe is what makes this a gate on the WEIGHT filter and
/// not merely on the scratch term (which is trivially proportional to
/// `layers`): with no block resident, only the embedding is on the card, and
/// a cost model that ignored the truncation would still report the whole
/// stack's weights here.
#[test]
fn truncation_is_visible_to_the_cost_model() {
    let te = qwen3::QwenConfig::qwen3_8b();
    let whole = te_bytes(&te, te.n_layers as usize, 512, false);
    let cut = te_bytes(&te, 27, 512, false);
    assert!(cut < whole, "truncated {:.1} GiB must be less than whole {:.1} GiB", gib(cut), gib(whole));
    let embed_only = te_bytes(&te, 0, 0, false);
    assert!(gib(embed_only) < 4.0, "with no block resident only the embedding is on the card: {:.1} GiB", gib(embed_only));
    assert!(embed_only * 5 < whole, "the block stack must dominate: {:.1} vs {:.1} GiB", gib(embed_only), gib(whole));
}

/// The DiT cost follows the architecture: 9B is bigger than 4B, int8 is
/// smaller than f32, and a bigger joint sequence costs more scratch. A cost
/// model that ignored any of these would place a 9B run as if it were a 4B
/// one.
#[test]
fn the_dit_cost_follows_the_architecture_and_the_numeric_tier() {
    let (four, nine) = (flux2::Flux2Config::klein_4b(), flux2::Flux2Config::klein_9b());
    let n = 512 + 3072;
    assert!(dit_bytes(&nine, Precision::F32, n, 1) > dit_bytes(&four, Precision::F32, n, 1), "9B must cost more than 4B");
    assert!(dit_bytes(&nine, Precision::Int8, n, 1) < dit_bytes(&nine, Precision::F32, n, 1), "int8 must cost less than f32");
    assert!(dit_bytes(&nine, Precision::Int8, 2 * n, 1) > dit_bytes(&nine, Precision::Int8, n, 1), "a longer joint sequence costs more scratch");
    // A 9B int8 DiT plus its scratch is most of a 24 GiB card - which is why
    // the text encoder cannot join it. Bracketed, not pinned to a
    // measurement: this is a placement input, not a performance claim.
    let real = dit_bytes(&nine, Precision::Int8, n, 1);
    assert!((8.0..20.0).contains(&gib(real)), "int8 9B DiT budget out of the plausible band: {:.1} GiB", gib(real));
}
