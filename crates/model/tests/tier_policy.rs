// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`model::ops::TierPolicy`] semantics, ahead of any model wiring it in
//! (M24). A per-leaf generalization of the single [`backend_api::DType`]
//! `qwen3::Qwen::new_shard_dt` takes - `Qwen35::new_impl_on`'s `i8: bool`
//! cannot express "Q4 on the MLP, F32 on the GDN decay/beta gates", and this
//! is the value that replaces it. Deliberately NOT a fourth `QTier` copy
//! (`wan::block::QTier`, `ltxv::block::QTier` are two private re-spellings of
//! the same two-variant subset of `DType` this policy is built on).

use gpu_core::select::Dtype as DType;
use model::ops::TierPolicy;

#[test]
fn uniform_returns_its_default_for_any_name() {
    let p = TierPolicy::uniform(DType::Q4);
    assert_eq!(p.want("blocks.0.mlp.gate.weight"), DType::Q4);
    assert_eq!(p.want("blocks.63.linear_attn.in_proj_a.weight"), DType::Q4);
    assert_eq!(p.want(""), DType::Q4);
}

#[test]
fn with_narrows_by_substring_match() {
    let p = TierPolicy::uniform(DType::Q4).with(&["in_proj_a", "in_proj_b"], DType::F32);
    assert_eq!(p.want("blocks.0.linear_attn.in_proj_a.weight"), DType::F32);
    assert_eq!(p.want("blocks.0.linear_attn.in_proj_b.weight"), DType::F32);
    assert_eq!(p.want("blocks.0.mlp.gate.weight"), DType::Q4, "unrelated leaves keep the default");
}

#[test]
fn later_rule_wins_over_an_earlier_broader_one() {
    // A broad rule then a narrower override, declaration order - the last
    // rule whose pattern matches wins, so a caller can carve an exception
    // out of an earlier blanket rule without reordering it.
    let p = TierPolicy::uniform(DType::F32).with(&["proj"], DType::Q4).with(&["in_proj_a"], DType::I8);
    assert_eq!(p.want("blocks.0.linear_attn.in_proj_a.weight"), DType::I8, "the later, narrower rule must win");
    assert_eq!(p.want("blocks.0.linear_attn.in_proj_b.weight"), DType::Q4, "still caught by the broader rule");
    assert_eq!(p.want("blocks.0.norm.weight"), DType::F32, "no rule matches -- falls through to the default");
}

#[test]
fn quantizes_anything_reflects_whether_any_leaf_ever_leaves_f32() {
    assert!(!TierPolicy::uniform(DType::F32).quantizes_anything());
    assert!(TierPolicy::uniform(DType::Q4).quantizes_anything());
    assert!(TierPolicy::uniform(DType::F32).with(&["mlp"], DType::Q4).quantizes_anything());
}

#[test]
fn parse_and_describe_round_trip() {
    for p in [
        TierPolicy::uniform(DType::F32),
        TierPolicy::uniform(DType::I8),
        TierPolicy::uniform(DType::Q4),
        TierPolicy::uniform(DType::Q4).with(&["in_proj_a", "in_proj_b"], DType::F32),
    ] {
        let s = p.describe();
        let back = TierPolicy::parse(&s).unwrap_or_else(|e| panic!("parse({s:?}) failed: {e}"));
        assert_eq!(back, p, "parse(describe(p)) must equal p, got describe = {s:?}");
    }
}

#[test]
fn parse_rejects_an_unknown_tier_name_loudly() {
    assert!(TierPolicy::parse("q9").is_err(), "an unknown bare tier must not silently default to f32");
    assert!(TierPolicy::parse("q4,in_proj_a=fp32").is_err(), "an unknown tier in a rule must also be rejected");
}

#[test]
fn parse_accepts_the_documented_grammar() {
    let p = TierPolicy::parse("q4,in_proj_a.weight=f32,in_proj_b.weight=f32").expect("valid grammar");
    assert_eq!(p.want("blocks.0.linear_attn.in_proj_a.weight"), DType::F32);
    assert_eq!(p.want("blocks.0.mlp.gate.weight"), DType::Q4);
}

/// Substring matching is a real hazard, not a hypothetical: `"up.weight"`
/// would also match a hypothetical `"group.weight"`. This test does not
/// exercise the hazard case (there is no `"group.weight"` leaf in this
/// crate), it pins the mechanism the hazard follows from, so a future reader
/// changing `want`'s match rule sees this break instead of a silent widen.
#[test]
fn substring_matching_is_not_anchored_to_a_full_path_segment() {
    let p = TierPolicy::uniform(DType::F32).with(&["up"], DType::Q4);
    assert_eq!(p.want("blocks.0.mlp.up.weight"), DType::Q4, "the intended match");
    assert_eq!(p.want("blocks.0.mlp.group.weight"), DType::Q4, "an unintended substring match -- by design, documented");
}
