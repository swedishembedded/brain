// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A Qwen3 dense checkpoint is served at the floating precision it is stored
//! in unless its caller chose a tier: the tier is read off the checkpoint
//! itself (`qwen3::stored_weight_tier`), and the default rule
//! (`qwen3::footprint::default_weight_tier`) keeps it, falling to int8 only
//! where the stored weights and the serving headroom do not fit the card.

use qwen3::export::{export_hf, HfDtype, HfExport};
use qwen3::{Dtype, QwenConfig};

#[test]
fn a_checkpoint_reports_the_floating_tier_its_linears_are_stored_in() {
    let cfg = QwenConfig::tiny();
    let weights = qwen3::init_weights(&cfg, 7);
    for (stored, want) in [(HfDtype::Bf16, Dtype::BF16), (HfDtype::F16, Dtype::F16), (HfDtype::F32, Dtype::F32)] {
        let dir = std::env::temp_dir().join(format!("brain-stored-tier-{stored:?}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        export_hf(&weights, &cfg, &dir, &HfExport { dtype: stored, ..Default::default() }, &mut || true).unwrap();
        let tier = qwen3::stored_weight_tier(dir.to_str().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(tier, Ok(want), "a {stored:?} checkpoint");
    }
}

#[test]
fn the_default_tier_of_a_checkpoint_is_the_one_it_is_stored_in() {
    let card = Some(96u64 << 30);
    for stored in [Dtype::BF16, Dtype::F16, Dtype::F32] {
        assert_eq!(qwen3::footprint::default_weight_tier(&QwenConfig::qwen3_1_7b(), stored, card), stored, "a small {stored:?} checkpoint");
        assert_eq!(qwen3::footprint::default_weight_tier(&QwenConfig::qwen3_8b(), stored, card), stored, "an 8B {stored:?} checkpoint on a card that holds it");
    }
    // A stored encoding that is no floating tier (FP8, a packed int8 file) keeps the fp32 default.
    assert_eq!(qwen3::footprint::default_weight_tier(&QwenConfig::qwen3_1_7b(), Dtype::F8E4M3, card), Dtype::F32);
}

#[test]
fn int8_is_the_default_only_where_the_stored_tier_does_not_fit() {
    let big = QwenConfig::qwen3_8b();
    let bf16_need = qwen3::footprint::held_weight_bytes(&big, Dtype::BF16) + qwen3::footprint::SERVE_HEADROOM_BYTES;
    let fp32_need = qwen3::footprint::held_weight_bytes(&big, Dtype::F32) + qwen3::footprint::SERVE_HEADROOM_BYTES;
    assert!(bf16_need < fp32_need);
    let tier = |stored, bytes| qwen3::footprint::default_weight_tier(&big, stored, bytes);
    assert_eq!(tier(Dtype::BF16, Some(bf16_need)), Dtype::BF16, "a card that holds the bf16 weights and the headroom serves them bf16");
    assert_eq!(tier(Dtype::BF16, Some(bf16_need - 1)), Dtype::I8);
    assert_eq!(tier(Dtype::F32, Some(bf16_need)), Dtype::I8, "the fp32 weights of the same model do not fit there");
    assert_eq!(tier(Dtype::F32, Some(fp32_need)), Dtype::F32);
    assert_eq!(tier(Dtype::BF16, None), Dtype::I8, "no device known: the size rule stands");
    assert_eq!(qwen3::footprint::default_weight_tier(&QwenConfig::qwen3_1_7b(), Dtype::BF16, Some(1 << 30)), Dtype::BF16, "below 6B parameters the default is never int8");
}
