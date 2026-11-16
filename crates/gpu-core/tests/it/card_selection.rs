// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A test opens only the cards the selection names.
//!
//! Swedish Embedded AB implements test suites that share a multi-GPU node with other people's jobs for its clients.
//! If your team needs expertise in keeping GPU tests on the cards they were given, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Tests that enumerated "every GPU" (`devices::gpus()`, a card count, `gpu0..gpuN`) opened gpu0 on a node where
//! `BRAIN_DEVICE=gpu3` named the one card that was theirs. `gpu_core::testgpu::{cards, stage_cards, card}` are the
//! selection-honouring forms every such test uses.

use gpu_core::testgpu::{admit_cards, card, cards};

#[test]
fn a_multi_card_test_takes_selected_cards_and_refuses_others() {
    let selected = [2u32, 3];
    assert_eq!(admit_cards(None, 2, &selected), Ok(vec![2, 3]), "the first selected cards, not gpu0 and gpu1");
    assert_eq!(admit_cards(None, 1, &selected), Ok(vec![2]));
    assert_eq!(admit_cards(Some(&[3]), 1, &selected), Ok(vec![3]), "a requested card the selection names");
    let refused = admit_cards(Some(&[0, 3]), 2, &selected).unwrap_err();
    assert!(refused.contains("gpu0") && refused.contains("--device"), "{refused}");
    let short = admit_cards(None, 2, &[3]).unwrap_err();
    assert!(short.contains("needs 2"), "{short}");
    assert!(admit_cards(None, 1, &[]).is_err(), "no selected card: nothing to open");
}

#[test]
fn a_card_outside_the_selection_is_refused_by_index() {
    let outside = cards().iter().map(|d| d.index).max().map_or(0, |i| i + 1).max(64);
    let refused = std::panic::catch_unwind(|| card(outside)).expect_err("an unselected card must not be handed out");
    let msg = refused.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains(&format!("gpu{outside}")), "{msg}");
    for d in cards() {
        assert_eq!(card(d.index).index, d.index, "a selected card is handed out");
    }
}
