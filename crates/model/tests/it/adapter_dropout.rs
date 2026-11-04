// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LoRA dropout (M11): the device path (a trainer with the adapter's input
//! activation `x` available) is where this can be implemented - none of
//! this workspace's device model crates wire a dropout config through yet
//! (tracked as open work), so this milestone's host-substrate scope is
//! narrower: `TargetHp::dropout` must never be a silently-ignored field.
//! `LoraPair::project` consumes a dense `dL/dW_eff` and never sees `x`, so
//! there is no dW to project once masking makes the adapter non-linear in
//! a single step - a nonzero dropout on this path is a hard, loud error.

use model::adapter::{AdapterKind, TargetHp, TargetSpec};
use model::lora::LoraPair;

#[test]
#[should_panic(expected = "dropout requires an activation-aware path")]
fn a_nonzero_dropout_on_the_host_path_is_a_hard_error_not_a_silent_no_op() {
    let mut hp = TargetHp::new(2, 4.0);
    hp.dropout = 0.5;
    let _ = LoraPair::new(TargetSpec::whole(4, 4), hp, &mut || 0.01);
}

#[test]
fn zero_dropout_is_unaffected() {
    let hp = TargetHp::new(2, 4.0);
    assert_eq!(hp.dropout, 0.0, "TargetHp::new's default must stay 0.0 - every existing adapter relies on it");
    let _ = LoraPair::new(TargetSpec::whole(4, 4), hp, &mut || 0.01);
}
