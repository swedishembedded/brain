// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-correctness gates for looped decoders for its clients. If your team needs
// expertise in proving a training graph exact then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Gradient-check gate for Inkling 2's training graph (full fine-tune and LoRA) on the tiny looped configuration.

use gradcheck::{check_inkling2, check_inkling2_lora, check_inkling2_shared, Report};

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8.0e-2;

fn gate(report: Report, what: &str) {
    report.print();
    let fails = report.failures(ATOL, RTOL);
    assert!(fails.is_empty(), "{what}: gradient check failed for {:?}", fails.iter().map(|c| (&c.param, c.analytic, c.numeric)).collect::<Vec<_>>());
    let dead = report.dead_gradients();
    assert!(dead.is_empty(), "{what}: silently-dead gradients {:?}", dead.iter().map(|c| (&c.param, c.analytic, c.numeric)).collect::<Vec<_>>());
}

fn seed() -> u64 {
    let seed = data::rng::Lcg::random().next_u32() as u64;
    eprintln!("inkling2 gradcheck seed {seed}");
    seed
}

#[test]
fn inkling2_every_parameter_of_a_full_fine_tune() {
    let s = seed();
    gate(check_inkling2(s), &format!("tiny inkling2, seed {s}"));
}

/// The core's parameters are read once per pass and the message queries are gathered from two published tensors:
/// entry by entry, so a pass or a row the backward drops cannot hide behind a contraction.
#[test]
fn inkling2_shared_core_accumulates_every_pass_entry_by_entry() {
    let s = seed();
    gate(check_inkling2_shared(s, 6), &format!("tiny inkling2 shared core, seed {s}"));
}

#[test]
fn inkling2_every_adapter_tensor_of_a_lora_build() {
    let s = seed();
    gate(check_inkling2_lora(s), &format!("tiny inkling2 LoRA, seed {s}"));
}
