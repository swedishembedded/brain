// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec gates for [`residency::plan`] - the pure, hardware-free policy that
//! answers "which device does each part of this model go on".
//!
//! Swedish Embedded AB implements automatic multi-device model placement for
//! its clients. If your team needs expertise in fitting large models across
//! the accelerators a machine actually has, you can procure our services by
//! sending an email to info@swedishembedded.com.

use residency::budget::Budgets;
use residency::plan::{plan, Affinity, Part};
use residency::{Device, MemCost};

const GIB: u64 = 1 << 30;

/// Two cards, one already occupied by somebody else. A model that fits only
/// the free card must be placed there - this is the reported bug, in one
/// assertion: brain used to take card 0 unconditionally and OOM.
#[test]
fn a_model_lands_on_the_card_that_can_hold_it_not_on_card_zero() {
    let mut b = Budgets::new();
    // `total` here is what is actually free right now (the CLI probes live
    // free VRAM), so an 18 GiB foreign allocation on gpu0 shows up as 5 GiB.
    b.set(Device::Gpu(0), 5 * GIB, 0);
    b.set(Device::Gpu(1), 24 * GIB, 0);
    b.set(Device::Cpu, 128 * GIB, 0);

    let parts = [Part::new("dit", MemCost::new(16 * GIB, 0))];
    let p = plan(&parts, &b).expect("16 GiB fits the free card");
    assert_eq!(p.of("dit"), Some(Device::Gpu(1)), "must pick the card with room, not card 0");
}

/// A pipeline of several parts spreads across the cards it has, instead of
/// piling onto one. `Apart` is the declaration a model makes; the engine does
/// the placing.
#[test]
fn a_two_part_pipeline_spreads_across_two_cards() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 2 * GIB);
    b.set(Device::Gpu(1), 24 * GIB, 2 * GIB);
    b.set(Device::Cpu, 128 * GIB, 0);

    let parts = [
        Part::new("dit", MemCost::new(16 * GIB, 0)).apart(),
        Part::new("te", MemCost::new(9 * GIB, 0)).apart(),
    ];
    let p = plan(&parts, &b).expect("both fit when spread");
    let (dit, te) = (p.of("dit").unwrap(), p.of("te").unwrap());
    assert_ne!(dit, te, "an int8 9B DiT and its text encoder must not co-reside: {p:?}");
    assert!(matches!(dit, Device::Gpu(_)) && matches!(te, Device::Gpu(_)), "both parts belong on cards: {p:?}");
}

/// Parts that must share a device say so, and the planner honours it even
/// when another device is emptier.
#[test]
fn parts_declared_together_share_one_device() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 20 * GIB, 0);
    b.set(Device::Gpu(1), 24 * GIB, 0);
    b.set(Device::Cpu, 128 * GIB, 0);

    let parts = [
        Part::new("te", MemCost::new(9 * GIB, 0)).apart(),
        Part::new("dit", MemCost::new(16 * GIB, 0)).apart(),
        // The VAE decodes the DiT's own latents: same card, always.
        Part::new("vae", MemCost::new(2 * GIB, 0)).with("dit"),
    ];
    let p = plan(&parts, &b).expect("plan");
    assert_eq!(p.of("vae"), p.of("dit"), "vae must follow the dit: {p:?}");
    assert_ne!(p.of("te"), p.of("dit"), "te was declared apart from the dit: {p:?}");
}

/// Successive parts see the bytes their predecessors took. Three 16 GiB parts
/// cannot be talked onto two 24 GiB cards.
#[test]
fn placement_charges_the_budget_so_a_third_part_cannot_be_double_booked() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    b.set(Device::Gpu(1), 24 * GIB, 0);

    let ok = [Part::new("a", MemCost::new(16 * GIB, 0)), Part::new("b", MemCost::new(16 * GIB, 0))];
    let p = plan(&ok, &b).expect("two fit on two cards");
    assert_ne!(p.of("a"), p.of("b"), "the second must not be booked on top of the first: {p:?}");

    let too_many = [
        Part::new("a", MemCost::new(16 * GIB, 0)),
        Part::new("b", MemCost::new(16 * GIB, 0)),
        Part::new("c", MemCost::new(16 * GIB, 0)),
    ];
    assert!(plan(&too_many, &b).is_err(), "48 GiB of cards cannot hold 48 GiB + 16 GiB");
}

/// No GPU at all (a CI box, `BRAIN_DEVICE=cpu`): a weight-holding part falls
/// back to the host RAM tier rather than failing.
#[test]
fn with_no_gpu_the_host_tier_takes_the_model() {
    let mut b = Budgets::new();
    b.set(Device::Cpu, 128 * GIB, 0);
    let parts = [Part::new("dit", MemCost::new(16 * GIB, 0))];
    let p = plan(&parts, &b).expect("cpu fallback");
    assert_eq!(p.of("dit"), Some(Device::Cpu));
}

/// When nothing fits, the refusal names the part, its size, and every
/// device's free capacity - a legible refusal instead of a raw driver OOM.
#[test]
fn an_impossible_plan_refuses_legibly_instead_of_oom() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 2 * GIB);
    b.set(Device::Gpu(1), 5 * GIB, 2 * GIB);
    b.set(Device::Cpu, 4 * GIB, 0);

    let parts = [Part::new("dit-fp32", MemCost::new(40 * GIB, 0))];
    let e = plan(&parts, &b).expect_err("40 GiB fits nothing here");
    let msg = e.to_string();
    for want in ["dit-fp32", "40", "gpu0", "gpu1", "22", "3"] {
        assert!(msg.contains(want), "refusal must name {want:?}; got:\n{msg}");
    }
}

/// A single part with no declared cost (the "no preference, just give me a
/// card" default every `Gpu::new` takes) still gets the emptiest card.
#[test]
fn an_unsized_part_still_gets_the_emptiest_card() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 2 * GIB, 0);
    b.set(Device::Gpu(1), 24 * GIB, 0);
    let p = plan(&[Part::unsized_("model")], &b).expect("plan");
    assert_eq!(p.of("model"), Some(Device::Gpu(1)));
}

/// Affinity is declarative, not positional: `with` works when the anchor is
/// declared *after* the follower.
#[test]
fn an_anchor_may_be_declared_after_the_part_that_follows_it() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    b.set(Device::Gpu(1), 24 * GIB, 0);
    let parts = [
        Part::new("vae", MemCost::new(2 * GIB, 0)).with("dit"),
        Part::new("dit", MemCost::new(16 * GIB, 0)),
    ];
    let p = plan(&parts, &b).expect("plan");
    assert_eq!(p.of("vae"), p.of("dit"));
    assert!(matches!(Affinity::Any, Affinity::Any));
}

/// `Apart` must be a real constraint, not a coincidence of most-free-wins.
/// On a machine whose emptiest card could hold BOTH parts, greedy placement
/// co-locates them; the declaration is what keeps them apart. This is the
/// measured FLUX.2 case in miniature - the two parts fitting arithmetically
/// is not the same as them running.
#[test]
fn apart_separates_parts_the_greedy_rule_would_co_locate() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 40 * GIB, 0);
    b.set(Device::Gpu(1), 20 * GIB, 0);
    let parts = [
        Part::new("dit", MemCost::new(16 * GIB, 0)).apart(),
        Part::new("te", MemCost::new(9 * GIB, 0)).apart(),
    ];
    let p = plan(&parts, &b).expect("plan");
    assert_eq!(p.of("dit"), Some(Device::Gpu(0)), "{p:?}");
    assert_eq!(p.of("te"), Some(Device::Gpu(1)), "apart must move it off the emptiest card: {p:?}");
}

/// ...and `Apart` is a preference, not a demand: one card means one card.
#[test]
fn apart_still_places_when_there_is_only_one_device() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 40 * GIB, 0);
    let parts = [
        Part::new("dit", MemCost::new(16 * GIB, 0)).apart(),
        Part::new("te", MemCost::new(9 * GIB, 0)).apart(),
    ];
    let p = plan(&parts, &b).expect("one card holds both");
    assert_eq!(p.of("dit"), Some(Device::Gpu(0)));
    assert_eq!(p.of("te"), Some(Device::Gpu(0)));
}

/// Parts joined by `With` are placed as ONE unit, sized to their sum.
///
/// This is not cosmetic: placing them one after another lets the anchor take
/// the last device that could have held the pair, after which the follower
/// has nowhere to go and a perfectly placeable plan is refused. A real
/// two-card FLUX.2 run found exactly this - the DiT took a card the VAE then
/// could not join.
#[test]
fn an_affinity_group_is_placed_as_a_unit_not_one_part_at_a_time() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 10 * GIB, 0);
    b.set(Device::Gpu(1), 8 * GIB, 0);

    let parts = [
        Part::new("te", MemCost::new(7 * GIB, 0)).apart(),
        Part::new("dit", MemCost::new(6 * GIB, 0)).apart(),
        Part::new("vae", MemCost::new(3 * GIB, 0)).with("dit"),
    ];
    let p = plan(&parts, &b).expect("dit+vae = 9 GiB on the 10 GiB card, te on the 8 GiB one");
    assert_eq!(p.of("dit"), Some(Device::Gpu(0)), "the 9 GiB pair needs the bigger card: {p:?}");
    assert_eq!(p.of("vae"), Some(Device::Gpu(0)), "{p:?}");
    assert_eq!(p.of("te"), Some(Device::Gpu(1)), "{p:?}");
}

/// ...and a group that genuinely cannot fit is refused naming its members,
/// not silently split.
#[test]
fn an_oversized_group_is_refused_naming_what_it_is_grouped_with() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 8 * GIB, 0);
    let parts = [Part::new("dit", MemCost::new(6 * GIB, 0)), Part::new("vae", MemCost::new(4 * GIB, 0)).with("dit")];
    let e = plan(&parts, &b).expect_err("10 GiB does not fit 8");
    let msg = e.to_string();
    assert!(msg.contains("dit") && msg.contains("vae"), "the refusal must name both members: {msg}");
}

/// Parts in different PHASES never co-reside: the caller frees phase k's
/// weights before allocating phase k+1's (a diffusion pipeline evicts its
/// denoiser before it builds the decode graph). A card is therefore charged
/// the MAX over phases, not the sum - which is what lets a 16 GiB denoiser
/// and a 16 GiB decode graph take turns on one 24 GiB card instead of
/// needing two.
#[test]
fn parts_in_different_phases_take_turns_on_one_card() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    let parts = [
        Part::new("dit", MemCost::new(16 * GIB, 0)).phase(1),
        Part::new("vae", MemCost::new(16 * GIB, 0)).phase(2),
    ];
    let p = plan(&parts, &b).expect("16 GiB then 16 GiB, never both: one card holds both");
    assert_eq!(p.of("dit"), Some(Device::Gpu(0)));
    assert_eq!(p.of("vae"), Some(Device::Gpu(0)));
}

/// Phases bound WHEN a part is live, not where it may go: two parts in the
/// SAME phase are two simultaneous residents and are charged as a sum, exactly
/// as unphased parts are.
#[test]
fn parts_in_the_same_phase_still_charge_as_a_sum() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    let parts = [
        Part::new("dit", MemCost::new(16 * GIB, 0)).phase(1),
        Part::new("enc", MemCost::new(16 * GIB, 0)).phase(1),
    ];
    let e = plan(&parts, &b).expect_err("32 GiB live at once does not fit 24");
    assert!(e.to_string().contains("enc"), "{e}");
}

/// A permanent part (no phase) is resident in EVERY phase, so it is charged
/// beside each of them - the decode graph takes the denoiser's place but
/// never the text encoder's.
#[test]
fn a_permanent_part_is_charged_beside_every_phase() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    let parts = [
        Part::new("te", MemCost::new(10 * GIB, 0)),
        Part::new("dit", MemCost::new(16 * GIB, 0)).phase(1),
    ];
    let e = plan(&parts, &b).expect_err("the te outlives the denoise, so 26 GiB is really live");
    assert!(e.to_string().contains("dit"), "{e}");

    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    let parts = [
        Part::new("te", MemCost::new(6 * GIB, 0)),
        Part::new("dit", MemCost::new(16 * GIB, 0)).phase(1),
        Part::new("vae", MemCost::new(16 * GIB, 0)).phase(2),
    ];
    let p = plan(&parts, &b).expect("6 + max(16, 16) = 22 GiB peak: fits");
    assert_eq!(p.of("dit"), Some(Device::Gpu(0)));
    assert_eq!(p.of("vae"), Some(Device::Gpu(0)));
}

/// The FLUX.2 shape end to end: a mixed-phase `With` group (the VAE's encode
/// graph lives while the denoiser does, its decode graph after the denoiser
/// is evicted) is charged per member phase, so the group can join the
/// denoiser's card.
#[test]
fn a_mixed_phase_group_is_charged_per_phase_not_as_a_sum() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 22 * GIB, 0);
    let parts = [
        Part::new("dit", MemCost::new(13 * GIB, 0)).phase(1),
        // The VAE's two graphs share one card; encode coexists with the
        // denoiser, decode does not.
        Part::new("vae_enc", MemCost::new(8 * GIB, 0)).phase(1).with("vae_dec"),
        Part::new("vae_dec", MemCost::new(11 * GIB, 0)).phase(2).with("vae_enc"),
    ];
    let p = plan(&parts, &b).expect("peak = max(13+8, 11) = 21 GiB: fits one card");
    assert_eq!(p.of("dit"), Some(Device::Gpu(0)));
    assert_eq!(p.of("vae_enc"), Some(Device::Gpu(0)));
    assert_eq!(p.of("vae_dec"), Some(Device::Gpu(0)));
}
