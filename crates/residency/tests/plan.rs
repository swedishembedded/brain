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

/// **The reported production failure, in one assertion.** Two cards busy with
/// somebody else's job, 45.6 GiB of host RAM idle, and a 14.3 GiB DiT: the run
/// died with `cannot place 'dit' ... free: gpu0=4.9 GiB gpu1=7.8 GiB
/// cpu=45.6 GiB`. The host tier is a documented execution tier, and the
/// operator's requirement is explicit - "it may still run slower if all of a
/// sudden vram is not available but it should never fail".
///
/// The CPU-spill gate used to read `budgets.gpus().is_empty()`, copied from
/// `place::pick_device` where returning `None` really does mean "evict a card
/// instead". `plan` has no eviction to fall back on, so there the same gate
/// meant a machine whose cards were momentarily busy was strictly worse off
/// than a machine with no cards at all.
#[test]
fn the_host_tier_takes_a_part_no_card_can_hold_right_now() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 4_900 * GIB / 1000, 0);
    b.set(Device::Gpu(1), 7_800 * GIB / 1000, 0);
    b.set(Device::Cpu, 45_600 * GIB / 1000, 0);
    let parts = [Part::new("dit", MemCost::new(14_300 * GIB / 1000, 0))];
    let p = plan(&parts, &b).expect("45.6 GiB of host RAM can hold a 14.3 GiB part");
    assert_eq!(p.of("dit"), Some(Device::Cpu));
}

/// A card with ONE free byte must not be better or worse than a card with
/// zero. It used to be strictly worse: `probe_free_vram` dropped a
/// fully-consumed card from the GPU list entirely, which emptied `gpus()` and
/// re-enabled the host tier - while a card with any free byte at all kept the
/// GPU class "non-empty" and so forced the hard failure. A more-contended
/// machine came out ahead of a less-contended one.
#[test]
fn a_uselessly_small_amount_of_free_vram_does_not_veto_the_host_tier() {
    let parts = [Part::new("dit", MemCost::new(14 * GIB, 0))];

    let mut no_cards = Budgets::new();
    no_cards.set(Device::Cpu, 128 * GIB, 0);
    assert_eq!(plan(&parts, &no_cards).unwrap().of("dit"), Some(Device::Cpu));

    let mut one_byte = Budgets::new();
    one_byte.set(Device::Gpu(0), 1, 0);
    one_byte.set(Device::Cpu, 128 * GIB, 0);
    assert_eq!(plan(&parts, &one_byte).unwrap().of("dit"), Some(Device::Cpu), "one useless free byte must not veto a 128 GiB host tier");
}

/// A part's declared host RAM is held wherever it is placed - `MemCost::ram`
/// says so in as many words - but only the device it landed on was ever
/// charged. So a part needing 40 GiB of host staging RAM was placed happily
/// onto a card belonging to a box with 4 GiB of RAM free.
#[test]
fn host_ram_is_checked_even_when_the_part_lands_on_a_card() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    b.set(Device::Cpu, 4 * GIB, 0);
    let parts = [Part::new("streamer", MemCost::new(8 * GIB, 40 * GIB))];
    assert!(plan(&parts, &b).is_err(), "40 GiB of host RAM against 4 GiB free is not placeable anywhere");

    // ...and with the host RAM actually there, the card placement stands.
    b.set(Device::Cpu, 64 * GIB, 0);
    assert_eq!(plan(&parts, &b).expect("both dimensions fit").of("streamer"), Some(Device::Gpu(0)));
}

/// ...and it is CHARGED there too, so the parts that follow see it. Two
/// GPU-placed parts holding 20 GiB of host RAM each used to leave the host
/// tier reporting itself completely empty.
#[test]
fn host_ram_is_charged_even_when_the_part_lands_on_a_card() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    b.set(Device::Gpu(1), 24 * GIB, 0);
    b.set(Device::Cpu, 30 * GIB, 0);
    let parts = [
        Part::new("a", MemCost::new(8 * GIB, 20 * GIB)).apart(),
        Part::new("b", MemCost::new(8 * GIB, 20 * GIB)).apart(),
        Part::new("host_only", MemCost::new(0, 25 * GIB)),
    ];
    assert!(plan(&parts, &b).is_err(), "20 + 20 + 25 GiB of host RAM does not fit a 30 GiB host tier");
}

/// The placement order must not compare host-RAM bytes against VRAM bytes -
/// they are different budgets on different devices. The old key was
/// `peak_vram.max(cost.ram).max(cost.npu)`, so a part that is large only in
/// host RAM sorted first and took the emptiest CARD, stranding the part that
/// actually needed it, even though a valid plan existed the other way round.
#[test]
fn a_host_ram_heavy_part_does_not_take_the_card_the_vram_heavy_part_needs() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 15 * GIB, 0);
    b.set(Device::Gpu(1), 7 * GIB, 0);
    b.set(Device::Cpu, 128 * GIB, 0);
    let parts = [
        Part::new("te", MemCost::new(6 * GIB, 20 * GIB)),
        Part::new("dit", MemCost::new(14 * GIB, 0)),
    ];
    let p = plan(&parts, &b).expect("dit(14)->gpu0, te(6)->gpu1 is a valid plan and must be found");
    assert_eq!(p.of("dit"), Some(Device::Gpu(0)), "{p:?}");
    assert_eq!(p.of("te"), Some(Device::Gpu(1)), "{p:?}");
}

/// A PERMANENT part gets its pick of the cards before a merely TRANSIENT one,
/// even when the transient one is bigger. This is the FLUX.1 shape that
/// produced the reported crash: a 14 GiB DiT that is resident for the whole
/// run, and an 18 GiB T5-XXL that exists for one `encode()` call and is
/// dropped. Biggest-first gave the encoder the only card that could hold the
/// DiT; permanent-first places the DiT and lets the encoder take the (slower)
/// host tier, which is the outcome the operator asked for.
#[test]
fn a_permanent_part_outranks_a_bigger_transient_one_for_the_cards() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 23 * GIB, 0);
    b.set(Device::Gpu(1), 8 * GIB, 0);
    b.set(Device::Cpu, 128 * GIB, 0);
    let parts = [
        Part::new("te", MemCost::new(18 * GIB, 0)).apart().phase(1),
        Part::new("dit", MemCost::new(14 * GIB, 0)).apart(),
    ];
    let p = plan(&parts, &b).expect("dit on the big card, te on the host tier");
    assert_eq!(p.of("dit"), Some(Device::Gpu(0)), "the permanent part must get the card: {p:?}");
    assert_eq!(p.of("te"), Some(Device::Cpu), "the transient part takes the slower tier: {p:?}");
}

/// An UNSIZED part holds real device bytes nobody costed. It is charged
/// nothing - an invented number would distort every sized part - which is only
/// safe if it is placed LAST and only onto a card with real slack. Priced at
/// one byte and placed in size order, it "fit" a card a 23.5 GiB part had
/// already filled, and the two were silently double-booked.
#[test]
fn an_unsized_part_does_not_double_book_a_card_a_sized_part_filled() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    b.set(Device::Cpu, 128 * GIB, 0);
    let parts = [Part::unsized_("dit"), Part::new("te", MemCost::new(24 * GIB - (GIB / 2), 0))];
    let p = plan(&parts, &b).expect("plans clean");
    assert_eq!(p.of("te"), Some(Device::Gpu(0)));
    assert_eq!(p.of("dit"), Some(Device::Cpu), "half a GiB of slack is not a home for an uncosted part: {p:?}");
}

/// ...and with room to spare it still takes the emptiest card, unchanged.
#[test]
fn an_unsized_part_still_takes_a_card_with_room_to_spare() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 24 * GIB, 0);
    b.set(Device::Gpu(1), 24 * GIB, 0);
    b.set(Device::Cpu, 128 * GIB, 0);
    let parts = [Part::unsized_("dit"), Part::new("te", MemCost::new(20 * GIB, 0))];
    let p = plan(&parts, &b).expect("plans clean");
    assert!(matches!(p.of("dit"), Some(Device::Gpu(_))), "{p:?}");
    assert_ne!(p.of("dit"), p.of("te"), "the emptier card was free: {p:?}");
}

/// With every card contended below the point where anything can be built on
/// it, the no-preference default (the bare `Gpu::new` case) must still resolve
/// somewhere rather than refuse - otherwise `gpu_core::devices::auto_gpu`
/// memoizes a permanent "no card" and every later build in the process
/// silently reverts to card 0, typically the most contended one.
#[test]
fn the_no_preference_default_resolves_even_with_every_card_contended() {
    let mut b = Budgets::new();
    b.set(Device::Gpu(0), 8 * GIB, 8 * GIB); // headroom clamps usable to 0
    b.set(Device::Gpu(1), 8 * GIB, 8 * GIB);
    b.set(Device::Cpu, 128 * GIB, 0);
    let p = plan(&[Part::unsized_("model")], &b).expect("the default must resolve somewhere");
    assert_eq!(p.of("model"), Some(Device::Cpu));
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
