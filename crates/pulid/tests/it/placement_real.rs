// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! What the placement plan charges the DiT's card for a PuLID-conditioned
//! generation, checked against the device bytes that card really takes: the
//! int8 DiT, the PuLID cross-attention module and the face-identity stack
//! (ArcFace with its SCRFD detector, BiSeNet, EVA-CLIP, IDFormer) pinned beside
//! them, built through the served action and run through one 256 px, one step
//! generation. The peak is the lowest free figure the driver reports on that
//! card while the action runs, less the free figure once the card's context
//! exists, so the driver context (which the planner's headroom covers, not the
//! parts' estimates) is not charged to them.
//!
//! Swedish Embedded AB implements solutions for fitting identity-conditioned
//! diffusion pipelines onto the accelerators a machine really has for its
//! clients. If your team needs expertise in device memory planning then you
//! can procure our services by sending an email to info@swedishembedded.com.

use capability::Provider;
use flux1::pipeline::{part_needs, plan_flux1, Flux1};
use flux1::{Flux1Config, Precision};
use gpu_core::devices::Home;
use gpu_core::Gpu;
use serde_json::json;

const MIB: f64 = (1u64 << 20) as f64;
const SIDE: u32 = 256;

/// Held on the card the DiT is not meant for, so the planner's choice of the
/// freest card is not decided by the driver contexts' own few MiB.
const BALLAST_BYTES: u64 = 1 << 29;

#[test]
#[ignore = "slow (real): builds FLUX.1-dev int8, PuLID and the identity stack on a GPU and runs one 256 px step"]
fn the_dit_card_peak_matches_the_plan_charge_for_the_identity_stack() {
    let store = |rel: &str| brain_testutil::model_dir(rel).filter(|d| std::path::Path::new(d).exists());
    let (Some(flux1_root), Some(adapter), Some(arcface), Some(clip), Some(bisenet)) = (
        store("black-forest-labs/FLUX.1-dev"),
        store("guozinan/PuLID").map(|d| format!("{d}/pulid_flux_v0.9.1.safetensors")).filter(|f| std::path::Path::new(f).exists()),
        store("DIAMONIK7777/antelopev2"),
        store("QuanSun/EVA-CLIP"),
        store("facexlib/pulid"),
    ) else {
        brain_testutil::skip("the pulid placement estimate needs FLUX.1-dev, guozinan/PuLID, DIAMONIK7777/antelopev2, QuanSun/EVA-CLIP and facexlib/pulid in the model store");
        return;
    };
    let photo_path = brain_testutil::testdata("pulid/face_source.jpg");
    if !std::path::Path::new(&photo_path).exists() {
        brain_testutil::skip(&format!("portrait {photo_path} absent (run `make testdata/fetch`)"));
        return;
    }
    let photo = imaging::load(&photo_path).unwrap_or_else(|e| panic!("reading {photo_path}: {e}"));
    let face = capability::blob::image_blob(&photo.to_hwc_unit(), photo.w, photo.h, 3);

    loader::install_default_placer(None, true);
    let cfg = Flux1Config::dev();
    let n_joint = Flux1::n_max(SIDE, SIDE).expect("joint-token budget") as u64;
    let extra = pulid::caps::dit_extra_bytes(&cfg, &arcface, &bisenet).expect("identity stack estimate");
    let estimate = part_needs(&cfg, Precision::Int8, n_joint, extra).into_iter().find(|n| n.name == "dit").expect("declared part").vram;
    // The planner gives the DiT the card with the most free bytes, so which
    // card that is has to be fixed before the action plans for itself: both
    // contexts exist, and ballast on the other card makes the choice
    // unambiguous, so the plan below is the plan the action will make.
    let mut cards: Vec<_> = gpu_core::capacity::probe_gpus().into_iter().filter(|g| g.measured).collect();
    cards.sort_by_key(|g| std::cmp::Reverse(g.free));
    let [dit_card, other_card] = &cards[..] else {
        brain_testutil::skip_unavailable("the placement estimate needs exactly two cards whose memory nvidia-smi can read");
        return;
    };
    let (card, other) = (dit_card.index, other_card.index);
    let _context = gpu_core::devices::with_gpu(card, || Gpu::new(pulid::KERNELS)).expect("card");
    let other_context = gpu_core::devices::with_gpu(other, || Gpu::new(pulid::KERNELS)).expect("card");
    let _ballast = other_context.storage(BALLAST_BYTES / 4);
    let homes = plan_flux1(&cfg, Precision::Int8, n_joint, extra).expect("placement plan");
    if homes.of("dit") != Some(Home::Gpu(card)) {
        brain_testutil::skip_unavailable(&format!("the plan did not put the DiT on the freest card ({})", homes.describe()));
        return;
    }
    let baseline = gpu_core::capacity::free_on(card).expect("probed card");
    let provider = pulid::caps::PulidProvider::new(flux1_root, adapter, arcface, clip, bisenet);
    let action = provider.action("text2image").expect("declared action");
    let request = capability::Invocation::new()
        .set("prompt", json!("a portrait photo, studio light"))
        .set("width", json!(SIDE))
        .set("height", json!(SIDE))
        .set("steps", json!(1))
        .set("precision", json!("int8"))
        .set("seed", json!(7))
        .blob("face_image", face);
    let (outcome, measured) = gpu_core::capacity::peak_taken_while(card, baseline, || action.run(&request, &mut |_| {}));
    outcome.expect("text2image run");

    eprintln!("dit + pulid ({}): estimate {:.0} MiB, identity stack {:.0} MiB of it, measured peak {:.0} MiB", homes.describe(), estimate as f64 / MIB, extra as f64 / MIB, measured as f64 / MIB);
    let bounds = gpu_core::capacity::estimate_bounds(estimate);
    assert!(
        bounds.contains(&measured),
        "the DiT card's measured peak {:.0} MiB is outside the {:.0}..{:.0} MiB band around the {:.0} MiB estimate",
        measured as f64 / MIB,
        *bounds.start() as f64 / MIB,
        *bounds.end() as f64 / MIB,
        estimate as f64 / MIB
    );
}
