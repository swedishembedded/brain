// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What this model tells a pipeline about itself.
//!
//! `brain-recon` plans a long capture without knowing what is reconstructing
//! it: it asks the model what shape it wants its frames in and how many of
//! them one pass may hold. Both answers are architecture facts - the ViT's
//! patch grid, and a trunk whose attention is global across frames - and both
//! are checkable with no checkpoint and no GPU, which is the point of having
//! them as plain functions rather than as something only a loaded model can
//! answer.
//!
//! Swedish Embedded AB implements feed-forward reconstruction that scales to
//! real captures. If your team needs that, you can procure our services by
//! sending an email to info@swedishembedded.com.

use recon::{plan_chunks, MIN_OVERLAP};
use worldmirror2::config::MirrorConfig;
use worldmirror2::recon_impl::{frame_budget, input_grid};

/// The grid has to land on the patch lattice whatever came in. A frame whose
/// grid is not a whole number of patches has a row the tokenizer cannot form,
/// and the cap is a CAP: a small photograph is not upsampled to meet it.
#[test]
fn the_input_grid_lands_on_the_patch_lattice_and_respects_the_cap() {
    let cfg = MirrorConfig::default();
    let p = cfg.patch as u32;
    for &(w, h) in &[(1920u32, 1080u32), (800, 800), (640, 480), (3000, 4000), (200, 137)] {
        for &cap in &[518usize, 952] {
            let (gw, gh) = input_grid(&cfg, cap, w, h);
            assert!(gw % p == 0 && gh % p == 0, "{w}x{h} at cap {cap} gave {gw}x{gh}, off the {p}-patch grid");
            assert!(gw > 0 && gh > 0, "{w}x{h} at cap {cap} gave an empty grid");
            assert!(
                gw.max(gh) as usize <= cap,
                "{w}x{h} at cap {cap} gave {gw}x{gh}, past the cap"
            );
            // Aspect is preserved to within the patch rounding, so the crop a
            // pipeline does against this grid does not throw away a third of
            // the picture.
            let (want, got) = (w as f64 / h as f64, gw as f64 / gh as f64);
            assert!(
                (want - got).abs() / want < 0.15,
                "{w}x{h} (aspect {want:.3}) became {gw}x{gh} (aspect {got:.3})"
            );
        }
    }
}

/// The budget is the reason chunking exists. Global attention means one pass
/// attends across every patch of every frame it holds, so the frames a pass
/// can afford must FALL as the frames get bigger - and whatever it returns has
/// to be a number a plan can actually be built from.
#[test]
fn the_frame_budget_falls_as_the_frames_grow_and_is_always_plannable() {
    let cfg = MirrorConfig::default();
    let grids: Vec<(u32, u32)> =
        [252u32, 518, 728, 952].iter().map(|&c| input_grid(&cfg, c as usize, 1920, 1080)).collect();
    let budgets: Vec<usize> = grids.iter().map(|&g| frame_budget(&cfg, g, 0)).collect();
    assert!(
        budgets.windows(2).all(|w| w[0] >= w[1]),
        "the budget did not fall with frame size: {grids:?} -> {budgets:?}"
    );
    assert!(budgets[0] > *budgets.last().unwrap(), "the budget is flat: {budgets:?}");
    for (g, b) in grids.iter().zip(&budgets) {
        assert!(
            *b > MIN_OVERLAP,
            "a budget of {b} at {g:?} leaves no room for the {MIN_OVERLAP}-frame minimum overlap"
        );
        plan_chunks(200, *b, MIN_OVERLAP + 1)
            .unwrap_or_else(|e| panic!("a 200-frame capture cannot be planned at {g:?}: {e}"));
    }
}

/// A pass that does not fit in one storage binding does not run slowly, it
/// does not run. The device's limit is therefore part of the budget, not
/// something the caller discovers when the first chunk dies.
#[test]
fn a_small_device_gets_a_smaller_budget() {
    let cfg = MirrorConfig::default();
    let grid = input_grid(&cfg, 518, 1920, 1080);
    let roomy = frame_budget(&cfg, grid, 0);
    let cramped = frame_budget(&cfg, grid, 64 << 20);
    assert!(
        cramped <= roomy,
        "a 64 MiB binding limit produced {cramped} frames against an unconstrained {roomy}"
    );
    assert!(cramped > MIN_OVERLAP, "a cramped device was left with an unplannable budget of {cramped}");
}
