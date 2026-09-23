// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Two properties a POOL of adapters rests on, neither of which was verified
//! before (continual-reader roadmap R3).
//!
//! **Stacking is exact.** A working set is several adapters active over one
//! base at once, and `RuntimeLora::new` merges the ones covering the same
//! rectangle into a single rank-concatenated correction:
//! `s1*B1*A1 + s2*B2*A2 = [s1B1|s2B2] . [A1;A2]`. That identity is what
//! makes a working set cost one correction rather than one per adapter, and
//! it was documented but never measured.
//!
//! **Optimiser moments belong to the adapter, not to the slot it occupied.**
//! In a pool, eviction is routine: an adapter is written out, something else
//! takes its place, and it comes back later to be trained further. If its
//! Adam moments do not come back with it, training carries on looking
//! healthy while every re-admitted adapter restarts its momentum - or worse,
//! inherits whatever happened to be in the slot. The one-shot finetune path
//! deliberately does not persist them (it trains, saves and folds once), so
//! a pool needs its own round trip and this is the test of it.

use data::rng::Lcg;
use model::lora::{resumable, Pair, Placement, RuntimeDelta, RuntimeLora};

const OUT: usize = 12;
const INN: usize = 8;

fn pair(r: usize, seed: u64) -> Pair {
    let mut rng = Lcg::new(seed);
    let mut p = Pair::new(OUT, INN, r, || rng.signed() * 0.05);
    // `B` starts at zero, so a fresh pair has no delta at all. One step with
    // arbitrary gradients gives both factors something to contribute.
    let da: Vec<f32> = (0..r * INN).map(|_| rng.signed() * 0.1).collect();
    let db: Vec<f32> = (0..OUT * r).map(|_| rng.signed() * 0.1).collect();
    p.adam_step(&da, &db, 1e-2, 1);
    p
}

fn delta_of(p: &Pair, scale: f32) -> Vec<f32> {
    let mut w = vec![0.0f32; OUT * INN];
    p.delta(scale, &mut w);
    w
}

/// The identity the working set is built on. Three adapters over one
/// rectangle must fold to exactly the sum of what each contributes alone.
#[test]
fn a_stacked_working_set_equals_the_sum_of_its_adapters_deltas() {
    let scale = 0.5f32;
    let ps = [pair(2, 1), pair(4, 2), pair(3, 3)];

    let mut expected = vec![0.0f32; OUT * INN];
    for p in &ps {
        let d = delta_of(p, scale);
        for (e, x) in expected.iter_mut().zip(&d) {
            *e += x;
        }
    }

    let deltas: Vec<RuntimeDelta> = ps.iter().map(|p| RuntimeDelta::from_placement(&Placement::whole("attn.q", p), scale)).collect();
    let rl = RuntimeLora::new(deltas);
    assert_eq!(rl.len(), 1, "adapters over the same rectangle must merge into ONE correction, got {}", rl.len());
    assert_eq!(rl.max_rank(), 2 + 4 + 3, "the merged rank must be the sum of the parts");

    let mut got = vec![0.0f32; OUT * INN];
    rl.deltas()[0].fold_into(&mut got);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-5, "element {i}: stacked {g} != summed {e}");
    }
}

/// Adapters over DIFFERENT rectangles must stay separate corrections - the
/// stay-silent half, so "merge" cannot quietly mean "merge everything".
#[test]
fn adapters_over_different_rectangles_are_not_merged() {
    let rl = RuntimeLora::new(vec![
        RuntimeDelta::from_placement(&Placement::whole("attn.q", &pair(2, 4)), 1.0),
        RuntimeDelta::from_placement(&Placement::whole("attn.k", &pair(2, 5)), 1.0),
    ]);
    assert_eq!(rl.len(), 2);
    assert_eq!(rl.max_rank(), 2);
}

/// The pool's round trip. After a save and a load, the NEXT optimiser step
/// must land exactly where it would have without the round trip - which is
/// only true if the moments travelled with the adapter.
#[test]
fn an_adapter_resumes_training_from_its_own_moments_after_a_round_trip() {
    let dir = std::env::temp_dir().join(format!("brain-pool-moments-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("adapter.safetensors");

    let mut kept = pair(3, 7);
    let mut rng = Lcg::new(99);
    // A few real steps, so the moments carry history worth losing.
    for t in 2..6 {
        let da: Vec<f32> = (0..3 * INN).map(|_| rng.signed() * 0.1).collect();
        let db: Vec<f32> = (0..OUT * 3).map(|_| rng.signed() * 0.1).collect();
        kept.adam_step(&da, &db, 1e-2, t);
    }

    resumable::save(path.to_str().unwrap(), &[("attn.q".to_string(), kept.clone())], 1.0, "pool-adapter", "base", "qwen")
        .expect("save");
    let loaded = resumable::load(path.to_str().unwrap()).expect("load");
    assert_eq!(loaded.len(), 1);
    let (key, mut restored) = loaded.into_iter().next().unwrap();
    assert_eq!(key, "attn.q");
    assert!(restored.has_moments(), "a pool adapter must come back with its optimiser state");

    // The same next step on both.
    let da: Vec<f32> = (0..3 * INN).map(|_| rng.signed() * 0.1).collect();
    let db: Vec<f32> = (0..OUT * 3).map(|_| rng.signed() * 0.1).collect();
    let mut never_saved = kept.clone();
    never_saved.adam_step(&da, &db, 1e-2, 6);
    restored.adam_step(&da, &db, 1e-2, 6);

    assert_eq!(restored.a, never_saved.a, "A after the resumed step must be bit-identical to never having saved");
    assert_eq!(restored.b, never_saved.b, "B after the resumed step must be bit-identical to never having saved");

    // The contrast that makes the assertion above worth making: the same
    // weights WITHOUT their moments take a different step, because Adam's
    // first step from empty state is not the fifth step of a run.
    let mut momentless = Pair::from_ab(OUT, INN, 3, kept.a.clone(), kept.b.clone());
    assert!(!momentless.has_moments(), "from_ab is the moment-free shape by design");
    momentless.restart_moments();
    momentless.adam_step(&da, &db, 1e-2, 6);
    assert_ne!(momentless.a, never_saved.a, "losing the moments must visibly change the step, or this test proves nothing");

    let _ = std::fs::remove_dir_all(&dir);
}
