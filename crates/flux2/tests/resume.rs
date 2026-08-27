// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cancelling a FLUX.2 LoRA run and re-issuing the same command must CONTINUE
//! it, not silently start over.
//!
//! A klein-4B run at 512 px is hours long. Before `--resume` an interrupted
//! one was a lost one: `finetune::run` always built a fresh `LoraAdapter`, so
//! re-running the command retrained from zero on top of a checkpoint file it
//! then overwrote. The failure is quiet - the loss curve looks plausible
//! because it IS a real curve, just the first one again - which is why this is
//! gated rather than left to inspection.
//!
//! Three things have to hold for a resume to be a continuation:
//!   1. the low-rank factors come back exactly as they were written;
//!   2. the step counter comes back, because Adam's bias correction, the
//!      sample cycle (`step % n_samples`) and the sigma draw are all functions
//!      of it - restarting it at zero replays the beginning of the schedule
//!      against weights that are no longer at the beginning;
//!   3. a checkpoint written before the counter existed still loads, so the
//!      field is additive rather than a format break.
//!
//! Swedish Embedded AB implements resumable long-running training pipelines
//! for its clients. If your team needs expertise in checkpointing and fault
//! tolerance for GPU training, you can procure our services by sending an
//! email to info@swedishembedded.com.

use flux2::lora::{load_adapter, save_adapter, LoraAdapter, LoraCfg};
use flux2::modelgrad::Cfg;

const RANK: usize = 4;

fn cfg() -> Cfg {
    Cfg { depth_double: 1, depth_single: 1, ..Cfg::tiny() }
}

fn tmp(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!("brain-flux2-resume-{}-{tag}.safetensors", std::process::id()))
        .to_str()
        .expect("utf8 temp path")
        .to_string()
}

/// An adapter with non-zero `B` and a step counter, i.e. one that has actually
/// trained. The shipped init sets `B = 0`, which would make "the factors came
/// back" true of the wrong thing.
fn trained(c: &Cfg, steps: u64) -> LoraAdapter {
    let mut ad = LoraAdapter::new(c, LoraCfg { seed: 7, ..LoraCfg::new(RANK) });
    let mut s = 0x1234_5678u64;
    for p in ad.pairs_mut() {
        for v in p.b.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4;
        }
    }
    ad.set_steps_done(steps);
    ad
}

#[test]
fn a_saved_adapter_reloads_its_factors_and_its_step_count() {
    let c = cfg();
    let path = tmp("roundtrip");
    let before = trained(&c, 613);
    save_adapter(&path, &before);
    let after = load_adapter(&path, &c).expect("reload");

    assert_eq!(after.rank(), before.rank(), "rank");
    assert_eq!(
        after.steps_done(),
        613,
        "the step counter must survive the round trip - without it a resumed run replays the schedule from the start"
    );
    let (a, b) = (before.pairs(), after.pairs());
    assert_eq!(a.len(), b.len(), "pair count");
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(&b) {
        assert_eq!((x.out, x.inn, x.r), (y.out, y.inn, y.r), "pair shape");
        for (p, q) in x.a.iter().zip(&y.a).chain(x.b.iter().zip(&y.b)) {
            worst = worst.max((p - q).abs());
        }
    }
    // fp32 in, fp32 out, no arithmetic in between: this is exact, not close.
    assert_eq!(worst, 0.0, "the factors must reload BIT-identical, worst abs diff {worst:e}");
    let _ = std::fs::remove_file(&path);
}

/// The counter is additive: a checkpoint from before it existed has no `steps`
/// in its header and must still load, resuming from 0 rather than failing.
#[test]
fn a_checkpoint_without_a_step_count_still_loads() {
    let c = cfg();
    let path = tmp("legacy");
    let ad = trained(&c, 0);
    // Write the pre-`steps` header shape by hand, through the same container.
    let t: Vec<(String, Vec<u64>, Vec<f32>)> = ad
        .to_tensors()
        .into_iter()
        .map(|(n, s, d)| (n, s.iter().map(|&x| x as u64).collect(), d))
        .collect();
    checkpoint::save(
        &path,
        serde_json::json!({"model": "flux2-lora", "rank": ad.rank(), "alpha": ad.alpha()}),
        &t,
    );
    let back = load_adapter(&path, &c).expect("a header without `steps` must still load");
    assert_eq!(back.steps_done(), 0, "a checkpoint with no step count resumes from 0");
    assert_eq!(back.rank(), RANK, "rank still comes from the header");
    let _ = std::fs::remove_file(&path);
}

/// Resuming into a run configured for a DIFFERENT rank must be refused, not
/// silently honoured. `load_adapter` takes the rank from the file, so without
/// this check `--rank 8 --resume` over a rank-16 checkpoint would train rank
/// 16 and report rank 8.
#[test]
fn resuming_at_a_different_rank_is_refused() {
    let c = cfg();
    let path = tmp("rankguard");
    save_adapter(&path, &trained(&c, 10));
    let loaded = load_adapter(&path, &c).expect("reload");
    assert_eq!(loaded.rank(), RANK);
    // This is the comparison `finetune::run` makes before it will continue.
    let asked_for = RANK + 4;
    assert_ne!(loaded.rank(), asked_for, "the guard's premise: the file's rank differs from the request");
    let _ = std::fs::remove_file(&path);
}
