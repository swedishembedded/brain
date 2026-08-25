// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The two cards must agree, bit for bit.** The real-weight half of the
//! gate on concurrent classifier-free-guidance dispatch: one generation with
//! both CFG branches on one card, one with them on two cards at the same
//! time, and every decoded byte identical.
//!
//! Swedish Embedded AB implements multi-GPU inference scheduling with
//! bit-exact reproducibility gates for production model pipelines. If your
//! team needs expertise in proving that a placement change moved work and
//! nothing else, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this is the right claim to gate
//!
//! `guidance > 1.0` runs two DiT forwards per denoise step at the same
//! latent - one against the prompt's context, one against the empty
//! prompt's. They share no intermediate value; the only thing that reads
//! both is the host-side fold `uncond + guidance·(cond - uncond)` after both
//! have returned. Moving one of them to another card therefore splits no
//! reduction and reassociates no sum: each card runs the identical dispatch
//! sequence over identical bytes it would have run alone.
//!
//! So bit-identity is the *prediction*, not a hope, and the gate is a
//! bit-pattern comparison rather than a tolerance. A failure here would mean
//! something real - a kernel reading uninitialised memory, a
//! nondeterministic reduction, a cache entry mutated by a concurrent
//! reader - and is worth failing on. Widening the bound would hide exactly
//! the class of defect this gate exists to catch.
//!
//! The checkpoint-scoped block-weight cache (`ltxv::weightcache`) is shared
//! by both branches, which is the other half of what makes this worth doing:
//! two cards pay ONE checkpoint read between them, not two, because the
//! store hands out `Arc<CachedQBlockWeights>` host bytes and each branch
//! uploads them to its own device independently.
//!
//! # Deliberately NOT using the real text encoder
//!
//! `Paths::resolve(None, dit, None, None)` forces the deterministic stub context,
//! so this test needs the 23.6 GB DiT and the VAE but not the 14 GB Gemma-4
//! encoder. CFG is still genuinely exercised: the stub's conditional and
//! unconditional contexts differ (the latter is all-zero), so the two
//! branches really do compute different velocities - asserted below, because
//! a gate where the two branches agreed would pass even if the dispatch
//! handed the same context to both.
//!
//! `#[ignore]`d: two full real 22B generations, weight-streaming bound. Run
//! explicitly:
//!
//! ```text
//! BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
//! BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
//! cargo test --release -p brain-ltxv --test cfg_parallel -- --ignored --nocapture
//! ```
//!
//! A missing checkpoint SKIPs rather than failing - same convention as every
//! other real-weight test in this crate.

use ltxv::devplan::DevicePlan;
use ltxv::pipeline::{generate, GenOpts, Paths};

/// 9 frames at 64x64 is 2 latent frames over a 2x2 latent grid = 8 DiT
/// tokens - this crate's established smallest real-weight smoke shape. The
/// claim under test is about PLACEMENT, which does not depend on token
/// count, so the cheapest shape that still runs all 48 real blocks is the
/// right one.
const FRAMES: usize = 9;
const SIDE: usize = 64;
const SEED: u64 = 20260821;

/// Above 1.0, so both branches run. Any value would do; a large one makes
/// the fold's contribution of each branch unmistakable in the output.
const GUIDANCE: f32 = 5.0;

fn base_opts() -> GenOpts {
    GenOpts {
        frames: FRAMES,
        width: SIDE,
        height: SIDE,
        seed: SEED,
        // Deterministic sampler: an ancestral one would still be reproducible
        // (the noise draw is seeded), but a gate on bit-identity should not
        // depend on that being true.
        eta: 0.0,
        guidance: GUIDANCE,
        dit_config: "ltx25_22b".into(),
        device: Some("gpu".into()),
        ..GenOpts::default()
    }
}

/// The real DiT + VAE, with the text encoder deliberately left out (see this
/// module's doc), or `None` (SKIP).
fn real_paths() -> Option<Paths> {
    let vae = std::env::var("BRAIN_LTXV_VAE").ok().filter(|s| !s.is_empty())?;
    let dit = std::env::var("BRAIN_LTXV_DIT").ok().filter(|s| !s.is_empty())?;
    Paths::resolve(Some(&vae), Some(&dit), None, None).ok()
}

fn run(plan: DevicePlan, paths: &Paths, label: &str) -> (Vec<Vec<u8>>, f32) {
    let o = GenOpts { devices: plan, ..base_opts() };
    let t0 = std::time::Instant::now();
    let (v, timings) = generate(paths, "a slow pan across a quiet harbour at dawn", &o, &Default::default(), |_, _, _| {}).unwrap_or_else(|e| panic!("{label}: {e}"));
    let secs = t0.elapsed().as_secs_f32();
    eprintln!("[{label}] {secs:.1} s wall, denoise {:.1} s, {} steps x {} forwards", timings.denoise, timings.steps, timings.forwards_per_step);
    (v.frames, secs)
}

/// The gate. Same seed, same prompt, same shape, same checkpoint - only the
/// device plan differs, and every decoded byte must match.
///
/// **Both timed arms run against an already-warm block cache**, because the
/// first generation against a checkpoint pays a cold read of minutes that has
/// nothing to do with placement. Timing the first arm against the second
/// without a warm-up measures the cache, not the cards - it reported a
/// "speedup" roughly double the honest concurrency figure on this box. The
/// warm-up run is discarded, and the two arms it precedes are
/// then comparable.
///
/// The wall times are reported, never asserted on: a correctness gate that
/// fails when the box is loaded is a flaky gate. Two Tesla P40s measure a
/// real but sub-linear wall-clock win at this shape, larger on the denoise
/// loop alone than end to end; those figures belong in the roadmap ledger,
/// not in an assertion here.
#[test]
#[ignore = "needs the real 22B LTX-2.5 checkpoint (BRAIN_LTXV_DIT) and VAE (BRAIN_LTXV_VAE)"]
fn concurrent_two_card_cfg_is_bit_identical_to_sequential_one_card() {
    let Some(paths) = real_paths() else {
        eprintln!("SKIP: set BRAIN_LTXV_DIT and BRAIN_LTXV_VAE to run this gate");
        return;
    };
    let n_gpus = gpu_core::devices::ambient_compute_set().gpus.len();
    eprintln!("schedulable GPUs: {n_gpus} (a genuinely concurrent arm needs 2 or more)");

    // Warm the checkpoint's shared block cache. Discarded: its only job is to
    // move the ~23.6 GB read + int8 quantize out of the arms being compared.
    let _ = run(DevicePlan::Single, &paths, "warm-up (discarded)");
    let (seq, seq_secs) = run(DevicePlan::Single, &paths, "sequential, one card");
    // The concurrent arm now really does have two threads reading ONE warm
    // store at the same time, which is the interleaving worth gating.
    let (par, par_secs) = run(DevicePlan::Auto, &paths, "concurrent, two cards");

    assert_eq!(seq.len(), par.len(), "same frame count");
    for (i, (a, b)) in seq.iter().zip(&par).enumerate() {
        assert_eq!(a.len(), b.len(), "frame {i}: same byte length");
        if a != b {
            let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
            let worst = a.iter().zip(b).map(|(&x, &y)| (x as i32 - y as i32).abs()).max().unwrap_or(0);
            panic!("frame {i}: concurrent CFG dispatch changed the output - {diff} of {} bytes differ, worst |delta| {worst}. This is a real defect, not a tolerance to widen (see this module's doc).", a.len());
        }
    }
    eprintln!("bit-identical over {} frames x {} bytes; {seq_secs:.1} s sequential vs {par_secs:.1} s concurrent", seq.len(), seq[0].len());

    // A clip of identical frames would make the comparison above vacuous:
    // any two runs of a frozen generator agree. Assert the clip carries real,
    // varying content before believing that agreeing on it means anything.
    let flat = seq.iter().all(|f| f == &seq[0]);
    assert!(!flat, "every frame is identical - the bit-identity assertion above is vacuous on a frozen clip");
}

/// The other half of the claim: the two branches must really be computing
/// different things. If the stub's conditional and unconditional contexts
/// produced the same velocity, the gate above would pass even for a
/// dispatch that ran the conditional forward twice.
///
/// Checked cheaply and without any checkpoint, at the tiny config, by
/// comparing a `guidance = 1.0` run (conditional only) against a
/// `guidance = 5.0` one (both branches, folded): the fold has to change the
/// output, or there is nothing for two cards to divide.
#[test]
fn the_two_cfg_branches_produce_different_velocities() {
    let Ok(vae) = std::env::var("BRAIN_LTXV_VAE") else {
        eprintln!("SKIP: BRAIN_LTXV_VAE is required even for the tiny DiT path (the VAE decode is real)");
        return;
    };
    let paths = Paths::resolve(Some(&vae), None, None, None).expect("a VAE path resolves");
    let one = GenOpts { guidance: 1.0, seed: SEED, ..GenOpts::default() };
    let five = GenOpts { guidance: GUIDANCE, ..one.clone() };
    let a = generate(&paths, "harbour", &one, &Default::default(), |_, _, _| {}).expect("tiny generation runs").0;
    let b = generate(&paths, "harbour", &five, &Default::default(), |_, _, _| {}).expect("tiny generation runs").0;
    assert_ne!(a.frames, b.frames, "guidance 1.0 and 5.0 must differ, or the unconditional branch contributes nothing and the concurrency gate proves nothing");
}
