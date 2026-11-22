// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sharded-training contract ([`crate::zero`]) for a model's own test to run: a data-parallel fine-tune whose
//! optimizer state (ZeRO-1) or optimizer state and gradients (ZeRO-2) are sharded over in-process ranks trains to the
//! weights of the replicated run within a stated tolerance, every rank ends with the same weights bit for bit, a
//! repeat of a sharded run is bit-identical, and a sharded run stopped and resumed from its per-rank state ends where
//! the uninterrupted one does, bit for bit. The ranks are threads over a [`crate::HostCollective`] on the test's
//! device, so the contract runs on the CPU backend in CI; the device collectives (NCCL) are exercised by the
//! cross-process tests of `crates/model/tests/it/zero_job.rs`.
//!
//! Swedish Embedded AB implements sharded data-parallel training for its clients. If your team needs expertise in
//! verifying that a sharded optimizer computes what a replicated one does, you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use crate::zero::Sharding;
use crate::{FitControl, FitOpts, GradSync, HostCollective, Model};

/// The largest difference between a sharded and the replicated run's weights after the contract's six steps, as a
/// fraction of the learning rate (the most one AdamW step moves an element): the reductions sum in another order and
/// the update runs on the host, and AdamW's normalisation turns a rounding difference in a near-zero gradient into
/// at most a small fraction of one step.
pub const SHARDED_STEP_FRACTION: f32 = 1e-2;

/// How far, relative to the loss (or absolutely below 1), a data-parallel step's reported loss may lie from the loss
/// one rank measures over the same records: the per-token mean summed in another order, on weights a rounding apart.
const STEP_LOSS_TOLERANCE: f32 = 1e-3;

/// One rank's weights, by name.
type Weights = Vec<(String, Vec<f32>)>;
/// What one rank of a run ended with: its weights, and the training loss each of its steps reported.
struct RankRun {
    weights: Weights,
    losses: Vec<f32>,
}

/// Write the masked prompt/answer corpus the contract trains on into `dir` (tokens 2..=15, vocabulary 16).
fn corpus(dir: &Path) -> std::io::Result<()> {
    use data::binio::{self, Meta};
    let (mut tokens, mut mask) = (Vec::new(), Vec::new());
    for i in 0..400u32 {
        tokens.extend_from_slice(&[2 + i % 5, 7 + i % 3, 9, 15]);
        mask.extend_from_slice(&[false, false, true, true]);
    }
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens)?;
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask)?;
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[])?;
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[])?;
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(16))
}

fn opts(block: u32, sharding: Sharding, host_state: bool, seed: u64) -> FitOpts {
    FitOpts {
        steps: 6,
        batch_size: 1,
        block_size: block,
        lr: 1e-2,
        min_lr: 1e-3,
        warmup: 2,
        decay_iters: 6,
        weight_decay: 0.1,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        seed,
        sharding,
        precision: crate::TrainPrecision { host_moments: host_state, ..Default::default() },
        ..Default::default()
    }
}

/// Run `world` ranks of a data-parallel fit as threads; `stop`/`state` stop each rank after that many steps with its
/// resume state at `state(rank)`, or resume from it.
#[allow(clippy::too_many_arguments)]
fn run<M: Model>(name: &str, build: &(impl Fn() -> M + Sync), dir: &Path, world: usize, opts: &FitOpts, stop: Option<u32>, state: Option<&(dyn Fn(usize) -> std::path::PathBuf + Sync)>) -> Result<Vec<RankRun>, String> {
    let coll = HostCollective::new(world);
    let out: Vec<Result<RankRun, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let coll = Arc::clone(&coll);
                s.spawn(move || {
                    let err = |e: std::io::Error| format!("{name}: rank {rank}: {e}");
                    let (train, val, bcfg, _, itos) = crate::load_dataset_with_itos(dir, opts).map_err(err)?;
                    let objective = crate::causal_lm::<M>(train, val, bcfg, itos);
                    let sync = GradSync { collective: &*coll, rank, checkpoint: None, observer: None };
                    let mut losses = Vec::new();
                    let mut stop_at = |r: &crate::StepReport| {
                        losses.push(r.loss);
                        stop.is_none_or(|n| r.step < n)
                    };
                    let path = state.map(|f| f(rank));
                    let control = FitControl { on_step: Some(&mut stop_at), state: path.as_deref(), state_every: 0, identity: serde_json::json!({ "test": name }), snapshots: None, grad_sync: Some(sync) };
                    let (_, model) = crate::fit_controlled(build(), objective, opts, None, control).map_err(err)?;
                    Ok(RankRun { weights: weights_of(&model), losses })
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(format!("{name}: a rank panicked")))).collect()
    });
    out.into_iter().collect()
}

/// A model's weights by name, in name order.
fn weights_of<M: Model>(model: &M) -> Weights {
    let mut names = model.param_names();
    names.sort();
    names.into_iter().map(|n| (n.clone(), model.read_weight(&n))).collect()
}

/// Whether a shard file of the sharded optimizer was saved for the resume state at `state`, wherever the fit loop files
/// the state's off-device part (beside the file, or in a distributed run's generations).
fn saved_shard(state: &Path) -> bool {
    fn walk(dir: &Path) -> bool {
        std::fs::read_dir(dir).into_iter().flatten().flatten().any(|e| {
            let path = e.path();
            if path.is_dir() {
                walk(&path)
            } else {
                path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("shard-") && n.ends_with(".safetensors"))
            }
        })
    }
    let parent = state.parent().unwrap_or(Path::new("."));
    let stem = state.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    std::fs::read_dir(parent).into_iter().flatten().flatten().any(|e| e.file_name().to_str().is_some_and(|n| n.starts_with(stem) && n != stem) && e.path().is_dir() && walk(&e.path()))
}

fn bits_equal(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|((na, wa), (nb, wb))| na == nb && wa.iter().map(|x| x.to_bits()).eq(wb.iter().map(|x| x.to_bits())))
}

/// The largest absolute difference between two runs' weights, and its tensor.
fn worst(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)]) -> (f32, String) {
    let mut out = (0f32, String::new());
    for ((n, x), (_, y)) in a.iter().zip(b) {
        let diff = x.iter().zip(y).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        if diff > out.0 {
            out = (diff, n.clone());
        }
    }
    out
}

/// Data parallelism changes where a step's records are computed, never the step: `world` replicated ranks of one
/// record each train what one rank accumulating the same `world` records a step trains (rank `r` keeps stream record
/// `n` when `n % world == r`, so step `s` of both covers records `s*world..(s+1)*world`), and the first update moves
/// no weight by more than one AdamW step of the scheduled rate. A rate scaled by the world size, an update applied
/// more than once a step, or ranks training the same records all break it. (A sum in place of the mean is invisible
/// to AdamW whenever the step is clipped, so the bound is on the update, not on the gradient.) The loss each step
/// reports is the step's: the per-token mean over all `world` records, the one-rank run's, not rank 0's share of it.
fn check_one_rank_equivalence<M: Model>(name: &str, build: &(impl Fn() -> M + Sync), scratch: &Path, world: usize, block: u32, seed: u64, ranks: &RankRun) -> Result<(), String> {
    let o = FitOpts { grad_accum: world as u32, ..opts(block, Sharding::Replicated, false, seed) };
    let one = run(name, build, scratch, 1, &o, None, None)?;
    let (diff, tensor) = worst(&one[0].weights, &ranks.weights);
    if diff / o.lr > SHARDED_STEP_FRACTION {
        return Err(format!("{name}: {world} replicated ranks end {diff:.2e} ({tensor}) from one rank accumulating the same {world} records a step, {:.2e} of a step, over {SHARDED_STEP_FRACTION:.0e}", diff / o.lr));
    }
    for (step, (a, b)) in one[0].losses.iter().zip(&ranks.losses).enumerate() {
        if (a - b).abs() > STEP_LOSS_TOLERANCE * a.abs().max(1.0) {
            return Err(format!("{name}: step {} of {world} ranks reports loss {b}, the step's loss over the same records is {a}", step + 1));
        }
    }
    if one[0].losses.len() != ranks.losses.len() {
        return Err(format!("{name}: {} steps reported by one rank, {} by the ranks", one[0].losses.len(), ranks.losses.len()));
    }
    let initial = weights_of(&build());
    let first = run(name, build, scratch, world, &opts(block, Sharding::Replicated, false, seed), Some(1), None)?;
    let step_lr = crate::cosine_lr(0, &o);
    let decay = initial.iter().flat_map(|(_, w)| w.iter()).fold(0f32, |m, w| m.max(w.abs())) * o.weight_decay * step_lr;
    let (moved, tensor) = worst(&first[0].weights, &initial);
    // AdamW's first step moves an element by lr * |g| / (|g| + eps) <= lr, plus its decoupled decay.
    if moved > (step_lr + decay) * (1.0 + 1e-3) {
        return Err(format!("{name}: the first step of {world} ranks moved {tensor} by {moved:.3e}, more than one AdamW step at the scheduled rate {step_lr:.3e} (+ decay {decay:.1e})"));
    }
    eprintln!("{name} x{world}: replicated ranks within {:.2e} of a step of one rank accumulating {world}; first step moves at most {moved:.3e} at rate {step_lr:.3e}", diff / o.lr);
    Ok(())
}

/// The sharded-training contract (module documentation) at `world` in-process ranks. `build` makes a fresh trainable
/// model (the same initial weights every call) for a vocabulary of at least 16, a batch of one and a block of `block`
/// tokens, on a device that registers [`crate::zero::KERNELS`]; `scratch` is an empty directory the runs may write.
pub fn check_sharded_training<M: Model>(name: &str, build: impl Fn() -> M + Sync, block: u32, world: usize, scratch: &Path) -> Result<(), String> {
    corpus(scratch).map_err(|e| format!("{name}: {e}"))?;
    let seed = data::rng::random_seed();
    let replicated = run(name, &build, scratch, world, &opts(block, Sharding::Replicated, false, seed), None, None)?;
    check_one_rank_equivalence(name, &build, scratch, world, block, seed, &replicated[0])?;
    // The replicated run is held to the same repeat and resume contract: it is the reference. The sharded state is
    // held on the cards, and in host memory (`moments=host`) for ZeRO-2.
    for (sharding, host_state) in [(Sharding::Replicated, false), (Sharding::Optimizer, false), (Sharding::Gradients, false), (Sharding::Gradients, true)] {
        let o = opts(block, sharding, host_state, seed);
        let first = run(name, &build, scratch, world, &o, None, None)?;
        for (rank, r) in first.iter().enumerate() {
            if !bits_equal(&r.weights, &first[0].weights) {
                return Err(format!("{name} {sharding:?}: rank {rank}'s weights differ from rank 0's"));
            }
            if r.losses.iter().map(|l| l.to_bits()).ne(first[0].losses.iter().map(|l| l.to_bits())) {
                return Err(format!("{name} {sharding:?}: rank {rank} reports other step losses than rank 0 ({:?} against {:?}): a step's loss is the group's", r.losses, first[0].losses));
            }
        }
        let (diff, tensor) = worst(&first[0].weights, &replicated[0].weights);
        let rel = diff / o.lr;
        if rel > SHARDED_STEP_FRACTION {
            return Err(format!("{name} {sharding:?}: {tensor} is {diff:.2e} from the replicated run, {rel:.2e} of a step, over {SHARDED_STEP_FRACTION:.0e}"));
        }
        let again = run(name, &build, scratch, world, &o, None, None)?;
        if !bits_equal(&again[0].weights, &first[0].weights) {
            return Err(format!("{name} {sharding:?}: a repeat of the run on the same layout is not bit-identical"));
        }
        // Stopped after three steps with every rank's state saved, then resumed in fresh models.
        let state = |rank: usize| scratch.join(format!("{sharding:?}-{host_state}-rank{rank}.state"));
        run(name, &build, scratch, world, &o, Some(3), Some(&state))?;
        for rank in 0..world {
            if sharding != Sharding::Replicated && !saved_shard(&state(rank)) {
                return Err(format!("{name} {sharding:?}: rank {rank} saved no shard of its optimizer state"));
            }
        }
        let resumed = run(name, &build, scratch, world, &o, None, Some(&state))?;
        if !bits_equal(&resumed[0].weights, &first[0].weights) {
            let (diff, t) = worst(&resumed[0].weights, &first[0].weights);
            return Err(format!("{name} {sharding:?}: the resumed run's weights differ from the uninterrupted run's ({t} by {diff:e})"));
        }
        eprintln!("{name} {sharding:?} (host state: {host_state}) x{world}: at most {rel:.2e} of a step from replicated ({tensor}); ranks, repeat and resume bit-identical");
    }
    Ok(())
}
