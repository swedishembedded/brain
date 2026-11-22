// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distributed training jobs that
// survive a lost rank without corrupting their state for its clients. If your
// team needs expertise in collective failure handling and checkpoint recovery
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A distributed GPT job that loses a rank, at every point a step can lose one, and is restarted.
//!
//! Two OS processes (this test binary re-executed by `model::conformance_faults`) train one plan over loopback, committing their exact-resume
//! state every step. In each case one rank fails at step [`FAULT_STEP`]: it is killed during the gradient
//! all-reduce, killed after the ranks agreed to apply the step (during the optimizer step), killed while the step's
//! state is being committed, or stalled inside the all-reduce. A job that shards its gradients and optimizer state
//! (ZeRO-2) loses a rank in the gradient reduce-scatter (killed or stalled) and in the parameter all-gather after
//! the vote. Then:
//!
//! - every surviving rank exits with the restart status, having applied no step its peers did not agree to;
//! - the newest committed generation holds every rank's state of the same step, with identical weights;
//! - the job started again with the same arguments resumes from that generation and ends with the weights of the
//!   uninterrupted run, bit for bit (the host transport sums in a fixed rank order, so the run is repeatable).

use std::path::{Path, PathBuf};

use gpt2::model::Gpt;
use gpt2::train::TrainOpts;
use gpt2::GptConfig;
use model::conformance_faults::{FaultJob, RankArgs};
use model::jobfault::RESTART_STATUS;
use model::zero::Sharding;
use model::{CheckpointIdentity, GradSync, Model};

const CHILD: &str = "distributed_faults::rank_process";
const STEPS: u32 = 6;
/// The step every fault happens in.
const FAULT_STEP: u32 = 4;
const BATCH: u32 = 2;
const BLOCK: u32 = 16;

fn config() -> GptConfig {
    GptConfig { vocab: 0, block_size: BLOCK, n_layers: 1, d_model: 32, n_heads: 2, d_ff: 64 }
}

/// The run's options; a sharded run on the host keeps its optimizer slices in host memory (the CPU device registers
/// no sharded-step kernels).
fn opts(sharding: Sharding, device: bool) -> TrainOpts {
    TrainOpts {
        sharding,
        precision: model::TrainPrecision { host_moments: !sharding.is_replicated() && !device, ..Default::default() },
        steps: STEPS,
        batch_size: BATCH,
        block_size: BLOCK,
        lr: 1e-2,
        warmup: 0,
        decay_iters: STEPS,
        eval_interval: 0,
        checkpoint_secs: 0,
        mask_before: Some('='),
        mask_per_line: true,
        seed: 11, // seed: every rank and every restart must start from the same weights and data order
        ..Default::default()
    }
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("brain-gpt2-faults-{}-{name}", std::process::id()))
}

/// A two-rank job of the GPT trainer on `data`, sharding `sharding`, reducing on the cards when `device`.
fn job(name: &str, data: &Path, device: bool, sharding: Sharding) -> FaultJob {
    let mut args = vec![format!("--data={}", data.display())];
    if device {
        args.push("--device-collective".into());
    }
    if !sharding.is_replicated() {
        args.push(format!("--sharding={}", sharding.stage()));
    }
    FaultJob::new(scratch(name), &format!("gpt2-faults-{name}"), 2, device, CHILD, args).unwrap()
}

fn replica(job: &FaultJob, rank: usize) -> PathBuf {
    job.dir.join(format!("replica-{rank}.safetensors"))
}

/// A model's trainable tensors by name.
type Weights = Vec<(String, Vec<f32>)>;

fn weights(path: &Path) -> Weights {
    let gpt = Gpt::load(path.to_str().unwrap(), BATCH, BLOCK);
    gpt.param_names().into_iter().map(|n| (n.clone(), gpt.read_weight(&n))).collect()
}

/// The newest generation the whole job committed, and every rank's weights in its own committed file of it.
fn committed_generation(job: &FaultJob) -> (u32, Vec<Weights>) {
    let (step, files) = job.committed("state.safetensors");
    let per_rank = files
        .iter()
        .enumerate()
        .map(|(rank, file)| {
            let st = checkpoint::st::load_safetensors(file.to_str().unwrap()).unwrap();
            assert_eq!(st.config()["step"], step, "rank {rank}'s committed state is of the committed step");
            let mut w: Weights = st.tensors.iter().filter(|(n, _)| n.starts_with("w:")).map(|(n, v)| (n.clone(), v.clone())).collect();
            w.sort_by(|a, b| a.0.cmp(&b.0));
            w
        })
        .collect();
    (step, per_rank)
}

fn dataset() -> PathBuf {
    let root = scratch("data");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let data = root.join("calculator");
    data::prepare::prepare(data::prepare::Dataset::Calculator, &data, 4000, 1).unwrap();
    data
}

/// Lose a rank to `fault`, check what the failure left, restart the job and compare it with `reference`.
#[allow(clippy::too_many_arguments)] // the case: its job, where and how it fails, and what it is compared with
fn lose_and_restart(name: &str, data: &Path, device: bool, sharding: Sharding, faulty_rank: usize, fault: &str, reference: &[(String, Vec<f32>)], survivor_may_apply: bool) {
    let job = job(name, data, device, sharding);
    let mut faults = [None, None];
    faults[faulty_rank] = Some(fault);
    let status = job.run(0, &faults);
    let survivor = 1 - faulty_rank;
    assert_eq!(status[survivor], Some(RESTART_STATUS), "{name}: the surviving rank exits with the restart status; log:\n{}", job.log(survivor, 0));
    assert_ne!(status[faulty_rank], Some(0), "{name}: the faulty rank did not finish");
    // No step the ranks did not all agree to was applied: before the agreement the survivor stops.
    let applied = format!("applied {FAULT_STEP}\n");
    if !survivor_may_apply {
        assert!(!job.phases(survivor).contains(&applied), "{name}: the survivor applied step {FAULT_STEP} although a peer was lost before agreeing to it");
    } else {
        // A fault the step may survive into its update comes after the vote: the survivor took part in it.
        assert!(job.phases(survivor).contains(&format!("reduced {FAULT_STEP}\n")), "{name}: the step's vote preceded its update; phases:\n{}", job.phases(survivor));
    }
    let (step, states) = committed_generation(&job);
    assert_eq!(step, FAULT_STEP - 1, "{name}: the newest commit is the last step every rank made durable");
    assert_eq!(states[0], states[1], "{name}: every rank's committed state holds the same weights");

    // Restart: the same command on every rank, no fault.
    let status = job.run(1, &[None, None]);
    assert_eq!(status, [Some(0), Some(0)], "{name}: the restarted job completes; logs:\n{}\n{}", job.log(0, 1), job.log(1, 1));
    assert!(job.log(0, 1).contains(&format!("rebuilding from committed generation {}", FAULT_STEP - 1)), "{name}: {}", job.log(0, 1));
    for rank in 0..2 {
        assert_eq!(weights(&replica(&job, rank)), reference, "{name}: rank {rank} ends with the uninterrupted run's weights");
    }
    std::fs::remove_dir_all(&job.dir).ok();
}

/// Every fault, then a restart, against the uninterrupted run; `device` reduces over NCCL on the cards.
fn every_fault(device: bool) {
    let data = dataset();
    let reference = |sharding: Sharding| {
        let job = job(&format!("reference-{}", sharding.stage()), &data, device, sharding);
        assert_eq!(job.run(0, &[None, None]), [Some(0), Some(0)], "the uninterrupted run ({sharding:?}); logs:\n{}\n{}", job.log(0, 0), job.log(1, 0));
        let weights0 = weights(&replica(&job, 0));
        assert_eq!(weights(&replica(&job, 1)), weights0, "the replicas agree ({sharding:?})");
        assert!(job.phases(0).contains(&format!("committed {STEPS}\n")), "every step was committed ({sharding:?})");
        std::fs::remove_dir_all(&job.dir).ok();
        weights0
    };

    let replicated = reference(Sharding::Replicated);
    let r = Sharding::Replicated;
    lose_and_restart("killed-in-all-reduce", &data, device, r, 1, "kill-in-reduce", &replicated, false);
    lose_and_restart("killed-in-optimizer-step", &data, device, r, 1, "kill-after-agreement", &replicated, true);
    lose_and_restart("killed-in-commit", &data, device, r, 1, "kill-after-state-saved", &replicated, true);
    lose_and_restart("stalled-in-all-reduce", &data, device, r, 1, "stall-in-reduce", &replicated, false);
    lose_and_restart("coordinator-killed-in-all-reduce", &data, device, r, 0, "kill-in-reduce", &replicated, false);

    // The sharded step is the same transaction: reduce-scatter, vote, then update and all-gather.
    let sharded = reference(Sharding::Gradients);
    let g = Sharding::Gradients;
    lose_and_restart("sharded-killed-in-reduce-scatter", &data, device, g, 1, "kill-in-reduce-scatter", &sharded, false);
    lose_and_restart("sharded-stalled-in-reduce-scatter", &data, device, g, 1, "stall-in-reduce-scatter", &sharded, false);
    lose_and_restart("sharded-killed-in-all-gather", &data, device, g, 1, "kill-in-all-gather", &sharded, true);
    std::fs::remove_dir_all(data.parent().unwrap()).ok();
}

#[test]
fn a_job_that_loses_a_rank_anywhere_in_a_step_resumes_from_its_last_commit_and_ends_where_it_would_have() {
    every_fault(false);
}

/// The same over NCCL, the gradients reduced on two cards: a rank lost mid-reduction aborts the communicator on the
/// survivor (its async error, or the op deadline for a stalled peer). Needs the CUDA backend selected, two cards
/// and NCCL; skips by name without them.
#[test]
fn a_job_reducing_on_the_cards_survives_the_same_faults() {
    if brain_testutil::gpu_unavailable(gpu_core::testgpu::missing_gpu()) {
        return;
    }
    if gpu_core::backend_name() != "cuda" {
        brain_testutil::skip_unavailable(&format!("needs the CUDA backend selected (BRAIN_BACKEND=cuda), not '{}'", gpu_core::backend_name()));
        return;
    }
    if let Err(e) = gpu_core::testgpu::stage_cards(None, 2) {
        brain_testutil::skip_unavailable(&format!("two ranks need two cards: {e}"));
        return;
    }
    let probe = gpu_core::Gpu::new_on(gpu_core::testgpu::cards()[0], &[]);
    if let Err(e) = gpu_core::devgroup::RankGroup::supported(&probe) {
        brain_testutil::skip_unavailable(&format!("needs NCCL's per-process communicator: {e}"));
        return;
    }
    every_fault(true);
}

/// The rank process: one rank of the job in `--dir`, with an optional `--fault`, its collective wrapped by the
/// harness so the fault fires in [`FAULT_STEP`] (the step before it has committed).
#[test]
#[ignore = "runs as a child process of the fault-injection test"]
fn rank_process() {
    let Some(args) = RankArgs::from_env() else { return };
    let data = args.flag("data").expect("--data");
    let sharding: Sharding = args.flag("sharding").map_or(Sharding::Replicated, |s| Sharding::parse(&s).expect("a sharding"));
    let device = args.has("device-collective");
    let rank = args.rank;
    let (context, faulty) = args.join(device, FAULT_STEP);
    let observe = |phase, step| faulty.observe(phase, step);
    let identity = context.plan().identity();
    let sync = GradSync { collective: &faulty, rank, checkpoint: Some(CheckpointIdentity { job: &context.plan().job, plan: &identity }), observer: Some(&observe) };
    let out = args.dir.join(format!("replica-{rank}.safetensors"));
    let state = args.dir.join("state.safetensors");
    let train = || gpt2::train::train_distributed_resumable(Path::new(&data), config(), &opts(sharding, device), None, sync, &state, 1);
    let trained = if device { gpu_core::devices::with_gpu(gpu_core::testgpu::cards()[rank].index, train).unwrap_or_else(|e| Err(std::io::Error::other(e))) } else { train() };
    args.exit(trained.map(|(_report, replica)| replica.save(out.to_str().expect("a UTF-8 path"))));
}
