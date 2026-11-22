// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for parameter-sharded training jobs that survive a lost rank without
// corrupting their state for its clients. If your team needs expertise in collective failure handling for ZeRO-3
// style training and checkpoint recovery then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A parameter-sharded (ZeRO-3) job of the tiny hybrid Qwen3.5 that loses a rank inside a step, and is restarted.
//!
//! Two OS processes (this test binary re-executed by `model::conformance_faults`) train one plan over loopback, each
//! holding its slices of every layer and committing its exact-resume state every step. The sharded step is not a
//! voted transaction: each layer is all-gathered where the forward and backward use it, and updated on its owners as
//! soon as its reduce-scatter completes, inside the backward. In each case one rank fails at step [`FAULT_STEP`]:
//! killed in the forward's first all-gather (before any update), killed or stalled in the backward's second
//! reduce-scatter (its peer has already updated the first layer of the step), the coordinator killed there, or killed
//! once its state of the step is durable (before the commit). Then:
//!
//! - the surviving rank exits with the restart status, not a panic, and never reports the step applied when it
//!   failed inside it;
//! - no generation after the last one every rank completed is committed: the newest commit is step `FAULT_STEP - 1`,
//!   every rank's file of it verified, so a half-updated rank's state can never be resumed from;
//! - the job started again with the same arguments resumes from that generation and writes the model the
//!   uninterrupted run writes, byte for byte (the failed step's partial updates are gone with the process).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use data::binio::{self, Meta};
use gpu_core::Gpu;
use model::conformance_faults::{FaultJob, RankArgs};
use model::jobfault::RESTART_STATUS;
use model::memplan::Capacity;
use model::resident_ft::{ClipPass, LayerWindow, ResidentFtConfig, ResidentPlan, ResidentValue};
use model::saved::SavedPolicy;
use model::zero::{ShardGroup, Sharding};
use model::{CheckpointIdentity, Collective, GradSync};
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const CHILD: &str = "sharded_faults::rank_process";
const T: u32 = 16;
const STEPS: u32 = 6;
/// The step every fault happens in.
const FAULT_STEP: u32 = 4;

fn cfg() -> Qwen35Config {
    Qwen35Config { block_size: T, max_position_embeddings: T, ..Qwen35Config::tiny() }
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("brain-qwen35-zero3-faults-{}-{name}", std::process::id()))
}

/// A short token cycle, so the run learns something every step.
fn write_dataset(dir: &Path) {
    let tokens: Vec<u32> = (0..800u32).map(|i| 3 + i % 7).collect();
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &vec![true; tokens.len()]).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(cfg().vocab as usize)).unwrap();
}

/// No clip: one backward a step that updates each layer as its gradient is reduced.
fn opts() -> model::FitOpts {
    model::FitOpts { steps: STEPS, batch_size: 1, block_size: T, lr: 1e-2, min_lr: 1e-3, warmup: 2, decay_iters: STEPS, weight_decay: 0.0, grad_clip: 0.0, eval_interval: 0, eval_batches: 0, seed: 11, sharding: Sharding::Parameters, ..Default::default() }
}

fn build(shard: ShardGroup) -> Qwen35 {
    let resident = ResidentFtConfig {
        value: ResidentValue::Bf16,
        master: MasterPolicy::F32,
        state: StateFormat::F32,
        rule: Rule::AdamW { adam: model::Adam::default(), wd: 0.0 },
        grad: GradDtype::Bf16,
        stochastic_state: false,
        seed: None,
        tile_elems: 256,
        state_dir: None,
        defer_publication: false,
        window: Some(LayerWindow { layers: 1 }),
        clip_pass: ClipPass::TwoPass,
        shard: Some(shard),
    };
    let plan = ResidentPlan { cfg: resident, capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 }, others: Vec::new() };
    let mut list = pipelines().to_vec();
    model::ops::extend_kernel_list(&mut list, model::zero::KERNELS);
    let gpu = Gpu::new(Box::leak(list.into_boxed_slice()));
    let c = cfg();
    let init = qwen35::init::init_weights(&c, 11);
    let train = TrainOpts { saved: SavedPolicy::Recompute, gemm_precision: gpu_core::provider::library::GemmPrecision::Fp32Reference, resident: Some(plan), ..TrainOpts::default() };
    Qwen35::new_resident(gpu, c, 1, T, &init, &train).expect("builds")
}

fn job(name: &str, data: &Path, device: bool) -> FaultJob {
    let mut args = vec![format!("--data={}", data.display())];
    if device {
        args.push("--device-collective".into());
    }
    FaultJob::new(scratch(name), &format!("qwen35-zero3-faults-{name}"), 2, device, CHILD, args).unwrap()
}

/// The model rank 0 wrote at the end of the run.
fn written(job: &FaultJob) -> Vec<u8> {
    std::fs::read(job.dir.join("model.safetensors")).expect("the written model")
}

/// Lose a rank to `fault`, check what the failure left, restart the job and compare its model with `reference`.
fn lose_and_restart(name: &str, data: &Path, device: bool, faulty_rank: usize, fault: &str, reference: &[u8], survivor_may_apply: bool) {
    let job = job(name, data, device);
    let mut faults = [None, None];
    faults[faulty_rank] = Some(fault);
    let status = job.run(0, &faults);
    let survivor = 1 - faulty_rank;
    let log = job.log(survivor, 0);
    assert_eq!(status[survivor], Some(RESTART_STATUS), "{name}: the surviving rank exits with the restart status; log:\n{log}");
    assert!(!log.contains("panicked"), "{name}: the survivor ended on a panic; log:\n{log}");
    assert_ne!(status[faulty_rank], Some(0), "{name}: the faulty rank did not finish");
    if !survivor_may_apply {
        assert!(!job.phases(survivor).contains(&format!("applied {FAULT_STEP}\n")), "{name}: the survivor reported step {FAULT_STEP} applied although it failed inside it");
    }
    let (step, _files) = job.committed("state");
    assert_eq!(step, FAULT_STEP - 1, "{name}: the newest commit is the last step every rank completed");

    let status = job.run(1, &[None, None]);
    assert_eq!(status, [Some(0), Some(0)], "{name}: the restarted job completes; logs:\n{}\n{}", job.log(0, 1), job.log(1, 1));
    assert!(job.log(0, 1).contains(&format!("rebuilding from committed generation {}", FAULT_STEP - 1)), "{name}: {}", job.log(0, 1));
    assert!(written(&job) == reference, "{name}: the restarted job wrote another model than the uninterrupted run");
    std::fs::remove_dir_all(&job.dir).ok();
}

fn every_fault(device: bool) {
    let data = scratch("data");
    let _ = std::fs::remove_dir_all(&data);
    std::fs::create_dir_all(&data).unwrap();
    write_dataset(&data);
    let reference = {
        let job = job("reference", &data, device);
        assert_eq!(job.run(0, &[None, None]), [Some(0), Some(0)], "the uninterrupted run; logs:\n{}\n{}", job.log(0, 0), job.log(1, 0));
        assert!(job.phases(0).contains(&format!("committed {STEPS}\n")), "every step was committed");
        let model = written(&job);
        std::fs::remove_dir_all(&job.dir).ok();
        model
    };
    lose_and_restart("killed-in-forward-all-gather", &data, device, 1, "kill-in-all-gather", &reference, false);
    lose_and_restart("killed-mid-update", &data, device, 1, "kill-in-reduce-scatter@2", &reference, false);
    lose_and_restart("stalled-mid-update", &data, device, 1, "stall-in-reduce-scatter@2", &reference, false);
    lose_and_restart("coordinator-killed-mid-update", &data, device, 0, "kill-in-reduce-scatter@2", &reference, false);
    lose_and_restart("killed-in-commit", &data, device, 1, "kill-after-state-saved", &reference, true);
    std::fs::remove_dir_all(&data).ok();
}

#[test]
fn a_parameter_sharded_job_that_loses_a_rank_inside_a_step_restarts_from_its_last_commit() {
    every_fault(false);
}

/// The same over NCCL, each rank's slices on its own card: a lost peer aborts the communicator on the survivor (its
/// async error, or the op deadline the watchdog bounds every enqueued collective with). Needs the CUDA backend
/// selected, two cards and NCCL; skips by name without them.
#[test]
fn a_parameter_sharded_job_on_the_cards_survives_the_same_faults() {
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
    let probe = Gpu::new_on(gpu_core::testgpu::cards()[0], &[]);
    if let Err(e) = gpu_core::devgroup::RankGroup::supported(&probe) {
        brain_testutil::skip_unavailable(&format!("needs NCCL's per-process communicator: {e}"));
        return;
    }
    every_fault(true);
}

/// One rank of the job in `--dir`, with an optional `--fault` armed for [`FAULT_STEP`].
#[test]
#[ignore = "runs as a child process of the fault-injection test"]
fn rank_process() {
    let Some(args) = RankArgs::from_env() else { return };
    let data = PathBuf::from(args.flag("data").expect("--data"));
    let device = args.has("device-collective");
    let rank = args.rank;
    let (context, faulty) = args.join(device, FAULT_STEP);
    let faulty = Arc::new(faulty);
    let observe = |phase, step| faulty.observe(phase, step);
    let identity = context.plan().identity();
    let train = || -> std::io::Result<()> {
        let group: Arc<dyn Collective> = faulty.clone();
        let m = build(ShardGroup { collective: group, rank });
        let opts = opts();
        let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(&data, &opts)?;
        let sync = GradSync { collective: &*faulty, rank, checkpoint: Some(CheckpointIdentity { job: &context.plan().job, plan: &identity }), observer: Some(&observe) };
        let state = args.dir.join("state");
        let control = model::FitControl { grad_sync: Some(sync), state: Some(&state), state_every: 1, identity: serde_json::json!({ "test": "zero3-faults" }), ..Default::default() };
        model::fit_controlled(m, model::causal_lm::<Qwen35>(train, val, bcfg, itos), &opts, Some(&args.dir.join("model.safetensors")), control).map(|_| ())
    };
    let trained = if device { gpu_core::devices::with_gpu(gpu_core::testgpu::cards()[rank].index, train).unwrap_or_else(|e| Err(std::io::Error::other(e))) } else { train() };
    args.exit(trained);
}
