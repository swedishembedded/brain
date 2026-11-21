// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full-parameter fine-tuning of hybrid linear-attention models larger than one
// accelerator, sharded over the cards of a cluster, for its clients. If your team needs expertise in ZeRO-3-style
// parameter sharding, you can procure our services by sending an email to info@swedishembedded.com.

//! The tiny hybrid Qwen3.5 trained with its parameters sharded (ZeRO-3) over in-process ranks through the generic fit
//! loop: a layer-streamed bf16 resident build whose layers and tables are gathered from the ranks' slices and
//! reduce-scattered to their owners. On a dataset whose every row is the same, every rank's gradient is the single
//! rank's, so the sharded run follows the single-rank layer-streamed run within the sharded update's rounding, and
//! every rank reports the same losses and global gradient norms bit for bit.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use data::binio::{self, Meta};
use gpu_core::Gpu;
use model::memplan::Capacity;
use model::resident_ft::{ClipPass, LayerWindow, ResidentFtConfig, ResidentPlan, ResidentValue};
use model::saved::SavedPolicy;
use model::zero::{ShardGroup, Sharding};
use model::{Collective, GradSync, HostCollective};
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const T: u32 = 16;

fn cfg() -> Qwen35Config {
    Qwen35Config { block_size: T, max_position_embeddings: T, ..Qwen35Config::tiny() }
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen35-zero3-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One token, repeated: every row drawn anywhere in the stream is the same, so every rank's batch is the single
/// rank's.
fn write_dataset(dir: &Path) {
    let tokens = vec![7u32; 800];
    let mask = vec![true; 800];
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(cfg().vocab as usize)).unwrap();
}

fn opts(sharding: Sharding, seed: u64) -> model::FitOpts {
    model::FitOpts { steps: 4, batch_size: 1, block_size: T, lr: 1e-2, min_lr: 1e-3, warmup: 2, decay_iters: 4, weight_decay: 0.0, grad_clip: 0.5, eval_interval: 0, eval_batches: 0, seed, sharding, ..Default::default() }
}

fn gpu() -> Gpu {
    let mut list = pipelines().to_vec();
    model::ops::extend_kernel_list(&mut list, model::zero::KERNELS);
    Gpu::new(Box::leak(list.into_boxed_slice()))
}

fn build(shard: Option<ShardGroup>) -> Qwen35 {
    build_with(shard, ClipPass::TwoPass)
}

fn build_with(shard: Option<ShardGroup>, clip_pass: ClipPass) -> Qwen35 {
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
        clip_pass,
        shard,
    };
    let plan = ResidentPlan { cfg: resident, capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 }, others: Vec::new() };
    let c = cfg();
    let init = qwen35::init::init_weights(&c, 11);
    let train = TrainOpts { saved: SavedPolicy::Recompute, gemm_precision: gpu_core::provider::library::GemmPrecision::Fp32Reference, resident: Some(plan), ..TrainOpts::default() };
    Qwen35::new_resident(gpu(), c, 1, T, &init, &train).expect("builds")
}

/// Each step's (loss, global gradient norm).
fn losses(dir: &Path, opts: &model::FitOpts, m: Qwen35, sync: Option<GradSync<'_>>) -> Vec<(f32, f32)> {
    run(dir, opts, m, sync, None, None, None)
}

/// [`losses`] writing the trained model to `out`, keeping the resume state at `state`, stopping after `stop` steps.
fn run(dir: &Path, opts: &model::FitOpts, m: Qwen35, sync: Option<GradSync<'_>>, out: Option<&Path>, state: Option<&Path>, stop: Option<u32>) -> Vec<(f32, f32)> {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    let mut seen = Vec::new();
    let mut on_step = |r: &model::StepReport| {
        seen.push((r.loss, r.grad_norm.unwrap_or(f32::NAN)));
        stop.is_none_or(|n| r.step < n)
    };
    let control = model::FitControl { on_step: Some(&mut on_step), grad_sync: sync, state, identity: serde_json::json!({ "test": "zero3" }), ..Default::default() };
    model::fit_controlled(m, model::causal_lm::<Qwen35>(train, val, bcfg, itos), opts, out, control).unwrap();
    seen
}

/// Run `world` ranks of a parameter-sharded fit as threads; every rank's (loss, norm) per step.
fn ranks(dir: &Path, world: usize, opts: &model::FitOpts, clip_pass: ClipPass, out: Option<&Path>, state: Option<&(dyn Fn(usize) -> PathBuf + Sync)>, stop: Option<u32>) -> Vec<Vec<(f32, f32)>> {
    let coll = HostCollective::new(world);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let coll = Arc::clone(&coll);
                s.spawn(move || {
                    let shared: Arc<dyn Collective> = coll.clone();
                    let m = build_with(Some(ShardGroup { collective: shared, rank }), clip_pass);
                    let sync = GradSync { collective: &*coll, rank, checkpoint: None, observer: None };
                    let path = state.map(|f| f(rank));
                    run(dir, opts, m, Some(sync), out, path.as_deref(), stop)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn bits(r: &[(f32, f32)]) -> Vec<(u32, u32)> {
    r.iter().map(|s| (s.0.to_bits(), s.1.to_bits())).collect()
}

/// A sharded run stopped after two steps with every rank's state committed, then restarted in fresh models, ends where
/// the uninterrupted run does bit for bit (its losses and its written model); the written model opens as a qwen35
/// checkpoint (bf16 linears and tables, f32 norms) whose weights are the single-rank run's within bf16 rounding.
#[test]
fn a_parameter_sharded_run_resumes_exactly_and_writes_a_checkpoint() {
    let dir = tmp("resume");
    write_dataset(&dir);
    let seed = data::rng::random_seed();
    let o = opts(Sharding::Parameters, seed);
    let (straight_out, resumed_out, single_out) = (dir.join("straight.safetensors"), dir.join("resumed.safetensors"), dir.join("single.safetensors"));
    let straight = ranks(&dir, 2, &o, ClipPass::TwoPass, Some(&straight_out), None, None);
    let state = |rank: usize| dir.join(format!("rank{rank}.state"));
    ranks(&dir, 2, &o, ClipPass::TwoPass, None, Some(&state), Some(2));
    let resumed = ranks(&dir, 2, &o, ClipPass::TwoPass, Some(&resumed_out), Some(&state), None);
    assert_eq!(bits(&resumed[0]), bits(&straight[0][2..]), "the resumed steps differ from the uninterrupted run's");
    assert_eq!(std::fs::read(&resumed_out).unwrap(), std::fs::read(&straight_out).unwrap(), "the resumed run wrote another model");
    // The written model opens like any qwen35 base and holds the single-rank run's weights within bf16 rounding.
    losses_to(&dir, &opts(Sharding::Replicated, seed), &single_out);
    let got = qwen35::base_source::open_base(&straight_out).unwrap();
    let want = qwen35::base_source::open_base(&single_out).unwrap();
    assert_eq!(got.cfg.n_layers, want.cfg.n_layers);
    for (name, numel) in cfg().param_list().into_iter().filter(|(n, _)| !model::adapter::device::is_adapter_param(n)) {
        let (a, b) = (read_all(&*got.src, &name, numel), read_all(&*want.src, &name, numel));
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let worst = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        // bf16 rounding of the written values, and the sharded update's rounding: AdamW's normalised step turns a
        // rounding difference in a near-zero gradient into up to a fraction of the learning rate.
        let bound = 4e-3 * scale + 0.1 * o.lr;
        assert!(worst <= bound, "{name}: {worst:e} from the single-rank run (bound {bound:e})");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

fn losses_to(dir: &Path, opts: &model::FitOpts, out: &Path) {
    run(dir, opts, build(None), None, Some(out), None, None);
}

fn read_all(src: &dyn checkpoint::TensorSource, name: &str, numel: usize) -> Vec<f32> {
    let mut out = vec![0f32; numel];
    assert!(checkpoint::read_range_into(src, name, 0, &mut out), "{name} is not in the checkpoint");
    out
}

/// The single-pass running clip: one backward a step, every rank the same losses and norms, and the run trains.
#[test]
fn a_parameter_sharded_run_clips_in_one_pass() {
    let dir = tmp("running");
    write_dataset(&dir);
    let got = ranks(&dir, 2, &opts(Sharding::Parameters, data::rng::random_seed()), ClipPass::Running, None, None, None);
    assert_eq!(bits(&got[0]), bits(&got[1]), "the ranks' losses or norms differ");
    assert!(got[0][3].0 < got[0][0].0 / 2.0, "the run did not train: {:?}", got[0]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_parameter_sharded_run_follows_the_single_rank_run() {
    let dir = tmp("follow");
    write_dataset(&dir);
    let seed = data::rng::random_seed();
    let single = losses(&dir, &opts(Sharding::Replicated, seed), build(None), None);
    let world = 2;
    let coll = HostCollective::new(world);
    let ranks: Vec<Vec<(f32, f32)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let coll = Arc::clone(&coll);
                let dir = &dir;
                s.spawn(move || {
                    let shared: Arc<dyn Collective> = coll.clone();
                    let m = build(Some(ShardGroup { collective: shared, rank }));
                    let sync = GradSync { collective: &*coll, rank, checkpoint: None, observer: None };
                    losses(dir, &opts(Sharding::Parameters, seed), m, Some(sync))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let bits = |r: &[(f32, f32)]| r.iter().map(|s| (s.0.to_bits(), s.1.to_bits())).collect::<Vec<_>>();
    assert_eq!(bits(&ranks[0]), bits(&ranks[1]), "the ranks' losses or global norms differ");
    for (step, ((a, na), (b, nb))) in single.iter().zip(&ranks[0]).enumerate() {
        assert!((a - b).abs() <= 2e-3 * a.abs().max(1e-2), "step {step}: sharded loss {b}, single {a} ({single:?} vs {:?})", ranks[0]);
        assert!((na - nb).abs() <= 2e-3 * na.abs(), "step {step}: sharded norm {nb}, single {na}");
    }
    assert!(ranks[0][3].0 < ranks[0][0].0 / 2.0, "the sharded run did not train: {:?}", ranks[0]);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The measurement harness of a parameter-sharded full fine-tune of a real checkpoint: one process per card, started
/// by a launcher with `-- --plan=P --rank=N --secret=S --base=DIR --data=DIR --steps=K --block=T --window=W` (the
/// launcher restricts each process to its card; `--layers=L` trains only the first `L` layers, for a smoke run;
/// `--clip=running` clips in one backward pass instead of two). Trains `K` steps on a synthetic token stream and prints each step's
/// time, phases, loss, global norm and device memory. Ignored: it needs the checkpoint, the cards and a job.
#[test]
#[ignore = "measurement harness: one process per card under a launcher"]
fn measure_sharded_parameters_rank() {
    let args: Vec<String> = std::env::args().collect();
    let Some(separator) = args.iter().position(|a| a == "--") else { return };
    let after = &args[separator + 1..];
    let flag = |name: &str| after.iter().find_map(|a| a.strip_prefix(name).map(str::to_string));
    let num = |name: &str, default: u32| flag(name).and_then(|v| v.parse().ok()).unwrap_or(default);
    let (Some(plan), Some(rank), Some(secret), Some(base), Some(data)) = (flag("--plan="), flag("--rank=").and_then(|r| r.parse::<usize>().ok()), flag("--secret="), flag("--base="), flag("--data=")) else {
        return;
    };
    let (steps, block, window) = (num("--steps=", 3), num("--block=", 1024), num("--window=", 2));
    let clip = flag("--clip=").unwrap_or_else(|| "two-pass".into());
    let started = std::time::Instant::now();
    let ctx = model::JobContext::from_plan_file(Path::new(&plan), rank, Path::new(&secret)).unwrap();
    let job: Arc<dyn Collective> = Arc::new(ctx.job_collective().unwrap());
    let qwen35::base_source::BaseCheckpoint { mut cfg, src } = qwen35::base_source::open_base(Path::new(&base)).unwrap();
    if let Some(layers) = flag("--layers=").and_then(|l| l.parse().ok()) {
        cfg.n_layers = layers;
    }
    // A synthetic stream over the model's vocabulary, the same on every rank.
    let dir = PathBuf::from(&data).join(format!("rank{rank}"));
    std::fs::create_dir_all(&dir).unwrap();
    let tokens: Vec<u32> = (0..64 * block).map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % cfg.vocab).collect();
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &vec![true; tokens.len()]).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(cfg.vocab as usize)).unwrap();
    let precision = model::TrainPrecision::parse(&format!("weights=bf16,grad=bf16,window={window},clip={clip},activations=recompute")).unwrap();
    let opts = model::FitOpts {
        steps,
        batch_size: 1,
        block_size: block,
        lr: 1e-5,
        min_lr: 1e-6,
        warmup: 0,
        decay_iters: steps,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        initial_probe_batches: 0,
        seed: 7,
        precision,
        sharding: Sharding::Parameters,
        parameter_group: Some(ShardGroup { collective: Arc::clone(&job), rank }),
        ..Default::default()
    };
    let resident = opts.resident_config().unwrap().expect("a resident build");
    let m = qwen35::finetune::build_resident(cfg, &opts, &*src, &resident).unwrap();
    eprintln!("rank {rank}: built in {:.1} s", started.elapsed().as_secs_f64());
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(&dir, &opts).unwrap();
    let mut on_step = |r: &model::StepReport| {
        let w = r.work;
        let mem = w.device_memory.map_or(String::new(), |m| format!(" peak {:.2} GiB", m.peak_live as f64 / (1u64 << 30) as f64));
        eprintln!(
            "rank {rank} step {} loss {:.4} norm {:?} {:.2} s (forward {:.2} backward {:.2} optimizer {:.2}){mem}",
            r.step,
            r.loss,
            r.grad_norm,
            w.step_secs(),
            w.forward_secs,
            w.backward_secs,
            w.optimizer_secs
        );
        true
    };
    let sync = GradSync { collective: &*job, rank, checkpoint: None, observer: None };
    let control = model::FitControl { on_step: Some(&mut on_step), grad_sync: Some(sync), ..Default::default() };
    model::fit_controlled(m, model::causal_lm::<Qwen35>(train, val, bcfg, itos), &opts, None, control).unwrap();
    eprintln!("rank {rank}: done in {:.1} s", started.elapsed().as_secs_f64());
}
