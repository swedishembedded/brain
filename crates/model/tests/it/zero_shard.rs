// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sharded data-parallel training and its checkpoints for its clients. If your team
// needs expertise in resuming a sharded run on another number of cards, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The sharded optimizer ([`model::zero::ZeroOptimizer`]) on raw device tensors over in-process ranks: a checkpoint
//! saved by one world size loads into another with the logical tensors unchanged, bf16 parameters shard and gather
//! to the same bytes on every rank, and what cannot be sharded is refused by what it lacks. A parameter-sharded unit
//! can be gathered ahead of its use and placed into the parameters only where it is used.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use gpu_core::{CollectiveDtype, Gpu};
use model::zero::{self, DeviceParam, DeviceParams, Sharding, StateTier, StepArgs, StepOutcome, ZeroOptimizer};
use model::{Batch, HostCollective, Model, ModelConfig};

/// Tensor sizes: odd f32 lengths, one larger than the unit cap below, one empty-ish small one.
pub(crate) const SIZES: [u64; 5] = [7, 130, 33, 2, 61];
pub(crate) const UNIT: u64 = 64;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-zero-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub(crate) fn values(seed: u32, n: u64) -> Vec<f32> {
    let mut lcg = data::rng::Lcg::new(u64::from(seed) + 1);
    (0..n).map(|_| lcg.unit() - 0.5).collect()
}

/// The tensors one rank trains, on `gpu`, with initial values from `seed` (the same on every rank for a seed).
pub(crate) fn params(gpu: &Gpu, seed: u32, dtype: CollectiveDtype) -> DeviceParams {
    let tensors = SIZES
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let n = if dtype == CollectiveDtype::Bf16 { n.div_ceil(2) * 2 } else { n };
            let words = (n * dtype.bytes()).div_ceil(4);
            let (value, grad) = (gpu.storage(words), gpu.storage(words));
            zero::write_elems(gpu, &value, 0, &values(seed * 31 + i as u32, n), dtype);
            DeviceParam { name: format!("p{i}"), value, value_dtype: dtype, grad, grad_dtype: dtype, numel: n, lr_mult: 1.0 }
        })
        .collect();
    DeviceParams { gpu: gpu.share(), tensors }
}

pub(crate) fn args(t: u32) -> StepArgs {
    StepArgs { t, lr: 1e-2, wd: 0.1, adam: model::Adam::default(), clip: Some(1.0), extra_scale: 1.0 }
}

/// Rank `rank`'s gradient of tensor `i` at step `t`.
pub(crate) fn grad(t: u32, rank: usize, i: usize, n: u64) -> Vec<f32> {
    values(1000 + t * 97 + rank as u32 * 13 + i as u32, n)
}

/// Run `world` ranks as threads: each builds its tensors from `seed`, takes `steps` sharded steps on rank-specific
/// gradients (after restoring from `restore` when given) and saves to `save` when given; returns every rank's
/// parameter values.
#[allow(clippy::too_many_arguments)] // the run's shape, written out at every call
fn ranks(world: usize, sharding: Sharding, tier: StateTier, dtype: CollectiveDtype, seed: u32, restore: Option<(&Path, u32)>, steps: u32, save: Option<&Path>) -> Vec<Vec<Vec<f32>>> {
    let coll = HostCollective::new(world);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let coll = Arc::clone(&coll);
                s.spawn(move || {
                    let gpu = gpu_core::testgpu::dev(zero::KERNELS);
                    let p = params(&gpu, seed, dtype);
                    let handles: Vec<DeviceParam> = p.tensors.clone();
                    let mut opt = ZeroOptimizer::new(p, sharding, tier, &*coll, rank, UNIT).unwrap();
                    let first = match restore {
                        Some((dir, step)) => {
                            opt.restore(&*coll, dir, step).unwrap();
                            step
                        }
                        None => 0,
                    };
                    for t in first + 1..=first + steps {
                        for (i, h) in handles.iter().enumerate() {
                            zero::write_elems(&gpu, &h.grad, 0, &grad(t, rank, i, h.numel), dtype);
                        }
                        assert!(matches!(opt.step(&*coll, 1.0 + rank as f32, 1.0, &args(t)).unwrap(), StepOutcome::Applied { .. }));
                    }
                    if let Some(dir) = save {
                        opt.save(dir, serde_json::json!({ "rank0": rank == 0 })).unwrap();
                    }
                    handles.iter().map(|h| zero::read_elems(&gpu, &h.value, 0, h.numel, dtype)).collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// A run saved at four ranks (state on the cards) resumes at two (state in host memory) and at four from the same
/// logical master and moments; the restored parameters are the saved master; a save at the new world holds the same
/// logical tensors; and the two tiers train the same steps to the same parameters within the update's arithmetic.
#[test]
fn a_checkpoint_saved_at_one_world_loads_into_another() {
    let dir = scratch("reshard");
    let (four, two) = (dir.join("four"), dir.join("two"));
    let saved = ranks(4, Sharding::Gradients, StateTier::Device, CollectiveDtype::F32, 3, None, 2, Some(&four));
    let logical = zero::read_logical(&four, 2).unwrap();
    assert_eq!(logical.step, 2);
    assert_eq!(logical.master, saved[0], "the logical master is what every rank's parameters hold");
    for (rank, w) in saved.iter().enumerate() {
        assert_eq!(w, &saved[0], "rank {rank}'s parameters differ from rank 0's");
    }
    // Built from other initial values, then restored: the checkpoint decides the parameters.
    let resumed = ranks(2, Sharding::Optimizer, StateTier::Host, CollectiveDtype::F32, 9, Some((&four, 2)), 0, Some(&two));
    assert_eq!(resumed[0], logical.master, "restored into two ranks, the parameters are the saved master");
    let again = zero::read_logical(&two, 2).unwrap();
    for (what, a, b) in [("master", &again.master, &logical.master), ("m", &again.m, &logical.m), ("v", &again.v, &logical.v)] {
        assert_eq!(a, b, "the {what} tensors changed across the world-size change");
    }
    let on_four = ranks(4, Sharding::Gradients, StateTier::Device, CollectiveDtype::F32, 9, Some((&four, 2)), 1, None);
    let on_four_again = ranks(4, Sharding::Gradients, StateTier::Device, CollectiveDtype::F32, 9, Some((&four, 2)), 1, None);
    assert_eq!(on_four, on_four_again, "a resume on an unchanged layout is repeatable bit for bit");
    let on_host = ranks(4, Sharding::Gradients, StateTier::Host, CollectiveDtype::F32, 9, Some((&four, 2)), 1, None);
    let worst = on_host[0].iter().flatten().zip(on_four[0].iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst <= 1e-2 * args(1).lr, "the host and the device update differ by {worst:e}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// bf16 parameters and gradients, on either tier: every rank gathers the same bytes, and they are the master rounded
/// to bf16.
#[test]
fn bf16_parameters_shard_and_gather_to_the_same_bytes() {
    for tier in [StateTier::Device, StateTier::Host] {
        let dir = scratch(&format!("bf16-{tier:?}"));
        let out = ranks(3, Sharding::Gradients, tier, CollectiveDtype::Bf16, 5, None, 2, Some(&dir));
        for w in &out {
            assert_eq!(w, &out[0]);
        }
        let logical = zero::read_logical(&dir, 2).unwrap();
        for (got, master) in out[0].iter().zip(&logical.master) {
            let rounded: Vec<f32> = model::half::pack_bf16(master).iter().flat_map(|w| [f32::from_bits(w << 16), f32::from_bits(w & 0xFFFF_0000)]).take(master.len()).collect();
            assert_eq!(got, &rounded, "{tier:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// A non-finite loss on one rank skips the step on every rank and changes nothing.
#[test]
fn a_non_finite_loss_on_one_rank_skips_the_step_everywhere() {
    let coll = HostCollective::new(2);
    let out: Vec<(StepOutcome, bool)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                let coll = Arc::clone(&coll);
                s.spawn(move || {
                    let gpu = gpu_core::testgpu::dev(zero::KERNELS);
                    let p = params(&gpu, 1, CollectiveDtype::F32);
                    let h = p.tensors[0].clone();
                    let before = zero::read_elems(&gpu, &h.value, 0, h.numel, CollectiveDtype::F32);
                    let mut opt = ZeroOptimizer::new(p, Sharding::Optimizer, StateTier::Device, &*coll, rank, UNIT).unwrap();
                    let outcome = opt.step(&*coll, 1.0, if rank == 0 { 1.0 } else { f32::NAN }, &args(1)).unwrap();
                    (outcome, zero::read_elems(&gpu, &h.value, 0, h.numel, CollectiveDtype::F32) == before && opt.step_count() == 0)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (outcome, unchanged) in out {
        assert!(matches!(outcome, StepOutcome::Skipped(ref why) if why.contains("non-finite loss")), "{outcome:?}");
        assert!(unchanged, "a skipped step changed a parameter or the step count");
    }
}

/// What cannot be sharded is refused by what it lacks.
#[test]
fn sharding_is_refused_by_what_is_missing() {
    let coll = HostCollective::new(1);
    let gpu = gpu_core::testgpu::dev(zero::KERNELS);
    let err = |sharding| ZeroOptimizer::new(params(&gpu, 1, CollectiveDtype::F32), sharding, StateTier::Device, &*coll, 0, UNIT).err().unwrap();
    assert!(err(Sharding::Replicated).contains("no state to shard"));
    assert!(err(Sharding::Parameters).contains("gathers each block's parameters"));
    // A CPU device without the sharded step's kernels cannot hold the state on the card; the host tier needs none.
    let bare = gpu_core::testgpu::dev(&[]);
    if bare.kind() == "cpu" {
        let why = ZeroOptimizer::new(params(&bare, 1, CollectiveDtype::F32), Sharding::Gradients, StateTier::Device, &*coll, 0, UNIT).err().unwrap();
        assert!(why.contains("moments=host"), "{why}");
    }
    assert!(ZeroOptimizer::new(params(&bare, 1, CollectiveDtype::F32), Sharding::Gradients, StateTier::Host, &*coll, 0, UNIT).is_ok());
    let mut odd = params(&gpu, 1, CollectiveDtype::Bf16);
    odd.tensors[0].numel = 7;
    let why = ZeroOptimizer::new(odd, Sharding::Optimizer, StateTier::Host, &*coll, 0, UNIT).err().unwrap();
    assert!(why.contains("whole number of 32-bit words"), "{why}");
}

// ---- the fit loop's refusals --------------------------------------------------------------------------------------

/// A model with no parameter store a sharded optimizer reaches.
#[derive(Clone)]
struct HostCfg;
impl ModelConfig for HostCfg {
    fn param_list(&self) -> Vec<(String, usize)> {
        vec![("w".into(), 2)]
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    fn from_json(_: &serde_json::Value) -> Self {
        HostCfg
    }
    fn vocab(&self) -> u32 {
        0
    }
    fn block_size(&self) -> u32 {
        0
    }
    fn finalize_for_dataset(self, _: u32, _: u32) -> Self {
        self
    }
}
struct HostModel(HostCfg);
impl Model for HostModel {
    type Config = HostCfg;
    fn new(cfg: HostCfg, _: u32, _: u32, _: &std::collections::HashMap<String, Vec<f32>>) -> Self {
        HostModel(cfg)
    }
    fn init_weights(_: &HostCfg, _: u64) -> std::collections::HashMap<String, Vec<f32>> {
        Default::default()
    }
    fn config(&self) -> &HostCfg {
        &self.0
    }
    fn set_batch(&self, _: Batch) {}
    fn forward(&self) -> f32 {
        0.0
    }
    fn backward(&self) {}
    fn zero_grads(&self) {}
    fn adamw_step(&self, _: u32, _: f32, _: f32, _: model::Adam, _: Option<f32>, _: f32) {}
    fn poll_wait(&self) {}
    fn param_names(&self) -> Vec<String> {
        vec!["w".into()]
    }
    fn read_weight(&self, _: &str) -> Vec<f32> {
        vec![0.0; 2]
    }
    fn write_weight(&self, _: &str, _: &[f32]) {}
    fn read_grad(&self, _: &str) -> Vec<f32> {
        vec![0.0; 2]
    }
    fn logits_all(&self, _: &[u32]) -> Option<Vec<f32>> {
        None
    }
    fn save(&self, _: &str) {}
    fn config_json(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

struct Nothing;
impl model::Objective<HostModel> for Nothing {
    fn regime(&self) -> &'static str {
        "nothing"
    }
    fn shard(&mut self, _: usize, _: usize) -> std::io::Result<()> {
        Ok(())
    }
    fn micro_step(&mut self, _: &HostModel, _: &mut data::rng::Rng) -> f32 {
        0.0
    }
    fn eval(&mut self, _: &HostModel, _: &mut data::rng::Rng, _: u32) -> Option<f32> {
        None
    }
}

fn fit_error(sharding: Sharding, group: bool, precision: &str) -> String {
    let coll = HostCollective::new(1);
    let opts = model::FitOpts { steps: 1, sharding, precision: if precision.is_empty() { model::TrainPrecision::default() } else { model::TrainPrecision::parse(precision).unwrap() }, ..Default::default() };
    let control = model::FitControl { publish: None, grad_sync: group.then(|| model::GradSync { collective: &*coll, rank: 0, checkpoint: None, observer: None }), ..Default::default() };
    match model::fit_controlled(HostModel(HostCfg), Nothing, &opts, None, control) {
        Ok(_) => "trained".into(),
        Err(e) => e.to_string(),
    }
}

/// Parameter sharding gathers each layer through a layer-streamed window into bf16 slots with bf16 gradient lanes: a
/// resident precision the sharded build does not run is refused by the rule a plan consults before a model is built,
/// with what to ask for, not when the build reaches its first layer.
#[test]
fn parameter_sharding_is_admitted_only_with_a_layer_window() {
    let admits = |spec: &str| Sharding::Parameters.admits(&model::TrainPrecision::parse(spec).unwrap());
    assert!(admits("weights=bf16,grad=bf16,window=2").is_ok());
    let why = admits("weights=bf16,grad=bf16").unwrap_err();
    assert!(why.contains("window=N"), "{why}");
    let why = admits("weights=bf16,window=2").unwrap_err();
    assert!(why.contains("grad=bf16"), "the build's bf16 gradient lanes: {why}");
    assert!(admits("moments=host").unwrap_err().contains("weights=bf16"));
}

#[test]
fn the_fit_loop_refuses_a_sharded_run_it_cannot_shard() {
    assert!(fit_error(Sharding::Gradients, false, "").contains("has none"), "no group");
    assert!(fit_error(Sharding::Gradients, true, "").contains("not where a sharded optimizer reaches them"), "no device parameters");
    let why = fit_error(Sharding::Optimizer, true, "state=bf16");
    assert!(why.contains("f32 AdamW"), "another state precision: {why}");
    let replicated = fit_error(Sharding::Replicated, true, "");
    assert!(!replicated.contains("sharded training"), "a replicated run is not refused for sharding: {replicated}");
}

#[test]
fn a_prefetched_unit_reaches_the_parameters_when_placed_or_before_its_staging_is_reused() {
    // Parameter sharding gathers the next layer while one computes. Writing the gathered values into the layer's
    // buffers at once would order the computing layer's work (which shares the stream) after that gather; they reach
    // the buffers when the layer is placed, or before anything else reuses the gather's staging.
    let world = 2;
    let coll = HostCollective::new(world);
    std::thread::scope(|s| {
        for rank in 0..world {
            let coll = Arc::clone(&coll);
            s.spawn(move || {
                let gpu = gpu_core::testgpu::dev(zero::KERNELS);
                let p = params(&gpu, 1, CollectiveDtype::F32);
                let initial: Vec<Vec<f32>> = SIZES.iter().enumerate().map(|(i, &n)| values(31 + i as u32, n)).collect();
                let read_initial = |t: usize, offset: u64, out: &mut [f32]| -> Result<(), String> {
                    out.copy_from_slice(&initial[t][offset as usize..offset as usize + out.len()]);
                    Ok(())
                };
                let mut read = read_initial;
                // One unit per tensor.
                let groups: Vec<u32> = (0..SIZES.len() as u32).collect();
                let values_of = |p: &DeviceParams| -> Vec<Vec<f32>> { p.tensors.iter().map(|t| zero::read_elems(&gpu, &t.value, 0, t.numel, CollectiveDtype::F32)).collect() };
                let blank = |p: &DeviceParams| p.tensors.iter().for_each(|t| zero::write_elems(&gpu, &t.value, 0, &vec![0.0; t.numel as usize], CollectiveDtype::F32));
                let handles = DeviceParams { gpu: gpu.share(), tensors: p.tensors.clone() };
                let opt = ZeroOptimizer::new_parameter_shards(p, &groups, &*coll, rank, &mut read).unwrap();
                blank(&handles);
                let (a, b) = (opt.unit_of(0), opt.unit_of(2));
                assert_eq!(a % 2, b % 2, "units 0 and 2 share a staging buffer");
                opt.prefetch_unit(&*coll, a).unwrap();
                opt.settle(&*coll).unwrap();
                assert!(values_of(&handles)[0].iter().all(|&x| x == 0.0), "rank {rank}: a prefetched unit is not in the parameters before it is placed");
                opt.place_unit(a);
                assert_eq!(values_of(&handles)[0], initial[0], "rank {rank}: placing the unit writes its gathered values");
                blank(&handles);
                // The next gather into the same staging places the waiting unit first.
                opt.prefetch_unit(&*coll, a).unwrap();
                opt.gather_unit(&*coll, b).unwrap();
                opt.settle(&*coll).unwrap();
                let got = values_of(&handles);
                assert_eq!((&got[0], &got[2]), (&initial[0], &initial[2]), "rank {rank}: reusing the staging placed the waiting unit");
            });
        }
    });
}
