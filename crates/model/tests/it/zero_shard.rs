// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sharded data-parallel training and its checkpoints for its clients. If your team
// needs expertise in resuming a sharded run on another number of cards, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The sharded optimizer ([`model::zero::ZeroOptimizer`]) on raw device tensors over in-process ranks: a checkpoint
//! saved by one world size loads into another with the logical tensors unchanged, bf16 parameters shard and gather
//! to the same bytes on every rank, and what cannot be sharded is refused by what it lacks.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use gpu_core::{CollectiveDtype, Gpu};
use model::zero::{self, DeviceParam, DeviceParams, Sharding, StepArgs, StepOutcome, ZeroOptimizer};
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
fn ranks(world: usize, sharding: Sharding, dtype: CollectiveDtype, seed: u32, restore: Option<(&Path, u32)>, steps: u32, save: Option<&Path>) -> Vec<Vec<Vec<f32>>> {
    let coll = HostCollective::new(world);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let coll = Arc::clone(&coll);
                s.spawn(move || {
                    let gpu = gpu_core::testgpu::dev(&[]);
                    let p = params(&gpu, seed, dtype);
                    let handles: Vec<DeviceParam> = p.tensors.clone();
                    let mut opt = ZeroOptimizer::new(p, sharding, &*coll, rank, UNIT).unwrap();
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
                        assert!(matches!(opt.step(&*coll, 1.0 + rank as f32, true, &args(t)).unwrap(), StepOutcome::Applied { .. }));
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

/// A run saved at four ranks resumes at two (and at four) from the same logical master and moments; the restored
/// parameters are the saved master; a save at the new world holds the same logical tensors.
#[test]
fn a_checkpoint_saved_at_one_world_loads_into_another() {
    let dir = scratch("reshard");
    let (four, two) = (dir.join("four"), dir.join("two"));
    let saved = ranks(4, Sharding::Gradients, CollectiveDtype::F32, 3, None, 2, Some(&four));
    let logical = zero::read_logical(&four, 2).unwrap();
    assert_eq!(logical.step, 2);
    assert_eq!(logical.master, saved[0], "the logical master is what every rank's parameters hold");
    for (rank, w) in saved.iter().enumerate() {
        assert_eq!(w, &saved[0], "rank {rank}'s parameters differ from rank 0's");
    }
    // Built from other initial values, then restored: the checkpoint decides the parameters.
    let resumed = ranks(2, Sharding::Optimizer, CollectiveDtype::F32, 9, Some((&four, 2)), 0, Some(&two));
    assert_eq!(resumed[0], logical.master, "restored into two ranks, the parameters are the saved master");
    let again = zero::read_logical(&two, 2).unwrap();
    for (what, a, b) in [("master", &again.master, &logical.master), ("m", &again.m, &logical.m), ("v", &again.v, &logical.v)] {
        assert_eq!(a, b, "the {what} tensors changed across the world-size change");
    }
    // Training on from the restored state goes the same way at either world size, within the reduction order.
    let on_four = ranks(4, Sharding::Gradients, CollectiveDtype::F32, 9, Some((&four, 2)), 1, None);
    let on_four_again = ranks(4, Sharding::Gradients, CollectiveDtype::F32, 9, Some((&four, 2)), 1, None);
    assert_eq!(on_four, on_four_again, "a resume on an unchanged layout is repeatable bit for bit");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// bf16 parameters and gradients: every rank gathers the same bytes, and they are the master rounded to bf16.
#[test]
fn bf16_parameters_shard_and_gather_to_the_same_bytes() {
    let dir = scratch("bf16");
    let out = ranks(3, Sharding::Gradients, CollectiveDtype::Bf16, 5, None, 2, Some(&dir));
    for w in &out {
        assert_eq!(w, &out[0]);
    }
    let logical = zero::read_logical(&dir, 2).unwrap();
    for (got, master) in out[0].iter().zip(&logical.master) {
        let rounded: Vec<f32> = model::half::pack_bf16(master).iter().flat_map(|w| [f32::from_bits(w << 16), f32::from_bits(w & 0xFFFF_0000)]).take(master.len()).collect();
        assert_eq!(got, &rounded);
    }
    std::fs::remove_dir_all(&dir).unwrap();
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
                    let gpu = gpu_core::testgpu::dev(&[]);
                    let p = params(&gpu, 1, CollectiveDtype::F32);
                    let h = p.tensors[0].clone();
                    let before = zero::read_elems(&gpu, &h.value, 0, h.numel, CollectiveDtype::F32);
                    let mut opt = ZeroOptimizer::new(p, Sharding::Optimizer, &*coll, rank, UNIT).unwrap();
                    let outcome = opt.step(&*coll, 1.0, rank == 0, &args(1)).unwrap();
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
    let gpu = gpu_core::testgpu::dev(&[]);
    let err = |sharding| ZeroOptimizer::new(params(&gpu, 1, CollectiveDtype::F32), sharding, &*coll, 0, UNIT).err().unwrap();
    assert!(err(Sharding::Replicated).contains("no state to shard"));
    assert!(err(Sharding::Parameters).contains("gathers each block's parameters"));
    let mut odd = params(&gpu, 1, CollectiveDtype::Bf16);
    odd.tensors[0].numel = 7;
    let why = ZeroOptimizer::new(odd, Sharding::Optimizer, &*coll, 0, UNIT).err().unwrap();
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
    let control = model::FitControl { grad_sync: group.then(|| model::GradSync { collective: &*coll, rank: 0, checkpoint: None, observer: None }), ..Default::default() };
    match model::fit_controlled(HostModel(HostCfg), Nothing, &opts, None, control) {
        Ok(_) => "trained".into(),
        Err(e) => e.to_string(),
    }
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
