// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full-parameter fine-tuning of models larger than one accelerator, sharded over the
// cards of a cluster, for its clients. If your team needs expertise in ZeRO-3-style parameter sharding, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec of parameter sharding (ZeRO-3) for a layer-streamed resident full fine-tune: a stack of layers trained over
//! in-process ranks, every layer gathered from the ranks' slices where the forward and the backward reach it,
//! reduce-scattered and updated by its owners once its gradient is complete, follows the single-rank layer-streamed
//! run (every rank given the same batch, so the group's mean gradient is that batch's) within the rounding of the
//! sharded update, with no clip, the two-pass global clip and the single-pass running clip, with a window of one or
//! two layers (the second prefetches), and every rank sees the same losses bit for bit. What
//! the sharded build does not run is refused by name.

use std::collections::HashMap;
use std::sync::Arc;

use data::rng::Lcg;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::{DeviceBuffer, Gpu};
use model::memplan::{Authorized, Capacity};
use model::ops::{training_kernel_list, GradDtype, Ops, Weight};
use model::resident_ft::{ClipPass, LayerWindow, ResidentBudget, ResidentFtConfig, ResidentFullFt, ResidentSpec, ResidentValue, StreamPass};
use model::zero::ShardGroup;
use model::{Collective, HostCollective};
use optim::rule::Rule;
use optim::Adam;
use paramstore::precision::{MasterPolicy, StateFormat};

const M: usize = 16;
const D: usize = 32;
const LAYERS: usize = 4;
const LR: f32 = 1e-2;
const ADAM: Adam = Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 };
/// The task's data: fixed, so every run compares the same trajectories.
const TASK_SEED: u64 = 0x2026_1122;
/// How far a sharded run's loss may be from the single-rank run's, relative to it: the sharded update's rounding.
const FOLLOW: f64 = 1e-3;

fn ops() -> Ops {
    let mut list = training_kernel_list().to_vec();
    model::ops::extend_kernel_list(&mut list, model::zero::KERNELS);
    let mut ops = Ops::new(Gpu::new(Box::leak(list.into_boxed_slice()))).expect("Ops::new");
    ops.set_gemm_precision(GemmPrecision::Fp32Reference);
    ops
}

fn specs() -> Vec<ResidentSpec> {
    (0..LAYERS).flat_map(|l| [("a", l), ("b", l)].map(|(leaf, l)| ResidentSpec::linear(format!("layer{l}.{leaf}"), D as u32, D as u32, l as u32))).collect()
}

struct Task {
    x: Vec<f32>,
    target: Vec<f32>,
    init: HashMap<String, Vec<f32>>,
}

fn task() -> Task {
    let mut rng = Lcg::new(TASK_SEED);
    let init = specs().iter().map(|s| (s.name.clone(), rng.vec_scaled(D * D, 0.15))).collect();
    Task { x: rng.vec_scaled(M * D, 1.0), target: rng.vec_scaled(M * D, 1.0), init }
}

fn config(shard: Option<ShardGroup>, window: u32) -> ResidentFtConfig {
    ResidentFtConfig {
        value: ResidentValue::Bf16,
        master: MasterPolicy::F32,
        state: StateFormat::F32,
        rule: Rule::AdamW { adam: ADAM, wd: 0.01 },
        grad: GradDtype::Bf16,
        stochastic_state: false,
        seed: None,
        tile_elems: 128,
        state_dir: None,
        defer_publication: false,
        window: Some(LayerWindow { layers: window }),
        clip_pass: ClipPass::TwoPass,
        shard,
    }
}

fn roomy() -> ResidentBudget<'static> {
    ResidentBudget { capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 }, others: &[], authorized: Authorized::none() }
}

fn upload(ops: &Ops, host: &[f32]) -> DeviceBuffer {
    let b = ops.gpu().storage(host.len() as u64);
    ops.gpu().write_f32(&b, host);
    b
}

fn forward(ops: &Ops, w: &Weight, x: &DeviceBuffer, y: &DeviceBuffer) {
    let mut s = Vec::new();
    let act = ops.act_f32(x, 0, M as u32, w.k());
    ops.matmul_rows(&mut s, w, (0, w.n()), &act, y, 0);
    ops.gpu().submit(&[], &s);
}

/// One layer-streamed step (the protocol a model drives): load each layer in the forward, then per backward pass
/// load, backward and finish each layer in reverse. Returns the loss before the update.
fn step(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, clip: Option<f32>, running: bool) -> f64 {
    let g = ops.gpu();
    let mut x = upload(ops, &task.x);
    let mut inputs = Vec::new();
    for l in 0..LAYERS {
        ft.load_layer(ops, l).unwrap();
        let (h, y) = (g.storage((M * D) as u64), g.storage((M * D) as u64));
        forward(ops, ft.weight(2 * l), &x, &h);
        forward(ops, ft.weight(2 * l + 1), &h, &y);
        inputs.push(x);
        inputs.push(h);
        x = y;
    }
    let out = g.read(&x, M * D);
    let loss = out.iter().zip(&task.target).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / (2.0 * M as f64);
    let backward = |ft: &mut ResidentFullFt, pass: StreamPass| {
        ft.zero_grads(ops);
        ft.begin_pass(pass);
        let d: Vec<f32> = out.iter().zip(&task.target).map(|(a, b)| (a - b) / M as f32).collect();
        let mut dy = upload(ops, &d);
        for l in (0..LAYERS).rev() {
            ft.begin_layer_backward(ops, l).unwrap();
            let (dh, dx) = (g.storage((M * D) as u64), g.storage((M * D) as u64));
            let mut s = Vec::new();
            ops.matmul_dx(&mut s, ft.weight(2 * l + 1), &dy, M as u32, &dh, false);
            ft.weight_grad(ops, &mut s, 2 * l + 1, &inputs[2 * l + 1], &dy, M as u32);
            ops.matmul_dx(&mut s, ft.weight(2 * l), &dh, M as u32, &dx, false);
            ft.weight_grad(ops, &mut s, 2 * l, &inputs[2 * l], &dh, M as u32);
            g.submit(&[], &s);
            g.flush();
            ft.finish_layer(ops, l).unwrap();
            dy = dx;
        }
        ft.end_pass().unwrap()
    };
    match clip {
        Some(_) if running => {
            backward(ft, StreamPass::Running { lr: LR, clip, extra_scale: 1.0 });
        }
        Some(_) => {
            let measured = backward(ft, StreamPass::Measure);
            backward(ft, StreamPass::Update { lr: LR, grad_multiplier: optim::grad_multiplier(measured.sum_sq, clip, 1.0) });
        }
        None => {
            backward(ft, StreamPass::Update { lr: LR, grad_multiplier: 1.0 });
        }
    }
    loss
}

/// Every rank of `world` trains the task for `steps` steps; each rank's losses.
fn sharded(world: usize, window: u32, task: &Task, clip: Option<f32>, running: bool, steps: usize) -> Vec<Vec<f64>> {
    let coll = HostCollective::new(world);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..world)
            .map(|rank| {
                let coll: Arc<dyn Collective> = coll.clone();
                s.spawn(move || {
                    let ops = ops();
                    let group = ShardGroup { collective: coll, rank };
                    let mut ft = ResidentFullFt::new(&ops, config(Some(group), window), specs(), &task.init, &roomy()).unwrap();
                    ft.set_shard_weight(1.0 / world as f32).unwrap();
                    (0..steps).map(|_| step(&ops, &mut ft, task, clip, running)).collect::<Vec<f64>>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

#[test]
fn a_parameter_sharded_step_follows_the_layer_streamed_step() {
    let task = task();
    // No clip, the exact two-pass clip, and the single-pass running clip (the group's norm of each layer summed as the
    // layers finish).
    for (clip, running) in [(None, false), (Some(1e-2f32), false), (Some(1e-2f32), true)] {
        let ops = ops();
        let mut single = ResidentFullFt::new(&ops, config(None, 1), specs(), &task.init, &roomy()).unwrap();
        let want: Vec<f64> = (0..5).map(|_| step(&ops, &mut single, &task, clip, running)).collect();
        // A window of two layers prefetches the next layer while one computes.
        for (world, window) in [(2, 1), (3, 2)] {
            let got = sharded(world, window, &task, clip, running, 5);
            for (rank, losses) in got.iter().enumerate() {
                assert_eq!(losses.iter().map(|l| l.to_bits()).collect::<Vec<_>>(), got[0].iter().map(|l| l.to_bits()).collect::<Vec<_>>(), "clip {clip:?} world {world}: rank {rank}'s losses differ from rank 0's");
            }
            for (step, (a, b)) in want.iter().zip(&got[0]).enumerate() {
                assert!((a - b).abs() <= FOLLOW * a.abs().max(1e-3), "clip {clip:?} world {world} step {step}: loss {b} sharded, {a} single");
            }
        }
        // Following is only evidence when the reference moves: its updates change the loss by far more than the
        // tolerance (a step that updated nothing would be followed trivially).
        let moved = want.iter().map(|l| (l - want[0]).abs()).fold(0.0, f64::max);
        assert!(moved > 10.0 * FOLLOW * want[0].abs(), "clip {clip:?}: the single-rank run's loss moved only {moved:e} over its steps: {want:?}");
    }
}

#[test]
fn what_the_sharded_build_does_not_run_is_refused() {
    let task = task();
    let ops = ops();
    let group = || Some(ShardGroup { collective: HostCollective::new(1), rank: 0 });
    let no_window = ResidentFtConfig { window: None, ..config(group(), 1) };
    assert!(ResidentFullFt::new(&ops, no_window, specs(), &task.init, &roomy()).err().unwrap().contains("window"));
    let f32_grads = ResidentFtConfig { grad: GradDtype::F32, ..config(group(), 1) };
    assert!(ResidentFullFt::new(&ops, f32_grads, specs(), &task.init, &roomy()).err().unwrap().contains("bf16 gradients"));
    let mut ft = ResidentFullFt::new(&ops, config(group(), 1), specs(), &task.init, &roomy()).unwrap();
    assert!(ft.read_master(&ops, 0).unwrap_err().to_string().contains("this rank's slices"));
}
