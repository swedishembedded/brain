// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models that do not fit on one
// accelerator for its clients. If your team needs expertise in training every weight of a model larger than the
// device memory, then you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of layer-streamed full fine-tuning (`model::resident_ft` with a [`LayerWindow`]): a stack of layers whose
//! weights do not all live on the device. A bounded window of layers is resident; every layer is loaded from its
//! host copy when the forward or the backward reaches it, and the update of a layer runs between the backward
//! segments, as soon as its gradient is complete.
//!
//! * the layer-streamed step gives the same bits as the resident step (clip off, and with the global clip in its
//!   two-pass form, which measures the norm on the device first);
//! * the device holds at most the window's layers, and the planner prices exactly that;
//! * the single-pass running clip never clips harder than the global clip and equals it when nothing clips;
//! * a restored run continues exactly.

use data::rng::Lcg;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::{DeviceBuffer, Gpu};
use model::memplan::{Authorized, Capacity};
use model::ops::{training_kernel_list, GradDtype, Ops, Weight};
use model::resident_ft::{ClipPass, LayerWindow, ResidentBudget, ResidentFtConfig, ResidentFullFt, ResidentSpec, ResidentValue, StreamPass};
use optim::checkpoint::CheckpointDir;
use optim::rule::Rule;
use optim::Adam;
use paramstore::precision::{MasterPolicy, StateFormat};
use std::collections::HashMap;

const M: usize = 16;
const D: usize = 32;
const LAYERS: usize = 4;
const LR: f32 = 1e-2;
const ADAM: Adam = Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 };

fn ops() -> Ops {
    let mut ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    ops.set_gemm_precision(GemmPrecision::Fp32Reference);
    ops
}

/// Tensor `2l` and `2l + 1` are layer `l`'s two square linears: the same shape twice in a layer, so a slot pool
/// must keep them apart.
fn specs() -> Vec<ResidentSpec> {
    (0..LAYERS).flat_map(|l| [("a", l), ("b", l)].map(|(leaf, l)| ResidentSpec::linear(format!("layer{l}.{leaf}"), D as u32, D as u32, l as u32))).collect()
}

struct Task {
    x: Vec<f32>,
    target: Vec<f32>,
    init: HashMap<String, Vec<f32>>,
}

fn task() -> Task {
    let mut rng = Lcg::random();
    let init = specs().iter().map(|s| (s.name.clone(), rng.vec_scaled(D * D, 0.15))).collect();
    Task { x: rng.vec_scaled(M * D, 1.0), target: rng.vec_scaled(M * D, 1.0), init }
}

fn config(value: ResidentValue, window: Option<u32>, clip_pass: ClipPass) -> ResidentFtConfig {
    ResidentFtConfig {
        value,
        // An f32 value is its own master.
        master: if value == ResidentValue::F32 { MasterPolicy::None } else { MasterPolicy::F32 },
        state: StateFormat::F32,
        rule: Rule::AdamW { adam: ADAM, wd: 0.0 },
        grad: GradDtype::F32,
        stochastic_state: false,
        seed: None,
        tile_elems: 128,
        state_dir: None,
        defer_publication: false,
        window: window.map(|layers| LayerWindow { layers }),
        clip_pass,
    }
}

fn roomy() -> ResidentBudget<'static> {
    ResidentBudget { capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 }, others: &[], authorized: Authorized::none() }
}

fn build(ops: &Ops, cfg: ResidentFtConfig, task: &Task) -> ResidentFullFt {
    ResidentFullFt::new(ops, cfg, specs(), &task.init, &roomy()).unwrap()
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

/// The forward of the stack. `load` runs before each layer's products (the layer-streamed build loads the layer
/// there). Returns the activations every backward segment needs: the input of each linear.
struct Acts {
    inputs: Vec<DeviceBuffer>,
    out: Vec<f32>,
}

fn run_forward(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, mut load: impl FnMut(&mut ResidentFullFt, usize)) -> Acts {
    let g = ops.gpu();
    let mut x = upload(ops, &task.x);
    let mut inputs = Vec::new();
    for l in 0..LAYERS {
        load(ft, l);
        let h = g.storage((M * D) as u64);
        let y = g.storage((M * D) as u64);
        forward(ops, ft.weight(2 * l), &x, &h);
        forward(ops, ft.weight(2 * l + 1), &h, &y);
        inputs.push(x);
        inputs.push(h);
        x = y;
    }
    let out = g.read(&x, M * D);
    Acts { inputs, out }
}

fn loss(out: &[f32], target: &[f32]) -> f64 {
    out.iter().zip(target).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / (2.0 * M as f64)
}

/// Backward of layer `l` given the gradient of its output: records the two weight gradients, returns the
/// gradient of its input.
fn layer_backward(ops: &Ops, ft: &ResidentFullFt, acts: &Acts, l: usize, dy: &DeviceBuffer) -> DeviceBuffer {
    let g = ops.gpu();
    let dh = g.storage((M * D) as u64);
    let dx = g.storage((M * D) as u64);
    let mut s = Vec::new();
    ops.matmul_dx(&mut s, ft.weight(2 * l + 1), dy, M as u32, &dh, false);
    ft.weight_grad(ops, &mut s, 2 * l + 1, &acts.inputs[2 * l + 1], dy, M as u32);
    ops.matmul_dx(&mut s, ft.weight(2 * l), &dh, M as u32, &dx, false);
    ft.weight_grad(ops, &mut s, 2 * l, &acts.inputs[2 * l], &dh, M as u32);
    g.submit(&[], &s);
    dx
}

fn dout(ops: &Ops, acts: &Acts, task: &Task) -> DeviceBuffer {
    let d: Vec<f32> = acts.out.iter().zip(&task.target).map(|(a, b)| (a - b) / M as f32).collect();
    upload(ops, &d)
}

/// One step with every layer resident: all gradients, the host-visible global clip, one update.
fn resident_step(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, clip: Option<f32>) -> f64 {
    let acts = run_forward(ops, ft, task, |_, _| {});
    let mut dy = dout(ops, &acts, task);
    for l in (0..LAYERS).rev() {
        dy = layer_backward(ops, ft, &acts, l, &dy);
    }
    let sum_sq = if clip.is_some() { ft.grad_sum_sq(ops).unwrap() } else { 0.0 };
    ft.step_scaled(ops, LR, optim::grad_multiplier(sum_sq, clip, 1.0)).unwrap();
    loss(&acts.out, &task.target)
}

/// One layer-streamed step: layers are loaded as the forward reaches them; the backward runs once per pass over the
/// layers in reverse, each layer loaded again, its gradient measured or applied as soon as it is complete.
fn streamed_step(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, clip: Option<f32>, pass: ClipPass) -> (f64, f32) {
    let acts = run_forward(ops, ft, task, |ft, l| ft.load_layer(ops, l).unwrap());
    let backward = |ft: &mut ResidentFullFt, p: StreamPass| {
        ft.begin_pass(p);
        let mut dy = dout(ops, &acts, task);
        for l in (0..LAYERS).rev() {
            ft.begin_layer_backward(ops, l).unwrap();
            dy = layer_backward(ops, ft, &acts, l, &dy);
            ft.finish_layer(ops, l).unwrap();
        }
        ft.end_pass().unwrap()
    };
    let report = match (clip, pass) {
        (Some(_), ClipPass::TwoPass) => {
            let measured = backward(ft, StreamPass::Measure);
            backward(ft, StreamPass::Update { lr: LR, grad_multiplier: optim::grad_multiplier(measured.sum_sq, clip, 1.0) })
        }
        (None, _) => backward(ft, StreamPass::Update { lr: LR, grad_multiplier: 1.0 }),
        (Some(_), ClipPass::Running) => backward(ft, StreamPass::Running { lr: LR, clip, extra_scale: 1.0 }),
    };
    (loss(&acts.out, &task.target), report.min_multiplier)
}

fn masters(ops: &Ops, ft: &mut ResidentFullFt) -> Vec<Vec<u32>> {
    (0..2 * LAYERS).map(|i| ft.read_master(ops, i).unwrap().into_iter().map(f32::to_bits).collect()).collect()
}

const VALUES: [ResidentValue; 5] = [ResidentValue::F32, ResidentValue::Bf16, ResidentValue::Fp8E4M3 { block: 128 }, ResidentValue::Int8, ResidentValue::Nf4];

/// Spec: layer streaming changes where the weights live, not the arithmetic: master weights after five steps are
/// bit-identical to the resident step's, for every storage format, windows of one and two layers, with no clip and
/// with a clip that is active.
#[test]
fn the_layer_streamed_step_gives_the_resident_steps_bits() {
    let ops = ops();
    let task = task();
    for value in VALUES {
        for clip in [None, Some(1e-3f32)] {
            for window in [1u32, 2] {
                let mut resident = build(&ops, config(value, None, ClipPass::TwoPass), &task);
                let mut streamed = build(&ops, config(value, Some(window), ClipPass::TwoPass), &task);
                for step in 0..5 {
                    let want = resident_step(&ops, &mut resident, &task, clip);
                    let (got, _) = streamed_step(&ops, &mut streamed, &task, clip, ClipPass::TwoPass);
                    assert_eq!(want.to_bits(), got.to_bits(), "{value:?} clip {clip:?} window {window} step {step}: loss");
                }
                assert_eq!(masters(&ops, &mut resident), masters(&ops, &mut streamed), "{value:?} clip {clip:?} window {window}: masters");
                assert_eq!(resident.step_count(), streamed.step_count());
            }
        }
    }
}

/// Spec: deferred publication of a layer's update (all-or-nothing per layer) changes no bits.
#[test]
fn deferred_publication_per_layer_changes_no_bits() {
    let ops = ops();
    let task = task();
    let mut direct = build(&ops, config(ResidentValue::Bf16, Some(1), ClipPass::TwoPass), &task);
    let mut deferred = build(&ops, ResidentFtConfig { defer_publication: true, ..config(ResidentValue::Bf16, Some(1), ClipPass::TwoPass) }, &task);
    for _ in 0..3 {
        streamed_step(&ops, &mut direct, &task, Some(1e-3), ClipPass::TwoPass);
        streamed_step(&ops, &mut deferred, &task, Some(1e-3), ClipPass::TwoPass);
    }
    assert_eq!(masters(&ops, &mut direct), masters(&ops, &mut deferred));
}

/// Spec: the device holds the window's layers and no more; the planner prices exactly that, so a budget that
/// holds a window but not the model is accepted by the streamed build and refused for the resident one.
#[test]
fn the_device_holds_at_most_the_window_and_the_planner_prices_it() {
    let ops = ops();
    let task = task();
    let layer_bytes = |value: ResidentValue| 2 * value.device_bytes(D as u64, D as u64);
    for value in VALUES {
        for window in [1u32, 2] {
            let ft = build(&ops, config(value, Some(window), ClipPass::TwoPass), &task);
            assert_eq!(ft.device_weight_bytes(), u64::from(window) * layer_bytes(value), "{value:?} window {window}");
        }
        let full = build(&ops, config(value, None, ClipPass::TwoPass), &task);
        assert_eq!(full.device_weight_bytes(), LAYERS as u64 * layer_bytes(value), "{value:?}: resident holds every layer");
    }
    // Weights of one layer plus one layer of f32 gradients fit; the whole model does not.
    let value = ResidentValue::Bf16;
    let one_layer = layer_bytes(value) + 2 * (D * D * 4) as u64;
    let tight = ResidentBudget { capacity: Capacity { device_bytes: one_layer, host_bytes: 1 << 30 }, others: &[], authorized: Authorized::none() };
    ResidentFullFt::new(&ops, config(value, Some(1), ClipPass::TwoPass), specs(), &task.init, &tight).expect("a one-layer window fits one layer");
    let err = ResidentFullFt::new(&ops, config(value, None, ClipPass::TwoPass), specs(), &task.init, &tight).err().expect("resident refused");
    assert!(err.contains("does not fit"), "{err}");
}

/// Spec: the single-pass running clip equals the global clip when nothing clips, and never clips harder than the
/// global clip when something does (the running norm only grows toward the global one).
#[test]
fn the_running_clip_never_clips_harder_than_the_global_clip() {
    let ops = ops();
    let task = task();
    let mut unclipped_ref = build(&ops, config(ResidentValue::Bf16, Some(1), ClipPass::TwoPass), &task);
    let mut running = build(&ops, config(ResidentValue::Bf16, Some(1), ClipPass::Running), &task);
    for _ in 0..3 {
        streamed_step(&ops, &mut unclipped_ref, &task, None, ClipPass::TwoPass);
        let (_, mult) = streamed_step(&ops, &mut running, &task, Some(1e9), ClipPass::Running);
        assert_eq!(mult, 1.0, "nothing clips at a huge bound");
    }
    assert_eq!(masters(&ops, &mut unclipped_ref), masters(&ops, &mut running));

    let mut global = build(&ops, config(ResidentValue::Bf16, Some(1), ClipPass::TwoPass), &task);
    let mut running = build(&ops, config(ResidentValue::Bf16, Some(1), ClipPass::Running), &task);
    let (_, global_mult) = streamed_step(&ops, &mut global, &task, Some(1e-3), ClipPass::TwoPass);
    let (_, running_min) = streamed_step(&ops, &mut running, &task, Some(1e-3), ClipPass::Running);
    assert!(running_min >= global_mult, "running multiplier {running_min} must not be below the global {global_mult}");
    assert!(global_mult < 1.0, "the test clip must be active");
}

/// Spec: a layer-streamed run restored from a committed checkpoint continues exactly: the host values, masters
/// and moments match the uninterrupted run bit for bit, and the stale device window is reloaded.
#[test]
fn a_restored_layer_streamed_run_continues_exactly() {
    let ops = ops();
    let task = task();
    let dir = std::env::temp_dir().join(format!("brain-layer-ft-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for value in VALUES {
        let cfg = || ResidentFtConfig { state: StateFormat::Int8Block { block: 64 }, ..config(value, Some(1), ClipPass::TwoPass) };
        let ckpt = CheckpointDir::new(dir.join(format!("{value:?}"))).unwrap();
        let mut a = build(&ops, cfg(), &task);
        for _ in 0..3 {
            streamed_step(&ops, &mut a, &task, Some(1e-3), ClipPass::TwoPass);
        }
        a.commit(&ops, &ckpt, &a.run_state(0, vec![])).unwrap();
        for _ in 0..3 {
            streamed_step(&ops, &mut a, &task, Some(1e-3), ClipPass::TwoPass);
        }
        let mut b = build(&ops, cfg(), &task);
        // A different history first, so the window holds stale layers when the checkpoint is restored.
        streamed_step(&ops, &mut b, &task, Some(1e-3), ClipPass::TwoPass);
        let restored = b.restore(&ops, &ckpt).unwrap().expect("a checkpoint exists");
        assert_eq!((restored.step, b.step_count()), (3, 3));
        for _ in 0..3 {
            streamed_step(&ops, &mut b, &task, Some(1e-3), ClipPass::TwoPass);
        }
        assert_eq!(masters(&ops, &mut a), masters(&ops, &mut b), "{value:?}");
        assert_eq!(a.step_count(), b.step_count());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

// ---- a table beside the streamed layers ------------------------------------------------------------------------

/// The table every head test adds after the layers: a `[D, D]` output projection held as a [`ResidentSpec::table`].
const HEAD: usize = 2 * LAYERS;

fn specs_with_head() -> Vec<ResidentSpec> {
    let mut specs = specs();
    specs.push(ResidentSpec::table("head", D as u32, D as u32));
    specs
}

fn task_with_head() -> Task {
    let mut task = task();
    task.init.insert("head".into(), Lcg::random().vec_scaled(D * D, 0.15));
    task
}

fn build_with_head(ops: &Ops, cfg: ResidentFtConfig, task: &Task) -> ResidentFullFt {
    ResidentFullFt::new(ops, cfg, specs_with_head(), &task.init, &roomy()).unwrap()
}

/// The stack, then the head: the head's input joins the activations and `out` is the head's output.
fn head_forward(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, load: impl FnMut(&mut ResidentFullFt, usize)) -> Acts {
    let mut acts = run_forward(ops, ft, task, load);
    let x = upload(ops, &acts.out);
    let y = ops.gpu().storage((M * D) as u64);
    forward(ops, ft.weight(HEAD), &x, &y);
    acts.out = ops.gpu().read(&y, M * D);
    acts.inputs.push(x);
    acts
}

fn head_backward(ops: &Ops, ft: &ResidentFullFt, acts: &Acts, dy: &DeviceBuffer) -> DeviceBuffer {
    let dx = ops.gpu().storage((M * D) as u64);
    let mut s = Vec::new();
    ops.matmul_dx(&mut s, ft.weight(HEAD), dy, M as u32, &dx, false);
    ft.weight_grad(ops, &mut s, HEAD, &acts.inputs[HEAD], dy, M as u32);
    ops.gpu().submit(&[], &s);
    dx
}

fn resident_head_step(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, clip: Option<f32>) -> f64 {
    let acts = head_forward(ops, ft, task, |_, _| {});
    let mut dy = head_backward(ops, ft, &acts, &dout(ops, &acts, task));
    for l in (0..LAYERS).rev() {
        dy = layer_backward(ops, ft, &acts, l, &dy);
    }
    let sum_sq = if clip.is_some() { ft.grad_sum_sq(ops).unwrap() } else { 0.0 };
    ft.step_scaled(ops, LR, optim::grad_multiplier(sum_sq, clip, 1.0)).unwrap();
    loss(&acts.out, &task.target)
}

/// A layer-streamed step with a table: each pass covers the layers; the table's gradient joins the global norm and the
/// table is updated once the backward is complete, by the step's final multiplier.
fn streamed_head_step(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, clip: Option<f32>) -> f64 {
    let acts = head_forward(ops, ft, task, |ft, l| ft.load_layer(ops, l).unwrap());
    let backward = |ft: &mut ResidentFullFt, p: StreamPass| {
        ft.zero_grads(ops);
        ft.begin_pass(p);
        let mut dy = head_backward(ops, ft, &acts, &dout(ops, &acts, task));
        for l in (0..LAYERS).rev() {
            ft.begin_layer_backward(ops, l).unwrap();
            dy = layer_backward(ops, ft, &acts, l, &dy);
            ft.finish_layer(ops, l).unwrap();
        }
        ft.end_pass().unwrap()
    };
    let multiplier = match clip {
        None => {
            backward(ft, StreamPass::Update { lr: LR, grad_multiplier: 1.0 });
            1.0
        }
        Some(_) => {
            let measured = backward(ft, StreamPass::Measure);
            let multiplier = optim::grad_multiplier(measured.sum_sq + ft.table_grad_sum_sq(ops), clip, 1.0);
            backward(ft, StreamPass::Update { lr: LR, grad_multiplier: multiplier });
            multiplier
        }
    };
    ft.update_tables(ops, LR, multiplier).unwrap();
    loss(&acts.out, &task.target)
}

fn all_masters(ops: &Ops, ft: &mut ResidentFullFt) -> Vec<Vec<u32>> {
    (0..=HEAD).map(|i| ft.read_master(ops, i).unwrap().into_iter().map(f32::to_bits).collect()).collect()
}

/// Spec: a table lives in the resident state in bf16 beside narrower linears and in f32 beside f32 ones, stays on the device while the
/// layers stream (the window's bytes plus the table's), trains, and a layer-streamed step with it gives the resident
/// step's bits, with no clip and with an active clip that spans the table.
#[test]
fn a_table_trains_beside_streamed_layers_with_the_resident_steps_bits() {
    let ops = ops();
    let task = task_with_head();
    for value in VALUES {
        let table = if value == ResidentValue::F32 { ResidentValue::F32 } else { ResidentValue::Bf16 };
        let table_bytes = table.device_bytes(D as u64, D as u64);
        for clip in [None, Some(1e-3f32)] {
            let mut resident = build_with_head(&ops, config(value, None, ClipPass::TwoPass), &task);
            let mut streamed = build_with_head(&ops, config(value, Some(1), ClipPass::TwoPass), &task);
            for ft in [&resident, &streamed] {
                let held = matches!((table, ft.weight(HEAD)), (ResidentValue::F32, Weight::F32 { .. }) | (ResidentValue::Bf16, Weight::BF16 { .. }));
                assert!(held, "{value:?}: the table is held in {table:?}");
            }
            assert_eq!(streamed.device_weight_bytes(), 2 * value.device_bytes(D as u64, D as u64) + table_bytes, "{value:?}: one layer and the table");
            for step in 0..4 {
                let want = resident_head_step(&ops, &mut resident, &task, clip);
                let got = streamed_head_step(&ops, &mut streamed, &task, clip);
                assert_eq!(want.to_bits(), got.to_bits(), "{value:?} clip {clip:?} step {step}: loss");
            }
            let (a, b) = (all_masters(&ops, &mut resident), all_masters(&ops, &mut streamed));
            assert_eq!(a, b, "{value:?} clip {clip:?}: masters");
            let init: Vec<u32> = task.init["head"].iter().map(|v| v.to_bits()).collect();
            assert_ne!(a[HEAD], init, "{value:?} clip {clip:?}: the table trained");
        }
    }
}

/// Spec: a layer-streamed run with a table restored from a committed checkpoint continues exactly: the restored table
/// reaches its device slot (it never reloads on its own), so the continued steps match the uninterrupted run.
#[test]
fn a_restored_layer_streamed_run_with_a_table_continues_exactly() {
    let ops = ops();
    let task = task_with_head();
    let dir = std::env::temp_dir().join(format!("brain-layer-ft-table-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ckpt = CheckpointDir::new(&dir).unwrap();
    let cfg = || config(ResidentValue::Fp8E4M3 { block: 128 }, Some(1), ClipPass::TwoPass);
    let mut a = build_with_head(&ops, cfg(), &task);
    for _ in 0..2 {
        streamed_head_step(&ops, &mut a, &task, Some(1e-3));
    }
    a.commit(&ops, &ckpt, &a.run_state(0, vec![])).unwrap();
    let mut b = build_with_head(&ops, cfg(), &task);
    // A different history first, so the table's device slot holds a stale value when the checkpoint is restored.
    for _ in 0..3 {
        streamed_head_step(&ops, &mut b, &task, None);
    }
    b.restore(&ops, &ckpt).unwrap().expect("a checkpoint exists");
    for step in 0..2 {
        let want = streamed_head_step(&ops, &mut a, &task, Some(1e-3));
        let got = streamed_head_step(&ops, &mut b, &task, Some(1e-3));
        assert_eq!(want.to_bits(), got.to_bits(), "step {step} after the restore: loss");
    }
    assert_eq!(all_masters(&ops, &mut a), all_masters(&ops, &mut b));
    std::fs::remove_dir_all(&dir).unwrap();
}
