// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of language models in bf16 and FP8 for its
// clients. If your team needs expertise in verifying that a low-precision training state learns what an fp32 one
// does, you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of resident low-precision full fine-tuning (`model::resident_ft`): the linears of a two-layer stack live
//! on the device as bf16 or FP8 `Weight`s, `Ops::weight_grad` produces their gradients, and the update runs
//! against an off-device master copy.
//!
//! * it learns what an fp32 AdamW reference learns on the same data, to the accuracy the format allows;
//! * the stored weight is exactly the codec's encoding of the master, and a master keeps updates smaller than the
//!   format's spacing alive;
//! * the layer-at-a-time step (deferred publication) equals the all-gradients step bit for bit;
//! * a configuration that does not fit is refused before anything is allocated;
//! * the streamed export in the chosen format reloads into the same resident weight and infers identically;
//! * commit and restore resume a run exactly.

use std::collections::HashMap;

use checkpoint::packed::{BoundedWorkspace, CodecRegistry, LoadConfig, LoadPolicy, Loader, MemoryBudget, PackedReader, SafetensorsSink};
use data::rng::Lcg;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::{DeviceBuffer, Gpu};
use model::memplan::{Authorized, Capacity};
use model::ops::{training_kernel_list, DeviceWeightSink, GradDtype, Ops, Weight};
use model::resident_ft::{ClipPass, ResidentBudget, ResidentFtConfig, ResidentFullFt, ResidentSpec, ResidentValue};
use optim::checkpoint::CheckpointDir;
use optim::rule::Rule;
use optim::Adam;
use paramstore::precision::{MasterPolicy, StateFormat};

const M: usize = 32;
const K0: usize = 32;
const H: usize = 32;
const O: usize = 16;

fn ops() -> Ops {
    let mut ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    ops.set_gemm_precision(GemmPrecision::Fp32Reference);
    ops
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain-resident-ft-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Task {
    x: Vec<f32>,
    target: Vec<f32>,
    init: HashMap<String, Vec<f32>>,
}

/// A teacher stack the student starts away from: the student must learn it.
fn task() -> Task {
    let mut rng = Lcg::random();
    let x = rng.vec_scaled(M * K0, 1.0);
    let t1 = rng.vec_scaled(H * K0, 0.3);
    let t2 = rng.vec_scaled(O * H, 0.3);
    let target = host_forward(&x, &t1, &t2).1;
    let mut init = HashMap::new();
    init.insert("l1.weight".to_string(), rng.vec_scaled(H * K0, 0.3));
    init.insert("l2.weight".to_string(), rng.vec_scaled(O * H, 0.3));
    Task { x, target, init }
}

fn matmul_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            y[i * n + j] = (0..k).map(|p| a[i * k + p] * b[j * k + p]).sum();
        }
    }
    y
}

fn host_forward(x: &[f32], w1: &[f32], w2: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let h = matmul_nt(x, w1, M, H, K0);
    let z = matmul_nt(&h, w2, M, O, H);
    (h, z)
}

fn loss(z: &[f32], t: &[f32]) -> f64 {
    z.iter().zip(t).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / (2.0 * M as f64)
}

const ADAM: Adam = Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 };
const LR: f32 = 1e-2;
const STEPS: u32 = 80;

/// The fp32 reference: host forward/backward, AdamW (the optimizer crate's own reference update).
fn reference(task: &Task) -> f64 {
    let mut w = [task.init["l1.weight"].clone(), task.init["l2.weight"].clone()];
    let mut st = [(vec![0f32; H * K0], vec![0f32; H * K0]), (vec![0f32; O * H], vec![0f32; O * H])];
    for t in 1..=STEPS {
        let (h, z) = host_forward(&task.x, &w[0], &w[1]);
        let dz: Vec<f32> = z.iter().zip(&task.target).map(|(a, b)| (a - b) / M as f32).collect();
        let dw2 = matmul_tn(&dz, &h, M, O, H);
        let dh = matmul_nn(&dz, &w[1], M, O, H);
        let dw1 = matmul_tn(&dh, &task.x, M, H, K0);
        for (i, g) in [dw1, dw2].iter().enumerate() {
            let (m, v) = &mut st[i];
            ADAM.update_slice(t, LR, 0.0, 1.0, &mut w[i], m, v, g);
        }
    }
    let z = host_forward(&task.x, &w[0], &w[1]).1;
    loss(&z, &task.target)
}

/// `dW[n,k] = dY[m,n]^T X[m,k]`.
fn matmul_tn(dy: &[f32], x: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut g = vec![0f32; n * k];
    for i in 0..m {
        for a in 0..n {
            for b in 0..k {
                g[a * k + b] += dy[i * n + a] * x[i * k + b];
            }
        }
    }
    g
}

/// `dX[m,k] = dY[m,n] W[n,k]`.
fn matmul_nn(dy: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut g = vec![0f32; m * k];
    for i in 0..m {
        for a in 0..n {
            for b in 0..k {
                g[i * k + b] += dy[i * n + a] * w[a * k + b];
            }
        }
    }
    g
}

fn config(value: ResidentValue, master: MasterPolicy, state: StateFormat, grad: GradDtype) -> ResidentFtConfig {
    ResidentFtConfig { value, master, state, rule: Rule::AdamW { adam: ADAM, wd: 0.0 }, grad, stochastic_state: false, seed: Some(7), tile_elems: 128, state_dir: None, defer_publication: false, window: None, clip_pass: ClipPass::TwoPass }
}

fn specs() -> Vec<ResidentSpec> {
    vec![ResidentSpec::linear("l1.weight", H as u32, K0 as u32, 0), ResidentSpec::linear("l2.weight", O as u32, H as u32, 1)]
}

fn roomy() -> ResidentBudget<'static> {
    ResidentBudget { capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 }, others: &[], authorized: Authorized::none() }
}

fn upload(ops: &Ops, host: &[f32]) -> DeviceBuffer {
    let b = ops.gpu().storage(host.len() as u64);
    ops.gpu().write_f32(&b, host);
    b
}

fn forward(ops: &Ops, w: &Weight, x: &DeviceBuffer, m: u32, y: &DeviceBuffer) {
    let mut s = Vec::new();
    let act = ops.act_f32(x, 0, m, w.k());
    ops.matmul_rows(&mut s, w, (0, w.n()), &act, y, 0);
    ops.gpu().submit(&[], &s);
}

/// One training step on the device: forward through the resident weights, host loss gradient, `dX` through the
/// tier, `dW` through `Ops::weight_grad`, then `update`. Returns the loss before the update.
fn train_step(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, update: impl FnOnce(&Ops, &mut ResidentFullFt, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer)) -> f64 {
    let g = ops.gpu();
    let x = upload(ops, &task.x);
    let h = g.storage((M * H) as u64);
    let z = g.storage((M * O) as u64);
    forward(ops, ft.weight(0), &x, M as u32, &h);
    forward(ops, ft.weight(1), &h, M as u32, &z);
    let zh = g.read(&z, M * O);
    let dz_host: Vec<f32> = zh.iter().zip(&task.target).map(|(a, b)| (a - b) / M as f32).collect();
    let dz = upload(ops, &dz_host);
    let dh = g.storage((M * H) as u64);
    let mut s = Vec::new();
    ops.matmul_dx(&mut s, ft.weight(1), &dz, M as u32, &dh, false);
    ft.weight_grad(ops, &mut s, 1, &h, &dz, M as u32);
    ft.weight_grad(ops, &mut s, 0, &x, &dh, M as u32);
    g.submit(&[], &s);
    let l = loss(&zh, &task.target);
    update(ops, ft, &x, &h, &dz);
    l
}

fn direct(ops: &Ops, ft: &mut ResidentFullFt, _: &DeviceBuffer, _: &DeviceBuffer, _: &DeviceBuffer) {
    ft.step(ops, LR, None, 1.0).unwrap();
}

fn layered(ops: &Ops, ft: &mut ResidentFullFt, _: &DeviceBuffer, _: &DeviceBuffer, _: &DeviceBuffer) {
    // Layer order of a backward pass: the last layer completes first.
    ft.layered_step(ops, LR, 1.0, |step| {
        step.update(1)?;
        step.update(0)
    })
    .unwrap();
}

fn final_loss(ops: &Ops, ft: &mut ResidentFullFt, task: &Task, update: fn(&Ops, &mut ResidentFullFt, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer)) -> f64 {
    let mut last = 0.0;
    for _ in 0..STEPS {
        last = train_step(ops, ft, task, update);
    }
    // the loss after the last update
    train_step(ops, ft, task, |ops, ft, _, _, _| ft.zero_grads(ops)).min(last)
}

fn build(ops: &Ops, cfg: ResidentFtConfig, task: &Task) -> ResidentFullFt {
    ResidentFullFt::new(ops, cfg, specs(), &task.init, &roomy()).unwrap()
}

fn device_bytes(ops: &Ops, w: &Weight) -> Vec<u32> {
    let (payload, scales) = match w {
        Weight::BF16 { w, .. } => (w, None),
        Weight::FP8 { w, s, .. } | Weight::I8 { w, s, .. } | Weight::NF4 { w, s, .. } => (w, Some(s)),
        _ => panic!("unexpected tier"),
    };
    let words = |b: &DeviceBuffer, n: usize| ops.gpu().read(b, n).into_iter().map(f32::to_bits).collect::<Vec<u32>>();
    let n = (w.n() * w.k()) as usize;
    let mut out = match w {
        Weight::BF16 { .. } => words(payload, n / 2),
        Weight::NF4 { .. } => words(payload, n.div_ceil(8)),
        _ => words(payload, n.div_ceil(4)),
    };
    if let Some(s) = scales {
        let scale_words = match w {
            Weight::FP8 { .. } => 1,
            _ => n / 32,
        };
        out.extend(ops.gpu().read(s, scale_words).into_iter().map(f32::to_bits));
    }
    out
}

/// Spec: a stack trained with resident bf16 or FP8 weights, an off-device f32 master and f32 or compressed
/// moments reaches the loss an fp32 AdamW reference reaches (well below where it started), to the accuracy the
/// format allows.
#[test]
fn resident_low_precision_weights_learn_what_the_fp32_reference_learns() {
    let ops = ops();
    let task = task();
    let want = reference(&task);
    let start = {
        let z = host_forward(&task.x, &task.init["l1.weight"], &task.init["l2.weight"]).1;
        loss(&z, &task.target)
    };
    assert!(want < 0.1 * start, "the reference itself learns the task ({start} -> {want})");
    let cases = [
        ("bf16, f32 master, f32 moments, f32 grads", config(ResidentValue::Bf16, MasterPolicy::F32, StateFormat::F32, GradDtype::F32), 0.05),
        ("bf16, f32 master, int8 moments, bf16 grads", config(ResidentValue::Bf16, MasterPolicy::F32, StateFormat::Int8Block { block: 64 }, GradDtype::Bf16), 0.1),
        ("bf16, f32 master, log8 moments, bf16 grads", config(ResidentValue::Bf16, MasterPolicy::F32, StateFormat::Log8Block { block: 64 }, GradDtype::Bf16), 0.1),
        ("fp8 e4m3, f32 master, f32 moments", config(ResidentValue::Fp8E4M3 { block: 128 }, MasterPolicy::F32, StateFormat::F32, GradDtype::F32), 0.15),
        ("fp8 e4m3, bf16 master, bf16 moments", config(ResidentValue::Fp8E4M3 { block: 128 }, MasterPolicy::Bf16, StateFormat::Bf16, GradDtype::F32), 0.2),
        ("QAT int8 (straight-through), f32 master", config(ResidentValue::Int8, MasterPolicy::F32, StateFormat::F32, GradDtype::F32), 0.15),
        ("QAT nf4 (straight-through), f32 master", config(ResidentValue::Nf4, MasterPolicy::F32, StateFormat::F32, GradDtype::F32), 0.6),
        ("bf16, f32 master, adafactor", ResidentFtConfig { rule: Rule::adafactor(0.0), ..config(ResidentValue::Bf16, MasterPolicy::F32, StateFormat::F32, GradDtype::F32) }, 0.3),
    ];
    for (label, cfg, bound) in cases {
        let mut ft = build(&ops, cfg, &task);
        let got = final_loss(&ops, &mut ft, &task, direct);
        eprintln!("{label}: loss {start:.4} -> {got:.5} (fp32 reference {want:.5})");
        assert!(got < 0.1 * start, "{label}: the loss fell from {start} to {got}");
        assert!(got < want + bound * start, "{label}: final loss {got} is not within {bound} of the start loss of the reference's {want}");
    }
}

/// Spec: the layer-at-a-time step with deferred publication gives exactly the weights the all-gradients step
/// gives, for every resident format.
#[test]
fn the_layered_step_equals_the_all_gradients_step() {
    let ops = ops();
    let task = task();
    for value in [ResidentValue::Bf16, ResidentValue::Fp8E4M3 { block: 128 }, ResidentValue::Int8, ResidentValue::Nf4] {
        let cfg = || config(value, MasterPolicy::F32, StateFormat::F32, GradDtype::F32);
        let (mut a, mut b) = (build(&ops, cfg(), &task), build(&ops, cfg(), &task));
        for _ in 0..5 {
            train_step(&ops, &mut a, &task, direct);
            train_step(&ops, &mut b, &task, layered);
        }
        for i in 0..2 {
            assert_eq!(device_bytes(&ops, a.weight(i)), device_bytes(&ops, b.weight(i)), "{value:?} tensor {i}");
        }
    }
}

/// Spec: the stored weight is the codec's encoding of the master, and a master keeps an update far below the
/// format's spacing alive: with plain SGD at a tiny rate the bf16 weight moves once the master has accumulated a
/// spacing's worth, which a master-less bf16 value never would.
#[test]
fn the_stored_weight_follows_a_master_that_accumulates_sub_spacing_updates() {
    let ops = ops();
    let task = task();
    let mut cfg = config(ResidentValue::Bf16, MasterPolicy::F32, StateFormat::F32, GradDtype::F32);
    cfg.rule = Rule::SgdMomentum { momentum: 0.0, wd: 0.0, nesterov: false };
    let mut ft = build(&ops, cfg, &task);
    let before = device_bytes(&ops, ft.weight(0));
    let changed = |now: &[u32]| now.iter().zip(&before).filter(|(a, b)| a != b).count() as f64 / before.len() as f64;
    // A gradient of 1e-3 everywhere at lr 1e-3: 1e-6 per step, far below bf16's 2^-8 relative spacing of the
    // weights. The init values are not on the bf16 grid, so a stray weight sitting at a rounding boundary may
    // flip; almost none do.
    let g = ft.grad_buffer(0).clone();
    let write_grad = |ops: &Ops| ops.gpu().write_f32(&g, &vec![1e-3; H * K0]);
    write_grad(&ops);
    ft.step(&ops, 1e-3, None, 1.0).unwrap();
    let one = changed(&device_bytes(&ops, ft.weight(0)));
    assert!(one < 0.05, "one sub-spacing step changes {one} of the stored words");
    for _ in 0..4000 {
        write_grad(&ops);
        ft.step(&ops, 1e-3, None, 1.0).unwrap();
    }
    let many = changed(&device_bytes(&ops, ft.weight(0)));
    assert!(many > 0.5, "4000 accumulated updates reach the stored weights through the master ({many})");
}

/// Spec: a configuration that does not fit is refused before anything is allocated, with the arithmetic; one
/// that fits only when the optimizer state is on the host is accepted because that is where the state lives.
#[test]
fn a_configuration_that_does_not_fit_is_refused_before_loading() {
    let ops = ops();
    let task = task();
    let cfg = config(ResidentValue::Bf16, MasterPolicy::F32, StateFormat::F32, GradDtype::F32);
    let weights = (H * K0 + O * H) as u64 * 2;
    let grads = (H * K0 + O * H) as u64 * 4;
    let tight = ResidentBudget { capacity: Capacity { device_bytes: weights + grads - 1, host_bytes: 1 << 30 }, others: &[], authorized: Authorized::none() };
    let err = ResidentFullFt::new(&ops, cfg.clone(), specs(), &task.init, &tight).err().expect("refused");
    assert!(err.contains("does not fit") && err.contains("Weights"), "{err}");
    let exact = ResidentBudget { capacity: Capacity { device_bytes: weights + grads, host_bytes: 1 << 30 }, others: &[], authorized: Authorized::none() };
    ResidentFullFt::new(&ops, cfg.clone(), specs(), &task.init, &exact).expect("fits exactly: master and moments are host bytes");
    let host_poor = ResidentBudget { capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1000 }, others: &[], authorized: Authorized::none() };
    let err = ResidentFullFt::new(&ops, cfg, specs(), &task.init, &host_poor).err().expect("refused");
    assert!(err.contains("OptimizerState"), "{err}");
}

/// Spec: a resident value without a master is refused by name (updates below the spacing would be lost), as is a
/// row length the format cannot hold.
#[test]
fn unsupported_configurations_are_refused_by_name() {
    let ops = ops();
    let task = task();
    let no_master = config(ResidentValue::Bf16, MasterPolicy::None, StateFormat::F32, GradDtype::F32);
    assert!(ResidentFullFt::new(&ops, no_master, specs(), &task.init, &roomy()).err().unwrap().contains("master"));
    let mut odd = specs();
    odd[0].k = 31;
    let cfg = config(ResidentValue::Bf16, MasterPolicy::F32, StateFormat::F32, GradDtype::F32);
    assert!(ResidentFullFt::new(&ops, cfg, odd, &task.init, &roomy()).err().unwrap().contains("l1.weight"));
}

/// Spec: the streamed export of the masters in the resident format, reloaded through the packed loader into
/// resident weights, is byte-identical to what trains, and infers the same outputs; an export into another
/// format (bf16 from an FP8 run) reloads to the master within that format's precision.
#[test]
fn the_exported_checkpoint_reloads_into_the_same_weights_and_infers_identically() {
    let ops = ops();
    let task = task();
    let dir = scratch("export");
    for value in [ResidentValue::Bf16, ResidentValue::Fp8E4M3 { block: 128 }, ResidentValue::Int8, ResidentValue::Nf4] {
        let mut ft = build(&ops, config(value, MasterPolicy::F32, StateFormat::F32, GradDtype::F32), &task);
        for _ in 0..6 {
            train_step(&ops, &mut ft, &task, direct);
        }
        let path = dir.join(format!("{value:?}.safetensors"));
        let enc = vec![value.encoding(); 2];
        let descs = ft.export_descriptors(&enc).unwrap();
        let mut sink = SafetensorsSink::create(&path, &descs, &serde_json::json!({})).unwrap();
        let report = ft.export(&ops, &enc, &mut sink).unwrap();
        sink.commit().unwrap();
        assert!(report.staging_bytes <= 8 * ft.tile_elems() * 4 + 4096, "export staging is a tile, not a tensor: {}", report.staging_bytes);

        // brain's word-packed int8 and NF4 payloads do not say what they are: the descriptors are declared.
        let reader = descs.iter().filter(|d| !matches!(d.encoding.element, checkpoint::packed::Element::Float(_))).fold(PackedReader::open(&path).unwrap(), |r, d| r.with_descriptor(d.clone()));
        let codecs = CodecRegistry::reference();
        let loader = Loader { reader: &reader, codecs: &codecs, workspace: BoundedWorkspace::new(MemoryBudget::new(64 << 20, 0)), config: LoadConfig::default() };
        let xdev = upload(&ops, &task.x);
        for i in 0..2 {
            let name = &specs()[i].name;
            let mut load_sink = DeviceWeightSink::new(&ops);
            loader.load(name, &LoadPolicy::Preserve, &mut load_sink).unwrap();
            let reloaded = load_sink.take_weight().unwrap();
            assert_eq!(device_bytes(&ops, &reloaded), device_bytes(&ops, ft.weight(i)), "{value:?} {name}: reload is byte-identical");
            let k = ft.weight(i).k();
            let rows = M as u32;
            let (ya, yb) = (ops.gpu().storage((M as u32 * ft.weight(i).n()) as u64), ops.gpu().storage((M as u32 * ft.weight(i).n()) as u64));
            let input = if i == 0 { xdev.clone() } else { upload(&ops, &task.x[..M * H]) };
            assert_eq!(k as usize, if i == 0 { K0 } else { H });
            forward(&ops, ft.weight(i), &input, rows, &ya);
            forward(&ops, &reloaded, &input, rows, &yb);
            assert_eq!(ops.gpu().read(&ya, M * ft.weight(i).n() as usize), ops.gpu().read(&yb, M * ft.weight(i).n() as usize), "{value:?} {name}: inference matches");
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Spec: a run resumed from a committed checkpoint continues exactly: masters, moments, step and the stored
/// weights match the uninterrupted run bit for bit.
#[test]
fn a_restored_run_continues_exactly() {
    let ops = ops();
    let task = task();
    let dir = scratch("resume");
    for value in [ResidentValue::Bf16, ResidentValue::Fp8E4M3 { block: 128 }, ResidentValue::Int8, ResidentValue::Nf4] {
        let cfg = || config(value, MasterPolicy::F32, StateFormat::Int8Block { block: 64 }, GradDtype::F32);
        let ckpt = CheckpointDir::new(dir.join(format!("{value:?}"))).unwrap();
        let mut a = build(&ops, cfg(), &task);
        for _ in 0..3 {
            train_step(&ops, &mut a, &task, direct);
        }
        let run = a.run_state(0, vec![("lr".into(), LR as f64)]);
        a.commit(&ops, &ckpt, &run).unwrap();
        for _ in 0..3 {
            train_step(&ops, &mut a, &task, direct);
        }
        let mut b = build(&ops, cfg(), &task);
        let restored = b.restore(&ops, &ckpt).unwrap().expect("a checkpoint exists");
        assert_eq!((restored.step, b.step_count()), (3, 3));
        for _ in 0..3 {
            train_step(&ops, &mut b, &task, direct);
        }
        for i in 0..2 {
            assert_eq!(device_bytes(&ops, a.weight(i)), device_bytes(&ops, b.weight(i)), "{value:?} tensor {i}");
        }
        assert_eq!(a.step_count(), b.step_count());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
