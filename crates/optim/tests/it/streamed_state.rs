// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in 8-bit optimiser state and resumable
// full fine-tuning then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec-level gates of the streamed optimiser on a small fixed task:
//! quality of 8-bit state against fp32 AdamW, resume parity through the
//! transactional checkpoint, crash safety of the commit, tile independence.

use std::io;
use std::path::PathBuf;

use optim::checkpoint::{CheckpointDir, RunState, Stage};
use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, FileBacking, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, StreamedOptimizer, TensorSpec};
use optim::Adam;
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, StateFormat, ValueDtype};

const TILE: usize = 256;

/// A scratch directory under the temp dir, removed when dropped.
struct Scratch(PathBuf);

impl std::ops::Deref for Scratch {
    type Target = std::path::Path;
    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(name: &str) -> Scratch {
    let d = std::env::temp_dir().join(format!("brain-optim-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    Scratch(d)
}

fn prec(state: StateFormat, sr: bool) -> ParamPrecision {
    ParamPrecision { value: ValueDtype::F32, grad: GradDtype::F32, state, master: MasterPolicy::None, stochastic_rounding: sr }
}

/// Ill-conditioned quadratic with a deterministic gradient perturbation:
/// loss = 0.5 sum d_i (w_i - t_i)^2, grad = d_i (w_i - t_i) + noise.
struct Task {
    d: Vec<f32>,
    t: Vec<f32>,
}

impl Task {
    fn new(n: usize) -> Task {
        let d = (0..n).map(|i| 0.05 * 10f32.powf(2.0 * ((i * 37 % n) as f32 / n as f32))).collect();
        let t = (0..n).map(|i| ((i * 13 % 29) as f32 / 29.0 - 0.5) * 2.0).collect();
        Task { d, t }
    }
    fn loss(&self, w: &[f32]) -> f64 {
        w.iter().zip(&self.d).zip(&self.t).map(|((w, d), t)| 0.5 * (*d as f64) * ((w - t) as f64).powi(2)).sum()
    }
    fn grad(&self, w: &[f32], step: u32) -> Vec<f32> {
        w.iter()
            .enumerate()
            .map(|(i, w)| self.d[i] * (w - self.t[i]) + 0.02 * (((i as u32 * 31 + step * 17) % 23) as f32 / 23.0 - 0.5))
            .collect()
    }
}

type Mem = BlockStateStorage<MemBacking>;

struct Rig {
    opt: StreamedOptimizer<Mem>,
    p: HostParams<MemBacking, MemBacking>,
    n: usize,
}

fn rig(n: usize, precision: ParamPrecision, tile: usize, w0: &[f32]) -> Rig {
    let rule = Rule::AdamW { adam: Adam::default(), wd: 0.0 };
    let specs = vec![TensorSpec { name: "w".into(), numel: n, precision, row_len: None }];
    let layout = state_layout(&rule, &specs);
    let st = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), 4096).unwrap();
    let opt = StreamedOptimizer::new(rule, specs, st, 5, tile).unwrap();
    let vl = tensor_layout(&[n], StateFormat::F32);
    let mut values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), 4096).unwrap();
    let grads = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), 4096).unwrap();
    for (c, ch) in w0.chunks(1024).enumerate() {
        values.write_tile(0, 0, c * 1024, ch, Rounding::NEAREST).unwrap();
    }
    Rig { opt, p: HostParams { values, grads }, n }
}

impl Rig {
    fn weights(&mut self) -> Vec<f32> {
        let mut w = vec![0f32; self.n];
        for (c, ch) in w.chunks_mut(1024).enumerate() {
            self.p.values.read_tile(0, 0, c * 1024, ch).unwrap();
        }
        w
    }
    fn train_step(&mut self, task: &Task) {
        let w = self.weights();
        let g = task.grad(&w, self.opt.step_count() + 1);
        for (c, ch) in g.chunks(1024).enumerate() {
            self.p.grads.write_tile(0, 0, c * 1024, ch, Rounding::NEAREST).unwrap();
        }
        self.opt.step(&mut self.p, 0.01, None, 1.0).unwrap();
    }
}

/// Gate: 8-bit blockwise moments track fp32 AdamW on a fixed task. Records the
/// loss curve gap and the per-step update difference; the thresholds are set
/// from the measured numbers printed by this test (measured on this task: int8 curve gap 0.061, update rel-L2 0.10-0.11).
#[test]
fn int8_moments_track_fp32_adamw_on_a_fixed_task() {
    let n = 4096;
    let task = Task::new(n);
    let w0 = vec![0.0f32; n];
    let steps = 300;
    let run = |p: ParamPrecision| {
        let mut r = rig(n, p, TILE, &w0);
        let (mut losses, mut updates) = (vec![], vec![]);
        for _ in 0..steps {
            let before = r.weights();
            r.train_step(&task);
            let after = r.weights();
            losses.push(task.loss(&after));
            updates.push(after.iter().zip(&before).map(|(a, b)| a - b).collect::<Vec<f32>>());
        }
        (losses, updates)
    };
    let (lref, uref) = run(prec(StateFormat::F32, false));
    let cases = [
        ("bf16", prec(StateFormat::Bf16, true)),
        ("int8 block64 SR", prec(StateFormat::Int8Block { block: 64 }, true)),
        ("int8 block256 SR", prec(StateFormat::Int8Block { block: 256 }, true)),
        ("int8 block64 nearest (informational)", prec(StateFormat::Int8Block { block: 64 }, false)),
    ];
    println!("loss fp32: start {:.4} end {:.6}", task.loss(&w0), lref[steps - 1]);
    let mut int8_gap = 0f64;
    for (name, p) in cases {
        let (l, u) = run(p);
        let curve_gap = l.iter().zip(&lref).map(|(a, b)| ((a - b) / b.max(1e-12)).abs()).fold(0f64, f64::max);
        let (mut num, mut den) = (0f64, 0f64);
        for (ua, ub) in u.iter().zip(&uref) {
            for (a, b) in ua.iter().zip(ub) {
                num += ((a - b) as f64).powi(2);
                den += (*b as f64).powi(2);
            }
        }
        let update_rel = (num / den).sqrt();
        println!("{name}: end loss {:.6}, max relative loss-curve gap {:.4}, update rel-L2 diff {:.4}", l[steps - 1], curve_gap, update_rel);
        if name.contains("informational") {
            continue;
        }
        assert!(l[steps - 1] < 0.02 * task.loss(&w0), "{name} must converge");
        assert!(curve_gap < 0.1, "{name}: loss curve gap {curve_gap}");
        if name.starts_with("int8") {
            int8_gap = int8_gap.max(update_rel);
            assert!(update_rel < 0.2, "{name}: update diff {update_rel}");
        }
    }
    assert!(int8_gap > 0.0);
}

/// Gate: the result of a step does not depend on the tile size, even with
/// 8-bit state and stochastic rounding (rounding is keyed on absolute indices).
#[test]
fn stochastic_int8_step_is_independent_of_tile_size() {
    let n = 3000;
    let task = Task::new(n);
    let w0 = vec![0.1f32; n];
    let p = prec(StateFormat::Int8Block { block: 64 }, true);
    let finish = |tile: usize| {
        let mut r = rig(n, p, tile, &w0);
        for _ in 0..5 {
            r.train_step(&task);
        }
        r.weights()
    };
    assert_eq!(finish(64), finish(1024));
}

fn file_rig(dir: &std::path::Path, n: usize, p: ParamPrecision, w0: &[f32]) -> (StreamedOptimizer<BlockStateStorage<FileBacking>>, HostParams<FileBacking, FileBacking>) {
    let rule = Rule::AdamW { adam: Adam::default(), wd: 0.01 };
    let specs = vec![TensorSpec { name: "w".into(), numel: n, precision: p, row_len: None }];
    let layout = state_layout(&rule, &specs);
    let st = BlockStateStorage::new(layout.clone(), FileBacking::create(&dir.join("opt.state"), layout.total_bytes()).unwrap(), 4096).unwrap();
    let opt = StreamedOptimizer::new(rule, specs, st, 99, TILE).unwrap();
    let vl = tensor_layout(&[n], StateFormat::F32);
    let mut values =
        BlockStateStorage::new(vl.clone(), FileBacking::create(&dir.join("values"), vl.total_bytes()).unwrap(), 4096).unwrap();
    let grads = BlockStateStorage::new(vl.clone(), FileBacking::create(&dir.join("grads"), vl.total_bytes()).unwrap(), 4096).unwrap();
    for (c, ch) in w0.chunks(1024).enumerate() {
        values.write_tile(0, 0, c * 1024, ch, Rounding::NEAREST).unwrap();
    }
    (opt, HostParams { values, grads })
}

fn fstep(opt: &mut StreamedOptimizer<BlockStateStorage<FileBacking>>, p: &mut HostParams<FileBacking, FileBacking>, task: &Task, n: usize) {
    let mut w = vec![0f32; n];
    for (c, ch) in w.chunks_mut(1024).enumerate() {
        p.values.read_tile(0, 0, c * 1024, ch).unwrap();
    }
    let g = task.grad(&w, opt.step_count() + 1);
    for (c, ch) in g.chunks(1024).enumerate() {
        p.grads.write_tile(0, 0, c * 1024, ch, Rounding::NEAREST).unwrap();
    }
    opt.step(p, 0.01, Some(1.0), 1.0).unwrap();
}

fn fweights(p: &mut HostParams<FileBacking, FileBacking>, n: usize) -> Vec<f32> {
    let mut w = vec![0f32; n];
    for (c, ch) in w.chunks_mut(1024).enumerate() {
        p.values.read_tile(0, 0, c * 1024, ch).unwrap();
    }
    w
}

/// Gate (T3): run 10 steps straight; run 6, checkpoint, throw the process
/// state away, restore into fresh files, run 4 more - bit-identical weights
/// and state, with stochastic int8 state on a streamed on-disk optimiser.
#[test]
fn resume_from_checkpoint_is_bit_identical_to_an_uninterrupted_run() {
    let n = 2000;
    let task = Task::new(n);
    let w0: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1).collect();
    let p = prec(StateFormat::Int8Block { block: 64 }, true);

    let da = scratch("resume-a");
    let (mut a, mut pa) = file_rig(&da, n, p, &w0);
    for _ in 0..10 {
        fstep(&mut a, &mut pa, &task, n);
    }

    let db = scratch("resume-b");
    let (mut b, mut pb) = file_rig(&db, n, p, &w0);
    for _ in 0..6 {
        fstep(&mut b, &mut pb, &task, n);
    }
    let ck = CheckpointDir::new(db.join("ckpt")).unwrap();
    let run = RunState { step: 6, seed: b.seed(), rng_counter: 42, schedule: vec![("lr_pos".into(), 6.0)] };
    ck.commit(&mut b, &mut pb, &run).unwrap();
    drop((b, pb));

    let dc = scratch("resume-c");
    let (mut c, mut pc) = file_rig(&dc, n, p, &vec![0.0; n]);
    // a differently seeded fresh optimiser: restore must bring back the seed
    c.set_position(0, 12345);
    let got = CheckpointDir::new(db.join("ckpt")).unwrap().restore(&mut c, &mut pc).unwrap().unwrap();
    assert_eq!(got, run);
    assert_eq!((c.step_count(), c.seed()), (6, 99));
    for _ in 0..4 {
        fstep(&mut c, &mut pc, &task, n);
    }
    assert_eq!(fweights(&mut pa, n), fweights(&mut pc, n));
    let (mut sa, mut sc) = (Vec::new(), Vec::new());
    a.state_mut().export(&mut sa).unwrap();
    c.state_mut().export(&mut sc).unwrap();
    assert!(sa == sc, "optimiser state must match bit for bit");
}

/// Gate: a crash at any point of a commit leaves the previous checkpoint
/// restorable, and a corrupted newest generation falls back to the older one.
#[test]
fn a_crashed_or_corrupted_commit_never_loses_the_last_good_checkpoint() {
    let n = 500;
    let task = Task::new(n);
    let p = prec(StateFormat::Int8Block { block: 64 }, false);
    let dir = scratch("crash");
    let (mut o, mut pp) = file_rig(&dir, n, p, &vec![0.2; n]);
    let ck = CheckpointDir::new(dir.join("ckpt")).unwrap();
    let seed: u64 = std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), 0u8);
    let rs = |s: u32| RunState { step: s, seed, rng_counter: 0, schedule: vec![] };
    for _ in 0..2 {
        fstep(&mut o, &mut pp, &task, n);
    }
    ck.commit(&mut o, &mut pp, &rs(2)).unwrap();
    let good = fweights(&mut pp, n);
    fstep(&mut o, &mut pp, &task, n);

    for stage in [Stage::DataWritten, Stage::MetaWritten] {
        ck.commit_until(&mut o, &mut pp, &rs(3), stage).unwrap();
        let (mut r, mut pr) = file_rig(&scratch("crash-r"), n, p, &vec![0.0; n]);
        let got = ck.restore(&mut r, &mut pr).unwrap().unwrap();
        assert_eq!(got.step, 2, "a crash after {stage:?} must leave step 2 as the newest");
        assert_eq!(fweights(&mut pr, n), good);
    }

    ck.commit(&mut o, &mut pp, &rs(3)).unwrap();
    // flip one byte of the newest generation's state: it must be rejected
    let newest = std::fs::read_dir(dir.join("ckpt")).unwrap().map(|e| e.unwrap().path()).find(|p| p.to_str().unwrap().contains("gen-0000000003")).unwrap();
    let sp = newest.join("state.bin");
    let mut bytes = std::fs::read(&sp).unwrap();
    bytes[10] ^= 0xff;
    std::fs::write(&sp, bytes).unwrap();
    let (mut r, mut pr) = file_rig(&scratch("crash-r2"), n, p, &vec![0.0; n]);
    assert_eq!(ck.restore(&mut r, &mut pr).unwrap().unwrap().step, 2);
    assert_eq!(fweights(&mut pr, n), good);
}

/// A checkpoint from a different tensor table or rule is refused, not
/// loaded into a mismatched layout.
#[test]
fn a_checkpoint_for_other_tensors_is_refused() {
    let dir = scratch("mismatch");
    let p = prec(StateFormat::F32, false);
    let (mut o, mut pp) = file_rig(&dir, 100, p, &vec![0.0; 100]);
    let ck = CheckpointDir::new(dir.join("ckpt")).unwrap();
    ck.commit(&mut o, &mut pp, &RunState { step: 1, seed: std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), 0u8), rng_counter: 0, schedule: vec![] }).unwrap();
    let dir2 = scratch("mismatch2");
    let (mut o2, mut p2) = file_rig(&dir2, 200, p, &vec![0.0; 200]);
    let err: io::Error = ck.restore(&mut o2, &mut p2).unwrap_err();
    assert!(err.to_string().contains("differs"), "{err}");
}
