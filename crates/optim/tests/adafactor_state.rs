// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in training with factored optimiser
// state then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec-level gates of Adafactor on the streamed optimiser: it trains a matrix with `rows + columns`
//! floats of second-moment state, a step does not depend on the tile size, and a run resumed from the
//! transactional checkpoint continues bit for bit.

use std::path::PathBuf;

use optim::checkpoint::{CheckpointDir, RunState};
use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, StreamedOptimizer, TensorSpec};
use optim::Adam;
use paramstore::precision::{MasterPolicy, ParamPrecision, StateFormat};

const ROWS: usize = 32;
const COLS: usize = 48;
const N: usize = ROWS * COLS;

struct Task {
    d: Vec<f32>,
    t: Vec<f32>,
}

impl Task {
    fn new() -> Task {
        let d = (0..N).map(|i| 0.1 * (1.0 + (i / COLS) as f32 * 0.1) * (1.0 + (i % COLS) as f32 * 0.05)).collect();
        let t = (0..N).map(|i| ((i * 13 % 29) as f32 / 29.0 - 0.5) * 2.0).collect();
        Task { d, t }
    }
    fn loss(&self, w: &[f32]) -> f64 {
        w.iter().zip(&self.d).zip(&self.t).map(|((w, d), t)| 0.5 * f64::from(*d) * f64::from(w - t).powi(2)).sum()
    }
    fn grad(&self, w: &[f32]) -> Vec<f32> {
        w.iter().enumerate().map(|(i, w)| self.d[i] * (w - self.t[i])).collect()
    }
}

struct Rig {
    opt: StreamedOptimizer<BlockStateStorage<MemBacking>>,
    p: HostParams<MemBacking, MemBacking>,
}

fn rig(rule: Rule, row_len: Option<usize>, master: MasterPolicy, tile: usize) -> Rig {
    let precision = ParamPrecision { master, ..ParamPrecision::FP32 };
    let specs = vec![TensorSpec { name: "w".into(), numel: N, precision, row_len }];
    let layout = state_layout(&rule, &specs);
    let st = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), 8192).unwrap();
    let opt = StreamedOptimizer::new(rule, specs, st, 5, tile).unwrap();
    let vl = tensor_layout(&[N], StateFormat::F32);
    let values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), 8192).unwrap();
    let grads = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), 8192).unwrap();
    Rig { opt, p: HostParams { values, grads } }
}

impl Rig {
    fn weights(&mut self) -> Vec<f32> {
        let mut w = vec![0f32; N];
        for (c, ch) in w.chunks_mut(1024).enumerate() {
            self.p.values.read_tile(0, 0, c * 1024, ch).unwrap();
        }
        w
    }
    fn step(&mut self, task: &Task, lr: f32) {
        let g = task.grad(&self.weights());
        for (c, ch) in g.chunks(1024).enumerate() {
            self.p.grads.write_tile(0, 0, c * 1024, ch, Rounding::NEAREST).unwrap();
        }
        self.opt.step(&mut self.p, lr, None, 1.0).unwrap();
    }
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("brain-adafactor-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Spec: Adafactor trains a matrix to a small loss, keeps `rows + columns` floats of second-moment state where
/// AdamW keeps `2 * rows * columns`, and reaches a loss in the neighbourhood of AdamW's on the same task.
#[test]
fn adafactor_trains_a_matrix_with_rows_plus_columns_of_state() {
    let task = Task::new();
    let start = task.loss(&vec![0.0; N]);
    let mut af = rig(Rule::adafactor(0.0), Some(COLS), MasterPolicy::None, 512);
    let mut adam = rig(Rule::AdamW { adam: Adam::default(), wd: 0.0 }, None, MasterPolicy::None, 512);
    assert_eq!(af.opt.state_physical_bytes(), 4 * (ROWS + COLS) as u64);
    assert_eq!(adam.opt.state_physical_bytes(), 2 * 4 * N as u64);
    for _ in 0..400 {
        af.step(&task, 0.02);
        adam.step(&task, 0.02);
    }
    let (la, lb) = (task.loss(&af.weights()), task.loss(&adam.weights()));
    println!("start {start:.4}, adafactor {la:.6}, adamw {lb:.6}");
    assert!(la < 0.02 * start, "adafactor must train: {la} from {start}");
    assert!(la < 5.0 * lb + 0.01 * start, "adafactor {la} should be in AdamW's neighbourhood {lb}");
}

/// Spec: a step does not depend on the tile size (tiles are whole rows), with a master copy beside the state.
#[test]
fn an_adafactor_step_is_independent_of_the_tile_size() {
    let task = Task::new();
    let run = |tile: usize| {
        let mut r = rig(Rule::adafactor(0.01), Some(COLS), MasterPolicy::F32, tile);
        for _ in 0..6 {
            r.step(&task, 0.02);
        }
        r.weights()
    };
    assert_eq!(run(48), run(100));
    assert_eq!(run(100), run(N));
}

/// Spec: a run committed through the transactional checkpoint and restored into a fresh optimiser continues
/// bit for bit, the factored statistics included.
#[test]
fn a_restored_adafactor_run_continues_exactly() {
    let task = Task::new();
    let dir = scratch("resume");
    let ckpt = CheckpointDir::new(&dir).unwrap();
    let mut a = rig(Rule::adafactor(0.0), Some(COLS), MasterPolicy::F32, 256);
    for _ in 0..5 {
        a.step(&task, 0.02);
    }
    let run = RunState { step: a.opt.step_count(), seed: a.opt.seed(), rng_counter: 0, schedule: vec![] };
    ckpt.commit(&mut a.opt, &mut a.p, &run).unwrap();
    for _ in 0..5 {
        a.step(&task, 0.02);
    }
    let mut b = rig(Rule::adafactor(0.0), Some(COLS), MasterPolicy::F32, 256);
    b.opt.init_master(&mut b.p).unwrap();
    let restored = ckpt.restore(&mut b.opt, &mut b.p).unwrap().expect("a checkpoint");
    assert_eq!(restored.step, 5);
    for _ in 0..5 {
        b.step(&task, 0.02);
    }
    assert_eq!(a.weights(), b.weights());
    std::fs::remove_dir_all(&dir).unwrap();
}
