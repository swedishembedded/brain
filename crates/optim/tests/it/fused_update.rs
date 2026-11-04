// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in host-side optimiser updates that keep
// up with a GPU, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The update of a tile done in place on addressable state, piece by piece across the CPU pool, gives the bits of the
//! staged tile loop (state read into scratch, updated, written back): same values, same state bytes, for every state
//! format, with and without a master copy, with nearest and stochastic rounding.

use std::io;

use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{Backing, BlockStateStorage, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, StreamedOptimizer, TensorSpec};
use optim::Adam;
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, StateFormat, ValueDtype};

/// Memory that is not addressable from outside: only reads and writes at an offset, so the optimizer stages its tiles.
struct Opaque(MemBacking);

impl Backing for Opaque {
    fn read_at(&mut self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        self.0.read_at(off, buf)
    }
    fn write_at(&mut self, off: u64, buf: &[u8]) -> io::Result<()> {
        self.0.write_at(off, buf)
    }
    fn len(&self) -> u64 {
        self.0.len()
    }
    fn sync(&mut self) -> io::Result<()> {
        self.0.sync()
    }
}

const N: usize = 40_000;
/// Not a multiple of the update piece, so a tile ends mid-piece and the last tile is short.
const TILE: usize = 12_288;
const STEPS: u32 = 3;

/// Three steps of AdamW on a fixed task over `backing`; returns the final values and the raw state bytes.
fn run<B: Backing>(backing: impl Fn(u64) -> B, precision: ParamPrecision) -> (Vec<f32>, Vec<u8>) {
    let rule = Rule::AdamW { adam: Adam { beta1: 0.9, beta2: 0.99, eps: 1e-6 }, wd: 0.01 };
    let specs = vec![TensorSpec { name: "w".into(), numel: N, precision, row_len: None }];
    let layout = state_layout(&rule, &specs);
    let state = BlockStateStorage::new(layout.clone(), backing(layout.total_bytes()), TILE).unwrap();
    let mut opt = StreamedOptimizer::new(rule, specs, state, 7, TILE).unwrap();
    let vl = tensor_layout(&[N], StateFormat::F32);
    let mut values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), TILE).unwrap();
    let mut grads = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), TILE).unwrap();
    let w0: Vec<f32> = (0..N).map(|i| ((i * 7 % 101) as f32 / 101.0) - 0.5).collect();
    let g0: Vec<f32> = (0..N).map(|i| (((i * 13) % 37) as f32 / 37.0 - 0.5) * 0.1).collect();
    for start in (0..N).step_by(TILE) {
        let end = (start + TILE).min(N);
        values.write_tile(0, 0, start, &w0[start..end], Rounding::NEAREST).unwrap();
        grads.write_tile(0, 0, start, &g0[start..end], Rounding::NEAREST).unwrap();
    }
    let mut params = HostParams { values, grads };
    if precision.master != MasterPolicy::None {
        opt.init_master(&mut params).unwrap();
    }
    for _ in 0..STEPS {
        opt.step(&mut params, 1e-2, None, 0.5).unwrap();
    }
    let mut out = vec![0f32; N];
    for start in (0..N).step_by(TILE) {
        let end = (start + TILE).min(N);
        params.values.read_tile(0, 0, start, &mut out[start..end]).unwrap();
    }
    let mut bytes = Vec::new();
    opt.state_mut().export(&mut bytes).unwrap();
    (out, bytes)
}

#[test]
fn the_in_place_update_gives_the_bits_of_the_staged_one() {
    let p = |state, master, sr| ParamPrecision { value: ValueDtype::F32, grad: GradDtype::F32, state, master, stochastic_rounding: sr };
    for (name, precision) in [
        ("f32 state with an f32 master", p(StateFormat::F32, MasterPolicy::F32, false)),
        ("f32 state, no master", p(StateFormat::F32, MasterPolicy::None, false)),
        ("bf16 state with a bf16 master, stochastic", p(StateFormat::Bf16, MasterPolicy::Bf16, true)),
        ("int8 state with an f32 master, stochastic", p(StateFormat::Int8Block { block: 32 }, MasterPolicy::F32, true)),
        ("log8 state, no master", p(StateFormat::Log8Block { block: 64 }, MasterPolicy::None, false)),
    ] {
        let (in_place_values, in_place_state) = run(MemBacking::zeroed, precision);
        let (staged_values, staged_state) = run(|n| Opaque(MemBacking::zeroed(n)), precision);
        assert_eq!(in_place_values, staged_values, "{name}: values");
        assert_eq!(in_place_state, staged_state, "{name}: state bytes");
        assert!(in_place_values.iter().any(|&v| v != 0.0), "{name}: the task moved the values");
    }
}

/// Seeding a master tile by tile from values the caller already holds is seeding it from the values' own store.
#[test]
fn seeding_a_master_tile_by_tile_equals_seeding_it_from_the_store() {
    let precision = ParamPrecision { value: ValueDtype::F32, grad: GradDtype::F32, state: StateFormat::F32, master: MasterPolicy::Bf16, stochastic_rounding: true };
    let rule = Rule::AdamW { adam: Adam::default(), wd: 0.0 };
    let build = || {
        let specs = vec![TensorSpec { name: "w".into(), numel: N, precision, row_len: None }];
        let layout = state_layout(&rule, &specs);
        let state = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), TILE).unwrap();
        StreamedOptimizer::new(rule, specs, state, 3, TILE).unwrap()
    };
    let w0: Vec<f32> = (0..N).map(|i| (i as f32).sin()).collect();
    let vl = tensor_layout(&[N], StateFormat::F32);
    let mut values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), TILE).unwrap();
    let grads = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), TILE).unwrap();
    for start in (0..N).step_by(TILE) {
        let end = (start + TILE).min(N);
        values.write_tile(0, 0, start, &w0[start..end], Rounding::NEAREST).unwrap();
    }
    let mut from_store = build();
    from_store.init_master(&mut HostParams { values, grads }).unwrap();
    let mut by_tile = build();
    for start in (0..N).step_by(TILE) {
        let end = (start + TILE).min(N);
        by_tile.init_master_tile(0, start, &w0[start..end]).unwrap();
    }
    let (mut a, mut b) = (Vec::new(), Vec::new());
    from_store.state_mut().export(&mut a).unwrap();
    by_tile.state_mut().export(&mut b).unwrap();
    assert_eq!(a, b);
}
