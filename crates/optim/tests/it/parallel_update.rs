// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in host-side optimiser updates that keep
// up with a GPU, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A tile large enough to be updated on several threads gives the bits the serial reference gives: the elements
//! are independent, so how the tile is cut must not show in the result.

use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, StreamedOptimizer, TensorSpec};
use optim::Adam;
use paramstore::precision::{ParamPrecision, StateFormat};

#[test]
fn a_large_tile_updates_on_several_threads_with_the_serial_result() {
    let n = 300_000usize;
    let tile = 1 << 18;
    let adam = Adam { beta1: 0.9, beta2: 0.99, eps: 1e-6 };
    let rule = Rule::AdamW { adam, wd: 0.01 };
    let specs = vec![TensorSpec { name: "w".into(), numel: n, precision: ParamPrecision::FP32, row_len: None }];
    let layout = state_layout(&rule, &specs);
    let st = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), tile).unwrap();
    let mut opt = StreamedOptimizer::new(rule, specs, st, 1, tile).unwrap();
    let vl = tensor_layout(&[n], StateFormat::F32);
    let mut values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), tile).unwrap();
    let mut grads = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), tile).unwrap();
    let w0: Vec<f32> = (0..n).map(|i| ((i * 7 % 101) as f32 / 101.0) - 0.5).collect();
    let g0: Vec<f32> = (0..n).map(|i| (((i * 13) % 37) as f32 / 37.0 - 0.5) * 0.1).collect();
    for start in (0..n).step_by(tile) {
        let end = (start + tile).min(n);
        values.write_tile(0, 0, start, &w0[start..end], Rounding::NEAREST).unwrap();
        grads.write_tile(0, 0, start, &g0[start..end], Rounding::NEAREST).unwrap();
    }
    let mut params = HostParams { values, grads };
    for _ in 0..3 {
        opt.step(&mut params, 1e-2, None, 1.0).unwrap();
    }

    let (mut w, mut m, mut v) = (w0.clone(), vec![0f32; n], vec![0f32; n]);
    for t in 1..=3u32 {
        adam.update_slice(t, 1e-2, 0.01, 1.0, &mut w, &mut m, &mut v, &g0);
    }
    let mut got = vec![0f32; n];
    for start in (0..n).step_by(tile) {
        let end = (start + tile).min(n);
        params.values.read_tile(0, 0, start, &mut got[start..end]).unwrap();
    }
    assert_eq!(got, w);
}
