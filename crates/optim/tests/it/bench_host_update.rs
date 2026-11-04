// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in host-side optimiser updates that keep
// up with a GPU, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Throughput of the host streamed update (`StreamedOptimizer::update_tensor`) per state format, in million parameters
//! per second, on one tensor large enough to be memory bound. The host update of an off-device trainer is judged
//! against this number: a model of P parameters costs P / rate seconds per step.
//!
//! ```text
//! taskset -c 88-143 cargo test --release -p brain-optim --test bench_host_update -- --ignored --nocapture
//! ```

use std::time::Instant;

use optim::codec::Rounding;
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, StreamedOptimizer, TensorSpec};
use optim::Adam;
use paramstore::precision::{MasterPolicy, ParamPrecision, StateFormat};

fn rate(rule: Rule, state: StateFormat, master: MasterPolicy, n: usize) -> f64 {
    let tile = 1 << 23;
    let precision = ParamPrecision { state, master, ..ParamPrecision::FP32 };
    let specs = vec![TensorSpec { name: "w".into(), numel: n, precision, row_len: matches!(rule, Rule::Adafactor { .. }).then_some(1024) }];
    let layout = state_layout(&rule, &specs);
    let st = BlockStateStorage::new(layout.clone(), MemBacking::zeroed(layout.total_bytes()), tile * 2).unwrap();
    let mut opt = StreamedOptimizer::new(rule, specs, st, 1, tile).unwrap();
    let tile = opt.tile_elems();
    let vl = tensor_layout(&[n], StateFormat::F32);
    let mut values = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), tile).unwrap();
    let mut grads = BlockStateStorage::new(vl.clone(), MemBacking::zeroed(vl.total_bytes()), tile).unwrap();
    let ones = vec![0.01f32; tile];
    for start in (0..n).step_by(tile) {
        let len = tile.min(n - start);
        values.write_tile(0, 0, start, &ones[..len], Rounding::NEAREST).unwrap();
        grads.write_tile(0, 0, start, &ones[..len], Rounding::NEAREST).unwrap();
    }
    let mut params = HostParams { values, grads };
    opt.init_master(&mut params).unwrap();
    opt.step(&mut params, 1e-3, None, 1.0).unwrap();
    let t = Instant::now();
    let steps = 3;
    for _ in 0..steps {
        opt.step(&mut params, 1e-3, None, 1.0).unwrap();
    }
    (n * steps) as f64 / t.elapsed().as_secs_f64() / 1e6
}

#[test]
#[ignore]
fn host_update_throughput_by_state_format() {
    let n = 128 << 20;
    let adamw = Rule::AdamW { adam: Adam::default(), wd: 0.0 };
    for (label, rule, state, master) in [
        ("adamw f32 state", adamw, StateFormat::F32, MasterPolicy::None),
        ("adamw f32 state + f32 master", adamw, StateFormat::F32, MasterPolicy::F32),
        ("adamw bf16 state + f32 master", adamw, StateFormat::Bf16, MasterPolicy::F32),
        ("adamw int8 state + f32 master", adamw, StateFormat::Int8Block { block: 64 }, MasterPolicy::F32),
        ("adafactor + f32 master", Rule::adafactor(0.0), StateFormat::F32, MasterPolicy::F32),
    ] {
        println!("{label}: {:.0} Mparam/s", rate(rule, state, master, n));
    }
}
