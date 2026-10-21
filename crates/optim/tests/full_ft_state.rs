// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in bf16, FP8 and quantization-aware full
// fine-tuning then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Gates of the generic full-fine-tuning state: layer-streamed updates with
//! deferred publication, values stored in any registered encoding (bf16, FP8,
//! int8, NF4) with and without a master copy, and streamed export of the
//! masters to a chosen format.

use checkpoint::packed::{CodecRegistry, MemorySink, QuantEncoding, TensorDescriptor};
use optim::codec::Rounding;
use optim::deferred::DeferredPublish;
use optim::encoded::{aligned_tile, export_values, EncodedParams, EncodedValues};
use optim::rule::Rule;
use optim::storage::{BlockStateStorage, MemBacking, OptimizerStateStorage};
use optim::streamed::{state_layout, tensor_layout, HostParams, MasterView, ParamTiles, StreamedOptimizer, TensorSpec};
use optim::Adam;
use paramstore::precision::{GradDtype, MasterPolicy, ParamPrecision, StateFormat, ValueDtype};

type Mem = BlockStateStorage<MemBacking>;

fn f32_store(numels: &[usize], max_tile: usize) -> Mem {
    let l = tensor_layout(numels, StateFormat::F32);
    BlockStateStorage::new(l.clone(), MemBacking::zeroed(l.total_bytes()), max_tile).unwrap()
}

fn read_all(p: &mut dyn ParamTiles, ti: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for (c, ch) in out.chunks_mut(256).enumerate() {
        p.read_value(ti, c * 256, ch).unwrap();
    }
    out
}

fn write_all(s: &mut Mem, ti: usize, vals: &[f32]) {
    for (c, ch) in vals.chunks(256).enumerate() {
        s.write_tile(ti, 0, c * 256, ch, Rounding::NEAREST).unwrap();
    }
}

fn optimizer(specs: Vec<TensorSpec>, tile: usize) -> StreamedOptimizer<Mem> {
    let rule = Rule::AdamW { adam: Adam::default(), wd: 0.0 };
    let l = state_layout(&rule, &specs);
    let st = BlockStateStorage::new(l.clone(), MemBacking::zeroed(l.total_bytes()), 4096).unwrap();
    StreamedOptimizer::new(rule, specs, st, 5, tile).unwrap()
}

fn fp32_spec(name: &str, n: usize) -> TensorSpec {
    TensorSpec { name: name.into(), numel: n, precision: ParamPrecision::FP32, row_len: None }
}

/// Coupled "model": the gradient of tensor i depends on the VALUES of the
/// other tensors, so a gradient taken after a neighbour was updated differs
/// from one taken against the step's own version.
fn coupled_grad(ti: usize, vals: &[Vec<f32>]) -> Vec<f32> {
    let other = &vals[(ti + 1) % vals.len()];
    vals[ti].iter().zip(other).map(|(w, o)| 0.5 * (w - 0.3) + 0.2 * o).collect()
}

/// Gate: layer-streamed updates with deferred publication give exactly the
/// whole-model step (every gradient against one parameter version); updating
/// in place as each gradient completes does not, so the deferral is load-bearing.
#[test]
fn deferred_publication_equals_the_whole_model_step() {
    let numels = [300usize, 513, 128];
    let init: Vec<Vec<f32>> = numels.iter().map(|&n| (0..n).map(|i| ((i * 7 % 23) as f32) * 0.05 - 0.5).collect()).collect();
    let specs: Vec<TensorSpec> = numels.iter().enumerate().map(|(i, &n)| fp32_spec(&format!("t{i}"), n)).collect();
    let fresh = |init: &[Vec<f32>]| {
        let mut p = HostParams { values: f32_store(&numels, 256), grads: f32_store(&numels, 256) };
        for (i, v) in init.iter().enumerate() {
            write_all(&mut p.values, i, v);
        }
        p
    };
    let set_grads = |p: &mut HostParams<MemBacking, MemBacking>, vals: &[Vec<f32>]| {
        for ti in 0..numels.len() {
            write_all(&mut p.grads, ti, &coupled_grad(ti, vals));
        }
    };

    // reference: all gradients from version v, one whole-model step
    let mut pr = fresh(&init);
    let mut opt_r = optimizer(specs.clone(), 128);
    set_grads(&mut pr, &init);
    opt_r.step(&mut pr, 0.05, None, 1.0).unwrap();
    let reference: Vec<Vec<f32>> = (0..3).map(|i| read_all(&mut pr, i, numels[i])).collect();

    // streamed with deferred publication
    let mut live = fresh(&init);
    let mut opt = optimizer(specs.clone(), 128);
    {
        let mut deferred = DeferredPublish::new(&mut live, f32_store(&numels, 256), numels.to_vec(), 128);
        opt.begin_step();
        for ti in 0..3 {
            let vals: Vec<Vec<f32>> = (0..3).map(|i| read_all(&mut deferred, i, numels[i])).collect();
            // the layer's gradient lands in the live gradient store
            let g = coupled_grad(ti, &vals);
            for (c, ch) in g.chunks(128).enumerate() {
                deferred.write_grad(ti, c * 128, ch).unwrap();
            }
            opt.update_tensor(&mut deferred, ti, 0.05, 1.0).unwrap();
        }
        assert_eq!(deferred.publish().unwrap(), 3);
    }
    for i in 0..3 {
        assert_eq!(read_all(&mut live, i, numels[i]), reference[i], "tensor {i}");
    }

    // negative control: publishing each layer immediately changes the result
    let mut inplace = fresh(&init);
    let mut opt2 = optimizer(specs, 128);
    opt2.begin_step();
    for ti in 0..3 {
        let vals: Vec<Vec<f32>> = (0..3).map(|i| read_all(&mut inplace, i, numels[i])).collect();
        write_all(&mut inplace.grads, ti, &coupled_grad(ti, &vals));
        opt2.update_tensor(&mut inplace, ti, 0.05, 1.0).unwrap();
    }
    // tensor 2's gradient reads tensor 0, which an in-place loop has already updated
    assert!(read_all(&mut inplace, 2, numels[2]) != reference[2], "in-place updates must diverge from the single-version step");
}

/// Quadratic task with per-coordinate curvature.
struct Task {
    d: Vec<f32>,
    t: Vec<f32>,
}

impl Task {
    fn new(n: usize) -> Task {
        Task {
            d: (0..n).map(|i| 0.2 + 2.0 * ((i * 37 % n) as f32 / n as f32)).collect(),
            t: (0..n).map(|i| ((i * 13 % 29) as f32 / 29.0 - 0.5) * 2.0).collect(),
        }
    }
    fn loss(&self, w: &[f32]) -> f64 {
        w.iter().zip(&self.d).zip(&self.t).map(|((w, d), t)| 0.5 * (*d as f64) * ((w - t) as f64).powi(2)).sum()
    }
    fn grad(&self, w: &[f32]) -> Vec<f32> {
        w.iter().enumerate().map(|(i, w)| self.d[i] * (w - self.t[i])).collect()
    }
}

const N: usize = 64 * 64;

/// Train `steps` of AdamW with values in `enc`; the gradient is taken at the
/// STORED (quantized) value and applied to the master when there is one
/// (the straight-through estimator). Returns the final loss of the stored value.
fn train_encoded(enc: QuantEncoding, master: MasterPolicy, steps: usize, registry: &CodecRegistry) -> f64 {
    let task = Task::new(N);
    let descs = vec![TensorDescriptor::new("w", vec![64, 64], enc)];
    let tile = aligned_tile(&descs, 512);
    let need = EncodedValues::<MemBacking>::required_bytes(&descs);
    let values = EncodedValues::new(descs, registry.clone(), MemBacking::zeroed(need), tile).unwrap();
    let gl = tensor_layout(&[N], StateFormat::F32);
    let grads = BlockStateStorage::new(gl.clone(), MemBacking::zeroed(gl.total_bytes()), tile).unwrap();
    let mut p = EncodedParams { values, grads };
    let precision = ParamPrecision { value: ValueDtype::Bf16, grad: GradDtype::F32, state: StateFormat::F32, master, stochastic_rounding: false };
    let specs = vec![TensorSpec { name: "w".into(), numel: N, precision, row_len: None }];
    let rule = Rule::AdamW { adam: Adam::default(), wd: 0.0 };
    let l = state_layout(&rule, &specs);
    let st = BlockStateStorage::new(l.clone(), MemBacking::zeroed(l.total_bytes()), tile).unwrap();
    let mut opt = StreamedOptimizer::new(rule, specs, st, 1, tile).unwrap();
    assert_eq!(opt.tile_elems() % tile, 0);
    let read = |p: &mut EncodedParams<MemBacking, MemBacking>| {
        let mut w = vec![0f32; N];
        for (c, ch) in w.chunks_mut(tile).enumerate() {
            p.read_value(0, c * tile, ch).unwrap();
        }
        w
    };
    // start from a quantization of 0.8 everywhere
    for c in 0..N.div_ceil(tile) {
        let n = tile.min(N - c * tile);
        p.values.write(0, c * tile, &vec![0.8; n]).unwrap();
    }
    opt.init_master(&mut p).unwrap();
    for _ in 0..steps {
        let w = read(&mut p);
        let g = task.grad(&w);
        for (c, ch) in g.chunks(tile).enumerate() {
            p.grads.write_tile(0, 0, c * tile, ch, Rounding::NEAREST).unwrap();
        }
        opt.step(&mut p, 0.02, None, 1.0).unwrap();
    }
    task.loss(&read(&mut p))
}

fn registry() -> CodecRegistry {
    CodecRegistry::reference()
}

/// Gate: with an fp32 master, values stored in bf16, block-scaled FP8, int8 and
/// NF4 all train the task to within a stated factor of the fp32 baseline (the
/// quantization-aware / straight-through case); without a master the
/// low-precision value cannot hold the small late updates and ends worse.
#[test]
fn encoded_values_with_a_master_train_like_fp32_and_without_one_do_not() {
    let r = registry();
    let base = train_encoded(QuantEncoding::f32(), MasterPolicy::None, 150, &r);
    println!("fp32 values: final loss {base:.5}");
    let start = Task::new(N).loss(&vec![0.8; N]);
    for (name, enc, bound, master_needed) in [
        ("bf16", QuantEncoding::bf16(), 1.5, false),
        ("fp8 e4m3 tile 32", QuantEncoding::fp8_e4m3_tiled(32), 4.0, true),
        ("int8 block 32", QuantEncoding::int8_blocked(32), 2.0, false),
        ("nf4 block 32", QuantEncoding::nf4_blocked(32), 40.0, true),
    ] {
        let with = train_encoded(enc.clone(), MasterPolicy::F32, 150, &r);
        let without = train_encoded(enc, MasterPolicy::None, 150, &r);
        println!("{name}: loss with master {with:.5}, without {without:.5} (start {start:.1})");
        assert!(with < 0.05 * start, "{name} with a master must train (loss {with})");
        assert!(with <= bound * base.max(1e-3), "{name}: loss {with} vs fp32 {base}, bound x{bound}");
        if master_needed {
            assert!(without > 10.0 * with, "{name}: without a master the loss {without} should stay far above {with}");
        }
    }
}

/// Gate: the export streams masters into another encoding through a TensorSink
/// with staging independent of the tensor size, and what arrives decodes to
/// the master within the target format's error.
#[test]
fn export_streams_masters_into_a_chosen_encoding_within_a_fixed_staging() {
    let r = registry();
    let measure = |rows: usize| {
        let n = rows * 64;
        let precision = ParamPrecision { master: MasterPolicy::F32, ..ParamPrecision::FP32 };
        let specs = vec![TensorSpec { name: "w".into(), numel: n, precision, row_len: None }];
        let mut opt = optimizer(specs, 256);
        let master: Vec<f32> = (0..n).map(|i| ((i * 29 % 101) as f32 - 50.0) * 0.01).collect();
        for (c, ch) in master.chunks(256).enumerate() {
            // slot 2 of an AdamW tensor with a master is the master (after m and v)
            opt.state_mut().write_tile(0, 2, c * 256, ch, Rounding::NEAREST).unwrap();
        }
        let mut values = HostParams { values: f32_store(&[n], 256), grads: f32_store(&[n], 256) };
        let dst = vec![TensorDescriptor::new("w", vec![rows as u64, 64], QuantEncoding::int8_blocked(32))];
        let mut view = MasterView { opt: &mut opt, values: &mut values };
        let mut sink = MemorySink::default();
        let rep = export_values(&mut view, &dst, &r, &mut sink, 256).unwrap();
        // decode what the sink holds and compare with the master
        let mut out = Vec::new();
        let tile = checkpoint::packed::Tile { elems: 0..n as u64, payload: &sink.payload, scales: sink.scales.iter().map(|s| s.as_slice()).collect() };
        r.decoder(&dst[0].encoding).unwrap().decode_tile(&dst[0], &tile, &mut out).unwrap();
        let worst = out.iter().zip(&master).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst <= 0.5 * 0.5 / 127.0 * 1.01 + 1e-6, "int8 export error {worst}");
        assert_eq!(rep.payload_bytes, n as u64);
        rep.staging_bytes
    };
    let (small, large) = (measure(16), measure(1024));
    println!("export staging: {small} B for 1024 elements, {large} B for 65536 elements");
    assert_eq!(small, large, "staging must not grow with the tensor");
}

/// The FP8 encoder reproduces values within one E4M3 step of their block
/// scale, never saturates a block's maximum, and agrees with the registry's decoder.
#[test]
fn fp8_encoder_round_trips_within_one_code_and_keeps_the_block_maximum() {
    let r = registry();
    let desc = TensorDescriptor::new("w", vec![32, 32], QuantEncoding::fp8_e4m3_tiled(32));
    let vals: Vec<f32> = (0..1024).map(|i| ((i * 37 % 211) as f32 - 105.0) * 0.013).collect();
    let mut buf = checkpoint::packed::TileBuf::default();
    r.encoder(&desc.encoding).unwrap().encode_tile(&desc, &(0..1024), &vals, &checkpoint::packed::codec::EncodeCtx::default(), &mut buf).unwrap();
    let tile = checkpoint::packed::Tile { elems: 0..1024, payload: &buf.payload, scales: buf.scales.iter().map(|s| s.as_slice()).collect() };
    let mut out = Vec::new();
    r.decoder(&desc.encoding).unwrap().decode_tile(&desc, &tile, &mut out).unwrap();
    let max = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
    for (a, b) in vals.iter().zip(&out) {
        assert!((a - b).abs() <= a.abs() * 0.0625 + max * 1e-3, "{a} vs {b}");
    }
    let omax = out.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!((omax - max).abs() <= max * 0.0625, "block maximum {max} came back as {omax}");
}
