// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving mixed FP8 and NVFP4 releases on the GPU its clients already own.
// If your team needs expertise in measuring quantized GEMV and tensor-core GEMM kernels against the memory and compute
// ceilings of a card then you can procure our services by sending an email to info@swedishembedded.com.

//! Device-timed cost of a stored FP8 and a stored NVFP4 weight against `rows` activation rows at the dense 27B shapes
//! (gate/up, down, the LM head). `#[ignore]`d: a measurement, not a gate (the gates are `qops_gemv_small_m.rs` and
//! `gpu-core/tests/qgemv_native.rs`).
//!
//! ```text
//! BRAIN_DEVICE=gpu1 BRAIN_BACKEND=cuda cargo test --release -p brain-model --test qops_nvfp4_dense_shapes -- --ignored --nocapture
//! ```
//!
//! Time is the sum of the device events of every kernel one product launches (the activation cast, the weight decode
//! into a bf16 tile and the vendor-library product count as parts of the NVFP4 product), never the host clock; the
//! weights are rotated through `COPIES` distinct buffers so each product meets them cold. The achieved rate is the
//! weight bytes over the time (the figure of merit of a decode step) and, for many rows, the tensor-core rate.

use checkpoint::packed::{QuantEncoding, TensorDescriptor, TensorSink, Tile};
use data::rng::Lcg;
use gpu_core::Gpu;
use model::ops::{training_kernel_list, DeviceWeightSink, Ops, Weight};

const COPIES: usize = 3;
const TRIALS: usize = 5;

/// A weight of encoding `enc` over random valid bytes.
fn random_weight(ops: &Ops, enc: QuantEncoding, n: usize, k: usize, rng: &mut Lcg) -> Weight {
    let desc = TensorDescriptor::new("w", vec![n as u64, k as u64], enc.clone());
    // E4M3 payloads below 0x40 are never NaN; E2M1 codes are all values.
    let fp8 = matches!(enc.element, checkpoint::packed::Element::Float(checkpoint::packed::FloatFormat::E4M3Fn));
    let payload: Vec<u8> = (0..desc.payload_bytes()).map(|_| if fp8 { (rng.next_u32() % 0x40) as u8 } else { rng.next_u32() as u8 }).collect();
    let scales: Vec<Vec<u8>> = enc
        .scales
        .iter()
        .enumerate()
        .map(|(l, s)| match s.format {
            checkpoint::packed::FloatFormat::F32 => 1.5f32.to_le_bytes().to_vec(),
            checkpoint::packed::FloatFormat::Bf16 => (0..desc.scale_numel(l)).flat_map(|_| half::bf16::from_f32(0.01).to_le_bytes()).collect(),
            _ => (0..desc.scale_bytes(l)).map(|_| 0x30 + (rng.next_u32() % 0x20) as u8).collect(),
        })
        .collect();
    let tile = Tile { elems: 0..desc.numel(), payload: &payload, scales: scales.iter().map(Vec::as_slice).collect() };
    let mut sink = DeviceWeightSink::new(ops);
    sink.begin(&desc).unwrap();
    sink.write_tile(&desc, &tile).unwrap();
    sink.finish(&desc).unwrap();
    sink.take_weight().unwrap()
}

/// Device milliseconds per kernel name of one product of `w` against `m` rows, best of `TRIALS` by total.
fn product_ms(ops: &Ops, weights: &[Weight], x: &gpu_core::DeviceBuffer, y: &gpu_core::DeviceBuffer, m: u32) -> Vec<(String, f64)> {
    let g = ops.gpu();
    let (n, k) = (weights[0].n(), weights[0].k());
    let mut best: Option<Vec<(String, f64)>> = None;
    for t in 0..TRIALS + 1 {
        let w = &weights[t % weights.len()];
        let act = ops.act_f32(x, 0, m, k);
        let mut steps = Vec::new();
        ops.matmul_rows(&mut steps, w, (0, n), &act, y, 0);
        g.reset_kernel_times();
        g.submit(&[], &steps);
        g.poll_wait();
        if t == 0 {
            continue; // compilation and library handles
        }
        let rows: Vec<(String, f64)> = g.kernel_times().expect("device timing").into_iter().map(|(name, ms, _)| (name, ms)).collect();
        let total = |r: &[(String, f64)]| r.iter().map(|(_, ms)| ms).sum::<f64>();
        if best.as_ref().is_none_or(|b| total(&rows) < total(b)) {
            best = Some(rows);
        }
    }
    best.unwrap()
}

#[test]
#[ignore = "GPU measurement"]
fn fp8_and_nvfp4_products_at_the_dense_27b_shapes() {
    let ops = Ops::new(Gpu::new(training_kernel_list())).expect("Ops::new");
    let g = ops.gpu();
    if g.kind() != "cuda" {
        return brain_testutil::skip_unavailable("device event timing is the CUDA backend's");
    }
    assert!(g.set_kernel_timing(true));
    let mut rng = Lcg::random();
    println!("{:<18} {:<7} {:>5} {:>10} {:>9} {:>9}  kernels (ms)", "shape", "tier", "rows", "total us", "GB/s", "TFLOPS");
    for (label, n, k, rows) in [
        ("gate/up 17408x5120", 17408usize, 5120usize, vec![1u32, 2, 3, 4, 8, 16, 64, 256, 2048, 8192]),
        ("down 5120x17408", 5120, 17408, vec![1, 2, 3, 4, 8, 16, 64, 256, 2048, 8192]),
        ("lm_head 248320x5120", 248_320, 5120, vec![1, 2, 3, 4, 8, 16, 64]),
    ] {
        let mut fp8_enc = QuantEncoding::fp8_e4m3_tiled(128);
        fp8_enc.scales[0].format = checkpoint::packed::FloatFormat::Bf16;
        let tiers = [("fp8", fp8_enc, (n * k) as f64 + (n.div_ceil(128) * k.div_ceil(128) * 4) as f64), ("nvfp4", QuantEncoding::nvfp4(), (n * k) as f64 / 2.0 + (n * k) as f64 / 16.0)];
        for (tier, enc, bytes) in tiers {
            let weights: Vec<Weight> = (0..COPIES).map(|_| random_weight(&ops, enc.clone(), n, k, &mut rng)).collect();
            for &m in &rows {
                let x = g.storage_init("x", &rng.vec_scaled(m as usize * k, 1.0));
                let y = g.storage(m as u64 * n as u64);
                let kernels = product_ms(&ops, &weights, &x, &y, m);
                let total_ms: f64 = kernels.iter().map(|(_, ms)| ms).sum();
                let flops = 2.0 * m as f64 * n as f64 * k as f64;
                let parts: Vec<String> = kernels.iter().map(|(name, ms)| format!("{}={ms:.3}", name.chars().take(34).collect::<String>())).collect();
                println!("{label:<18} {tier:<7} {m:>5} {:>10.1} {:>9.0} {:>9.1}  {}", total_ms * 1e3, bytes / (total_ms * 1e-3) / 1e9, flops / (total_ms * 1e-3) / 1e12, parts.join(" "));
            }
        }
    }
}
