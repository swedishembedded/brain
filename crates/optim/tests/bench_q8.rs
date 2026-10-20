// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in profiling optimiser kernels against
// the memory roofline then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Steady-state step time of the fp32-moment AdamW against the fused
//! int8-moment AdamW (`adamw_q8`), on one large tensor set, with the bytes
//! each moves per element so the result can be read against the device's
//! memory bandwidth (`brain roofline gpu`).
//!
//! ```text
//! BRAIN_BACKEND=cuda BRAIN_DEVICE=gpu0 cargo test --release -p brain-optim --test bench_q8 -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::time::Instant;

use gpu_core::Gpu;
use optim::{Adam, Optim};
use paramstore::precision::{ParamPrecision, PrecisionPolicy, StateFormat};
use paramstore::{ParamStore, Role};

/// Elements per tensor and tensor count: 64 MiElem x 4 = 256 Mi parameters.
const TENSOR: usize = 64 << 20;
const TENSORS: usize = 4;
const STEPS: u32 = 20;

fn run(gpu: &Gpu, policy: &PrecisionPolicy) -> f64 {
    let names: Vec<String> = (0..TENSORS).map(|i| format!("w{i}")).collect();
    let init: HashMap<String, Vec<f32>> = names.iter().map(|n| (n.clone(), vec![0.01f32; TENSOR])).collect();
    let roles: Vec<(String, usize, Role)> = names.iter().map(|n| (n.clone(), TENSOR, Role::Trainable)).collect();
    let ps = ParamStore::new_with_policy_src(gpu, roles, &init, policy);
    let g: Vec<f32> = (0..TENSOR).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 0.01).collect();
    for n in &names {
        gpu.write_f32(ps.g(n), &g);
    }
    let opt = Optim::new(0, 1, 2, 3, 4);
    opt.step(gpu, &ps, 1, 1e-3, 0.01, Adam::default(), None, 1.0);
    gpu.read(ps.w("w0"), 1); // drain
    let t0 = Instant::now();
    for t in 2..=STEPS + 1 {
        opt.step(gpu, &ps, t, 1e-3, 0.01, Adam::default(), None, 1.0);
    }
    gpu.read(ps.w("w0"), 1);
    t0.elapsed().as_secs_f64() / STEPS as f64
}

#[test]
#[ignore]
fn bench_fp32_against_int8_moments() {
    static KERNELS: &[(&str, &str)] = &[
        ("adamw", kernels::ADAMW),
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("grad_scale", kernels::GRAD_SCALE),
        ("clip_coef", kernels::CLIP_COEF),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ("adamw_q8", kernels::ADAMW_Q8),
        ("adamw_q8_wg", kernels::ADAMW_Q8_WG),
    ];
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let params = (TENSOR * TENSORS) as f64;
    let int8 = ParamPrecision { state: StateFormat::Int8Block { block: 64 }, stochastic_rounding: true, ..ParamPrecision::FP32 };
    for (name, policy, bytes_per_elem) in [
        ("fp32 moments", PrecisionPolicy::uniform(ParamPrecision::FP32), 28.0),
        ("int8 moments", PrecisionPolicy::uniform(int8), 16.0 + 4.0 / 32.0),
    ] {
        let s = run(&gpu, &policy);
        println!(
            "{name}: {:.2} ms/step for {:.0} Mi params, {:.1} GB/s effective ({bytes_per_elem:.2} B/elem moved)",
            s * 1e3,
            params / (1 << 20) as f64,
            params * bytes_per_elem / s / 1e9
        );
    }
}
