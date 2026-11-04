// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed cost of the stored-format decode GEMVs at the Qwen3.6-35B-A3B shapes, in GB/s of weight bytes against
//! the card's HBM bandwidth.
//!
//! Swedish Embedded AB implements measured throughput contracts for quantized decode kernels. If your team needs
//! expertise in knowing how close a 4-bit or 8-bit GEMV runs to the bandwidth ceiling of your GPU, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a gate (the gate is `qgemv_native.rs`):
//!
//! ```text
//! cargo test --release -p brain-gpu-core --test bench_qgemv -- --ignored --nocapture
//! ```
//!
//! Time is the kernel's own, from the backend's device events, never the host clock (the card is shared). Each shape
//! is timed with the weights cold (a buffer larger than L2 is rotated through), as a decode token meets them.

use gpu_core::{Dispatch, Gpu};
use std::sync::OnceLock;

fn kernel_list() -> &'static [(&'static str, &'static str)] {
    static LIST: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    LIST.get_or_init(|| {
        let f8 = kernels::template::interned("matmul_fp8_gemv", kernels::MATMUL_FP8_GEMV, &[("MAXM", 1)]).expect("MAXM");
        let f4 = kernels::template::interned("matmul_nvfp4_gemv", kernels::MATMUL_NVFP4_GEMV, &[("MAXM", 1)]).expect("MAXM");
        vec![f8, f4, ("moe_nvfp4_gemv_gather", kernels::MOE_NVFP4_GEMV_GATHER)]
    })
}

const TRIALS: usize = 9;

fn timed(gpu: &Gpu, step: &gpu_core::Step, what: &str) -> f64 {
    gpu.submit(&[], std::slice::from_ref(step));
    gpu.poll_wait();
    let mut per_call = Vec::new();
    for _ in 0..TRIALS {
        gpu.reset_kernel_times();
        gpu.submit(&[], std::slice::from_ref(step));
        gpu.poll_wait();
        let rows = gpu.kernel_times().expect("device timing");
        let (_, ms, calls) = rows.iter().find(|(name, _, _)| name.contains(what)).unwrap_or_else(|| panic!("the kernel {what} ran: {rows:?}"));
        per_call.push(ms / *calls as f64);
    }
    per_call.sort_by(f64::total_cmp);
    per_call[0] * 1e3
}

#[test]
#[ignore]
fn decode_gemv_cost_at_the_35b_a3b_shapes() {
    let gpu = Gpu::new(kernel_list());
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    let x = gpu.storage_init("x", &vec![0.5f32; 4096]);
    println!("{:<40} {:>10} {:>10}", "shape", "best us", "GB/s");
    // FP8 projections: GDN in_proj_qkv, in_proj_z, out_proj, GQA q, k, o.
    for (n, k) in [(8192u32, 2048u32), (4096, 2048), (2048, 4096), (8192, 2048), (512, 2048), (2048, 4096)] {
        let w = gpu.storage(u64::from(n) * u64::from(k) / 4);
        let grid = u64::from(n.div_ceil(128) * k.div_ceil(128));
        let s = gpu.storage_init("s", &vec![0.01f32; grid as usize]);
        let y = gpu.storage(u64::from(n));
        let params = [1, k, n, 0, 128, k.div_ceil(128), 0, 1];
        let step = gpu.dispatch(0, &[&x, &w, &s, &y], &params, Dispatch::Workgroups(n.div_ceil(16)));
        let us = timed(&gpu, &step, "matmul_fp8_gemv");
        let bytes = u64::from(n) * u64::from(k) + grid * 4;
        println!("fp8 [{n}, {k}]{:<27} {:>10.1} {:>10.0}", "", us, bytes as f64 / (us * 1e-6) / 1e9);
    }
    // The NVFP4 LM head and a shared-expert-sized projection.
    for (n, k) in [(248_320u32, 2048u32), (512, 2048)] {
        let w = gpu.storage(u64::from(n) * u64::from(k) / 8);
        let s = gpu.storage(u64::from(n) * u64::from(k) / 64);
        let y = gpu.storage(u64::from(n));
        let params = [1, k, n, 0, 0, 1.0f32.to_bits(), 1, 0];
        let step = gpu.dispatch(1, &[&x, &w, &s, &y], &params, Dispatch::Workgroups(n.div_ceil(16)));
        let us = timed(&gpu, &step, "matmul_nvfp4_gemv");
        let bytes = u64::from(n) * u64::from(k) / 2 + u64::from(n) * u64::from(k) / 16;
        println!("nvfp4 [{n}, {k}]{:<24} {:>10.1} {:>10.0}", "", us, bytes as f64 / (us * 1e-6) / 1e9);
    }
    // The expert gathers of a decode token: 9 slots (8 routed + shared) of gate/up [512, 2048] and down [2048, 512].
    for (n, k, xdiv) in [(512u32, 2048u32, 9u32), (2048, 512, 1)] {
        let slots = 9u32;
        let experts = 257u32;
        let w = gpu.storage(u64::from(experts) * u64::from(n) * u64::from(k) / 8);
        let s = gpu.storage(u64::from(experts) * u64::from(n) * u64::from(k) / 64);
        let gl = gpu.storage_init("g", &vec![1.0f32; experts as usize]);
        let ids = gpu.storage(u64::from(slots));
        gpu.write(&ids, &[3, 17, 29, 64, 101, 150, 201, 250, 256]);
        let out = gpu.storage(u64::from(slots) * u64::from(n));
        let params = [slots, k, n, xdiv];
        let step = gpu.dispatch(2, &[&x, &ids, &w, &s, &gl, &out], &params, Dispatch::Workgroups(slots * n.div_ceil(16)));
        let us = timed(&gpu, &step, "moe_nvfp4_gemv_gather");
        let bytes = u64::from(slots) * u64::from(n) * u64::from(k) * 9 / 16;
        println!("gather 9 slots [{n}, {k}]{:<15} {:>10.1} {:>10.0}", "", us, bytes as f64 / (us * 1e-6) / 1e9);
    }
}
