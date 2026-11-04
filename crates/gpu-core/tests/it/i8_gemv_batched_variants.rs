// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for getting quantised decode onto a
// client's memory roofline. If your team needs expertise in GPU kernel
// selection and tensor-core usage then you can procure our services by sending
// an email to info@swedishembedded.com.

//! The int8 decode GEMV's own variants at a BATCHED row count (8): its
//! tensor-core block, its dp4a block, and the tensor-core block at two other
//! load-unroll depths, interleaved so every variant sees the same card
//! contention at the real Qwen3.8-27B projection shapes. `#[ignore]`d - a
//! measurement, not a gate (the gate is
//! `i8_gemv_native_bench::eight_rows_stream_the_weights_at_almost_the_one_row_rate`).
//!
//! Run on a CUDA box:
//!
//! ```text
//! cargo test --release -p brain-gpu-core --test i8_gemv_batched_variants -- --ignored --nocapture
//! ```
//!
//! What it settled (GH200, 2026-10-25): at eight rows the tensor-core block
//! streams 1.48-1.99 TB/s of weights, 1.5-1.6x the dp4a block's 1.0-1.26, and
//! load-unroll depth 4 beats both 2 and 8. So the batched step's weight rate is
//! not a missing tensor-core path, a per-row weight re-read, or an unroll
//! choice - it is the block's own streaming efficiency, which the profile puts
//! at 67% of a batched decode step.

use backend_api::DeviceBuffer;
use gpu_core::{Gpu, Step};

const KERNELS: &[(&str, &str)] = &[("matmul_i8_gemv", kernels::MATMUL_I8_GEMV)];
const FOOTPRINT_BYTES: u64 = 768 << 20;
const TRIALS: usize = 9;

struct Case<'a> {
    steps: Vec<Step>,
    bytes: u64,
    calls: usize,
    // Kept alive for the steps that name them: the kernel reads these buffers.
    _xq: DeviceBuffer,
    _sx: DeviceBuffer,
    _out: DeviceBuffer,
    _w: Vec<(DeviceBuffer, DeviceBuffer)>,
    gpu: &'a Gpu,
}

fn setup<'a>(gpu: &'a Gpu, id: backend_api::NativeId, m: u32, k: u32, n: u32) -> Case<'a> {
    let kg = k / 4;
    let wbytes = u64::from(n) * u64::from(kg) * 4 + u64::from(n) * u64::from(kg / 8) * 4;
    let copies = (FOOTPRINT_BYTES / wbytes).max(1) as usize;
    let xq = gpu.storage(u64::from(m * kg));
    let sx = gpu.storage(u64::from(m * kg / 8));
    let out = gpu.storage(u64::from(m * n));
    gpu.write(&xq, &vec![0x01020304u32; (m * kg) as usize]);
    gpu.write_f32(&sx, &vec![1.0; (m * kg / 8) as usize]);
    // Decode reads every weight from HBM once per token: cycle enough distinct
    // copies to exceed the L2 several times, or a small projection reports L2.
    let w: Vec<(DeviceBuffer, DeviceBuffer)> = (0..copies)
        .map(|i| {
            let wb = gpu.storage(u64::from(n) * u64::from(kg));
            let sb = gpu.storage(u64::from(n) * u64::from(kg / 8));
            if i == 0 {
                gpu.write(&wb, &vec![0x7f01fe03u32; (u64::from(n) * u64::from(kg)).min(1 << 22) as usize]);
                gpu.write_f32(&sb, &vec![0.01; (u64::from(n) * u64::from(kg / 8)).min(1 << 20) as usize]);
            }
            (wb, sb)
        })
        .collect();
    let calls = copies.max(16);
    let blocks = m.div_ceil(8) * n.div_ceil(16);
    let steps: Vec<_> = (0..calls)
        .map(|i| {
            let (wb, sb) = &w[i % copies];
            let bufs: [&DeviceBuffer; 5] = [&xq, wb, &sx, sb, &out];
            let offs = [(0, u64::from(m * kg)), (0, 0), (0, u64::from(m * kg / 8)), (0, 0), (0, u64::from(m * n))];
            gpu.step_native_sliced(id, &bufs, &offs, &[m, kg, n], blocks).expect("native step")
        })
        .collect();
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    Case { steps, bytes: wbytes, calls, _xq: xq, _sx: sx, _out: out, _w: w, gpu }
}

impl Case<'_> {
    fn timed(&self) -> f64 {
        self.gpu.reset_kernel_times();
        self.gpu.submit(&[], &self.steps);
        self.gpu.poll_wait();
        let rows = self.gpu.kernel_times().expect("device timing");
        rows.iter().map(|(_, ms, _)| ms).sum::<f64>() / self.calls as f64
    }
}

#[test]
#[ignore]
fn tensor_core_block_against_dp4a_block_and_unroll_depths() {
    let gpu = Gpu::new(KERNELS);
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    let src = kernels_cuda::get("matmul_i8_gemv").expect("registered").src;
    let bindings: &'static [backend_api::BindKind] = &[
        backend_api::BindKind::Uniform,
        backend_api::BindKind::StorageRead,
        backend_api::BindKind::StorageRead,
        backend_api::BindKind::StorageRead,
        backend_api::BindKind::StorageRead,
        backend_api::BindKind::StorageReadWrite,
    ];
    let register = |name: &'static str, text: &'static str, block_dim: u32| {
        gpu.register_native(&backend_api::NativeSpec::Cuda {
            name,
            src: text,
            entry: "brain_matmul_i8_gemv",
            block_dim,
            bindings,
            shared_bytes: 0,
            launch: backend_api::CudaLaunch::NONE,
        })
        .expect("the gemv kernel registers")
    };
    let default = register("native:gemv_variant_default", Box::leak(src.to_string().into_boxed_str()), 128);
    let no_mma: &'static str = Box::leak(format!("#define BRAIN_I8G_NO_MMA\n{src}").into_boxed_str());
    let dp4a = register("native:gemv_variant_dp4a", no_mma, 128);
    let u8: &'static str = Box::leak(format!("#define BRAIN_I8G_MMA_UNROLL 8\n{src}").into_boxed_str());
    let unroll8 = register("native:gemv_variant_u8", u8, 128);
    let u2: &'static str = Box::leak(format!("#define BRAIN_I8G_MMA_UNROLL 2\n{src}").into_boxed_str());
    let unroll2 = register("native:gemv_variant_u2", u2, 128);

    println!("{:<26} {:>3} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}", "shape", "m", "default us", "u8 us", "u2 us", "dp4a us", "default GB/s", "dp4a GB/s");
    for (label, k, n) in [("gdn qkv 5120x10240", 5120u32, 10240u32), ("ffn gate/up 5120x17408", 5120, 17408), ("ffn down 17408x5120", 17408, 5120), ("attn out 6144x5120", 6144, 5120)] {
        for m in [8u32] {
            let a = setup(&gpu, default, m, k, n);
            let b = setup(&gpu, dp4a, m, k, n);
            let c = setup(&gpu, unroll8, m, k, n);
            let d = setup(&gpu, unroll2, m, k, n);
            let (mut dflt, mut dp, mut e8, mut e2) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
            for _ in 0..TRIALS {
                dflt = dflt.min(a.timed());
                dp = dp.min(b.timed());
                e8 = e8.min(c.timed());
                e2 = e2.min(d.timed());
            }
            println!(
                "{label:<26} {m:>3} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>10.0}",
                dflt * 1e3,
                e8 * 1e3,
                e2 * 1e3,
                dp * 1e3,
                a.bytes as f64 / (dflt * 1e-3) / 1e9,
                b.bytes as f64 / (dp * 1e-3) / 1e9
            );
        }
    }
}
