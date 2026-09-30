// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of large language models on
// consumer GPUs for its clients. If your team needs expertise in GPU
// kernels for mixed-precision training then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The register-tiled input-gradient kernel reads a weight (fp32, or bf16
//! packed two to a word) the way the naive one does: at shapes that are not multiples of its 128x128 tile, with
//! both the overwriting and the accumulating write, it gives the naive
//! kernel's answer.

use data::rng::Lcg;
use gpu_core::{Dispatch, Gpu};
use model::half::pack_bf16;

enum Weight {
    F32(Vec<f32>),
    Bf16(Vec<u32>),
}

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn dx(gpu: &Gpu, kernel: &str, dispatch: impl Fn(u32, u32) -> Dispatch, weight: &Weight, dy: &[f32], (m, k, n): (u32, u32, u32), prior: &[f32], accumulate: u32) -> Vec<f32> {
    let kind = gpu.kernel_index(kernel).unwrap_or_else(|| panic!("{kernel} is not registered"));
    let (dy_b, dx_b) = (gpu.storage(dy.len() as u64), gpu.storage((m * k) as u64));
    let w_b = match weight {
        Weight::F32(w) => {
            let b = gpu.storage(w.len() as u64);
            gpu.write_f32(&b, w);
            b
        }
        Weight::Bf16(packed) => {
            let b = gpu.storage(packed.len() as u64);
            gpu.write(&b, packed);
            b
        }
    };
    gpu.write_f32(&dy_b, dy);
    gpu.write_f32(&dx_b, prior);
    gpu.submit(&[], &[gpu.dispatch(kind, &[&dy_b, &w_b, &dx_b], &[m, k, n, accumulate], dispatch(m, k))]);
    gpu.read(&dx_b, (m * k) as usize)
}

#[test]
fn the_register_tiled_input_gradient_matches_the_naive_one() {
    if gpu_disabled() {
        return;
    }
    let gpu = Gpu::new(qwen3::model::pipelines());
    let mut rng = Lcg::new(17);
    for (m, k, n) in [(200u32, 384u32, 320u32), (130, 130, 70), (1, 256, 256), (257, 64, 513)] {
        let w: Vec<f32> = (0..n * k).map(|_| rng.signed()).collect();
        let dy: Vec<f32> = (0..m * n).map(|_| rng.signed()).collect();
        let prior: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.25).collect();
        for (tier, weight, naive, reg) in [
            ("bf16", Weight::Bf16(pack_bf16(&w)), "matmul_dx#w=bf16", "matmul_dx_reg#w=bf16"),
            ("f32", Weight::F32(w.clone()), "matmul_dx", "matmul_dx_reg"),
        ] {
            for accumulate in [0u32, 1] {
                let want = dx(&gpu, naive, |m, k| Dispatch::Threads(m * k), &weight, &dy, (m, k, n), &prior, accumulate);
                let got = dx(&gpu, reg, |m, k| Dispatch::Workgroups(m.div_ceil(128) * k.div_ceil(128)), &weight, &dy, (m, k, n), &prior, accumulate);
                let scale = want.iter().fold(0.0f32, |s, x| s.max(x.abs())).max(1e-6);
                let worst = got.iter().zip(&want).fold(0.0f32, |s, (g, w)| s.max((g - w).abs() / scale));
                assert!(worst < 1e-4, "{tier} m={m} k={k} n={n} accumulate={accumulate}: relative error {worst}");
            }
        }
    }
}
