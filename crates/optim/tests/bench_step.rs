// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Whole `Optim::step` wall-clock, at real model parameter-tensor
//! distributions (kernel-performance.md M6.4). `bench_gradnorm.rs` already
//! measures the grad-norm reduction alone; this measures the STEP a model
//! actually pays - grad-norm + clip + AdamW, end to end, steady state (after
//! the first call, which also pays the one-off graph build).
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-optim --test bench_step -- --ignored --nocapture
//! ```

use std::collections::HashMap;

use gpu_core::Gpu;
use optim::Optim;
use paramstore::ParamStore;

/// `(numel, count)` distributions, matching `bench_gradnorm.rs`.
type ParamDist = &'static [(usize, usize)];

const GPT2_SMALL: ParamDist = &[
    (768, 74),
    (2304, 12),
    (3072, 12),
    (786_432, 1),
    (589_824, 12),
    (1_769_472, 12),
    (2_359_296, 24),
    (38_597_376, 2),
];

const QWEN_0B6: ParamDist = &[
    (128, 56),
    (1024, 57),
    (1_048_576, 56),
    (2_097_152, 56),
    (3_145_728, 84),
    (155_582_464, 1),
];

fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| ((((i * 37 + s * 13) % 197) as f32 / 197.0) - 0.5) * 0.01).collect()
}

#[test]
#[ignore]
fn bench_step() {
    static KERNELS: &[(&str, &str)] = &[
        ("adamw", kernels::ADAMW),
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("grad_scale", kernels::GRAD_SCALE),
        ("clip_coef", kernels::CLIP_COEF),
        ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
        ("gradnorm_part", kernels::GRADNORM_PART),
        ("clip_coef_wg", kernels::CLIP_COEF_WG),
    ];

    for (model, dist) in [("GPT-2-small (148 tensors)", GPT2_SMALL), ("Qwen3-0.6B (311 tensors)", QWEN_0B6)] {
        let gpu = Gpu::new_wgpu(KERNELS);
        let opt = Optim::new(0, 1, 2, 3, 4);
        let mut shapes = Vec::new();
        let mut init = HashMap::new();
        let mut idx = 0usize;
        for &(numel, count) in dist {
            for _ in 0..count {
                let name = format!("p{idx}");
                idx += 1;
                init.insert(name.clone(), fill(numel.min(1024), 1).into_iter().cycle().take(numel).collect());
                shapes.push((name, numel));
            }
        }
        let n_tensors = shapes.len();
        let ps = ParamStore::new(&gpu, shapes.clone(), &init);
        for (name, numel) in &shapes {
            gpu.write(ps.g(name), bytemuck::cast_slice(&fill(*numel, 7)));
        }

        let stats = || gpu.stats().expect("wgpu backend always reports DeviceStats");

        // Warm-up: builds the graph (one-off dispatch registration + writes).
        opt.step(&gpu, &ps, 1, 1e-3, 0.01, 0.9, 0.999, 1e-8, Some(1.0), 1.0);
        gpu.poll_wait();

        let reps = 10;
        let d0 = stats();
        let t0 = std::time::Instant::now();
        for t in 2..2 + reps {
            opt.step(&gpu, &ps, t, 1e-3, 0.01, 0.9, 0.999, 1e-8, Some(1.0), 1.0);
        }
        gpu.poll_wait();
        let elapsed = t0.elapsed().as_secs_f64() / reps as f64;
        let d1 = stats();

        println!(
            "{model}: {n_tensors} tensors, {:.3} ms/step steady-state, {} dispatches/step, {} writes/step",
            elapsed * 1e3,
            (d1.dispatches - d0.dispatches) / reps as u64,
            (d1.writes - d0.writes) / reps as u64,
        );
    }
}
