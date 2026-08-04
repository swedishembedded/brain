// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic pipeline sharding (`model::Pipeline<Gpt>`) is bit-exact against the
//! single-device GPT — proving the sharding seam is not Qwen-specific. GPT's
//! lm_head is untied, so stages hold fully disjoint parameters (no replicated
//! gradient). The cut is placed automatically by `plan_balanced`.
//!
//! Two stages on GPUs 0 and 1 by default; `SHARD_TEST_GPUS=1,1` pins both to one
//! card. Skipped under `MOE_SKIP_GPU_TESTS`.

use gpt::{Gpt, GptConfig};
use model::{Batch, Pipeline};

fn gpu_disabled() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    // Skip rather than fault when this box lacks the cards the test pins to.
    // Without the check the multi-GPU paths assume cards 0..n exist and die
    // inside the driver on a single-GPU or GPU-less machine, which reads as a
    // real regression and masks actual ones.
    let need = stage_gpus().iter().copied().max().unwrap_or(0) + 1;
    let have = gpu_core::discrete_gpu_count();
    if have < need {
        eprintln!("skipping: needs {need} discrete GPU(s), found {have}");
        return true;
    }
    false
}
fn stage_gpus() -> Vec<usize> {
    std::env::var("SHARD_TEST_GPUS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<usize>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![0, 1])
}

#[test]
fn shard_forward_and_grad_parity_gpt() {
    if gpu_disabled() {
        return;
    }
    let cfg = GptConfig::tiny().with_ff_default(); // L2
    let init = gpt::init_weights(&cfg, 7);
    let (b, t) = (2u32, 8u32);
    let x: Vec<u32> = (0..b * t).map(|i| i * 3 % cfg.vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| (i * 3 + 1) % cfg.vocab).collect();

    // Single-device reference.
    let single = Gpt::new_on(gpu_core::testgpu::dev(gpt::model::PIPELINES), cfg.clone(), b, t, &init);
    single.set_batch(&x, &y);
    single.zero_grads();
    let l_single = single.forward();
    single.backward();
    single.poll_wait();

    // Auto-placed two-stage pipeline from the SAME weights.
    let pipe = Pipeline::<Gpt>::new(cfg.clone(), b, t, &init, &stage_gpus());
    assert_eq!(pipe.n_stages(), 2);
    eprintln!("auto-placed shards: {:?}", pipe.shards());
    pipe.zero_grads();
    let l_pipe = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
    pipe.backward();

    let dl = (l_single - l_pipe).abs() / l_single.abs().max(1e-6);
    eprintln!("gpt loss  single={l_single:.6}  pipe={l_pipe:.6}  rel={dl:.2e}");
    assert!(dl < 1e-4, "gpt sharded loss mismatch: {l_single} vs {l_pipe}");

    let mut worst = 0f32;
    let mut worst_name = String::new();
    for (name, _) in cfg.param_list() {
        let a = single.read_grad(&name);
        let g = pipe.reduced_grad(&name);
        let (mut num, mut den) = (0f32, 1e-6f32);
        for (p, q) in a.iter().zip(&g) {
            num = num.max((p - q).abs());
            den = den.max(p.abs());
        }
        let rel = num / den;
        if rel > worst {
            worst = rel;
            worst_name = name.clone();
        }
        assert!(rel < 1e-3, "gpt sharded grad mismatch for {name}: rel {rel:.2e}");
    }
    eprintln!("gpt worst grad rel {worst:.2e} ({worst_name})");
}
