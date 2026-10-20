// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision full fine-tuning for its clients. If your team needs
// expertise in training real models with 8-bit optimizer state then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Bounded real-model check of the full fine-tune precision seam: a short full
//! fine-tune of Qwen3-0.6B with the AdamW moments host-offloaded in fp32, bf16
//! and blockwise int8, on a synthetic next-token rule, reporting loss before and
//! after on train and held-out sequences, step time, peak device memory (this
//! process, as the driver reports it), host peak RSS and the host optimizer state.
//!
//! ```text
//! card --card 1 --gb 16 -- cargo test --release -p brain-qwen3 --test real_full_ft_precision -- --ignored --nocapture
//! ```

use std::time::Instant;

use paramstore::precision::{ParamPrecision, PrecisionPolicy, StateFormat};

const T: u32 = 64;
const STEPS: u32 = 24;

fn sequence(seed: u32, vocab: u32) -> (Vec<u32>, Vec<u32>) {
    // a learnable rule over a small token subset: next = (7 * cur + 3) mod 997
    let mut cur = (seed * 131 + 17) % 997;
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for _ in 0..T {
        let nxt = (7 * cur + 3) % 997;
        x.push(cur % vocab);
        y.push(nxt % vocab);
        cur = nxt;
    }
    (x, y)
}

fn mean_loss(m: &qwen3::Qwen, seqs: &[(Vec<u32>, Vec<u32>)]) -> f32 {
    let mut s = 0.0;
    for (x, y) in seqs {
        m.set_batch(x, y);
        s += m.forward();
    }
    s / seqs.len() as f32
}

fn run(name: &str, path: &str, policy: &PrecisionPolicy) -> (f32, f32) {
    let (cfg, init) = qwen3::open_checkpoint(path).expect("open checkpoint");
    let vocab = cfg.vocab;
    let opts = model::FitOpts { batch_size: 1, block_size: T, ..Default::default() };
    let m = model::offload::with_precision_policy(policy, || {
        model::offload::with_adam_offloaded(true, || qwen3::finetune::build_for_training(cfg, &opts, &*init, qwen3::Dtype::F32))
    })
    .expect("build");
    drop(init);
    let train: Vec<_> = (0..8).map(|i| sequence(i, vocab)).collect();
    let held: Vec<_> = (100..108).map(|i| sequence(i, vocab)).collect();
    let (t0_train, t0_held) = (mean_loss(&m, &train), mean_loss(&m, &held));
    let mut vram = brain_testutil::own_gpu_memory_mib().unwrap_or(0);
    let t = Instant::now();
    for step in 1..=STEPS {
        let (x, y) = &train[(step as usize - 1) % train.len()];
        m.zero_grads();
        m.set_batch(x, y);
        m.forward();
        m.backward();
        m.adamw_step(step, 1e-5, 0.0, Default::default(), Some(1.0), 1.0);
        vram = vram.max(brain_testutil::own_gpu_memory_mib().unwrap_or(0));
    }
    m.poll_wait();
    let ms = t.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
    let (t1_train, t1_held) = (mean_loss(&m, &train), mean_loss(&m, &held));
    let (state, staging) = m.offload_host_bytes().unwrap_or((0, 0));
    println!(
        "{name}: train loss {t0_train:.3} -> {t1_train:.3}, held-out {t0_held:.3} -> {t1_held:.3}, {ms:.0} ms/step, device peak {vram} MiB, host state {:.0} MiB, staging {:.0} MiB",
        state as f64 / 1048576.0,
        staging as f64 / 1048576.0
    );
    brain_testutil::mem(name);
    assert!(t1_train < t0_train, "{name}: training loss must fall");
    (t1_train, t1_held)
}

#[test]
#[ignore]
fn qwen3_0_6b_full_fine_tune_with_fp32_bf16_and_int8_moments() {
    let Some(path) = brain_testutil::model_dir("Qwen/Qwen3-0.6B") else { return };
    if !std::path::Path::new(&path).exists() {
        brain_testutil::skip_unavailable("Qwen/Qwen3-0.6B is not in the model store");
        return;
    }
    let with = |state| PrecisionPolicy::uniform(ParamPrecision { state, stochastic_rounding: true, ..ParamPrecision::FP32 });
    let (f_train, f_held) = run("fp32 moments", &path, &with(StateFormat::F32));
    let (b_train, b_held) = run("bf16 moments", &path, &with(StateFormat::Bf16));
    let (q_train, q_held) = run("int8 moments", &path, &with(StateFormat::Int8Block { block: 64 }));
    for (n, tr, he) in [("bf16", b_train, b_held), ("int8", q_train, q_held)] {
        assert!(tr < f_train + 0.5 && he < f_held + 0.5, "{n} moments fall behind fp32: train {tr} vs {f_train}, held {he} vs {f_held}");
    }
}
