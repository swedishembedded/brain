// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where one training step of the timeline model spends its time, on the
//! device `BRAIN_BACKEND` selects: batch assembly on the host, the upload,
//! forward, backward and the optimiser step, each timed to completion, then
//! (under `BRAIN_PROFILE=1`) the per-kernel table.
//!
//! The shape is a cohort model's: 256 subjects of 96 token rows, two layers
//! of width 64, 21 hazard knots, three outcome codes. Kernel shapes depend on
//! the padded rows, not on how many tokens a subject really has, so the
//! synthetic population gives the device cost of real data.
//!
//! ```text
//! cargo run --release -p brain-horizon --example step_profile [-- STEPS]
//! ```

use std::time::Instant;

use data::rng::Rng;
use horizon::batch::assemble;
use horizon::encode::{encode, Encoded};
use horizon::synthetic::{population, CODES};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let (subjects, _) = population(4096, 1);
    let codes: Vec<String> = CODES.iter().map(|c| c.to_string()).collect();
    let vocab = Vocab::fit(&subjects, &codes, &codes[..2], &FitOptions::default()).expect("vocab");
    let mut cfg = HorizonConfig::default_for(vocab.len(), CODES.len() as u32);
    cfg.max_tokens = 96;
    cfg.knots = (0..=20).map(|k| k as f32).collect();
    let b = 256u32;
    let enc: Vec<Encoded> = subjects.iter().map(|s| encode(s, &vocab, &cfg)).collect();
    let model = Horizon::new(cfg.clone(), b, &horizon::init_weights(&cfg, 1));
    let mut rng = Rng::new(7);
    let (mut host, mut upload, mut fwd, mut bwd, mut opt) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for step in 0..steps {
        let t = Instant::now();
        let picks: Vec<&Encoded> = (0..b as usize)
            .map(|_| &enc[(rng.next_u64() % enc.len() as u64) as usize])
            .collect();
        let hb = assemble(&cfg, &picks, b as usize, 0.3, &mut rng);
        let t1 = Instant::now();
        model.set_batch(&hb);
        model.gpu.poll_wait();
        let t2 = Instant::now();
        model.forward();
        let t3 = Instant::now();
        model.zero_grads();
        model.backward();
        model.gpu.poll_wait();
        let t4 = Instant::now();
        model.adamw_step(
            step as u32 + 1,
            1e-3,
            0.1,
            model::Adam::default(),
            Some(1.0),
            1.0,
        );
        model.gpu.poll_wait();
        let t5 = Instant::now();
        if step == 4 {
            model.gpu.set_kernel_timing(true);
            model.gpu.reset_kernel_times();
        }
        if step >= 5 {
            host += (t1 - t).as_secs_f64();
            upload += (t2 - t1).as_secs_f64();
            fwd += (t3 - t2).as_secs_f64();
            bwd += (t4 - t3).as_secs_f64();
            opt += (t5 - t4).as_secs_f64();
        }
    }
    let n = (steps.saturating_sub(5)).max(1) as f64 * 1e-3;
    println!(
        "per step (ms): assemble {:.2}  upload {:.2}  forward+loss {:.2}  backward {:.2}  adamw {:.2}  total {:.2}",
        host / n,
        upload / n,
        fwd / n,
        bwd / n,
        opt / n,
        (host + upload + fwd + bwd + opt) / n
    );
    // Per kernel, device time per step, the largest first.
    if let Some(mut times) = model.gpu.kernel_times() {
        times.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f64 = times.iter().map(|t| t.1).sum();
        println!(
            "device kernels: {:.2} ms per step in all",
            total / (n * 1e3)
        );
        for (name, secs, calls) in times.iter().take(25) {
            println!(
                "  {name:<28} {:>8.3} ms  {:>5} launches",
                secs / (n * 1e3),
                *calls as f64 / (n * 1e3)
            );
        }
    }
}
