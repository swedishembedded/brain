// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measured host-f32 trainer step time AND whole-process memory at REAL
//! klein-4B dims (random-init weights, 256×256 image → 16×16 latent grid, full
//! 512-token text pad). Ignored by default - it reports what a step costs on
//! the machine it is run on and asserts nothing about it, so it is a
//! measurement instrument rather than a gate; the CI-sized memory gate is
//! `tests/streamed_grads.rs`. Run with:
//! `cargo test -p brain-flux2 --release --test step_time -- --ignored --nocapture`
//!
//! Memory is read from `/proc/self/status`, whole-process, because that is the
//! number that decides whether a run survives the box: a heap counter cannot
//! see a mapping's faulted pages and a peak that has already been freed is
//! still a peak. `BRAIN_FLUX2_STEP_COLLECT=1` runs the step the old way
//! (`grads` + `step`, collecting a whole-model `ModelGrads`) so the two peaks
//! can be measured back to back on the same machine.

use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{grads, grads_into, init_model, make_flow_batch, Cfg};
use flux2::Flux2Config;

#[test]
#[ignore = "multi-minute measurement, not a gate"]
fn klein_4b_host_step_time() {
    let fc = Flux2Config::klein_4b();
    let cfg = Cfg::from_flux2(&fc, 16, 16); // 256×256 px → 256 latent tokens
    eprintln!("init klein-4b random weights…");
    let t0 = std::time::Instant::now();
    let base = init_model::<f32>(&cfg, 1);
    eprintln!("init: {:.1} s", t0.elapsed().as_secs_f64());
    brain_testutil::mem("frozen base resident");

    let x0 = model::hostmath::randn(cfg.n_img() * cfg.in_channels, 2);
    let ctx = model::hostmath::randn(cfg.txt_len * cfg.context_in_dim, 3);
    let noise = model::hostmath::randn(x0.len(), 4);
    let b = make_flow_batch(&cfg, &x0, &ctx, 0.5, &noise);

    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(16));
    brain_testutil::reset_peak_rss();
    let t0 = std::time::Instant::now();
    let w_eff = ad.apply(&base);
    let t_apply = t0.elapsed().as_secs_f64();
    brain_testutil::mem("+ effective weights (apply)");

    let collect = std::env::var("BRAIN_FLUX2_STEP_COLLECT").is_ok_and(|v| v != "0");
    let t0 = std::time::Instant::now();
    let (loss, t_step) = if collect {
        let (loss, g) = grads(&cfg, &w_eff, &b);
        let t_grads = t0.elapsed().as_secs_f64();
        brain_testutil::mem("+ whole-model ModelGrads");
        let t1 = std::time::Instant::now();
        ad.step(&g, 1e-4);
        eprintln!("  (collecting route: fwd/bwd {t_grads:.1} s)");
        (loss, t1.elapsed().as_secs_f64())
    } else {
        let mut s = ad.stepper(1e-4);
        let (loss, _globals) = grads_into(&cfg, &w_eff, &b, &mut s);
        (loss, 0.0)
    };
    let t_grads = t0.elapsed().as_secs_f64();
    brain_testutil::mem("after the step");
    eprintln!(
        "klein-4b host f32 LoRA step @256×256 ({}): apply {t_apply:.1} s + fwd/bwd/adam {t_grads:.1} s (adam {t_step:.1} s, loss {loss:.4})",
        if collect { "collected grads" } else { "streamed grads" }
    );
}
