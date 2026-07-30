// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measured host-f32 trainer step time at REAL klein-4B dims (random-init
//! weights, 256×256 image → 16×16 latent grid, full 512-token text pad).
//! Ignored by default — this is the measurement backing the step-time numbers
//! documented in `flux2::finetune`'s module doc, not a CI gate. Run with:
//! `cargo test -p brain-flux2 --release --test step_time -- --ignored --nocapture`

use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{grads, init_model, make_flow_batch, Cfg};
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

    let x0 = model::hostmath::randn(cfg.n_img() * cfg.in_channels, 2);
    let ctx = model::hostmath::randn(cfg.txt_len * cfg.context_in_dim, 3);
    let noise = model::hostmath::randn(x0.len(), 4);
    let b = make_flow_batch(&cfg, &x0, &ctx, 0.5, &noise);

    let mut ad = LoraAdapter::new(&cfg, LoraCfg::new(16));
    let t0 = std::time::Instant::now();
    let w_eff = ad.apply(&base);
    let t_apply = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let (loss, g) = grads(&cfg, &w_eff, &b);
    let t_grads = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    ad.step(&g, 1e-4);
    let t_step = t0.elapsed().as_secs_f64();
    eprintln!(
        "klein-4b host f32 LoRA step @256×256: apply {t_apply:.1} s + fwd/bwd {t_grads:.1} s + adam {t_step:.1} s (loss {loss:.4})"
    );
}
