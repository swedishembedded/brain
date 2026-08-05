// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8n @640 forward on the P40: conv-as-GEMM (im2col + matmul_reg3 +
//! epilogue) vs the direct register-tiled conv (`conv_act_reg`).
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-yolo --test bench_forward_p40 -- --ignored --nocapture
//! ```
//!
//! Conv is ~95% of YOLOv8n's cost. `conv_act_reg` collapses on the deep
//! small-spatial stages (measured below naive there); on a compute-bound GPU the
//! im2col+GEMM path runs those 2-5× faster (`backend-wgpu` `bench_conv_gemm`).
//! This measures the whole-network effect. `BRAIN_CONV_GEMM=0` forces the direct
//! conv; the default (unset) uses the GEMM path where eligible.

use std::collections::HashMap;

use gpu_core::{set_default_backend, Backend};
use yolo::{Yolo, YoloConfig};

fn randimg(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5).collect()
}

fn forward_ms(gemm: bool, init: &HashMap<String, Vec<f32>>, reps: usize) -> f64 {
    std::env::set_var("BRAIN_CONV_GEMM", if gemm { "1" } else { "0" });
    set_default_backend(Backend::Wgpu);
    let cfg = YoloConfig::yolov8n();
    let m = Yolo::new_on(gpu_core::testgpu::dev(yolo::net::PIPELINES), cfg.clone(), 1, cfg.input, init);
    m.set_eval(true);
    let img = randimg((3 * cfg.input * cfg.input) as usize);
    m.set_image(&img);
    m.forward_net_pub();
    m.gpu.poll_wait(); // warm

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        m.forward_net_pub();
        m.gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

#[test]
#[ignore]
fn yolov8n_forward_p40() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = YoloConfig::yolov8n();
    let init = yolo::init_weights(&cfg, 7);
    let reps = 15;

    let direct = forward_ms(false, &init, reps);
    let gemm = forward_ms(true, &init, reps);

    println!("\n=== YOLOv8n @640 forward on P40 (wgpu) ===");
    println!("  direct conv (conv_act_reg):  {direct:8.1} ms   {:.1} fps", 1e3 / direct);
    println!("  conv-as-GEMM (im2col+reg3):  {gemm:8.1} ms   {:.1} fps", 1e3 / gemm);
    println!("  speedup: {:.2}x", direct / gemm);

    assert!(direct.is_finite() && gemm > 0.0);
}
