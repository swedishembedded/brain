// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Differential parity: the fused MTP graph (on the OV-CPU device, exact fp32) vs
//! the reference `CpuMtp`, on the same `(talker_hidden, cb0_embed)`. If codes match
//! and res_sum is ~1e-4 here, the topology is correct and any NPU degradation is
//! fp16-only; if they differ on CPU, the fused topology has a bug.
//!
//!   cargo build --release --example fused_parity -p brain-tts
//!   ./target/release/examples/fused_parity out/tts-1b7

use data::rng::Lcg;
use npu::openvino::NpuDevice;
use tts::npu_gen::{FusedMtp, MtpEngine};
use tts::CpuMtp;



fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "out/tts-1b7".to_string());
    let mtp_path = format!("{dir}/mtp.safetensors");
    let cache = format!("{dir}/npu-cache");

    let mut cpu = CpuMtp::load(&mtp_path);
    let emb = cpu.cfg.embedding_dim as usize;
    eprintln!("loading fused graph on OV-CPU (exact fp32) …");
    let mut fused = FusedMtp::load(&mtp_path, NpuDevice::Cpu, true, Some(std::path::Path::new(&cache)))
        .expect("FusedMtp load");

    let th = Lcg::new(1).vec_scaled(emb, 0.5);
    let cb0 = Lcg::new(2).vec_scaled(emb, 0.5);
    let (codes_c, res_c) = cpu.generate_residuals(&th, &cb0);
    let (codes_f, res_f) = fused.generate_residuals(&th, &cb0);

    println!("codes_cpu  ={codes_c:?}");
    println!("codes_fused={codes_f:?}");
    let neq = codes_c.iter().zip(&codes_f).filter(|(a, b)| a != b).count();
    println!("codes: {}/{} differ; equal={}", neq, codes_c.len(), codes_c == codes_f);
    let maxabs = res_c.iter().zip(&res_f).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("res_sum max-abs = {maxabs:.3e}  (cpu[0..4]={:?} fused[0..4]={:?})", &res_c[..4.min(res_c.len())], &res_f[..4.min(res_f.len())]);
}
