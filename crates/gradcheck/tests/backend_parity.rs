// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cross-backend forward parity (#10): brain's from-scratch Qwen3 forward must
//! produce the same logits on the CPU backend and the GPU (Vulkan) backend, for
//! the same weights + input. Together with the gradcheck suite running green on
//! both backends (analytic grads == finite differences) and the TTS NPU==CPU
//! parity tests, this is the CPU == Vulkan == NPU parity gate.
//!
//! The backend is chosen per `Gpu` via `set_default_backend`, so we build the
//! same model twice (CPU then GPU) in one process and diff the logits. If Vulkan
//! is unavailable the GPU `Gpu` falls back to wgpu, which is still a valid
//! CPU-vs-GPU parity check.

use std::collections::HashMap;

use gpu_core::{set_default_backend, Backend};
use qwen::{Qwen, QwenConfig};

fn logits_on(backend: Backend, cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>, x: &[u32]) -> Vec<f32> {
    set_default_backend(backend);
    // batch 1 (logits_all requires it); the model + its Gpu drop at end of scope.
    let m = Qwen::new(cfg.clone(), 1, x.len() as u32, init);
    m.logits_all(x)
}

#[test]
fn cpu_gpu_forward_parity() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let init = qwen::init_weights(&cfg, 7);
    let x: Vec<u32> = (0..12).map(|i| ((i * 5 + 1) % 23) as u32).collect();

    let lc = logits_on(Backend::Cpu, &cfg, &init, &x);
    let lg = logits_on(Backend::Vulkan, &cfg, &init, &x);
    assert_eq!(lc.len(), lg.len(), "logit count mismatch across backends");

    let maxd = lc.iter().zip(&lg).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let scale = lc.iter().fold(1f32, |m, &v| m.max(v.abs()));
    let rel = maxd / scale;
    eprintln!("CPU vs GPU forward logits: max-abs {maxd:.3e}, scale {scale:.2e}, rel {rel:.3e} ({} logits)", lc.len());
    // fp32 GPU vs CPU reduction-order differences only.
    assert!(rel < 5e-3, "cross-backend logits diverge: max-abs {maxd}, rel {rel}");
}
