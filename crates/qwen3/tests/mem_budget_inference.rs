// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Regression gate for the inference-only (`train=false`) construction path:
//! `Qwen::new_shard(cfg, 1, cfg.block_size, &init, false, shard)` must not
//! request device memory anywhere near what a real batched TRAINING build
//! needs, because it never runs a backward pass and never needs more than one
//! layer's forward activations alive at once.
//!
//! At real Qwen3-0.6B shape this used to allocate a full backward-shaped
//! scratch set (gated only by the unrelated `decode_only` flag, never by
//! `train`) plus one permanently-resident activation set PER LAYER (28 of
//! them), instead of pooling identically-shaped per-layer scratch the way a
//! pure forward pass can. On a real 15.2 GiB card that is an OOM for a model
//! whose weights are only ~2.27 GiB - `./target/debug/brain models profile
//! Qwen/Qwen3-0.6B-Q8_0` is the real command that hit it.
//!
//! [`gpu_core::Gpu::charged_bytes`] (backed by `memauth`'s process-wide
//! ceiling) is the existing device-bytes-requested accounting this test
//! reads - no new instrumentation. It reports `0` unless a ceiling is
//! published (`BRAIN_LIMIT_VRAM_TOTAL`/`BRAIN_LIMIT_RAM_TOTAL`), and that
//! ceiling is a process-wide `OnceLock` (`memauth::limits()`), so - exactly
//! like `crates/gpu-core/tests/memory_limit.rs` - the measurement runs in a
//! freshly spawned child process rather than risk another test in this
//! binary resolving it first.
//!
//! Swedish Embedded AB implements memory-bounded inference engines for teams
//! shipping real transformer checkpoints onto real VRAM budgets. If your team
//! needs expertise in making a model's construction cost match what it
//! actually computes, rather than what its training twin would, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::process::Command;

/// Re-run this test binary as a child process, executing only the named
/// `#[ignore]`d helper, with a generous (64 GiB) ceiling published on BOTH
/// memory classes so `charged_bytes()` is populated regardless of which
/// backend (GPU or CPU JIT) the ambient device selection resolves to here.
/// Returns the child's stdout; panics with its full output on failure.
fn measure(helper: &str) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", helper, "--ignored", "--nocapture", "--test-threads=1"]);
    cmd.env("BRAIN_LIMIT_VRAM_TOTAL", "64G");
    cmd.env("BRAIN_LIMIT_RAM_TOTAL", "64G");
    let out = cmd.output().expect("spawn subprocess");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "child {helper} exited {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}", out.status.code());
    stdout
}

fn marker(stdout: &str, name: &str) -> String {
    let needle = format!("{name}=");
    stdout
        .lines()
        .find_map(|l| l.split_once(&needle).map(|(_, rest)| rest.trim().to_string()))
        .unwrap_or_else(|| panic!("child never printed {needle}; stdout:\n{stdout}"))
}

/// The regression gate: a real Qwen3-0.6B-scale `train=false` build must stay
/// well under a generous device-bytes ceiling. Weights alone are ~2.27 GiB
/// (fp32-dequantized) at this shape, so 5 GiB leaves headroom for the head
/// buffers and KV cache a pure-forward build legitimately needs, while still
/// catching the unpooled per-layer activations (~11+ GiB at real GGUF
/// `block_size`, several GiB even at this preset's smaller `block_size`) and
/// the unconditionally-allocated backward scratch this regression is about.
#[test]
fn inference_only_build_stays_under_a_generous_device_budget() {
    let out = measure("child_measure_inference_charged_bytes");
    let charged: u64 = marker(&out, "CHARGED").parse().expect("CHARGED must be a byte count");
    let weights_only: u64 = marker(&out, "WEIGHTS_ONLY").parse().expect("WEIGHTS_ONLY must be a byte count");
    const GIB: u64 = 1 << 30;
    assert!(
        charged >= weights_only,
        "a real build must charge at least its own weights ({weights_only} bytes); got {charged}"
    );
    assert!(
        charged < 5 * GIB,
        "train=false Qwen3-0.6B-scale build requested {charged} bytes ({:.2} GiB) - \
         expected well under 5 GiB given weights alone are only ~2.27 GiB; \
         this is the per-layer-activation / backward-scratch regression",
        charged as f64 / GIB as f64
    );
}

#[test]
#[ignore = "child process helper, driven by inference_only_build_stays_under_a_generous_device_budget"]
fn child_measure_inference_charged_bytes() {
    let cfg = qwen3::QwenConfig::qwen3_0_6b();
    let init = qwen3::init_weights(&cfg, 0);
    let shard = qwen3::Shard::whole(cfg.n_layers as usize);
    let weights_only: u64 = cfg.param_list().iter().map(|(_, n)| *n as u64 * 4).sum();
    let m = qwen3::Qwen::new_shard(cfg.clone(), 1, cfg.block_size, &init, false, shard);
    println!("WEIGHTS_ONLY={weights_only}");
    println!("CHARGED={}", m.gpu.charged_bytes());
}
