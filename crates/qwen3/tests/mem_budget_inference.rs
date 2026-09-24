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
/// Two of these run in one binary and each spawns a child that builds a
/// real 0.6B model on the device. Left to the harness's default threads they
/// overlap, and the second one fails for want of memory the first is holding
/// - a fact about the schedule, not about either build. One at a time.
static ONE_CHILD_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn measure(helper: &str) -> String {
    let _serialised = ONE_CHILD_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
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

/// The same gate over the SEAM a generic caller reaches for.
///
/// `Model::new` builds the training shape, and it is the only constructor
/// the trait had: every generic consumer that merely scores or decodes -
/// `rl::improve::decode_checkpoint`, and so the continual reader's whole
/// battery - got the backward scratch and the per-layer activation copies
/// through it, with no way to ask for anything else. `Model::new_inference`
/// is that way, and this pins that the architecture actually overrides it
/// rather than inheriting the default that forwards to `new`.
#[test]
fn the_generic_inference_seam_stays_under_the_same_budget() {
    let out = measure("child_measure_trait_inference_charged_bytes");
    let charged: u64 = marker(&out, "CHARGED").parse().expect("CHARGED must be a byte count");
    let training: u64 = marker(&out, "TRAINING_ESTIMATE").parse().expect("TRAINING_ESTIMATE must be a byte count");
    const GIB: u64 = 1 << 30;
    assert!(
        charged < 5 * GIB,
        "Model::new_inference at Qwen3-0.6B scale requested {charged} bytes ({:.2} GiB); \
         the trait default forwards to Model::new, which is the training shape",
        charged as f64 / GIB as f64
    );
    assert!(
        charged < training,
        "new_inference ({charged}) must ask for less than the training build it exists to avoid ({training})"
    );
}

#[test]
#[ignore = "child process helper, driven by the_generic_inference_seam_stays_under_the_same_budget"]
fn child_measure_trait_inference_charged_bytes() {
    use model::Model;
    let cfg = qwen3::QwenConfig::qwen3_0_6b();
    let init = qwen3::init_weights(&cfg, 0);
    let block = cfg.block_size;
    let m = <qwen3::Qwen as Model>::new_inference(cfg.clone(), 1, block, &init);
    println!("CHARGED={}", m.gpu().charged_bytes());
    // The training shape's own demand, measured rather than asserted from a
    // remembered number: `new` keeps one copy of every per-layer activation
    // and the backward scratch beside it. Built at a SMALL block so the
    // comparison can be made on a card this one would not fit on, and
    // scaled to the block the inference build above used - the per-layer
    // activations this is about are linear in it.
    let small = 128u32;
    let mut tiny = cfg.clone();
    tiny.block_size = small;
    let t = <qwen3::Qwen as Model>::new(tiny, 1, small, &init);
    let scaled = t.gpu().charged_bytes().saturating_mul(u64::from(block) / u64::from(small));
    println!("TRAINING_ESTIMATE={scaled}");
}

/// `new_inference` is an allocation decision and never a numerical one: the
/// same weights through the same forward arithmetic must give the same
/// logits, or every score taken through the cheaper build is about a
/// different model than the one `new` would have scored.
#[test]
fn the_inference_build_computes_what_the_training_build_computes() {
    use model::Model;
    let cfg = qwen3::QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, 7);
    let tokens: Vec<u32> = (0..8u32).map(|i| i % cfg.vocab).collect();

    let train = <qwen3::Qwen as Model>::new(cfg.clone(), 1, cfg.block_size, &init);
    let infer = <qwen3::Qwen as Model>::new_inference(cfg.clone(), 1, cfg.block_size, &init);
    let a = train.logits_all(&tokens);
    let b = infer.logits_all(&tokens);
    assert_eq!(a.len(), b.len(), "the two builds must produce the same logit shape");
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 1e-4, "logit {i} differs: {x} vs {y}");
    }
}
