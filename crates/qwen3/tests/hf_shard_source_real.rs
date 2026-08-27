// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The spec for the shard-aware streaming import, measured against a REAL
//! HuggingFace Qwen3 checkpoint rather than a synthetic fixture.
//!
//! Swedish Embedded AB implements streaming checkpoint import and
//! device-resident model loading for its clients. If your team needs expertise
//! in large-model weight loading then you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! `crates/qwen3/src/import.rs`'s unit tests pin the same properties on a
//! hand-built 4-layer checkpoint, where every tensor is a few dozen elements.
//! That fixture cannot catch anything that only appears at real scale: a
//! sharded `weight_map` spread over several files, a bf16 source dtype, an
//! embedding table of hundreds of millions of elements, or a shard boundary
//! that falls inside a shard FILE rather than neatly between two.
//!
//! So this test asserts the same thing the fixture does - **equality of bits**
//! between the eager whole-checkpoint import and the shard-streamed one - over
//! a checkpoint that has all four of those properties. It is the evidence
//! behind the claim that swapping a caller from `brain_init_from_hf` to
//! `hf_shard_source` cannot move a single output value: identical weights in
//! means identical device buffers, which means identical numerics.
//!
//! Env-gated because it needs a multi-GB checkpoint on disk that this
//! repository does not ship: point `BRAIN_QWEN3_HF_DIR` at an HF Qwen3 model
//! directory (`config.json` + `model.safetensors`, single or sharded). With
//! the variable unset the test reports that it is skipping and passes - it is
//! not a silent no-op.
//!
//! NOTE ON MEMORY: the comparison deliberately holds the EAGER import (the
//! whole model, fp32) so it can compare against it, so running this needs
//! host RAM of roughly four times the checkpoint's bf16 size. That cost is
//! the very thing the change under test removes from the production path; it
//! is paid here only to prove the removal was lossless.

use checkpoint::TensorSource;

/// Reset this process's peak-RSS high-water mark, so a later `VmHWM` read
/// reports the peak of one measured phase rather than of the whole process so
/// far. `5` is the "clear peak RSS" selector. Best-effort: on a kernel or
/// sandbox that refuses the write, the phase peaks simply become cumulative
/// and are reported as such rather than the test failing.
fn reset_peak_rss() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

/// This process's peak resident set size in bytes, from the kernel's own
/// high-water mark - not a sampled estimate.
fn peak_rss_bytes() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[test]
fn shard_streamed_and_eager_imports_agree_bit_for_bit_on_a_real_checkpoint() {
    let Ok(dir) = std::env::var("BRAIN_QWEN3_HF_DIR") else {
        eprintln!("skipping: set BRAIN_QWEN3_HF_DIR to an HF Qwen3 model directory to run this");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let cfg = qwen3::import::config_from_hf(&cfg_json).expect("parse config.json");

    // A genuinely partial shard, shaped like the one a mid-stack-tap text
    // encoder builds: the embedding plus the leading three quarters of the
    // layers, no final norm and no LM head. For a 36-layer Qwen3-8B that is
    // exactly the `end: 27` shard the FLUX.2 pipeline asks for.
    let end = (cfg.n_layers as usize) * 3 / 4;
    let shard = qwen3::Shard { start: 0, end, embed: true, head: false, gpu_index: qwen3::Shard::ANY_GPU };
    let want = qwen3::shard_param_list(&cfg, &shard);
    let full = cfg.param_list();
    assert!(want.len() < full.len(), "the shard must be a real truncation of a {}-layer config", cfg.n_layers);

    let want_elems: usize = want.iter().map(|(_, n)| n).sum();
    let full_elems: usize = full.iter().map(|(_, n)| n).sum();
    let biggest = want.iter().map(|(_, n)| *n).max().unwrap_or(0);
    eprintln!(
        "shard: {} of {} tensors, {:.2} of {:.2} G params, largest tensor {:.2} GiB at fp32",
        want.len(),
        full.len(),
        want_elems as f64 / 1e9,
        full_elems as f64 / 1e9,
        gib((biggest * 4) as u64),
    );

    // --- Phase 1: the streamed path alone, so its peak is its own. ---
    reset_peak_rss();
    let t0 = std::time::Instant::now();
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(&dir).expect("open checkpoint");
    let src = qwen3::import::hf_shard_source(&reader, &cfg, &shard).expect("shard-aware coverage check");
    // Touch every tensor the shard needs, exactly as a build would, and fold
    // each into an accumulator so nothing can be optimised away.
    let mut acc = 0f64;
    let mut seen = 0usize;
    for (name, numel) in &want {
        let mut got = 0usize;
        assert!(
            src.with_tensor(name, &mut |d| {
                got = d.len();
                acc += d.first().copied().unwrap_or(0.0) as f64;
                acc += d.last().copied().unwrap_or(0.0) as f64;
            }),
            "streamed source could not produce {name}"
        );
        assert_eq!(got, *numel, "{name}: element count");
        seen += 1;
    }
    let streamed_peak = peak_rss_bytes();
    let streamed_wall = t0.elapsed();
    assert_eq!(seen, want.len());
    eprintln!("streamed shard import: peak RSS {:.2} GiB, {:.1} s", gib(streamed_peak), streamed_wall.as_secs_f64());

    // --- Phase 2: the eager whole-checkpoint path, for the comparison. ---
    reset_peak_rss();
    let t1 = std::time::Instant::now();
    let eager_ts = checkpoint::safetensors::read_model_dir(&dir).expect("eager read_model_dir");
    let eager = qwen3::import::brain_init_from_hf(eager_ts, &cfg).expect("eager brain_init_from_hf");
    let eager_peak = peak_rss_bytes();
    let eager_wall = t1.elapsed();
    eprintln!("eager whole import:    peak RSS {:.2} GiB, {:.1} s", gib(eager_peak), eager_wall.as_secs_f64());

    // The measurement is only meaningful if the two phases really differ; a
    // streamed peak at or above the eager one means the streaming did not
    // happen and the rest of this test would be reassuring for no reason.
    assert!(
        streamed_peak < eager_peak,
        "streamed peak {:.2} GiB must be below the eager peak {:.2} GiB",
        gib(streamed_peak),
        gib(eager_peak)
    );

    // --- Phase 3: equality of bits, tensor by tensor. ---
    // Bits, not a tolerance. Both routes decode the same bf16 source bytes
    // with the same `bf16_to_f32`; the only difference is WHERE the decoded
    // f32 lives. A tolerance here would pass a reader that dropped a tail
    // element or mixed up two shard files.
    for (name, _) in &want {
        let mine = eager.get(name).unwrap_or_else(|| panic!("{name}: absent from the eager import"));
        let mut equal = false;
        assert!(src.with_tensor(name, &mut |d| equal = d == mine.as_slice()), "streamed source lost {name}");
        assert!(equal, "{name}: streamed and eager values must be identical");
    }
    eprintln!("verified {} tensors bit-for-bit; acc {acc:e}", want.len());
}
