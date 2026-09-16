// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **An UNPOOLED chunk round must not outrun the device's reclaim ceiling.**
//!
//! Swedish Embedded AB implements device-memory lifetime management for
//! inference engines for its clients. If your team needs expertise in GPU
//! allocator behaviour and host-side pipeline stalls then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! `Qwen35::run_prefill_chunk_stage` runs a round in one of two scratch
//! regimes (`CHUNK_ARENA_MIN_ROWS`). The POOLED one recycles every layer's
//! temporaries and drains at each layer boundary because that recycling
//! requires it. The UNPOOLED one recycles nothing, so it owes no drain for
//! aliasing - but it still DROPS a layer's temporaries as it takes the next
//! layer's, and `backend_wgpu::WgpuBackend::track` refuses the next allocation
//! outright once more than the device's own ceiling is sitting
//! dropped-and-unreclaimed. That refusal is a panic, by design, so that this
//! failure is localised instead of surfacing as a generic OOM later.
//!
//! This was found the expensive way: a 128-row unpooled round on the real 27B
//! crossed a P40's 4.29 GB ceiling partway through its 34 layers. The fix is a
//! drain on BYTES (`Gpu::pending_reclaim_bytes` against
//! `Gpu::reclaim_ceiling_bytes`) rather than on a layer count, so a round pays
//! nothing at the 1-8 rows a speculative verify uses and is merely bounded at
//! a size that would otherwise have been fatal.
//!
//! Gating that needs the ceiling to be genuinely CROSSED, which in production
//! takes gigabytes. `BRAIN_GPU_RECLAIM_CEILING` lowers it so a `tiny()`-dims
//! round crosses it in kilobytes instead - the same mechanism, the same code
//! path, a second rather than a multi-GB allocation. It is resolved once per
//! process (`backend_api::hardware::reclaim_ceiling`), so the work runs in a
//! CHILD process, exactly as `gpu_core/tests/transient_reclaim.rs` does it.
//!
//! The child asserts more than "did not panic": it also checks the round is
//! still numerically what a token-by-token replay produces, because a drain
//! inserted in the wrong place would be equally silent and far worse.

use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

/// Chosen against the measured per-layer figure, not guessed. At `tiny()` dims
/// one layer of this round drops 32 KiB (GQA) to 91 KiB (GDN), and the 14-token
/// / chunk-4 replay below accumulates ~2.4 MiB across its four rounds. The
/// drain fires at half the ceiling, so the ceiling has to sit ABOVE twice one
/// layer's drop (or a single layer would cross it before the check at that
/// layer's end could run - the check `WgpuBackend::track` makes is per
/// ALLOCATION) and BELOW the whole call's accumulation (or nothing is ever
/// crossed and the gate proves nothing). 512 KiB is comfortably inside that
/// window on both sides: 256 KiB of budget against a 91 KiB layer, and 512 KiB
/// against 2.4 MiB of total drops.
///
/// The same window is enormous in production - a 2.1 GB budget against a
/// ~15 MB layer at the largest unpooled row count - which is the point: this
/// fixture reproduces the RELATIONSHIP at a scale a test can afford, not the
/// numbers.
const CEILING: u64 = 512 * 1024;

/// The prompt/tail/chunk shape is `tests/chunked_prefill.rs`', deliberately:
/// this gate is that gate's claim plus a ceiling, and a different fixture would
/// make a disagreement between them hard to attribute.
fn replay_matches(pooled: bool) {
    let cfg = Qwen35Config { n_layers: 8, ..Qwen35Config::tiny() };
    let init = qwen35::init::init_weights(&cfg, 7);
    let m = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, cfg.block_size, &init);
    m.set_chunk_arena_min_rows(if pooled { 1 } else { u32::MAX });

    let prompt: Vec<u32> = (0..14).map(|i| (i * 5 + 3) % cfg.vocab).collect();
    let tail: Vec<u32> = (0..3).map(|i| (i * 7 + 1) % cfg.vocab).collect();

    m.reset_decode_cache();
    let mut want_last = Vec::new();
    for &tok in &prompt {
        want_last = m.step(tok);
    }
    let want_tail: Vec<Vec<f32>> = tail.iter().map(|&tok| m.step(tok)).collect();

    m.reset_decode_cache();
    let got_last = m.prefill_chunked(&prompt, 4);
    let got_tail: Vec<Vec<f32>> = tail.iter().map(|&tok| m.step(tok)).collect();

    let maxabs = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    let err = maxabs(&got_last, &want_last);
    assert!(err < 1e-5, "pooled={pooled}: under a lowered reclaim ceiling the round's last hidden state moved, maxabs={err}");
    for (i, (got, want)) in got_tail.iter().zip(&want_tail).enumerate() {
        let e = maxabs(got, want);
        assert!(e < 1e-5, "pooled={pooled}: continuation token {i} maxabs={e}");
    }
    println!("reclaim-ceiling round (pooled={pooled}) completed and matched replay, worst maxabs {err:e}");
}

#[test]
#[ignore = "child-process helper, driven by the test below under a lowered ceiling"]
fn unpooled_round_under_a_lowered_ceiling_helper() {
    replay_matches(false);
}

#[test]
#[ignore = "child-process helper, driven by the test below under a lowered ceiling"]
fn pooled_round_under_a_lowered_ceiling_helper() {
    replay_matches(true);
}

fn child(helper: &str) -> (bool, String) {
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args(["--exact", helper, "--ignored", "--nocapture", "--test-threads=1"])
        .env("BRAIN_GPU_RECLAIM_CEILING", CEILING.to_string())
        .output()
        .expect("spawn subprocess");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

/// The gate. Both regimes must survive a ceiling their own round crosses -
/// the pooled one because it drains every layer, the unpooled one because it
/// drains on bytes.
///
/// Mutation-checked: deleting the `pending_reclaim_bytes` branch from
/// `run_prefill_chunk_stage`'s layer loop makes the unpooled child die with
/// "dropped without an intervening poll_wait()".
#[test]
fn a_chunk_round_never_outruns_the_devices_reclaim_ceiling() {
    if gpu_core::discrete_gpu_count() == 0 {
        brain_testutil::skip_unavailable("chunk_round_reclaim: needs a real device whose reclaim is deferred");
        return;
    }
    for helper in ["unpooled_round_under_a_lowered_ceiling_helper", "pooled_round_under_a_lowered_ceiling_helper"] {
        let (ok, out) = child(helper);
        assert!(
            ok,
            "{helper} must complete under a {CEILING}-byte reclaim ceiling - a chunk round has to bound its own \
             dropped-but-unreclaimed scratch at every row count, not only at the ones that happen to fit; output:\n{out}"
        );
    }
}
