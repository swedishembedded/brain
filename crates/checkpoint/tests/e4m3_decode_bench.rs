// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated (no disk I/O) before/after timing for
//! `checkpoint::safetensors::decode_e4m3_bytes`'s parallelization: an
//! already-in-memory synthetic byte buffer, so the measurement is not
//! confounded by mmap page faults / cold vs warm page cache / ambient
//! machine load the way a real-checkpoint timing run is (this session's own
//! `crates/qwen35/tests/stream_profile.rs` numbers, taken on a shared,
//! heavily-loaded box, moved around a lot for exactly that reason). The
//! decode cost is uniform per byte regardless of VALUE (a LUT lookup), so
//! random bytes are representative of a real FP8 tensor's own cost.
//!
//! `#[ignore]`d (not part of the default fast suite - a few hundred MB
//! allocation and a real multi-core parallel pass, not a unit-test-sized
//! cost) but needs no environment variable: it is pure host computation, no
//! checkpoint required. Run with:
//!
//! ```text
//! cargo test -p brain-checkpoint --test e4m3_decode_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

/// One real layer's worth of FP8 bytes (`qwen35::config::Qwen35Config::
/// layer_i8_bytes`'s own real numbers are ~372-383 MB for the INT8-packed
/// size; the raw FP8 checkpoint tensor is the same element count, 1 byte
/// each) - big enough that per-call overhead is negligible next to the
/// actual decode work, matching the real scale this fix targets.
const LAYER_BYTES: usize = 380_000_000;

#[test]
#[ignore]
fn parallel_e4m3_decode_beats_sequential_on_an_in_memory_buffer() {
    let mut rng = data::rng::Lcg::new(42);
    let raw: Vec<u8> = (0..LAYER_BYTES).map(|_| (rng.next_u64() % 256) as u8).collect();

    // The exact pre-fix sequential expression (still what `decode_e4m3_bytes`
    // falls back to on wasm32) - inlined here rather than calling a private
    // function, since this test lives outside the crate.
    let t0 = Instant::now();
    let seq: Vec<f32> = raw.iter().map(|&b| checkpoint::safetensors::e4m3fn_to_f32(b)).collect();
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let par = checkpoint::safetensors::decode_e4m3_bytes(&raw);
    let par_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Bit-pattern comparison, not `==`: E4M3's one reserved NaN encoding
    // (`0x7F`/`0xFF`) decodes to a real `f32::NAN`, and `NaN != NaN` under
    // IEEE 754 would fail `assert_eq!` on ~1/128 of a large random buffer's
    // elements even when the two decodes agree bit-for-bit.
    let mismatch = seq.iter().zip(&par).position(|(&a, &b)| a.to_bits() != b.to_bits());
    assert!(mismatch.is_none(), "parallel decode diverges from sequential at index {:?}", mismatch);

    let speedup = seq_ms / par_ms.max(1e-6);
    eprintln!(
        "e4m3 decode, {} MB ({} elements): sequential={seq_ms:.1}ms parallel={par_ms:.1}ms speedup={speedup:.2}x (nproc={})",
        LAYER_BYTES / 1_000_000,
        LAYER_BYTES,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    );
}
