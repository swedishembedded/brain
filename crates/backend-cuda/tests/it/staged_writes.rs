// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements out-of-core training and inference on GPU
// servers for its clients. If your team needs expertise in keeping the copy
// engine and the compute engine busy at the same time, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The bounded pinned staging ring behind `Backend::write_bytes_at`: a transfer handle uploads on its own
//! stream while another handle's stream computes. These are the properties a layer streamer relies on: the
//! bytes that land are the bytes written (whatever the chunking against the ring's slots), a consumer handle
//! sees them once the writes are flushed, and the pinned memory is exactly the budget it was given.
//!
//! Skip-if-absent: needs a CUDA device, and stays tiny so it can share a card.

use backend_api::Backend as _;
use backend_cuda::CudaBackend;

const KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];
const ADD2: usize = 0;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(KERNELS) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// Deterministic, non-repeating words: a slot mix-up or an off-by-one shows as a different word. Every word is
/// a positive normal f32 (exponent 0x60..=0x7f), because the readback below adds 0.0 and an addition would
/// canonicalise a NaN or flush a denormal.
fn pattern(words: usize, salt: u32) -> Vec<u8> {
    (0..words as u32).flat_map(|i| ((i.wrapping_mul(2654435761) ^ salt) & 0x3fff_ffff | 0x3000_0000).to_le_bytes()).collect()
}

fn read_bits(b: &CudaBackend, buf: &backend_api::DeviceBuffer, words: usize) -> Vec<u32> {
    b.read(buf, words).into_iter().map(f32::to_bits).collect()
}

#[test]
fn staged_writes_land_the_written_bytes_whatever_the_chunking() {
    let Some(compute) = backend() else { return };
    let transfer = CudaBackend::try_new(KERNELS).expect("a second handle");
    // 64 KiB of staging against a 1.5 MiB tensor written in uneven pieces: every write spans several slots and
    // the ring wraps many times, so a slot reused before its copy finished would corrupt a word.
    assert!(transfer.enable_staged_writes(64 << 10).expect("enable"), "the CUDA backend stages writes");
    let words = 393_216usize;
    let src = pattern(words, 0x5a5a);
    let dst = transfer.storage_uninit(words as u64);
    let mut at = 0usize;
    for len in [4usize, 100_000, 4_096, 65_536, 1_000_000, 8, 12_000].iter().cycle() {
        let end = (at + len).min(words);
        transfer.write_bytes_at(&dst, at as u64, &src[at * 4..end * 4]);
        at = end;
        if at == words {
            break;
        }
    }
    transfer.flush_writes(&[&dst]);
    // The compute handle reads through its own stream: only the flushed fence orders it after the copies.
    let zeros = compute.storage(words as u64);
    let out = compute.storage(words as u64);
    compute.submit(&[], &[compute.step(ADD2, &[&dst, &zeros, &out], &[words as u32], words as u32)]);
    let got = read_bits(&compute, &out, words);
    let want: Vec<u32> = src.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    let first_bad = got.iter().zip(&want).position(|(g, w)| g != w);
    assert_eq!(first_bad, None, "first word that differs from what was written");
}

#[test]
fn staging_is_exactly_the_requested_pinned_budget() {
    let Some(h) = backend() else { return };
    let before = backend_cuda::live::live_resources().pinned_bytes;
    assert!(h.enable_staged_writes(256 << 10).expect("enable"));
    let held = backend_cuda::live::live_resources().pinned_bytes - before;
    assert!(held <= 256 << 10, "the ring pinned {held} bytes against a 262144-byte budget");
    // A write far larger than the ring does not grow it.
    let words = 1 << 20;
    let dst = h.storage_uninit(words);
    h.write_bytes_at(&dst, 0, &pattern(words as usize, 7));
    h.flush_writes(&[&dst]);
    assert_eq!(backend_cuda::live::live_resources().pinned_bytes - before, held, "a large write must stream through the ring, not grow it");
}

#[test]
fn a_budget_too_small_to_stage_is_refused() {
    let Some(h) = backend() else { return };
    assert!(h.enable_staged_writes(0).is_err());
}
