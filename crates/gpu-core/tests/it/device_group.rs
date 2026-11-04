// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `DeviceGroup`: all-reduce and all-gather over device buffers, on the device
//! collective library when there is one (NCCL for CUDA) and through the host
//! otherwise - the same exact results either way.
//!
//! Swedish Embedded AB implements multi-GPU training and inference runtimes for
//! its clients. If your team needs expertise in keeping tensor-parallel and
//! data-parallel traffic on the GPU interconnect instead of staging it through
//! host memory, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The CPU test needs no hardware. The two-card test is skip-if-absent
//! (`brain_testutil::skip_unavailable`): fewer than two CUDA cards, or no
//! `libnccl.so.2`, leaves nothing to compare. Buffers are 1 MiB.

use gpu_core::devgroup::DeviceGroup;
use gpu_core::Gpu;

const WORDS: usize = 1 << 18;

/// Values of mixed sign and magnitude, so the sum rounds and a wrong reduction order or a dropped rank shows.
fn rank_data(rank: u32) -> Vec<f32> {
    (0..WORDS as u32).map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(rank * 40_503) % 20_011) as f32 * 0.37 - 3_700.0 * (rank as f32 + 1.0)).collect()
}

/// Reduce and gather across `members`, then check every rank holds the exact sum / concatenation.
fn collectives_are_exact(members: &[&Gpu]) -> &'static str {
    let group = DeviceGroup::open(members);
    let n = members.len();
    let data: Vec<Vec<f32>> = (0..n as u32).map(rank_data).collect();

    let bufs: Vec<_> = members.iter().zip(&data).map(|(g, d)| g.storage_init("x", d)).collect();
    let refs: Vec<_> = bufs.iter().collect();
    group.all_reduce_sum(&refs, WORDS as u64).expect("all_reduce_sum");
    // Left-to-right in rank order, which for two ranks is the only order there is.
    let want: Vec<f32> = (0..WORDS).map(|i| data.iter().skip(1).fold(data[0][i], |s, d| s + d[i])).collect();
    for (rank, (g, b)) in members.iter().zip(&bufs).enumerate() {
        let got = g.read(b, WORDS);
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "rank {rank}: all_reduce_sum is not the exact sum ({})", group.provider());
    }

    let piece = 4096usize;
    let send: Vec<_> = members.iter().zip(&data).map(|(g, d)| g.storage_init("s", &d[..piece])).collect();
    let recv: Vec<_> = members.iter().map(|g| g.storage((piece * n) as u64)).collect();
    let (send_refs, recv_refs): (Vec<_>, Vec<_>) = (send.iter().collect(), recv.iter().collect());
    group.all_gather(&send_refs, &recv_refs, piece as u64).expect("all_gather");
    let want: Vec<f32> = data.iter().flat_map(|d| d[..piece].iter().copied()).collect();
    for (rank, (g, b)) in members.iter().zip(&recv).enumerate() {
        let got = g.read(b, piece * n);
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "rank {rank}: all_gather is not the concatenation ({})", group.provider());
    }
    group.provider()
}

#[test]
fn the_cpu_backend_reduces_and_gathers_exactly_through_the_host() {
    let (a, b) = (Gpu::new_cpu(&[]), Gpu::new_cpu(&[]));
    assert_eq!(collectives_are_exact(&[&a, &b]), "host");
}

#[test]
fn a_wrong_number_of_buffers_is_refused() {
    let (a, b) = (Gpu::new_cpu(&[]), Gpu::new_cpu(&[]));
    let group = DeviceGroup::open(&[&a, &b]);
    let x = a.storage(8);
    assert!(group.all_reduce_sum(&[&x], 8).is_err(), "one buffer for two ranks must be refused");
}

#[test]
fn two_cards_reduce_and_gather_exactly_on_nccl() {
    if gpu_core::devices::gpus().len() < 2 {
        brain_testutil::skip_unavailable("fewer than two GPUs");
        return;
    }
    if let Err(e) = backend_cuda::nccl::availability() {
        brain_testutil::skip_unavailable(&format!("no NCCL: {e}"));
        return;
    }
    let open = |i| gpu_core::devices::with_gpu(i, || Gpu::try_new_cuda(&[])).and_then(|g| g);
    let (Ok(a), Ok(b)) = (open(0), open(1)) else {
        brain_testutil::skip_unavailable("no usable CUDA backend on two cards");
        return;
    };
    assert_eq!(collectives_are_exact(&[&a, &b]), "nccl", "NCCL is present, so the group must use it");
}
