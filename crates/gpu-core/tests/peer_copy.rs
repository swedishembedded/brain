// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Gpu::copy_to`: a byte range of a buffer on one device lands, exactly, in a
//! buffer of another, by the fastest path the pair has - straight over the
//! device-to-device link when the driver offers one, through the host otherwise.
//!
//! Swedish Embedded AB implements multi-GPU inference runtimes for its clients.
//! If your team needs expertise in splitting a model across accelerators without
//! paying for a host round trip at every boundary, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! The CPU test needs no hardware. The two-card test is skip-if-absent
//! (`brain_testutil::skip_unavailable`): a box with fewer than two CUDA cards
//! has no pair to copy between. Buffers are a few MiB at most.

use gpu_core::link::Path;
use gpu_core::Gpu;

/// Words that exercise every bit position, including NaN payloads and -0.0, so
/// a copy that went through float arithmetic anywhere would show.
fn pattern(words: usize, salt: u32) -> Vec<f32> {
    (0..words as u32).map(|i| f32::from_bits(i.wrapping_mul(0x9E37_79B1) ^ salt)).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Copy a middle range between two handles and check the range arrived whole
/// and its neighbours did not move. Returns the path the copy took.
fn copy_is_exact(from: &Gpu, to: &Gpu) -> Path {
    const WORDS: usize = 1 << 18; // 1 MiB
    let (src_off, dst_off, n) = (1000usize, 70_000usize, 150_000usize);
    let src_data = pattern(WORDS, 0x5bd1_e995);
    let dst_data = pattern(WORDS, 0x1b87_3593);
    let src = from.storage_init("src", &src_data);
    let dst = to.storage_init("dst", &dst_data);

    let path = from.copy_to(&src, src_off as u64 * 4, to, &dst, dst_off as u64 * 4, n as u64 * 4);

    let got = bits(&to.read(&dst, WORDS));
    let mut want = bits(&dst_data);
    want[dst_off..dst_off + n].copy_from_slice(&bits(&src_data)[src_off..src_off + n]);
    assert!(got == want, "the destination differs from the source range (path {path:?})");
    assert_eq!(bits(&from.read(&src, WORDS)), bits(&src_data), "the source was changed (path {path:?})");
    path
}

#[test]
fn the_cpu_backend_copies_exactly_through_the_host() {
    let (a, b) = (Gpu::new_cpu(&[]), Gpu::new_cpu(&[]));
    let link = a.peer_link(&b);
    assert!(!link.direct, "the CPU backend has no device-to-device path");
    assert!(link.reason.is_some(), "a missing direct path must say why");
    assert_eq!(copy_is_exact(&a, &b), Path::HostStaged);
}

#[test]
fn a_misaligned_range_is_refused() {
    let (a, b) = (Gpu::new_cpu(&[]), Gpu::new_cpu(&[]));
    let (src, dst) = (a.storage(16), b.storage(16));
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.copy_to(&src, 2, &b, &dst, 0, 8)));
    assert!(refused.is_err(), "a copy that is not word-aligned must be refused, not rounded");
}

/// One CUDA handle per card, the first two cards of the registry.
fn two_cuda_cards(kernels: &[(&str, &str)]) -> Option<(Gpu, Gpu)> {
    if gpu_core::devices::gpus().len() < 2 {
        brain_testutil::skip_unavailable("fewer than two GPUs");
        return None;
    }
    let open = |i| gpu_core::devices::with_gpu(i, || Gpu::try_new_cuda(kernels)).and_then(|g| g);
    match (open(0), open(1)) {
        (Ok(a), Ok(b)) => Some((a, b)),
        (Err(e), _) | (_, Err(e)) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend on two cards: {e}"));
            None
        }
    }
}

#[test]
fn two_cards_copy_exactly_and_take_the_peer_path_when_the_driver_offers_it() {
    let Some((a, b)) = two_cuda_cards(&[]) else { return };
    let dir = std::env::temp_dir().join(format!("brain-peer-copy-{}", std::process::id()));
    gpu_core::roof::set_cache_dir(Some(dir.clone()));

    for (from, to) in [(&a, &b), (&b, &a)] {
        let link = from.peer_link(to);
        let path = copy_is_exact(from, to);
        assert_eq!(
            path == Path::Peer,
            link.direct,
            "the path must follow the driver's answer ({link:?}) and fall back only without a direct link"
        );
        let bw = gpu_core::link::bandwidth(from, to).expect("link bandwidth");
        eprintln!(
            "{} -> {}: direct={} rank={:?} peer={:?} GB/s host-staged={:.2} GB/s",
            from.identity().unwrap().name,
            to.identity().unwrap().name,
            link.direct,
            link.performance_rank,
            bw.peer_gbs,
            bw.host_gbs
        );
        assert_eq!(bw.peer_gbs.is_some(), link.direct);
        if let Some(peer) = bw.peer_gbs {
            assert!(peer > bw.host_gbs, "the direct link ({peer} GB/s) must beat host staging ({} GB/s)", bw.host_gbs);
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_copy_waits_for_the_kernel_that_wrote_its_source() {
    // `axpy`: out[i] += s * in[i]. The copy is issued with no host wait after the dispatch, so only the
    // stream/event ordering can make the destination see the finished result.
    let Some((a, b)) = two_cuda_cards(&[("axpy", kernels::AXPY)]) else { return };
    const N: u32 = 1 << 18;
    let x = a.storage_init("x", &(0..N).map(|i| i as f32).collect::<Vec<_>>());
    let y = a.storage(N as u64);
    let step = a.step(0, &[&y, &x], &[N, 2.0f32.to_bits()], N);
    a.submit(&[], &[step]);
    let landed = b.storage(N as u64);
    a.copy_to(&y, 0, &b, &landed, 0, N as u64 * 4);
    // The consumer on the destination card reads `landed` without any host wait either.
    let out = b.storage(N as u64);
    let step = b.step(0, &[&out, &landed], &[N, 1.0f32.to_bits()], N);
    b.submit(&[], &[step]);
    let got = b.read(&out, N as usize);
    assert!(got.iter().enumerate().all(|(i, &v)| v == 2.0 * i as f32), "the copy raced the producer or the consumer");
}
