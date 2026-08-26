// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A readback's staging buffer is kept between calls, and a REUSED buffer is
//! usually bigger than the read that borrows it. This pins the one property
//! that makes reuse safe: a read returns its OWN bytes and nothing else.
//!
//! Swedish Embedded AB implements GPU compute and memory-transfer paths for
//! its clients. If your team needs expertise in Vulkan/wgpu readback and
//! staging-memory behaviour, you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! `WgpuBackend::read` used to allocate a fresh `MAP_READ` buffer per call.
//! Host memory is expensive to ALLOCATE - the driver pins the pages - which is
//! the same finding that turned upload staging into a recycled buffer; the
//! read direction had kept allocating. Reusing it introduces exactly one new
//! way to be wrong, and it is a silent one: the cached buffer holds the
//! PREVIOUS read's bytes past the end of the current one, so a read that maps
//! the whole buffer instead of its own `n * 4` range returns a plausible
//! tail rather than failing. Every assertion here is about that.
//!
//! The no-reuse arm is the same binary under
//! `BRAIN_GPU_NO_READ_STAGING_REUSE=1` - the switch is read once per process,
//! so the two arms cannot be two cases in one test function. Both must pass.
//!
//! ```text
//! cargo test --release -p brain-backend-wgpu --test read_staging_reuse
//! BRAIN_GPU_NO_READ_STAGING_REUSE=1 cargo test --release -p brain-backend-wgpu --test read_staging_reuse
//! ```

use backend_api::{Backend, BufUsage};
use backend_wgpu::WgpuBackend;

fn backend() -> WgpuBackend {
    WgpuBackend::new(&[("axpy", kernels::AXPY)])
}

/// A device buffer holding `n` words of a distinctive, position-dependent
/// pattern, so a byte that came from the wrong place is identifiable rather
/// than merely different.
fn filled(b: &WgpuBackend, tag: u32, n: usize) -> (backend_api::DeviceBuffer, Vec<f32>) {
    let want: Vec<f32> = (0..n).map(|i| (tag as f32) * 1e6 + i as f32).collect();
    let buf = Backend::buffer(b, "src", (n * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    let words: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
    Backend::write_at(b, &buf, 0, &words);
    Backend::flush(b);
    (buf, want)
}

/// The shrinking sequence: a big read, then a small one, then a smaller one.
///
/// This is the ONE ordering a reused buffer can get wrong. Growing reads
/// cannot see a stale tail (the buffer is reallocated), and equal-size reads
/// overwrite every byte they map. Only a read that is SHORTER than the cached
/// buffer can be handed bytes the copy did not write.
#[test]
fn a_shorter_read_after_a_longer_one_returns_only_its_own_bytes() {
    let b = backend();
    // Deliberately not powers of two and not multiples of each other: a
    // rounding bug in the mapped range is invisible when every length divides
    // the last one.
    for n in [65_537usize, 4099, 1021, 7, 1] {
        let (buf, want) = filled(&b, n as u32, n);
        let got = Backend::read(&b, &buf, n);
        assert_eq!(got.len(), n, "read({n}) returned {} words", got.len());
        assert_eq!(got, want, "read({n}) after a longer read returned bytes it did not ask for");
    }
}

/// The growing sequence must work too - the cached buffer has to be replaced
/// rather than truncated - and each answer is still exact.
#[test]
fn a_longer_read_after_a_shorter_one_grows_the_cache() {
    let b = backend();
    for n in [3usize, 1024, 40_961, 131_072] {
        let (buf, want) = filled(&b, n as u32, n);
        assert_eq!(Backend::read(&b, &buf, n), want, "read({n}) after a shorter read");
    }
}

/// Two DIFFERENT source buffers read at the same length in a row: the second
/// answer must be the second buffer's, not a cached copy of the first.
///
/// Without this the test above passes for a `read` that never re-copies at
/// all - it would keep returning the first payload, which has the right
/// length and the right shape.
#[test]
fn every_read_re_copies_rather_than_returning_the_cached_payload() {
    let b = backend();
    let n = 8192;
    let (first, want_first) = filled(&b, 1, n);
    let (second, want_second) = filled(&b, 2, n);
    assert_eq!(Backend::read(&b, &first, n), want_first);
    assert_eq!(Backend::read(&b, &second, n), want_second, "the second read returned the first buffer's bytes");
    assert_eq!(Backend::read(&b, &first, n), want_first, "reading back the first buffer no longer agrees with itself");
    assert_ne!(want_first, want_second, "the two payloads must differ, or the assertion above proves nothing");
}

/// A zero-length read is empty, not a validation error. `read` has to special
/// case it: a zero-size copy and a zero-size map are both illegal in wgpu, and
/// the pre-reuse path sidestepped that only because it allocated a zero-size
/// buffer and never reached them.
#[test]
fn a_zero_length_read_is_empty() {
    let b = backend();
    let (buf, _) = filled(&b, 9, 16);
    assert!(Backend::read(&b, &buf, 0).is_empty());
    // ...and the device is still usable afterwards.
    assert_eq!(Backend::read(&b, &buf, 4).len(), 4);
}

/// The reuse is a PERFORMANCE property, and nothing above can see it: every
/// assertion so far passes just as well for a `read` that allocates fresh
/// every time. So gate the mechanism itself on the one thing that is
/// observable - how many staging buffers were allocated.
///
/// Without this, a revert to per-call allocation is green everywhere and
/// shows up only as a wall clock nobody is watching.
#[test]
fn a_loop_over_one_shape_pins_its_staging_pages_once() {
    let b = backend();
    let n = 4096;
    let (buf, want) = filled(&b, 5, n);
    let before = b.read_staging_allocations();
    for _ in 0..8 {
        assert_eq!(Backend::read(&b, &buf, n), want);
    }
    let allocs = b.read_staging_allocations() - before;
    let opted_out = std::env::var("BRAIN_GPU_NO_READ_STAGING_REUSE").map(|v| v != "0").unwrap_or(false);
    if opted_out {
        assert_eq!(allocs, 8, "the opt-out arm must allocate per read, or the comparison it exists for is not a comparison");
    } else {
        assert_eq!(allocs, 1, "eight reads of one shape allocated {allocs} staging buffers; the cache is not being reused");
    }
}
