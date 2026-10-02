// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A buffer larger than 4 GiB is addressed correctly by the generated tier.
//!
//! Swedish Embedded AB implements large-memory GPU compute for its clients,
//! where an index that wraps at 32 bits writes into the wrong tensor without a
//! fault. If your team needs expertise in proving an accelerator runtime
//! addresses every byte it allocates, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! # What would go wrong, and how this notices
//!
//! A byte offset or an element-to-byte conversion done in 32 bits wraps at 4 GiB:
//! an access at byte `4 GiB + x` lands on byte `x`. The test therefore writes a
//! sentinel at element `e` and a DIFFERENT sentinel at `e + 2^30` (exactly 4 GiB
//! further on), and reads both back: an aliased pair reads back as the later
//! write twice. It does the same through a sub-range binding whose offset is
//! past 4 GiB, and at the very last element of the buffer.
//!
//! The 5 GiB buffer is allocated only when the device has the room with a margin
//! for whatever else is resident, and is freed before the test returns; the live
//! counters must come back to baseline.

use backend_api::Backend as _;
use backend_cuda::{live_resources, CudaBackend};

/// One thread writes `p.val` at element `p.idx`; another reads element `p.idx`
/// into `out[0]`. Single-thread kernels, so every index is an explicit value
/// rather than something derived from the grid.
const POKE: &str = r#"
struct P { idx: u32, val: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    x[p.idx] = p.val;
}
"#;
const PEEK: &str = r#"
struct P { idx: u32, val: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[0] = x[p.idx];
}
"#;
const KERNELS: &[(&str, &str)] = &[("poke", POKE), ("peek", PEEK)];
const POKE_K: usize = 0;
const PEEK_K: usize = 1;

const GIB: u64 = 1 << 30;

fn poke(b: &CudaBackend, x: &backend_api::DeviceBuffer, idx: u64, val: u32) {
    b.submit(&[], &[b.step(POKE_K, &[x], &[idx as u32, val], 1)]);
}

fn peek(b: &CudaBackend, x: &backend_api::DeviceBuffer, out: &backend_api::DeviceBuffer, idx: u64) -> u32 {
    b.submit(&[], &[b.step(PEEK_K, &[x, out], &[idx as u32, 0], 1)]);
    b.read(out, 1)[0].to_bits()
}

#[test]
fn a_buffer_past_4_gib_is_addressed_without_a_32_bit_wrap() {
    let before = live_resources();
    let Ok(b) = CudaBackend::try_new(KERNELS) else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let words = 5 * GIB / 4;
    if b.max_buffer_bytes() < 5 * GIB + 2 * GIB {
        brain_testutil::skip_unavailable("less than 7 GiB free on the device");
        return;
    }
    {
        let x = b.storage(words);
        let out = b.storage(1);

        // 4 GiB apart: element + 2^30 words.
        let (lo, hi) = (12_345u64, 12_345 + (1 << 30));
        assert!(hi * 4 > u32::MAX as u64, "the second sentinel must be past 4 GiB");
        poke(&b, &x, lo, 0xA1A1_A1A1);
        poke(&b, &x, hi, 0xB2B2_B2B2);
        assert_eq!(peek(&b, &x, &out, lo), 0xA1A1_A1A1, "the write past 4 GiB landed on the element 4 GiB below it");
        assert_eq!(peek(&b, &x, &out, hi), 0xB2B2_B2B2);

        // The very last element, which is where a clamp that wrapped would show.
        poke(&b, &x, words - 1, 0xC3C3_C3C3);
        assert_eq!(peek(&b, &x, &out, words - 1), 0xC3C3_C3C3);
        assert_eq!(peek(&b, &x, &out, lo), 0xA1A1_A1A1, "the last-element write disturbed another element");

        // A binding whose offset is past 4 GiB (a sliced step's offset is in
        // words): element 0 of the slice is element `slice_elem` of the buffer.
        let slice_elem = (4 * GIB + 1024) / 4;
        assert!(slice_elem * 4 > u32::MAX as u64);
        let step = b.step_sliced(POKE_K, &[&x], &[(slice_elem, 0)], &[7, 0xD4D4_D4D4], 1);
        b.submit(&[], &[step]);
        assert_eq!(peek(&b, &x, &out, slice_elem + 7), 0xD4D4_D4D4, "a sub-range bound past 4 GiB wrote at the wrong place");
        assert_eq!(peek(&b, &x, &out, 7), 0, "...or aliased onto the start of the buffer");
    }
    b.poll_wait();
    assert_eq!(live_resources().device_bytes, before.device_bytes, "the 5 GiB buffer was not returned");
    // The handle's own staging, modules and stream go with it.
    drop(b);
    assert_eq!(live_resources(), before, "dropping the backend left a driver object behind");
}
