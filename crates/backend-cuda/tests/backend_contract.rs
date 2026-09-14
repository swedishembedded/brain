// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The parts of the [`backend_api::Backend`] contract a whole-model parity
//! run cannot see.
//!
//! `crates/gpt2/tests/cuda_backend_parity.rs` runs a real forward on this
//! backend and holds it to the CPU and Vulkan answers, which is the strongest
//! statement available about the arithmetic. It is a weak instrument for the
//! *plumbing*, and measurably so: with the four mutations below applied one at
//! a time - storage handed back unzeroed, a sub-range binding ignored, the
//! grid laid out at a fixed 64 rather than the kernel's own work-group size,
//! and a `write_at` offset dropped - that model test still passed every time,
//! because the shape it runs never exercises any of them. A test that cannot
//! fail is not evidence, so these four get their own.
//!
//! Swedish Embedded AB implements accelerator backends and the contract tests
//! that hold them honest for its clients. If your team needs expertise in
//! proving a device layer correct rather than assuming it, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! Skip-if-absent, and deliberately tiny: a correctness gate, never a
//! benchmark, and it must not contend with anything else resident on the card.

use backend_api::Backend as _;
use backend_cuda::CudaBackend;

/// The catalogue subset these tests drive. `gn_stats_wg` is here for one
/// reason: it is the only kernel in the generator's supported subset whose
/// `@workgroup_size` is NOT 64, so it is the only one that can tell a grid
/// laid out at the kernel's own size from one laid out at a hardcoded 64.
const KERNELS: &[(&str, &str)] = &[
    ("add2", kernels::ADD2),
    ("gn_stats_wg", kernels::GN_STATS_WG),
    ("mul", kernels::MUL),
];

const ADD2: usize = 0;
const GN_STATS_WG: usize = 1;

/// A backend to test, or the reason there is none.
fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(KERNELS) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

/// `cuMemAlloc` promises nothing about its contents. Every other backend in
/// this engine hands back zeroes (wgpu maps buffers zero-filled, the CPU
/// backend allocates a zeroed `Vec`) and model code relies on it - an
/// accumulator is allocated and then added into, never written first.
///
/// **This assertion is NOT known to have teeth**, and that is recorded here
/// rather than left to be rediscovered. It dirties a region, frees it and
/// re-allocates the same size - twenty times over, to make address reuse
/// likely rather than hoped for - and with `storage`'s explicit zeroing
/// removed it still passed on the driver this was written against, which
/// evidently scrubs a freed allocation before handing it out again. That is a
/// driver's courtesy, not an API guarantee, so the zeroing stays; but read
/// this test as a statement of the contract, not as evidence the backend
/// honours it. Do not delete the dirtying loop on the grounds that it proves
/// nothing: it is what a driver that does NOT scrub would need.
#[test]
fn storage_is_handed_back_zeroed() {
    let Some(b) = backend() else { return };
    const N: u64 = 4096;

    for round in 0..20 {
        let dirty = b.storage(N);
        b.write(&dirty, &vec![0xDEAD_BEEFu32; N as usize]);
        b.poll_wait();
        assert_eq!(b.read(&dirty, 4)[0].to_bits(), 0xDEAD_BEEF, "the dirtying write did not land");
        drop(dirty);

        let fresh = b.storage(N);
        let got = b.read(&fresh, N as usize);
        assert!(
            got.iter().all(|v| v.to_bits() == 0),
            "round {round}: new storage came back holding {} non-zero words",
            got.iter().filter(|v| v.to_bits() != 0).count()
        );
    }
}

/// `write_at` must start at the WORD offset it is given and touch nothing
/// before it - it exists so a multi-gigabyte host upload can be chunked, and
/// a chunk written at the wrong place corrupts a weight tensor silently.
#[test]
fn write_at_starts_at_the_word_offset_and_leaves_the_rest_alone() {
    let Some(b) = backend() else { return };
    let base: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let buf = b.storage_init("x", &base);

    let patch: Vec<u32> = [100.0f32, 200.0, 300.0].iter().map(|v| v.to_bits()).collect();
    b.write_at(&buf, 5, &patch);

    let got = b.read(&buf, 16);
    let mut want = base.clone();
    want[5..8].copy_from_slice(&[100.0, 200.0, 300.0]);
    assert_eq!(got, want, "write_at wrote somewhere other than word 5");
}

/// A sliced step binds `(word_offset, word_len)` of each buffer, which on this
/// API is the base address plus `4 * word_offset` and nothing else. If the
/// offset were dropped the kernel would read and write the buffer's HEAD -
/// still finite, still plausible, and wrong everywhere a tiled head or a
/// per-expert weight slice is bound.
///
/// The window is deliberately several work-groups wide rather than the one it
/// would take to make the offset point: it also makes this the test that a
/// grid covers its whole dispatch, since a grid laid out at any size larger
/// than this kernel's own 64 leaves the tail of the window uncomputed.
#[test]
fn a_sliced_step_binds_the_sub_range_and_not_the_head() {
    let Some(b) = backend() else { return };
    const N: usize = 1024;
    const OFF: u64 = 256;
    const WIN: usize = 512;

    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let c: Vec<f32> = (0..N).map(|i| 100.0 + i as f32).collect();
    let sentinel = -7.0f32;

    let ba = b.storage_init("a", &a);
    let bc = b.storage_init("c", &c);
    let out = b.storage_init("out", &vec![sentinel; N]);

    let offsets = [(OFF, WIN as u64); 3];
    let step = b.step_sliced(ADD2, &[&ba, &bc, &out], &offsets, &[WIN as u32], WIN as u32);
    b.submit(&[], &[step]);

    let got = b.read(&out, N);
    for (i, g) in got.iter().enumerate() {
        let want = if (OFF as usize..OFF as usize + WIN).contains(&i) {
            a[i] + c[i]
        } else {
            sentinel
        };
        assert_eq!(*g, want, "element {i}: a sliced step wrote outside, or at the wrong place in, its window");
    }
}

/// A block must be as wide as the kernel's OWN `@workgroup_size`.
///
/// Almost every kernel in the catalogue declares 64, so a backend that wrote
/// 64 down instead of reading it would be right almost everywhere and wrong
/// on the reductions - exactly the kernels whose `__shared__` tile and strided
/// walk are written against their own size. This runs the one kernel in the
/// generator's supported subset that declares 256, and it checks the numbers
/// rather than the launch arithmetic.
///
/// The group has to be WIDER than one wrong block for the difference to be
/// visible, which is the part that is easy to get wrong in the test rather
/// than in the code: at `m <= 64` a 64-thread block still covers every element
/// of the strided walk and still reduces the right total, because the emitter
/// zero-initialises the `__shared__` slots no lane wrote. At `m = 320` the
/// elements a 64-wide block never reaches are missing from both passes, and
/// the mean and the variance both move.
#[test]
fn a_kernel_that_is_not_64_wide_is_launched_at_its_own_workgroup_size() {
    let Some(b) = backend() else { return };
    // N=1, C=G so each group is exactly one channel of H*W elements, and
    // H*W = 320 so a group spans more than one 256-thread stride.
    const N: u32 = 1;
    const C: u32 = 4;
    const H: u32 = 16;
    const W: u32 = 20;
    const G: u32 = 4;
    const EPS: f32 = 1e-5;
    let m = (C / G * H * W) as usize;

    let x: Vec<f32> = (0..(N * C * H * W) as usize).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let bx = b.storage_init("x", &x);
    let stats = b.storage(2 * (N * G) as u64);

    let params = [N, C, H, W, G, EPS.to_bits()];
    let threads = N * G * 256;
    let step = b.step(GN_STATS_WG, &[&bx, &stats], &params, threads);
    b.submit(&[], &[step]);
    let got = b.read(&stats, 2 * (N * G) as usize);

    for g in 0..G as usize {
        let grp = &x[g * m..(g + 1) * m];
        let mean = grp.iter().sum::<f32>() / m as f32;
        let var = grp.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / m as f32;
        let rstd = 1.0 / (var + EPS).sqrt();
        assert!(
            (got[2 * g] - mean).abs() < 1e-6,
            "group {g} mean: got {}, want {mean}",
            got[2 * g]
        );
        assert!(
            (got[2 * g + 1] - rstd).abs() < 1e-4 * rstd.abs(),
            "group {g} rstd: got {}, want {rstd}",
            got[2 * g + 1]
        );
    }
}

/// Compilation is per `kind`, on first dispatch, and once.
///
/// Eager compilation of a registered catalogue is the thing this backend
/// exists NOT to do (`backend-vulkan` builds every pipeline at `Factory`
/// time, and a few hundred NVRTC invocations is minutes of cold start). A
/// regression to it is invisible except as a slow start, so it gets an
/// assertion rather than a comment.
#[test]
fn a_kernel_compiles_on_first_dispatch_and_only_once() {
    let Some(b) = backend() else { return };
    assert_eq!(b.compiled_kernel_count(), 0, "registering a catalogue compiled something");

    let a = b.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let c = b.storage_init("c", &[10.0, 20.0, 30.0, 40.0]);
    let out = b.storage(4);
    let run = || {
        let s = b.step(ADD2, &[&a, &c, &out], &[4], 4);
        b.submit(&[], &[s]);
        b.read(&out, 4)
    };

    assert_eq!(run(), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(b.compiled_kernel_count(), 1, "the dispatched kernel did not get compiled");

    assert_eq!(run(), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(b.compiled_kernel_count(), 1, "a second dispatch of the same kind compiled it again");
    // The other two registered kernels were never asked for, so they must
    // still be uncompiled - that is the whole claim.
    assert_eq!(KERNELS.len(), 3);
}

/// The host-cost counter counts HOST work, and it is a different quantity from
/// the recorded dispatch count even while the two happen to be equal.
///
/// `DeviceStats::dispatches` is a property of the model's graph: it is the same
/// number however the backend chooses to issue those steps. `host_launches` and
/// `host_nanos` are properties of how the *host* issued them, and they are what
/// any batched-submission mechanism has to move. Pinning the relation now -
/// one launch call per recorded step, and a submit that costs measurable host
/// time - is what makes a later divergence between the two readable as a
/// result rather than as a bug.
#[test]
fn the_host_cost_of_a_submit_is_counted_separately_from_its_dispatches() {
    let Some(b) = backend() else { return };
    let a = b.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
    let c = b.storage_init("c", &[10.0, 20.0, 30.0, 40.0]);
    let out = b.storage(4);

    let before = b.launch_stats();
    const ROUNDS: u64 = 8;
    const PER_SUBMIT: u64 = 3;
    for _ in 0..ROUNDS {
        let steps: Vec<_> = (0..PER_SUBMIT).map(|_| b.step(ADD2, &[&a, &c, &out], &[4], 4)).collect();
        b.submit(&[], &steps);
    }
    b.poll_wait();
    let after = b.launch_stats();

    assert_eq!(after.submits - before.submits, ROUNDS);
    assert_eq!(after.dispatches - before.dispatches, ROUNDS * PER_SUBMIT);
    assert_eq!(
        after.host_launches - before.host_launches,
        ROUNDS * PER_SUBMIT,
        "one driver launch call per recorded step is what an unbatched submission does"
    );
    assert!(
        after.host_nanos > before.host_nanos,
        "submitting {} dispatches was recorded as costing the host no time at all",
        ROUNDS * PER_SUBMIT
    );
    // The read-back proves the counted launches were real work and not a
    // counter incremented next to a launch that never happened.
    assert_eq!(b.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
}
