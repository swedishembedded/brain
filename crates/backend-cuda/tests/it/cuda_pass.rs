// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A pass - many differently shaped submissions bracketed by `begin_pass` and
//! `end_pass` - is issued to the card as ONE submission, so a decode step
//! whose layers each submit their own handful of dispatches can still be
//! captured whole and replayed.
//!
//! Swedish Embedded AB implements low-latency single-stream inference for its
//! clients. If your team needs expertise in cutting host launch cost out of a
//! token loop without changing what it computes, you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! What has to hold, and why each part is asserted:
//!
//! - the pass is replayable as one graph although its submissions are all
//!   different shapes (a per-layer submit never repeats consecutively, so
//!   without batching a graph never forms: that was measured on the real
//!   decode, where one capture happened in a whole run);
//! - program order survives the batching. A `write` or a `read` between two
//!   submissions of a pass observes exactly the dispatches submitted before it
//!   and none after, because deferral that reordered them would be a wrong
//!   number rather than a slow one.

use backend_api::Backend as _;
use backend_cuda::CudaBackend;

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const AXPY: usize = 0;
const CHAIN: usize = 32;
const N: usize = 256;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::try_new(KERNELS) {
        Ok(b) => Some(b),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA backend: {e}"));
            None
        }
    }
}

fn input() -> Vec<f32> {
    (0..N).map(|i| 1.0 + i as f32 * 0.125).collect()
}

/// One pass: `CHAIN` submissions of ONE `axpy` each, the way a layer loop
/// submits.
fn pass(b: &CudaBackend, outs: &[backend_api::DeviceBuffer], inp: &backend_api::DeviceBuffer, s: f32) {
    let params = [N as u32, s.to_bits()];
    b.begin_pass();
    for o in outs {
        b.submit(&[], &[b.step(AXPY, &[o, inp], &params, N as u32)]);
    }
    b.end_pass();
}

#[test]
fn a_pass_of_distinct_submissions_is_captured_once_and_replayed() {
    let Some(b) = backend() else { return };
    const ROUNDS: usize = 6;
    const S: f32 = 0.5;
    let inp = b.storage_init("inp", &input());
    let outs: Vec<_> = (0..CHAIN).map(|_| b.storage(N as u64)).collect();

    let before = b.launch_stats();
    for _ in 0..ROUNDS {
        pass(&b, &outs, &inp, S);
        b.poll_wait();
    }
    let after = b.launch_stats();

    assert_eq!(after.dispatches - before.dispatches, (ROUNDS * CHAIN) as u64);
    assert_eq!(after.graph_captures - before.graph_captures, 1, "the pass was not captured as one graph");
    assert!(after.graph_replays - before.graph_replays >= (ROUNDS - 2) as u64, "the pass was not replayed");
    assert_eq!(
        after.host_launches - before.host_launches,
        (2 * CHAIN) as u64,
        "the host kept launching per dispatch after the pass had a graph"
    );
    assert_eq!(
        after.graph_param_copies - before.graph_param_copies,
        1,
        "the captured graph carries more than one host-to-device parameter copy: each is a node the device \
         executes between kernels on every replay (~12 us apiece on the real decode, where 2500 of them \
         made the replayed token slower than launching one dispatch at a time)"
    );
    let want = input();
    for (j, o) in outs.iter().enumerate() {
        let got = b.read(o, N);
        for i in 0..N {
            let expect = ROUNDS as f32 * S * want[i];
            assert!((got[i] - expect).abs() <= 1e-5 * expect.abs().max(1.0), "output {j} element {i}: got {}, want {expect}", got[i]);
        }
    }
}

/// `write` between two submissions of a pass lands between them.
#[test]
fn a_write_inside_a_pass_is_ordered_after_the_submissions_before_it() {
    let Some(b) = backend() else { return };
    let inp = b.storage_init("inp", &input());
    let out = b.storage(N as u64);
    let params = [N as u32, 1.0f32.to_bits()];
    for round in 0..4 {
        b.begin_pass();
        b.submit(&[], &[b.step(AXPY, &[&out, &inp], &params, N as u32)]);
        b.write(&out, &vec![0u32; N]);
        b.submit(&[], &[b.step(AXPY, &[&out, &inp], &params, N as u32)]);
        b.end_pass();
        let got = b.read(&out, N);
        let want = input();
        for i in 0..N {
            assert!(
                (got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0),
                "round {round} element {i}: got {}, want {} - the write was not ordered between the two submissions",
                got[i],
                want[i]
            );
        }
    }
}

/// `read` inside a pass sees every submission made before it.
#[test]
fn a_read_inside_a_pass_sees_the_pending_submissions() {
    let Some(b) = backend() else { return };
    let inp = b.storage_init("inp", &input());
    let out = b.storage(N as u64);
    let params = [N as u32, 1.0f32.to_bits()];
    b.begin_pass();
    b.submit(&[], &[b.step(AXPY, &[&out, &inp], &params, N as u32)]);
    let got = b.read(&out, N);
    b.submit(&[], &[b.step(AXPY, &[&out, &inp], &params, N as u32)]);
    b.end_pass();
    let want = input();
    for i in 0..N {
        assert!((got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0), "element {i}: read {} before the second axpy, want {}", got[i], want[i]);
    }
    let after = b.read(&out, N);
    for i in 0..N {
        assert!((after[i] - 2.0 * want[i]).abs() <= 1e-5 * want[i].abs().max(1.0));
    }
}

/// A submission with clears closes the pending run first: the clear must not
/// be hoisted above dispatches submitted before it.
#[test]
fn a_clear_inside_a_pass_zeroes_after_the_earlier_submissions() {
    let Some(b) = backend() else { return };
    let inp = b.storage_init("inp", &input());
    let out = b.storage(N as u64);
    let params = [N as u32, 1.0f32.to_bits()];
    b.begin_pass();
    b.submit(&[], &[b.step(AXPY, &[&out, &inp], &params, N as u32)]);
    b.submit(&[&out], &[b.step(AXPY, &[&out, &inp], &params, N as u32)]);
    b.end_pass();
    let got = b.read(&out, N);
    let want = input();
    for i in 0..N {
        assert!((got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0), "element {i}: got {}, want {}", got[i], want[i]);
    }
}

/// Passes nest by counting: only the outermost `end_pass` issues the work.
#[test]
fn passes_nest_and_only_the_outermost_end_issues() {
    let Some(b) = backend() else { return };
    let inp = b.storage_init("inp", &input());
    let out = b.storage(N as u64);
    let params = [N as u32, 1.0f32.to_bits()];
    let s0 = b.launch_stats().host_launches;
    b.begin_pass();
    b.begin_pass();
    b.submit(&[], &[b.step(AXPY, &[&out, &inp], &params, N as u32)]);
    b.end_pass();
    assert_eq!(b.launch_stats().host_launches, s0, "the inner end_pass issued the work");
    b.end_pass();
    assert_eq!(b.launch_stats().host_launches, s0 + 1, "the outer end_pass did not issue the work");
}

/// A decode token is two shapes in alternation - the layer stack, then the
/// head after a readback - and neither repeats CONSECUTIVELY. A capture
/// trigger that only remembered the last shape never fired for either: the
/// real decode ran 2500 individual launches per token with graphs enabled.
#[test]
fn two_alternating_shapes_are_each_captured_and_replayed() {
    let Some(b) = backend() else { return };
    const ROUNDS: usize = 6;
    const S: f32 = 0.5;
    let inp = b.storage_init("inp", &input());
    let stack: Vec<_> = (0..CHAIN).map(|_| b.storage(N as u64)).collect();
    let head: Vec<_> = (0..3).map(|_| b.storage(N as u64)).collect();

    let before = b.launch_stats();
    for _ in 0..ROUNDS {
        pass(&b, &stack, &inp, S);
        // The token's readback: what separates the two shapes in time.
        let _ = b.read(&stack[0], 1);
        pass(&b, &head, &inp, S);
        let _ = b.read(&head[0], 1);
    }
    let after = b.launch_stats();
    assert_eq!(after.graph_captures - before.graph_captures, 2, "each of the two alternating shapes should be captured once");
    assert!(
        after.graph_replays - before.graph_replays >= 2 * (ROUNDS as u64 - 2),
        "only {} replays across {ROUNDS} rounds of two shapes",
        after.graph_replays - before.graph_replays
    );
    let want = input();
    for o in stack.iter().chain(&head) {
        let got = b.read(o, N);
        for i in 0..N {
            let expect = ROUNDS as f32 * S * want[i];
            assert!((got[i] - expect).abs() <= 1e-5 * expect.abs().max(1.0), "element {i}: got {}, want {expect}", got[i]);
        }
    }
}

/// Two dispatches of the SAME shape (one kernel, the same buffers) with
/// DIFFERENT parameters in one pass must each see their own: they share a
/// uniform allocation when issued one at a time, and a captured graph that
/// merged their parameter storage would run both with the last one's.
#[test]
fn two_steps_of_one_shape_with_different_parameters_keep_their_own() {
    let Some(b) = backend() else { return };
    let inp = b.storage_init("inp", &input());
    let out = b.storage(N as u64);
    for round in 0..5 {
        b.begin_pass();
        for s in [1.0f32, 4.0] {
            let params = [N as u32, s.to_bits()];
            b.submit(&[], &[b.step(AXPY, &[&out, &inp], &params, N as u32)]);
        }
        b.end_pass();
        let got = b.read(&out, N);
        let want = input();
        for i in 0..N {
            let expect = (round + 1) as f32 * 5.0 * want[i];
            assert!((got[i] - expect).abs() <= 1e-4 * expect.abs().max(1.0), "round {round} element {i}: got {}, want {expect}", got[i]);
        }
    }
    assert!(b.launch_stats().graph_replays > 0, "the pass was never replayed, so this proves nothing about replay");
}

/// A `flush` inside a pass hands the work held so far to the card WITHOUT
/// waiting, once enough is held to be worth a launch - so the host can keep
/// building the rest of the step while the card runs the first part. Without
/// it the card sat idle for the whole host-side build of a decode token
/// (~2.5 ms of a ~16): the pass was one submission, issued at the end.
///
/// The chunks must be the same chunks every token, or no graph would repeat:
/// the cut depends only on how much has been held, never on timing.
#[test]
fn a_flush_inside_a_pass_issues_chunks_that_replay() {
    let Some(b) = backend() else { return };
    const ROUNDS: usize = 5;
    const S: f32 = 0.5;
    let (first, chunk) = (backend_cuda::PASS_FIRST_FLUSH_STEPS, backend_cuda::PASS_FLUSH_STEPS);
    // A short first chunk, two full ones and a remainder.
    let n = first + 2 * chunk + chunk / 2;
    let inp = b.storage_init("inp", &input());
    let outs: Vec<_> = (0..n).map(|_| b.storage(N as u64)).collect();
    let params = [N as u32, S.to_bits()];

    let before = b.launch_stats();
    for _ in 0..ROUNDS {
        b.begin_pass();
        for o in &outs {
            b.submit(&[], &[b.step(AXPY, &[o, &inp], &params, N as u32)]);
            b.flush();
        }
        b.end_pass();
        b.poll_wait();
    }
    let after = b.launch_stats();
    // The short first chunk, two full ones and the remainder, each its own graph.
    assert_eq!(after.graph_captures - before.graph_captures, 4, "the pass was not cut into its four fixed chunks");
    assert!(after.graph_replays - before.graph_replays >= 4 * (ROUNDS as u64 - 2), "the chunks were not replayed");
    let want = input();
    for o in &outs {
        let got = b.read(o, N);
        for i in 0..N {
            let expect = ROUNDS as f32 * S * want[i];
            assert!((got[i] - expect).abs() <= 1e-5 * expect.abs().max(1.0), "element {i}: got {}, want {expect}", got[i]);
        }
    }
}

/// Outside a pass `flush` has nothing to hold and nothing to do.
#[test]
fn a_flush_outside_a_pass_is_a_no_op() {
    let Some(b) = backend() else { return };
    let before = b.launch_stats().host_launches;
    b.flush();
    assert_eq!(b.launch_stats().host_launches, before);
}

/// Submissions the caller makes on its own, outside a pass, are captured only
/// when one repeats the one before it - the original trigger. A prefill round
/// is hundreds of small submissions in a fixed order, some of which recur every
/// few layers; capturing each of those paid a graph instantiation
/// (milliseconds) apiece, and a one-row chunk round on the real 27B went from
/// 143 ms to 1473 ms when the wider window that serves passes was applied to
/// them too.
#[test]
fn alternating_submissions_outside_a_pass_are_not_captured() {
    let Some(b) = backend() else { return };
    let inp = b.storage_init("inp", &input());
    let outs: Vec<_> = (0..4).map(|_| b.storage(N as u64)).collect();
    let params = [N as u32, 0.5f32.to_bits()];
    let before = b.launch_stats();
    for _ in 0..8 {
        for o in &outs {
            b.submit(&[], &[b.step(AXPY, &[o, &inp], &params, N as u32)]);
        }
        b.poll_wait();
    }
    let after = b.launch_stats();
    assert_eq!(after.graph_captures - before.graph_captures, 0, "a submission that never repeated back to back was captured");
    assert_eq!(after.host_launches - before.host_launches, 32, "each submission should have been launched on its own");
}
