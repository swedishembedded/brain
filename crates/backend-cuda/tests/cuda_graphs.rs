// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A repeated submission is captured once and replayed, and a replay computes
//! the same answers as issuing every dispatch one at a time.
//!
//! Swedish Embedded AB implements accelerator runtimes and the tests that hold
//! their optimisations to the unoptimised answer for its clients. If your team
//! needs expertise in batched GPU submission without giving up correctness,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What has to be true, and why each part of it is here
//!
//! Batching submission is only worth anything if the host stops making one
//! driver call per dispatch, and it is only *allowed* if the answers do not
//! move. Those are two different claims and neither implies the other, so both
//! are asserted:
//!
//! - the host-launch count stops growing while the dispatch count keeps
//!   growing (`launch_stats`, which exists for exactly this comparison), and
//! - every buffer still holds what the unbatched path would have left in it,
//!   including after the per-submit parameters change and after the dispatch
//!   grid grows.
//!
//! `axpy` is the kernel throughout because it is the one that can tell those
//! apart. Its uniform carries a float, so a replay that re-ran the captured
//! parameters instead of this submit's would produce a wrong *value* rather
//! than a wrong size; and it accumulates into its output, so a replay that
//! silently did not execute at all leaves a number that is too small rather
//! than a number that happens to be right.
//!
//! Skip-if-absent, and deliberately tiny: a correctness gate, never a
//! benchmark, and it must not contend with anything else resident on the card.

use backend_api::Backend as _;
use backend_cuda::CudaBackend;

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const AXPY: usize = 0;

/// Dispatches per submission. Large enough that the per-launch host cost is
/// the dominant term in a submit, small enough that the whole test is a few
/// hundred microseconds of device time on a tiny buffer.
const CHAIN: usize = 64;
/// Elements per buffer - four work-groups of the kernel's own 64.
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

/// The input every `axpy` reads: distinct per element, so a dispatch that
/// wrote the wrong element is visible.
fn input() -> Vec<f32> {
    (0..N).map(|i| 1.0 + i as f32 * 0.125).collect()
}

/// One submission of `CHAIN` `axpy` steps, `out[j] += s * inp` for each output.
fn round(b: &CudaBackend, outs: &[backend_api::DeviceBuffer], inp: &backend_api::DeviceBuffer, n: usize, s: f32) {
    let params = [n as u32, s.to_bits()];
    let steps: Vec<_> = outs.iter().map(|o| b.step(AXPY, &[o, inp], &params, n as u32)).collect();
    b.submit(&[], &steps);
}

/// The whole claim in one test: a submission whose shape repeats is captured
/// once and thereafter costs the host one driver call instead of `CHAIN`, and
/// the numbers do not move.
///
/// The accumulate is what makes the second half of that checkable. `axpy`
/// reads its output, so after `ROUNDS` submissions every element must be
/// exactly `ROUNDS * s * inp[i]`; a replay that was enqueued but never ran, or
/// ran once for several replays, leaves a smaller number, and a replay that
/// ran twice leaves a bigger one.
#[test]
fn a_repeated_submission_is_captured_once_and_then_replayed() {
    let Some(b) = backend() else { return };
    const ROUNDS: usize = 6;
    const S: f32 = 0.5;

    let inp = b.storage_init("inp", &input());
    let outs: Vec<_> = (0..CHAIN).map(|_| b.storage(N as u64)).collect();

    let before = b.launch_stats();
    for _ in 0..ROUNDS {
        round(&b, &outs, &inp, N, S);
    }
    b.poll_wait();
    let after = b.launch_stats();

    assert_eq!(
        after.dispatches - before.dispatches,
        (ROUNDS * CHAIN) as u64,
        "every step of every submission is still a dispatch, whatever issued it"
    );
    assert_eq!(after.graph_captures - before.graph_captures, 1, "the same shape was captured more than once");
    assert!(
        after.graph_replays - before.graph_replays >= (ROUNDS - 2) as u64,
        "{ROUNDS} identical submissions produced only {} replays",
        after.graph_replays - before.graph_replays
    );
    // One eager round, one capturing round (a capture records the launches, so
    // it pays for them), and nothing after that.
    assert_eq!(
        after.host_launches - before.host_launches,
        (2 * CHAIN) as u64,
        "the host kept issuing per-dispatch launch calls after the graph existed"
    );

    let want = input();
    for (j, o) in outs.iter().enumerate() {
        let got = b.read(o, N);
        for i in 0..N {
            let expect = ROUNDS as f32 * S * want[i];
            assert!(
                (got[i] - expect).abs() <= 1e-5 * expect.abs().max(1.0),
                "output {j} element {i}: got {}, want {expect} - a replay did not run exactly once per submission",
                got[i]
            );
        }
    }
}

/// A replay must use the parameters of the submission being replayed, not the
/// ones that happened to be current when the graph was captured.
///
/// This is the single assumption the whole design rests on: the uniform
/// allocation is keyed WITHOUT the parameter values, so a per-token parameter
/// change reuses the same device address and the graph stays valid, and each
/// step's first graph node re-copies that step's own parameters into it. Drop
/// that copy node and this test sees every round compute with round one's
/// scale - which is still a plausible number, just the wrong one.
#[test]
fn a_replay_computes_with_this_submission_s_parameters() {
    let Some(b) = backend() else { return };
    let scales: [f32; 6] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];

    let inp = b.storage_init("inp", &input());
    let outs: Vec<_> = (0..4).map(|_| b.storage(N as u64)).collect();

    for s in scales {
        round(&b, &outs, &inp, N, s);
    }
    b.poll_wait();

    assert!(b.launch_stats().graph_replays > 0, "nothing was replayed, so this proves nothing");

    let want = input();
    let total: f32 = scales.iter().sum();
    for (j, o) in outs.iter().enumerate() {
        let got = b.read(o, N);
        for i in 0..N {
            let expect = total * want[i];
            assert!(
                (got[i] - expect).abs() <= 1e-4 * expect.abs().max(1.0),
                "output {j} element {i}: got {}, want {expect} - a replay reused the captured parameters",
                got[i]
            );
        }
    }
}

/// A dispatch grid that grows must be re-pointed inside the already
/// instantiated graph, not answered with a second graph.
///
/// This is the case a decode loop hits every token: the thread count tracks
/// the sequence length, so it grows while the shape of the submission does
/// not. Re-instantiating per position was measured to cost more than the
/// batching saves, so the requirement is stronger than "still correct" - the
/// capture count must stay at one.
#[test]
fn a_grid_that_grows_is_re_pointed_rather_than_re_instantiated() {
    let Some(b) = backend() else { return };
    // The first two are equal so a graph exists before the grid moves; every
    // later one is a different number of work-groups AND a tail that is not a
    // multiple of the kernel's 64, so a stale grid leaves elements untouched.
    let widths = [N, N, N + 64, N + 160, N + 224];
    const S: f32 = 1.5;

    let big = widths.iter().copied().max().unwrap();
    let inp_host: Vec<f32> = (0..big).map(|i| 1.0 + i as f32 * 0.0625).collect();
    let inp = b.storage_init("inp", &inp_host);
    let outs: Vec<_> = (0..4).map(|_| b.storage(big as u64)).collect();

    let before = b.launch_stats();
    for w in widths {
        round(&b, &outs, &inp, w, S);
    }
    b.poll_wait();
    let after = b.launch_stats();

    assert_eq!(
        after.graph_captures - before.graph_captures,
        1,
        "a grid change re-captured instead of re-pointing the instantiated graph"
    );
    assert!(
        after.grid_updates - before.grid_updates > 0,
        "the grid changed three times and no node was re-pointed"
    );

    for (j, o) in outs.iter().enumerate() {
        let got = b.read(o, big);
        for i in 0..big {
            let times = widths.iter().filter(|w| i < **w).count() as f32;
            let expect = times * S * inp_host[i];
            assert!(
                (got[i] - expect).abs() <= 1e-4 * expect.abs().max(1.0),
                "output {j} element {i}: got {}, want {expect} - a replayed grid covered the wrong extent",
                got[i]
            );
        }
    }
}

/// Freeing a device allocation invalidates any captured graph.
///
/// A graph node holds a device *address*. The driver may hand a freed address
/// straight back to the next allocation, so a graph that outlived a free could
/// be replayed against memory that now belongs to something else - reading and
/// writing a live tensor at full speed, with no fault and no wrong-looking
/// number until much later. The guard is coarse on purpose: any free at all,
/// not an analysis of which addresses moved.
#[test]
fn freeing_an_allocation_invalidates_the_captured_graph() {
    let Some(b) = backend() else { return };
    const S: f32 = 1.0;
    let inp = b.storage_init("inp", &input());

    let outs: Vec<_> = (0..4).map(|_| b.storage(N as u64)).collect();
    for _ in 0..3 {
        round(&b, &outs, &inp, N, S);
    }
    b.poll_wait();
    let captured = b.launch_stats();
    assert_eq!(captured.graph_captures, 1, "nothing was captured, so this proves nothing");

    // Free them, then build the same shape again. The addresses the graph
    // recorded are now the allocator's to reuse.
    drop(outs);
    let outs: Vec<_> = (0..4).map(|_| b.storage(N as u64)).collect();
    for _ in 0..3 {
        round(&b, &outs, &inp, N, S);
    }
    b.poll_wait();
    let after = b.launch_stats();
    assert_eq!(after.graph_captures, 2, "the graph survived a free of the buffers it names");

    let want = input();
    for (j, o) in outs.iter().enumerate() {
        let got = b.read(o, N);
        for i in 0..N {
            let expect = 3.0 * S * want[i];
            assert!(
                (got[i] - expect).abs() <= 1e-5 * expect.abs().max(1.0),
                "output {j} element {i}: got {}, want {expect} after re-capture",
                got[i]
            );
        }
    }
}

/// The decode loop's own shape: submit, read, submit, read.
///
/// A stream capture forbids synchronising calls inside the captured region,
/// and brain reads logits every single token - so if capture spanned anything
/// wider than one submission this pattern would either fail outright or, worse,
/// invalidate the capture and quietly fall back forever. Asserting the graph
/// still forms under a read between every submission is what makes the claim
/// "capture is confined to `submit`" a tested one rather than a comment.
#[test]
fn a_read_between_every_submission_does_not_prevent_capture() {
    let Some(b) = backend() else { return };
    const S: f32 = 0.5;
    let inp = b.storage_init("inp", &input());
    let outs: Vec<_> = (0..8).map(|_| b.storage(N as u64)).collect();

    let want = input();
    for r in 1..=5u32 {
        round(&b, &outs, &inp, N, S);
        // Exactly what a decode step does with its logits.
        let got = b.read(&outs[0], 1);
        let expect = r as f32 * S * want[0];
        assert!((got[0] - expect).abs() <= 1e-5, "round {r}: read back {}, want {expect}", got[0]);
    }
    assert!(
        b.launch_stats().graph_replays > 0,
        "no graph was ever replayed under a read-per-submission loop"
    );
}

/// Replaying a captured submission costs the host less than issuing the same
/// dispatches one at a time.
///
/// The comparison is between two handles on the same card doing the same work,
/// one with capture turned off, and it is measured in HOST time inside
/// `submit` - which never waits for the device, so this is not a device
/// benchmark and does not depend on what else is resident.
///
/// **The margin asserted is small on purpose, and that is a result rather than
/// caution.** What a replay removes is the driver calls - one per dispatch
/// plus one per parameter upload, collapsed to a single graph launch, which
/// `a_repeated_submission_is_captured_once_and_then_replayed` asserts exactly.
/// What it does not remove is everything the backend does before it can decide
/// to replay: resolving each step to its kernel and argument list, and
/// building the submission's signature to compare against the captured one.
/// Both paths pay that, it is now the larger term at this shape, and it is
/// where the next host-cost work is. Asserting a large ratio here would be
/// asserting something that is not true.
///
/// The loop drains between submissions because that is what a decode step
/// does - it reads its logits every token - and because the alternative is not
/// a measurement of this mechanism at all: a replay whose predecessor is still
/// running has to wait for the device before it may overwrite its parameter
/// staging, and what that would measure is the kernels. `staging_waits` is
/// asserted at zero so that the distinction is checked rather than assumed.
///
/// The statistic is a MEDIAN over per-submission samples, not a mean. This runs
/// on shared machines, and a single submission that loses the CPU to something
/// else adds its whole stall to a mean - which failed this exact assertion once
/// while the same test passed on its own moments later. A median of many
/// samples is unmoved by a few stalls, and both sides are measured the same
/// way.
#[test]
fn replaying_costs_the_host_less_than_launching_each_dispatch() {
    let Some(eager) = backend() else { return };
    let Some(graphed) = backend() else { return };
    let eager = eager.with_graph_capture(false);
    const ROUNDS: usize = 17;
    const S: f32 = 0.5;

    let mut ns = [0u64; 2];
    for (slot, b) in ns.iter_mut().zip([&eager, &graphed]) {
        let inp = b.storage_init("inp", &input());
        let outs: Vec<_> = (0..CHAIN).map(|_| b.storage(N as u64)).collect();
        // Warm up: compile the kernel, and on the capturing handle get past
        // the eager round and the capturing round.
        for _ in 0..3 {
            round(b, &outs, &inp, N, S);
            b.poll_wait();
        }
        let mut samples = Vec::with_capacity(ROUNDS);
        for _ in 0..ROUNDS {
            let t0 = b.launch_stats().host_nanos;
            round(b, &outs, &inp, N, S);
            // Outside the sample: `host_nanos` only counts time inside
            // `submit`, and this wait is what makes the next submission the
            // decode loop's rather than a tight loop's.
            b.poll_wait();
            samples.push(b.launch_stats().host_nanos - t0);
        }
        samples.sort_unstable();
        *slot = samples[samples.len() / 2];
    }

    assert_eq!(eager.launch_stats().graph_captures, 0, "capture was supposed to be off on this handle");
    assert!(graphed.launch_stats().graph_replays >= ROUNDS as u64, "the graphed handle did not replay");
    assert_eq!(
        graphed.launch_stats().staging_waits,
        0,
        "a replay waited for the device, so this measured kernels rather than host cost"
    );
    println!(
        "cuda_graphs: {CHAIN} dispatches per submission - host time {} us per submission launching each, \
         {} us per submission replaying",
        ns[0] as f64 / 1000.0,
        ns[1] as f64 / 1000.0
    );
    // Observed repeatedly between 0.62 and 0.78 of the unbatched cost; 0.9 is
    // the bar, which leaves room for the box and still fails if replaying ever
    // stops being cheaper.
    assert!(
        ns[1] * 10 < ns[0] * 9,
        "replaying cost {} ns of host time against {} ns for the same dispatches issued one at a time",
        ns[1],
        ns[0]
    );
}
