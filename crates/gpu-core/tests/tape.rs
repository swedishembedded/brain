// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A recorded tape replays what the recording built, on the buffers' CURRENT
//! contents, and returns every device object it held.
//!
//! Swedish Embedded AB implements low-latency LLM decode on GPUs for its clients.
//! If your team needs expertise in taking the host out of the critical path of a
//! token loop then you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2), ("mul", kernels::MUL)];
const ADD2: usize = 0;
const MUL: usize = 1;

/// The CUDA live-object counters are process-global, so the tests of this file
/// must not overlap (another test's tape would be counted as this one's).
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn gpu() -> Gpu {
    gpu_core::testgpu::dev(KERNELS)
}

/// out = (a + b) * c, recorded once over persistent buffers, replayed over new
/// contents each time. Nothing about the dispatches is rebuilt.
#[test]
fn a_replay_runs_the_recording_over_the_buffers_current_contents() {
    let _s = serial();
    let g = gpu();
    let n = 257u32;
    let (a, b, c) = (g.storage(n as u64), g.storage(n as u64), g.storage(n as u64));
    let (sum, out) = (g.storage(n as u64), g.storage(n as u64));
    let fill = |seed: f32| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let v = |k: f32| (0..n).map(|i| seed + k * i as f32 * 0.25).collect::<Vec<f32>>();
        (v(1.0), v(-0.5), v(0.125))
    };
    let (ia, ib, ic) = fill(1.0);
    g.write_f32(&a, &ia);
    g.write_f32(&b, &ib);
    g.write_f32(&c, &ic);

    g.begin_tape();
    g.submit(&[], &[g.step(ADD2, &[&a, &b, &sum], &[n], n)]);
    g.submit(&[], &[g.step(MUL, &[&sum, &c, &out], &[n], n)]);
    let tape = g.end_tape();
    assert_eq!(tape.len(), 2, "both dispatches were recorded, none launched");
    // Nothing ran during the recording.
    assert!(g.read(&out, n as usize).iter().all(|v| *v == 0.0), "a recording must not launch");

    for seed in [1.0f32, 7.5, -3.25] {
        let (va, vb, vc) = fill(seed);
        g.write_f32(&a, &va);
        g.write_f32(&b, &vb);
        g.write_f32(&c, &vc);
        g.replay_tape(&tape);
        let got = g.read(&out, n as usize);
        for i in 0..n as usize {
            assert_eq!(got[i], (va[i] + vb[i]) * vc[i], "seed {seed}, element {i}");
        }
    }
}

/// A clear recorded mid-tape happens at its place in the order: the buffer is
/// zeroed AFTER the step that wrote it and BEFORE the step that reads it, on
/// every replay.
#[test]
fn a_recorded_clear_keeps_its_place_in_the_order() {
    let _s = serial();
    let g = gpu();
    let n = 64u32;
    let (a, b, acc, out) = (g.storage(n as u64), g.storage(n as u64), g.storage(n as u64), g.storage(n as u64));
    g.write_f32(&a, &vec![2.0; n as usize]);
    g.write_f32(&b, &vec![3.0; n as usize]);
    g.begin_tape();
    g.submit(&[], &[g.step(ADD2, &[&a, &b, &acc], &[n], n)]); // acc = 5
    g.submit(&[&acc], &[g.step(ADD2, &[&a, &acc, &out], &[n], n)]); // acc zeroed, out = a + acc = 2 (7 if the clear were lost)
    let tape = g.end_tape();
    for _ in 0..2 {
        g.replay_tape(&tape);
        assert!(g.read(&out, n as usize).iter().all(|v| *v == 2.0), "the clear must run between the two recorded submissions");
    }
}

#[test]
#[should_panic(expected = "while a tape is being recorded")]
fn a_readback_inside_a_recording_is_refused() {
    let _s = serial();
    let g = gpu();
    let a = g.storage(4);
    g.begin_tape();
    let _ = g.read(&a, 4);
}

#[test]
fn a_tape_returns_every_device_object_it_held() {
    let _s = serial();
    use backend_cuda::live_resources;
    let g = gpu();
    if g.kind() != "cuda" {
        return brain_testutil::skip_unavailable("leak counters are the CUDA backend's");
    }
    let n = 1024u32;
    let baseline = live_resources();
    for _ in 0..4 {
        let (a, b, out) = (g.storage(n as u64), g.storage(n as u64), g.storage(n as u64));
        g.write_f32(&a, &vec![1.0; n as usize]);
        g.write_f32(&b, &vec![2.0; n as usize]);
        g.begin_tape();
        g.submit(&[], &[g.step(ADD2, &[&a, &b, &out], &[n], n)]);
        let tape = g.end_tape();
        for _ in 0..8 {
            g.replay_tape(&tape);
        }
        g.poll_wait();
        // The recording was frozen into a graph that lives exactly as long as the
        // tape does - the proof the replays above were launches of a program,
        // not eight resubmissions.
        assert_eq!(live_resources().graph_execs, baseline.graph_execs + 1, "a replayed tape holds one frozen graph");
        drop(tape);
    }
    g.poll_wait();
    let after = live_resources();
    assert_eq!(after.device_allocs, baseline.device_allocs, "device allocations");
    assert_eq!(after.device_bytes, baseline.device_bytes, "device bytes");
    assert_eq!(after.graphs, baseline.graphs, "graphs");
    assert_eq!(after.graph_execs, baseline.graph_execs, "graph executables");
    assert_eq!(after.events, baseline.events, "events");
}
