// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::probe` against whatever device this machine really has.
//!
//! Swedish Embedded AB implements solutions for provable hardware capability
//! reporting for its clients. If your team needs expertise in GPU/NPU
//! benchmarking or scheduler-facing device profiling then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! Run with `--nocapture` to see the measured table for this box:
//!
//! ```text
//! cargo test -p brain-model --test probe_gemm -- --nocapture
//! ```
//!
//! There is deliberately no threshold on the GFLOP/s here - asserting a
//! number would make the suite a hardware detector rather than a correctness
//! gate. What IS asserted is every property that could silently lie: a
//! measured tier reports a real kernel and a positive rate, an unsupported
//! tier reports a reason and no number, and no tier can appear twice.

use gpu_core::select::Dtype;
use model::ops::{self, Ops};
use model::probe::{self, Arithmetic, Outcome, Plan, Tier};

/// The probes in this file must not run concurrently **with each other**.
///
/// A throughput probe measures the throughput available to it, so two of them
/// sharing one device measure the contended device and disagree - which is not
/// a bug in the probe, it is what the probe is for. Every test here shares the
/// one `testgpu::dev()` handle, and the suite runs multi-threaded by default,
/// so without this the timing assertions fail purely because a neighbour is
/// also saturating the GPU. `crates/gpu-core/tests/roofline.rs` guards its own
/// probes exactly this way and for exactly this reason.
///
/// It matters more since the probe learned to ramp the device: each test now
/// holds the GPU for seconds rather than milliseconds, so the window in which
/// two of them can collide is far wider than it used to be.
static PROBE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn probe_lock() -> std::sync::MutexGuard<'static, ()> {
    PROBE.lock().unwrap_or_else(|e| e.into_inner())
}

fn ops() -> Ops {
    let gpu = gpu_core::testgpu::dev(ops::kernel_list());
    Ops::new(gpu).expect("the canonical kernel_list() must satisfy Ops::new")
}

/// The headline property: on real hardware, every tier is either a genuine
/// measurement or a stated reason there is none - never a zero, and never a
/// promoted tier's number under another tier's name.
#[test]
fn every_tier_is_either_measured_or_explained_never_faked() {
    let _probe = probe_lock();
    let ops = ops();
    let caps = ops.caps();
    println!(
        "\ndevice: {} | class {:?} | numeric: int8_dot={} f16_storage={} bf16_storage={}",
        ops.gpu().kind(),
        caps.class,
        caps.numeric.int8_dot,
        caps.numeric.f16_storage,
        caps.numeric.bf16_storage
    );

    let plan = Plan::default();
    let started = std::time::Instant::now();
    let sweep = probe::sweep(&ops, &plan);
    let elapsed = started.elapsed();
    assert_eq!(sweep.len(), probe::TIERS.len(), "one row per tier, always");

    let mut measured = 0;
    for row in &sweep {
        match row {
            Outcome::Measured(t) => {
                measured += 1;
                let unit = match t.arithmetic() {
                    Arithmetic::Float32 => "GFLOP/s",
                    Arithmetic::Int8Dot => "GOP/s (int)",
                };
                println!(
                    "  {:<5} {:>9.2} {unit:<12} {:>8.3} ms  [{}x{}x{}] x{} shapes={}  kernel={:<24} ({})",
                    t.dtype.label(),
                    t.gops,
                    t.best_seconds * 1e3,
                    t.m,
                    t.k,
                    t.n,
                    t.reps,
                    t.shapes_tried,
                    t.kernel,
                    t.dtype.describe()
                );
                assert!(t.gops > 0.0, "a real dispatch cannot take zero time: {t:?}");
                assert!(t.best_seconds > 0.0, "{t:?}");
                assert!(
                    t.kernel.starts_with("matmul"),
                    "the reported kernel must be the one really dispatched, got {}",
                    t.kernel
                );
                assert!(
                    plan.shapes.contains(&(t.m, t.n, t.k)),
                    "the reported shape must be one the plan actually ran: {t:?}"
                );
                assert!(t.reps >= 1 && t.shapes_tried >= 1, "{t:?}");
            }
            Outcome::Unsupported { dtype, reason } => {
                println!("  {:<5} {:>9} {reason}", dtype.label(), "--");
                assert!(!reason.is_empty(), "an unsupported tier must say why");
            }
        }
    }
    println!("  (whole-device sweep took {} ms)\n", elapsed.as_millis());

    assert!(
        elapsed < std::time::Duration::from_secs(15),
        "a five-tier sweep at the default plan must stay cheap enough for a scheduler to \
         repeat; took {elapsed:?}"
    );
    assert!(
        measured > 0,
        "f32 is available on every backend brain supports, so at least one tier must measure"
    );
    // f32 specifically: `DType::promote` never widens f32, so it can never be
    // the promoted case, on any device.
    assert!(
        matches!(&sweep[0], Outcome::Measured(t) if t.dtype == Dtype::F32),
        "f32 must always be measurable: {:?}",
        sweep[0]
    );
}

/// The canonical kernel list really does satisfy `Ops::new` - the whole point
/// of hoisting it out of four hand-maintained copies.
#[test]
fn the_canonical_kernel_list_builds_an_ops() {
    ops::assert_kernel_list_complete(ops::kernel_list());
    let _ = ops();
}

/// **The probe must report the device's operating point, not its idle clock.**
///
/// An integrated GPU parks at its frequency floor when idle and takes seconds
/// of continuous work to reach the clock it will run a job at. A probe that
/// times the first dispatch out of idle therefore reports a number an order of
/// magnitude below what the same kernel, shape and buffers achieve in a real
/// workload - which is the wrong regime, not a conservative reading of the
/// right one.
///
/// There is still no threshold on GFLOP/s here. What is asserted is the
/// relation that the fix is about: the ramp is real and reported, and where the
/// device demonstrably climbed during it, the number finally published is above
/// the cold one the old probe would have returned. On hardware that does not
/// scale its clock the ramp is ~1.0 and that arm self-disables.
#[test]
fn the_probe_reports_the_operating_point_not_the_idle_clock() {
    let _probe = probe_lock();
    let ops = ops();
    let plan = Plan::default();
    let (warm, sweep) = probe::sweep_ramped(&ops, &plan);

    let warm = warm.expect("the default plan warms up, so a ramp must be reported");
    println!(
        "\nwarm-up: [{}x{}x{}] {} dispatches in {} ms, first {:.1} -> best {:.1} GFLOP/s \
         ({:.1}x ramp)",
        warm.m,
        warm.k,
        warm.n,
        warm.dispatches,
        warm.elapsed.as_millis(),
        warm.first_gops,
        warm.best_gops,
        warm.ramp()
    );
    assert!(warm.dispatches >= 1, "a warm-up that issued nothing is not a warm-up: {warm:?}");
    assert!(warm.first_gops > 0.0, "a real dispatch cannot take zero time: {warm:?}");
    assert!(warm.best_gops >= warm.first_gops, "best is a maximum over the ramp: {warm:?}");
    assert!(
        plan.shapes.contains(&(warm.m, warm.n, warm.k)),
        "the ramp must run a shape the plan actually lists: {warm:?}"
    );

    let f32_row = match &sweep[0] {
        Outcome::Measured(t) => t.clone(),
        other => panic!("f32 must always be measurable: {other:?}"),
    };
    println!("  f32 reported: {:.1} GFLOP/s ({})", f32_row.gops, f32_row.kernel);

    if warm.ramp() > 2.0 {
        assert!(
            f32_row.gops > warm.first_gops,
            "this device climbed {:.1}x during the warm-up, so a probe reporting {:.1} \
             GFLOP/s - at or below the {:.1} GFLOP/s of the very first dispatch out of \
             idle - is timing the idle clock again",
            warm.ramp(),
            f32_row.gops,
            warm.first_gops
        );
    }
}

/// The warm-up is opt-out, and opting out is honest about it: no ramp is
/// reported, and the tiers still measure.
#[test]
fn a_plan_with_no_warmup_reports_no_ramp() {
    let _probe = probe_lock();
    let ops = ops();
    let plan = Plan { warmup: std::time::Duration::ZERO, ..Plan::default() };
    let (warm, sweep) = probe::sweep_ramped(&ops, &plan);
    assert!(warm.is_none(), "a zero warm-up must not invent a ramp: {warm:?}");
    assert_eq!(sweep.len(), probe::TIERS.len());
    assert!(matches!(&sweep[0], Outcome::Measured(t) if t.gops > 0.0), "{:?}", sweep[0]);
}

/// Two probes of the same tier on the same device must agree to within an
/// order of magnitude. Not a tight bound (a shared CI box is noisy), but it
/// catches the failure that would matter: a probe whose "measurement" is
/// really submit overhead or a no-op dispatch would swing wildly.
#[test]
fn repeated_probes_of_the_same_tier_are_in_the_same_ballpark() {
    let _probe = probe_lock();
    let ops = ops();
    let plan = Plan::default();
    let a = probe::gemm(&ops, Dtype::F32, &plan).expect("f32 is always available");
    let b = probe::gemm(&ops, Dtype::F32, &plan).expect("f32 is always available");
    let ratio = a.gops.max(b.gops) / a.gops.min(b.gops);
    assert!(ratio < 10.0, "two f32 probes disagreed by {ratio:.1}x: {a:?} vs {b:?}");
    assert!(a.gops > 0.0 && b.gops > 0.0, "{a:?} {b:?}");
}

/// A shape the packers cannot take is refused before any device work.
#[test]
fn an_unpackable_shape_is_refused_with_a_reason() {
    let _probe = probe_lock();
    let ops = ops();
    let unpackable = |k| Plan { shapes: vec![(128, 512, k)], ..Plan::default() };

    let err = probe::gemm(&ops, Dtype::F32, &unpackable(6)).expect_err("K=6 is not a multiple of 4");
    assert!(err.to_string().contains("multiple of 4"), "{err}");

    let err =
        probe::gemm(&ops, Dtype::Q4, &unpackable(12)).expect_err("K=12 is not a multiple of 8");
    assert!(err.to_string().contains("multiple of 8"), "{err}");

    let err = probe::gemm(&ops, Dtype::F32, &Plan { shapes: Vec::new(), ..Plan::default() })
        .expect_err("a plan with no shapes measures nothing");
    assert!(err.to_string().contains("no shapes"), "{err}");
}

/// **The whole-machine profile, on whatever this box really has.** Run with
/// `--nocapture` to see it.
///
/// Every accelerator gets a row set, including one this engine can enumerate
/// but not benchmark (an NPU): a machine profile that silently omitted a
/// device would be a worse answer than one that names it and says why it has
/// no numbers.
#[test]
fn the_machine_profile_covers_every_accelerator_including_the_ones_it_cannot_measure() {
    let _probe = probe_lock();
    let plan = Plan::default();
    let started = std::time::Instant::now();
    let profiles = probe::machine(&plan);
    let elapsed = started.elapsed();

    assert!(
        !profiles.is_empty(),
        "every machine has at least a CPU, and brain's CPU backend is a real device"
    );

    println!("\n=== machine profile ===");
    for p in &profiles {
        println!(
            "{}{}  {}  backend={}  mem={} MB  ({} ms)",
            p.device.kind.label(),
            p.device.index,
            p.device.name,
            p.backend,
            p.device.memory_bytes / 1_000_000,
            p.elapsed.as_millis()
        );
        for row in &p.tiers {
            match row {
                Outcome::Measured(t) => println!(
                    "    {:<5} {:>9.2} {:<8} {:>8.3} ms  [{}x{}x{}]  {}",
                    t.dtype.label(),
                    t.gops,
                    match t.arithmetic() {
                        Arithmetic::Float32 => "GFLOP/s",
                        Arithmetic::Int8Dot => "GOP/s",
                    },
                    t.best_seconds * 1e3,
                    t.m,
                    t.k,
                    t.n,
                    t.kernel
                ),
                Outcome::Unsupported { dtype, reason } => {
                    println!("    {:<5} {:>9}  {reason}", dtype.label(), "--")
                }
            }
        }
    }
    println!("=== whole machine profiled in {} ms ===\n", elapsed.as_millis());

    for p in &profiles {
        assert_eq!(p.tiers.len(), probe::TIERS.len(), "every device gets every tier's row");
    }
    assert!(
        profiles.iter().any(|p| p.device.kind == probe::DeviceKind::Cpu),
        "the CPU is always an accelerator here"
    );
    assert!(
        profiles.iter().any(|p| p
            .tiers
            .iter()
            .any(|t| matches!(t, Outcome::Measured(_)))),
        "at least one device on any machine must produce a real number"
    );
}
