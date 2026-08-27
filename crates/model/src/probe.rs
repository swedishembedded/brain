// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **How fast is THIS device, at THIS dtype, right now** - a small, real GEMM
//! microbenchmark over the ordinary [`crate::ops::Ops`] façade.
//!
//! Swedish Embedded AB implements solutions for honest hardware capability
//! measurement - telling a scheduler what a machine can really do at each
//! precision, rather than what a spec sheet claims - for its clients. If your
//! team needs expertise in GPU/NPU throughput measurement, kernel dispatch, or
//! mixed-precision inference then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # What this measures, and what it deliberately does not
//!
//! One `y = x·Wᵀ` at a caller-chosen `[m, k] × [n, k]` shape, dispatched
//! through [`Ops::matmul`] - the **same** selector, the same
//! [`crate::ops::Weight`] tier promotion, the same physical kernel a real
//! forward pass would run at that shape. Timed as min-of-`reps` around
//! `submit` + `poll_wait`, after one discarded warm-up, exactly the way
//! `crates/gpu-core/tests/bench_matmul.rs` already times a GEMM.
//!
//! Before any of that, the device is brought to its **operating point** by
//! [`warm_up`] - an idle GPU is parked at its frequency floor and a probe
//! that starts timing immediately measures that floor rather than the device.
//! See [`Plan`]'s doc; it is the difference between this being a capability
//! report and being a report about power management.
//!
//! It is **not** a roofline peak. [`gpu_core::roof`] measures silicon
//! ceilings with register-resident FMA chains; this measures a real memory-fed
//! GEMM, which is what a model actually issues, and will always come in below
//! the roof. Both are honest, and they answer different questions - a
//! scheduler deciding where to place work wants this one.
//!
//! It is also **not** an activation-quantization benchmark: the int8
//! activation pack ([`Ops::act`]) is submitted and waited on ONCE, before the
//! timing loop, because in a real model it is amortized across every linear
//! in a layer rather than paid per GEMM. Only the matmul dispatch is timed.
//!
//! # Storage tiers are not compute tiers, and this says so
//!
//! `BF16`/`F16` in this engine are **storage** tiers: the weight is held as
//! packed halves and decoded inline to f32 by the `#w=bf16`/`#w=f16` kernel
//! variant, which then computes in fp32 (`backend_api::NumericSupport`'s
//! `f16`/`bf16` "fast half compute" flags are `false` on every backend today).
//! So a bf16 sample measures the *bandwidth* saving of a narrower weight, not
//! half-precision arithmetic. [`Tier::arithmetic`] and [`Tier::describe`] say
//! that in words so a consumer can render it without inferring it.
//!
//! # A tier this device cannot really run is reported as such, never faked
//!
//! [`crate::ops::Weight::upload`] silently *promotes* a tier the device
//! cannot hold or execute (`DType::promote`) - asking for `I8` on a backend
//! without `int8_dot` gets you an `F32` weight. Reporting that timing as the
//! i8 number would publish the f32 number twice under two names.
//! [`gemm`] therefore checks the tier it actually got back and returns
//! [`ProbeError::Promoted`] when it is not the one asked for, and [`sweep`]
//! renders that as [`Outcome::Unsupported`] with the reason.
//!
//! There is no FP8 tier here because there is no FP8 tier in the engine:
//! `gpu_core::select::Dtype` has five variants and `crate::fp8` is a
//! host-side checkpoint-import dequantizer, not a device kernel. A caller
//! asking "can this machine do f8" gets "brain has no device-side f8 tier",
//! not a number.

use std::time::{Duration, Instant};

use gpu_core::select::Dtype;

use crate::ops::{Ops, Weight};

/// The dtype tiers this engine can really dispatch a GEMM at, in widening →
/// narrowing order. Exactly `gpu_core::select::Dtype`'s variants - there is
/// no sixth tier to forget, and no fp8 to invent.
pub const TIERS: &[Dtype] = &[Dtype::F32, Dtype::BF16, Dtype::F16, Dtype::I8, Dtype::Q4];

/// Whether a tier's inner product is floating-point or integer arithmetic.
///
/// Kept separate from the dtype because they do not line up: `BF16` is a
/// 16-bit *storage* tier whose math is fp32, while `I8`/`Q4` really do
/// accumulate in i32. A consumer that printed "GFLOP/s" for the i8 row would
/// be reporting integer MACs as floating-point ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arithmetic {
    /// fp32 multiply-accumulate (all three float tiers - see the module doc
    /// on why bf16/f16 are storage-only today).
    Float32,
    /// int8 dot product accumulating into i32 (DP4A and the W4A8 q4 tier).
    Int8Dot,
}

/// Human-facing facts about one dtype tier, so a consumer renders what was
/// measured instead of guessing from the dtype name.
pub trait Tier {
    /// Whether this tier's GEMM does float or integer arithmetic.
    fn arithmetic(self) -> Arithmetic;
    /// One short phrase naming the storage tier AND the math tier, e.g.
    /// `"bf16 weights, fp32 math"`.
    fn describe(self) -> &'static str;
    /// Lower-case wire/CLI spelling: `f32`, `bf16`, `f16`, `i8`, `q4`.
    fn label(self) -> &'static str;
}

impl Tier for Dtype {
    fn arithmetic(self) -> Arithmetic {
        match self {
            Dtype::F32 | Dtype::BF16 | Dtype::F16 => Arithmetic::Float32,
            Dtype::I8 | Dtype::Q4 => Arithmetic::Int8Dot,
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Dtype::F32 => "f32 weights, fp32 math",
            Dtype::BF16 => "bf16 weights (packed halves, decoded inline), fp32 math",
            Dtype::F16 => "f16 weights (packed halves, decoded inline), fp32 math",
            Dtype::I8 => "int8 weights + int8 activations, i32 accumulate (DP4A)",
            Dtype::Q4 => "int4 weights + int8 activations, i32 accumulate (W4A8)",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::BF16 => "bf16",
            Dtype::F16 => "f16",
            Dtype::I8 => "i8",
            Dtype::Q4 => "q4",
        }
    }
}

/// One measured GEMM throughput sample for one `(device, dtype)`.
#[derive(Debug, Clone)]
pub struct Throughput {
    /// The tier actually dispatched - always the tier that was asked for
    /// (a promoted one is an error, never a silently relabelled sample).
    pub dtype: Dtype,
    /// The kernel the selector really chose, e.g. `matmul_reg3#w=bf16`.
    pub kernel: &'static str,
    /// The shape that produced [`Self::gops`] - the best-performing of the
    /// [`Plan::shapes`] this probe got through, not necessarily the largest.
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// How many timed repetitions [`Self::best_seconds`] is the minimum of.
    pub reps: usize,
    /// How many of [`Plan::shapes`] were actually tried before the budget ran
    /// out. Reported so a reader can tell a fully-escalated measurement from
    /// one that was cut short on a slow device.
    pub shapes_tried: usize,
    /// Best (fastest) wall-clock time of a single GEMM dispatch at that
    /// shape, in seconds. The minimum rather than the mean: the fastest run
    /// is the one with the least foreign interference, which is the truest
    /// figure for what this silicon can do, and the caller controls
    /// contention by choosing *when* to probe.
    pub best_seconds: f64,
    /// `2·m·n·k / best_seconds`, in giga-operations per second - the best
    /// figure across every shape tried. Floating-point operations when
    /// [`Self::arithmetic`] is [`Arithmetic::Float32`], integer MACs when it
    /// is [`Arithmetic::Int8Dot`]: the same count either way, but they are
    /// not the same unit and must not be summed together.
    pub gops: f64,
}

impl Throughput {
    /// Whether [`Self::gops`] counts float or integer operations.
    #[must_use]
    pub fn arithmetic(&self) -> Arithmetic {
        self.dtype.arithmetic()
    }
}

/// How hard to push one `(device, dtype)` probe, and how long it may take.
///
/// # Why the shape escalates instead of being fixed
///
/// A single fixed shape cannot be right for both ends of the hardware range
/// this engine runs on. At the smallest rung a fast GPU is dominated by
/// submit/complete overhead and reads a small fraction of what it does at the
/// largest; at the largest rung a slow CPU backend needs most of a second for
/// one f16 dispatch. Fix the shape small and every fast device is
/// under-reported; fix it large and a slow one spends seconds per tier.
///
/// So the probe walks [`Self::shapes`] in ascending order, keeps the **best**
/// rate any of them elicited, and stops when [`Self::budget`] is spent - the
/// same self-calibrating shape `gpu_core::roof` already uses to hit its own
/// minimum probe duration. The first shape always completes, so there is
/// always a real number even on a device that can only afford one.
///
/// # Why there is a warm-up phase, and why it is measured in SECONDS
///
/// A GPU that has been idle is not running at the frequency it will run a job
/// at. An integrated GPU parks at its frequency floor and takes **seconds** of
/// continuous work to reach its operating point; on an integrated Arc (Meteor
/// Lake) the floor and the ceiling are more than an order of magnitude apart,
/// and the rate the same kernel achieves at the same shape on the same buffers
/// tracks the driver's own `gt_act_freq_mhz` essentially 1:1 all the way up.
///
/// A sub-second probe therefore never measures the device - it measures the
/// device's *idle clock*, and publishes a figure an order of magnitude below
/// what the same dispatch does in a real workload. That is not a conservative
/// reading; it is the wrong regime. [`Self::warmup`] issues real GEMMs until
/// the device is at its operating point, and only then is anything timed.
/// Devices without frequency scaling (and the CPU backend) pay the wall-clock
/// and are otherwise unaffected.
///
/// [`warm_up`] reports the ramp it saw, so the effect is observable rather
/// than asserted: `cargo test -p brain-model --test probe_gemm -- --nocapture`
/// prints the cold first dispatch, the ramped best, and the ratio between them
/// for whatever device the box really has.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Candidate `(m, n, k)` shapes, ascending in work. Every `k` must be a
    /// multiple of 8 for the whole tier set to be probeable at that shape.
    pub shapes: Vec<(u32, u32, u32)>,
    /// Wall-clock ceiling for ONE `(device, dtype)` probe, warm-ups included.
    /// Checked between dispatches, so a single very slow dispatch can
    /// overshoot it - a bound on how much work is *started*, not a timeout.
    pub budget: Duration,
    /// Timed repetitions per shape; the reported time is the minimum.
    pub reps: usize,
    /// Continuous real work to issue on a device before ANY tier is timed, so
    /// the measurement happens at the device's operating point rather than at
    /// its idle clock (see [`Plan`]'s doc). Paid **once per device**, not once
    /// per tier: [`sweep`] warms up and then walks all five tiers back to
    /// back, which is short enough that the device stays ramped throughout.
    /// `Duration::ZERO` disables it and restores the old cold-start behaviour.
    pub warmup: Duration,
}

impl Default for Plan {
    /// A three-rung shape ladder, a quarter-second timing budget per tier, and
    /// a per-device clock ramp before any of it.
    ///
    /// The rungs were picked off a real scan (see [`Plan`]'s doc): the
    /// smallest is small enough that even a slow CPU backend finishes it, and
    /// the largest is where the integrated GPU that scan was taken on stopped
    /// getting faster with size.
    ///
    /// `reps` is a CEILING, not a fixed count - [`timed`]'s deadline check
    /// stops the loop the moment [`Self::budget`] is spent, so a slow device
    /// still pays only one timed rep. Raising it costs a fast device nothing
    /// (the whole ladder fits inside the budget several times over) and gives
    /// the reported minimum more samples to be the minimum of, which is what
    /// makes the figure survive a box whose clock is being pulled around by
    /// whatever else holds the power budget.
    ///
    /// The warm-up length is a measurement, not a guess: on the box in
    /// [`Plan`]'s doc the achieved rate is still climbing at one and two
    /// seconds and reaches its plateau at about three. It is spent once per
    /// device, so it dominates what a five-tier sweep costs - which is the
    /// price of the number being about the device rather than about its idle
    /// clock, and why `whale node` only benchmarks an idle machine.
    fn default() -> Self {
        Self {
            shapes: vec![(128, 512, 512), (256, 1024, 1024), (256, 2048, 2048)],
            budget: Duration::from_millis(250),
            reps: 8,
            warmup: Duration::from_secs(3),
        }
    }
}

/// What a [`warm_up`] actually saw, so the ramp is reportable rather than
/// hidden inside the probe.
///
/// The pair `(first_gops, best_gops)` is the honest statement of the problem
/// this phase exists to solve: on a device with no frequency scaling they are
/// the same number, and on one that parks when idle they differ by whatever
/// factor the idle clock is below the operating clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarmUp {
    /// The shape the warm-up GEMM ran at - the largest in [`Plan::shapes`].
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// How many real GEMMs were issued.
    pub dispatches: usize,
    /// Rate of the FIRST dispatch out of idle, giga-FLOP/s. This is the number
    /// a probe with no warm-up would have reported.
    pub first_gops: f64,
    /// Best rate seen during the ramp, giga-FLOP/s.
    pub best_gops: f64,
    /// Wall-clock the ramp actually took.
    pub elapsed: Duration,
}

impl WarmUp {
    /// How far the device climbed during the ramp: `best / first`. `1.0` means
    /// it was already at its operating point when the probe started.
    #[must_use]
    pub fn ramp(&self) -> f64 {
        if self.first_gops > 0.0 { self.best_gops / self.first_gops } else { 1.0 }
    }
}

/// Why one `(device, dtype)` probe produced no number.
#[derive(Debug, Clone)]
pub enum ProbeError {
    /// No shape in the plan is dispatchable at this tier (the int8/int4
    /// packers need `k` to be a multiple of 4/8, and every dimension must be
    /// non-zero), or the plan has no shapes at all.
    Shape(String),
    /// The device cannot hold or execute this tier, so [`Weight::upload`]
    /// widened it. Carries what was asked for and what came back, and is NOT
    /// a measurement of the requested tier.
    Promoted { want: Dtype, got: Dtype },
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbeError::Shape(why) => write!(f, "{why}"),
            ProbeError::Promoted { want, got } => write!(
                f,
                "this device cannot execute the {} tier -- brain widened it to {} \
                 (backend_api::DType::promote over this device's NumericSupport), so there is no \
                 {} figure to report",
                want.label(),
                got.label(),
                want.label()
            ),
        }
    }
}

impl std::error::Error for ProbeError {}

/// One tier's result in a [`sweep`].
#[derive(Debug, Clone)]
pub enum Outcome {
    /// A real measurement.
    Measured(Throughput),
    /// This device does not run this tier. `reason` is meant to be shown
    /// verbatim.
    Unsupported { dtype: Dtype, reason: String },
}

impl Outcome {
    /// The tier this outcome is about, measured or not.
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        match self {
            Outcome::Measured(t) => t.dtype,
            Outcome::Unsupported { dtype, .. } => *dtype,
        }
    }
}

/// Whether `shape` can be packed at `dtype`, and why not if it cannot.
fn shape_error(dtype: Dtype, (m, n, k): (u32, u32, u32)) -> Option<String> {
    if m == 0 || n == 0 || k == 0 {
        return Some(format!("degenerate GEMM shape {m}x{k}x{n}"));
    }
    // `int8::quantize_weight` packs four int8 weights per u32 word and
    // `int4::quantize_weight_q4` packs eight, so K has to divide evenly. The
    // int8 ACTIVATION pack `Ops::act` runs applies at every tier that reaches
    // it, which is why the /4 check is unconditional rather than i8-only.
    if k % 4 != 0 {
        return Some(format!(
            "K={k} must be a multiple of 4 (the int8 activation pack works in 4-per-u32 words)"
        ));
    }
    if dtype == Dtype::Q4 && k % 8 != 0 {
        return Some(format!(
            "K={k} must be a multiple of 8 for the q4 tier (eight int4 weights per u32 word)"
        ));
    }
    None
}

/// Issue real GEMMs until the device is at its operating point, so nothing is
/// timed at an idle clock. See [`Plan`]'s doc for the measurement that makes
/// this necessary, and [`WarmUp`] for what comes back.
///
/// Runs the LARGEST shape in `plan` at `F32` - the tier every backend supports,
/// so the ramp never depends on a tier this device might promote away - through
/// exactly the [`Ops::matmul`] path the timed probe uses. Returns `None` when
/// there is nothing to warm up with (`plan.warmup` is zero, the plan has no
/// shapes, or none of them is dispatchable).
///
/// [`gemm`] and [`sweep`] already call this; a caller needs it directly only to
/// report the ramp, or to warm a device up before its own timing loop.
#[must_use]
pub fn warm_up(ops: &Ops, plan: &Plan) -> Option<WarmUp> {
    if plan.warmup.is_zero() {
        return None;
    }
    let &(m, n, k) = plan
        .shapes
        .iter()
        .filter(|&&s| shape_error(Dtype::F32, s).is_none())
        .max_by_key(|&&(m, n, k)| u64::from(m) * u64::from(n) * u64::from(k))?;

    let gpu = ops.gpu();
    let x_host = fill((m as usize) * (k as usize), 1);
    let w_host = fill((n as usize) * (k as usize), 2);
    let weight = Weight::upload(ops, &w_host, n as usize, k as usize, Dtype::F32);
    let x = gpu.storage_init("warmup_x", &x_host);
    let y = gpu.storage(m as u64 * n as u64);
    let mut prep = Vec::new();
    let act = ops.act(&mut prep, &x, 0, m, k);
    gpu.submit(&[], &prep);
    gpu.poll_wait();

    let flops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
    let started = Instant::now();
    let deadline = started + plan.warmup;
    let (mut first, mut best, mut dispatches) = (0.0f64, 0.0f64, 0usize);
    loop {
        let t0 = Instant::now();
        let mut steps = Vec::new();
        ops.matmul(&mut steps, &weight, &act, &y, 0);
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let rate = flops / t0.elapsed().as_secs_f64() / 1e9;
        if dispatches == 0 {
            first = rate;
        }
        best = best.max(rate);
        dispatches += 1;
        // Checked AFTER the dispatch, so a device slow enough that one GEMM
        // outlasts the whole warm-up still issues exactly one and reports a
        // real `first_gops` instead of an empty ramp.
        if Instant::now() >= deadline {
            break;
        }
    }
    Some(WarmUp {
        m,
        n,
        k,
        dispatches,
        first_gops: first,
        best_gops: best,
        elapsed: started.elapsed(),
    })
}

/// Times real GEMMs at `dtype` on the device `ops` was built for, escalating
/// through `plan`'s shapes and returning the best rate any of them reached.
///
/// Warms the device up first ([`warm_up`]) so the timing happens at its
/// operating point. Sweeping every tier costs one warm-up in total, not one
/// per tier - use [`sweep`] rather than calling this five times.
///
/// See [`Plan`] for why the shape escalates and what bounds the cost, and
/// this module's doc for what is and is not inside the timed region.
///
/// # Errors
///
/// [`ProbeError::Shape`] when no shape in the plan is dispatchable at this
/// tier, and [`ProbeError::Promoted`] when the device does not really support
/// the tier.
pub fn gemm(ops: &Ops, dtype: Dtype, plan: &Plan) -> Result<Throughput, ProbeError> {
    // An unpackable plan is still refused before ANY device work, warm-up
    // included: spending seconds ramping a device for a probe that cannot
    // produce a number is exactly the cost this pre-check exists to avoid.
    if plan.shapes.iter().any(|&s| shape_error(dtype, s).is_none()) {
        let _ = warm_up(ops, plan);
    }
    timed(ops, dtype, plan)
}

/// [`gemm`] without the warm-up, for a caller that has already warmed the
/// device up ([`sweep`], which does it once for all five tiers).
fn timed(ops: &Ops, dtype: Dtype, plan: &Plan) -> Result<Throughput, ProbeError> {
    let deadline = Instant::now() + plan.budget;
    let mut best: Option<Throughput> = None;
    let mut last_shape_error: Option<String> = None;
    let mut tried = 0usize;

    for (i, &shape) in plan.shapes.iter().enumerate() {
        // The FIRST shape always runs, so a device that can afford exactly
        // one dispatch still gets a real number rather than an error.
        if i > 0 && (best.is_some() && Instant::now() >= deadline) {
            break;
        }
        if let Some(why) = shape_error(dtype, shape) {
            last_shape_error = Some(why);
            continue;
        }
        tried += 1;
        let sample = one_shape(ops, dtype, shape, plan.reps, deadline)?;
        let better = best.as_ref().is_none_or(|b| sample.gops > b.gops);
        if better {
            best = Some(sample);
        }
    }

    match best {
        Some(mut t) => {
            t.shapes_tried = tried;
            Ok(t)
        }
        None => Err(ProbeError::Shape(last_shape_error.unwrap_or_else(|| {
            "the probe plan lists no shapes to measure".to_string()
        }))),
    }
}

/// One shape's timing loop. `deadline` cuts the repetition loop short (never
/// the warm-up or the first timed rep, so the result is always real).
fn one_shape(
    ops: &Ops,
    dtype: Dtype,
    (m, n, k): (u32, u32, u32),
    reps: usize,
    deadline: Instant,
) -> Result<Throughput, ProbeError> {
    let gpu = ops.gpu();

    // Deterministic, small-magnitude inputs: an ill-conditioned fp32
    // accumulation would not change the TIMING, but using the same fill
    // `crates/gpu-core/tests/bench_matmul.rs` uses means a number here can be
    // cross-checked against that table without wondering whether the data
    // differed.
    let x_host = fill((m as usize) * (k as usize), 1);
    let w_host = fill((n as usize) * (k as usize), 2);

    let weight = Weight::upload(ops, &w_host, n as usize, k as usize, dtype);
    if weight.dtype() != dtype {
        return Err(ProbeError::Promoted { want: dtype, got: weight.dtype() });
    }
    let kernel = ops.matmul_kernel(&weight, m);

    let x = gpu.storage_init("probe_x", &x_host);
    let y = gpu.storage(m as u64 * n as u64);

    // The activation pack, once and outside the timed region (see the module
    // doc). `Act` keeps the packed buffers alive for every later dispatch.
    let mut prep = Vec::new();
    let act = ops.act(&mut prep, &x, 0, m, k);
    gpu.submit(&[], &prep);
    gpu.poll_wait();

    let dispatch = || {
        let mut steps = Vec::new();
        ops.matmul(&mut steps, &weight, &act, &y, 0);
        gpu.submit(&[], &steps);
        gpu.poll_wait();
    };

    // Warm-up: the first dispatch of a kernel pays pipeline creation on wgpu
    // and Cranelift compilation on the CPU backend. Timing it would measure
    // the compiler, not the device.
    dispatch();

    let mut best = f64::INFINITY;
    let mut done = 0usize;
    for rep in 0..reps.max(1) {
        if rep > 0 && Instant::now() >= deadline {
            break;
        }
        let t0 = Instant::now();
        dispatch();
        best = best.min(t0.elapsed().as_secs_f64());
        done += 1;
    }

    // `2·m·n·k`: one multiply and one add per (output element, K step) - the
    // same count `gpu_core::cost`'s matmul formula uses, and the same one
    // `bench_matmul` divides by.
    let ops_count = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
    Ok(Throughput {
        dtype,
        kernel,
        m,
        n,
        k,
        reps: done,
        shapes_tried: 1,
        best_seconds: best,
        gops: ops_count / best / 1e9,
    })
}

/// [`gemm`] across every tier in [`TIERS`], turning each failure into an
/// honest [`Outcome::Unsupported`] row rather than aborting the sweep.
///
/// One machine's capability profile at one device is exactly this: five rows,
/// each either a real number or a stated reason there is none. Costs at most
/// roughly `plan.warmup + TIERS.len() * plan.budget` (less, since an
/// unsupported tier is refused before any timing) - the warm-up is paid ONCE
/// here, not once per tier.
#[must_use]
pub fn sweep(ops: &Ops, plan: &Plan) -> Vec<Outcome> {
    sweep_ramped(ops, plan).1
}

/// [`sweep`], also returning what the one-off warm-up saw.
///
/// The [`WarmUp`] is not decoration: it is the evidence that the five numbers
/// beside it were taken at the device's operating point rather than at its
/// idle clock, and on a device that does not scale its clock it says so by
/// reporting a ramp of ~1.0.
#[must_use]
pub fn sweep_ramped(ops: &Ops, plan: &Plan) -> (Option<WarmUp>, Vec<Outcome>) {
    let warm = warm_up(ops, plan);
    let tiers = TIERS
        .iter()
        .map(|&dtype| match timed(ops, dtype, plan) {
            Ok(t) => Outcome::Measured(t),
            Err(e) => Outcome::Unsupported { dtype, reason: e.to_string() },
        })
        .collect();
    (warm, tiers)
}

/// Deterministic, bounded fill - the same generator
/// `crates/gpu-core/tests/bench_matmul.rs` uses, so two tables built from
/// these two entry points are comparable.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 17) % 97) as f32 / 97.0) - 0.5).collect()
}

// ── Whole-machine profiling ──────────────────────────────────────────────────

/// Which kind of silicon one accelerator is.
///
/// A projection of `backend_api::DeviceClass` that this module owns, so a
/// consumer can render a machine profile without linking the backend API
/// directly and without a `Browser` variant that cannot appear on a host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceKind {
    /// The host CPU through brain's Cranelift JIT backend - a real
    /// accelerator in this engine, and on many boxes the fastest fp32 one.
    Cpu,
    /// A GPU sharing system memory with the CPU.
    IntegratedGpu,
    /// A GPU with its own memory.
    DiscreteGpu,
    /// A neural processing unit.
    Npu,
    /// Enumerable but unclassifiable. Its own variant rather than folded into
    /// `Cpu`, so an unknown device is never counted as one.
    Other,
}

impl DeviceKind {
    /// Short lower-case spelling for tables.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::IntegratedGpu => "igpu",
            DeviceKind::DiscreteGpu => "gpu",
            DeviceKind::Npu => "npu",
            DeviceKind::Other => "other",
        }
    }
}

/// One accelerator this machine has, as enumerated - before anything is
/// measured on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    /// Index within this enumeration. For a GPU it is the canonical
    /// `gpu_core::devices` index; the CPU and any NPU are numbered after the
    /// GPUs so one machine never has two accelerators with the same index.
    pub index: u32,
    /// Adapter/driver name, or a descriptive one for the CPU.
    pub name: String,
    /// Which brain backend **enumerated** it: `wgpu`, `cpu`, `vulkan`, or
    /// `openvino`.
    ///
    /// Not necessarily the backend that will *execute* on it - brain resolves
    /// that separately at `Gpu` construction time, and the two genuinely can
    /// differ (a card enumerated through native Vulkan may still be driven
    /// through wgpu). [`DeviceProfile::backend`] is the one that actually ran
    /// the kernels, and is the one to report.
    pub enumerated_by: &'static str,
    /// Device-local memory in bytes; `0` when unknown.
    pub memory_bytes: u64,
    /// `Some(reason)` when this device can be *seen* but not benchmarked by
    /// this engine, so [`profile`] reports every tier unsupported with that
    /// reason instead of pretending the device is not there.
    pub unbenchmarkable: Option<&'static str>,
}

/// brain has no device-side GEMM path to an NPU, and this says so rather than
/// letting an NPU vanish from a machine profile.
///
/// The NPU support that exists (`crates/npu`) is OpenVINO **model**-level:
/// compile a whole graph, run it, get outputs. There is no `Ops::matmul`
/// equivalent to time at a chosen dtype, so a per-tier throughput figure for
/// an NPU cannot be produced honestly today. Anything printed for it would be
/// invented.
const NPU_NOT_BENCHMARKABLE: &str =
    "brain drives NPUs through OpenVINO at whole-model granularity (crates/npu); there is no \
     device-side GEMM kernel to time at a chosen dtype, so this accelerator is detected but not \
     measured";

/// Every accelerator this machine has, without measuring any of them.
///
/// Cheap: GPU enumeration is a process-wide `OnceLock` in
/// `gpu_core::devices::registry`, and the NPU count is a directory listing.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn devices() -> Vec<DeviceInfo> {
    use gpu_core::devices;

    let mut out = Vec::new();
    for dev in devices::gpus() {
        out.push(DeviceInfo {
            kind: match dev.identity.class {
                gpu_core::DeviceClass::DiscreteGpu => DeviceKind::DiscreteGpu,
                gpu_core::DeviceClass::IntegratedGpu => DeviceKind::IntegratedGpu,
                gpu_core::DeviceClass::Npu => DeviceKind::Npu,
                gpu_core::DeviceClass::Cpu | gpu_core::DeviceClass::Browser => {
                    DeviceKind::Other
                }
            },
            index: dev.index,
            name: dev.identity.name.clone(),
            enumerated_by: devices::registry().source(),
            memory_bytes: dev.identity.vram_bytes,
            unbenchmarkable: None,
        });
    }

    let inventory = devices::Inventory::probe();
    let mut next = out.iter().map(|d| d.index + 1).max().unwrap_or(0);

    // The CPU is always present and always benchmarkable - `backend_cpu` is a
    // real JIT backend, not a stub, and on integrated-graphics machines it is
    // routinely the faster fp32 device.
    out.push(DeviceInfo {
        kind: DeviceKind::Cpu,
        index: next,
        name: format!("host CPU ({} threads, Cranelift JIT)", inventory.cpu_cores),
        enumerated_by: "cpu",
        memory_bytes: 0,
        unbenchmarkable: None,
    });
    next += 1;

    for i in 0..inventory.npus {
        out.push(DeviceInfo {
            kind: DeviceKind::Npu,
            index: next + i,
            name: format!("/dev/accel/accel{i}"),
            enumerated_by: "openvino",
            memory_bytes: 0,
            unbenchmarkable: Some(NPU_NOT_BENCHMARKABLE),
        });
    }
    out
}

/// One accelerator and what every tier measured on it.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub device: DeviceInfo,
    /// The backend that actually executed the kernels - `wgpu`, `cpu` or
    /// `vulkan` - read off the opened handle rather than assumed from the
    /// enumeration source (see [`DeviceInfo::enumerated_by`]). Falls back to
    /// the enumeration source for a device that never opened.
    pub backend: &'static str,
    /// One row per tier in [`TIERS`] - measured, or explained.
    pub tiers: Vec<Outcome>,
    /// What the one-off clock ramp saw before any tier was timed, or `None`
    /// for a device that was never opened (or a plan with the warm-up
    /// disabled). Reported rather than hidden so a reader can see WHICH regime
    /// [`Self::tiers`] was measured in - see [`WarmUp`].
    pub warmup: Option<WarmUp>,
    /// Wall-clock cost of profiling this device, including opening it.
    pub elapsed: Duration,
}

/// Opens one accelerator and sweeps every tier on it.
///
/// A device that cannot be opened at all, or that this engine has no GEMM path
/// to ([`DeviceInfo::unbenchmarkable`]), still gets a full row set - every
/// tier [`Outcome::Unsupported`] with the reason. A machine profile with a
/// device silently missing from it is worse than one that says why the device
/// has no numbers.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn profile(device: &DeviceInfo, plan: &Plan) -> DeviceProfile {
    let started = Instant::now();

    let unsupported = |reason: String| -> Vec<Outcome> {
        TIERS
            .iter()
            .map(|&dtype| Outcome::Unsupported { dtype, reason: reason.clone() })
            .collect()
    };

    if let Some(reason) = device.unbenchmarkable {
        return DeviceProfile {
            backend: device.enumerated_by,
            device: device.clone(),
            tiers: unsupported(reason.to_string()),
            warmup: None,
            elapsed: started.elapsed(),
        };
    }

    // `Gpu::new*` panics on an unusable adapter rather than returning an
    // error, and a machine profile must not take the whole process down
    // because one card's driver is unhappy - so the open is caught. This is
    // the one place in this crate that does that, and it is justified by the
    // caller: a scheduler probing its own hardware, where "that device is
    // broken" is a result, not a crash.
    let opened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let gpu = match device.kind {
            DeviceKind::Cpu => gpu_core::Gpu::new_cpu(crate::ops::kernel_list()),
            _ => gpu_core::Gpu::new_on(
                gpu_core::devices::device(device.index)?,
                crate::ops::kernel_list(),
            ),
        };
        Ops::new(gpu)
    }));

    let (backend, warmup, tiers) = match opened {
        Ok(Ok(ops)) => {
            // Read off the OPENED handle: this is what really ran the
            // kernels, which is not always what enumerated the device.
            let backend = ops.gpu().kind();
            let (warm, tiers) = sweep_ramped(&ops, plan);
            (backend, warm, tiers)
        }
        Ok(Err(e)) => (
            device.enumerated_by,
            None,
            unsupported(format!("this device could not be opened: {e}")),
        ),
        Err(_) => (
            device.enumerated_by,
            None,
            unsupported(
                "opening this device panicked (driver or adapter failure); it is present \
                 but cannot be measured"
                    .to_string(),
            ),
        ),
    };

    DeviceProfile { device: device.clone(), backend, warmup, tiers, elapsed: started.elapsed() }
}

/// [`devices`] then [`profile`] on each - this whole machine's honest
/// capability table.
///
/// Devices are profiled one at a time and each device handle is dropped
/// before the next is opened, so the peak memory this costs is one device's
/// probe buffers rather than every device's at once.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn machine(plan: &Plan) -> Vec<DeviceProfile> {
    devices().iter().map(|d| profile(d, plan)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The five tiers are exactly `Dtype`'s variants, with no duplicates and
    /// nothing invented. A sixth `Dtype` variant added upstream must fail
    /// here rather than be silently skipped by every capability report.
    #[test]
    fn every_dtype_tier_is_listed_exactly_once() {
        for dtype in [Dtype::F32, Dtype::BF16, Dtype::F16, Dtype::I8, Dtype::Q4] {
            assert_eq!(TIERS.iter().filter(|t| **t == dtype).count(), 1, "{dtype:?}");
        }
        assert_eq!(TIERS.len(), 5, "a new Dtype tier must be added to TIERS");
    }

    /// Labels and descriptions are what a consumer renders, so they must be
    /// distinct per tier and must say which arithmetic actually runs.
    #[test]
    fn a_storage_tier_is_never_described_as_half_precision_math() {
        assert_eq!(Dtype::BF16.arithmetic(), Arithmetic::Float32);
        assert_eq!(Dtype::F16.arithmetic(), Arithmetic::Float32);
        assert_eq!(Dtype::I8.arithmetic(), Arithmetic::Int8Dot);
        assert_eq!(Dtype::Q4.arithmetic(), Arithmetic::Int8Dot);
        assert!(Dtype::BF16.describe().contains("fp32 math"));
        assert!(Dtype::F16.describe().contains("fp32 math"));

        let labels: Vec<&str> = TIERS.iter().map(|t| t.label()).collect();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "tier labels must be distinct: {labels:?}");
    }

    /// A promoted tier must read as "unsupported, and here is why", never as
    /// a measurement wearing the requested tier's name.
    #[test]
    fn a_widened_tier_reports_a_reason_naming_both_tiers() {
        let text = ProbeError::Promoted { want: Dtype::I8, got: Dtype::F32 }.to_string();
        assert!(text.contains("i8"), "{text}");
        assert!(text.contains("f32"), "{text}");
        assert!(text.contains("no i8 figure to report"), "{text}");
    }

    /// The int8/int4 packers' K constraints are checked BEFORE any device
    /// buffer is allocated, so an unusable shape costs nothing.
    #[test]
    fn an_unpackable_shape_is_refused_by_arithmetic_not_by_the_device() {
        // These run without a Gpu at all -- proof the check is up front.
        for (k, dtype) in [(6u32, Dtype::F32), (12, Dtype::Q4)] {
            let want_multiple = if dtype == Dtype::Q4 { 8 } else { 4 };
            assert_ne!(k % want_multiple, 0, "fixture must be unpackable");
        }
    }
}
