// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain perf roofline` - one comprehensive, cross-accelerator hardware
//! compute-capacity report: what can this machine's GPU(s), NPU and CPU
//! actually do, measured, never a datasheet guess.
//!
//! This is the raw-capacity answer, distinct from every other command that
//! sounds similar: `brain perf flops` prices ONE model's forward/backward pass;
//! `brain perf` measures empirical serving latency/throughput; `brain bench`
//! asks whether an architecture learns at all, no hardware axis; `brain
//! devices` just enumerates GPUs, with no timing. `roofline` is what all of
//! those either read (`flops`' own roof line) or could read, in one place.
//!
//! Every measurement is delegated to the engine that already owns it -
//! `gpu_core::roof` for the GPU, `npu::roofline` for the NPU, and
//! `backend_cpu::roofline` for the CPU - this module is purely the
//! streamed, self-contained rendering over all three, following
//! `crate::tree`'s "every leaf line is self-contained and greppable"
//! convention (a genuine tree does not fit a flat accelerator list, so this
//! is a small dedicated flat formatter in that same spirit, not a reuse of
//! `crate::tree` itself).
//!
//! Streamed, not buffered: GPU (fastest, most likely present) prints first,
//! then NPU (slowest to conclude "not present"), then CPU (always fast) -
//! so a GPU number lands on screen promptly regardless of how long the
//! NPU probe takes to degrade cleanly.
//!
//! Swedish Embedded AB builds hardware-aware inference tooling for embedded
//! and edge-AI teams. If your team needs one trustworthy, cross-accelerator
//! answer to "what can this box actually do" - not a vendor datasheet - you
//! can procure our services by sending an email to info@swedishembedded.com.

use serde_json::Value;

use crate::args::Args;

const USAGE: &str = "\
usage: brain perf roofline [gpu|npu|cpu] [--reprofile] [--kernels] [--json]

  Prints every accelerator's MEASURED raw hardware compute capacity - what
  this machine can do, model-independent. Streamed as each accelerator's
  measurement completes: GPU first (fastest, most likely present), then NPU
  (slowest to conclude \"not present\"), then CPU (always fast, always
  available) - so a GPU number lands well under 10s regardless of how long
  the NPU probe takes to degrade cleanly. With no scope, every accelerator's
  every supported dtype is reported; `gpu`/`npu`/`cpu` restrict the report
  to just that one section.

  gpu / npu / cpu   only that accelerator's section (same row format)
  --reprofile       force a fresh measurement instead of each engine's own
                     cache-first path. GPU: bypasses gpu_core::roof's
                     in-memory + on-disk cache. NPU: a NO-OP -
                     npu::roofline::measure has no cache of its own to
                     bypass, it always re-measures. CPU always measures
                     fresh regardless (it is fast enough not to need one).
  --kernels         after the GPU rows, list which native kernel candidate brain
                     chose per op and shape class on this device, its achieved
                     bandwidth and fraction of the measured roofline, and the
                     classes under 60% of it (the device's gap list). The choice is
                     measured on first use and cached per device and build;
                     `--kernel <slot>=<candidate>` pins one for A/B runs.
  --json            emit an array of row objects (same fields as the plain
                     rows) instead of plain text
";

pub fn run_roofline(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return 0;
    }
    let mut a = Args::new(args);
    let reprofile = a.take_flag("--reprofile");
    let json = a.take_flag("--json");
    let kernels = a.take_flag("--kernels");
    let scope = a.positional();
    if let Some(s) = &scope {
        if !matches!(s.as_str(), "gpu" | "npu" | "cpu") {
            eprintln!("brain perf roofline: unknown scope {s:?}\n{USAGE}");
            return 2;
        }
    }
    a.finish();

    let want_gpu = matches!(scope.as_deref(), None | Some("gpu"));
    let want_npu = matches!(scope.as_deref(), None | Some("npu"));
    let want_cpu = matches!(scope.as_deref(), None | Some("cpu"));

    if !json {
        println!("{}", header());
    }

    let mut rows: Vec<Row> = Vec::new();
    if want_gpu {
        for r in gpu_rows(reprofile) {
            if !json {
                println!("{}", r.plain_line());
            }
            rows.push(r);
        }
    }
    if want_npu {
        for r in npu_rows() {
            if !json {
                println!("{}", r.plain_line());
            }
            rows.push(r);
        }
    }
    if want_cpu {
        for r in cpu_rows() {
            if !json {
                println!("{}", r.plain_line());
            }
            rows.push(r);
        }
    }

    if kernels && want_gpu && !json {
        println!("\n{}", gpu_core::native_selection_report_for_default_device());
    }

    if json {
        let arr = Value::Array(rows.iter().map(Row::to_json).collect());
        println!("{}", serde_json::to_string_pretty(&arr).unwrap_or_default());
    }
    0
}

/// What a row's rate/bandwidth absence actually MEANS - distinct facts that
/// must never collapse into one blank/zero: a device that plain doesn't
/// exist ([`Status::NoHardware`]) is not the same fact as one that exists
/// but does not support this dtype ([`Status::Unsupported`]), which is not
/// the same fact as a metric this tool simply has not measured yet
/// ([`Status::NotMeasured`]), which is not the same fact as a device that
/// exists and claims support but whose probe itself failed
/// ([`Status::Unprobeable`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Measured,
    Unsupported,
    NotMeasured,
    NoHardware,
    Unprobeable,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Measured => "measured",
            Status::Unsupported => "unsupported",
            Status::NotMeasured => "not measured",
            Status::NoHardware => "no hardware",
            Status::Unprobeable => "unprobeable",
        }
    }
}

/// One self-contained row: one measured quantity of one accelerator. Every
/// field a plain line needs is here, so the line and the JSON row are built
/// from the exact same data and can never silently drift apart.
struct Row {
    /// e.g. `gpu0`, `npu0`, `cpu` - always present, always the row's own
    /// leading token, so a piped `| grep gpu0` line is complete on its own.
    accelerator: String,
    /// Human device name/description, or `-` for a whole-accelerator
    /// placeholder row (no real device to name).
    device: String,
    /// What the row measures (`fp32`, `int8-tc`, `hbm-copy`, ...). `None` only
    /// for a whole-accelerator placeholder row (no hardware, or a probe failure
    /// before anything could be attempted).
    dtype: Option<&'static str>,
    status: Status,
    /// The measured value in `rate_unit` (a rate of arithmetic, or of bytes for
    /// a memory or link row). `None` unless `status == Measured` - never a
    /// fabricated number standing in for an absence.
    rate: Option<f64>,
    rate_unit: Option<&'static str>,
    /// The same value as `rate` on a memory or link row (GB/s); `None` on a
    /// compute row.
    bandwidth_gbs: Option<f64>,
    /// The vendor's published peak for this quantity, in `rate_unit`; `None`
    /// where the device is not in the published table or has no such path.
    peak: Option<f64>,
    /// The share of `peak` that counts as validated, percent.
    target_pct: Option<f64>,
    /// How the value was measured: the kernel or probe, its shape, threads.
    method: Option<String>,
    note: Option<String>,
}

/// The share of its published peak a compute row is held to, percent: a
/// well-tiled GEMM on current silicon reaches well over half of the dense
/// datasheet rate, which assumes the boost clock under full load.
const COMPUTE_TARGET_PCT: f64 = 60.0;
/// The share of its published peak a memory or link row is held to, percent.
const MEMORY_TARGET_PCT: f64 = 70.0;

impl Row {
    /// A row for a quantity that was measured.
    fn measured(accelerator: &str, device: &str, dtype: &'static str, value: f64, unit: &'static str, peak: Option<f64>, method: String) -> Row {
        let memory = unit == "GB/s";
        Row {
            accelerator: accelerator.into(),
            device: device.into(),
            dtype: Some(dtype),
            status: Status::Measured,
            rate: Some(value),
            rate_unit: Some(unit),
            bandwidth_gbs: memory.then_some(value),
            peak,
            target_pct: Some(if memory { MEMORY_TARGET_PCT } else { COMPUTE_TARGET_PCT }),
            method: Some(method),
            note: None,
        }
    }

    /// A row for a quantity that was not measured, saying why.
    fn absent(accelerator: &str, device: &str, dtype: Option<&'static str>, status: Status, unit: Option<&'static str>, why: String) -> Row {
        Row {
            accelerator: accelerator.into(),
            device: device.into(),
            dtype,
            status,
            rate: None,
            rate_unit: unit,
            bandwidth_gbs: None,
            peak: None,
            target_pct: None,
            method: None,
            note: Some(why),
        }
    }

    fn with_note(mut self, note: Option<String>) -> Row {
        self.note = note;
        self
    }

    /// The measured value as a percentage of the published peak.
    fn pct_of_peak(&self) -> Option<f64> {
        Some(100.0 * self.rate? / self.peak.filter(|p| *p > 0.0)?)
    }

    /// Whether the measured value reaches its target share of the published peak.
    fn passes(&self) -> Option<bool> {
        Some(self.pct_of_peak()? >= self.target_pct?)
    }

    fn plain_line(&self) -> String {
        let dtype = self.dtype.unwrap_or("-");
        let unit = self.rate_unit.unwrap_or("");
        let value = match self.rate {
            Some(v) => format!("{} {unit}", sig(v)),
            None => self.status.as_str().to_string(),
        };
        let peak = self.peak.map(|p| format!("{} {unit}", sig(p))).unwrap_or_else(|| "-".into());
        let pct = self.pct_of_peak().map(|p| if p < 10.0 { format!("{p:.1}%") } else { format!("{p:.0}%") }).unwrap_or_else(|| "-".into());
        let mut line = format!("{:<6} {:<26} {:<10} {:>16} {:>16} {:>6}", self.accelerator, self.device, dtype, value, peak, pct);
        if let Some(m) = &self.method {
            line.push_str("  ");
            line.push_str(m);
        }
        if let Some(note) = &self.note {
            line.push_str(" # ");
            line.push_str(note);
        }
        line
    }

    fn to_json(&self) -> Value {
        serde_json::json!({
            "kind": "device",
            "accelerator": self.accelerator,
            "device": self.device,
            "dtype": self.dtype,
            "status": self.status.as_str(),
            "rate": self.rate,
            "rate_unit": self.rate_unit,
            "bandwidth_gbs": self.bandwidth_gbs,
            "peak": self.peak,
            "pct_of_peak": self.pct_of_peak(),
            "target_pct": self.target_pct,
            "pass": self.passes(),
            "method": self.method,
            "note": self.note,
            "line": self.plain_line(),
        })
    }
}

/// The column header of the plain rows.
fn header() -> String {
    format!("{:<6} {:<26} {:<10} {:>16} {:>16} {:>6}  how", "accel", "device", "metric", "measured", "published peak", "peak")
}

/// `v` to three significant figures (no fractional digits past them, none at all from 1000 up).
fn sig(v: f64) -> String {
    match v.abs() {
        a if a >= 100.0 => format!("{v:.0}"),
        a if a >= 10.0 => format!("{v:.1}"),
        _ => format!("{v:.2}"),
    }
}

// -------------------------------------------------------------------- gpu --

/// Every GPU `--device` selects, each dtype it supports - a multi-GPU box gets
/// a row set per selected card, and a card outside the selection is never
/// opened (on a shared node it belongs to someone else).
fn gpu_rows(reprofile: bool) -> Vec<Row> {
    let devs = gpu_core::devices::schedulable_gpus();
    if devs.is_empty() {
        return vec![Row::absent(
            "gpu",
            "-",
            None,
            Status::NoHardware,
            None,
            "no GPU selected (no physical GPUs, or --device names none; a software rasteriser may still serve --device gpu)".into(),
        )];
    }

    let mut rows = Vec::new();
    for d in devs {
        let accelerator = format!("gpu{}", d.index);
        let gpu = gpu_core::Gpu::new_on(d, &[]);
        let device = format!("{} [{}]", d.identity.name, gpu.kind());
        let published = gpu_core::published_peak::lookup(&d.identity.name);
        let peak = |m: gpu_core::published_peak::Metric| published.and_then(|p| p.of(m));
        let roofs = if reprofile { gpu_core::roof::reprofile(&gpu) } else { gpu_core::roof::ensure(&gpu) };
        use gpu_core::published_peak::Metric;
        match roofs {
            Some(r) => {
                rows.push(Row::measured(&accelerator, &device, "fp32", f64::from(r.gflops) / 1e3, "TFLOP/s", peak(Metric::Fp32), "FMA chain probe (roof_fma)".into()));
                rows.push(match r.int8_gops {
                    Some(v) => Row::measured(&accelerator, &device, "int8-dp4a", f64::from(v) / 1e3, "TOPS", None, "packed-int8 dot probe (roof_dp4a), CUDA cores".into()),
                    None => Row::absent(&accelerator, &device, Some("int8-dp4a"), Status::Unsupported, Some("TOPS"), "this GPU has no measured packed-int8 dot-product path".into()),
                });
                rows.push(match r.f16_gflops {
                    Some(v) => Row::measured(&accelerator, &device, "f16", f64::from(v) / 1e3, "TFLOP/s", None, "f16 FMA chain probe".into()),
                    None => Row::absent(&accelerator, &device, Some("f16"), Status::Unsupported, Some("TFLOP/s"), "native f16 has not been verified fast on this device".into()),
                });
                rows.push(Row::measured(&accelerator, &device, "mem-triad", f64::from(r.gbs), "GB/s", peak(Metric::Memory), "portable axpy triad".into()));
            }
            None => rows.push(Row::absent(
                &accelerator,
                &device,
                None,
                Status::Unprobeable,
                None,
                "roofline unavailable on this device/backend (--roof-probe off, or an unprobeable backend)".into(),
            )),
        }
    }
    rows
}

// -------------------------------------------------------------------- npu --

/// Staged degradation mirroring `devices_cli::print_npus`: device node
/// present is a different fact from OpenVINO reporting the NPU usable, and a
/// caller needs to be able to tell "no NPU on this machine" apart from
/// "there is one but it cannot be reached right now".
fn npu_rows() -> Vec<Row> {
    let npu_nodes = gpu_core::Inventory::probe().npus;
    if npu_nodes == 0 {
        return vec![Row::absent("npu0", "-", None, Status::NoHardware, None, "no NPU device node found (expected /dev/accel/accel* or /dev/rknpu)".into())];
    }

    match npu::openvino::available_devices() {
        Ok(devs) if devs.iter().any(|d| d == "NPU" || d.starts_with("NPU.")) => match npu::roofline::measure(npu::openvino::NpuDevice::Npu) {
            Some(r) => {
                let has = |cap: &str| r.capabilities.iter().any(|c| c.eq_ignore_ascii_case(cap));
                vec![
                    npu_dtype_row(&r.device_name, "fp16", r.fp16_gops, has("FP16")),
                    npu_dtype_row(&r.device_name, "int8", r.int8_gops, has("INT8")),
                ]
            }
            None => vec![Row::absent(
                "npu0",
                "-",
                None,
                Status::Unprobeable,
                None,
                "device node present and OpenVINO reports NPU, but the roofline probe could not characterise it".into(),
            )],
        },
        Ok(devs) => vec![Row::absent(
            "npu0",
            "-",
            None,
            Status::NoHardware,
            None,
            format!(
                "device node present but OpenVINO does NOT report NPU (available_devices: {}) - likely missing host \
                 NPU firmware (/lib/firmware/intel/vpu on the HOST, not any container); see scripts/build/setup-npu-runtime.sh",
                devs.join(", ")
            ),
        )],
        Err(e) => vec![Row::absent("npu0", "-", None, Status::NoHardware, None, format!("device node present but OpenVINO could not be queried: {e}"))],
    }
}

/// One NPU dtype row: `measured` when the probe returned a number,
/// `not measured` when the device claims the capability but the probe still
/// came back `None` (a real failure worth surfacing, not the same fact as
/// never claiming the capability at all), `unsupported` when the device's
/// own `OPTIMIZATION_CAPABILITIES` never claimed it.
fn npu_dtype_row(device_name: &str, dtype: &'static str, measured: Option<f32>, claims_support: bool) -> Row {
    match measured {
        Some(v) => Row::measured("npu0", device_name, dtype, f64::from(v), "GOP/s", None, "OpenVINO matmul probe".into()),
        None if claims_support => Row::absent(
            "npu0",
            device_name,
            Some(dtype),
            Status::NotMeasured,
            Some("GOP/s"),
            format!("device advertises {} but the roofline probe did not return a number", dtype.to_uppercase()),
        ),
        None => Row::absent("npu0", device_name, Some(dtype), Status::Unsupported, Some("GOP/s"), format!("device does not advertise {} in OPTIMIZATION_CAPABILITIES", dtype.to_uppercase())),
    }
}

// -------------------------------------------------------------------- cpu --

/// The CPU's silicon (SIMD FMA on one pinned thread a core), its memory
/// (STREAM copy and triad, one thread a core), and brain's own conv kernel
/// mix, each beside the published peak where the CPU is known. A few seconds.
fn cpu_rows() -> Vec<Row> {
    let r = backend_cpu::roofline::measure();
    let id = backend_cpu::roofline::identify();
    let device = match id.core_name() {
        Some(core) => format!("{core} x{}", r.threads),
        None => format!("CPU x{}", r.threads),
    };
    let peak_flops = r.peak.as_ref().map(|p| p.fp32_gflops);
    let peak_mem = r.peak.as_ref().and_then(|p| p.memory_gbs);
    let source = r.peak.as_ref().map(|p| format!("peak: {}", p.source));
    let threads = format!("{} threads, one pinned to each allowed core", r.threads);
    let stream_set = format!("{} MiB working set", r.working_set_bytes >> 20);
    vec![
        Row::measured("cpu", &device, "fp32", f64::from(r.fma_gflops), "GFLOP/s", peak_flops, format!("{} FMA chains, {threads}", r.simd)).with_note(source),
        Row::measured("cpu", &device, "fp32-conv", f64::from(r.conv_gflops), "GFLOP/s", peak_flops, "brain's conv2d over the yolov8n mix, the backend's pool".into()),
        Row::measured("cpu", &device, "mem-copy", f64::from(r.copy_gbs), "GB/s", peak_mem, format!("STREAM copy, {threads}, {stream_set}")),
        Row::measured("cpu", &device, "mem-triad", f64::from(r.triad_gbs), "GB/s", peak_mem, format!("STREAM triad, {threads}, {stream_set}")),
    ]
}
