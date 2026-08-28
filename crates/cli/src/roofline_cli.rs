// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain roofline` - one comprehensive, cross-accelerator hardware
//! compute-capacity report: what can this machine's GPU(s), NPU and CPU
//! actually do, measured, never a datasheet guess.
//!
//! This is the raw-capacity answer, distinct from every other command that
//! sounds similar: `brain flops` prices ONE model's forward/backward pass;
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
//! so a GPU number lands on screen well under 10s regardless of how long the
//! NPU probe takes to degrade cleanly.
//!
//! Swedish Embedded AB builds hardware-aware inference tooling for embedded
//! and edge-AI teams. If your team needs one trustworthy, cross-accelerator
//! answer to "what can this box actually do" - not a vendor datasheet - you
//! can procure our services by sending an email to info@swedishembedded.com.

use serde_json::Value;

use crate::args::Args;

const USAGE: &str = "\
usage: brain roofline [gpu|npu|cpu] [--reprofile] [--json]

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
    let scope = a.positional();
    if let Some(s) = &scope {
        if !matches!(s.as_str(), "gpu" | "npu" | "cpu") {
            eprintln!("brain roofline: unknown scope {s:?}\n{USAGE}");
            return 2;
        }
    }
    a.finish();

    let want_gpu = matches!(scope.as_deref(), None | Some("gpu"));
    let want_npu = matches!(scope.as_deref(), None | Some("npu"));
    let want_cpu = matches!(scope.as_deref(), None | Some("cpu"));

    if !json {
        println!("{:<6} {:<40} {:<6} {:<20} {:<24}", "accel", "device", "dtype", "rate", "bandwidth");
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

/// One self-contained row: every field a plain line needs is here, so the
/// line and the JSON row are built from the exact same data and can never
/// silently drift apart.
struct Row {
    /// e.g. `gpu0`, `npu0`, `cpu` - always present, always the row's own
    /// leading token, so a piped `| grep gpu0` line is complete on its own.
    accelerator: String,
    /// Human device name/description, or `-` for a whole-accelerator
    /// placeholder row (no real device to name).
    device: String,
    /// `None` only for a whole-accelerator placeholder row (no hardware, or
    /// a probe failure before any dtype could even be attempted).
    dtype: Option<&'static str>,
    status: Status,
    /// GFLOP/s or GOP/s as `rate_unit` says. `None` unless `status ==
    /// Measured` - never a fabricated number standing in for an absence.
    rate: Option<f64>,
    rate_unit: Option<&'static str>,
    /// GB/s of DRAM bandwidth. `None` where genuinely not measured (NPU: no
    /// bandwidth probe exists at all today; CPU: the lifted conv-throughput
    /// methodology does not isolate one) - rendered as "not measured", never
    /// blank or zero.
    bandwidth_gbs: Option<f64>,
    note: Option<String>,
}

impl Row {
    fn plain_line(&self) -> String {
        let dtype = self.dtype.unwrap_or("-");
        let rate = match self.rate {
            Some(v) => format!("{v:.0} {}", self.rate_unit.unwrap_or("")),
            None => self.status.as_str().to_string(),
        };
        let bandwidth = match self.bandwidth_gbs {
            Some(v) => format!("{v:.0} GB/s"),
            None => match self.status {
                Status::NoHardware | Status::Unprobeable => "-".to_string(),
                _ => "not measured".to_string(),
            },
        };
        let mut line = format!("{:<6} {:<40} {:<6} {:<20} {:<24}", self.accelerator, self.device, dtype, rate, bandwidth);
        if let Some(note) = &self.note {
            line.push_str(" # ");
            line.push_str(note);
        }
        line
    }

    fn to_json(&self) -> Value {
        serde_json::json!({
            "accelerator": self.accelerator,
            "device": self.device,
            "dtype": self.dtype,
            "status": self.status.as_str(),
            "rate": self.rate,
            "rate_unit": self.rate_unit,
            "bandwidth_gbs": self.bandwidth_gbs,
            "note": self.note,
            "line": self.plain_line(),
        })
    }
}

// -------------------------------------------------------------------- gpu --

/// Every physical GPU on the machine, each dtype it supports - not just the
/// ambient/first device, so a multi-GPU box gets a row set per card.
fn gpu_rows(reprofile: bool) -> Vec<Row> {
    let devs = gpu_core::devices::registry().devices();
    if devs.is_empty() {
        return vec![Row {
            accelerator: "gpu".into(),
            device: "-".into(),
            dtype: None,
            status: Status::NoHardware,
            rate: None,
            rate_unit: None,
            bandwidth_gbs: None,
            note: Some("no physical GPUs (a software rasteriser may still serve --device gpu)".into()),
        }];
    }

    let mut rows = Vec::new();
    for d in devs {
        let accelerator = format!("gpu{}", d.index);
        let gpu = gpu_core::Gpu::new_on(d, &[]);
        let device = format!("{} [{}]", d.identity.name, gpu.kind());
        let roofs = if reprofile { gpu_core::roof::reprofile(&gpu) } else { gpu_core::roof::ensure(&gpu) };
        match roofs {
            Some(r) => {
                rows.push(Row {
                    accelerator: accelerator.clone(),
                    device: device.clone(),
                    dtype: Some("fp32"),
                    status: Status::Measured,
                    rate: Some(r.gflops as f64),
                    rate_unit: Some("GFLOP/s"),
                    bandwidth_gbs: Some(r.gbs as f64),
                    note: None,
                });
                rows.push(match r.int8_gops {
                    Some(v) => Row {
                        accelerator: accelerator.clone(),
                        device: device.clone(),
                        dtype: Some("int8"),
                        status: Status::Measured,
                        rate: Some(v as f64),
                        rate_unit: Some("GOP/s"),
                        bandwidth_gbs: Some(r.gbs as f64),
                        note: None,
                    },
                    None => Row {
                        accelerator: accelerator.clone(),
                        device: device.clone(),
                        dtype: Some("int8"),
                        status: Status::Unsupported,
                        rate: None,
                        rate_unit: Some("GOP/s"),
                        bandwidth_gbs: Some(r.gbs as f64),
                        note: Some("this GPU has no measured packed-int8 dot-product path".into()),
                    },
                });
                rows.push(match r.f16_gflops {
                    Some(v) => Row {
                        accelerator: accelerator.clone(),
                        device: device.clone(),
                        dtype: Some("f16"),
                        status: Status::Measured,
                        rate: Some(v as f64),
                        rate_unit: Some("GFLOP/s"),
                        bandwidth_gbs: Some(r.gbs as f64),
                        note: None,
                    },
                    None => Row {
                        accelerator: accelerator.clone(),
                        device: device.clone(),
                        dtype: Some("f16"),
                        status: Status::Unsupported,
                        rate: None,
                        rate_unit: Some("GFLOP/s"),
                        bandwidth_gbs: Some(r.gbs as f64),
                        note: Some("native f16 has not been verified fast on this device".into()),
                    },
                });
            }
            None => rows.push(Row {
                accelerator,
                device,
                dtype: None,
                status: Status::Unprobeable,
                rate: None,
                rate_unit: None,
                bandwidth_gbs: None,
                note: Some("roofline unavailable on this device/backend (BRAIN_NO_ROOF, or an unprobeable backend)".into()),
            }),
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
        return vec![Row {
            accelerator: "npu0".into(),
            device: "-".into(),
            dtype: None,
            status: Status::NoHardware,
            rate: None,
            rate_unit: None,
            bandwidth_gbs: None,
            note: Some("no NPU device node found (expected /dev/accel/accel*)".into()),
        }];
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
            None => vec![Row {
                accelerator: "npu0".into(),
                device: "-".into(),
                dtype: None,
                status: Status::Unprobeable,
                rate: None,
                rate_unit: None,
                bandwidth_gbs: None,
                note: Some("device node present and OpenVINO reports NPU, but the roofline probe could not characterise it".into()),
            }],
        },
        Ok(devs) => vec![Row {
            accelerator: "npu0".into(),
            device: "-".into(),
            dtype: None,
            status: Status::NoHardware,
            rate: None,
            rate_unit: None,
            bandwidth_gbs: None,
            note: Some(format!(
                "device node present but OpenVINO does NOT report NPU (available_devices: {}) - likely missing host \
                 NPU firmware (/lib/firmware/intel/vpu on the HOST, not any container); see scripts/build/setup-npu-runtime.sh",
                devs.join(", ")
            )),
        }],
        Err(e) => vec![Row {
            accelerator: "npu0".into(),
            device: "-".into(),
            dtype: None,
            status: Status::NoHardware,
            rate: None,
            rate_unit: None,
            bandwidth_gbs: None,
            note: Some(format!("device node present but OpenVINO could not be queried: {e}")),
        }],
    }
}

/// One NPU dtype row: `measured` when the probe returned a number,
/// `not measured` when the device claims the capability but the probe still
/// came back `None` (a real failure worth surfacing, not the same fact as
/// never claiming the capability at all), `unsupported` when the device's
/// own `OPTIMIZATION_CAPABILITIES` never claimed it.
fn npu_dtype_row(device_name: &str, dtype: &'static str, measured: Option<f32>, claims_support: bool) -> Row {
    let base = Row {
        accelerator: "npu0".into(),
        device: device_name.into(),
        dtype: Some(dtype),
        status: Status::Measured,
        rate: None,
        rate_unit: Some("GOP/s"),
        bandwidth_gbs: None,
        note: None,
    };
    match measured {
        Some(v) => Row { rate: Some(v as f64), status: Status::Measured, ..base },
        None if claims_support => Row {
            status: Status::NotMeasured,
            note: Some(format!("device advertises {} but the roofline probe did not return a number", dtype.to_uppercase())),
            ..base
        },
        None => Row {
            status: Status::Unsupported,
            note: Some(format!("device does not advertise {} in OPTIMIZATION_CAPABILITIES", dtype.to_uppercase())),
            ..base
        },
    }
}

// -------------------------------------------------------------------- cpu --

/// Always available, always fast (~0.2s) - no capability gating needed.
fn cpu_rows() -> Vec<Row> {
    let r = backend_cpu::roofline::measure();
    vec![Row {
        accelerator: "cpu".into(),
        device: "CPU (all cores)".into(),
        dtype: Some("fp32"),
        status: Status::Measured,
        rate: Some(r.gflops as f64),
        rate_unit: Some("GFLOP/s"),
        bandwidth_gbs: r.bandwidth_gbs.map(|v| v as f64),
        note: None,
    }]
}
