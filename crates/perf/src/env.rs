// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The result fingerprint: *what hardware and what build produced this number.*
//!
//! A performance number without its fingerprint cannot be compared, and this
//! suite exists to compare across models and hardware. The load-bearing field is
//! [`Env::adapter_software`]: a machine with no real GPU still serves
//! `--device gpu` through a software rasteriser (llvmpipe / lavapipe /
//! SwiftShader), and a run like that reported as a "GPU result" is worse than no
//! result at all.
//!
//! Everything here is read from the OS at run time — nothing is configured, so a
//! fingerprint can never drift out of sync with the machine that produced it.

use serde_json::{json, Value};

/// The machine + build a result was produced on.
#[derive(Clone, Debug)]
pub struct Env {
    pub commit: Option<String>,
    pub dirty: bool,
    /// The device the caller asked for (`cpu`/`gpu`/`vulkan`/`npu`).
    pub device_requested: String,
    /// The backend that was actually resolved (`wgpu`/`cpu`/`vulkan`).
    pub backend: String,
    /// The wgpu adapter description, when a wgpu backend was built.
    pub adapter: Option<String>,
    /// True when that adapter is a software rasteriser — i.e. not a real GPU.
    pub adapter_software: Option<bool>,
    pub cpu_model: Option<String>,
    pub cpu_cores: usize,
    pub ram_gb: Option<f64>,
    pub os: String,
    pub build: &'static str,
    /// Environment flags that change performance, captured so a run is reproducible.
    pub flags: Vec<(String, Option<String>)>,
}

/// Flags that materially change how fast brain runs. Recorded on every result.
const TRACKED_FLAGS: &[&str] =
    &["BRAIN_DEVICE", "BRAIN_PROFILE", "BRAIN_NO_FASTCONV", "MOE_SKIP_GPU_TESTS", "RAYON_NUM_THREADS"];

impl Env {
    /// Capture the current environment. Call this *after* the backend has been
    /// built, so the adapter is known — before that, `adapter` reads `None`.
    pub fn capture(device_requested: &str) -> Env {
        let (adapter, adapter_software) = match gpu_core::adapter_info() {
            Some((desc, sw)) => (Some(desc), Some(sw)),
            None => (None, None),
        };
        let (commit, dirty) = git_head();
        Env {
            commit,
            dirty,
            device_requested: device_requested.to_string(),
            backend: gpu_core::backend_name().to_string(),
            adapter,
            adapter_software,
            cpu_model: cpu_model(),
            cpu_cores: cpu_cores(),
            ram_gb: ram_gb(),
            os: os_string(),
            build: if cfg!(debug_assertions) { "debug" } else { "release" },
            flags: TRACKED_FLAGS.iter().map(|f| (f.to_string(), std::env::var(f).ok())).collect(),
        }
    }

    /// True when this result must not be presented as a hardware-accelerated
    /// number. Reported prominently rather than hidden.
    pub fn is_software_gpu(&self) -> bool {
        self.adapter_software.unwrap_or(false)
    }

    /// A short human label for tables: `gpu[0] Tesla`, `cpu[4 core(s)]`, or
    /// `gpu llvmpipe (software)`.
    ///
    /// `device_requested` is already the *resolved* compute set when the CLI
    /// filled it in (e.g. `gpu[0,1] + cpu[48 core(s)]`), so the core count is
    /// only appended when the label would otherwise say nothing about the CPU.
    pub fn label(&self) -> String {
        let resolved = self.device_requested.contains('[');
        let base = if resolved {
            self.device_requested.clone()
        } else {
            format!("{}/{}c", self.device_requested, self.cpu_cores)
        };
        match (&self.adapter, self.adapter_software) {
            (Some(a), Some(true)) => {
                let short = a.split_whitespace().next().unwrap_or(a);
                format!("{base} {short} (software)")
            }
            (Some(a), _) => {
                let short = a.split_whitespace().next().unwrap_or(a);
                format!("{base} {short}")
            }
            _ => base,
        }
    }

    /// The axes that must match for two results to be comparable. `compare`
    /// warns on every one of these that differs.
    pub fn comparison_axes(&self) -> Vec<(&'static str, String)> {
        vec![
            // The resolved schedulable-compute set, e.g. "gpu[0,1] + cpu[48 core(s)]".
            // Two runs that used different hardware sets are not comparable, and
            // that is invisible from the backend name alone.
            ("device", self.device_requested.clone()),
            ("backend", self.backend.clone()),
            ("adapter", self.adapter.clone().unwrap_or_else(|| "-".into())),
            ("cpu_cores", self.cpu_cores.to_string()),
            ("build", self.build.to_string()),
        ]
    }

    pub fn to_json(&self) -> Value {
        let flags: serde_json::Map<String, Value> = self
            .flags
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().map(Value::from).unwrap_or(Value::Null)))
            .collect();
        json!({
            "commit": self.commit.clone().map(Value::from).unwrap_or(Value::Null),
            "dirty": self.dirty,
            "device": self.device_requested,
            "backend": self.backend,
            "adapter": self.adapter.clone().map(Value::from).unwrap_or(Value::Null),
            "adapter_is_software": self.adapter_software.map(Value::from).unwrap_or(Value::Null),
            "cpu": {
                "model": self.cpu_model.clone().map(Value::from).unwrap_or(Value::Null),
                "cores": self.cpu_cores,
            },
            "ram_gb": self.ram_gb.map(Value::from).unwrap_or(Value::Null),
            "os": self.os,
            "build": self.build,
            "flags": Value::Object(flags),
        })
    }
}

fn git_head() -> (Option<String>, bool) {
    let head = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok();
    let commit = head.and_then(|o| {
        if o.status.success() {
            Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
        } else {
            None
        }
    });
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    (commit, dirty)
}

fn cpu_model() -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            return rest.split(':').nth(1).map(|s| s.trim().to_string());
        }
    }
    None
}

fn cpu_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn ram_gb() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: f64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            return Some(crate::stats::r3(kb / 1024.0 / 1024.0));
        }
    }
    None
}

fn os_string() -> String {
    let rel = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into());
    format!("{} {}", std::env::consts::OS, rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_fills_the_machine_fields() {
        let e = Env::capture("cpu");
        assert!(e.cpu_cores >= 1);
        assert_eq!(e.device_requested, "cpu");
        // These come from the OS, so they must be present on any Linux CI box.
        #[cfg(target_os = "linux")]
        {
            assert!(e.ram_gb.is_some(), "ram must be readable from /proc/meminfo");
            assert!(e.os.starts_with("linux"));
        }
    }

    #[test]
    fn json_keeps_unmeasured_fields_null_not_zero() {
        let e = Env {
            commit: None,
            dirty: false,
            device_requested: "cpu".into(),
            backend: "cpu".into(),
            adapter: None,
            adapter_software: None,
            cpu_model: None,
            cpu_cores: 4,
            ram_gb: None,
            os: "linux".into(),
            build: "release",
            flags: vec![],
        };
        let j = e.to_json();
        assert!(j["adapter"].is_null());
        assert!(j["adapter_is_software"].is_null(), "unknown must not read as 'real GPU'");
        assert!(j["ram_gb"].is_null());
        assert_eq!(j["cpu"]["cores"], 4);
    }

    #[test]
    fn software_adapter_is_flagged_and_labelled() {
        let mut e = Env::capture("gpu");
        e.adapter = Some("llvmpipe (LLVM 20.1.2, 256 bits) (Cpu, Vulkan)".into());
        e.adapter_software = Some(true);
        assert!(e.is_software_gpu());
        assert!(e.label().contains("software"), "label must not hide a software rasteriser");
    }
}
