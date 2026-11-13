// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The OpenVINO run options an action offers when it compiles a graph for the
//! Intel NPU: which device, the performance hint, the compile cache, turbo,
//! CPU/GPU fallback and profiling - one declaration, so every NPU action of
//! every sample takes the same flags with the same defaults and bounds.
//!
//! Swedish Embedded AB implements model deployment to Intel NPUs for its
//! clients. If your team needs expertise in compiling and tuning graphs for
//! OpenVINO then you can procure our services by sending an email to
//! info@swedishembedded.com.

use capability::{Invocation, ParamSpec, ParamType};
use serde_json::json;

use crate::openvino::{NpuConfig, NpuDevice, PerfHint};

const DEVICES: [&str; 4] = ["npu", "cpu", "gpu", "auto"];
const HINTS: [&str; 2] = ["latency", "throughput"];

/// The params of an NPU action.
pub fn params() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new("ov_device", ParamType::Enum(DEVICES.map(String::from).to_vec()), "OpenVINO device the graph compiles for").default(json!("npu")),
        ParamSpec::new("hint", ParamType::Enum(HINTS.map(String::from).to_vec()), "latency- or throughput-oriented compilation").default(json!("latency")),
        ParamSpec::new("allow_fallback", ParamType::Bool, "fall back to CPU/GPU when the requested device is absent").default(json!(false)),
        ParamSpec::new("cache_dir", ParamType::Str, "directory the compiled graphs are cached in, so a later run skips the compile").host_resolved(),
        ParamSpec::new("turbo", ParamType::Bool, "NPU turbo mode").default(json!(false)),
        ParamSpec::new("profile", ParamType::Bool, "collect per-node profiling").default(json!(false)),
    ]
}

/// The compile configuration an invocation names.
pub fn config(inv: &Invocation) -> Result<NpuConfig, String> {
    let device = inv.get_str("ov_device").unwrap_or_else(|| "npu".into());
    Ok(NpuConfig {
        device: NpuDevice::parse(&device).ok_or_else(|| format!("ov_device: expected npu|cpu|gpu|auto, got {device:?}"))?,
        perf_hint: if inv.get_str("hint").as_deref() == Some("throughput") { PerfHint::Throughput } else { PerfHint::Latency },
        cache_dir: inv.get_str("cache_dir").filter(|d| !d.is_empty()).map(std::path::PathBuf::from),
        turbo: inv.get_bool("turbo").unwrap_or(false),
        profiling: inv.get_bool("profile").unwrap_or(false),
        allow_fallback: inv.get_bool("allow_fallback").unwrap_or(false),
        ..NpuConfig::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_the_npu_with_no_fallback_and_a_named_device_is_taken() {
        let c = config(&Invocation::new()).unwrap();
        assert_eq!((c.device, c.perf_hint, c.allow_fallback, c.turbo, c.profiling), (NpuDevice::Npu, PerfHint::Latency, false, false, false));
        let c = config(&Invocation::new().set("ov_device", json!("cpu")).set("hint", json!("throughput")).set("allow_fallback", json!(true)).set("cache_dir", json!("/c"))).unwrap();
        assert_eq!((c.device, c.perf_hint, c.allow_fallback), (NpuDevice::Cpu, PerfHint::Throughput, true));
        assert_eq!(c.cache_dir.as_deref(), Some(std::path::Path::new("/c")));
        assert!(config(&Invocation::new().set("ov_device", json!("tpu"))).is_err());
    }
}
