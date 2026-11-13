// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structural validation of an exported ONNX graph, plus an attempted compile
//! where OpenVINO and a device are present. One report for `brain npu check`
//! and for every sample action that exports graphs and wants to say what it
//! wrote (op histogram, whether the device accepts it).
//!
//! Swedish Embedded AB implements model deployment onto NPUs and other
//! fixed-function accelerators for its clients. If your team needs expertise in
//! finding out which operators an accelerator's compiler rejects, before a
//! network is committed to it, then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

use crate::openvino::{self, NpuConfig, NpuSession};

/// The report for the graph at `onnx_path`: node, initializer, input and output
/// counts and the operator histogram (always available), then the devices
/// OpenVINO sees and whether the graph compiles on `cfg`'s. A graph that cannot
/// be read or decoded is reported, not an error, so a caller checking several
/// graphs in one run sees every result.
pub fn structural_report(onnx_path: &str, cfg: &NpuConfig) -> String {
    let mut out = String::new();
    let bytes = match std::fs::read(onnx_path) {
        Ok(b) => b,
        Err(e) => return format!("reading {onnx_path}: {e}\n"),
    };
    match onnx::decode_model(&bytes) {
        Ok(m) => {
            let g = m.graph.unwrap_or_default();
            let mut ops: BTreeMap<&str, usize> = BTreeMap::new();
            for n in &g.node {
                *ops.entry(n.op_type.as_str()).or_insert(0) += 1;
            }
            let _ = writeln!(out, "onnx: {} nodes, {} initializers, {} inputs, {} outputs", g.node.len(), g.initializer.len(), g.input.len(), g.output.len());
            let _ = writeln!(out, "ops:");
            for (op, c) in &ops {
                let _ = writeln!(out, "  {op:<20} {c}");
            }
        }
        Err(e) => return format!("malformed ONNX: {e}\n"),
    }
    match openvino::available_devices() {
        Ok(devs) if devs.is_empty() => out.push_str("openvino: no devices (runtime not installed); compile/op-coverage check skipped\n"),
        Ok(devs) => {
            let _ = writeln!(out, "openvino devices: {devs:?}");
            match NpuSession::load(Path::new(onnx_path), cfg) {
                Ok(s) => {
                    let _ = writeln!(out, "compiled OK on {} (input {:?})", s.device(), s.input_shape());
                }
                Err(e) => {
                    let _ = writeln!(out, "compile failed: {e}");
                }
            }
        }
        Err(e) => {
            let _ = writeln!(out, "openvino unavailable: {e}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unreadable_or_malformed_graph_is_reported_not_raised() {
        let dir = std::env::temp_dir().join(format!("brain-npu-check-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = NpuConfig::default();
        let missing = dir.join("missing.onnx");
        assert!(structural_report(missing.to_str().unwrap(), &cfg).starts_with("reading "));
        let bad = dir.join("bad.onnx");
        std::fs::write(&bad, [0xff, 0xff, 0xff]).unwrap();
        assert!(structural_report(bad.to_str().unwrap(), &cfg).starts_with("malformed ONNX"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
