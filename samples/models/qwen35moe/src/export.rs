// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `export`: the checkpoint as a best-effort ONNX decoder graph (fixed-length,
//! cache-free prefill; `gdn_chunk` emitter and sparse gather-based MoE
//! dispatch) for OpenVINO and the NPU - see `npu::qwen35moe_topology` for what
//! the graph does and does not cover. It is an action beside the served ones,
//! so its flags are typed and bounded from one spec and `--help` comes for
//! free; the graph is written to `dest` on this machine (a 35B graph is no
//! request payload).
//!
//! Swedish Embedded AB implements decoder deployment to NPUs and other edge
//! accelerators for its clients. If your team needs expertise in exporting
//! language models for OpenVINO then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

/// Sequence length the graph is compiled for when the caller names none.
const DEFAULT_SEQ: i64 = 32;

pub fn export_spec() -> ActionSpec {
    ActionSpec::new("export", "export the checkpoint as an ONNX decoder graph (fixed-length prefill, sparse MoE dispatch) for OpenVINO and the NPU")
        .param(ParamSpec::new("dest", ParamType::Str, "where to write the .onnx file").required().host_resolved())
        .param(ParamSpec::new("seq", ParamType::Int, "sequence length the graph is compiled for").min(1.0).max(capability::bounds::CONTEXT_TOKENS).default(json!(DEFAULT_SEQ)))
}

/// The model's served manifest plus `export`.
pub fn manifest() -> Manifest {
    let mut m = qwen35moe::caps::manifest();
    m.actions.push(export_spec());
    m
}

/// The served provider, extended with `export` over the same checkpoint.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    let served = qwen35moe::caps::provider(assembly)?;
    Ok(Arc::new(WithExport { served, weights: assembly.role_path("weights")? }))
}

struct WithExport {
    served: Arc<dyn Provider>,
    weights: String,
}

impl Provider for WithExport {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        if name == "export" {
            return Some(Arc::new(Export { weights: self.weights.clone() }));
        }
        self.served.action(name)
    }
}

struct Export {
    weights: String,
}

impl Action for Export {
    fn spec(&self) -> ActionSpec {
        export_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dest = inv.get_str("dest").ok_or("qwen35moe export: missing required param 'dest'")?;
        let seq = inv.get_i64("seq").unwrap_or(DEFAULT_SEQ).max(1) as usize;
        progress(Progress::step(0, 1, format!("exporting {} at seq {seq}", self.weights)));
        npu::qwen35moe_export::export_qwen35_fp32(&self.weights, &dest, seq).map_err(|e| format!("qwen35moe export: {e}"))?;
        progress(Progress::step(1, 1, "written"));
        let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        Ok(Outcome::new().set("dest", json!(dest)).set("seq", json!(seq)).set("bytes", json!(bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The program's manifest is the served one plus `export`, and `export`
    /// refuses a missing destination before touching a checkpoint.
    #[test]
    fn export_extends_the_served_manifest_and_needs_a_destination() {
        let names: Vec<String> = manifest().actions.into_iter().map(|a| a.name).collect();
        assert_eq!(names, ["generate", "export"]);
        assert!(export_spec().validate(Invocation::new()).unwrap_err().contains("dest"));
        assert!(export_spec().validate(Invocation::new().set("dest", json!("/tmp/x.onnx")).set("seq", json!(0))).is_err());
    }
}
