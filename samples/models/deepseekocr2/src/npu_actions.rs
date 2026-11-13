// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `npu-export`: write DeepSeek-OCR-2's ONNX-eligible pieces - the resampler,
//! one graph per view size (`resampler_local.onnx`, `resampler_global.onnx`),
//! and the decoder (`decoder.onnx`) at a fixed sequence length - for a
//! best-effort OpenVINO and NPU compile. Pure Rust: it runs anywhere and needs
//! neither OpenVINO nor a device.
//!
//! The checkpoint directory is the one `deepseekocr2::import::Files::locate`
//! reads: the two GGUFs, their already-expanded fp32 forms, or an HF
//! checkpoint. The SAM vision backbone is not exported (see
//! `npu::deepseekocr2_topology` for the unresolved reason: no windowed
//! attention or relative-position-bias ONNX precedent exists in the NPU crate).
//!
//! Swedish Embedded AB implements vision-language model deployment to NPUs and
//! other edge accelerators for its clients. If your team needs expertise in
//! splitting a multimodal model into the graphs an accelerator's compiler
//! accepts then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Assembly, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

const GRAPHS: [&str; 3] = ["resampler_local.onnx", "resampler_global.onnx", "decoder.onnx"];

pub fn export_spec() -> ActionSpec {
    ActionSpec::new("npu-export", "export the resampler (both view sizes) and the decoder as ONNX graphs for OpenVINO; the SAM vision backbone is not exported")
        .param(ParamSpec::new("seq", ParamType::Int, "the decoder's fixed export sequence length").min(1.0).max(capability::bounds::TOKENS).default(json!(512)))
        .param(ParamSpec::new("out_dir", ParamType::Str, "directory the three graphs are written to, on this machine").default(json!("out/deepseekocr2-onnx")).host_resolved())
}

/// The model's served manifest plus `npu-export`.
pub fn manifest() -> Manifest {
    let mut m = deepseekocr2::caps::manifest();
    m.actions.push(export_spec());
    m
}

/// The served provider (built over the checkpoint directory on `generate`),
/// extended with `npu-export` over the same directory.
pub fn provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(WithExport { dir: ocrserve::dir_from_assembly(assembly)? }))
}

struct WithExport {
    dir: String,
}

impl Provider for WithExport {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "npu-export" => Some(Arc::new(Export { dir: self.dir.clone() })),
            _ => deepseekocr2::caps::DeepseekOcr2Provider::new(self.dir.clone())?.action(name),
        }
    }
}

struct Export {
    dir: String,
}

impl Action for Export {
    fn spec(&self) -> ActionSpec {
        export_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let seq = inv.get_i64("seq").unwrap_or(512).max(1) as usize;
        let out_dir = inv.get_str("out_dir").filter(|d| !d.is_empty()).unwrap_or_else(|| "out/deepseekocr2-onnx".to_string());
        progress(Progress::step(0, 1, format!("exporting the resampler and the decoder at S={seq}")));
        npu::deepseekocr2_export::export_all(&self.dir, seq, &out_dir).map_err(|e| format!("deepseekocr2 npu-export: {e}"))?;
        Ok(Outcome::new().set("out_dir", json!(out_dir)).set("graphs", json!(GRAPHS)).set("seq", json!(seq)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_program_offers_generate_and_npu_export_with_every_knob() {
        let m = manifest();
        assert!(m.actions.iter().any(|a| a.name == "generate"));
        let names: Vec<String> = export_spec().params.into_iter().map(|p| p.name).collect();
        assert_eq!(names, ["seq", "out_dir"]);
    }

    #[test]
    fn a_directory_without_a_checkpoint_is_refused_with_the_layout_it_expects() {
        let dir = std::env::temp_dir().join(format!("brain-deepseekocr2-sample-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let e = Export { dir: dir.to_string_lossy().into_owned() }.run(&Invocation::new().set("out_dir", json!(dir.join("out").to_string_lossy())), &mut |_| {}).unwrap_err();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(e.contains("holds neither"), "{e}");
    }
}
