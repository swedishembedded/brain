// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `npu-export`: write Qwen3-Omni's three NPU-eligible pieces - the audio
//! tower, the vision tower and the Code2Wav vocoder, not the 30B-parameter MoE
//! Thinker decoder, which is not an NPU target - as fp32 ONNX graphs
//! (`audio_tower.onnx`, `vision_tower.onnx`, `codec_decoder.onnx`), and run the
//! structural check `brain npu check` runs on each, with an OpenVINO compile
//! where a runtime and device are present. Pure Rust export: it runs anywhere.
//! See `qwen3omnimoe::npu_export` for the scope (single-merger vision path, no
//! DeepStack).
//!
//! A piece that fails to export is reported and the others still run, so one
//! run shows every graph's result; the action then fails if any piece did.
//!
//! Swedish Embedded AB implements multimodal model deployment to NPUs and other
//! edge accelerators for its clients. If your team needs expertise in splitting
//! an omni-modal model into the graphs an accelerator's compiler accepts then
//! you can procure our services by sending an email to info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Assembly, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use serde_json::json;

/// Largest vision patch grid side one export may ask for.
const MAX_GRID: f64 = 4096.0;

pub fn export_spec() -> ActionSpec {
    let mut spec = ActionSpec::new("npu-export", "export the audio tower, the vision tower and the Code2Wav vocoder as ONNX graphs and check each (and compile it, where OpenVINO and a device are present)")
        .param(ParamSpec::new("out_dir", ParamType::Str, "directory the graphs are written to, on this machine").default(json!("out/omni")).host_resolved())
        .param(ParamSpec::new("n_audio", ParamType::Int, "audio frames the audio tower is exported for").min(1.0).max(capability::bounds::TOKENS).default(json!(100)))
        .param(ParamSpec::new("grid_h", ParamType::Int, "vision patch grid height the vision tower is exported for").min(1.0).max(MAX_GRID).default(json!(16)))
        .param(ParamSpec::new("grid_w", ParamType::Int, "vision patch grid width the vision tower is exported for").min(1.0).max(MAX_GRID).default(json!(16)))
        .param(ParamSpec::new("code_len", ParamType::Int, "codec frames the vocoder is exported for").min(1.0).max(capability::bounds::TOKENS).default(json!(32)))
        .output(BlobSpec::new("report", Media::Text, "per graph: where it was written and its structural and compile check"));
    spec.params.extend(npu::run_options::params());
    spec
}

/// The model's served manifest plus `npu-export`.
pub fn manifest() -> Manifest {
    let mut m = qwen3omnimoe::caps::direct_manifest();
    m.actions.push(export_spec());
    m
}

/// The served provider, extended with `npu-export` over the same directory.
pub fn provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    let served = qwen3omnimoe::caps::provider(assembly)?;
    Ok(Arc::new(WithExport { served, dir: assembly.role_path("weights").ok() }))
}

struct WithExport {
    served: Arc<dyn Provider>,
    dir: Option<String>,
}

impl Provider for WithExport {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "npu-export" => Some(Arc::new(Export { dir: self.dir.clone() })),
            _ => self.served.action(name),
        }
    }
}

struct Export {
    dir: Option<String>,
}

impl Action for Export {
    fn spec(&self) -> ActionSpec {
        export_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = self.dir.as_deref().ok_or("qwen3omnimoe npu-export: no checkpoint directory in the model store (brain pull Qwen/Qwen3-Omni-30B-A3B-Instruct, or --weights DIR)")?;
        let out_dir = inv.get_str("out_dir").filter(|d| !d.is_empty()).unwrap_or_else(|| "out/omni".to_string());
        let n_audio = inv.get_i64("n_audio").unwrap_or(100).max(1) as u32;
        let (grid_h, grid_w) = (inv.get_i64("grid_h").unwrap_or(16).max(1) as u32, inv.get_i64("grid_w").unwrap_or(16).max(1) as u32);
        let code_len = inv.get_i64("code_len").unwrap_or(32).max(1) as usize;
        let ncfg = npu::run_options::config(inv)?;
        let fail = |what: &str, e: &dyn std::fmt::Display| format!("qwen3omnimoe npu-export: {what}: {e}");

        std::fs::create_dir_all(&out_dir).map_err(|e| fail(&format!("create {out_dir}"), &e))?;
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(Path::new(dir)).map_err(|e| fail(&format!("open {dir}"), &e))?;
        let config_json = std::fs::read_to_string(Path::new(dir).join("config.json")).map_err(|e| fail("read config.json", &e))?;
        let root: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| fail("parse config.json", &e))?;
        let cfg = qwen3omnimoe::config::OmniConfig::from_json(&root);

        let path = |name: &str| format!("{out_dir}/{name}");
        let pieces = [
            (
                "audio tower",
                format!("n_audio={n_audio}"),
                path("audio_tower.onnx"),
                Box::new(|p: &str| qwen3omnimoe::npu_export::export_audio_onnx(&reader, p, n_audio)) as Box<dyn Fn(&str) -> std::io::Result<()>>,
            ),
            ("vision tower", format!("grid {grid_h}x{grid_w}"), path("vision_tower.onnx"), Box::new(|p: &str| qwen3omnimoe::npu_export::export_vision_onnx(&reader, p, grid_h, grid_w))),
            ("codec decoder", format!("code_len={code_len}"), path("codec_decoder.onnx"), Box::new(|p: &str| qwen3omnimoe::npu_export::export_codec_onnx(&reader, &cfg.code2wav, p, code_len))),
        ];

        let total = pieces.len() as u32;
        let mut report = String::new();
        let mut failed = Vec::new();
        for (i, (label, detail, file, export)) in pieces.into_iter().enumerate() {
            progress(Progress::step(i as u32, total, format!("exporting the {label}")));
            match export(&file) {
                Ok(()) => report.push_str(&format!("== {label} -> {file} ({detail}) ==\n{}", npu::check::structural_report(&file, &ncfg))),
                Err(e) => {
                    report.push_str(&format!("{label} export failed: {e}\n"));
                    failed.push(label);
                }
            }
        }
        if !failed.is_empty() {
            return Err(format!("{report}qwen3omnimoe npu-export: failed to export: {}", failed.join(", ")));
        }
        Ok(Outcome::new().set("out_dir", json!(out_dir)).blob("report", Blob::new(Media::Text, report.into_bytes())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_program_offers_the_served_actions_and_every_export_knob() {
        let actions: Vec<String> = manifest().actions.into_iter().map(|a| a.name).collect();
        for want in ["generate", "speak", "converse", "import", "npu-export"] {
            assert!(actions.iter().any(|a| a == want), "{want} in {actions:?}");
        }
        let names: Vec<String> = export_spec().params.into_iter().map(|p| p.name).collect();
        for want in ["out_dir", "n_audio", "grid_h", "grid_w", "code_len", "ov_device", "allow_fallback"] {
            assert!(names.iter().any(|n| n == want), "{want} in {names:?}");
        }
    }

    #[test]
    fn an_export_without_a_checkpoint_says_how_to_name_one() {
        let e = Export { dir: None }.run(&Invocation::new(), &mut |_| {}).unwrap_err();
        assert!(e.contains("--weights"), "{e}");
    }

    #[test]
    fn an_unreadable_checkpoint_directory_is_an_error_not_a_partial_export() {
        let out = std::env::temp_dir().join(format!("brain-omni-sample-{}", std::process::id()));
        let inv = Invocation::new().set("out_dir", json!(out.to_string_lossy()));
        let e = Export { dir: Some("/nonexistent/omni".into()) }.run(&inv, &mut |_| {}).unwrap_err();
        let _ = std::fs::remove_dir_all(&out);
        assert!(e.contains("open /nonexistent/omni"), "{e}");
    }
}
