// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The NPU route as actions beside the served ones: `npu-export` (fp32 ONNX),
//! `npu-quantize` (INT8 Q/DQ ONNX from calibration images), `npu-sim` (fp32
//! against simulated-INT8 mAP, no NPU needed) and `npu-run` (detect through
//! OpenVINO on the NPU, from an exported graph or straight from the checkpoint).
//!
//! They are actions, not workflows, so their flags are typed and bounded from
//! one spec like every served action's, `--help` and `--json` come for free,
//! and the ONNX files are named outputs (`--out onnx=PATH`). The program's
//! manifest is the model's served one plus these; [`provider`] answers both.
//!
//! `export`, `quantize` and `sim` are pure Rust and run anywhere; `run` needs
//! OpenVINO and an Intel NPU at run time and says so when it has neither.
//!
//! Swedish Embedded AB implements detector deployment to NPUs and other edge
//! accelerators for its clients. If your team needs expertise in exporting and
//! INT8-quantizing vision models for OpenVINO then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use npu::openvino::{NpuConfig, NpuDevice, PerfHint};
use serde_json::json;
use yolov8::identity::FaceVerifier;

/// Most calibration images one quantization reads.
const MAX_CALIB: f64 = 100_000.0;
/// Highest ONNX opset the writer is asked for.
const MAX_OPSET: f64 = 32.0;
/// Most classes / largest DFL width a bare ONNX graph can be told it has.
const MAX_CLASSES: f64 = 10_000.0;
const MAX_REG: f64 = 64.0;

fn input_param(what: &str) -> ParamSpec {
    ParamSpec::new("input", ParamType::Int, &format!("{what}, x32; 0 = the checkpoint's own")).min(0.0).max(capability::bounds::SIZE_PX).default(json!(0))
}

fn opset_param() -> ParamSpec {
    ParamSpec::new("opset", ParamType::Int, "ONNX opset to write").min(1.0).max(MAX_OPSET).default(json!(onnx::DEFAULT_OPSET))
}

fn num_calib_param() -> ParamSpec {
    ParamSpec::new("num_calib", ParamType::Int, "most calibration images to read").min(1.0).max(MAX_CALIB).default(json!(300))
}

pub fn export_spec() -> ActionSpec {
    ActionSpec::new("npu-export", "export the checkpoint as an fp32 ONNX graph with a static input, for OpenVINO")
        .param(input_param("static input side of the graph"))
        .param(opset_param())
        .output(BlobSpec::new("onnx", Media::Bytes, "the fp32 ONNX graph").required())
}

pub fn quantize_spec() -> ActionSpec {
    ActionSpec::new("npu-quantize", "calibrate INT8 activation scales on images and export an INT8 Q/DQ ONNX graph")
        .param(ParamSpec::new("calib", ParamType::Str, "calibration folder on this machine: images (PPM, PNG, JPEG, name order) or a detection dataset directory").required().host_resolved())
        .param(num_calib_param())
        .param(input_param("input side the calibration images are letterboxed to and the graph is exported at"))
        .param(opset_param())
        .output(BlobSpec::new("onnx", Media::Bytes, "the INT8 Q/DQ ONNX graph").required())
        .output(BlobSpec::new("scales", Media::Text, "the calibrated activation scales as JSON"))
}

pub fn sim_spec() -> ActionSpec {
    ActionSpec::new("npu-sim", "fp32 against simulated-INT8 mAP@0.5 over a detection dataset, in brain's own engine: INT8 accuracy with no NPU and no OpenVINO")
        .param(ParamSpec::new("data", ParamType::Str, "detection dataset directory on this machine that is scored").required().host_resolved())
        .param(ParamSpec::new("calib", ParamType::Str, "calibration folder on this machine (images or a dataset directory); empty = the scored dataset's own images").default(json!("")).host_resolved())
        .param(num_calib_param())
        .param(ParamSpec::new("conf", ParamType::Float, "confidence threshold").min(0.0).max(capability::bounds::UNIT).default(json!(0.25)))
        .param(ParamSpec::new("iou", ParamType::Float, "NMS IoU threshold").min(0.0).max(capability::bounds::UNIT).default(json!(0.45)))
}

pub fn run_spec() -> ActionSpec {
    ActionSpec::new("npu-run", "detect through OpenVINO (the Intel NPU by default): an exported graph, or the checkpoint exported in memory when no graph is named")
        .from_scratch()
        .param(ParamSpec::new("onnx", ParamType::Str, "ONNX graph file on this machine from npu-export or npu-quantize; empty = export the checkpoint in memory").default(json!("")).host_resolved())
        .param(ParamSpec::new("conf", ParamType::Float, "confidence threshold").min(0.0).max(capability::bounds::UNIT).default(json!(0.25)))
        .param(ParamSpec::new("iou", ParamType::Float, "NMS IoU threshold").min(0.0).max(capability::bounds::UNIT).default(json!(0.45)))
        .param(ParamSpec::new("nc", ParamType::Int, "class count of a bare graph, which does not carry it; 0 = 80 (yolov8n)").min(0.0).max(MAX_CLASSES).default(json!(0)))
        .param(ParamSpec::new("reg_max", ParamType::Int, "DFL width of a bare graph; 0 = 16 (yolov8n)").min(0.0).max(MAX_REG).default(json!(0)))
        .param(input_param("input side the image is letterboxed to; 0 = the graph's or checkpoint's own"))
        .param(ParamSpec::new("ov_device", ParamType::Enum(["npu", "cpu", "gpu", "auto"].map(String::from).to_vec()), "OpenVINO device the graph compiles for").default(json!("npu")))
        .param(ParamSpec::new("hint", ParamType::Enum(["latency", "throughput"].map(String::from).to_vec()), "compile for latency or throughput").default(json!("latency")))
        .param(ParamSpec::new("cache_dir", ParamType::Str, "compiled-blob cache folder on this machine; empty = none").default(json!("")).host_resolved())
        .param(ParamSpec::new("turbo", ParamType::Bool, "NPU turbo mode").default(json!(false)))
        .param(ParamSpec::new("allow_fallback", ParamType::Bool, "fall back to CPU/GPU when the requested device is absent").default(json!(false)))
        .param(ParamSpec::new("profile", ParamType::Bool, "collect OpenVINO profiling counters").default(json!(false)))
        .input(BlobSpec::new("image", Media::Image, "the image to run detection on, at its own resolution").required())
        .output(BlobSpec::new("detections", Media::Text, "JSON array of {bbox:[x1,y1,x2,y2], conf, class} in image coords"))
}

/// The model's served manifest plus the NPU actions.
pub fn manifest() -> Manifest {
    let mut m = yolov8::caps::manifest();
    m.actions.extend([export_spec(), quantize_spec(), sim_spec(), run_spec()]);
    m
}

/// The served provider, extended with the NPU actions over the same checkpoint.
pub fn provider(assembly: &capability::Assembly, faces: Arc<dyn FaceVerifier>) -> Result<Arc<dyn Provider>, String> {
    let served = yolov8::caps::provider(assembly, faces)?;
    Ok(Arc::new(WithNpu { served, weights: assembly.role_path("weights").ok() }))
}

struct WithNpu {
    served: Arc<dyn Provider>,
    weights: Option<String>,
}

impl Provider for WithNpu {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let weights = self.weights.clone();
        let kind = match name {
            "npu-export" => Kind::Export,
            "npu-quantize" => Kind::Quantize,
            "npu-sim" => Kind::Sim,
            "npu-run" => Kind::Run,
            _ => return self.served.action(name),
        };
        Some(Arc::new(NpuAction { kind, weights }))
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Export,
    Quantize,
    Sim,
    Run,
}

struct NpuAction {
    kind: Kind,
    weights: Option<String>,
}

fn side(inv: &Invocation) -> Option<u32> {
    inv.get_i64("input").filter(|&n| n > 0).map(|n| n as u32)
}

fn opset(inv: &Invocation) -> i64 {
    inv.get_i64("opset").unwrap_or(onnx::DEFAULT_OPSET)
}

fn num_calib(inv: &Invocation) -> usize {
    inv.get_i64("num_calib").unwrap_or(300).max(1) as usize
}

fn nonempty(inv: &Invocation, key: &str) -> Option<String> {
    inv.get_str(key).filter(|s| !s.is_empty())
}

impl NpuAction {
    fn need_weights(&self, name: &str) -> Result<&str, String> {
        self.weights.as_deref().ok_or_else(|| format!("yolov8 {name}: no YOLOv8 checkpoint (pull one, or name it with --weights)"))
    }

    fn export(&self, inv: &Invocation) -> ActionResult {
        let bytes = npu::try_build_fp32_bytes(self.need_weights("npu-export")?, side(inv), opset(inv)).map_err(|e| format!("yolov8 npu-export: {e}"))?;
        Ok(Outcome::new().set("bytes", json!(bytes.len())).blob("onnx", Blob::new(Media::Bytes, bytes)))
    }

    /// Calibrate on `calib` and return the scales with the number of images used.
    fn calibrate(weights: &str, calib: &str, input: u32, max: usize, verb: &str) -> Result<(npu::Quant, usize), String> {
        let images = npu::load_calib_images(calib, input, max).map_err(|e| format!("yolov8 {verb}: loading calibration images from {calib}: {e}"))?;
        if images.is_empty() {
            return Err(format!("yolov8 {verb}: no calibration images found in {calib}"));
        }
        let quant = npu::try_calibrate_from_weights(weights, &images).map_err(|e| format!("yolov8 {verb}: {e}"))?;
        Ok((quant, images.len()))
    }

    fn quantize(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = self.need_weights("npu-quantize")?;
        let calib = inv.get_str("calib").ok_or("yolov8 npu-quantize: missing required param 'calib'")?;
        let input = match side(inv) {
            Some(s) => s,
            None => npu::try_config_of(weights)?.input,
        };
        progress(Progress::step(0, 1, format!("calibrating on up to {} images at {input}px", num_calib(inv))));
        let (quant, images) = Self::calibrate(weights, &calib, input, num_calib(inv), "npu-quantize")?;
        let bytes = npu::try_build_int8_bytes(weights, &quant, side(inv), opset(inv)).map_err(|e| format!("yolov8 npu-quantize: {e}"))?;
        let scales = serde_json::to_vec_pretty(&quant.to_json()).map_err(|e| e.to_string())?;
        Ok(Outcome::new()
            .set("images", json!(images))
            .set("calibrated_convs", json!(quant.len()))
            .blob("onnx", Blob::new(Media::Bytes, bytes))
            .blob("scales", Blob::new(Media::Text, scales)))
    }

    fn sim(&self, inv: &Invocation) -> ActionResult {
        let weights = self.need_weights("npu-sim")?;
        let data = inv.get_str("data").ok_or("yolov8 npu-sim: missing required param 'data'")?;
        let cfg = npu::try_config_of(weights)?;
        // Calibrate on `calib` if given, else reuse the scored dataset's images.
        let calib = nonempty(inv, "calib").unwrap_or_else(|| data.clone());
        let (quant, images) = Self::calibrate(weights, &calib, cfg.input, num_calib(inv), "npu-sim")?;
        let conf = inv.get_f64("conf").unwrap_or(0.25) as f32;
        let iou = inv.get_f64("iou").unwrap_or(0.45) as f32;
        let (fp32, int8) = npu::try_simulate_map(weights, &data, &quant, conf, iou).map_err(|e| format!("yolov8 npu-sim: {e}"))?;
        Ok(Outcome::new().set("map50_fp32", json!(fp32)).set("map50_int8_sim", json!(int8)).set("delta", json!(fp32 - int8)).set("calibration_images", json!(images)))
    }

    fn detect(&self, inv: &Invocation) -> ActionResult {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let conf = inv.get_f64("conf").unwrap_or(0.25) as f32;
        let iou = inv.get_f64("iou").unwrap_or(0.45) as f32;
        let device = inv.get_str("ov_device").unwrap_or_else(|| "npu".into());
        let hint = match inv.get_str("hint").as_deref() {
            Some("throughput") => PerfHint::Throughput,
            _ => PerfHint::Latency,
        };
        let cfg = NpuConfig {
            device: NpuDevice::parse(&device).ok_or_else(|| format!("yolov8 npu-run: ov_device expects npu|cpu|gpu|auto (got {device:?})"))?,
            perf_hint: hint,
            cache_dir: nonempty(inv, "cache_dir").map(std::path::PathBuf::from),
            turbo: inv.get_bool("turbo").unwrap_or(false),
            tiles: None,
            compilation_params: Some("optimization-level=2 performance-hint-override=latency".into()),
            qdq_opt: true,
            profiling: inv.get_bool("profile").unwrap_or(false),
            allow_fallback: inv.get_bool("allow_fallback").unwrap_or(false),
        };
        let positive = |key: &str| inv.get_i64(key).filter(|&n| n > 0).map(|n| n as u32);
        let mut session = match nonempty(inv, "onnx") {
            Some(path) => npu::YoloSession::from_onnx(Path::new(&path), positive("nc"), positive("reg_max"), side(inv), &cfg),
            None => npu::YoloSession::from_weights(self.need_weights("npu-run")?, side(inv), &cfg),
        }
        .map_err(|e| format!("yolov8 npu-run: {e}"))?;
        let dets = session.detect(&hwc, w, h, conf, iou).map_err(|e| format!("yolov8 npu-run: {e}"))?;
        let objects: Vec<_> = dets.iter().map(|d| json!({"bbox": [d[0], d[1], d[2], d[3]], "conf": d[4], "class": d[5] as u32})).collect();
        let text = serde_json::to_string(&objects).map_err(|e| e.to_string())?;
        Ok(Outcome::new().set("count", json!(objects.len())).set("device", json!(session.device())).set("detections", json!(objects)).blob("detections", Blob::new(Media::Text, text.into_bytes())))
    }
}

impl Action for NpuAction {
    fn spec(&self) -> ActionSpec {
        match self.kind {
            Kind::Export => export_spec(),
            Kind::Quantize => quantize_spec(),
            Kind::Sim => sim_spec(),
            Kind::Run => run_spec(),
        }
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.kind {
            Kind::Export => self.export(inv),
            Kind::Quantize => self.quantize(inv, progress),
            Kind::Sim => self.sim(inv),
            Kind::Run => self.detect(inv),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_checkpoint(dir: &Path) -> String {
        use model::ModelConfig;
        let mut cfg = yolov8::YoloConfig::tiny(1);
        cfg.input = 64;
        let init = yolov8::init_weights(&cfg, 3);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(n, c)| (n.clone(), vec![c as u64], init[&n].clone())).collect();
        let path = dir.join("tiny.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
        path.to_string_lossy().into_owned()
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("yolov8-npu-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn act(kind: Kind, weights: Option<String>) -> NpuAction {
        NpuAction { kind, weights }
    }

    fn run(kind: Kind, weights: Option<String>, inv: Invocation) -> ActionResult {
        let a = act(kind, weights);
        let inv = a.spec().validate(inv)?;
        a.run(&inv, &mut |_| {})
    }

    #[test]
    fn the_manifest_is_the_served_actions_then_the_npu_ones() {
        let names: Vec<String> = manifest().actions.into_iter().map(|a| a.name).collect();
        assert_eq!(names, ["detect", "train", "finetune", "eval", "npu-export", "npu-quantize", "npu-sim", "npu-run"]);
        assert!(run_spec().from_scratch && !export_spec().from_scratch, "only npu-run may start without a checkpoint");
    }

    #[test]
    fn export_writes_an_onnx_graph_and_needs_a_checkpoint() {
        let dir = scratch("export");
        let out = run(Kind::Export, Some(tiny_checkpoint(&dir)), Invocation::new()).unwrap();
        let bytes = &out.blobs["onnx"].bytes;
        assert!(bytes.len() > 1000 && out.outputs["bytes"] == bytes.len());
        assert!(run(Kind::Export, None, Invocation::new()).unwrap_err().contains("no YOLOv8 checkpoint"));
        assert!(run(Kind::Export, Some("/nonexistent/w.safetensors".into()), Invocation::new()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantize_and_sim_calibrate_on_a_dataset_and_report() {
        let dir = scratch("quant");
        let weights = tiny_checkpoint(&dir);
        let data = dir.join("data");
        data::gen_detect::write_dataset(&data, data::gen_detect::Preset::MultiObject, 4, 64, 64, 1, 5).unwrap();
        let inv = Invocation::new().set("calib", json!(data.to_string_lossy())).set("num_calib", json!(2));
        let out = run(Kind::Quantize, Some(weights.clone()), inv).unwrap();
        assert_eq!(out.outputs["images"], 2);
        assert!(out.outputs["calibrated_convs"].as_u64().unwrap() > 0);
        assert!(out.blobs["onnx"].bytes.len() > 1000);
        assert!(serde_json::from_slice::<serde_json::Value>(&out.blobs["scales"].bytes).is_ok());

        let sim = run(Kind::Sim, Some(weights.clone()), Invocation::new().set("data", json!(data.to_string_lossy())).set("num_calib", json!(2)).set("conf", json!(0.001))).unwrap();
        let (a, b, d) = (sim.outputs["map50_fp32"].as_f64().unwrap(), sim.outputs["map50_int8_sim"].as_f64().unwrap(), sim.outputs["delta"].as_f64().unwrap());
        assert!((a - b - d).abs() < 1e-6);

        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        let err = run(Kind::Quantize, Some(weights), Invocation::new().set("calib", json!(empty.to_string_lossy()))).unwrap_err();
        assert!(err.contains("no calibration images"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_names_a_bad_device_and_a_missing_graph() {
        let img = Blob::new(Media::Image, vec![0u8; 12]).with_meta(json!({"w": 1, "h": 1, "c": 3}));
        let err = run(Kind::Run, None, Invocation::new().blob("image", img.clone()).set("ov_device", json!("tpu"))).unwrap_err();
        assert!(err.contains("ov_device"), "{err}");
        let err = run(Kind::Run, None, Invocation::new().blob("image", img)).unwrap_err();
        assert!(err.contains("no YOLOv8 checkpoint"), "{err}");
    }
}
