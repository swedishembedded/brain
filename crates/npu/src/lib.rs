// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain-npu` — deploy brain's YOLO detector to the Intel NPU.
//!
//! Pipeline: **export** the trained model to ONNX, brain-native **INT8
//! post-training quantization** (calibrate → Q/DQ ONNX), then **compile + run**
//! on the Intel NPU via OpenVINO (behind the optional `openvino` feature). DFL
//! decode + NMS stay on the host in Rust.
//!
//! The export / quantize / fake-quant-simulate core is pure Rust and builds &
//! tests anywhere (no NPU, no OpenVINO). Only `run`/`bench` need the `openvino`
//! feature + an Intel NPU.

// Pure-Rust core (always compiled, hardware-free).
pub mod calib;
pub mod codec_export;
pub mod topo;
pub mod codec_topology;
pub mod decode;
pub mod export;
pub mod fold;
pub mod quant;
pub mod qwen_decode;
pub mod qwen_export;
pub mod qwen_topology;
pub mod glm_export;
pub mod glm_decode;
pub mod glm_topology;
pub mod lfm_topology;
pub mod upscale_topology;
pub mod lfm_export;
pub mod chronos2_topology;
pub mod chronos2_export;
pub mod kronos_topology;
pub mod kronos_export;
pub mod fincast_topology;
pub mod fincast_export;
pub mod sim;
pub mod topology;
pub mod wm_topology;
pub mod mirror_topology;
pub mod nemotron_topology;
pub mod nemotron_export;
pub mod qwen_asr_topology;

// OpenVINO runtime seam (real on x86_64 linux/windows, stub elsewhere).
pub mod openvino;

pub use calib::{calibrate, calibrate_from_weights, load_calib_images, RangeCollector};
pub use codec_export::{
    build_codec_fp32_bytes, export_codec_back_stream_fp32, export_codec_front_fp32, export_codec_fp32,
};
pub use decode::{decode_npu_outputs, detect_image, detect_weights_on_npu};
pub use export::{build_fp32_bytes, config_of, export_fp32, export_int8};
pub use quant::Quant;
pub use sim::{reference_logits, simulate_logits, simulate_map, FakeQuantTap};
pub use topology::{build_graph, WeightSource};
pub mod depth_topology;
pub use depth_topology::{build_depth_graph, build_depth_graph_hw};
pub use wm_topology::{build_diamond_graph, WmSession, WmUnetConfig};
pub use nemotron_topology::{build_nemotron_encoder, build_nemotron_head, build_subsampling, NemotronTopo};
pub use qwen_asr_topology::{build_qwen_asr_head, QwenAsrTopo};

/// The one per-model NPU seam (see `docs/npu-residency.md`). A model implements
/// [`build`](NpuModel::build) — its device-heavy forward, composed from the shared
/// `topo`/`onnx` block library — plus a [`cache_key`](NpuModel::cache_key); the
/// generic [`openvino::NpuGraph`] and the residency NPU instance do compile / cache /
/// infer / evict. This is what turns the NPU into a **uniform, reusable** compute
/// target: adding a model to the NPU is implementing this trait + a thin residency
/// adapter, not a new session or runtime code. fp16 is the NPU's native path
/// (OpenVINO compiles the fp32 bytes to fp16, no calibration); INT8/INT4 are
/// orthogonal opt-ins.
pub trait NpuModel: Send + Sync {
    /// Build the model's forward into `g`, declaring its named inputs/outputs
    /// (`g.input_f32`/`output_f32`). Reuse the shared emitters.
    fn build(&self, g: &mut onnx::GraphBuilder) -> Result<(), String>;

    /// A stable identity for the OpenVINO compiled-blob cache (model + precision +
    /// shape), so a residency warm-start skips recompilation.
    fn cache_key(&self) -> String;

    /// The fp32 ONNX bytes for this model (default: `build` then serialize).
    fn onnx_bytes(&self) -> Result<Vec<u8>, String> {
        let mut g = onnx::GraphBuilder::new(&self.cache_key());
        self.build(&mut g)?;
        Ok(g.finish_with(onnx::DEFAULT_OPSET, onnx::DEFAULT_IR_VERSION))
    }

    /// Compile this model onto `cfg.device` (NPU by default), returning the generic
    /// named-tensor runner — the single call a residency NPU instance makes.
    fn compile(&self, cfg: &openvino::NpuConfig) -> Result<openvino::NpuGraph, String> {
        openvino::NpuGraph::compile_bytes(&self.onnx_bytes()?, cfg).map_err(|e| e.to_string())
    }
}
