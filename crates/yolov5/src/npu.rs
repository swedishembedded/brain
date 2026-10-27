// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for real-time object detection on
// the NPUs of low-cost camera SoCs for its clients. If your team needs
// expertise in embedded vision on Rockchip-class hardware then you can procure
// our services by sending an email to info@swedishembedded.com.

//! YOLOv5 on the Rockchip RV1103/RV1106 NPU, through brain's own compiler
//! (`crates/rknpu`): the checkpoint's layer list lowered to an int8 network,
//! compiled and run on the board, its three heads decoded in place.
//!
//! Two artifacts make a model ready for the NPU, both next to each other in
//! the model directory:
//!
//! - the checkpoint (`model.safetensors`), read through a memory map one
//!   layer at a time, so loading never holds the model's float weights;
//! - its calibration (`model.npu-calibration.json`, [`calibration_path`]):
//!   the value range of every tensor over representative images at the input
//!   size the board runs, measured once on a host by [`calibrate`]. A board
//!   has neither the RAM nor the time to measure it.
//!
//! The heads leave the NPU as int8 logits; the sigmoid the float graph applies
//! is applied by the decode, and only to what it reads (every anchor's
//! objectness, the rest only for anchors that pass the threshold).

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::mmap::MmapSafetensors;
use imaging::nms::Detection;
use imaging::Letterbox;
use rknpu::graph::{Calibration, ConvParams, Graph as NpuGraph};
use rknpu::layout::Shape;
use rknpu::model::{Model, Timing};
use rknpu::plan::Plan;
use rknpu::quant::QParams;
use serde::{Deserialize, Serialize};

use crate::config::{Activation, Yolov5Config};
use crate::decode;
use crate::error::Error;
use crate::graph::{Epilogue, Graph, Op};
use crate::weights::{open_mapped_checkpoint, MappedCheckpoint};

/// The letterbox fill, as the reference preprocessing uses (grey 114).
pub const PAD: u8 = 114;

/// Pixels map to int8 as `p - 128`: scale 1/255 over the unit range.
const INPUT_Q: QParams = QParams { scale: 1.0 / 255.0, zero_point: -128 };

/// Where the calibration of `checkpoint` lives: beside it, named after it.
pub fn calibration_path(checkpoint: &Path) -> PathBuf {
    let stem = checkpoint.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    checkpoint.with_file_name(format!("{stem}.npu-calibration.json"))
}

/// The calibration file: one `[min, max]` per graph value (value 0 is the
/// image, node `i` produces value `i + 1`), measured at `input` x `input`.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationFile {
    pub arch: String,
    pub input: u32,
    pub images: usize,
    pub ranges: Vec<[f32; 2]>,
}

impl CalibrationFile {
    pub fn read(path: &Path) -> Result<CalibrationFile, Error> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            Error::Npu(format!("{}: {e}; calibrate the checkpoint on a host first (yolov5::npu::calibrate, e.g. `make crates/yolov5/examples/npu-calibrate/run`)", path.display()))
        })?;
        let file: CalibrationFile = serde_json::from_str(&text).map_err(|e| Error::Npu(format!("{}: {e}", path.display())))?;
        if file.arch != "yolov5" {
            return Err(Error::Npu(format!("{}: calibration is for {:?}, not yolov5", path.display(), file.arch)));
        }
        if file.ranges.iter().any(|[lo, hi]| !(lo.is_finite() && hi.is_finite() && lo <= hi)) {
            return Err(Error::Npu(format!("{}: a range is not a finite [min, max]", path.display())));
        }
        Ok(file)
    }

    pub fn write(&self, path: &Path) -> Result<(), Error> {
        let text = serde_json::to_string_pretty(self).map_err(|e| Error::Npu(e.to_string()))?;
        std::fs::write(path, text).map_err(|e| Error::Npu(format!("{}: {e}", path.display())))
    }
}

/// A convolution's parameters, read from the mapped checkpoint when asked.
struct MappedConv {
    file: Arc<MmapSafetensors>,
    weight: String,
    bias: Option<String>,
    out_channels: usize,
}

/// Elements per chunk a mapped tensor is read in: bounded memory however
/// large the layer.
const READ_CHUNK: usize = 16 * 1024;

impl ConvParams for MappedConv {
    fn weights(&self, sink: &mut dyn FnMut(&[f32])) {
        // The checkpoint was validated against the graph when it was opened.
        let found = self.file.with_tensor_chunks(&self.weight, READ_CHUNK, &mut |_, chunk| sink(chunk));
        assert!(found, "validated tensor {} is unreadable", self.weight);
    }

    fn bias(&self) -> Cow<'_, [f32]> {
        let mut bias = Vec::with_capacity(self.out_channels);
        match &self.bias {
            Some(name) => {
                let found = self.file.with_tensor_chunks(name, READ_CHUNK, &mut |_, chunk| bias.extend_from_slice(chunk));
                assert!(found, "validated tensor {name} is unreadable");
            }
            None => bias.resize(self.out_channels, 0.0),
        }
        Cow::Owned(bias)
    }
}

/// The checkpoint's network at `size` x `size` as an NPU graph. Value `v` of
/// the checkpoint graph is tensor `v` of the result. Head convolutions keep
/// their logits (the decode applies the sigmoid).
pub fn npu_graph(ckpt: &MappedCheckpoint, size: usize) -> Result<NpuGraph, Error> {
    if size == 0 || !size.is_multiple_of(32) {
        return Err(Error::Input(format!("NPU input size {size} must be a positive multiple of 32")));
    }
    let graph: &Graph = &ckpt.graph;
    let mut g = NpuGraph::new(Shape::new(size, size, 3));
    let npu = |e: rknpu::Error| Error::Npu(e.to_string());
    for (i, node) in graph.nodes.iter().enumerate() {
        let x = |k: usize| node.inputs[k];
        let id = match &node.op {
            Op::Conv(c) => {
                let relu = match c.epilogue {
                    Epilogue::None | Epilogue::Sigmoid => false,
                    Epilogue::Act(Activation::Relu) => true,
                    Epilogue::Act(Activation::Silu) => return Err(Error::Npu(format!("{}: SiLU has no NPU lowering yet; use a ReLU checkpoint", c.name))),
                };
                let params = Arc::new(MappedConv { file: Arc::clone(&ckpt.file), weight: c.weight_name(), bias: c.bias.then(|| c.bias_name()), out_channels: c.cout as usize });
                let p = c.pad as usize;
                g.conv(x(0), params, c.cout as usize, c.k as usize, c.stride as usize, [p; 4], relu).map_err(npu)?
            }
            Op::Add => g.add(x(0), x(1)).map_err(npu)?,
            Op::Concat => g.concat(&node.inputs).map_err(npu)?,
            Op::MaxPool { k, stride, pad } => g.max_pool(x(0), *k as usize, *stride as usize, *pad as usize).map_err(npu)?,
            Op::Upsample2x => g.upsample2x(x(0)),
        };
        debug_assert_eq!(id, i + 1, "NPU tensor ids follow the checkpoint's value ids");
    }
    for &o in &graph.outputs {
        g.mark_output(o);
    }
    Ok(g)
}

/// Letterbox an RGB8 image into `size` x `size` (nearest-neighbour scaling,
/// [`PAD`] fill) and say how to map boxes back. An image already of that size
/// passes through untouched.
pub fn letterbox_rgb8(rgb: &[u8], w: u32, h: u32, size: u32) -> (Cow<'_, [u8]>, Letterbox) {
    let lb = Letterbox::compute(w, h, size);
    if (w, h) == (size, size) {
        return (Cow::Borrowed(rgb), lb);
    }
    let mut out = vec![PAD; (size * size * 3) as usize];
    for y in 0..lb.new_h {
        let sy = ((y as f32 + 0.5) / lb.scale) as u32;
        let sy = sy.min(h - 1);
        for x in 0..lb.new_w {
            let sx = (((x as f32 + 0.5) / lb.scale) as u32).min(w - 1);
            let (dx, dy) = (x + lb.pad_x as u32, y + lb.pad_y as u32);
            let s = ((sy * w + sx) * 3) as usize;
            let d = ((dy * size + dx) * 3) as usize;
            out[d..d + 3].copy_from_slice(&rgb[s..s + 3]);
        }
    }
    (Cow::Owned(out), lb)
}

/// Measure the calibration of the checkpoint at `path` over `images` (RGB8,
/// any size; letterboxed to `size` like the board does) with the float
/// reference, and write it to [`calibration_path`]. Host only: it holds the
/// float activations of the whole network.
pub fn calibrate(path: &Path, size: u32, images: &[(Vec<u8>, u32, u32)]) -> Result<CalibrationFile, Error> {
    let ckpt = open_mapped_checkpoint(path)?;
    let graph = npu_graph(&ckpt, size as usize)?;
    let inputs: Vec<Vec<f32>> = images.iter().map(|(rgb, w, h)| letterbox_rgb8(rgb, *w, *h, size).0.iter().map(|&p| f32::from(p) / 255.0).collect()).collect();
    let measured = Calibration::measure(&graph, inputs.iter().map(Vec::as_slice)).map_err(|e| Error::Npu(e.to_string()))?;
    let file = CalibrationFile { arch: "yolov5".into(), input: size, images: images.len(), ranges: measured.ranges.iter().map(|&(lo, hi)| [lo, hi]).collect() };
    file.write(&calibration_path(path))?;
    Ok(file)
}

/// YOLOv5 compiled for and resident on the NPU.
pub struct NpuDetector {
    model: Model,
    config: Yolov5Config,
    size: u32,
}

impl NpuDetector {
    /// Compile the checkpoint at `path` with its calibration (at the
    /// calibration's input size) and load it onto the NPU.
    pub fn load(path: &Path) -> Result<NpuDetector, Error> {
        let calibration = CalibrationFile::read(&calibration_path(path))?;
        let ckpt = open_mapped_checkpoint(path)?;
        let graph = npu_graph(&ckpt, calibration.input as usize)?;
        if calibration.ranges.len() != graph.tensor_count() {
            return Err(Error::Npu(format!("calibration has {} ranges, the checkpoint's graph has {} values; recalibrate it", calibration.ranges.len(), graph.tensor_count())));
        }
        let ranges = Calibration { ranges: calibration.ranges.iter().map(|&[lo, hi]| (lo, hi)).collect() };
        let npu = |e: rknpu::Error| Error::Npu(e.to_string());
        let plan = Plan::new(&graph, &ranges, INPUT_Q).map_err(npu)?;
        let model = Model::load(rknpu::driver::Device::open().map_err(npu)?, plan).map_err(npu)?;
        Ok(NpuDetector { model, config: ckpt.config, size: calibration.input })
    }

    /// The square input size the model was compiled for.
    pub fn input_size(&self) -> u32 {
        self.size
    }

    /// Bytes of NPU memory the model holds.
    pub fn resident_bytes(&self) -> usize {
        self.model.resident_bytes()
    }

    /// Detect in an RGB8 image of `w` x `h`: letterbox (a no-op for an image
    /// already the input size), run, decode, NMS, map back. Detections are
    /// `[x1, y1, x2, y2, conf, class]` in image pixels, best first.
    pub fn detect(&mut self, rgb: &[u8], w: u32, h: u32, conf: f32, iou: f32, max_det: usize) -> Result<(Vec<Detection>, Timing), Error> {
        if rgb.len() != (w * h * 3) as usize || w == 0 || h == 0 {
            return Err(Error::Input(format!("{} bytes is not a {w}x{h} RGB8 image", rgb.len())));
        }
        let (input, letterbox) = letterbox_rgb8(rgb, w, h, self.size);
        let timing = self.model.run_rgb8(&input).map_err(|e| Error::Npu(e.to_string()))?;
        let heads = [self.model.output(0), self.model.output(1), self.model.output(2)];
        let sizes = heads.map(|v| (v.placement().shape.h, v.placement().shape.w));
        let found = decode_logits(&self.config, sizes, |scale, channel, y, x| heads[scale].value(y, x, channel), &letterbox, (w, h), conf, iou, max_det)?;
        Ok((found, timing))
    }
}

/// What the NPU computes for one image, computed on the host with the same
/// integers ([`Plan::simulate`]): the board's detections, exactly. Host only
/// (it holds the plan's whole activation region and re-reads every layer).
pub fn simulate_detect(path: &Path, rgb: &[u8], w: u32, h: u32, conf: f32, iou: f32, max_det: usize) -> Result<Vec<Detection>, Error> {
    let calibration = CalibrationFile::read(&calibration_path(path))?;
    let ckpt = open_mapped_checkpoint(path)?;
    let graph = npu_graph(&ckpt, calibration.input as usize)?;
    let ranges = Calibration { ranges: calibration.ranges.iter().map(|&[lo, hi]| (lo, hi)).collect() };
    let npu = |e: rknpu::Error| Error::Npu(e.to_string());
    let plan = Plan::new(&graph, &ranges, INPUT_Q).map_err(npu)?;
    let (input, letterbox) = letterbox_rgb8(rgb, w, h, calibration.input);
    let p = plan.placement(plan.input());
    let codes: Vec<i8> = input.iter().map(|&v| INPUT_Q.quantize(f32::from(v) / 255.0)).collect();
    let mut data = vec![0u8; plan.activation_bytes()];
    p.shape.pack(&codes, &mut data[p.offset..]);
    plan.simulate(&mut data).map_err(npu)?;
    let heads: Vec<_> = plan.outputs().iter().map(|&o| plan.placement(o)).collect();
    let sizes = [0, 1, 2].map(|i| (heads[i].shape.h, heads[i].shape.w));
    decode_logits(&ckpt.config, sizes, |scale, channel, y, x| heads[scale].value(&data, y, x, channel), &letterbox, (w, h), conf, iou, max_det)
}

/// Decode head logits read through `logit(scale, channel, y, x)`: sigmoid,
/// anchor decode, NMS, un-letterbox.
#[allow(clippy::too_many_arguments)]
fn decode_logits(cfg: &Yolov5Config, sizes: [(usize, usize); 3], logit: impl Fn(usize, usize, usize, usize) -> f32, letterbox: &Letterbox, orig: (u32, u32), conf: f32, iou: f32, max_det: usize) -> Result<Vec<Detection>, Error> {
    let candidates = decode::decode_candidates_with(cfg, sizes, conf, |scale, channel, y, x| sigmoid(logit(scale, channel, y, x)))?;
    Ok(decode::finish(&candidates, letterbox, orig, iou, max_det))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
