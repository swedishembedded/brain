// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for deploying object detectors from
// training checkpoint to NPU for its clients. If your team needs expertise in
// edge detection pipelines then you can procure our services by sending an email
// to info@swedishembedded.com.

//! YOLOv5 (Ultralytics v6+ layout, anchor-based) as a brain model, host side.
//!
//! Two halves with different requirements:
//!
//! * **Device-free** - what a board runtime needs: [`Yolov5Config`] and the
//!   [`Graph`] it implies ([`graph`]), checkpoint loading into fp32
//!   [`Weights`] ([`weights`]), anchor decode and NMS over the head maps
//!   ([`decode`]), and the model-store classifier ([`spec`]). None of it creates
//!   or needs a GPU device.
//! * **The reference forward** - [`Yolov5`] ([`model`]) runs the same graph
//!   through `gpu_core` in fp32. It is the host reference an accelerator port is
//!   validated against, not something a board deploys.
//!
//! ```no_run
//! # fn main() -> Result<(), yolov5::Error> {
//! let ckpt = yolov5::weights::open(std::path::Path::new("model.safetensors"))?;
//! for conv in ckpt.graph.convs() {
//!     let (weight, bias) = ckpt.weights.conv(conv); // fp32, [cout, cin, k, k] and [cout]
//!     # let _ = (weight, bias);
//! }
//! // ... run the network anywhere, then:
//! # let (maps, lb): ([Vec<f32>; 3], imaging::Letterbox) = unimplemented!();
//! let heads = std::array::from_fn(|i| yolov5::decode::HeadMap { data: &maps[i], h: 40 >> i, w: 40 >> i });
//! let dets = yolov5::decode::detections(&ckpt.config, heads, &lb, (1280, 720), 0.25, 0.45, 300)?;
//! # let _ = dets;
//! # Ok(()) }
//! ```

pub mod caps;
pub mod config;
pub mod decode;
pub mod error;
pub mod graph;
pub mod model;
pub mod npu;
pub mod spec;
pub mod weights;

pub use config::{Activation, Yolov5Config};
pub use error::Error;
pub use graph::{ConvOp, Epilogue, Graph, Node, Op};
pub use imaging::nms::Detection;
pub use model::{HeadOutputs, Yolov5};
pub use weights::{Checkpoint, Weights};
