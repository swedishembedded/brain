// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR-2 as a served model: its identity, its measured footprint and
//! the loader that builds it. The `generate` action, the placement of the two
//! halves, the continuous-batching front end and the residency adapter are the
//! ones every OCR model shares (`ocrserve`); the decoder is the same
//! `deepseek2::serve::Engine` DeepSeek-OCR runs on. This module only says what
//! is specific to v2.

use ocrserve::{DecoderDevice, Footprint, OcrModel};
use residency::Instance;

use crate::import::Files;

/// The catalog id. Case-exact, the real upstream repo.
pub const MODEL: &str = "deepseek-ai/DeepSeek-OCR-2";

pub use ocrserve::{DEFAULT_INSTRUCTION, DEFAULT_MAX_NEW, MAX_CONTEXT};

/// What the model holds while hot. The decoder is DeepSeek-OCR's, byte for byte,
/// so its figures are DeepSeek-OCR's (`deepseek2ocr::caps`); the vision side is
/// SAM plus a 24-layer Qwen2 resampler.
pub const FOOTPRINT: Footprint = Footprint { vision_device: 15u64 << 30, decoder_device: 18u64 << 30, host_gpu_decoder: 3u64 << 30, host_cpu_decoder: 14u64 << 30 };

/// DeepSeek-OCR-2, as the serving layer sees it.
pub struct DeepseekOcr2;

impl OcrModel for DeepseekOcr2 {
    const ID: &'static str = MODEL;
    const SUMMARY: &'static str = "DeepSeek-OCR-2 -- document image in, text/markdown out. SAM ViT-B feeding a 24-layer Qwen2 GQA \
         resampler under a prefix-LM mask, spliced into the DeepSeek-V2 MoE decoder. Greedy, continuously batched.";
    const FOOTPRINT: Footprint = FOOTPRINT;

    fn locate(dir: &str) -> Result<(), String> {
        Files::locate(dir).map(|_| ())
    }

    fn load(dir: &str, placement: DecoderDevice) -> Result<Box<dyn Instance>, String> {
        crate::serve::load(dir, placement).map(|i| Box::new(i) as Box<dyn Instance>)
    }
}

pub fn manifest() -> capability::Manifest {
    ocrserve::caps::manifest::<DeepseekOcr2>()
}

/// The manifest for the RESIDENT/scheduled service: the checkpoint directory is
/// service-side configuration, so the served action carries only per-request parameters.
pub fn manifest_resident() -> capability::Manifest {
    ocrserve::caps::manifest_resident::<DeepseekOcr2>()
}

/// A stateless provider over [`DeepseekOcr2`].
pub type DeepseekOcr2Provider = ocrserve::caps::OcrProvider<DeepseekOcr2>;
