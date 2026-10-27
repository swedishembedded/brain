// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-OCR as a served model: its identity, its measured footprint and the
//! loader that builds it. The `generate` action, the placement of the two halves,
//! the continuous-batching front end and the residency adapter are the ones every
//! OCR model shares (`ocrserve`); this module only says what is specific to v1.
//!
//! * **Real preprocessing.** [`crate::preprocess::preprocess_image`] turns the
//!   request's decoded pixels into the normalized tensor with the checkpoint's
//!   own `mean = std = 0.5` affine and the reference's aspect-preserving
//!   fit-and-pad - not a stretch, not a borrowed CLIP normalization.
//! * **The real 273-row prompt.** [`crate::prompt::build_prompt`] assembles
//!   `BOS ++ <image>x273 ++ text`, and the DeepEncoder's rows - the 16
//!   `image_newline` rows and the one `view_separator` row carrying the learned
//!   vectors - stand at the image positions ([`crate::serve::PageEncoder`]).

use crate::import::Files;
use ocrserve::{DecoderDevice, Footprint, OcrModel};
use residency::Instance;

/// The catalog id. Case-exact, because it is a real upstream repo
/// (`deepseek-ai/DeepSeek-OCR`) rather than one of the `brain/<family>`
/// placeholders `crates/modelref/src/alias.rs` documents.
pub const MODEL: &str = "deepseek-ai/DeepSeek-OCR";

pub use ocrserve::{DEFAULT_INSTRUCTION, DEFAULT_MAX_NEW, MAX_CONTEXT};

/// Device bytes the vision tower (SAM ViT-B at 1024² + the 16x compressor +
/// CLIP-L + the projector) holds while hot.
///
/// MEASURED on a Tesla P40, on the card, not inferred from host RSS: a real
/// page through the production [`crate::serve::load`] peaks that card at
/// **6.27 GiB** (`nvidia-smi memory.used`, sampled for the life of the
/// process, with the decoder placed elsewhere so the figure is the tower's
/// alone). 7 GiB rounds that up.
///
/// The residency adapter budgets it ([`FOOTPRINT`]).
///
/// The serving build holds `ENCODE_WORKERS` (2) tower sets with SAM
/// activation scratch shared between them, and each set carries TF32/fp16
/// serving copies; the served replica measured **32.6 GiB** in total on a
/// GH200 (`nvidia-smi`, 64 rows, 128k-token KV pool, decode tapes), against
/// 22.7 GiB with a single serial tower. The vision share is budgeted at
/// 15 GiB and the decoder at 18 GiB (together 33 GiB).
pub const VISION_DEVICE_BYTES: u64 = 15u64 << 30;

/// Device bytes the decoder half holds while hot, at the default context.
///
/// MEASURED the same way, on the same run: **13.25 GiB** on its own Tesla P40
/// at `ctx = 8192`, `chunk = 512` (the served defaults). It decomposes as
/// ~11.4 GiB of weights (2234 tensors, 2.9 B parameters at fp32), ~1.0 GiB of
/// KV cache (12 layers x 2 x 8192 x 1280 fp32), ~0.5 GiB of per-expert MoE
/// activation scratch at a 512-row round, and ~0.3 GiB of attention
/// score/probability slabs. 14 GiB rounds that up.
pub const DECODER_DEVICE_BYTES: u64 = 18u64 << 30;

/// Host bytes the whole composite holds once the decoder is on a card:
/// tokenizer, the transient mmproj import, the CPU-side preprocessor and the
/// per-token host work. MEASURED at **2.63 GiB** of VmHWM over a real page.
pub const HOST_BYTES_GPU_DECODER: u64 = 3u64 << 30;

/// Host bytes the whole composite holds with the decoder on the CPU backend -
/// the ~11.4 GiB of fp32 weights that would otherwise be on a card, plus
/// [`HOST_BYTES_GPU_DECODER`]'s working set. MEASURED at **13.51 GiB** of
/// VmHWM over the same page.
pub const HOST_BYTES_CPU_DECODER: u64 = 14u64 << 30;

/// What the model holds while hot.
pub const FOOTPRINT: Footprint =
    Footprint { vision_device: VISION_DEVICE_BYTES, decoder_device: DECODER_DEVICE_BYTES, host_gpu_decoder: HOST_BYTES_GPU_DECODER, host_cpu_decoder: HOST_BYTES_CPU_DECODER };

/// DeepSeek-OCR, as the serving layer sees it.
pub struct DeepseekOcr;

impl OcrModel for DeepseekOcr {
    const ID: &'static str = MODEL;
    const SUMMARY: &'static str = "DeepSeek-OCR -- document image in, text/markdown out. DeepEncoder (SAM ViT-B + 16x compressor \
         -> CLIP-L) spliced into a DeepSeek-V2 MoE decoder. Greedy, continuously batched, both halves on the GPU where one fits.";
    const FOOTPRINT: Footprint = FOOTPRINT;

    fn locate(dir: &str) -> Result<(), String> {
        Files::locate(dir).map(|_| ())
    }

    fn load(dir: &str, placement: DecoderDevice) -> Result<Box<dyn Instance>, String> {
        crate::serve::load(dir, placement).map(|i| Box::new(i) as Box<dyn Instance>)
    }
}

pub fn manifest() -> capability::Manifest {
    ocrserve::caps::manifest::<DeepseekOcr>()
}

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor, HTTP): the
/// checkpoint directory is service-side configuration, so the served action
/// carries only real per-request parameters.
pub fn manifest_resident() -> capability::Manifest {
    ocrserve::caps::manifest_resident::<DeepseekOcr>()
}

/// A stateless provider over [`DeepseekOcr`].
pub type DeepseekOcrProvider = ocrserve::caps::OcrProvider<DeepseekOcr>;
