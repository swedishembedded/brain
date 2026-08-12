// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **DeepSeek-OCR** -- the composite: a DeepEncoder feeding a DeepSeek-V2 MHA
//! decoder through the shared vision-language embedding splice.
//!
//! ```text
//! image ──► sam1::SamEncoder            [1, c_out, gh/4, gw/4]   (SAM ViT-B + neck + 16x compressor)
//!             │ NCHW ──► NLC
//!             ├──────────────────────────────────────────┐
//!             ▼                                          │
//!        clip::ClipVision(PatchSource::Tokens)           │   (patch embed BYPASSED)
//!             │ drop the class-token row                 │
//!             ▼                                          ▼
//!        clip_spatial [N, clip_width] ── concat ── compressor_flat [N, c_out]
//!                                          │
//!                                Linear(clip_width + c_out, d_model)
//!                                          ▼
//!            deepseekv2::DeepseekV2, image rows spliced over the placeholder
//!            token embeddings (`model::vlm::splice_fwd`/`splice_bwd`)
//! ```
//!
//! **The concat order is `[clip_spatial, compressor_flat]`** -- CLIP's output
//! first, the compressor's own pre-CLIP features second. It is not a guess: it
//! is what the checkpoint-free golden dumper records and asserts by slicing the
//! halves back out (`findings.vision_concat`), and independently what
//! llama.cpp's own consumer of this GGUF does
//! (`ggml_concat(ctx0, clip_out, sam_out, 0)`, concatenating along the row
//! width). At real scale `c_out == clip_width == 1024`, so a swap would be
//! arithmetically invisible; `crates/deepseekocr/tests/tiny_ref.rs` runs at
//! widths 11 and 14 and asserts each half against its own source tap.
//!
//! ## Modules
//!
//! * [`config`] -- the three sub-configs plus the projector's shape facts, and
//!   the real-scale invariant `compressor_out == clip_width`.
//! * [`encoder`] -- the DeepEncoder (forward + backward).
//! * [`model`] -- encoder + splice + decoder (forward + backward, plus
//!   `O(T²)`-recompute greedy decode: the image is encoded once and every step
//!   re-splices the same projected tokens).
//! * [`preprocess`] -- **real images in**: decoded RGB of any extent -> the
//!   `[3, 1024, 1024]` normalized tensor [`DeepEncoder::forward`] takes. The
//!   square, the `mean = std = 0.5` normalization and the aspect-preserving
//!   fit-and-pad are each read off the shipped mmproj or off llama.cpp's own
//!   preprocessor, not borrowed from a sibling model.
//! * [`rows`] -- the **multi-view row layout**: pure host index math, no GPU and
//!   no weights, for the resolution modes a later preprocessing phase needs.
//! * [`layout`] -- what fills that layout: a row-table gather (`splice` +
//!   `embed`, adjoint `embed` + `emb_bwd`) turning the projector's output plus
//!   the two learned vectors into the interleaved block the decoder splices.
//! * [`prompt`] -- the decoder-side prompt: the LM GGUF's own tokenizer, this
//!   model's reserved token ids, and `text ++ image rows ++ text` assembled
//!   into the id sequence plus the `(row0, n_rows)` the splice takes.
//! * [`import`] -- the shipped `ggml-org/DeepSeek-OCR-GGUF` pair (plus the
//!   decoder's cached fp32 expansion) turned into the two weight sources, the
//!   config and the tokenizer the composite takes. Production code; this
//!   crate's real-weight tests are thin wrappers over it.
//! * [`caps`] -- the serving surface: one `generate` action (image + instruction
//!   in, streamed text out) behind `capability::Provider`, plus the [`caps::Session`]
//!   `crates/cli/src/resident_deepseekocr.rs` owns.
//! * [`train`] -- LoRA training glue: merging a real (or checkpoint-free
//!   fixture) base weight map with freshly-initialised adapter tensors for a
//!   `cfg.decoder.lora`-configured composite. The adapter mechanism itself
//!   (frozen base + trainable low-rank delta on the decoder's four attention
//!   projections) lives in `deepseekv2::config::LoraCfg`/
//!   `deepseekv2::model::DeepseekV2`, reused unchanged -- [`DeepseekOcr::new`]/
//!   [`DeepseekOcr::new_split`] need no change at all to build a LoRA-adapted
//!   composite once `cfg.decoder.lora` is set.
//!
//! ## What is and is not claimed
//!
//! * The composite forward is gated per stage against a checkpoint-free
//!   reference dump (`tools/goldens/deepseek_ocr_dump_reference.py`) at
//!   deliberately non-coincidental dims, from the SAM patch embed through to the
//!   decoder's logits.
//! * The composite backward exists end to end: the decoder's cross-entropy
//!   gradient reaches the input image through the splice, the projector, the
//!   concat, CLIP's injected-token seam and the whole SAM tower.
//! * **Batch is 1.** `sam1` is a single-image tower (its windowed attention
//!   spans' storage-binding offsets are not 256 B aligned across a batch
//!   stride), so the composite is too, and the golden fixture is generated at
//!   `batch = 1` to match.
//! * **Single contiguous image run.** The decoder splice covers one
//!   `[row0, row0 + n_rows)` run -- the golden's own scope. [`prompt`] builds
//!   the global-view prompt over that one run; [`rows`]'s multi-view
//!   (Base/Gundam) layouts still have no consumer, because they need one splice
//!   call per run (`layout::RowGather` itself is already indifferent to that;
//!   `deepseekv2::DeepseekV2::enable_mm_splice` is not).
//! * **The real 273-row block is assembled, and the 256-row one is still the
//!   parity path.** [`DeepseekOcr::new_with_prompt`] sizes the splice at
//!   [`build_prompt`]'s own `n_rows` (273 at the real geometry) and fills it
//!   through [`layout::RowGather`], which places the mmproj's
//!   `vision.image_newline` at the 16 newline rows and `vision.view_separator`
//!   at the one separator row -- **bit-identically**, it is a copy. The
//!   backward is the exact adjoint: an inverse gather for the projector rows,
//!   and an accumulating sum onto each learned vector over every row that read
//!   it. [`DeepseekOcr::new`]/[`DeepseekOcr::new_split`] keep the contiguous
//!   256-row splice, because that is the checkpoint-free fixture's own scope
//!   and every parity number here was measured on it.
//! * **Stages exchange host buffers.** Each sub-model owns its own `Gpu` with
//!   its own pipeline list, so SAM → CLIP → projector → decoder round-trip
//!   through `Vec<f32>` -- the same arrangement `crates/qwenvl` uses. A
//!   same-device fast path is additive (the splice seam already exposes
//!   `img_embeds_buf`/`d_img_embeds_buf` for it) and is not built here.
//! * **The composed decode loop has no multimodal oracle.** The text decoder's
//!   greedy loop is matched token for token against llama.cpp
//!   (`crates/deepseekv2/tests/generate.rs`); the image+decoder loop is gated
//!   only on completing, on finite logits, and on causal self-consistency
//!   (`tests/real_weight_generate.rs`), because llama.cpp's debug callback
//!   segfaults inside this model's CLIP graph and no post-image token-id capture
//!   exists to compare against.
//! * **The serving contract is met** ([`caps`], plus
//!   `crates/cli/src/resident_deepseekocr.rs`): one `generate` action over
//!   `brain caps`/`brain do`, the residency scheduler, D-Bus and the
//!   OpenAI/Anthropic surfaces, with real per-token streaming and real token
//!   counts. `run_batch` is the serial default and says why.
//! * **LoRA fine-tuning exists** ([`train::lora_init_map`] plus
//!   `deepseekv2`'s own adapter mechanism): the decoder's base weights frozen,
//!   only its `.lora_a`/`.lora_b` adapters trainable, gated by a descent smoke
//!   test (`tests/tiny_ref.rs::composite_lora_backward_freezes_the_base_and_descends`)
//!   proving a LoRA-only optimizer step measurably lowers the composite loss.
//!   Not done: a `finetune`-style CLI-driven training LOOP (`qwen3::finetune`'s
//!   shape) over a real dataset -- this phase proves the wiring descends, not a
//!   production fine-tune.
//! * Not done: INT8, KV-cached decode, EOS early-stop, sampling beyond
//!   greedy, a dedicated `brain deepseekocr` verb, and the wgpu backend (see
//!   [`caps`]'s header for the `crates/sam1` corruption that forces CPU).

pub mod caps;
pub mod config;
pub mod encoder;
pub mod import;
pub mod layout;
pub mod model;
pub mod preprocess;
pub mod prompt;
pub mod rows;
pub mod train;

pub use config::DeepseekOcrConfig;
pub use encoder::{DeepEncoder, GLUE_PIPELINES};
pub use layout::{RowGather, RowGatherIds, LAYOUT_PIPELINES};
pub use model::DeepseekOcr;
pub use preprocess::{preprocess_image, Fit};
pub use prompt::{build_prompt, ImageTokens, Prompt};
pub use rows::{row_plan, RowPlan, Src, ViewGrid};

/// Stage wall time to stderr when `BRAIN_PROFILE` is set -- the coarse timeline
/// above the per-kernel `BRAIN_PROFILE` table (which only ever instruments GPU
/// kernel dispatch time, not host-side work like weight streaming or a
/// SAM/CLIP/glue round-trip through `Vec<f32>`). Shared by [`caps`] (the
/// load/generate timeline) and [`encoder`] (the vision-tower breakdown), so
/// both print through the same gate and format.
pub(crate) fn stage_time(name: &str, since: std::time::Instant) {
    if std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false) {
        eprintln!("stage {name}: {:.1} ms", since.elapsed().as_secs_f64() * 1e3);
    }
}

/// How a composite gets a device for each sub-model's pipeline list.
///
/// Every stage here is a separate `gpu_core::Gpu` because every stage has its
/// own kernel set; a test hands in `gpu_core::testgpu::dev` (the pooled test
/// device, keyed on the pipeline slice address) and production hands in
/// `Gpu::new`. Passing four `Gpu` handles positionally instead would make the
/// call sites unreadable and the ORDER load-bearing.
pub type DeviceFactory<'a> = &'a dyn Fn(&'static [(&'static str, &'static str)]) -> gpu_core::Gpu;
