// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLIP-family text and image encoders: CLIP-L, OpenCLIP-bigG and EVA-CLIP
//! behind one config-driven graph.
//!
//! * [`config`] — the reference configurations and canonical tensor manifests.
//! * [`import`] — checkpoint import with two-way coverage validation.
//! * [`init`] — random init for the text tower (tests / gradient checks only).
//! * [`model`] — the graphs (`ClipText`, `EvaVision`).
//!
//! The **text** tower is trainable: `ClipText::new_train_on` adds the reverse
//! pass over the same SSA forward, gated by `gradcheck::check_clip`. The EVA
//! **image** tower is still forward-only, as are the `capability` Provider /
//! residency adapter / D-Bus surface and the CLIP BPE tokenizer (which belongs
//! in `crates/data` next to the GPT-2 and Qwen BPEs, not here).

pub mod caps;
pub mod config;
pub mod import;
pub mod init;
pub mod model;
