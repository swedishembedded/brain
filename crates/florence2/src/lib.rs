// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Florence-2-base: DaViT vision encoder + BART-style encoder-decoder,
//! native text-conditioned open-vocabulary detection/phrase grounding.
//!
//! Chosen as brain's first UI-element visual-grounding oracle: 0.23B
//! params, CPU-viable on hardware too small for a 4B+ VLM. DaViT and the
//! BART-style encoder-decoder are built from existing brain kernels/builders
//! piece by piece (windowed attention, channel-attention-via-transpose,
//! cross-attention-over-a-fixed-encoder-output) rather than new
//! from-scratch kernel authoring - see each module's own doc for which
//! existing piece it reuses.

pub mod grounding;
pub mod text;
pub mod tokenizer;
pub mod vision;
