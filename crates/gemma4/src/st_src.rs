// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Gemma-4 text tower's original safetensors checkpoint, read on demand.
//!
//! Swedish Embedded AB implements streamed loading of large model checkpoints
//! for its clients. If your team needs expertise in running big text encoders
//! within a fixed host and device memory budget then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The released LTX encoder (`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`)
//! is 26 GB of bf16. Read whole, it becomes ~49 GB of host f32 before the
//! first token is embedded, and an encode spends its time copying that. This
//! is the same checkpoint memory-mapped and presented in this crate's
//! canonical name space through a declarative [`RemapSource`] plan - the
//! [`crate::gguf_src`] treatment for the file as shipped, with no conversion
//! on disk - so [`crate::forward_streamed`] decodes one layer at a time.

use std::collections::HashMap;

use checkpoint::mmap::MmapSafetensors;
use checkpoint::remap::{Fetch, RemapSource};

use crate::config::Gemma4Config;
use crate::gguf_src::source_name;
use crate::import::gemma4_tensor_manifest;

/// The embedded tokenizer's tensor name, the same in every LTX encoder file.
const TOKENIZER_TENSOR: &str = "tokenizer_json";

/// The checkpoint at `path`, memory-mapped and presented in this crate's
/// canonical tensor names. Validated - names and sizes only, both ways
/// ([`crate::import::check_checkpoint_names`]) - before any tensor byte is
/// decoded.
pub fn open_safetensors(path: &str, cfg: &Gemma4Config) -> Result<RemapSource<'static>, String> {
    let m = MmapSafetensors::open(path)?;
    crate::import::check_checkpoint_names(path, cfg, m.names().iter().map(String::as_str), |name| m.numel(name))?;
    let plan: HashMap<String, Fetch> = gemma4_tensor_manifest(cfg).into_iter().map(|(name, _)| { let src = source_name(&name); (name, Fetch::Whole(src)) }).collect();
    Ok(RemapSource::owning(Box::new(m), plan))
}

/// The tokenizer embedded in the checkpoint at `path` (its `tokenizer_json`
/// tensor), the safetensors twin of [`crate::Gemma4GgufSource::tokenizer`].
pub fn safetensors_tokenizer(path: &str) -> Result<data::llama_bpe::LlamaBpe, String> {
    let m = MmapSafetensors::open(path)?;
    let mut bytes = Vec::new();
    if !m.with_tensor_chunks(TOKENIZER_TENSOR, 1 << 20, &mut |_, chunk| bytes.extend_from_slice(chunk)) {
        return Err(format!("gemma4: {path} carries no {TOKENIZER_TENSOR} tensor"));
    }
    crate::tokenizer::parse_tokenizer(&crate::tokenizer::bytes_from_f32(TOKENIZER_TENSOR, &bytes)?)
}
