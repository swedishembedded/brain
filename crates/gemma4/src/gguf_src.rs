// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A quantized GGUF of the Gemma-4 text tower, read on demand.
//!
//! Swedish Embedded AB implements streamed, quantized model loading for its
//! clients. If your team needs expertise in running large checkpoints on
//! memory- and bandwidth-constrained accelerators then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # Why this exists
//!
//! The only way to load this model was `checkpoint::safetensors::read` on the
//! real bf16 checkpoint, which decodes **every** tensor to f32 up front - a
//! 26 GB file becoming ~49 GB of resident host `Vec<f32>` before the first
//! token is embedded. Two costs follow from that and only one of them is
//! obvious. The memory is the obvious one. The other is that a text encode
//! reads 26 GB of checkpoint that nothing has touched, and cold sequential
//! storage, not arithmetic, is what sets how long that takes.
//!
//! A `Q8_0` GGUF produced by `brain quantize` addresses both: it is roughly
//! half the bytes, and it is memory-mapped, so a caller that consumes one
//! layer at a time never materializes more than one layer.
//!
//! # Name space
//!
//! The GGUF carries the checkpoint's own names (`model.layers.0...`); this
//! crate's canonical space is the post-`model.`-strip one
//! (`layers.0...`) that `crate::import::gemma4_tensor_manifest` defines and
//! `crate::block`'s `tget` calls use. The mapping is exactly
//! `crate::import::classify`'s, run in reverse, and it is applied here so
//! that a caller reading through [`checkpoint::TensorSource`] sees the same
//! names an `import_gemma4` map would have given it. Nothing downstream
//! needs to know which of the two loaders it is talking to.

use std::collections::{BTreeSet, HashMap};

use checkpoint::gguf::MmapGguf;
use checkpoint::gguf_src::GgufSource;

use crate::config::Gemma4Config;
use crate::import::gemma4_tensor_manifest;

/// `general.architecture` a Gemma-4 text-tower GGUF must declare - what
/// `brain quantize --arch gemma4` writes, and what this refuses to open
/// without. Opening some other model's GGUF and finding the tensors missing
/// one at a time is a far worse error than one refusal at the top.
pub const GGUF_ARCHITECTURE: &str = "gemma4";

/// A memory-mapped Gemma-4 GGUF, presented in this crate's canonical tensor
/// name space.
pub struct Gemma4GgufSource {
    src: GgufSource,
}

/// This crate's canonical name -> the checkpoint name a GGUF stores it under.
/// The exact inverse of `crate::import::classify`'s `Weight` arm: everything
/// the text tower owns is `model.`-prefixed in the checkpoint except the
/// `text_embedding_projection.*` head, which LTX adds at the top level.
fn source_name(canonical: &str) -> String {
    if canonical.starts_with("text_embedding_projection.") {
        canonical.to_string()
    } else {
        format!("model.{canonical}")
    }
}

impl Gemma4GgufSource {
    /// Open and validate. Validation is **shape-only** and happens before a
    /// single tensor byte is decoded, mirroring `ltxv::import`'s own
    /// GGUF-shape gate: a wrong or truncated file is refused at open, not
    /// three layers into a forward pass.
    ///
    /// Two-way, over the text tower's own names: every manifest tensor must
    /// be present at the right element count, and every tensor the file
    /// carries must be one this crate recognizes - a weight, one of the
    /// deliberately-unused sibling towers, or an asset blob. An unrecognized
    /// name is an error, exactly as it is on the safetensors path, so the
    /// skip lists stay closed named sets rather than a catch-all.
    pub fn open(path: &str, cfg: &Gemma4Config) -> Result<Gemma4GgufSource, String> {
        let mg = MmapGguf::open(path)?;
        let arch = mg.kv().get("general.architecture").and_then(|v| v.as_str()).unwrap_or("");
        if arch != GGUF_ARCHITECTURE {
            return Err(format!("gemma4 gguf: {path} declares general.architecture '{arch}', expected '{GGUF_ARCHITECTURE}'"));
        }

        let manifest = gemma4_tensor_manifest(cfg);
        let mut expected: BTreeSet<String> = BTreeSet::new();
        for (name, shape) in &manifest {
            let src = source_name(name);
            let want: usize = shape.iter().product();
            match mg.shape(&src) {
                None => return Err(format!("gemma4 gguf: {path} is missing tensor {src}")),
                Some(s) => {
                    let got: usize = s.iter().product();
                    if got != want {
                        return Err(format!("gemma4 gguf: {src} holds {got} elements, expected {want} (shape {s:?} vs {shape:?})"));
                    }
                }
            }
            expected.insert(src);
        }

        // The reverse direction. `classify` owns which non-manifest names are
        // legitimately present (the vision/audio/multi-modal towers this
        // text-only path never reads, and the embedded asset blobs).
        let mut unknown: Vec<&str> = Vec::new();
        for name in mg.names() {
            if expected.contains(name) {
                continue;
            }
            if crate::import::is_recognized_non_weight(name) {
                continue;
            }
            unknown.push(name);
        }
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(format!("gemma4 gguf: {path} carries unrecognized tensors: {unknown:?}"));
        }

        let plan: HashMap<String, String> = manifest.into_iter().map(|(name, _)| { let src = source_name(&name); (name, src) }).collect();
        Ok(Gemma4GgufSource { src: GgufSource::renaming(mg, plan) })
    }

    /// The underlying reader, for metadata a caller wants directly.
    pub fn gguf(&self) -> &MmapGguf {
        self.src.gguf()
    }

    /// How each tensor is stored, at this crate's canonical name - `"Q8_0"`,
    /// `"F32"`, etc. Used by the quantized loader to tell an
    /// already-quantized weight from one it must quantize itself.
    pub fn dtype(&self, canonical: &str) -> Option<&'static str> {
        self.src.gguf().dtype(self.src.source_name(canonical)?)
    }

    /// The checkpoint-embedded `tokenizer.json` bytes. Read straight off the
    /// mapping under its OWN on-disk name - not part of the canonical
    /// rename plan, which covers only the text tower's weights.
    pub fn tokenizer_json(&self) -> Result<Vec<u8>, String> {
        let data = self
            .src
            .gguf()
            .tensor("tokenizer_json")
            .ok_or("gemma4 gguf: missing tokenizer_json tensor")?
            .map_err(|e| format!("gemma4 gguf: decoding tokenizer_json: {e}"))?;
        crate::tokenizer::bytes_from_f32("tokenizer_json", &data)
    }

    /// Extract + parse, the GGUF twin of `crate::tokenizer::load_tokenizer`.
    pub fn tokenizer(&self) -> Result<data::qwen_tokenizer::QwenBpe, String> {
        let bytes = self.tokenizer_json()?;
        data::qwen_tokenizer::QwenBpe::from_json_bytes(&bytes)
    }
}

impl checkpoint::TensorSource for Gemma4GgufSource {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        self.src.with_tensor(name, f)
    }

    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.src.raw_words(name)
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        self.src.with_tensor_chunks(name, max_elems, f)
    }

    fn raw_blocks(&self, name: &str) -> Option<(checkpoint::gguf::BlockLayout, &[u8])> {
        self.src.raw_blocks(name)
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.src.numel(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_name_mapping_is_the_inverse_of_the_importers_own_classification() {
        // A layer weight and the top-level LTX head - the two arms that
        // differ. If these ever disagreed with `classify`, the GGUF loader
        // and the safetensors loader would present different name spaces for
        // the same checkpoint and only one of them would work.
        assert_eq!(source_name("layers.0.mlp.gate_proj.weight"), "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(source_name("embed_tokens.weight"), "model.embed_tokens.weight");
        assert_eq!(
            source_name("text_embedding_projection.video_aggregate_embed.weight"),
            "text_embedding_projection.video_aggregate_embed.weight"
        );
        for canonical in ["layers.3.self_attn.q_proj.weight", "norm.weight", "text_embedding_projection.audio_aggregate_embed.bias"] {
            assert_eq!(
                crate::import::canonical_weight_name(&source_name(canonical)).as_deref(),
                Some(canonical),
                "round trip through the importer's own classification"
            );
        }
    }
}
