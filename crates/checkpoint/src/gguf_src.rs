// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A memory-mapped GGUF presented under a MODEL's own tensor names.
//!
//! Swedish Embedded AB implements checkpoint-loading infrastructure for its
//! clients. If your team needs expertise in loading a quantized GGUF straight
//! into a model's own tensor-name space, with no ahead-of-time conversion and
//! no whole-model host materialization, then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! Three model crates (`wan`, `ltxv`, `gemma4`) each hand-rolled a
//! `TensorSource` wrapper over [`crate::gguf::MmapGguf`] that translates a
//! canonical (model-native) tensor name to the name the GGUF actually stores
//! it under, and forwards every read through. The three differed in exactly
//! ONE expression - the translation itself (a lookup table, the identity, a
//! prefix rewrite) - and were otherwise five verbatim forwarding methods
//! each. Which is how one of them shipped without `raw_words` or
//! `with_tensor_chunks` at all: every tensor materialized whole on every
//! read path, with nothing to catch it, because there were three copies of
//! the boilerplate to keep in sync and only two of them got both methods.
//! [`GgufSource`] is the one implementation; a method added to
//! [`crate::TensorSource`] (like `raw_blocks`) now reaches every GGUF-backed
//! model that uses this at once.

use std::collections::HashMap;

use crate::gguf::{BlockLayout, MmapGguf};
use crate::TensorSource;

/// A memory-mapped GGUF, read under a caller-chosen tensor name space rather
/// than the file's own. OWNS its mapping (unlike [`crate::remap::RemapSource`],
/// which borrows an inner source) - a loader opens a path and hands the
/// result around.
pub struct GgufSource {
    mg: MmapGguf,
    /// canonical (model) name -> the name the GGUF stores it under.
    plan: HashMap<String, String>,
}

impl GgufSource {
    /// `plan` maps each canonical name this model wants to read to the name
    /// the GGUF actually carries it under. A canonical name with no entry
    /// simply is not readable through this source (`with_tensor` -> `false`,
    /// not a panic) - a model's own coverage check reports that in the
    /// model's own vocabulary, not here.
    pub fn renaming(mg: MmapGguf, plan: HashMap<String, String>) -> GgufSource {
        GgufSource { mg, plan }
    }

    /// Every tensor the file carries, under its own name unchanged - for a
    /// checkpoint whose on-disk names already ARE the model's canonical
    /// names (no rename step at all).
    pub fn identity(mg: MmapGguf) -> GgufSource {
        let plan = mg.names().iter().map(|n| (n.clone(), n.clone())).collect();
        GgufSource { mg, plan }
    }

    /// The underlying mapped reader, for metadata a caller wants directly
    /// (KV, `ModelCard`, the tokenizer, …).
    pub fn gguf(&self) -> &MmapGguf {
        &self.mg
    }

    /// `canonical`'s on-disk name, if this source's plan covers it - the
    /// hook a caller needs to ask the reader something name-keyed that
    /// [`TensorSource`] itself does not expose (`gemma4::gguf_src`'s
    /// `dtype` is the worked example).
    pub fn source_name(&self, canonical: &str) -> Option<&str> {
        self.plan.get(canonical).map(String::as_str)
    }
}

impl TensorSource for GgufSource {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.source_name(name) {
            Some(src) => self.mg.with_tensor(src, f),
            None => false,
        }
    }

    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.mg.raw_words(self.source_name(name)?)
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        match self.source_name(name) {
            Some(src) => self.mg.with_tensor_chunks(src, max_elems, f),
            None => false,
        }
    }

    fn raw_blocks(&self, name: &str) -> Option<(BlockLayout, &[u8])> {
        self.mg.raw_blocks(self.source_name(name)?)
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.mg.numel(self.source_name(name)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_write::{write, TensorOut};

    fn synthetic(path: &str, tensors: &[(&str, &[f32])]) {
        let out: Vec<TensorOut> =
            tensors.iter().map(|(n, v)| TensorOut { name: n.to_string(), shape: vec![v.len()], ty: 0, data: v.iter().flat_map(|f| f.to_le_bytes()).collect() }).collect();
        write(path, &[], &out, 32).unwrap();
    }

    fn tmp(tag: &str) -> String {
        std::env::temp_dir().join(format!("brain-gguf-src-{tag}-{}.gguf", std::process::id())).to_string_lossy().into_owned()
    }

    #[test]
    fn renaming_translates_every_read_path_including_raw_blocks() {
        let path = tmp("renaming");
        synthetic(&path, &[("model.w", &[1.0, 2.0, 3.0, 4.0])]);
        let mg = MmapGguf::open(&path).unwrap();
        let mut plan = HashMap::new();
        plan.insert("w".to_string(), "model.w".to_string());
        let src = GgufSource::renaming(mg, plan);

        let mut got = None;
        assert!(src.with_tensor("w", &mut |d| got = Some(d.to_vec())));
        assert_eq!(got.unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(src.numel("w"), Some(4));
        assert!(src.raw_words("w").is_some(), "plain F32 must be zero-copyable through the rename");

        // The canonical-space caller never sees the on-disk name.
        assert!(!src.with_tensor("model.w", &mut |_| panic!("must not resolve the on-disk name directly")));
        assert!(!src.with_tensor("does.not.exist", &mut |_| panic!("must not call")));
        assert_eq!(src.source_name("w"), Some("model.w"));
        assert_eq!(src.source_name("does.not.exist"), None);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn identity_needs_no_plan_entries_written_by_hand() {
        let path = tmp("identity");
        synthetic(&path, &[("a", &[1.0]), ("b", &[2.0, 3.0])]);
        let mg = MmapGguf::open(&path).unwrap();
        let src = GgufSource::identity(mg);

        for (name, want) in [("a", vec![1.0]), ("b", vec![2.0, 3.0])] {
            let mut got = None;
            assert!(src.with_tensor(name, &mut |d| got = Some(d.to_vec())), "{name}");
            assert_eq!(got.unwrap(), want, "{name}");
        }
        assert!(!src.with_tensor("c", &mut |_| panic!("must not call")));
        std::fs::remove_file(&path).ok();
    }

    /// The capability the three original hand-rolled wrappers did not all
    /// have (one of them had neither `raw_words` nor `with_tensor_chunks`,
    /// and none had `raw_blocks` - it did not exist yet): a Q8_0 tensor's
    /// already-quantized blocks are reachable through the rename, not just
    /// its dequantized f32.
    #[test]
    fn raw_blocks_is_reachable_through_a_rename() {
        let block = crate::quant::quantize_par(crate::gguf::TYPE_Q8_0, &[5.0f32; 32]).unwrap();
        let path = tmp("rawblocks");
        write(&path, &[], &[TensorOut { name: "on.disk.name".to_string(), shape: vec![32], ty: crate::gguf::TYPE_Q8_0, data: block }], 32).unwrap();
        let mg = MmapGguf::open(&path).unwrap();
        let mut plan = HashMap::new();
        plan.insert("canonical.name".to_string(), "on.disk.name".to_string());
        let src = GgufSource::renaming(mg, plan);

        let (layout, bytes) = src.raw_blocks("canonical.name").expect("must resolve through the rename");
        assert_eq!(layout.ty, crate::gguf::GgmlType::Q8_0);
        assert_eq!(layout.numel, 32);
        assert_eq!(bytes.len(), 34, "one Q8_0 block");
        assert!(src.raw_blocks("on.disk.name").is_none(), "the on-disk name is not itself a canonical name");

        std::fs::remove_file(&path).ok();
    }
}
