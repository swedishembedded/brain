// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Direct GGUF loading for the Wan DiT - a [`checkpoint::TensorSource`] over a
//! [`checkpoint::gguf::MmapGguf`] that never materializes a converted
//! safetensors file on disk.
//!
//! Wraps the mmap plus the reference-name -> source-name table
//! [`crate::import::source_map`] already extracted from [`crate::import::
//! import_gguf`], so this loader and the ahead-of-time converter share one
//! naming table and cannot silently drift apart. Every read - `with_tensor`,
//! `raw_words`, `numel` - just translates the caller's native name through
//! that table and delegates to `MmapGguf`, which is what makes an F32 tensor
//! in the file (city96's release leaves every 1-D tensor plain F32) zero-copy
//! all the way to a device upload, and a quantized one (Q3_K, …) dequantize
//! on demand, one tensor at a time.

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;

use crate::config::WanConfig;
use crate::import::{dit_config_from_shapes, source_map};

/// A Wan DiT GGUF, opened for direct streaming access - no ahead-of-time
/// conversion, no whole-model host materialization.
pub struct WanGgufSource {
    mg: MmapGguf,
    /// Reference (native) tensor name -> the GGUF's own name carrying it.
    source_of: HashMap<String, String>,
    cfg: WanConfig,
}

impl WanGgufSource {
    /// Open `path`, derive the DiT variant from tensor SHAPES alone (no
    /// dequant), and build the reference<->source name map. Errors if the
    /// file is not a Wan transformer checkpoint the config table recognizes,
    /// or if its two name spaces disagree (see [`source_map`]).
    pub fn open(path: &str) -> Result<WanGgufSource, String> {
        let mg = MmapGguf::open(path)?;
        Self::from_mmap(mg)
    }

    /// [`Self::open`] over an already-mapped [`MmapGguf`] - the seam a caller
    /// that already opened the file (e.g. to read its `ModelCard`) uses to
    /// avoid a second `mmap()`.
    pub fn from_mmap(mg: MmapGguf) -> Result<WanGgufSource, String> {
        let shapes: Vec<(String, Vec<usize>)> =
            mg.names().iter().map(|n| (n.clone(), mg.shape(n).map(<[usize]>::to_vec).unwrap_or_default())).collect();
        let cfg = dit_config_from_shapes(&shapes)?;
        let source_of = source_map(&mg)?;
        Ok(WanGgufSource { mg, source_of, cfg })
    }

    /// The DiT variant this checkpoint's tensor shapes name.
    pub fn config(&self) -> &WanConfig {
        &self.cfg
    }

    fn resolve<'a>(&'a self, native: &str) -> Option<&'a str> {
        self.source_of.get(native).map(String::as_str)
    }
}

impl checkpoint::TensorSource for WanGgufSource {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.resolve(name) {
            Some(src) => self.mg.with_tensor(src, f),
            None => false,
        }
    }

    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.mg.raw_words(self.resolve(name)?)
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        match self.resolve(name) {
            Some(src) => self.mg.with_tensor_chunks(src, max_elems, f),
            None => false,
        }
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.mg.numel(self.resolve(name)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::dit_manifest;
    use checkpoint::gguf_write::{write, TensorOut};
    use checkpoint::TensorSource;

    /// Build a tiny native-spelled GGUF for `WanConfig::t2v_1_3b`'s full
    /// manifest, each tensor filled with distinct deterministic data.
    fn synthetic_gguf(path: &str) {
        let cfg = WanConfig::t2v_1_3b();
        let manifest = dit_manifest(&cfg);
        let mut seed = 0u64;
        let tensors: Vec<TensorOut> = manifest
            .into_iter()
            .map(|(name, shape)| {
                seed += 1;
                let n: usize = shape.iter().product();
                let data: Vec<u8> = (0..n).flat_map(|i| (((i as u64 + seed) % 997) as f32 * 0.01 - 5.0).to_le_bytes()).collect();
                TensorOut { name, shape, ty: 0u32 /* ggml type id: F32 */, data }
            })
            .collect();
        write(path, &[], &tensors, 32).unwrap();
    }

    #[test]
    fn open_derives_the_config_and_resolves_every_manifest_tensor() {
        let path = std::env::temp_dir()
            .join(format!("wan-gguf-src-test-{}.gguf", std::process::id()))
            .to_string_lossy()
            .into_owned();
        synthetic_gguf(&path);

        let src = WanGgufSource::open(&path).unwrap();
        assert_eq!(src.config().name, "t2v-1.3B");

        for (name, shape) in dit_manifest(src.config()) {
            let n: usize = shape.iter().product();
            let mut got = None;
            assert!(src.with_tensor(&name, &mut |d| got = Some(d.to_vec())), "missing {name}");
            assert_eq!(got.unwrap().len(), n, "{name}");
            assert_eq!(src.numel(&name), Some(n), "{name}");
        }
        assert!(!src.with_tensor("does.not.exist", &mut |_| panic!("must not call")));
        assert_eq!(src.numel("does.not.exist"), None);
        assert!(src.raw_words("does.not.exist").is_none());

        std::fs::remove_file(&path).ok();
    }

    /// Every F32 tensor in the synthetic file must be zero-copyable (the
    /// point of `raw_words` translating through the name map).
    #[test]
    fn raw_words_zero_copies_through_the_name_map() {
        let path = std::env::temp_dir()
            .join(format!("wan-gguf-src-rawwords-{}.gguf", std::process::id()))
            .to_string_lossy()
            .into_owned();
        synthetic_gguf(&path);
        let src = WanGgufSource::open(&path).unwrap();

        for (name, _) in dit_manifest(src.config()) {
            let words = src.raw_words(&name).unwrap_or_else(|| panic!("{name} should be zero-copyable (plain F32)"));
            let mut via_with_tensor = None;
            src.with_tensor(&name, &mut |d| via_with_tensor = Some(d.to_vec()));
            let via_words: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
            assert_eq!(via_words, via_with_tensor.unwrap(), "{name}");
        }
        std::fs::remove_file(&path).ok();
    }

    /// A file whose shapes don't name any known variant must fail at `open`,
    /// not silently succeed with a bogus config.
    #[test]
    fn open_rejects_a_checkpoint_that_is_not_a_wan_transformer() {
        let path = std::env::temp_dir()
            .join(format!("wan-gguf-src-bad-{}.gguf", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let tensors = vec![TensorOut { name: "not.a.wan.tensor".to_string(), shape: vec![4], ty: 0u32 /* ggml type id: F32 */, data: vec![0u8; 16] }];
        write(&path, &[], &tensors, 32).unwrap();
        let e = WanGgufSource::open(&path).err().expect("must be rejected");
        assert!(e.contains("patch_embedding.weight"), "{e}");
        std::fs::remove_file(&path).ok();
    }
}
