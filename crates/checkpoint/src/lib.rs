// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared weight container, backed by safetensors (see [`st`]). Used for
//! inference weights and training checkpoints. The public `Container` API is
//! preserved: `header` carries `{"config": ...}` and every tensor has role `""`
//! (safetensors has no role concept). Reads/writes delegate to [`st`].

use std::collections::HashMap;

use serde_json::Value;

pub mod safetensors;
pub mod st;
pub mod gguf;
pub mod quant;
#[cfg(not(target_arch = "wasm32"))]
pub mod gguf_write;
#[cfg(not(target_arch = "wasm32"))]
pub mod mmap;
#[cfg(not(target_arch = "wasm32"))]
pub mod weightio;
pub mod remap;
pub mod torchpt;
pub mod zipread;

/// A by-name source of f32 tensor data for **streaming** model construction.
/// Both the eager `HashMap<String, Vec<f32>>` (a whole-model host copy) and the
/// mmap-backed [`weightio::WeightReader`] (one tensor decoded on demand)
/// implement it, so a builder can pull each weight, upload it to the device, and
/// drop it — peak host allocation ≈ one tensor of f32 — without ever holding the
/// entire model as a map (the second host copy the eager `by_role("")` load kept
/// alongside the device weights).
pub trait TensorSource {
    /// Invoke `f` with tensor `name`'s f32 data if present; returns whether it
    /// was found. The slice is valid only for the call — a `WeightReader`
    /// decodes into a temporary that is dropped on return (streaming), a
    /// `HashMap` lends its stored vec (no copy).
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool;

    /// Zero-copy path: `name`'s bytes AS ALREADY-STORED `u32` words, borrowed
    /// straight from wherever the source keeps them — no allocation, no
    /// decode. Only `Some` when the on-disk (or in-memory) representation is
    /// already exactly what the device binds (`F32`/packed-int8 `U32`) and,
    /// for an mmap-backed source, the byte range is 4-byte aligned. `None`
    /// means "not zero-copyable here" — the caller falls back to
    /// [`with_tensor_chunks`](Self::with_tensor_chunks) or
    /// [`with_tensor`](Self::with_tensor). Default: never zero-copyable (safe
    /// for any implementor that doesn't override it).
    fn raw_words(&self, _name: &str) -> Option<&[u32]> {
        None
    }

    /// Ordered chunks of at most `max_elems` f32, each decoded into a SINGLE
    /// reused scratch buffer owned by this call — so peak *extra* host
    /// allocation is `O(max_elems)`, never `O(tensor)`. Returns whether the
    /// tensor was found (`f` is never called if not). Default: materializes
    /// the whole tensor via [`with_tensor`](Self::with_tensor) and hands it
    /// over as one chunk at offset 0 — i.e. reproduces exactly what every
    /// implementor already does today. An mmap-backed source overrides this
    /// to decode incrementally instead.
    fn with_tensor_chunks(&self, name: &str, _max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        self.with_tensor(name, &mut |d| f(0, d))
    }

    /// Element count of `name`, if cheaply known without decoding. Default:
    /// unknown (`None`) — callers that need it fall back to `with_tensor`.
    fn numel(&self, _name: &str) -> Option<usize> {
        None
    }
}

impl TensorSource for HashMap<String, Vec<f32>> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.get(name) {
            Some(v) => {
                f(v);
                true
            }
            None => false,
        }
    }
    /// Already f32 in host memory — a bit-cast view, not a new allocation.
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.get(name).map(|v| bytemuck::cast_slice::<f32, u32>(v))
    }
    fn numel(&self, name: &str) -> Option<usize> {
        self.get(name).map(|v| v.len())
    }
}

/// One tensor read from a container (role is "" if the header omits it).
pub struct LoadedTensor {
    pub name: String,
    pub role: String,
    pub data: Vec<f32>,
}

pub struct Container {
    pub header: Value,
    pub tensors: Vec<LoadedTensor>,
}

impl Container {
    /// Tensors whose role matches `role`, keyed by name.
    pub fn by_role(&self, role: &str) -> HashMap<String, Vec<f32>> {
        self.tensors
            .iter()
            .filter(|t| t.role == role)
            .map(|t| (t.name.clone(), t.data.clone()))
            .collect()
    }
    pub fn find(&self, name: &str, role: &str) -> Option<&Vec<f32>> {
        self.tensors
            .iter()
            .find(|t| t.name == name && t.role == role)
            .map(|t| &t.data)
    }
}

/// Parse a weight container from an in-memory byte slice. This is the portable
/// core: native `load` reads the file then calls this; the browser entry point
/// (`web::run_inference`) passes the fetched bytes directly, since there is no
/// `std::fs` in a browser. Backed by safetensors; every tensor gets role `""`.
pub fn parse(bytes: &[u8]) -> Container {
    let m = st::parse_safetensors(bytes).expect("parse safetensors");
    let header = serde_json::json!({ "config": m.config() });
    let tensors = m
        .tensors
        .into_iter()
        .map(|(name, data)| LoadedTensor { name, role: String::new(), data })
        .collect();
    Container { header, tensors }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load(path: &str) -> Container {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {path}"));
    parse(&bytes)
}

/// Write a checkpoint: `config` is the model config object, `tensors` is an
/// ordered list of (name, shape, data). Delegates to [`st::save_safetensors`],
/// which writes atomically (tmp + rename).
#[cfg(not(target_arch = "wasm32"))]
pub fn save(path: &str, config: Value, tensors: &[(String, Vec<u64>, Vec<f32>)]) {
    st::save_safetensors(path, tensors, &config, None).expect("save safetensors");
}

/// Same as [`save`], but attaches a [`st::ModelCard`] to the checkpoint's
/// metadata — the family/id every servable model needs for
/// `crates/cli/src/model_dir.rs::discover()` to auto-register it. An
/// additive sibling, not a `save` signature change: `save`'s ~30 existing
/// call sites (training/research crates that were never meant to be
/// servable) stay untouched; only a family's real save path that wants to
/// be auto-discoverable switches to this one.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_carded(path: &str, config: Value, tensors: &[(String, Vec<u64>, Vec<f32>)], card: &st::ModelCard) {
    st::save_safetensors(path, tensors, &config, Some(card)).expect("save safetensors");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("moe_rs_ckpt_test_{}.bin", std::process::id()));
        let p = path.to_str().unwrap();
        let cfg = serde_json::json!({"d_model": 8, "n_layers": 1});
        let tensors = vec![
            ("a".to_string(), vec![2u64, 2], vec![1.0f32, -2.5, 3.25, 4.0]),
            ("b".to_string(), vec![3u64], vec![0.1f32, 0.2, 0.3]),
        ];
        save(p, cfg, &tensors);
        let c = load(p);
        assert_eq!(c.header["config"]["d_model"].as_u64().unwrap(), 8);
        let map = c.by_role(""); // saved tensors have no role
        assert_eq!(map["a"], vec![1.0, -2.5, 3.25, 4.0]);
        assert_eq!(map["b"], vec![0.1, 0.2, 0.3]);
        assert_eq!(c.find("a", "").unwrap().len(), 4);
        assert!(c.find("a", "init").is_none()); // role filter works
        std::fs::remove_file(p).ok();
    }

    /// `save_carded` must round-trip the card through `st::read_card` -- the
    /// exact reader `model_dir::discover()` uses to find a servable model --
    /// while `save`'s own plain (uncarded) path must keep reporting `None`,
    /// so switching one family to `save_carded` cannot accidentally make an
    /// untouched family's checkpoints look carded.
    #[test]
    fn save_carded_roundtrips_the_card_plain_save_does_not() {
        let dir = std::env::temp_dir();
        let carded_path = dir.join(format!("checkpoint-carded-test-{}.safetensors", std::process::id()));
        let plain_path = dir.join(format!("checkpoint-plain-test-{}.safetensors", std::process::id()));
        let (cp, pp) = (carded_path.to_str().unwrap(), plain_path.to_str().unwrap());
        let cfg = serde_json::json!({"d_model": 8});
        let tensors = vec![("w".to_string(), vec![2u64], vec![1.0f32, 2.0])];
        let card = st::ModelCard::new("brain/toy", "toy");

        save_carded(cp, cfg.clone(), &tensors, &card);
        let got = st::read_card(cp).unwrap();
        assert_eq!(got, Some(card), "save_carded must round-trip the exact card written");

        save(pp, cfg, &tensors);
        assert_eq!(st::read_card(pp).unwrap(), None, "plain save must stay cardless");

        std::fs::remove_file(cp).ok();
        std::fs::remove_file(pp).ok();
    }
}
