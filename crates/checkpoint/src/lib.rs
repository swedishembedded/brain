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
/// Native only: it writes a GGUF, and `gguf_write` needs `std::fs`.
#[cfg(not(target_arch = "wasm32"))]
pub mod quantize;
#[cfg(not(target_arch = "wasm32"))]
pub mod mmap;
#[cfg(not(target_arch = "wasm32"))]
pub mod weightio;
/// Native only: it wraps `gguf::MmapGguf`, which needs `std::fs`/mmap.
#[cfg(not(target_arch = "wasm32"))]
pub mod gguf_src;
#[cfg(not(target_arch = "wasm32"))]
pub mod load_progress;
pub mod remap;
pub mod split;
pub mod srccheck;
pub mod torchpt;
pub mod zipread;
#[cfg(test)]
mod testalloc;

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

    /// Ordered chunks of at most `max_elems` **packed `u32` words** - the
    /// bounded reader for brain's int8-native storage convention
    /// (`model::int8::quantize_weight`'s 4-int8-per-`u32` layout, written by
    /// `weightio::StWriter::write_u32`; its scale sibling is a separate
    /// `[n, k/32]` F32 tensor - `weightio::PACKED_INT8_LAYOUT`). Same `O(max_elems)` peak guarantee as
    /// [`with_tensor_chunks`](Self::with_tensor_chunks), for the dtype that
    /// one cannot serve (a packed tensor has no meaningful f32 decoding).
    ///
    /// Returns whether the tensor was found AND could be served as packed
    /// words. The default satisfies it from [`raw_words`](Self::raw_words)
    /// when that lends a zero-copy view, and otherwise reports `false` - a
    /// source with no packed-word representation (a plain f32 `HashMap`, a
    /// GGUF) genuinely has nothing to give here, and saying so is better than
    /// silently reinterpreting float bits as packed lanes.
    fn with_tensor_u32_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[u32])) -> bool {
        match self.raw_words(name) {
            Some(words) => {
                let chunk = if max_elems == 0 { words.len().max(1) } else { max_elems };
                for (i, part) in words.chunks(chunk).enumerate() {
                    f((i * chunk) as u64, part);
                }
                true
            }
            None => false,
        }
    }

    /// Element count of `name`, if cheaply known without decoding. Default:
    /// unknown (`None`) — callers that need it fall back to `with_tensor`.
    fn numel(&self, _name: &str) -> Option<usize> {
        None
    }

    /// [`raw_words`](Self::raw_words)'s sibling for storage that is NOT what
    /// the device binds but IS something a caller can consume without an fp32
    /// round trip - `name`'s bytes exactly as the source holds them, plus the
    /// block format that decodes them. The zero-fp32 path a Q8_0 tensor takes
    /// on its way to brain's packed-int8 layout
    /// (`gguf::int8_direct::try_i8_rect`), or a K-quant tensor uploaded
    /// verbatim to a kernel that decodes blocks in WGSL, both go through
    /// this.
    ///
    /// The lend is the WHOLE tensor; a caller wanting a sub-rectangle slices
    /// it itself using `layout.block_elems()`/`block_bytes()` to find the cut
    /// points - a quant block is the smallest independently decodable unit,
    /// so there is no meaningful finer-grained lend. Borrowed from the
    /// source's own storage: no allocation, valid for the lifetime of `&self`.
    ///
    /// Unlike [`raw_words`](Self::raw_words) there is no alignment
    /// precondition - `&[u8]` has nothing to align - so a correct
    /// implementation reports `Some` for every tensor whose on-disk
    /// representation is a known quantized block format, not selectively.
    ///
    /// Default: `None`, "this source has no quantized representation to
    /// lend". Correct for every f32-backed source (a plain `HashMap`,
    /// safetensors) and - the case that matters - for ANY wrapper that
    /// transforms tensor VALUES on read. A transform applied on
    /// [`with_tensor`](Self::with_tensor) but skipped here would hand the
    /// caller PRE-transform bytes, silently, and it is weights: see
    /// `qwen35::int8_gguf_resident::SsmALogFix`'s explicit refusal for the
    /// worked example, and `remap::RemapSource`'s `Fetch::Concat` arm for the
    /// structural case (a destination assembled from several source pieces
    /// has no single contiguous block range to lend).
    fn raw_blocks(&self, _name: &str) -> Option<(gguf::BlockLayout, &[u8])> {
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

/// The shape-carrying eager map several model crates use for a small,
/// wholly-materialized checkpoint (`s3dit::block::Tensors`, `vae`'s import
/// map) — the same role as `HashMap<String, Vec<f32>>` above, plus a shape
/// alongside each tensor's data. Defined here, not in each of those crates,
/// because the orphan rule blocks a foreign crate from implementing a
/// foreign trait for `HashMap` regardless of its type parameters — this is
/// the one place that can be done, for every crate that needs it.
impl TensorSource for HashMap<String, (Vec<usize>, Vec<f32>)> {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.get(name) {
            Some((_, data)) => {
                f(data);
                true
            }
            None => false,
        }
    }
    /// Already f32 in host memory — a bit-cast view, not a new allocation.
    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.get(name).map(|(_, data)| bytemuck::cast_slice::<f32, u32>(data))
    }
    fn numel(&self, name: &str) -> Option<usize> {
        self.get(name).map(|(_, data)| data.len())
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
    container(st::parse_safetensors(bytes).expect("parse safetensors"))
}

/// Repackage a parsed [`st::StModel`] as the public [`Container`] shape.
fn container(m: st::StModel) -> Container {
    let header = serde_json::json!({ "config": m.config() });
    let tensors = m
        .tensors
        .into_iter()
        .map(|(name, data)| LoadedTensor { name, role: String::new(), data })
        .collect();
    Container { header, tensors }
}

/// Read a whole checkpoint into host memory.
///
/// Eager by design - the ~100 call sites that use this want every tensor as
/// f32 - but eager about the *tensors* only. It goes through
/// [`st::load_safetensors`]'s mapping rather than reading the file into an
/// owned buffer first, so the peak is one copy of the model instead of the
/// file plus the model. A caller that wants only the config should reach for
/// [`read_config`], and one that wants one tensor at a time for
/// [`weightio::WeightReader`].
#[cfg(not(target_arch = "wasm32"))]
pub fn load(path: &str) -> Container {
    container(st::load_safetensors(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}")))
}

/// The model config of the checkpoint at `path`, for the cost of a header
/// parse - no tensor data is read.
///
/// This is what a call site wanting `block_size`, `vocab` or a whole
/// `SomeConfig::from_json` should use. The eager alternative,
/// `load(path).header["config"]`, materializes every tensor in the file as
/// f32 first and throws them away; on a real checkpoint that is a multi-GB
/// allocation to reach a few hundred bytes of JSON.
///
/// Reads **both** container formats, sniffed by content: a brain-native
/// safetensors answers with its `brain.config`, a GGUF with its KV map (which
/// is where a GGUF keeps the same information).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_config(path: &str) -> Value {
    weightio::WeightReader::open(path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
        .config()
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
    use crate::testalloc::peak_live;

    /// A checkpoint whose tensor blob is big enough that holding it twice is
    /// unmistakable in the peak, plus the byte counts to compare against.
    /// Returns `(path, config, tensor_bytes)`.
    fn big_checkpoint(tag: &str) -> (String, Value, usize) {
        let p = std::env::temp_dir()
            .join(format!("brain-ckpt-{}-{}-{}.safetensors", tag, std::process::id(), line!()))
            .to_str()
            .unwrap()
            .to_string();
        let cfg = serde_json::json!({"d_model": 64, "n_layers": 3, "name": "big"});
        let (n, ntensors) = (200_000usize, 8usize);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            (0..ntensors).map(|i| (format!("t{i}"), vec![n as u64], vec![i as f32; n])).collect();
        save(&p, cfg.clone(), &tensors);
        (p, cfg, n * ntensors * 4)
    }

    /// `load` must not hold the raw file bytes AND the decoded tensors at the
    /// same time. The tensors are F32 on disk, so the file is essentially the
    /// tensor blob: a `std::fs::read` + decode holds that blob twice over,
    /// while decoding straight out of a mapping holds it once.
    ///
    /// This is a *heap* measurement ([`crate::testalloc`]), which is what makes
    /// the distinction visible at all - a mapping's pages are not heap, so the
    /// owned-buffer copy is the only thing that can show up twice.
    #[test]
    fn load_does_not_hold_the_file_bytes_and_the_decoded_tensors_at_once() {
        let (p, cfg, tensor_bytes) = big_checkpoint("doubleread");

        let (c, peak) = peak_live(|| load(&p));

        // Still a correct, complete load - the cheaper route must change
        // nothing a caller can observe.
        assert_eq!(c.header["config"], cfg);
        assert_eq!(c.tensors.len(), 8);
        assert_eq!(c.find("t3", "").unwrap(), &vec![3.0f32; 200_000]);

        assert!(
            peak < tensor_bytes * 3 / 2,
            "load peak {peak} is not within ~one copy of the tensor blob ({tensor_bytes}) - the file bytes are being held alongside the decoded tensors"
        );
        std::fs::remove_file(&p).ok();
    }

    /// Reading a checkpoint's config must cost a header parse, not a whole
    /// model. Every call site that only wants `block_size`/`vocab` out of a
    /// multi-GB file rides on this.
    #[test]
    fn read_config_does_not_materialize_the_tensors() {
        let (p, cfg, tensor_bytes) = big_checkpoint("cfgonly");

        let (got, peak) = peak_live(|| read_config(&p));

        assert_eq!(got, cfg, "the config must be the same value the eager load reports");
        assert!(
            peak < tensor_bytes / 50,
            "read_config peak {peak} is not a header-sized read of a {tensor_bytes}-byte blob"
        );
        std::fs::remove_file(&p).ok();
    }

    /// ...and it must agree with what the eager path reports, exactly, for
    /// both a brain-native safetensors checkpoint and a GGUF - the two formats
    /// `weightio::WeightReader` sniffs between.
    #[test]
    fn read_config_agrees_with_the_eager_header_and_reads_gguf_too() {
        let (p, cfg, _) = big_checkpoint("agree");
        assert_eq!(read_config(&p), load(&p).header["config"]);
        assert_eq!(read_config(&p), cfg);
        std::fs::remove_file(&p).ok();

        // A GGUF has no `brain.config`; its KV map is its config, and the same
        // accessor must produce it rather than failing or reporting Null.
        let g = std::env::temp_dir().join(format!("brain-ckpt-agree-{}.gguf", std::process::id()));
        let g = g.to_str().unwrap();
        gguf_write::write(
            g,
            &[("general.architecture".to_string(), gguf::GgufValue::String("toy".to_string()))],
            &[gguf_write::TensorOut { name: "w".to_string(), shape: vec![2], ty: 0, data: vec![0u8; 8] }],
            32,
        )
        .unwrap();
        assert_eq!(read_config(g)["general.architecture"], serde_json::json!("toy"));
        std::fs::remove_file(g).ok();
    }

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
