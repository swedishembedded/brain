// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Prefix-scoped reads of a GGUF's KV metadata.
//!
//! GGUF's own convention is that (almost) every hyperparameter lives under an
//! architecture prefix with a **standardized suffix**: `{arch}.block_count`,
//! `{arch}.embedding_length`, `{arch}.attention.head_count`,
//! `{arch}.attention.layer_norm_rms_epsilon`, `{arch}.rope.freq_base`, … The
//! prefix is the only per-model part, so hand-building
//! `format!("{arch}.block_count")` at every call site - which is what the
//! first GGUF importer in this tree did - repeats the architecture name once
//! per field and gets the typing (`as_u64` vs the `F32` variant) wrong in a
//! different way each time.
//!
//! [`ArchKv`] binds the prefix once; a config-extraction function then reads
//! as a declarative list of suffixes. It is deliberately *thin*: the escape
//! hatch ([`ArchKv::gguf`]) is right there for the architecture-specific rest
//! (tensor shapes as ground truth, non-conforming keys, unprefixed keys),
//! because no fixed schema survives contact with a real converter - the
//! DeepSeek-OCR mmproj in this crate's own tests declares a
//! `clip.vision.feed_forward_length` that its tensors flatly contradict.

use checkpoint::gguf::{GgufValue, MmapGguf};

/// The `general.architecture` value, the primary dispatch key of any GGUF.
pub fn architecture(mg: &MmapGguf) -> Option<&str> {
    mg.kv().get("general.architecture").and_then(|v| v.as_str())
}

/// A prefix-scoped view of one GGUF's KV metadata: every lookup is
/// `{prefix}.{suffix}`.
pub struct ArchKv<'a> {
    mg: &'a MmapGguf,
    prefix: String,
}

impl<'a> ArchKv<'a> {
    /// Scope reads to an explicit prefix (usually the architecture name, but
    /// any dotted prefix works - see [`ArchKv::scoped`]).
    pub fn new(mg: &'a MmapGguf, prefix: &str) -> ArchKv<'a> {
        ArchKv { mg, prefix: prefix.to_string() }
    }

    /// Scope reads to the file's own `general.architecture`.
    pub fn from_architecture(mg: &'a MmapGguf) -> Result<ArchKv<'a>, String> {
        let arch = architecture(mg).ok_or("gguf: general.architecture missing or not a string")?;
        Ok(ArchKv { mg, prefix: arch.to_string() })
    }

    /// Scope reads to `arch`, failing when the file declares something else -
    /// the guard every per-architecture `config_from_gguf` opens with, so a
    /// mismatched file is rejected by name instead of silently yielding a
    /// config full of defaults.
    pub fn expect_architecture(mg: &'a MmapGguf, arch: &str) -> Result<ArchKv<'a>, String> {
        let got = architecture(mg).unwrap_or("");
        if got != arch {
            return Err(format!("gguf: expected general.architecture={arch:?}, got {got:?}"));
        }
        Ok(ArchKv { mg, prefix: arch.to_string() })
    }

    /// A deeper view: `self.scoped("vision")` on prefix `clip` reads
    /// `clip.vision.*`. Sub-towers (`clip.vision`, `clip.vision.sam`) are the
    /// reason this exists.
    pub fn scoped(&self, sub: &str) -> ArchKv<'a> {
        ArchKv { mg: self.mg, prefix: format!("{}.{sub}", self.prefix) }
    }

    /// The prefix every lookup is made under.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The underlying file - the escape hatch for tensor shapes, unprefixed
    /// keys and anything else that does not follow the suffix convention.
    pub fn gguf(&self) -> &'a MmapGguf {
        self.mg
    }

    /// The raw value at `{prefix}.{suffix}`.
    pub fn get(&self, suffix: &str) -> Option<&'a GgufValue> {
        self.mg.kv().get(&format!("{}.{suffix}", self.prefix))
    }

    /// An unsigned-integer scalar (any width).
    pub fn u32(&self, suffix: &str) -> Option<u32> {
        self.get(suffix).and_then(|v| v.as_u64()).map(|v| v as u32)
    }

    /// [`u32`](ArchKv::u32), or `default` when the key is absent.
    pub fn u32_or(&self, suffix: &str, default: u32) -> u32 {
        self.u32(suffix).unwrap_or(default)
    }

    /// [`u32`](ArchKv::u32), erroring by full key name when absent.
    pub fn req_u32(&self, suffix: &str) -> Result<u32, String> {
        self.u32(suffix).ok_or_else(|| format!("gguf: missing {}.{suffix}", self.prefix))
    }

    /// A float scalar. Integers are accepted and widened, because converters
    /// write e.g. `rope.freq_base` as either.
    pub fn f32(&self, suffix: &str) -> Option<f32> {
        match self.get(suffix) {
            Some(GgufValue::F32(v)) => Some(*v),
            Some(GgufValue::F64(v)) => Some(*v as f32),
            Some(v) => v.as_u64().map(|v| v as f32),
            None => None,
        }
    }

    /// [`f32`](ArchKv::f32), or `default` when the key is absent.
    pub fn f32_or(&self, suffix: &str, default: f32) -> f32 {
        self.f32(suffix).unwrap_or(default)
    }

    /// [`f32`](ArchKv::f32), erroring by full key name when absent.
    pub fn req_f32(&self, suffix: &str) -> Result<f32, String> {
        self.f32(suffix).ok_or_else(|| format!("gguf: missing {}.{suffix}", self.prefix))
    }

    /// A string scalar.
    pub fn str(&self, suffix: &str) -> Option<&'a str> {
        self.get(suffix).and_then(|v| v.as_str())
    }

    /// A boolean scalar.
    pub fn bool(&self, suffix: &str) -> Option<bool> {
        match self.get(suffix) {
            Some(GgufValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// A float array (`image_mean`/`image_std` and friends). `None` when the
    /// key is absent or is not an array; non-float elements widen like
    /// [`f32`](ArchKv::f32).
    pub fn f32_array(&self, suffix: &str) -> Option<Vec<f32>> {
        let GgufValue::Array(items) = self.get(suffix)? else { return None };
        items
            .iter()
            .map(|v| match v {
                GgufValue::F32(x) => Some(*x),
                GgufValue::F64(x) => Some(*x as f32),
                other => other.as_u64().map(|x| x as f32),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::gguf_write::{write, TensorOut};

    /// A synthetic GGUF whose KV exercises every accessor, written to a temp
    /// path (the reader is mmap-only, so there is no in-memory constructor).
    fn synthetic(path: &str) {
        let kvs = vec![
            ("general.architecture".to_string(), GgufValue::String("toy".to_string())),
            ("toy.block_count".to_string(), GgufValue::U32(7)),
            ("toy.attention.head_count".to_string(), GgufValue::U16(3)),
            ("toy.attention.layer_norm_rms_epsilon".to_string(), GgufValue::F32(1e-5)),
            ("toy.rope.freq_base".to_string(), GgufValue::U32(10_000)),
            ("toy.rope.scaling.type".to_string(), GgufValue::String("yarn".to_string())),
            ("toy.vision.embedding_length".to_string(), GgufValue::U32(64)),
            ("toy.has_encoder".to_string(), GgufValue::Bool(true)),
            (
                "toy.image_mean".to_string(),
                GgufValue::Array(vec![GgufValue::F32(0.5), GgufValue::F32(0.25), GgufValue::F32(0.0)]),
            ),
        ];
        let t = TensorOut { name: "w".to_string(), shape: vec![2], ty: 0, data: vec![0u8; 8] };
        write(path, &kvs, &[t], 32).unwrap();
    }

    fn temp(tag: &str) -> String {
        std::env::temp_dir().join(format!("gguf-kv-{tag}-{}.gguf", std::process::id())).to_string_lossy().into_owned()
    }

    #[test]
    fn scoped_reads_resolve_the_standard_suffix_convention() {
        let path = temp("suffix");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();

        let kv = ArchKv::from_architecture(&mg).unwrap();
        assert_eq!(kv.prefix(), "toy");
        assert_eq!(kv.req_u32("block_count").unwrap(), 7);
        // Width-agnostic: head_count is stored U16, read as u32.
        assert_eq!(kv.u32("attention.head_count"), Some(3));
        assert_eq!(kv.req_f32("attention.layer_norm_rms_epsilon").unwrap(), 1e-5);
        // An integer-typed float key still reads as f32.
        assert_eq!(kv.f32("rope.freq_base"), Some(10_000.0));
        assert_eq!(kv.str("rope.scaling.type"), Some("yarn"));
        assert_eq!(kv.bool("has_encoder"), Some(true));
        assert_eq!(kv.f32_array("image_mean"), Some(vec![0.5, 0.25, 0.0]));
        // A sub-tower prefix.
        assert_eq!(kv.scoped("vision").req_u32("embedding_length").unwrap(), 64);
        assert_eq!(kv.scoped("vision").prefix(), "toy.vision");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn absent_keys_default_or_error_by_full_name() {
        let path = temp("absent");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let kv = ArchKv::new(&mg, "toy");

        assert_eq!(kv.u32("expert_count"), None);
        assert_eq!(kv.u32_or("expert_count", 0), 0);
        assert_eq!(kv.f32_or("rope.scaling.factor", 1.0), 1.0);
        // The error must name the FULL key, or a missing-key report sends the
        // reader hunting for the wrong string in the file.
        let err = kv.req_u32("expert_count").unwrap_err();
        assert!(err.contains("toy.expert_count"), "error must name the full key, got {err:?}");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn architecture_mismatch_is_rejected_up_front() {
        let path = temp("arch");
        synthetic(&path);
        let mg = MmapGguf::open(&path).unwrap();

        assert_eq!(architecture(&mg), Some("toy"));
        assert!(ArchKv::expect_architecture(&mg, "toy").is_ok());
        let Err(err) = ArchKv::expect_architecture(&mg, "other") else {
            panic!("a mismatched architecture must be rejected");
        };
        assert!(err.contains("\"other\"") && err.contains("\"toy\""), "got {err:?}");

        std::fs::remove_file(&path).ok();
    }
}
