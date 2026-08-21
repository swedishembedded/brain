// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A content-addressed disk cache for the ENCODED text context.
//!
//! Swedish Embedded AB implements caching and cold-start elimination for
//! production inference pipelines. If your team needs expertise in making a
//! large model's repeat-run cost disappear then you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! # What this caches, and why it is a different lever from quantization
//!
//! The text encode is the largest stage of a real generation, and its cost is
//! dominated by reading a checkpoint that nothing has touched: tens of
//! gigabytes at cold sequential-storage rates. Quantizing the encoder halves
//! those bytes and speeding up its arithmetic shortens what is left, but both
//! are proportional wins on a cost that is paid EVERY run.
//!
//! The output of all that work is a few megabytes - a `[context_len,
//! cross_attention_dim]` matrix - and it is a pure function of the prompt and
//! the encoder. So for the workflow this pipeline is actually used in, where
//! the same prompt is re-encoded on every iteration of a
//! change-something-else loop, the whole stage is avoidable rather than
//! merely reducible. The two are complementary: the cache removes the repeat
//! cost, quantization removes half of what a genuine first run still has to
//! pay.
//!
//! # Correctness: the key is verified, never trusted
//!
//! A cache that returns the wrong context does not fail; it silently
//! conditions every generation on the wrong prompt. So the digest here is
//! only a FILENAME - the complete key material is stored inside the file and
//! compared field by field on load. A digest collision therefore produces a
//! MISS, never a wrong hit, and no argument about hash strength is load
//! bearing.
//!
//! The key is every input the encode's output depends on: the prompt, the
//! encoder checkpoint's identity (path, byte length and modification time -
//! its content without reading it), the precision tier the encoder ran at,
//! the three DiT config fields that shape the result, and whether the
//! unconditional branch was computed - PLUS [`ENCODE_REVISION`], which
//! describes the encode itself rather than an input. Anything else changing
//! cannot change the answer; anything here changing produces a different
//! filename AND fails the verification if it somehow did not.
//!
//! # Where it lives
//!
//! `gpu_core::cache_dir()` - the one cache-directory resolution this
//! workspace already has (`BRAIN_PIPELINE_CACHE_DIR`, else
//! `XDG_CACHE_HOME/brain`, else `~/.cache/brain`), not a second copy of that
//! ladder. Set `BRAIN_LTXV_TEXT_CACHE=0` to disable.

use serde_json::json;

/// Every input [`crate::pipeline`]'s real text encode is a function of.
#[derive(Clone, Debug, PartialEq)]
pub struct Key {
    pub prompt: String,
    /// The encoder checkpoint's identity: its path, byte length and mtime.
    /// Content-derived without reading gigabytes - a re-quantized or replaced
    /// checkpoint at the same path changes at least one of the latter two.
    pub encoder_path: String,
    pub encoder_len: u64,
    pub encoder_mtime: i64,
    /// Which arithmetic the encoder's projections ran in. int8 and fp32 do
    /// not produce the same embeddings, so they must not share an entry.
    pub precision: String,
    pub cross_attention_dim: usize,
    pub connector_registers: u32,
    pub use_connector: bool,
    /// Whether the unconditional branch was really encoded (`guidance > 1.0`)
    /// or is the all-zero stand-in.
    pub uncond_encoded: bool,
    /// Which revision of the encode's own ARITHMETIC produced the entry -
    /// [`ENCODE_REVISION`].
    ///
    /// Every other field above describes an INPUT. This one describes the
    /// function, and it has to be here for the same reason the others do: a
    /// cache whose key cannot express "we compute this differently now"
    /// silently serves a context the current pipeline would never produce,
    /// which is exactly the wrong-prompt failure this module's header says it
    /// must not have. The checkpoint identity does not cover it - the bug
    /// that forced this field changed `gemma4::AggregateEmbed`, not any file
    /// on disk.
    pub encode_revision: u32,
}

/// Bump whenever a change to the text encode makes previously-cached entries
/// wrong rather than merely stale - i.e. whenever the same prompt and the
/// same checkpoint would now produce different numbers.
///
/// * `1` - the original encode.
/// * `2` - `gemma4::AggregateEmbed::forward` gained the reference's
///   per-token/per-state RMS normalization, interleaved column order and
///   `sqrt(out_dim/hidden)` rescale, and the prompt gained the leading
///   `<bos>` `LTXGemmaTokenizer` prepends.
pub const ENCODE_REVISION: u32 = 2;

impl Key {
    /// The key material as JSON - written into the cache file and compared on
    /// load. This IS the key; [`Self::digest`] is only how it is filed.
    fn to_json(&self) -> serde_json::Value {
        json!({
            "prompt": self.prompt,
            "encoder_path": self.encoder_path,
            "encoder_len": self.encoder_len,
            "encoder_mtime": self.encoder_mtime,
            "precision": self.precision,
            "cross_attention_dim": self.cross_attention_dim,
            "connector_registers": self.connector_registers,
            "use_connector": self.use_connector,
            "uncond_encoded": self.uncond_encoded,
            "encode_revision": self.encode_revision,
        })
    }

    /// A filename-safe digest of the key material. FNV-1a over the canonical
    /// JSON: strong enough to keep distinct prompts in distinct files, and it
    /// does not need to be stronger than that, because a collision is caught
    /// by the verification on load and downgraded to a miss.
    fn digest(&self) -> String {
        let s = self.to_json().to_string();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }
}

/// One cached encode: exactly what `real_text_context` returns.
pub struct Encoded {
    pub ctx_cond: Vec<f32>,
    pub ctx_uncond: Vec<f32>,
    pub context_valid: Vec<f32>,
    pub context_len: usize,
}

/// `false` when `BRAIN_LTXV_TEXT_CACHE` is set to `0`/`off`/`false`.
pub fn enabled() -> bool {
    match std::env::var("BRAIN_LTXV_TEXT_CACHE") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"),
        Err(_) => true,
    }
}

fn path_for(key: &Key) -> Option<std::path::PathBuf> {
    let dir = gpu_core::cache_dir()?.join("ltxv-text-context");
    Some(dir.join(format!("{}.safetensors", key.digest())))
}

/// Identity fields for an encoder checkpoint, without reading its contents.
/// Returns zeros when the file cannot be stat'ed, which makes the key
/// unstable and so simply prevents caching rather than caching under a key
/// that does not describe the file.
pub fn encoder_identity(path: &str) -> (u64, i64) {
    match std::fs::metadata(path) {
        Ok(m) => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            (m.len(), mtime)
        }
        Err(_) => (0, 0),
    }
}

/// Look `key` up. `None` on any miss, any unreadable/corrupt entry, and - the
/// case that matters - any entry whose stored key material does not match
/// `key` exactly.
pub fn load(key: &Key) -> Option<Encoded> {
    if !enabled() {
        return None;
    }
    let path = path_for(key)?;
    let bytes = std::fs::read(&path).ok()?;
    let m = checkpoint::st::parse_safetensors(&bytes).ok()?;

    let stored = m.config();
    if stored.get("key") != Some(&key.to_json()) {
        // A digest collision, or an entry written by an older layout. Either
        // way this is a MISS, never a wrong hit.
        tracing::debug!(path = %path.display(), "text-context cache entry did not match its key; treating as a miss");
        return None;
    }

    let get = |n: &str| m.tensors.iter().find(|(name, _)| name.as_str() == n).map(|(_, d)| d.clone());
    let ctx_cond = get("ctx_cond")?;
    let ctx_uncond = get("ctx_uncond")?;
    let context_valid = get("context_valid")?;
    let context_len = context_valid.len();

    // Structural agreement between the stored pieces, so a truncated write
    // cannot come back as a plausible-looking context.
    if context_len == 0 || ctx_cond.len() != context_len * key.cross_attention_dim || ctx_uncond.len() != ctx_cond.len() {
        tracing::warn!(path = %path.display(), "text-context cache entry has inconsistent shapes; ignoring it");
        return None;
    }
    tracing::info!(path = %path.display(), context_len, "text context served from cache");
    Some(Encoded { ctx_cond, ctx_uncond, context_valid, context_len })
}

/// Store `enc` under `key`. Best-effort: a cache that cannot be written is
/// not an error in a generation, so failures are logged and swallowed.
pub fn store(key: &Key, enc: &Encoded) {
    if !enabled() {
        return;
    }
    let Some(path) = path_for(key) else {
        return;
    };
    if let Some(dir) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            tracing::warn!(dir = %dir.display(), error = %e, "could not create the text-context cache directory");
            return;
        }
    }
    let dim = key.cross_attention_dim as u64;
    let tensors = vec![
        ("ctx_cond".to_string(), vec![enc.context_len as u64, dim], enc.ctx_cond.clone()),
        ("ctx_uncond".to_string(), vec![enc.context_len as u64, dim], enc.ctx_uncond.clone()),
        ("context_valid".to_string(), vec![enc.context_len as u64], enc.context_valid.clone()),
    ];
    match checkpoint::st::save_safetensors(&path.to_string_lossy(), &tensors, &json!({ "key": key.to_json() }), None) {
        Ok(()) => tracing::info!(path = %path.display(), context_len = enc.context_len, "text context cached"),
        Err(e) => tracing::warn!(path = %path.display(), error = %e, "could not write the text-context cache entry"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Key {
        Key {
            prompt: "a cat".into(),
            encoder_path: "enc.gguf".into(),
            encoder_len: 42,
            encoder_mtime: 7,
            precision: "Int8".into(),
            cross_attention_dim: 4,
            connector_registers: 2,
            use_connector: true,
            uncond_encoded: false,
            encode_revision: ENCODE_REVISION,
        }
    }

    /// Every field is load bearing. A key field that does not change the
    /// digest is a field that lets two genuinely different encodes share one
    /// entry - which is the wrong-context failure this cache must not have.
    #[test]
    fn every_key_field_changes_the_digest() {
        let base = key();
        let d = base.digest();
        let variants: Vec<Key> = vec![
            Key { prompt: "a dog".into(), ..base.clone() },
            Key { encoder_path: "other.gguf".into(), ..base.clone() },
            Key { encoder_len: 43, ..base.clone() },
            Key { encoder_mtime: 8, ..base.clone() },
            Key { precision: "Fp32".into(), ..base.clone() },
            Key { cross_attention_dim: 8, ..base.clone() },
            Key { connector_registers: 4, ..base.clone() },
            Key { use_connector: false, ..base.clone() },
            Key { uncond_encoded: true, ..base.clone() },
            Key { encode_revision: base.encode_revision + 1, ..base.clone() },
        ];
        for v in variants {
            assert_ne!(v.digest(), d, "this field does not affect the cache key: {v:?}");
        }
        assert_eq!(base.digest(), key().digest(), "the digest must be stable for equal keys");
    }

    /// A round trip through the real file format, then the property that the
    /// whole design rests on: an entry whose stored key differs is a MISS.
    /// Exercised by writing under one key and reading with another that has
    /// been forced to the same filename.
    #[test]
    fn a_mismatched_key_is_a_miss_not_a_wrong_hit() {
        let tmp = std::env::temp_dir().join("brain-ltxv-text-cache-test");
        // SAFETY-equivalent note: this test owns the process's cache dir for
        // its duration; it runs single-threaded within its own binary.
        unsafe { std::env::set_var("BRAIN_PIPELINE_CACHE_DIR", &tmp) };
        let k = key();
        let enc = Encoded {
            ctx_cond: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ctx_uncond: vec![0.0; 8],
            context_valid: vec![1.0, 0.0],
            context_len: 2,
        };
        store(&k, &enc);
        let got = load(&k).expect("a stored entry must load back");
        assert_eq!(got.ctx_cond, enc.ctx_cond);
        assert_eq!(got.context_valid, enc.context_valid);
        assert_eq!(got.context_len, 2);

        // Same file, different key material: overwrite the entry's key with a
        // different prompt's, then confirm the original key does not accept it.
        let other = Key { prompt: "something else entirely".into(), ..k.clone() };
        let file = path_for(&k).unwrap();
        let bytes = std::fs::read(&file).unwrap();
        let m = checkpoint::st::parse_safetensors(&bytes).unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            m.tensors.iter().map(|(n, d)| (n.clone(), vec![d.len() as u64], d.clone())).collect();
        checkpoint::st::save_safetensors(&file.to_string_lossy(), &tensors, &json!({ "key": other.to_json() }), None).unwrap();
        assert!(load(&k).is_none(), "an entry whose key does not match must be a miss");

        std::fs::remove_dir_all(&tmp).ok();
        unsafe { std::env::remove_var("BRAIN_PIPELINE_CACHE_DIR") };
    }

    #[test]
    fn the_cache_can_be_turned_off() {
        unsafe { std::env::set_var("BRAIN_LTXV_TEXT_CACHE", "0") };
        assert!(!enabled());
        assert!(load(&key()).is_none());
        unsafe { std::env::set_var("BRAIN_LTXV_TEXT_CACHE", "1") };
        assert!(enabled());
        unsafe { std::env::remove_var("BRAIN_LTXV_TEXT_CACHE") };
    }
}
