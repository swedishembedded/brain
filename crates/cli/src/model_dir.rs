// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The global model directory scan — brain's catalog source of truth.
//!
//! [`resolve`] picks the directory (`--models-dir` / `BRAIN_MODELS_DIR`, else
//! `$XDG_DATA_HOME/brain/models`, else `$HOME/.local/share/brain/models` — no
//! absolute-path literal, always computed from env). [`discover`] scans it and
//! turns every servable weight file into its OWN [`ResidentModel`], keyed by its
//! model-card id, so a base model and a finetune/LoRA sitting side by side are
//! two distinct selectable models.
//!
//! Non-fatal throughout: an unreadable file, a cardless safetensors, an unknown
//! `card.family`, or a missing directory is logged and skipped — the scan never
//! fails serving. HF-sharded safetensors register once (the index / first shard
//! carries the card; the `-00001-of-*` shards are grouped, not re-registered).
//! GGUF embeds its tokenizer in KV; a safetensors model reuses a sibling
//! `tokenizer.json` (HF convention) and registers even without one.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::st::{self, ModelCard};
use residency::ResidentModel;

/// Resolve the models directory. Precedence: the `--models-dir` flag, then
/// `BRAIN_MODELS_DIR`, then `$XDG_DATA_HOME/brain/models`, then
/// `$HOME/.local/share/brain/models`. `None` only when the flag and all three
/// env vars are unset (no HOME) — the scan is then simply skipped.
pub fn resolve(flag: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = flag.filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("BRAIN_MODELS_DIR").filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(p));
    }
    if let Some(x) = std::env::var_os("XDG_DATA_HOME").filter(|s| !s.is_empty()) {
        return Some(Path::new(&x).join("brain").join("models"));
    }
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(|h| Path::new(&h).join(".local").join("share").join("brain").join("models"))
}

/// If `fname` is an HF shard (`<base>-<NNNNN>-of-<MMMMM>.safetensors`), return
/// `(base, index)`; else `None` (a plain `.safetensors` is its own model).
fn shard_of(fname: &str) -> Option<(String, u32)> {
    let stem = fname.strip_suffix(".safetensors")?;
    let (left, total) = stem.rsplit_once("-of-")?;
    if total.is_empty() || !total.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (base, idx) = left.rsplit_once('-')?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((base.to_string(), idx.parse().ok()?))
}

/// Scan `dir` and build one resident per discovered model-card id. Deduplicates
/// by id (first wins). See the module docs for the skip/warn policy.
pub fn discover(dir: &Path) -> Vec<Arc<dyn ResidentModel>> {
    let mut out: Vec<Arc<dyn ResidentModel>> = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("brain: models dir {} not scanned ({e})", dir.display());
            return out;
        }
    };

    // Enumerate weight files. Plain `*.safetensors` and `*.gguf` are single
    // models; sharded `*-NNNNN-of-*.safetensors` group by base and register once
    // (the lowest-index shard carries the card, the `.index.json` handle
    // preferred when present). Everything else (incl. bare `.index.json`,
    // `tokenizer.json`) is ignored as a model file.
    let mut singles: Vec<PathBuf> = Vec::new();
    let mut shards: BTreeMap<String, Vec<(u32, PathBuf)>> = BTreeMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let fname = match path.file_name().and_then(|s| s.to_str()) {
            Some(f) => f.to_string(),
            None => continue,
        };
        if fname.ends_with(".safetensors") {
            match shard_of(&fname) {
                Some((base, idx)) => shards.entry(base).or_default().push((idx, path)),
                None => singles.push(path),
            }
        } else if fname.ends_with(".gguf") {
            singles.push(path);
        }
    }

    let mut seen: BTreeSet<String> = BTreeSet::new();
    // Sorted for a deterministic catalog order.
    singles.sort();
    for path in &singles {
        register(path, &card_of(path), dir, &mut seen, &mut out);
    }
    for (base, mut group) in shards {
        group.sort_by_key(|(idx, _)| *idx);
        let first = &group[0].1; // lowest-index shard carries the card
        // Prefer the `<base>.safetensors.index.json` handle if it exists.
        let index = dir.join(format!("{base}.safetensors.index.json"));
        let handle = if index.exists() { index } else { first.clone() };
        register(&handle, &card_of(first), dir, &mut seen, &mut out);
    }
    out
}

/// Read a file's card: `read_card` (metadata-only) for safetensors, the KV-
/// synthesized card for GGUF (always `Some`). `None` on I/O error or a cardless
/// safetensors.
fn card_of(path: &Path) -> Option<ModelCard> {
    let p = path.to_string_lossy();
    if p.ends_with(".gguf") {
        match checkpoint::weightio::WeightReader::open(&p) {
            Ok(r) => r.card(),
            Err(e) => {
                eprintln!("brain: skip {p} ({e})");
                None
            }
        }
    } else {
        match st::read_card(&p) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("brain: skip {p} ({e})");
                None
            }
        }
    }
}

/// Dispatch one carded file to its family's resident and push it (deduped by id).
fn register(weights: &Path, card: &Option<ModelCard>, dir: &Path, seen: &mut BTreeSet<String>, out: &mut Vec<Arc<dyn ResidentModel>>) {
    let wp = weights.to_string_lossy();
    let card = match card {
        Some(c) => c,
        None => {
            eprintln!("brain: skip {wp} (no model card)");
            return;
        }
    };
    if !seen.insert(card.id.clone()) {
        eprintln!("brain: skip {wp} (duplicate model id '{}')", card.id);
        return;
    }
    // GGUF carries its tokenizer in KV; a safetensors model reuses a sibling
    // tokenizer.json (HF convention). Absent it, register anyway (limited chat).
    let sibling = dir.join("tokenizer.json");
    let tokenizer: Option<String> = if wp.ends_with(".gguf") {
        None
    } else if sibling.exists() {
        Some(sibling.to_string_lossy().into_owned())
    } else {
        eprintln!("brain: {wp}: no sibling tokenizer.json ({} chat-templating may be limited)", card.id);
        None
    };

    match resident_for(&wp, card, tokenizer.as_deref()) {
        Some(r) => out.push(r),
        None => {
            seen.remove(&card.id); // free the id if the family declined
        }
    }
}

/// Construct the resident matching `card.family`, or log + `None` for an unknown
/// / not-yet-dispatchable family.
fn resident_for(weights: &str, card: &ModelCard, tokenizer: Option<&str>) -> Option<Arc<dyn ResidentModel>> {
    match card.family.as_str() {
        "gpt" => Some(Arc::new(crate::resident_llm::GptResident::from_card(weights, card, tokenizer))),
        "glm" => Some(Arc::new(crate::resident_llm::GlmResident::from_card(weights, card, tokenizer))),
        "qwen" => Some(Arc::new(crate::resident_llm::QwenResident::from_card(weights, card, tokenizer))),
        "lfm" => match crate::resident_lfm::LfmResident::from_card(weights, card, tokenizer) {
            Ok(l) => Some(Arc::new(l)),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        other => {
            eprintln!("brain: skip {} (family '{other}' not servable from the model dir yet)", card.id);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("brain-modeldir-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A tiny carded safetensors: one 1-element tensor + a `ModelCard`.
    fn write_st(dir: &Path, file: &str, id: &str, family: &str) {
        let card = ModelCard::new(id, family);
        let p = dir.join(file);
        st::save_safetensors(p.to_str().unwrap(), &[("w".into(), vec![1u64], vec![0.0f32])], &serde_json::json!({}), Some(&card)).unwrap();
    }

    fn write_tokenizer(dir: &Path) {
        std::fs::write(dir.join("tokenizer.json"), br#"{"model":{"vocab":{"a":0}}}"#).unwrap();
    }

    fn put_str(v: &mut Vec<u8>, s: &str) {
        v.extend((s.len() as u64).to_le_bytes());
        v.extend(s.as_bytes());
    }

    /// A minimal GGUF (one f32 tensor) whose KV synthesizes a card: family =
    /// `general.architecture`, id = `general.name`.
    fn write_gguf(dir: &Path, file: &str, arch: &str, name: &str) {
        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes()); // version
        h.extend(1u64.to_le_bytes()); // tensor count
        h.extend(2u64.to_le_bytes()); // kv count
        put_str(&mut h, "general.architecture");
        h.extend(8u32.to_le_bytes()); // value type: string
        put_str(&mut h, arch);
        put_str(&mut h, "general.name");
        h.extend(8u32.to_le_bytes());
        put_str(&mut h, name);
        // tensor info: "w", 1 dim [4], type F32, offset 0
        put_str(&mut h, "w");
        h.extend(1u32.to_le_bytes());
        h.extend(4u64.to_le_bytes());
        h.extend(0u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            h.extend(v.to_le_bytes());
        }
        std::fs::write(dir.join(file), &h).unwrap();
    }

    /// A minimal GGUF whose KV synthesizes a `qwen` card AND embeds a gpt2
    /// byte-level BPE tokenizer (`tokenizer.ggml.*`) — the shape a real Qwen3
    /// GGUF ships, minus the weights. Enough for the discovery/registration path
    /// to build a chat-capable Qwen entry with no sibling tokenizer.json.
    fn write_gguf_qwen(dir: &Path, file: &str, name: &str) {
        fn put_str_arr(v: &mut Vec<u8>, key: &str, items: &[&str]) {
            put_str(v, key);
            v.extend(9u32.to_le_bytes()); // array
            v.extend(8u32.to_le_bytes()); // element type: string
            v.extend((items.len() as u64).to_le_bytes());
            for s in items {
                put_str(v, s);
            }
        }
        fn put_i32_arr(v: &mut Vec<u8>, key: &str, items: &[i32]) {
            put_str(v, key);
            v.extend(9u32.to_le_bytes()); // array
            v.extend(5u32.to_le_bytes()); // element type: int32
            v.extend((items.len() as u64).to_le_bytes());
            for x in items {
                v.extend(x.to_le_bytes());
            }
        }
        fn put_u32_kv(v: &mut Vec<u8>, key: &str, x: u32) {
            put_str(v, key);
            v.extend(4u32.to_le_bytes());
            v.extend(x.to_le_bytes());
        }

        let mut kv: Vec<u8> = Vec::new();
        put_str(&mut kv, "general.architecture");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "qwen");
        put_str(&mut kv, "general.name");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, name);
        put_str(&mut kv, "tokenizer.ggml.model");
        kv.extend(8u32.to_le_bytes());
        put_str(&mut kv, "gpt2");
        put_str_arr(&mut kv, "tokenizer.ggml.tokens", &["<|endoftext|>", "<|im_start|>", "<|im_end|>", "h", "i", "hi"]);
        put_str_arr(&mut kv, "tokenizer.ggml.merges", &["h i"]);
        put_i32_arr(&mut kv, "tokenizer.ggml.token_type", &[3, 3, 3, 1, 1, 1]);
        put_u32_kv(&mut kv, "tokenizer.ggml.bos_token_id", 0);
        put_u32_kv(&mut kv, "tokenizer.ggml.eos_token_id", 2);

        let mut h: Vec<u8> = Vec::new();
        h.extend(b"GGUF");
        h.extend(3u32.to_le_bytes()); // version
        h.extend(1u64.to_le_bytes()); // tensor count
        h.extend(8u64.to_le_bytes()); // kv count
        h.extend(&kv);
        // tensor info: "w", 1 dim [4], type F32, offset 0
        put_str(&mut h, "w");
        h.extend(1u32.to_le_bytes());
        h.extend(4u64.to_le_bytes());
        h.extend(0u32.to_le_bytes());
        h.extend(0u64.to_le_bytes());
        let data_start = h.len().div_ceil(32) * 32;
        h.resize(data_start, 0);
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            h.extend(v.to_le_bytes());
        }
        std::fs::write(dir.join(file), &h).unwrap();
    }

    fn ids(residents: &[Arc<dyn ResidentModel>]) -> Vec<String> {
        residents.iter().map(|r| r.manifest().model).collect()
    }

    #[test]
    fn discovers_each_file_as_a_distinct_catalog_entry() {
        let dir = tmp_dir("variants");
        // Two checkpoints of ONE family → two distinct selectable models.
        write_st(&dir, "base.safetensors", "toy-base", "gpt");
        write_st(&dir, "ft.safetensors", "toy-ft", "gpt");
        // Unknown family → skipped (not fatal).
        write_st(&dir, "mystery.safetensors", "toy-unknown", "mystery");
        // A sibling tokenizer.json lets the encoder register (it embeds no vocab).
        write_tokenizer(&dir);
        write_st(&dir, "enc.safetensors", "toy-enc", "lfm");
        // GGUF card is synthesized from KV and dispatched by family.
        write_gguf(&dir, "toy.gguf", "gpt", "toy-gguf");

        let got = ids(&discover(&dir));
        assert!(got.contains(&"toy-base".to_string()), "base missing: {got:?}");
        assert!(got.contains(&"toy-ft".to_string()), "ft missing: {got:?}");
        assert!(!got.contains(&"toy-unknown".to_string()), "unknown family not skipped: {got:?}");
        assert!(got.contains(&"toy-enc".to_string()), "sibling tokenizer.json not picked up (lfm missing): {got:?}");
        assert!(got.contains(&"toy-gguf".to_string()), "gguf card not discovered: {got:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lfm_skipped_without_sibling_tokenizer() {
        // No tokenizer.json → the encoder cannot construct → skipped (not fatal).
        let dir = tmp_dir("notok");
        write_st(&dir, "enc.safetensors", "toy-enc", "lfm");
        let got = ids(&discover(&dir));
        assert!(!got.contains(&"toy-enc".to_string()), "lfm registered without a tokenizer: {got:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gguf_qwen_with_embedded_tokenizer_registers_chat_capable() {
        // A Qwen GGUF carrying its own gpt2 tokenizer registers with NO sibling
        // tokenizer.json, and advertises the streaming chat `generate` action.
        let dir = tmp_dir("ggufqwen");
        write_gguf_qwen(&dir, "qwen3.gguf", "toy-qwen-gguf");
        let residents = discover(&dir);
        let got = ids(&residents);
        assert!(got.contains(&"toy-qwen-gguf".to_string()), "qwen gguf not registered: {got:?}");

        // The registered entry is chat-capable: its `generate` action streams and
        // takes chat `messages` (generate_spec(chat=true)).
        let r = residents.iter().find(|r| r.manifest().model == "toy-qwen-gguf").unwrap();
        let m = r.manifest();
        let gen = m.actions.iter().find(|a| a.name == "generate").expect("generate action");
        assert!(gen.streaming, "qwen generate must stream");
        assert!(gen.params.iter().any(|p| p.name == "messages"), "qwen generate must accept chat messages");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_prefers_the_flag() {
        // The explicit flag wins over env/default; empty flag falls through.
        assert!(resolve(Some("flagdir")).unwrap().ends_with("flagdir"));
    }
}
