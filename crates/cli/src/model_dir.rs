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
//!
//! Two layouts are scanned, preferred-first: the `<vendor>/<repo>` store
//! layout ([`brain_modelstore::Store::scan`]) — each entry already knows its
//! own directory, so its sibling `tokenizer.json` is never shared across
//! models — and, for back-compat, the original flat single-level directory of
//! `*.safetensors`/`*.gguf` files. A flat-layout hit logs a one-time-per-process
//! warning recommending migration to the store layout.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};

use checkpoint::st::{self, ModelCard};
use residency::ResidentModel;

/// Resolve the models directory. Precedence: the `--models-dir` flag, then
/// [`brain_modelstore::default_root`] (`BRAIN_MODELS_DIR`, then
/// `$XDG_DATA_HOME/brain/models`, then `$HOME/.local/share/brain/models`).
/// `None` only when the flag and all three env vars are unset (no HOME) — the
/// scan is then simply skipped.
pub fn resolve(flag: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = flag.filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(p));
    }
    brain_modelstore::default_root()
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
/// by id (first wins, store layout before flat layout). See the module docs
/// for the layout precedence and the skip/warn policy.
pub fn discover(dir: &Path) -> Vec<Arc<dyn ResidentModel>> {
    let mut out: Vec<Arc<dyn ResidentModel>> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();

    let mut locals = brain_modelstore::Store::new(dir.to_path_buf()).scan();
    locals.sort_by(|a, b| a.weights.cmp(&b.weights)); // deterministic catalog order
    for local in &locals {
        register(&local.weights, &local.card, &local.dir, &mut seen, &mut out);
    }

    discover_flat(dir, &mut seen, &mut out);
    out
}

static FLAT_LAYOUT_WARNING: Once = Once::new();

/// The original single-level flat scan, predating the `<vendor>/<repo>` store
/// layout. Kept for back-compat; a hit warns once per process.
fn discover_flat(dir: &Path, seen: &mut BTreeSet<String>, out: &mut Vec<Arc<dyn ResidentModel>>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("brain: models dir {} not scanned ({e})", dir.display());
            return;
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

    if !singles.is_empty() || !shards.is_empty() {
        FLAT_LAYOUT_WARNING.call_once(|| {
            eprintln!(
                "brain: {} holds models directly (the flat legacy layout); \
                 migrate to <vendor>/<repo>/ (see docs/models/naming.md) -- \
                 this warning prints once per process",
                dir.display()
            );
        });
    }

    // Sorted for a deterministic catalog order.
    singles.sort();
    for path in &singles {
        register(path, &card_of(path), dir, seen, out);
    }
    for (base, mut group) in shards {
        group.sort_by_key(|(idx, _)| *idx);
        let first = &group[0].1; // lowest-index shard carries the card
        // Prefer the `<base>.safetensors.index.json` handle if it exists.
        let index = dir.join(format!("{base}.safetensors.index.json"));
        let handle = if index.exists() { index } else { first.clone() };
        register(&handle, &card_of(first), dir, seen, out);
    }
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

/// GGUF cards carry `general.architecture` verbatim as `family` (the KV's own
/// reported name, e.g. llama.cpp's `"qwen3"`) — brain's own family names
/// (`crate::resident_llm`) differ, so alias the ones brain's implementation
/// actually matches. brain's `qwen` crate targets the Qwen3 architecture
/// specifically (QK-norm + GQA); a plain `"qwen2"` GGUF is a different
/// architecture and stays unaliased rather than silently mis-served.
fn brain_family(reported: &str) -> &str {
    match reported {
        "qwen3" => "qwen",
        other => other,
    }
}

/// Construct the resident matching `card.family` (normalized via
/// [`brain_family`]), or log + `None` for an unknown / not-yet-dispatchable
/// family.
fn resident_for(weights: &str, card: &ModelCard, tokenizer: Option<&str>) -> Option<Arc<dyn ResidentModel>> {
    match brain_family(&card.family) {
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

/// [`resident_for`] over a [`brain_modelstore::LocalModel`] -- the same
/// family dispatch [`discover`] uses for the store's scan, reused by the
/// auto-fetch supplier (`crate::supply`) so there is exactly one
/// "weights file -> resident" mapping regardless of how the file was found.
pub(crate) fn resident_for_local(local: &brain_modelstore::LocalModel) -> Option<Arc<dyn ResidentModel>> {
    let card = local.card.as_ref()?;
    let weights = local.weights.to_str()?;
    let tokenizer = local.tokenizer.as_deref().and_then(|p| p.to_str());
    resident_for(weights, card, tokenizer)
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
        // The real value llama.cpp writes for Qwen3 (verified against an
        // actual downloaded Qwen3-0.6B-GGUF) -- brain's own family name is
        // "qwen"; see `brain_family`'s alias.
        put_str(&mut kv, "qwen3");
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
    fn qwen3_gguf_architecture_aliases_to_the_qwen_family() {
        assert_eq!(brain_family("qwen3"), "qwen");
        assert_eq!(brain_family("qwen"), "qwen");
        assert_eq!(brain_family("llama"), "llama"); // unaliased families pass through
    }

    #[test]
    fn gguf_qwen_with_embedded_tokenizer_registers_chat_capable() {
        // A Qwen3 GGUF (general.architecture = "qwen3", llama.cpp's own naming)
        // carrying its own gpt2 tokenizer registers with NO sibling
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

    #[test]
    fn store_layout_scans_vendor_repo_dirs_with_independent_tokenizers() {
        // Regression test for the shared-tokenizer bug: two different
        // <vendor>/<repo> dirs each own their sibling tokenizer.json, so one
        // repo's tokenizer must never leak into another's.
        let dir = tmp_dir("store-layout");
        let a = dir.join("VendorA").join("RepoX");
        std::fs::create_dir_all(&a).unwrap();
        let card_a = ModelCard::new("VendorA/RepoX", "lfm");
        st::save_safetensors(
            a.join("model.brain.safetensors").to_str().unwrap(),
            &[("w".into(), vec![1u64], vec![0.0f32])],
            &serde_json::json!({}),
            Some(&card_a),
        )
        .unwrap();
        std::fs::write(a.join("tokenizer.json"), br#"{"model":{"vocab":{"a":0}}}"#).unwrap();

        // A second repo with NO tokenizer of its own: it must NOT register (lfm
        // requires one), proving VendorA's tokenizer wasn't picked up for it.
        let b = dir.join("VendorB").join("RepoY");
        std::fs::create_dir_all(&b).unwrap();
        let card_b = ModelCard::new("VendorB/RepoY", "lfm");
        st::save_safetensors(
            b.join("model.brain.safetensors").to_str().unwrap(),
            &[("w".into(), vec![1u64], vec![0.0f32])],
            &serde_json::json!({}),
            Some(&card_b),
        )
        .unwrap();

        let got = ids(&discover(&dir));
        assert!(got.contains(&"VendorA/RepoX".to_string()), "A missing: {got:?}");
        assert!(
            !got.contains(&"VendorB/RepoY".to_string()),
            "B registered without its own tokenizer -- tokenizer leaked across repos: {got:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
