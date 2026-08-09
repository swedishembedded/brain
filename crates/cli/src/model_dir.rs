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
//!
//! [`resident_for`] dispatches by `card.family`: `qwen`, `gpt`, `glm`, `lfm`,
//! `yolo`, `depth` today. Each family's own checkpoint-writing code must
//! attach a real [`ModelCard`] (`checkpoint::save_carded`, not the plain
//! `checkpoint::save`, which never carries one) for a checkpoint to be
//! reachable here at all — a `from_card` resident constructor alone is not
//! enough (`gpt`/`glm` had working dispatch arms for a while with no carded
//! checkpoint ever able to reach them). Single-file models only: `z-image`
//! and `flux2` are each FOUR distinct-role files (DiT/VAE/text-encoder/
//! tokenizer) with no directory/manifest registration shape here yet.

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
        register(&local.weights, &local.card, &local.dir, local.adapter.as_deref(), &mut seen, &mut out);
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
        register(path, &card_of(path), dir, None, seen, out);
    }
    for (base, mut group) in shards {
        group.sort_by_key(|(idx, _)| *idx);
        let first = &group[0].1; // lowest-index shard carries the card
        // Prefer the `<base>.safetensors.index.json` handle if it exists.
        let index = dir.join(format!("{base}.safetensors.index.json"));
        let handle = if index.exists() { index } else { first.clone() };
        register(&handle, &card_of(first), dir, None, seen, out);
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

/// Dispatch one carded file to its family's resident and push it (deduped by
/// id). `adapter` is the adapter's own weight file when `card.id` names one
/// (`brain_modelstore::LocalModel::adapter`) -- `None` for a plain base/quant.
fn register(weights: &Path, card: &Option<ModelCard>, dir: &Path, adapter: Option<&Path>, seen: &mut BTreeSet<String>, out: &mut Vec<Arc<dyn ResidentModel>>) {
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

    let adapter = adapter.map(|p| p.to_string_lossy().into_owned());
    match resident_for(&wp, card, tokenizer.as_deref(), adapter.as_deref()) {
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
/// family. `adapter` (a named LoRA adapter's own weight file) is meaningful
/// only to the `qwen` family today -- other families ignore it.
fn resident_for(weights: &str, card: &ModelCard, tokenizer: Option<&str>, adapter: Option<&str>) -> Option<Arc<dyn ResidentModel>> {
    match brain_family(&card.family) {
        "gpt" => Some(Arc::new(crate::resident_llm::GptResident::from_card(weights, card, tokenizer))),
        "glm" => Some(Arc::new(crate::resident_llm::GlmResident::from_card(weights, card, tokenizer))),
        "qwen" => Some(Arc::new(crate::resident_llm::QwenResident::from_card(weights, card, tokenizer, adapter))),
        "lfm" => match crate::resident_lfm::LfmResident::from_card(weights, card, tokenizer) {
            Ok(l) => Some(Arc::new(l)),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        "yolo" => Some(Arc::new(crate::resident::YoloResident::from_card(weights, card, tokenizer))),
        "depth" => Some(Arc::new(crate::resident_depth::DepthResident::from_card(weights, card, tokenizer))),
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
/// A compound (multi-file, `local.roles.is_some()`) model has no single
/// weights path, so it dispatches separately, by family, before the
/// single-path case below even applies.
pub(crate) fn resident_for_local(local: &brain_modelstore::LocalModel) -> Option<Arc<dyn ResidentModel>> {
    let card = local.card.as_ref()?;
    if let Some(roles) = &local.roles {
        return resident_for_compound(card, roles);
    }
    let weights = local.weights.to_str()?;
    let tokenizer = local.tokenizer.as_deref().and_then(|p| p.to_str());
    let adapter = local.adapter.as_deref().and_then(|p| p.to_str());
    resident_for(weights, card, tokenizer, adapter)
}

/// The compound-model counterpart of [`resident_for`]: family dispatch keyed
/// on named roles (a directory or file each) rather than one weights path.
fn resident_for_compound(card: &ModelCard, roles: &std::collections::BTreeMap<String, PathBuf>) -> Option<Arc<dyn ResidentModel>> {
    match brain_family(&card.family) {
        "zimage" => match zimage_paths_from_roles(roles) {
            Ok(paths) => match crate::resident::ZImageResident::from_paths(card.id.clone(), paths) {
                Ok(z) => Some(Arc::new(z)),
                Err(e) => {
                    eprintln!("brain: skip {} ({e})", card.id);
                    None
                }
            },
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        other => {
            eprintln!("brain: skip {} (compound family '{other}' not servable from the model dir yet)", card.id);
            None
        }
    }
}

/// Build a `zimage::pipeline::Paths` from a compound manifest's roles --
/// reads the SAME role names `brain_modelstore::recipe::ZimageRecipe::ROLES`
/// writes, from a `brain.manifest.json` this crate's `supply::convert`
/// (`"zimage"` arm) produced.
fn zimage_paths_from_roles(roles: &std::collections::BTreeMap<String, PathBuf>) -> Result<zimage::pipeline::Paths, String> {
    let get = |role: &str| roles.get(role).and_then(|p| p.to_str()).map(str::to_string).ok_or_else(|| format!("compound manifest missing role {role:?}"));
    Ok(zimage::pipeline::Paths { dit: get("dit")?, vae: get("vae")?, qwen: get("text_encoder")?, tokenizer: get("tokenizer")? })
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

    /// gpt was already proven by `discovers_each_file_as_a_distinct_catalog_entry`
    /// above (its `write_st(..., "gpt")` cases) -- this covers the three
    /// families that joined `resident_for`'s dispatch table alongside gpt's
    /// checkpoint-save fix (glm, yolo, depth): each constructs with nothing
    /// but a path + card (no real weight loading happens until `activate()`),
    /// so a bare `write_st` is enough to prove discovery end to end, exactly
    /// like the "unknown family" contrast case above proves the negative.
    #[test]
    fn discovers_glm_yolo_and_depth_checkpoints() {
        let dir = tmp_dir("glm-yolo-depth");
        write_st(&dir, "glm.safetensors", "toy-glm", "glm");
        write_st(&dir, "yolo.safetensors", "toy-yolo", "yolo");
        write_st(&dir, "depth.safetensors", "toy-depth", "depth");

        let residents = discover(&dir);
        let got = ids(&residents);
        assert!(got.contains(&"toy-glm".to_string()), "glm missing: {got:?}");
        assert!(got.contains(&"toy-yolo".to_string()), "yolo missing: {got:?}");
        assert!(got.contains(&"toy-depth".to_string()), "depth missing: {got:?}");
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

    /// A base Qwen model and a named LoRA adapter trained on top of it must
    /// register as TWO distinct, independently selectable catalog entries
    /// (mirrors `brain_modelstore::Store`'s own
    /// `scan_finds_a_base_model_and_its_adapter_as_two_distinct_entries`,
    /// but as the CLI-facing gate: this is what `brain caps`/`brain do`
    /// actually walk).
    #[test]
    fn discover_surfaces_a_base_and_its_named_adapter_as_two_distinct_entries() {
        let dir = tmp_dir("adapter-catalog");
        let repo = dir.join("Qwen").join("Qwen3-Toy");
        std::fs::create_dir_all(&repo).unwrap();
        let base_card = ModelCard::new("Qwen/Qwen3-Toy", "qwen");
        st::save_safetensors(
            repo.join("model.brain.safetensors").to_str().unwrap(),
            &[("w".into(), vec![1u64], vec![0.0f32])],
            &serde_json::json!({}),
            Some(&base_card),
        )
        .unwrap();
        write_tokenizer(&repo);

        let adapter_dir = repo.join("adapters").join("swedishembedded-com").join("generic-sft").join("latest");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        let mut adapter_card = ModelCard::new("Qwen/Qwen3-Toy:swedishembedded-com:generic-sft:latest", "qwen");
        adapter_card.variant_of = Some("Qwen/Qwen3-Toy".to_string());
        adapter_card.adapter = Some(st::Adapter { kind: "lora".into(), rank: Some(4), base: Some("Qwen/Qwen3-Toy".to_string()), alpha: Some(8.0), targets: Some(vec!["wq".into()]), dataset_id: None });
        st::save_safetensors(
            adapter_dir.join("adapter.brain.safetensors").to_str().unwrap(),
            &[("blocks.0.attn.wq.weight.lora_a".into(), vec![1u64], vec![0.1f32])],
            &serde_json::json!({"rank": 4, "alpha": 8.0}),
            Some(&adapter_card),
        )
        .unwrap();

        let residents = discover(&dir);
        let got = ids(&residents);
        assert!(got.contains(&"Qwen/Qwen3-Toy".to_string()), "base missing: {got:?}");
        assert!(got.contains(&"Qwen/Qwen3-Toy:swedishembedded-com:generic-sft:latest".to_string()), "adapter missing: {got:?}");
        assert_eq!(got.len(), 2, "expected exactly base + adapter, got: {got:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
