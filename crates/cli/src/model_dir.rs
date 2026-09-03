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
//! `yolo`, `depth`, `qwen35` today. A raw `.gguf` reaches that dispatch at all
//! only for the families whose resident reads GGUF itself
//! ([`family_reads_gguf`]) - every other `.gguf` is answered BEFORE the match
//! with [`gguf_advice`], because a GGUF card's family is its
//! `general.architecture` verbatim and so may spell any family's name. A
//! `.gguf` whose architecture has a registered importer
//! ([`crate::gguf_import`]) but is not itself servable is reported with the
//! one-time `brain import-gguf` command that makes it servable - see that
//! module's doc for why the scan does not convert it in place.
//!
//! Each family's own checkpoint-writing code must
//! attach a real [`ModelCard`] (`checkpoint::save_carded`, not the plain
//! `checkpoint::save`, which never carries one) for a checkpoint to be
//! reachable here at all — a `from_card` resident constructor alone is not
//! enough (`gpt`/`glm` had working dispatch arms for a while with no carded
//! checkpoint ever able to reach them).
//!
//! A model that is FOUR distinct-role files (DiT/VAE/text-encoder/tokenizer)
//! has no single weights path, so it registers through
//! [`resident_for_compound`] instead, keyed on the roles a
//! `brain.manifest.json` names: `zimage` and `wan` today. `flux2` is the same
//! shape and still has no arm there -- it is reachable only through its own
//! `BRAIN_FLUX2_*` variables.

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
    // models; a sharded `*-NNNNN-of-MMMMM.safetensors` or split
    // `*-NNNNN-of-MMMMM.gguf` set groups by base and registers once (the
    // lowest-index part carries the card - for safetensors, the
    // `.index.json` handle is preferred when present; for GGUF, part 1's
    // path is what `MmapGguf::open` needs to find every sibling). Everything
    // else (incl. bare `.index.json`, `tokenizer.json`) is ignored as a
    // model file. `checkpoint::split::split_name` is the one parser for
    // both extensions' identical `-NNNNN-of-MMMMM` convention.
    let mut singles: Vec<PathBuf> = Vec::new();
    let mut shards: BTreeMap<String, Vec<(u32, PathBuf)>> = BTreeMap::new();
    let mut gguf_splits: BTreeMap<String, Vec<(u32, PathBuf)>> = BTreeMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let fname = match path.file_name().and_then(|s| s.to_str()) {
            Some(f) => f.to_string(),
            None => continue,
        };
        if fname.ends_with(".safetensors") {
            match checkpoint::split::split_name(&fname, "safetensors") {
                Some((base, part, _count, _width)) => shards.entry(base.to_string()).or_default().push((part, path)),
                None => singles.push(path),
            }
        } else if fname.ends_with(".gguf") {
            match checkpoint::split::split_name(&fname, "gguf") {
                Some((base, part, _count, _width)) => gguf_splits.entry(base.to_string()).or_default().push((part, path)),
                None => singles.push(path),
            }
        }
    }

    if !singles.is_empty() || !shards.is_empty() || !gguf_splits.is_empty() {
        FLAT_LAYOUT_WARNING.call_once(|| {
            eprintln!(
                "brain: {} holds models directly (the flat legacy layout); \
                 migrate to <vendor>/<repo>/ -- \
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
    for (_base, mut group) in gguf_splits {
        group.sort_by_key(|(part, _)| *part);
        let part1 = &group[0].1; // MmapGguf::open finds every sibling from part 1's path
        register(part1, &card_of(part1), dir, None, seen, out);
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

/// Whether a family's resident can open a raw `.gguf` weights path ITSELF.
///
/// The opt-in half of [`resident_for`]'s GGUF gate, and deliberately an
/// allowlist rather than a list of exclusions: a GGUF card's `family` is
/// `general.architecture` verbatim (see [`card_of`]), so ANY architecture
/// string is free to collide with ANY family name here, and the safe answer
/// for a family nobody has checked is "no" - the caller then prints the
/// `brain import-gguf` advice instead of handing GGUF bytes to a resident
/// that cannot parse them.
///
/// The three listed all reach their weights through
/// `checkpoint::weightio::WeightReader`, which sniffs the container and reads
/// safetensors and GGUF alike; `qwen` additionally has a documented `.gguf`
/// decode path and an embedded-tokenizer fallback
/// (`crate::resident_llm::QwenResident::activate`). Every other family loads
/// through `checkpoint::load` / `checkpoint::torchpt`, which are
/// safetensors-only: `qwen35`/`qwen35moe`'s serving `Engine`, yolo's
/// `yolov8::Yolo::load`, depth's `zipdepth::import::load`.
fn family_reads_gguf(family: &str) -> bool {
    matches!(family, "gpt" | "glm" | "qwen")
}

/// What to do with a raw `.gguf` no family resident can open - the one line
/// [`resident_for`] logs before declining it, keyed on the file's own
/// `general.architecture`.
///
/// A `.gguf` whose architecture IS in the importer table is not a dead end -
/// name what to do with it instead of reporting it as unservable. Which advice
/// is right comes from the table's own direct-load column: an architecture
/// that streams its GGUF at inference has no conversion to run, and telling
/// someone to convert one would send them to a command that refuses. See
/// `crate::gguf_import`'s module doc for why discovery does not run a
/// conversion itself.
fn gguf_advice(weights: &str, architecture: &str) -> String {
    match crate::gguf_import::importer_for(architecture) {
        Some(entry) if entry.loads_directly() => {
            format!("{weights}: GGUF architecture '{architecture}' loads directly through its own verb, not the model-dir scan")
        }
        Some(_) => format!("{weights}: GGUF architecture '{architecture}' needs a one-time conversion -- run `brain import-gguf {weights}`"),
        None => format!("family '{architecture}' not servable from the model dir yet"),
    }
}

/// Construct the resident matching `card.family` (normalized via
/// [`brain_family`]), or log + `None` for an unknown / not-yet-dispatchable
/// family. `adapter` (a named LoRA adapter's own weight file) is meaningful
/// only to the `qwen` family today -- other families ignore it.
fn resident_for(weights: &str, card: &ModelCard, tokenizer: Option<&str>, adapter: Option<&str>) -> Option<Arc<dyn ResidentModel>> {
    let family = brain_family(&card.family);
    // The GGUF gate, BEFORE the family match: a GGUF card's family is its
    // `general.architecture` copied verbatim, so an architecture string is
    // free to spell a brain-native family name and fall into that family's arm
    // below - handing a resident that only reads safetensors a file it cannot
    // parse, which surfaces as a low-level error at activate() instead of the
    // actionable import advice. Gating here rather than per-arm is what makes
    // the whole collision class impossible: an arm is reached by a `.gguf`
    // only if its family opted in via `family_reads_gguf`.
    if weights.ends_with(".gguf") && !family_reads_gguf(family) {
        eprintln!("brain: skip {} ({})", card.id, gguf_advice(weights, &card.family));
        return None;
    }
    match family {
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
        // What `crate::gguf_import`'s registry-driven conversion stamps on the
        // checkpoint it writes - so an imported GGUF is picked up by the very
        // next scan with no env vars and no per-model wiring.
        "qwen35moe" => match crate::resident_qwen35moe::Qwen35Resident::from_card(weights, card, tokenizer) {
            Ok(q) => Some(Arc::new(q)),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        // What `Qwen35::save` (`crates/qwen35`, the dense sibling) stamps on
        // its own checkpoints - a distinct family from "qwen35moe" above
        // (a pre-existing collision between the two crates, fixed in the
        // same commit that added this arm).
        "qwen35" => match crate::resident_qwen35::Qwen35Resident::from_card(weights, card, tokenizer) {
            Ok(q) => Some(Arc::new(q)),
            Err(e) => {
                eprintln!("brain: skip {} ({e})", card.id);
                None
            }
        },
        // Only a brain-native (safetensors) checkpoint reaches here: the gate
        // above already answered every `.gguf` whose family has no arm.
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
        "wan" => match wan_paths_from_roles(roles) {
            // Unlike zimage's, this construction cannot fail: the four roles
            // ARE the model, and the weights are read lazily at activate().
            Ok(paths) => Some(Arc::new(crate::resident_wan::WanResident::from_paths(card.id.clone(), paths))),
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

/// Build a `s3dit::pipeline::Paths` from a compound manifest's roles --
/// reads the SAME role names `brain_modelstore::recipe::ZimageRecipe::ROLES`
/// writes, from a `brain.manifest.json` this crate's `supply::convert`
/// (`"zimage"` arm) produced.
fn zimage_paths_from_roles(roles: &std::collections::BTreeMap<String, PathBuf>) -> Result<s3dit::pipeline::Paths, String> {
    let get = |role: &str| roles.get(role).and_then(|p| p.to_str()).map(str::to_string).ok_or_else(|| format!("compound manifest missing role {role:?}"));
    Ok(s3dit::pipeline::Paths { dit: get("dit")?, vae: get("vae")?, qwen: get("text_encoder")?, tokenizer: get("tokenizer")? })
}

/// Build a `wan::Paths` from a compound manifest's roles -- the SAME role
/// names `brain_modelstore::recipe::WanRecipe::ROLES` writes, from a
/// `brain.manifest.json` this crate's `supply::convert_wan` produced. The
/// `text_encoder` role is umT5-XXL rather than z-image's Qwen, but the role
/// NAMES are deliberately the same four as zimage's, so the two compound
/// families read alike.
fn wan_paths_from_roles(roles: &std::collections::BTreeMap<String, PathBuf>) -> Result<wan::Paths, String> {
    let get = |role: &str| roles.get(role).and_then(|p| p.to_str()).map(str::to_string).ok_or_else(|| format!("compound manifest missing role {role:?}"));
    Ok(wan::Paths { dit: get("dit")?, vae: get("vae")?, t5: get("text_encoder")?, tokenizer: get("tokenizer")? })
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

    /// A 3-part split GGUF carrying the same "qwen" card shape
    /// `write_gguf_qwen` writes as one file - so this test proves the split
    /// path registers a real, chat-capable resident, not just that
    /// `discover_flat`'s grouping runs without panicking. Each part gets its
    /// own distinctly-named tensor (`MmapGguf::open`'s cross-part merge
    /// refuses a repeated name), and only part 1 carries the KV a card is
    /// synthesized from - split.no/split.count/split.tensors.count come
    /// from `write_split` itself.
    fn write_gguf_qwen_split(dir: &Path, base: &str, name: &str) -> String {
        use checkpoint::gguf::GgufValue as V;
        let kv = vec![
            ("general.architecture".to_string(), V::String("qwen3".to_string())),
            ("general.name".to_string(), V::String(name.to_string())),
            ("tokenizer.ggml.model".to_string(), V::String("gpt2".to_string())),
            (
                "tokenizer.ggml.tokens".to_string(),
                V::Array(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "h", "i", "hi"].into_iter().map(|s| V::String(s.to_string())).collect()),
            ),
            ("tokenizer.ggml.merges".to_string(), V::Array(vec![V::String("h i".to_string())])),
            ("tokenizer.ggml.token_type".to_string(), V::Array([3, 3, 3, 1, 1, 1].into_iter().map(V::I32).collect())),
            ("tokenizer.ggml.bos_token_id".to_string(), V::U32(0)),
            ("tokenizer.ggml.eos_token_id".to_string(), V::U32(2)),
        ];
        let tensor = |n: &str| checkpoint::gguf_write::TensorOut {
            name: n.to_string(),
            shape: vec![4],
            ty: checkpoint::gguf::GgmlType::F32.id(),
            data: [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let parts = vec![vec![tensor("w0")], vec![tensor("w1")], vec![tensor("w2")]];
        checkpoint::gguf_write::write_split(dir.to_str().unwrap(), base, &kv, &parts, 32).unwrap()
    }

    /// The split path must register exactly ONE resident from a 3-part
    /// split, chat-capable exactly like `gguf_qwen_with_embedded_tokenizer_
    /// registers_chat_capable`'s single-file case.
    #[test]
    fn a_split_gguf_registers_once_as_a_chat_capable_qwen_resident() {
        let dir = tmp_dir("ggufqwensplit");
        write_gguf_qwen_split(&dir, "qwen3-split", "toy-qwen-gguf-split");

        let residents = discover(&dir);
        let matching: Vec<&Arc<dyn ResidentModel>> = residents.iter().filter(|r| r.manifest().model == "toy-qwen-gguf-split").collect();
        assert_eq!(matching.len(), 1, "the 3-part split must register exactly once, not once per part");

        let m = matching[0].manifest();
        let gen = m.actions.iter().find(|a| a.name == "generate").expect("generate action");
        assert!(gen.streaming, "qwen generate must stream");
        assert!(gen.params.iter().any(|p| p.name == "messages"), "qwen generate must accept chat messages");
        std::fs::remove_dir_all(&dir).ok();
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

    /// A raw `.gguf` must never be handed to a brain-native family resident
    /// just because its `general.architecture` happens to spell that family's
    /// name. `card_of` copies the architecture into `card.family` VERBATIM, so
    /// a `qwen35moe`/`qwen35` GGUF collides with the two arms serving those
    /// families' brain-native checkpoints -- residents whose `Engine` reads
    /// `checkpoint::load` (safetensors ONLY) and cannot open GGUF bytes at
    /// all. The collision must resolve to the `brain import-gguf` guidance,
    /// not to a resident that dies with a low-level parse error at activate().
    /// A GGUF architecture whose resident DOES read GGUF (`qwen3` -> `qwen`)
    /// is unaffected.
    #[test]
    fn a_raw_gguf_never_reaches_a_safetensors_only_family_resident() {
        let dir = tmp_dir("gguf-family-collision");
        // The store layout hands a sibling tokenizer.json over, so the two
        // residents below would construct happily -- the tokenizer is not what
        // is supposed to be stopping them.
        write_tokenizer(&dir);
        let tok_path = dir.join("tokenizer.json");
        let tok = tok_path.to_str();

        for arch in ["qwen35moe", "qwen35"] {
            let file = format!("{arch}.gguf");
            write_gguf(&dir, &file, arch, &format!("toy-{arch}-gguf"));
            let p = dir.join(&file);
            let card = card_of(&p).expect("gguf card");
            assert_eq!(card.family, arch, "a GGUF card's family IS general.architecture, verbatim");
            assert!(
                resident_for(p.to_str().unwrap(), &card, tok, None).is_none(),
                "raw {arch}.gguf routed into the brain-native '{arch}' resident, which cannot read GGUF bytes"
            );
        }

        // ...and the guidance is the actionable one, not a bare "unservable":
        // an architecture WITH a registered importer names the exact command.
        assert!(gguf_advice("m.gguf", "qwen35moe").contains("run `brain import-gguf m.gguf`"), "{}", gguf_advice("m.gguf", "qwen35moe"));

        // The GGUF path that DOES work must keep working: qwen3 aliases to the
        // `qwen` family, whose resident streams a `.gguf` directly.
        write_gguf_qwen(&dir, "qwen3.gguf", "toy-qwen-gguf");
        let p = dir.join("qwen3.gguf");
        let card = card_of(&p).expect("gguf card");
        assert!(resident_for(p.to_str().unwrap(), &card, None, None).is_some(), "a qwen3 GGUF must still serve from the scan");

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
