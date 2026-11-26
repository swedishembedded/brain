// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for releasing fine-tuned speech
// models that load unchanged in their upstream tooling for its clients. If
// your team needs expertise in text-to-speech training and model release then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Export a trained Qwen3-TTS checkpoint in the UPSTREAM layout: the layout of
//! the base it was trained from, which brain serves as downloaded and the
//! upstream package loads unchanged.
//!
//! The destination holds:
//! - `model.safetensors`: every tensor of the base under its upstream name and
//!   dtype (BF16 for the released checkpoints, rounded to nearest even), the
//!   trained ones replaced and the rest copied byte for byte
//!   ([`checkpoint::rewrite::rewrite_safetensors`]). A named-voice release
//!   leaves out the speaker encoder (`speaker_encoder.*`): upstream builds no
//!   speaker encoder for any checkpoint but `base`, and its own fine-tuning
//!   writes a `custom_voice` checkpoint without one, so the file holds what
//!   its loader builds (brain refuses cloning on a non-Base checkpoint too).
//! - `config.json`: the base's, with `talker_config.codec_language_id`
//!   extended by the trained languages, `spk_id` by the named voices,
//!   `spk_is_dialect` naming every voice (`false` for a new one: upstream reads
//!   it for every named speaker when the language is `chinese` or `auto`), and
//!   `tts_model_type` `custom_voice` once any voice is named (upstream refuses
//!   `generate_custom_voice` on any other kind).
//! - `generation_config.json`, the tokenizer files and `speech_tokenizer/`,
//!   carried over unchanged.
//! - `brain.manifest.json`, as `brain pull` writes it for a Qwen3-TTS repo.
//!
//! The export is atomic: everything is written into a temporary sibling of the
//! destination, synced, and the directory renamed into place. A destination
//! that exists is never overwritten.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::prompt::{CheckpointKind, TtsSpecials};

/// What a training run hands the export.
pub struct Trained<'a> {
    /// Trained tensors under their upstream names (`talker.model.
    /// codec_embedding.weight`, ...), with the base's shapes, as f32.
    pub tensors: &'a [(String, Vec<u64>, Vec<f32>)],
    /// Language name -> codec row, added to (or already in) the base's
    /// `codec_language_id`.
    pub languages: &'a BTreeMap<String, u32>,
    /// Voice name -> codec row, added to (or already in) the base's `spk_id`.
    pub voices: &'a BTreeMap<String, u32>,
}

/// What [`export`] wrote.
#[derive(Debug)]
pub struct Exported {
    /// The kind the exported config declares.
    pub kind: CheckpointKind,
    /// Tensors of the base left out of the export.
    pub left_out: usize,
}

/// The base's tensors a checkpoint of `kind` does not carry.
fn leaves_out(kind: &CheckpointKind, name: &str) -> bool {
    *kind != CheckpointKind::Base && name.starts_with("speaker_encoder.")
}

/// Files carried over from the base when present, beside the ones every
/// release ships ([`REQUIRED_FILES`]).
const OPTIONAL_FILES: &[&str] = &["tokenizer.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", crate::speaker_projection::FILE];
/// Files every Qwen3-TTS release ships and the export carries over.
const REQUIRED_FILES: &[&str] = &["generation_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json"];
/// The codec checkpoint's directory, carried over whole.
const CODEC_DIR: &str = "speech_tokenizer";
/// Longest language or voice name accepted.
const MAX_NAME: usize = 64;

/// Export `trained` over the base checkpoint at `base` into the new directory
/// `out`, registering it under the model id `id` in its manifest. `keep_going`
/// is asked before each tensor is written; once it says no nothing is left
/// behind and the answer is the contract's cancelled error.
///
/// Refused before anything is written: a destination that exists, a trained
/// tensor the base does not hold or of another shape, a language or voice
/// name that is not lowercase `[a-z0-9_-]`, a codec row outside the codec
/// vocabulary, on a special token, or already naming another language or
/// voice, and a name the base already maps to another row.
pub fn export(base: &Path, out: &Path, id: &str, trained: &Trained<'_>, keep_going: &mut dyn FnMut() -> bool) -> Result<Exported, String> {
    if out.exists() {
        return Err(format!("export: {} exists; an export never overwrites a directory", out.display()));
    }
    let base_str = base.to_str().ok_or_else(|| format!("export: {} is not UTF-8", base.display()))?;
    let mut config: Value = serde_json::from_slice(&std::fs::read(base.join("config.json")).map_err(|e| format!("export: read {}/config.json: {e}", base.display()))?)
        .map_err(|e| format!("export: parse {}/config.json: {e}", base.display()))?;
    let specials = TtsSpecials::from_config_dir(base_str)?;
    let kind = extend_config(&mut config, &specials, trained)?;
    for f in REQUIRED_FILES {
        if !base.join(f).is_file() {
            return Err(format!("export: the base has no {f}"));
        }
    }
    if !base.join(CODEC_DIR).join("config.json").is_file() {
        return Err(format!("export: the base has no {CODEC_DIR}/"));
    }
    check_tensors(&base.join("model.safetensors"), trained.tensors)?;

    let leave_out = |n: &str| leaves_out(&kind, n);
    let mut model = |dst: &Path, keep_going: &mut dyn FnMut() -> bool| checkpoint::rewrite::rewrite_safetensors(&base.join("model.safetensors"), dst, trained.tensors, &leave_out, keep_going).map_err(|e| format!("export: {e}"));
    place(out, |tmp| write_release(base, tmp, id, &config, &mut model, &[], keep_going))?;
    let left_out = checkpoint::mmap::MmapSafetensors::open(base.join("model.safetensors"))?.names().iter().filter(|n| leaves_out(&kind, n)).count();
    Ok(Exported { kind, left_out })
}

/// Write a release with `write` into a temporary sibling of `out`, then
/// rename it into place; nothing is left behind on failure or cancellation.
fn place(out: &Path, write: impl FnOnce(&Path) -> Result<bool, String>) -> Result<(), String> {
    let name = out.file_name().and_then(|n| n.to_str()).ok_or_else(|| format!("export: {} names no directory", out.display()))?;
    let tmp = out.with_file_name(format!(".{name}.export-{}.tmp", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    match write(&tmp) {
        Ok(true) => {
            std::fs::rename(&tmp, out).map_err(|e| {
                let _ = std::fs::remove_dir_all(&tmp);
                format!("export: rename into {}: {e}", out.display())
            })?;
            sync_dir(out.parent().unwrap_or(Path::new(".")));
            Ok(())
        }
        Ok(false) => {
            let _ = std::fs::remove_dir_all(&tmp);
            Err(capability::failure::CANCELLED.to_string())
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            Err(e)
        }
    }
}

/// A checkpoint edited into other shapes (narrower layers, fewer of them, a
/// trimmed vocabulary), in the base's tensor names.
pub struct Edited<'a> {
    /// The whole `config.json`, stating every new width.
    pub config: &'a Value,
    /// Every tensor of the edited checkpoint under its upstream name, f32;
    /// each is stored in the dtype the base stores that name in.
    pub tensors: &'a [(String, Vec<u64>, Vec<f32>)],
    /// Tokenizer files rewritten for a trimmed vocabulary, written in place
    /// of the base's: only [`TOKENIZER_FILES`].
    pub files: &'a [(String, Vec<u8>)],
}

/// The tokenizer files an edited export may rewrite.
pub const TOKENIZER_FILES: &[&str] = &["vocab.json", "merges.txt", "tokenizer_config.json"];

/// Export the edited checkpoint `edited` of the base at `base` into the new
/// directory `out`, in the base's layout: `model.safetensors` holding exactly
/// the edited tensors in the base's dtypes, the edited `config.json`, the
/// rewritten tokenizer files, every other file of the base carried over and
/// the manifest under `id`. The result is then opened through brain's own
/// readers ([`crate::import::talker_view`], [`crate::import::mtp_view`]),
/// which check every tensor against the config, and refused (nothing left
/// behind) if they do not accept it.
pub fn export_edited(base: &Path, out: &Path, id: &str, edited: &Edited<'_>, keep_going: &mut dyn FnMut() -> bool) -> Result<(), String> {
    if out.exists() {
        return Err(format!("export: {} exists; an export never overwrites a directory", out.display()));
    }
    if let Some((f, _)) = edited.files.iter().find(|(f, _)| !TOKENIZER_FILES.contains(&f.as_str())) {
        return Err(format!("export: {f} is not a tokenizer file an edit rewrites"));
    }
    for f in REQUIRED_FILES {
        if !base.join(f).is_file() {
            return Err(format!("export: the base has no {f}"));
        }
    }
    let mut model = |dst: &Path, keep_going: &mut dyn FnMut() -> bool| checkpoint::rewrite::write_safetensors_like(&base.join("model.safetensors"), dst, edited.tensors, keep_going).map_err(|e| format!("export: {e}"));
    place(out, |tmp| {
        if !write_release(base, tmp, id, edited.config, &mut model, edited.files, keep_going)? {
            return Ok(false);
        }
        crate::import::talker_view(tmp).and_then(|_| crate::import::mtp_view(tmp)).map_err(|e| format!("export: brain does not read the edited checkpoint back: {e}"))?;
        Ok(true)
    })
}

/// Every trained tensor is one the base holds, at the base's shape. The dtype
/// and value count are checked again by the rewrite itself.
fn check_tensors(model: &Path, tensors: &[(String, Vec<u64>, Vec<f32>)]) -> Result<(), String> {
    let st = checkpoint::mmap::MmapSafetensors::open(model)?;
    for (name, shape, _) in tensors {
        let base: Vec<u64> = st.shape(name).ok_or_else(|| format!("export: {name}: the base checkpoint has no such tensor"))?.iter().map(|&d| d as u64).collect();
        if *shape != base {
            return Err(format!("export: {name}: shape {shape:?}, the base stores {base:?}"));
        }
    }
    Ok(())
}

/// A language or voice name as upstream looks one up (it lowercases the
/// request, so a key with an upper-case letter could never be named).
fn check_name(what: &str, name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > MAX_NAME || !name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-') {
        return Err(format!("export: {what} {name:?}: a name is 1-{MAX_NAME} lowercase letters, digits, '_' or '-'"));
    }
    Ok(())
}

/// Apply the trained languages and voices to the base `config`, returning the
/// kind it now declares.
fn extend_config(config: &mut Value, specials: &TtsSpecials, trained: &Trained<'_>) -> Result<CheckpointKind, String> {
    let vocab = config["talker_config"]["vocab_size"].as_u64().ok_or("export: config.json has no talker_config.vocab_size")?;
    let reserved = [specials.codec_nothink, specials.codec_think, specials.codec_think_bos, specials.codec_think_eos, specials.codec_pad, specials.codec_bos, specials.codec_eos];
    // Every row already named, by what names it, so a new entry cannot take one.
    let mut taken: BTreeMap<u32, String> = BTreeMap::new();
    for (n, r) in &specials.lang {
        taken.insert(*r, format!("language {n:?}"));
    }
    for (n, r) in &specials.spk_id {
        taken.insert(*r, format!("voice {n:?}"));
    }
    let mut claim = |what: &str, name: &str, row: u32, existing: &std::collections::HashMap<String, u32>| -> Result<bool, String> {
        check_name(what, name)?;
        if let Some(&had) = existing.get(name) {
            if had != row {
                return Err(format!("export: {what} {name:?} is codec row {had} in the base, not {row}"));
            }
            return Ok(false);
        }
        if u64::from(row) >= vocab {
            return Err(format!("export: {what} {name:?}: codec row {row} is outside the codec vocabulary ({vocab})"));
        }
        if reserved.contains(&row) {
            return Err(format!("export: {what} {name:?}: codec row {row} is a special token"));
        }
        if let Some(owner) = taken.insert(row, format!("{what} {name:?}")) {
            return Err(format!("export: {what} {name:?}: codec row {row} already names {owner}"));
        }
        Ok(true)
    };
    let mut new_languages = Vec::new();
    for (name, &row) in trained.languages {
        if name.contains("dialect") {
            return Err(format!("export: language {name:?}: upstream hides a language whose name says dialect"));
        }
        if claim("language", name, row, &specials.lang)? {
            new_languages.push((name.clone(), row));
        }
    }
    let mut new_voices = Vec::new();
    for (name, &row) in trained.voices {
        if claim("voice", name, row, &specials.spk_id)? {
            new_voices.push((name.clone(), row));
        }
    }

    let talker = config["talker_config"].as_object_mut().ok_or("export: config.json has no talker_config")?;
    let langs = table(talker, "codec_language_id")?;
    for (name, row) in new_languages {
        langs.insert(name, row.into());
    }
    let spk = table(talker, "spk_id")?;
    for (name, row) in new_voices {
        spk.insert(name, row.into());
    }
    let voices: Vec<String> = spk.keys().cloned().collect();
    let dialect = table(talker, "spk_is_dialect")?;
    for name in &voices {
        dialect.entry(name.clone()).or_insert(Value::Bool(false));
    }
    let kind = if voices.is_empty() { specials.checkpoint.as_ref().map_or(CheckpointKind::Base, |c| c.kind.clone()) } else { CheckpointKind::CustomVoice };
    if !voices.is_empty() {
        config["tts_model_type"] = Value::from("custom_voice");
    }
    Ok(kind)
}

/// The `talker_config` name table `key`, created empty when the base has none.
fn table<'a>(talker: &'a mut serde_json::Map<String, Value>, key: &str) -> Result<&'a mut serde_json::Map<String, Value>, String> {
    talker.entry(key).or_insert_with(|| Value::Object(Default::default())).as_object_mut().ok_or_else(|| format!("export: talker_config.{key} is not an object"))
}

/// Writes `model.safetensors` at the given path; `Ok(false)` once `keep_going` said no.
type ModelWriter<'a> = dyn FnMut(&Path, &mut dyn FnMut() -> bool) -> Result<bool, String> + 'a;

/// Write the whole release into `dir` (created here), every file synced:
/// the model through `model`, `files` in place of the base's files of their
/// names, every other file carried over. `Ok(false)` once `keep_going` said no.
fn write_release(base: &Path, dir: &Path, id: &str, config: &Value, model: &mut ModelWriter<'_>, files: &[(String, Vec<u8>)], keep_going: &mut dyn FnMut() -> bool) -> Result<bool, String> {
    let io = |what: &Path, e: std::io::Error| format!("export: {}: {e}", what.display());
    std::fs::create_dir_all(dir.join(CODEC_DIR)).map_err(|e| io(dir, e))?;
    if !model(&dir.join("model.safetensors"), keep_going)? {
        return Ok(false);
    }
    let rewritten = |f: &Path| files.iter().any(|(n, _)| Path::new(n) == f);
    let mut carried: Vec<PathBuf> = REQUIRED_FILES.iter().chain(OPTIONAL_FILES).map(PathBuf::from).filter(|f| base.join(f).is_file() && !rewritten(f)).collect();
    for entry in std::fs::read_dir(base.join(CODEC_DIR)).map_err(|e| io(&base.join(CODEC_DIR), e))? {
        let entry = entry.map_err(|e| io(&base.join(CODEC_DIR), e))?;
        let file = entry.file_name();
        let Some(file) = file.to_str() else { continue };
        // The store's own integrity sidecars describe the base download.
        if entry.file_type().map_err(|e| io(&entry.path(), e))?.is_file() && !file.starts_with('.') && !file.ends_with(".sha256") {
            carried.push(Path::new(CODEC_DIR).join(file));
        }
    }
    for f in &carried {
        if !keep_going() {
            return Ok(false);
        }
        std::fs::copy(base.join(f), dir.join(f)).map_err(|e| io(&base.join(f), e))?;
        sync_file(&dir.join(f))?;
    }
    for (name, bytes) in files {
        write_synced(&dir.join(name), bytes)?;
    }
    write_synced(&dir.join("config.json"), &serde_json::to_vec_pretty(config).map_err(|e| format!("export: config: {e}"))?)?;
    let (family, roles) = brain_modelstore::recipe::files_recipe_roles("qwen3tts").ok_or("export: the model store has no qwen3tts recipe")?;
    let manifest = brain_modelstore::CompoundManifest { id: id.to_string(), family: family.to_string(), roles: roles.iter().map(|(r, p)| (r.to_string(), p.to_string())).collect() };
    write_synced(&dir.join(brain_modelstore::MANIFEST_FILE), &serde_json::to_vec_pretty(&manifest).map_err(|e| format!("export: manifest: {e}"))?)?;
    sync_dir(&dir.join(CODEC_DIR));
    sync_dir(dir);
    Ok(true)
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("export: {}: {e}", path.display()))?;
    sync_file(path)
}

fn sync_file(path: &Path) -> Result<(), String> {
    std::fs::File::open(path).and_then(|f| f.sync_all()).map_err(|e| format!("export: sync {}: {e}", path.display()))
}

/// Best effort: not every platform or file system can open a directory.
fn sync_dir(dir: &Path) {
    if let Ok(d) = std::fs::File::open(dir) {
        d.sync_all().ok();
    }
}
