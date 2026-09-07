// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kronos's [`ArchSpec`]: `decoder` (the autoregressive net,
//! `NeoQuasar/Kronos-base`) and `tokenizer` (the BSQ tokenizer,
//! `NeoQuasar/Kronos-Tokenizer-base`) - two byte-identical-looking plain
//! transformers repos, told apart only by their own `config.json` content
//! (never the filename or which repo published them): the tokenizer's own
//! config always carries `n_enc_layers`/`n_dec_layers`/`d_in` (the BSQ
//! encoder/decoder stage widths - see [`kronos::config::KronosTokenizerConfig`]),
//! the decoder's always carries `n_layers`/`dep_n_heads` (the AR stack depth
//! and its dependency-layer head count - see [`kronos::config::KronosConfig`]),
//! and neither config ever carries the other's fields. Classification never
//! cares which repo a candidate came from - the same general cross-repo
//! merge FLUX.2's own `classify()` already demonstrates for its
//! cross-vendor text_encoder/tokenizer pairing, just a second instance of
//! it, not a new mechanism.

use std::collections::BTreeMap;
use std::path::Path;

use brain_modelstore::inventory::{ArtifactKind, ArtifactRecord};
use brain_modelstore::resolve::{ArchSpec, AssembleOutcome, AssembledVariant, Confidence};
use capability::Assembly;

use crate::config::{KronosConfig, KronosTokenizerConfig};

pub struct KronosSpec;

const ROLES: &[&str] = &["decoder", "tokenizer"];

fn read_config_json(dir: &Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(dir.join("config.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

impl ArchSpec for KronosSpec {
    fn arch(&self) -> &'static str {
        "kronos"
    }

    fn roles(&self) -> &'static [&'static str] {
        ROLES
    }

    fn classify(&self, records: &[ArtifactRecord], _inventory_root: &Path) -> Vec<(usize, String, Confidence)> {
        let mut out = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            if !rec.usable() || rec.kind != ArtifactKind::HfDir {
                continue;
            }
            let Some(config) = read_config_json(&rec.path) else { continue };
            let has = |k: &str| config.get(k).is_some();
            if has("n_enc_layers") && has("n_dec_layers") && has("d_in") {
                out.push((idx, "tokenizer".to_string(), Confidence::Declared));
            } else if has("n_layers") && has("dep_n_heads") {
                out.push((idx, "decoder".to_string(), Confidence::Declared));
            }
        }
        out
    }

    fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
        let decoder_idx = *chosen.get("decoder").ok_or("kronos assemble: no decoder chosen")?;
        let decoder_path = &records[decoder_idx].path;
        // `<vendor>/<repo>` when the decoder came out of a canonical
        // two-level store layout, else the bare directory name - matches
        // `kronos::forecaster::version_from_path`'s own provenance rule
        // rather than a synthesized id nothing else agrees with.
        let name = decoder_path.file_name().and_then(|n| n.to_str());
        let vendor = decoder_path.parent().and_then(|d| d.file_name()).and_then(|n| n.to_str());
        let id = match (vendor, name) {
            (Some(v), Some(n)) if !v.is_empty() => format!("{v}/{n}"),
            (_, Some(n)) => n.to_string(),
            _ => "local/kronos".to_string(),
        };
        Ok(AssembleOutcome::Assembled(AssembledVariant { id, variant: None }))
    }

    fn validate(&self, assembly: &Assembly) -> Result<(), String> {
        let decoder_path = assembly.roles.get("decoder").ok_or("kronos validate: assembly has no decoder role")?;
        let tokenizer_path = assembly.roles.get("tokenizer").ok_or("kronos validate: assembly has no tokenizer role")?;
        let dc = read_config_json(decoder_path)
            .ok_or_else(|| format!("kronos validate: {} has no readable config.json", decoder_path.display()))
            .and_then(|v| KronosConfig::from_hf(&v))?;
        let tc = read_config_json(tokenizer_path)
            .ok_or_else(|| format!("kronos validate: {} has no readable config.json", tokenizer_path.display()))
            .and_then(|v| KronosTokenizerConfig::from_hf(&v))?;
        // The decoder's dual head predicts INTO the tokenizer's own (s1, s2)
        // codebooks - a header-only check that a decoder trained against one
        // tokenizer's bit width was not paired with a different one's,
        // before either checkpoint's tensor bytes are ever read.
        if dc.s1_bits != tc.s1_bits || dc.s2_bits != tc.s2_bits {
            return Err(format!(
                "kronos validate: decoder (s1_bits={}, s2_bits={}) does not match tokenizer (s1_bits={}, s2_bits={}) - decoder={}, tokenizer={}",
                dc.s1_bits,
                dc.s2_bits,
                tc.s1_bits,
                tc.s2_bits,
                decoder_path.display(),
                tokenizer_path.display()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::Completeness;
    use brain_modelstore::resolve::{resolve, Resolution};
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("brain-kronos-spec-test-{tag}-{}-{n}", std::process::id()))
    }

    fn complete(path: PathBuf, kind: ArtifactKind) -> ArtifactRecord {
        ArtifactRecord { path, size: 1, mtime_ns: 0, kind, completeness: Completeness::Complete }
    }

    fn write_hfdir(dir: &Path, config: &serde_json::Value) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
        // hfdir_record requires either an index or at least one loose
        // model*.safetensors shard beside config.json to collapse to an
        // HfDir record at all - a real Kronos repo ships exactly one.
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
    }

    fn tokenizer_json(s1_bits: u64, s2_bits: u64) -> serde_json::Value {
        serde_json::json!({"d_in": 6, "d_model": 256, "n_heads": 4, "ff_dim": 512, "n_enc_layers": 4, "n_dec_layers": 4, "s1_bits": s1_bits, "s2_bits": s2_bits, "group_size": 4})
    }

    fn decoder_json(s1_bits: u64, s2_bits: u64) -> serde_json::Value {
        serde_json::json!({"d_model": 512, "n_layers": 8, "n_heads": 8, "ff_dim": 1024, "s1_bits": s1_bits, "s2_bits": s2_bits, "learn_te": true, "dep_n_heads": 4, "max_context": 512})
    }

    /// The whole point of this migration: two separate repos (real
    /// `NeoQuasar/Kronos-base` + `NeoQuasar/Kronos-Tokenizer-base` shape)
    /// resolve through the model-store resolver with ZERO `BRAIN_KRONOS_*`
    /// environment variables set - content alone tells them apart.
    #[test]
    fn resolves_two_separate_repos_by_content_with_no_env_vars_set() {
        let dir = tmp("resolves-clean");
        write_hfdir(&dir.join("NeoQuasar").join("Kronos-base"), &decoder_json(10, 10));
        write_hfdir(&dir.join("NeoQuasar").join("Kronos-Tokenizer-base"), &tokenizer_json(10, 10));

        let records = brain_modelstore::inventory::scan(&dir);
        let spec = KronosSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("kronos", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Resolved(a) => {
                assert_eq!(a.id, "NeoQuasar/Kronos-base");
                assert_eq!(a.roles["decoder"], dir.join("NeoQuasar").join("Kronos-base"));
                assert_eq!(a.roles["tokenizer"], dir.join("NeoQuasar").join("Kronos-Tokenizer-base"));
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    /// Content, never the filename: a repo directory named exactly like the
    /// real tokenizer release but whose `config.json` is actually the
    /// DECODER's shape must classify as `decoder`, not `tokenizer`.
    #[test]
    fn classify_uses_content_not_the_repo_name() {
        let dir = tmp("content-not-name");
        let suggestive = dir.join("NeoQuasar").join("Kronos-Tokenizer-base");
        write_hfdir(&suggestive, &decoder_json(10, 10));

        let records = brain_modelstore::inventory::scan(&dir);
        let out = KronosSpec.classify(&records, dir.as_path());
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].1, "decoder", "{out:?}");
    }

    /// A decoder trained against a different tokenizer bit width must be
    /// rejected before any tensor is read - the header-only cross-check
    /// `validate` exists for.
    #[test]
    fn a_decoder_and_tokenizer_with_mismatched_bit_widths_is_rejected_by_validate() {
        let dir = tmp("mismatched-validate");
        let decoder_dir = dir.join("NeoQuasar").join("Kronos-base");
        let tokenizer_dir = dir.join("NeoQuasar").join("Kronos-Tokenizer-base");
        std::fs::create_dir_all(&decoder_dir).unwrap();
        std::fs::create_dir_all(&tokenizer_dir).unwrap();
        std::fs::write(decoder_dir.join("config.json"), serde_json::to_vec(&decoder_json(10, 10)).unwrap()).unwrap();
        std::fs::write(tokenizer_dir.join("config.json"), serde_json::to_vec(&tokenizer_json(8, 8)).unwrap()).unwrap();

        let assembly = Assembly {
            id: "local/kronos".to_string(),
            arch: "kronos".to_string(),
            variant: None,
            roles: BTreeMap::from([("decoder".to_string(), decoder_dir.clone()), ("tokenizer".to_string(), tokenizer_dir.clone())]),
            provenance: Vec::new(),
        };
        let err = KronosSpec.validate(&assembly).unwrap_err();
        assert!(err.contains("s1_bits=10") && err.contains("s1_bits=8"), "{err}");
    }

    /// Two candidates for the SAME role (a real store commonly has more than
    /// one decoder variant around, e.g. `Kronos-small` beside `Kronos-base`)
    /// must be ambiguous, never a silent pick.
    #[test]
    fn two_decoder_candidates_at_equal_confidence_is_ambiguous_not_a_silent_pick() {
        let dir = tmp("two-decoders");
        write_hfdir(&dir.join("NeoQuasar").join("Kronos-base"), &decoder_json(10, 10));
        write_hfdir(&dir.join("NeoQuasar").join("Kronos-small"), &decoder_json(10, 10));
        write_hfdir(&dir.join("NeoQuasar").join("Kronos-Tokenizer-base"), &tokenizer_json(10, 10));

        let records = brain_modelstore::inventory::scan(&dir);
        let spec = KronosSpec;
        let specs: Vec<&dyn ArchSpec> = vec![&spec];
        let out = resolve("kronos", &records, &specs, &BTreeMap::new());
        match out {
            Resolution::Ambiguous(a) => assert_eq!(a.choices.len(), 2, "{a:?}"),
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }
}
