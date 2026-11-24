// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A trained checkpoint exported in the upstream layout (`qwen3tts::export`)
//! on the real Base checkpoint: an export with nothing trained keeps every
//! tensor byte for byte; trained tensors come back through brain's own loaders
//! as their BF16-rounded values; the config names the new language and voices
//! and validates; a wrong name or shape is refused and leaves nothing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::mmap::MmapSafetensors;
use qwen3tts::export::{export, Trained};
use qwen3tts::prompt::{CheckpointKind, TtsSpecials};

/// The store's Base checkpoint as downloaded, or a skip.
fn base() -> Option<PathBuf> {
    let paths = crate::store::store_paths()?;
    let dir = PathBuf::from(&paths.ckpt_dir);
    if !dir.join("model.safetensors").is_file() {
        brain_testutil::skip("the store's Qwen3-TTS checkpoint is not in the downloaded layout");
        return None;
    }
    Some(dir)
}

fn bytes_of<'a>(st: &'a MmapSafetensors, name: &str) -> &'a [u8] {
    st.raw_range(name, 0..st.nbytes(name).unwrap() as usize).unwrap()
}

fn bf16_round(v: f32) -> f32 {
    f32::from_bits(u32::from(model::half::f32_to_bf16(v)) << 16)
}

fn config(dir: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap()
}

#[test]
fn an_export_with_nothing_trained_keeps_every_tensor_and_carried_file_byte_for_byte() {
    let Some(base) = base() else { return };
    let scratch = brain_testutil::Scratch::dir("qwen3tts-export-unchanged");
    let out = scratch.path().join("release");
    let (none_l, none_v) = (BTreeMap::new(), BTreeMap::new());
    let done = export(&base, &out, "local/unchanged", &Trained { tensors: &[], languages: &none_l, voices: &none_v }, &mut || true).unwrap();
    assert_eq!(done.kind, CheckpointKind::Base);

    let (a, b) = (MmapSafetensors::open(base.join("model.safetensors")).unwrap(), MmapSafetensors::open(out.join("model.safetensors")).unwrap());
    let (mut na, mut nb) = (a.names().to_vec(), b.names().to_vec());
    na.sort();
    nb.sort();
    assert_eq!(na, nb, "every tensor of the base");
    for name in &na {
        assert_eq!((a.dtype(name), a.shape(name)), (b.dtype(name), b.shape(name)), "{name}");
        assert!(bytes_of(&a, name) == bytes_of(&b, name), "{name} is copied byte for byte");
    }
    for f in ["generation_config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors"] {
        assert!(std::fs::read(base.join(f)).unwrap() == std::fs::read(out.join(f)).unwrap(), "{f} is carried over unchanged");
    }
    assert_eq!(config(&base), config(&out));
    let manifest: brain_modelstore::CompoundManifest = serde_json::from_slice(&std::fs::read(out.join(brain_modelstore::MANIFEST_FILE)).unwrap()).unwrap();
    assert_eq!((manifest.id.as_str(), manifest.family.as_str()), ("local/unchanged", "qwen3tts"));
    assert_eq!(manifest.roles.get("ckpt").map(String::as_str), Some("."));
}

#[test]
fn trained_tensors_reload_through_brain_as_their_bf16_values_with_the_new_language_and_voices_named() {
    let Some(base) = base() else { return };
    let scratch = brain_testutil::Scratch::dir("qwen3tts-export-trained");
    let out = scratch.path().join("release");
    let src = MmapSafetensors::open(base.join("model.safetensors")).unwrap();
    let changed = |name: &str, by: f32| -> (String, Vec<u64>, Vec<f32>) {
        let v: Vec<f32> = src.tensor_f32(name).unwrap().iter().enumerate().map(|(i, x)| x + by * ((i % 7) as f32 - 3.0) * 1e-3).collect();
        (name.to_string(), src.shape(name).unwrap().iter().map(|&d| d as u64).collect(), v)
    };
    let tensors = vec![changed("talker.model.codec_embedding.weight", 1.0), changed("talker.code_predictor.model.codec_embedding.0.weight", -2.0)];
    let languages = BTreeMap::from([("swedish".to_string(), 2087)]);
    let voices = BTreeMap::from([("astrid".to_string(), 2861), ("nils".to_string(), 2862)]);
    let done = export(&base, &out, "local/trained", &Trained { tensors: &tensors, languages: &languages, voices: &voices }, &mut || true).unwrap();
    assert_eq!(done.kind, CheckpointKind::CustomVoice);

    let dir = out.to_str().unwrap();
    let talker = qwen3tts::import::load_talker(dir);
    let mtp = qwen3tts::import::load_mtp(dir);
    let want = |i: usize| tensors[i].2.iter().map(|&v| bf16_round(v)).collect::<Vec<f32>>();
    assert!(talker.find("tok.weight", "").unwrap() == &want(0), "the Talker's codec table reloads as its BF16 values");
    assert!(mtp.find("codec_embedding.0.weight", "").unwrap() == &want(1), "the code predictor's table reloads as its BF16 values");
    let untouched = "talker.model.layers.0.mlp.up_proj.weight";
    let after = MmapSafetensors::open(out.join("model.safetensors")).unwrap();
    assert!(bytes_of(&src, untouched) == bytes_of(&after, untouched), "an untrained tensor is the base's bytes");
    assert!(after.names().iter().all(|n| !n.starts_with("speaker_encoder.")), "a named-voice release carries no speaker encoder, as upstream builds none for it");

    let specials = TtsSpecials::from_config_dir(dir).unwrap();
    assert_eq!(specials.language_id("Swedish").unwrap(), Some(2087));
    assert_eq!(specials.language_id("english").unwrap(), Some(2050), "the base's languages stay");
    assert_eq!((specials.speaker_id("astrid"), specials.speaker_id("nils")), (Some(2861), Some(2862)));
    assert_eq!(specials.checkpoint.as_ref().map(|c| &c.kind), Some(&CheckpointKind::CustomVoice));
    let t = &config(&out)["talker_config"];
    assert_eq!(t["spk_is_dialect"], serde_json::json!({"astrid": false, "nils": false}), "upstream reads a dialect flag for every named voice");
    assert!(config(&out).get("speaker_encoder_config").is_some(), "upstream reads the speaker encoder's sample rate from every config");
    let paths = qwen3tts::TtsPaths::new(&out, dir);
    paths.require(false).unwrap();
    assert!(paths.speaker.is_empty());
}

/// Trained tensors, languages, voices, and a word the refusal names.
type Case = (Vec<(String, Vec<u64>, Vec<f32>)>, BTreeMap<String, u32>, BTreeMap<String, u32>, &'static str);

#[test]
fn an_export_naming_a_wrong_tensor_shape_or_id_is_refused_and_leaves_nothing() {
    let Some(base) = base() else { return };
    let scratch = brain_testutil::Scratch::dir("qwen3tts-export-refused");
    let out = scratch.path().join("release");
    let src = MmapSafetensors::open(base.join("model.safetensors")).unwrap();
    let norm = "talker.model.norm.weight";
    let n = src.numel(norm).unwrap();
    let empty = BTreeMap::new();
    let cases: Vec<Case> = vec![
        (vec![("talker.model.not_a_tensor".to_string(), vec![n as u64], vec![0.0; n])], empty.clone(), empty.clone(), "no such tensor"),
        (vec![(norm.to_string(), vec![1, n as u64], vec![0.0; n])], empty.clone(), empty.clone(), "shape"),
        (vec![], BTreeMap::from([("english".to_string(), 2087)]), empty.clone(), "english"),
        (vec![], BTreeMap::from([("swedish".to_string(), 2050)]), empty.clone(), "2050"),
        (vec![], BTreeMap::from([("Swedish".to_string(), 2087)]), empty.clone(), "lowercase"),
        (vec![], empty.clone(), BTreeMap::from([("astrid".to_string(), 99_999)]), "99999"),
        (vec![], BTreeMap::from([("swedish".to_string(), 2087)]), BTreeMap::from([("astrid".to_string(), 2087)]), "2087"),
        (vec![], empty.clone(), BTreeMap::from([("astrid".to_string(), 2149)]), "2149"),
    ];
    for (tensors, languages, voices, why) in cases {
        let e = export(&base, &out, "local/refused", &Trained { tensors: &tensors, languages: &languages, voices: &voices }, &mut || true).unwrap_err();
        assert!(e.contains(why), "{why}: {e}");
        assert_eq!(std::fs::read_dir(scratch.path()).unwrap().count(), 0, "{why}: nothing is left behind");
    }
    std::fs::create_dir(&out).unwrap();
    let e = export(&base, &out, "local/refused", &Trained { tensors: &[], languages: &empty, voices: &empty }, &mut || true).unwrap_err();
    assert!(e.contains("exists"), "an existing destination is never overwritten: {e}");
}
