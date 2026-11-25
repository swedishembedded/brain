// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `speech_sft` end to end on the CPU backend: a Qwen3-TTS base in the upstream layout in the model store, a training
//! shard as a blob or a digest, a real executor. The run streams its stages and steps, leaves as a manifest of its
//! files in the store, installs into the model store where the server serves it under its own id with its voices and
//! languages, resumes by key, and refuses at the door what it cannot take (a path, an unknown key, a digest the store
//! does not hold, a shard larger than the job may stage).

use std::path::Path;

use serde_json::{json, Value};

use crate::support::{data_digest, rig, Frames, Rig};

/// The model-store name the base is put under.
pub const TTS_BASE: &str = "local/tts-base";

/// A Qwen3-TTS base in the layout upstream releases it: the root config, tokenizer and generation files, a BF16
/// `model.safetensors` (a handful of tensors under upstream names, the speaker encoder's among them) and the codec in
/// `speech_tokenizer/`, with the manifest `brain pull` writes.
pub fn write_tts_base(models: &Path) {
    let dir = models.join(TTS_BASE);
    std::fs::create_dir_all(dir.join("speech_tokenizer")).unwrap();
    let config = json!({
        "architectures": ["Qwen3TTSForConditionalGeneration"], "tts_model_type": "base", "tts_model_size": "0b6",
        "speaker_encoder_config": {"enc_dim": 4, "sample_rate": 24000},
        "talker_config": {"vocab_size": 3072, "num_code_groups": 16, "codec_language_id": {"english": 2050, "german": 2053}, "spk_id": {}, "spk_is_dialect": {}},
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    for (file, body) in [("generation_config.json", "{}"), ("tokenizer_config.json", "{}"), ("vocab.json", "{}"), ("merges.txt", "#version: 0.2\n"), ("preprocessor_config.json", "{}"), ("speech_tokenizer/config.json", "{}")] {
        std::fs::write(dir.join(file), body).unwrap();
    }
    let bf16 = |n: usize| -> Vec<u8> { (0..n).flat_map(|i| half_bits(i as f32 * 0.25)).collect() };
    let (a, b, c) = (bf16(3072 * 2), bf16(8), bf16(4));
    let views = vec![
        ("talker.model.codec_embedding.weight", safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![3072, 2], &a).unwrap()),
        ("talker.model.norm.weight", safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![8], &b).unwrap()),
        ("speaker_encoder.fc.bias", safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![4], &c).unwrap()),
    ];
    std::fs::write(dir.join("model.safetensors"), safetensors::serialize(views, None).unwrap()).unwrap();
    std::fs::write(dir.join("speech_tokenizer/model.safetensors"), safetensors::serialize(Vec::<(&str, safetensors::tensor::TensorView)>::new(), None).unwrap()).unwrap();
    let manifest = json!({"id": TTS_BASE, "family": "qwen3tts", "roles": {"ckpt": ".", "weights_dir": "."}});
    std::fs::write(dir.join(brain_modelstore::MANIFEST_FILE), manifest.to_string()).unwrap();
}

fn half_bits(v: f32) -> [u8; 2] {
    // Round to nearest even, as a release stores BF16.
    let bits = v.to_bits();
    (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16).to_le_bytes()
}

/// A shard of one Swedish utterance by the voice `astrid`, at the base's codebooks and x-vector width.
pub fn shard() -> Vec<u8> {
    use qwen3tts::shard::{Record, RecordId, ShardMeta, ShardWriter, FORMAT};
    let meta = ShardMeta { format: FORMAT.into(), codebooks: 16, codebook_size: 2048, xvector_dim: 4, frame_rate: 12.5, languages: vec!["swedish".into()], voices: vec!["astrid".into()], records: 0, producers: Default::default() };
    let mut w = ShardWriter::new(meta);
    let codes: Vec<u16> = (0..32).collect();
    let id = RecordId { id: "u1".into(), speaker: "s1".into(), source: "test".into(), text: "hej".into() };
    w.push(&Record { id, text_ids: &[151644, 77091, 198, 9707, 151645], codes: &codes, xvector: &[0.1, 0.2, 0.3, 0.4], voice: 0, language: 0, complete: true }).unwrap();
    w.to_bytes()
}

fn tts_rig(tag: &str) -> Rig {
    let rig = rig(tag, 16 << 30, true);
    write_tts_base(&rig.dir.join("models"));
    rig
}

pub const LANGUAGES: &str = r#"{"swedish": {"new": true, "init_from": ["english", "german"]}}"#;
pub const VOICES: &str = r#"{"astrid": {"new": true}}"#;

fn params(extra: Value) -> Value {
    let mut p = json!({"base": TTS_BASE, "languages": LANGUAGES, "voices": VOICES, "steps": 2, "seed": 5, "allow_cpu": true});
    for (k, v) in extra.as_object().unwrap() {
        p[k] = v.clone();
    }
    p
}

fn run(rig: &Rig, params: Value, blobs: &[(&str, Vec<u8>)], frames: &Frames) -> Result<capability::Outcome, String> {
    crate::support::run_action(rig, trainjob::speech::ACTION, &trainjob::speech::spec(), params, blobs, frames)
}

#[test]
fn a_speech_fine_tune_leaves_as_an_upstream_layout_checkpoint_installed_and_served_under_its_own_id() {
    let rig = tts_rig("speech-run");
    let frames = Frames::default();
    let out = run(&rig, params(json!({"install_as": "sv-voices"})), &[("dataset", shard())], &frames).unwrap();
    let report = &out.outputs;
    assert_eq!((report["status"].as_str(), report["objective"].as_str(), report["tts_model_type"].as_str()), (Some("completed"), Some("speech_sft"), Some("custom_voice")), "{report}");
    assert!(frames.events().iter().any(|e| e["type"] == "stage"), "the run streams its stages");

    // It leaves as a manifest of its files by digest.
    let manifest_digest = trainjob::Digest::parse(report["checkpoint_digest"].as_str().unwrap()).unwrap();
    let manifest: Value = serde_json::from_slice(&std::fs::read(rig.store.root().join(manifest_digest.hex())).unwrap()).unwrap();
    assert_eq!(manifest["format"], "brain-checkpoint-manifest-v1");
    let names: Vec<&str> = manifest["files"].as_array().unwrap().iter().map(|f| f["name"].as_str().unwrap()).collect();
    for want in ["model.safetensors", "config.json", "generation_config.json", "vocab.json", "speech_tokenizer/model.safetensors", "brain.manifest.json"] {
        assert!(names.contains(&want), "{want} is published: {names:?}");
    }

    // Installed, it names its language and voice, and the server serves it as a model of its own.
    assert_eq!(report["installed_as"], "local/sv-voices");
    let installed = rig.dir.join("models").join("local").join("sv-voices");
    let specials = qwen3tts::TtsSpecials::from_config_dir(installed.to_str().unwrap()).unwrap();
    let (lang, voice) = (report["languages"]["swedish"].as_u64().unwrap() as u32, report["voices"]["astrid"].as_u64().unwrap() as u32);
    assert_eq!((specials.language_id("swedish").unwrap(), specials.speaker_id("astrid")), (Some(lang), Some(voice)));
    let served = catalog::resident_tts::TtsResident::all_from_store(&loader::served::ServedStore::new(Some(rig.dir.join("models"))));
    let ids: Vec<String> = served.iter().map(|r| residency::ResidentModel::manifest(r).model).collect();
    assert!(ids.contains(&"local/sv-voices".to_string()) && ids.contains(&TTS_BASE.to_string()), "the trained checkpoint is served beside its base: {ids:?}");
    crate::support::assert_reservation_returned(&rig);
}

#[test]
fn a_speech_fine_tune_by_digest_resumes_by_key_and_answers_a_completed_run_again() {
    let rig = tts_rig("speech-resume");
    let digest = data_digest(&rig, &shard());
    let p = params(json!({"dataset_digest": digest.hex(), "resume_key": "sv-run-1"}));
    let first = run(&rig, p.clone(), &[], &Frames::default()).unwrap();
    assert_eq!(first.outputs["resume"]["state"], "fresh", "{}", first.outputs);
    let again = run(&rig, p, &[], &Frames::default()).unwrap();
    assert_eq!(again.outputs["resume"]["state"], "completed");
    assert_eq!(again.outputs["checkpoint_digest"], first.outputs["checkpoint_digest"], "the recorded result, not a second run");
}

#[test]
fn validate_only_checks_the_shards_against_the_names_and_trains_nothing() {
    let rig = tts_rig("speech-validate");
    let out = run(&rig, params(json!({"validate_only": true})), &[("dataset", shard())], &Frames::default()).unwrap();
    assert_eq!((out.outputs["status"].as_str(), out.outputs["records"].as_u64()), (Some("valid"), Some(1)), "{}", out.outputs);
    let refused = run(&rig, params(json!({"validate_only": true, "voices": "{}"})), &[("dataset", shard())], &Frames::default()).unwrap_err();
    assert!(refused.contains("astrid") && refused.contains("voices"), "a shard voice the request does not name: {refused}");
    let garbage = run(&rig, params(json!({"validate_only": true})), &[("dataset", b"not a shard".to_vec())], &Frames::default()).unwrap_err();
    assert!(garbage.contains("dataset"), "{garbage}");
}

/// A call's input blobs, by name.
type Blobs = Vec<(&'static str, Vec<u8>)>;

#[test]
fn a_speech_fine_tune_is_refused_at_the_door_for_paths_unknown_keys_missing_digests_and_wrong_names() {
    let rig = tts_rig("speech-door");
    let missing = "ab".repeat(32);
    let cases: Vec<(Value, Blobs, &str)> = vec![
        (params(json!({"out": "x"})), vec![("dataset", shard())], "unknown param"),
        (params(json!({"weights_dir": "models"})), vec![("dataset", shard())], "unknown param"),
        (params(json!({"dataset_digest": missing})), vec![], "not in the artifact store"),
        (params(json!({"dataset_digest": "../../../etc/shadow"})), vec![], "dataset_digest"),
        (params(json!({"base": "local/toy"})), vec![("dataset", shard())], "not a Qwen3-TTS checkpoint"),
        (params(json!({"base": "../../etc"})), vec![("dataset", shard())], "base"),
        (params(json!({"languages": r#"{"english": {"new": true}}"#})), vec![("dataset", shard())], "the base's own"),
        (params(json!({"languages": r#"{"swedish": {"new": false}}"#})), vec![("dataset", shard())], "not the base's"),
        (params(json!({"languages": r#"{"swedish": {"new": true, "init_from": ["klingon"]}}"#})), vec![("dataset", shard())], "klingon"),
        (params(json!({})), vec![("dataset", shard()), ("ref", vec![1])], "unknown input"),
    ];
    for (p, blobs, why) in cases {
        let e = run(&rig, p.clone(), &blobs, &Frames::default()).unwrap_err();
        assert!(e.contains(why), "{p}: expected {why:?}, got {e}");
    }
    assert!(crate::support::work_is_clean(&rig.work_root), "a refusal stages nothing");
}

#[test]
fn a_shard_larger_than_the_job_may_stage_is_refused_and_leaves_nothing() {
    let rig = crate::support::rig_with_quotas("speech-quota", trainjob::Quotas { max_job_staging_bytes: 1 << 20, min_free_bytes: 0, ..trainjob::Quotas::default() });
    write_tts_base(&rig.dir.join("models"));
    let e = run(&rig, params(json!({})), &[("dataset", vec![0u8; 4 << 20])], &Frames::default()).unwrap_err();
    assert!(e.contains("bytes"), "{e}");
    assert!(crate::support::work_is_clean(&rig.work_root));
}

/// On the real Base checkpoint (skips when the model store holds none): the exported release is served under its
/// own id and speaks in its new voice and language; the base's tensors come back byte for byte while the objective
/// is the stand-in, so this proves the path, not the training.
#[test]
fn a_trained_release_of_the_real_base_is_served_with_its_voice_and_language() {
    let Some(real) = brain_testutil::model_dir("Qwen/Qwen3-TTS-12Hz-0.6B-Base").filter(|d| Path::new(d).join("model.safetensors").is_file()) else {
        brain_testutil::skip("Qwen/Qwen3-TTS-12Hz-0.6B-Base is not in the model store");
        return;
    };
    let rig = rig("speech-real", 64 << 30, true);
    let vendor = rig.dir.join("models").join("Qwen");
    std::fs::create_dir_all(&vendor).unwrap();
    std::os::unix::fs::symlink(&real, vendor.join("Qwen3-TTS-12Hz-0.6B-Base")).unwrap();
    let meta = qwen3tts::shard::ShardMeta { format: qwen3tts::shard::FORMAT.into(), codebooks: 16, codebook_size: 2048, xvector_dim: 1024, frame_rate: 12.5, languages: vec!["swedish".into()], voices: vec!["astrid".into()], records: 0, producers: Default::default() };
    let mut w = qwen3tts::shard::ShardWriter::new(meta);
    let codes: Vec<u16> = (0..32).collect();
    let id = qwen3tts::shard::RecordId { id: "u1".into(), speaker: "s1".into(), source: "test".into(), text: "hej".into() };
    w.push(&qwen3tts::shard::Record { id, text_ids: &[151644, 77091, 198, 9707, 151645], codes: &codes, xvector: &[0.01; 1024], voice: 0, language: 0, complete: true }).unwrap();
    let p = json!({"base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "languages": LANGUAGES, "voices": VOICES, "steps": 1, "install_as": "sv-real", "allow_cpu": true});
    let out = run(&rig, p, &[("dataset", w.to_bytes())], &Frames::default()).unwrap();
    assert_eq!(out.outputs["installed_as"], "local/sv-real", "{}", out.outputs);

    let served = catalog::resident_tts::TtsResident::all_from_store(&loader::served::ServedStore::new(Some(rig.dir.join("models"))));
    let sv = served.iter().find(|r| residency::ResidentModel::manifest(*r).model == "local/sv-real").expect("the release is served under its own id");
    let inv = capability::Invocation::new().set("text", json!("Hej, det här är ett test.")).set("speaker", json!("astrid")).set("lang", json!("swedish")).set("max_frames", json!(6)).set("seed", json!(1));
    let key = residency::ResidentModel::instance_key(sv, "design", &inv);
    let mut inst = residency::ResidentModel::activate(sv, &key, residency::Device::Cpu).unwrap();
    let spoken = inst.run("design", &inv, &mut |_| {}).unwrap();
    let wav = audio::wav::parse(&spoken.blobs["audio"].bytes).unwrap();
    assert!(!wav.samples.is_empty(), "it speaks as its voice, in its language");
    let refused = inst.run("design", &inv.clone().set("speaker", json!("nobody")), &mut |_| {}).unwrap_err();
    assert!(refused.contains("nobody") || refused.contains("speaker"), "{refused}");
}
