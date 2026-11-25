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

/// The tiny release's widths: a 2-layer Talker 16 wide over 48 codec rows, a 2-layer code predictor over 4 codebooks
/// of 16, x-vectors 16 wide.
const D: u32 = 16;
const GROUPS: u32 = 4;
const CODEBOOK: u32 = 16;
/// The base's one language row and its codec specials, above the codebook.
const ENGLISH: u32 = 30;
const SPECIALS: [(&str, u32); 7] = [("codec_nothink_id", 20), ("codec_think_id", 21), ("codec_think_bos_id", 22), ("codec_think_eos_id", 23), ("codec_pad_id", 24), ("codec_bos_id", 25), ("codec_eos_token_id", 26)];

/// A tiny trainable Qwen3-TTS base in the layout upstream releases it: `config.json`, `model.safetensors` under
/// upstream names (its named codec rows trained, the rest of the special band untrained, as in a release), the
/// tokenizer and generation files and the codec in `speech_tokenizer/`, with the manifest `brain pull` writes.
pub fn write_tts_base(models: &Path) {
    use model::Model;
    use qwen3tts::speech_sft::{SftShape, SpeechSftModel};
    let dir = models.join(TTS_BASE);
    std::fs::create_dir_all(dir.join("speech_tokenizer")).unwrap();
    let talker = qwen3tts::TalkerConfig { vocab: 48, d_model: D, num_code_groups: GROUPS, ..qwen3tts::TalkerConfig::tiny() };
    let mtp = qwen3tts::MtpConfig { vocab: CODEBOOK, embedding_dim: D, num_code_groups: GROUPS, ..qwen3tts::MtpConfig::tiny() };
    let mut specials = qwen3tts_specials();
    specials.lang.insert("english".into(), ENGLISH);
    let shape = SftShape { talker: talker.clone(), mtp: mtp.clone(), specials, batch: 1, block: 1, frames: 1, mtp_weight: 0.3, lr: Default::default(), new_rows: Vec::new(), text_inter: 20, gemm: qwen3tts::speech_sft::GemmPrecision::Fp32Reference, gt_weight: 1.0, kd: None };
    let mut tensors = SpeechSftModel::init_weights(&shape, 0x7a6); // seed: a fixed fixture
    let table = tensors.get_mut(qwen3tts::speech_sft::model::TALKER_TABLE).unwrap();
    let named: Vec<u32> = SPECIALS.iter().map(|(_, id)| *id).chain([ENGLISH]).collect();
    for id in CODEBOOK..48 {
        let scale = if named.contains(&id) { 1.0 } else { 0.002 };
        table[id as usize * D as usize..(id as usize + 1) * D as usize].fill(scale);
    }
    let mut talker_cfg = json!({
        "num_hidden_layers": talker.n_layers, "hidden_size": D, "head_dim": talker.head_dim, "num_attention_heads": talker.n_heads,
        "num_key_value_heads": talker.n_kv_heads, "intermediate_size": talker.d_ff, "vocab_size": 48, "num_code_groups": GROUPS,
        "text_hidden_size": talker.text_hidden_size, "text_vocab_size": talker.text_vocab_size,
        "codec_language_id": {"english": ENGLISH}, "spk_id": {}, "spk_is_dialect": {},
        "code_predictor_config": {"num_hidden_layers": mtp.n_layers, "hidden_size": mtp.d_model, "head_dim": mtp.head_dim, "num_attention_heads": mtp.n_heads, "num_key_value_heads": mtp.n_kv_heads, "intermediate_size": mtp.d_ff, "vocab_size": CODEBOOK, "num_code_groups": GROUPS},
    });
    for (key, id) in SPECIALS {
        talker_cfg[key] = json!(id);
    }
    let config = json!({"architectures": ["Qwen3TTSForConditionalGeneration"], "tts_bos_token_id": 0, "tts_eos_token_id": 1, "tts_pad_token_id": 2, "tts_model_type": "base", "tts_model_size": "tiny", "speaker_encoder_config": {"enc_dim": D, "sample_rate": 24000}, "talker_config": talker_cfg});
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let plan: Vec<(String, Vec<u64>)> = names.iter().map(|n| ((*n).clone(), vec![tensors[*n].len() as u64])).collect();
    let mut w = checkpoint::weightio::StWriter::create(dir.join("model.safetensors").to_str().unwrap(), &plan, &Value::Null, None).unwrap();
    for n in names {
        w.write(n, &tensors[n]).unwrap();
    }
    w.finish().unwrap();
    for (file, body) in [("generation_config.json", "{}"), ("tokenizer_config.json", "{}"), ("vocab.json", "{}"), ("merges.txt", "#version: 0.2\n"), ("preprocessor_config.json", "{}"), ("speech_tokenizer/config.json", "{}")] {
        std::fs::write(dir.join(file), body).unwrap();
    }
    std::fs::write(dir.join("speech_tokenizer/model.safetensors"), safetensors::serialize(Vec::<(&str, safetensors::tensor::TensorView)>::new(), None).unwrap()).unwrap();
    let manifest = json!({"id": TTS_BASE, "family": "qwen3tts", "roles": {"ckpt": ".", "weights_dir": "."}});
    std::fs::write(dir.join(brain_modelstore::MANIFEST_FILE), manifest.to_string()).unwrap();
}

/// The tiny release's specials, as its config names them.
fn qwen3tts_specials() -> qwen3tts::TtsSpecials {
    let get = |k: &str| SPECIALS.iter().find(|(n, _)| *n == k).unwrap().1;
    qwen3tts::TtsSpecials {
        tts_bos: 0,
        tts_eos: 1,
        tts_pad: 2,
        codec_nothink: get("codec_nothink_id"),
        codec_think: get("codec_think_id"),
        codec_think_bos: get("codec_think_bos_id"),
        codec_think_eos: get("codec_think_eos_id"),
        codec_pad: get("codec_pad_id"),
        codec_bos: get("codec_bos_id"),
        codec_eos: get("codec_eos_token_id"),
        lang: Default::default(),
        spk_id: Default::default(),
        checkpoint: None,
    }
}

/// A shard of three Swedish utterances, two by the voice `astrid`, at the tiny base's codebooks and x-vector width.
pub fn shard() -> Vec<u8> {
    shard_drawn(0x56) // seed: a fixed fixture
}

/// [`shard`] with its codes and x-vectors drawn from `seed`: another seed is a held-out shard of the same names.
fn shard_drawn(seed: u64) -> Vec<u8> {
    use qwen3tts::shard::{Record, RecordId, ShardMeta, ShardWriter, FORMAT, NO_VOICE};
    let meta = ShardMeta { format: FORMAT.into(), codebooks: GROUPS, codebook_size: CODEBOOK, xvector_dim: D, frame_rate: 12.5, languages: vec!["swedish".into()], voices: vec!["astrid".into()], records: 0, producers: Default::default() };
    let mut w = ShardWriter::new(meta);
    let mut rng = data::rng::Lcg::new(seed);
    for (i, (text, frames, voice)) in [(&[11u32, 12, 13][..], 4usize, true), (&[14, 15][..], 3, false), (&[16, 17, 18, 19][..], 5, true)].into_iter().enumerate() {
        let ids: Vec<u32> = [3, 4, 5].iter().chain(text).chain(&[6, 7, 8, 9, 10]).copied().collect();
        let codes: Vec<u16> = (0..frames * GROUPS as usize).map(|_| (rng.next_u64() % u64::from(CODEBOOK)) as u16).collect();
        let xvector = rng.vec_scaled(D as usize, 0.5);
        let id = RecordId { id: format!("u{i}"), speaker: "s1".into(), source: "test".into(), text: String::new() };
        w.push(&Record { id, text_ids: &ids, codes: &codes, xvector: &xvector, voice: if voice { 0 } else { NO_VOICE }, language: 0, complete: true }).unwrap();
    }
    w.to_bytes()
}

fn tts_rig(tag: &str) -> Rig {
    let rig = rig(tag, 16 << 30, true);
    write_tts_base(&rig.dir.join("models"));
    rig
}

pub const LANGUAGES: &str = r#"{"swedish": {"new": true, "init_from": ["english"]}}"#;
pub const VOICES: &str = r#"{"astrid": {"new": true}}"#;

fn params(extra: Value) -> Value {
    let mut p = json!({"base": TTS_BASE, "languages": LANGUAGES, "voices": VOICES, "steps": 2, "batch_utterances": 2, "max_frames": 8, "lr": 1e-2, "seed": 5, "allow_cpu": true});
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
    let steps = frames.steps();
    assert_eq!(steps.len(), 2, "one frame per step: {steps:?}");
    for step in &steps {
        for metric in ["loss", "talker_ce", "code_predictor_ce", "eos_accuracy", "grad_norm", "lr", "frames_per_s"] {
            assert!(step[metric].is_number(), "a step frame carries {metric}: {step}");
        }
    }
    assert!(report["trained_tensors"].as_u64().unwrap() > 0 && report["steps_completed"] == 2, "{report}");

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
    let (before, after) = (checkpoint::mmap::MmapSafetensors::open(rig.dir.join("models").join(TTS_BASE).join("model.safetensors")).unwrap(), checkpoint::mmap::MmapSafetensors::open(installed.join("model.safetensors")).unwrap());
    let head = "talker.codec_head.weight";
    assert_ne!(before.tensor_f32(head), after.tensor_f32(head), "the trained weights are the release's");
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
    assert_eq!((out.outputs["status"].as_str(), out.outputs["records"].as_u64()), (Some("valid"), Some(3)), "{}", out.outputs);
    let charged = |p: Value| run(&rig, params(json!({"validate_only": true}).as_object().unwrap().clone().into_iter().chain(p.as_object().unwrap().clone()).collect()), &[("dataset", shard())], &Frames::default()).unwrap().outputs["device_bytes"].as_u64().unwrap();
    let (small, large) = (charged(json!({"batch_utterances": 2})), charged(json!({"batch_utterances": 4})));
    assert!(large > small && charged(json!({"batch_utterances": 2, "max_frames": 8})) == small && charged(json!({"batch_utterances": 2, "max_frames": 4})) < small, "a rank's charge grows with the frames a micro-batch can hold: {small} {large}");
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

/// A server holding the tiny Qwen3-TTS base, given the cards `gpus` (computing on the CPU backend, as the suite does)
/// and, with `job_access`, taking part in jobs spread over several servers.
fn tts_server(tag: &str, gpus: &[u32], job_access: Option<trainjob::job::JobAccess>) -> Rig {
    let rig = crate::support::rig_qwen3_serving_jobs(tag, 16 << 30, gpus, false, job_access);
    write_tts_base(&rig.dir.join("models"));
    rig
}

/// A loopback address no one listens on now.
fn free_address() -> String {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().to_string()
}

/// A speech fine-tune trains data-parallel: on two cards of one server, and as one job over two servers of one card
/// each, the two spreads train the same checkpoint (every rank its share of the records, the same rank-ordered
/// gradient sums); rank 0's server publishes it and the other server answers for its ranks only.
#[test]
fn a_speech_fine_tune_on_two_cards_trains_what_one_job_over_two_servers_trains_and_rank_zeros_server_publishes_it() {
    let two = tts_server("speech-two-cards", &[0, 1], None);
    let frames = Frames::default();
    let alone = run(&two, params(json!({"cards": 2, "steps": 3})), &[("dataset", shard())], &frames).expect("two cards of one server").outputs;
    assert_eq!((alone["status"].as_str(), alone["steps_completed"].as_u64(), alone["parallel"]["world_size"].as_u64()), (Some("completed"), Some(3), Some(2)), "{alone}");
    assert_eq!(frames.steps().len(), 3, "rank 0 streams one frame a step");
    crate::support::assert_reservation_returned(&two);

    let secret = brain::JobSecret::random();
    let coordinator = free_address();
    let access = |coordinator: Option<&str>| Some(trainjob::job::JobAccess { secret: secret.clone(), coordinator: coordinator.map(str::to_string) });
    let (first, second) = (tts_server("speech-job-a", &[0], access(Some(&coordinator))), tts_server("speech-job-b", &[0], access(None)));
    let plan = json!({"job": "speech-test", "world_size": 2, "tp": 1, "pp": 1, "dp": 2, "coordinator": coordinator, "handshake_timeout_ms": 60000, "op_timeout_ms": 60000});
    let part = |first_rank: usize| params(json!({"steps": 3, "job": json!({"plan": plan, "first_rank": first_rank, "secret_sha256": secret.digest_hex()}).to_string()}));
    let (published, worker) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| run(&second, part(1), &[("dataset", shard())], &Frames::default()));
        (run(&first, part(0), &[("dataset", shard())], &Frames::default()), worker.join().unwrap())
    });
    let (published, worker) = (published.expect("rank 0's server").outputs, worker.expect("the other server").outputs);
    assert_eq!(published["checkpoint_digest"], alone["checkpoint_digest"], "one job over two servers trains what two cards of one server train");
    assert_eq!(worker, json!({"status": "completed", "objective": "speech_sft", "job": "speech-test", "ranks": [1, 1], "steps_completed": 3}), "the other server answers for its ranks only");
    crate::support::assert_reservation_returned(&first);
    crate::support::assert_reservation_returned(&second);
}

/// Selecting on held-out loss: with a patience the run stops once the held-out loss has not improved for that many
/// evaluations, and publishes the best evaluation's checkpoint, the one a run stopped at that step publishes; the
/// report carries the held-out curve and the chosen step and loss. On two cards, every rank stops at the same step.
#[test]
fn a_speech_fine_tune_with_patience_stops_early_and_publishes_its_best_evaluations_checkpoint() {
    let rig = tts_server("speech-best", &[0, 1], None);
    let blobs = [("dataset", shard()), ("held_out", shard_drawn(0x99))];
    // A constant rate, so a run of fewer steps trains exactly the first steps of a longer one.
    let selecting = params(json!({"cards": 2, "steps": 40, "lr": 0.2, "schedule": "constant", "eval_every": 1, "patience": 2}));
    let out = run(&rig, selecting, &blobs, &Frames::default()).expect("a selecting run").outputs;
    let curve = out["held_out_curve"].as_array().unwrap();
    let (best_at, best) = curve.iter().map(|e| (e["step"].as_u64().unwrap(), e["held_out_loss"].as_f64().unwrap())).fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
    assert_eq!((out["checkpoint"].as_str(), out["best"]["step"].as_u64(), out["best"]["held_out_loss"].as_f64()), (Some("best"), Some(best_at), Some(best)), "{out}");
    assert_eq!(out["stopped_early"], true, "the held-out loss of a tiny model overfitting three utterances rises: {out}");
    let stopped = out["steps_completed"].as_u64().unwrap();
    assert!(stopped < 40 && curve.len() as u64 == stopped && stopped == best_at + 2, "stopped two evaluations past the best: {out}");

    let at_best = run(&rig, params(json!({"cards": 2, "steps": best_at, "lr": 0.2, "schedule": "constant"})), &blobs[..1], &Frames::default()).expect("a run to the best step").outputs;
    assert_eq!((out["checkpoint_digest"].as_str(), at_best["checkpoint"].as_str()), (at_best["checkpoint_digest"].as_str(), Some("last")), "the published checkpoint is the best step's");
    crate::support::assert_reservation_returned(&rig);
}
