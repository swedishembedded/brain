// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The same job over the bus: `Subscribe` with the dataset as a file descriptor, step frames on the stream,
//! the adapter back as a file descriptor, and `Cancel`. Needs a session bus; skips without one (run under
//! `dbus-run-session`).

use std::collections::HashMap;
use std::io::IoSliceMut;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

use nix::sys::socket::{recvmsg, ControlMessageOwned, MsgFlags};
use serde_json::{json, Value};
use zbus::OwnedFd as ZOwnedFd;

use crate::support::{plain, rig, rig_qwen3, rollouts, run_action, work_is_clean, Frames, BASE};

/// One datagram of the event stream: its JSON header and the descriptor it carried, if any.
fn next_frame(stream: &OwnedFd) -> Option<(Value, Option<OwnedFd>)> {
    let mut buffer = vec![0u8; 1 << 16];
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    loop {
        let mut cmsg = nix::cmsg_space!([RawFd; 1]);
        let mut iov = [IoSliceMut::new(&mut buffer)];
        // The stream is non-blocking at both ends: wait for the next frame.
        let got = match recvmsg::<()>(stream.as_raw_fd(), &mut iov, Some(&mut cmsg), MsgFlags::empty()) {
            Ok(got) => got,
            Err(nix::errno::Errno::EAGAIN) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }
            Err(_) => return None,
        };
        let fd = got.cmsgs().ok()?.find_map(|c| match c {
            ControlMessageOwned::ScmRights(fds) => fds.first().map(|&raw| unsafe { OwnedFd::from_raw_fd(raw) }),
            _ => None,
        });
        let bytes = got.bytes;
        if bytes == 0 {
            return None;
        }
        return Some((serde_json::from_slice(&buffer[..bytes]).expect("a JSON frame"), fd));
    }
}

fn memfd(bytes: &[u8]) -> ZOwnedFd {
    brain_dbus::fd::memfd_seal("train-in", bytes).unwrap().into()
}

fn session_bus_reachable() -> bool {
    let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return false };
    rt.block_on(zbus::Connection::session()).is_ok()
}

#[test]
fn a_job_over_the_bus_takes_its_dataset_as_a_descriptor_streams_its_steps_returns_the_adapter_as_one_and_can_be_cancelled() {
    if !session_bus_reachable() {
        brain_testutil::skip_unavailable("no reachable session bus (run under `dbus-run-session -- cargo test ...`)");
        return;
    }
    let rig = rig("dbus-job", 16 << 30, false);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let manager = brain_dbus::service::Manager::new(rig.exec.clone()).with_training(true);
        let name = format!("com.swedishembedded.Brain1.traintest{}", std::process::id());
        let _conn = zbus::connection::Builder::session().name(name.as_str()).serve_at(brain_dbus::OBJECT_PATH, manager).build().await.unwrap();
        let client = zbus::Connection::session().await.unwrap();
        let proxy = zbus::Proxy::new(&client, name.as_str(), brain_dbus::OBJECT_PATH, "com.swedishembedded.Brain1.Manager").await.unwrap();

        // The service is discoverable like any model, and offers a caller no path.
        let listed: String = proxy.call("Manifests", &()).await.unwrap();
        let manifests: Value = serde_json::from_str(&listed).unwrap();
        let train = manifests.as_array().unwrap().iter().find(|m| m["model"] == trainjob::MODEL).expect("brain/train is listed");
        assert_eq!(train["actions"][0]["name"], "train");

        // ---- a run whose dataset is a file descriptor ----
        let params = json!({"base": BASE, "arch": "qwen35", "objective": "policy_gradient", "behaviour": "start_of_run_proxy", "steps": 2, "grad_accum": 2, "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "return_adapter": true, "data": json!({"train": {"blob": "dataset"}}).to_string()});
        let mut fds: HashMap<String, ZOwnedFd> = HashMap::new();
        fds.insert("dataset".into(), memfd(&rollouts()));
        let (job, stream): (u64, ZOwnedFd) = proxy.call("Subscribe", &(trainjob::MODEL, trainjob::ACTION, params.to_string(), fds, r#"{"dataset":{"media":"bytes"}}"#)).await.unwrap();
        let stream: OwnedFd = stream.into();
        let (mut steps, mut blobs, mut done) = (Vec::new(), HashMap::new(), None);
        let stream = std::sync::Arc::new(stream);
        loop {
            let reader = stream.clone();
            let Some((frame, fd)) = tokio::task::spawn_blocking(move || next_frame(&reader)).await.unwrap() else { break };
            match frame["type"].as_str().unwrap() {
                "progress" if frame["event"]["type"] == "step" => steps.push(frame["event"].clone()),
                "blob" => {
                    blobs.insert(frame["name"].as_str().unwrap().to_string(), brain_dbus::fd::read_owned_to_vec(&fd.expect("a blob frame carries its descriptor")).unwrap());
                }
                "done" => {
                    done = Some(frame["result"].clone());
                    break;
                }
                "error" => panic!("the job failed: {frame}"),
                _ => {}
            }
        }
        let report = done.expect("a done frame");
        assert_eq!(report["status"], "completed");
        assert!(!steps.is_empty() && steps.iter().all(|s| s["pg"]["kl"].is_number()), "step frames carry the policy-gradient statistics: {steps:?}");
        assert!(!blobs["adapter"].is_empty() && !blobs["record"].is_empty(), "the adapter and its record came back as descriptors");
        assert_eq!(report["adapter_digest"], Value::Null, "this server has no artifact store");
        let _ = job;

        // ---- Cancel ends a run at its next step ----
        let params = json!({"base": BASE, "arch": "qwen35", "steps": 1000, "rank": 2, "seed": 7, "max_block": 128, "return_adapter": true, "data": json!({"train": {"blob": "dataset"}}).to_string()});
        let mut fds: HashMap<String, ZOwnedFd> = HashMap::new();
        fds.insert("dataset".into(), memfd(&plain()));
        let (job, stream): (u64, ZOwnedFd) = proxy.call("Subscribe", &(trainjob::MODEL, trainjob::ACTION, params.to_string(), fds, r#"{"dataset":{"media":"bytes"}}"#)).await.unwrap();
        let stream = std::sync::Arc::new(OwnedFd::from(stream));
        let mut cancelled = false;
        loop {
            let reader = stream.clone();
            let Some((frame, _)) = tokio::task::spawn_blocking(move || next_frame(&reader)).await.unwrap() else { panic!("the stream ended without a terminal frame") };
            match frame["type"].as_str().unwrap() {
                "progress" if frame["event"]["type"] == "step" && !cancelled => {
                    assert!(proxy.call::<_, _, bool>("Cancel", &(job,)).await.unwrap(), "the job is in flight");
                    cancelled = true;
                }
                "error" => {
                    assert_eq!(frame["message"], "cancelled");
                    break;
                }
                "done" => panic!("a cancelled run completed"),
                _ => {}
            }
        }
        assert!(cancelled);
    });
    assert!(work_is_clean(&rig.work_root));
}

#[test]
fn the_bus_refuses_a_job_that_names_what_only_the_host_may_choose() {
    if !session_bus_reachable() {
        brain_testutil::skip_unavailable("no reachable session bus (run under `dbus-run-session -- cargo test ...`)");
        return;
    }
    let rig = rig("dbus-refuse", 16 << 30, false);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let manager = brain_dbus::service::Manager::new(rig.exec.clone()).with_training(true);
        let name = format!("com.swedishembedded.Brain1.trainrefuse{}", std::process::id());
        let _conn = zbus::connection::Builder::session().name(name.as_str()).serve_at(brain_dbus::OBJECT_PATH, manager).build().await.unwrap();
        let client = zbus::Connection::session().await.unwrap();
        let proxy = zbus::Proxy::new(&client, name.as_str(), brain_dbus::OBJECT_PATH, "com.swedishembedded.Brain1.Manager").await.unwrap();
        for extra in ["out_dir", "weights", "path"] {
            let params = json!({"base": BASE, "arch": "qwen35", "data": json!({"train": {"blob": "dataset"}}).to_string(), extra: "../escape"});
            let mut fds: HashMap<String, ZOwnedFd> = HashMap::new();
            fds.insert("dataset".into(), memfd(&plain()));
            let refused = proxy.call::<_, _, (u64, ZOwnedFd)>("Subscribe", &(trainjob::MODEL, trainjob::ACTION, params.to_string(), fds, r#"{"dataset":{"media":"bytes"}}"#)).await;
            assert!(refused.unwrap_err().to_string().contains("unknown param"), "{extra}");
        }
    });
}

/// `check_dataset` over the bus's `Run`: the same per-record counts the executor returns, and the same typed refusal of
/// a page over the cap as over HTTP.
#[test]
fn check_dataset_lists_per_record_counts_over_the_bus_and_refuses_an_unbounded_page() {
    if !session_bus_reachable() {
        brain_testutil::skip_unavailable("no reachable session bus (run under `dbus-run-session -- cargo test ...`)");
        return;
    }
    let rig = rig_qwen3("dbus-check", 16 << 30);
    let mut dataset = plain();
    dataset.extend_from_slice(b"\nnot json\n");
    let params = json!({"base": BASE, "per_record": true});
    let direct = run_action(&rig, "check_dataset", &trainjob::dryrun::spec(), params.clone(), &[("dataset", dataset.clone())], &Frames::default()).unwrap().outputs;
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let manager = brain_dbus::service::Manager::new(rig.exec.clone()).with_training(true);
        let name = format!("com.swedishembedded.Brain1.checktest{}", std::process::id());
        let _conn = zbus::connection::Builder::session().name(name.as_str()).serve_at(brain_dbus::OBJECT_PATH, manager).build().await.unwrap();
        let client = zbus::Connection::session().await.unwrap();
        let proxy = zbus::Proxy::new(&client, name.as_str(), brain_dbus::OBJECT_PATH, "com.swedishembedded.Brain1.Manager").await.unwrap();
        let run = |params: Value| {
            let mut fds: HashMap<String, ZOwnedFd> = HashMap::new();
            fds.insert("dataset".into(), memfd(&dataset));
            let proxy = &proxy;
            async move { proxy.call::<_, _, (String, HashMap<String, ZOwnedFd>, String)>("Run", &(trainjob::MODEL, "check_dataset", params.to_string(), fds, r#"{"dataset":{"media":"bytes"}}"#, "memfd")).await }
        };
        let (result, _, _) = run(params).await.expect("the report");
        assert_eq!(serde_json::from_str::<Value>(&result).unwrap(), direct);
        let refused = run(json!({"base": BASE, "per_record": true, "limit": u64::MAX})).await.unwrap_err();
        assert!(refused.to_string().contains("limit"), "{refused}");
    });
}

/// `speech_sft` over the bus: the shard is a descriptor of `Subscribe`, the stages stream, the result is the
/// checkpoint's manifest digest, and a param naming a path is a bus error before any job exists.
#[test]
fn a_speech_fine_tune_over_the_bus_takes_its_shard_as_a_descriptor() {
    if !session_bus_reachable() {
        brain_testutil::skip_unavailable("no reachable session bus (run under `dbus-run-session -- cargo test ...`)");
        return;
    }
    use crate::speech::{shard, write_tts_base, LANGUAGES, TTS_BASE, VOICES};
    let rig = rig("dbus-speech", 16 << 30, true);
    write_tts_base(&rig.dir.join("models"));
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let manager = brain_dbus::service::Manager::new(rig.exec.clone()).with_training(true);
        let name = format!("com.swedishembedded.Brain1.speechtest{}", std::process::id());
        let _conn = zbus::connection::Builder::session().name(name.as_str()).serve_at(brain_dbus::OBJECT_PATH, manager).build().await.unwrap();
        let client = zbus::Connection::session().await.unwrap();
        let proxy = zbus::Proxy::new(&client, name.as_str(), brain_dbus::OBJECT_PATH, "com.swedishembedded.Brain1.Manager").await.unwrap();
        let params = json!({"base": TTS_BASE, "languages": LANGUAGES, "voices": VOICES, "steps": 1, "allow_cpu": true});

        let mut bad = params.clone();
        bad["out"] = json!("/etc");
        let mut fds: HashMap<String, ZOwnedFd> = HashMap::new();
        fds.insert("dataset".into(), memfd(&shard()));
        let refused = proxy.call::<_, _, (u64, ZOwnedFd)>("Subscribe", &(trainjob::MODEL, trainjob::speech::ACTION, bad.to_string(), fds, r#"{"dataset":{"media":"bytes"}}"#)).await;
        assert!(refused.is_err(), "a path param is refused before a job exists");

        let mut fds: HashMap<String, ZOwnedFd> = HashMap::new();
        fds.insert("dataset".into(), memfd(&shard()));
        let (_job, stream): (u64, ZOwnedFd) = proxy.call("Subscribe", &(trainjob::MODEL, trainjob::speech::ACTION, params.to_string(), fds, r#"{"dataset":{"media":"bytes"}}"#)).await.unwrap();
        let stream = std::sync::Arc::new(OwnedFd::from(stream));
        let (mut stages, mut done) = (0, None);
        loop {
            let reader = stream.clone();
            let Some((frame, _)) = tokio::task::spawn_blocking(move || next_frame(&reader)).await.unwrap() else { break };
            match frame["type"].as_str().unwrap() {
                "progress" if frame["event"]["type"] == "stage" => stages += 1,
                "done" => {
                    done = Some(frame["result"].clone());
                    break;
                }
                "error" => panic!("the job failed: {frame}"),
                _ => {}
            }
        }
        let report = done.expect("a done frame");
        assert!(stages > 0, "the stages stream");
        assert_eq!(report["status"], "completed", "{report}");
        assert!(trainjob::Digest::parse(report["checkpoint_digest"].as_str().unwrap_or_default()).is_ok(), "{report}");
    });
    assert!(work_is_clean(&rig.work_root));
}

/// The real 0.6B Base trained over the bus with the real objective on the CPU backend (a handful of steps on one
/// recording, encoded by the model's own `encode`), installed, then served and spoken in its new voice and language:
/// the release's trained tensors are the run's, not the base's. Needs a session bus, the checkpoint in the model store
/// and the voice-clone clip under testdata; slow (it trains a 0.6B on the CPU).
#[test]
#[ignore = "slow: trains the real 0.6B on the CPU backend"]
fn the_real_base_trains_over_the_bus_and_its_release_speaks_in_its_new_voice() {
    if !session_bus_reachable() {
        brain_testutil::skip_unavailable("no reachable session bus (run under `dbus-run-session -- cargo test ...`)");
        return;
    }
    let Some(real) = brain_testutil::model_dir("Qwen/Qwen3-TTS-12Hz-0.6B-Base").filter(|d| std::path::Path::new(d).join("model.safetensors").is_file()) else {
        brain_testutil::skip("Qwen/Qwen3-TTS-12Hz-0.6B-Base is not in the model store");
        return;
    };
    let Ok(clip) = std::fs::read(brain_testutil::testdata("tts/voice-clone-example-voice.wav")) else {
        brain_testutil::skip("testdata/tts/voice-clone-example-voice.wav absent (make testdata/fetch)");
        return;
    };
    // The shard: one recording as the served `encode` gives it.
    let paths = qwen3tts::TtsPaths::new(&real, real.as_str());
    let encoder = qwen3tts::encode::Encoder::load(&paths).unwrap();
    let text = "Det här är en mening på svenska.";
    let req = qwen3tts::encode::Request::parse(&clip, Some(text.to_string())).unwrap();
    let enc = encoder.encode(&req);
    let meta = qwen3tts::shard::ShardMeta { format: qwen3tts::shard::FORMAT.into(), codebooks: 16, codebook_size: 2048, xvector_dim: enc.xvector.len() as u32, frame_rate: 12.5, languages: vec!["swedish".into()], voices: vec!["astrid".into()], records: 0, producers: encoder.producers().clone() };
    let mut w = qwen3tts::shard::ShardWriter::new(meta);
    let id = qwen3tts::shard::RecordId { id: "u1".into(), speaker: "s1".into(), source: "test".into(), text: text.into() };
    w.push(&qwen3tts::shard::Record { id, text_ids: enc.text_ids.as_deref().unwrap(), codes: &enc.codes, xvector: &enc.xvector, voice: 0, language: 0, complete: true }).unwrap();
    let shard = w.to_bytes();

    let rig = rig("dbus-speech-real", 96 << 30, true);
    let vendor = rig.dir.join("models").join("Qwen");
    std::fs::create_dir_all(&vendor).unwrap();
    std::os::unix::fs::symlink(&real, vendor.join("Qwen3-TTS-12Hz-0.6B-Base")).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    let report = rt.block_on(async {
        let manager = brain_dbus::service::Manager::new(rig.exec.clone()).with_training(true);
        let name = format!("com.swedishembedded.Brain1.speechreal{}", std::process::id());
        let _conn = zbus::connection::Builder::session().name(name.as_str()).serve_at(brain_dbus::OBJECT_PATH, manager).build().await.unwrap();
        let client = zbus::Connection::session().await.unwrap();
        let proxy = zbus::Proxy::new(&client, name.as_str(), brain_dbus::OBJECT_PATH, "com.swedishembedded.Brain1.Manager").await.unwrap();
        let params = json!({"base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "languages": r#"{"swedish": {"new": true, "init_from": ["english", "german"]}}"#, "voices": r#"{"astrid": {"new": true}}"#,
                            "steps": 3, "batch_utterances": 1, "max_frames": 64, "lr": 1e-4, "new_row_lr_multiplier": 10.0, "seed": 3, "install_as": "sv-real", "allow_cpu": true});
        let mut fds: HashMap<String, ZOwnedFd> = HashMap::new();
        fds.insert("dataset".into(), memfd(&shard));
        let (_job, stream): (u64, ZOwnedFd) = proxy.call("Subscribe", &(trainjob::MODEL, trainjob::speech::ACTION, params.to_string(), fds, r#"{"dataset":{"media":"bytes"}}"#)).await.unwrap();
        let stream = std::sync::Arc::new(OwnedFd::from(stream));
        let (mut steps, mut done) = (Vec::new(), None);
        loop {
            let reader = stream.clone();
            let Some((frame, _)) = tokio::task::spawn_blocking(move || next_frame(&reader)).await.unwrap() else { break };
            match frame["type"].as_str().unwrap() {
                "progress" if frame["event"]["type"] == "step" => {
                    eprintln!("step {}", frame["event"]);
                    steps.push(frame["event"].clone());
                }
                "done" => {
                    done = Some(frame["result"].clone());
                    break;
                }
                "error" => panic!("the job failed: {frame}"),
                _ => {}
            }
        }
        assert_eq!(steps.len(), 3, "a frame per step");
        for step in &steps {
            for metric in ["loss", "talker_ce", "code_predictor_ce", "eos_accuracy", "grad_norm", "lr", "frames_per_s"] {
                assert!(step[metric].is_number(), "{metric}: {step}");
            }
        }
        done.expect("a done frame")
    });
    eprintln!("report {report}");
    assert!(report["trained_tensors"].as_u64().unwrap() > 0, "{report}");
    let installed = rig.dir.join("models").join("local").join("sv-real");
    let (before, after) = (checkpoint::mmap::MmapSafetensors::open(std::path::Path::new(&real).join("model.safetensors")).unwrap(), checkpoint::mmap::MmapSafetensors::open(installed.join("model.safetensors")).unwrap());
    let head = "talker.codec_head.weight";
    assert_ne!(before.raw_range(head, 0..before.nbytes(head).unwrap() as usize), after.raw_range(head, 0..after.nbytes(head).unwrap() as usize), "the release carries the trained weights");

    let served = catalog::resident_tts::TtsResident::all_from_store(&loader::served::ServedStore::new(Some(rig.dir.join("models"))));
    let sv = served.iter().find(|r| residency::ResidentModel::manifest(*r).model == "local/sv-real").expect("the release is served under its own id");
    let inv = capability::Invocation::new().set("text", json!("Hej, det här är ett test.")).set("speaker", json!("astrid")).set("lang", json!("swedish")).set("max_frames", json!(8)).set("seed", json!(1));
    let mut inst = residency::ResidentModel::activate(sv, &residency::ResidentModel::instance_key(sv, "design", &inv), residency::Device::Cpu).unwrap();
    let spoken = inst.run("design", &inv, &mut |_| {}).unwrap();
    assert!(!audio::wav::parse(&spoken.blobs["audio"].bytes).unwrap().samples.is_empty(), "it speaks in its new voice and language");
}

/// A streaming job over the bus: `Subscribe` creates it, `Input` hands it a group and then `finish`, its receipts and
/// published version ride the event stream, and `Input` to a job that is gone (or another user's) is refused.
#[test]
fn a_stream_job_over_the_bus_takes_its_records_by_input_and_finishes() {
    if !session_bus_reachable() {
        brain_testutil::skip_unavailable("no reachable session bus (run under `dbus-run-session -- cargo test ...`)");
        return;
    }
    let rig = crate::support::rig_stream("dbus-stream");
    let sizes = crate::stream::sizes(&rig);
    let group = crate::stream::group(sizes, "g1", 0);
    let tokens = (sizes[0] + sizes[1]) as u64;
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let manager = brain_dbus::service::Manager::new(rig.exec.clone()).with_training(true);
        let name = format!("com.swedishembedded.Brain1.trainstream{}", std::process::id());
        let _conn = zbus::connection::Builder::session().name(name.as_str()).serve_at(brain_dbus::OBJECT_PATH, manager).build().await.unwrap();
        let client = zbus::Connection::session().await.unwrap();
        let proxy = zbus::Proxy::new(&client, name.as_str(), brain_dbus::OBJECT_PATH, "com.swedishembedded.Brain1.Manager").await.unwrap();
        let stream = json!({"publish_adapter": crate::stream::ADAPTER, "publish_every": 1, "step_tokens": tokens, "queue_tokens": tokens});
        let params = json!({"base": BASE, "objective": "policy_gradient", "loss_agg": "token-sum", "rank": 2, "lr": 2e-2, "seed": 7, "max_block": 128, "stream": stream.to_string()});
        let (job, events): (u64, ZOwnedFd) = proxy.call("Subscribe", &(trainjob::MODEL, trainjob::ACTION, params.to_string(), HashMap::<String, ZOwnedFd>::new(), "{}")).await.unwrap();
        let input = |kind: &'static str, body: Vec<u8>| {
            let proxy = &proxy;
            async move { proxy.call::<_, _, String>("Input", &(job, kind, "{}", memfd(&body))).await.map(|reply| serde_json::from_str::<Value>(&reply).unwrap()) }
        };
        let accepted = input("records", group.clone()).await.unwrap();
        assert_eq!((accepted["accepted"].clone(), accepted["answer"]["accepted"].clone()), (json!(true), json!(["g1"])), "{accepted}");
        let refused = input("records", b"not json".to_vec()).await.unwrap();
        assert_eq!((refused["accepted"].clone(), refused["refusal"].clone()), (json!(false), json!("invalid")), "{refused}");
        assert_eq!(input("finish", Vec::new()).await.unwrap()["answer"]["finishing"], json!(true));
        let events = std::sync::Arc::new(OwnedFd::from(events));
        let mut kinds = Vec::new();
        loop {
            let reader = events.clone();
            let Some((frame, _)) = tokio::task::spawn_blocking(move || next_frame(&reader)).await.unwrap() else { panic!("the stream ended without a terminal frame") };
            match frame["type"].as_str().unwrap() {
                "progress" => kinds.push(frame["event"]["type"].as_str().unwrap_or_default().to_string()),
                "done" => break,
                "error" => panic!("the job failed: {frame}"),
                _ => {}
            }
        }
        for kind in ["stream", "applied", "published", "step", "committed"] {
            assert!(kinds.iter().any(|k| k == kind), "{kind} in {kinds:?}");
        }
        let gone = proxy.call::<_, _, String>("Input", &(job, "records", "{}", memfd(&group))).await;
        assert!(gone.unwrap_err().to_string().contains("no such job"));
    });
}
