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

use crate::support::{plain, rig, rollouts, work_is_clean, BASE};

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
