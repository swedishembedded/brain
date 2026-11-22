// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving models and their
// adapters from node-local copies on training and serving clusters for its
// clients. If your team needs expertise in distributing checkpoints across a
// cluster then you can procure our services by sending an email to
// info@swedishembedded.com.

//! `--stage-models DIR` answers a store path from a node-local copy of its
//! entry. The copy is the same model as the store entry: a server whose
//! `--checkpoint` names the copy serves that base once, under the same id, and
//! offers the adapters of `--serve-adapters` on it exactly as it does without
//! staging.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use brain_testutil::Scratch;

const BASE_ID: &str = "acme/tiny";
const ADAPTER_ID: &str = "acme/tiny:acme:support:v1";

/// A store holding one Qwen3 base entry (`acme/tiny`) and, beside it, a directory with one adapter trained on it.
fn store_with_base_and_adapter(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let models = dir.join("models");
    let entry = models.join(BASE_ID);
    std::fs::create_dir_all(&entry).unwrap();
    let weights = entry.join("model.brain.safetensors");
    let card = checkpoint::st::ModelCard::new(BASE_ID, "qwen");
    checkpoint::st::save_safetensors(weights.to_str().unwrap(), &[("embed.weight".to_string(), vec![4], vec![1.0; 4])], &serde_json::json!({}), Some(&card)).unwrap();
    let adapters = dir.join("adapters");
    std::fs::create_dir_all(&adapters).unwrap();
    let digest = brain_modelstore::fetch::base_digest(&weights).unwrap();
    brain_testutil::adapters::write_adapter(&adapters.join("support.safetensors"), ADAPTER_ID, BASE_ID, Some(digest), 0.5);
    (models, entry, adapters)
}

/// `brain serve` over `models`, `--checkpoint` naming the store entry, every adapter of `adapters` offered; with
/// `stage` the entry is answered from a copy under it.
fn serve(dir: &Path, models: &Path, entry: &Path, adapters: &Path, stage: Option<&Path>, run: &Path) -> Child {
    let mut cmd = Command::new(brain_testutil::package_bin("brain"));
    cmd.args(["--device", "cpu"]);
    if let Some(stage) = stage {
        cmd.arg("--stage-models").arg(stage).arg("--stage-from").arg(models);
    }
    cmd.args(["serve", "--seed", "1", "--openai", "127.0.0.1:0", "--models-dir"])
        .arg(models)
        .arg("--checkpoint")
        .arg(entry)
        .arg("--serve-adapters")
        .arg(adapters)
        .arg("--runtime-dir")
        .arg(run)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        // Nothing of the caller's environment reaches the server: what it serves comes from its flags alone.
        .env_clear()
        .env("HOME", dir.join("home"));
    cmd.spawn().expect("spawn brain serve")
}

/// The URL and key of the one HTTP surface `child` publishes in `run`, once it listens.
fn wait_for_surface(child: &mut Child, run: &Path) -> (String, String) {
    let record = run.join(format!("serve-{}.json", child.id()));
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if let Some(value) = std::fs::read_to_string(&record).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) {
            let surface = &value["surfaces"][0];
            return (surface["url"].as_str().unwrap().to_string(), surface["key"].as_str().unwrap().to_string());
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("brain serve exited with {status} before listening; stderr:\n{}", stop(child));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("brain serve never published {}; stderr:\n{}", record.display(), stop(child));
}

/// The ids `/v1/models` lists, sorted.
fn listed_models(url: &str, key: &str) -> Vec<String> {
    let host = url.strip_prefix("http://").unwrap().split('/').next().unwrap();
    let mut stream = std::net::TcpStream::connect(host).expect("connect to the published URL");
    stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    write!(stream, "GET /v1/models HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {key}\r\nConnection: close\r\n\r\n").unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).ok();
    let body = response.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or_default();
    let value: serde_json::Value = serde_json::from_str(body).unwrap_or_else(|e| panic!("/v1/models answered no JSON ({e}): {response}"));
    let mut ids: Vec<String> = value["data"].as_array().expect("a data array").iter().map(|m| m["id"].as_str().unwrap().to_string()).collect();
    ids.sort();
    ids
}

/// Stop `child` and return what it wrote on stderr.
fn stop(child: &mut Child) -> String {
    // SAFETY: signalling our own child by pid.
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    let deadline = Instant::now() + Duration::from_secs(60);
    while child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    child.kill().ok();
    child.wait().ok();
    let mut stderr = String::new();
    child.stderr.take().map(|mut s| s.read_to_string(&mut stderr));
    stderr
}

/// What a server lists over the fixture, and its stderr.
fn served(dir: &Path, stage: Option<&Path>, run: &str) -> (Vec<String>, String) {
    let (models, entry, adapters) = (dir.join("models"), dir.join("models").join(BASE_ID), dir.join("adapters"));
    let run = dir.join(run);
    let mut child = serve(dir, &models, &entry, &adapters, stage, &run);
    let (url, key) = wait_for_surface(&mut child, &run);
    let listed = listed_models(&url, &key);
    (listed, stop(&mut child))
}

/// The cluster regression: with staging on, the base was registered once as the copy and once as the store
/// entry, and every adapter was refused because its base was "served several times".
#[test]
fn a_staged_checkpoint_is_the_store_entry_it_copies() {
    let dir = Scratch::dir("brain-serve-staged");
    std::fs::create_dir_all(dir.join("home")).unwrap();
    store_with_base_and_adapter(&dir);
    let stage = dir.join("stage");

    let (plain, plain_err) = served(&dir, None, "run-plain");
    assert!(plain.iter().any(|id| id == ADAPTER_ID), "the adapter is offered without staging: {plain:?}; stderr:\n{plain_err}");

    let (staged, staged_err) = served(&dir, Some(&stage), "run-staged");
    assert!(stage.join(BASE_ID).join("model.brain.safetensors").is_file(), "the entry was staged; stderr:\n{staged_err}");
    assert!(!staged_err.contains("not offering the adapter"), "the adapter is refused with staging on; stderr:\n{staged_err}");
    assert_eq!(staged.iter().filter(|id| id.as_str() == BASE_ID).count(), 1, "one served base: {staged:?}");
    assert_eq!(staged, plain, "/v1/models is the same with and without staging; staged stderr:\n{staged_err}");
}
