// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-serving setups that explain what is
// wrong before a model is loaded, for its clients. If your team needs
// expertise in making local inference deployments diagnosable, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Process-level tests for `brain doctor`: the real binary, a scratch model
//! store, no weights and no GPU needed. A check that fails exits 1 and names
//! what to run; a usage error exits 2; `--json` prints the same report. Every
//! test names its store with `--models-dir`, which outranks the environment.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-doctor-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn doctor(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_brain"))
        .arg("doctor")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("run brain doctor {args:?}: {e}"));
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn an_empty_store_warns_how_to_fill_it_and_does_not_stop_a_server_from_starting() {
    let store = scratch("empty");
    let (code, out, err) = doctor(&["--models-dir", store.to_str().unwrap()]);
    assert_eq!(code, 0, "{out}\n{err}");
    assert!(out.contains("warn store:") && out.contains("brain pull"), "{out}");
    assert!(out.contains("ready: brain can serve here"), "{out}");
    std::fs::remove_dir_all(&store).ok();
}

#[test]
fn a_store_that_is_not_there_fails_naming_the_flag_and_exits_1() {
    let missing = std::env::temp_dir().join(format!("brain-doctor-cli-missing-{}", std::process::id()));
    let (code, out, _) = doctor(&["--models-dir", missing.to_str().unwrap()]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("FAIL store:") && out.contains("--models-dir"), "{out}");
    assert!(out.contains("not ready"), "{out}");
}

#[test]
fn a_model_that_is_not_pulled_fails_with_the_command_that_fetches_it() {
    let store = scratch("unpulled");
    let (code, out, _) = doctor(&["--models-dir", store.to_str().unwrap(), "--model", "Qwen/Qwen3-0.6B"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("FAIL model:") && out.contains("brain pull Qwen/Qwen3-0.6B"), "{out}");
    std::fs::remove_dir_all(&store).ok();
}

#[test]
fn json_prints_the_same_report_for_a_script() {
    let missing = std::env::temp_dir().join(format!("brain-doctor-cli-json-{}", std::process::id()));
    let (code, out, _) = doctor(&["--models-dir", missing.to_str().unwrap(), "--json"]);
    assert_eq!(code, 1);
    let report: serde_json::Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("not JSON ({e}): {out}"));
    assert_eq!(report["ready"], false);
    assert_eq!(report["checks"][0]["name"], "store");
    assert_eq!(report["checks"][0]["verdict"], "fail");
    assert!(report["checks"][0]["fix"].as_str().unwrap().contains("--models-dir"));
}

#[test]
fn a_stray_argument_is_a_usage_error() {
    let (code, _, err) = doctor(&["stray"]);
    assert_eq!(code, 2);
    assert!(err.contains("usage: brain doctor"), "{err}");
}
