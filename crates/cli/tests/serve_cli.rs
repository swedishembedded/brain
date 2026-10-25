// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Process-level tests for `brain serve`'s flag parsing: unknown flags must
//! be a hard error (not the old warn-and-continue), and `--help` must
//! actually print usage instead of falling into the blocking stdio loop.
//! `brain run` used to be an alias for this same command; it is now freed up
//! (see `the_former_run_alias_is_no_longer_a_recognized_command` below).
//!
//! Regression coverage for an incident where `brain serve --listen HOST:PORT`
//! (a flag that never existed) used to be silently ignored, exit 0, and
//! never open a listener.
//!
//! IMPORTANT: every test here passes `.stdin(Stdio::null())`. Without it, a
//! regression back to warn-and-continue falls into the blocking stdio JSONL
//! loop, which reads from whatever stdin `cargo test` gave the child — this is
//! what turns a silent regression into a LOUD, fast test failure (wrong exit
//! code) instead of either a hang or a flaky pass, since `Stdio::null()`
//! guarantees stdin reads EOF immediately either way.

use std::process::{Command, Stdio};

fn bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("brain");
    path.to_string_lossy().into_owned()
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(bin())
        .args(args)
        .stdin(Stdio::null())
        .env("BRAIN_DEVICE", "cpu")
        // These tests pin the CPU device. An ambient backend choice (a CUDA
        // lane exporting BRAIN_BACKEND) would make the binary warn that the
        // backend needs a GPU, which is not what is under test.
        .env_remove("BRAIN_BACKEND")
        .output()
        .unwrap_or_else(|e| panic!("run brain {args:?}: {e}"))
}

#[test]
fn unknown_flag_is_a_hard_error_with_usage() {
    // The exact flag from the bench-integration-friction incident.
    let out = run(&["serve", "--listen", "0.0.0.0:8788"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown flag"), "stderr: {stderr}");
    assert!(stderr.contains("--listen"), "stderr: {stderr}");
    // The usage text (printed on the error path) must actually help the reader
    // find the real flag.
    assert!(stderr.contains("--openai"), "stderr: {stderr}");
}

/// `brain run` used to be an alias for `brain serve`; the stdio controller it
/// selected by default is now reached explicitly via `brain serve --stdio`
/// (see `run_cli`'s module doc), which frees "run" to mean nothing special --
/// it is not a verb any architecture recognizes and not an architecture id,
/// so it falls through to the generic "unknown command" path.
#[test]
fn the_former_run_alias_is_no_longer_a_recognized_command() {
    let out = run(&["run", "--nope"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown command 'run'"));
}

#[test]
fn non_numeric_port_errors_instead_of_silently_defaulting() {
    // Before this change, a non-numeric token after --openai was left
    // unconsumed by take_port, silently fell through with the default port
    // 8788 bound, and "foo" was then dropped on the floor by the old
    // warn-and-continue arm.
    let out = run(&["serve", "--openai", "foo"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("foo"));
}

#[test]
fn out_of_range_port_errors() {
    let out = run(&["serve", "--openai", "99999"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn bare_positional_is_an_error() {
    let out = run(&["serve", "models.safetensors"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn value_taking_flag_with_no_value_errors() {
    let out = run(&["serve", "--models-dir"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--models-dir needs a value"));
}

/// The admission deadlines are flags (the environment variables they replaced
/// are no longer read). A value that is not a positive number of milliseconds
/// is a typo, not a request for the default and never an unbounded wait.
#[test]
fn admission_deadline_flags_reject_values_that_are_not_positive_milliseconds() {
    for flag in ["--admit-deadline-ms", "--cold-build-admit-deadline-ms"] {
        for bad in ["0", "soon", "-5"] {
            let out = run(&["serve", flag, bad]);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert_eq!(out.status.code(), Some(2), "{flag} {bad}: stderr: {stderr}");
            assert!(!stderr.contains("unknown flag"), "{flag} must be a known flag: {stderr}");
            assert!(stderr.contains(flag) && stderr.contains("positive"), "{flag} {bad}: the error names the flag and the rule: {stderr}");
        }
    }
}

#[test]
fn serve_help_exits_zero_with_usage_on_stdout() {
    for flag in ["--help", "-h"] {
        let out = run(&["serve", flag]);
        assert!(out.status.success(), "brain serve {flag}: {}", String::from_utf8_lossy(&out.stderr));
        assert!(out.stderr.is_empty(), "stderr should be empty for {flag}: {}", String::from_utf8_lossy(&out.stderr));
        let stdout = String::from_utf8_lossy(&out.stdout);
        for f in ["--openai", "--anthropic", "--openrouter", "--dbus", "--models-dir", "--api-keys-out", "--reserve-gb", "--ready-file"] {
            assert!(stdout.contains(f), "brain serve {flag} stdout missing {f}:\n{stdout}");
        }
    }
}

/// `--stdio` is the explicit spelling of the event-driven controller `brain
/// serve` used to fall into implicitly whenever no surface flag was given;
/// it must be a recognized flag, not "unknown flag --stdio".
#[test]
fn stdio_flag_is_recognized_and_still_reaches_help() {
    let out = run(&["serve", "--stdio", "--help"]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("--openai"));
}

/// `--status` is a query, so its answer has to be in the exit code: a script
/// that asks "is a server running?" should not have to grep English prose.
/// This is the `systemctl is-active` convention (3 = inactive), which is why
/// the script driving a server can write `if brain serve --status; then`.
#[test]
fn status_exits_nonzero_when_no_server_is_running() {
    let state = std::env::temp_dir().join(format!("brain-status-test-{}", std::process::id()));
    std::fs::create_dir_all(&state).unwrap();

    let out = Command::new(bin())
        .args(["serve", "--status"])
        .stdin(Stdio::null())
        .env("BRAIN_DEVICE", "cpu")
        .env("XDG_RUNTIME_DIR", &state)
        .output()
        .expect("run brain serve --status");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("not running"), "stdout: {stdout}");
    assert_eq!(
        out.status.code(),
        Some(3),
        "--status must report absence through the exit code; stdout: {stdout}"
    );

    std::fs::remove_dir_all(&state).ok();
}

/// The lifecycle flags work whatever global flag stands before `serve`, and
/// answer from the directory `--runtime-dir` names.
#[test]
fn status_reads_the_runtime_dir_flag_after_a_leading_global_flag() {
    let dir = Scratch::new("status-runtime-dir");
    let out = Command::new(bin())
        .args(["--device", "cpu", "serve", "--status", "--runtime-dir"])
        .arg(&dir.0)
        .stdin(Stdio::null())
        .env_remove("BRAIN_DEVICE")
        .output()
        .expect("run brain serve --status");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(3), "stdout: {stdout}; stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("not running"), "stdout: {stdout}");
}

/// A scratch directory for one test, removed when it ends.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("brain-serve-cli-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// The card id a trainer records: `<vendor>/<repo>:<owner>:<name>:<tag>`.
const ADAPTER_ID: &str = "brain/qwen3:acme:support:v3";

/// A stand-in base file (startup verification reads only its bytes) and a
/// LoRA adapter whose card records the digest of `trained_on` (default: that
/// base) as the base it was trained against.
fn write_base_and_adapter(dir: &std::path::Path, fill: f32, trained_on: Option<&std::path::Path>) -> (std::path::PathBuf, std::path::PathBuf) {
    write_named_adapter(dir, fill, trained_on, ADAPTER_ID)
}

/// [`write_base_and_adapter`] with the adapter card's id chosen by the test.
fn write_named_adapter(dir: &std::path::Path, fill: f32, trained_on: Option<&std::path::Path>, id: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let base = dir.join(format!("base-{fill}.safetensors"));
    checkpoint::st::save_safetensors(base.to_str().unwrap(), &[("embed.weight".to_string(), vec![4], vec![fill; 4])], &serde_json::json!({}), None).unwrap();
    let adapter = dir.join("adapter.safetensors");
    let mut card = checkpoint::st::ModelCard::new(id, "qwen");
    card.adapter = Some(checkpoint::st::Adapter { kind: "lora".to_string(), rank: Some(2), base: Some("local/base".to_string()), alpha: Some(4.0), ..Default::default() });
    card.training = Some(checkpoint::st::TrainingProvenance {
        code_revision: "test".to_string(),
        regime: "sft_lora".to_string(),
        seed: data::rng::random_seed(),
        hyperparams: serde_json::Value::Null,
        environment: "cpu".to_string(),
        gate: None,
        trained_from: None,
        base_digest: Some(brain_modelstore::fetch::file_digest(trained_on.unwrap_or(&base)).unwrap()),
        cycle: 0,
    });
    let tensors = vec![("blocks.0.attn.wq.weight.lora_a".to_string(), vec![2, 4], vec![0.5; 8]), ("blocks.0.attn.wq.weight.lora_b".to_string(), vec![4, 2], vec![0.5; 8])];
    checkpoint::st::save_safetensors(adapter.to_str().unwrap(), &tensors, &serde_json::json!({"rank": 2}), Some(&card)).unwrap();
    (base, adapter)
}

fn serve_command(dir: &std::path::Path, base: &std::path::Path, extra: &[&str]) -> Command {
    let models = dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let mut cmd = Command::new(bin());
    cmd.args(["serve", "--openai", "127.0.0.1:0", "--seed", "1", "--models-dir", models.to_str().unwrap(), "--checkpoint", base.to_str().unwrap()])
        .arg("--runtime-dir")
        .arg(dir.join("run"))
        .args(extra)
        .stdin(Stdio::null())
        .env("BRAIN_DEVICE", "cpu")
        .env_remove("BRAIN_AUTO_FETCH");
    cmd
}

/// `--adapter FILE` pins one release: the startup line names it by the
/// digest a fine-tune reports for that file.
#[test]
fn a_pinned_adapter_is_reported_by_digest_at_startup() {
    let dir = Scratch::new("pin");
    let (base, adapter) = write_base_and_adapter(&dir.0, 1.0, None);
    let digest = brain_modelstore::fetch::file_digest(&adapter).unwrap();
    let mut child = serve_command(&dir.0, &base, &["--adapter", adapter.to_str().unwrap()]).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().expect("spawn brain serve");

    let expected = format!("brain serve: {ADAPTER_ID} adapter={ADAPTER_ID} digest={digest}");
    let (tx, rx) = std::sync::mpsc::channel();
    let stderr = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        use std::io::BufRead;
        for line in std::io::BufReader::new(stderr).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let mut seen = Vec::new();
    let found = loop {
        match rx.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now())) {
            Ok(line) if line == expected => break true,
            Ok(line) => seen.push(line),
            Err(_) => break false,
        }
    };
    // Exit status before the kill: a server that died on its own says how.
    let exited = child.try_wait().ok().flatten();
    child.kill().ok();
    child.wait().ok();
    assert!(found, "expected {expected:?} on stderr; server exit status {exited:?}; saw:\n{}", seen.join("\n"));
}

/// The checkpoint is named by `--checkpoint` alone: `BRAIN_QWEN_WEIGHTS` in
/// the environment selects nothing, so an adapter with no flag is refused.
#[test]
fn the_environment_names_no_checkpoint() {
    let dir = Scratch::new("pin-env");
    let (base, adapter) = write_base_and_adapter(&dir.0, 1.0, None);
    let models = dir.0.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let run_dir = dir.0.join("run");
    let out = Command::new(bin())
        .args(["serve", "--openai", "127.0.0.1:0", "--seed", "1", "--models-dir", models.to_str().unwrap(), "--runtime-dir", run_dir.to_str().unwrap(), "--adapter", adapter.to_str().unwrap()])
        .stdin(Stdio::null())
        .env("BRAIN_DEVICE", "cpu")
        .env("BRAIN_QWEN_WEIGHTS", &base)
        .env_remove("BRAIN_AUTO_FETCH")
        .output()
        .expect("run brain serve");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the variable must not select the checkpoint; stderr:\n{stderr}");
    assert!(stderr.contains("--checkpoint"), "the remedy is the flag; stderr:\n{stderr}");
}

/// With no flag and no variable, `--adapter` says how to name the checkpoint,
/// by the flag.
#[test]
fn an_adapter_with_no_checkpoint_names_the_flag_that_selects_one() {
    let dir = Scratch::new("pin-no-base");
    let (_, adapter) = write_base_and_adapter(&dir.0, 1.0, None);
    let models = dir.0.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let out = Command::new(bin())
        .args(["serve", "--openai", "127.0.0.1:0", "--seed", "1", "--models-dir", models.to_str().unwrap(), "--adapter", adapter.to_str().unwrap()])
        .arg("--runtime-dir")
        .arg(dir.0.join("run"))
        .stdin(Stdio::null())
        .env("BRAIN_DEVICE", "cpu")
        .env_remove("BRAIN_QWEN_WEIGHTS")
        .env_remove("BRAIN_AUTO_FETCH")
        .output()
        .expect("run brain serve");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "no checkpoint must not serve; stderr:\n{stderr}");
    assert!(stderr.contains("--checkpoint"), "the remedy is the flag; stderr:\n{stderr}");
}

/// An adapter whose card id is not an adapter model reference is refused
/// at startup with that reason on stderr and a non-zero exit, never silently.
#[test]
fn a_pinned_adapter_with_an_unusable_id_is_refused_with_its_reason() {
    let dir = Scratch::new("pin-bad-id");
    let (base, adapter) = write_named_adapter(&dir.0, 1.0, None, "support:v3");
    let out = serve_command(&dir.0, &base, &["--adapter", adapter.to_str().unwrap()]).output().expect("run brain serve");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "an unusable adapter id must not serve; stderr:\n{stderr}");
    assert!(stderr.contains("support:v3") && stderr.contains("not a model reference"), "stderr must say why; got:\n{stderr}");
}

/// An adapter trained against another base is refused before anything
/// binds, naming both digests.
#[test]
fn a_pinned_adapter_for_another_base_is_a_startup_error() {
    let dir = Scratch::new("pin-mismatch");
    let trained_on = dir.0.join("trained-on.safetensors");
    checkpoint::st::save_safetensors(trained_on.to_str().unwrap(), &[("embed.weight".to_string(), vec![4], vec![9.0; 4])], &serde_json::json!({}), None).unwrap();
    let (base, adapter) = write_base_and_adapter(&dir.0, 1.0, Some(&trained_on));
    let out = serve_command(&dir.0, &base, &["--adapter", adapter.to_str().unwrap()]).output().expect("run brain serve");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a mismatched base must not serve; stderr:\n{stderr}");
    for file in [&trained_on, &base] {
        let digest = brain_modelstore::fetch::file_digest(file).unwrap();
        assert!(stderr.contains(&digest), "the error must name {digest}; stderr:\n{stderr}");
    }
}

/// A Qwen3.8 adapter (card family `qwen35`) is not a Qwen3 adapter: pinned on
/// a Qwen3 base it is refused by family, with both families named, never
/// loaded into a bank that cannot hold its linears.
#[test]
fn a_pinned_qwen35_adapter_is_refused_on_a_qwen3_base_by_family() {
    let dir = Scratch::new("pin-qwen35");
    let (base, _) = write_base_and_adapter(&dir.0, 1.0, None);
    let adapter = dir.0.join("qwen35-adapter.safetensors");
    let linears = [brain_testutil::adapters::Linear { base: "blocks.0.linear_attn.in_proj_qkv.weight".to_string(), out: 4, inn: 2 }];
    brain_testutil::adapters::write_lora_adapter_of("qwen35", &adapter, ADAPTER_ID, "local/base", 2, &linears, 1);
    let out = serve_command(&dir.0, &base, &["--adapter", adapter.to_str().unwrap()]).output().expect("run brain serve");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a Qwen3.8 adapter must not serve on a Qwen3 base; stderr:\n{stderr}");
    assert!(stderr.contains("qwen35") && stderr.contains("qwen"), "stderr must name both families; got:\n{stderr}");
}

/// A pinned adapter cannot be combined with a mode that would replace it,
/// nor given where no Qwen3 is served.
#[test]
fn a_pinned_adapter_cannot_also_follow_releases() {
    let out = run(&["serve", "--adapter", "a.safetensors", "--watch-adapters", "adapters/"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let out = run(&["serve", "--adapter", "a.safetensors", "--adapter-manifest", "release.json"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    // The stdio controller serves no Qwen3: the flag would be ignored.
    let out = run(&["serve", "--stdio", "--adapter", "a.safetensors"]);
    assert_eq!(out.status.code(), Some(2), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

/// `--models-dir` is the one answer to "which store" for everything `brain
/// serve` resolves at startup, not only its own catalog scan: every
/// architecture resolver (qwen35, FLUX.2, the catalog residents, the imaging
/// pipeline's stages, ...) must scan the flag's directory, never the store
/// `BRAIN_MODELS_DIR` names. A resolver that scanned the environment's store
/// instead would, on a real machine, sit in startup inventorying a
/// multi-terabyte store the operator explicitly pointed away from.
///
/// Observable through the inventory scan's own on-disk cache, which every
/// scan writes into the directory it scanned: the environment's store must
/// come out of startup untouched, while the flag's store must have been
/// scanned (the positive control that proves the resolvers ran at all).
#[test]
fn every_startup_resolver_scans_the_models_dir_flag_not_the_environment_store() {
    let dir = Scratch::new("models-dir-flag");
    let flag_store = dir.0.join("flag-store");
    let env_store = dir.0.join("env-store");
    let home = dir.0.join("home");
    for d in [&flag_store, &env_store, &home] {
        std::fs::create_dir_all(d).unwrap();
    }
    let ready = dir.0.join("ready");
    let mut child = Command::new(bin())
        .args(["serve", "--openai", "127.0.0.1:0", "--models-dir", flag_store.to_str().unwrap(), "--ready-file", ready.to_str().unwrap()])
        .arg("--runtime-dir")
        .arg(dir.0.join("run"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .env("BRAIN_DEVICE", "cpu")
        .env("BRAIN_MODELS_DIR", &env_store)
        .env("HOME", &home)
        .env_remove("XDG_DATA_HOME")
        .env_remove("BRAIN_AUTO_FETCH")
        .spawn()
        .expect("spawn brain serve");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while !ready.exists() && std::time::Instant::now() < deadline {
        if child.try_wait().ok().flatten().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let bound = ready.exists();
    child.kill().ok();
    let out = child.wait_with_output().expect("collect brain serve output");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(bound, "brain serve never became ready; stderr:\n{stderr}");

    let cache = ".brain-inventory.json";
    assert!(
        !env_store.join(cache).exists(),
        "a startup resolver scanned the BRAIN_MODELS_DIR store although --models-dir named another; stderr:\n{stderr}"
    );
    assert!(flag_store.join(cache).exists(), "no startup resolver scanned the --models-dir store; stderr:\n{stderr}");
}

/// `--exact-decode` is a model-agnostic serving policy switch: accepted by the
/// parser, documented in `--help`, and reported as part of the policy line.
/// (Which families honour it is their own declaration; the Qwen3 families that
/// do not are refused by name when a model is actually served.)
#[test]
fn exact_decode_is_a_serving_policy_switch_and_appears_in_help() {
    let out = run(&["serve", "--help"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--exact-decode"), "brain serve --help must document --exact-decode:\n{stdout}");

    let out = run(&["serve", "--exact-decode", "--help"]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).is_empty(), "--exact-decode must be a known flag: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn a_bad_serving_policy_value_names_its_flag_and_exits_2() {
    for (flag, value) in [("--kv-cache", "fp4"), ("--gemm", "tf32"), ("--weight-format", "q9"), ("--ctx", "many")] {
        let out = run(&["serve", flag, value]);
        assert_eq!(out.status.code(), Some(2), "{flag} {value}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(flag), "{flag} {value}: stderr:\n{stderr}");
    }
}

#[test]
fn a_policy_no_served_family_can_honour_is_refused_naming_the_families() {
    // The Qwen3 engine has no nf4 weights and the Qwen3.8 resident has no --gemm.
    let out = run(&["serve", "--weight-format", "nf4", "--gemm", "fp32"]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("qwen3") && stderr.contains("qwen35") && stderr.contains("supported"), "stderr:\n{stderr}");
}

#[test]
fn a_deprecated_architecture_prefixed_flag_still_works_and_says_what_replaces_it() {
    let out = run(&["serve", "--qwen-kv-fp8", "--help"]);
    // --help ends the parse after the alias was applied and warned about.
    assert_eq!(out.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--qwen-kv-fp8 is deprecated") && stderr.contains("--kv-cache fp8"), "stderr:\n{stderr}");
}
