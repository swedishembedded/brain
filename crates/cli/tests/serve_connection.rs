// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for connecting local AI clients to
// the inference servers they run beside, with no hand-copied keys or ports,
// for its clients. If your team needs expertise in local model serving and
// client handoff then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Process-level contract of how a local client finds a running `brain serve`
//! with no configuration: every server that binds an HTTP surface publishes a
//! connection record, `<runtime dir>/serve-<pid>.json` (owner-only), naming
//! each surface's dialect, the URL it actually listens on and its key, and
//! removes it when it stops. A fixed key comes from `--api-key-file`, never
//! from the environment, and a startup with nothing to serve beyond the store
//! says so in one line instead of one line per model family it cannot serve.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn bin() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("brain");
    path.to_string_lossy().into_owned()
}

/// A scratch directory for one test, removed when it ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("brain-serve-connection-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(dir.join("models")).unwrap();
        std::fs::create_dir_all(dir.join("home")).unwrap();
        Scratch(dir)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

/// `brain --device cpu serve <args>` against an empty store, a private home
/// and a private runtime directory, so nothing on the host leaks in.
fn serve(dir: &Scratch, args: &[&str]) -> Command {
    let mut cmd = Command::new(bin());
    cmd.args(["--device", "cpu", "serve", "--seed", "1", "--models-dir"])
        .arg(dir.path("models"))
        .arg("--runtime-dir")
        .arg(dir.path("run"))
        .args(args)
        .stdin(Stdio::null())
        // Nothing of the caller's environment reaches the server: what it
        // serves and where it publishes come from its flags alone.
        .env_clear()
        .env("HOME", dir.path("home"));
    cmd
}

/// Wait for the server's connection record and return it parsed.
fn wait_for_record(child: &mut Child, run_dir: &Path) -> (PathBuf, serde_json::Value) {
    let record = run_dir.join(format!("serve-{}.json", child.id()));
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if let Ok(text) = std::fs::read_to_string(&record) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                return (record, value);
            }
        }
        if let Ok(Some(status)) = child.try_wait() {
            let mut stderr = String::new();
            child.stderr.take().map(|mut s| s.read_to_string(&mut stderr));
            panic!("brain serve exited with {status} before publishing {}; stderr:\n{stderr}", record.display());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    child.kill().ok();
    panic!("brain serve never published {}", record.display());
}

/// One blocking HTTP/1.1 GET; returns the status code.
fn get_status(url: &str, path: &str, bearer: &str) -> u16 {
    let rest = url.strip_prefix("http://").expect("an http URL");
    let host = rest.split('/').next().unwrap();
    let mut stream = std::net::TcpStream::connect(host).expect("connect to the published URL");
    stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    use std::io::Write;
    write!(stream, "GET {path} HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {bearer}\r\nConnection: close\r\n\r\n").unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).ok();
    response.split_whitespace().nth(1).and_then(|code| code.parse().ok()).unwrap_or(0)
}

fn surfaces(record: &serde_json::Value) -> Vec<(String, String, String)> {
    record["surfaces"]
        .as_array()
        .expect("a surfaces array")
        .iter()
        .map(|s| (s["provider"].as_str().unwrap().to_string(), s["url"].as_str().unwrap().to_string(), s["key"].as_str().unwrap().to_string()))
        .collect()
}

fn stop(mut child: Child) -> String {
    // SAFETY: signalling our own child by pid.
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    let deadline = Instant::now() + Duration::from_secs(60);
    while child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    child.kill().ok();
    let out = child.wait_with_output().expect("collect brain serve output");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A local client needs nothing but the runtime directory: the record names
/// the URL the surface really listens on (never the `:0` it was asked for),
/// and the key in it opens the API. It is owner-only, and it is gone once the
/// server stops, so a client never connects to a server that has left.
#[test]
fn a_running_server_publishes_where_it_listens_and_its_key() {
    let dir = Scratch::new("record");
    let mut child = serve(&dir, &["--openai", "127.0.0.1:0"]).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().expect("spawn brain serve");
    let (record, value) = wait_for_record(&mut child, &dir.path("run"));

    assert_eq!(value["pid"].as_u64(), Some(u64::from(child.id())), "the record names the serving process: {value}");
    let found = surfaces(&value);
    assert_eq!(found.len(), 1, "one surface was asked for: {value}");
    let (provider, url, key) = &found[0];
    assert_eq!(provider, "openai");
    assert!(url.starts_with("http://127.0.0.1:") && !url.starts_with("http://127.0.0.1:0/"), "the bound port, not :0: {url}");
    assert!(key.starts_with("sk-brain-"), "a generated key: {key}");
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(&record).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "the record holds a key, so only its owner may read it");
    assert_eq!(get_status(url, "/v1/models", key), 200, "the published key opens the published URL");
    assert_eq!(get_status(url, "/v1/models", "sk-brain-wrong"), 401, "the API still refuses any other key");

    let stderr = stop(child);
    assert!(!record.exists(), "a stopped server leaves no record behind; stderr:\n{stderr}");
}

/// The variable that used to pin the key, set to prove it no longer does.
const RETIRED_KEY_VARIABLE: &str = "BRAIN_API_KEY";

/// `--api-key-file` pins one key for every dialect the server binds; the
/// environment names no key any more.
#[test]
fn a_key_file_pins_one_key_for_every_dialect_and_the_environment_names_none() {
    let dir = Scratch::new("key-file");
    let key_file = dir.path("key");
    std::fs::write(&key_file, "sk-brain-pinned-for-the-test\n").unwrap();
    let mut child = serve(&dir, &["--openai", "127.0.0.1:0", "--anthropic", "127.0.0.1:0", "--api-key-file", key_file.to_str().unwrap()])
        .env(RETIRED_KEY_VARIABLE, "sk-brain-from-the-environment")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn brain serve");
    let (_, value) = wait_for_record(&mut child, &dir.path("run"));
    let found = surfaces(&value);
    stop(child);
    assert_eq!(found.len(), 2, "two dialects were asked for: {value}");
    for (provider, _, key) in &found {
        assert_eq!(key, "sk-brain-pinned-for-the-test", "{provider} serves the file's key, trimmed, not the environment's");
    }
}

/// An empty key file is a mistake, not a request for no key: refused at
/// startup, naming the flag.
#[test]
fn an_empty_key_file_is_refused_naming_the_flag() {
    let dir = Scratch::new("empty-key");
    let key_file = dir.path("key");
    std::fs::write(&key_file, "  \n").unwrap();
    let out = serve(&dir, &["--openai", "127.0.0.1:0", "--api-key-file", key_file.to_str().unwrap()]).output().expect("run brain serve");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "stderr:\n{stderr}");
    assert!(stderr.contains("--api-key-file"), "the error names the flag; stderr:\n{stderr}");
}

/// A model family with nothing on disk is not an event worth a line at the
/// default verbosity: startup names what is served and counts what is not,
/// and `-v` lists each one with its reason.
#[test]
fn startup_counts_the_families_it_cannot_serve_instead_of_listing_them() {
    let dir = Scratch::new("quiet");
    let mut child = serve(&dir, &["--openai", "127.0.0.1:0"]).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().expect("spawn brain serve");
    wait_for_record(&mut child, &dir.path("run"));
    let stderr = stop(child);
    assert!(!stderr.lines().any(|l| l.contains("not served")), "no per-family line at the default level; stderr:\n{stderr}");
    assert!(stderr.lines().any(|l| l.contains("model families") && l.contains("-v")), "one line counts them and says how to list them; stderr:\n{stderr}");

    let dir = Scratch::new("verbose");
    let mut child = serve(&dir, &["--openai", "127.0.0.1:0", "-v"]).stdout(Stdio::null()).stderr(Stdio::piped()).spawn().expect("spawn brain serve -v");
    wait_for_record(&mut child, &dir.path("run"));
    let stderr = stop(child);
    assert!(stderr.lines().any(|l| l.contains("not served")), "-v lists each family it cannot serve; stderr:\n{stderr}");
}

/// The stdio controller starts with no checkpoint at all: a capability whose
/// weights are absent is left out, never a panic.
#[test]
fn the_stdio_controller_starts_with_an_empty_model_store() {
    let dir = Scratch::new("stdio");
    let out = serve(&dir, &["--stdio"]).output().expect("run brain serve --stdio");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdin at EOF ends the loop cleanly; stderr:\n{stderr}");
    assert!(!stderr.contains("panicked"), "stderr:\n{stderr}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("ready"), "the loop announced itself");
}
