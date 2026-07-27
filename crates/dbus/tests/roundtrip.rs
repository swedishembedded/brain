// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end D-Bus round-trip: serve a tiny capability provider, then call `Run`
//! over the bus as a client and verify the result comes back through a file
//! descriptor. Needs a session bus, so it **skips** when `DBUS_SESSION_BUS_ADDRESS`
//! is unset — run it under one:
//!
//!     dbus-run-session -- cargo test -p brain-dbus --test roundtrip -- --nocapture
//!
//! This exercises the whole surface (worker thread → memfd → `a{sh}` reply →
//! client mmap) without any GPU or model weights.

use std::collections::HashMap;
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, Invocation, Manifest, Media, Outcome, Progress, Provider, Registry};
use serde_json::json;
use zbus::zvariant::OwnedFd as ZOwnedFd;

// ---- a no-weights provider: `bytes.reverse(text)` -> a Bytes blob "out" ----
struct RevProvider;
struct RevAction;

impl Action for RevAction {
    fn spec(&self) -> ActionSpec {
        use capability::{BlobSpec, ParamSpec, ParamType};
        ActionSpec::new("reverse", "reverse the bytes of `text`")
            .param(ParamSpec::new("text", ParamType::Str, "text to reverse").required())
            .output(BlobSpec::new("out", Media::Bytes, "the reversed bytes"))
    }
    fn run(&self, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        let t = inv.get_str("text").unwrap_or_default();
        let rev: Vec<u8> = t.bytes().rev().collect();
        Ok(Outcome::new().set("len", json!(rev.len())).blob("out", Blob::new(Media::Bytes, rev)))
    }
}

/// `cat` — echoes an **input blob** back out, proving input-fd passing: the client
/// sends bytes as an fd, the action reads them, the result returns as an fd.
struct CatAction;
impl Action for CatAction {
    fn spec(&self) -> ActionSpec {
        use capability::BlobSpec;
        ActionSpec::new("cat", "echo the input blob back")
            .input(BlobSpec::new("data", Media::Bytes, "input bytes").required())
            .output(BlobSpec::new("out", Media::Bytes, "the same bytes"))
    }
    fn run(&self, inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
        let b = inv.get_blob("data").ok_or("missing input blob 'data'")?;
        Ok(Outcome::new().set("len", json!(b.bytes.len())).blob("out", Blob::new(Media::Bytes, b.bytes.clone())))
    }
}

impl Provider for RevProvider {
    fn manifest(&self) -> Manifest {
        Manifest::new("rev", "byte reverser (test provider)", vec![RevAction.spec(), CatAction.spec()])
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "reverse" => Some(Arc::new(RevAction) as Arc<dyn Action>),
            "cat" => Some(Arc::new(CatAction) as Arc<dyn Action>),
            _ => None,
        }
    }
}

/// Create a sealed memfd holding `data` for use as an input fd.
fn memfd(data: &[u8]) -> std::os::fd::OwnedFd {
    brain_dbus::fd::memfd_seal("test-in", data).unwrap()
}

#[test]
fn run_roundtrips_a_result_over_an_fd() {
    if std::env::var("DBUS_SESSION_BUS_ADDRESS").map(|s| s.is_empty()).unwrap_or(true) {
        eprintln!("SKIP: no session bus (run under `dbus-run-session -- cargo test ...`)");
        return;
    }
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(async {
        // ---- service side ----
        let mut reg = Registry::new();
        reg.register(Arc::new(RevProvider));
        let (handle, _join) = brain_dbus::worker::spawn(reg);
        let manager = brain_dbus::service::Manager::new(handle);
        // Unique name so parallel test runs don't collide.
        let name = format!("com.swedishembedded.Brain1.test{}", std::process::id());
        let _conn = zbus::connection::Builder::session()
            .unwrap()
            .name(name.as_str())
            .unwrap()
            .serve_at(brain_dbus::OBJECT_PATH, manager)
            .unwrap()
            .build()
            .await
            .unwrap();

        // ---- client side ----
        let client = zbus::Connection::session().await.unwrap();
        let proxy = zbus::Proxy::new(&client, name.as_str(), brain_dbus::OBJECT_PATH, "com.swedishembedded.Brain1.Manager")
            .await
            .unwrap();

        let empty: HashMap<String, ZOwnedFd> = HashMap::new();
        let (result, out_fds, out_meta): (String, HashMap<String, ZOwnedFd>, String) = proxy
            .call("Run", &("rev", "reverse", r#"{"text":"brain"}"#, empty, "", "memfd"))
            .await
            .unwrap();

        // Scalar result + metadata.
        assert_eq!(serde_json::from_str::<serde_json::Value>(&result).unwrap()["len"], 5);
        assert!(out_meta.contains("\"transport\":\"memfd\""), "meta: {out_meta}");

        // The output blob came back as an fd — mmap it and check the bytes.
        let fd = out_fds.get("out").expect("out fd present");
        let bytes = brain_dbus::fd::read_owned_to_vec(fd).unwrap();
        assert_eq!(bytes, b"niarb", "reversed bytes via fd: {:?}", String::from_utf8_lossy(&bytes));
        eprintln!("roundtrip ok: Run(rev.reverse) -> fd -> {:?}", String::from_utf8_lossy(&bytes));

        // ---- input-fd: send bytes as an fd, get them echoed back through an fd ----
        let payload = b"input via file descriptor".to_vec();
        let in_fd: ZOwnedFd = memfd(&payload).into();
        let mut in_fds: HashMap<String, ZOwnedFd> = HashMap::new();
        in_fds.insert("data".to_string(), in_fd);
        let in_meta = r#"{"data":{"media":"bytes"}}"#;
        let (result2, out_fds2, _meta2): (String, HashMap<String, ZOwnedFd>, String) =
            proxy.call("Run", &("rev", "cat", "{}", in_fds, in_meta, "memfd")).await.unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(&result2).unwrap()["len"], payload.len());
        let echoed = brain_dbus::fd::read_owned_to_vec(out_fds2.get("out").expect("out fd")).unwrap();
        assert_eq!(echoed, payload, "input fd not echoed correctly");
        eprintln!("roundtrip ok: Run(rev.cat) input-fd -> output-fd -> {:?}", String::from_utf8_lossy(&echoed));
    });
}
