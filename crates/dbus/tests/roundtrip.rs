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

use capability::{Action, ActionResult, ActionSpec, Blob, Invocation, Manifest, Media, Outcome, Progress, Provider};
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

/// `slow` — a long-running action that polls the invocation's cancel token each
/// step, so `Cancel(job)` can be verified end-to-end: uncancelled it takes ~10 s,
/// cancelled it aborts within one 20 ms step.
struct SlowAction;
impl Action for SlowAction {
    fn spec(&self) -> ActionSpec {
        ActionSpec::new("slow", "sleep in steps, polling the cancel token").streaming()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        for step in 0..500 {
            if inv.cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            progress(Progress::step(step, 500, "working"));
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        Ok(Outcome::new())
    }
}

impl Provider for RevProvider {
    fn manifest(&self) -> Manifest {
        Manifest::new("rev", "byte reverser (test provider)", vec![RevAction.spec(), CatAction.spec(), SlowAction.spec()])
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "reverse" => Some(Arc::new(RevAction) as Arc<dyn Action>),
            "cat" => Some(Arc::new(CatAction) as Arc<dyn Action>),
            "slow" => Some(Arc::new(SlowAction) as Arc<dyn Action>),
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
        // ---- service side: RevProvider as a (stateless) resident model behind the
        // shared Executor, exactly as `brain serve --dbus` wires real models. ----
        let rev: Arc<dyn residency::ResidentModel> = Arc::new(residency::bridge::ProviderResident::stateless(Arc::new(RevProvider)));
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(residency::Device::Gpu(0), 24 << 30, 0);
        let executor = residency::Executor::start(vec![rev], budgets, residency::Policy::default());
        let manager = brain_dbus::service::Manager::new(executor);
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

        // ---- Cancel: bogus id -> false; a live Subscribe job -> true, and the
        // polling action must actually abort (its registry entry drains long before
        // the 10 s an uncancelled run would take). ----
        let bogus: bool = proxy.call("Cancel", &(u64::MAX,)).await.unwrap();
        assert!(!bogus, "Cancel on a bogus job id must return false");

        let empty2: HashMap<String, ZOwnedFd> = HashMap::new();
        let (job, _event_fd): (u64, ZOwnedFd) = proxy.call("Subscribe", &("rev", "slow", "{}", empty2, "")).await.unwrap();
        // Give the executor a moment to start the action, then cancel it.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let hit: bool = proxy.call("Cancel", &(job,)).await.unwrap();
        assert!(hit, "Cancel on a live job id must return true");
        // The job leaves the registry when its (cancelled) reply fires.
        let t0 = std::time::Instant::now();
        loop {
            let still: bool = proxy.call("Cancel", &(job,)).await.unwrap();
            if !still {
                break;
            }
            assert!(t0.elapsed() < std::time::Duration::from_secs(5), "cancelled action did not abort");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        eprintln!("cancel ok: Subscribe(rev.slow) aborted in {:?}", t0.elapsed());
    });
}

/// Regression test for the shutdown deadlock: before the fix, `run_stats_stream`
/// held its own `zbus::Connection` clone for its whole life and only released it
/// when a signal emit *failed* — which cannot happen while the connection is
/// alive — so `Connection::graceful_shutdown()` awaited a drop event that could
/// never fire. `serve`/`serve_with_shutdown` would print "shutting down" and then
/// hang forever. This test drives `serve_with_shutdown` on its own thread, waits
/// for it to actually claim the bus name (so the fix is proven against a *live*
/// connection with the stats task running, not just an early return), fires the
/// shutdown token, and asserts the thread joins within a bound. On the old code
/// this test hangs until the harness's own timeout kills it.
#[test]
fn dbus_serve_stops_promptly_once_shutdown_fires() {
    if std::env::var("DBUS_SESSION_BUS_ADDRESS").map(|s| s.is_empty()).unwrap_or(true) {
        eprintln!("SKIP: no session bus (run under `dbus-run-session -- cargo test ...`)");
        return;
    }
    // A stateless resident is enough — this test is about the shutdown handshake,
    // not about serving a real action.
    let rev: Arc<dyn residency::ResidentModel> = Arc::new(residency::bridge::ProviderResident::stateless(Arc::new(RevProvider)));
    let mut budgets = residency::budget::Budgets::new();
    budgets.set(residency::Device::Gpu(0), 24 << 30, 0);
    let executor = residency::Executor::start(vec![rev], budgets, residency::Policy::default());

    let name = format!("com.swedishembedded.Brain1.shutdowntest{}", std::process::id());
    let opts = brain_dbus::DbusOpts { bus: brain_dbus::BusKind::Session, name: name.clone() };
    let (trigger, shutdown) = brain_shutdown::channel();

    let handle = std::thread::spawn(move || brain_dbus::serve_with_shutdown(executor, opts, shutdown));

    // Wait for the server to actually claim the bus name (proving a *live*
    // connection, stats task and all, unwinds cleanly) before firing shutdown.
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let ready = rt.block_on(async {
        let client = zbus::Connection::session().await.unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Ok(proxy) = zbus::Proxy::new(&client, name.as_str(), brain_dbus::OBJECT_PATH, "com.swedishembedded.Brain1.Manager").await {
                if proxy.call::<_, _, Vec<String>>("ListModels", &()).await.is_ok() {
                    return true;
                }
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    });
    assert!(ready, "server never claimed {name} on the bus within 10s");

    trigger.fire();
    let t0 = std::time::Instant::now();
    loop {
        if handle.is_finished() {
            break;
        }
        assert!(t0.elapsed() < std::time::Duration::from_secs(5), "brain_dbus::serve_with_shutdown did not stop within 5s of shutdown firing");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    handle.join().unwrap().unwrap();
}
