// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference servers that release
// their GPUs promptly on shutdown for its clients. If your team needs
// expertise in graceful shutdown of long-running model servers then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A live server stops promptly on shutdown even with a generation in flight:
//! the job is cancelled, and a job that ignores cancellation cannot hold the
//! process past the drain bound.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::{Duration, Instant};

use apiserve::{serve_all, Provider, ServeOpts, Surface, SHUTDOWN_DRAIN};
use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use residency::budget::Budgets;
use residency::{Device, Executor, Instance, InstanceKey, MemCost, Policy, ResidentModel};

const KEY: &str = "sk-brain-test-key";

/// A generation that runs for a minute unless cancelled, polling its token as
/// a real decode loop does between steps. With `honours_cancel` false it
/// models a step that cannot be interrupted.
struct Endless {
    started: Arc<AtomicBool>,
    honours_cancel: bool,
}
struct EndlessInstance {
    started: Arc<AtomicBool>,
    honours_cancel: bool,
}

impl ResidentModel for Endless {
    fn manifest(&self) -> Manifest {
        Manifest::new(
            "brain-chat",
            "a chat model",
            vec![ActionSpec::new("generate", "generate text")
                .streaming()
                .param(ParamSpec::new("prompt", ParamType::Str, "the prompt").required())
                .output(BlobSpec::new("text", Media::Text, "generated text"))],
        )
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("brain-chat", "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        MemCost::default()
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        Ok(Box::new(EndlessInstance { started: self.started.clone(), honours_cancel: self.honours_cancel }))
    }
}

impl Instance for EndlessInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.started.store(true, Ordering::SeqCst);
        let until = Instant::now() + Duration::from_secs(60);
        while Instant::now() < until {
            if self.honours_cancel && inv.cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(Outcome::new().blob("text", Blob::new(Media::Text, b"done".to_vec())))
    }
}

/// A free loopback address (bound and released, so `serve_all` can take it).
fn free_addr() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0").expect("bind").local_addr().expect("addr")
}

/// Serve `Endless` on a fresh port, put one chat request in flight, fire the
/// shutdown, and return how long `serve_all` took to return.
fn time_to_stop_with_a_request_in_flight(honours_cancel: bool) -> Duration {
    let started = Arc::new(AtomicBool::new(false));
    let models: Vec<Arc<dyn ResidentModel>> = vec![Arc::new(Endless { started: started.clone(), honours_cancel })];
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 1 << 30, 0);
    let exec = Executor::start(models, budgets, Policy::default());

    let addr = free_addr();
    let (trigger, shutdown) = brain_shutdown::channel();
    let (done_tx, done_rx) = channel();
    std::thread::spawn(move || {
        let result = serve_all(exec, vec![Surface::new(Provider::OpenAI, addr, KEY)], ServeOpts::new().with_shutdown(shutdown));
        let _ = done_tx.send((Instant::now(), result));
    });

    let body = r#"{"model":"brain-chat","messages":[{"role":"user","content":"hi"}]}"#;
    let request = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    let mut client = (0..100)
        .find_map(|_| {
            TcpStream::connect(addr).ok().or_else(|| {
                std::thread::sleep(Duration::from_millis(50));
                None
            })
        })
        .expect("the server never started listening");
    client.write_all(request.as_bytes()).expect("send the request");

    let until = Instant::now() + Duration::from_secs(5);
    while !started.load(Ordering::SeqCst) {
        assert!(Instant::now() < until, "the request never reached the model");
        std::thread::sleep(Duration::from_millis(10));
    }
    let fired = Instant::now();
    trigger.fire();

    let (returned, result) = done_rx.recv_timeout(SHUTDOWN_DRAIN + Duration::from_secs(5)).expect("serve_all never returned");
    result.expect("a shutdown is a clean stop");
    // The client sees the connection end (a reply or a reset), never a hang.
    client.set_read_timeout(Some(Duration::from_secs(2))).expect("timeout");
    let mut sink = Vec::new();
    let _ = client.read_to_end(&mut sink);
    returned.duration_since(fired)
}

#[test]
fn shutdown_cancels_the_generation_in_flight_and_stops_at_once() {
    let took = time_to_stop_with_a_request_in_flight(true);
    assert!(took < Duration::from_secs(2), "a cancelled generation must not hold the server: {took:?}");
}

#[test]
fn a_generation_that_cannot_be_interrupted_holds_shutdown_only_for_the_drain_bound() {
    let took = time_to_stop_with_a_request_in_flight(false);
    assert!(took < SHUTDOWN_DRAIN + Duration::from_secs(2), "the drain bound must end the wait: {took:?}");
}
