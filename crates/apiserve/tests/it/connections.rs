// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference servers that stay
// available when a hostile host shares their network for its clients. If your
// team needs expertise in hardening network services against resource
// exhaustion then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A peer cannot hold the server's connections: an idle or slow-header
//! connection is closed at the header deadline, a body that trickles in is cut
//! at the body deadline, and connections beyond the cap are refused at once.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use apiserve::{serve_all, ConnectionLimits, Provider, ServeOpts, Surface};
use residency::budget::Budgets;
use residency::{Device, Executor, Policy};

const KEY: &str = "sk-brain-test-key";

/// A server with no models on a fresh loopback port, under `limits`. It runs
/// until the test process exits.
fn serve(limits: ConnectionLimits) -> SocketAddr {
    let addr = TcpListener::bind("127.0.0.1:0").expect("bind").local_addr().expect("addr");
    let mut budgets = Budgets::new();
    budgets.set(Device::Cpu, 1 << 30, 0);
    let exec = Executor::start(Vec::new(), budgets, Policy::default());
    let (trigger, shutdown) = brain_shutdown::channel();
    std::thread::spawn(move || {
        let _keep = trigger;
        let _ = serve_all(exec, vec![Surface::new(Provider::OpenAI, addr, KEY)], ServeOpts::new().with_shutdown(shutdown).with_limits(limits));
    });
    let until = Instant::now() + Duration::from_secs(10);
    while TcpStream::connect(addr).is_err() {
        assert!(Instant::now() < until, "the server never started listening");
        std::thread::sleep(Duration::from_millis(20));
    }
    addr
}

/// How long until the server ends `stream` (EOF or reset), at most `within`.
/// `None` when the connection is still open at `within`.
fn closed_after(stream: &mut TcpStream, within: Duration) -> Option<Duration> {
    let start = Instant::now();
    stream.set_read_timeout(Some(within)).unwrap();
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) if start.elapsed() < within => return Some(start.elapsed()),
            Ok(0) | Err(_) => return None,
            Ok(_) => continue,
        }
    }
}

fn limits(header_read: Duration, body: Duration, max_connections: usize) -> ConnectionLimits {
    ConnectionLimits { header_read, body, max_connections, ..ConnectionLimits::default() }
}

#[test]
fn an_idle_connection_is_closed_at_the_header_deadline() {
    let addr = serve(limits(Duration::from_millis(300), Duration::from_secs(30), 64));
    let mut idle = TcpStream::connect(addr).unwrap();
    let took = closed_after(&mut idle, Duration::from_secs(5)).expect("an idle connection must not be held open");
    assert!(took >= Duration::from_millis(200), "closed before the deadline: {took:?}");
}

#[test]
fn a_connection_that_never_finishes_its_headers_is_closed() {
    let addr = serve(limits(Duration::from_millis(300), Duration::from_secs(30), 64));
    let mut slow = TcpStream::connect(addr).unwrap();
    slow.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n").unwrap();
    assert!(closed_after(&mut slow, Duration::from_secs(5)).is_some(), "a slow-header peer must not hold the connection");
}

#[test]
fn a_body_that_trickles_in_is_cut_at_the_body_deadline() {
    let addr = serve(limits(Duration::from_secs(30), Duration::from_millis(500), 64));
    let mut slow = TcpStream::connect(addr).unwrap();
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n"
    );
    slow.write_all(head.as_bytes()).unwrap();
    // One byte at a time, each well inside any per-frame idle timeout: only a
    // deadline on the whole body ends this.
    let writer = slow.try_clone().unwrap();
    std::thread::spawn(move || {
        let mut writer = writer;
        for _ in 0..4096 {
            if writer.write_all(b" ").is_err() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    let start = Instant::now();
    slow.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut response = Vec::new();
    let _ = slow.read_to_end(&mut response);
    let took = start.elapsed();
    assert!(took < Duration::from_secs(5), "the trickling body held the request for {took:?}");
    let response = String::from_utf8_lossy(&response);
    assert!(response.is_empty() || response.starts_with("HTTP/1.1 4"), "a cut body is a client error, got {response:?}");
}

/// A `GET /v1/models` on a fresh connection, read through its whole response; the response, with the stream kept
/// open, or `None` when the server ended the connection without answering (it held no slot for it).
fn models(addr: SocketAddr) -> Option<(TcpStream, String)> {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.write_all(format!("GET /v1/models HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\n\r\n").as_bytes()).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut got = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let text = String::from_utf8_lossy(&got);
        if let Some(head) = text.find("\r\n\r\n") {
            let length = text[..head].lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()));
            if got.len() >= head + 4 + length.expect("the response states its length") {
                return Some((stream, text.into_owned()));
            }
        }
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => return None,
            Ok(n) => got.extend_from_slice(&buf[..n]),
        }
    }
}

/// [`models`] retried until the server answers: a slot freed by a closed connection is released when the server
/// sees the close, which it does on its own schedule.
fn models_once_a_slot_is_free(addr: SocketAddr) -> (TcpStream, String) {
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(answered) = models(addr) {
            return answered;
        }
        assert!(Instant::now() < until, "no slot came free within 10 s");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn connections_over_the_cap_are_refused_and_a_freed_slot_is_reused() {
    let addr = serve(limits(Duration::from_secs(30), Duration::from_secs(30), 2));
    // Each held connection has been answered, so it holds a slot and keeps it (keep-alive). `serve`'s readiness
    // probe may still hold one until the server sees it close, which the retry waits out.
    let held: Vec<TcpStream> = (0..2)
        .map(|_| {
            let (stream, response) = models_once_a_slot_is_free(addr);
            assert!(response.starts_with("HTTP/1.1 200"), "{response:?}");
            stream
        })
        .collect();
    let mut over = TcpStream::connect(addr).unwrap();
    assert!(closed_after(&mut over, Duration::from_secs(2)).is_some(), "a connection over the cap must be refused");

    drop(held);
    let (_, response) = models_once_a_slot_is_free(addr);
    assert!(response.starts_with("HTTP/1.1 200"), "a freed slot serves again: {response:?}");
}
