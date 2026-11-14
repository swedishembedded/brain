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

#[test]
fn connections_over_the_cap_are_refused_and_a_freed_slot_is_reused() {
    let addr = serve(limits(Duration::from_secs(30), Duration::from_secs(30), 2));
    // `serve`'s readiness probe holds a slot until the server sees it close.
    std::thread::sleep(Duration::from_millis(200));
    let held: Vec<TcpStream> = (0..2).map(|_| TcpStream::connect(addr).unwrap()).collect();
    // The accept loop takes the held connections in order before the third.
    std::thread::sleep(Duration::from_millis(200));
    let mut over = TcpStream::connect(addr).unwrap();
    assert!(closed_after(&mut over, Duration::from_secs(2)).is_some(), "a connection over the cap must be refused");

    drop(held);
    std::thread::sleep(Duration::from_millis(200));
    let mut again = TcpStream::connect(addr).unwrap();
    again.write_all(format!("GET /v1/models HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
    again.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut response = Vec::new();
    let _ = again.read_to_end(&mut response);
    assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200"), "a freed slot serves again: {:?}", String::from_utf8_lossy(&response));
}
