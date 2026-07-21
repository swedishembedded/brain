// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Listening transports: TCP and Unix-socket accept loops, thread-per-connection
//! with a concurrency cap. Both drive [`pump_connection`](crate::pump_connection)
//! with a fresh [`Session`](crate::Session) per connection, built by a factory so
//! per-instance model state never crosses threads.

use crate::{pump_connection, Session};
use std::io::{self, BufReader};
use std::net::{TcpListener, ToSocketAddrs};
use std::os::unix::net::UnixListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

/// A factory building one fresh [`Session`] per connection. `Send + Sync` so it
/// can be shared across accept threads; its *return* stays on the connection
/// thread and need not be `Send`.
pub type SessionFactory = Arc<dyn Fn() -> Box<dyn Session> + Send + Sync>;

/// Serving options.
#[derive(Clone, Debug)]
pub struct ServeOpts {
    /// Maximum concurrent connections. Excess connections are dropped
    /// immediately with a short error line — admission control, so a flood of
    /// clients can't spawn unbounded threads.
    pub max_connections: usize,
}

impl Default for ServeOpts {
    fn default() -> Self {
        ServeOpts { max_connections: 64 }
    }
}

/// The over-capacity rejection line (a structured error the client can read
/// before the socket closes).
fn busy_line() -> String {
    serde_json::json!({
        "event": "error",
        "code": "server_busy",
        "message": "connection limit reached",
        "retryable": true,
    })
    .to_string()
}

/// Accept + dispatch loop shared by both listeners. `accept` yields streams that
/// can be split into an independent writer; each gets a thread and a fresh
/// session.
fn accept_loop<S, A>(mut accept: A, make: SessionFactory, opts: ServeOpts) -> io::Result<()>
where
    S: CloneStream,
    A: FnMut() -> io::Result<S>,
{
    let live = Arc::new(AtomicUsize::new(0));
    loop {
        let mut stream = accept()?;
        if live.load(Ordering::Relaxed) >= opts.max_connections {
            let _ = writeln!(stream, "{}", busy_line());
            let _ = stream.flush();
            continue; // stream dropped -> connection closed
        }
        live.fetch_add(1, Ordering::Relaxed);
        let make = make.clone();
        let live_c = live.clone();
        // Each connection: its own reader/writer split and its own session.
        let handle = thread::Builder::new().name("brain-conn".into()).spawn(move || {
            let writer = match stream.dup() {
                Ok(w) => w,
                Err(_) => {
                    live_c.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
            };
            let reader = BufReader::new(stream);
            let mut session = make();
            let _ = pump_connection(reader, writer, session.as_mut());
            live_c.fetch_sub(1, Ordering::Relaxed);
        });
        if handle.is_err() {
            live.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// A duplex stream we can split into an independent writer. TCP and Unix streams
/// both support `try_clone`; this trait unifies them.
trait CloneStream: io::Read + io::Write + Send + 'static {
    fn dup(&self) -> io::Result<Box<dyn io::Write + Send>>;
}

impl CloneStream for std::net::TcpStream {
    fn dup(&self) -> io::Result<Box<dyn io::Write + Send>> {
        Ok(Box::new(self.try_clone()?))
    }
}
impl CloneStream for std::os::unix::net::UnixStream {
    fn dup(&self) -> io::Result<Box<dyn io::Write + Send>> {
        Ok(Box::new(self.try_clone()?))
    }
}

/// Serve the `events::Event` protocol over a Unix domain socket. Removes a stale
/// socket file at `path` first. Blocks forever (run on its own thread).
pub fn serve_unix<P: AsRef<std::path::Path>>(
    path: P,
    make: SessionFactory,
    opts: ServeOpts,
) -> io::Result<()> {
    let path = path.as_ref();
    let _ = std::fs::remove_file(path); // clear a stale socket
    let listener = UnixListener::bind(path)?;
    accept_loop(move || listener.accept().map(|(s, _)| s), make, opts)
}

/// Serve the `events::Event` protocol over TCP. Blocks forever.
pub fn serve_tcp<A: ToSocketAddrs>(
    addr: A,
    make: SessionFactory,
    opts: ServeOpts,
) -> io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    accept_loop(move || listener.accept().map(|(s, _)| s), make, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    #[test]
    fn unix_socket_round_trips_a_line() {
        // A minimal session so we don't pull the whole runtime here.
        struct Echo;
        impl Session for Echo {
            fn on_line(&mut self, line: &str) -> Vec<String> {
                vec![format!("{{\"echo\":{}}}", line.len())]
            }
            fn greeting(&mut self) -> Vec<String> {
                vec!["{\"event\":\"ready\"}".into()]
            }
        }

        let dir = std::env::temp_dir();
        let path = dir.join(format!("brain-server-test-{}.sock", std::process::id()));
        let p2 = path.clone();
        let make: SessionFactory = Arc::new(|| Box::new(Echo));
        // server on a detached thread (accept loop blocks forever)
        thread::spawn(move || {
            let _ = serve_unix(&p2, make, ServeOpts::default());
        });

        // wait for the socket to appear
        let mut client = None;
        for _ in 0..200 {
            if let Ok(s) = UnixStream::connect(&path) {
                client = Some(s);
                break;
            }
            thread::sleep(std::time::Duration::from_millis(5));
        }
        let stream = client.expect("server never came up");
        let mut writer = stream.try_clone().unwrap();
        let mut reader = BufReader::new(stream);

        // greeting first
        let mut ready = String::new();
        reader.read_line(&mut ready).unwrap();
        assert!(ready.contains("ready"), "greeting: {ready:?}");

        // send a line, read the echo
        writeln!(writer, "hello").unwrap();
        writer.flush().unwrap();
        let mut resp = String::new();
        reader.read_line(&mut resp).unwrap();
        assert!(resp.contains("\"echo\":5"), "resp: {resp:?}");

        let _ = std::fs::remove_file(&path);
    }
}
