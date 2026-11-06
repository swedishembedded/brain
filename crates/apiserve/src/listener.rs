// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference servers that stay
// available when a hostile host shares their network for its clients. If your
// team needs expertise in hardening network services against resource
// exhaustion then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The accept loop every HTTP surface is served by.
//!
//! `axum::serve` sets no timer, so hyper's header deadline never runs, and it
//! accepts without limit: a peer with no key could open sockets, send nothing,
//! and hold them until the process ran out of descriptors. This loop bounds
//! what any peer, keyed or not, can hold:
//!
//! - **Connections** are capped ([`ConnectionLimits::max_connections`]); one
//!   over the cap is closed at accept, before a byte is read.
//! - **Headers** must arrive within [`ConnectionLimits::header_read`]. hyper
//!   starts that timer whenever it waits for a request head, so it also ends a
//!   connection that sits idle between keep-alive requests.
//! - **Bodies** must arrive whole within [`ConnectionLimits::body`], a deadline
//!   on the entire body rather than between frames, so a body trickled in one
//!   byte at a time cannot hold a request open either.
//!
//! Only HTTP/1.1 is served: HTTP/2 has no header deadline, so an h2 peer that
//! answers keep-alive pings could hold a connection without ever sending a
//! request. Response streams (SSE) stay open as long as their job runs; the
//! connection cap bounds them.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use axum::Router;
use futures::StreamExt;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

/// What one peer may hold of a surface. See the module docs for each bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectionLimits {
    /// How long a connection may take to send a request head, including the
    /// wait for the next request on a kept-alive connection.
    pub header_read: Duration,
    /// How long a request may take to send its whole body.
    pub body: Duration,
    /// How many connections the surface holds at once.
    pub max_connections: usize,
}

impl Default for ConnectionLimits {
    /// Ten seconds for headers is generous for any client that means to send a
    /// request. Five minutes for a body covers the largest accepted body
    /// (`MAX_BODY_BYTES` on the upload routes, 64 MiB) at under 2 Mbit/s.
    fn default() -> ConnectionLimits {
        ConnectionLimits { header_read: Duration::from_secs(10), body: Duration::from_secs(300), max_connections: 1024 }
    }
}

/// The error a request body yields when its deadline passes. The extractor
/// reading it fails, so the handler answers with a client error.
#[derive(Debug)]
struct BodyDeadline(Duration);

impl std::fmt::Display for BodyDeadline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "request body not received within {:?}", self.0)
    }
}

impl std::error::Error for BodyDeadline {}

/// Give `request`'s body a deadline of `limit` from now, after which reading
/// it fails.
async fn body_deadline(limit: Duration, request: Request, next: Next) -> Response {
    let deadline = tokio::time::Instant::now() + limit;
    let (parts, body) = request.into_parts();
    let mut data = body.into_data_stream();
    let bounded = async_stream::stream! {
        loop {
            match tokio::time::timeout_at(deadline, data.next()).await {
                Ok(Some(chunk)) => yield chunk,
                Ok(None) => break,
                Err(_) => {
                    yield Err(axum::Error::new(BodyDeadline(limit)));
                    break;
                }
            }
        }
    };
    next.run(Request::from_parts(parts, Body::from_stream(bounded))).await
}

/// Serve `app` on `listener` under `limits` until `shutdown` fires, then stop
/// accepting and let open connections finish. The caller bounds that drain.
pub(crate) async fn serve(listener: TcpListener, app: Router, limits: ConnectionLimits, shutdown: brain_shutdown::Shutdown) {
    let app = app.layer(axum::middleware::from_fn(move |request, next| body_deadline(limits.body, request, next)));
    let slots = Arc::new(Semaphore::new(limits.max_connections));
    let graceful = GracefulShutdown::new();
    let stop = shutdown.wait();
    tokio::pin!(stop);
    loop {
        let stream = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _peer)) => stream,
                Err(e) => {
                    // Out of descriptors, or a connection reset before accept:
                    // neither ends the surface. The pause keeps a persistent
                    // error from spinning a core.
                    eprintln!("brain apiserve: accept failed: {e}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            },
            _ = &mut stop => break,
        };
        let Ok(slot) = slots.clone().try_acquire_owned() else {
            drop(stream);
            continue;
        };
        let mut http = hyper::server::conn::http1::Builder::new();
        http.timer(TokioTimer::new()).header_read_timeout(limits.header_read);
        let connection = http.serve_connection(TokioIo::new(stream), TowerToHyperService::new(app.clone()));
        let watched = graceful.watch(connection);
        tokio::spawn(async move {
            // A peer that resets or times out is that connection's end, not an
            // error of the surface.
            let _ = watched.await;
            drop(slot);
        });
    }
    graceful.shutdown().await;
}
