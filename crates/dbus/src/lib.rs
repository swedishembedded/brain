// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Optional D-Bus control surface for brain.
//!
//! Exposes the synchronous [`capability::Registry`] over the bus name
//! `com.swedishembedded.Brain1`, so local Linux apps can discover models, run
//! actions, and exchange images/streams/results as **file descriptors**
//! (memfd/mmap, and dmabuf where available) instead of bytes over D-Bus.
//!
//! Layering (kept deliberately thin — no model code here):
//! - [`fd`] — memfd/mmap FD transport.
//! - `worker` — a dedicated thread that owns the `Registry` and runs the blocking
//!   inference off the async/D-Bus threads.
//! - `service` — the zbus `Manager` interface (validate → enqueue → reply).
//! - [`serve`] — wires a Tokio runtime + the worker + a zbus connection together.

pub mod fd;
pub mod service;
pub mod stream;

use residency::Executor;

/// Which bus to connect to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BusKind {
    /// Per-user session bus (default; no policy file / root needed).
    #[default]
    Session,
    /// System-wide bus (needs a `system.d` policy to be callable by non-root).
    System,
}

/// Options for [`serve`].
#[derive(Clone, Debug)]
pub struct DbusOpts {
    pub bus: BusKind,
    /// Well-known name to request (default `com.swedishembedded.Brain1`).
    pub name: String,
}

impl Default for DbusOpts {
    fn default() -> DbusOpts {
        DbusOpts { bus: BusKind::Session, name: "com.swedishembedded.Brain1".to_string() }
    }
}

/// The object path the `Manager` is served at.
pub const OBJECT_PATH: &str = "/com/swedishembedded/Brain1";

/// Serve the [`Executor`] over D-Bus until Ctrl-C / SIGTERM. Builds a multi-threaded
/// Tokio runtime, connects to the chosen bus, requests the well-known name, and
/// serves the [`service::Manager`] at [`OBJECT_PATH`]. The executor owns the
/// residency manager + scheduler worker (inference runs there, off the bus threads).
/// Blocks the calling thread for the lifetime of the service.
pub fn serve(executor: Executor, opts: DbusOpts) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        // A cheap executor clone drives the background stats stream, so the
        // snapshot source is independent of the served `Manager` instance.
        let stats_executor = executor.clone();
        let manager = service::Manager::new(executor);
        let builder = match opts.bus {
            BusKind::Session => zbus::connection::Builder::session()?,
            BusKind::System => zbus::connection::Builder::system()?,
        };
        let conn = builder
            .name(opts.name.as_str())?
            .serve_at(OBJECT_PATH, manager)?
            .build()
            .await?;
        eprintln!("brain: serving {} on the {:?} bus at {OBJECT_PATH}", opts.name, opts.bus);
        // Push the self-describing stats snapshot as the `StatsStream` signal at
        // >=2 Hz (see `service::STATS_INTERVAL`), so braintop subscribes instead of
        // polling. Detached; it stops when the connection drops.
        tokio::spawn(service::run_stats_stream(conn.clone(), stats_executor, OBJECT_PATH));
        // Run until interrupted; graceful shutdown releases the name cleanly.
        tokio::signal::ctrl_c().await?;
        eprintln!("brain: shutting down D-Bus service");
        conn.graceful_shutdown().await;
        Ok::<(), anyhow::Error>(())
    })
}
