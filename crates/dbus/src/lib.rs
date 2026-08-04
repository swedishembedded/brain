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
///
/// Installs its own SIGINT/SIGTERM handling via [`brain_shutdown`]. If this is the
/// only serving surface in the process, that is exactly right; when it runs
/// alongside an HTTP surface, use [`serve_with_shutdown`] instead so both surfaces
/// share one shutdown source — see that function's docs for why a second,
/// independent `tokio::signal::ctrl_c()` registration is a bug, not a redundancy.
pub fn serve(executor: Executor, opts: DbusOpts) -> anyhow::Result<()> {
    serve_with_shutdown(executor, opts, brain_shutdown::Shutdown::from_signals())
}

/// Serve the [`Executor`] over D-Bus until `shutdown` fires. Identical to [`serve`]
/// except the caller supplies (and owns the lifetime of) the shutdown signal —
/// the shape needed when D-Bus runs alongside another surface (see
/// `crates/cli/src/run_cli.rs::run_apis`), so exactly one SIGINT/SIGTERM
/// registration is shared by every surface in the process instead of each surface
/// racing to install its own.
///
/// No auto-fetch supplier — a plain `"no model '…'"` reply for an unresolved
/// model, exactly today's behavior. Use [`serve_with_shutdown_and_supplier`] to
/// enable transparent auto-fetch (`crates/cli/src/run_cli.rs::run_apis` does,
/// building a `StoreSupplier`).
pub fn serve_with_shutdown(executor: Executor, opts: DbusOpts, shutdown: brain_shutdown::Shutdown) -> anyhow::Result<()> {
    serve_with_shutdown_and_supplier(executor, opts, shutdown, None)
}

/// Like [`serve_with_shutdown`], but `Run`/`Subscribe`/`StreamTranscribe` first
/// try `supplier` (if given) for a model that isn't already resident, blocking
/// until it's fetched and registered before dispatching — see
/// `service::Manager::ensure_resident`.
pub fn serve_with_shutdown_and_supplier(
    executor: Executor,
    opts: DbusOpts,
    shutdown: brain_shutdown::Shutdown,
    supplier: Option<std::sync::Arc<dyn residency::ModelSupplier>>,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        // A cheap executor clone drives the background stats stream, so the
        // snapshot source is independent of the served `Manager` instance.
        let stats_executor = executor.clone();
        let manager = service::Manager::new(executor).with_supplier(supplier);
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
        // polling. It holds its own `conn` clone and exits on `shutdown` — see
        // `run_stats_stream`'s doc comment for why that exit path must exist.
        let stats = tokio::spawn(service::run_stats_stream(conn.clone(), stats_executor, OBJECT_PATH, shutdown.clone()));
        shutdown.wait().await;
        eprintln!("brain: shutting down D-Bus service");
        // Wait for the stats task to actually observe `shutdown` and return before
        // asking for a graceful shutdown: `graceful_shutdown()` awaits `conn`'s
        // drop event, which fires only once every `Connection` clone — including
        // the stats task's — is gone. Racing the two here is exactly the bug this
        // function exists to fix.
        let _ = stats.await;
        conn.graceful_shutdown().await;
        Ok::<(), anyhow::Error>(())
    })
}
