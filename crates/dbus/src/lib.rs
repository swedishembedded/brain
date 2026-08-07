// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Optional D-Bus control surface for brain.
//!
//! Exposes the shared [`residency::Executor`] over the bus name
//! `com.swedishembedded.Brain1`, so local Linux apps can discover models, run
//! actions, and exchange images/streams/results as **file descriptors**
//! (memfd/mmap, and dmabuf where available) instead of bytes over D-Bus.
//! Every method only validates + translates: it builds a `capability::
//! Invocation` from the params + in_fds, arms a `CancelToken`, submits a
//! `residency::Job`, and returns the outcome fds — the SAME executor
//! `crates/apiserve`'s HTTP surfaces submit to, so scheduling/residency/
//! batching stay uniform across both.
//!
//! Layering (kept deliberately thin — no model code here):
//! - [`fd`] — memfd/mmap FD transport.
//! - [`service`] — the zbus `Manager` interface (validate → build an
//!   `Invocation` → submit a `residency::Job` → reply/stream frames).
//! - [`stream`] — the `Subscribe`/`StreamTranscribe` SEQPACKET frame protocol.
//! - `serve` — wires a Tokio runtime + a zbus connection + the executor together.

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

/// Everything [`serve`] needs beyond the executor and the bus options. Mirrors
/// `apiserve::ServeOpts` — see that type's docs for why a shared shutdown source
/// matters when D-Bus runs alongside an HTTP surface, and why the readiness gate
/// is a marker file rather than a bus method.
#[derive(Default)]
pub struct ServeOpts {
    /// `None` installs a private shutdown source via
    /// [`brain_shutdown::Shutdown::from_signals`] — right for a D-Bus-only
    /// process, wrong when it runs alongside HTTP (see `apiserve::ServeOpts::shutdown`).
    pub shutdown: Option<brain_shutdown::Shutdown>,
    /// `Run`/`Subscribe`/`StreamTranscribe` first try this (if given) for a model
    /// that isn't already resident, blocking until it's fetched and registered —
    /// see `service::Manager::ensure_resident`. `None` is today's default: a
    /// plain `"no model '…'"` reply for an unresolved model.
    pub supplier: Option<std::sync::Arc<dyn residency::ModelSupplier>>,
    /// Notified once, after the well-known bus name is acquired. Default is
    /// [`brain_shutdown::ready::Gate::disabled`] (a no-op).
    pub ready: brain_shutdown::ready::Gate,
}

impl ServeOpts {
    pub fn new() -> ServeOpts {
        ServeOpts::default()
    }
    pub fn with_shutdown(mut self, shutdown: brain_shutdown::Shutdown) -> ServeOpts {
        self.shutdown = Some(shutdown);
        self
    }
    pub fn with_supplier(mut self, supplier: Option<std::sync::Arc<dyn residency::ModelSupplier>>) -> ServeOpts {
        self.supplier = supplier;
        self
    }
    pub fn with_ready(mut self, ready: brain_shutdown::ready::Gate) -> ServeOpts {
        self.ready = ready;
        self
    }
}

/// Serve the [`Executor`] over D-Bus until `opts.shutdown` fires (or, with
/// `opts.shutdown: None`, until this process's own Ctrl-C/SIGTERM). Builds a
/// multi-threaded Tokio runtime, connects to the chosen bus, requests the
/// well-known name, and serves the [`service::Manager`] at [`OBJECT_PATH`]. The
/// executor owns the residency manager + scheduler worker (inference runs there,
/// off the bus threads). Blocks the calling thread for the lifetime of the service.
pub fn serve(executor: Executor, opts: DbusOpts, serve_opts: ServeOpts) -> anyhow::Result<()> {
    let shutdown = serve_opts.shutdown.unwrap_or_else(brain_shutdown::Shutdown::from_signals);
    let ServeOpts { supplier, ready, .. } = serve_opts;
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
        match opts.bus {
            BusKind::Session => {
                eprintln!("brain: connect with: braintop --name {}", opts.name);
                // zbus::connection::Builder::session() (used above) resolves this
                // same address: $DBUS_SESSION_BUS_ADDRESS, else $XDG_RUNTIME_DIR/bus
                // -- see zbus::address::Address::session(). Print the resolved
                // value so a DIFFERENT shell/process (one that did not inherit this
                // process's environment -- e.g. this was launched under
                // `dbus-run-session`, whose private bus is otherwise invisible
                // anywhere else) can still reach it.
                match std::env::var("DBUS_SESSION_BUS_ADDRESS") {
                    Ok(addr) => eprintln!("brain:   (session bus address: {addr} -- export DBUS_SESSION_BUS_ADDRESS=\"{addr}\" in another shell, or pass braintop --address \"{addr}\")"),
                    Err(_) => eprintln!("brain:   (DBUS_SESSION_BUS_ADDRESS is unset here; zbus falls back to $XDG_RUNTIME_DIR/bus -- braintop must resolve the SAME fallback to connect, e.g. run it from a shell with the same $XDG_RUNTIME_DIR)"),
                }
            }
            BusKind::System => eprintln!("brain: connect with: braintop --system --name {}", opts.name),
        }
        // The well-known name is owned and the object is served: this is a true
        // "up" point. zbus 5's `Builder::name` acquires the name during `build()`
        // and errors above if it cannot, so reaching here means both succeeded.
        ready.bound("dbus");
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
