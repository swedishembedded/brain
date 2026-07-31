// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The D-Bus side: connect to `com.swedishembedded.Brain1`, **subscribe** to the
//! `StatsStream` signal, and **fall back to polling** `stats_snapshot()` at ~2 Hz
//! if no signal is flowing. Also a one-shot [`fetch_once`] for `--cli`.
//!
//! All of this is confined here so the rest of braintop (app/ui/cli) stays a pure
//! function of a [`StatsSnapshot`] value and needs no bus in its tests.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use brain_stats::StatsSnapshot;
use futures::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

/// Default well-known name brain requests.
pub const DEFAULT_NAME: &str = "com.swedishembedded.Brain1";
/// Object path the `Manager` interface is served at (mirrors `brain-dbus`).
pub const DEFAULT_PATH: &str = "/com/swedishembedded/Brain1";
/// Poll cadence for the `stats_snapshot()` fallback (2 Hz — the stream's floor).
const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// If no signal arrived within this window, poll once to stay ≥2 Hz.
const SIGNAL_GRACE: Duration = Duration::from_millis(900);
/// Backoff between reconnect attempts.
const RECONNECT_DELAY: Duration = Duration::from_secs(1);

/// Which bus (or explicit address) to connect to.
#[derive(Clone, Debug)]
pub enum Bus {
    Session,
    System,
    Address(String),
}

/// Where and how to reach brain.
#[derive(Clone, Debug)]
pub struct ConnOpts {
    pub bus: Bus,
    pub name: String,
    pub path: String,
}

impl Default for ConnOpts {
    fn default() -> ConnOpts {
        ConnOpts { bus: Bus::Session, name: DEFAULT_NAME.to_string(), path: DEFAULT_PATH.to_string() }
    }
}

/// A transport event pushed to the app.
#[derive(Clone, Debug)]
pub enum Update {
    Connected,
    Disconnected(String),
    Snapshot(StatsSnapshot),
}

/// The `com.swedishembedded.Brain1.Manager` client proxy — just the stats surface
/// braintop reads (the one-shot pull method and the live signal).
#[zbus::proxy(
    interface = "com.swedishembedded.Brain1.Manager",
    default_service = "com.swedishembedded.Brain1",
    default_path = "/com/swedishembedded/Brain1"
)]
pub trait Manager {
    /// One-shot pull of the full self-describing stats snapshot (JSON).
    fn stats_snapshot(&self) -> zbus::Result<String>;

    /// Live snapshot pushed by brain at ~2 Hz (the same JSON document).
    #[zbus(signal)]
    fn stats_stream(&self, snapshot: String) -> zbus::Result<()>;
}

async fn connect(opts: &ConnOpts) -> Result<zbus::Connection> {
    let conn = match &opts.bus {
        Bus::Session => zbus::Connection::session().await.context("session bus")?,
        Bus::System => zbus::Connection::system().await.context("system bus")?,
        Bus::Address(addr) => zbus::connection::Builder::address(addr.as_str())?.build().await.context("address")?,
    };
    Ok(conn)
}

async fn build_proxy<'a>(conn: &zbus::Connection, opts: &ConnOpts) -> Result<ManagerProxy<'a>> {
    let proxy = ManagerProxy::builder(conn)
        .destination(opts.name.clone())?
        .path(opts.path.clone())?
        .build()
        .await
        .context("building Manager proxy")?;
    Ok(proxy)
}

/// One-shot fetch of a single snapshot — the `--cli` path. No TUI, no subscribe.
pub async fn fetch_once(opts: &ConnOpts) -> Result<StatsSnapshot> {
    let conn = connect(opts).await?;
    let proxy = build_proxy(&conn, opts).await?;
    let json = proxy.stats_snapshot().await.context("calling stats_snapshot()")?;
    StatsSnapshot::from_json_str(&json).context("parsing snapshot JSON")
}

/// The long-running client loop: reconnect forever, subscribing to `StatsStream`
/// and polling `stats_snapshot()` as a fallback, pushing every update to `tx`.
/// Returns only if the channel receiver is dropped (app exiting).
pub async fn run(opts: ConnOpts, tx: UnboundedSender<Update>) {
    loop {
        match serve_once(&opts, &tx).await {
            Ok(()) => return, // receiver gone → app exiting
            Err(why) => {
                if tx.send(Update::Disconnected(why.to_string())).is_err() {
                    return;
                }
                tokio::time::sleep(RECONNECT_DELAY).await;
            }
        }
    }
}

/// One connected session: connect, prime with a poll, subscribe, then race the
/// signal stream against the poll timer. Returns `Ok(())` only when `tx` is
/// closed; any transport error bubbles up so [`run`] reconnects.
async fn serve_once(opts: &ConnOpts, tx: &UnboundedSender<Update>) -> Result<()> {
    let conn = connect(opts).await?;
    let proxy = build_proxy(&conn, opts).await?;
    if tx.send(Update::Connected).is_err() {
        return Ok(());
    }

    // Prime immediately so the UI shows data without waiting for the first tick.
    if let Ok(json) = proxy.stats_snapshot().await {
        if let Ok(snap) = StatsSnapshot::from_json_str(&json) {
            if tx.send(Update::Snapshot(snap)).is_err() {
                return Ok(());
            }
        }
    }

    let mut signals = proxy.receive_stats_stream().await.context("subscribing to StatsStream")?;
    let mut poll = tokio::time::interval(POLL_INTERVAL);
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_signal = Instant::now();

    loop {
        tokio::select! {
            sig = signals.next() => {
                let Some(sig) = sig else {
                    // Signal stream ended → treat as a disconnect and reconnect.
                    anyhow::bail!("StatsStream ended");
                };
                if let Ok(args) = sig.args() {
                    last_signal = Instant::now();
                    if let Ok(snap) = StatsSnapshot::from_json_str(&args.snapshot) {
                        if tx.send(Update::Snapshot(snap)).is_err() {
                            return Ok(());
                        }
                    }
                }
            }
            _ = poll.tick() => {
                // Only poll when signals have gone quiet, so a healthy stream is
                // not doubled up — but a stalled stream still yields ≥2 Hz.
                if last_signal.elapsed() >= SIGNAL_GRACE {
                    let json = proxy.stats_snapshot().await.context("polling stats_snapshot()")?;
                    if let Ok(snap) = StatsSnapshot::from_json_str(&json) {
                        if tx.send(Update::Snapshot(snap)).is_err() {
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
}
