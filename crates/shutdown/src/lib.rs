// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Process lifecycle signalling for every serving surface: one shutdown source
//! ([`Shutdown`], end of life) and one readiness latch ([`ready::Gate`], start of
//! life). `brain serve` can run several surfaces at once (an HTTP dialect per
//! port, plus D-Bus), each on its own thread/runtime, and both signals are the
//! same shape for that reason: "has everything started" and "should everything
//! stop" are both an AND/OR over surfaces that only a shared, cross-thread token
//! can express — see [`ready`]'s module docs for the readiness half.
//!
//! `brain serve` can run the D-Bus surface and one or more HTTP surfaces at once,
//! each on its own tokio runtime (`crates/dbus` builds a multi-thread runtime on
//! its own thread; `crates/apiserve` builds one on the calling thread). SIGINT/
//! SIGTERM disposition is **process-wide** — if each surface independently calls
//! `tokio::signal::ctrl_c()`, only the first registration actually receives the
//! signal, and it is unspecified which surface that is. The prior bug in this
//! repo was exactly that: `crates/dbus`'s stats-stream task held a `Connection`
//! clone forever, so `Connection::graceful_shutdown()` awaited a drop event that
//! could never fire — a plain deadlock — and when D-Bus ran on a background
//! thread alongside an HTTP surface, that same deadlocked runtime had already
//! claimed the one SIGINT registration, so Ctrl-C did nothing at all.
//!
//! The fix is one shutdown source, installed once, awaitable from any runtime:
//! [`channel`] returns a [`Trigger`] (fire it from anywhere, no runtime required)
//! and a [`Shutdown`] (clone it into every surface; `await` [`Shutdown::wait`]
//! wherever that surface would otherwise block forever). [`install_signals`]
//! owns the one SIGINT/SIGTERM registration on a dedicated thread, so it does not
//! matter which surface's runtime is built first or which one owns the process's
//! main thread.

pub mod ready;

use tokio::sync::watch;

/// The write side of a shutdown signal. Cheap to clone; [`Trigger::fire`] needs
/// no async runtime, so it can be called from a plain OS thread (this is what
/// [`install_signals`] does).
#[derive(Clone)]
pub struct Trigger {
    tx: watch::Sender<bool>,
}

impl Trigger {
    /// Fire the shutdown signal. Idempotent: firing twice is harmless, and every
    /// [`Shutdown`] clone observes it exactly once (or immediately, if it starts
    /// watching after the fire).
    pub fn fire(&self) {
        let _ = self.tx.send(true);
    }
}

/// The read side of a shutdown signal. `Clone`, and awaitable from any tokio
/// runtime — this is what lets two independently-built runtimes (the D-Bus
/// surface's and the HTTP surface's) share one shutdown source with neither
/// owning the other.
#[derive(Clone)]
pub struct Shutdown {
    rx: watch::Receiver<bool>,
}

impl Shutdown {
    /// A shutdown channel with SIGINT/SIGTERM handling already installed on a
    /// dedicated thread. The common case for a single-surface server.
    pub fn from_signals() -> Shutdown {
        let (trigger, shutdown) = channel();
        install_signals(trigger);
        shutdown
    }

    /// Resolve immediately if the signal has already fired; otherwise wait for it.
    pub async fn wait(&self) {
        let mut rx = self.rx.clone();
        if *rx.borrow() {
            return;
        }
        // `changed()` errors only if every `Trigger` clone has been dropped
        // without ever firing. A `Trigger` returned by `install_signals` is held
        // for the process lifetime by its signal thread's async block, so that
        // path is not reachable from a caller wired the intended way; treat it
        // the same as "never fires" rather than returning spuriously.
        while rx.changed().await.is_ok() {
            if *rx.borrow() {
                return;
            }
        }
        std::future::pending::<()>().await;
    }

    /// Non-blocking check of the current state.
    pub fn is_shutdown(&self) -> bool {
        *self.rx.borrow()
    }
}

/// A fresh, unfired shutdown channel.
/// End the process **now**, with `code`, running no `atexit` handler and no
/// shared-library finalizer - only the standard streams are flushed first.
///
/// The ordinary way for a Rust program to end (returning from `main`, or
/// `std::process::exit`) calls libc `exit(3)`, which runs every registered
/// destructor and then unmaps shared libraries. That is safe only if no thread
/// is still executing code from those libraries - and a process that has
/// talked to a GPU does not control that: the graphics driver keeps worker
/// threads of its own that a program can neither see nor join.
///
/// This is a measured hazard on this engine's hardware, not a precaution. A
/// server that had placed a model across TWO GPUs faulted intermittently on
/// SIGTERM - always inside a driver thread, always as a jump to an address
/// that is no longer mapped (so no Rust frame appears in the trace at all),
/// and never in the same run restricted to ONE GPU. Nothing in that final
/// teardown is needed: the kernel reclaims every allocation, every mapping and
/// every device handle when the process dies either way.
///
/// Use it only as the LAST statement of a process's own shutdown path, once
/// every surface has drained - it skips destructors, so anything that must be
/// flushed or committed has to have happened already.
pub fn exit_now(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(unix)]
    // SAFETY: `_exit` is async-signal-safe and never returns. The streams
    // above are the only buffered state this process owns at this point.
    unsafe {
        libc::_exit(code)
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

pub fn channel() -> (Trigger, Shutdown) {
    let (tx, rx) = watch::channel(false);
    (Trigger { tx }, Shutdown { rx })
}

/// Install SIGINT/SIGTERM handling on one dedicated OS thread and fire `trigger`
/// on the first of either. A **second** signal after that exits the process
/// immediately with status 130 — a wedged shutdown must always be escapable by a
/// human at the terminal, and a test harness must never need `kill -9`.
///
/// Owning the registration on its own thread — rather than inside whichever
/// runtime happens to call a signal future first — is the point: call this once,
/// before building any surface's runtime, and delivery no longer depends on
/// which surfaces are active or in what order they start.
///
/// **Blocks the calling thread until the OS-level handler is actually
/// installed** (or a short bound elapses). This closes a real race, not a
/// theoretical one: a non-interactive shell backgrounding a command (`cmd &` in
/// a script, which is exactly how every serving surface here is started —
/// including by the very bats harness that caught this) sets SIGINT to `SIG_IGN`
/// for that child *before* `exec`ing it, and `SIG_IGN` survives `exec`. Our own
/// `sigaction()` call overrides that inherited disposition — but only once it
/// has actually run. Before this function waited for that confirmation, a
/// signal arriving in the gap between "thread spawned" and "handler installed"
/// (a gap easily lost against a fast HTTP listener bind) was silently
/// swallowed: not delivered to us, and not defaulted to terminating the
/// process either, because it was already ignored. The fix registers via the
/// synchronous [`tokio::signal::unix::signal`] (which installs eagerly, at call
/// time — unlike `tokio::signal::ctrl_c()`, an `async fn` that only registers on
/// its first poll) for *both* signals, and acks over a channel before returning.
pub fn install_signals(trigger: Trigger) {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    let spawned = std::thread::Builder::new().name("brain-signals".to_string()).spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread().enable_io().build() {
            Ok(rt) => rt,
            Err(err) => {
                eprintln!("brain: could not start the signal-handling runtime ({err}); Ctrl-C/SIGTERM will not stop the server cleanly");
                drop(ready_tx); // wake the waiting caller with a closed channel, not a hang
                return;
            }
        };
        rt.block_on(signal_loop(trigger, ready_tx));
    });
    match spawned {
        Ok(_) => {
            // Bounded: if the thread panicked before acking, `ready_tx` drops and
            // `recv` returns `Err` immediately — this is not a wait for a slow
            // machine, it is a wait for one `sigaction()` call, so a 2s cap is
            // generous headroom, not a real limit in the healthy case.
            if ready_rx.recv_timeout(std::time::Duration::from_secs(2)).is_err() {
                eprintln!("brain: signal handler installation did not confirm in time; Ctrl-C/SIGTERM may not stop the server cleanly");
            }
        }
        Err(err) => eprintln!("brain: could not spawn the signal-handling thread ({err}); Ctrl-C/SIGTERM will not stop the server cleanly"),
    }
}

#[cfg(unix)]
async fn signal_loop(trigger: Trigger, ready: std::sync::mpsc::Sender<()>) {
    use tokio::signal::unix::{signal, SignalKind};
    // Both via the synchronous, eagerly-registering `signal()` — NOT
    // `tokio::signal::ctrl_c()`, whose registration is deferred to its first
    // poll and would reopen exactly the race this function exists to close.
    let mut sigint = match signal(SignalKind::interrupt()) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("brain: could not install a SIGINT handler ({err}); Ctrl-C/SIGTERM will not stop the server cleanly");
            drop(ready); // ack failure by closing the channel — the caller does not hang
            return;
        }
    };
    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("brain: could not install a SIGTERM handler ({err}); Ctrl-C/SIGTERM will not stop the server cleanly");
            drop(ready);
            return;
        }
    };
    // Both handlers are live: safe to let the caller proceed now.
    let _ = ready.send(());

    let first = tokio::select! {
        _ = sigint.recv() => "SIGINT",
        _ = sigterm.recv() => "SIGTERM",
    };
    eprintln!("brain: received {first}, shutting down (send it again to force-exit)");
    trigger.fire();
    tokio::select! {
        _ = sigint.recv() => {}
        _ = sigterm.recv() => {}
    }
    eprintln!("brain: received a second signal, force-exiting");
    std::process::exit(130);
}

#[cfg(not(unix))]
async fn signal_loop(trigger: Trigger, ready: std::sync::mpsc::Sender<()>) {
    // No SIGTERM concept off Unix, and `ctrl_c()` is the only handle available —
    // its registration is deferred to first poll, so the ack here is best-effort
    // rather than a proof the handler is live (this platform is not brain's
    // primary target; see `docs/engine/*` for the supported platform list).
    let _ = ready.send(());
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("brain: received Ctrl-C, shutting down (send it again to force-exit)");
    trigger.fire();
    let _ = tokio::signal::ctrl_c().await;
    eprintln!("brain: received a second Ctrl-C, force-exiting");
    std::process::exit(130);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn wait_returns_immediately_once_already_fired() {
        let (trigger, shutdown) = channel();
        trigger.fire();
        tokio::time::timeout(Duration::from_millis(200), shutdown.wait()).await.expect("wait must return immediately once fired");
    }

    #[tokio::test]
    async fn wait_wakes_when_fired_later() {
        let (trigger, shutdown) = channel();
        assert!(!shutdown.is_shutdown());
        let waiter = tokio::spawn(async move {
            shutdown.wait().await;
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        trigger.fire();
        tokio::time::timeout(Duration::from_millis(200), waiter).await.expect("wait must wake once fired").unwrap();
    }

    #[tokio::test]
    async fn clones_are_independent_but_share_the_signal() {
        let (trigger, shutdown) = channel();
        let a = shutdown.clone();
        let b = shutdown.clone();
        assert!(!a.is_shutdown());
        assert!(!b.is_shutdown());
        trigger.fire();
        tokio::time::timeout(Duration::from_millis(200), a.wait()).await.unwrap();
        tokio::time::timeout(Duration::from_millis(200), b.wait()).await.unwrap();
        assert!(a.is_shutdown());
        assert!(b.is_shutdown());
    }

    #[tokio::test]
    async fn never_fired_wait_does_not_return() {
        let (_trigger, shutdown) = channel();
        let res = tokio::time::timeout(Duration::from_millis(50), shutdown.wait()).await;
        assert!(res.is_err(), "wait must not return before the trigger fires");
    }

    #[test]
    fn is_shutdown_transitions() {
        let (trigger, shutdown) = channel();
        assert!(!shutdown.is_shutdown());
        trigger.fire();
        assert!(shutdown.is_shutdown());
    }
}
