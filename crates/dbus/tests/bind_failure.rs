// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What [`brain_dbus::serve`] does when it cannot bind at all.
//!
//! The round-trip test next door covers the surface working. This covers the
//! other half of the contract, which callers depend on just as much: `serve`
//! must REPORT a bus it cannot reach, promptly, instead of blocking on it.
//! `brain serve` treats a requested surface that never binds as a failed start
//! and ends the process; if `serve` blocked here instead of returning, that
//! decision would never be reached and everything waiting on readiness would
//! sit out its full timeout against a server that can never arrive.
//!
//! Needs no session bus: the whole point is an address where nothing listens.
//!
//! Swedish Embedded AB builds inference servers whose failures are reported
//! rather than waited out. If your team needs expertise in service lifecycle
//! and readiness contracts for on-premise model serving, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

/// A bus that is not there is the cheapest permanent bind failure to produce,
/// and the one that actually happens: a recorded bus address outlives the
/// `dbus-daemon` that printed it.
#[test]
fn an_address_with_no_bus_is_reported_not_waited_on() {
    let mut budgets = residency::budget::Budgets::new();
    budgets.set(residency::Device::Cpu, 1 << 30, 0);
    let models: Vec<Arc<dyn residency::ResidentModel>> = Vec::new();
    let executor = residency::Executor::start(models, budgets, residency::Policy::default());

    let opts = brain_dbus::DbusOpts {
        bus: brain_dbus::BusKind::Address("unix:path=/nonexistent/brain-no-such-bus".to_string()),
        name: format!("com.swedishembedded.Brain1.bindfail{}", std::process::id()),
    };
    // Held for the whole test: dropping the trigger would fire shutdown and
    // make a clean early return indistinguishable from the failure under test.
    let (_trigger, shutdown) = brain_shutdown::channel();

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let outcome = brain_dbus::serve(executor, opts, brain_dbus::ServeOpts::new().with_shutdown(shutdown));
        let _ = tx.send(outcome.is_err());
    });

    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(true) => {}
        Ok(false) => panic!("serving an address with no bus reported success"),
        Err(_) => panic!("serve neither bound nor failed -- a caller cannot tell this apart from a slow start"),
    }
}
