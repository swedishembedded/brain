// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The client side of [`crate::service`]'s `com.swedishembedded.Brain1.Manager`
//! interface.
//!
//! Lives beside the service it mirrors, deliberately: a proxy declaration is a
//! second copy of a method's signature, and the only reliable defence against
//! the two drifting is for them to be edited in the same file tree, compiled
//! by the same build, and reviewed together.
//!
//! Not every client can use this. `braintop` keeps its own two-method proxy
//! because depending on this crate would drag `residency` and the whole model
//! layer into a terminal UI -- a layering regression the crate-layers gate
//! exists to prevent. This module serves the clients that ALREADY depend on
//! brain's serving stack, `brain-cli` first among them.
//!
//! Swedish Embedded AB implements Linux system integration for teams putting
//! AI into a product rather than behind a web API. If your team needs
//! expertise in bus services and local IPC around an inference daemon, you can
//! procure our services by sending an email to info@swedishembedded.com.

use crate::{BusKind, DbusOpts, OBJECT_PATH};

/// The subset of `com.swedishembedded.Brain1.Manager` a control client needs.
///
/// Read-only by design: running an action is what the fd protocol in
/// [`crate::service`] is for, and it needs far more than a proxy declaration.
#[zbus::proxy(
    interface = "com.swedishembedded.Brain1.Manager",
    default_service = "com.swedishembedded.Brain1",
    default_path = "/com/swedishembedded/Brain1"
)]
pub trait Manager {
    /// What would happen if this ran here -- a `residency::RunPlan` as JSON.
    /// See `Manager::plan` in [`crate::service`] for what it does and does
    /// not promise.
    fn plan(&self, model: &str, action: &str, params: &str) -> zbus::Result<String>;

    /// The served model names.
    fn list_models(&self) -> zbus::Result<Vec<String>>;

    /// Every model's manifest as a JSON array.
    fn manifests(&self) -> zbus::Result<String>;
}

/// Blocking one-shot: connect, ask for a plan, hand back its JSON.
///
/// Owns its own current-thread runtime, exactly as [`crate::serve`] owns the
/// multi-threaded one -- so a synchronous caller (`brain plan`, and anything
/// else that is a plain command rather than a server) needs no async runtime
/// and no tokio dependency of its own.
///
/// # Errors
///
/// A human-readable message naming which step failed: starting a runtime,
/// reaching the bus, or the daemon's own refusal (an unknown model or action
/// arrives here as the `residency::PlanError`'s own wording).
pub fn plan_blocking(
    opts: &DbusOpts,
    model: &str,
    action: &str,
    params: &str,
) -> Result<String, String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("starting a runtime: {e}"))?;
    rt.block_on(async {
        let proxy = connect(opts)
            .await
            .map_err(|e| format!("connecting to {}: {e}", opts.name))?;
        proxy
            .plan(model, action, params)
            .await
            .map_err(|e| e.to_string())
    })
}

/// Connects to a running brain daemon and builds a `Manager` proxy.
///
/// Takes the SAME [`DbusOpts`] the serving side does, so "where does the
/// daemon live" is described once for both halves and a `--bus`/`--name` flag
/// means the same thing whichever end of the connection a caller is on.
///
/// # Errors
///
/// A `zbus::Error` when the bus itself is unreachable, or when no process owns
/// the requested name. Both are left distinct rather than flattened: they are
/// the two cases a caller has to tell a user apart ("no bus here" versus "no
/// brain running on it").
pub async fn connect(opts: &DbusOpts) -> zbus::Result<ManagerProxy<'static>> {
    let conn = match &opts.bus {
        BusKind::Session => zbus::Connection::session().await?,
        BusKind::System => zbus::Connection::system().await?,
        BusKind::Address(addr) => {
            zbus::connection::Builder::address(addr.as_str())?
                .build()
                .await?
        }
    };
    ManagerProxy::builder(&conn)
        .destination(opts.name.clone())?
        .path(OBJECT_PATH)?
        .build()
        .await
}
