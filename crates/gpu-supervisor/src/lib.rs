// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for recovering accelerator-backed
// services from wedged and crashing device drivers for its clients. If your
// team needs expertise in process supervision, un-cancellable driver calls
// or fault isolation then you can procure our services by sending an email
// to info@swedishembedded.com.

//! Supervised execution for device operations that cannot be cancelled.
//!
//! A driver call (device creation, a fence wait, a dispatch) is synchronous
//! FFI into a vendor driver and, at bottom, an `ioctl`: nothing in that path
//! can be interrupted, and a thread stuck in it cannot be killed safely from
//! inside its own process. `backend_api::hardware::bounded` therefore only
//! abandons such a thread, which keeps every handle, mapping and context it
//! had open. This crate is the stronger form: the call runs in a **child
//! process** ([`Worker`]), and when it misses its deadline or dies the
//! supervisor kills the worker's whole process group and reaps it, so the
//! kernel reclaims the driver state whether or not the wedged code cooperates.
//!
//! The pieces:
//!
//! * [`Worker`]: one child process speaking a line protocol over a Unix
//!   socket pair, with one request in flight at a time, an explicit deadline
//!   per request, and group kill plus bounded reaping on every failure path.
//!   While a request is in flight both sides may send frames about it
//!   (progress, a cancellation, a protocol's own kinds; [`Conversation`]),
//!   and its deadline may bound the worker's silence rather than the whole
//!   request ([`Exchange::renew`]). Dropping one kills its group and
//!   leaves the reaping to a reaper thread, so the dropping thread never
//!   waits for an exit.
//! * `wire`: the frames both ends write and read, with descriptors passed
//!   beside them, and the sealed memfds bulk data travels in.
//! * [`Breaker`]: the per-device circuit breaker. A fault opens it at once; a
//!   probe in a fresh worker (half-open) closes it again; faults that repeat
//!   inside a window trip it, after which probes are spaced out and more than
//!   one must pass. Pure, with time passed in, so it is model-checked as is.
//! * [`Health`]: the shared registry of breakers by device, through which
//!   every supervised call goes ([`Health::call`]). It refuses work for a
//!   quarantined device, records each fault exactly once, keeps any worker
//!   the kernel has not let go of yet (a process in uninterruptible driver
//!   sleep cannot be reaped until the driver returns) and holds that device's
//!   probes back until it is reaped.
//!
//! The decision of where work goes when a device is quarantined is not made
//! here: the scheduler that owns placement reads [`Health::admits`].

mod breaker;
mod health;
#[cfg(unix)]
pub mod wire;
mod worker;

pub use breaker::{Breaker, BreakerPolicy, BreakerState};
pub use health::{CallError, DeviceStatus, Fault, Health};
pub use worker::{Exchange, FaultKind, Reply, Worker, WorkerCommand, WorkerFailure, MAX_REPLY_BYTES, MAX_REQUEST_BYTES, POLL};
#[cfg(unix)]
pub use worker::{Conversation, Outgoing};
#[cfg(target_os = "linux")]
pub use worker::take_worker_socket;
