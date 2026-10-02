// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What an embedding application can attach to every call: a say in whether it
//! runs, and a record of what it cost.
//!
//! [`RequestHooks::begin`] runs before a call is submitted and may refuse it
//! (out of credit, over a limit, not permitted). What it returns is a
//! [`Ticket`], and the ticket is settled by consuming it, so a call that began
//! is settled exactly once: [`Ticket::finish`] takes `self` by value, and the
//! bridge calls it on every path out -- a result, a failure, a call that was
//! never admitted, and a streamed answer whose client walked away.
//!
//! Nothing here names who the caller is. A [`Principal`](crate::Principal) is
//! whatever the embedder's [`Authenticator`](crate::Authenticator) produced, and
//! the embedder downcasts it back.

use capability::{Invocation, Outcome};

use crate::auth::Principal;
use crate::error::ApiError;
use crate::surface::Provider;

/// What is about to run.
pub struct Call<'a> {
    /// The dialect the call arrived on.
    pub provider: Provider,
    /// Who made it; `None` when the surface authenticates with a static key,
    /// which identifies nobody in particular.
    pub caller: Option<&'a Principal>,
    pub model: &'a str,
    pub action: &'a str,
    /// The call as it will be submitted, for an estimate of its worst case.
    pub invocation: &'a Invocation,
}

/// How a call ended.
pub enum CallResult<'a> {
    /// It ran to completion; the outcome carries what it consumed.
    Done(&'a Outcome),
    /// It was admitted and then failed or was cancelled. The text is for the
    /// embedder's log, never for the caller.
    Failed(&'a str),
    /// It never ran: it was not admitted in time, or its model could not be
    /// fetched.
    Refused,
}

/// A call that began and has not yet been settled.
pub trait Ticket: Send {
    /// Settles the call. An error means the settlement could not be recorded,
    /// and fails the request: an answer nobody is accountable for is not served.
    ///
    /// # Errors
    /// The embedder's refusal to let the call stand.
    fn finish(self: Box<Self>, result: CallResult<'_>) -> Result<(), ApiError>;
}

/// The embedder's view of every call a surface serves.
pub trait RequestHooks: Send + Sync {
    /// Decides whether the call may run, and returns what settles it.
    ///
    /// # Errors
    /// A provider-shaped refusal, sent to the caller as is; the model is never
    /// reached and there is nothing to settle.
    fn begin(&self, call: &Call<'_>) -> Result<Box<dyn Ticket>, ApiError>;
}
