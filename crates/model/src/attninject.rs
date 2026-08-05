// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Injecting an extra term into a backbone's **cross-attention**, generically.
//!
//! `crates/controlnet` already solves the other half of this problem: a
//! ControlNet's residuals are handed to the backbone as **pre-supplied device
//! inputs**, which works because a control residual does not depend on anything
//! the backbone computes.
//!
//! An identity adapter is not like that. IP-Adapter-FaceID's ID branch attends
//! the **per-site query** — a tensor produced inside the backbone's own
//! attention block — so its dispatches must be recorded *inside* the graph, at
//! the point where that query exists. This trait is that seam.
//!
//! # Where the contribution goes, and why it is the context
//!
//! The reference (`ip_adapter/attention_processor.py::IPAttnProcessor`) computes
//! the text-conditioned context, adds `scale * ip_context`, and only then applies
//! the block's shared `to_out`:
//!
//! ```text
//!   ctx  = attend(q, k_text, v_text)
//!   ctx += scale * attend(q, k_id, v_id)     <-- [`CrossAttnInject::inject`]
//!   out  = to_out(ctx)
//! ```
//!
//! So [`inject`](CrossAttnInject::inject) is called with the context **before**
//! `to_out` and adds into it in place. Adding after `to_out`, or concatenating
//! the ID tokens onto the text tokens, both run and both produce a plausible
//! image — they are simply not what the weights were trained for.
//!
//! # Backbone-agnostic by construction
//!
//! Nothing here names a UNet, a site count or a channel layout: an implementor
//! receives the query, the context, and the two dimensions it needs. A DiT with
//! per-block cross-attention implements the same trait unchanged.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// An extra attention term a backbone adds into each cross-attention context.
///
/// # The adapter dispatches on the BACKBONE's device
///
/// [`inject`](CrossAttnInject::inject) receives the backbone's [`Gpu`], so it
/// can only dispatch kernels that device was built with. An adapter therefore
/// declares what it needs via [`kernels`](CrossAttnInject::kernels), and the
/// caller builds the device from the UNION of the backbone's set and the
/// adapter's — the same `const fn` append that
/// `facenet::caps::SERVING_PIPELINES` uses to extend `model::PIPELINES` at the
/// tail, which keeps ONE kernel index space and leaves every existing index
/// valid.
///
/// The backbone checks this at construction, so a missing kernel is an error
/// naming it rather than a panic from `Ki::resolve` deep in a forward.
pub trait CrossAttnInject: Send + Sync {
    /// Kernels this adapter dispatches. The backbone verifies its device has
    /// every one before recording.
    fn kernels(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    /// How many cross-attention sites this adapter serves. The backbone asserts
    /// this against its own count, so a checkpoint built for a different
    /// backbone fails at construction with a number rather than mid-forward with
    /// a shape.
    fn sites(&self) -> usize;

    /// Append dispatches that add this adapter's contribution into `ctx`.
    ///
    /// * `k` — the site index, in the backbone's own cross-attention order.
    /// * `q` — that site's queries, `[t, c]`, already projected.
    /// * `ctx` — the context to add into, `[t, c]`, **before** `to_out`.
    ///
    /// The implementor owns any scratch it needs; `ctx` is the only buffer it
    /// may write, and it must ADD rather than overwrite.
    fn inject(&self, steps: &mut Vec<Step>, gpu: &Gpu, k: usize, q: &DeviceBuffer, ctx: &DeviceBuffer, t: u32, c: u32);
}
