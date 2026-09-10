// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-iteration device objects that reclaim their own memory.
//!
//! Swedish Embedded AB implements device-memory lifetime management for
//! inference engines for its clients. If your team needs expertise in GPU
//! allocator behaviour, weight streaming and per-layer residency then you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! ## The failure this removes
//!
//! A model too big to hold resident runs its stack one layer at a time:
//! build layer `l` (which uploads that layer's weights into fresh device
//! buffers), run it, throw it away, build layer `l+1`. Written the obvious
//! way that loop leaks the whole model:
//!
//! ```ignore
//! for l in 0..n {
//!     let layer = build_layer(&gpu, l);   // fresh device buffers
//!     x = layer.forward(&x);
//! }                                        // `layer` dropped here...
//! ```
//!
//! Dropping the last host handle to a device buffer does NOT return its
//! memory. wgpu may only reclaim a buffer once the GPU is known to be done
//! with every submission that could still reference it, which it learns from
//! a completed [`Gpu::poll_wait`]. Until then the bytes are merely *pending*
//! (`backend_api::Backend::pending_reclaim_bytes`), and a loop that never
//! polls accumulates every layer it has ever run - a real 12B text encoder
//! and a real 22B DiT both reached multiple gigabytes of abandoned-but-live
//! device memory this way, on the same day, in two different crates.
//!
//! ## Why the obvious fix is not the fix
//!
//! The trap, and the reason this module exists rather than a lesson entry
//! alone: adding the poll *inside* the loop body looks right and does
//! nothing.
//!
//! ```ignore
//! for l in 0..n {
//!     let layer = build_layer(&gpu, l);
//!     x = layer.forward(&x);
//!     gpu.poll_wait();                     // WRONG - `layer` is still alive
//! }
//! ```
//!
//! At the moment that poll runs, `layer` has not been dropped: it is still a
//! live binding, and its buffers are not pending anything. The poll therefore
//! waits on a device that has nothing new to hand back, `layer` drops
//! immediately afterwards, and its bytes sit pending until the NEXT poll -
//! which, on the next iteration, comes only after that iteration's weights
//! have already been allocated. The freed bytes land one iteration late,
//! forever, and the ceiling check in `backend_wgpu::WgpuBackend::track` still
//! fires. Order is the whole content of the fix: **drop, then poll**.
//!
//! ## What to use
//!
//! [`Transient`] is the answer for the ordinary case - one object per
//! iteration that owns that iteration's device buffers. It is an RAII guard
//! that drops its payload and *then* polls, so the order cannot be written
//! wrongly, and it is what a per-layer constructor returns:
//!
//! ```ignore
//! // `LtxBlock::on` returns `Transient<'_, LtxBlock>`; there is no way to
//! // obtain a bare `LtxBlock`, so there is no way to write the loop above.
//! for l in lo..hi {
//!     let blk = LtxBlock::on(gpu, &cfg, &w, l, t, ctx);
//!     x = blk.forward(&x);
//! }                                        // drop, THEN poll - by type
//! ```
//!
//! That is the load-bearing property: the type of the constructor, not the
//! discipline of the caller, is what keeps the loop correct. Nothing about
//! holding two of them alive at once (a residual that spans two layers, a
//! parity test comparing block `l` in two tiers) is restricted - each guard
//! reclaims when it individually goes out of scope.
//!
//! [`reclaiming`] covers the case with no single owning object: a loop body
//! that allocates several ad-hoc device buffers directly. It takes the body
//! as a closure so every buffer the body created is dropped, by the language,
//! before the poll runs - the same guarantee without a payload type.
//!
//! ## What NOT to use it for
//!
//! Weights that are deliberately RESIDENT - built once, kept for the life of
//! the model - are not transient and must not be wrapped: nothing about them
//! is ever pending, and polling per layer at build time only serialises the
//! upload pipeline. This is for objects a loop throws away.

use crate::Gpu;

/// A device object that belongs to ONE iteration: dropped, and its device
/// memory actually reclaimed, when it goes out of scope.
///
/// Constructed by whatever builds the per-iteration object (see the module
/// doc: per-layer constructors return this instead of the bare layer, which
/// is what makes the unsafe loop unwritable). Derefs to the payload, so a
/// caller calls `blk.forward(..)` exactly as before and never names this type.
///
/// There is deliberately no way to move the payload back out. A caller that
/// could do so could keep it past the guard, which is precisely the bug -
/// and a caller that genuinely wants the object to outlive the iteration
/// wants a resident model, not a transient one.
pub struct Transient<'g, T> {
    /// `Option` only so [`Drop`] can take the payload and destroy it BEFORE
    /// polling. Never `None` while the guard is alive.
    obj: Option<T>,
    gpu: &'g Gpu,
}

impl<'g, T> Transient<'g, T> {
    /// Wrap `obj` - built on `gpu`, owning device buffers of `gpu`'s device -
    /// so that dropping it reclaims those buffers.
    ///
    /// `gpu` must be a handle on the same device the payload allocated from
    /// (a [`Gpu::share`] of it is fine, and is what a constructor that moves
    /// a handle into the payload will have): the poll has to run on the
    /// device whose memory is pending, and a poll on an unrelated device
    /// reclaims nothing.
    pub fn on(gpu: &'g Gpu, obj: T) -> Transient<'g, T> {
        Transient { obj: Some(obj), gpu }
    }
}

impl<T> std::ops::Deref for Transient<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.obj.as_ref().expect("Transient payload is only taken in Drop")
    }
}

impl<T> std::ops::DerefMut for Transient<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.obj.as_mut().expect("Transient payload is only taken in Drop")
    }
}

impl<T> Drop for Transient<'_, T> {
    fn drop(&mut self) {
        // Explicit, and in this order, because the order IS the fix: the
        // payload's device buffers must have no live host handle before the
        // poll runs, or the poll observes nothing to reclaim and the bytes
        // wait for whatever polls next (see the module doc).
        drop(self.obj.take());
        reclaim(self.gpu);
    }
}

/// Run `f`, then reclaim every device buffer it dropped.
///
/// The closure boundary is what orders it: every binding `f` made is dropped
/// when `f` returns, before this polls - so a loop body written as
/// `reclaiming(&gpu, || { .. })` cannot poll too early however its body is
/// arranged. For the common case of ONE per-iteration object, prefer
/// [`Transient`], which pushes the same guarantee into the constructor's type
/// where a caller cannot skip it.
///
/// Whatever `f` returns is returned unchanged - and note that returning a
/// device buffer out of the closure deliberately keeps it alive past the
/// poll, which is correct for a value the next iteration still needs.
pub fn reclaiming<R>(gpu: &Gpu, f: impl FnOnce() -> R) -> R {
    let r = f();
    reclaim(gpu);
    r
}

/// The one poll both forms above perform, so "what reclaiming means" is
/// written once.
///
/// Skipped while a panic is unwinding: [`Gpu::poll_wait`] can itself panic (a
/// wedged or lost device), and a panic raised during an unwind is an
/// unconditional process abort - which would replace a reported failure with
/// a bare `SIGABRT` and lose the original message. A process that is already
/// unwinding is also not one whose next allocation needs the memory.
fn reclaim(gpu: &Gpu) {
    #[cfg(not(target_arch = "wasm32"))]
    if !std::thread::panicking() {
        gpu.poll_wait();
    }
    // wasm: the browser drives the device from its own event loop and there
    // is no blocking poll to call (see `backend_wgpu::WgpuBackend::poll_wait`,
    // which is native-only for the same reason). Model code stays identical
    // on both targets.
    #[cfg(target_arch = "wasm32")]
    let _ = gpu;
}
