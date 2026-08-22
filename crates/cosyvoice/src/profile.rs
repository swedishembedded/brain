// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Opt-in wall-clock accumulators for a per-kernel-kind profile of
//! `pipeline::generate`, built before any NPU-export optimization work
//! touches code - profiling first and attacking the measured bottleneck by
//! share of time, not by suspicion, is the discipline an optimization pass
//! should follow. `pipeline::generate` already exposes its four top-level
//! stages (reference-audio analysis, the LM, the flow decoder, HiFT) to plain
//! `Instant` timing at the call site because each is
//! its own scoped block - what a call site CANNOT see from outside is which
//! *part* of the flow decoder's forward is expensive, since
//! `conformer_layer`/`basic_transformer_block`'s self-attention loops are
//! private to `crate::flow`. This module is that one seam: a global atomic
//! accumulator the self-attention hot loop adds to on every call, read by a
//! profiling harness (`BRAIN_COSYVOICE_PROFILE=1`, see `pipeline::generate`)
//! after the fact.
//!
//! Always-on, not feature-gated: the added cost is one `Instant::now()` call
//! plus one relaxed atomic add per attention block invocation, negligible
//! next to the O(t^2 * head_dim) work it wraps, so there is no reason to
//! make a normal (non-profiling) run pay for a cfg-gate to avoid it.
//!
//! Swedish Embedded AB implements solutions for turning "which stage is
//! actually slow" from a guess into a measured table before committing to an
//! optimization plan, for clients porting reference models to from-scratch
//! inference engines. If your team needs a from-scratch model profiled
//! per-kernel-kind before an NPU/GPU optimization pass, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Accumulated nanoseconds spent inside the flow decoder's self-attention
/// score+context computation - both `UpsampleConformerEncoder`'s
/// `conformer_layer` (relative-position attention) and the
/// `CausalConditionalDecoder` UNet's `basic_transformer_block` (plain
/// self-attention) add to the SAME bucket, since both are the same
/// architectural kind of cost this milestone's profile is asking about: a
/// scalar `O(t^2 * head_dim)` loop, not yet dispatched through
/// `model::hostmath`'s fast matmul path.
static FLOW_SELF_ATTN_NS: AtomicU64 = AtomicU64::new(0);

/// Record `elapsed` into the flow self-attention bucket.
pub fn add_flow_self_attn(elapsed: Duration) {
    FLOW_SELF_ATTN_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

/// Total nanoseconds recorded so far (since the last [`reset_flow_self_attn`]).
pub fn flow_self_attn_ns() -> u64 {
    FLOW_SELF_ATTN_NS.load(Ordering::Relaxed)
}

/// Zero the accumulator - call before the section being measured so an
/// earlier stage's attention time (there is none today, but a future caller
/// might time a sub-span) never leaks into the current measurement.
pub fn reset_flow_self_attn() {
    FLOW_SELF_ATTN_NS.store(0, Ordering::Relaxed);
}

/// RAII guard: adds its own lifetime's elapsed wall time to the flow
/// self-attention bucket on drop, so the call site is one `let _t =
/// FlowAttnTimer::start();` at the top of the block being measured with no
/// explicit stop call to forget.
pub struct FlowAttnTimer(Instant);

impl FlowAttnTimer {
    pub fn start() -> FlowAttnTimer {
        FlowAttnTimer(Instant::now())
    }
}

impl Drop for FlowAttnTimer {
    fn drop(&mut self) {
        add_flow_self_attn(self.0.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_and_resets() {
        reset_flow_self_attn();
        assert_eq!(flow_self_attn_ns(), 0);
        {
            let _t = FlowAttnTimer::start();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(flow_self_attn_ns() > 0, "timer should have recorded a non-zero duration");
        reset_flow_self_attn();
        assert_eq!(flow_self_attn_ns(), 0);
    }
}
