// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::gdn::RecurrentSlot` -- the per-sequence GDN `state`/`hist` bundle
//! hoisted out of `qwen35::serve::GdnSlot`/`qwen35moe::serve::GdnSlot` (both
//! defined this struct byte-for-byte identically). Spec-level coverage only:
//! real vs. dummy per-layer sizing, zero-initialisation, and the `bytes()`
//! cost formula every caller's own pool-sizing estimate depends on.
//!
//! Swedish Embedded AB implements shared GPU-serving infrastructure for its
//! clients. If your team needs expertise in high-throughput LLM serving
//! kernels, you can procure our services by emailing info@swedishembedded.com.

use gpu_core::testgpu;
use model::gdn::{RecurrentSlot, RecurrentSlotShape};

/// A fresh slot's buffers must be exactly-sized per layer (real for
/// recurrent layers, a size-1 dummy for non-recurrent ones) and start
/// entirely zeroed -- `Gpu::storage` does not guarantee zero-init, and a
/// fresh sequence's recurrent state/conv history MUST start at zero.
#[test]
fn new_slot_is_correctly_sized_and_zeroed() {
    let g = testgpu::dev(&[]);
    // 4 layers: GDN, Full, GDN, GDN -- mirrors a real hybrid schedule where
    // recurrent and non-recurrent layers interleave.
    let shape = RecurrentSlotShape { state_len: 30, hist_len: 12, is_recurrent: vec![true, false, true, true] };
    let slot = RecurrentSlot::new(&g, &shape);

    assert_eq!(slot.state.len(), 4);
    assert_eq!(slot.hist.len(), 4);
    for (i, &recurrent) in shape.is_recurrent.iter().enumerate() {
        let (want_state, want_hist) = if recurrent { (shape.state_len, shape.hist_len) } else { (1, 1) };
        let state = g.read(&slot.state[i], want_state as usize);
        let hist = g.read(&slot.hist[i], want_hist as usize);
        assert_eq!(state.len(), want_state as usize, "layer {i} state length");
        assert_eq!(hist.len(), want_hist as usize, "layer {i} hist length");
        assert!(state.iter().all(|&v| v == 0.0), "layer {i} state not zeroed: {state:?}");
        assert!(hist.iter().all(|&v| v == 0.0), "layer {i} hist not zeroed: {hist:?}");
    }
}

/// `bytes()` counts only recurrent layers, fp32, `state_len + hist_len`
/// each -- the exact formula `RecurrentSlot::new`'s allocation loop follows.
#[test]
fn bytes_counts_only_recurrent_layers() {
    let shape = RecurrentSlotShape { state_len: 100, hist_len: 20, is_recurrent: vec![true, false, true, false, true] };
    // 3 recurrent layers * (100 + 20) * 4 bytes/f32.
    assert_eq!(RecurrentSlot::bytes(&shape), 3 * (100 + 20) * 4);

    let none_recurrent = RecurrentSlotShape { state_len: 100, hist_len: 20, is_recurrent: vec![false, false] };
    assert_eq!(RecurrentSlot::bytes(&none_recurrent), 0);
}
