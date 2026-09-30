// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reference-exact image preprocessing.
//!
//! The upstream pipeline is: PIL `Image.resize(BICUBIC)` on uint8 RGB →
//! `ToTensor()` (`/255`) → center-crop → (inside the model) ImageNet
//! mean/std. The resize is `imaging::host::resize_bicubic_pil`, Pillow's
//! fixed-point resampler byte for byte (T1 gates it against a PIL golden);
//! this module sizes it.

/// `_calculate_resize_dims` (crop strategy): longest side → `target`, the
/// other side scaled and rounded to a multiple of `patch`.
pub fn resize_dims(orig_w: usize, orig_h: usize, target: usize, patch: usize) -> (usize, usize) {
    if orig_w >= orig_h {
        let new_h = ((orig_h as f64 * (target as f64 / orig_w as f64) / patch as f64).round()
            as usize)
            * patch;
        (target, new_h)
    } else {
        let new_w = ((orig_w as f64 * (target as f64 / orig_h as f64) / patch as f64).round()
            as usize)
            * patch;
        (new_w, target)
    }
}

/// `compute_adaptive_target_size`: min(longest edge, cap) floored to /patch.
pub fn adaptive_target(orig_w: usize, orig_h: usize, cap: usize, patch: usize) -> usize {
    let effective = orig_w.max(orig_h).min(cap) / patch * patch;
    effective.max(patch * 2)
}
