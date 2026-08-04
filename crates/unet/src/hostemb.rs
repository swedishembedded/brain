// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The one genuinely host-side piece of the UNet's conditioning that is not
//! already shared: SDXL's added-conditioning concat.
//!
//! It qualifies under AGENTS.md's host-math rule ("`m=1` decode steps,
//! references and glue"): it is a memcpy of 2816 floats once per forward.
//! Everything downstream of it — the two MLPs, the per-resnet `time_emb_proj`,
//! the broadcast add — runs on the device.
//!
//! The sinusoid itself is **`model::hostmath::timestep_embedding`** and is not
//! reimplemented here. This module originally carried its own copy on the
//! argument that `flip_sin_to_cos` / `downscale_freq_shift` are diffusers
//! conventions rather than general math; they are not — the shared function's
//! own doc already claimed to be diffusers'
//! `Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)`, so the two
//! knobs belong on it and the copy was deleted.

use model::hostmath::timestep_embedding;

/// SDXL's `text_time` added conditioning, exactly as
/// `UNet2DConditionModel.get_aug_embed` builds it:
///
/// ```text
/// time_embeds = add_time_proj(time_ids.flatten()).reshape(B, -1)
/// add_embeds  = concat([text_embeds, time_embeds], dim=-1)
/// ```
///
/// So the POOLED TEXT COMES FIRST and the six micro-conditioning sinusoids
/// follow in `time_ids` order — `(original_h, original_w, crop_top, crop_left,
/// target_h, target_w)`. The reverse order is the classic mistake here: it type-
/// checks (2816 either way), trains nothing, and silently changes composition
/// and crop framing at inference. Verified against the reference, not assumed.
pub fn added_cond(pooled: &[f32], time_ids: &[f32], dim: u32, flip_sin_to_cos: bool, freq_shift: f32) -> Vec<f32> {
    let mut v = pooled.to_vec();
    for &id in time_ids {
        v.extend_from_slice(&timestep_embedding(
            id,
            dim as usize,
            flip_sin_to_cos,
            freq_shift as f64,
            10_000.0,
        ));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn added_cond_puts_pooled_text_first() {
        let pooled = vec![7.0f32; 4];
        let v = added_cond(&pooled, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 8, true, 0.0);
        assert_eq!(v.len(), 4 + 6 * 8);
        assert_eq!(&v[..4], &pooled[..]);
        // Each of the six blocks is the t=0 embedding.
        for k in 0..6 {
            assert_eq!(v[4 + k * 8], 1.0, "block {k} does not start with cos(0)");
            assert_eq!(v[4 + k * 8 + 4], 0.0, "block {k} sin half is not 0");
        }
    }
}
