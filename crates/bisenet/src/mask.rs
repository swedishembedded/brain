// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The background-whiten / face-grayscale mask PuLID builds from BiSeNet's
//! class map - transcribed verbatim from the official
//! `pulid/pipeline_flux.py::PuLIDPipeline.get_id_embedding`:
//!
//! ```python
//! parsing_out = parsing_out.argmax(dim=1, keepdim=True)
//! bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
//! bg = sum(parsing_out == i for i in bg_label).bool()
//! white_image = torch.ones_like(input)
//! face_features_image = torch.where(bg, white_image, self.to_gray(input))
//! ```
//!
//! A per-pixel argmax over 19 classes plus an elementwise select is cheap
//! (512x512, once per identity, not once per denoise step) and has no
//! natural batch/spatial-reduction shape a device kernel would help with -
//! host-side, matching this workspace's convention for exactly this kind of
//! infrequent postprocessing (NMS, etc).

/// The 19 BiSeNet classes upstream treats as "not face" for this mask -
/// background (0) plus hat/neck/necklace/cloth/earring/ear-adjacent classes.
/// The exact list, not a semantic re-derivation of it.
pub const BG_LABELS: [u32; 8] = [0, 16, 18, 7, 8, 9, 14, 15];

/// `logits`: `[19, H, W]` BiSeNet class logits (any monotonic score works -
/// only the per-pixel ARGMAX is read, so pre- or post-softmax is the same
/// answer). `orig`: `[3, H, W]` RGB `[0,1]`, the SAME un-normalized aligned
/// crop that was imagenet-normalized before being fed to BiSeNet (masking
/// reads the original pixels, never the normalized copy - the reference's
/// `input` variable is deliberately the pre-normalization tensor).
///
/// Returns `[3, H, W]` RGB `[0,1]`: background pixels are pure white
/// `(1,1,1)`; face-region pixels are `to_gray(orig)` (`0.299R + 0.587G +
/// 0.114B`, replicated across all three channels) - never the class-colored
/// map itself, which would be a segmentation VISUALIZATION, not this mask.
pub fn whiten_and_gray(logits: &[f32], orig: &[f32], h: u32, w: u32) -> Vec<f32> {
    let (n_class, hw) = (19usize, (h * w) as usize);
    assert_eq!(logits.len(), n_class * hw, "logits must be [19, H, W]");
    assert_eq!(orig.len(), 3 * hw, "orig must be [3, H, W]");

    let mut out = vec![0.0f32; 3 * hw];
    for p in 0..hw {
        let cls = (0..n_class as u32)
            .max_by(|&a, &b| logits[a as usize * hw + p].total_cmp(&logits[b as usize * hw + p]))
            .expect("n_class > 0");
        if BG_LABELS.contains(&cls) {
            out[p] = 1.0;
            out[hw + p] = 1.0;
            out[2 * hw + p] = 1.0;
        } else {
            let (r, g, b) = (orig[p], orig[hw + p], orig[2 * hw + p]);
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            out[p] = gray;
            out[hw + p] = gray;
            out[2 * hw + p] = gray;
        }
    }
    out
}
