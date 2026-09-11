// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reference-image conditioning: the **one** construction of the joint token
//! sequence a FLUX.2 DiT evaluation runs on, shared by generation
//! ([`crate::pipeline`]) and by training ([`crate::modelgrad`],
//! [`crate::finetune`]).
//!
//! FLUX.2 conditions on a reference photograph by VAE-encoding it and
//! **concatenating** its latent tokens onto the image half of the joint
//! sequence, with a distinct t-axis RoPE id (`10·(i+1)` for reference `i`) that
//! separates them from the generated image's own axis. Nothing about that is
//! implied by the weights: it is a convention, held in two places if it is
//! written twice.
//!
//! It *was* written twice. Generation built reference tokens and their position
//! ids; training built neither - `make_flow_batch` passed an empty reference
//! list to `position_ids`, and `Cfg` had no slot for a reference at all - so an
//! adapter trained on a `--ref` workflow was fitted against an input
//! distribution the deployed path never presents. The two halves of that bug
//! (the token layout and the latent packing) are the two halves of this module,
//! and both callers go through them:
//!
//! * [`JointLayout`] - how many tokens of which kind, in which order, with
//!   which position ids and therefore which RoPE tables. [`crate::position_ids`]
//!   is this type's [`JointLayout::ids`].
//! * [`pack_tokens`] - a VAE latent mean to packed DiT tokens `[lh·lw, cin]`,
//!   the transform `Pipeline::encode_image` and `finetune::encode_samples` each
//!   used to spell out for themselves.
//!
//! Swedish Embedded AB implements train/inference-consistent conditioning for
//! diffusion models for its clients. If your team needs expertise in keeping a
//! training pipeline and its deployed sampler provably on the same input
//! distribution, you can procure our services by sending an email to
//! info@swedishembedded.com.

/// The joint token layout of ONE FLUX.2 DiT evaluation: `txt_len` text rows,
/// then the generated image's `lh×lw` latent tokens, then each reference
/// image's own latent grid, in the order the references were supplied.
///
/// This ordering is not a detail: the head reads the FIRST `n_gen` image rows
/// (`Flux2Model::forward_batch`'s `n_pred`), so a layout that put references
/// first would predict a velocity for the photograph instead of for the image
/// being generated.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JointLayout {
    /// Text conditioning rows (the model's fixed text window).
    pub txt_len: usize,
    /// The generated image's latent grid, in tokens (pixels / 16 per axis).
    pub lh: usize,
    pub lw: usize,
    /// Each reference image's latent grid `(h, w)` in tokens, in sequence
    /// order. Empty is the plain caption-only / text-to-image layout.
    pub refs: Vec<(usize, usize)>,
}

impl JointLayout {
    /// A layout with no reference images - text-to-image, and caption-only
    /// training.
    pub fn unpaired(txt_len: usize, lh: usize, lw: usize) -> JointLayout {
        JointLayout { txt_len, lh, lw, refs: Vec::new() }
    }

    /// A layout conditioned on `refs`, each a latent grid `(h, w)` in tokens.
    pub fn with_refs(txt_len: usize, lh: usize, lw: usize, refs: Vec<(usize, usize)>) -> JointLayout {
        JointLayout { txt_len, lh, lw, refs }
    }

    /// Tokens of the image being generated - the rows the head predicts a
    /// velocity for, and the only rows the flow-matching loss is defined on.
    pub fn n_gen(&self) -> usize {
        self.lh * self.lw
    }

    /// Conditioning tokens the references contribute.
    pub fn n_ref(&self) -> usize {
        self.refs.iter().map(|&(h, w)| h * w).sum()
    }

    /// Rows in the image half of the joint sequence: generated **plus**
    /// reference. Attention, the image embedder and the RoPE tables are sized
    /// in these; the head is not.
    pub fn n_img(&self) -> usize {
        self.n_gen() + self.n_ref()
    }

    /// Rows in the whole joint sequence.
    pub fn n(&self) -> usize {
        self.txt_len + self.n_img()
    }

    /// The 4-axis position ids, text rows first.
    ///
    /// Text tokens: `(0,0,0,l)`; generated image: `(0,h,w,0)` raster-major;
    /// reference `i`: `(10·(i+1), h, w, 0)`. The t-axis offset is what keeps a
    /// reference token from colliding with the generated token at the same
    /// spatial position.
    pub fn ids(&self) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.n() * 4);
        for l in 0..self.txt_len {
            ids.extend([0, 0, 0, l as u32]);
        }
        for h in 0..self.lh {
            for w in 0..self.lw {
                ids.extend([0, h as u32, w as u32, 0]);
            }
        }
        for (i, &(rh, rw)) in self.refs.iter().enumerate() {
            let t = 10 * (i as u32 + 1);
            for h in 0..rh {
                for w in 0..rw {
                    ids.extend([t, h as u32, w as u32, 0]);
                }
            }
        }
        ids
    }

    /// The interleaved-RoPE cos/sin tables for this layout, `[n · head_dim/2]`
    /// each.
    pub fn rope(&self, axes_dim: [usize; 4], theta: f64) -> dit::rope::RopeTables {
        let rc = dit::rope::RopeConfig {
            axes_dims: axes_dim.iter().map(|&a| a as u32).collect(),
            axes_lens: vec![4096, 4096, 4096, 4096],
            theta,
        };
        dit::rope::tables_for_ids(&rc, &self.ids(), 4)
    }

    /// The image half of the joint sequence: the generated image's tokens
    /// followed by every reference's, `[n_img · cin]`.
    ///
    /// `gen` is `[n_gen · cin]` (the noised latent under training, the current
    /// latent under sampling) and `refs` is the concatenation of every
    /// reference's tokens in layout order, exactly what repeated
    /// [`pack_tokens`] calls produce.
    pub fn joint_tokens(&self, gen: &[f32], refs: &[f32], cin: usize) -> Vec<f32> {
        assert_eq!(gen.len(), self.n_gen() * cin, "generated tokens");
        assert_eq!(refs.len(), self.n_ref() * cin, "reference tokens");
        let mut out = Vec::with_capacity(self.n_img() * cin);
        out.extend_from_slice(gen);
        out.extend_from_slice(refs);
        out
    }
}

/// A VAE latent **mean** at an `lh8×lw8` 8x-downsampled grid to packed DiT
/// tokens `[(lh8/2)·(lw8/2), cin]`, row-major - the token order
/// [`JointLayout::ids`] assigns position ids in.
///
/// One implementation for generation and training. It is two steps that must
/// agree exactly and are easy to spell differently: `vae::latent::pack` folds
/// the 2×2 pixel-unshuffle and the eval-BatchNorm affine into `[cin, lh, lw]`,
/// and this transposes that to token-major. A transposed second copy is not a
/// crash, it is an adapter trained on a channel-shuffled image.
pub fn pack_tokens(
    mean: &[f32],
    lh8: usize,
    lw8: usize,
    bn_mean: &[f32],
    bn_var: &[f32],
    eps: f32,
    cin: usize,
) -> Vec<f32> {
    // The latent is `cin/4` channels before the 2x2 unshuffle packs it to `cin`.
    let packed = vae::latent::pack(mean, cin / 4, lh8, lw8, bn_mean, bn_var, eps);
    let (lh, lw) = (lh8 / 2, lw8 / 2);
    let mut tokens = vec![0.0f32; lh * lw * cin];
    for c in 0..cin {
        for y in 0..lh {
            for x in 0..lw {
                tokens[(y * lw + x) * cin + c] = packed[(c * lh + y) * lw + x];
            }
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The t-axis offset is the whole point of a reference id: without it a
    /// reference token and the generated token at the same (h, w) would carry
    /// identical position ids and therefore identical RoPE phase, and the
    /// model could not tell the photograph from the canvas.
    #[test]
    fn reference_ids_carry_the_t_axis_offset() {
        let l = JointLayout::with_refs(2, 2, 2, vec![(2, 2), (1, 1)]);
        let ids = l.ids();
        assert_eq!(ids.len(), l.n() * 4);
        let row = |i: usize| &ids[i * 4..(i + 1) * 4];
        assert_eq!(row(0), [0, 0, 0, 0]); // text
        assert_eq!(row(1), [0, 0, 0, 1]);
        assert_eq!(row(2), [0, 0, 0, 0]); // generated (0,0)
        assert_eq!(row(5), [0, 1, 1, 0]); // generated (1,1)
        assert_eq!(row(6), [10, 0, 0, 0]); // reference 0 (0,0)
        assert_eq!(row(9), [10, 1, 1, 0]);
        assert_eq!(row(10), [20, 0, 0, 0]); // reference 1
    }

    /// Counting is the sizing contract every buffer in the trainer and the
    /// denoiser is allocated from; the head's `n_pred` is `n_gen`, never
    /// `n_img`.
    #[test]
    fn the_counts_separate_generated_from_conditioning_rows() {
        let l = JointLayout::with_refs(3, 4, 5, vec![(2, 2), (1, 3)]);
        assert_eq!(l.n_gen(), 20);
        assert_eq!(l.n_ref(), 7);
        assert_eq!(l.n_img(), 27);
        assert_eq!(l.n(), 30);
        let u = JointLayout::unpaired(3, 4, 5);
        assert_eq!((u.n_gen(), u.n_ref(), u.n_img(), u.n()), (20, 0, 20, 23));
    }
}
