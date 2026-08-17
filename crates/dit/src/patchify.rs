// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Space-to-depth patchify/unpatchify over a `[C,F,H,W]` latent, inner order
//! **channel-innermost** `[pF,pH,pW,C]` - diffusers' `_patchify_image`/
//! `_unpatchify` convention (Z-Image), and (at `pf=1`) the row layout a DiT
//! head's own linear projection emits (`view(*patch_size, c)`, Wan).
//!
//! This is only ONE of the two space-to-depth orderings Wan needs. Its
//! `patch_embedding` is `Conv3d(in, dim, (1,2,2), stride=(1,2,2))`: kernel
//! equals stride and the temporal extent is 1, so it is also a per-frame 2x2
//! space-to-depth, but a `Conv3d` weight row flattens **channel-outermost**
//! (`[c][kh][kw]`), the opposite of the head's own output row. That forward-
//! direction patchify stays in `wan::model` (see its module doc, `the_two_
//! patch_orderings_are_not_the_same`) - folding a second, incompatible inner
//! order into this function's signature would just relocate that one
//! caller's special case into a shared crate instead of removing it, so it is
//! deliberately left where it is. Wan's OWN unpatchify, however, IS this
//! ordering (at `pf=1`) and is what [`unpatchify`] replaces.

/// `[C,F,H,W] -> [tokens, pF·pH·pW·C]`. Tokens walk the patch grid `(f,h,w)`
/// row-major (frame slowest, width fastest - the same order RoPE's grid ids
/// use), the patch inner order is `[pF,pH,pW,C]` (channel innermost).
pub fn patchify(latent: &[f32], c: usize, f: usize, h: usize, w: usize, pf: usize, ph: usize, pw: usize) -> Vec<f32> {
    let (ft, ht, wt) = (f / pf, h / ph, w / pw);
    let patch = pf * ph * pw * c;
    let mut out = vec![0f32; ft * ht * wt * patch];
    for fi in 0..ft {
        for hi in 0..ht {
            for wi in 0..wt {
                let tok = ((fi * ht + hi) * wt + wi) * patch;
                for a in 0..pf {
                    for b in 0..ph {
                        for d in 0..pw {
                            for ci in 0..c {
                                let src = ((ci * f + (fi * pf + a)) * h + (hi * ph + b)) * w + (wi * pw + d);
                                out[tok + ((a * ph + b) * pw + d) * c + ci] = latent[src];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`]: `[tokens, pF·pH·pW·C] -> [C,F,H,W]`. `(ft,ht,wt)`
/// is the patch grid `patchify` would have produced for the original
/// `(f,h,w)` - i.e. the caller supplies the grid it already knows rather than
/// the full extent, which is what lets a caller with only the grid (Wan,
/// whose `patch_grid` never reconstructs the pre-patch extent) call this
/// without back-computing `f,h,w` first.
pub fn unpatchify(tokens: &[f32], c: usize, ft: usize, ht: usize, wt: usize, pf: usize, ph: usize, pw: usize) -> Vec<f32> {
    let (f, h, w) = (ft * pf, ht * ph, wt * pw);
    let patch = pf * ph * pw * c;
    let mut out = vec![0f32; c * f * h * w];
    for fi in 0..ft {
        for hi in 0..ht {
            for wi in 0..wt {
                let tok = ((fi * ht + hi) * wt + wi) * patch;
                for a in 0..pf {
                    for b in 0..ph {
                        for d in 0..pw {
                            for ci in 0..c {
                                let v = tokens[tok + ((a * ph + b) * pw + d) * c + ci];
                                out[((ci * f + (fi * pf + a)) * h + (hi * ph + b)) * w + (wi * pw + d)] = v;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `patchify` then `unpatchify` at the same shape is the identity - unlike
    /// Wan's forward-direction `patchify`, this ordering IS its own inverse.
    #[test]
    fn patchify_and_unpatchify_are_inverses() {
        let (c, f, h, w, pf, ph, pw) = (2usize, 2usize, 4usize, 4usize, 1usize, 2usize, 2usize);
        let x: Vec<f32> = (0..(c * f * h * w)).map(|i| i as f32).collect();
        let toks = patchify(&x, c, f, h, w, pf, ph, pw);
        let back = unpatchify(&toks, c, f / pf, h / ph, w / pw, pf, ph, pw);
        assert_eq!(back, x);
    }

    /// A temporal patch (`pf=2`, Z-Image's `f_patch_size`) folds two frames
    /// into one token, still channel-innermost, still its own inverse.
    #[test]
    fn temporal_patch_is_also_its_own_inverse() {
        let (c, f, h, w, pf, ph, pw) = (1usize, 4usize, 2usize, 2usize, 2usize, 1usize, 1usize);
        let x: Vec<f32> = (0..(c * f * h * w)).map(|i| i as f32).collect();
        let toks = patchify(&x, c, f, h, w, pf, ph, pw);
        assert_eq!(toks.len(), (f / pf) * (h / ph) * (w / pw) * pf * ph * pw * c);
        let back = unpatchify(&toks, c, f / pf, h / ph, w / pw, pf, ph, pw);
        assert_eq!(back, x);
    }
}
