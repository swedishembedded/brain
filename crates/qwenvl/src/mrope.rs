// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Interleaved multi-axis RoPE (M-RoPE) for Qwen3-VL — a pure host table builder.
//!
//! Qwen3-VL's only positional novelty is that each rotary channel is driven by
//! one of three position axes (T, H, W) in an **interleaved** layout, rather than
//! the single sequential position of ordinary RoPE. Because the rotation itself
//! is the standard half-split `rotate_half`, this needs **no new device kernel**:
//! the host builds the per-token `(cos, sin)` tables with the right axis feeding
//! each channel, and the existing `rope2d.wgsl` (table-driven, half-split, pairs
//! `(d, d+half)`) applies them — exactly as `crates/dit` does for Z-Image, but
//! with the interleaved stride-3 slot→axis map instead of contiguous blocks.
//!
//! Layout (matching HF `apply_interleaved_mrope`, `mrope_section = [T,H,W]`, sum
//! = head_dim/2): start every channel on axis T, then for axis H overwrite
//! channels `1, 4, 7, …` and for axis W `2, 5, 8, …`, each for its section count.
//! Net: channel `d` is owned by axis `d % 3` until the H/W sections are exhausted,
//! after which the tail belongs to T. For 4B `[24,20,20]`: T = {0,3,…,57} ∪
//! {60,61,62,63} (24), H = {1,4,…,58} (20), W = {2,5,…,59} (20).

/// Per-channel axis owner (0 = T, 1 = H, 2 = W) over the `half = head_dim/2`
/// rotary channels, from the interleaved `mrope_section` layout.
pub fn axis_map(mrope_section: [u32; 3], half: usize) -> Vec<usize> {
    let mut m = vec![0usize; half]; // default: temporal
    for axis in 1..3usize {
        for i in 0..mrope_section[axis] as usize {
            let slot = axis + 3 * i;
            if slot < half {
                m[slot] = axis;
            }
        }
    }
    m
}

/// Build the interleaved-M-RoPE `(cos, sin)` tables for a sequence of 3-axis
/// position ids. `positions[t] = [t_pos, h_pos, w_pos]`; returns row-major
/// `[seq, half]` cos and sin (`half = head_dim/2`) ready to upload as the two
/// `rope2d` table buffers. `inv_freq[d] = theta^(-2d/head_dim)`, matching HF.
pub fn mrope_tables(positions: &[[u32; 3]], mrope_section: [u32; 3], head_dim: u32, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = (head_dim / 2) as usize;
    let amap = axis_map(mrope_section, half);
    let seq = positions.len();
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for (t, p) in positions.iter().enumerate() {
        for d in 0..half {
            let inv_freq = theta.powf(-2.0 * d as f32 / head_dim as f32);
            let angle = p[amap[d]] as f32 * inv_freq;
            cos[t * half + d] = angle.cos();
            sin[t * half + d] = angle.sin();
        }
    }
    (cos, sin)
}

/// Compute the 3-axis (T,H,W) position ids for interleaved M-RoPE from a token
/// stream and the per-image grids, porting HF `get_rope_index`.
///
/// `tokens` is the full decoder input; each image occupies a contiguous run of
/// `image_token_id` slots. `grids` lists each image's `(t, h, w)` in **merged
/// (LLM) units** (i.e. after the 2×2 PatchMerger), in order of appearance;
/// `sum(t·h·w)` must equal the number of image tokens. Returns one `[t,h,w]`
/// position per token:
/// - a text run advances all three axes together (diagonal), like plain RoPE;
/// - an image block places `t·h·w` tokens on a `(t,h,w)` meshgrid anchored at the
///   running position, then advances the running position past the block's
///   spatial extent (`max(h,w)`), so the next text token starts clear of it.
///
/// This is what lets 2-D image layout survive into the 1-D decoder stream.
/// (Video timestamp handling is deferred — images use `t = 1`; the temporal axis
/// simply counts frames from the anchor.)
///
/// A thin single-placeholder-type wrapper over [`get_rope_index_multi`] — see
/// that function's doc for the generalization (multiple placeholder token
/// types, e.g. Qwen3-Omni's image/audio/video, in one scan).
pub fn get_rope_index(tokens: &[u32], image_token_id: u32, grids: &[(u32, u32, u32)]) -> Vec<[u32; 3]> {
    get_rope_index_multi(tokens, &[(image_token_id, grids)])
}

/// [`get_rope_index`] generalized to MULTIPLE placeholder token types in one
/// scan — Qwen3-Omni's Thinker has three (image, audio, video); a "text" run
/// is any token that isn't one of them. `placeholders` lists `(token_id,
/// grids)` pairs; a token stream may freely interleave runs of different
/// placeholder types, each consuming its own `grids` list (in order of
/// appearance) exactly as the single-type version does for images.
///
/// An audio run's "grid" has no real 2-D spatial extent — model it as
/// `(n_audio, 1, 1)`: the meshgrid loop already degenerates to advancing only
/// the T axis (one step per audio embedding row) with H/W pinned at the
/// anchor when `h = w = 1`. A video run is `(n_frames, h, w)` — the SAME
/// meshgrid math images use, just with `t > 1` (one real frame per T step,
/// each frame's own merged patch grid on H/W). This is a documented
/// approximation of HF's exact reference (which additionally scales audio's T
/// axis by wall-clock seconds via `position_id_per_seconds`), not a byte-exact
/// port — see `crate::generate`'s module doc / `docs/models/omni/status.md`'s
/// multimodal entry.
///
/// The post-run anchor advance is `max(t, h, w)` (not `max(h, w)` — the
/// single-type version's original formula, which only ever saw `t = 1`
/// images and so never needed to consider it): the running position must
/// clear whichever axis actually grew, and for audio/video that is `t`, not
/// `h`/`w`. Identical to the old formula whenever `t = 1`, so every existing
/// image-only caller is unaffected.
pub fn get_rope_index_multi(tokens: &[u32], placeholders: &[(u32, &[(u32, u32, u32)])]) -> Vec<[u32; 3]> {
    let mut pos = Vec::with_capacity(tokens.len());
    let mut cp = 0u32; // running anchor ("current_pos")
    let mut gi = vec![0usize; placeholders.len()];
    let placeholder_index = |tok: u32| placeholders.iter().position(|&(id, _)| id == tok);
    let mut i = 0usize;
    while i < tokens.len() {
        match placeholder_index(tokens[i]) {
            None => {
                let start = i;
                while i < tokens.len() && placeholder_index(tokens[i]).is_none() {
                    i += 1;
                }
                for k in 0..(i - start) as u32 {
                    pos.push([cp + k, cp + k, cp + k]);
                }
                cp += (i - start) as u32;
            }
            Some(pi) => {
                let tok_id = tokens[i];
                // One contiguous run may cover several media items of the SAME
                // placeholder type back-to-back (mirrors the single-type version).
                while i < tokens.len() && tokens[i] == tok_id {
                    let (t, h, w) = placeholders[pi].1[gi[pi]];
                    gi[pi] += 1;
                    let st = cp;
                    for ti in 0..t {
                        for hi in 0..h {
                            for wi in 0..w {
                                pos.push([st + ti, st + hi, st + wi]);
                            }
                        }
                    }
                    i += (t * h * w) as usize;
                    cp = st + t.max(h).max(w); // advance past whichever axis actually grew
                }
            }
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_map_interleaved_4b() {
        let m = axis_map([24, 20, 20], 64);
        // Interleaved head: T,H,W,T,H,W…
        assert_eq!(&m[..6], &[0, 1, 2, 0, 1, 2]);
        // H section ends at channel 1+3·19 = 58; W at 2+3·19 = 59.
        assert_eq!(m[58], 1);
        assert_eq!(m[59], 2);
        // Tail [60,64) reverts to T (H/W sections exhausted).
        assert_eq!(&m[60..64], &[0, 0, 0, 0]);
        // Per-axis counts equal the section sizes.
        let count = |a: usize| m.iter().filter(|&&x| x == a).count();
        assert_eq!((count(0), count(1), count(2)), (24, 20, 20));
    }

    #[test]
    fn diagonal_positions_collapse_to_plain_rope() {
        // When all three axes share a position (text tokens), M-RoPE must equal
        // ordinary RoPE at that position regardless of the axis map.
        let (hd, theta) = (128u32, 5_000_000.0f32);
        let (cos, sin) = mrope_tables(&[[7, 7, 7]], [24, 20, 20], hd, theta);
        let half = (hd / 2) as usize;
        for d in 0..half {
            let inv = theta.powf(-2.0 * d as f32 / hd as f32);
            let ang = 7.0 * inv;
            assert!((cos[d] - ang.cos()).abs() < 1e-6);
            assert!((sin[d] - ang.sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn image_positions_route_by_axis() {
        // Distinct axis positions: channel 0 uses T, 1 uses H, 2 uses W.
        let (hd, theta) = (128u32, 5_000_000.0f32);
        let (cos, _) = mrope_tables(&[[10, 20, 30]], [24, 20, 20], hd, theta);
        let inv = |d: u32| theta.powf(-2.0 * d as f32 / hd as f32);
        assert!((cos[0] - (10.0 * inv(0)).cos()).abs() < 1e-6); // T
        assert!((cos[1] - (20.0 * inv(1)).cos()).abs() < 1e-6); // H
        assert!((cos[2] - (30.0 * inv(2)).cos()).abs() < 1e-6); // W
        assert!((cos[3] - (10.0 * inv(3)).cos()).abs() < 1e-6); // back to T
    }

    #[test]
    fn table_shape() {
        let (cos, sin) = mrope_tables(&[[0, 0, 0], [1, 2, 3], [4, 5, 6]], [24, 20, 20], 128, 5e6);
        assert_eq!(cos.len(), 3 * 64);
        assert_eq!(sin.len(), 3 * 64);
    }

    #[test]
    fn rope_index_all_text_is_diagonal() {
        let pos = get_rope_index(&[10, 11, 12, 13], 999, &[]);
        assert_eq!(pos, vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    }

    #[test]
    fn rope_index_text_image_text() {
        // 2 text, then a 1×2×2 image (4 tokens, id=999), then 1 text.
        const IMG: u32 = 999;
        let tokens = [7, 7, IMG, IMG, IMG, IMG, 7];
        let pos = get_rope_index(&tokens, IMG, &[(1, 2, 2)]);
        assert_eq!(
            pos,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                // image anchored at cp=2: meshgrid over h∈{0,1}, w∈{0,1}
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3],
                // next text starts past the spatial extent: cp = 2 + max(2,2) = 4
                [4, 4, 4],
            ]
        );
        assert_eq!(pos.len(), tokens.len());
    }

    #[test]
    fn rope_index_two_adjacent_images() {
        // Two 1×1×2 images back-to-back (no text between): each advances the anchor.
        const IMG: u32 = 999;
        let tokens = [IMG, IMG, IMG, IMG];
        let pos = get_rope_index(&tokens, IMG, &[(1, 1, 2), (1, 1, 2)]);
        // img0 at cp=0: [0,0,0],[0,0,1]; advance cp += max(1,2)=2.
        // img1 at cp=2: [2,2,2],[2,2,3].
        assert_eq!(pos, vec![[0, 0, 0], [0, 0, 1], [2, 2, 2], [2, 2, 3]]);
    }

    #[test]
    fn multi_text_audio_text_advances_by_audio_duration() {
        // 1 text, then a 3-frame audio run (grid (3,1,1): T advances, H/W pinned),
        // then 1 text. Audio's post-run advance is max(3,1,1)=3, NOT max(1,1)=1 --
        // the bug the single-type formula (`h.max(w)`) would have hit.
        const AUDIO: u32 = 500;
        let tokens = [7, AUDIO, AUDIO, AUDIO, 7];
        let pos = get_rope_index_multi(&tokens, &[(AUDIO, &[(3, 1, 1)])]);
        assert_eq!(
            pos,
            vec![
                [0, 0, 0],
                // audio anchored at cp=1: T advances 1..4, H/W pinned at 1.
                [1, 1, 1],
                [2, 1, 1],
                [3, 1, 1],
                // next text starts past the full audio duration: cp = 1 + max(3,1,1) = 4
                [4, 4, 4],
            ]
        );
    }

    #[test]
    fn multi_image_and_audio_interleave_independently() {
        // A 1x1x2 image run then a 2-frame audio run, each consuming its OWN
        // grids list in order -- proves per-placeholder-type `gi` bookkeeping
        // doesn't cross-contaminate.
        const IMG: u32 = 999;
        const AUDIO: u32 = 500;
        let tokens = [IMG, IMG, AUDIO, AUDIO];
        let pos = get_rope_index_multi(&tokens, &[(IMG, &[(1, 1, 2)]), (AUDIO, &[(2, 1, 1)])]);
        assert_eq!(
            pos,
            vec![
                // image at cp=0: [0,0,0],[0,0,1]; advance cp += max(1,1,2)=2.
                [0, 0, 0],
                [0, 0, 1],
                // audio at cp=2: T advances 2..4, H/W pinned at 2.
                [2, 2, 2],
                [3, 2, 2],
            ]
        );
    }

    #[test]
    fn multi_video_run_is_a_t_h_w_meshgrid_like_a_multiframe_image() {
        // A 2-frame video, each frame a 1x1 merged patch grid: grid (2,1,1) --
        // same shape as the audio case, proving video needs no separate code
        // path (it's the existing meshgrid with t>1, matching real frames).
        const VIDEO: u32 = 600;
        let tokens = [VIDEO, VIDEO];
        let pos = get_rope_index_multi(&tokens, &[(VIDEO, &[(2, 1, 1)])]);
        assert_eq!(pos, vec![[0, 0, 0], [1, 0, 0]]);
    }

    #[test]
    fn multi_single_type_matches_get_rope_index_exactly() {
        // get_rope_index must still be a pure wrapper: identical output to
        // calling get_rope_index_multi with a one-entry placeholders list.
        const IMG: u32 = 999;
        let tokens = [7, 7, IMG, IMG, IMG, IMG, 7];
        let grids = [(1u32, 2u32, 2u32)];
        assert_eq!(get_rope_index(&tokens, IMG, &grids), get_rope_index_multi(&tokens, &[(IMG, &grids)]));
    }
}
