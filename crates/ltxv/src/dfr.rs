// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DFR (Diffusion Fidelity Rendering) geometry: the keyframe-segment canvas
//! grid, temporal tile boundaries with their lead-in, and generated-keyframe-
//! slot token-append bookkeeping. Ported from `dfr_layout.py` and
//! `VideoGeneratedKeyframeSlots.apply_to`
//! (`scratchpad/reference/ltxv/packages/ltx-pipelines/src/ltx_pipelines/
//! dfr_layout.py`, `.../ltx-core/src/ltx_core/conditioning/types/
//! keyframe_slots.py`) - PURE geometry/bookkeeping, no weights, no device,
//! every function here is real and unit-tested (several pinned against
//! numbers read directly off a live run of the reference `dfr_layout.py`,
//! not hand-derived - see this module's tests for the exact python
//! invocations each case reproduces).
//!
//! # What this module does NOT do
//!
//! It does not run a model, does not know about LoRA, and does not
//! implement the real per-token/partial-strength "anchor keyframe" carry-
//! forward `_ANCHOR_KEYFRAME_STRENGTH=0.95` mechanism (that needs per-token
//! Euler-ancestral stepping the pipeline layer doesn't build - see
//! `pipeline.rs`'s `generate_dfr` doc for exactly what's wired and what
//! isn't). [`TileRange::anchor_kf_global`] is still computed correctly here
//! (real geometry, real test coverage) even though `generate_dfr` does not
//! act on it - a future milestone that adds per-token partial denoising can
//! consume it without touching this file.
//!
//! # Position units: this port's own fractional-latent-grid convention, not
//! # real LTX's pixel/fps time
//!
//! Real LTX-2.5 normalizes RoPE time positions to `pixel_frame / fps` so an
//! 8-pixel-frame video token and a 1-pixel-frame keyframe-slot token sit on
//! the same real-time axis (`keyframe_slots.py::_slot_positions`). This
//! port's own RoPE convention (`pipeline::grid_positions`, parity-proven
//! against the M3 golden) never divides by fps at all - it uses plain
//! INTEGER latent-grid coordinates, one unit per latent frame. This module
//! extends that SAME convention fractionally rather than introducing a
//! second, fps-normalized position space: a keyframe slot at pixel frame `p`
//! (temporal scale `s`) gets latent-grid position `[p/s, p/s + 1/s)` - the
//! natural generalization of "1 pixel frame is `1/s` of a latent frame's
//! span" in a scheme where a whole latent frame already spans `[k, k+1)`.
//! RoPE already supports fractional positions (`rope.rs`'s "fractional
//! position -> midpoint" construction, M3), so no new machinery is needed.
//! This is a genuine, documented DEVIATION from upstream's pixel/fps units -
//! reworking the whole pipeline onto fps-normalized positions was judged out
//! of scope for a smoke-level milestone (it would touch the
//! already-parity-tested M3 position convention used by every other tap).

/// Candidate keyframe-segment lengths, in pixel frames - `dfr_layout.py`'s
/// `SEGMENT_CANDIDATES`.
pub const SEGMENT_CANDIDATES: [usize; 2] = [24, 32];

/// Lead-in for non-first tiles, in canvas segments - `dfr_layout.py`'s
/// `TILE_LEAD_SEGMENTS`. See [`tile_ranges`]'s doc for why 1.
pub const TILE_LEAD_SEGMENTS: usize = 1;

/// This port's causal video VAE's temporal stride (`VIDEO_SCALE_FACTORS.
/// time` upstream) - one latent frame per 8 pixel frames, except latent
/// frame 0 which covers pixel frame 0 alone. Matches
/// `vae3d::LtxVaeConfig::latent_frames`'s own `1 + 8k` contract.
pub const VIDEO_TEMPORAL_SCALE: usize = 8;

/// Pick the keyframe segment length from [`SEGMENT_CANDIDATES`], preferring
/// whichever pads `content_frames` least; ties keep the larger segment.
/// `content_frames` is `num_frames - 1` (frame 0 is never part of a segment
/// - see [`resolve_canvas`]). Mirrors `dfr_layout.py::choose_segment_length`
/// exactly.
pub fn choose_segment_length(content_frames: usize) -> Result<usize, String> {
    if content_frames < 1 {
        return Err(format!("content_frames must be >= 1, got {content_frames}"));
    }
    let pad = |segment: usize| (segment - content_frames % segment) % segment;
    let mut best = SEGMENT_CANDIDATES[0];
    let mut best_pad = pad(best);
    for &candidate in &SEGMENT_CANDIDATES[1..] {
        let p = pad(candidate);
        if p < best_pad || (p == best_pad && candidate > best) {
            best = candidate;
            best_pad = p;
        }
    }
    Ok(best)
}

/// Pad `(num_frames - 1)` up to a multiple of the chosen segment length,
/// returning `(padded_num_frames, segment, positions)`. `positions` are
/// `[segment, 2*segment, ..., padded_num_frames - 1]` - frame 0 is excluded
/// (it is already its own single-pixel-frame latent token under causal
/// encoding) and the terminal frame is always included, even when it lands
/// exactly on a segment boundary. Mirrors `dfr_layout.py::resolve_canvas`.
pub fn resolve_canvas(num_frames: usize, temporal_scale: usize) -> Result<(usize, usize, Vec<usize>), String> {
    if num_frames < 1 {
        return Err(format!("num_frames must be >= 1, got {num_frames}"));
    }
    if (num_frames - 1) % temporal_scale != 0 {
        return Err(format!("num_frames must satisfy (num_frames - 1) % {temporal_scale} == 0 (got {num_frames})"));
    }
    let content = num_frames - 1;
    if content == 0 {
        return Err("the canvas needs at least 2 pixel frames".into());
    }
    let segment = choose_segment_length(content)?;
    let content_padded = content + (segment - content % segment) % segment;
    let positions: Vec<usize> = (1..=content_padded / segment).map(|i| segment * i).collect();
    Ok((content_padded + 1, segment, positions))
}

/// Map an x-`temporal_scale`-border pixel frame to its latent index.
/// Mirrors `dfr_layout.py::pixel_to_latent_index`.
pub fn pixel_to_latent_index(pixel_frame: usize, temporal_scale: usize) -> Result<usize, String> {
    if pixel_frame != 0 && pixel_frame % temporal_scale != 0 {
        return Err(format!("pixel_frame {pixel_frame} is not on the x{temporal_scale} latent border"));
    }
    Ok(pixel_frame / temporal_scale)
}

/// One temporal denoise tile in global pixel/latent coordinates. `pixel_start`/
/// `pixel_end` are inclusive, `latent_end_exclusive` is half-open. Non-first
/// tiles start [`TILE_LEAD_SEGMENTS`] segments before the region they own, so
/// the seam shared with the previous tile falls inside the window.
/// `anchor_kf_global` are the seam keyframes in the window (every tile but
/// the first has at least one - frame 0 is not itself a keyframe, so the
/// first tile's window start contributes no anchor). `slot_kf_global` are
/// the mid-segment positions this window invents. Mirrors `dfr_layout.py`'s
/// `TileRange` NamedTuple field-for-field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileRange {
    pub pixel_start: usize,
    pub pixel_end: usize,
    pub latent_start: usize,
    pub latent_end_exclusive: usize,
    pub anchor_kf_global: Vec<usize>,
    pub slot_kf_global: Vec<usize>,
    pub drop_latent_prefix: usize,
}

/// Split `n_segments` into `num_tiles` contiguous owned runs, largest first.
/// Mirrors `dfr_layout.py::_owned_segment_counts`.
fn owned_segment_counts(n_segments: usize, num_tiles: usize) -> Vec<usize> {
    let base = n_segments / num_tiles;
    let remainder = n_segments % num_tiles;
    (0..num_tiles).map(|index| base + usize::from(index < remainder)).collect()
}

/// One window owning segments `[own_lo, own_hi)`, preceded by `lead_segments`
/// of lead-in. Mirrors `dfr_layout.py::_build_tile`.
fn build_tile(boundaries: &[usize], own_lo: usize, own_hi: usize, lead_segments: usize, temporal_scale: usize) -> Result<TileRange, String> {
    let window_lo = own_lo.saturating_sub(lead_segments);
    let pixel_start = boundaries[window_lo];
    let pixel_end = boundaries[own_hi];
    let latent_start = pixel_to_latent_index(pixel_start, temporal_scale)?;

    // Handover is exactly at the shared keyframe: the previous tile keeps
    // the seam latent (the 8-frame token ending on the KF mark) and this
    // tile resumes strictly after it, so the prefix drops the lead-in plus
    // that seam latent.
    let mut drop_latent_prefix = pixel_to_latent_index(boundaries[own_lo], temporal_scale)? - latent_start;
    if own_lo > 0 {
        drop_latent_prefix += 1;
    }

    let anchor_kf_global: Vec<usize> = (window_lo..=own_hi).filter(|&i| boundaries[i] != 0).map(|i| boundaries[i]).collect();
    let slot_kf_global: Vec<usize> = (window_lo..own_hi).map(|i| (boundaries[i] + boundaries[i + 1]) / 2).collect();

    Ok(TileRange {
        pixel_start,
        pixel_end,
        latent_start,
        latent_end_exclusive: pixel_to_latent_index(pixel_end, temporal_scale)? + 1,
        anchor_kf_global,
        slot_kf_global,
        drop_latent_prefix,
    })
}

/// Partition the canvas into `num_tiles` keyframe-seam tiles, gapless, with a
/// lead-in. Owned segment runs are contiguous; each non-first window
/// additionally reaches `lead_segments` back so it denoises through the
/// shared seam keyframe. `num_tiles` is clamped to the segment count.
/// Mirrors `dfr_layout.py::tile_ranges`.
pub fn tile_ranges(seam_positions: &[usize], num_frames: usize, num_tiles: usize, temporal_scale: usize, lead_segments: usize) -> Result<Vec<TileRange>, String> {
    if num_frames < 2 {
        return Err(format!("num_frames must be >= 2, got {num_frames}"));
    }
    if seam_positions.is_empty() {
        return Err("seam_positions must be non-empty".into());
    }
    if seam_positions[seam_positions.len() - 1] != num_frames - 1 {
        return Err(format!("last seam must be the terminal frame {}, got {}", num_frames - 1, seam_positions[seam_positions.len() - 1]));
    }
    if lead_segments < 1 {
        return Err(format!("lead_segments must be >= 1, got {lead_segments}"));
    }
    if num_tiles < 1 {
        return Err(format!("num_tiles must be >= 1, got {num_tiles}"));
    }

    let mut boundaries: Vec<usize> = Vec::with_capacity(seam_positions.len() + 1);
    boundaries.push(0);
    boundaries.extend_from_slice(seam_positions);
    for i in 1..boundaries.len() {
        let span = boundaries[i] as isize - boundaries[i - 1] as isize;
        if span <= 0 {
            return Err(format!("seam_positions must be strictly increasing, got {seam_positions:?}"));
        }
        let span = span as usize;
        if span % temporal_scale != 0 {
            return Err(format!("segment span {span} is not a multiple of temporal scale {temporal_scale}"));
        }
        if span / temporal_scale < 2 {
            return Err(format!("segment span {span} is under 2 latent frames, too short to carry a tile lead-in"));
        }
    }

    let n_segments = boundaries.len() - 1;
    let mut tiles = Vec::new();
    let mut own_lo = 0usize;
    for (index, count) in owned_segment_counts(n_segments, num_tiles.min(n_segments)).into_iter().enumerate() {
        let lead = if index > 0 { lead_segments } else { 0 };
        tiles.push(build_tile(&boundaries, own_lo, own_lo + count, lead, temporal_scale)?);
        own_lo += count;
    }
    Ok(tiles)
}

/// Shift global pixel indices into a tile-local frame (`local = global -
/// pixel_start`). Mirrors `dfr_layout.py::remap_positions_to_local`.
pub fn remap_positions_to_local(positions: &[usize], pixel_start: usize) -> Vec<usize> {
    positions.iter().map(|&p| p - pixel_start).collect()
}

/// Crop `[start, end)` out of a `[c, t, h, w]` row-major (channel-major)
/// buffer along the `t` axis. Channel-major slicing along a non-outermost
/// axis is not a contiguous sub-slice (unlike token-major, see
/// `pipeline::tc_to_chw`'s doc), so this is a real per-channel gather, not a
/// borrow.
pub fn slice_time_chw(x: &[f32], c: usize, t: usize, h: usize, w: usize, start: usize, end: usize) -> Vec<f32> {
    assert!(end <= t && start <= end, "slice_time_chw: [{start},{end}) out of bounds for t={t}");
    let hw = h * w;
    let keep = end - start;
    let mut out = vec![0f32; c * keep * hw];
    for ci in 0..c {
        let src = &x[(ci * t + start) * hw..(ci * t + end) * hw];
        out[ci * keep * hw..(ci + 1) * keep * hw].copy_from_slice(src);
    }
    out
}

/// Concatenate tile video latents along `T`, each tile contributing
/// `latent[drop_latent_prefix:]` - the stitch handover lands exactly on the
/// shared seam keyframe (see [`tile_ranges`]'s doc). Every tile latent is
/// `[c, latent_end_exclusive - latent_start, h, w]` channel-major. Mirrors
/// `dfr_layout.py::stitch_tile_latents`.
pub fn stitch_tile_latents(tile_latents: &[Vec<f32>], ranges: &[TileRange], c: usize, h: usize, w: usize) -> Result<Vec<f32>, String> {
    if tile_latents.len() != ranges.len() {
        return Err(format!("expected {} tile latents, got {}", ranges.len(), tile_latents.len()));
    }
    if tile_latents.is_empty() {
        return Err("tile_latents must be non-empty".into());
    }
    let hw = h * w;
    let mut pieces: Vec<Vec<f32>> = Vec::with_capacity(tile_latents.len());
    for (latent, tile) in tile_latents.iter().zip(ranges) {
        let expected_t = tile.latent_end_exclusive - tile.latent_start;
        if latent.len() != c * expected_t * hw {
            return Err(format!("tile latent has {} values, expected {} for range [{}, {})", latent.len(), c * expected_t * hw, tile.latent_start, tile.latent_end_exclusive));
        }
        if tile.drop_latent_prefix >= expected_t {
            return Err(format!("drop_latent_prefix={} invalid for tile T={expected_t}", tile.drop_latent_prefix));
        }
        pieces.push(slice_time_chw(latent, c, expected_t, h, w, tile.drop_latent_prefix, expected_t));
    }
    let out_t: usize = pieces.iter().zip(ranges).map(|(_, r)| (r.latent_end_exclusive - r.latent_start) - r.drop_latent_prefix).sum();
    let mut out = vec![0f32; c * out_t * hw];
    let mut t_off = 0usize;
    for (piece, tile) in pieces.iter().zip(ranges) {
        let keep = (tile.latent_end_exclusive - tile.latent_start) - tile.drop_latent_prefix;
        for ci in 0..c {
            let dst = (ci * out_t + t_off) * hw;
            let src = ci * keep * hw;
            out[dst..dst + keep * hw].copy_from_slice(&piece[src..src + keep * hw]);
        }
        t_off += keep;
    }
    Ok(out)
}

/// The final frame-count contract every DFR call honours regardless of how
/// much the canvas padded its tail: `(requested - 1) * 2^rounds + 1`.
/// `requested - 1` is always a multiple of the VAE temporal scale (enforced
/// by [`resolve_canvas`]), so the corresponding latent trim always lands on
/// a latent boundary. Mirrors the formula stated in `dfr_pipeline.py`'s
/// class doc and its `target_frames` computation.
pub fn target_frame_count(requested_frames: usize, rounds: u32) -> usize {
    (requested_frames - 1) * 2usize.pow(rounds) + 1
}

/// Geometry for appending `pixel_frame_indices.len()` generated-keyframe (or,
/// with `marked=false`, ordinary anchor-image) tokens after `base_t` existing
/// tokens, at the target's `(lh, lw)` spatial grid. Content (the actual
/// latent values) is assembled by the caller - this is deliberately
/// content-free, matching this module's "pure geometry" scope; `pipeline.rs`
/// builds the token buffer itself, in the same order this lays out
/// `positions`/`keyframes_mask` (base tokens first, then slots in
/// `(slot, h, w)` raster order, [`Self::tokens_per_slot`] tokens per slot).
#[derive(Debug)]
pub struct KeyframeSlotLayout {
    /// `base_t + pixel_frame_indices.len() * tokens_per_slot`.
    pub total_tokens: usize,
    /// `lh * lw` - one latent frame's worth of tokens at the target's
    /// spatial resolution (`keyframe_slots.py`'s `tokens_per_keyframe`).
    pub tokens_per_slot: usize,
    /// Index of the first appended token - `base_t`.
    pub first_token: usize,
    /// `[total_tokens]`, non-zero marks a token [`crate::dit::LtxDit::
    /// forward`]'s `keyframes_mask` should add
    /// `keyframes_abs_pos_embedding` to. All-zero on `[0, base_t)`
    /// (unmarked base tokens carry no keyframe/anchor semantics of their
    /// own) then `marked as f32` on the appended range.
    pub keyframes_mask: Vec<f32>,
    /// `[3, total_tokens, 2]` row-major RoPE position bounds, same layout
    /// `pipeline::grid_positions` produces - `base_positions` copied
    /// unchanged into `[0, base_t)` of every axis block, appended slot
    /// positions after. See this module's doc for the fractional-latent-grid
    /// unit convention.
    pub positions: Vec<f32>,
}

/// Build [`KeyframeSlotLayout`] for `pixel_frame_indices` appended after
/// `base_positions` (`[3, base_t, 2]`, [`crate::pipeline::grid_positions`]'s
/// layout). `marked` distinguishes a real generated-keyframe slot
/// (`VideoGeneratedKeyframeSlots`, `marked=true`) from an ordinary anchor/
/// image keyframe (`VideoConditionByKeyframeIndex`, `marked=false`) - see
/// this crate's `dit.rs` doc on `keyframes_mask` for why only the former
/// receives the learned embedding.
pub fn keyframe_slots(base_t: usize, base_positions: &[f32], lh: usize, lw: usize, pixel_frame_indices: &[usize], temporal_scale: usize, marked: bool) -> Result<KeyframeSlotLayout, String> {
    if base_positions.len() != 3 * base_t * 2 {
        return Err(format!("base_positions has {} values, expected {}", base_positions.len(), 3 * base_t * 2));
    }
    if pixel_frame_indices.is_empty() {
        return Err("pixel_frame_indices must be non-empty".into());
    }
    for w in pixel_frame_indices.windows(2) {
        if w[1] <= w[0] {
            return Err(format!("pixel_frame_indices must be strictly increasing, got {pixel_frame_indices:?}"));
        }
    }

    let tokens_per_slot = lh * lw;
    let k = pixel_frame_indices.len();
    let num_new = tokens_per_slot * k;
    let total_tokens = base_t + num_new;

    let mut keyframes_mask = vec![0f32; total_tokens];
    keyframes_mask[..base_t].copy_from_slice(&vec![0f32; base_t]); // explicit: base tokens are never marked here
    if marked {
        for v in &mut keyframes_mask[base_t..] {
            *v = 1.0;
        }
    }

    let mut positions = vec![0f32; 3 * total_tokens * 2];
    for axis in 0..3 {
        let src = &base_positions[axis * base_t * 2..(axis + 1) * base_t * 2];
        let dst_off = axis * total_tokens * 2;
        positions[dst_off..dst_off + base_t * 2].copy_from_slice(src);
    }
    let mut local = 0usize;
    for &p in pixel_frame_indices {
        let fstart = p as f32 / temporal_scale as f32;
        let fend = fstart + 1.0 / temporal_scale as f32;
        for hi in 0..lh {
            for wi in 0..lw {
                let axis_vals = [(fstart, fend), (hi as f32, hi as f32 + 1.0), (wi as f32, wi as f32 + 1.0)];
                for (axis, &(s, e)) in axis_vals.iter().enumerate() {
                    let off = axis * total_tokens * 2 + (base_t + local) * 2;
                    positions[off] = s;
                    positions[off + 1] = e;
                }
                local += 1;
            }
        }
    }

    Ok(KeyframeSlotLayout { total_tokens, tokens_per_slot, first_token: base_t, keyframes_mask, positions })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------- choose_segment_length

    /// Every case pinned against a live `python3 -c "... choose_segment_length(...)"`
    /// run of the actual reference `dfr_layout.py` (not hand-derived).
    #[test]
    fn choose_segment_length_matches_reference_numbers() {
        let cases = [(1, 24), (8, 24), (16, 24), (23, 24), (24, 24), (25, 32), (32, 32), (33, 24), (40, 24), (48, 24), (64, 32), (96, 32), (100, 24)];
        for (content, want) in cases {
            assert_eq!(choose_segment_length(content).unwrap(), want, "content_frames={content}");
        }
    }

    #[test]
    fn choose_segment_length_rejects_zero() {
        assert!(choose_segment_length(0).is_err());
    }

    // ------------------------------------------------------------ resolve_canvas

    /// Pinned against a live reference `resolve_canvas(...)` run.
    #[test]
    fn resolve_canvas_matches_reference_numbers() {
        let cases: [(usize, (usize, usize, &[usize])); 8] = [
            (9, (25, 24, &[24])),
            (17, (25, 24, &[24])),
            (25, (25, 24, &[24])),
            (41, (49, 24, &[24, 48])),
            (49, (49, 24, &[24, 48])),
            (65, (65, 32, &[32, 64])),
            (97, (97, 32, &[32, 64, 96])),
            (129, (129, 32, &[32, 64, 96, 128])),
        ];
        for (num_frames, (want_n, want_seg, want_positions)) in cases {
            let (n, seg, positions) = resolve_canvas(num_frames, VIDEO_TEMPORAL_SCALE).unwrap();
            assert_eq!((n, seg, positions.as_slice()), (want_n, want_seg, want_positions), "num_frames={num_frames}");
        }
    }

    #[test]
    fn resolve_canvas_rejects_a_frame_count_off_the_temporal_border() {
        let e = resolve_canvas(10, 8).unwrap_err();
        assert!(e.contains("% 8 == 0"), "{e}");
    }

    #[test]
    fn resolve_canvas_rejects_a_single_frame() {
        // (num_frames - 1) % 8 == 0 and num_frames == 1 both hold, but a
        // 1-frame canvas has no content to segment at all.
        let e = resolve_canvas(1, 8).unwrap_err();
        assert!(e.contains("at least 2 pixel frames"), "{e}");
    }

    // -------------------------------------------------------------- tile_ranges

    /// The 2-tile case pinned against a live reference `tile_ranges([24, 48],
    /// 49, 2)` run - includes the degenerate "lead-in eats the whole prior
    /// canvas" case that a 2-segment/2-tile split produces.
    #[test]
    fn tile_ranges_two_tiles_matches_reference_numbers() {
        let tiles = tile_ranges(&[24, 48], 49, 2, VIDEO_TEMPORAL_SCALE, TILE_LEAD_SEGMENTS).unwrap();
        assert_eq!(tiles.len(), 2);
        assert_eq!(
            tiles[0],
            TileRange { pixel_start: 0, pixel_end: 24, latent_start: 0, latent_end_exclusive: 4, anchor_kf_global: vec![24], slot_kf_global: vec![12], drop_latent_prefix: 0 }
        );
        assert_eq!(
            tiles[1],
            TileRange { pixel_start: 0, pixel_end: 48, latent_start: 0, latent_end_exclusive: 7, anchor_kf_global: vec![24, 48], slot_kf_global: vec![12, 36], drop_latent_prefix: 4 }
        );
    }

    /// The 4-tile/5-segment case pinned against a live reference
    /// `tile_ranges([24, 48, 72, 96, 120], 121, 4)` run - a non-degenerate
    /// split where every tile's lead-in is a genuine PARTIAL reach-back, not
    /// the whole canvas.
    #[test]
    fn tile_ranges_four_tiles_five_segments_matches_reference_numbers() {
        let tiles = tile_ranges(&[24, 48, 72, 96, 120], 121, 4, VIDEO_TEMPORAL_SCALE, TILE_LEAD_SEGMENTS).unwrap();
        assert_eq!(tiles.len(), 4);
        assert_eq!(
            tiles[0],
            TileRange { pixel_start: 0, pixel_end: 48, latent_start: 0, latent_end_exclusive: 7, anchor_kf_global: vec![24, 48], slot_kf_global: vec![12, 36], drop_latent_prefix: 0 }
        );
        assert_eq!(
            tiles[1],
            TileRange { pixel_start: 24, pixel_end: 72, latent_start: 3, latent_end_exclusive: 10, anchor_kf_global: vec![24, 48, 72], slot_kf_global: vec![36, 60], drop_latent_prefix: 4 }
        );
        assert_eq!(
            tiles[2],
            TileRange { pixel_start: 48, pixel_end: 96, latent_start: 6, latent_end_exclusive: 13, anchor_kf_global: vec![48, 72, 96], slot_kf_global: vec![60, 84], drop_latent_prefix: 4 }
        );
        assert_eq!(
            tiles[3],
            TileRange { pixel_start: 72, pixel_end: 120, latent_start: 9, latent_end_exclusive: 16, anchor_kf_global: vec![72, 96, 120], slot_kf_global: vec![84, 108], drop_latent_prefix: 4 }
        );
    }

    #[test]
    fn tile_ranges_clamps_num_tiles_to_the_segment_count() {
        // Only 1 segment exists, so requesting 4 tiles must still yield 1.
        let tiles = tile_ranges(&[24], 25, 4, VIDEO_TEMPORAL_SCALE, TILE_LEAD_SEGMENTS).unwrap();
        assert_eq!(tiles.len(), 1);
    }

    #[test]
    fn tile_ranges_rejects_a_seam_list_not_ending_on_the_terminal_frame() {
        let e = tile_ranges(&[24, 40], 49, 2, VIDEO_TEMPORAL_SCALE, TILE_LEAD_SEGMENTS).unwrap_err();
        assert!(e.contains("terminal frame"), "{e}");
    }

    // --------------------------------------------------- remap_positions_to_local

    #[test]
    fn remap_positions_to_local_matches_reference_numbers() {
        assert_eq!(remap_positions_to_local(&[24, 48], 10), vec![14, 38]);
    }

    // ----------------------------------------------------- stitch_tile_latents

    #[test]
    fn stitch_tile_latents_drops_each_prefix_and_concatenates_along_t() {
        // 2 tiles from the pinned 2-tile case above: tile0 T=4 drop=0 (keep
        // 4), tile1 T=7 drop=4 (keep 3) -> stitched T=7, matching
        // expected_t=(49-1)/8+1=7 the pipeline itself asserts.
        let tiles = tile_ranges(&[24, 48], 49, 2, VIDEO_TEMPORAL_SCALE, TILE_LEAD_SEGMENTS).unwrap();
        let (c, h, w) = (2usize, 1usize, 1usize);
        // Tile0: 4 frames of 2 channels, values c*100 + t.
        let t0: Vec<f32> = (0..c).flat_map(|ci| (0..4).map(move |t| (ci * 100 + t) as f32)).collect();
        // Tile1: 7 frames, values c*1000 + t (distinguishable from tile0).
        let t1: Vec<f32> = (0..c).flat_map(|ci| (0..7).map(move |t| (ci * 1000 + t) as f32)).collect();
        let out = stitch_tile_latents(&[t0, t1], &tiles, c, h, w).unwrap();
        assert_eq!(out.len(), c * 7 * h * w);
        // Channel 0: tile0 keeps frames [0..4) = [0,1,2,3], tile1 keeps
        // frames [4..7) (drop_latent_prefix=4) = [4,5,6].
        assert_eq!(&out[0..7], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Channel 1: tile0 base 100, tile1 base 1000.
        assert_eq!(&out[7..14], &[100.0, 101.0, 102.0, 103.0, 1004.0, 1005.0, 1006.0]);
    }

    #[test]
    fn stitch_tile_latents_rejects_a_length_mismatch() {
        // This tile's own `latent_end_exclusive - latent_start` is 4, so a
        // correctly-shaped `[c=1, t=4, h=1, w=1]` latent has exactly 4
        // values - deliberately pass 5 (a real mismatch), not the tile's own
        // correct length.
        let tiles = tile_ranges(&[24], 25, 1, VIDEO_TEMPORAL_SCALE, TILE_LEAD_SEGMENTS).unwrap();
        assert_eq!(tiles[0].latent_end_exclusive - tiles[0].latent_start, 4);
        let e = stitch_tile_latents(&[vec![0.0; 5]], &tiles, 1, 1, 1).unwrap_err();
        assert!(e.contains("values, expected"), "{e}");
    }

    // -------------------------------------------------------- target_frame_count

    /// Pinned against a live reference `(requested - 1) * 2**rounds + 1`
    /// computation for the same `(requested, rounds)` pairs the pipeline
    /// exercises.
    #[test]
    fn target_frame_count_matches_the_documented_formula() {
        let cases = [(9, 0, 9), (9, 1, 17), (9, 2, 33), (17, 0, 17), (17, 1, 33), (17, 2, 65), (25, 0, 25), (25, 1, 49), (25, 2, 97), (41, 0, 41), (41, 1, 81), (41, 2, 161)];
        for (requested, rounds, want) in cases {
            assert_eq!(target_frame_count(requested, rounds), want, "requested={requested} rounds={rounds}");
        }
    }

    // ------------------------------------------------------------ keyframe_slots

    #[test]
    fn keyframe_slots_extends_length_marks_the_new_range_and_leaves_base_untouched() {
        let base_t = 2 * 2 * 2; // lat_t=2, lh=2, lw=2
        let base_positions = crate::pipeline::grid_positions(2, 2, 2);
        let base_positions_before = base_positions.clone();
        let layout = keyframe_slots(base_t, &base_positions, 2, 2, &[8, 16], VIDEO_TEMPORAL_SCALE, true).unwrap();
        assert_eq!(layout.tokens_per_slot, 4); // lh*lw
        assert_eq!(layout.first_token, base_t);
        assert_eq!(layout.total_tokens, base_t + 2 * 4);
        assert_eq!(layout.keyframes_mask.len(), layout.total_tokens);
        assert!(layout.keyframes_mask[..base_t].iter().all(|&v| v == 0.0), "base tokens must be unmarked");
        assert!(layout.keyframes_mask[base_t..].iter().all(|&v| v == 1.0), "every appended slot token must be marked");
        // The base axis blocks must be copied through unchanged (spot-check
        // axis 0 and axis 2).
        for axis in [0usize, 2] {
            let old = &base_positions_before[axis * base_t * 2..(axis + 1) * base_t * 2];
            let new = &layout.positions[axis * layout.total_tokens * 2..axis * layout.total_tokens * 2 + base_t * 2];
            assert_eq!(old, new, "axis {axis} base block must be unchanged");
        }
        // Slot 0 (pixel frame 8, temporal_scale 8) sits at latent-grid
        // [1.0, 1.125) on the frame axis - one full latent-frame step in
        // (matches a REGULAR latent frame boundary exactly, since 8/8=1).
        let frame_axis = 0usize;
        let tok = base_t; // first token of slot 0
        let off = frame_axis * layout.total_tokens * 2 + tok * 2;
        assert_eq!((layout.positions[off], layout.positions[off + 1]), (1.0, 1.125));
        // Slot 1 (pixel frame 16) sits at [2.0, 2.125).
        let tok1 = base_t + 4; // first token of slot 1
        let off1 = frame_axis * layout.total_tokens * 2 + tok1 * 2;
        assert_eq!((layout.positions[off1], layout.positions[off1 + 1]), (2.0, 2.125));
        // Spatial axes of a slot token cover the same integer grid as a
        // regular token at that (h, w).
        let h_axis = 1usize;
        let off_h = h_axis * layout.total_tokens * 2 + tok * 2; // slot 0, token (h=0,w=0)
        assert_eq!((layout.positions[off_h], layout.positions[off_h + 1]), (0.0, 1.0));
    }

    #[test]
    fn keyframe_slots_unmarked_leaves_the_mask_all_zero() {
        let base_t = 1;
        let base_positions = vec![0f32; 3 * base_t * 2];
        let layout = keyframe_slots(base_t, &base_positions, 1, 1, &[8], VIDEO_TEMPORAL_SCALE, false).unwrap();
        assert!(layout.keyframes_mask.iter().all(|&v| v == 0.0), "marked=false must leave every token unmarked");
    }

    #[test]
    fn keyframe_slots_rejects_non_increasing_indices() {
        let base_positions = vec![0f32; 3 * 2];
        let e = keyframe_slots(1, &base_positions, 1, 1, &[8, 8], VIDEO_TEMPORAL_SCALE, true).unwrap_err();
        assert!(e.contains("strictly increasing"), "{e}");
    }

    #[test]
    fn keyframe_slots_rejects_a_base_positions_length_mismatch() {
        let e = keyframe_slots(2, &vec![0f32; 3 * 2], 1, 1, &[8], VIDEO_TEMPORAL_SCALE, true).unwrap_err();
        assert!(e.contains("values, expected"), "{e}");
    }
}
