// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DFR (Diffusion Fidelity Rendering) geometry: the keyframe-segment canvas
//! grid and the temporal tile boundaries with their lead-in. Ported from
//! `ltx_pipelines.dfr_helpers.layout` - PURE geometry, no weights, no device,
//! every function here is real and unit-tested (several pinned against
//! numbers read directly off a live run of the reference `dfr_layout.py`,
//! not hand-derived - see this module's tests for the exact python
//! invocations each case reproduces).
//!
//! The generated keyframe slots themselves are appended by the pipeline's
//! stage machinery (`pipeline::append_keyframe_slots`, in the real
//! fps-normalized position convention every real stage uses); this module
//! only says where they go.

/// Candidate keyframe-segment lengths, in pixel frames - `dfr_layout.py`'s
/// `SEGMENT_CANDIDATES`.
pub const SEGMENT_CANDIDATES: [usize; 2] = [24, 32];

/// This port's causal video VAE's temporal stride (`VIDEO_SCALE_FACTORS.
/// time` upstream) - one latent frame per 8 pixel frames, except latent
/// frame 0 which covers pixel frame 0 alone. Matches
/// `vae3d::LtxVaeConfig::latent_frames`'s own `1 + 8k` contract.
pub const VIDEO_TEMPORAL_SCALE: usize = 8;

/// Pick the keyframe segment length from [`SEGMENT_CANDIDATES`], preferring
/// whichever pads `content_frames` least; ties keep the larger segment.
/// `content_frames` is `num_frames - 1` (frame 0 is never part of a
/// segment - see [`resolve_canvas`]). Mirrors
/// `dfr_layout.py::choose_segment_length` exactly.
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
    if !(num_frames - 1).is_multiple_of(temporal_scale) {
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
    if pixel_frame != 0 && !pixel_frame.is_multiple_of(temporal_scale) {
        return Err(format!("pixel_frame {pixel_frame} is not on the x{temporal_scale} latent border"));
    }
    Ok(pixel_frame / temporal_scale)
}

/// One temporal-round tile (`dfr_helpers.layout.TemporalTile`): the latent
/// cells `[start, end)` it owns, their pixel span `[pixel_start, pixel_end]`
/// (inclusive), the seams it anchors on and the segment midpoints it
/// generates slots at - all in the round's global pixel frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalTile {
    pub start: usize,
    pub end: usize,
    pub pixel_start: usize,
    pub pixel_end: usize,
    pub anchors: Vec<usize>,
    pub slots: Vec<usize>,
}

/// `TemporalTilePlan`: split a round's `num_frames`-frame canvas into at most
/// `num_tiles` tiles on its keyframe seams (`split_at_seams` with no
/// overlap - segments dealt largest-first, leftovers to the leading tiles),
/// each owning exactly its segments, and hand every seam and every canvas
/// segment midpoint to the tile that contains it. `seams` are the round's
/// keyframe pixel frames, the last one the canvas's last frame.
pub fn temporal_tile_plan(seams: &[usize], num_frames: usize, num_tiles: usize, temporal_scale: usize) -> Result<Vec<TemporalTile>, String> {
    if num_tiles == 0 {
        return Err("a temporal round needs at least one tile".into());
    }
    let mut boundaries = vec![0usize];
    for &p in seams {
        boundaries.push(pixel_to_latent_index(p, temporal_scale)?);
    }
    if boundaries.windows(2).any(|w| w[1] <= w[0]) {
        return Err(format!("seams must be strictly increasing and past frame 0, got {seams:?}"));
    }
    let latent_len = (num_frames - 1) / temporal_scale + 1;
    if *boundaries.last().expect("holds 0") != latent_len - 1 {
        return Err(format!("the last seam must be the canvas's last frame ({}), got {seams:?}", num_frames - 1));
    }
    let segments = boundaries.len() - 1;
    let tiles = num_tiles.min(segments);
    let (base, leftover) = (segments / tiles, segments % tiles);
    let mut midpoints = Vec::with_capacity(seams.len());
    let mut left = 0;
    for &right in seams {
        midpoints.push((left + right) / 2);
        left = right;
    }
    let mut out = Vec::with_capacity(tiles);
    let mut cursor = 0;
    for ti in 0..tiles {
        let start = if ti == 0 { 0 } else { boundaries[cursor] + 1 };
        cursor += base + usize::from(ti < leftover);
        let end = boundaries[cursor] + 1;
        let (pixel_start, pixel_end) = (start * temporal_scale, (end - 1) * temporal_scale);
        let within = |v: &[usize]| v.iter().copied().filter(|&p| (pixel_start..=pixel_end).contains(&p)).collect();
        out.push(TemporalTile { start, end, pixel_start, pixel_end, anchors: within(seams), slots: within(&midpoints) });
    }
    Ok(out)
}

/// Where a non-first tile starts (`dfr_helpers.ops.TilePrefix`): on the
/// one-frame keyframe plane at the last plane before its seam, then the
/// canvas cells between that plane and the seam, all pinned, so the tile
/// reads as an image-to-video clip and resumes on the frame after the seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePrefix {
    /// The plane that becomes the tile's latent frame 0.
    pub keyframe_position: usize,
    /// The first canvas cell of the pinned video run, right after the plane.
    pub video_start_cell: usize,
    /// Pinned latent frames, the plane included.
    pub cells: usize,
    /// The first pixel frame the tile denoises.
    pub resume_pixel: usize,
}

/// `tile_prefix`: the prefix a tile resuming after `seam_pixel` starts from,
/// out of the planes the round holds so far.
pub fn tile_prefix(seam_pixel: usize, planes: impl IntoIterator<Item = usize>, temporal_scale: usize) -> Result<TilePrefix, String> {
    let keyframe = planes.into_iter().filter(|&p| p < seam_pixel).max().ok_or_else(|| format!("no keyframe plane before seam {seam_pixel} to start the tile on"))?;
    if !keyframe.is_multiple_of(temporal_scale) || !seam_pixel.is_multiple_of(temporal_scale) {
        return Err(format!("keyframe {keyframe} and seam {seam_pixel} must both sit on the x{temporal_scale} border"));
    }
    Ok(TilePrefix { keyframe_position: keyframe, video_start_cell: keyframe / temporal_scale + 1, cells: 1 + (seam_pixel - keyframe) / temporal_scale, resume_pixel: seam_pixel + 1 })
}

/// The frame rate a round's transformer positions run at
/// (`dfr_helpers.ops.conditioning_fps`): above 30 it snaps to 60, the
/// playback rate itself is unchanged.
pub fn conditioning_fps(playback_fps: f64) -> f64 {
    if playback_fps > 30.0 {
        60.0
    } else {
        playback_fps
    }
}

/// The video latent frame a slot at local `position` is seeded from
/// (`slot_initials_from_video`): the nearest one, Python's halves-to-even
/// rounding, clamped into the `latent_frames` the tile holds.
pub fn slot_seed_frame(position: usize, temporal_scale: usize, latent_frames: usize) -> usize {
    ((position as f64 / temporal_scale as f64).round_ties_even() as usize).min(latent_frames - 1)
}

/// `audio_latent_for_tile`: the stage-1 audio for one temporal tile - its
/// playback window `[pixel_start, pixel_start + local_frames) / playback_fps`
/// as a fraction of the stage-1 `source_seconds`, linearly resampled onto
/// `out_tokens` (the tile's own token count at its conditioning rate).
/// `audio` is `[T, token_dim]` token-major.
pub fn audio_for_tile(audio: &[f32], token_dim: usize, pixel_start: usize, local_frames: usize, playback_fps: f64, source_seconds: f64, out_tokens: usize) -> Result<Vec<f32>, String> {
    let full_t = audio.len() / token_dim;
    if full_t == 0 || out_tokens == 0 || local_frames == 0 {
        return Err(format!("cannot window a {full_t}-token audio latent into {out_tokens} tokens over {local_frames} frames"));
    }
    let src_start = pixel_start as f64 / playback_fps / source_seconds * full_t as f64;
    let src_end = (pixel_start + local_frames) as f64 / playback_fps / source_seconds * full_t as f64;
    // The positions in fp32, as the reference's `src_start + step *
    // torch.arange(...)` computes them.
    let (start, step) = (src_start as f32, ((src_end - src_start) / out_tokens as f64) as f32);
    let mut out = Vec::with_capacity(out_tokens * token_dim);
    for k in 0..out_tokens {
        let pos = (start + step * k as f32).clamp(0.0, (full_t - 1) as f32);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(full_t - 1);
        let w = pos - lo as f32;
        out.extend(audio[lo * token_dim..(lo + 1) * token_dim].iter().zip(&audio[hi * token_dim..(hi + 1) * token_dim]).map(|(&a, &b)| a * (1.0 - w) + b * w));
    }
    Ok(out)
}

/// One axis interval of a spatial tile split (`ltx_core.tiling.
/// DimensionInterval`): cells `[start, end)` and the ramps blending it into
/// its neighbours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisInterval {
    pub start: usize,
    pub end: usize,
    pub left_ramp: usize,
    pub right_ramp: usize,
}

/// `split_by_size`: tiles of `size` cells sharing `overlap`, the last one
/// shorter when the axis does not divide evenly.
fn split_by_size(size: usize, overlap: usize, dim: usize) -> Vec<AxisInterval> {
    if dim <= size {
        return vec![AxisInterval { start: 0, end: dim, left_ramp: 0, right_ramp: 0 }];
    }
    let amount = (dim + size - 2 * overlap - 1) / (size - overlap);
    let mut out = vec![AxisInterval { start: 0, end: size, left_ramp: 0, right_ramp: overlap }];
    for i in 1..amount - 1 {
        out.push(AxisInterval { start: i * (size - overlap), end: i * (size - overlap) + size, left_ramp: overlap, right_ramp: overlap });
    }
    out.push(AxisInterval { start: (amount - 1) * (size - overlap), end: dim, left_ramp: overlap, right_ramp: 0 });
    out
}

/// `split_by_count`: `num_tiles` tiles sharing `overlap` cells, the first
/// `remainder` tiles one cell longer.
pub fn split_by_count(num_tiles: usize, overlap: usize, dim: usize) -> Result<Vec<AxisInterval>, String> {
    if num_tiles == 0 || num_tiles > dim {
        return Err(format!("{num_tiles} tiles cannot split {dim} cells"));
    }
    if num_tiles == 1 {
        return Ok(vec![AxisInterval { start: 0, end: dim, left_ramp: 0, right_ramp: 0 }]);
    }
    let total = dim + overlap * (num_tiles - 1);
    let (size, remainder) = (total / num_tiles, total % num_tiles);
    if size <= overlap {
        return Err(format!("{num_tiles} tiles of {size} cells cannot share {overlap} over {dim}"));
    }
    Ok(split_by_size(size, overlap, dim - remainder)
        .into_iter()
        .enumerate()
        .map(|(i, iv)| {
            let (shift, grow) = (i.min(remainder), usize::from(i < remainder));
            AxisInterval { start: iv.start + shift, end: iv.end + shift + grow, ..iv }
        })
        .collect())
}

/// `compute_trapezoidal_mask_1d`: ones with linear ramps in over
/// `left_ramp` cells and out over `right_ramp` (neither reaching 0 or 1 at
/// its end, as `linspace(...)[1:-1]` builds them).
pub fn trapezoid_mask(iv: &AxisInterval) -> Vec<f32> {
    let len = iv.end - iv.start;
    let (l, r) = (iv.left_ramp.min(len), iv.right_ramp.min(len));
    let mut m = vec![1f32; len];
    // `torch.linspace(0, 1, l + 2)[1:-1]` and `linspace(1, 0, r + 2)[1:-1]`.
    let ramp = |n: usize, k: usize| (k as f64 / (n + 1) as f64) as f32;
    for (k, v) in m[..l].iter_mut().enumerate() {
        *v *= ramp(l, k + 1);
    }
    for (k, v) in m[len - r..].iter_mut().enumerate() {
        *v *= 1.0 - ramp(r, k + 1);
    }
    m
}

/// The epilogue's spatial tile count along a `cells`-cell latent axis
/// (`epilogue_tile_count`): the fewest 32-cell tiles overlapping by 2 that
/// cover it.
pub fn epilogue_tile_count(cells: usize) -> usize {
    const TILE: usize = 32;
    const OVERLAP: usize = 2;
    cells.saturating_sub(OVERLAP).div_ceil(TILE - OVERLAP).max(1)
}

/// `clamp_dim_tiling`: a tile count and overlap the axis can hold - one tile
/// when it is shorter than the count, the overlap clamped to `dim - n`.
pub fn clamp_axis_tiling(num_tiles: usize, overlap: usize, dim: usize) -> (usize, usize) {
    if num_tiles <= 1 || dim < num_tiles {
        return (1, 0);
    }
    (num_tiles, overlap.min(dim - num_tiles))
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


/// The final frame-count contract every DFR call honours regardless of how
/// much the canvas padded its tail: `(requested - 1) * 2^rounds + 1`.
/// `requested - 1` is always a multiple of the VAE temporal scale (enforced
/// by [`resolve_canvas`]), so the corresponding latent trim always lands on
/// a latent boundary. Mirrors the formula stated in `dfr_pipeline.py`'s
/// class doc and its `target_frames` computation.
pub fn target_frame_count(requested_frames: usize, rounds: u32) -> usize {
    (requested_frames - 1) * 2usize.pow(rounds) + 1
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

    /// One [`resolve_canvas`] case: `num_frames` in, and the
    /// `(padded_frames, segment, seam_positions)` it must return.
    type CanvasCase = (usize, (usize, usize, &'static [usize]));

    /// Pinned against a live reference `resolve_canvas(...)` run.
    #[test]
    fn resolve_canvas_matches_reference_numbers() {
        let cases: [CanvasCase; 8] = [
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

    // ------------------------------------------------------- temporal rounds

    /// Every plan, prefix and conditioning rate against
    /// `tools/goldens/ltxv_dfr_temporal_dump_reference.py`'s live run of the
    /// reference's `TemporalTilePlan`, `tile_prefix` and `conditioning_fps`,
    /// over canvases with leftover segments and more tiles than segments.
    #[test]
    fn temporal_rounds_match_the_reference_geometry() {
        let path = brain_testutil::testdata("golden/ltxv/dfr_temporal/temporal.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            brain_testutil::skip(&format!("{path} absent - run tools/goldens/ltxv_dfr_temporal_dump_reference.py"));
            return;
        };
        let golden: serde_json::Value = serde_json::from_str(&text).unwrap();
        let nums = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<_>>();
        let n = |v: &serde_json::Value, k: &str| v[k].as_u64().unwrap() as usize;
        for plan in golden["plans"].as_array().unwrap() {
            let (seams, frames, tiles) = (nums(&plan["seams"]), n(plan, "frames"), n(plan, "num_tiles"));
            let got = temporal_tile_plan(&seams, frames, tiles, VIDEO_TEMPORAL_SCALE).unwrap();
            let want = plan["tiles"].as_array().unwrap();
            assert_eq!(got.len(), want.len(), "{plan}");
            let mut planes: Vec<usize> = seams.clone();
            for (ti, (g, w)) in got.iter().zip(want).enumerate() {
                let tile = TemporalTile { start: n(w, "start"), end: n(w, "end"), pixel_start: n(w, "pixel_start"), pixel_end: n(w, "pixel_end"), anchors: nums(&w["anchors"]), slots: nums(&w["slots"]) };
                assert_eq!(g, &tile, "tile {ti} of {plan}");
                if ti > 0 {
                    let p = &w["prefix"];
                    let want = TilePrefix { keyframe_position: n(p, "keyframe_position"), video_start_cell: n(p, "video_start_cell"), cells: n(p, "cells"), resume_pixel: n(p, "resume_pixel") };
                    assert_eq!(tile_prefix((g.start - 1) * VIDEO_TEMPORAL_SCALE, planes.iter().copied(), VIDEO_TEMPORAL_SCALE).unwrap(), want, "prefix of tile {ti} of {plan}");
                }
                planes.extend(&g.slots);
            }
        }
        for (fps, want) in golden["conditioning_fps"].as_object().unwrap() {
            assert_eq!(conditioning_fps(fps.parse().unwrap()), want.as_f64().unwrap(), "conditioning_fps({fps})");
        }
    }

    #[test]
    fn a_tile_plan_rejects_seams_that_miss_the_last_frame() {
        assert!(temporal_tile_plan(&[48, 96], 145, 2, VIDEO_TEMPORAL_SCALE).is_err());
        assert!(temporal_tile_plan(&[96, 48], 97, 2, VIDEO_TEMPORAL_SCALE).is_err());
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
}

