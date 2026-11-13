// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

use super::geometry::*;
use super::stills::pick_still;
use super::*;

    /// The canvas holds the whole actor with margin, is at least 3:4 wide and
    /// stays inside the frame sideways; only an actor the bottom edge cuts
    /// off gets canvas below the frame.
    #[test]
    fn the_canvas_holds_the_actor_and_room_for_what_the_frame_cuts_off() {
        for actor in [Box2 { x: 300, y: 100, w: 80, h: 200 }, Box2 { x: 0, y: 200, w: 120, h: 248 }, Box2 { x: 700, y: 0, w: 68, h: 448 }, Box2 { x: 250, y: 30, w: 300, h: 418 }] {
            let below = if actor.y + actor.h >= 448 { actor.h } else { 0 };
            let c = canvas_around(actor, below, 768, 448);
            assert!(c.x <= actor.x && c.y <= actor.y && c.x + c.w >= actor.x + actor.w && c.y + c.h >= actor.y + actor.h, "{actor:?} -> {c:?}");
            assert!(c.x + c.w <= 768, "{c:?} leaves the frame sideways");
            assert!(c.w * 4 >= c.h * 3 || c.w == 768, "{c:?} is narrower than 3:4");
            if actor.y + actor.h < 448 {
                assert!(c.y + c.h <= 448, "{actor:?} is whole on the frame, so is its canvas {c:?}");
            } else {
                assert_eq!(c.y + c.h, 448 + below, "{actor:?} is cut off by the bottom edge: its canvas extends below it by what is missing");
            }
        }
        assert_eq!(edit_size(300, 400, 768), (576, 768));
    }

    /// The appearance reference is the top of the actor's silhouette - head,
    /// hair and clothes, with a tenth of its size as margin - and not the
    /// legs, whose pose changes from guide to guide. It stays inside the
    /// frame, and an actor who is not there has no appearance to carry.
    #[test]
    fn the_appearance_reference_is_the_head_hair_and_clothes_band() {
        let (w, h) = (100usize, 100usize);
        let actor: Vec<bool> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                ((20..40).contains(&y) && (40..60).contains(&x)) || ((40..95).contains(&y) && (10..90).contains(&x))
            })
            .collect();
        let b = appearance_box(&actor, w, h).unwrap();
        let full = bounding_box(&actor, w, h).unwrap();
        assert!(b.y <= 20 && b.y + b.h < full.y + full.h, "the head and clothes, not the legs: {b:?} of {full:?}");
        assert!(b.w >= 20 && b.h >= 20, "the band is the character's own: {b:?}");
        assert!(b.x + b.w <= w && b.y + b.h <= h, "{b:?} leaves the frame");
        let edge = appearance_box(&vec![true; w * h], w, h).unwrap();
        assert_eq!((edge.x, edge.y, edge.x + edge.w), (0, 0, w), "an actor on every edge clamps to the frame: {edge:?}");
        assert_eq!(appearance_box(&vec![false; w * h], w, h), None, "no actor, no appearance");
    }

    /// How much body the bottom edge cuts off is the silhouette's expected
    /// height (`ratio` times its median row width) less what is on the
    /// frame, at most one more actor height; a figure standing whole on the
    /// edge, or not reaching it, is missing nothing.
    #[test]
    fn the_body_below_the_frame_is_estimated_from_the_silhouette() {
        let (w, h) = (64, 64);
        let actor = |rect: &dyn Fn(usize, usize) -> bool| (0..w * h).map(|i| rect(i % w, i / w)).collect::<Vec<bool>>();
        // 54 rows of a 10-wide body standing on the edge: 4 x 10 = 40 < 54.
        let standing = actor(&|x, y| y >= 10 && (20..30).contains(&x));
        // 24 rows of a 16-wide torso cut by the edge: 4 x 16 - 24 = 40, capped at 24.
        let cut = actor(&|x, y| y >= 40 && (20..36).contains(&x));
        // 12 rows of a 10-wide body: 4 x 10 - 12 = 28 would be missing, but it does not reach the edge.
        let floating = actor(&|x, y| (10..22).contains(&y) && (20..30).contains(&x));
        for (name, m, want) in [("standing", &standing, 0), ("cut", &cut, 24), ("floating", &floating, 0)] {
            let b = bounding_box(m, w, h).unwrap();
            assert_eq!(body_below_frame(m, b, w, h, 4.0), want, "{name}");
        }
    }

    /// Among the edits that cover the actor, the one whose face is closest
    /// to the sheet's wins; without a measurable face, the best overlap; and
    /// when none covers the actor, the best overlap (dropped or refused by
    /// the caller).
    #[test]
    fn the_still_whose_face_matches_the_sheet_best_is_kept() {
        assert_eq!(pick_still(&[(0.9, Some(0.1)), (0.6, Some(0.4)), (0.8, Some(0.3))], 0.5), 1);
        assert_eq!(pick_still(&[(0.9, Some(0.1)), (0.4, Some(0.6))], 0.5), 0, "a face on a figure that misses the actor does not count");
        assert_eq!(pick_still(&[(0.7, None), (0.9, None)], 0.5), 1, "no face measured: the best overlap");
        assert_eq!(pick_still(&[(0.7, None), (0.6, Some(0.2))], 0.5), 1, "a measured face beats none");
        assert_eq!(pick_still(&[(0.3, Some(0.5)), (0.4, None)], 0.5), 1, "nothing covers the actor: the best overlap");
    }

    /// A figure follows the actor: shifted with the actor's centroid and
    /// scaled with the square root of the area change.
    #[test]
    fn a_figure_is_carried_along_the_actor_track() {
        let (w, h) = (64, 64);
        let square = |x0: usize, y0: usize, s: usize| (0..w * h).map(|i| (x0..x0 + s).contains(&(i % w)) && (y0..y0 + s).contains(&(i / w))).collect::<Vec<bool>>();
        let carried = carry(&square(10, 10, 8), &square(10, 10, 8), &square(30, 20, 16), w, h);
        assert_eq!(bounding_box(&carried, w, h), Some(Box2 { x: 30, y: 20, w: 16, h: 16 }));
        assert_eq!(count(&carry(&square(10, 10, 8), &vec![false; w * h], &square(30, 20, 16), w, h)), 0, "no actor, nothing carried");
    }

    /// Guide frames: the default spacing ends on the last frame, and an
    /// asked-for list is sorted, never frame 0, never past the clip.
    #[test]
    fn guide_frames_are_inside_the_clip_and_in_order() {
        assert_eq!(guide_frames(None, 40, 121).unwrap(), vec![40, 80, 120]);
        assert_eq!(guide_frames(None, 32, 97).unwrap(), vec![32, 64, 96]);
        assert_eq!(guide_frames(Some(&[96, 32]), 40, 121).unwrap(), vec![32, 96]);
        assert!(guide_frames(Some(&[]), 40, 121).unwrap().is_empty());
        for bad in [&[0][..], &[121], &[8, 8]] {
            assert!(guide_frames(Some(bad), 40, 121).is_err(), "{bad:?}");
        }
    }

    /// A clip LTX-2.5 cannot take is trimmed at its end to 1 + 8k frames and
    /// centre-cropped to multiples of 32, the clicks moving with the crop;
    /// one it can take is untouched, and `--no-conform` refuses instead.
    #[test]
    fn a_clip_is_conformed_to_the_latent_grid_with_its_clicks() {
        let clip = |w: u32, h: u32, n: usize| crate::pipeline::Video {
            width: w,
            height: h,
            fps: 24,
            frames: (0..n).map(|f| (0..w * h * 3).map(|i| (i as usize + f) as u8).collect()).collect(),
            audio: None,
        };
        let Conformed { clip: c, points, note, .. } = conform(clip(100, 70, 20), &[(50.0, 40.0, 1.0)], None, true).unwrap();
        assert_eq!((c.width, c.height, c.frames.len()), (96, 64, 17));
        assert_eq!(points, vec![(48.0, 37.0, 1.0)], "the clicks move with the centre crop (offset 2, 3)");
        assert_eq!(c.frames[16][0], clip(100, 70, 20).frames[16][(3 * 100 + 2) * 3], "frame 16's first pixel is the source's (2, 3)");
        assert!(note.is_some());
        let note = conform(clip(64, 32, 9), &[(1.0, 1.0, 1.0)], None, true).unwrap().note;
        assert!(note.is_none(), "a representable clip is untouched");
        assert!(conform(clip(100, 70, 20), &[(50.0, 40.0, 1.0)], None, false).err().is_some(), "conform=false refuses");
        assert!(conform(clip(100, 70, 20), &[(1.0, 1.0, 1.0)], None, true).is_err(), "a click the crop cuts away is an error");
        assert!(conform(clip(64, 64, 8), &[(1.0, 1.0, 1.0)], None, true).is_err(), "fewer than 9 frames");
    }

    /// A track written to the work directory reads back as it was: it is
    /// what a resumed run takes in place of tracking again.
    #[test]
    fn a_written_track_reads_back_unchanged() {
        let (w, h, n) = (32usize, 32usize, 3usize);
        let track: Vec<Vec<bool>> = (0..n).map(|f| (0..w * h).map(|i| i % w >= 4 + f && i / w >= 8 && i / w < 20).collect()).collect();
        let dir = std::env::temp_dir().join(format!("brain-replace-track-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_masks(&dir, &track, w, h, 24).unwrap();
        let back = track_of(&MaskClip::read_dir(&dir).unwrap(), w as u32, h as u32, n);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(back.unwrap(), track);
    }

    #[test]
    fn clicks_spread_over_the_silhouette() {
        let (w, h) = (32, 32);
        let m: Vec<bool> = (0..w * h).map(|i| (8..24).contains(&(i % w)) && (4..28).contains(&(i / w))).collect();
        let pts = spread_points(&m, w, h);
        assert_eq!(pts.len(), 10);
        assert!(pts.iter().all(|&(x, y)| m[y as usize * w + x.round() as usize]), "{pts:?}");
    }
