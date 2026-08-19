// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Write a captioned video-clip dataset: a [`crate::episode`] dataset (one
//! episode per clip, action 0 on every frame, no rewards) plus a
//! `captions.json` array in the same episode order - the exact layout
//! `wan::finetune::ClipSet::load_dir` reads.

use std::path::Path;

use crate::episode::EpisodeWriter;

/// `[h,w,3]` `f32` `[0,1]` -> CHW `u8` - the layout [`EpisodeWriter::push`]
/// wants (`frames.u8` is raw CHW, per `crate::episode`'s module doc).
fn hwc_f32_to_chw_u8(hwc: &[f32], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0u8; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                out[c * h * w + y * w + x] = (hwc[(y * w + x) * 3 + c].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

/// Write `clips` (`(caption, frames)` pairs, each frame HWC `f32` `[0,1]`) to
/// `out_dir` as a `data::episode` dataset plus `captions.json`. One episode
/// per clip, action 0 on every frame, no rewards. Returns the clip count.
///
/// Errors if `clips` is empty or a frame's size does not match `w*h*3`; the
/// clip's own frame count is free to vary (a `data::episode` dataset does not
/// require uniform episode lengths - `wan::finetune::ClipSet` windows within
/// whatever each clip actually has).
pub fn write_clipset(out_dir: &Path, clips: &[(String, Vec<Vec<f32>>)], w: u32, h: u32, fps: u32) -> Result<usize, String> {
    if clips.is_empty() {
        return Err("videoset: no clips to write".into());
    }
    let mut wr = EpisodeWriter::create(out_dir, 3, h, w, 1, fps)?;
    for (i, (_, frames)) in clips.iter().enumerate() {
        if frames.is_empty() {
            return Err(format!("videoset: clip {i} has no frames"));
        }
        for (j, f) in frames.iter().enumerate() {
            if f.len() != (w * h * 3) as usize {
                return Err(format!("videoset: clip {i} frame {j} is {} values, expected {}", f.len(), w * h * 3));
            }
            wr.push(&hwc_f32_to_chw_u8(f, w, h), 0, None)?;
        }
        wr.end_episode();
    }
    let n = clips.len();
    wr.finalize()?;
    let captions: Vec<&str> = clips.iter().map(|(c, _)| c.as_str()).collect();
    let json = serde_json::to_string(&captions).map_err(|e| format!("videoset: captions.json: {e}"))?;
    std::fs::write(out_dir.join("captions.json"), json).map_err(|e| format!("videoset: writing captions.json: {e}"))?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::EpisodeDataset;

    fn solid_frame(w: u32, h: u32, color: [f32; 3]) -> Vec<f32> {
        (0..(w * h) as usize).flat_map(|_| color).collect()
    }

    #[test]
    fn write_clipset_round_trips_frames_actions_and_captions_in_order() {
        let dir = std::env::temp_dir().join(format!("videoset_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (w, h) = (4u32, 3u32);
        let clips = vec![
            ("clip a".to_string(), vec![solid_frame(w, h, [1.0, 0.0, 0.0]); 5]),
            ("clip b".to_string(), vec![solid_frame(w, h, [0.0, 1.0, 0.0]); 3]),
        ];
        let n = write_clipset(&dir, &clips, w, h, 12).expect("write");
        assert_eq!(n, 2);

        let ds = EpisodeDataset::open(&dir).expect("open");
        assert_eq!((ds.n, ds.c, ds.h, ds.w, ds.fps), (8, 3, h, w, 12));
        assert_eq!(ds.episodes.len(), 2);
        assert_eq!(ds.episodes[0].len, 5);
        assert_eq!(ds.episodes[1].len, 3);
        assert!(ds.actions().iter().all(|&a| a == 0));
        assert!(ds.rewards().is_none());

        // The first clip's frame 0 must be pure red in CHW/255 layout.
        let f0 = ds.frame_f32(0).unwrap();
        let plane = (w * h) as usize;
        assert!(f0[..plane].iter().all(|&v| (v - 1.0).abs() < 1e-3), "R plane");
        assert!(f0[plane..2 * plane].iter().all(|&v| v.abs() < 1e-3), "G plane");

        let caps: Vec<String> = serde_json::from_str(&std::fs::read_to_string(dir.join("captions.json")).unwrap()).unwrap();
        assert_eq!(caps, vec!["clip a", "clip b"]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_clipset_rejects_a_mis_sized_frame_and_an_empty_set() {
        let dir = std::env::temp_dir().join(format!("videoset_bad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let bad = vec![("x".to_string(), vec![vec![0.0f32; 5]])];
        assert!(write_clipset(&dir, &bad, 4, 3, 8).is_err());
        assert!(!dir.exists(), "a rejected write must leave no dataset dir");
        assert!(write_clipset(&dir, &[], 4, 3, 8).is_err());
    }
}
