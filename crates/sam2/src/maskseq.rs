// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain/sam2-maskseq/1` - the per-frame mask sequence [`crate::video`] emits
//! and other models consume.
//!
//! One DIRECTORY per tracked object:
//!
//! ```text
//! <dir>/masks.json          the manifest below
//! <dir>/mask_000000.png     frame 0, 8-bit, all three channels equal
//! <dir>/mask_000001.png     frame 1
//! ...
//! ```
//!
//! A directory rather than a container file, on purpose: `ffmpeg -i
//! mask_%06d.png` round-trips it in both directions with no new container code,
//! and the manifest carries per-frame confidence that a video file cannot.
//!
//! # Polarity is DECLARED, never inferred
//!
//! [`Polarity::ObjectWhite`] is what SAM 2 means: 255 is the tracked object.
//! Some consumers want the inverse - LTX-2.5's `VideoConditionByMask` reads
//! `1` as "keep this region clean, exclude it from denoising", so replacing a
//! character means masking the BACKGROUND white - and [`Polarity::ObjectBlack`]
//! writes that directly.
//!
//! Getting this backwards regenerates the entire background and preserves the
//! subject, the exact inverse of the intent, and it does so without erroring.
//! So the contract is: **a consumer MUST read `polarity` from `masks.json` and
//! act on it, and MUST hard-error if the key is missing or unrecognised.**
//! Never infer polarity from the pixels, never assume a default.
//! [`MaskSeq::read`] is that reader, and it refuses rather than guesses.
//!
//! # `object_score`
//!
//! `per_frame[i].object_score` is SAM 2's occlusion logit. **Negative means the
//! model believes the object is absent or fully occluded on that frame**, and
//! its mask is then legitimately empty. A consumer that treats an empty mask as
//! a bug will chase a phantom; one that ignores the score will condition on
//! nothing. The meaning is written into the manifest itself (`notes`), not only
//! into this doc comment, because the manifest is what travels.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// Which value means "the tracked object".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// 255 = object, 0 = background. SAM 2's native meaning, and the default.
    ObjectWhite,
    /// 0 = object, 255 = background - the inverse, for a consumer whose `1`
    /// means "keep, do not regenerate" (LTX-2.5 masked conditioning).
    ObjectBlack,
}

impl Polarity {
    /// The exact string that goes in, and comes out of, `masks.json`.
    pub fn tag(self) -> &'static str {
        match self {
            Polarity::ObjectWhite => "object=255",
            Polarity::ObjectBlack => "object=0",
        }
    }

    pub fn parse(tag: &str) -> Result<Polarity, String> {
        match tag {
            "object=255" => Ok(Polarity::ObjectWhite),
            "object=0" => Ok(Polarity::ObjectBlack),
            other => Err(format!(
                "maskseq: unrecognised polarity {other:?} (expected \"object=255\" or \"object=0\"); \
                 refusing to guess - the two are exact inverses"
            )),
        }
    }
}

/// The format tag written to, and required of, `masks.json`.
pub const FORMAT: &str = "brain/sam2-maskseq/1";
/// The `printf` pattern for the frame files.
pub const PATTERN: &str = "mask_%06d.png";
/// The manifest's file name inside the sequence directory.
pub const MANIFEST: &str = "masks.json";

/// One frame's row in the manifest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameInfo {
    pub frame: usize,
    /// SAM 2's occlusion logit. `<= 0` means the object is believed absent.
    pub object_score: f32,
    /// The predicted IoU of the mask that was kept.
    pub iou: f32,
    /// Object pixels after thresholding, in the emitted polarity's terms (i.e.
    /// always the count of OBJECT pixels, whichever value they carry).
    pub area_px: u32,
}

/// A mask sequence being written, or one that was read back.
#[derive(Debug, Clone)]
pub struct MaskSeq {
    pub dir: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub polarity: Polarity,
    /// Hard 0/255 (the default) versus the soft sigmoid ramp.
    pub binary: bool,
    /// The logit threshold used when `binary`. SAM 2's own is 0.
    pub threshold: f32,
    pub object_id: u32,
    /// `(source file name, its frame count as the segmenter saw it)`.
    pub source: (String, usize),
    /// `(x, y, label)` in SOURCE-image pixels, and the frame they sat on.
    pub prompt_frame: usize,
    pub prompt_points: Vec<(f32, f32, f32)>,
    pub frames: Vec<FrameInfo>,
}

impl MaskSeq {
    /// A sequence about to be written into `dir`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dir: impl Into<PathBuf>,
        width: u32,
        height: u32,
        fps: f64,
        polarity: Polarity,
        object_id: u32,
        source: (String, usize),
        prompt_frame: usize,
        prompt_points: Vec<(f32, f32, f32)>,
    ) -> MaskSeq {
        MaskSeq {
            dir: dir.into(),
            width,
            height,
            fps,
            polarity,
            binary: true,
            threshold: 0.0,
            object_id,
            source,
            prompt_frame,
            prompt_points,
            frames: Vec::new(),
        }
    }

    /// Path of frame `i`'s PNG.
    pub fn frame_path(&self, i: usize) -> PathBuf {
        self.dir.join(format!("mask_{i:06}.png"))
    }

    /// Write one frame from `logits` - the mask logits at `width x height`, row
    /// major - and record its row.
    ///
    /// `binary` thresholds at [`MaskSeq::threshold`] (SAM 2's `logit > 0`);
    /// otherwise the sigmoid is written as a 0..255 ramp. Either way the value
    /// is replicated across R, G and B, so a consumer can read channel 0 - the
    /// convention `capability::blob::decode_plane` already uses.
    pub fn write_frame(&mut self, i: usize, logits: &[f32], object_score: f32, iou: f32) -> Result<(), String> {
        let (w, h) = (self.width as usize, self.height as usize);
        if logits.len() != w * h {
            return Err(format!("maskseq: frame {i} has {} logits, expected {}x{} = {}", logits.len(), w, h, w * h));
        }
        let mut px = vec![0u8; w * h * 3];
        let mut area = 0u32;
        for (p, &l) in logits.iter().enumerate() {
            let is_obj = l > self.threshold;
            if is_obj {
                area += 1;
            }
            // The stored byte is always "how much object is here"; the polarity
            // flip is the LAST step, so `area_px` counts object pixels in both.
            let mut v = if self.binary {
                if is_obj {
                    255u8
                } else {
                    0u8
                }
            } else {
                let s = 1.0 / (1.0 + (-l).exp());
                (s * 255.0 + 0.5).clamp(0.0, 255.0) as u8
            };
            if self.polarity == Polarity::ObjectBlack {
                v = 255 - v;
            }
            px[p * 3] = v;
            px[p * 3 + 1] = v;
            px[p * 3 + 2] = v;
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("maskseq: {}: {e}", self.dir.display()))?;
        let img = imaging::Rgb8::new(self.width, self.height, px)?;
        imaging::save_png(self.frame_path(i), &img)?;
        self.frames.push(FrameInfo { frame: i, object_score, iou, area_px: area });
        Ok(())
    }

    /// Write `masks.json`. Fails if the recorded frames are not `0..n`
    /// contiguous, or do not match the source's frame count: a consumer that
    /// lines masks up with a clip cannot recover from a silent gap.
    pub fn write_manifest(&self) -> Result<PathBuf, String> {
        for (i, f) in self.frames.iter().enumerate() {
            if f.frame != i {
                return Err(format!("maskseq: frame rows are not contiguous from 0 (row {i} is frame {})", f.frame));
            }
        }
        if self.frames.len() != self.source.1 {
            return Err(format!(
                "maskseq: wrote {} frames but the source {} has {} - a mask sequence must cover every source frame, \
                 and truncating or padding it silently would misalign every frame after the gap",
                self.frames.len(),
                self.source.0,
                self.source.1
            ));
        }
        let per_frame: Vec<Value> = self
            .frames
            .iter()
            .map(|f| json!({"frame": f.frame, "object_score": f.object_score, "iou": f.iou, "area_px": f.area_px}))
            .collect();
        let points: Vec<Value> = self.prompt_points.iter().map(|(x, y, l)| json!([x, y, l])).collect();
        let m = json!({
            "format": FORMAT,
            "pattern": PATTERN,
            "frames": self.frames.len(),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "polarity": self.polarity.tag(),
            "binary": self.binary,
            "threshold": self.threshold,
            "object_id": self.object_id,
            "source": {"name": self.source.0, "frames": self.source.1},
            "prompt": {"frame": self.prompt_frame, "points": points, "box": Value::Null},
            "notes": {
                "polarity": "REQUIRED reading. \"object=255\" means white is the tracked object; \
                             \"object=0\" is the exact inverse. A consumer must act on this key and must \
                             hard-error if it is missing or unrecognised - never infer it from the pixels.",
                "object_score": "SAM 2's occlusion logit. <= 0 means the object is believed absent or fully \
                                 occluded on that frame, so an empty mask there is the model's answer, not a bug.",
                "resolution": "Masks are at SOURCE pixel resolution, one PNG per source frame, contiguous from 0. \
                               Any spatial or temporal downsampling is the consumer's, done against its own grid.",
                "channels": "8-bit; R, G and B carry the same value, so channel 0 is the mask."
            },
            "per_frame": per_frame,
        });
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("maskseq: {}: {e}", self.dir.display()))?;
        let path = self.dir.join(MANIFEST);
        std::fs::write(&path, serde_json::to_vec_pretty(&m).map_err(|e| e.to_string())?)
            .map_err(|e| format!("maskseq: {}: {e}", path.display()))?;
        Ok(path)
    }

    /// Read a sequence's manifest back.
    ///
    /// Every failure mode here is an ERROR, never a default: an unreadable
    /// manifest, a missing or unrecognised `polarity`, a wrong `format`. That is
    /// the whole point of the contract - see the module docs.
    pub fn read(dir: impl AsRef<Path>) -> Result<MaskSeq, String> {
        let dir = dir.as_ref().to_path_buf();
        let path = dir.join(MANIFEST);
        let raw = std::fs::read(&path).map_err(|e| {
            format!("maskseq: cannot read {} ({e}) - refusing to consume a mask sequence with no manifest", path.display())
        })?;
        let m: Value = serde_json::from_slice(&raw).map_err(|e| format!("maskseq: {} is not JSON: {e}", path.display()))?;
        let fmt = m["format"].as_str().unwrap_or_default();
        if fmt != FORMAT {
            return Err(format!("maskseq: {} declares format {fmt:?}, expected {FORMAT:?}", path.display()));
        }
        let polarity = match m["polarity"].as_str() {
            Some(t) => Polarity::parse(t)?,
            None => return Err(format!("maskseq: {} has no \"polarity\" key; refusing to assume one", path.display())),
        };
        let u = |k: &str| -> Result<u64, String> {
            m[k].as_u64().ok_or_else(|| format!("maskseq: {} has no integer {k:?}", path.display()))
        };
        let frames: Vec<FrameInfo> = m["per_frame"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|r| FrameInfo {
                        frame: r["frame"].as_u64().unwrap_or_default() as usize,
                        object_score: r["object_score"].as_f64().unwrap_or_default() as f32,
                        iou: r["iou"].as_f64().unwrap_or_default() as f32,
                        area_px: r["area_px"].as_u64().unwrap_or_default() as u32,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let n = u("frames")? as usize;
        if frames.len() != n {
            return Err(format!("maskseq: {} says {n} frames but lists {} rows", path.display(), frames.len()));
        }
        let points = m["prompt"]["points"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|p| {
                        let g = |i: usize| p[i].as_f64().unwrap_or_default() as f32;
                        (g(0), g(1), g(2))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(MaskSeq {
            dir,
            width: u("width")? as u32,
            height: u("height")? as u32,
            fps: m["fps"].as_f64().unwrap_or(0.0),
            polarity,
            binary: m["binary"].as_bool().unwrap_or(true),
            threshold: m["threshold"].as_f64().unwrap_or(0.0) as f32,
            object_id: m["object_id"].as_u64().unwrap_or(0) as u32,
            source: (
                m["source"]["name"].as_str().unwrap_or_default().to_string(),
                m["source"]["frames"].as_u64().unwrap_or(n as u64) as usize,
            ),
            prompt_frame: m["prompt"]["frame"].as_u64().unwrap_or(0) as usize,
            prompt_points: points,
            frames,
        })
    }

    /// Frames the model believes the object is absent from, by index.
    pub fn occluded_frames(&self) -> BTreeMap<usize, f32> {
        self.frames.iter().filter(|f| f.object_score <= 0.0).map(|f| (f.frame, f.object_score)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("brain-sam2-maskseq/{name}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A logit map with a known object area, so the round trip is checkable.
    fn logits() -> Vec<f32> {
        let mut v = vec![-4.0f32; 4 * 3];
        v[0] = 2.0;
        v[5] = 0.5;
        v
    }

    #[test]
    fn round_trips_through_the_manifest() {
        let d = tmpdir("roundtrip");
        let mut s = MaskSeq::new(&d, 4, 3, 24.0, Polarity::ObjectWhite, 7, ("clip.mp4".into(), 2), 0, vec![(10.0, 20.0, 1.0)]);
        s.write_frame(0, &logits(), 3.5, 0.9).unwrap();
        s.write_frame(1, &logits(), -1.25, 0.4).unwrap();
        s.write_manifest().unwrap();

        let r = MaskSeq::read(&d).unwrap();
        assert_eq!(r.polarity, Polarity::ObjectWhite);
        assert_eq!((r.width, r.height, r.object_id), (4, 3, 7));
        assert_eq!(r.source, ("clip.mp4".to_string(), 2));
        assert_eq!(r.frames.len(), 2);
        assert_eq!(r.frames[0].area_px, 2);
        assert_eq!(r.occluded_frames().keys().copied().collect::<Vec<_>>(), vec![1]);
        assert!(s.frame_path(1).exists());
    }

    /// The polarity flip must invert the PIXELS and leave `area_px` counting
    /// object pixels, so a consumer cannot silently read the inverse.
    #[test]
    fn polarity_inverts_pixels_but_not_the_area_count() {
        for (pol, want) in [(Polarity::ObjectWhite, 255u8), (Polarity::ObjectBlack, 0u8)] {
            let d = tmpdir(&format!("pol{}", pol.tag()));
            let mut s = MaskSeq::new(&d, 4, 3, 24.0, pol, 0, ("c.mp4".into(), 1), 0, vec![]);
            s.write_frame(0, &logits(), 1.0, 1.0).unwrap();
            s.write_manifest().unwrap();
            assert_eq!(s.frames[0].area_px, 2, "area is object pixels regardless of polarity");
            let img = imaging::load(s.frame_path(0)).unwrap();
            assert_eq!(img.px[0], want, "{}: object pixel", pol.tag());
            assert_eq!(img.px[3], 255 - want, "{}: background pixel", pol.tag());
            assert_eq!(MaskSeq::read(&d).unwrap().polarity, pol);
        }
    }

    #[test]
    fn a_missing_polarity_is_an_error_not_a_default() {
        let d = tmpdir("nopolarity");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(MANIFEST), format!(r#"{{"format":"{FORMAT}","frames":0,"width":1,"height":1,"per_frame":[]}}"#)).unwrap();
        let e = MaskSeq::read(&d).unwrap_err();
        assert!(e.contains("polarity"), "{e}");
        // ...and an unrecognised one is refused rather than guessed.
        assert!(Polarity::parse("white").unwrap_err().contains("refusing to guess"));
    }

    #[test]
    fn a_short_sequence_is_an_error() {
        let d = tmpdir("short");
        let mut s = MaskSeq::new(&d, 4, 3, 24.0, Polarity::ObjectWhite, 0, ("clip.mp4".into(), 9), 0, vec![]);
        s.write_frame(0, &logits(), 1.0, 1.0).unwrap();
        let e = s.write_manifest().unwrap_err();
        assert!(e.contains("must cover every source frame"), "{e}");
    }
}
