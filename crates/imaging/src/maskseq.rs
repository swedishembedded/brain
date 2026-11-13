// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain/maskseq/1` - one object's mask on every frame of a clip.
//!
//! A tracker writes it, a masked video editor reads it. It has two forms that
//! carry exactly the same information:
//!
//! - **a blob** (`capability::Media::Video`, one channel): the frames as f32
//!   in `[0,1]`, meta `{frames, w, h, c: 1, fps, maskseq: {...}}` where
//!   `maskseq` is the manifest below. This is what a served action takes and
//!   returns.
//! - **a directory**, for a person to look at and for other tools:
//!
//!   ```text
//!   <dir>/masks.json          the manifest
//!   <dir>/mask_000000.png     frame 0, 8-bit, R = G = B
//!   <dir>/mask_000001.png     frame 1
//!   ```
//!
//!   `ffmpeg -i mask_%06d.png` round-trips the frames, and the manifest keeps
//!   the per-frame confidence a video file cannot.
//!
//! # Polarity is declared, never inferred
//!
//! [`Polarity::ObjectWhite`]: 1 (255 on disk) is the tracked object.
//! [`Polarity::ObjectBlack`] is the inverse, for a consumer whose 1 means
//! "keep this region": LTX-2.5's masked conditioning reads 1 as "hold", so
//! replacing a person means masking the background white. Read backwards, a
//! sequence regenerates the whole background and keeps the person, without
//! an error. So a reader takes `polarity` from the manifest and refuses one
//! that is missing or unknown; nothing here infers it from the pixels.
//!
//! # `object_score`
//!
//! The tracker's occlusion logit per frame: `<= 0` means the object is
//! believed absent there, so an empty mask on that frame is the model's
//! answer, not a defect. The manifest says so in its `notes`, because the
//! manifest is what travels.
//!
//! Swedish Embedded AB implements video segmentation and masked video editing
//! pipelines for its clients. If your team needs expertise in object tracking
//! and generative video editing, you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use capability::{Blob, Media};
use serde_json::{json, Value};

/// The format tag a manifest carries and a reader requires.
pub const FORMAT: &str = "brain/maskseq/1";
/// The manifest's file name inside a sequence directory.
pub const MANIFEST: &str = "masks.json";

/// Which value means "the tracked object".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// 1 (255) is the object, 0 the background: a tracker's own meaning.
    ObjectWhite,
    /// 0 is the object, 1 (255) the background.
    ObjectBlack,
}

impl Polarity {
    /// The exact string in the manifest.
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
            other => Err(format!("maskseq: unknown polarity {other:?} (\"object=255\" or \"object=0\"); refusing to guess - the two are exact inverses")),
        }
    }

    /// How much of the object a stored value in this polarity means.
    pub fn objectness(self, stored: f32) -> f32 {
        match self {
            Polarity::ObjectWhite => stored,
            Polarity::ObjectBlack => 1.0 - stored,
        }
    }
}

/// One frame's row in the manifest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameInfo {
    /// The tracker's occlusion logit; `<= 0` means the object is believed absent.
    pub object_score: f32,
    /// The tracker's predicted IoU for the mask it kept.
    pub iou: f32,
    /// Object pixels (stored objectness above one half), whatever the polarity.
    pub area_px: u32,
}

/// A mask sequence in memory.
#[derive(Debug, Clone)]
pub struct MaskClip {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub polarity: Polarity,
    /// Hard 0/1 masks, or the soft probability ramp.
    pub binary: bool,
    pub object_id: u32,
    /// The clip the masks were made from: its file name, as the tracker saw it.
    pub source: String,
    /// The frame the prompt sat on, and its `(x, y, label)` clicks in source
    /// pixels (label 1 on the object, 0 off it).
    pub prompt_frame: usize,
    pub prompt_points: Vec<(f32, f32, f32)>,
    /// One `width * height` plane per frame, stored values in `[0,1]` in
    /// [`Self::polarity`].
    pub frames: Vec<Vec<f32>>,
    pub info: Vec<FrameInfo>,
}

impl MaskClip {
    /// An empty sequence of `width x height` frames; [`Self::push_logits`]
    /// adds them.
    pub fn new(width: u32, height: u32, fps: f64, polarity: Polarity, binary: bool) -> MaskClip {
        MaskClip { width, height, fps, polarity, binary, object_id: 0, source: String::new(), prompt_frame: 0, prompt_points: Vec::new(), frames: Vec::new(), info: Vec::new() }
    }

    /// Add the next frame from mask logits at `width x height` (positive is
    /// the object): thresholded at 0 when [`Self::binary`], else the sigmoid.
    pub fn push_logits(&mut self, logits: &[f32], object_score: f32, iou: f32) -> Result<(), String> {
        let n = self.plane();
        if logits.len() != n {
            return Err(format!("maskseq: frame {} has {} values, expected {}x{}", self.frames.len(), logits.len(), self.width, self.height));
        }
        let mut area = 0u32;
        let plane = logits
            .iter()
            .map(|&l| {
                area += u32::from(l > 0.0);
                let object = if self.binary { f32::from(u8::from(l > 0.0)) } else { 1.0 / (1.0 + (-l).exp()) };
                self.polarity.objectness(object)
            })
            .collect();
        self.frames.push(plane);
        self.info.push(FrameInfo { object_score, iou, area_px: area });
        Ok(())
    }

    fn plane(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Frames the tracker believes the object is absent from, with their score.
    pub fn occluded_frames(&self) -> BTreeMap<usize, f32> {
        self.info.iter().enumerate().filter(|(_, f)| f.object_score <= 0.0).map(|(i, f)| (i, f.object_score)).collect()
    }

    /// The manifest: everything but the pixels.
    pub fn manifest(&self) -> Value {
        let per_frame: Vec<Value> = self.info.iter().enumerate().map(|(i, f)| json!({"frame": i, "object_score": f.object_score, "iou": f.iou, "area_px": f.area_px})).collect();
        json!({
            "format": FORMAT,
            "frames": self.frames.len(),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "polarity": self.polarity.tag(),
            "binary": self.binary,
            "object_id": self.object_id,
            "source": self.source,
            "prompt": {"frame": self.prompt_frame, "points": self.prompt_points.iter().map(|(x, y, l)| json!([x, y, l])).collect::<Vec<_>>()},
            "notes": {
                "polarity": "Required reading: \"object=255\" means white is the tracked object, \"object=0\" is the exact inverse. A reader acts on this key and refuses a sequence without it.",
                "object_score": "The tracker's occlusion logit: <= 0 means the object is believed absent on that frame, so an empty mask there is the model's answer.",
                "resolution": "One mask per source frame at the source's size, contiguous from 0."
            },
            "per_frame": per_frame,
        })
    }

    /// Everything [`Self::manifest`] holds, read back; `frames` is filled by
    /// the caller. Every missing or unknown key that matters is an error.
    fn from_manifest(m: &Value, origin: &str) -> Result<MaskClip, String> {
        let fmt = m["format"].as_str().unwrap_or_default();
        if fmt != FORMAT {
            return Err(format!("maskseq: {origin} declares format {fmt:?}, expected {FORMAT:?}"));
        }
        let polarity = Polarity::parse(m["polarity"].as_str().ok_or_else(|| format!("maskseq: {origin} has no \"polarity\"; refusing to assume one"))?)?;
        let int = |k: &str| m[k].as_u64().ok_or_else(|| format!("maskseq: {origin} has no integer {k:?}"));
        let n = int("frames")? as usize;
        let info: Vec<FrameInfo> = m["per_frame"]
            .as_array()
            .map(|rows| rows.iter().map(|r| FrameInfo { object_score: r["object_score"].as_f64().unwrap_or(1.0) as f32, iou: r["iou"].as_f64().unwrap_or(0.0) as f32, area_px: r["area_px"].as_u64().unwrap_or(0) as u32 }).collect())
            .unwrap_or_default();
        if info.len() != n {
            return Err(format!("maskseq: {origin} says {n} frames but lists {} rows", info.len()));
        }
        let points = m["prompt"]["points"]
            .as_array()
            .map(|a| a.iter().map(|p| (p[0].as_f64().unwrap_or(0.0) as f32, p[1].as_f64().unwrap_or(0.0) as f32, p[2].as_f64().unwrap_or(1.0) as f32)).collect())
            .unwrap_or_default();
        Ok(MaskClip {
            width: int("width")? as u32,
            height: int("height")? as u32,
            fps: m["fps"].as_f64().filter(|f| *f > 0.0).ok_or_else(|| format!("maskseq: {origin} has no frame rate"))?,
            polarity,
            binary: m["binary"].as_bool().unwrap_or(true),
            object_id: m["object_id"].as_u64().unwrap_or(0) as u32,
            source: m["source"].as_str().unwrap_or_default().to_string(),
            prompt_frame: m["prompt"]["frame"].as_u64().unwrap_or(0) as usize,
            prompt_points: points,
            frames: Vec::with_capacity(n),
            info,
        })
    }

    /// The blob form: a one-channel clip whose meta carries the manifest.
    pub fn to_blob(&self) -> Blob {
        let bytes: Vec<u8> = self.frames.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        Blob::new(Media::Video, bytes).with_meta(json!({"frames": self.frames.len(), "w": self.width, "h": self.height, "c": 1, "fps": self.fps, "maskseq": self.manifest()}))
    }

    /// Whether `b` is a mask sequence (rather than an ordinary clip).
    pub fn is_blob(b: &Blob) -> bool {
        b.media == Media::Video && b.meta.get("maskseq").is_some()
    }

    /// A blob back into a sequence; a clip without a manifest is refused,
    /// because its polarity would have to be guessed.
    pub fn from_blob(b: &Blob) -> Result<MaskClip, String> {
        let m = b.meta.get("maskseq").ok_or("maskseq: the clip carries no mask-sequence manifest (meta.maskseq); a plain video is not a mask sequence")?;
        let mut clip = MaskClip::from_manifest(m, "the blob")?;
        let n = clip.info.len();
        let plane = clip.plane();
        if b.bytes.len() != n * plane * 4 {
            return Err(format!("maskseq: the blob holds {} bytes, expected {n} frames of {}x{} f32", b.bytes.len(), clip.width, clip.height));
        }
        let values: Vec<f32> = b.bytes.as_chunks::<4>().0.iter().map(|q| f32::from_le_bytes(*q)).collect();
        clip.frames = values.chunks(plane.max(1)).map(<[f32]>::to_vec).collect();
        clip.frames.truncate(n);
        Ok(clip)
    }

    /// Write the directory form into `dir` (created if missing). Returns the
    /// manifest's path.
    pub fn write_dir(&self, dir: impl AsRef<Path>) -> Result<PathBuf, String> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| format!("maskseq: {}: {e}", dir.display()))?;
        for (i, plane) in self.frames.iter().enumerate() {
            let px = plane.iter().flat_map(|v| [((v.clamp(0.0, 1.0) * 255.0) + 0.5) as u8; 3]).collect();
            crate::save_png(frame_path(dir, i), &crate::Rgb8::new(self.width, self.height, px)?)?;
        }
        let path = dir.join(MANIFEST);
        std::fs::write(&path, serde_json::to_vec_pretty(&self.manifest()).map_err(|e| e.to_string())?).map_err(|e| format!("maskseq: {}: {e}", path.display()))?;
        Ok(path)
    }

    /// Read the directory form: the manifest, then every frame it lists, each
    /// checked against the declared size.
    pub fn read_dir(dir: impl AsRef<Path>) -> Result<MaskClip, String> {
        let dir = dir.as_ref();
        let path = dir.join(MANIFEST);
        let raw = std::fs::read(&path).map_err(|e| format!("maskseq: cannot read {} ({e}); a mask sequence needs its manifest", path.display()))?;
        let m: Value = serde_json::from_slice(&raw).map_err(|e| format!("maskseq: {} is not JSON: {e}", path.display()))?;
        let mut clip = MaskClip::from_manifest(&m, &path.display().to_string())?;
        for i in 0..clip.info.len() {
            let p = frame_path(dir, i);
            let img = crate::load(&p)?;
            if (img.w, img.h) != (clip.width, clip.height) {
                return Err(format!("maskseq: {} is {}x{}, the manifest says {}x{}", p.display(), img.w, img.h, clip.width, clip.height));
            }
            clip.frames.push(img.px.as_chunks::<3>().0.iter().map(|c| f32::from(c[0]) / 255.0).collect());
        }
        Ok(clip)
    }
}

/// Whether `dir` holds a mask sequence's manifest.
pub fn is_dir(dir: impl AsRef<Path>) -> bool {
    dir.as_ref().join(MANIFEST).is_file()
}

fn frame_path(dir: &Path, i: usize) -> PathBuf {
    dir.join(format!("mask_{i:06}.png"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("brain-maskseq-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// Logits with two object pixels out of twelve.
    fn logits() -> Vec<f32> {
        let mut v = vec![-4.0f32; 4 * 3];
        v[0] = 2.0;
        v[5] = 0.5;
        v
    }

    fn clip(polarity: Polarity) -> MaskClip {
        let mut c = MaskClip::new(4, 3, 24.0, polarity, true);
        c.object_id = 7;
        c.source = "clip.mp4".into();
        c.prompt_points = vec![(1.0, 2.0, 1.0)];
        c.push_logits(&logits(), 3.5, 0.9).unwrap();
        c.push_logits(&logits(), -1.25, 0.4).unwrap();
        c
    }

    /// Blob and directory forms carry the same sequence, and polarity flips
    /// the pixels while `area_px` keeps counting object pixels.
    #[test]
    fn both_forms_round_trip_and_polarity_flips_only_the_pixels() {
        for pol in [Polarity::ObjectWhite, Polarity::ObjectBlack] {
            let c = clip(pol);
            let b = MaskClip::from_blob(&c.to_blob()).unwrap();
            assert_eq!((b.frames, b.polarity, b.object_id, b.source.as_str()), (c.frames.clone(), pol, 7, "clip.mp4"));
            let d = tmpdir(pol.tag());
            c.write_dir(&d).unwrap();
            assert!(is_dir(&d));
            let r = MaskClip::read_dir(&d).unwrap();
            assert_eq!(r.frames, c.frames);
            assert_eq!(r.info[0].area_px, 2);
            assert_eq!(r.occluded_frames().keys().copied().collect::<Vec<_>>(), vec![1]);
            assert_eq!(pol.objectness(r.frames[0][0]), 1.0, "{}: the object pixel", pol.tag());
            assert_eq!(pol.objectness(r.frames[0][1]), 0.0, "{}: a background pixel", pol.tag());
            let _ = std::fs::remove_dir_all(d);
        }
    }

    /// A sequence whose polarity is missing or unknown is refused, and a
    /// plain clip is not taken for a mask sequence.
    #[test]
    fn a_missing_polarity_or_a_plain_clip_is_refused() {
        let d = tmpdir("nopolarity");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(MANIFEST), format!(r#"{{"format":"{FORMAT}","frames":0,"width":1,"height":1,"fps":24,"per_frame":[]}}"#)).unwrap();
        assert!(MaskClip::read_dir(&d).unwrap_err().contains("polarity"));
        assert!(Polarity::parse("white").unwrap_err().contains("refusing to guess"));
        let plain = capability::blob::video_blob(&[(vec![0.0; 12], 2, 2)]).unwrap();
        assert!(!MaskClip::is_blob(&plain));
        assert!(MaskClip::from_blob(&plain).is_err());
        let _ = std::fs::remove_dir_all(d);
    }
}
