// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The face half of YOLOv8's two-stage detect-then-verify recognizer: SCRFD
//! finds every face in the ORIGINAL full-resolution frame in one pass, each
//! face is attributed to the person box its centre falls inside, and that face
//! is aligned and embedded by ArcFace. `yolov8::identity` owns the gate that
//! turns those embeddings into names; this is the [`FaceVerifier`] that feeds it.
//!
//! The face search runs ONCE per frame and its result is reused for every box,
//! both because that is N times cheaper than a detector pass per box and
//! because it is more accurate: SCRFD must see a face at its native capture
//! scale, and a per-box crop is upscaled onto the detector's canvas (see
//! `yolov8::identity`'s module docs for the measured numbers). Attribution is
//! then geometric, which is what "the largest face in the frame" cannot express
//! once a frame holds several people.
//!
//! No second face-embedding path lives here: detection is
//! `ArcFaceSession::detect_faces` and alignment + embedding is
//! `ArcFaceSession::embed_face_chw`, the two halves `embed_raw_chw` composes and
//! `arcface embed` serves. The models come from the model store through
//! [`capability::Components`]; this crate, not the detector, links them, so a
//! detector-only consumer carries neither.
//!
//! Swedish Embedded AB implements camera pipelines that tell people apart on
//! the client's own hardware. If your team needs expertise in face recognition
//! behind a detector then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use arcface::caps::ArcFaceSession;
use capability::hot::Hot;
use capability::{Assembly, Components};
use scrfd::Face;
use yolov8::identity::{verify_detections, BoxEmbedder, FaceVerifier, Labeled, VerifyRequest};

/// The model the identity stage needs from a host's [`Components`], and the
/// resolver spec of it.
pub fn component_specs() -> Vec<(&'static str, &'static dyn brain_modelstore::resolve::ArchSpec)> {
    vec![("arcface", &arcface::spec::ArcFaceSpec)]
}

/// The directory holding `glintr100.onnx` and `scrfd_10g_bnkps.onnx`, from the
/// resolved `arcface` assembly: the embedder's `weights` role sits next to the
/// detector it is released with.
fn arcface_dir(assembly: &Assembly) -> Result<String, String> {
    let weights = assembly.role_path("weights")?;
    let dir = Path::new(&weights).parent().ok_or_else(|| format!("{weights} has no directory"))?;
    Ok(dir.to_string_lossy().into_owned())
}

/// The faces whose CENTRE lies inside `bbox` (`[x1,y1,x2,y2]`).
fn faces_inside(faces: &[Face], bbox: [f32; 4]) -> Vec<Face> {
    faces
        .iter()
        .filter(|f| {
            let (cx, cy) = ((f.bbox[0] + f.bbox[2]) * 0.5, (f.bbox[1] + f.bbox[3]) * 0.5);
            cx >= bbox[0] && cx <= bbox[2] && cy >= bbox[1] && cy <= bbox[3]
        })
        .copied()
        .collect()
}

/// An [`ArcFaceSession`] behind the detector's embedder seam, for ONE frame:
/// the frame's faces are found on the first box and reused for the rest.
struct FrameFaces<'a> {
    session: &'a ArcFaceSession,
    faces: RefCell<Option<Vec<Face>>>,
}

impl BoxEmbedder for FrameFaces<'_> {
    fn embed_in_box(&self, chw: &[f32], w: u32, h: u32, bbox: [f32; 4]) -> Result<Vec<f32>, String> {
        let mut cache = self.faces.borrow_mut();
        if cache.is_none() {
            *cache = Some(self.session.detect_faces(chw, w, h)?);
        }
        let inside = faces_inside(cache.as_deref().unwrap_or_default(), bbox);
        if inside.is_empty() {
            return Err("no face inside this box".into());
        }
        // Several faces inside one person box (a crowd behind the subject) is
        // resolved the reference recipe's way: the largest is the subject.
        self.session.embed_face_chw(chw, w, h, &arcface::caps::primary_face(&inside))
    }
}

/// SCRFD + ArcFace resolved from `components`, resident across calls (keyed by
/// the directory they were loaded from).
pub struct StoreFaces {
    components: Arc<dyn Components>,
    session: Hot<String, ArcFaceSession>,
}

impl StoreFaces {
    pub fn new(components: Arc<dyn Components>) -> StoreFaces {
        StoreFaces { components, session: Hot::new() }
    }
}

impl FaceVerifier for StoreFaces {
    /// The ArcFace session rides on the detector's own device (`Gpu::new_like`):
    /// one process, one device, two kernel lists, because each crate resolves
    /// its pipeline indices against the list it registered.
    fn verify(&self, r: &VerifyRequest) -> Result<Vec<Labeled>, String> {
        let dir = arcface_dir(&self.components.resolve("arcface", r.arcface_roles)?)?;
        self.session.try_get(dir.clone(), || ArcFaceSession::load(&dir, r.gpu.new_like(&arcface::caps::SERVING_PIPELINES)), |session| {
            let frame = FrameFaces { session, faces: RefCell::new(None) };
            Ok(verify_detections(r.gate, &frame, r.dets, r.hwc, r.w, r.h))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(bbox: [f32; 4]) -> Face {
        Face { bbox, score: 0.9, kps: [[0.0; 2]; 5] }
    }

    /// A face belongs to the person box its CENTRE is in, not to any box it
    /// merely overlaps: two people side by side keep their own faces.
    #[test]
    fn a_face_is_attributed_to_the_box_holding_its_centre() {
        let faces = [face([10.0, 10.0, 30.0, 30.0]), face([90.0, 10.0, 110.0, 30.0])];
        let left = faces_inside(&faces, [0.0, 0.0, 60.0, 100.0]);
        let right = faces_inside(&faces, [60.0, 0.0, 120.0, 100.0]);
        assert_eq!((left.len(), right.len()), (1, 1));
        assert_eq!(left[0].bbox[0], 10.0);
        assert!(faces_inside(&faces, [25.0, 0.0, 60.0, 100.0]).is_empty(), "the centre decides, not overlap");
    }

    #[test]
    fn the_detector_directory_is_the_embedders_own() {
        let a = Assembly { id: "x".into(), arch: "arcface".into(), variant: None, roles: [("weights".to_string(), "/models/antelope/glintr100.onnx".into())].into(), provenance: Vec::new() };
        assert_eq!(arcface_dir(&a).unwrap(), "/models/antelope");
        assert!(arcface_dir(&Assembly { roles: Default::default(), ..a }).is_err());
    }
}
