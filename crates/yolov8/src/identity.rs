// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements two-stage detect-then-verify recognition
// pipelines - a category detector supplying regions, a specialist embedder
// deciding WHO is in them - for its clients. If your team needs expertise in
// on-device object detection or face-identity verification, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Detect-then-verify: turning a CATEGORY detection into an IDENTITY one.
//!
//! # Why this exists as a second stage rather than a class in the head
//!
//! A COCO-preserving YOLOv8 class head cannot express personal identity, and
//! that is a property of the features, not of the training recipe. The shared
//! backbone was trained to separate *categories* (car / dog / person); an added
//! class 80 sitting on those frozen features is a linear probe on a
//! representation that discards exactly the within-category variation identity
//! is made of. Unfreezing them so the head CAN learn it destroys the
//! preservation the head was frozen for. Both halves of that trade were
//! measured on this repo's own identity pipeline and neither is acceptable.
//!
//! So identity is decided where the signal actually lives: the detector finds
//! *a person*, and a face embedder decides *which* person, on a crop of the
//! ORIGINAL image. This is the same ArcFace-cosine test the training-data
//! generator already applies when it gates which generated images are allowed
//! into the dataset - extended from data-generation time to inference time,
//! rather than reimplemented.
//!
//! # The seam
//!
//! [`BoxEmbedder`] is deliberately the only thing this module knows about the
//! embedder: the WHOLE image plus one box, one embedding out. `crates/yolov8`
//! therefore gains no dependency on `crates/arcface`, `crates/scrfd` or any
//! other identity model, and the geometry + threshold logic here is testable
//! with no weights at all (see this module's tests).
//!
//! # Why the embedder gets the whole image and not a crop
//!
//! The obvious shape - cut the box out and hand over the pixels - is measurably
//! worse, and the reason is worth stating because it is not obvious. A face
//! detector resizes whatever it is given onto its own square canvas, so a small
//! crop is UPSCALED; and a marginal face survives that badly. Measured on one
//! held-out frame, on a face that is 17x24 source pixels, with SCRFD's 640²
//! canvas: found at score 0.595 when the detector sees the full 512² frame,
//! and found by NOTHING at 400², 300² or 235² crops around the same face. The
//! smaller the crop, the further the face is resampled UP, and interpolated
//! detail is not detail. So the face search runs ONCE over the frame at its
//! native scale, and the box is used to decide which of the faces found there
//! belongs to which detection - which is also N-times cheaper than a detector
//! pass per box.
//!
//! # Coordinates
//!
//! [`Yolo::detect`](crate::Yolo::detect) already returns boxes in
//! **original-image pixel coordinates** (it letterboxes internally and inverts
//! the transform on the way out), so everything here is in the caller's own
//! full-resolution frame. Pre-letterboxing the input before `detect` - which
//! the detector would only redo - additionally throws away the resolution the
//! face embedder needs, and is the one loss in this pipeline that is not
//! recoverable downstream.

use crate::nms::Detection;

/// Embeds the subject inside one box of an image into a comparable identity
/// vector.
///
/// The image arrives as **CHW RGB** floats at its own resolution, in the units
/// the caller's image is in (this repo's wire convention is `[0,1]`); `bbox` is
/// `[x1,y1,x2,y2]` in those same pixels. `Err` is the ordinary, expected
/// outcome for a box that holds no usable face - a person seen from behind, or
/// too small to register - and callers treat it as "not this identity", never
/// as a failure of the whole detection.
pub trait BoxEmbedder {
    fn embed_in_box(&self, chw: &[f32], w: u32, h: u32, bbox: [f32; 4]) -> Result<Vec<f32>, String>;
}

/// The identity test applied to one detected class.
///
/// Exactly one class is verified ([`class`](IdentityGate::class), `0` = COCO
/// `person`): the gate refines what a detection of that class is CALLED, and
/// never touches a detection of any other class. Detections of the gated class
/// that clear [`threshold`](IdentityGate::threshold) are labelled
/// [`name`](IdentityGate::name); the rest keep
/// [`class_name`](IdentityGate::class_name), so a stranger stays a correctly
/// detected generic person rather than becoming a non-detection.
pub struct IdentityGate {
    /// The identity's reference embedding, from the same embedder. Need not be
    /// L2-normalised - the comparison is a cosine.
    pub reference: Vec<f32>,
    /// Cosine floor for "this IS the reference identity".
    pub threshold: f32,
    /// Label for a detection that clears the floor (e.g. `"einstein"`).
    pub name: String,
    /// Label for a gated-class detection that does not (e.g. `"person"`).
    pub class_name: String,
    /// The detected class the gate applies to.
    pub class: u32,
    /// Fraction of the box's own width/height to widen the search box by on
    /// each side. A detector box is tight around the subject, and a face whose
    /// hairline sits exactly on the box edge is easier to attribute to it with
    /// a little slack than without.
    pub margin: f32,
}

impl IdentityGate {
    /// A gate over COCO `person` (class 0) with this repo's default labels.
    pub fn new(reference: Vec<f32>, threshold: f32, name: impl Into<String>) -> IdentityGate {
        IdentityGate {
            reference,
            threshold,
            name: name.into(),
            class_name: "person".into(),
            class: 0,
            margin: 0.0,
        }
    }
}

/// One detection after the gate has run.
pub struct Labeled {
    /// The detection, unchanged - `[x1,y1,x2,y2,conf,class]` in original-image
    /// coords. The gate never renumbers a class: a verified person is still a
    /// person detection, now carrying a name.
    pub det: Detection,
    /// The human label to render, or `None` for a class the gate does not
    /// cover - in which case a renderer falls back to the numeric class id
    /// exactly as it did before identity verification existed.
    pub label: Option<String>,
    /// Cosine against the reference, or `None` when the crop held no face the
    /// embedder could use.
    pub similarity: Option<f32>,
    /// Whether this detection IS the reference identity.
    pub verified: bool,
}

/// Widen `bbox` by `margin` of its own size on each side and clamp it to the
/// `w0 x h0` frame.
///
/// `None` when the result has no area inside the frame - a degenerate NMS
/// output or a box entirely off-image, which is a skip rather than an error.
pub fn expand_box(bbox: [f32; 4], margin: f32, w0: u32, h0: u32) -> Option<[f32; 4]> {
    let (mx, my) = ((bbox[2] - bbox[0]) * margin, (bbox[3] - bbox[1]) * margin);
    let b = [
        (bbox[0] - mx).max(0.0),
        (bbox[1] - my).max(0.0),
        (bbox[2] + mx).min(w0 as f32),
        (bbox[3] + my).min(h0 as f32),
    ];
    (b[2] > b[0] && b[3] > b[1]).then_some(b)
}

/// Run `gate` over `dets`, asking `embedder` who is inside each gated-class
/// detection of `src` (interleaved-RGB **HWC**, `w0 x h0`).
///
/// Every input detection appears in the output in the same order. Detections of
/// other classes pass through untouched and unlabelled. The HWC -> CHW
/// transpose the embedder wants happens ONCE for the frame, not once per box.
pub fn verify_detections(
    gate: &IdentityGate,
    embedder: &dyn BoxEmbedder,
    dets: &[Detection],
    src: &[f32],
    w0: u32,
    h0: u32,
) -> Vec<Labeled> {
    let gated = dets.iter().any(|d| d[5] as u32 == gate.class);
    let chw = if gated { imaging::pixels::hwc_to_chw(src, 3, h0 as usize, w0 as usize) } else { Vec::new() };
    dets.iter()
        .map(|&det| {
            if det[5] as u32 != gate.class {
                return Labeled { det, label: None, similarity: None, verified: false };
            }
            let sim = expand_box([det[0], det[1], det[2], det[3]], gate.margin, w0, h0)
                .and_then(|b| embedder.embed_in_box(&chw, w0, h0, b).ok())
                .filter(|e| e.len() == gate.reference.len())
                .map(|e| model::hostmath::cosine(&gate.reference, &e));
            let verified = sim.is_some_and(|s| s >= gate.threshold);
            let label = Some(if verified { gate.name.clone() } else { gate.class_name.clone() });
            Labeled { det, label, similarity: sim, verified }
        })
        .collect()
}

/// The gate's output as the box JSON `brain imageops draw_boxes` reads:
/// `{"bbox":[x1,y1,x2,y2],"conf":c,"class":n}` plus the optional `"label"` that
/// makes the rendered annotation read `einstein:0.87` instead of `0:0.87`, and
/// `"similarity"` when a face was found. Keeping the schema additive is what
/// lets every pre-existing `draw_boxes` caller keep working unchanged.
pub fn to_json(labeled: &[Labeled]) -> serde_json::Value {
    serde_json::Value::Array(
        labeled
            .iter()
            .map(|l| {
                let mut o = serde_json::json!({
                    "bbox": [l.det[0], l.det[1], l.det[2], l.det[3]],
                    "conf": l.det[4],
                    "class": l.det[5] as u32,
                });
                if let Some(name) = &l.label {
                    o["label"] = serde_json::json!(name);
                }
                if let Some(s) = l.similarity {
                    o["similarity"] = serde_json::json!(s);
                }
                o
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A weights-free stand-in for the SCRFD + ArcFace path: it reports the
    /// box it was asked about and answers with a vector chosen by the mean
    /// pixel value INSIDE that box, so a test can put "the target" and "a
    /// stranger" in one frame as two solid patches and check the gate tells
    /// them apart - and that it asked about the right region.
    struct PatchEmbedder {
        seen: std::cell::RefCell<Vec<[f32; 4]>>,
    }

    impl PatchEmbedder {
        fn new() -> PatchEmbedder {
            PatchEmbedder { seen: std::cell::RefCell::new(Vec::new()) }
        }
    }

    impl BoxEmbedder for PatchEmbedder {
        fn embed_in_box(&self, chw: &[f32], w: u32, h: u32, bbox: [f32; 4]) -> Result<Vec<f32>, String> {
            self.seen.borrow_mut().push(bbox);
            assert_eq!(chw.len(), (w * h * 3) as usize, "the embedder is handed the WHOLE frame");
            let (x0, y0) = (bbox[0] as u32, bbox[1] as u32);
            let (x1, y1) = (bbox[2] as u32, bbox[3] as u32);
            let (mut sum, mut n) = (0.0f32, 0usize);
            for c in 0..3u32 {
                for y in y0..y1 {
                    for x in x0..x1 {
                        sum += chw[((c * h + y) * w + x) as usize];
                        n += 1;
                    }
                }
            }
            let mean = if n == 0 { 0.0 } else { sum / n as f32 };
            // 0.8 = the target, 0.4 = a different person, 0.0 = no face at all.
            if mean < 0.1 {
                return Err("no face detected in the box".into());
            }
            Ok(if mean > 0.6 { vec![1.0, 0.0, 0.0] } else { vec![0.0, 1.0, 0.0] })
        }
    }

    /// A `w x h` RGB HWC frame with `patches` of `(x0,y0,x1,y1,value)` painted in.
    fn frame(w: u32, h: u32, patches: &[(u32, u32, u32, u32, f32)]) -> Vec<f32> {
        let mut px = vec![0.0f32; (w * h * 3) as usize];
        for &(x0, y0, x1, y1, v) in patches {
            for y in y0..y1 {
                for x in x0..x1 {
                    for c in 0..3 {
                        px[((y * w + x) * 3 + c) as usize] = v;
                    }
                }
            }
        }
        px
    }

    fn gate() -> IdentityGate {
        IdentityGate::new(vec![1.0, 0.0, 0.0], 0.5, "einstein")
    }

    /// The whole point of the design: two person-class detections in one frame,
    /// one of them the reference identity. The identity one is named, the other
    /// stays a correctly-detected generic person - NOT a dropped detection.
    #[test]
    fn the_reference_identity_is_named_and_a_stranger_stays_a_person() {
        let src = frame(64, 32, &[(0, 0, 32, 32, 0.8), (32, 0, 64, 32, 0.4)]);
        let dets: Vec<Detection> =
            vec![[0.0, 0.0, 32.0, 32.0, 0.91, 0.0], [32.0, 0.0, 64.0, 32.0, 0.77, 0.0]];
        let out = verify_detections(&gate(), &PatchEmbedder::new(), &dets, &src, 64, 32);

        assert_eq!(out.len(), 2, "every detection survives the gate");
        assert!(out[0].verified);
        assert_eq!(out[0].label.as_deref(), Some("einstein"));
        assert!(out[0].similarity.unwrap() > 0.99, "same identity is cosine ~1");
        assert!(!out[1].verified, "a different person must not clear the floor");
        assert_eq!(out[1].label.as_deref(), Some("person"), "still a person detection");
        assert!(out[1].similarity.unwrap() < 0.5);
    }

    /// A person box with no findable face is the expected case the operators
    /// pipeline must survive (a subject seen from behind): it degrades to a
    /// generic person detection with no similarity, never to an error.
    #[test]
    fn a_box_with_no_face_degrades_to_a_generic_detection() {
        let src = frame(32, 32, &[]);
        let dets: Vec<Detection> = vec![[0.0, 0.0, 32.0, 32.0, 0.6, 0.0]];
        let out = verify_detections(&gate(), &PatchEmbedder::new(), &dets, &src, 32, 32);
        assert_eq!(out.len(), 1);
        assert!(!out[0].verified);
        assert_eq!(out[0].similarity, None, "no face means no similarity, not a similarity of 0");
        assert_eq!(out[0].label.as_deref(), Some("person"));
    }

    /// The gate covers ONE class. A dog is not a candidate identity and must
    /// not be embedded or labelled - it renders exactly as before.
    #[test]
    fn other_classes_are_never_embedded_or_labelled() {
        let src = frame(64, 32, &[(0, 0, 32, 32, 0.8), (32, 0, 64, 32, 0.8)]);
        let dets: Vec<Detection> =
            vec![[0.0, 0.0, 32.0, 32.0, 0.9, 16.0], [32.0, 0.0, 64.0, 32.0, 0.9, 0.0]];
        let emb = PatchEmbedder::new();
        let out = verify_detections(&gate(), &emb, &dets, &src, 64, 32);
        assert_eq!(out[0].label, None, "a non-gated class keeps the numeric fallback");
        assert_eq!(out[0].similarity, None);
        assert_eq!(out[1].label.as_deref(), Some("einstein"));
        assert_eq!(emb.seen.borrow().len(), 1, "only the gated class is embedded");
    }

    /// The embedder is asked about the ORIGINAL frame at original resolution,
    /// for the box expanded by the requested margin and clamped to the frame.
    #[test]
    fn the_search_box_is_expanded_by_margin_and_clamped() {
        let src = frame(100, 80, &[(0, 0, 100, 80, 0.8)]);
        let mut g = gate();
        g.margin = 0.25;
        let emb = PatchEmbedder::new();
        // Interior box: 40x40 grown by 25% on each side -> [20,10,80,70].
        // Edge box: 20x20 at the origin grown by 5px -> clamped to [0,0,25,25].
        let dets: Vec<Detection> =
            vec![[30.0, 20.0, 70.0, 60.0, 0.9, 0.0], [0.0, 0.0, 20.0, 20.0, 0.9, 0.0]];
        verify_detections(&g, &emb, &dets, &src, 100, 80);
        assert_eq!(&*emb.seen.borrow(), &[[20.0, 10.0, 80.0, 70.0], [0.0, 0.0, 25.0, 25.0]]);

        // ... and a box entirely outside the frame is skipped, not embedded.
        let emb2 = PatchEmbedder::new();
        let off: Vec<Detection> = vec![[120.0, 90.0, 140.0, 110.0, 0.9, 0.0]];
        let out = verify_detections(&g, &emb2, &off, &src, 100, 80);
        assert!(emb2.seen.borrow().is_empty());
        assert_eq!(out[0].label.as_deref(), Some("person"), "still reported, just unverifiable");
    }

    /// The rendered-box schema stays additive: `label` appears only when the
    /// gate assigned one, so a pre-identity `draw_boxes` caller is unaffected.
    #[test]
    fn the_box_json_carries_the_label_only_when_there_is_one() {
        let src = frame(64, 32, &[(0, 0, 32, 32, 0.8)]);
        let dets: Vec<Detection> =
            vec![[0.0, 0.0, 32.0, 32.0, 0.5, 0.0], [32.0, 0.0, 64.0, 32.0, 0.5, 16.0]];
        let j = to_json(&verify_detections(&gate(), &PatchEmbedder::new(), &dets, &src, 64, 32));
        assert_eq!(j[0]["label"], "einstein");
        assert_eq!(j[0]["class"], 0);
        assert_eq!(j[0]["bbox"], serde_json::json!([0.0, 0.0, 32.0, 32.0]));
        assert!(j[1].get("label").is_none(), "an unlabelled box carries no 'label' key at all");
        assert!(j[1].get("similarity").is_none());
    }
}
