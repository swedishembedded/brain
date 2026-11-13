// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The edited stills of `replace`: the character painted over the actor on
//! frame 0 and on the guide frames by FLUX.2's served `edit`, the new figure
//! found by SAM 2, and the edit to keep chosen by overlap and by how close
//! its face is to the sheet's (ArcFace).

use std::path::Path;
use std::sync::Arc;

use capability::{Components, Invocation, Provider};
use imaging::{Ctx, Filter, Rgb8};
use serde_json::json;

use super::geometry::*;
use super::resume::{digest, digest_mask, Resume};
use super::{adapters_record, track, Request};

/// The still recipe: how a still is shaped outside the inputs its key records
/// (the composite's blend, the figure's segmentation). Bump it whenever that
/// recipe changes, so a resumed run repaints instead of reusing a still the
/// old recipe made.
const STILL_RECIPE: u32 = 2;

/// Frame 0's character as this clip shows them: the head, hair and clothes
/// crop every guide still is painted against, and that crop's digest - what a
/// resumed run keys the guides by.
pub(super) struct Appearance {
    pub image: Rgb8,
    pub digest: u64,
}

/// A source frame with the character painted in over the actor.
pub(super) struct Still {
    pub frame: usize,
    pub image: Rgb8,
    /// Where the new figure stands on this still.
    pub figure: Vec<bool>,
    /// How well the figure covers the actor ([`iou`]).
    pub iou: f32,
}

/// ArcFace and the character sheet's face embedding: how close a painted
/// still's face is to the character's.
pub(super) struct Identity {
    session: arcface::caps::ArcFaceSession,
    sheet: Vec<f32>,
}

impl Identity {
    /// The identity model from the store (or the `arcface.weights`
    /// component) and the sheet's face. `None` - said, not silent - when the
    /// store holds no ArcFace or the sheet shows no face a detector finds (a
    /// drawn or masked character): the stills are then kept by overlap alone.
    pub(super) fn load(req: &Request, components: &dyn Components, device: &gpu_core::Gpu) -> Result<Option<Identity>, String> {
        if !req.identity {
            return Ok(None);
        }
        let named = req.component("arcface");
        let assembly = match components.resolve("arcface", &named) {
            Ok(a) => a,
            // A component named outright that does not resolve is the
            // caller's error; one the store simply lacks is a choice the run
            // can make without.
            Err(e) if named.is_empty() => {
                tracing::warn!("ltxv replace: no identity model ({e}) - stills are kept by overlap alone, not by how close their face is to the sheet's (brain pull https://huggingface.co/DIAMONIK7777/antelopev2/blob/main/glintr100.onnx and .../scrfd_10g_bnkps.onnx to compare faces)");
                return Ok(None);
            }
            Err(e) => return Err(format!("the identity model: {e}")),
        };
        let weights = assembly.role_path("weights")?;
        let dir = Path::new(&weights).parent().ok_or_else(|| format!("{weights} has no directory"))?.to_string_lossy().into_owned();
        let session = arcface::caps::ArcFaceSession::load(&dir, device.new_like(&arcface::caps::SERVING_PIPELINES))?;
        let sheet = &req.character;
        let chw = imaging::pixels::hwc_to_chw(&sheet.to_hwc_unit(), 3, sheet.h as usize, sheet.w as usize);
        match session.embed_raw_chw(&chw, sheet.w, sheet.h, true, true) {
            Ok((raw, _)) => Ok(Some(Identity { session, sheet: model::hostmath::l2_normalize(&raw) })),
            Err(_) => {
                tracing::warn!("ltxv replace: no face found on the character sheet - stills are kept by overlap alone");
                Ok(None)
            }
        }
    }

    /// Cosine similarity of the largest face on `still` to the sheet's;
    /// `None` when no face is found there.
    fn similarity(&self, still: &Still) -> Result<Option<f32>, String> {
        let chw = imaging::pixels::hwc_to_chw(&still.image.to_hwc_unit(), 3, still.image.h as usize, still.image.w as usize);
        let faces = self.session.detect_faces(&chw, still.image.w, still.image.h)?;
        // The face on the figure, not a passer-by's: the largest one inside it.
        let w = still.image.w as usize;
        let on_figure: Vec<_> = faces
            .into_iter()
            .filter(|f| {
                let (cx, cy) = (((f.bbox[0] + f.bbox[2]) / 2.0) as usize, ((f.bbox[1] + f.bbox[3]) / 2.0) as usize);
                still.figure.get(cy * w + cx).copied().unwrap_or(false)
            })
            .collect();
        if on_figure.is_empty() {
            return Ok(None);
        }
        let face = arcface::caps::primary_face(&on_figure);
        let e = model::hostmath::l2_normalize(&self.session.embed_face_chw(&chw, still.image.w, still.image.h, &face)?);
        Ok(Some(e.iter().zip(&self.sheet).map(|(a, b)| a * b).sum()))
    }
}

/// What every still edit shares.
pub(super) struct Editor<'a> {
    pub req: &'a Request,
    /// FLUX.2's served actions, which paint the stills.
    pub flux2: Arc<dyn Provider>,
    pub resume: &'a Resume,
    pub tracker: &'a sam2::Sam2,
    /// The face the stills are matched against, when there is one to match.
    pub identity: Option<Identity>,
    pub ctx: &'a Ctx<'a>,
    pub w: usize,
    pub h: usize,
}

impl Editor<'_> {
    /// Up to `--edit-attempts` edits of frame `k`, each with its own seed,
    /// kept by [`pick_still`]: with an identity model, every attempt is made
    /// and the covering edit whose face is closest to the sheet's wins;
    /// without one, the first covering edit. The winner is `still_<k>.png`.
    ///
    /// `appearance` is frame 0's accepted still, cut to the character's head
    /// and torso ([`appearance_box`]), and given to every later edit as an
    /// extra reference; it carries who the character is as this clip already
    /// shows them, so a guide cannot come back in different clothes from the
    /// frame the video starts on.
    pub(super) fn best_still(&self, k: usize, src: &[u8], actor: &[bool], appearance: Option<&Appearance>) -> Result<Still, String> {
        let (step, key) = (format!("still_{k}"), self.still_key(k, src, actor, appearance.map(|a| a.digest)));
        if self.resume.holds(&step, &key) {
            if let Ok(still) = self.load_still(k, actor) {
                tracing::info!("ltxv replace: frame {k} reused from {}, figure/actor overlap {:.2}", self.req.work.display(), still.iou);
                return Ok(still);
            }
        }
        let t = &self.req.tuning;
        let mut stills: Vec<(Still, Option<f32>)> = Vec::new();
        for attempt in 0..t.edit_attempts {
            let seed = self.req.seed.wrapping_add(k as u64 * t.edit_attempts + attempt);
            let still = self.edit(k, src, actor, seed, appearance)?;
            let face = self.identity.as_ref().map(|id| id.similarity(&still)).transpose()?.flatten();
            let face_note = match (&self.identity, face) {
                (None, _) => String::new(),
                (Some(_), Some(f)) => format!(", face/sheet similarity {f:.2}"),
                (Some(_), None) => ", no face found".to_string(),
            };
            tracing::info!("ltxv replace: frame {k} edit {} (seed {seed}): figure/actor overlap {:.2}{face_note}", attempt + 1, still.iou);
            let covers = still.iou >= t.min_overlap;
            stills.push((still, face));
            // Without a face to compare, the first edit that covers the actor
            // is as good as any later one.
            if covers && self.identity.is_none() {
                break;
            }
        }
        let i = pick_still(&stills.iter().map(|(s, f)| (s.iou, *f)).collect::<Vec<_>>(), t.min_overlap);
        let best = stills.swap_remove(i).0;
        imaging::save_png(self.req.work.join(format!("still_{k}.png")), &best.image)?;
        let figure: Vec<u8> = best.figure.iter().flat_map(|&f| [if f { 255 } else { 0 }; 3]).collect();
        imaging::save_png(self.req.work.join(format!("figure_{k}.png")), &Rgb8::new(self.w as u32, self.h as u32, figure)?)?;
        self.resume.record(&step, &key)?;
        Ok(best)
    }

    /// Everything the winning still of frame `k` depends on. The source
    /// frame and the actor's mask are named by their pixels, so a changed
    /// clip, click or track gives another key without the key knowing how
    /// the track was made. A guide also names the appearance reference it
    /// was painted against, so repainting frame 0 repaints the guides.
    fn still_key(&self, k: usize, src: &[u8], actor: &[bool], appearance: Option<u64>) -> String {
        let r = self.req;
        serde_json::json!({
            "frame": k, "recipe": STILL_RECIPE, "source": digest(src.iter().copied()), "actor": digest_mask(actor), "character": digest(r.character.px.iter().copied()),
            "appearance": appearance,
            "edit_prompt": r.edit_prompt, "seed": r.seed, "editor": r.component("flux2"), "edit_adapters": adapters_record(&r.edit_adapters),
            "identity": r.identity, "identity_model": r.component("arcface"), "tuning": r.tuning.edit_record(),
        })
        .to_string()
    }

    /// The still [`Editor::best_still`] saved: its image and its figure.
    fn load_still(&self, k: usize, actor: &[bool]) -> Result<Still, String> {
        let image = imaging::load(self.req.work.join(format!("still_{k}.png")))?;
        let figure: Vec<bool> = imaging::load(self.req.work.join(format!("figure_{k}.png")))?.px.as_chunks::<3>().0.iter().map(|p| p[0] > 127).collect();
        if (image.w as usize, image.h as usize, figure.len()) != (self.w, self.h, self.w * self.h) {
            return Err(format!("saved still {k} is not {}x{}", self.w, self.h));
        }
        Ok(Still { frame: k, iou: iou(&figure, actor), image, figure })
    }

    fn edit(&self, k: usize, src: &[u8], actor: &[bool], seed: u64, appearance: Option<&Appearance>) -> Result<Still, String> {
        let (w, h, ctx, work) = (self.w, self.h, self.ctx, &self.req.work);
        let bbox = bounding_box(actor, w, h).ok_or_else(|| format!("the actor is not visible on frame {k}: nothing to paint the character over (click them on a frame they are in: prompt_frame)"))?;
        let canvas = canvas_around(bbox, body_below_frame(actor, bbox, w, h, self.req.tuning.body_ratio), w, h);
        let (cw, ch) = (canvas.w, canvas.h);
        let visible = ch.min(h - canvas.y);
        // What FLUX.2 repaints: the actor, grown, and every row below the
        // frame. The rest of the canvas is kept exactly, so the figure has to
        // take the actor's place, size and pose; the rows below the frame
        // give it room to draw a whole body at the actor's scale where the
        // frame cuts the actor off, which it otherwise does by shrinking it.
        let repaint: Vec<bool> = (0..ch * cw).map(|i| i / cw >= visible || actor[(canvas.y + i / cw) * w + canvas.x + i % cw]).collect();
        let repaint = grow(ctx, &repaint, cw, ch, (bbox.w as f32 * self.req.tuning.repaint_grow).round() as u32);
        let (ew, eh) = edit_size(cw, ch, self.req.tuning.edit_long_side);
        let canvas_img = Rgb8::new(ew, eh, resample(ctx, &cut(src, w, h, canvas), cw, ch, ew as usize, eh as usize, Filter::Bilinear))?;
        imaging::save_png(work.join(format!("canvas_{k}.png")), &canvas_img)?;
        let repaint_px: Vec<u8> = repaint.iter().flat_map(|&r| [if r { 255 } else { 0 }; 3]).collect();
        let repaint_img = Rgb8::new(ew, eh, resample(ctx, &repaint_px, cw, ch, ew as usize, eh as usize, Filter::Nearest))?;
        imaging::save_png(work.join(format!("repaint_{k}.png")), &repaint_img)?;
        let edit_path = work.join(format!("edit_{k}_{seed}.png"));
        // FLUX.2's served `edit`: the canvas is the image it starts from
        // (anchored by the mask), the sheet and frame 0's appearance are
        // further references, each bounded to the edit size.
        let blob = |img: &Rgb8| capability::blob::image_blob(&img.to_hwc_unit(), img.w, img.h, 3);
        let mut inv = Invocation::new()
            .set("prompt", json!(self.req.edit_prompt))
            .set("seed", json!(seed))
            .set("width", json!(ew))
            .set("height", json!(eh))
            .set("ref_size", json!(self.req.tuning.edit_long_side))
            .blob("image", blob(&canvas_img))
            .blob("mask", capability::Blob { media: capability::Media::Mask, ..blob(&repaint_img) })
            .blob("image0", blob(&self.req.character));
        if let Some(a) = appearance {
            inv = inv.blob("image1", blob(&a.image));
        }
        if !self.req.edit_adapters.is_empty() {
            let list: Vec<String> = self.req.edit_adapters.iter().map(|a| format!("{}@{}", a.path, a.scale)).collect();
            inv = inv.set("adapter", json!(list.join(",")));
        }
        if let Some(v) = self.req.component("flux2").get("variant") {
            inv = inv.set("variant", json!(v));
        }
        let act = self.flux2.action("edit").ok_or("FLUX.2 has no edit action")?;
        let out = act.run(&act.spec().validate(inv)?, &mut |_| {}).map_err(|e| format!("editing frame {k}: {e}"))?;
        let (hwc, ow, oh) = capability::blob::decode_image(&Invocation::new().blob("image", out.blobs.get("image").ok_or("the FLUX.2 edit made no image")?.clone()), "image")?;
        let edited_img = Rgb8::new(ow, oh, hwc.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect())?;
        imaging::save_png(&edit_path, &edited_img)?;
        let edited = fit(ctx, &edited_img, cw, ch);
        let mut painted = src.to_vec();
        paste(&mut painted, w, Box2 { h: visible, ..canvas }, &edited[..visible * cw * 3]);

        // The new figure: SAM 2 asked at spots spread over the actor's
        // silhouette - the character was asked to stand there - in ONE
        // multi-point call. Ten single-point calls unioned let the clicks that
        // land on the bed each come back as a large mask under the size guard,
        // and their union then covered the frame (a 2x-the-actor figure whose
        // low IoU drops an otherwise good still). The same clicks asked
        // together segment the character coherently. A mask much larger than
        // the actor is still dropped: that is a click that picked the set
        // instead of the character.
        let painted_unit: Vec<f32> = painted.iter().map(|&v| v as f32 / 255.0).collect();
        let actor_px = count(actor);
        let points: Vec<(f32, f32, f32)> = spread_points(actor, w, h).into_iter().map(|(x, y)| (x, y, 1.0)).collect();
        let figure: Vec<bool> = if points.is_empty() {
            vec![false; w * h]
        } else {
            let pick = track(self.tracker, std::slice::from_ref(&painted_unit), w as u32, h as u32, 0, &points)?.remove(0);
            if count(&pick) <= 2 * actor_px {
                pick
            } else {
                vec![false; w * h]
            }
        };
        let joined: Vec<bool> = actor.iter().zip(&figure).map(|(&a, &f)| a || f).collect();
        let alpha = feather(ctx, &grow(ctx, &joined, w, h, self.req.tuning.paste_grow_px), w, h, self.req.tuning.feather_px);
        let image = Rgb8::new(w as u32, h as u32, blend(&painted, src, &alpha))?;
        Ok(Still { frame: k, image, iou: iou(&figure, actor), figure })
    }

    /// The appearance reference every guide still after frame 0 is painted
    /// against: frame 0's accepted still, cut to the character's head, hair
    /// and clothes ([`appearance_box`]). Saved in the work directory because
    /// a person inspecting a run wants to see it; keyed by the crop's own pixels,
    /// not the file it lands in, so re-saving the same crop does not
    /// invalidate the guides a resumed run would reuse, while a changed crop
    /// or a changed frame 0 does.
    pub(super) fn appearance_reference(&self, first: &Still, actor0: &[bool]) -> Result<Appearance, String> {
        let b = appearance_box(actor0, self.w, self.h)
            .ok_or("the actor is not visible on frame 0: there is no appearance to carry into the guides")?;
        let crop = cut(&first.image.px, self.w, self.h, b);
        let image = Rgb8::new(b.w as u32, b.h as u32, crop.clone())?;
        imaging::save_png(self.req.work.join("appearance.png"), &image)?;
        Ok(Appearance { image, digest: digest(crop.iter().copied()) })
    }
}

/// Which of a frame's edits to keep, given each one's `(overlap with the
/// actor, face similarity to the sheet)`: among those that cover the actor
/// (`min_overlap`), the closest face, a measured face before none, then the
/// best overlap; when none covers the actor, the best overlap.
pub(super) fn pick_still(candidates: &[(f32, Option<f32>)], min_overlap: f32) -> usize {
    let covering = candidates.iter().any(|c| c.0 >= min_overlap);
    let rank = |c: &(f32, Option<f32>)| -> (bool, f32, f32) {
        if covering && c.0 >= min_overlap {
            (true, c.1.unwrap_or(f32::NEG_INFINITY), c.0)
        } else {
            (false, f32::NEG_INFINITY, c.0)
        }
    };
    (0..candidates.len())
        .max_by(|&a, &b| rank(&candidates[a]).partial_cmp(&rank(&candidates[b])).unwrap_or(std::cmp::Ordering::Equal))
        .expect("at least one edit")
}

/// Intersection over union of two masks; 0 when both are empty.
pub(super) fn iou(a: &[bool], b: &[bool]) -> f32 {
    let (inter, union) = a.iter().zip(b).fold((0usize, 0usize), |(i, u), (&x, &y)| (i + (x && y) as usize, u + (x || y) as usize));
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}
