// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `replace` - a source clip and a character sheet in, the same clip out with
//! the person clicked on replaced by the character: their path, timing and
//! the whole background kept.
//!
//! Swedish Embedded AB implements video-editing pipelines that put a new
//! character into existing footage for its clients. If your team needs
//! expertise in generative video editing then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! # The pipeline
//!
//! 1. **Track the actor.** SAM 2.1 follows the clicked person through every
//!    frame ([`sam2::track::track_clip`]).
//! 2. **Paint the character in on a few stills.** FLUX.2's served `edit`,
//!    with the sheet as a reference, repaints the actor's grown silhouette on
//!    a canvas around them and keeps the rest of the canvas exactly, so the
//!    character has to take the actor's place, size and pose instead of being
//!    staged somewhere else. Where the frame cuts the actor off, the canvas
//!    extends below it, so the model can draw a whole body at the actor's
//!    scale rather than shrinking it to fit. Frame 0 is painted against the
//!    sheet alone; every guide frame is painted against the sheet AND frame
//!    0's accepted still, cut to the head, hair and clothes, so the
//!    character's face, hair and clothes are the same at the clip's last frame
//!    as at its first instead of drifting into another depiction. SAM 2
//!    segments the new figure on the edit, and the edit is composited back
//!    over the source inside the union of the two silhouettes. Frame 0 is
//!    always one of them; the others are the guide frames.
//! 3. **Regenerate the actor's region.** LTX-2.5 masked video-to-video
//!    ([`crate::pipeline::generate_masked`]) holds every latent token outside
//!    the region and gets the edited stills after frame 0 as keyframe guides,
//!    held at half strength (`guide_strength`). Frame 0 is the edited still,
//!    held inside the figure and redrawn across its outline, where the edit
//!    blends the clothes into the street. The region on each frame is the
//!    actor's silhouette joined with the new figure's (carried from the
//!    nearest edited still along the actor's track and scaled with the
//!    actor's size), grown to whole latent tokens
//!    ([`crate::maskcond::snap_to_latent_tokens`]) - a token held only in part
//!    keeps part of the old costume.
//! 4. **Composite.** Outside the region every output pixel is the source's
//!    own; inside it, the generated clip's, across a feathered edge.
//!
//! The models other than LTX-2.5 are found through [`Components`]: the host
//! resolves them from its model store, and the `components` param names one
//! outright as `<model>.<role>=<path>`.
//!
//! Every intermediate (the actor track, the edited stills, the region mask
//! sequence, the clip the video pass sees, the generated clip) is written to
//! the work directory, so a bad result can be traced to the step that
//! produced it. The run is resumable: each expensive step records the key of
//! everything its output depends on ([`resume`]), and a run that finds the
//! same key loads the output instead of making it again. A run that failed in
//! the video pass, say out of memory, repeated with the same request goes
//! straight back to that pass; `fresh` redoes every step.

mod geometry;
mod resume;
mod stills;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{Components, Invocation};
use imaging::maskseq::{MaskClip, Polarity};
use imaging::{Ctx, Rgb8};
use model::adapter::AdapterSpec;

use crate::pipeline::{Paths, Video};
use geometry::{blend, carry, cut, feather, grow, shrink, Box2};
use resume::{digest, digest_mask, Resume};
use stills::{Editor, Identity, Still};

/// The other models `replace` runs, and the resolver spec of each: what a
/// host's [`Components`] has to be able to resolve.
pub fn component_specs() -> Vec<(&'static str, &'static dyn brain_modelstore::resolve::ArchSpec)> {
    vec![("sam2", &sam2::spec::Sam2Spec), ("flux2", &flux2::spec::Flux2Spec), ("arcface", &arcface::spec::ArcFaceSpec)]
}

/// Every dial of the pipeline. The defaults are the values it was tuned at;
/// each is a param of the `replace` action.
#[derive(Clone, Debug)]
pub struct Tuning {
    /// Without `guide_frames`, a guide every this many frames: about every
    /// second and a half at the usual clip rate, often enough that the figure's identity has no time
    /// to drift between guides.
    pub guide_every: usize,
    /// How hard the video pass holds a guide. Guides held at 1.0 are each a
    /// hard target the clip has to hit, and between two of them it can
    /// cross-fade one figure into the next instead of moving it - a ghosted,
    /// half-transparent figure. Held at half strength they still carry the
    /// identity and a pose like a wave, and the motion stays the clip's.
    pub guide_strength: f32,
    /// How many seeds an edit gets before its still is given up on.
    pub edit_attempts: u64,
    /// The least overlap (intersection over union) the painted figure must
    /// have with the actor's silhouette. A guide whose figure stands
    /// elsewhere, or at another size, pulls the clip toward its own staging
    /// and away from the actor's motion: it is worse than no guide.
    pub min_overlap: f32,
    /// The stills are rendered with this long side, then fitted back to the
    /// canvas's own size.
    pub edit_long_side: u32,
    /// How far past the actor's silhouette an edit repaints, as a fraction of
    /// the actor's width. The character's clothes are not the actor's - a
    /// coat flares wider than a hoodie - and an edge the repaint region cuts
    /// through is smeared into the background; the silhouette still fixes
    /// where the figure stands and how big it is.
    pub repaint_grow: f32,
    /// A standing person's height over their silhouette's median row width:
    /// how much body an actor the bottom edge cuts off is missing.
    pub body_ratio: f32,
    /// The margin around actor and figure a still is pasted back in, so the
    /// edit's own edge blends in rather than being cut on the silhouette.
    pub paste_grow_px: u32,
    /// The margin around actor and figure the video pass regenerates: the
    /// figure's stride and arm swing are not the actor's, and an arm that
    /// leaves the region is cut off by the composite.
    pub region_grow_px: u32,
    /// How far inside the figure's outline frame 0 stops being held: the
    /// band an edit blends across its repaint edge, and then some.
    pub frame0_inset_px: u32,
    /// The feather across every composite edge.
    pub feather_px: u32,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning {
            guide_every: 40,
            guide_strength: 0.5,
            edit_attempts: 3,
            min_overlap: 0.5,
            edit_long_side: 768,
            repaint_grow: 0.2,
            body_ratio: 4.0,
            paste_grow_px: 6,
            region_grow_px: 12,
            frame0_inset_px: 16,
            feather_px: 3,
        }
    }
}

impl Tuning {
    /// The dials an invocation sets, checked; unset ones keep their default.
    pub fn from_invocation(inv: &Invocation) -> Result<Tuning, String> {
        let d = Tuning::default();
        let int = |k: &str, v: u64| inv.get_i64(k).map_or(v, |n| n.max(0) as u64);
        let flt = |k: &str, v: f32| inv.get_f64(k).map_or(v, |x| x as f32);
        let t = Tuning {
            guide_every: int("guide_every", d.guide_every as u64) as usize,
            guide_strength: flt("guide_strength", d.guide_strength),
            edit_attempts: int("edit_attempts", d.edit_attempts),
            min_overlap: flt("min_overlap", d.min_overlap),
            edit_long_side: int("edit_size", d.edit_long_side as u64) as u32,
            repaint_grow: flt("repaint_grow", d.repaint_grow),
            body_ratio: flt("body_ratio", d.body_ratio),
            paste_grow_px: int("paste_grow", d.paste_grow_px as u64) as u32,
            region_grow_px: int("region_grow", d.region_grow_px as u64) as u32,
            frame0_inset_px: int("frame0_inset", d.frame0_inset_px as u64) as u32,
            feather_px: int("feather", d.feather_px as u64) as u32,
        };
        for (name, v) in [("guide_strength", t.guide_strength), ("min_overlap", t.min_overlap)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("{name} {v} is outside 0..1"));
            }
        }
        if t.guide_every == 0 || t.edit_attempts == 0 || t.edit_long_side < 64 || t.repaint_grow.is_nan() || t.repaint_grow < 0.0 || t.body_ratio.is_nan() || t.body_ratio <= 0.0 {
            return Err("guide_every and edit_attempts must be at least 1, edit_size at least 64, repaint_grow not negative, body_ratio positive".into());
        }
        Ok(t)
    }

    fn record(&self) -> serde_json::Value {
        serde_json::json!({
            "guide_every": self.guide_every, "guide_strength": self.guide_strength, "edit_attempts": self.edit_attempts,
            "min_overlap": self.min_overlap, "edit_size": self.edit_long_side, "repaint_grow": self.repaint_grow, "body_ratio": self.body_ratio,
            "paste_grow": self.paste_grow_px, "region_grow": self.region_grow_px, "frame0_inset": self.frame0_inset_px, "feather": self.feather_px,
        })
    }

    /// The settings an edited still depends on; the video pass's settings
    /// are left out, so changing one does not repaint the stills.
    fn edit_record(&self) -> serde_json::Value {
        serde_json::json!({
            "edit_attempts": self.edit_attempts, "min_overlap": self.min_overlap, "edit_size": self.edit_long_side, "repaint_grow": self.repaint_grow,
            "body_ratio": self.body_ratio, "paste_grow": self.paste_grow_px, "feather": self.feather_px,
        })
    }
}

/// One `replace` call, decoded and checked.
pub struct Request {
    /// The clip as it should look: what LTX-2.5 renders the region against.
    pub prompt: String,
    /// The instruction for the still edits.
    pub edit_prompt: String,
    /// The character sheet.
    pub character: Rgb8,
    /// `(x, y, label)` clicks in source pixels on [`Self::prompt_frame`].
    pub points: Vec<(f32, f32, f32)>,
    /// The actor's track given outright, instead of clicks.
    pub actor_mask: Option<MaskClip>,
    pub prompt_frame: usize,
    pub work: PathBuf,
    /// Reuse nothing from the work directory.
    pub fresh: bool,
    pub guide_frames: Option<Vec<usize>>,
    pub tuning: Tuning,
    /// Trim and crop a clip onto LTX-2.5's grid, instead of refusing it.
    pub conform: bool,
    /// The video pass's adapters.
    pub adapters: Vec<AdapterSpec>,
    /// The still edits' adapters.
    pub edit_adapters: Vec<AdapterSpec>,
    pub seed: u64,
    /// Keep the stills whose face is closest to the sheet's (ArcFace), not by
    /// overlap alone.
    pub identity: bool,
    /// Roles of the other models named outright: model -> role -> path.
    pub components: BTreeMap<String, BTreeMap<String, String>>,
    pub dit_config: Option<String>,
    pub steps: Option<usize>,
}

impl Request {
    /// The roles of `model` named outright.
    pub fn component(&self, model: &str) -> BTreeMap<String, String> {
        self.components.get(model).cloned().unwrap_or_default()
    }

    /// Decode a `replace` invocation (validated against its spec). The work
    /// directory, when none is given, is a fixed place under the system's
    /// temporary directory named by the clip and the sheet, so the same
    /// request repeated resumes.
    pub fn from_invocation(inv: &Invocation, clip: &Video) -> Result<Request, String> {
        let prompt = inv.get_str("prompt").ok_or("'prompt' is required")?;
        let (hwc, w, h) = capability::blob::decode_image(inv, "character")?;
        let character = Rgb8::new(w, h, hwc.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect())?;
        let clicks = |key: &str, label: f32| -> Result<Vec<(f32, f32, f32)>, String> {
            inv.get_str(key)
                .unwrap_or_default()
                .split(';')
                .map(str::trim)
                .filter(|g| !g.is_empty())
                .map(|g| {
                    let v: Vec<f32> = g.split(',').map(|p| p.trim().parse::<f32>()).collect::<Result<_, _>>().map_err(|e| format!("{key}: '{g}' is not 'x,y': {e}"))?;
                    match v[..] {
                        [x, y] => Ok((x, y, label)),
                        _ => Err(format!("{key}: '{g}' is not 'x,y'")),
                    }
                })
                .collect()
        };
        let mut points = clicks("points", 1.0)?;
        points.extend(clicks("exclude", 0.0)?);
        let actor_mask = inv.get_blob("actor_mask").map(MaskClip::from_blob).transpose()?;
        if actor_mask.is_none() && !points.iter().any(|p| p.2 == 1.0) {
            return Err("replace: give clicks on the person (points), or their track (actor_mask)".into());
        }
        let guide_frames = match inv.get_str("guide_frames").filter(|s| !s.trim().is_empty()) {
            None => None,
            Some(s) if s.trim() == "none" => Some(Vec::new()),
            Some(s) => Some(s.split(',').map(|f| f.trim().parse::<usize>().map_err(|e| format!("guide_frames: '{f}': {e}"))).collect::<Result<_, _>>()?),
        };
        let seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
        let work = match inv.get_str("work_dir").filter(|d| !d.is_empty()) {
            Some(d) => PathBuf::from(d),
            None => {
                let id = digest(clip.frames.iter().flatten().copied().chain(character.px.iter().copied()));
                std::env::temp_dir().join(format!("brain-ltxv-replace-{id:016x}"))
            }
        };
        let lora_scale = inv.get_f64("lora_scale").unwrap_or(1.0) as f32;
        let edit_scale = inv.get_f64("edit_lora_scale").unwrap_or(1.0) as f32;
        Ok(Request {
            edit_prompt: inv.get_str("edit_prompt").filter(|s| !s.is_empty()).unwrap_or_else(|| default_edit_prompt(&prompt)),
            prompt,
            character,
            points,
            actor_mask,
            prompt_frame: inv.get_i64("prompt_frame").unwrap_or(0).max(0) as usize,
            work,
            fresh: inv.get_bool("fresh").unwrap_or(false),
            guide_frames,
            tuning: Tuning::from_invocation(inv)?,
            conform: inv.get_bool("conform").unwrap_or(true),
            adapters: AdapterSpec::parse_list(&inv.get_str("adapter").unwrap_or_default(), lora_scale)?,
            edit_adapters: AdapterSpec::parse_list(&inv.get_str("edit_adapter").unwrap_or_default(), edit_scale)?,
            seed,
            identity: inv.get_bool("identity").unwrap_or(true),
            components: capability::component_overrides(&inv.get_str("components").unwrap_or_default())?,
            dit_config: inv.get_str("dit_config").filter(|s| !s.is_empty()),
            steps: inv.get_i64("steps").filter(|&s| s > 0).map(|s| s as usize),
        })
    }
}

/// The adapters as a key records them.
fn adapters_record(adapters: &[AdapterSpec]) -> serde_json::Value {
    serde_json::json!(adapters.iter().map(|a| serde_json::json!({"path": a.path, "scale": a.scale})).collect::<Vec<_>>())
}

/// The edit instruction when none is given.
fn default_edit_prompt(prompt: &str) -> String {
    format!(
        "Replace the person in image 1 with the character from image 2, keeping the character's face, hair and clothes exactly as in image 2. \
         The character stands exactly where the person stands, in the same pose, size and stride. Keep the background, lighting and every \
         other detail of image 1 exactly the same. The scene: {prompt}"
    )
}

/// What a run produced: the clip, and what it did that the caller should
/// know (conforming, dropped guides).
pub struct Replaced {
    pub video: Video,
    pub notes: Vec<String>,
    pub work: PathBuf,
}

/// Run `req` over `clip` (the source, its soundtrack attached when there is
/// one) with LTX-2.5 at `paths` and the other models from `components`.
pub fn run(paths: &Paths, components: &dyn Components, req: &Request, clip: Video, cancel: &capability::CancelToken, progress: &mut dyn FnMut(u32, u32, &str)) -> Result<Replaced, String> {
    std::fs::create_dir_all(&req.work).map_err(|e| format!("work_dir {}: {e}", req.work.display()))?;
    let mut notes = Vec::new();
    let mut decoded = clip;
    let sound = decoded.audio.take();
    let source_digest = digest(decoded.frames.iter().flatten().copied());
    // Before any model loads: a clip LTX-2.5 cannot take costs milliseconds,
    // not a tracking pass and four edits.
    let given_track = req.actor_mask.as_ref().map(|m| track_of(m, decoded.width, decoded.height, decoded.frames.len())).transpose()?;
    let Conformed { clip: source, points, track: given_track, note } = conform(decoded, &req.points, given_track, req.conform)?;
    if let Some(note) = note {
        notes.push(format!("{note}; the output is the conformed clip"));
    }
    let (w, h, n) = (source.width as usize, source.height as usize, source.frames.len());
    let grid = crate::maskcond::latent_grid(n, w, h)?;
    let guides = guide_frames(req.guide_frames.as_deref(), req.tuning.guide_every, n)?;
    tracing::info!("ltxv replace: {n} frames at {w}x{h}, edited stills at frames {:?}", std::iter::once(0).chain(guides.iter().copied()).collect::<Vec<_>>());
    let steps_total = (2 + guides.len() + 1) as u32;
    progress(0, steps_total, "tracking the actor");

    let tracker = load_tracker(components, &req.component("sam2"))?;
    let resume = Resume::new(&req.work, req.fresh);
    let track_key = serde_json::json!({
        "input": source_digest, "conform": req.conform, "frames": n, "size": [w, h], "points": points,
        "prompt_frame": req.prompt_frame, "tracker": req.component("sam2"),
    })
    .to_string();
    let actor_dir = req.work.join("actor");
    let actor: Vec<Vec<bool>> = match given_track {
        Some(track) => track,
        None if resume.holds("actor", &track_key) => {
            tracing::info!("ltxv replace: the actor's track is reused from {}", actor_dir.display());
            track_of(&MaskClip::read_dir(&actor_dir)?, w as u32, h as u32, n)?
        }
        None => {
            let unit: Vec<Vec<f32>> = source.frames.iter().map(|px| px.iter().map(|&v| v as f32 / 255.0).collect()).collect();
            track(&tracker, &unit, w as u32, h as u32, req.prompt_frame, &points, cancel)?
        }
    };
    if !resume.holds("actor", &track_key) {
        write_masks(&actor_dir, &actor, w, h, source.fps)?;
        if req.actor_mask.is_none() {
            resume.record("actor", &track_key)?;
        }
    }

    // The edited stills: frame 0 (the clip's own first frame) and the guides.
    let masks_gpu = gpu_core::Gpu::new(imaging::PIPELINES);
    let ctx = Ctx::new(&masks_gpu);
    let identity = Identity::load(req, components, &tracker.gpu)?;
    let flux2: Arc<dyn capability::Provider> = flux2::caps::provider(&components.resolve("flux2", &req.component("flux2"))?)?;
    let editor = Editor { req, flux2, resume: &resume, tracker: &tracker, identity, ctx: &ctx, w, h };
    let mut stills: Vec<Still> = Vec::new();
    // Frame 0's character, carried into every guide as an appearance
    // reference: without it a guide's edit can come back in different
    // clothes, and the video pass then dissolves the character into them.
    let mut appearance = None;
    for (i, &k) in std::iter::once(&0).chain(&guides).enumerate() {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        progress(1 + i as u32, steps_total, &format!("editing frame {k}"));
        if k > 0 && !actor[k].contains(&true) {
            notes.push(format!("frame {k} dropped as a guide: the actor is not visible on it"));
            continue;
        }
        let still = editor.best_still(k, &source.frames[k], &actor[k], appearance.as_ref())?;
        if k == 0 {
            appearance = Some(editor.appearance_reference(&still, &actor[0])?);
        }
        let iou = still.iou;
        let (attempts, min) = (req.tuning.edit_attempts, req.tuning.min_overlap);
        if k == 0 || iou >= min {
            tracing::info!("ltxv replace: frame {k} edited, figure/actor overlap {iou:.2}");
            stills.push(still);
        } else {
            notes.push(format!("frame {k} dropped as a guide: the best of {attempts} edits overlaps the actor only {iou:.2} (< {min})"));
        }
    }
    if stills[0].iou < req.tuning.min_overlap {
        return Err(format!(
            "frame 0: the best of {} edits overlaps the actor only {:.2} (< {}); it is the clip's first frame and cannot be dropped - see {} and try another seed, edit_prompt or min_overlap",
            req.tuning.edit_attempts,
            stills[0].iou,
            req.tuning.min_overlap,
            req.work.display()
        ));
    }
    // The edit models are done: the video pass gets the memory they held.
    drop(editor);
    drop(tracker);

    // The region to regenerate on every frame. Frame 0 is the edited still,
    // held inside the figure - who the character is - and regenerated in a
    // ring across its outline: the edit blends whatever crosses the edge of
    // the region it repainted, and the video pass redraws that edge
    // consistently with the held figure.
    let regions: Vec<Vec<bool>> = (0..n)
        .map(|f| {
            if f == 0 {
                let still = &stills[0];
                let joined: Vec<bool> = actor[0].iter().zip(&still.figure).map(|(&a, &c)| a || c).collect();
                let held = shrink(&ctx, &still.figure, w, h, req.tuning.frame0_inset_px);
                return grow(&ctx, &joined, w, h, req.tuning.region_grow_px).iter().zip(&held).map(|(&r, &k)| r && !k).collect();
            }
            let near = stills.iter().min_by_key(|s| s.frame.abs_diff(f)).expect("frame 0 is always edited");
            let carried = carry(&near.figure, &actor[near.frame], &actor[f], w, h);
            let joined: Vec<bool> = actor[f].iter().zip(&carried).map(|(&a, &c)| a || c).collect();
            grow(&ctx, &joined, w, h, req.tuning.region_grow_px)
        })
        .collect();
    let regions = crate::maskcond::snap_to_latent_tokens(&regions, w, h, grid);
    let region = masks(&regions, w, h, source.fps)?;
    region.write_dir(req.work.join("region"))?;

    let mut spliced = source;
    spliced.frames[0] = stills[0].image.px.clone();
    // The clip the video pass sees, kept so the pass can be run again alone
    // (`ltxv v2v --video <work>/input.mp4 --mask <work>/region ...`).
    write_clip(&spliced, &req.work.join("input.mp4"))?;
    let generated_path = req.work.join("generated.mp4");
    let video_key = serde_json::json!({
        "input": source_digest, "regions": digest(regions.iter().map(|r| digest_mask(r)).flat_map(u64::to_le_bytes)),
        "stills": stills.iter().map(|s| (s.frame, digest(s.image.px.iter().copied()))).collect::<Vec<_>>(),
        "prompt": req.prompt, "seed": req.seed, "steps": req.steps, "dit_config": req.dit_config, "guide_strength": req.tuning.guide_strength,
        "adapters": adapters_record(&req.adapters), "video_model": paths.dit,
    })
    .to_string();
    let reused = if resume.holds("generated", &video_key) { read_clip(&generated_path, spliced.fps).ok().filter(|v| v.frames.len() == n) } else { None };
    let generated = match reused {
        Some(video) => {
            tracing::info!("ltxv replace: the video pass is reused from {}", generated_path.display());
            video
        }
        None => {
            let video = generate(paths, req, &spliced, region, &stills[1..], cancel, &mut |done, total, phase| {
                progress(steps_total - 1, steps_total, &format!("video [{done}/{total}] {phase}"))
            })?;
            write_clip(&video, &generated_path)?;
            resume.record("generated", &video_key)?;
            video
        }
    };

    // Outside the region every pixel is the source's - the still's, on frame 0.
    let mut out = spliced;
    out.audio = match sound {
        Some(s) => Some(fit_soundtrack(s, n, out.fps)?),
        None => {
            notes.push("the result is silent: give the clip's sound as the soundtrack input to carry it".into());
            None
        }
    };
    for ((frame, region), generated) in out.frames.iter_mut().zip(&regions).zip(&generated.frames) {
        *frame = blend(generated, frame, &feather(&ctx, region, w, h, req.tuning.feather_px));
    }
    progress(steps_total, steps_total, "composited");
    tracing::info!("ltxv replace: tuning {}", req.tuning.record());
    Ok(Replaced { video: out, notes, work: req.work.clone() })
}

/// A clip on LTX-2.5's grid, with its clicks and actor track moved onto it.
struct Conformed {
    clip: Video,
    points: Vec<(f32, f32, f32)>,
    track: Option<Vec<Vec<bool>>>,
    /// What conforming did, `None` when the clip already fit.
    note: Option<String>,
}

/// Fit `clip` to LTX-2.5's grid: trim it at its end to 1 + 8k frames and
/// centre-crop it to multiples of 32, moving the clicks (`(x, y, label)` in
/// source pixels) and cropping the given actor track with it. With `allowed`
/// false a clip that needs it is an error instead.
fn conform(clip: Video, points: &[(f32, f32, f32)], track: Option<Vec<Vec<bool>>>, allowed: bool) -> Result<Conformed, String> {
    let (w, h, n) = (clip.width as usize, clip.height as usize, clip.frames.len());
    if n < 9 || w < 32 || h < 32 {
        return Err(format!("{n} frames at {w}x{h}: LTX-2.5 needs at least 9 frames (1 + 8k) and 32x32 pixels"));
    }
    let (n2, w2, h2) = (1 + (n - 1) / 8 * 8, w / 32 * 32, h / 32 * 32);
    if (n2, w2, h2) == (n, w, h) {
        return Ok(Conformed { clip, points: points.to_vec(), track, note: None });
    }
    let (x0, y0) = ((w - w2) / 2, (h - h2) / 2);
    let note = format!("{n} frames at {w}x{h} conformed to {n2} frames at {w2}x{h2} (the last {} frame(s) trimmed, cropped from ({x0}, {y0}))", n - n2);
    if !allowed {
        return Err(format!("{n} frames at {w}x{h} is not 1 + 8k frames at multiples of 32, and conform=false refuses to trim it to {n2} frames at {w2}x{h2}"));
    }
    let moved: Vec<(f32, f32, f32)> = points.iter().map(|&(x, y, l)| (x - x0 as f32, y - y0 as f32, l)).collect();
    if let Some(&(x, y, _)) = points.iter().zip(&moved).find(|(_, m)| !(0.0..w2 as f32).contains(&m.0) || !(0.0..h2 as f32).contains(&m.1)).map(|(p, _)| p) {
        return Err(format!("the click ({x}, {y}) falls outside the centre crop to {w2}x{h2}: click inside it"));
    }
    let crop = Box2 { x: x0, y: y0, w: w2, h: h2 };
    let frames = clip.frames.into_iter().take(n2).map(|f| cut(&f, w, h, crop)).collect();
    let track = track.map(|t| t.into_iter().take(n2).map(|m| (y0..y0 + h2).flat_map(|y| m[y * w + x0..y * w + x0 + w2].to_vec()).collect()).collect());
    Ok(Conformed { clip: Video { width: w2 as u32, height: h2 as u32, frames, ..clip }, points: moved, track, note: Some(note) })
}

/// An actor track given as a mask sequence, checked against the clip it
/// tracks; a frame the tracker marked occluded is an empty mask, so the
/// actor there is left as it is.
fn track_of(seq: &MaskClip, w: u32, h: u32, n: usize) -> Result<Vec<Vec<bool>>, String> {
    if (seq.width, seq.height, seq.frames.len()) != (w, h, n) {
        return Err(format!("actor_mask: {} frames at {}x{}, the clip has {n} at {w}x{h}", seq.frames.len(), seq.width, seq.height));
    }
    Ok(seq
        .frames
        .iter()
        .zip(&seq.info)
        .map(|(plane, info)| if info.object_score <= 0.0 { vec![false; (w * h) as usize] } else { plane.iter().map(|&v| seq.polarity.objectness(v) > 0.5).collect() })
        .collect())
}

/// The soundtrack cut to `frames` at `fps`; one too short is an error.
fn fit_soundtrack(mut clip: crate::audio::AudioClip, frames: usize, fps: usize) -> Result<crate::audio::AudioClip, String> {
    let want = (frames as u64 * clip.sample_rate as u64).div_ceil(fps as u64) as usize;
    if clip.samples_per_channel() < want {
        return Err(format!("the soundtrack holds {} samples per channel, the clip needs {want} ({frames} frames at {fps} fps)", clip.samples_per_channel()));
    }
    clip.channels.iter_mut().for_each(|c| c.truncate(want));
    Ok(clip)
}

/// The guide frames: the ones asked for, checked, or the default spacing.
fn guide_frames(asked: Option<&[usize]>, every: usize, n: usize) -> Result<Vec<usize>, String> {
    let Some(asked) = asked else {
        return Ok((1..n).filter(|f| f % every == 0 || *f == n - 1).collect());
    };
    if let Some(bad) = asked.iter().find(|&&f| f == 0 || f >= n) {
        return Err(format!("guide_frames: frame {bad} is not inside 1..{} (frame 0 is always edited: it is the clip's first frame)", n - 1));
    }
    let mut sorted = asked.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|p| p[0] == p[1]) {
        return Err(format!("guide_frames names a frame twice: {asked:?}"));
    }
    Ok(sorted)
}

/// The LTX-2.5 masked pass over `clip`, regenerating `region` and guided by
/// `guides`.
fn generate(paths: &Paths, req: &Request, clip: &Video, region: MaskClip, guides: &[Still], cancel: &capability::CancelToken, progress: &mut dyn FnMut(u32, u32, &str)) -> Result<Video, String> {
    let mut o = crate::pipeline::MaskedOpts { mask: Some(region), ..Default::default() };
    o.base.seed = req.seed;
    o.base.fps = clip.fps;
    o.base.conditioning_strength = req.tuning.guide_strength;
    o.base.adapters = req.adapters.clone();
    o.base.future_anchors = guides.iter().map(|s| (s.frame, req.work.join(format!("still_{}.png", s.frame)).to_string_lossy().into_owned())).collect();
    if let Some(d) = &req.dit_config {
        o.base.dit_config = d.clone();
    }
    crate::pipeline::resolve_defaults(&mut o.base, paths, crate::pipeline::Named { dit_config: req.dit_config.is_some(), frames: true, size: true, fps: true });
    if let Some(s) = req.steps {
        o.base.steps = s;
    }
    let (video, _) = crate::pipeline::generate_masked(paths, &req.prompt, clip, &o, cancel, |done, total, phase| progress(done, total, phase))?;
    Ok(video)
}

/// Masks as a mask sequence, object = 1, every frame seen.
fn masks(masks: &[Vec<bool>], w: usize, h: usize, fps: usize) -> Result<MaskClip, String> {
    let mut seq = MaskClip::new(w as u32, h as u32, fps as f64, Polarity::ObjectWhite, true);
    for m in masks {
        seq.push_logits(&m.iter().map(|&r| if r { 1.0 } else { -1.0 }).collect::<Vec<f32>>(), 1.0, 1.0)?;
    }
    Ok(seq)
}

fn write_masks(dir: &Path, m: &[Vec<bool>], w: usize, h: usize, fps: usize) -> Result<(), String> {
    masks(m, w, h, fps)?.write_dir(dir).map(|_| ())
}

/// The SAM 2.1 tracking model the store and `overrides` resolve to.
fn load_tracker(components: &dyn Components, overrides: &BTreeMap<String, String>) -> Result<sam2::Sam2, String> {
    let assembly = components.resolve("sam2", overrides)?;
    let variant = assembly.variant.clone().ok_or("ltxv replace: the resolved SAM 2 checkpoint has no variant")?;
    sam2::track::load_video(&assembly.role_path("weights")?, sam2::caps::variant_config(&variant)?, gpu_core::Gpu::new(sam2::PIPELINES))
}

/// The object `points` pick, tracked through `frames`: one mask per frame,
/// empty where SAM 2 believes the object is occluded.
fn track(m: &sam2::Sam2, frames: &[Vec<f32>], w: u32, h: u32, prompt_frame: usize, points: &[(f32, f32, f32)], cancel: &capability::CancelToken) -> Result<Vec<Vec<bool>>, String> {
    let req = sam2::track::TrackRequest { points: points.to_vec(), prompt_frame, object_id: 0, polarity: Polarity::ObjectWhite, binary: true };
    let masks = sam2::track::track_clip(m, frames, w, h, 0.0, &req, &mut |_, _| !cancel.is_cancelled())?;
    Ok(masks.frames.iter().zip(&masks.info).map(|(p, i)| if i.object_score > 0.0 { p.iter().map(|&v| v > 0.5).collect() } else { vec![false; p.len()] }).collect())
}

/// A clip in the work directory, read back at `fps`.
fn read_clip(path: &Path, fps: usize) -> Result<Video, String> {
    let decoded = imaging::video::decode_frames_rgb8(path, &imaging::video::VideoDecodeOpts { fps: None, max_frames: 0, spread: 0 })?;
    let first = decoded.first().ok_or_else(|| format!("{} has no frames", path.display()))?;
    let (w, h) = (first.w, first.h);
    Ok(Video { width: w, height: h, fps, frames: decoded.into_iter().map(|f| f.px).collect(), audio: None })
}

/// A clip written to the work directory, with its sound when it has one.
fn write_clip(video: &Video, path: &Path) -> Result<(), String> {
    let track = video.audio.as_ref().map(|a| imaging::video::AudioTrack { channels: a.channels.clone(), sample_rate: a.sample_rate });
    let frames: Vec<Rgb8> = video.frames.iter().map(|px| Rgb8::new(video.width, video.height, px.clone())).collect::<Result<_, _>>()?;
    imaging::video::encode_frames(&frames, path, video.fps as f64, &imaging::video::VideoEncodeOpts { audio: track, ..Default::default() }).map(|_| ())
}

#[cfg(test)]
mod tests;
