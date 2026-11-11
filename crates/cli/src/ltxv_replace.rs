// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain ltxv replace` - a source clip and a character sheet in, the same
//! clip out with the person clicked on replaced by the character: their
//! path, timing and the whole background kept.
//!
//! Swedish Embedded AB implements video-editing pipelines that put a new
//! character into existing footage for its clients. If your team needs
//! expertise in generative video editing then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! # The pipeline
//!
//! 1. **Track the actor.** SAM 2.1 follows the clicked person through every
//!    frame ([`crate::sam2_cli::track_frames`]).
//! 2. **Paint the character in on a few stills.** FLUX.2, with the sheet as
//!    its reference, repaints the actor's grown silhouette on a canvas around
//!    them and keeps the rest of the canvas exactly, so the character has to
//!    take the actor's place, size and pose instead of being staged somewhere
//!    else. Where the frame cuts the actor off, the canvas extends below it,
//!    so the model can draw a whole body at the actor's scale rather than
//!    shrinking it to fit. SAM 2 segments the new figure on the edit,
//!    and the edit is composited back over the source inside the union of
//!    the two silhouettes. Frame 0 is always one of them; the others are the
//!    guide frames.
//! 3. **Regenerate the actor's region.** LTX-2.5 masked video-to-video
//!    ([`ltxv::pipeline::generate_masked`]) holds every latent token outside
//!    the region and gets the edited stills after frame 0 as keyframe
//!    guides, held at half strength (`--guide-strength`). Frame 0 is the
//!    edited still, held inside the figure and redrawn across its outline,
//!    where the edit blends the clothes into the street. The region on each
//!    frame is the actor's silhouette joined with the new figure's (carried
//!    from the nearest edited still along the actor's track and scaled with
//!    the actor's size), grown to whole latent tokens
//!    ([`ltxv::maskcond::snap_to_latent_tokens`]) - a token held only in part
//!    keeps part of the old costume.
//! 4. **Composite.** Outside the region every output pixel is the source's
//!    own; inside it, the generated clip's, across a feathered edge.
//!
//! Every intermediate (the actor track, the edited stills, the region mask
//! sequence, the clip the video pass sees, the generated clip) is written to
//! `--work-dir`, so a bad result
//! can be traced to the step that produced it.
//!
//! The run is resumable: each expensive step (the track, each edited still,
//! the video pass) records the key of everything its output depends on
//! ([`crate::ltxv_replace_resume`]), and a run that finds the same key
//! loads the output instead of making it again. A run that failed in the
//! video pass, say out of memory, repeated with the same arguments goes
//! straight back to that pass. Change a setting and only the steps that
//! depend on it are redone; `--fresh` redoes all of them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use brain_modelstore::resolve::ArchSpec;
use imaging::{AlignCorners, Ctx, Filter, Rgb8, Shape};

use crate::args::Args;
use crate::ltxv_replace_resume::{digest, digest_mask, stamp, Resume};

pub(crate) const HELP: &str = "\
brain ltxv replace - replace a person in a clip with a character from a sheet

USAGE:
  brain ltxv replace --input <clip> --character <sheet.png> --point <x,y>
                     --prompt <text> --output-path <out.mp4> [options]

  --input <clip>          the source video, any container brain decodes
                          (.mp4 / .webm / .mkv / .mov). LTX-2.5 takes 1 + 8k
                          frames with both sides a multiple of 32; a clip that
                          is not is trimmed at its end and centre-cropped to
                          the nearest that is, and the run says so
  --no-conform            refuse such a clip instead of trimming and cropping
  --fps <n>               the source's frame rate, when its container does
                          not state one
  --character <image>     the character sheet: who the actor becomes
  --point <x,y>           a click on the actor, in source pixels (repeatable;
                          a few clicks over coat, trousers and head pick the
                          whole person)
  --exclude <x,y>         a click on something that is NOT the actor (repeatable)
  --prompt-frame <n>      the frame the clicks sit on (default 0)
  --actor-mask <dir>      the actor's track as a brain/sam2-maskseq/1 directory
                          (`brain sam2 track` writes one), instead of clicks
  --prompt <text>         the clip as it should look: the character's looks
                          and clothes, what they do, the scene - LTX-2.5
                          renders the region against this
  --edit-prompt <text>    the instruction for the still edits (default: put
                          the character from the sheet in the actor's place,
                          pose and size, the rest of the image unchanged)
  --output-path <file>    the result (.mp4 / .webm / .mkv)
  --soundtrack <wav>      the sound to carry into the result, cut to the
                          clip's length; brain reads no sound out of a
                          video container, so without it the result is silent
  --work-dir <dir>        where the intermediates go (default: <output>.work).
                          A run repeated with the same work dir reuses every
                          step whose inputs and settings are unchanged
  --fresh                 redo every step, reusing nothing from the work dir
  --seed <n>              seed for the edits and the video pass

Guides:
  --guide-frames <list>   the frames, besides frame 0, painted as keyframe
                          guides: comma-separated, or 'none'
  --guide-every <n>       without --guide-frames: a guide every n frames, and
                          the last frame (default 40)
  --guide-strength <s>    how hard the video pass holds each guide, 0..1
                          (default 0.5: held fully, the clip dissolves
                          between stills instead of moving between them)

Stills:
  --edit-attempts <n>     edits made per still (default 3). With an identity
                          model in the store (ArcFace), all are made and the
                          one covering the actor whose face is closest to
                          the sheet's is kept; without one, the first that
                          covers the actor
  --no-identity           keep stills by overlap alone, without ArcFace
  --min-overlap <f>       the least intersection over union of painted figure
                          and actor a still needs, 0..1 (default 0.5); a guide
                          below it is dropped, frame 0 below it stops the run
  --edit-size <px>        the long side the stills are painted at (default 768)
  --repaint-grow <f>      how far past the actor's outline an edit may paint,
                          as a fraction of the actor's width (default 0.2)
  --body-ratio <f>        a standing person's height over their silhouette's
                          median width: how much body is painted below the
                          frame where it cuts the actor off (default 4)
  --paste-grow <px>       margin around actor and figure a still is pasted
                          back in (default 6)
  --edit-adapter <path>   a LoRA for the still edits (repeatable; each may be
                          followed by --edit-lora-scale <s>, default 1.0)

Video pass:
  --region-grow <px>      margin around actor and figure that is regenerated
                          (default 12)
  --frame0-inset <px>     how far inside the figure frame 0 stops being held
                          and is redrawn (default 16)
  --feather <px>          the feather across every composite edge (default 3)
  --adapter <path>        a LoRA for the video pass (repeatable; each may be
                          followed by --lora-scale <s>, default 1.0)
  --dit-config <name>     the LTX-2.5 DiT (default: the resolved checkpoint's)
  --steps <n>             video denoising steps (default: the checkpoint's)
  --device <spec>         the video pass's placement, as `ltxv v2v` takes it

Model roles resolve from the models directory like every other command.
Name LTX-2.5's outright with --dit/--vae/--text-encoder/--tokenizer, and
the other models' with --component <model>.<role>=<path> (repeatable):
flux2.dit, flux2.vae, flux2.text_encoder, flux2.tokenizer, flux2.variant
(the stills), sam2.weights, sam2.variant (the tracking), arcface.weights
(the face comparison).
";

/// Every dial of the pipeline. The defaults are the values the pipeline was
/// tuned at; each is a flag (see [`HELP`]).
struct Tuning {
    /// Without `--guide-frames`, a guide every this many frames: about every
    /// 1.7 s at 24 fps, often enough that the figure's identity has no time
    /// to drift between guides.
    guide_every: usize,
    /// How hard the video pass holds a guide. Guides held at 1.0 are each a
    /// hard target the clip has to hit, and between two of them it can
    /// cross-fade one figure into the next instead of moving it - a ghosted,
    /// half-transparent figure. Held at half strength they still carry the
    /// identity and a pose like a wave, and the motion stays the clip's.
    guide_strength: f32,
    /// How many seeds an edit gets before its still is given up on.
    edit_attempts: u64,
    /// The least overlap (intersection over union) the painted figure must
    /// have with the actor's silhouette. A guide whose figure stands
    /// elsewhere, or at another size, pulls the clip toward its own staging
    /// and away from the actor's motion: it is worse than no guide.
    min_overlap: f32,
    /// The stills are rendered with this long side, then fitted back to the
    /// canvas's own size.
    edit_long_side: u32,
    /// How far past the actor's silhouette an edit repaints, as a fraction of
    /// the actor's width. The character's clothes are not the actor's - a
    /// coat flares wider than a hoodie - and an edge the repaint region cuts
    /// through is smeared into the background; the silhouette still fixes
    /// where the figure stands and how big it is.
    repaint_grow: f32,
    /// A standing person's height over their silhouette's median row width:
    /// how much body an actor the bottom edge cuts off is missing.
    body_ratio: f32,
    /// The margin around actor and figure a still is pasted back in, so the
    /// edit's own edge blends in rather than being cut on the silhouette.
    paste_grow_px: u32,
    /// The margin around actor and figure the video pass regenerates: the
    /// figure's stride and arm swing are not the actor's, and an arm that
    /// leaves the region is cut off by the composite.
    region_grow_px: u32,
    /// How far inside the figure's outline frame 0 stops being held: the
    /// band an edit blends across its repaint edge, and then some.
    frame0_inset_px: u32,
    /// The feather across every composite edge.
    feather_px: u32,
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

    fn parse(a: &mut Args) -> Result<Tuning, String> {
        let d = Tuning::default();
        let t = Tuning {
            guide_every: a.parsed("--guide-every").unwrap_or(d.guide_every),
            guide_strength: a.parsed("--guide-strength").unwrap_or(d.guide_strength),
            edit_attempts: a.parsed("--edit-attempts").unwrap_or(d.edit_attempts),
            min_overlap: a.parsed("--min-overlap").unwrap_or(d.min_overlap),
            edit_long_side: a.parsed("--edit-size").unwrap_or(d.edit_long_side),
            repaint_grow: a.parsed("--repaint-grow").unwrap_or(d.repaint_grow),
            body_ratio: a.parsed("--body-ratio").unwrap_or(d.body_ratio),
            paste_grow_px: a.parsed("--paste-grow").unwrap_or(d.paste_grow_px),
            region_grow_px: a.parsed("--region-grow").unwrap_or(d.region_grow_px),
            frame0_inset_px: a.parsed("--frame0-inset").unwrap_or(d.frame0_inset_px),
            feather_px: a.parsed("--feather").unwrap_or(d.feather_px),
        };
        for (flag, v) in [("--guide-strength", t.guide_strength), ("--min-overlap", t.min_overlap)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("{flag} {v} is outside 0..1"));
            }
        }
        if t.guide_every == 0 || t.edit_attempts == 0 || t.edit_long_side < 64 || t.repaint_grow.is_nan() || t.repaint_grow < 0.0 || t.body_ratio.is_nan() || t.body_ratio <= 0.0 {
            return Err("--guide-every and --edit-attempts must be at least 1, --edit-size at least 64, --repaint-grow not negative, --body-ratio positive".into());
        }
        Ok(t)
    }
}

struct Request {
    input: String,
    character: String,
    points: Vec<(f32, f32, f32)>,
    /// The actor's track given outright, instead of clicks.
    actor_mask: Option<PathBuf>,
    prompt_frame: usize,
    prompt: String,
    edit_prompt: String,
    out: String,
    work: PathBuf,
    /// `--fresh`: reuse nothing from the work directory.
    fresh: bool,
    guide_frames: Option<Vec<usize>>,
    tuning: Tuning,
    conform: bool,
    fps: Option<usize>,
    soundtrack: Option<String>,
    /// `--adapter <path> [--lora-scale <s>]` for the video pass.
    adapters: Vec<model::adapter::AdapterSpec>,
    /// The still edits' adapters, as `flux2 generate` flags.
    edit_adapter_flags: Vec<String>,
    seed: u64,
    flux2_roles: Vec<String>,
    sam2_overrides: BTreeMap<String, String>,
    arcface_overrides: BTreeMap<String, String>,
    /// `--no-identity`: keep the stills by overlap alone.
    no_identity: bool,
    ltxv_overrides: BTreeMap<String, String>,
    dit_config: Option<String>,
    steps: Option<usize>,
    device: Option<String>,
}

fn parse(args: &[String]) -> Result<Request, String> {
    let (ltxv_overrides, rest) = crate::resolver_cli::extract_role_overrides(&ltxv::spec::LtxvSpec, args);
    let (seed, rest) = crate::args::split_seed(&rest, "ltxv replace");
    let Adapters { video: adapters, edit_flags: edit_adapter_flags, rest } = take_adapters(&rest)?;
    let mut a = Args::new(&rest);
    let take = |a: &mut Args, flag: &str| a.take_str(flag).ok_or_else(|| format!("{flag} is required\n\n{HELP}"));
    let input = take(&mut a, "--input")?;
    let character = take(&mut a, "--character")?;
    let prompt = take(&mut a, "--prompt")?;
    let out = crate::args::strip_out_name_prefix(&take(&mut a, "--output-path")?, "video").to_string();
    let mut points = Vec::new();
    for (flag, label) in [("--point", 1.0), ("--exclude", 0.0)] {
        while let Some(p) = a.take_str(flag) {
            let (x, y) = crate::sam2_cli::parse_point(&p)?;
            points.push((x, y, label));
        }
    }
    let actor_mask = a.take_str("--actor-mask").map(PathBuf::from);
    match (points.iter().any(|p| p.2 == 1.0), &actor_mask) {
        (false, None) => return Err(format!("--point <x,y> or --actor-mask <dir> is required: it is what picks the actor\n\n{HELP}")),
        (true, Some(_)) => return Err("--point and --actor-mask both pick the actor: give one".into()),
        _ => {}
    }
    let guide_frames = match a.take_str("--guide-frames").as_deref() {
        None => None,
        Some("none") => Some(Vec::new()),
        Some(list) => Some(list.split(',').map(|f| f.trim().parse::<usize>().map_err(|e| format!("--guide-frames: '{f}': {e}"))).collect::<Result<_, _>>()?),
    };
    let (mut flux2_roles, mut sam2_overrides, mut arcface_overrides) = (Vec::new(), BTreeMap::new(), BTreeMap::new());
    while let Some(c) = a.take_str("--component") {
        let (model, role, path) = parse_component(&c)?;
        match model {
            Component::Editor => flux2_roles.extend([format!("--{}", role.replace('_', "-")), path]),
            Component::Tracker => drop(sam2_overrides.insert(role, path)),
            Component::Identity => drop(arcface_overrides.insert(role, path)),
        }
    }
    let req = Request {
        work: a.take_str("--work-dir").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("{out}.work"))),
        fresh: a.take_flag("--fresh"),
        prompt_frame: a.usize_or("--prompt-frame", 0),
        edit_prompt: a.take_str("--edit-prompt").unwrap_or_else(|| default_edit_prompt(&prompt)),
        dit_config: a.take_str("--dit-config"),
        steps: a.parsed("--steps"),
        tuning: Tuning::parse(&mut a)?,
        conform: !a.take_flag("--no-conform"),
        fps: a.parsed("--fps"),
        soundtrack: a.take_str("--soundtrack"),
        adapters,
        edit_adapter_flags,
        device: a.take_str("--device"),
        input,
        character,
        points,
        actor_mask,
        prompt,
        out,
        guide_frames,
        seed,
        flux2_roles,
        sam2_overrides,
        arcface_overrides,
        no_identity: a.take_flag("--no-identity"),
        ltxv_overrides,
    };
    a.finish();
    Ok(req)
}

/// The adapter flags of a command line, and the rest of it.
struct Adapters {
    /// `--adapter <path> [--lora-scale <s>]`, for the video pass.
    video: Vec<model::adapter::AdapterSpec>,
    /// `--edit-adapter`/`--edit-lora-scale` as `flux2 generate` flags.
    edit_flags: Vec<String>,
    rest: Vec<String>,
}

/// Pull the adapter flags out of `args` in order - a `--lora-scale` belongs
/// to the `--adapter` it follows, which a flag-at-a-time parser cannot see.
fn take_adapters(args: &[String]) -> Result<Adapters, String> {
    let (mut video, mut scaled, mut edit, mut rest) = (Vec::new(), 0usize, Vec::new(), Vec::new());
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--adapter" => video.push(model::adapter::AdapterSpec::new(value()?)),
            "--lora-scale" => {
                let v = value()?;
                crate::args::attach_lora_scale(&mut video, &mut scaled, v.parse().map_err(|e| format!("--lora-scale '{v}': {e}"))?)?;
            }
            "--edit-adapter" => edit.extend(["--adapter".to_string(), value()?]),
            "--edit-lora-scale" => {
                if edit.len() < 2 || edit[edit.len() - 2] != "--adapter" {
                    return Err("--edit-lora-scale applies to the --edit-adapter it follows, and follows none here".into());
                }
                edit.extend(["--lora-scale".to_string(), value()?]);
            }
            _ => rest.push(flag.clone()),
        }
    }
    Ok(Adapters { video, edit_flags: edit, rest })
}

/// The pipeline's models besides LTX-2.5, as `--component` names them.
#[derive(Debug, PartialEq)]
enum Component {
    /// FLUX.2, which paints the stills.
    Editor,
    /// SAM 2.1, which tracks the actor and segments the painted figure.
    Tracker,
    /// ArcFace, which compares a painted face with the sheet's.
    Identity,
}

/// `--component <model>.<role>=<path>`: one role of one of the pipeline's
/// other models, named outright - `flux2.dit=<file>`, `sam2.weights=<file>`.
/// The role must be one the model's resolver declares (or its `variant`),
/// so a misspelt role is an error rather than a flag nobody reads.
fn parse_component(text: &str) -> Result<(Component, String, String), String> {
    let bad = || format!("--component '{text}' is not <model>.<role>=<path> (models: flux2, sam2, arcface)");
    let (name, rest) = text.split_once('.').ok_or_else(bad)?;
    let (role, path) = rest.split_once('=').ok_or_else(bad)?;
    let role = role.replace('-', "_");
    let (component, roles): (Component, &[&str]) = match name {
        "flux2" => (Component::Editor, flux2::spec::Flux2Spec.roles()),
        "sam2" => (Component::Tracker, sam2::spec::Sam2Spec.roles()),
        "arcface" => (Component::Identity, arcface::spec::ArcFaceSpec.roles()),
        _ => return Err(bad()),
    };
    if role != "variant" && !roles.contains(&role.as_str()) {
        return Err(format!("--component {text}: {name} has no role '{role}' (its roles: {}, variant)", roles.join(", ")));
    }
    if path.is_empty() {
        return Err(bad());
    }
    Ok((component, role, path.to_string()))
}

fn default_edit_prompt(prompt: &str) -> String {
    format!(
        "Replace the person in image 1 with the character from image 2, keeping the character's face, hair and clothes exactly as in image 2. \
         The character stands exactly where the person stands, in the same pose, size and stride. Keep the background, lighting and every \
         other detail of image 1 exactly the same. The scene: {prompt}"
    )
}

pub(crate) fn replace(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{HELP}");
        return Ok(());
    }
    let req = parse(args)?;
    let t0 = std::time::Instant::now();
    std::fs::create_dir_all(&req.work).map_err(|e| format!("--work-dir {}: {e}", req.work.display()))?;
    let decoded = crate::ltxv_cli::read_clip(&req.input, crate::ltxv_cli::source_fps(&req.input, req.fps)?)?;
    // Before any model loads: a clip LTX-2.5 cannot take costs milliseconds,
    // not a tracking pass and four edits.
    let given_track = req.actor_mask.as_deref().map(|dir| read_track(dir, decoded.width, decoded.height, decoded.frames.len())).transpose()?;
    let Conformed { clip: source, points, track: given_track, note: conformed } =
        conform(decoded, &req.points, given_track, req.conform).map_err(|e| format!("{}: {e}", req.input))?;
    if let Some(note) = &conformed {
        eprintln!("ltxv replace: {}: {note}; the output is the conformed clip", req.input);
    }
    let (w, h, n) = (source.width as usize, source.height as usize, source.frames.len());
    let source_fps = source.fps;
    let grid = ltxv::maskcond::latent_grid(n, w, h)?;
    let guides = guide_frames(req.guide_frames.as_deref(), req.tuning.guide_every, n)?;
    eprintln!("ltxv replace: {n} frames at {w}x{h}, edited stills at frames {:?}", std::iter::once(0).chain(guides.iter().copied()).collect::<Vec<_>>());

    let tracker = crate::sam2_cli::load_tracker(&req.sam2_overrides)?;
    let resume = Resume::new(&req.work, req.fresh);
    let track_key = serde_json::json!({
        "input": stamp(&req.input), "conform": req.conform, "frames": n, "size": [w, h], "points": points,
        "prompt_frame": req.prompt_frame, "tracker": req.sam2_overrides,
    })
    .to_string();
    let actor_dir = req.work.join("actor");
    let actor: Vec<Vec<bool>> = match given_track {
        Some(track) => track,
        None if resume.holds("actor", &track_key) => {
            eprintln!("ltxv replace: the actor's track is reused from {}", actor_dir.display());
            read_track(&actor_dir, w as u32, h as u32, n)?
        }
        None => {
            let unit: Vec<Vec<f32>> = source.frames.iter().map(|px| px.iter().map(|&v| v as f32 / 255.0).collect()).collect();
            crate::sam2_cli::track_frames(&tracker, &unit, w as u32, h as u32, req.prompt_frame, &points, 0)?
                .into_iter()
                .map(|t| if t.object_score > 0.0 { t.logits.iter().map(|&l| l > 0.0).collect() } else { vec![false; w * h] })
                .collect()
        }
    };
    if !resume.holds("actor", &track_key) {
        write_masks(&actor_dir, &actor, w, h, source.fps, &req.input)?;
        if req.actor_mask.is_none() {
            resume.record("actor", &track_key)?;
        }
    }

    // The edited stills: frame 0 (the clip's own first frame) and the guides.
    let masks_gpu = gpu_core::Gpu::new(imaging::PIPELINES);
    let ctx = Ctx::new(&masks_gpu);
    let identity = Identity::load(&req, &tracker.gpu)?;
    let editor = Editor { req: &req, resume: &resume, tracker: &tracker, identity, ctx: &ctx, w, h };
    let mut stills: Vec<Still> = Vec::new();
    for &k in std::iter::once(&0).chain(&guides) {
        if k > 0 && !actor[k].contains(&true) {
            eprintln!("ltxv replace: frame {k} dropped as a guide: the actor is not visible on it");
            continue;
        }
        let still = editor.best_still(k, &source.frames[k], &actor[k])?;
        let iou = still.iou;
        let (attempts, min) = (req.tuning.edit_attempts, req.tuning.min_overlap);
        if k == 0 || iou >= min {
            eprintln!("ltxv replace: frame {k} edited, figure/actor overlap {iou:.2}");
            stills.push(still);
        } else {
            eprintln!("ltxv replace: frame {k} dropped as a guide: the best of {attempts} edits overlaps the actor only {iou:.2} (< {min})");
        }
    }
    if stills[0].iou < req.tuning.min_overlap {
        return Err(format!(
            "frame 0: the best of {} edits overlaps the actor only {:.2} (< {}); it is the clip's first frame and cannot be dropped - see {} and try another --seed, --edit-prompt or --min-overlap",
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
    let regions = ltxv::maskcond::snap_to_latent_tokens(&regions, w, h, grid);
    let region_dir = req.work.join("region");
    write_masks(&region_dir, &regions, w, h, source.fps, &req.input)?;

    let mut spliced = source;
    spliced.frames[0] = stills[0].image.px.clone();
    // The clip the video pass sees, kept so the pass can be run again alone
    // (`ltxv v2v --input <work>/input.mp4 --mask <work>/region ...`).
    crate::ltxv_cli::write_clip(spliced.clone(), &req.work.join("input.mp4").to_string_lossy())?;
    let generated_path = req.work.join("generated.mp4");
    let video_key = serde_json::json!({
        "input": stamp(&req.input), "regions": digest(regions.iter().map(|r| digest_mask(r)).flat_map(u64::to_le_bytes)),
        "stills": stills.iter().map(|s| (s.frame, digest(s.image.px.iter().copied()))).collect::<Vec<_>>(),
        "prompt": req.prompt, "seed": req.seed, "steps": req.steps, "dit_config": req.dit_config, "guide_strength": req.tuning.guide_strength,
        "adapters": req.adapters.iter().map(|a| serde_json::json!({"path": a.path, "scale": a.scale})).collect::<Vec<_>>(),
        "video_models": req.ltxv_overrides,
    })
    .to_string();
    let reused = if resume.holds("generated", &video_key) {
        crate::ltxv_cli::read_clip(&generated_path.to_string_lossy(), source_fps).ok().filter(|v| v.frames.len() == n)
    } else {
        None
    };
    let generated = match reused {
        Some(video) => {
            eprintln!("ltxv replace: the video pass is reused from {}", generated_path.display());
            video
        }
        None => {
            let video = generate(&req, &spliced, &region_dir, &stills[1..])?;
            crate::ltxv_cli::write_clip(video.clone(), &generated_path.to_string_lossy())?;
            resume.record("generated", &video_key)?;
            video
        }
    };

    // Outside the region every pixel is the source's - the still's, on frame 0.
    let mut out = spliced;
    out.audio = match &req.soundtrack {
        Some(path) => Some(soundtrack(path, n, out.fps)?),
        None => {
            eprintln!("ltxv replace: the result is silent - brain reads no sound out of {}; give the sound as --soundtrack <wav> to carry it", req.input);
            None
        }
    };
    for ((frame, region), generated) in out.frames.iter_mut().zip(&regions).zip(&generated.frames) {
        *frame = blend(generated, frame, &feather(&ctx, region, w, h, req.tuning.feather_px));
    }
    eprintln!("ltxv replace: wrote {} in {:.0}s", crate::ltxv_cli::write_clip(out, &req.out)?, t0.elapsed().as_secs_f32());
    crate::run_record::write_run_record(
        &req.out,
        "ltxv replace",
        req.seed,
        serde_json::json!({
            "input": req.input, "character": req.character, "points": points, "prompt_frame": req.prompt_frame,
            "prompt": req.prompt, "edit_prompt": req.edit_prompt, "guide_frames": guides, "work_dir": req.work,
            "adapters": req.adapters.iter().map(|a| serde_json::json!({"path": a.path, "scale": a.scale})).collect::<Vec<_>>(),
            "edit_adapters": req.edit_adapter_flags, "tuning": req.tuning.record(), "conformed": conformed,
        }),
    )
}

/// A clip on LTX-2.5's grid, with its clicks and actor track moved onto it.
struct Conformed {
    clip: ltxv::pipeline::Video,
    points: Vec<(f32, f32, f32)>,
    track: Option<Vec<Vec<bool>>>,
    /// What conforming did, `None` when the clip already fit.
    note: Option<String>,
}

/// Fit `clip` to LTX-2.5's grid: trim it at its end to 1 + 8k frames and
/// centre-crop it to multiples of 32, moving the clicks (`(x, y, label)` in
/// source pixels) and cropping the given actor track with it. Returns what was done, `None` when nothing
/// was; with `allowed` false a clip that needs it is an error instead.
fn conform(clip: ltxv::pipeline::Video, points: &[(f32, f32, f32)], track: Option<Vec<Vec<bool>>>, allowed: bool) -> Result<Conformed, String> {
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
        return Err(format!("{n} frames at {w}x{h} is not 1 + 8k frames at multiples of 32, and --no-conform refuses to trim it to {n2} frames at {w2}x{h2}"));
    }
    let moved: Vec<(f32, f32, f32)> = points.iter().map(|&(x, y, l)| (x - x0 as f32, y - y0 as f32, l)).collect();
    if let Some(&(x, y, _)) = points.iter().zip(&moved).find(|(_, m)| !(0.0..w2 as f32).contains(&m.0) || !(0.0..h2 as f32).contains(&m.1)).map(|(p, _)| p) {
        return Err(format!("the click ({x}, {y}) falls outside the centre crop to {w2}x{h2}: click inside it"));
    }
    let crop = Box2 { x: x0, y: y0, w: w2, h: h2 };
    let frames = clip.frames.into_iter().take(n2).map(|f| cut(&f, w, h, crop)).collect();
    let track = track.map(|t| t.into_iter().take(n2).map(|m| (y0..y0 + h2).flat_map(|y| m[y * w + x0..y * w + x0 + w2].to_vec()).collect()).collect());
    Ok(Conformed { clip: ltxv::pipeline::Video { width: w2 as u32, height: h2 as u32, frames, ..clip }, points: moved, track, note: Some(note) })
}

/// An actor track given as a mask sequence, checked against the clip it
/// tracks; a frame the tracker marked occluded is an empty mask, so the
/// actor there is left as it is.
fn read_track(dir: &Path, w: u32, h: u32, n: usize) -> Result<Vec<Vec<bool>>, String> {
    let seq = sam2::MaskSeq::read(dir)?;
    if (seq.width, seq.height, seq.frames.len()) != (w, h, n) {
        return Err(format!("--actor-mask {}: {} frames at {}x{}, the clip has {n} at {w}x{h}", dir.display(), seq.frames.len(), seq.width, seq.height));
    }
    let object = |v: u8| match seq.polarity {
        sam2::Polarity::ObjectWhite => v > 127,
        sam2::Polarity::ObjectBlack => v <= 127,
    };
    seq.frames
        .iter()
        .enumerate()
        .map(|(i, info)| {
            if info.object_score <= 0.0 {
                return Ok(vec![false; (w * h) as usize]);
            }
            let img = imaging::load(seq.frame_path(i))?;
            Ok(img.px.as_chunks::<3>().0.iter().map(|p| object(p[0])).collect())
        })
        .collect()
}

/// The WAV at `path`, cut to `frames` frames at `fps`. A soundtrack shorter
/// than the clip is an error: padding it would put silence under the end of
/// the picture without saying so.
fn soundtrack(path: &str, frames: usize, fps: usize) -> Result<ltxv::audio::AudioClip, String> {
    let mut clip = ltxv::audio::read_wav(path)?;
    let want = (frames as u64 * clip.sample_rate as u64).div_ceil(fps as u64) as usize;
    if clip.samples_per_channel() < want {
        return Err(format!("--soundtrack {path} holds {} samples per channel, the clip needs {want} ({frames} frames at {fps} fps)", clip.samples_per_channel()));
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
        return Err(format!("--guide-frames: frame {bad} is not inside 1..{} (frame 0 is always edited: it is the clip's first frame)", n - 1));
    }
    let mut sorted = asked.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|p| p[0] == p[1]) {
        return Err(format!("--guide-frames names a frame twice: {asked:?}"));
    }
    Ok(sorted)
}

/// A source frame with the character painted in over the actor.
struct Still {
    frame: usize,
    image: Rgb8,
    /// Where the new figure stands on this still.
    figure: Vec<bool>,
    /// How well the figure covers the actor ([`iou`]).
    iou: f32,
}

/// ArcFace and the character sheet's face embedding: how close a painted
/// still's face is to the character's.
struct Identity {
    session: arcface::caps::ArcFaceSession,
    sheet: Vec<f32>,
}

impl Identity {
    /// The identity model from the store (or `--component arcface.weights`)
    /// and the sheet's face. `None` - said, not silent - when the store holds
    /// no ArcFace or the sheet shows no face a detector finds (a drawn or
    /// masked character): the stills are then kept by overlap alone.
    fn load(req: &Request, device: &gpu_core::Gpu) -> Result<Option<Identity>, String> {
        if req.no_identity {
            return Ok(None);
        }
        let spec = arcface::spec::ArcFaceSpec;
        let assembly = match crate::resolver_cli::try_resolve(loader::model_dir::resolve(None).as_deref(), "arcface", &spec, &req.arcface_overrides) {
            Ok(a) => a,
            Err(loader::resolver::ResolveFailure::Missing(_)) => {
                eprintln!("ltxv replace: no ArcFace in the model store - stills are kept by overlap alone, not by how close their face is to the sheet's (brain pull https://huggingface.co/DIAMONIK7777/antelopev2/blob/main/glintr100.onnx and .../scrfd_10g_bnkps.onnx to compare faces)");
                return Ok(None);
            }
            Err(e) => return Err(format!("the identity model: {}", e.message())),
        };
        let weights = assembly.role_path("weights")?;
        let dir = Path::new(&weights).parent().ok_or_else(|| format!("{weights} has no directory"))?.to_string_lossy().into_owned();
        let session = arcface::caps::ArcFaceSession::load(&dir, device.new_like(&arcface::caps::SERVING_PIPELINES))?;
        let sheet = imaging::load(&req.character)?;
        let chw = imaging::pixels::hwc_to_chw(&sheet.to_hwc_unit(), 3, sheet.h as usize, sheet.w as usize);
        match session.embed_raw_chw(&chw, sheet.w, sheet.h, true, true) {
            Ok((raw, _)) => Ok(Some(Identity { session, sheet: model::hostmath::l2_normalize(&raw) })),
            Err(_) => {
                eprintln!("ltxv replace: no face found on {} - stills are kept by overlap alone", req.character);
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
struct Editor<'a> {
    req: &'a Request,
    resume: &'a Resume,
    tracker: &'a sam2::Sam2,
    /// The face the stills are matched against, when there is one to match.
    identity: Option<Identity>,
    ctx: &'a Ctx<'a>,
    w: usize,
    h: usize,
}

impl Editor<'_> {
    /// Up to `--edit-attempts` edits of frame `k`, each with its own seed,
    /// kept by [`pick_still`]: with an identity model, every attempt is made
    /// and the covering edit whose face is closest to the sheet's wins;
    /// without one, the first covering edit. The winner is `still_<k>.png`.
    fn best_still(&self, k: usize, src: &[u8], actor: &[bool]) -> Result<Still, String> {
        let (step, key) = (format!("still_{k}"), self.still_key(k, src, actor));
        if self.resume.holds(&step, &key) {
            if let Ok(still) = self.load_still(k, actor) {
                eprintln!("ltxv replace: frame {k} reused from {}, figure/actor overlap {:.2}", self.req.work.display(), still.iou);
                return Ok(still);
            }
        }
        let t = &self.req.tuning;
        let mut stills: Vec<(Still, Option<f32>)> = Vec::new();
        for attempt in 0..t.edit_attempts {
            let seed = self.req.seed.wrapping_add(k as u64 * t.edit_attempts + attempt);
            let still = self.edit(k, src, actor, seed)?;
            let face = self.identity.as_ref().map(|id| id.similarity(&still)).transpose()?.flatten();
            let face_note = match (&self.identity, face) {
                (None, _) => String::new(),
                (Some(_), Some(f)) => format!(", face/sheet similarity {f:.2}"),
                (Some(_), None) => ", no face found".to_string(),
            };
            eprintln!("ltxv replace: frame {k} edit {} (seed {seed}): figure/actor overlap {:.2}{face_note}", attempt + 1, still.iou);
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
    /// the track was made.
    fn still_key(&self, k: usize, src: &[u8], actor: &[bool]) -> String {
        let r = self.req;
        serde_json::json!({
            "frame": k, "source": digest(src.iter().copied()), "actor": digest_mask(actor), "character": stamp(&r.character),
            "edit_prompt": r.edit_prompt, "seed": r.seed, "editor": r.flux2_roles, "edit_adapters": r.edit_adapter_flags,
            "no_identity": r.no_identity, "identity": r.arcface_overrides, "tuning": r.tuning.edit_record(),
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

    fn edit(&self, k: usize, src: &[u8], actor: &[bool], seed: u64) -> Result<Still, String> {
        let (w, h, ctx, work) = (self.w, self.h, self.ctx, &self.req.work);
        let bbox = bounding_box(actor, w, h).ok_or_else(|| format!("the actor is not visible on frame {k}: nothing to paint the character over (click them on a frame they are in: --prompt-frame)"))?;
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
        let canvas_path = work.join(format!("canvas_{k}.png"));
        imaging::save_png(&canvas_path, &Rgb8::new(ew, eh, resample(ctx, &cut(src, w, h, canvas), cw, ch, ew as usize, eh as usize, Filter::Bilinear))?)?;
        let repaint_px: Vec<u8> = repaint.iter().flat_map(|&r| [if r { 255 } else { 0 }; 3]).collect();
        let repaint_path = work.join(format!("repaint_{k}.png"));
        imaging::save_png(&repaint_path, &Rgb8::new(ew, eh, resample(ctx, &repaint_px, cw, ch, ew as usize, eh as usize, Filter::Nearest))?)?;
        let edit_path = work.join(format!("edit_{k}_{seed}.png"));
        let mut argv: Vec<String> = [self.req.flux2_roles.clone(), self.req.edit_adapter_flags.clone()].concat();
        argv.extend(
            [
                "--ref", &canvas_path.to_string_lossy(), "--ref", &self.req.character, "--ref-size", &self.req.tuning.edit_long_side.to_string(),
                "--mask", &repaint_path.to_string_lossy(), "--width", &ew.to_string(), "--height", &eh.to_string(),
                "--seed", &seed.to_string(), "--prompt", &self.req.edit_prompt, "--out", &edit_path.to_string_lossy(),
            ]
            .map(String::from),
        );
        crate::flux2_cli::generate(&argv).map_err(|e| format!("editing frame {k}: {e}"))?;
        let edited = fit(ctx, &imaging::load(&edit_path)?, cw, ch);
        let mut painted = src.to_vec();
        paste(&mut painted, w, Box2 { h: visible, ..canvas }, &edited[..visible * cw * 3]);

        // The new figure: SAM 2 clicked at spots spread over the actor's
        // silhouette - the character was asked to stand there. A click that
        // lands on background picks the street, not the figure, so a pick
        // much larger than the actor is dropped.
        let painted_unit: Vec<f32> = painted.iter().map(|&v| v as f32 / 255.0).collect();
        let actor_px = count(actor);
        let mut figure = vec![false; w * h];
        for p in spread_points(actor, w, h) {
            let t = crate::sam2_cli::track_frames(self.tracker, std::slice::from_ref(&painted_unit), w as u32, h as u32, 0, &[(p.0, p.1, 1.0)], 0)?;
            let pick: Vec<bool> = t[0].logits.iter().map(|&l| l > 0.0).collect();
            if t[0].object_score > 0.0 && count(&pick) <= 2 * actor_px {
                figure.iter_mut().zip(&pick).for_each(|(f, &p)| *f |= p);
            }
        }
        let joined: Vec<bool> = actor.iter().zip(&figure).map(|(&a, &f)| a || f).collect();
        let alpha = feather(ctx, &grow(ctx, &joined, w, h, self.req.tuning.paste_grow_px), w, h, self.req.tuning.feather_px);
        let image = Rgb8::new(w as u32, h as u32, blend(&painted, src, &alpha))?;
        Ok(Still { frame: k, image, iou: iou(&figure, actor), figure })
    }
}

/// Which of a frame's edits to keep, given each one's `(overlap with the
/// actor, face similarity to the sheet)`: among those that cover the actor
/// (`min_overlap`), the closest face, a measured face before none, then the
/// best overlap; when none covers the actor, the best overlap.
fn pick_still(candidates: &[(f32, Option<f32>)], min_overlap: f32) -> usize {
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
fn iou(a: &[bool], b: &[bool]) -> f32 {
    let (inter, union) = a.iter().zip(b).fold((0usize, 0usize), |(i, u), (&x, &y)| (i + (x && y) as usize, u + (x || y) as usize));
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}

fn generate(req: &Request, clip: &ltxv::pipeline::Video, region: &Path, guides: &[Still]) -> Result<ltxv::pipeline::Video, String> {
    let mut o = ltxv::pipeline::MaskedOpts { mask_dir: region.to_string_lossy().into_owned(), ..Default::default() };
    o.base.seed = req.seed;
    o.base.fps = clip.fps;
    o.base.device = req.device.clone();
    o.base.conditioning_strength = req.tuning.guide_strength;
    o.base.adapters = req.adapters.clone();
    o.base.future_anchors = guides.iter().map(|s| (s.frame, req.work.join(format!("still_{}.png", s.frame)).to_string_lossy().into_owned())).collect();
    if let Some(d) = &req.dit_config {
        o.base.dit_config = d.clone();
    }
    let paths = crate::ltxv_cli::resolve_ltxv(&req.ltxv_overrides, None)?;
    ltxv::pipeline::resolve_defaults(&mut o.base, &paths, ltxv::pipeline::Named { dit_config: req.dit_config.is_some(), frames: true, size: true, fps: true });
    if let Some(s) = req.steps {
        o.base.steps = s;
    }
    let cancel = capability::CancelToken::default();
    let (video, _) = ltxv::pipeline::generate_masked(&paths, &req.prompt, clip, &o, &cancel, |done, total, phase| {
        eprint!("\rltxv replace: video [{done}/{total}] {phase}                    ");
    })?;
    eprintln!();
    Ok(video)
}

/// A mask sequence on disk, object = 255, every frame seen.
fn write_masks(dir: &Path, masks: &[Vec<bool>], w: usize, h: usize, fps: usize, source: &str) -> Result<(), String> {
    let mut seq = sam2::MaskSeq::new(dir, w as u32, h as u32, fps as f64, sam2::Polarity::ObjectWhite, 0, (source.to_string(), masks.len()), 0, Vec::new());
    for (f, m) in masks.iter().enumerate() {
        let logits: Vec<f32> = m.iter().map(|&r| if r { 1.0 } else { -1.0 }).collect();
        seq.write_frame(f, &logits, 1.0, 1.0)?;
    }
    seq.write_manifest().map(|_| ())
}

fn count(m: &[bool]) -> usize {
    m.iter().filter(|&&v| v).count()
}

/// An axis-aligned pixel box.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Box2 {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

fn bounding_box(m: &[bool], w: usize, h: usize) -> Option<Box2> {
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for (i, _) in m.iter().enumerate().filter(|(_, &v)| v) {
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    (x1 > x0).then_some(Box2 { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
}

/// How many rows of body the bottom edge of the frame cuts off `actor`
/// (boxed by `b`): a standing person's silhouette is about `ratio` times as
/// tall as its median row width, so what that height exceeds the visible
/// one by is below the frame - at most one more actor height, and nothing
/// when the actor does not reach the edge or stands whole on it.
fn body_below_frame(actor: &[bool], b: Box2, w: usize, h: usize, ratio: f32) -> usize {
    if b.y + b.h < h {
        return 0;
    }
    let mut widths: Vec<usize> = (b.y..b.y + b.h).map(|y| count(&actor[y * w..(y + 1) * w])).collect();
    widths.sort_unstable();
    let expected = (ratio * widths[widths.len() / 2] as f32).round() as usize;
    expected.saturating_sub(b.h).min(b.h)
}

/// The canvas an edit works on: the actor's box with an eighth of its height
/// of margin above and below, at least 3:4 wide so a portrait figure has
/// room, inside the frame sideways. An actor the bottom edge cuts off gets
/// `below` more rows of canvas BELOW the frame ([`body_below_frame`]), room
/// for the rest of a body at the actor's own scale; those rows are only
/// ever repainted.
fn canvas_around(actor: Box2, below: usize, w: usize, h: usize) -> Box2 {
    let margin = actor.h / 8;
    let y = actor.y.saturating_sub(margin);
    let cut_off = below > 0;
    let ch = if cut_off { h - y + below } else { (actor.h + 2 * margin).min(h) };
    let y = if cut_off { y } else { y.min(h - ch) };
    let cw = (actor.w + actor.w / 5).max((ch * 3).div_ceil(4)).min(w);
    Box2 { x: (actor.x + actor.w / 2).saturating_sub(cw / 2).min(w - cw), y, w: cw, h: ch }
}

/// The edit's render size: the canvas's aspect with the long side at
/// `long_side`, both sides multiples of 16 (FLUX.2's patch grid).
fn edit_size(cw: usize, ch: usize, long_side: u32) -> (u32, u32) {
    let long = cw.max(ch) as f32;
    let side = |s: usize| ((s as f32 * long_side as f32 / long / 16.0).round().max(1.0) * 16.0) as u32;
    (side(cw), side(ch))
}

/// The pixels of `b`; rows below the `h`-row frame are mid grey.
fn cut(px: &[u8], w: usize, h: usize, b: Box2) -> Vec<u8> {
    (b.y..b.y + b.h).flat_map(|y| if y < h { px[(y * w + b.x) * 3..(y * w + b.x + b.w) * 3].to_vec() } else { vec![128; b.w * 3] }).collect()
}

fn paste(px: &mut [u8], w: usize, b: Box2, tile: &[u8]) {
    for (row, y) in (b.y..b.y + b.h).enumerate() {
        px[(y * w + b.x) * 3..(y * w + b.x + b.w) * 3].copy_from_slice(&tile[row * b.w * 3..(row + 1) * b.w * 3]);
    }
}

/// Ten clicks spread over a silhouette: two per row at five heights, 30 % and
/// 70 % across the row, so head, torso and legs are all asked about.
fn spread_points(m: &[bool], w: usize, h: usize) -> Vec<(f32, f32)> {
    let rows: Vec<usize> = (0..h).filter(|&y| m[y * w..(y + 1) * w].contains(&true)).collect();
    if rows.is_empty() {
        return Vec::new();
    }
    let mut points = Vec::new();
    for q in [0.15, 0.35, 0.55, 0.75, 0.9] {
        let y = rows[((rows.len() - 1) as f32 * q).round() as usize];
        let row = &m[y * w..(y + 1) * w];
        let (x0, x1) = (row.iter().position(|&v| v).unwrap_or(0), row.iter().rposition(|&v| v).unwrap_or(0));
        for f in [0.3, 0.7] {
            points.push((x0 as f32 + f * (x1 - x0) as f32, y as f32));
        }
    }
    points
}

/// Carry `figure`, drawn where the actor stood as `from`, to where the actor
/// stands as `to`: shifted with the actor's centroid and scaled about it by
/// the square root of the actor's change in area. An actor not visible on
/// either frame carries nothing.
fn carry(figure: &[bool], from: &[bool], to: &[bool], w: usize, h: usize) -> Vec<bool> {
    let (Some((fx, fy, fa)), Some((tx, ty, ta))) = (centroid(from, w), centroid(to, w)) else {
        return vec![false; w * h];
    };
    let s = (ta / fa).sqrt();
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            // Inverse map: where on `figure` this pixel comes from.
            let (sx, sy) = (fx + (x as f32 - tx) / s, fy + (y as f32 - ty) / s);
            let (ix, iy) = (sx.round() as isize, sy.round() as isize);
            if (0..w as isize).contains(&ix) && (0..h as isize).contains(&iy) {
                out[y * w + x] = figure[iy as usize * w + ix as usize];
            }
        }
    }
    out
}

fn centroid(m: &[bool], w: usize) -> Option<(f32, f32, f32)> {
    let (mut sx, mut sy, mut n) = (0f64, 0f64, 0f64);
    for (i, _) in m.iter().enumerate().filter(|(_, &v)| v) {
        sx += (i % w) as f64;
        sy += (i / w) as f64;
        n += 1.0;
    }
    (n > 0.0).then(|| ((sx / n) as f32, (sy / n) as f32, n as f32))
}

fn to_plane(ctx: &Ctx, m: &[bool]) -> gpu_core::DeviceBuffer {
    ctx.upload("ltxv.replace.mask", &m.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect::<Vec<f32>>())
}

fn shrink(ctx: &Ctx, m: &[bool], w: usize, h: usize, radius: u32) -> Vec<bool> {
    let s = Shape::new(1, 1, h as u32, w as u32);
    let shrunk = imaging::mask::erode(ctx, &to_plane(ctx, m), s, radius);
    ctx.download(&shrunk, s.numel()).iter().map(|&v| v > 0.5).collect()
}

fn grow(ctx: &Ctx, m: &[bool], w: usize, h: usize, radius: u32) -> Vec<bool> {
    let s = Shape::new(1, 1, h as u32, w as u32);
    let grown = imaging::mask::dilate(ctx, &to_plane(ctx, m), s, radius);
    ctx.download(&grown, s.numel()).iter().map(|&v| v > 0.5).collect()
}

/// A `[0, 1]` alpha: `m` with its edge ramped over `radius` pixels.
fn feather(ctx: &Ctx, m: &[bool], w: usize, h: usize, radius: u32) -> Vec<f32> {
    let s = Shape::new(1, 1, h as u32, w as u32);
    let soft = imaging::mask::feather(ctx, &to_plane(ctx, m), s, radius);
    ctx.download(&soft, s.numel())
}

/// Resample interleaved RGB8 `px` (`w x h`) to `tw x th`.
fn resample(ctx: &Ctx, px: &[u8], w: usize, h: usize, tw: usize, th: usize, filter: Filter) -> Vec<u8> {
    let unit: Vec<f32> = px.iter().map(|&v| v as f32 / 255.0).collect();
    let chw = imaging::pixels::hwc_to_chw(&unit, 3, h, w);
    let (out, os) = ctx.resize(&ctx.upload("ltxv.replace.canvas", &chw), Shape::new(1, 3, h as u32, w as u32), th as u32, tw as u32, filter, AlignCorners::HalfPixel);
    interleave(&ctx.download(&out, os.numel()), tw * th)
}

/// Planar `[3, hw]` values in `[0, 1]` to interleaved RGB8.
fn interleave(plane: &[f32], hw: usize) -> Vec<u8> {
    (0..hw).flat_map(|i| (0..3).map(move |c| i + c * hw)).map(|j| (plane[j] * 255.0).round().clamp(0.0, 255.0) as u8).collect()
}

/// `img` at `tw x th`: area-averaged when it shrinks, so the downscale does
/// not alias, bilinear when it grows (a canvas taller than the edit's render).
fn fit(ctx: &Ctx, img: &Rgb8, tw: usize, th: usize) -> Vec<u8> {
    let (w, h) = (img.w as usize, img.h as usize);
    if tw > w || th > h {
        return resample(ctx, &img.px, w, h, tw, th, Filter::Bilinear);
    }
    let s = Shape::new(1, 3, img.h, img.w);
    let chw = imaging::pixels::hwc_to_chw(&img.to_hwc_unit(), 3, h, w);
    let (small, ss) = imaging::mask::downsample(ctx, &ctx.upload("ltxv.replace.edit", &chw), s, th as u32, tw as u32);
    interleave(&ctx.download(&small, ss.numel()), tw * th)
}

/// `alpha * over + (1 - alpha) * under`, per pixel.
fn blend(over: &[u8], under: &[u8], alpha: &[f32]) -> Vec<u8> {
    over.as_chunks::<3>()
        .0
        .iter()
        .zip(under.as_chunks::<3>().0)
        .zip(alpha)
        .flat_map(|((o, u), &a)| (0..3).map(move |c| (a * o[c] as f32 + (1.0 - a) * u[c] as f32).round() as u8))
        .collect()
}

#[cfg(test)]
mod tests {
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

    /// A component names a declared role of a known model; anything else is
    /// an error, never a flag nobody reads.
    #[test]
    fn a_component_names_a_declared_role_of_a_known_model() {
        assert_eq!(parse_component("flux2.text-encoder=/m/te").unwrap(), (Component::Editor, "text_encoder".into(), "/m/te".into()));
        assert_eq!(parse_component("sam2.variant=large").unwrap(), (Component::Tracker, "variant".into(), "large".into()));
        for bad in ["flux2.ditt=/m", "wan.dit=/m", "flux2.dit", "flux2.dit=", "sam2"] {
            assert!(parse_component(bad).is_err(), "{bad}");
        }
    }

    /// A clip LTX-2.5 cannot take is trimmed at its end to 1 + 8k frames and
    /// centre-cropped to multiples of 32, the clicks moving with the crop;
    /// one it can take is untouched, and `--no-conform` refuses instead.
    #[test]
    fn a_clip_is_conformed_to_the_latent_grid_with_its_clicks() {
        let clip = |w: u32, h: u32, n: usize| ltxv::pipeline::Video {
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
        assert!(conform(clip(100, 70, 20), &[(50.0, 40.0, 1.0)], None, false).err().is_some(), "--no-conform refuses");
        assert!(conform(clip(100, 70, 20), &[(1.0, 1.0, 1.0)], None, true).is_err(), "a click the crop cuts away is an error");
        assert!(conform(clip(64, 64, 8), &[(1.0, 1.0, 1.0)], None, true).is_err(), "fewer than 9 frames");
    }

    /// Every flag the parser takes is in --help, and every flag --help
    /// names is parsed.
    #[test]
    fn every_flag_is_documented() {
        let src = include_str!("ltxv_replace.rs");
        let parsed = [
            "--input", "--character", "--prompt", "--output-path", "--point", "--exclude", "--guide-frames", "--component", "--work-dir",
            "--prompt-frame", "--edit-prompt", "--dit-config", "--steps", "--device", "--guide-strength", "--guide-every", "--edit-attempts",
            "--min-overlap", "--edit-size", "--repaint-grow", "--body-ratio", "--paste-grow", "--region-grow", "--frame0-inset", "--feather", "--no-conform",
            "--fps", "--adapter", "--lora-scale", "--edit-adapter", "--edit-lora-scale", "--actor-mask", "--soundtrack", "--no-identity", "--fresh",
        ];
        for flag in parsed {
            assert!(HELP.contains(flag), "{flag} is parsed but not in --help");
            assert!(src.contains(&format!("\"{flag}\"")), "{flag} is in --help but not parsed");
        }
    }

    /// A track written to the work directory reads back as it was: it is
    /// what a resumed run takes in place of tracking again.
    #[test]
    fn a_written_track_reads_back_unchanged() {
        let (w, h, n) = (32usize, 32usize, 3usize);
        let track: Vec<Vec<bool>> = (0..n).map(|f| (0..w * h).map(|i| i % w >= 4 + f && i / w >= 8 && i / w < 20).collect()).collect();
        let dir = std::env::temp_dir().join(format!("brain-replace-track-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_masks(&dir, &track, w, h, 24, "clip.mp4").unwrap();
        let back = read_track(&dir, w as u32, h as u32, n);
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
}
