// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain sam2 track` - clicks on one frame in, a per-frame mask sequence
//! out.
//!
//! ```text
//! brain sam2 track --video clip.mp4 --point 640,300 --out masks/
//! ```
//!
//! Every OTHER `sam2` verb (`segment`) still goes through the generic
//! capability dispatch: this module forwards anything it does not recognise
//! to `crate::resolver_cli::run_generic_migrated`, so adding a dedicated
//! handler did not take the image path off its shared code path - and `track`
//! (below) resolves its own `weights`/`variant` through the SAME
//! `sam2::spec::Sam2Spec` resolver `segment` does, so both entry points agree
//! on which checkpoint a given models-directory state means.
//!
//! # Why this is not a `capability::Action`
//!
//! `segment` is one image in, one mask blob out, which is exactly the shape the
//! capability wire format has. Tracking is a whole clip in and a DIRECTORY of
//! per-frame PNGs plus a manifest out - see `sam2::maskseq`, which defines that
//! format and is what a downstream consumer reads. Squeezing a mask sequence
//! through a single blob would either lose the per-frame confidence or invent a
//! container, so the verb is a CLI command over the same `sam2::Tracker` API a
//! future capability would call.
//!
//! # Polarity
//!
//! What comes out is SAM 2's own meaning: **white (255) is the tracked
//! object**. `--invert` writes the inverse instead, for a consumer whose `1`
//! means "keep this region, do not regenerate" - LTX-2.5's masked conditioning
//! is the case in hand, and inverting there means masking the BACKGROUND white
//! to replace a character. Either way `masks.json` states which one is on disk,
//! and `sam2::MaskSeq::read` refuses to guess if it does not.

use std::path::{Path, PathBuf};

use imaging::{AlignCorners, Ctx, Filter, Shape};
use sam2::{MaskSeq, Polarity, Prompt, Sam2, Scope, Tracker};

use crate::args::Args;

const HELP: &str = "\
brain sam2 track - propagate a point prompt through a video (SAM 2.1 memory bank)

USAGE:
  brain sam2 track --video <clip> --point <x,y> --out <dir> [options]
  brain sam2 track --frames <dir> --point <x,y> --out <dir> [options]

  --video <path>        source clip, decoded at its native rate
  --frames <dir>        a directory of numbered image files, instead of --video
  --point <x,y>         a click on the object, in SOURCE pixels (repeatable:
                        several clicks pick one object together, for a person
                        whose coat and trousers SAM 2 would otherwise take apart)
  --exclude <x,y>       a click on something that is NOT the object (repeatable)
  --prompt-frame <n>    frame the clicks sit on (default 0)
  --out <dir>           mask-sequence directory (masks.json + mask_%06d.png)

  --weights <path>      state the checkpoint outright, as the models directory
                        scan already found it (default: resolved from the
                        models directory - --brain-data-dir / BRAIN_MODELS_DIR)
  --variant <tiny|large>  state the checkpoint's own size outright; checked
                        against its real trunk width, not trusted blindly
                        (default: derived from that width)
  --max-frames <n>      stop after n frames (default: the whole clip)
  --fps <f>             resample the clip to this rate before tracking; with
                        --frames, the rate the frames were taken at (required)
  --object-id <n>       recorded in masks.json for multi-character work
  --invert              emit object=0 (LTX masked-conditioning polarity)
  --soft                emit the sigmoid ramp instead of a hard 0/255 mask

Masks come out at the SOURCE frame size, one PNG per source frame, contiguous
from 0. Any downsampling is the consumer's, against its own grid.
";

pub fn run_sam2(args: &[String]) {
    let verb = args.first().map(String::as_str).unwrap_or("");
    if verb != "track" {
        // Everything else is the image path, reached generically - through
        // the SAME resolver `track` below uses, so both entry points agree
        // on which checkpoint a given models-directory state means.
        std::process::exit(
            crate::resolver_cli::run_generic_migrated("sam2", sam2::caps::MODEL, args).expect("sam2 is a resolver-migrated architecture"),
        );
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        return;
    }
    if let Err(e) = track(&args[1..]) {
        eprintln!("brain sam2 track: {e}");
        std::process::exit(1);
    }
}

/// One `(x, y)` click in source pixels.
pub(crate) fn parse_point(s: &str) -> Result<(f32, f32), String> {
    let v: Vec<f32> = s.split(',').map(|p| p.trim().parse::<f32>()).collect::<Result<_, _>>().map_err(|e| format!("--point wants 'x,y': {e}"))?;
    match v[..] {
        [x, y] => Ok((x, y)),
        _ => Err(format!("--point wants 'x,y', got {} numbers", v.len())),
    }
}

/// A decoded source clip.
///
/// Named fields rather than a tuple on purpose: `width` and `height` are two
/// `u32`s that travel together into every resize and into `masks.json`, and a
/// tuple lets a caller swap them silently - a mask sequence emitted at the
/// transposed resolution is wrong in a way nothing downstream can detect.
struct Clip {
    /// One interleaved `h * w * 3` RGB frame in `[0, 1]` per source frame, in
    /// order - the form every model preprocessor starts from.
    frames: Vec<Vec<f32>>,
    width: u32,
    height: u32,
    /// What the sequence records as its source, so a consumer can check the
    /// masks against the clip it is conditioning.
    name: String,
    fps: f64,
}

impl Clip {
    /// Decode from a video file, or read a directory of numbered images.
    ///
    /// Frames that disagree in size are an error here rather than a surprise
    /// three resizes later.
    fn load(video: Option<&str>, dir: Option<&str>, fps: Option<f64>, max: u32) -> Result<Clip, String> {
        let (raw, name, rate) = match (video, dir) {
            (Some(v), _) => {
                let p = Path::new(v);
                let rate = fps.or_else(|| imaging::video::probe_fps(p)).ok_or_else(|| format!("{v}: its frame rate cannot be read from the container; give it with --fps"))?;
                let opts = imaging::video::VideoDecodeOpts { fps, max_frames: max, spread: 0 };
                let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| v.to_string());
                (imaging::video::decode_frames(p, &opts)?, name, rate)
            }
            (None, Some(d)) => {
                let mut paths: Vec<PathBuf> = std::fs::read_dir(d)
                    .map_err(|e| format!("{d}: {e}"))?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension().is_some_and(|x| ["ppm", "png", "jpg", "jpeg"].contains(&x.to_string_lossy().to_lowercase().as_str()))
                    })
                    .collect();
                paths.sort();
                if paths.is_empty() {
                    return Err(format!("{d} holds no .ppm/.png/.jpg frames"));
                }
                if max > 0 {
                    paths.truncate(max as usize);
                }
                let f = paths
                    .iter()
                    .map(|p| imaging::load(p).map(|img| (img.to_hwc_unit(), img.w, img.h)))
                    .collect::<Result<Vec<_>, String>>()?;
                // The BASENAME, like the `--video` arm: `masks.json` travels
                // with the sequence, and a full machine path in it is both
                // noise and something a shared artifact should not carry.
                let name = Path::new(d).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| d.to_string());
                (f, name, fps.ok_or("--frames has no frame rate of its own: give it with --fps")?)
            }
            (None, None) => return Err("one of --video or --frames is required".into()),
        };
        let (width, height) = (raw[0].1, raw[0].2);
        if let Some(i) = raw.iter().position(|f| (f.1, f.2) != (width, height)) {
            return Err(format!("frame {i} is {}x{}, but frame 0 is {width}x{height}", raw[i].1, raw[i].2));
        }
        Ok(Clip { frames: raw.into_iter().map(|f| f.0).collect(), width, height, name, fps: rate })
    }

    fn len(&self) -> usize {
        self.frames.len()
    }
}

fn track(args: &[String]) -> Result<(), String> {
    // The same resolver `segment` (the ARCH_TO_MODEL/generic path, above)
    // uses: an explicit `--weights`/`--variant` is a stated claim, checked
    // against the checkpoint's own real trunk width rather than trusted
    // outright (`sam2::spec::Sam2Spec::assemble`) - the derived variant wins
    // when neither is given.
    let spec = sam2::spec::Sam2Spec;
    let (mut overrides, remaining) = crate::resolver_cli::extract_role_overrides(&spec, args);
    let mut a = Args::new(&remaining);
    let video = a.take_str("--video");
    let frames_dir = a.take_str("--frames");
    let mut points = Vec::new();
    while let Some(p) = a.take_str("--point") {
        let (x, y) = parse_point(&p)?;
        points.push((x, y, 1.0));
    }
    while let Some(p) = a.take_str("--exclude") {
        let (x, y) = parse_point(&p)?;
        points.push((x, y, 0.0));
    }
    if !points.iter().any(|p| p.2 == 1.0) {
        return Err("--point <x,y> is required: it is what picks the object".into());
    }
    let prompt_frame = a.usize_or("--prompt-frame", 0);
    let out = a.take_str("--out").ok_or("--out <dir> is required")?;
    let variant_flag = a.take_str("--variant");
    let max_frames = a.u32_or("--max-frames", 0);
    let fps = a.parsed::<f64>("--fps");
    let object_id = a.u32_or("--object-id", 0);
    let invert = a.take_flag("--invert");
    let soft = a.take_flag("--soft");
    a.finish();
    if let Some(v) = variant_flag {
        overrides.insert("variant".to_string(), v);
    }

    let m = load_tracker(&overrides)?;
    let clip = Clip::load(video.as_deref(), frames_dir.as_deref(), fps, max_frames)?;
    let (w, h) = (clip.width, clip.height);
    eprintln!("sam2 track: {} frames at {w}x{h}, {} click(s) on frame {prompt_frame}", clip.len(), points.len());
    let tracked = track_frames(&m, &clip.frames, w, h, prompt_frame, &points, object_id)?;

    let polarity = if invert { Polarity::ObjectBlack } else { Polarity::ObjectWhite };
    let mut seq = MaskSeq::new(&out, w, h, clip.fps, polarity, object_id, (clip.name.clone(), clip.len()), prompt_frame, points.clone());
    seq.binary = !soft;
    for (f, t) in tracked.iter().enumerate() {
        seq.write_frame(f, &t.logits, t.object_score, t.iou)?;
    }
    let manifest = seq.write_manifest()?;
    let occluded = seq.occluded_frames();
    eprintln!(
        "sam2 track: wrote {} masks to {} ({}), manifest {}{}",
        tracked.len(),
        out,
        polarity.tag(),
        manifest.display(),
        if occluded.is_empty() { String::new() } else { format!(", {} occluded frame(s)", occluded.len()) }
    );
    Ok(())
}

/// Load the whole SAM 2.1 checkpoint (memory bank included) the models
/// directory and `overrides` resolve to.
pub(crate) fn load_tracker(overrides: &std::collections::BTreeMap<String, String>) -> Result<Sam2, String> {
    let assembly = crate::resolver_cli::resolve_or_exit("sam2", &sam2::spec::Sam2Spec, overrides);
    let weights = assembly.role_path("weights")?;
    let variant = assembly.variant.clone().ok_or("sam2 track: resolved assembly has no variant")?;
    let cfg = sam2::caps::variant_config(&variant)?;
    // The WHOLE checkpoint: `Scope::Video` skips nothing and errors on an
    // unmatched key in either direction, so a memory bank left at random init
    // is not a reachable state.
    let raw: Vec<(String, Vec<usize>, Vec<f32>)> = if weights.ends_with(".safetensors") {
        checkpoint::safetensors::read(&weights)?.into_iter().map(|t| (t.name, t.shape, t.data)).collect()
    } else {
        checkpoint::torchpt::read(&weights)?.into_iter().map(|t| (t.name, t.shape, t.data)).collect()
    };
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> =
        raw.into_iter().map(|(nm, s, d)| (nm.strip_prefix("model.").unwrap_or(&nm).to_string(), s, d)).collect();
    let (weights, rep) = sam2::import_scoped(tensors, &cfg, Scope::Video)?;
    eprintln!("sam2 track: imported {} of {} checkpoint tensors, {} skipped", rep.imported, rep.source, rep.skipped_video);
    Ok(Sam2::new_video(gpu_core::Gpu::new(sam2::PIPELINES), cfg, &weights))
}

/// One frame's tracking result: mask logits on the source grid (positive is
/// the object), SAM 2's object score (`<= 0` means it believes the object is
/// occluded) and its predicted IoU.
pub(crate) struct Tracked {
    pub logits: Vec<f32>,
    pub object_score: f32,
    pub iou: f32,
}

/// Track the object `points` (`(x, y, label)` in source pixels, label 1 on
/// the object, 0 off it) pick on `prompt_frame` through `frames` (interleaved
/// `h * w * 3` RGB in `[0, 1]`). Every frame gets a result: frames before
/// the prompt frame take its mask, because reverse tracking is not ported.
pub(crate) fn track_frames(m: &Sam2, frames: &[Vec<f32>], w: u32, h: u32, prompt_frame: usize, points: &[(f32, f32, f32)], object_id: u32) -> Result<Vec<Tracked>, String> {
    let n = frames.len();
    if prompt_frame >= n {
        return Err(format!("--prompt-frame {prompt_frame} is past the {n}-frame clip"));
    }
    if let Some(&(x, y, _)) = points.iter().find(|&&(x, y, _)| !(0.0..w as f32).contains(&x) || !(0.0..h as f32).contains(&y)) {
        return Err(format!("the click ({x}, {y}) is outside the {w}x{h} frame"));
    }
    let s = m.cfg.image_size;
    // The reference resizes each frame to `image_size²` without preserving
    // aspect ratio (`Resize((1024, 1024))`), so each axis scales on its own.
    let (sx, sy) = (s as f32 / w as f32, s as f32 / h as f32);
    // The reference asks for the 3-way ambiguity head only for a single
    // click; several clicks already disambiguate (`_use_multimask`).
    let prompt = Prompt {
        coords: points.iter().map(|&(x, y, _)| (x * sx, y * sy)).collect(),
        labels: points.iter().map(|p| p.2).collect(),
        mask_lowres: None,
        multimask_output: points.len() == 1,
    };
    let mut tr = Tracker::new(m, n, object_id);
    let mut out: Vec<Option<Tracked>> = (0..n).map(|_| None).collect();
    // The conditioning frame first: everything after it attends to its memory.
    for f in prompt_frame..n {
        let ctx = Ctx::new(&m.gpu);
        let chw = imaging::pixels::hwc_to_chw(&frames[f], 3, h as usize, w as usize);
        let src = ctx.upload("sam2.track.src", &chw);
        let (dev, shape) = ctx.resize(&src, Shape::new(1, 3, h, w), s, s, Filter::Bilinear, AlignCorners::HalfPixel);
        let resized = ctx.download(&dev, shape.numel());
        let enc = m.encode(&m.preprocess(&resized));
        let step = if f == prompt_frame { tr.prompt(&enc, f, &prompt) } else { tr.track(&enc, f) };

        // The reference resamples the LOW-resolution logits straight to the
        // video grid (`_get_orig_video_res_output` is handed `pred_masks`), not
        // the image-resolution ones - one bilinear step, not two.
        let low = 4 * m.cfg.image_embedding_size();
        let (up, ushape) = ctx.resize(&step.low_res_mask, Shape::new(1, 1, low, low), h, w, Filter::Bilinear, AlignCorners::HalfPixel);
        if step.object_score <= 0.0 {
            eprintln!("sam2 track: frame {f} - object score {:.3} <= 0, SAM 2 believes the object is occluded here", step.object_score);
        }
        out[f] = Some(Tracked { logits: ctx.download(&up, ushape.numel()), object_score: step.object_score, iou: step.iou });
    }
    // Frames before the clicks take their mask, and `masks.json` says so
    // through `prompt.frame`: a mask sequence must cover EVERY source frame
    // or the consumer's frame indices silently shift.
    let seed = out[prompt_frame].as_ref().map(|t| (t.logits.clone(), t.object_score, t.iou)).ok_or("the prompt frame produced no mask")?;
    for slot in &mut out[..prompt_frame] {
        *slot = Some(Tracked { logits: seed.0.clone(), object_score: seed.1, iou: seed.2 });
    }
    out.into_iter().enumerate().map(|(f, t)| t.ok_or_else(|| format!("frame {f} produced no mask"))).collect()
}
