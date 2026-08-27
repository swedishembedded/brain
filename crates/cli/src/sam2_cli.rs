// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain sam2 track` - one point on one frame in, a per-frame mask sequence
//! out.
//!
//! ```text
//! brain sam2 track --video clip.mp4 --point 640,300 --out masks/
//! ```
//!
//! Every OTHER `sam2` verb (`segment`) still goes through the generic
//! capability dispatch, unchanged: this module forwards anything it does not
//! recognise to `caps_cli::run_do`, so adding a dedicated handler did not take
//! the image path off its shared code path.
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

  --video <path>        source clip; decoded with ffmpeg at its native rate
  --frames <dir>        a directory of numbered image files, instead of --video
  --point <x,y>         the click that picks the object, in SOURCE pixels
  --label <0|1>         1 = foreground (default), 0 = background
  --prompt-frame <n>    frame the click sits on (default 0)
  --out <dir>           mask-sequence directory (masks.json + mask_%06d.png)

  --weights <path>      sam2.1_hiera_*.pt (default: $BRAIN_SAM2_WEIGHTS)
  --variant <tiny|large>  checkpoint variant (default tiny)
  --max-frames <n>      stop after n frames (default: the whole clip)
  --fps <f>             resample the clip to this rate before tracking
  --object-id <n>       recorded in masks.json for multi-character work
  --invert              emit object=0 (LTX masked-conditioning polarity)
  --soft                emit the sigmoid ramp instead of a hard 0/255 mask

Masks come out at the SOURCE frame size, one PNG per source frame, contiguous
from 0. Any downsampling is the consumer's, against its own grid.
";

pub fn run_sam2(args: &[String]) {
    let verb = args.first().map(String::as_str).unwrap_or("");
    if verb != "track" {
        // Everything else is the image path, reached generically.
        let mut do_args = vec![sam2::caps::MODEL.to_string()];
        do_args.extend_from_slice(args);
        std::process::exit(crate::caps_cli::run_do(&do_args));
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
fn parse_point(s: &str) -> Result<(f32, f32), String> {
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
                let rate = fps.or_else(|| imaging::video::probe_fps(p)).unwrap_or(24.0);
                let opts = imaging::video::VideoDecodeOpts { fps, max_frames: max };
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
                (f, name, fps.unwrap_or(24.0))
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
    let mut a = Args::new(args);
    let video = a.take_str("--video");
    let frames_dir = a.take_str("--frames");
    let point = a.take_str("--point").ok_or("--point <x,y> is required: it is what picks the object")?;
    let label = a.f32_or("--label", 1.0);
    let prompt_frame = a.usize_or("--prompt-frame", 0);
    let out = a.take_str("--out").ok_or("--out <dir> is required")?;
    let variant = a.str_or("--variant", "tiny");
    let weights = a
        .take_str("--weights")
        .or_else(|| std::env::var("BRAIN_SAM2_WEIGHTS").ok())
        .ok_or("--weights <sam2.1_hiera_*.pt> (or BRAIN_SAM2_WEIGHTS) is required")?;
    let max_frames = a.u32_or("--max-frames", 0);
    let fps = a.take_str("--fps").and_then(|s| s.parse::<f64>().ok());
    let object_id = a.u32_or("--object-id", 0);
    let invert = a.take_flag("--invert");
    let soft = a.take_flag("--soft");
    a.finish();

    let (px, py) = parse_point(&point)?;
    let cfg = sam2::caps::variant_config(&variant)?;
    let clip = Clip::load(video.as_deref(), frames_dir.as_deref(), fps, max_frames)?;
    let n = clip.len();
    if prompt_frame >= n {
        return Err(format!("--prompt-frame {prompt_frame} is past the {n}-frame clip"));
    }
    let (w, h) = (clip.width, clip.height);
    eprintln!("sam2 track: {n} frames at {w}x{h}, prompt ({px}, {py}) on frame {prompt_frame}");

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
    let gpu = gpu_core::Gpu::new(sam2::PIPELINES);
    let m = Sam2::new_video(gpu, cfg, &weights);

    let s = m.cfg.image_size;
    // The reference resizes each frame to `image_size²` without preserving
    // aspect ratio (`Resize((1024, 1024))`), so each axis scales on its own.
    let (sx, sy) = (s as f32 / w as f32, s as f32 / h as f32);
    let prompt = Prompt { coords: vec![(px * sx, py * sy)], labels: vec![label], mask_lowres: None, multimask_output: true };

    let polarity = if invert { Polarity::ObjectBlack } else { Polarity::ObjectWhite };
    let mut seq = MaskSeq::new(&out, w, h, clip.fps, polarity, object_id, (clip.name.clone(), n), prompt_frame, vec![(px, py, label)]);
    seq.binary = !soft;

    let mut tr = Tracker::new(&m, n, object_id);
    // The conditioning frame first: everything after it attends to its memory.
    let order: Vec<usize> = std::iter::once(prompt_frame).chain((0..n).filter(|f| *f != prompt_frame)).collect();
    let mut logits: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
    let mut rows: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
    for f in order {
        if f < prompt_frame {
            // Reverse tracking is not ported; frames before the click keep the
            // click frame's own mask rather than silently getting an empty one.
            continue;
        }
        let ctx = Ctx::new(&m.gpu);
        let chw = imaging::pixels::hwc_to_chw(&clip.frames[f], 3, h as usize, w as usize);
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
        logits[f] = Some(ctx.download(&up, ushape.numel()));
        rows[f] = (step.object_score, step.iou);
        if step.object_score <= 0.0 {
            eprintln!("sam2 track: frame {f} - object score {:.3} <= 0, SAM 2 believes the object is occluded here", step.object_score);
        }
    }
    // Frames before the click take its mask, and `masks.json` says so through
    // `prompt.frame`: a mask sequence must cover EVERY source frame or the
    // consumer's frame indices silently shift.
    let seed = logits[prompt_frame].clone().ok_or("the prompt frame produced no mask")?;
    let seed_row = rows[prompt_frame];
    for f in 0..prompt_frame {
        logits[f] = Some(seed.clone());
        rows[f] = seed_row;
    }
    for f in 0..n {
        let l = logits[f].as_ref().ok_or_else(|| format!("frame {f} produced no mask"))?;
        seq.write_frame(f, l, rows[f].0, rows[f].1)?;
    }
    let manifest = seq.write_manifest()?;
    let occluded = seq.occluded_frames();
    eprintln!(
        "sam2 track: wrote {n} masks to {} ({}), manifest {}{}",
        out,
        polarity.tag(),
        manifest.display(),
        if occluded.is_empty() { String::new() } else { format!(", {} occluded frame(s)", occluded.len()) }
    );
    Ok(())
}
