// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `chain`: numbered stills in, one clip out.
//!
//! Each pair of consecutive stills becomes one `t2v` segment, the first as
//! its start frame and the second as its end frame (with `--mid`, triples:
//! start, mid and end, overlapping on their last still). Because a segment's
//! end is the next one's start, the segments already agree at every seam;
//! the shared still appears once in the result, and the sound, when the clip
//! has some, is joined the same way.
//!
//! The prompt is `--prompt`, or `prompt.txt` in the folder. Every other flag
//! is `t2v`'s own, applied to every segment; the segments are kept beside the
//! result in `<out>.segments/`.

use std::path::{Path, PathBuf};

use capability::{Invocation, Media};

pub fn run(call: &invoke::WorkflowCall) -> Result<(), String> {
    let folder = call.args.first().filter(|a| !a.starts_with('-')).ok_or("give the folder of stills first")?;
    let mut rest: Vec<String> = call.args[1..].to_vec();
    let mid = take_flag(&mut rest, "--mid");
    let out = take_value(&mut rest, "--out")?.ok_or("give the result as --out video=<file>")?;
    let out = out.strip_prefix("video=").unwrap_or(&out).to_string();

    let t2v = call.provider.action("t2v").ok_or("the model has no t2v action")?;
    let spec = t2v.spec();
    let parsed = invoke::args::check(call.program, &spec, &rest).map_err(|_| "see the flag error above".to_string())?;
    let prompt = match parsed.invocation.get_str("prompt") {
        Some(p) => p,
        None => std::fs::read_to_string(Path::new(folder).join("prompt.txt")).map_err(|e| format!("no --prompt and no {folder}/prompt.txt ({e})"))?.trim().to_string(),
    };

    let stills = stills(Path::new(folder))?;
    let step = if mid { 2 } else { 1 };
    if stills.len() < step + 1 || (mid && (stills.len() - 1) % 2 != 0) {
        return Err(format!(
            "{} stills: {} needs {}",
            stills.len(),
            if mid { "--mid" } else { "chaining" },
            if mid { "an odd count of at least 3 (start/mid/end triples sharing their ends)" } else { "at least 2" }
        ));
    }
    let segments = PathBuf::from(format!("{}.segments", out.rsplit_once('.').map_or(out.as_str(), |(stem, _)| stem)));
    let mut frames: Vec<(Vec<f32>, u32, u32)> = Vec::new();
    let mut sound: Vec<Vec<f32>> = Vec::new();
    let (mut fps, mut rate) = (0.0f64, 0u32);
    for (n, i) in (0..stills.len() - step).step_by(step).enumerate() {
        let path = |k: usize| serde_json::json!(stills[k].to_string_lossy());
        let mut inv = parsed.invocation.clone().set("prompt", serde_json::json!(prompt)).set("start_frame", path(i)).set("end_frame", path(i + step));
        if mid {
            inv = inv.set("mid_frame", path(i + 1));
        }
        eprintln!("{}: segment {n}: {} -> {}", call.program, stills[i].display(), stills[i + step].display());
        let outcome = t2v.run(&spec.validate(inv)?, &mut |p| {
            if p.total > 0 {
                eprint!("\r\x1b[2Ksegment {n} [{}/{}] {}", p.step, p.total, p.message);
            }
        })?;
        eprintln!();
        let video = outcome.blobs.get("video").ok_or("t2v made no video")?;
        invoke::media::save_blob(video, &segments.join(format!("clip-{n:02}.mp4")).to_string_lossy(), outcome.blobs.get("audio"))?;
        fps = video.meta.get("fps").and_then(|v| v.as_f64()).unwrap_or(fps);
        let decoded = capability::blob::decode_video(&Invocation::new().blob("v", video.clone()), "v")?;
        // A segment's first frame is the previous one's last: the shared still.
        let skip = usize::from(n > 0);
        frames.extend(decoded.into_iter().skip(skip));
        if let Some(a) = outcome.blobs.get("audio").filter(|b| b.media == Media::Audio) {
            let (r, channels) = audio::wav::parse_planar(&a.bytes).map_err(|e| format!("segment {n}'s sound: {e}"))?;
            rate = r;
            let drop = if n > 0 && fps > 0.0 { (r as f64 / fps).round() as usize } else { 0 };
            sound.resize(channels.len(), Vec::new());
            for (all, ch) in sound.iter_mut().zip(channels) {
                all.extend(ch.into_iter().skip(drop));
            }
        }
    }
    let mut clip = capability::blob::video_blob(&frames)?;
    clip.meta["fps"] = serde_json::json!(fps);
    let track = (!sound.is_empty()).then(|| {
        let refs: Vec<&[f32]> = sound.iter().map(Vec::as_slice).collect();
        capability::Blob::new(Media::Audio, audio::wav::encode_multi(&refs, rate)).with_meta(serde_json::json!({"format": "wav", "sample_rate": rate}))
    });
    invoke::media::save_blob(&clip, &out, track.as_ref())?;
    eprintln!("{}: wrote {out}: {} frames from {} segments (kept in {})", call.program, frames.len(), (stills.len() - 1) / step, segments.display());
    Ok(())
}

/// `folder`'s `image-NN.*` stills, in name order.
fn stills(folder: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(folder)
        .map_err(|e| format!("{}: {e}", folder.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            name.starts_with("image-") && [".png", ".jpg", ".jpeg"].iter().any(|x| name.ends_with(x))
        })
        .collect();
    found.sort();
    Ok(found)
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    let before = args.len();
    args.retain(|a| a != flag);
    args.len() != before
}

fn take_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            let v = args.get(i + 1).cloned().ok_or_else(|| format!("{flag} needs a value"))?;
            args.drain(i..=i + 1);
            Ok(Some(v))
        }
        None => Ok(None),
    }
}
