// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Photographs, a clip, or both - one ordered frame set.
//!
//! A capture is rarely one thing. Someone walks a room with a phone, then
//! takes a handful of stills of the part that mattered. Both are views of one
//! scene, and a reconstruction that can only take one of them throws away the
//! evidence the other carries.
//!
//! Sources are concatenated in the order given and never interleaved: the
//! order frames arrive in IS the structure the chunk overlap relies on, so
//! shuffling a clip's frames in among a folder's would put consecutive chunks
//! on views that have nothing to do with each other.

use std::path::{Path, PathBuf};

use imaging::Rgb8;

use crate::PipelineError;

/// Where frames come from.
#[derive(Clone, Debug)]
pub enum Source {
    /// Every image in a directory, in filename order - which is a capture's
    /// own order, since that is how cameras name files.
    Dir(PathBuf),
    /// Exactly these images, in exactly this order.
    Images(Vec<PathBuf>),
    /// A clip. `spread` asks the decoder for that many frames spaced over the
    /// WHOLE clip (0 = leave it to `fps`); `fps` resamples to a rate instead
    /// when it is above zero. Spreading happens inside the decoder, so the
    /// frames nobody wants are never converted and never staged on disk.
    Video { path: PathBuf, spread: u32, fps: f64 },
}

/// One frame, and where it came from. The label survives the whole pipeline,
/// so a report can say which photograph registered badly rather than which
/// index did.
#[derive(Clone, Debug)]
pub struct Frame {
    pub image: Rgb8,
    pub label: String,
    /// Index into the [`Source`] list this frame was ingested from.
    pub source: usize,
}

impl Frame {
    pub fn new(image: Rgb8, label: impl Into<String>, source: usize) -> Frame {
        Frame { image, label: label.into(), source }
    }
}

/// Extensions treated as images when a whole directory is ingested. The decode
/// itself sniffs the BYTES (`imaging::load`); this list only decides what is a
/// candidate, so a `notes.txt` sitting beside the photographs is not an error.
const IMAGE_EXTS: &[&str] = &["ppm", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"];

fn ext_of(p: &Path) -> String {
    p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

/// Read every source into one ordered frame set.
pub fn ingest(sources: &[Source]) -> Result<Vec<Frame>, PipelineError> {
    let mut out = Vec::new();
    for (si, src) in sources.iter().enumerate() {
        match src {
            Source::Dir(dir) => {
                let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
                    .map_err(|e| PipelineError::Ingest {
                        source: dir.display().to_string(),
                        reason: e.to_string(),
                    })?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && IMAGE_EXTS.contains(&ext_of(p).as_str()))
                    .collect();
                paths.sort();
                for p in paths {
                    out.push(load_frame(&p, si)?);
                }
            }
            Source::Images(paths) => {
                for p in paths {
                    out.push(load_frame(p, si)?);
                }
            }
            Source::Video { path, spread, fps } => {
                let opts = imaging::video::VideoDecodeOpts {
                    fps: if *fps > 0.0 { Some(*fps) } else { None },
                    max_frames: 0,
                    spread: *spread,
                };
                let frames = imaging::video::decode_frames_rgb8(path, &opts)
                    .map_err(|e| PipelineError::Ingest { source: path.display().to_string(), reason: e })?;
                for (i, image) in frames.into_iter().enumerate() {
                    out.push(Frame::new(image, format!("{}#{i:05}", path.display()), si));
                }
            }
        }
    }
    if out.is_empty() {
        return Err(PipelineError::NoFrames);
    }
    Ok(out)
}

fn load_frame(path: &Path, source: usize) -> Result<Frame, PipelineError> {
    let image = imaging::load(path)
        .map_err(|e| PipelineError::Ingest { source: path.display().to_string(), reason: e })?;
    Ok(Frame::new(image, path.display().to_string(), source))
}
