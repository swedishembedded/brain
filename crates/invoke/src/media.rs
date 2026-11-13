// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Files to blobs and blobs to files, in the shared blob conventions
//! (`capability::blob`): images and masks are interleaved HWC f32 in `[0,1]`
//! with meta `{w,h,c}`, audio is 16 kHz mono f32 PCM or a complete WAV, a
//! clip is its frames (`capability::blob::video_blob`).

use std::path::Path;

use capability::{ActionSpec, Blob, Media};
use serde_json::json;

/// Load `path` as the input blob `name` of `spec`, in the media that input
/// declares. A video input is one of:
///
/// - a mask-sequence directory (`imaging::maskseq`), as that sequence;
/// - a directory of numbered frames (PNG/JPEG/PPM, in name order), with no
///   frame rate of its own - the action takes it from a param;
/// - a container file, demuxed whole and at its own rate when the action
///   works on the clip ([`ActionSpec::is_clip_input`]), otherwise sampled at
///   the decoder's defaults as context for a prompt.
pub fn load_blob(spec: &ActionSpec, name: &str, path: &str) -> Result<Blob, String> {
    let media = spec.inputs.iter().find(|b| b.name == name).map(|b| b.media).ok_or_else(|| format!("the action has no input '{name}'"))?;
    let p = Path::new(path);
    if !p.exists() {
        return Err(format!("input '{name}': no file at {path}"));
    }
    match media {
        Media::Image | Media::Mask => {
            let img = imaging::load(p).map_err(|e| format!("input '{name}': {path}: {e}"))?;
            let hwc = img.to_hwc_unit();
            let c = hwc.len() / (img.w as usize * img.h as usize);
            let bytes: Vec<u8> = hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
            Ok(Blob::new(media, bytes).with_meta(json!({"w": img.w, "h": img.h, "c": c})))
        }
        Media::Audio => audio_blob(std::fs::read(p).map_err(|e| format!("input '{name}': {path}: {e}"))?),
        Media::Video if imaging::maskseq::is_dir(p) => Ok(imaging::maskseq::MaskClip::read_dir(p)?.to_blob()),
        Media::Video if p.is_dir() => frames_dir_blob(p),
        Media::Video if spec.is_clip_input(name) => imaging::video::clip_from_file(p, capability::video::MAX_EDIT_CLIP_FRAMES),
        Media::Video => capability::blob::video_blob(&imaging::video::decode_frames(p, &Default::default())?),
        _ => std::fs::read(p).map(|b| Blob::new(media, b)).map_err(|e| format!("input '{name}': {path}: {e}")),
    }
}

/// A directory of numbered frames as a clip blob, frames in file-name order,
/// all of one size.
fn frames_dir_blob(dir: &Path) -> Result<Blob, String> {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| ["png", "jpg", "jpeg", "ppm"].contains(&x.to_string_lossy().to_lowercase().as_str())))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("{} holds no .png/.jpg/.ppm frames", dir.display()));
    }
    let frames = paths.iter().map(|p| imaging::load(p).map(|img| (img.to_hwc_unit(), img.w, img.h))).collect::<Result<Vec<_>, String>>()?;
    capability::blob::video_blob(&frames)
}

/// An audio file's bytes as the `audio` blob: a RIFF/WAVE file is decoded to
/// 16 kHz mono f32 (feeding a model the header as samples is silent
/// garbage); anything else is taken to already be that raw PCM.
pub fn audio_blob(bytes: Vec<u8>) -> Result<Blob, String> {
    if audio::asr_caps::is_wav(&bytes) {
        audio::asr_caps::audio_blob_from_wav(&bytes)
    } else {
        Ok(Blob::new(Media::Audio, bytes))
    }
}

/// Write `b` to `path`: a mask sequence as its directory, an image or mask as
/// PNG/PPM (one channel rendered as grey), audio as WAV, a clip through a
/// container encoder carrying `sound` as its track, anything else as its raw
/// bytes.
pub fn save_blob(b: &Blob, path: &str, sound: Option<&Blob>) -> Result<(), String> {
    if imaging::maskseq::MaskClip::is_blob(b) {
        return imaging::maskseq::MaskClip::from_blob(b)?.write_dir(path).map(|_| ());
    }
    if let Some(dir) = Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    match b.media {
        Media::Video => imaging::video::encode_clip(b, sound, Path::new(path), Default::default()).map(|_| ()),
        Media::Image | Media::Mask => {
            let w = b.meta["w"].as_u64().ok_or("image blob has no w")? as u32;
            let h = b.meta["h"].as_u64().ok_or("image blob has no h")? as u32;
            let c = b.meta["c"].as_u64().unwrap_or(3) as usize;
            let hwc = f32s(&b.bytes);
            let img = imaging::pixels::hwc_to_rgb8(&hwc, w, h, c, imaging::ChannelPolicy::ReplicateFirst)?;
            imaging::save(path, &img)
        }
        // A producer either packs a whole WAV (`meta.format == "wav"`, or a
        // RIFF header) or headerless f32 PCM at `meta.sample_rate`; the
        // latter is wrapped here so the file opens in a player.
        Media::Audio if b.meta.get("format").and_then(|v| v.as_str()) == Some("wav") || audio::asr_caps::is_wav(&b.bytes) => {
            std::fs::write(path, &b.bytes).map_err(|e| e.to_string())
        }
        Media::Audio => {
            let rate = b.meta.get("sample_rate").and_then(|v| v.as_u64()).ok_or("audio blob has no sample_rate and is not a WAV: cannot write a header")? as u32;
            audio::wav::write(path, &f32s(&b.bytes), rate).map_err(|e| e.to_string())
        }
        _ => std::fs::write(path, &b.bytes).map_err(|e| e.to_string()),
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|q| f32::from_le_bytes(*q)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::BlobSpec;

    fn audio_spec() -> ActionSpec {
        ActionSpec::new("transcribe", "transcribe").input(BlobSpec::new("audio", Media::Audio, "raw mono f32 LE PCM at 16 kHz").required())
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("brain-invoke-{}-{name}", std::process::id()))
    }

    /// A WAV file is decoded to the 16 kHz mono blob the ASR models check
    /// for; raw PCM passes through untouched.
    #[test]
    fn audio_inputs_are_decoded_or_passed_through() {
        let wav = tmp("in.wav");
        audio::wav::write(wav.to_str().unwrap(), &vec![0.5f32; 32_000], 32_000).unwrap();
        let b = load_blob(&audio_spec(), "audio", wav.to_str().unwrap()).unwrap();
        assert_eq!(b.meta["sample_rate"], json!(16_000));
        assert_eq!(b.bytes.len(), 16_000 * 4);
        let raw = tmp("in.pcm");
        let pcm: Vec<u8> = [0.25f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&raw, &pcm).unwrap();
        assert_eq!(load_blob(&audio_spec(), "audio", raw.to_str().unwrap()).unwrap().bytes, pcm);
        let _ = (std::fs::remove_file(wav), std::fs::remove_file(raw));
        assert!(load_blob(&audio_spec(), "audio", "/nonexistent/x.wav").unwrap_err().contains("no file"));
    }

    /// A clip that made its picture and sound together is written as one
    /// file carrying both, stereo kept stereo.
    #[test]
    fn a_clip_is_written_with_its_own_sound_track() {
        let frames: Vec<(Vec<f32>, u32, u32)> = (0..4).map(|i| (vec![i as f32 / 4.0; 32 * 16 * 3], 32, 16)).collect();
        let video = capability::blob::video_blob(&frames).unwrap();
        let wav = audio::wav::encode_multi(&[&[0.25f32; 3200], &[-0.25f32; 3200]], 32_000);
        let sound = Blob::new(Media::Audio, wav).with_meta(json!({"format": "wav", "sample_rate": 32_000}));
        let path = tmp("av.mkv");
        save_blob(&video, path.to_str().unwrap(), Some(&sound)).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let head = bytes.windows(8).position(|w| w == b"OpusHead").expect("the clip carries an Opus sound track");
        assert_eq!(bytes[head + 9], 2, "both channels of the track are kept");
    }
}
