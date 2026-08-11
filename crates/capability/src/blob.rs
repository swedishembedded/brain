// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shared HWC-f32 image ↔ [`Blob`] codec — the ONE implementation of brain's
//! image-blob wire format (raw interleaved HWC f32 in `[0,1]`, little-endian,
//! meta `{"w","h","c"}`). Every provider and resident adapter decodes/encodes
//! images through here; never re-implement it locally (that is how
//! per-model encodings drift apart).

use serde_json::json;

use crate::{Blob, Invocation, Media};

/// Encode an interleaved HWC f32 `[0,1]` image (`c` channels) as an image
/// [`Blob`]: f32-LE bytes + the standard `{"w","h","c"}` metadata.
pub fn image_blob(hwc: &[f32], w: u32, h: u32, c: u32) -> Blob {
    debug_assert_eq!(hwc.len(), w as usize * h as usize * c as usize);
    let bytes: Vec<u8> = hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
    Blob::new(Media::Image, bytes).with_meta(json!({"w": w, "h": h, "c": c}))
}

/// Decode a named HWC-f32 blob to `(hwc, w, h, c)`. Validates the media (an
/// image-like blob: `image`/`mask`, or untyped `bytes` from a client that sent no
/// media tag), the `{w,h}` metadata, and that the payload is a whole number of
/// `w×h` f32 planes (`c` is inferred from the length).
pub fn decode_hwc(inv: &Invocation, name: &str) -> Result<(Vec<f32>, u32, u32, usize), String> {
    let b = inv.get_blob(name).ok_or_else(|| format!("missing required input '{name}'"))?;
    if matches!(b.media, Media::Audio | Media::Text) {
        return Err(format!("'{name}' must be an image blob (got {})", b.media.name()));
    }
    let dim = |k: &str| b.meta.get(k).and_then(|v| v.as_u64()).ok_or_else(|| format!("'{name}' blob missing {k}"));
    let (w, h) = (dim("w")? as u32, dim("h")? as u32);
    let px = w as usize * h as usize;
    let hwc: Vec<f32> = b.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
    if px == 0 || b.bytes.len() % 4 != 0 || !hwc.len().is_multiple_of(px) || hwc.is_empty() {
        return Err(format!("'{name}' payload ({} bytes) is not a whole number of {w}×{h} f32 planes", b.bytes.len()));
    }
    let c = hwc.len() / px;
    Ok((hwc, w, h, c))
}

/// Decode an RGB image blob (`c` must be 3). Returns `(hwc, w, h)`.
pub fn decode_image(inv: &Invocation, name: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let (hwc, w, h, c) = decode_hwc(inv, name)?;
    if c != 3 {
        return Err(format!("'{name}' must be a 3-channel RGB image (got {c} channels)"));
    }
    Ok((hwc, w, h))
}

/// Decode a mask-style blob to its channel-0 plane `[h·w]` in `[0,1]`.
/// Returns `(plane, w, h)`.
pub fn decode_plane(inv: &Invocation, name: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let (hwc, w, h, c) = decode_hwc(inv, name)?;
    let plane: Vec<f32> = (0..w as usize * h as usize).map(|i| hwc[i * c]).collect();
    Ok((plane, w, h))
}

/// Decode a multi-frame video blob: `frames` interleaved-HWC f32 RGB planes
/// (each `w×h×3`) concatenated into ONE payload, with `{"frames","w","h","c"}`
/// metadata. Returns one `(hwc, w, h)` tuple per frame, in order — the exact
/// shape `omni::mm::encode_video_frames` takes.
///
/// This is `decode_hwc` with `c` read from metadata rather than inferred from
/// the payload length: an N-frame video and a single `(3N)`-channel image are
/// indistinguishable by length alone (`decode_hwc`'s `c = len/(w*h)` would
/// misread N RGB frames as one `3N`-channel image), so a video blob needs its
/// own decode path rather than reusing `decode_hwc` with a post-hoc split.
///
/// Wire shape chosen over a repeated blob because `Invocation::blobs` is a
/// `BTreeMap<String, Blob>` — one blob per declared name, not a list — and
/// extending that to a repeated/dynamic blob would ripple into the D-Bus
/// fd-map and every HTTP transport for one input kind. Uses `Media::Bytes`
/// (there is no `Media::Video`), matching `crates/vqgan`'s precedent for a
/// non-image payload shipped that way.
pub fn decode_video_hwc(inv: &Invocation, name: &str) -> Result<Vec<(Vec<f32>, u32, u32)>, String> {
    let b = inv.get_blob(name).ok_or_else(|| format!("missing required input '{name}'"))?;
    let dim = |k: &str| b.meta.get(k).and_then(|v| v.as_u64()).ok_or_else(|| format!("'{name}' blob missing {k}"));
    let frames = dim("frames")? as usize;
    let (w, h, c) = (dim("w")? as u32, dim("h")? as u32, dim("c")? as usize);
    if c != 3 {
        return Err(format!("'{name}' must be 3-channel RGB per frame (got c={c})"));
    }
    if frames == 0 || w == 0 || h == 0 {
        return Err(format!("'{name}': frames/w/h must all be > 0 (got frames={frames} w={w} h={h})"));
    }
    let per_frame = w as usize * h as usize * c;
    let want_bytes = frames * per_frame * 4;
    if b.bytes.len() != want_bytes {
        return Err(format!(
            "'{name}' payload is {} bytes, expected {frames}×{w}×{h}×{c}×4 = {want_bytes}",
            b.bytes.len()
        ));
    }
    let all: Vec<f32> = b.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
    Ok(all.chunks_exact(per_frame).map(|chunk| (chunk.to_vec(), w, h)).collect())
}

/// Encode N RGB frames as a video [`Blob`]: concatenated interleaved-HWC f32
/// planes + `{"frames","w","h","c"}` metadata — the encoder counterpart of
/// [`decode_video_hwc`], for a caller that has real frames (e.g.
/// `imaging::video::decode_frames`'s output) and wants to send them over a
/// `generate`/`converse` `video` input. Every frame must share the SAME
/// `w`/`h` (the wire format carries no per-frame dimensions, matching
/// [`decode_video_hwc`]'s own single `{w,h}` pair).
pub fn video_blob(frames: &[(Vec<f32>, u32, u32)]) -> Result<Blob, String> {
    let (_, w, h) = frames.first().ok_or("video_blob: at least one frame required")?;
    let (w, h) = (*w, *h);
    let per_frame = w as usize * h as usize * 3;
    let mut bytes = Vec::with_capacity(frames.len() * per_frame * 4);
    for (i, (hwc, fw, fh)) in frames.iter().enumerate() {
        if (*fw, *fh) != (w, h) {
            return Err(format!("video_blob: frame {i} is {fw}x{fh}, expected {w}x{h} -- every frame must share dims"));
        }
        if hwc.len() != per_frame {
            return Err(format!("video_blob: frame {i} has {} elements, expected {w}x{h}x3={per_frame}", hwc.len()));
        }
        bytes.extend(hwc.iter().flat_map(|f| f.to_le_bytes()));
    }
    Ok(Blob::new(Media::Bytes, bytes).with_meta(json!({"frames": frames.len(), "w": w, "h": h, "c": 3})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_roundtrips_through_the_wire_format() {
        let hwc: Vec<f32> = (0..2 * 2 * 3).map(|i| i as f32 / 12.0).collect();
        let b = image_blob(&hwc, 2, 2, 3);
        assert_eq!(b.media, Media::Image);
        assert_eq!(b.meta, json!({"w": 2, "h": 2, "c": 3}));
        assert_eq!(b.bytes.len(), 2 * 2 * 3 * 4);
        let inv = Invocation::new().blob("image", b);
        let (back, w, h) = decode_image(&inv, "image").unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(back, hwc);
        let (_, _, _, c) = decode_hwc(&inv, "image").unwrap();
        assert_eq!(c, 3);
    }

    #[test]
    fn plane_takes_channel_zero() {
        let hwc: Vec<f32> = vec![1.0, 9.0, 9.0, 0.5, 9.0, 9.0]; // 2×1 RGB
        let inv = Invocation::new().blob("mask", image_blob(&hwc, 2, 1, 3));
        let (plane, w, h) = decode_plane(&inv, "mask").unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(plane, [1.0, 0.5]);
    }

    #[test]
    fn decode_validates_media_meta_and_size() {
        // missing blob
        assert!(decode_hwc(&Invocation::new(), "image").unwrap_err().contains("missing required input"));
        // wrong media
        let inv = Invocation::new().blob("image", Blob::new(Media::Audio, vec![0; 16]).with_meta(json!({"w":2,"h":2})));
        assert!(decode_hwc(&inv, "image").unwrap_err().contains("must be an image blob"));
        // missing meta
        let inv = Invocation::new().blob("image", Blob::new(Media::Image, vec![0; 16]));
        assert!(decode_hwc(&inv, "image").unwrap_err().contains("missing w"));
        // payload not whole w×h planes
        let inv = Invocation::new().blob("image", Blob::new(Media::Image, vec![0; 20]).with_meta(json!({"w":2,"h":2})));
        assert!(decode_hwc(&inv, "image").unwrap_err().contains("planes"));
        // not 3-channel where RGB is required
        let inv = Invocation::new().blob("image", image_blob(&[0.0; 4], 2, 2, 1));
        assert!(decode_image(&inv, "image").unwrap_err().contains("3-channel"));
    }

    #[test]
    fn video_decodes_n_frames_and_is_not_misread_as_one_wide_image() {
        // 3 frames, 2×1 RGB each -- distinct constant colors per frame so a
        // misinterpretation (e.g. as one 2×1×9 "image") is easy to catch.
        let mut bytes = Vec::new();
        let frames_f32: [[f32; 6]; 3] = [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], // frame 0: red, red
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], // frame 1: green, green
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0], // frame 2: blue, blue
        ];
        for f in &frames_f32 {
            for v in f {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let b = Blob::new(Media::Bytes, bytes).with_meta(json!({"frames": 3, "w": 2, "h": 1, "c": 3}));
        let inv = Invocation::new().blob("video", b);
        let frames = decode_video_hwc(&inv, "video").unwrap();
        assert_eq!(frames.len(), 3);
        for ((hwc, w, h), expect) in frames.iter().zip(frames_f32.iter()) {
            assert_eq!((*w, *h), (2, 1));
            assert_eq!(hwc.as_slice(), expect.as_slice());
        }
    }

    #[test]
    fn video_validates_meta_and_exact_payload_size() {
        // missing blob
        assert!(decode_video_hwc(&Invocation::new(), "video").unwrap_err().contains("missing required input"));
        // missing meta
        let inv = Invocation::new().blob("video", Blob::new(Media::Bytes, vec![0; 24]));
        assert!(decode_video_hwc(&inv, "video").unwrap_err().contains("missing frames"));
        // not 3-channel
        let inv = Invocation::new().blob(
            "video",
            Blob::new(Media::Bytes, vec![0; 8]).with_meta(json!({"frames": 1, "w": 1, "h": 1, "c": 1})),
        );
        assert!(decode_video_hwc(&inv, "video").unwrap_err().contains("3-channel"));
        // payload size mismatch (declares 2 frames, ships bytes for 1)
        let inv = Invocation::new().blob(
            "video",
            Blob::new(Media::Bytes, vec![0; 12]).with_meta(json!({"frames": 2, "w": 1, "h": 1, "c": 3})),
        );
        assert!(decode_video_hwc(&inv, "video").unwrap_err().contains("expected"));
    }

    #[test]
    fn video_blob_roundtrips_through_decode_video_hwc() {
        let frames: Vec<(Vec<f32>, u32, u32)> = vec![(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 2, 1), (vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0], 2, 1)];
        let b = video_blob(&frames).unwrap();
        assert_eq!(b.media, Media::Bytes);
        assert_eq!(b.meta, json!({"frames": 2, "w": 2, "h": 1, "c": 3}));
        let inv = Invocation::new().blob("video", b);
        let back = decode_video_hwc(&inv, "video").unwrap();
        assert_eq!(back, frames);
    }

    #[test]
    fn video_blob_rejects_mismatched_frame_dims() {
        let frames: Vec<(Vec<f32>, u32, u32)> = vec![(vec![0.0; 6], 2, 1), (vec![0.0; 12], 4, 1)];
        let err = video_blob(&frames).unwrap_err();
        assert!(err.contains("expected 2x1"), "error should name the mismatch: {err}");
    }

    #[test]
    fn video_blob_rejects_empty_frame_list() {
        assert!(video_blob(&[]).unwrap_err().contains("at least one frame"));
    }
}
