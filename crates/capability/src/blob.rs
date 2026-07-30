// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shared HWC-f32 image ↔ [`Blob`] codec — the ONE implementation of brain's
//! image-blob wire format (raw interleaved HWC f32 in `[0,1]`, little-endian,
//! meta `{"w","h","c"}`; see `docs/serving-contract.md` §1). Every provider and
//! resident adapter decodes/encodes images through here; never re-implement it
//! locally (that is how per-model encodings drift apart).

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
    if px == 0 || b.bytes.len() % 4 != 0 || hwc.len() % px != 0 || hwc.is_empty() {
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
}
