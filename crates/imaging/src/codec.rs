// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Image decode / encode — the one front door for turning bytes on disk into
//! pixels.
//!
//! ## P6 is re-exported, not re-implemented
//!
//! `events::ppm::{encode_p6, decode_p6}` is the workspace's canonical binary-PPM
//! codec: a full ASCII header tokenizer (`#` comments, arbitrary whitespace),
//! maxval checked, and it never panics. There were two independent P6 parsers
//! (`events` and `worldmirror2::preprocess::load_ppm`) plus six inline
//! `format!("P6\n{w} {h}\n255\n")` header writers. This module adds a
//! **zero**th: it re-exports the codec, so `imaging` is the front door while
//! `events` remains the implementation.
//!
//! Intended end state, for the migrator: once `crates/events` can be edited, the
//! `ppm` module moves *into* this file and `events` re-exports it from here,
//! flipping this dependency edge. Do that with `imaging`'s `image` dependency
//! behind a default feature that `events` turns off — otherwise the wasm build
//! (`crates/web` -> `events`) grows a JPEG decoder it cannot use. That is the
//! deliberate decision the survey's §6.1 asks for; it is not automatic.
//!
//! ## PNG / JPEG
//!
//! `events::decode_pixels` returns `"PNG is not supported (no decoder)"` while
//! `crates/data` has decoded PNG and JPEG all along (`image 0.25`, used by
//! `data::imageset::decode_rgb`). The capability was present and merely
//! unreachable. `imaging` owning the `image` dependency is what makes it
//! reachable from `events`, `cli` and every future capability.
//!
//! The same dependency carries `image`'s PNG **and JPEG encoders**, so
//! [`save_png`] and [`save_jpeg`] are wiring, not new code and not a new
//! dependency. Writing a DCT/Huffman encoder by hand here would be a second
//! implementation of something already linked into every binary in the
//! workspace - the `rmsnorm`-was-seven-times failure mode this crate exists to
//! undo. `imaging` is dependency-light about what enters the *runtime*, not
//! about refusing a pure-Rust codec it already ships.
//!
//! P6 is sniffed **first** so it always takes the `events` path, never `image`'s
//! PNM decoder — one format, one implementation, whatever cargo's feature
//! unification decides to compile.

use std::path::Path;

use crate::pixels::Rgb8;

pub use events::ppm::{decode_p6, encode_p6};

/// Decode PPM (P6), PNG or JPEG from memory, dispatching on the magic bytes.
///
/// Errors name the format when one is recognised but undecodable, and say what
/// *is* supported when the magic matches nothing — an unrecognised blob is
/// usually a truncated download or a path mix-up, not an exotic format.
pub fn decode(bytes: &[u8]) -> Result<Rgb8, String> {
    decode_bounded(bytes, u64::MAX)
}

/// [`decode`] for bytes from an untrusted caller: an image of more than `max_pixels` pixels is refused from its
/// header, before a pixel is decoded (a PNG a few kilobytes long can declare billions of them), and the decoder's own
/// allocation is bounded to the same figure.
pub fn decode_bounded(bytes: &[u8], max_pixels: u64) -> Result<Rgb8, String> {
    let too_big = |w: u64, h: u64| format!("image of {w}x{h} pixels is over the limit of {max_pixels} pixels");
    if bytes.starts_with(b"P6") {
        let (px, w, h) = decode_p6(bytes)?;
        if u64::from(w) * u64::from(h) > max_pixels {
            return Err(too_big(w.into(), h.into()));
        }
        return Rgb8::new(w, h, px);
    }
    let fmt = if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        image::ImageFormat::Png
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        image::ImageFormat::Jpeg
    } else {
        return Err(format!(
            "unsupported image: leading bytes {:02x?} match none of P6 / PNG / JPEG",
            &bytes[..bytes.len().min(4)]
        ));
    };
    let reader = || image::ImageReader::with_format(std::io::Cursor::new(bytes), fmt);
    let (w, h) = reader().into_dimensions().map_err(|e| format!("decoding {fmt:?}: {e}"))?;
    if u64::from(w) * u64::from(h) > max_pixels {
        return Err(too_big(w.into(), h.into()));
    }
    let mut decoder = reader();
    let mut limits = image::Limits::default();
    // The decoded image and its working copies: 8 bytes a pixel covers an RGBA16 source.
    limits.max_alloc = Some(max_pixels.saturating_mul(8).max(1 << 20));
    decoder.limits(limits);
    let img = decoder.decode().map_err(|e| format!("decoding {fmt:?}: {e}"))?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    Rgb8::new(w, h, img.into_raw())
}

/// A single-channel 16-bit image: the format depth sensors and depth datasets store a map in (a count per pixel,
/// whose unit the dataset states), which [`Rgb8`] cannot hold without throwing away the low byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gray16 {
    pub w: u32,
    pub h: u32,
    /// Row-major, `w * h` values.
    pub px: Vec<u16>,
}

/// Decode a 16-bit grayscale PNG from untrusted bytes. Anything else (an 8-bit, RGB or palette PNG, a JPEG) is
/// refused naming what it is: reading it as 16-bit would reinterpret or widen values the caller did not mean.
/// An image of more than `max_pixels` pixels is refused from its header before a pixel is decoded.
pub fn decode_gray16(bytes: &[u8], max_pixels: u64) -> Result<Gray16, String> {
    if !bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        return Err(format!("not a PNG: leading bytes {:02x?}", &bytes[..bytes.len().min(4)]));
    }
    let reader = || image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Png);
    let (w, h) = reader().into_dimensions().map_err(|e| format!("decoding PNG: {e}"))?;
    if u64::from(w) * u64::from(h) > max_pixels {
        return Err(format!("image of {w}x{h} pixels is over the limit of {max_pixels} pixels"));
    }
    let mut decoder = reader();
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(max_pixels.saturating_mul(8).max(1 << 20));
    decoder.limits(limits);
    match decoder.decode().map_err(|e| format!("decoding PNG: {e}"))? {
        image::DynamicImage::ImageLuma16(img) => Ok(Gray16 { w, h, px: img.into_raw() }),
        other => Err(format!("expected a 16-bit grayscale PNG, found {:?}", other.color())),
    }
}

/// Write a [`Gray16`] as a 16-bit grayscale PNG, creating the parent directory.
pub fn save_gray16_png(path: impl AsRef<Path>, img: &Gray16) -> Result<(), String> {
    let path = path.as_ref();
    create_parent_dir(path)?;
    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(img.w, img.h, img.px.clone())
        .ok_or_else(|| format!("{}: pixel buffer does not match {}x{}", path.display(), img.w, img.h))?
        .save_with_format(path, image::ImageFormat::Png)
        .map_err(|e| format!("writing {}: {e}", path.display()))
}

/// A PPM, PNG or JPEG file's `(width, height)`, read off its header without
/// decoding a pixel.
pub fn dimensions(path: impl AsRef<Path>) -> Result<(u32, u32), String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.starts_with(b"P6") {
        return decode_p6(&bytes).map(|(_, w, h)| (w, h)).map_err(|e| format!("{}: {e}", path.display()));
    }
    image::ImageReader::new(std::io::Cursor::new(&bytes)).with_guessed_format().map_err(|e| format!("{}: {e}", path.display()))?.into_dimensions().map_err(|e| format!("{}: {e}", path.display()))
}

/// Read and [`decode`] a file.
pub fn load(path: impl AsRef<Path>) -> Result<Rgb8, String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write an [`Rgb8`] as a binary PPM, creating the parent directory.
///
/// PPM because it is the format brain can both read and write with no encoder
/// dependency; `image` was decode-only here for a long time on purpose - see
/// [`save_png`] and [`save`] for why that stopped being enough.
pub fn save_ppm(path: impl AsRef<Path>, img: &Rgb8) -> Result<(), String> {
    let path = path.as_ref();
    create_parent_dir(path)?;
    std::fs::write(path, encode_p6(&img.px, img.w, img.h))
        .map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Write an [`Rgb8`] as PNG, creating the parent directory.
///
/// PPM has no viewer in a browser, a chat client, or GitHub's own markdown
/// renderer - exactly the audiences a generated demo image is FOR. `image`
/// (already a decode dependency here, `png` feature already enabled for
/// [`decode`]) makes PNG encoding free to add rather than a new dependency.
pub fn save_png(path: impl AsRef<Path>, img: &Rgb8) -> Result<(), String> {
    let path = path.as_ref();
    create_parent_dir(path)?;
    image::RgbImage::from_raw(img.w, img.h, img.px.clone())
        .ok_or_else(|| format!("{}: pixel buffer does not match {}x{}", path.display(), img.w, img.h))?
        .save_with_format(path, image::ImageFormat::Png)
        .map_err(|e| format!("writing {}: {e}", path.display()))
}

/// The quality [`save`] encodes a `.jpg`/`.jpeg` path at, on libjpeg's 1-100
/// scale.
///
/// 92 is the "high quality" end of the scale that ImageMagick, GIMP's export
/// dialog and most photo tools present as their own default-for-keeping, as
/// opposed to libjpeg's own 75, which is tuned for shipping a photograph over a
/// slow link. The difference matters here because brain's `.jpg` writes are
/// *generated* images, not camera captures: flat regions, hard synthetic edges
/// and text overlays are exactly what shows quantisation ringing first, and a
/// generated frame is frequently the input to the next stage (an upscale, a
/// caption, a parity comparison) rather than a final artefact. 92 keeps the
/// order-of-magnitude size win over PNG while leaving no artefact a viewer
/// would notice; a caller that wants a different point on the curve calls
/// [`save_jpeg`] directly rather than being stuck with this one.
pub const JPEG_QUALITY: u8 = 92;

/// Write an [`Rgb8`] as JPEG at `quality` (1-100, libjpeg's scale), creating the
/// parent directory.
///
/// The encoder is `image`'s (`image::codecs::jpeg`), reached through the `jpeg`
/// feature this crate already enables for [`decode`] - the same "already a
/// dependency, so the encoder is free" argument [`save_png`] makes. Baseline
/// JPEG, 4:4:4, no chroma subsampling.
///
/// JPEG is **lossy** and its dimensions are 16-bit: an image wider or taller
/// than 65535 px is an error here, not a silent crop. Nothing is written unless
/// the encode succeeds.
pub fn save_jpeg(path: impl AsRef<Path>, img: &Rgb8, quality: u8) -> Result<(), String> {
    let path = path.as_ref();
    // `JpegEncoder::encode` asserts on a mismatched buffer, and `Rgb8`'s fields
    // are public, so a hand-built struct could reach it. Same guard `save_png`
    // gets for free from `RgbImage::from_raw`.
    let need = img.w as usize * img.h as usize * 3;
    if img.px.len() != need {
        return Err(format!("{}: pixel buffer does not match {}x{}", path.display(), img.w, img.h));
    }
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, quality)
        .encode(&img.px, img.w, img.h, image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("encoding {} as JPEG: {e}", path.display()))?;
    create_parent_dir(path)?;
    std::fs::write(path, bytes).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Write an [`Rgb8`], choosing the encoder from `path`'s extension
/// (case-insensitively): `.png` -> [`save_png`], `.jpg`/`.jpeg` ->
/// [`save_jpeg`] at [`JPEG_QUALITY`], `.ppm` or no extension at all ->
/// [`save_ppm`]. This is the front door every CLI `--out name=path` write
/// should go through: the extension the caller typed is the format that lands
/// on disk.
///
/// Any **other** extension is an error naming it, and nothing is written. The
/// alternative - falling back to P6 - is worse than refusing: `--out
/// photo.webp` then produces a P6 wearing a `.webp` suffix, which no viewer
/// opens and which misreports its own format to everything downstream that
/// trusts the name. An unsupported extension is a typo or an unimplemented
/// format, and both want to be said out loud.
pub fn save(path: impl AsRef<Path>, img: &Rgb8) -> Result<(), String> {
    let path = path.as_ref();
    // `to_string_lossy`, not `to_str`: a non-UTF-8 extension is still an
    // extension, and must not fall through the `None` arm into a P6 the caller
    // did not ask for.
    let ext = path.extension().map(|e| e.to_string_lossy().into_owned());
    match ext.as_deref() {
        None => save_ppm(path, img),
        Some(e) if e.eq_ignore_ascii_case("ppm") => save_ppm(path, img),
        Some(e) if e.eq_ignore_ascii_case("png") => save_png(path, img),
        Some(e) if e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg") => save_jpeg(path, img, JPEG_QUALITY),
        Some(e) => Err(format!(
            "{}: brain cannot write '.{e}' images - supported are .png, .jpg/.jpeg and .ppm (a path with no extension writes .ppm's P6)",
            path.display()
        )),
    }
}

fn create_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Rgb8 {
        Rgb8::new(2, 1, vec![255, 0, 0, 0, 255, 0]).unwrap()
    }

    #[test]
    fn gray16_png_round_trips_every_16_bit_value() {
        let dir = std::env::temp_dir().join(format!("imaging-gray16-{}", std::process::id()));
        let img = Gray16 { w: 3, h: 2, px: vec![0, 1, 255, 256, 40000, 65535] };
        let path = dir.join("d.png");
        save_gray16_png(&path, &img).unwrap();
        let back = decode_gray16(&std::fs::read(&path).unwrap(), 6).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(back, img);
    }

    #[test]
    fn gray16_refuses_what_is_not_16_bit_gray_and_what_is_too_large() {
        let dir = std::env::temp_dir().join(format!("imaging-gray16-refuse-{}", std::process::id()));
        let rgb = dir.join("rgb.png");
        save_png(&rgb, &tiny()).unwrap();
        let bytes = std::fs::read(&rgb).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let err = decode_gray16(&bytes, u64::MAX).unwrap_err();
        assert!(err.contains("16-bit grayscale") && err.contains("Rgb8"), "{err}");
        assert!(decode_gray16(b"P6\n", u64::MAX).unwrap_err().contains("not a PNG"));
        let big = Gray16 { w: 4, h: 4, px: vec![7; 16] };
        let path = std::env::temp_dir().join(format!("imaging-gray16-big-{}.png", std::process::id()));
        save_gray16_png(&path, &big).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(decode_gray16(&bytes, 15).unwrap_err().contains("over the limit"));
    }

    #[test]
    fn p6_round_trips_through_the_shared_codec() {
        let img = tiny();
        let bytes = encode_p6(&img.px, img.w, img.h);
        assert!(bytes.starts_with(b"P6\n2 1\n255\n"));
        assert_eq!(decode(&bytes).unwrap(), img);
    }

    #[test]
    fn p6_with_comments_and_odd_whitespace_decodes() {
        // The header tokenizer is why P6 must not be re-parsed by hand: this is
        // valid PPM and the naive `split_whitespace` parsers get it wrong.
        let mut bytes = b"P6 # a comment\n  2\t1\n255\n".to_vec();
        bytes.extend_from_slice(&[255, 0, 0, 0, 255, 0]);
        assert_eq!(decode(&bytes).unwrap(), tiny());
    }

    #[test]
    fn truncated_p6_is_an_error_not_a_panic() {
        let mut bytes = b"P6\n2 1\n255\n".to_vec();
        bytes.push(255);
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn unknown_magic_names_what_is_supported() {
        let e = decode(b"GIF89a").unwrap_err();
        assert!(e.contains("P6"), "error should list the supported formats, got: {e}");
    }

    /// A PNG that is only a signature, a valid IHDR and the start of its pixel stream, declaring `w x h`: all a decompression bomb needs to announce.
    fn png_declaring(w: u32, h: u32) -> Vec<u8> {
        let crc = |bytes: &[u8]| {
            let mut c = 0xFFFF_FFFFu32;
            for &b in bytes {
                c ^= u32::from(b);
                for _ in 0..8 {
                    c = (c >> 1) ^ (0xEDB8_8320 & (c & 1).wrapping_neg());
                }
            }
            !c
        };
        let mut ihdr = b"IHDR".to_vec();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13];
        png.extend_from_slice(&ihdr);
        png.extend_from_slice(&crc(&ihdr).to_be_bytes());
        // The first bytes of the pixel stream, which is where a decoder stops reading the header.
        let idat = b"IDAT\x78\x01".to_vec();
        png.extend_from_slice(&[0, 0, 0, 2]);
        png.extend_from_slice(&idat);
        png.extend_from_slice(&crc(&idat).to_be_bytes());
        png
    }

    #[test]
    fn an_image_over_the_pixel_limit_is_refused_from_its_header() {
        let bomb = png_declaring(60_000, 60_000);
        let e = decode_bounded(&bomb, 36_000_000).unwrap_err();
        assert!(e.contains("60000x60000") && e.contains("limit"), "{e}");
        // Within the limit it reaches the decoder, which fails on the missing data - not on the limit.
        let e = decode_bounded(&png_declaring(100, 100), 36_000_000).unwrap_err();
        assert!(!e.contains("limit"), "{e}");
    }

    #[test]
    fn a_decoded_image_over_the_limit_is_refused_for_every_format() {
        let img = tiny();
        let ppm = encode_p6(&img.px, img.w, img.h);
        assert!(decode_bounded(&ppm, 2).is_ok());
        assert!(decode_bounded(&ppm, 1).unwrap_err().contains("limit"));
    }

    #[test]
    fn png_magic_reaches_the_decoder() {
        // A bare signature is not a valid PNG; the point is that it is routed to
        // the PNG decoder and fails *there*, rather than being rejected as
        // "no decoder" the way `events::decode_pixels` does.
        let e = decode(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]).unwrap_err();
        assert!(e.contains("Png"), "expected a PNG decode error, got: {e}");
    }

    #[test]
    fn save_ppm_creates_the_directory_and_reloads() {
        // Only ever touch OUR OWN leaf dir, never `dir.parent()` - the parent
        // (`brain-imaging-codec-test`) is shared by every test in this file,
        // each in its own leaf subdirectory; removing the shared parent races
        // with and deletes a concurrently-running sibling test's directory
        // out from under it (`cargo test`'s default multi-threaded run hits
        // this - see `save_dispatches_on_extension`'s own correct `&dir`
        // pattern, which this used to diverge from).
        let dir = std::env::temp_dir().join("brain-imaging-codec-test/nested");
        let path = dir.join("tiny.ppm");
        let _ = std::fs::remove_dir_all(&dir);
        save_ppm(&path, &tiny()).unwrap();
        assert_eq!(load(&path).unwrap(), tiny());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_png_round_trips_through_the_image_decoder() {
        // See `save_ppm_creates_the_directory_and_reloads`'s comment - only
        // ever remove our own leaf dir, never the shared parent.
        let dir = std::env::temp_dir().join("brain-imaging-codec-test/png");
        let path = dir.join("tiny.png");
        let _ = std::fs::remove_dir_all(&dir);
        save_png(&path, &tiny()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']), "not a PNG file");
        assert_eq!(decode(&bytes).unwrap(), tiny());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_dispatches_on_extension() {
        let dir = std::env::temp_dir().join("brain-imaging-codec-test/dispatch");
        let _ = std::fs::remove_dir_all(&dir);

        let png = dir.join("out.png");
        save(&png, &tiny()).unwrap();
        assert!(std::fs::read(&png).unwrap().starts_with(&[0x89, b'P', b'N', b'G']), "out.png should be PNG bytes");

        // `.ppm` and an absent extension keep today's PPM behaviour - no
        // existing `--out foo.ppm`/`--out foo` caller's output changes. An
        // extension nobody supports is an error instead of a mislabelled P6;
        // `tests/save_formats.rs` owns that case and the byte-level regression
        // fence around these two.
        let ppm = dir.join("out.ppm");
        save(&ppm, &tiny()).unwrap();
        assert!(std::fs::read(&ppm).unwrap().starts_with(b"P6"), "out.ppm should still be P6");

        let no_ext = dir.join("out");
        save(&no_ext, &tiny()).unwrap();
        assert!(std::fs::read(&no_ext).unwrap().starts_with(b"P6"), "extensionless path should still be P6");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
