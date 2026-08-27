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
    if bytes.starts_with(b"P6") {
        let (px, w, h) = decode_p6(bytes)?;
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
    let img = image::load_from_memory_with_format(bytes, fmt)
        .map_err(|e| format!("decoding {fmt:?}: {e}"))?
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    Rgb8::new(w, h, img.into_raw())
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
