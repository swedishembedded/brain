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

/// Write an [`Rgb8`], choosing the encoder from `path`'s extension:
/// `.png`/`.PNG` -> [`save_png`], anything else (including no extension,
/// `.ppm`, or an extension nobody asked for) -> [`save_ppm`]. This is the
/// front door every CLI `--out name=path` write should go through: the
/// extension the caller typed is the format that lands on disk, rather than
/// always being a P6 regardless of what the path says (the previous
/// behaviour of every `imaging::save_ppm` call site).
pub fn save(path: impl AsRef<Path>, img: &Rgb8) -> Result<(), String> {
    let path = path.as_ref();
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("png") => save_png(path, img),
        _ => save_ppm(path, img),
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
        let dir = std::env::temp_dir().join("brain-imaging-codec-test/nested");
        let path = dir.join("tiny.ppm");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        save_ppm(&path, &tiny()).unwrap();
        assert_eq!(load(&path).unwrap(), tiny());
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn save_png_round_trips_through_the_image_decoder() {
        let dir = std::env::temp_dir().join("brain-imaging-codec-test/png");
        let path = dir.join("tiny.png");
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
        save_png(&path, &tiny()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']), "not a PNG file");
        assert_eq!(decode(&bytes).unwrap(), tiny());
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn save_dispatches_on_extension() {
        let dir = std::env::temp_dir().join("brain-imaging-codec-test/dispatch");
        let _ = std::fs::remove_dir_all(&dir);

        let png = dir.join("out.png");
        save(&png, &tiny()).unwrap();
        assert!(std::fs::read(&png).unwrap().starts_with(&[0x89, b'P', b'N', b'G']), "out.png should be PNG bytes");

        // An unrecognised (or absent) extension keeps today's PPM behaviour -
        // no existing `--out foo.ppm`/`--out foo` caller's output changes.
        let ppm = dir.join("out.ppm");
        save(&ppm, &tiny()).unwrap();
        assert!(std::fs::read(&ppm).unwrap().starts_with(b"P6"), "out.ppm should still be P6");

        let no_ext = dir.join("out");
        save(&no_ext, &tiny()).unwrap();
        assert!(std::fs::read(&no_ext).unwrap().starts_with(b"P6"), "extensionless path should still be P6");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
