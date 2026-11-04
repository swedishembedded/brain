// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What lands on disk when a caller names a path.
//!
//! [`imaging::save`] is the front door every CLI `--out name=path` write goes
//! through, so the thing worth pinning is not "did a file appear" but "are the
//! bytes in it the format the extension promised". Three separate claims, each
//! with its own failure mode:
//!
//! * **Routing** - the extension the caller typed picks the encoder, and an
//!   extension nobody supports is an error rather than a P6 wearing someone
//!   else's suffix.
//! * **Regression** - `.png` and `.ppm` bytes are compared against literals
//!   captured from the implementation that predates JPEG routing. A cosine or a
//!   "starts_with(b\"P6\")" would not notice a re-encode; only the whole byte
//!   string does.
//! * **Validity and fidelity of the JPEG** - the markers and an *independent*
//!   decoder say it is a JPEG, and cosine plus rel_l2 say the pixels survived.
//!   JPEG is lossy, so an equality assertion on pixels is not available and a
//!   scale-blind cosine on its own is not enough.

use imaging::Rgb8;
use brain_testutil::parity::{compare, rel_l2};

/// A deterministic 7x5 image, small enough that its whole PNG and PPM encodings
/// fit in this file as literals.
fn tiny() -> Rgb8 {
    let (w, h) = (7u32, 5u32);
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            px.push((x * 31 + y * 7) as u8);
            px.push((x * 3 + y * 53) as u8);
            px.push(255u8.wrapping_sub((x * 11 + y * 17) as u8));
        }
    }
    Rgb8::new(w, h, px).unwrap()
}

/// A 64x48 stand-in for a photograph: smooth low-frequency colour ramps (what a
/// DCT reproduces well) plus a hard-edged bright rectangle (what it does not).
/// A pure gradient would flatter the encoder; flat noise would libel it.
fn photo() -> Rgb8 {
    let (w, h) = (64u32, 48u32);
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let inside = (16..40).contains(&x) && (12..30).contains(&y);
            let bump = if inside { 90.0 } else { 0.0 };
            px.push((30.0 + 200.0 * fx + bump).clamp(0.0, 255.0) as u8);
            px.push((40.0 + 170.0 * fy).clamp(0.0, 255.0) as u8);
            px.push((200.0 - 150.0 * fx * fy + bump).clamp(0.0, 255.0) as u8);
        }
    }
    Rgb8::new(w, h, px).unwrap()
}

fn as_f32(img: &Rgb8) -> Vec<f32> {
    img.px.iter().map(|&b| b as f32).collect()
}

fn workdir(leaf: &str) -> std::path::PathBuf {
    // Only ever touch our own leaf directory - the shared parent is used
    // concurrently by every other test in this file.
    let dir = std::env::temp_dir().join("brain-imaging-save-formats").join(leaf);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// The exact PNG `save` wrote before `.jpg` routing existed.
const PNG_BEFORE: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 7, 0, 0, 0, 5, 8, 2, 0, 0, 0, 6, 248, 97,
    143, 0, 0, 0, 121, 73, 68, 65, 84, 120, 1, 1, 110, 0, 145, 255, 0, 0, 0, 255, 31, 3, 244, 62, 6, 233, 93, 9, 222,
    124, 12, 211, 155, 15, 200, 186, 18, 189, 0, 7, 53, 238, 38, 56, 227, 69, 59, 216, 100, 62, 205, 131, 65, 194, 162,
    68, 183, 193, 71, 172, 0, 14, 106, 221, 45, 109, 210, 76, 112, 199, 107, 115, 188, 138, 118, 177, 169, 121, 166,
    200, 124, 155, 0, 21, 159, 204, 52, 162, 193, 83, 165, 182, 114, 168, 171, 145, 171, 160, 176, 174, 149, 207, 177,
    138, 0, 28, 212, 187, 59, 215, 176, 90, 218, 165, 121, 221, 154, 152, 224, 143, 183, 227, 132, 214, 230, 121, 216,
    139, 56, 15, 246, 43, 103, 201, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

/// The exact PPM `save` wrote before `.jpg` routing existed.
const PPM_BEFORE: &[u8] = &[
    80, 54, 10, 55, 32, 53, 10, 50, 53, 53, 10, 0, 0, 255, 31, 3, 244, 62, 6, 233, 93, 9, 222, 124, 12, 211, 155, 15,
    200, 186, 18, 189, 7, 53, 238, 38, 56, 227, 69, 59, 216, 100, 62, 205, 131, 65, 194, 162, 68, 183, 193, 71, 172,
    14, 106, 221, 45, 109, 210, 76, 112, 199, 107, 115, 188, 138, 118, 177, 169, 121, 166, 200, 124, 155, 21, 159, 204,
    52, 162, 193, 83, 165, 182, 114, 168, 171, 145, 171, 160, 176, 174, 149, 207, 177, 138, 28, 212, 187, 59, 215, 176,
    90, 218, 165, 121, 221, 154, 152, 224, 143, 183, 227, 132, 214, 230, 121,
];

/// The regression fence. `save` is the front door for every CLI image write, so
/// the bytes it produced for the two formats that already worked have to be the
/// same bytes, not merely the same format.
#[test]
fn png_and_ppm_bytes_are_unchanged() {
    let dir = workdir("unchanged");
    let png = dir.join("a.png");
    imaging::save(&png, &tiny()).unwrap();
    assert_eq!(std::fs::read(&png).unwrap(), PNG_BEFORE, "the PNG encoding changed");

    let ppm = dir.join("a.ppm");
    imaging::save(&ppm, &tiny()).unwrap();
    assert_eq!(std::fs::read(&ppm).unwrap(), PPM_BEFORE, "the PPM encoding changed");

    // No extension at all is P6 too, and it is the same P6.
    let bare = dir.join("a");
    imaging::save(&bare, &tiny()).unwrap();
    assert_eq!(std::fs::read(&bare).unwrap(), PPM_BEFORE, "the extensionless encoding changed");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Every routing case, including the ones that differ only by letter case: an
/// `ext == "jpg"` comparison passes the lowercase tests and silently mislabels
/// `PHOTO.JPG`.
#[test]
fn save_routes_on_the_extension() {
    let dir = workdir("routing");
    let img = tiny();

    for name in ["a.png", "a.PNG", "a.Png"] {
        let p = dir.join(name);
        imaging::save(&p, &img).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert!(b.starts_with(&[0x89, b'P', b'N', b'G']), "{name} should be PNG bytes");
    }

    for name in ["a.jpg", "a.JPG", "a.jpeg", "a.JPEG", "a.Jpg"] {
        let p = dir.join(name);
        imaging::save(&p, &img).unwrap();
        let b = std::fs::read(&p).unwrap();
        assert!(b.starts_with(&[0xFF, 0xD8]), "{name} should be JPEG bytes, got {:02x?}", &b[..4.min(b.len())]);
    }

    for name in ["a.ppm", "a.PPM", "a"] {
        let p = dir.join(name);
        imaging::save(&p, &img).unwrap();
        assert!(std::fs::read(&p).unwrap().starts_with(b"P6"), "{name} should be P6 bytes");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// An extension nobody supports must not become a P6 wearing that suffix. The
/// error has to name the extension (so the typo is visible) and list what is
/// supported (so the fix is visible), and no file may be left behind.
#[test]
fn an_unsupported_extension_is_an_error_and_writes_nothing() {
    let dir = workdir("unsupported");
    for (name, ext) in [("a.webp", "webp"), ("a.bmp", "bmp"), ("a.tiff", "tiff"), ("a.txt", "txt")] {
        let p = dir.join(name);
        let e = imaging::save(&p, &tiny()).unwrap_err();
        assert!(e.contains(ext), "error should name the extension {ext}, got: {e}");
        for supported in [".png", ".jpg", ".ppm"] {
            assert!(e.contains(supported), "error should list {supported}, got: {e}");
        }
        assert!(!p.exists(), "{name} must not be written at all");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The bytes have to be a JPEG according to something other than the encoder
/// that produced them. Three independent witnesses: the container markers, the
/// workspace's own decoder (`zune-jpeg`, a different implementation from
/// `image`'s encoder), and `ffprobe` when the box has it.
#[test]
fn the_jpeg_is_a_jpeg_to_an_independent_decoder() {
    let dir = workdir("validity");
    let path = dir.join("a.jpg");
    let img = photo();
    imaging::save(&path, &img).unwrap();
    let bytes = std::fs::read(&path).unwrap();

    assert_eq!(&bytes[..2], &[0xFF, 0xD8], "no SOI marker");
    assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9], "no EOI marker");

    let back = imaging::decode(&bytes).unwrap();
    assert_eq!((back.w, back.h), (img.w, img.h), "decoded dimensions differ");

    match std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0"])
        .arg(&path)
        .output()
    {
        Ok(out) if out.status.success() => {
            let got = String::from_utf8_lossy(&out.stdout).trim().to_string();
            assert_eq!(got, format!("mjpeg,{},{}", img.w, img.h), "ffprobe disagrees about the file");
        }
        _ => eprintln!("ffprobe is not on PATH - the external decode check did not run"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// JPEG is lossy, so the gate is a distance, not an equality. Cosine alone is
/// scale-invariant - an encoder that halved every sample would still score
/// 1.0000 - so rel_l2 is asserted alongside it.
///
/// The bounds are for the default quality. At that setting the quantisation
/// tables are scaled well below the point where ringing is visible, and the
/// encoder writes 4:4:4 (no chroma subsampling), so a few percent of relative
/// energy is the whole budget; the numbers a real run produces sit comfortably
/// under it. The bound is a floor that catches a broken port - a wrong channel
/// order, a transposed block, a quality collapsed to single digits - not a
/// transcription of one measurement.
#[test]
fn a_jpeg_round_trip_stays_close() {
    let dir = workdir("roundtrip");
    let path = dir.join("a.jpg");
    let img = photo();
    imaging::save(&path, &img).unwrap();
    let back = imaging::decode(&std::fs::read(&path).unwrap()).unwrap();

    let (want, got) = (as_f32(&img), as_f32(&back));
    let (cos, _max) = compare(&got, &want);
    let rel = rel_l2(&got, &want);
    assert!(cos >= 0.999, "cosine {cos:.6} below the round-trip floor");
    assert!(rel <= 0.03, "rel_l2 {rel:.6} above the round-trip ceiling");

    let _ = std::fs::remove_dir_all(&dir);
}

/// The quality parameter has to reach the encoder, and the default `save` picks
/// has to be a high one. Two things could go wrong silently: `save_jpeg`
/// ignoring `quality` altogether (every file the same size), and `save`
/// defaulting to something like libjpeg's 75 or lower while the constant claims
/// otherwise. Size is the observable that separates them - a lossy encoder that
/// honours quality spends more bytes for more of them - and fidelity is what
/// says which end of the scale the default sits at.
#[test]
fn quality_reaches_the_encoder_and_the_default_is_a_high_one() {
    let dir = workdir("quality");
    let img = photo();
    let want = as_f32(&img);

    let mut sizes = Vec::new();
    for q in [10u8, 50, 92] {
        let p = dir.join(format!("q{q}.jpg"));
        imaging::save_jpeg(&p, &img, q).unwrap();
        sizes.push(std::fs::read(&p).unwrap().len());
    }
    assert!(sizes[0] < sizes[1] && sizes[1] < sizes[2], "quality must change the encoding, got sizes {sizes:?}");

    // The low end has to be visibly worse than the round-trip gate allows,
    // otherwise that gate's bound proves nothing about the default.
    let low = imaging::decode(&std::fs::read(dir.join("q10.jpg")).unwrap()).unwrap();
    assert!(rel_l2(&as_f32(&low), &want) > 0.03, "quality 10 should breach the round-trip bound");

    // What `save` writes with no quality argument is what `JPEG_QUALITY` says.
    let via_save = dir.join("default.jpg");
    imaging::save(&via_save, &img).unwrap();
    let explicit = dir.join("explicit.jpg");
    imaging::save_jpeg(&explicit, &img, imaging::JPEG_QUALITY).unwrap();
    assert_eq!(
        std::fs::read(&via_save).unwrap(),
        std::fs::read(&explicit).unwrap(),
        "`save` must encode at JPEG_QUALITY"
    );
    const { assert!(imaging::JPEG_QUALITY >= 85, "the default must be a keep-quality setting, not a bandwidth one") };

    let _ = std::fs::remove_dir_all(&dir);
}
