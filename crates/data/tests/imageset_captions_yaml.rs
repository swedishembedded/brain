// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec for the `captions.yaml` caption file: the multiline (block-scalar)
//! form that makes a long, hand-editable caption possible, and the guarantee
//! that every single-line form that already worked still parses to the exact
//! same bytes.
//!
//! Swedish Embedded AB implements dataset tooling and training-data pipelines
//! for its clients. If your team needs expertise in machine-learning dataset
//! curation then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::PathBuf;

use data::imageset::{read_captions_yaml, write_captions_yaml};

/// A private scratch directory, named per test so a parallel run cannot
/// collide with another test's file.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("imageset_yaml_{}_{}", std::process::id(), name));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn parse(text: &str) -> BTreeMap<String, String> {
    let d = scratch(&format!("p{:x}", text.len()));
    let f = d.join("captions.yaml");
    std::fs::write(&f, text).unwrap();
    let out = read_captions_yaml(&f, &mut |_| {});
    std::fs::remove_dir_all(&d).ok();
    out
}

/// **The regression gate.** Every single-line spelling the loader documented
/// before block scalars existed must still parse to byte-identical values.
/// This corpus is the one from `imageset`'s own in-file test, plus the forms
/// its doc comment advertises.
#[test]
fn single_line_forms_are_unchanged() {
    let caps = parse(
        "# subject cat\n\
         a.png: a photo of sks cat\n\
         b.jpg: \"a photo of sks cat, closeup\"  # trailing comment\n\
         c.png: prompt with a \"#hashtag\" inside\n\
         \n\
         d.png: 'single quoted, with: a colon inside'\n\
         \"e.png\": quoted key\n",
    );
    assert_eq!(caps["a.png"], "a photo of sks cat");
    assert_eq!(caps["b.jpg"], "a photo of sks cat, closeup");
    assert_eq!(caps["c.png"], "prompt with a \"#hashtag\" inside");
    assert_eq!(caps["d.png"], "single quoted, with: a colon inside");
    assert_eq!(caps["e.png"], "quoted key");
    assert_eq!(caps.len(), 5);
}

/// A literal block scalar (`|`) keeps its own line breaks, and `#` inside the
/// body is caption text, not a comment - a detailed caption will contain both.
#[test]
fn literal_block_scalar_keeps_line_breaks() {
    let caps = parse(
        "a.png: |\n  \
           first line\n  \
           second line # not a comment\n\
         b.png: a single line\n",
    );
    assert_eq!(caps["a.png"], "first line\nsecond line # not a comment\n");
    assert_eq!(caps["b.png"], "a single line");
    assert_eq!(caps.len(), 2);
}

/// The three chomping indicators, which decide what happens to the trailing
/// newline: `-` strips it, bare clips to one, `+` keeps them all.
#[test]
fn block_scalar_chomping_indicators() {
    let caps = parse(
        "strip.png: |-\n  one\n  two\n\
         clip.png: |\n  one\n  two\n\n\n\
         keep.png: |+\n  one\n  two\n\n\n\
         last.png: tail\n",
    );
    assert_eq!(caps["strip.png"], "one\ntwo");
    assert_eq!(caps["clip.png"], "one\ntwo\n");
    assert_eq!(caps["keep.png"], "one\ntwo\n\n\n");
    assert_eq!(caps["last.png"], "tail");
}

/// A folded block scalar (`>`) joins wrapped lines with a space and turns a
/// blank line into a paragraph break - the form a human reaches for when they
/// want a long caption soft-wrapped in the editor.
#[test]
fn folded_block_scalar_joins_wrapped_lines() {
    let caps = parse(
        "a.png: >-\n  \
           a wide sunlit room\n  \
           with a rattan chair\n\n  \
           and a jute rug\n",
    );
    assert_eq!(caps["a.png"], "a wide sunlit room with a rattan chair\nand a jute rug");
}

/// Blank lines and deeper indentation inside a literal block belong to the
/// value; the block ends at the first line indented no further than its key.
#[test]
fn block_scalar_keeps_blank_lines_and_relative_indent() {
    let caps = parse(
        "a.png: |-\n  \
           para one\n\
         \n  \
           para two\n    \
             indented further\n\
         b.png: after\n",
    );
    assert_eq!(caps["a.png"], "para one\n\npara two\n  indented further");
    assert_eq!(caps["b.png"], "after");
}

/// **The round trip.** Write a caption set, read it back, and require exact
/// string equality including every embedded newline. This is the property the
/// labeler depends on: what it wrote is what the trainer reads, and what a
/// human edits in between stays readable.
#[test]
fn round_trip_is_exact_for_multiline_captions() {
    let mut caps = BTreeMap::new();
    caps.insert(
        "room-01.jpg".to_string(),
        "A wide-angle photograph of a living room in bohemian style.\n\
         The low rattan sofa is layered with a rust-orange linen throw and three\n\
         kilim cushions in ochre and deep red; a jute rug covers the oak floor.\n\
         \n\
         Warm low-angle afternoon light enters from a tall window on the left,\n\
         casting long shadows across the #1 focal wall."
            .to_string(),
    );
    caps.insert("room-02.jpg".to_string(), "A single-line caption, in bohemian style.".to_string());
    caps.insert(
        "room-03.webp".to_string(),
        "Trailing newline preserved, in bohemian style.\n".to_string(),
    );
    // A line of nothing but spaces: treating it as blank would silently eat
    // them, and a caption is only round-trip-safe if EVERY byte survives.
    caps.insert("room-04.png".to_string(), "first, in bohemian style\n   \nlast".to_string());

    let d = scratch("roundtrip");
    let f = d.join("captions.yaml");
    write_captions_yaml(&f, &caps).unwrap();

    // The writer must actually emit block scalars - a writer that folded the
    // caption onto one line would round-trip only by destroying the newlines.
    let text = std::fs::read_to_string(&f).unwrap();
    assert!(text.contains("room-01.jpg: |"), "expected a block scalar header, got:\n{text}");

    let back = read_captions_yaml(&f, &mut |w| panic!("unexpected warning: {w}"));
    assert_eq!(back, caps, "round trip changed the captions");

    // Mutation-sensitivity: a writer or reader that ignored its input would
    // pass an equality check against a fixture but not against a value the
    // test derives from the file it just wrote.
    assert_eq!(back["room-01.jpg"].lines().count(), 6);
    assert!(back["room-01.jpg"].contains("#1 focal wall"));
    assert!(back["room-03.webp"].ends_with('\n'));

    std::fs::remove_dir_all(&d).ok();
}

/// The writer must survive a caption whose first line is indented - the case
/// where inferring the block indent from the body would silently eat spaces.
#[test]
fn round_trip_is_exact_when_a_caption_line_is_indented() {
    let mut caps = BTreeMap::new();
    caps.insert("a.png".to_string(), "   leading spaces, in bohemian style\nsecond line".to_string());
    caps.insert("b.png".to_string(), "plain".to_string());

    let d = scratch("indent");
    let f = d.join("captions.yaml");
    write_captions_yaml(&f, &caps).unwrap();
    let back = read_captions_yaml(&f, &mut |w| panic!("unexpected warning: {w}"));
    assert_eq!(back, caps);
    std::fs::remove_dir_all(&d).ok();
}

/// A block-scalar caption reaches the trainer through the real loader, not
/// only through the parser - `load_dir` is the consumer that matters.
#[test]
fn load_dir_reads_a_block_scalar_caption() {
    let d = scratch("loaddir");
    let mut img = image::RgbImage::new(8, 6);
    for (x, _y, p) in img.enumerate_pixels_mut() {
        *p = image::Rgb([(x * 30) as u8, 90, 180]);
    }
    img.save(d.join("x.png")).unwrap();
    std::fs::write(d.join("captions.yaml"), "x.png: |-\n  line one, in bohemian style\n  line two\n").unwrap();

    let s = data::imageset::load_dir(&d, 16, |_| {}).unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].prompt, "line one, in bohemian style\nline two");
    std::fs::remove_dir_all(&d).ok();
}
