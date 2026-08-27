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

/// Constructs a hand-rolled line scanner gets wrong, and a real parser does
/// not. Each of these was either silently mis-parsed or silently dropped by the
/// `split_once(':')` scanner this module used to have; the point of depending
/// on a YAML implementation is that they are simply correct.
#[test]
fn constructs_a_line_scanner_would_mis_parse() {
    // An escape inside a double-quoted scalar is a real newline, not the two
    // characters backslash-n. The old scanner only stripped the quotes.
    let caps = parse("a.png: \"line one\\nline two\"\n");
    assert_eq!(caps["a.png"], "line one\nline two");

    // A colon inside a quoted KEY. Splitting on the first colon cut this file
    // name in half.
    let caps = parse("\"a:b.png\": caption here\n");
    assert_eq!(caps["a:b.png"], "caption here");

    // An apostrophe is not an opening quote, so the `#` after it still starts a
    // comment. The old scanner toggled its in-quote flag on the apostrophe and
    // kept the comment as caption text.
    let caps = parse("a.png: a cat's paw # note\n");
    assert_eq!(caps["a.png"], "a cat's paw");

    // Anchors and aliases: shared boilerplate across a caption set.
    let caps = parse("base: &c a shared caption\na.png: *c\n");
    assert_eq!(caps["a.png"], "a shared caption");

    // A leading document marker, as any tool that emits YAML will write.
    let caps = parse("---\na.png: one\n");
    assert_eq!(caps["a.png"], "one");

    // CRLF, as a file edited on Windows arrives.
    let caps = parse("a.png: one\r\nb.png: |-\r\n  two\r\n  three\r\n");
    assert_eq!(caps["a.png"], "one");
    assert_eq!(caps["b.png"], "two\nthree");
}

/// A file that is not YAML at all is REPORTED, not silently treated as empty
/// and not a panic. The old scanner had no failure mode: every line it could
/// not understand was skipped, so a wholly malformed file looked exactly like
/// an uncaptioned folder.
#[test]
fn a_malformed_caption_file_is_reported_with_its_location() {
    let d = scratch("malformed");
    let f = d.join("captions.yaml");
    // A tab where YAML requires spaces - the classic hand-edit accident.
    std::fs::write(&f, "a.png: |\n\tone\n").unwrap();
    let mut warnings = Vec::new();
    let caps = read_captions_yaml(&f, &mut |w| warnings.push(w.to_string()));
    assert!(caps.is_empty());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("tab"), "the message must name the problem: {}", warnings[0]);
    assert!(warnings[0].contains("line 2"), "and where it is: {}", warnings[0]);
    std::fs::remove_dir_all(&d).ok();
}

/// `captions.jsonl` is the scripted-override lane, and it is deserialized into
/// [`data::imageset::CaptionLine`] rather than picked apart by hand. A line
/// that does not fit the schema is reported and skipped; it never becomes a
/// caption, and it never aborts the folder.
#[test]
fn jsonl_overrides_are_typed_and_bad_lines_are_reported() {
    let d = scratch("jsonl");
    let mut img = image::RgbImage::new(4, 4);
    for p in img.pixels_mut() {
        *p = image::Rgb([10, 20, 30]);
    }
    img.save(d.join("a.png")).unwrap();
    img.save(d.join("b.png")).unwrap();
    std::fs::write(d.join("captions.yaml"), "a.png: from yaml\nb.png: also from yaml\n").unwrap();
    std::fs::write(
        d.join("captions.jsonl"),
        concat!(
            "{\"file\":\"a.png\",\"prompt\":\"OVERRIDDEN\"}\n",
            "\n",
            "{\"file\":\"b.png\",\"prompt\":\"\"}\n",
            "{\"file\":\"b.png\"}\n",
            "not json at all\n",
        ),
    )
    .unwrap();

    let mut warnings = Vec::new();
    let s = data::imageset::load_dir(&d, 16, |w| warnings.push(w.to_string())).unwrap();
    let by_name: BTreeMap<&str, &str> =
        s.iter().map(|x| (x.path.file_name().unwrap().to_str().unwrap(), x.prompt.as_str())).collect();

    assert_eq!(by_name["a.png"], "OVERRIDDEN", "a well-formed override must win");
    assert_eq!(by_name["b.png"], "also from yaml", "an empty prompt must NOT overwrite a real caption");
    assert_eq!(warnings.len(), 3, "empty prompt, missing field, and non-JSON: {warnings:?}");
    assert!(warnings.iter().all(|w| w.starts_with("captions.jsonl:")), "{warnings:?}");
    // The schema is what produces this message. A loose `Value` lookup would
    // default the absent field to an empty string and report only that it was
    // empty, losing the name of what is actually wrong with the line.
    assert!(
        warnings.iter().any(|w| w.contains("missing field") && w.contains("prompt")),
        "a line missing a schema field must be named as such: {warnings:?}"
    );
    std::fs::remove_dir_all(&d).ok();
}

/// The schema is what makes a structurally wrong caption file an ERROR rather
/// than a quiet omission. Walking a free-form YAML value and keeping the
/// string-shaped entries would drop these silently, and a caption that
/// vanished between the editor and the trainer is the hardest kind of dataset
/// bug to notice.
#[test]
fn a_value_that_is_not_a_caption_is_an_error_not_a_silent_drop() {
    let d = scratch("schema");
    let f = d.join("captions.yaml");

    for (name, text, want) in [
        ("nested mapping", "a.png:\n  nested: thing\nb.png: fine\n", "invalid type: map"),
        ("sequence", "a.png:\n  - one\n  - two\nb.png: fine\n", "invalid type: sequence"),
        ("top-level sequence", "- one\n- two\n", "expected a map"),
    ] {
        std::fs::write(&f, text).unwrap();
        let mut warnings = Vec::new();
        let caps = read_captions_yaml(&f, &mut |w| warnings.push(w.to_string()));
        assert!(caps.is_empty(), "{name}: a file that does not fit the schema yields no captions");
        assert_eq!(warnings.len(), 1, "{name}: {warnings:?}");
        assert!(warnings[0].contains(want), "{name}: expected {want:?} in {}", warnings[0]);
    }
    std::fs::remove_dir_all(&d).ok();
}
