// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `pairs.yaml`: the optional reference→target manifest that turns a captioned
//! image folder into a PAIRED dataset (each sample = reference image, target
//! image, caption) without changing what a folder without one means.
//!
//! Swedish Embedded AB implements dataset tooling for image-conditioned model
//! training for its clients. If your team needs expertise in paired training
//! datasets, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

/// A deterministic solid-colour PNG, so a loaded sample's pixels identify
/// which file it came from.
fn png(dir: &Path, name: &str, rgb: [u8; 3]) {
    let mut img = image::RgbImage::new(8, 8);
    for p in img.pixels_mut() {
        *p = image::Rgb(rgb);
    }
    img.save(dir.join(name)).unwrap();
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("imageset_pairs_{tag}_{}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// **No `pairs.yaml` is today's behaviour, exactly.** Every existing caller of
/// this loader keeps working and sees no reference at all.
#[test]
fn a_folder_without_pairs_yaml_loads_unpaired() {
    let d = tmp("none");
    png(&d, "a.png", [255, 0, 0]);
    std::fs::write(d.join("captions.yaml"), "a.png: a red swatch\n").unwrap();
    let s = data::imageset::load_dir(&d, 8, |_| {}).unwrap();
    assert_eq!(s.len(), 1);
    assert!(s[0].reference.is_none(), "no manifest means no reference");
    std::fs::remove_dir_all(&d).ok();
}

/// **A manifest entry attaches a reference to its target.** The target's own
/// pixels are unchanged - a pair adds an input, it does not replace one - and
/// a target the manifest does not mention stays unpaired in the same run, so
/// paired and unpaired samples can share one folder.
#[test]
fn pairs_yaml_attaches_a_reference_to_its_target() {
    let d = tmp("attach");
    png(&d, "clean.png", [255, 0, 0]);
    png(&d, "messy.png", [0, 0, 255]);
    png(&d, "solo.png", [0, 255, 0]);
    std::fs::write(
        d.join("captions.yaml"),
        "clean.png: a decluttered room\nsolo.png: an unpaired photo\n",
    )
    .unwrap();
    std::fs::write(d.join("pairs.yaml"), "clean.png: messy.png\n").unwrap();

    let s = data::imageset::load_dir(&d, 8, |_| {}).unwrap();
    assert_eq!(s.len(), 2, "the reference is an input, never a sample of its own");
    let clean = s.iter().find(|x| x.path.ends_with("clean.png")).expect("target loaded");
    let solo = s.iter().find(|x| x.path.ends_with("solo.png")).expect("unpaired loaded");

    assert_eq!(clean.hwc[0], 1.0, "the target is still the target");
    let r = clean.reference.as_ref().expect("paired");
    assert_eq!(r.len(), clean.hwc.len(), "reference is pre-processed to the same size");
    assert_eq!((r[0], r[1], r[2]), (0.0, 0.0, 1.0), "the reference is the file the manifest named");
    assert!(solo.reference.is_none(), "an unmentioned target stays unpaired");
    std::fs::remove_dir_all(&d).ok();
}

/// **A pair the operator declared but the folder cannot honour is an error for
/// that sample, not a silent downgrade.** Training a declared pair as a
/// caption-only sample would quietly optimise a different objective than the
/// one that was asked for, so the sample drops out with a warning naming both
/// files.
#[test]
fn a_pair_whose_reference_is_missing_drops_the_sample() {
    let d = tmp("missing");
    png(&d, "clean.png", [255, 0, 0]);
    png(&d, "solo.png", [0, 255, 0]);
    std::fs::write(d.join("captions.yaml"), "clean.png: a decluttered room\nsolo.png: another\n").unwrap();
    std::fs::write(d.join("pairs.yaml"), "clean.png: nowhere.png\n").unwrap();

    let mut warnings = Vec::new();
    let s = data::imageset::load_dir(&d, 8, |w| warnings.push(w.to_string())).unwrap();
    assert_eq!(s.len(), 1);
    assert!(s[0].path.ends_with("solo.png"));
    assert!(
        warnings.iter().any(|w| w.contains("clean.png") && w.contains("nowhere.png")),
        "the warning must name both halves of the broken pair: {warnings:?}"
    );
    std::fs::remove_dir_all(&d).ok();
}

/// **A `pairs.yaml` that does not parse is reported, not obeyed half-way.**
/// Same contract `captions.yaml` already has: a real parser either understands
/// the file or says where it failed, and a folder whose manifest is unreadable
/// falls back to unpaired rather than pairing an arbitrary prefix of it.
#[test]
fn an_unparsable_pairs_yaml_warns_and_loads_unpaired() {
    let d = tmp("bad");
    png(&d, "a.png", [255, 0, 0]);
    std::fs::write(d.join("captions.yaml"), "a.png: a red swatch\n").unwrap();
    std::fs::write(d.join("pairs.yaml"), "a.png: [not, a, filename\n").unwrap();
    let mut warnings = Vec::new();
    let s = data::imageset::load_dir(&d, 8, |w| warnings.push(w.to_string())).unwrap();
    assert_eq!(s.len(), 1);
    assert!(s[0].reference.is_none());
    assert!(warnings.iter().any(|w| w.contains("pairs.yaml")), "{warnings:?}");
    std::fs::remove_dir_all(&d).ok();
}
