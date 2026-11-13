// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--autofetch` is a flag of the command that fetches, written anywhere on
//! the command line: a model's action takes it as a run-level flag (never as
//! an action param it would refuse), and a verb with no use for it says so.

use std::process::Command;

fn brain(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_brain")).args(args).output().expect("run brain")
}

#[test]
fn a_model_action_takes_autofetch_before_or_after_its_verb() {
    let dir = std::env::temp_dir().join(format!("brain-autofetch-flag-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (i, args) in [["--autofetch", "imageops", "gradient"], ["imageops", "gradient", "--autofetch"]].iter().enumerate() {
        let out = dir.join(format!("g{i}.ppm"));
        let target = format!("image={}", out.display());
        let mut argv = args.to_vec();
        argv.extend(["--width", "16", "--height", "16", "--out", &target]);
        let run = brain(&argv);
        assert!(run.status.success(), "{argv:?}: {}", String::from_utf8_lossy(&run.stderr));
        assert!(out.is_file(), "{argv:?} wrote no image");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_verb_without_a_fetch_says_autofetch_has_no_effect() {
    let run = brain(&["--autofetch", "caps"]);
    assert!(run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("--autofetch has no effect"), "{}", String::from_utf8_lossy(&run.stderr));
}
