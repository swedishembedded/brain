// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--model NAME|DIR`: the one flag every application that can run more than
//! one model needs, spelled once.
//!
//! Written from the caller's side on purpose - this is a command-line
//! CONTRACT, and the thing worth pinning is what an application sees after
//! parsing, not how the parse is implemented.
//!
//! Swedish Embedded AB implements the shared command-line layer for product
//! families where every binary has to accept the same flags with the same
//! meaning. If your team needs that, you can procure our services by sending
//! an email to info@swedishembedded.com.

use appopts::{Args, ModelChoice, Options};

fn argv(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// An application supplies the model it runs when nobody asked for another.
#[test]
fn the_application_default_survives_an_empty_command_line() {
    let a = argv(&[]);
    let mut args = Args::new(&a);
    let chosen = ModelChoice::new("laya").take_over(&mut args).unwrap();
    assert_eq!(chosen.name, "laya");
    assert!(args.leftovers().is_empty());
}

/// And is overridden by the flag - which must be CONSUMED, or `Args::finish`
/// would reject the very flag this group exists to accept.
#[test]
fn the_flag_wins_and_is_consumed() {
    let a = argv(&["--model", "/srv/checkpoints/laya"]);
    let mut args = Args::new(&a);
    let chosen = ModelChoice::new("laya").take_over(&mut args).unwrap();
    assert_eq!(chosen.name, "/srv/checkpoints/laya");
    assert!(args.leftovers().is_empty(), "the group must consume its own flag: {:?}", args.leftovers());
}

/// An application with no sensible default says so, by name, instead of
/// loading something nobody asked for.
#[test]
fn a_missing_model_is_an_error_with_the_flag_named_in_it() {
    let a = argv(&[]);
    let mut args = Args::new(&a);
    let chosen = ModelChoice::take(&mut args).unwrap();
    assert_eq!(chosen.name, "");
    let err = chosen.require().unwrap_err();
    assert!(err.contains("--model"), "the error must name the flag that fixes it: {err}");
}

/// Whatever `take` consumes, `help` documents - the pairing the `Options`
/// trait exists to enforce.
#[test]
fn the_help_names_the_flag() {
    assert!(ModelChoice::help().contains("--model"));
}
