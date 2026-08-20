// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `--trace-*` argument grammar, derived from [`crate::registry`].
//!
//! Shaped like the `brain` CLI's other GLOBAL option (`--device`): the flags
//! are recognised anywhere in the command line, consumed, and the remaining
//! tokens handed back to the subcommand dispatcher - so `brain ltxv t2v
//! --trace-ltxv 5 --prompt "..."` and `brain --trace-ltxv 5 ltxv t2v ...`
//! are the same command, and no subcommand parser needs to know these flags
//! exist.
//!
//! ```text
//! --trace-<family> <0-5>     sugar, one per registry family (--trace-gpu, --trace-ltxv)
//! --trace <family>=<level>   the generic form; repeatable, works for every
//!                            registered family with or without a sugar flag
//! --trace-format text|json   default text
//! --trace-output -|<path>    default - (stdout)
//! ```
//!
//! The sugar flags are *generated* from the registry rather than listed
//! here, which is what makes a new family a one-entry change: there is no
//! second list of flag names to keep in step.

use crate::{family, names, Config, Format, Output, MAX_LEVEL};

/// The environment variable that sets the same family levels as the flags,
/// for entry points that have no CLI in the loop (a test binary, a bench, a
/// library embedding). Same grammar as `--trace`, comma-separated:
/// `BRAIN_TRACE=ltxv=5,gpu=3`. Any `--trace*` flag on the command line wins
/// over it outright rather than merging, so a flag never has to fight an
/// exported variable.
pub const BRAIN_TRACE_ENV: &str = "BRAIN_TRACE";

/// Parse and REMOVE every `--trace*` flag from `argv`, returning the config
/// they describe plus the untouched remainder.
///
/// Reads [`BRAIN_TRACE_ENV`] - and this is the only place in the workspace
/// that does - when no flag named a family.
pub fn strip_args(argv: Vec<String>) -> Result<(Config, Vec<String>), String> {
    let mut cfg = Config::default();
    let mut rest = Vec::with_capacity(argv.len());
    let mut from_flags = false;
    let mut i = 0;
    while i < argv.len() {
        let tok = argv[i].as_str();
        // `--trace-format`/`--trace-output` are matched before the generic
        // `--trace-<family>` prefix below: they share that prefix but are not
        // families, and falling through would report them as unknown ones.
        let taken = match tok {
            "--trace-format" => {
                cfg.format = parse_format(&value(&argv, i, tok)?)?;
                true
            }
            "--trace-output" => {
                cfg.output = parse_output(&value(&argv, i, tok)?);
                true
            }
            "--trace" => {
                let v = value(&argv, i, tok)?;
                let (name, level) = parse_pair(&v)?;
                cfg.families.push((name, level));
                from_flags = true;
                true
            }
            _ => match tok.strip_prefix("--trace-") {
                Some(name) => {
                    let f = family(name).ok_or_else(|| unknown_family(name))?;
                    let level = parse_level(&value(&argv, i, tok)?)?;
                    cfg.families.push((f.name.to_string(), level));
                    from_flags = true;
                    true
                }
                None => false,
            },
        };
        if taken {
            i += 2;
        } else {
            rest.push(argv[i].clone());
            i += 1;
        }
    }

    if !from_flags {
        if let Ok(spec) = std::env::var(BRAIN_TRACE_ENV) {
            for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                let (name, level) = parse_pair(part).map_err(|e| format!("{BRAIN_TRACE_ENV}: {e}"))?;
                cfg.families.push((name, level));
            }
        }
    }
    Ok((cfg, rest))
}

/// `<family>=<level>`, the generic `--trace` form.
fn parse_pair(spec: &str) -> Result<(String, u8), String> {
    let (name, level) = spec
        .split_once('=')
        .ok_or_else(|| format!("--trace expects <family>=<level>, got {spec:?} (families: {})", family_list()))?;
    let f = family(name).ok_or_else(|| unknown_family(name))?;
    Ok((f.name.to_string(), parse_level(level)?))
}

/// A verbosity on the documented 0-5 scale. Out-of-range is an error, never a
/// clamp: a user who typed `--trace-ltxv 9` meant something this tool cannot
/// do, and silently giving them 5 hides that.
pub fn parse_level(s: &str) -> Result<u8, String> {
    let n: u8 = s
        .parse()
        .map_err(|_| format!("trace level must be an integer 0-{MAX_LEVEL} (0 = off, {MAX_LEVEL} = everything), got {s:?}"))?;
    if n > MAX_LEVEL {
        return Err(format!("trace level must be 0-{MAX_LEVEL} (0 = off, {MAX_LEVEL} = everything), got {n}"));
    }
    Ok(n)
}

fn parse_format(s: &str) -> Result<Format, String> {
    match s {
        "text" => Ok(Format::Text),
        "json" => Ok(Format::Json),
        other => Err(format!("--trace-format must be text or json, got {other:?}")),
    }
}

fn parse_output(s: &str) -> Output {
    if s == "-" {
        Output::Stdout
    } else {
        Output::File(s.into())
    }
}

fn value(argv: &[String], i: usize, flag: &str) -> Result<String, String> {
    argv.get(i + 1).cloned().ok_or_else(|| format!("{flag} needs a value"))
}

fn unknown_family(name: &str) -> String {
    format!("unknown trace family {name:?} (known: {})", family_list())
}

fn family_list() -> String {
    names().collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// EVERY registry family is reachable through its generated sugar flag
    /// and through the generic escape hatch, at every level on the scale -
    /// the gate that the two halves cannot drift as families are added.
    #[test]
    fn every_registry_family_is_reachable_both_ways() {
        for name in names() {
            for level in 0..=MAX_LEVEL {
                let (cfg, rest) = strip_args(argv(&["brain", &format!("--trace-{name}"), &level.to_string()])).expect("sugar flag parses");
                assert_eq!(cfg.level_of(name), level, "--trace-{name} {level}");
                assert_eq!(rest, argv(&["brain"]), "the trace flag must be consumed");

                let (cfg, _) = strip_args(argv(&["brain", "--trace", &format!("{name}={level}")])).expect("generic flag parses");
                assert_eq!(cfg.level_of(name), level, "--trace {name}={level}");
            }
        }
    }

    /// The flags are global: recognised wherever they appear, and everything
    /// else is handed back untouched and in order for the subcommand parser.
    #[test]
    fn trace_flags_are_stripped_from_anywhere_and_the_rest_survives() {
        let (cfg, rest) = strip_args(argv(&[
            "brain", "ltxv", "t2v", "--trace-ltxv", "5", "--prompt", "a cat", "--trace-format", "json", "--trace-output", "/dev/null", "--steps", "4",
        ]))
        .expect("parses");
        assert_eq!(cfg.level_of("ltxv"), 5);
        assert_eq!(cfg.format, Format::Json);
        assert_eq!(cfg.output, Output::File("/dev/null".into()));
        assert_eq!(rest, argv(&["brain", "ltxv", "t2v", "--prompt", "a cat", "--steps", "4"]));
    }

    /// Defaults, stated as a test because they are a user-facing contract:
    /// text to stdout, and nothing traced unless asked.
    #[test]
    fn defaults_are_text_to_stdout_and_off() {
        let (cfg, rest) = strip_args(argv(&["brain", "caps"])).expect("parses");
        assert_eq!(cfg.format, Format::Text);
        assert_eq!(cfg.output, Output::Stdout);
        assert!(cfg.is_off());
        assert_eq!(rest, argv(&["brain", "caps"]));
        let (cfg, _) = strip_args(argv(&["brain", "--trace-output", "-"])).expect("parses");
        assert_eq!(cfg.output, Output::Stdout);
    }

    /// A typo must be reported, not ignored: a silently-dropped
    /// `--trace-quen` produces a run with no trace output and no explanation
    /// for why, which is indistinguishable from the feature being broken.
    #[test]
    fn a_bad_family_level_or_format_is_an_error_not_a_default() {
        for bad in [
            argv(&["brain", "--trace-nosuch", "5"]),
            argv(&["brain", "--trace", "nosuch=5"]),
            argv(&["brain", "--trace", "ltxv"]),
            argv(&["brain", "--trace-ltxv", "9"]),
            argv(&["brain", "--trace-ltxv", "loud"]),
            argv(&["brain", "--trace-format", "yaml"]),
            argv(&["brain", "--trace-ltxv"]),
        ] {
            assert!(strip_args(bad.clone()).is_err(), "{bad:?} should not parse");
        }
    }
}
