// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An action's command line, built from its [`ActionSpec`] alone.
//!
//! Every param becomes a typed `--<name>` (with a `--dashed-name` alias for
//! an underscored one); every input blob becomes `--<name> PATH`, the
//! shorthand for `--in name=PATH`; `--out name=PATH` names where an output
//! blob goes; `--json` prints the scalar outputs as one JSON object. A param
//! whose flag is not given is not set, so the action's own default applies.

use capability::{ActionSpec, Invocation, ParamType};
use clap::{Arg, ArgAction, ArgMatches, Command};
use serde_json::{json, Value};

/// The command line of `action` on `model`, named `program` in its help.
pub fn command(program: &str, spec: &ActionSpec) -> Command {
    let mut cmd = Command::new(program.to_string()).no_binary_name(true).about(spec.summary.clone());
    for p in &spec.params {
        let mut help = p.help.clone();
        if let Some(d) = &p.default {
            help.push_str(&format!(" [default: {d}]"));
        }
        let mut arg = with_dashed_alias(Arg::new(p.name.clone()).long(p.name.clone()).help(help), &p.name);
        if p.ty == ParamType::Bool {
            // `--flag` is `true`, and `--flag false` turns a default-on switch
            // off: without the optional value a param whose default is `true`
            // could not be cleared from a command line at all.
            arg = arg.action(ArgAction::Set).value_name("BOOL").num_args(0..=1).default_missing_value("true").value_parser(["true", "false", "1", "0"]);
        } else {
            arg = arg.action(ArgAction::Set).value_name(p.ty.name().to_uppercase());
            if let ParamType::Enum(vals) = &p.ty {
                arg = arg.value_parser(vals.clone());
            }
            if p.required && p.default.is_none() {
                arg = arg.required(true);
            }
        }
        cmd = cmd.arg(arg);
    }
    for b in spec.inputs.iter().filter(|b| has_blob_flag(spec, &b.name)) {
        let req = if b.required { " [required]" } else { "" };
        let help = format!("input {} (same as --in {}=PATH){req}: {}", b.media.name(), b.name, b.help);
        cmd = cmd.arg(with_dashed_alias(Arg::new(blob_arg_id(&b.name)).long(b.name.clone()).action(ArgAction::Set).value_name("PATH").help(help), &b.name));
    }
    let in_help = if spec.inputs.is_empty() {
        "named binary input, e.g. image=in.png".to_string()
    } else {
        format!("named binary input ({})", spec.inputs.iter().map(|b| format!("{}=<{}>", b.name, b.media.name())).collect::<Vec<_>>().join(", "))
    };
    cmd = cmd.arg(Arg::new("in").long("in").action(ArgAction::Append).value_name("NAME=PATH").help(in_help));
    if !has_out_param(spec) {
        let outs = spec.outputs.iter().map(|b| format!("{}=<{}>", b.name, b.media.name())).collect::<Vec<_>>().join(", ");
        cmd = cmd.arg(Arg::new("out").long("out").action(ArgAction::Append).value_name("NAME=PATH").help(format!("write an output blob to a file ({outs})")));
    }
    cmd.arg(Arg::new("json").long("json").action(ArgAction::SetTrue).help("print the scalar outputs as one JSON object"))
}

/// What a parsed command line asks for: the params as an [`Invocation`]
/// (blobs not yet loaded), the input files and the output files.
#[derive(Debug)]
pub struct Parsed {
    pub invocation: Invocation,
    /// `(input name, path)`, in the order given.
    pub inputs: Vec<(String, String)>,
    /// `(output name, path)`, in the order given.
    pub outputs: Vec<(String, String)>,
    pub json: bool,
}

/// Read `matches` (from [`command`] over the same `spec`) into a [`Parsed`].
/// A malformed `--in`/`--out` value is an error naming it, before anything
/// runs.
pub fn parse(spec: &ActionSpec, matches: &ArgMatches) -> Result<Parsed, String> {
    let mut parsed = Parsed { invocation: Invocation::new(), inputs: Vec::new(), outputs: Vec::new(), json: matches.get_flag("json") };
    for p in &spec.params {
        if let Some(v) = matches.get_one::<String>(&p.name) {
            parsed.invocation = std::mem::take(&mut parsed.invocation).set(&p.name, coerce(&p.ty, v));
        }
    }
    for v in matches.get_many::<String>("in").unwrap_or_default() {
        parsed.inputs.push(name_path("--in", v)?);
    }
    for b in spec.inputs.iter().filter(|b| has_blob_flag(spec, &b.name)) {
        if let Some(path) = matches.get_one::<String>(&blob_arg_id(&b.name)) {
            parsed.inputs.push((b.name.clone(), path.clone()));
        }
    }
    if !has_out_param(spec) {
        for v in matches.get_many::<String>("out").unwrap_or_default() {
            let (name, path) = name_path("--out", v)?;
            if !spec.outputs.iter().any(|b| b.name == name) {
                let known = spec.outputs.iter().map(|b| b.name.as_str()).collect::<Vec<_>>().join(", ");
                return Err(format!("--out {v:?}: the action has no output '{name}' (it has: {known})"));
            }
            parsed.outputs.push((name, path));
        }
    }
    Ok(parsed)
}

fn name_path(flag: &str, v: &str) -> Result<(String, String), String> {
    match v.split_once('=') {
        Some((name, path)) if !name.is_empty() && !path.is_empty() => Ok((name.to_string(), path.to_string())),
        _ => Err(format!("{flag} wants NAME=PATH, got {v:?}")),
    }
}

/// A command-line word as the JSON value `ty` expects. A word that does not
/// parse is passed on as a string, so the action's own validation names the
/// param and its type rather than this layer guessing.
pub fn coerce(ty: &ParamType, s: &str) -> Value {
    match ty {
        ParamType::Int => s.parse::<i64>().map(|n| json!(n)).unwrap_or_else(|_| json!(s)),
        ParamType::Float => s.parse::<f64>().map(|x| json!(x)).unwrap_or_else(|_| json!(s)),
        ParamType::Bool => json!(s == "true" || s == "1"),
        _ => json!(s),
    }
}

/// The clap id of the `--<name> PATH` shorthand for the input blob `name`.
fn blob_arg_id(name: &str) -> String {
    format!("blob:{name}")
}

/// An action with a param named `out` (a directory to write, say) owns the
/// `--out` flag; its output blobs then have no file flag.
fn has_out_param(spec: &ActionSpec) -> bool {
    spec.params.iter().any(|p| p.name == "out")
}

/// Whether the input blob `name` also gets a `--<name> PATH` flag: not when a
/// param or a reserved flag already owns the word.
fn has_blob_flag(spec: &ActionSpec, name: &str) -> bool {
    !matches!(name, "in" | "out" | "json") && !spec.params.iter().any(|p| p.name == name)
}

/// `--held-out` beside `--held_out`, as command lines are written.
fn with_dashed_alias(arg: Arg, name: &str) -> Arg {
    match name.contains('_') {
        true => arg.alias(name.replace('_', "-")),
        false => arg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{BlobSpec, Media, ParamSpec};

    fn spec() -> ActionSpec {
        ActionSpec::new("act", "summary")
            .param(ParamSpec::new("held_out", ParamType::Int, "rows held out").default(json!(4)))
            .param(ParamSpec::new("prompt", ParamType::Str, "what to make").required())
            .input(BlobSpec::new("image", Media::Image, "the picture").required())
            .output(BlobSpec::new("mask", Media::Mask, "the mask"))
    }

    fn parsed(argv: &[&str]) -> Result<Parsed, String> {
        let s = spec();
        let m = command("t", &s).try_get_matches_from(argv).map_err(|e| e.to_string())?;
        parse(&s, &m)
    }

    /// A flag left out is a param left out: the action, not this layer,
    /// supplies the default. Inputs come from either spelling.
    #[test]
    fn unset_flags_leave_the_default_to_the_action() {
        let p = parsed(&["--prompt", "x", "--image", "a.png"]).unwrap();
        assert_eq!(p.invocation.get_str("prompt").as_deref(), Some("x"));
        assert!(p.invocation.get_i64("held_out").is_none());
        assert_eq!(p.inputs, vec![("image".to_string(), "a.png".to_string())]);
        let p = parsed(&["--prompt", "x", "--held-out", "2", "--in", "image=b.png"]).unwrap();
        assert_eq!(p.invocation.get_i64("held_out"), Some(2));
        assert_eq!(p.inputs, vec![("image".to_string(), "b.png".to_string())]);
    }

    /// A required param with no default is refused by the parser, and an
    /// output the action does not make is refused before anything runs.
    #[test]
    fn missing_required_and_unknown_outputs_are_refused() {
        assert!(parsed(&["--image", "a.png"]).is_err());
        let e = parsed(&["--prompt", "x", "--out", "picture=o.png"]).unwrap_err();
        assert!(e.contains("no output 'picture'") && e.contains("mask"), "{e}");
        assert!(parsed(&["--prompt", "x", "--out", "mask="]).unwrap_err().contains("NAME=PATH"));
    }

    /// A param owns its word: an input blob of the same name is reachable
    /// only as `--in name=PATH`, and a param named `out` (a directory) takes
    /// `--out` from the output blobs.
    #[test]
    fn a_param_owns_its_flag_over_a_blob_of_the_same_name() {
        let s = ActionSpec::new("train", "summary")
            .param(ParamSpec::new("out", ParamType::Str, "a directory"))
            .param(ParamSpec::new("image", ParamType::Str, "a param sharing a blob's name"))
            .input(BlobSpec::new("image", Media::Text, "owned by the param"))
            .input(BlobSpec::new("held_out", Media::Text, "subjects"));
        let m = command("t", &s).try_get_matches_from(["--out", "dir", "--image", "p", "--in", "image=x.txt", "--held-out", "h.jsonl"]).unwrap();
        let p = parse(&s, &m).unwrap();
        assert_eq!(p.invocation.get_str("out").as_deref(), Some("dir"));
        assert_eq!(p.invocation.get_str("image").as_deref(), Some("p"));
        assert_eq!(p.inputs, vec![("image".to_string(), "x.txt".to_string()), ("held_out".to_string(), "h.jsonl".to_string())]);
        assert!(p.outputs.is_empty());
    }
}
