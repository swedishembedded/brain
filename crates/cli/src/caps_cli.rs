// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain caps` - the generalized capability interface's discovery half, and
//! [`run_do`] - its execution half, reached from the CLI as
//! `brain <architecture> <action>` (or `brain <action> <architecture>`; see
//! `crate::resolve`) for every architecture with no dedicated `_cli.rs`
//! module of its own. `run_do` itself is architecture-agnostic - it always
//! took `(model id, action name, ...flags)`, unchanged since the days it was
//! reachable as the standalone `brain do <model> <action>` command.
//!
//! `brain caps [model-or-arch-id] [--json]` lists what every supported
//! architecture can do (static manifests - no weights loaded); an id from
//! `brain_arch` resolves through the same table `crate::resolve` dispatches
//! with (`crate::resolve::model_for_arch`), so discovery and execution agree
//! on what names an architecture.
//!
//! Neither command knows anything model-specific: both go through
//! `capability::Registry`. A new model shows up here the moment it provides a
//! `capability::Manifest` (discovery) and a `Provider` (execution) - see
//! `crate::catalog` - ONE entry per model, so the list and the constructor
//! cannot drift apart (see that module's docs).

use std::io::IsTerminal;

use capability::{Manifest, ParamType};
use serde_json::Value;

// ---------------------------------------------------------------- brain caps

pub fn run_caps(argv: &[String]) -> i32 {
    let json_out = argv.iter().any(|a| a == "--json");
    // A `brain_arch` id (e.g. "scrfd") resolves through the same table
    // `brain <arch id> <action>` uses, so discovery and dispatch agree on
    // what names an architecture. Architectures dispatched through their own
    // `_cli.rs` module (`crate::resolve::ARCH_HANDLERS`, not
    // `model_for_arch`'s `ARCH_TO_MODEL`) still register a real catalog
    // entry for a handful of cases (qwen3, qwen35moe, qwen3omnimoe, lfm2,
    // qwen3tts, yolov8, zipdepth) -- their catalog id is exactly
    // `brain/<arch id>`, so that is the second candidate tried when the
    // first two miss. A filter that is neither is tried as a literal model
    // id unchanged.
    let filter = argv.iter().find(|a| !a.starts_with("--"));
    let candidates: Vec<String> = match filter {
        Some(m) => {
            let mut c = vec![crate::resolve::model_for_arch(m).map(str::to_string).unwrap_or_else(|| m.clone())];
            if brain_arch::by_id(m).is_some() {
                c.push(crate::resolve::catalog_id_for_arch_handler(m));
            }
            c
        }
        None => vec![],
    };
    let mans: Vec<Manifest> = catalog::manifests().into_iter().filter(|m| filter.is_none() || candidates.iter().any(|c| c == &m.model)).collect();
    if mans.is_empty() {
        eprintln!("no such model '{}' (try `brain caps`)", filter.map(String::as_str).unwrap_or_default());
        return 1;
    }
    if json_out {
        println!("{}", Value::Array(mans.iter().map(|m| m.to_json()).collect()));
        return 0;
    }
    print!("{}", render_listing(&mans, Style::for_stdout()));
    println!("run one with:  brain <architecture> <action> [--param value]… [--in name=path]… [--out name=path]…");
    0
}

/// How [`render_listing`] marks up model and action names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Style {
    Plain,
    Ansi,
}

impl Style {
    /// Colour only on a terminal, and never when `NO_COLOR` is set to a
    /// non-empty value (https://no-color.org).
    fn for_stdout() -> Style {
        Style::resolve(std::io::stdout().is_terminal(), std::env::var_os("NO_COLOR").as_deref())
    }

    fn resolve(stdout_is_terminal: bool, no_color: Option<&std::ffi::OsStr>) -> Style {
        if stdout_is_terminal && no_color.is_none_or(|v| v.is_empty()) {
            Style::Ansi
        } else {
            Style::Plain
        }
    }

    fn model(self, name: &str) -> String {
        match self {
            Style::Plain => name.to_string(),
            Style::Ansi => format!("\x1b[1m{name}\x1b[0m"),
        }
    }

    fn action(self, name: &str) -> String {
        match self {
            Style::Plain => name.to_string(),
            Style::Ansi => format!("\x1b[36m{name}\x1b[0m"),
        }
    }
}

/// The human-readable `brain caps` listing of `mans`.
fn render_listing(mans: &[Manifest], style: Style) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for m in mans {
        let _ = writeln!(out, "{} - {}", style.model(&m.model), m.summary);
        for a in &m.actions {
            let stream = if a.streaming { " (streaming)" } else { "" };
            let _ = writeln!(out, "  {}{stream}: {}", style.action(&a.name), a.summary);
            for p in &a.params {
                let req = if p.required { " [required]" } else { "" };
                let def = p.default.as_ref().map(|d| format!(" = {d}")).unwrap_or_default();
                let vals = match &p.ty {
                    ParamType::Enum(v) => format!(" {{{}}}", v.join("|")),
                    _ => String::new(),
                };
                let _ = writeln!(out, "      --{} <{}>{vals}{req}{def}  {}", p.name, p.ty.name(), p.help);
            }
            for b in a.inputs.iter() {
                let req = if b.required { " [required]" } else { "" };
                let _ = writeln!(out, "      --in {}=<{}>{req}  {}", b.name, b.media.name(), b.help);
            }
            for b in a.outputs.iter() {
                let _ = writeln!(out, "      --out {}=<{}>  {}", b.name, b.media.name(), b.help);
            }
        }
        out.push('\n');
    }
    out
}

// -------------------------------------------------------- generic dispatch

/// `brain <model> <action> [flags]`: one served action run on this machine,
/// through the catalog entry of `argv[0]` (`invoke::run_model`).
pub fn run_do(argv: &[String]) -> i32 {
    let Some(model) = argv.first().filter(|m| !m.starts_with("--")) else {
        eprintln!("usage: brain <architecture> <action> [--param value]… [--<input> PATH]… [--out name=PATH]…");
        return 2;
    };
    // A legacy short name (e.g. "mock") is a deprecation, not a second id: it
    // resolves to the canonical `brain/<name>` before dispatch, but is never
    // itself what gets registered or listed (see modelref::alias's module docs).
    let model = brain_modelref::alias::canonical(model).map(str::to_string).unwrap_or_else(|| model.clone());
    let Some(manifest) = catalog::manifests().into_iter().find(|m| m.model == model) else {
        eprintln!("brain: unknown model '{model}' (see `brain caps`)");
        return 1;
    };
    let build = |assembly: &capability::Assembly| catalog::provider_from_assembly(&model, assembly);
    let entry = invoke::Model { manifest, weights: catalog::resolver_spec_for(&model), build: &build };
    let shown = crate::resolve::arch_for_model(&model).unwrap_or(&model);
    invoke::run_model(&format!("brain {shown}"), &entry, &argv[1..])
}

// ---------------------------------------------------------------- built-in demo provider

#[cfg(test)]
mod tests {
    use super::*;

    /// `--out result=` (an output name with no path) is refused before the
    /// action runs.
    #[test]
    fn an_out_flag_with_no_path_is_refused_before_the_action_runs() {
        let argv: Vec<String> = ["brain/demo", "echo", "--text", "hi", "--out", "result="].into_iter().map(str::to_string).collect();
        assert_eq!(run_do(&argv), 2);
    }

    fn demo_manifests() -> Vec<Manifest> {
        catalog::manifests().into_iter().filter(|m| m.model == catalog::demo::MODEL).collect()
    }

    #[test]
    fn colour_is_for_a_terminal_that_has_not_opted_out() {
        use std::ffi::OsStr;
        assert_eq!(Style::resolve(true, None), Style::Ansi);
        assert_eq!(Style::resolve(true, Some(OsStr::new(""))), Style::Ansi, "an empty NO_COLOR is unset");
        assert_eq!(Style::resolve(true, Some(OsStr::new("1"))), Style::Plain);
        assert_eq!(Style::resolve(false, None), Style::Plain, "a pipe or file is never coloured");
    }

    #[test]
    fn the_plain_listing_has_no_escape_codes_and_the_ansi_one_only_adds_them() {
        let mans = demo_manifests();
        assert!(!mans.is_empty());
        let plain = render_listing(&mans, Style::Plain);
        let ansi = render_listing(&mans, Style::Ansi);
        assert!(!plain.contains('\x1b'), "{plain:?}");
        assert!(ansi.contains('\x1b'));
        let stripped: String = {
            let mut out = String::new();
            let mut chars = ansi.chars();
            while let Some(c) = chars.next() {
                if c == '\x1b' {
                    for e in chars.by_ref() {
                        if e == 'm' {
                            break;
                        }
                    }
                } else {
                    out.push(c);
                }
            }
            out
        };
        assert_eq!(stripped, plain);
    }
}
