// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The architecture-namespace CLI resolver: `brain <verb> <arch> …` and
//! `brain <arch> <verb> …` are the SAME invocation.
//!
//! One lookup, both orders: given `brain <a> <b> …`, if `b` is a known
//! `brain_arch` id, `a` is the verb and `b` names the architecture -- the
//! reverse of the direct form (`brain <arch> <verb> …`, where `a` itself is
//! the id). Both land on identical `(arch, rest)`, where `rest[0]` is always
//! the verb -- the exact shape every existing per-architecture `_cli.rs`
//! handler (`gpt_cli::run_gpt`, `yolo_cli::run_yolo`, …) already expects, so
//! nothing about their own verb parsing changes.
//!
//! An architecture reaches the CLI one of two ways:
//! - a dedicated handler in [`ARCH_HANDLERS`], for architectures with their
//!   own `_cli.rs` module and verb vocabulary (`train`/`infer`/`import`/…,
//!   including whatever long-tail verbs that module already supports -- this
//!   resolver does not enumerate or restrict them);
//! - the generic [`capability::Provider`] dispatch in [`ARCH_TO_MODEL`], for
//!   architectures with no dedicated CLI module: `rest[0]` becomes the
//!   capability ACTION name directly, and the rest of `rest` is handed to
//!   [`crate::caps_cli::run_do`] verbatim -- the exact machinery that used to
//!   sit behind `brain do <model> <action>`, just reached by architecture id
//!   instead of by typing the canonical model id.
//!
//! `brain import <FILE>` (no architecture token) is the one standing
//! exception: when the second token isn't a recognized architecture id,
//! `import` falls through to the generic GGUF importer
//! ([`crate::gguf_import`]), which picks the architecture from the file's own
//! `general.architecture` header instead of from the command line.

use crate::{caps_cli, gguf_import};

type Handler = fn(&[String]);

/// Architectures reachable through their own dedicated CLI module. Order
/// matches `AGENTS.md`'s model grouping; add a row here when a new
/// architecture gets its own `_cli.rs`.
const ARCH_HANDLERS: &[(&str, Handler)] = &[
    ("gpt2", crate::gpt_cli::run_gpt),
    ("qwen3", crate::qwen_cli::run_qwen),
    ("qwen35moe", crate::qwen35moe_cli::run_qwen35moe),
    ("qwen3omnimoe", crate::omni_cli::run_omni),
    ("glmdsa", crate::glm_cli::run_glm),
    ("lfm2", crate::lfm_cli::run_lfm),
    ("qwen3tts", crate::tts_cli::run_tts),
    ("yolov8", crate::yolo_cli::run_yolo),
    ("zipdepth", crate::depth_cli::run_depth),
    ("flux2", crate::flux2_cli::run_flux2),
    ("worldmirror2", crate::mirror_cli::run_mirror),
    ("splat", crate::splat_cli::run_splat),
    // wm_cli's own `--arch`/`--model` flags (not this resolver) pick
    // fake-vs-diamond within `play`/`import`/`export` -- diamond is its one
    // real served architecture, so that's the id this dispatches from.
    ("diamond", crate::wm_cli::run_wm),
    ("toypid", crate::pid_cli::run_pid),
    ("toymoe", run_toymoe),
];

/// The bare sparse-MoE toy model used to be three unrelated top-level
/// commands (`brain train`, `brain eval`, `brain generate`, no shared verb
/// dispatch). Folded into one handler so it fits [`ARCH_HANDLERS`]'s shape;
/// `toymoe::run_train`/`run_eval`/`run_generate` themselves are untouched.
fn run_toymoe(args: &[String]) {
    match args.first().map(String::as_str) {
        Some("train") => toymoe::run_train(&args[1..]),
        Some("eval") => toymoe::run_eval(&args[1..]),
        Some("infer" | "generate" | "gen" | "sample") => toymoe::run_generate(),
        other => eprintln!("usage: brain toymoe <train|eval|infer> ...  (got {other:?})"),
    }
}

/// Architectures with no dedicated CLI module, reached generically through
/// their [`capability::Provider`] instead: `<arch id>, <canonical model id>`.
/// One row per architecture, each naming its own served model: the face pair
/// (`scrfd` detection, `arcface` identity embedding) are two crates and two
/// models, and `arcface embed`'s default path reaches the detector itself.
const ARCH_TO_MODEL: &[(&str, &str)] = &[
    ("s3dit", "brain/z-image"),
    ("fastvlm", "brain/fastvlm"),
    ("qwen3vl", "brain/qwenvl"),
    ("sam2", "brain/sam2"),
    ("scrfd", "brain/scrfd"),
    ("arcface", "brain/arcface"),
    ("vqgan", "brain/vqgan"),
    ("codeformer", "brain/restore"),
    ("rrdbnet", "brain/upscale"),
    ("clip", "brain/clip"),
    ("deepseek2ocr", "deepseek-ai/DeepSeek-OCR"),
    ("nemotronasr", "brain/nemotron"),
    ("qwen3asr", "brain/qwen-asr"),
    ("chronos2", "brain/chronos2"),
    ("fincast", "brain/fincast"),
    ("kronos", "brain/kronos"),
];

enum Resolved {
    /// `arch` is `brain_arch`'s canonical id (`'static`, from the registry
    /// itself -- never borrowed from `argv`); `rest[0]` (if present) is the verb.
    Arch { arch: &'static str, rest: Vec<String> },
    /// `brain import <FILE> …` -- no architecture token, dispatched by the
    /// file's own GGUF header instead.
    ImportFile { rest: Vec<String> },
    Unknown(String),
    Empty,
}

fn resolve(argv: &[String]) -> Resolved {
    let Some(first) = argv.first() else {
        return Resolved::Empty;
    };
    if let Some(a) = brain_arch::by_id(first) {
        return Resolved::Arch { arch: a.id, rest: argv[1..].to_vec() };
    }
    if let Some(second) = argv.get(1) {
        if let Some(a) = brain_arch::by_id(second) {
            let mut rest = vec![first.clone()];
            rest.extend_from_slice(&argv[2..]);
            return Resolved::Arch { arch: a.id, rest };
        }
    }
    if first == "import" {
        return Resolved::ImportFile { rest: argv[1..].to_vec() };
    }
    Resolved::Unknown(first.clone())
}

/// Entry point: `argv` is everything after the `brain` binary name (so
/// `argv[0]` is the first real token, e.g. `"train"` or `"gpt2"`). Exits the
/// process on every path except a successfully dispatched, void-returning
/// architecture handler.
pub fn dispatch(argv: &[String], help: &str) {
    match resolve(argv) {
        Resolved::Arch { arch, rest } => dispatch_arch(arch, rest),
        Resolved::ImportFile { rest } => gguf_import::run_import_gguf(&rest),
        Resolved::Unknown(tok) => {
            eprintln!("brain: unknown command '{tok}'\n");
            print!("{help}");
            std::process::exit(2);
        }
        Resolved::Empty => print!("{help}"),
    }
}

/// The canonical model id an architecture without its own dedicated CLI
/// module serves under, if any -- what [`dispatch_arch`]'s generic path
/// translates through, and what lets `brain caps <arch id>` (in
/// `crate::caps_cli`) resolve an arch id the same way `brain <arch id>
/// <action>` already does, rather than requiring the model id spelled out.
pub(crate) fn model_for_arch(arch: &str) -> Option<&'static str> {
    ARCH_TO_MODEL.iter().find(|(id, _)| *id == arch).map(|(_, model)| *model)
}

fn dispatch_arch(arch: &str, rest: Vec<String>) {
    if let Some((_, handler)) = ARCH_HANDLERS.iter().find(|(id, _)| *id == arch) {
        let rest = maybe_inject_default_weights(arch, rest);
        return handler(&rest);
    }
    if let Some((_, model)) = ARCH_TO_MODEL.iter().find(|(id, _)| *id == arch) {
        // `run_do` expects `[model, action, ...flags]`; `rest` is already
        // `[verb, ...flags]` with the verb doubling as the action name, so
        // prepending the model id is the whole translation.
        let mut do_args = vec![model.to_string()];
        do_args.extend(rest);
        std::process::exit(caps_cli::run_do(&do_args));
    }
    eprintln!("brain: architecture {arch:?} is registered but not reachable via the CLI yet (see `brain caps` and `brain serve`)");
    std::process::exit(1);
}

/// For an `infer`-shaped verb with no `--weights` already given, auto-fetch
/// the architecture's default checkpoint
/// ([`brain_arch::Arch::default_ref`], via
/// [`crate::supply::ensure_default_weights`]) and inject `--weights <path>`
/// (plus `--tokenizer <path>`, when the fetched checkpoint has one and
/// `--tokenizer` was not already given) -- what makes `brain infer zipdepth
/// --in image=x.jpg` (no flags beyond the input) resolve a real checkpoint on
/// its own. Passes `rest` through completely unchanged for every other verb,
/// for an architecture with no `default_ref`, or when `--weights` is already
/// present -- this never silently overrides an explicit flag with a fetched
/// one.
fn maybe_inject_default_weights(arch: &str, rest: Vec<String>) -> Vec<String> {
    let is_infer = rest.first().is_some_and(|v| crate::args::canon_verb(v) == "infer");
    if !is_infer || rest.iter().any(|a| a == "--weights") {
        return rest;
    }
    match crate::supply::ensure_default_weights(arch) {
        Ok(got) => {
            let mut rest = rest;
            rest.push("--weights".to_string());
            rest.push(got.weights);
            if !rest.iter().any(|a| a == "--tokenizer") {
                if let Some(tokenizer) = got.tokenizer {
                    rest.push("--tokenizer".to_string());
                    rest.push(tokenizer);
                }
            }
            rest
        }
        Err(e) => {
            eprintln!("brain: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn both_word_orders_resolve_to_the_identical_rest() {
        let a = resolve(&s(&["gpt2", "train", "data/calculator"]));
        let b = resolve(&s(&["train", "gpt2", "data/calculator"]));
        let (Resolved::Arch { arch: arch_a, rest: rest_a }, Resolved::Arch { arch: arch_b, rest: rest_b }) = (a, b) else {
            panic!("both must resolve to Resolved::Arch");
        };
        assert_eq!(arch_a, "gpt2");
        assert_eq!(arch_a, arch_b);
        assert_eq!(rest_a, rest_b);
        assert_eq!(rest_a, s(&["train", "data/calculator"]));
    }

    #[test]
    fn a_bare_arch_id_with_no_verb_resolves_with_empty_rest() {
        let Resolved::Arch { arch, rest } = resolve(&s(&["zipdepth"])) else {
            panic!("expected Resolved::Arch");
        };
        assert_eq!(arch, "zipdepth");
        assert!(rest.is_empty());
    }

    #[test]
    fn import_with_a_file_argument_is_not_mistaken_for_an_architecture() {
        let Resolved::ImportFile { rest } = resolve(&s(&["import", "model-Q4_K_M.gguf", "--out", "out.safetensors"])) else {
            panic!("expected Resolved::ImportFile");
        };
        assert_eq!(rest, s(&["model-Q4_K_M.gguf", "--out", "out.safetensors"]));
    }

    #[test]
    fn import_with_a_real_arch_id_routes_to_that_archs_own_import_verb() {
        let Resolved::Arch { arch, rest } = resolve(&s(&["import", "qwen3", "--hf", "dir", "--out", "f"])) else {
            panic!("expected Resolved::Arch (qwen3 has a dedicated handler with its own import verb)");
        };
        assert_eq!(arch, "qwen3");
        assert_eq!(rest, s(&["import", "--hf", "dir", "--out", "f"]));
    }

    #[test]
    fn an_arch_specific_long_tail_verb_works_in_both_orders_with_no_registry_entry() {
        // "calib" is not a standard verb anywhere in this module -- the
        // resolver never enumerates verbs, only architecture ids, so any
        // word works as a verb as long as the OTHER token is a real id.
        let a = resolve(&s(&["zipdepth", "calib"]));
        let b = resolve(&s(&["calib", "zipdepth"]));
        let (Resolved::Arch { rest: rest_a, .. }, Resolved::Arch { rest: rest_b, .. }) = (a, b) else {
            panic!("both must resolve to Resolved::Arch");
        };
        assert_eq!(rest_a, rest_b);
        assert_eq!(rest_a, s(&["calib"]));
    }

    #[test]
    fn an_unrecognized_first_token_with_no_matching_second_token_is_unknown() {
        assert!(matches!(resolve(&s(&["totally-bogus"])), Resolved::Unknown(_)));
        assert!(matches!(resolve(&s(&["totally-bogus", "also-bogus"])), Resolved::Unknown(_)));
    }

    #[test]
    fn empty_argv_is_empty() {
        assert!(matches!(resolve(&s(&[])), Resolved::Empty));
    }

    #[test]
    fn every_arch_handlers_id_is_a_real_registry_entry() {
        for (id, _) in ARCH_HANDLERS {
            assert!(brain_arch::by_id(id).is_some(), "{id:?} in ARCH_HANDLERS has no brain_arch row");
        }
    }

    #[test]
    fn every_arch_to_model_id_is_a_real_registry_entry() {
        for (id, _) in ARCH_TO_MODEL {
            assert!(brain_arch::by_id(id).is_some(), "{id:?} in ARCH_TO_MODEL has no brain_arch row");
        }
    }

    /// ARCH_HANDLERS and ARCH_TO_MODEL partition disjointly -- an id in both
    /// would mean the generic capability path is silently unreachable for it
    /// (ARCH_HANDLERS is checked first), which is exactly the kind of
    /// drift a table like this is supposed to make impossible to miss.
    #[test]
    fn arch_handlers_and_arch_to_model_do_not_overlap() {
        for (id, _) in ARCH_HANDLERS {
            assert!(!ARCH_TO_MODEL.iter().any(|(m, _)| m == id), "{id:?} is in both ARCH_HANDLERS and ARCH_TO_MODEL");
        }
    }
}
