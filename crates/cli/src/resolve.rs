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
//! handler (`gpt_cli::run_gpt`, …) already expects, so
//! nothing about their own verb parsing changes.
//!
//! An architecture reaches the CLI through the generic
//! [`capability::Provider`] dispatch in [`ARCH_TO_MODEL`]: `rest[0]` becomes
//! the capability ACTION name directly, and the rest of `rest` is handed to
//! [`crate::caps_cli::run_do`] verbatim -- the exact machinery that used to
//! sit behind `brain do <model> <action>`, just reached by architecture id
//! instead of by typing the canonical model id.
//!
//! `brain import <FILE>` (no architecture token) is the one standing
//! exception: when the second token isn't a recognized architecture id,
//! `import` falls through to the generic GGUF importer
//! ([`serving::gguf_import`]), which picks the architecture from the file's own
//! `general.architecture` header instead of from the command line.

use crate::{caps_cli, optimize_cli, prune_cli, quantize_cli};

/// Architectures reached generically through their
/// [`capability::Provider`]: `<arch id>, <canonical model id>`.
/// One row per architecture, each naming its own served model: the face pair
/// (`scrfd` detection, `arcface` identity embedding) are two crates and two
/// models, and `arcface embed`'s default path reaches the detector itself.
const ARCH_TO_MODEL: &[(&str, &str)] = &[
    // The decision family (`decide` and Laya, which has no arch row of its
    // own): `brain decide decide --state ... --questions ...` answers typed
    // questions from the best installed model, `brain decide models` lists
    // what is installed without loading it.
    ("decide", "brain/decision"),
    ("s3dit", "brain/s3dit"),
    ("fastvlm", "brain/fastvlm"),
    ("llava", "brain/llava"),
    ("qwen3vl", "brain/qwen3vl"),
    ("edgerock", "brain/edgerock"),
    ("ricursive", "brain/ricursive"),
    ("sam2", "brain/sam2"),
    ("lpips", "brain/lpips"),
    ("scrfd", "brain/scrfd"),
    ("florence2", "brain/florence2"),
    ("mirage", "Cactus-Compute/mirage"),
    ("arcface", "brain/arcface"),
    ("vqgan", "brain/vqgan"),
    ("codeformer", "brain/codeformer"),
    ("rrdbnet", "brain/rrdbnet"),
    ("clip", "brain/clip"),
    ("deepseek2ocr", "deepseek-ai/DeepSeek-OCR"),
    ("deepseekocr2", "deepseek-ai/DeepSeek-OCR-2"),
    ("nemotronasr", "brain/nemotronasr"),
    ("qwen3asr", "brain/qwen3asr"),
    ("chronos2", "brain/chronos2"),
    ("fincast", "brain/fincast"),
    ("kronos", "brain/kronos"),
    ("timesfm3", "brain/timesfm3"),
    ("horizon", "brain/horizon"),
    ("minimaxmusic3", "brain/minimaxmusic3"),
    ("cosyvoice", "brain/cosyvoice"),
    ("supir", "brain/supir"),
    ("sdxlunet", "brain/sdxl"),
    ("controlnet", "brain/sdxl-controlnet"),
    ("t5encoder", "brain/t5encoder"),
    ("moondream3", "brain/moondream3"),
    ("minimaxh3", "brain/minimaxh3"),
    // Same shape as `supir`: one action each (`flux1::caps`/`pulid::caps`
    // both declare exactly `text2image`), every other registration point
    // (catalog entry, residency adapter, D-Bus) already exists - only this
    // row was missing. `pulid`'s model id is `brain/flux1-pulid`, not
    // `brain/pulid` (`pulid::caps::MODEL`).
    ("flux1", "brain/flux1"),
    ("flux2", "brain/flux2-klein"),
    ("wan", "brain/wan"),
    ("ltxv", "brain/ltxv"),
    ("splat", "brain/splat"),
    ("zipdepth", "brain/zipdepth"),
    ("diamond", "brain/diamond"),
    ("worldmirror2", "brain/worldmirror2"),
    ("lfm2", "brain/lfm2"),
    ("deepseekvl", "brain/deepseekvl"),
    ("januspro", "brain/januspro"),
    ("minicpmv", "brain/minicpmv"),
    ("yolov8", "brain/yolov8"),
    ("gpt2", "brain/gpt2"),
    ("glmdsa", "brain/glm"),
    ("qwen3", "brain/qwen3"),
    ("toypid", "brain/toypid"),
    ("toymoe", "brain/toymoe"),
    ("huginn", "brain/huginn"),
    ("ouro", "brain/ouro"),
    ("qwen35", "brain/qwen35"),
    ("qwen35moe", "brain/qwen35moe"),
    ("qwen3omnimoe", "brain/qwen3omnimoe"),
    ("qwen3tts", "brain/qwen3tts"),
    ("pulid", "brain/flux1-pulid"),
    // No-weights utility models: listed by `brain caps` (via `catalog::MODELS`)
    // but, before this row existed, unreachable from the CLI - the same
    // listed-but-unreachable gap `catalog.rs`'s own module docs warn about
    // (`ai-forever/Real-ESRGAN` shipped with a manifest and a provider and was
    // still unreachable because only the residency list had been updated).
    ("imageops", "brain/imageops"),
    ("demo", "brain/demo"),
    // Same gap again: imgpipe was documented as CLI-reachable, and every
    // example used the long-dead `brain do brain/imgpipe run ...` spelling -
    // with no `brain_arch` row and no entry here, `brain imgpipe run ...`
    // was never actually reachable.
    ("imgpipe", "brain/imgpipe"),
];

enum Resolved {
    /// `arch` is `brain_arch`'s canonical id (`'static`, from the registry
    /// itself -- never borrowed from `argv`); `rest[0]` (if present) is the verb.
    Arch { arch: &'static str, rest: Vec<String> },
    /// `brain import <FILE> …` -- no architecture token, dispatched by the
    /// file's own GGUF header instead.
    ImportFile { rest: Vec<String> },
    /// `brain quantize <SRC> --out …` -- the export direction. Also has no
    /// architecture token, and unlike `import` needs none at all: the policy
    /// is structural plus whatever `--keep` names.
    QuantizeFile { rest: Vec<String> },
    /// `brain optimize <BASE> --dataset … --out …` -- the weight precision
    /// search; the architecture is a flag of the SDK call, not a token.
    Optimize { rest: Vec<String> },
    /// `brain finetune <BASE> --dataset … --out …` -- supervised fine-tuning through the SDK's generic trainers; the
    /// architecture is a flag of the SDK call, not a token.
    Finetune { rest: Vec<String> },
    /// `brain prune <BASE> --dataset … --out …` -- structured pruning.
    Prune { rest: Vec<String> },
    /// `brain lsa distill|eval|prefill …` -- latent sparse attention for any engine that captures its attention; the
    /// architecture is read from the checkpoint, not a token.
    Lsa { rest: Vec<String> },
    Unknown(String),
    Empty,
}

/// Every id [`dispatch_arch`] can actually route, from any of the three
/// tables this resolver draws on. `brain_arch::ARCHS` covers the common case
/// (real architectures with a crate, an HF/GGUF fetch story); this resolver's
/// OWN [`ARCH_TO_MODEL`] additionally lists a couple of
/// no-weights utility models (`imageops`, `demo`) that intentionally have no
/// `brain_arch::Arch` row at all - no crate, nothing to fetch - so gating on
/// `by_id` alone made them silently unreachable: `dispatch_arch` already
/// checks `ARCH_TO_MODEL`, but `resolve` never got that far because it never
/// recognized the token as an arch in the first place. Same "listed but
/// unreachable" bug class `catalog.rs`'s module docs warn about, just one
/// layer up (the CLI's OWN word, not the model catalog).
fn known_arch_id(s: &str) -> Option<&'static str> {
    if let Some(a) = brain_arch::by_id(s) {
        return Some(a.id);
    }
    if let Some((id, _)) = ARCH_TO_MODEL.iter().find(|(id, _)| *id == s) {
        return Some(id);
    }
    None
}

fn resolve(argv: &[String]) -> Resolved {
    let Some(first) = argv.first() else {
        return Resolved::Empty;
    };
    if let Some(id) = known_arch_id(first) {
        return Resolved::Arch { arch: id, rest: argv[1..].to_vec() };
    }
    if let Some(second) = argv.get(1) {
        if let Some(id) = known_arch_id(second) {
            let mut rest = vec![first.clone()];
            rest.extend_from_slice(&argv[2..]);
            return Resolved::Arch { arch: id, rest };
        }
    }
    if first == "import" {
        return Resolved::ImportFile { rest: argv[1..].to_vec() };
    }
    if first == "quantize" {
        return Resolved::QuantizeFile { rest: argv[1..].to_vec() };
    }
    if first == "optimize" {
        return Resolved::Optimize { rest: argv[1..].to_vec() };
    }
    if crate::args::canon_verb(first) == "finetune" {
        return Resolved::Finetune { rest: argv[1..].to_vec() };
    }
    if first == "prune" {
        return Resolved::Prune { rest: argv[1..].to_vec() };
    }
    if first == "lsa" {
        return Resolved::Lsa { rest: argv[1..].to_vec() };
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
        Resolved::ImportFile { rest } => crate::import_gguf_cli::run_import_gguf(&rest),
        Resolved::QuantizeFile { rest } => quantize_cli::run_quantize(&rest),
        Resolved::Optimize { rest } => optimize_cli::run_optimize(&rest),
        Resolved::Finetune { rest } => crate::finetune_cli::run_finetune(&rest),
        Resolved::Prune { rest } => prune_cli::run_prune(&rest),
        Resolved::Lsa { rest } => crate::lsa_cli::run_lsa(&rest),
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

/// The architecture id a user types for the served `model`, if it has one -
/// what `brain <arch> <action>` shows as its own name.
pub(crate) fn arch_for_model(model: &str) -> Option<&'static str> {
    ARCH_TO_MODEL.iter().find(|(_, m)| *m == model).map(|(id, _)| *id)
}

/// Architectures without a dedicated CLI module that `brain finetune` trains through the SDK's generic trainers.
const GENERIC_FINETUNE_ARCHS: &[&str] = &["ricursive"];

fn dispatch_arch(arch: &str, rest: Vec<String>) {
    // Skipped for `-h`/`--help`: help text must never block on a network
    // fetch (or hang, if `BRAIN_MODELS_DIR` points somewhere with no local
    // weights and the network is slow/unreachable) just to print itself.
    let wants_help = rest.iter().any(|a| a == "-h" || a == "--help");
    if !wants_help {
        // Weight-load progress for this command's own runs, named after the
        // architecture (`ltxv load ...`). Infra verbs keep their own output.
        crate::load_line::install(arch);
    }
    // `brain ricursive finetune ...` is `brain finetune --arch ricursive ...`: the generic trainers serve this
    // architecture, which has no dedicated CLI module of its own.
    if GENERIC_FINETUNE_ARCHS.contains(&arch) && rest.first().is_some_and(|v| crate::args::canon_verb(v) == "finetune") {
        let mut args = rest[1..].to_vec();
        args.extend(["--arch".to_string(), arch.to_string()]);
        crate::finetune_cli::run_finetune(&args);
    }
    if let Some((_, model)) = ARCH_TO_MODEL.iter().find(|(id, _)| *id == arch) {
        // `run_do` resolves the model's `--<role>` override flags and its
        // `Assembly` through the catalog's spec. It expects `[model, action, ...flags]`; `rest` is already
        // `[verb, ...flags]` with the verb doubling as the action name, so
        // prepending the model id is the whole translation.
        let mut do_args = vec![model.to_string()];
        do_args.extend(rest);
        let code = caps_cli::run_do(&do_args);
        // `run_do` may have built a real device (any architecture's action
        // that touches a GPU) - see `crate::drain_before_exit`'s own doc for
        // the measured segfault this avoids: this exact command
        // (`brain qwen3vl generate`) is the one that reproduced it.
        crate::drain_before_exit(code);
    }
    eprintln!("brain: architecture {arch:?} is registered but not reachable via the CLI yet (see `brain caps` and `brain serve`)");
    std::process::exit(1);
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
    fn a_no_weights_utility_model_with_no_brain_arch_row_still_resolves() {
        // `imageops`/`demo` are listed by `brain caps` (via `catalog::MODELS`)
        // but carry no `brain_arch::Arch` row (no crate, nothing to fetch) --
        // `known_arch_id` must fall through to `ARCH_TO_MODEL` for these, or
        // `brain imageops gradient` regresses to "unknown command" even
        // though `dispatch_arch` has always known how to route it.
        for id in ["imageops", "demo"] {
            assert!(brain_arch::by_id(id).is_none(), "{id} was added to brain_arch::ARCHS -- this test (and the ARCH_TO_MODEL fallback) can be deleted");
            let Resolved::Arch { arch, rest } = resolve(&s(&[id, "gradient"])) else {
                panic!("expected {id} to resolve as Resolved::Arch");
            };
            assert_eq!(arch, id);
            assert_eq!(rest, s(&["gradient"]));
        }
    }

    #[test]
    fn import_with_a_file_argument_is_not_mistaken_for_an_architecture() {
        let Resolved::ImportFile { rest } = resolve(&s(&["import", "model-Q4_K_M.gguf", "--out", "out.safetensors"])) else {
            panic!("expected Resolved::ImportFile");
        };
        assert_eq!(rest, s(&["model-Q4_K_M.gguf", "--out", "out.safetensors"]));
    }

    #[test]
    fn import_with_a_real_arch_id_routes_to_that_archs_own_import_action() {
        let Resolved::Arch { arch, rest } = resolve(&s(&["import", "qwen35", "--source", "dir", "--dest", "f"])) else {
            panic!("expected Resolved::Arch (qwen35 serves its own import action)");
        };
        assert_eq!(arch, "qwen35");
        assert_eq!(rest, s(&["import", "--source", "dir", "--dest", "f"]));
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

    /// A served model whose catalog entry resolves its weights through an
    /// architecture's spec is a finished port: `brain <arch> <verb>` must
    /// reach it. Without a row the command prints "registered but not
    /// reachable via the CLI", and nothing else says the wiring was skipped.
    #[test]
    fn every_catalog_model_with_an_arch_spec_is_reachable_from_the_cli() {
        let unreachable: Vec<String> = catalog::models()
            .iter()
            .filter_map(|m| m.spec.map(|(arch, _)| arch))
            .filter(|arch| brain_arch::by_id(arch).is_some())
            .filter(|arch| !ARCH_TO_MODEL.iter().any(|(id, _)| id == arch))
            .map(str::to_string)
            .collect();
        assert!(unreachable.is_empty(), "served with a resolver spec but unreachable as `brain <arch> <verb>`: {unreachable:?}");
    }

    /// `ARCH_TO_MODEL` ids with no `brain_arch::Arch` row: no-weights utility
    /// models (`imgpipe` composes OTHER architectures' own weights) with no
    /// crate-with-a-port story of their own, so no row makes sense for them.
    const NO_ARCH_ROW: &[&str] = &["imageops", "demo", "imgpipe"];

    #[test]
    fn every_arch_to_model_id_is_a_real_registry_entry() {
        // `NO_ARCH_ROW` is the documented exception (see its own doc comment):
        // every other `ARCH_TO_MODEL` id is a real architecture.
        for (id, _) in ARCH_TO_MODEL {
            if NO_ARCH_ROW.contains(id) {
                assert!(brain_arch::by_id(id).is_none(), "{id:?} was added to brain_arch::ARCHS -- remove it from NO_ARCH_ROW");
                continue;
            }
            assert!(brain_arch::by_id(id).is_some(), "{id:?} in ARCH_TO_MODEL has no brain_arch row (if intentional, add it to NO_ARCH_ROW above)");
        }
    }

    /// qwen35's verbs are its served actions, reached generically: the
    /// removed `infer`/`sample-adapter` spellings are not among them.
    #[test]
    fn qwen35_verbs_are_its_served_actions() {
        assert_eq!(model_for_arch("qwen35"), Some(qwen35::caps::MODEL));
        assert_eq!(arch_for_model(qwen35::caps::MODEL), Some("qwen35"));
        let actions: Vec<String> = qwen35::caps::manifest().actions.into_iter().map(|a| a.name).collect();
        for v in ["generate", "finetune", "sample_adapter", "import"] {
            assert!(actions.iter().any(|a| a == v), "qwen35 should serve {v:?}: {actions:?}");
        }
        for v in ["infer", "sample-adapter", "t2i"] {
            assert!(!actions.iter().any(|a| a == v), "qwen35 has no {v:?} action");
        }
    }

    /// `pulid`'s `ARCH_TO_MODEL` row - weights-free, no `Bundle`/`Flux1` build
    /// involved. Pins the model id (`brain/flux1-pulid`, not `brain/pulid`)
    /// and that `text2image` resolves against `pulid::caps::manifest()`.
    #[test]
    fn pulid_resolves_to_its_manifest_model_id() {
        assert_eq!(model_for_arch("pulid"), Some(pulid::caps::MODEL));
    }
}
