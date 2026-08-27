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

use crate::{caps_cli, gguf_import, quantize_cli};

type Handler = fn(&[String]);

/// Architectures reachable through their own dedicated CLI module. Order
/// matches `AGENTS.md`'s model grouping; add a row here when a new
/// architecture gets its own `_cli.rs`.
const ARCH_HANDLERS: &[(&str, Handler)] = &[
    ("gpt2", crate::gpt_cli::run_gpt),
    ("qwen3", crate::qwen_cli::run_qwen),
    ("qwen35", crate::qwen35_cli::run_qwen35),
    ("qwen35moe", crate::qwen35moe_cli::run_qwen35moe),
    ("qwen3omnimoe", crate::omni_cli::run_omni),
    ("glmdsa", crate::glm_cli::run_glm),
    ("lfm2", crate::lfm_cli::run_lfm),
    ("qwen3tts", crate::tts_cli::run_tts),
    ("yolov8", crate::yolo_cli::run_yolo),
    ("zipdepth", crate::depth_cli::run_depth),
    ("flux2", crate::flux2_cli::run_flux2),
    ("wan", crate::wan_cli::run_wan),
    // `sam2 track` (the video memory bank) writes a mask-sequence DIRECTORY,
    // which no single capability blob can carry; every other sam2 verb is
    // forwarded straight back to the generic path by `run_sam2` itself, so the
    // image path keeps its `ARCH_TO_MODEL` row below and its shared code.
    ("sam2", crate::sam2_cli::run_sam2),
    ("ltxv", crate::ltxv_cli::run_ltxv),
    ("worldmirror2", crate::mirror_cli::run_mirror),
    ("splat", crate::splat_cli::run_splat),
    // wm_cli's own `--arch`/`--model` flags (not this resolver) pick
    // fake-vs-diamond within `play`/`import`/`export` -- diamond is its one
    // real served architecture, so that's the id this dispatches from.
    ("diamond", crate::wm_cli::run_wm),
    ("toypid", crate::pid_cli::run_pid),
    ("toymoe", run_toymoe),
];

/// For an [`ARCH_HANDLERS`] id whose crate's own catalog `MODEL` id is NOT
/// simply `brain/<id>` -- `caps_cli::run_caps`'s "is this arch id ALSO a
/// catalog entry" guess otherwise assumes that pattern, which every
/// `ARCH_HANDLERS` architecture but this one follows. Without this row,
/// `brain caps flux2` (and `brain caps flux2-klein`, which is not an
/// `ARCH_HANDLERS`/`ARCH_TO_MODEL` id at all) both report "no such model" even
/// though `crates/flux2/src/caps.rs` has a real, listed manifest.
const ARCH_HANDLER_CATALOG_ID_OVERRIDES: &[(&str, &str)] = &[("flux2", "brain/flux2-klein")];

/// The catalog model id an [`ARCH_HANDLERS`] architecture ALSO registers under
/// (most do, for `brain caps`/discovery, even though dispatch never routes
/// through it) -- the override above when one is needed, else the
/// `brain/<id>` pattern every other row follows.
pub(crate) fn catalog_id_for_arch_handler(id: &str) -> String {
    ARCH_HANDLER_CATALOG_ID_OVERRIDES.iter().find(|(a, _)| *a == id).map(|(_, m)| m.to_string()).unwrap_or_else(|| format!("brain/{id}"))
}

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
    ("s3dit", "brain/s3dit"),
    ("fastvlm", "brain/fastvlm"),
    ("qwen3vl", "brain/qwen3vl"),
    ("sam2", "brain/sam2"),
    ("scrfd", "brain/scrfd"),
    ("arcface", "brain/arcface"),
    ("vqgan", "brain/vqgan"),
    ("codeformer", "brain/codeformer"),
    ("rrdbnet", "brain/rrdbnet"),
    ("clip", "brain/clip"),
    ("deepseek2ocr", "deepseek-ai/DeepSeek-OCR"),
    ("nemotronasr", "brain/nemotronasr"),
    ("qwen3asr", "brain/qwen3asr"),
    ("chronos2", "brain/chronos2"),
    ("fincast", "brain/fincast"),
    ("kronos", "brain/kronos"),
    ("minimaxmusic3", "brain/minimaxmusic3"),
    ("cosyvoice", "brain/cosyvoice"),
    // No-weights utility models: listed by `brain caps` (via `catalog::MODELS`)
    // but, before this row existed, unreachable from the CLI - the same
    // listed-but-unreachable gap `catalog.rs`'s own module docs warn about
    // (`ai-forever/Real-ESRGAN` shipped with a manifest and a provider and was
    // still unreachable because only the residency list had been updated).
    // `flux2-klein`'s `text2image`/`edit`/`lora_train` have the same gap today
    // and are tracked separately, not fixed here.
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
    Unknown(String),
    Empty,
}

/// Every id [`dispatch_arch`] can actually route, from any of the three
/// tables this resolver draws on. `brain_arch::ARCHS` covers the common case
/// (real architectures with a crate, an HF/GGUF fetch story); this resolver's
/// OWN [`ARCH_HANDLERS`]/[`ARCH_TO_MODEL`] additionally list a couple of
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
    if let Some((id, _)) = ARCH_HANDLERS.iter().find(|(id, _)| *id == s) {
        return Some(id);
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
        Resolved::QuantizeFile { rest } => quantize_cli::run_quantize(&rest),
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
    // Unconditional and first: covers BOTH halves of this resolver.
    // `ARCH_TO_MODEL` architectures have no `--weights` flag at all (`run_do`'s
    // params are the action's own schema) and always need this; a handful of
    // `ARCH_HANDLERS` architectures (`qwen3tts`) ALSO read `BRAIN_*` env vars
    // as their own flags' defaults (`--ckpt` defaults to `$BRAIN_QWEN3TTS_CKPT`)
    // rather than taking `--weights` the way `maybe_inject_default_weights`
    // below expects, so this can't be scoped to just the `ARCH_TO_MODEL`
    // branch. No-ops instantly for every architecture with an empty
    // `weights_env` (everything else today).
    //
    // Skipped for `-h`/`--help`: help text must never block on a network
    // fetch (or hang, if `BRAIN_MODELS_DIR` points somewhere with no local
    // weights and the network is slow/unreachable) just to print itself.
    let wants_help = rest.iter().any(|a| a == "-h" || a == "--help");
    if !wants_help && !weights_already_named(arch, &rest) {
        crate::supply::ensure_env_weights(arch);
    }
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

/// The flag twin of a `weights_env` variable, under the naming rule a
/// multi-role architecture's own CLI follows: `BRAIN_<ARCH>_<ROLE>` is
/// reachable as `--<role>`, so `BRAIN_WAN_DIT` is `--dit`. A variable that
/// does not follow the pattern (`rrdbnet`'s `BRAIN_ESRGAN_WEIGHTS`) yields a
/// name no command line will contain, which is the safe direction: it simply
/// never suppresses the fetch.
fn flag_twin(arch: &str, var: &str) -> String {
    let prefix = format!("BRAIN_{}_", arch.to_ascii_uppercase());
    format!("--{}", var.strip_prefix(&prefix).unwrap_or(var).to_ascii_lowercase())
}

/// True when this invocation has already supplied EVERY weight role the
/// architecture declares - each one either set in the environment or named by
/// its flag twin on the command line.
///
/// Without this, an architecture whose `default_ref` is tens of gigabytes
/// (`wan`: 17.6 GB) starts downloading it for a command that named every path
/// explicitly, because [`crate::supply::ensure_env_weights`] can only see the
/// environment. The flag has to win over the variable AND over the fetch.
fn weights_already_named(arch: &str, rest: &[String]) -> bool {
    let Some(a) = brain_arch::by_id(arch) else { return false };
    if a.weights_env.is_empty() {
        return false; // nothing to name; `ensure_env_weights` no-ops anyway
    }
    a.weights_env.iter().all(|(var, _)| {
        std::env::var_os(var).is_some_and(|v| !v.is_empty()) || rest.iter().any(|t| *t == flag_twin(arch, var))
    })
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
///
/// A small, explicit per-arch allowlist extends this past `canon_verb`'s
/// generic "infer" mapping for architectures whose own dedicated CLI never
/// uses that spelling at all: `lfm2`'s real verbs are `fill-mask`/`embed`,
/// both genuinely inference-shaped (never trains from them), but neither
/// canonicalizes to "infer" - so without this, `default_ref: Some(...)` on
/// its `Arch` entry would be dead weight, unreachable from any verb its own
/// CLI actually supports. Kept local and explicit rather than widening
/// `canon_verb` itself, which is shared by every architecture and would
/// risk giving some future, unrelated arch's own "embed" verb a training
/// path this injection was never meant to touch.
fn wants_default_weights(arch: &str, verb: Option<&str>) -> bool {
    // A `Arch::weights_env` architecture whose vars the caller has ALREADY
    // exported has fully specified its weights, so there is nothing to fetch
    // and nothing to inject. `supply::ensure_default_weights` below would
    // download the whole `default_ref` regardless: unlike
    // `supply::ensure_env_weights`, it consults neither those vars nor
    // `BRAIN_AUTO_FETCH`. `brain flux2 generate` hit exactly that -- `canon_verb`
    // maps `generate` to `infer`, so a klein-9b run with all four
    // `BRAIN_FLUX2_*` paths exported still fetched the 4B `default_ref` it can
    // never use, then appended a `--weights` flag `flux2_cli` rejects.
    //
    // Keyed on "every var is set", NOT on "declares weights_env": those two are
    // NOT disjoint. `qwen35`, `qwen3vl` and `s3dit` all declare `weights_env`
    // and still depend on this injection when the vars are unset, so skipping
    // for every `weights_env` row would break them. This mirrors
    // `ensure_env_weights`'s own "every var the caller needs is already set"
    // early return, keeping one rule in both entry points.
    if brain_arch::by_id(arch).is_some_and(|a| {
        !a.weights_env.is_empty()
            && a.weights_env.iter().all(|(var, _)| std::env::var_os(var).is_some_and(|v| !v.is_empty()))
    }) {
        return false;
    }
    let verb = verb.map(crate::args::canon_verb);
    verb.is_some_and(|v| v == "infer")
        || (arch == "lfm2" && verb.is_some_and(|v| v == "fill-mask" || v == "embed"))
}

fn maybe_inject_default_weights(arch: &str, rest: Vec<String>) -> Vec<String> {
    let is_infer = wants_default_weights(arch, rest.first().map(String::as_str));
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
    fn wants_default_weights_recognizes_lfm2s_own_verbs() {
        // `lfm2`'s real verbs are `fill-mask`/`embed`, both genuinely
        // inference-shaped, but neither canonicalizes to "infer" the way
        // `generate`/`gen`/`sample` do for other architectures - so without
        // this arch-specific allowlist, `default_ref: Some("LiquidAI/
        // LFM2.5-350M")` on `lfm2`'s `Arch` entry was unreachable dead weight
        // from any verb its own CLI actually supports.
        assert!(wants_default_weights("lfm2", Some("fill-mask")));
        assert!(wants_default_weights("lfm2", Some("embed")));
        // Not for a verb lfm2's own CLI does not define, even one another
        // architecture's dedicated CLI happens to use for training.
        assert!(!wants_default_weights("lfm2", Some("train")));
        assert!(!wants_default_weights("lfm2", Some("finetune")));
        assert!(!wants_default_weights("lfm2", None));
        // The `fill-mask`/`embed` allowlist is scoped to `lfm2` alone - some
        // future, unrelated architecture's own "embed" verb (if one exists)
        // must not silently start auto-fetching a default checkpoint it
        // never asked for.
        assert!(!wants_default_weights("clip", Some("embed")));
    }

    #[test]
    fn wants_default_weights_still_recognizes_the_generic_infer_mapping() {
        assert!(wants_default_weights("zipdepth", Some("infer")));
        assert!(wants_default_weights("qwen3vl", Some("generate")));
        assert!(wants_default_weights("s3dit", Some("gen")));
        assert!(!wants_default_weights("qwen3", Some("train")));
    }

    /// A caller who has exported every `Arch::weights_env` path has fully
    /// specified its weights, so no `default_ref` may be fetched and no
    /// `--weights` injected. `brain flux2 generate` downloaded the 4B
    /// `default_ref` with all four `BRAIN_FLUX2_*` paths set, because
    /// `canon_verb` maps `generate` to `infer` and `ensure_default_weights`
    /// (unlike `ensure_env_weights`) consults neither those vars nor
    /// `BRAIN_AUTO_FETCH`.
    ///
    /// The partially-configured and unset cases must still fetch: several rows
    /// (`s3dit`, `qwen3vl`, `qwen35`, ...) declare `weights_env` AND depend on
    /// default fetching, which is why the rule keys on "every var set" rather
    /// than on "declares weights_env".
    #[test]
    fn a_fully_configured_weights_env_architecture_skips_the_default_fetch() {
        let a = brain_arch::by_id("flux2").expect("flux2 row");
        let vars: Vec<&str> = a.weights_env.iter().map(|(v, _)| *v).collect();
        assert!(vars.len() >= 2, "flux2 should declare several roles");

        // Nothing exported: the default-fetch path stays available.
        for v in &vars {
            std::env::remove_var(v);
        }
        assert!(wants_default_weights("flux2", Some("generate")), "unset env must still fetch");

        // Every path exported: nothing to fetch, nothing to inject.
        for v in &vars {
            std::env::set_var(v, "/nonexistent/for-test");
        }
        assert!(!wants_default_weights("flux2", Some("generate")), "fully configured must not fetch");
        assert!(!wants_default_weights("flux2", Some("infer")));

        // Partially configured is NOT fully specified, so it still fetches.
        std::env::remove_var(vars[0]);
        assert!(wants_default_weights("flux2", Some("generate")), "partial env must still fetch");

        for v in &vars {
            std::env::remove_var(v);
        }
    }

    /// `weights_env` and a `--weights` flag are NOT mutually exclusive, so the
    /// rule above cannot be simplified to "declares weights_env => never
    /// fetch": that regresses every row which declares both and relies on
    /// default fetching. Recorded because assuming disjointness here looked
    /// obviously right and is simply false.
    #[test]
    fn weights_env_and_the_weights_flag_are_not_mutually_exclusive() {
        // `qwen35_cli.rs` parses `--weights`; the `qwen35` row also declares
        // `weights_env` (and a `default_ref`).
        let a = brain_arch::by_id("qwen35").expect("qwen35 row");
        assert!(
            !a.weights_env.is_empty(),
            "qwen35 is the standing counterexample to the disjointness assumption; \
             if this row changed, re-check wants_default_weights' comment"
        );
        // And plenty of rows pair weights_env with a default_ref they still need.
        for id in ["s3dit", "qwen3vl", "sam2", "rrdbnet"] {
            let a = brain_arch::by_id(id).expect("row exists");
            assert!(!a.weights_env.is_empty() && a.default_ref.is_some(), "{id} should declare both");
            assert!(wants_default_weights(id, Some("infer")), "{id}: default fetch must survive");
        }
    }

    #[test]
    fn maybe_inject_default_weights_leaves_an_explicit_weights_flag_untouched() {
        // `--weights` already present must short-circuit BEFORE the
        // network-dependent `ensure_default_weights` call, for both the
        // generic "infer" path and the lfm2-specific allowlist above -
        // never silently override an explicit flag with a fetched one.
        let rest = s(&["infer", "--weights", "explicit.safetensors"]);
        assert_eq!(maybe_inject_default_weights("zipdepth", rest.clone()), rest);
        let rest = s(&["fill-mask", "--weights", "explicit.safetensors", "--text", "hi"]);
        assert_eq!(maybe_inject_default_weights("lfm2", rest.clone()), rest);
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
    fn explicit_weight_flags_suppress_the_auto_fetch() {
        // `wan` declares four roles; naming all four on the command line must
        // stop the 17.6 GB default-ref fetch, and naming three must not.
        assert_eq!(flag_twin("wan", "BRAIN_WAN_DIT"), "--dit");
        assert_eq!(flag_twin("wan", "BRAIN_WAN_TOKENIZER"), "--tokenizer");
        // A variable that does not follow the pattern yields a flag nothing
        // will match, so it never suppresses the fetch by accident.
        assert_eq!(flag_twin("rrdbnet", "BRAIN_ESRGAN_WEIGHTS"), "--brain_esrgan_weights");

        for (var, _) in brain_arch::by_id("wan").expect("wan row").weights_env {
            std::env::remove_var(var);
        }
        let all = s(&["t2v", "--dit", "d", "--vae", "v", "--t5", "t", "--tokenizer", "k", "--prompt", "x"]);
        assert!(weights_already_named("wan", &all));
        let three = s(&["t2v", "--dit", "d", "--vae", "v", "--t5", "t", "--prompt", "x"]);
        assert!(!weights_already_named("wan", &three));
        // An architecture with no `weights_env` is unaffected either way.
        assert!(!weights_already_named("gpt2", &all));
    }

    #[test]
    fn every_arch_handlers_id_is_a_real_registry_entry() {
        for (id, _) in ARCH_HANDLERS {
            assert!(brain_arch::by_id(id).is_some(), "{id:?} in ARCH_HANDLERS has no brain_arch row");
        }
    }

    #[test]
    fn every_arch_to_model_id_is_a_real_registry_entry() {
        // `imageops`/`demo`/`imgpipe` are the documented exception (see
        // `known_arch_id`'s doc comment): no-weights utility models (`imgpipe`
        // composes OTHER architectures' own weights) with no crate-with-a-port
        // story of their own, so no `brain_arch::Arch` row makes sense for
        // them. Every other `ARCH_TO_MODEL` id is a real architecture.
        const NO_ARCH_ROW: &[&str] = &["imageops", "demo", "imgpipe"];
        for (id, _) in ARCH_TO_MODEL {
            if NO_ARCH_ROW.contains(id) {
                assert!(brain_arch::by_id(id).is_none(), "{id:?} was added to brain_arch::ARCHS -- remove it from NO_ARCH_ROW");
                continue;
            }
            assert!(brain_arch::by_id(id).is_some(), "{id:?} in ARCH_TO_MODEL has no brain_arch row (if intentional, add it to NO_ARCH_ROW above)");
        }
    }

    /// ARCH_HANDLERS and ARCH_TO_MODEL partition disjointly -- an id in both
    /// would mean the generic capability path is silently unreachable for it
    /// (ARCH_HANDLERS is checked first), which is exactly the kind of
    /// drift a table like this is supposed to make impossible to miss.
    ///
    /// One shape is exempt: a handler that exists only to add a verb the
    /// capability wire format cannot carry, and that forwards every OTHER
    /// verb straight back to the generic path. That arch keeps both rows on
    /// purpose, and the exemption is checked rather than assumed -- it must
    /// still have the ARCH_TO_MODEL row it claims to be forwarding to.
    #[test]
    fn arch_handlers_and_arch_to_model_do_not_overlap() {
        // `sam2 track` returns a mask-sequence DIRECTORY, which no single
        // capability blob can carry, so it needs a handler; `run_sam2`
        // forwards everything that is not `track` to `caps_cli::run_do`, so
        // the image path keeps its ARCH_TO_MODEL row and stays reachable.
        const FORWARDS_TO_GENERIC: &[&str] = &["sam2"];
        for (id, _) in ARCH_HANDLERS {
            let has_model_row = ARCH_TO_MODEL.iter().any(|(m, _)| m == id);
            if FORWARDS_TO_GENERIC.contains(id) {
                assert!(has_model_row, "{id:?} is exempted as forwarding to the generic path, but has no ARCH_TO_MODEL row to forward to");
                continue;
            }
            assert!(!has_model_row, "{id:?} is in both ARCH_HANDLERS and ARCH_TO_MODEL");
        }
    }
}
