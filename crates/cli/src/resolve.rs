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
    ("llava", "brain/llava"),
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
    ("timesfm3", "brain/timesfm3"),
    ("minimaxmusic3", "brain/minimaxmusic3"),
    ("cosyvoice", "brain/cosyvoice"),
    // `sdxlunet`/`controlnet` chose no CLI shortcut at all (only `brain do
    // brain/sdxl …`/the residency-served transports) - SUPIR's own single
    // action makes a one-line shortcut cheap enough to add here instead,
    // matching `llava`'s row just above it: `brain supir restore …` reaches
    // the exact same `supir::caps` this row's model id already serves
    // through `brain do`/D-Bus.
    ("supir", "brain/supir"),
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

/// The closed verb vocabulary of an [`ARCH_HANDLERS`] entry, in the form
/// [`crate::args::canon_verb`] normalizes an input verb to -- checked
/// BEFORE [`dispatch_arch`] acquires any weights, so a mistyped verb (`brain
/// flux2 t2i`: "t2i" is not a real flux2 verb, the real one is
/// `generate`/`infer`) reports "unknown subcommand" instead of first trying
/// to pull a multi-GB default checkpoint for a command that could never run.
///
/// Populated only for the handlers whose `weights_env` makes that fetch
/// real (non-empty, per `brain_arch::by_id`): for every other
/// `ARCH_HANDLERS` entry `ensure_env_weights` is already a guaranteed
/// instant no-op, so there is nothing expensive to guard and adding a row
/// here would just be a second verb list to keep in sync with each
/// handler's own match arms for no behavioral gain. `qwen3tts` and `sam2`
/// are the two exceptions among THOSE: both forward any verb their own match
/// arm does not recognize to the generic capability dispatcher
/// (`crate::caps_cli::run_do`) instead of rejecting it outright, so there is
/// no fixed vocabulary to check them against here either -- they stay
/// ungated, unchanged from before this table existed.
const ARCH_HANDLER_VERBS: &[(&str, &[&str])] = &[
    ("flux2", &["generate", "infer", "finetune"]),
    // Deliberately NOT `infer` -- `wan_cli::run_wan`'s own module doc: that
    // canonicalizes from `generate`/`gen`/`sample`, which would inject a
    // single `--weights` flag onto a command that takes four weight roles.
    ("wan", &["t2v", "text2video", "finetune"]),
    ("ltxv", &["t2v", "text2video", "upscale", "v2v", "dfr"]),
];

/// Whether `verb` is one `arch` is known to accept, checked before any
/// weight acquisition is attempted.
///
/// `Some(true)`/`Some(false)` for an architecture this gate has an opinion
/// about ([`ARCH_HANDLER_VERBS`], or an [`ARCH_TO_MODEL`] architecture
/// checked against its own static capability manifest -- no weights loaded,
/// the same manifest `brain caps` prints and `run_do` itself validates the
/// action name against, just consulted earlier). `None` for everything
/// else: no opinion, so the caller must treat the verb as fine and proceed
/// exactly as it did before this gate existed.
fn verb_is_known(arch: &str, verb: &str) -> Option<bool> {
    if let Some((_, verbs)) = ARCH_HANDLER_VERBS.iter().find(|(id, _)| *id == arch) {
        return Some(verbs.contains(&crate::args::canon_verb(verb)));
    }
    if let Some(model) = model_for_arch(arch) {
        return Some(crate::catalog::manifests().iter().any(|m| m.model == model && m.actions.iter().any(|a| a.name == verb)));
    }
    None
}

/// Whether [`dispatch_arch`] should attempt to acquire `arch`'s weights for
/// this invocation at all -- the single gate in front of
/// `ensure_env_weights`'s network/filesystem work.
///
/// `false` for `-h`/`--help` (unchanged: help text must never block on a
/// fetch, or hang, if `BRAIN_MODELS_DIR` points somewhere with no local
/// weights and the network is slow/unreachable, just to print itself), for
/// no verb at all (bare `brain flux2` is exactly as help-shaped), and now
/// also for a verb [`verb_is_known`] says the handler will reject -- the
/// real bug this closes: `brain flux2 t2i` used to fetch flux2's default
/// checkpoint before `flux2_cli::run_flux2`'s own dispatch ever got a
/// chance to say "unknown subcommand t2i".
fn wants_weight_acquisition(arch: &str, rest: &[String]) -> bool {
    if rest.iter().any(|a| a == "-h" || a == "--help") {
        return false;
    }
    match rest.first() {
        None => false,
        Some(verb) => verb_is_known(arch, verb) != Some(false),
    }
}

/// Architectures whose weight resolution has fully moved to
/// `brain_modelstore::resolve` (a `--model`/per-role-flag/store-inventory
/// based resolver, wired into that architecture's own handler) and must
/// never again go through EITHER legacy env-var mechanism below --
/// `ensure_env_weights` (keyed on `Arch::weights_env`, already a no-op once
/// that's empty) AND, independently, `maybe_inject_default_weights` (keyed
/// only on the verb being infer-shaped, so it still fires for an arch with
/// an EMPTY `weights_env` too -- that's the whole point of it for
/// `weights_env`-free architectures like `zipdepth`/`gpt2`, which is exactly
/// why emptying `weights_env` alone does not opt an architecture out: the
/// resolver-migrated case needs an explicit marker, not an overload of a
/// field whose "empty" already means something else for those others).
/// Grown by one entry per architecture as it migrates.
const RESOLVER_MIGRATED_ARCHS: &[&str] =
    &["flux2", "qwen3tts", "kronos", "ltxv", "qwen35", "qwen3vl", "fastvlm", "moondream3", "deepseek2ocr"];

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
    let resolver_migrated = RESOLVER_MIGRATED_ARCHS.contains(&arch);
    // Unconditional (past the gates above) and first: covers BOTH halves of
    // this resolver. `ARCH_TO_MODEL` architectures have no `--weights` flag
    // at all (`run_do`'s params are the action's own schema) and always
    // need this; a handful of `ARCH_HANDLERS` architectures (`qwen3tts`)
    // ALSO read `BRAIN_*` env vars as their own flags' defaults (`--ckpt`
    // defaults to `$BRAIN_QWEN3TTS_CKPT`) rather than taking `--weights` the
    // way `maybe_inject_default_weights` below expects, so this can't be
    // scoped to just the `ARCH_TO_MODEL` branch. No-ops instantly for every
    // architecture with an empty `weights_env` (everything else today).
    let attempt_fetch = !resolver_migrated && wants_weight_acquisition(arch, &rest);
    if attempt_fetch && !weights_already_named(arch, &rest) {
        crate::supply::ensure_env_weights(arch);
    }
    if let Some((_, handler)) = ARCH_HANDLERS.iter().find(|(id, _)| *id == arch) {
        let rest = if attempt_fetch { maybe_inject_default_weights(arch, rest) } else { rest };
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

/// A `(variable, real CLI flag)` override for the handful of `weights_env`
/// variables whose flag genuinely doesn't follow the `BRAIN_<ARCH>_<ROLE>` ->
/// `--<role>` naming rule [`flag_twin`] otherwise derives. The `weights_env`
/// tuple's own second field ("role") is NOT a safe substitute here: it names
/// a shared semantic role used elsewhere (`crate::supply`'s `FilesRecipe`
/// layout) that several architectures deliberately spell differently on
/// their own dedicated CLI -- e.g. `wan`'s `role="text_encoder"` is reached
/// as `--t5`, not `--text_encoder`. Only list a variable here once its real
/// flag is confirmed to actually diverge from the derived guess; anything
/// absent keeps falling through to the derivation below.
const FLAG_TWIN_OVERRIDES: &[(&str, &str)] = &[
    // qwen3tts's dedicated CLI (`tts_cli.rs::parse_common`) reaches
    // `BRAIN_QWEN3TTS_WEIGHTS` as `--weights-dir`; the derived guess
    // (`--weights`, the variable's own suffix) never appears on a real
    // command line, so `weights_already_named` could never see an explicit
    // `--weights-dir --ckpt` pair and always fell through to the auto-fetch
    // gate even when both paths were named.
    ("BRAIN_QWEN3TTS_WEIGHTS", "--weights-dir"),
];

/// The flag twin of a `weights_env` variable, under the naming rule a
/// multi-role architecture's own CLI follows: `BRAIN_<ARCH>_<ROLE>` is
/// reachable as `--<role>`, so `BRAIN_WAN_DIT` is `--dit`. [`FLAG_TWIN_OVERRIDES`]
/// wins first for the variables known to diverge from that rule. A variable
/// that follows neither (`rrdbnet`'s `BRAIN_ESRGAN_WEIGHTS`) yields a name no
/// command line will contain, which is the safe direction: it simply never
/// suppresses the fetch.
fn flag_twin(arch: &str, var: &str) -> String {
    if let Some((_, flag)) = FLAG_TWIN_OVERRIDES.iter().find(|(v, _)| *v == var) {
        return flag.to_string();
    }
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
    weights_already_named_with(arch, brain_arch::by_id(arch), rest)
}

/// [`weights_already_named`]'s logic, taking an already-looked-up `Arch` -
/// see [`wants_default_weights_with`] for why a test needs this seam.
fn weights_already_named_with(arch: &str, a: Option<&brain_arch::Arch>, rest: &[String]) -> bool {
    let Some(a) = a else { return false };
    if a.weights_env.is_empty() {
        return false; // nothing to name; `ensure_env_weights` no-ops anyway
    }
    // `--model` (the generic resolver, `model_flag`) names the PRIMARY
    // weights outright -- the component `weights_env[0]`'s variable carries --
    // so it satisfies that one the way its flag twin would: a `--model` run
    // must not re-fetch the default ref's primary component it is about to
    // override. The resolver fetches its own when the name needs it.
    let model_names_primary = rest.iter().any(|t| t == "--model");
    a.weights_env.iter().enumerate().all(|(i, (var, _))| {
        let twin = flag_twin(arch, var);
        // Both spellings count. A dedicated CLI writes the flag with hyphens
        // (`--weights-dir`); the GENERIC capability dispatcher derives its
        // flags from param NAMES, which are underscored (`--weights_dir`) - so
        // an architecture that has both entry points (qwen3tts) would
        // otherwise have its generic invocations fall through to the
        // auto-fetch gate while its dedicated ones suppress it, for the same
        // path named the same way.
        let underscored = format!("--{}", twin.trim_start_matches('-').replace('-', "_"));
        std::env::var_os(var).is_some_and(|v| !v.is_empty())
            || rest.iter().any(|t| *t == twin || *t == underscored)
            || (i == 0 && model_names_primary)
    })
}

/// For an `infer`-shaped verb with no `--weights` already given, auto-fetch
/// the architecture's default checkpoint
/// ([`brain_arch::Arch::default_ref`], via
/// [`crate::supply::ensure_default_weights`]; fetching is opt-in, so with it
/// off an unpulled default errors instead) and inject `--weights <path>`
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
    wants_default_weights_with(arch, brain_arch::by_id(arch), verb)
}

/// [`wants_default_weights`]'s logic, taking an already-looked-up `Arch`
/// instead of resolving one itself - so a test can exercise the decision
/// against a hand-built fixture instead of a real, live registry row, which a
/// registry-wide migration (every `weights_env` row moving to the resolver,
/// as one eventually will) would otherwise leave with no real example left
/// to test against at all.
fn wants_default_weights_with(arch: &str, a: Option<&brain_arch::Arch>, verb: Option<&str>) -> bool {
    // A `Arch::weights_env` architecture whose vars the caller has ALREADY
    // exported has fully specified its weights, so there is nothing to fetch
    // and nothing to inject. `supply::ensure_default_weights` below would
    // still resolve the whole `default_ref` regardless: unlike
    // `supply::ensure_env_weights`, it consults none of those vars.
    // `brain flux2 generate` hit exactly that -- `canon_verb`
    // maps `generate` to `infer`, so a klein-9b run with all four
    // `BRAIN_FLUX2_*` paths exported still fetched the 4B `default_ref` it can
    // never use, then appended a `--weights` flag `flux2_cli` rejects.
    //
    // Keyed on "every var is set", NOT on "declares weights_env": those two are
    // NOT disjoint. Several rows declare `weights_env` and still depend on
    // this injection when the vars are unset, so skipping for every
    // `weights_env` row would break them. This mirrors
    // `ensure_env_weights`'s own "every var the caller needs is already set"
    // early return, keeping one rule in both entry points.
    if a.is_some_and(|a| {
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
    // `--model` names the weights the way `--weights` does (the generic
    // resolver, `model_flag`): an invocation that carries it has already said
    // what to load, and the injected flag would only be rejected by a parser
    // that does not take `--weights` at all. Help is not an inference
    // invocation at all, whatever the verb says (`canon_verb("generate")` is
    // "infer"): usage must never block on a fetch, the same rule
    // `dispatch_arch` applies to `ensure_env_weights`.
    if !is_infer
        || rest.iter().any(|a| a == "--weights")
        || rest.iter().any(|a| a == "--model")
        || rest.iter().any(|a| a == "-h" || a == "--help")
    {
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
    use brain_testutil::env_lock;

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
        assert!(wants_default_weights("moondream3", Some("gen")));
        assert!(!wants_default_weights("qwen3", Some("train")));
    }

    /// A synthetic multi-role `Arch`, for the tests below that need to
    /// exercise `wants_default_weights`/`weights_already_named` against a
    /// declared `weights_env` + `default_ref` pair. This registry migrated
    /// one real architecture at a time onto the resolver until none was left
    /// that still paired the two (every `weights_env` row that ALSO carried a
    /// `default_ref` moved to the resolver over the course of this
    /// migration) - a test pinned to whichever real row was left "still on
    /// the env-var path" kept going stale as the next one migrated. A
    /// hand-built fixture, exercised through the `_with` seam that takes an
    /// already-looked-up `Arch` instead of resolving one from the live
    /// registry, can never go stale that way.
    fn multi_role_fixture() -> brain_arch::Arch {
        brain_arch::Arch {
            id: "resolvetestarch",
            display: "Resolve test fixture",
            domain: brain_arch::Domain::Toy,
            source: brain_arch::Source::Toy,
            package: "brain-toy",
            gguf: None,
            hf: &[],
            default_ref: Some("test-vendor/test-repo"),
            extra_refs: &[],
            weights_env: &[("BRAIN_RESOLVETESTARCH_DIT", "dit"), ("BRAIN_RESOLVETESTARCH_VAE", "vae")],
            variants: &[],
            families: &[],
        }
    }

    /// A caller who has exported every `Arch::weights_env` path has fully
    /// specified its weights, so no `default_ref` may be fetched and no
    /// `--weights` injected. `brain flux2 generate` downloaded the 4B
    /// `default_ref` with all four `BRAIN_FLUX2_*` paths set, because
    /// `canon_verb` maps `generate` to `infer` and `ensure_default_weights`
    /// (unlike `ensure_env_weights`) consults none of those vars.
    ///
    /// The partially-configured and unset cases must still fetch: a
    /// `weights_env` row can still depend on default fetching, which is why
    /// the rule keys on "every var set" rather than on "declares
    /// weights_env" - see [`multi_role_fixture`].
    #[test]
    fn a_fully_configured_weights_env_architecture_skips_the_default_fetch() {
        let _serial = env_lock();
        let fixture = multi_role_fixture();
        let vars: Vec<&str> = fixture.weights_env.iter().map(|(v, _)| *v).collect();
        assert!(vars.len() >= 2, "the fixture should declare several roles");

        // Nothing exported: the default-fetch path stays available.
        for v in &vars {
            std::env::remove_var(v);
        }
        assert!(wants_default_weights_with(fixture.id, Some(&fixture), Some("generate")), "unset env must still fetch");

        // Every path exported: nothing to fetch, nothing to inject.
        for v in &vars {
            std::env::set_var(v, "/nonexistent/for-test");
        }
        assert!(!wants_default_weights_with(fixture.id, Some(&fixture), Some("generate")), "fully configured must not fetch");
        assert!(!wants_default_weights_with(fixture.id, Some(&fixture), Some("infer")));

        // Partially configured is NOT fully specified, so it still fetches.
        std::env::remove_var(vars[0]);
        assert!(wants_default_weights_with(fixture.id, Some(&fixture), Some("generate")), "partial env must still fetch");

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
        let fixture = multi_role_fixture();
        assert!(!fixture.weights_env.is_empty() && fixture.default_ref.is_some(), "the fixture should declare both");
        assert!(wants_default_weights_with(fixture.id, Some(&fixture), Some("infer")), "default fetch must survive when unset");
    }

    /// The five architectures this migration moved onto the resolver empty
    /// their `weights_env` - `wants_default_weights`'s "every var already
    /// set" early return can no longer apply to them (nothing to check), and
    /// `RESOLVER_MIGRATED_ARCHS` is what stops `dispatch_arch` from calling
    /// either legacy path at all for them, exactly as it already does for
    /// `flux2`.
    #[test]
    fn every_migrated_arch_has_an_empty_weights_env_and_is_resolver_migrated() {
        for id in ["qwen35", "qwen3vl", "fastvlm", "moondream3", "deepseek2ocr"] {
            let a = brain_arch::by_id(id).expect(id);
            assert!(a.weights_env.is_empty(), "{id} should have emptied weights_env");
            assert!(RESOLVER_MIGRATED_ARCHS.contains(&id), "{id} should be in RESOLVER_MIGRATED_ARCHS");
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

    /// `--model` names the weights too (the generic resolver): an invocation
    /// carrying it must short-circuit the same way `--weights` does -- before
    /// the network-dependent `ensure_default_weights` call -- rather than
    /// have a fetched default injected beside it.
    #[test]
    fn maybe_inject_default_weights_leaves_a_model_flag_untouched() {
        let rest = s(&["infer", "--model", "black-forest-labs/FLUX.2-klein-4B"]);
        assert_eq!(maybe_inject_default_weights("zipdepth", rest.clone()), rest);
    }

    /// Help is not an inference invocation, even when the verb is
    /// (`canon_verb("generate")` is "infer"): `brain flux2 generate --help`
    /// fetched the default ref over the network just to print usage, the
    /// same way `ensure_env_weights`'s own help gate exists to prevent.
    #[test]
    fn maybe_inject_default_weights_ignores_a_help_request() {
        for rest in [s(&["generate", "--help"]), s(&["generate", "-h"]), s(&["--help"])] {
            assert_eq!(maybe_inject_default_weights("flux2", rest.clone()), rest);
        }
    }

    /// `--model` stands in for the PRIMARY weights role (`weights_env[0]`'s
    /// variable): with the auxiliary roles set, naming the primary via
    /// `--model` must stop the default-ref fetch, which would otherwise
    /// re-download the very component `--model` overrides. A role that is
    /// neither set nor named still leaves the fetch in place.
    ///
    /// Uses [`multi_role_fixture`] as its example architecture - see that
    /// fixture's own doc for why a real registry row cannot serve here any
    /// more.
    #[test]
    fn a_model_flag_counts_as_naming_the_primary_weights() {
        let _serial = env_lock();
        let fixture = multi_role_fixture();
        let vars: Vec<_> = fixture.weights_env.iter().map(|(v, _)| *v).collect();
        for &var in &vars {
            std::env::remove_var(var);
        }
        let with_model = s(&["infer", "--model", "some/dit", "--prompt", "p"]);
        assert!(!weights_already_named_with(fixture.id, Some(&fixture), &with_model), "the auxiliary roles are still unnamed");

        // Every role but the primary (`weights_env[0]`, the DiT) set via its
        // own variable.
        for &var in &vars[1..] {
            std::env::set_var(var, "x");
        }
        assert!(weights_already_named_with(fixture.id, Some(&fixture), &with_model), "--model names the DiT; the rest are in the env");
        assert!(
            !weights_already_named_with(fixture.id, Some(&fixture), &s(&["infer", "--prompt", "p"])),
            "without --model the unset primary still wants the fetch"
        );

        for &var in &vars {
            std::env::remove_var(var);
        }
    }

    /// `flux2` moved to the resolver: its `weights_env` is empty, so
    /// `weights_already_named` has nothing to name (its own documented
    /// "nothing to name; `ensure_env_weights` no-ops anyway" early return) --
    /// and that no-op is exactly what lets `dispatch_arch` reach the
    /// resolver-based handler at all, instead of failing on unset
    /// `BRAIN_FLUX2_*` vars that no longer mean anything.
    #[test]
    fn flux2_no_longer_wants_env_based_weight_acquisition() {
        let _serial = env_lock();
        assert!(brain_arch::by_id("flux2").expect("flux2 row").weights_env.is_empty());
        assert!(!weights_already_named("flux2", &s(&["generate", "--prompt", "p"])));
    }

    /// `wan` moved to the resolver too - same gate as `flux2`'s own test
    /// above, on `t2v` instead of `generate`.
    #[test]
    fn wan_no_longer_wants_env_based_weight_acquisition() {
        let _serial = env_lock();
        assert!(brain_arch::by_id("wan").expect("wan row").weights_env.is_empty());
        assert!(!weights_already_named("wan", &s(&["t2v", "--prompt", "p"])));
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

    /// Uses [`multi_role_fixture`] rather than a real registry row - see that
    /// fixture's own doc for why.
    #[test]
    fn explicit_weight_flags_suppress_the_auto_fetch() {
        let _serial = env_lock();
        // Naming every role on the command line must stop the default-ref
        // fetch, and naming all but one must not.
        assert_eq!(flag_twin("wan", "BRAIN_WAN_DIT"), "--dit");
        assert_eq!(flag_twin("wan", "BRAIN_WAN_TOKENIZER"), "--tokenizer");
        // A variable that does not follow the pattern yields a flag nothing
        // will match, so it never suppresses the fetch by accident.
        assert_eq!(flag_twin("rrdbnet", "BRAIN_ESRGAN_WEIGHTS"), "--brain_esrgan_weights");
        // qwen3tts's `BRAIN_QWEN3TTS_WEIGHTS` is a listed override: its real
        // flag (`--weights-dir`) doesn't follow the derivation rule either,
        // but unlike rrdbnet's case this one DOES have a real dedicated CLI
        // flag that must be recognized, not silently missed.
        assert_eq!(flag_twin("qwen3tts", "BRAIN_QWEN3TTS_WEIGHTS"), "--weights-dir");

        let fixture = multi_role_fixture();
        let roles = fixture.weights_env;
        for (var, _) in roles {
            std::env::remove_var(var);
        }
        let mut all = vec!["infer".to_string()];
        for (var, _) in roles {
            all.push(flag_twin(fixture.id, var));
            all.push(format!("path-for-{var}"));
        }
        assert!(weights_already_named_with(fixture.id, Some(&fixture), &all));

        let mut missing_one = vec!["infer".to_string()];
        for (var, _) in &roles[..roles.len() - 1] {
            missing_one.push(flag_twin(fixture.id, var));
            missing_one.push(format!("path-for-{var}"));
        }
        assert!(!weights_already_named_with(fixture.id, Some(&fixture), &missing_one));
        // An architecture with no `weights_env` is unaffected either way.
        assert!(!weights_already_named("gpt2", &all));
    }

    /// `qwen3tts` moved to the resolver, the same way `flux2` did (see
    /// `flux2_no_longer_wants_env_based_weight_acquisition`): its
    /// `weights_env` is empty, so `weights_already_named` has nothing to
    /// name (its own documented "nothing to name; `ensure_env_weights`
    /// no-ops anyway" early return) - `tts_cli.rs`'s own `--weights-dir`/
    /// `--ckpt` flags are the resolver's role overrides now, not a pair this
    /// legacy env-var gate needs to recognize at all.
    #[test]
    fn qwen3tts_no_longer_wants_env_based_weight_acquisition() {
        let _serial = env_lock();
        assert!(brain_arch::by_id("qwen3tts").expect("qwen3tts row").weights_env.is_empty());
        assert!(!weights_already_named("qwen3tts", &s(&["synth", "--weights-dir", "D", "--ckpt", "C", "--text", "hi"])));
    }

    /// `kronos` moved to the resolver too - see
    /// `qwen3tts_no_longer_wants_env_based_weight_acquisition`'s own doc.
    /// kronos has no `ARCH_HANDLERS` entry of its own (it dispatches
    /// generically via `ARCH_TO_MODEL`), but the same env-var auto-fetch
    /// gate this row exempts it from would otherwise still apply to `brain
    /// kronos ...`/`brain <verb> kronos` invocations.
    #[test]
    fn kronos_no_longer_wants_env_based_weight_acquisition() {
        let _serial = env_lock();
        assert!(brain_arch::by_id("kronos").expect("kronos row").weights_env.is_empty());
        assert!(brain_arch::by_id("kronos").expect("kronos row").extra_refs.is_empty());
        assert!(!weights_already_named("kronos", &s(&["speak"])));
    }

    /// `ltxv` moved to the resolver too - see
    /// `qwen3tts_no_longer_wants_env_based_weight_acquisition`'s own doc.
    #[test]
    fn ltxv_no_longer_wants_env_based_weight_acquisition() {
        let _serial = env_lock();
        assert!(brain_arch::by_id("ltxv").expect("ltxv row").weights_env.is_empty());
        assert!(!weights_already_named("ltxv", &s(&["t2v", "--vae", "v", "--prompt", "p"])));
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

    /// The reported bug: `brain flux2 t2i` ("t2i" is not a real flux2 verb --
    /// the real one is `generate`/`infer`) printed "not pulled: BRAIN_FLUX2_DIT,
    /// ... unset and no local copy of black-forest-labs/FLUX.2-klein-4B" and
    /// exited 1, because `dispatch_arch` ran `ensure_env_weights`
    /// unconditionally before ever reaching `flux2_cli::run_flux2`'s own
    /// "unknown subcommand" branch. `verb_is_known` is the gate that gets
    /// checked FIRST now: `Some(false)` for a verb the handler will reject,
    /// so `wants_weight_acquisition` (below) can skip the fetch entirely
    /// with zero filesystem/network side effects.
    #[test]
    fn verb_is_known_rejects_flux2s_invalid_verb() {
        assert_eq!(verb_is_known("flux2", "t2i"), Some(false));
        assert_eq!(verb_is_known("flux2", "bogus"), Some(false));
    }

    /// flux2's real verbs (`generate`/`infer`/`finetune`) and their
    /// `canon_verb` synonyms (`gen`/`sample` -> infer, `fine-tune` ->
    /// finetune) must still be recognized -- this is the regression guard: a
    /// gate that is too strict would turn a real invocation into a false
    /// "unknown subcommand".
    #[test]
    fn verb_is_known_accepts_flux2s_real_verbs_and_synonyms() {
        for v in ["generate", "infer", "finetune", "gen", "sample", "fine-tune"] {
            assert_eq!(verb_is_known("flux2", v), Some(true), "flux2 should accept {v:?}");
        }
    }

    /// `wan` deliberately does NOT accept the generic `generate`/`infer`
    /// spelling (its own module doc: those canonicalize to "infer", which
    /// would wrongly inject a single `--weights` flag onto a command that
    /// takes four weight roles) -- only `t2v`/`text2video` and `finetune`.
    /// The regression guard for the OTHER architecture this bug class hits:
    /// a bogus wan verb must be rejected too, and wan's real verbs must stay
    /// accepted.
    #[test]
    fn verb_is_known_matches_wans_own_closed_verb_set() {
        assert_eq!(verb_is_known("wan", "t2i"), Some(false));
        for v in ["t2v", "text2video", "finetune"] {
            assert_eq!(verb_is_known("wan", v), Some(true), "wan should accept {v:?}");
        }
        // `generate`/`infer` canonicalize to "infer", which wan's own CLI
        // does not accept -- must stay rejected, not silently allowed.
        assert_eq!(verb_is_known("wan", "generate"), Some(false));
        assert_eq!(verb_is_known("wan", "infer"), Some(false));
    }

    /// An architecture this gate has no opinion about (no `weights_env`, so
    /// `ensure_env_weights` is already a guaranteed instant no-op -- nothing
    /// expensive to guard) is untouched: `None`, not `Some(false)`, so
    /// `wants_weight_acquisition` treats it exactly as before this gate
    /// existed.
    #[test]
    fn verb_is_known_has_no_opinion_on_an_ungated_handler() {
        assert_eq!(verb_is_known("gpt2", "t2i"), None);
        assert_eq!(verb_is_known("gpt2", "train"), None);
    }

    /// The actual gate `dispatch_arch` consults before calling
    /// `ensure_env_weights`: this is the function whose `false` means "zero
    /// filesystem/network side effects" for the reported bug. Exercised
    /// directly (rather than through `dispatch_arch`, which calls
    /// `std::process::exit` on every real code path) so a regression here
    /// fails a normal test instead of killing the test process.
    #[test]
    fn wants_weight_acquisition_skips_the_fetch_for_an_unknown_verb() {
        assert!(!wants_weight_acquisition("flux2", &s(&["t2i"])));
        assert!(!wants_weight_acquisition("wan", &s(&["t2i"])));
        // No verb at all (bare `brain flux2`) is help-shaped too -- must not
        // block on a fetch just to print the handler's own usage.
        assert!(!wants_weight_acquisition("flux2", &s(&[])));
        assert!(!wants_weight_acquisition("flux2", &s(&["--help"])));
    }

    /// The regression guard: every real verb (and its `canon_verb`
    /// synonyms) must still reach the fetch gate exactly as before this
    /// change.
    #[test]
    fn wants_weight_acquisition_still_proceeds_for_every_real_verb() {
        for v in ["generate", "infer", "finetune", "gen", "sample", "fine-tune"] {
            assert!(wants_weight_acquisition("flux2", &s(&[v])), "flux2 {v:?} should still attempt weight acquisition");
        }
        for v in ["t2v", "text2video", "finetune"] {
            assert!(wants_weight_acquisition("wan", &s(&[v])), "wan {v:?} should still attempt weight acquisition");
        }
        // An arch with no verb table entry is unaffected: unchanged
        // (pre-existing) behavior, gated only by help/no-verb.
        assert!(wants_weight_acquisition("gpt2", &s(&["t2i"])));
        assert!(wants_weight_acquisition("gpt2", &s(&["train"])));
    }

    /// `ARCH_TO_MODEL` architectures validate against their own capability
    /// manifest's action names (no separate hardcoded list) -- `s3dit`'s
    /// manifest is a real, populated one to check this against.
    #[test]
    fn verb_is_known_validates_arch_to_model_verbs_against_the_manifest() {
        assert_eq!(verb_is_known("s3dit", "not-a-real-action"), Some(false));
        let model = model_for_arch("s3dit").expect("s3dit has an ARCH_TO_MODEL row");
        let real_action = crate::catalog::manifests()
            .into_iter()
            .find(|m| m.model == model)
            .and_then(|m| m.actions.first().map(|a| a.name.clone()))
            .expect("s3dit's manifest should list at least one action");
        assert_eq!(verb_is_known("s3dit", &real_action), Some(true));
    }
}
