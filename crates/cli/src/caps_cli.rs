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
use std::sync::Arc;

use capability::{ActionSpec, Blob, Invocation, Manifest, Media, ParamType, Progress, Provider, Registry};
use clap::{Arg, ArgAction, Command};
use serde_json::{json, Value};

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

/// `argv` is `[model id, action, ...flags]` - what `crate::resolve` builds
/// from `brain <architecture> <action> ...` (the arch id translated to its
/// model id first) before calling this.
pub fn run_do(argv: &[String]) -> i32 {
    run_do_impl(argv, None)
}

/// [`run_do`], from an already-resolved [`capability::Assembly`] instead of
/// the placeholder empty one every other model's provider still ignores -
/// what `crate::resolver_cli::run_generic_migrated` calls once it has
/// resolved a migrated architecture's weights.
pub fn run_do_with_assembly(argv: &[String], assembly: &capability::Assembly) -> i32 {
    run_do_impl(argv, Some(assembly))
}

/// [`catalog::provider`], resolving through the CLI's own flagless answer for
/// the model store (`--brain-data-dir`, then `BRAIN_MODELS_DIR`, then XDG/HOME),
/// where the library's version consults only an explicitly configured one. An
/// `Ambiguous` or `Missing` outcome, or no store at all, is this function's own
/// `Err`, never an "unknown model": a listed model may legitimately fail for
/// want of weights.
fn provider_from_store(model: &str) -> Result<Arc<dyn Provider>, String> {
    let assembly = match catalog::resolver_spec_for(model) {
        Some((arch, spec)) => {
            loader::resolver::try_resolve(loader::model_dir::resolve(None).as_deref(), arch, spec, &Default::default()).map_err(|e| e.message().to_string())?
        }
        None => capability::Assembly { id: String::new(), arch: String::new(), variant: None, roles: Default::default(), provenance: Vec::new() },
    };
    catalog::provider_from_assembly(model, &assembly)
}

fn run_do_impl(argv: &[String], assembly: Option<&capability::Assembly>) -> i32 {
    let (model, action) = match (argv.first(), argv.get(1)) {
        (Some(m), Some(a)) if !m.starts_with("--") && !a.starts_with("--") => (m.clone(), a.clone()),
        _ => {
            eprintln!("usage: brain <architecture> <action> [--param value]… [--in name=path]… [--out name=path]…");
            return 2;
        }
    };
    // A legacy short name (e.g. "mock") is a deprecation, not a second id: it
    // resolves to the canonical `brain/<name>` before dispatch, but is never
    // itself what gets registered or listed (see modelref::alias's module docs).
    let model = brain_modelref::alias::canonical(&model).map(str::to_string).unwrap_or(model);
    let built = match assembly {
        Some(a) => catalog::provider_from_assembly(&model, a),
        None => provider_from_store(&model),
    };
    let reg = match built.map(|p| {
        let mut r = Registry::new();
        r.register(p);
        r
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("brain: {e}");
            return 1;
        }
    };
    let act = match reg.find(&model, &action) {
        Some(a) => a,
        None => {
            eprintln!("brain: model {model:?} has no action {action:?} (see `brain caps {model}`)");
            return 1;
        }
    };
    let spec = act.spec();

    // Build a clap parser *from the action's schema* - no hand-rolled arg loop.
    // Each param becomes a typed `--name`, plus `--in`/`--out name=path` and `--json`.
    let matches = match build_parser(&model, &action, &spec).try_get_matches_from(&argv[2..]) {
        Ok(m) => m,
        Err(e) => {
            let _ = e.print();
            return if matches!(e.kind(), clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion) { 0 } else { 2 };
        }
    };

    let mut inv = Invocation::new();
    for p in &spec.params {
        if let Some(v) = matches.get_one::<String>(&p.name) {
            inv = inv.set(&p.name, coerce(&p.ty, v));
        }
    }
    let mut inputs: Vec<(String, String)> = Vec::new();
    for spec_val in matches.get_many::<String>("in").unwrap_or_default() {
        let Some((name, path)) = spec_val.split_once('=') else {
            eprintln!("brain: --in must be name=path (got {spec_val:?})");
            return 2;
        };
        inputs.push((name.to_string(), path.to_string()));
    }
    for b in spec.inputs.iter().filter(|b| has_blob_flag(&spec, &b.name)) {
        if let Some(path) = matches.get_one::<String>(&blob_arg_id(&b.name)) {
            inputs.push((b.name.clone(), path.clone()));
        }
    }
    for (name, path) in &inputs {
        match load_blob(&spec, name, path) {
            Ok(b) => inv = inv.blob(name, b),
            Err(e) => {
                eprintln!("brain: {e}");
                return 1;
            }
        }
    }
    let mut out_paths: Vec<(String, String)> = Vec::new();
    // An action with a param called `out` owns the word: no output-blob flag.
    let blob_out_flag = !has_out_param(&spec);
    for spec_val in matches.get_many::<String>("out").filter(|_| blob_out_flag).into_iter().flatten() {
        let Some((name, path)) = spec_val.split_once('=') else {
            eprintln!("brain: --out must be name=path (got {spec_val:?})");
            return 2;
        };
        if path.is_empty() {
            eprintln!("brain: --out {spec_val:?} names no output path; write it as '--out {name}=<path>'");
            return 2;
        }
        out_paths.push((name.to_string(), path.to_string()));
    }
    let json_out = matches.get_flag("json");

    // run (progress → stderr)
    let mut progress = |p: Progress| {
        if p.total > 0 {
            eprint!("\r\x1b[2K{} [{}/{}] {}", action, p.step, p.total, p.message);
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
    };
    let outcome = match reg.run(&model, &action, inv, &mut progress) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("\nbrain: {e}");
            return 1;
        }
    };
    eprintln!();

    // write output blobs
    for (name, path) in &out_paths {
        match outcome.blobs.get(name) {
            Some(b) => {
                if let Err(e) = save_blob(b, path) {
                    eprintln!("brain: writing {path}: {e}");
                    return 1;
                }
                eprintln!("wrote {name} → {path} ({} bytes)", b.bytes.len());
            }
            None => eprintln!("brain: action produced no output {name:?}"),
        }
    }
    // scalar outputs
    if json_out {
        println!("{}", outcome.outputs);
    } else if let Some(obj) = outcome.outputs.as_object() {
        for (k, v) in obj {
            println!("{k}: {v}");
        }
    }
    0
}

/// The clap id of the `--<name> PATH` shorthand for the input blob `name`.
fn blob_arg_id(name: &str) -> String {
    format!("blob:{name}")
}

/// Whether the action has a param named `out` (a directory to write, say),
/// which then takes the `--out` flag from the generic `--out NAME=PATH`.
fn has_out_param(spec: &ActionSpec) -> bool {
    spec.params.iter().any(|p| p.name == "out")
}

/// The `--flag` words of `name`: itself, and with dashes for underscores
/// (`--held-out` for `held_out`), as command lines are written.
fn with_dashed_alias(arg: Arg, name: &str) -> Arg {
    match name.contains('_') {
        true => arg.alias(name.replace('_', "-")),
        false => arg,
    }
}

/// Whether the input blob `name` also gets a `--<name> PATH` flag, the
/// shorthand for `--in name=PATH`. Not when a param or a reserved flag
/// already owns the word.
fn has_blob_flag(spec: &ActionSpec, name: &str) -> bool {
    !matches!(name, "in" | "out" | "json") && !spec.params.iter().any(|p| p.name == name)
}

/// Build a clap parser directly from an [`ActionSpec`]: one typed `--<param>` per
/// param (required/enum/bool honoured by clap), plus repeatable `--in`/`--out
/// name=path` (every input blob also takes `--<name> path`) and `--json`. All
/// argument parsing goes through clap - no bespoke loop.
fn build_parser(model: &str, action: &str, spec: &ActionSpec) -> Command {
    let mut cmd = Command::new(format!("brain {model} {action}")).no_binary_name(true).about(spec.summary.clone());
    for p in &spec.params {
        let mut arg = with_dashed_alias(Arg::new(p.name.clone()).long(p.name.clone()).help(p.help.clone()), &p.name);
        if p.ty == ParamType::Bool {
            // A bool takes an OPTIONAL value: `--flag` is still `true` (every
            // existing call site keeps working), and `--flag false` / `--flag=0`
            // can now turn one OFF. Without that, a param whose schema default
            // is `true` - `arcface embed --aligned`, `sam2 segment --multimask`
            // - was unreachable from `brain do` while being perfectly settable
            // over D-Bus, i.e. the CLI silently exposed a smaller API than the
            // manifest advertises.
            arg = arg
                .action(ArgAction::Set)
                .value_name("BOOL")
                .num_args(0..=1)
                .default_missing_value("true")
                .value_parser(["true", "false", "1", "0"]);
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
        let help = format!("input {} (same as --in {}=PATH): {}", b.media.name(), b.name, b.help);
        cmd = cmd.arg(with_dashed_alias(Arg::new(blob_arg_id(&b.name)).long(b.name.clone()).action(ArgAction::Set).value_name("PATH").help(help), &b.name));
    }
    let in_help = if spec.inputs.is_empty() { "named binary input, e.g. image=in.ppm".to_string() } else { format!("named binary input ({})", spec.inputs.iter().map(|b| format!("{}=<{}>", b.name, b.media.name())).collect::<Vec<_>>().join(", ")) };
    cmd = cmd.arg(Arg::new("in").long("in").action(ArgAction::Append).value_name("NAME=PATH").help(in_help));
    if !has_out_param(spec) {
        cmd = cmd.arg(Arg::new("out").long("out").action(ArgAction::Append).value_name("NAME=PATH").help("write a named output blob to a file, e.g. image=out.ppm"));
    }
    cmd.arg(Arg::new("json").long("json").action(ArgAction::SetTrue).help("print scalar outputs as JSON"))
}

/// Coerce a CLI string to the JSON value the param type expects.
fn coerce(ty: &ParamType, s: &str) -> Value {
    match ty {
        ParamType::Int => s.parse::<i64>().map(|n| json!(n)).unwrap_or_else(|_| json!(s)),
        ParamType::Float => s.parse::<f64>().map(|x| json!(x)).unwrap_or_else(|_| json!(s)),
        ParamType::Bool => json!(s == "true" || s == "1"),
        _ => json!(s), // Str / Enum
    }
}

/// Load a file into a [`Blob`] with the media the action's input spec declares.
/// Images/masks are decoded to raw **HWC f32** planes in `[0,1]` (meta `{w,h,c}`);
/// a WAV audio file is decoded to raw 16 kHz mono f32 PCM (meta
/// `{"sample_rate":16000}`); a video container is demuxed to RGB frames
/// (`capability::blob::video_blob`'s wire format); text/bytes are read raw.
fn load_blob(spec: &ActionSpec, name: &str, path: &str) -> Result<Blob, String> {
    let media = spec.inputs.iter().find(|b| b.name == name).map(|b| b.media).ok_or_else(|| format!("action has no input '{name}'"))?;
    match media {
        Media::Image | Media::Mask => {
            let (hwc, w, h) = crate::image_io::load_image(path)?;
            let c = hwc.len() / (w as usize * h as usize);
            let bytes: Vec<u8> = hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
            Ok(Blob::new(media, bytes).with_meta(json!({"w": w, "h": h, "c": c})))
        }
        Media::Audio => {
            let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
            load_audio_bytes(bytes)
        }
        // A `video` input is a CLIP, not a file: the wire format is decoded
        // frames, so the container has to be demuxed here (the model crates
        // deliberately have no ffmpeg dependency). Defaults are
        // `VideoDecodeOpts`'s -- one frame per second, at most 32 frames, the scale a
        // multimodal prompt is validated at.
        Media::Video => {
            let frames = imaging::video::decode_frames(std::path::Path::new(path), &Default::default())?;
            capability::blob::video_blob(&frames)
        }
        _ => std::fs::read(path).map(|b| Blob::new(media, b)).map_err(|e| e.to_string()),
    }
}

/// An `--in audio=FILE` payload → the `audio` blob wire format.
///
/// A container file (RIFF/WAVE) is DECODED - downmixed to mono and resampled to
/// 16 kHz - through the same `audio::asr_caps::audio_blob_from_wav` the HTTP
/// `input_audio` content part uses; feeding a model the literal RIFF header
/// reinterpreted as f32 samples is silent garbage, which is what happened before.
/// Anything else is passed through untouched: raw headerless 16 kHz mono f32-LE
/// PCM is the `audio` blob's own wire format and stays accepted as-is, with no
/// meta so an already-correct payload isn't relabelled.
fn load_audio_bytes(bytes: Vec<u8>) -> Result<Blob, String> {
    if audio::asr_caps::is_wav(&bytes) {
        audio::asr_caps::audio_blob_from_wav(&bytes)
    } else {
        Ok(Blob::new(Media::Audio, bytes))
    }
}

/// Write a [`Blob`] to a file: images (raw HWC f32 + `{w,h,c}` meta) → binary PPM
/// (P6, the brain image convention) or PNG (see [`imaging::save`]); audio →
/// a WAV file; video (raw HWC f32 frames + `{frames,w,h,c}` meta) → a
/// container via `imaging::video::encode_frames`; everything else → raw bytes.
fn save_blob(b: &Blob, path: &str) -> Result<(), String> {
    match b.media {
        // A clip, not bytes: writing the raw f32 frames out would produce a
        // file no player opens, which is the same class of bug the audio arm
        // below exists to fix. The frame rate rides in the blob's own meta,
        // because only the producing action knows it; the fallback is the one
        // value every current producer would have written anyway.
        Media::Video => {
            let frames = capability::blob::decode_video(&Invocation::new().blob("video", b.clone()), "video")?;
            let fps = b.meta.get("fps").and_then(|v| v.as_f64()).unwrap_or(16.0);
            let rgb: Vec<imaging::Rgb8> = frames
                .iter()
                .map(|(hwc, w, h)| imaging::pixels::hwc_to_rgb8(hwc, *w, *h, 3, imaging::ChannelPolicy::ReplicateFirst))
                .collect::<Result<_, _>>()?;
            match imaging::video::encode_frames(&rgb, std::path::Path::new(path), fps, &Default::default())? {
                imaging::video::Encoded::Video(_) => Ok(()),
                imaging::video::Encoded::Frames { dir, command, audio: _ } => {
                    eprintln!("brain: ffmpeg is not on PATH, so the {} frames are numbered PPMs in {}", rgb.len(), dir.display());
                    eprintln!("brain: finish the job with:\n  {command}");
                    Ok(())
                }
            }
        }
        Media::Image | Media::Mask => {
            let w = b.meta["w"].as_u64().ok_or("image blob missing w")? as u32;
            let h = b.meta["h"].as_u64().ok_or("image blob missing h")? as u32;
            let c = b.meta["c"].as_u64().unwrap_or(3) as usize;
            let hwc: Vec<f32> = b.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
            // A depth map or a mask is one channel; the CLI's policy is to render
            // it as visible grey rather than refuse to save it. That is a real
            // choice, so it is spelled out rather than implied by the code.
            let img = imaging::pixels::hwc_to_rgb8(&hwc, w, h, c, imaging::ChannelPolicy::ReplicateFirst)?;
            imaging::save(path, &img)
        }
        // Two conventions coexist among audio-producing actions: `qwen3tts
        // synth` already packs a complete WAV byte stream (`meta.format ==
        // "wav"`, or sniffable via the RIFF header), while `qwen3omnimoe`'s
        // `speak`/`converse` emit headerless mono f32-LE PCM at `meta.
        // sample_rate` (the same wire convention `--in audio=` reads on the
        // way in). Writing the latter raw silently produced a file with no
        // WAV header a player could not open; wrap it here instead of
        // guessing at every actions's own meta shape a second time.
        Media::Audio if b.meta.get("format").and_then(|v| v.as_str()) == Some("wav") || audio::asr_caps::is_wav(&b.bytes) => {
            std::fs::write(path, &b.bytes).map_err(|e| e.to_string())
        }
        Media::Audio => {
            let sample_rate = b.meta.get("sample_rate").and_then(|v| v.as_u64()).ok_or("audio blob has no sample_rate and is not already a WAV -- cannot write a header")? as u32;
            let samples: Vec<f32> = b.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
            audio::wav::write(path, &samples, sample_rate).map_err(|e| e.to_string())
        }
        _ => std::fs::write(path, &b.bytes).map_err(|e| e.to_string()),
    }
}

// ---------------------------------------------------------------- built-in demo provider

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{BlobSpec, ParamSpec};

    fn audio_spec() -> ActionSpec {
        ActionSpec::new("transcribe", "transcribe").input(BlobSpec::new("audio", Media::Audio, "raw mono f32 LE PCM at 16 kHz").required())
    }

    /// A `--in audio=clip.wav` must be DECODED, not handed to the model as the
    /// literal RIFF bytes: same samples as a direct `wav::parse` +
    /// `resample_linear`, and tagged with the 16 kHz meta the ASR guards check.
    #[test]
    fn an_input_blob_is_also_a_flag_unless_a_param_owns_the_name() {
        let spec = ActionSpec::new("act", "summary")
            .param(ParamSpec::new("image", ParamType::Str, "a param that shares a blob's name"))
            .input(BlobSpec::new("image", Media::Text, "owned by the param"))
            .input(BlobSpec::new("history", Media::Text, "a text input"));
        assert!(has_blob_flag(&spec, "history"));
        assert!(!has_blob_flag(&spec, "image"), "the param owns the word");
        assert!(!has_blob_flag(&spec, "json"));
        let m = build_parser("m", "act", &spec)
            .try_get_matches_from(["--history", "patient.json", "--in", "image=x.txt", "--json"])
            .unwrap();
        assert_eq!(m.get_one::<String>(&blob_arg_id("history")).map(String::as_str), Some("patient.json"));
        assert_eq!(m.get_many::<String>("in").unwrap().collect::<Vec<_>>(), ["image=x.txt"]);
    }

    /// A param called `out` takes the `--out` flag from the output-blob one,
    /// and underscored names are also written with dashes.
    #[test]
    fn a_param_named_out_owns_the_flag_and_underscores_take_dashes() {
        let spec = ActionSpec::new("train", "summary")
            .param(ParamSpec::new("out", ParamType::Str, "a directory"))
            .param(ParamSpec::new("next_events", ParamType::Str, "codes"))
            .input(BlobSpec::new("held_out", Media::Text, "subjects"));
        assert!(has_out_param(&spec));
        let m = build_parser("m", "train", &spec)
            .try_get_matches_from(["--out", "dir", "--next-events", "a,b", "--held-out", "h.jsonl"])
            .unwrap();
        assert_eq!(m.get_one::<String>("out").map(String::as_str), Some("dir"));
        assert_eq!(m.get_one::<String>("next_events").map(String::as_str), Some("a,b"));
        assert_eq!(m.get_one::<String>(&blob_arg_id("held_out")).map(String::as_str), Some("h.jsonl"));
        let blobs = ActionSpec::new("act", "summary");
        assert!(!has_out_param(&blobs));
        assert!(build_parser("m", "act", &blobs).try_get_matches_from(["--out", "x=y"]).is_ok());
    }

    #[test]
    fn load_blob_decodes_a_wav_file_to_16khz_f32_pcm() {
        let src_rate = 8000u32; // not 16 kHz, so the resample is actually exercised
        let samples: Vec<f32> = (0..48).map(|i| (i as f32 / 48.0) - 0.5).collect();
        let wav_bytes = audio::wav::encode(&samples, src_rate);

        let dir = std::env::temp_dir().join(format!("brain-caps-cli-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.wav");
        std::fs::write(&path, &wav_bytes).unwrap();

        let blob = load_blob(&audio_spec(), "audio", path.to_str().unwrap()).expect("wav loads");
        std::fs::remove_dir_all(&dir).ok();

        let parsed = audio::wav::parse(&wav_bytes).expect("fixture parses");
        let want = audio::resample_linear(&parsed.samples, parsed.sample_rate, 16000);
        assert!(!want.is_empty());
        assert_eq!(blob.media, Media::Audio);
        assert_eq!(blob.bytes.len(), want.len() * 4, "one f32 per resampled sample");
        assert_eq!(blob.meta["sample_rate"], 16000);
        let got: Vec<f32> = blob.bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(got, want);
        // The header must be gone: raw pass-through would have kept 44+ bytes of it.
        assert_ne!(blob.bytes.len(), wav_bytes.len());
        // And it round-trips through the ASR blob decoder the models use.
        assert_eq!(audio::asr_caps::wav_from_blob(&blob).unwrap(), want);
    }

    /// Backward compatibility: a headerless raw-PCM file (the documented
    /// `clip.pcm`) is still passed through byte-for-byte, with no meta invented.
    #[test]
    fn load_blob_passes_a_non_wav_audio_file_through_unchanged() {
        let raw: Vec<u8> = (0..64u8).collect();
        let dir = std::env::temp_dir().join(format!("brain-caps-cli-pcm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.pcm");
        std::fs::write(&path, &raw).unwrap();

        let blob = load_blob(&audio_spec(), "audio", path.to_str().unwrap()).expect("raw pcm loads");
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(blob.bytes, raw);
        assert!(blob.meta.get("sample_rate").is_none(), "no sample_rate invented for raw PCM: {}", blob.meta);
    }

    /// `--out result=` (the blob name with nothing after `=`) used to pass
    /// `split_once('=')`'s check and run the whole action anyway, only to
    /// fail deep inside [`save_blob`] on the empty path - or, for a blob
    /// kind `save_blob` opens permissively, write a file literally named
    /// after the flag. It is now refused up front, before the action runs
    /// at all, the same way an unparseable `--out` already was.
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
