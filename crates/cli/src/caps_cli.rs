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

use std::sync::Arc;

use capability::{Action, ActionSpec, Blob, Invocation, Manifest, Media, ParamType, Progress, Provider, Registry};
use clap::{Arg, ArgAction, Command};
use serde_json::{json, Value};

/// Every model's static capability manifest (discovery - no weights). Add a model
/// here and it appears in `brain caps` immediately.
/// The catalog id of the trivial always-available demo model (no `caps.rs` of
/// its own -- this const is its single source of truth).
const DEMO_MODEL: &str = "brain/demo";

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
                c.push(format!("brain/{m}"));
            }
            c
        }
        None => vec![],
    };
    let mans: Vec<Manifest> = crate::catalog::manifests().into_iter().filter(|m| filter.is_none() || candidates.iter().any(|c| c == &m.model)).collect();
    if mans.is_empty() {
        eprintln!("no such model '{}' (try `brain caps`)", filter.map(String::as_str).unwrap_or_default());
        return 1;
    }
    if json_out {
        println!("{}", Value::Array(mans.iter().map(|m| m.to_json()).collect()));
        return 0;
    }
    for m in &mans {
        println!("\x1b[1m{}\x1b[0m - {}", m.model, m.summary);
        for a in &m.actions {
            let stream = if a.streaming { " (streaming)" } else { "" };
            println!("  \x1b[36m{}\x1b[0m{stream}: {}", a.name, a.summary);
            for p in &a.params {
                let req = if p.required { " [required]" } else { "" };
                let def = p.default.as_ref().map(|d| format!(" = {d}")).unwrap_or_default();
                let vals = match &p.ty {
                    ParamType::Enum(v) => format!(" {{{}}}", v.join("|")),
                    _ => String::new(),
                };
                println!("      --{} <{}>{vals}{req}{def}  {}", p.name, p.ty.name(), p.help);
            }
            for b in a.inputs.iter() {
                let req = if b.required { " [required]" } else { "" };
                println!("      --in {}=<{}>{req}  {}", b.name, b.media.name(), b.help);
            }
            for b in a.outputs.iter() {
                println!("      --out {}=<{}>  {}", b.name, b.media.name(), b.help);
            }
        }
        println!();
    }
    println!("run one with:  brain <architecture> <action> [--param value]… [--in name=path]… [--out name=path]…");
    0
}

// -------------------------------------------------------- generic dispatch

/// `argv` is `[model id, action, ...flags]` - what `crate::resolve` builds
/// from `brain <architecture> <action> ...` (the arch id translated to its
/// model id first) before calling this.
pub fn run_do(argv: &[String]) -> i32 {
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
    let reg = match crate::catalog::provider(&model).map(|p| {
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
    for spec_val in matches.get_many::<String>("in").unwrap_or_default() {
        let Some((name, path)) = spec_val.split_once('=') else {
            eprintln!("brain: --in must be name=path (got {spec_val:?})");
            return 2;
        };
        match load_blob(&spec, name, path) {
            Ok(b) => inv = inv.blob(name, b),
            Err(e) => {
                eprintln!("brain: {e}");
                return 1;
            }
        }
    }
    let mut out_paths: Vec<(String, String)> = Vec::new();
    for spec_val in matches.get_many::<String>("out").unwrap_or_default() {
        let Some((name, path)) = spec_val.split_once('=') else {
            eprintln!("brain: --out must be name=path (got {spec_val:?})");
            return 2;
        };
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

/// Build a clap parser directly from an [`ActionSpec`]: one typed `--<param>` per
/// param (required/enum/bool honoured by clap), plus repeatable `--in`/`--out
/// name=path` and `--json`. All argument parsing goes through clap - no bespoke loop.
fn build_parser(model: &str, action: &str, spec: &ActionSpec) -> Command {
    let mut cmd = Command::new(format!("brain {model} {action}")).no_binary_name(true).about(spec.summary.clone());
    for p in &spec.params {
        let mut arg = Arg::new(p.name.clone()).long(p.name.clone()).help(p.help.clone());
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
    let in_help = if spec.inputs.is_empty() { "named binary input, e.g. image=in.ppm".to_string() } else { format!("named binary input ({})", spec.inputs.iter().map(|b| format!("{}=<{}>", b.name, b.media.name())).collect::<Vec<_>>().join(", ")) };
    cmd.arg(Arg::new("in").long("in").action(ArgAction::Append).value_name("NAME=PATH").help(in_help))
        .arg(Arg::new("out").long("out").action(ArgAction::Append).value_name("NAME=PATH").help("write a named output blob to a file, e.g. image=out.ppm"))
        .arg(Arg::new("json").long("json").action(ArgAction::SetTrue).help("print scalar outputs as JSON"))
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
/// `{"sample_rate":16000}`); text/bytes are read raw.
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
/// a WAV file; everything else → raw bytes.
fn save_blob(b: &Blob, path: &str) -> Result<(), String> {
    match b.media {
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

/// A trivial always-available model so the generic dispatch path (and the
/// tests) work with no weights - and as a worked example of the
/// [`Provider`]/[`Action`] pattern.
pub(crate) struct DemoModel;
struct EchoAction;

impl Action for EchoAction {
    fn spec(&self) -> ActionSpec {
        use capability::{BlobSpec, ParamSpec};
        ActionSpec::new("echo", "repeat text, optionally upper/lower-cased")
            .param(ParamSpec::new("text", ParamType::Str, "the text").required())
            .param(ParamSpec::new("times", ParamType::Int, "repeat count").default(json!(1)))
            .param(ParamSpec::new("mode", ParamType::Enum(vec!["as-is".into(), "upper".into(), "lower".into()]), "casing").default(json!("as-is")))
            .output(BlobSpec::new("result", Media::Text, "the echoed text"))
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> capability::ActionResult {
        use capability::Outcome;
        let text = inv.get_str("text").unwrap_or_default();
        let n = inv.get_i64("times").unwrap_or(1).max(0) as usize;
        let s = match inv.get_str("mode").as_deref() {
            Some("upper") => text.to_uppercase(),
            Some("lower") => text.to_lowercase(),
            _ => text,
        };
        progress(Progress::step(1, 1, "echoing"));
        let out = s.repeat(n);
        Ok(Outcome::new().set("chars", json!(out.len())).blob("result", Blob::new(Media::Text, out.into_bytes())))
    }
}

impl Provider for DemoModel {
    fn manifest(&self) -> Manifest {
        Manifest::new(DEMO_MODEL, "a trivial always-available model (no weights) - a worked example of the capability interface", vec![EchoAction.spec()])
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "echo").then(|| Arc::new(EchoAction) as Arc<dyn Action>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::BlobSpec;

    fn audio_spec() -> ActionSpec {
        ActionSpec::new("transcribe", "transcribe").input(BlobSpec::new("audio", Media::Audio, "raw mono f32 LE PCM at 16 kHz").required())
    }

    /// A `--in audio=clip.wav` must be DECODED, not handed to the model as the
    /// literal RIFF bytes: same samples as a direct `wav::parse` +
    /// `resample_linear`, and tagged with the 16 kHz meta the ASR guards check.
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
}
