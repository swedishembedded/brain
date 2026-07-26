// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain caps` / `brain do` — the generalized capability interface on the CLI.
//!
//! `brain caps [model] [--json]` lists what every supported model can do (static
//! manifests — no weights loaded). `brain do <model> <action> [--param value]…
//! [--in name=path]… [--out name=path]… [--json]` validates the arguments against
//! the action's schema, runs it, and writes the output blobs to files.
//!
//! Neither command knows anything model-specific: both go through
//! `capability::Registry`. A new model shows up here the moment it provides a
//! `capability::Manifest` (discovery) and a `Provider` (execution) — see
//! [`static_manifests`] and [`build_registry`].

use std::sync::Arc;

use capability::{Action, ActionSpec, Blob, Invocation, Manifest, Media, ParamType, Progress, Provider, Registry};
use serde_json::{json, Value};

/// Every model's static capability manifest (discovery — no weights). Add a model
/// here and it appears in `brain caps` immediately.
fn static_manifests() -> Vec<Manifest> {
    vec![zimage::caps::manifest(), DemoModel.manifest()]
}

/// Build an executable registry for `model` (loads what that model needs). `do`
/// only constructs the one model it was asked to run.
fn build_registry(model: &str) -> Result<Registry, String> {
    let mut reg = Registry::new();
    match model {
        "demo" => reg.register(Arc::new(DemoModel)),
        zimage::caps::MODEL => reg.register(Arc::new(zimage::caps::ZImageProvider::load()?)),
        other => return Err(format!("unknown model '{other}' (see `brain caps`)")),
    }
    Ok(reg)
}

// ---------------------------------------------------------------- brain caps

pub fn run_caps(argv: &[String]) -> i32 {
    let json_out = argv.iter().any(|a| a == "--json");
    let model = argv.iter().find(|a| !a.starts_with("--")).cloned();
    let mans: Vec<Manifest> = static_manifests().into_iter().filter(|m| model.as_deref().is_none_or(|w| w == m.model)).collect();
    if mans.is_empty() {
        eprintln!("no such model '{}' (try `brain caps`)", model.unwrap_or_default());
        return 1;
    }
    if json_out {
        println!("{}", Value::Array(mans.iter().map(|m| m.to_json()).collect()));
        return 0;
    }
    for m in &mans {
        println!("\x1b[1m{}\x1b[0m — {}", m.model, m.summary);
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
    println!("run one with:  brain do <model> <action> [--param value]… [--in name=path]… [--out name=path]…");
    0
}

// ---------------------------------------------------------------- brain do

pub fn run_do(argv: &[String]) -> i32 {
    let (model, action) = match (argv.first(), argv.get(1)) {
        (Some(m), Some(a)) if !m.starts_with("--") && !a.starts_with("--") => (m.clone(), a.clone()),
        _ => {
            eprintln!("usage: brain do <model> <action> [--param value]… [--in name=path]… [--out name=path]…");
            return 2;
        }
    };
    let reg = match build_registry(&model) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("brain do: {e}");
            return 1;
        }
    };
    let act = match reg.find(&model, &action) {
        Some(a) => a,
        None => {
            eprintln!("brain do: model '{model}' has no action '{action}' (see `brain caps {model}`)");
            return 1;
        }
    };
    let spec = act.spec();

    // parse --param value / --in name=path / --out name=path / --json
    let mut inv = Invocation::new();
    let mut out_paths: Vec<(String, String)> = Vec::new();
    let mut json_out = false;
    let mut i = 2;
    while i < argv.len() {
        let a = &argv[i];
        match a.as_str() {
            "--json" => {
                json_out = true;
                i += 1;
            }
            "--in" => {
                let Some(kv) = argv.get(i + 1).and_then(|s| s.split_once('=')) else {
                    eprintln!("brain do: --in needs name=path");
                    return 2;
                };
                match load_blob(&spec, kv.0, kv.1) {
                    Ok(b) => inv = inv.blob(kv.0, b),
                    Err(e) => {
                        eprintln!("brain do: {e}");
                        return 1;
                    }
                }
                i += 2;
            }
            "--out" => {
                let Some(kv) = argv.get(i + 1).and_then(|s| s.split_once('=')) else {
                    eprintln!("brain do: --out needs name=path");
                    return 2;
                };
                out_paths.push((kv.0.to_string(), kv.1.to_string()));
                i += 2;
            }
            flag if flag.starts_with("--") => {
                let name = &flag[2..];
                let Some(ps) = spec.params.iter().find(|p| p.name == name) else {
                    eprintln!("brain do: unknown param --{name} (see `brain caps {model}`)");
                    return 1;
                };
                // bool params act as flags; others consume the next token.
                if ps.ty == ParamType::Bool {
                    inv = inv.set(name, json!(true));
                    i += 1;
                } else {
                    let Some(val) = argv.get(i + 1) else {
                        eprintln!("brain do: --{name} needs a value");
                        return 2;
                    };
                    inv = inv.set(name, coerce(&ps.ty, val));
                    i += 2;
                }
            }
            other => {
                eprintln!("brain do: unexpected argument '{other}'");
                return 2;
            }
        }
    }

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
            eprintln!("\nbrain do: {e}");
            return 1;
        }
    };
    eprintln!();

    // write output blobs
    for (name, path) in &out_paths {
        match outcome.blobs.get(name) {
            Some(b) => {
                if let Err(e) = save_blob(b, path) {
                    eprintln!("brain do: writing {path}: {e}");
                    return 1;
                }
                eprintln!("wrote {name} → {path} ({} bytes)", b.bytes.len());
            }
            None => eprintln!("brain do: action produced no output '{name}'"),
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
/// text/bytes are read raw.
fn load_blob(spec: &ActionSpec, name: &str, path: &str) -> Result<Blob, String> {
    let media = spec.inputs.iter().find(|b| b.name == name).map(|b| b.media).ok_or_else(|| format!("action has no input '{name}'"))?;
    match media {
        Media::Image | Media::Mask => {
            let (hwc, w, h) = crate::image_io::load_image(path)?;
            let c = hwc.len() / (w as usize * h as usize);
            let bytes: Vec<u8> = hwc.iter().flat_map(|f| f.to_le_bytes()).collect();
            Ok(Blob::new(media, bytes).with_meta(json!({"w": w, "h": h, "c": c})))
        }
        _ => std::fs::read(path).map(|b| Blob::new(media, b)).map_err(|e| e.to_string()),
    }
}

/// Write a [`Blob`] to a file: images (raw HWC f32 + `{w,h,c}` meta) → binary PPM
/// (P6, the brain image convention); everything else → raw bytes.
fn save_blob(b: &Blob, path: &str) -> Result<(), String> {
    match b.media {
        Media::Image | Media::Mask => {
            let w = b.meta["w"].as_u64().ok_or("image blob missing w")? as u32;
            let h = b.meta["h"].as_u64().ok_or("image blob missing h")? as u32;
            let c = b.meta["c"].as_u64().unwrap_or(3) as usize;
            let hwc: Vec<f32> = b.bytes.chunks_exact(4).map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]])).collect();
            // to interleaved u8 RGB (replicate grayscale to 3 channels for P6).
            let n = w as usize * h as usize;
            let mut rgb = vec![0u8; n * 3];
            for i in 0..n {
                for ch in 0..3 {
                    let v = if c >= 3 { hwc[i * c + ch] } else { hwc[i * c] };
                    rgb[i * 3 + ch] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
            std::fs::write(path, events::ppm::encode_p6(&rgb, w, h)).map_err(|e| e.to_string())
        }
        _ => std::fs::write(path, &b.bytes).map_err(|e| e.to_string()),
    }
}

// ---------------------------------------------------------------- built-in demo provider

/// A trivial always-available model so `brain do` (and the tests) work with no
/// weights — and as a worked example of the [`Provider`]/[`Action`] pattern.
struct DemoModel;
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
        progress(Progress { step: 1, total: 1, message: "echoing".into() });
        let out = s.repeat(n);
        Ok(Outcome::new().set("chars", json!(out.len())).blob("result", Blob::new(Media::Text, out.into_bytes())))
    }
}

impl Provider for DemoModel {
    fn manifest(&self) -> Manifest {
        Manifest::new("demo", "a trivial always-available model (no weights) — a worked example of the capability interface", vec![EchoAction.spec()])
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "echo").then(|| Arc::new(EchoAction) as Arc<dyn Action>)
    }
}
