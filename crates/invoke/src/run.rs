// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One action, from its command line to its written outputs.

use std::sync::Arc;

use capability::{Manifest, Media, Outcome, Progress, Provider};

use crate::{args, media};

/// A front end's name for itself and the model it drives, as its help and
/// errors print them: `brain do brain/sam2`, or a sample's binary name.
pub struct Program<'a> {
    pub name: &'a str,
}

/// Run the action `argv[0]` of `provider` with the flags after it, write the
/// outputs it was asked for and print its scalar outputs. Returns the
/// process exit code: 0 on success, 2 for a malformed command line, 1 for a
/// failed run. `-h`/`--help` (or no action) lists the actions.
pub fn run(program: &Program, provider: Arc<dyn Provider>, argv: &[String]) -> i32 {
    let manifest = provider.manifest();
    let Some(action) = argv.first().filter(|a| !a.starts_with('-')) else {
        print!("{}", usage(program, &manifest));
        return if argv.iter().any(|a| a == "-h" || a == "--help") { 0 } else { 2 };
    };
    match run_with(program, provider, action, &argv[1..]) {
        Ok(_) => 0,
        Err(code) => code,
    }
}

/// [`run`] for a named `action`, returning its [`Outcome`] to a caller that
/// goes on to use it (a sample chaining actions). The error is the exit code;
/// its message is already on stderr.
pub fn run_with(program: &Program, provider: Arc<dyn Provider>, action: &str, flags: &[String]) -> Result<Outcome, i32> {
    let manifest = provider.manifest();
    let Some(act) = provider.action(action) else {
        eprintln!("{}: no action '{action}'; it has: {}", program.name, names(&manifest));
        return Err(2);
    };
    let spec = act.spec();
    let cmd = args::command(&format!("{} {action}", program.name), &spec);
    let matches = match cmd.try_get_matches_from(flags) {
        Ok(m) => m,
        Err(e) => {
            let _ = e.print();
            let help = matches!(e.kind(), clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion);
            return Err(if help { 0 } else { 2 });
        }
    };
    let parsed = args::parse(&spec, &matches).map_err(|e| {
        eprintln!("{}: {e}", program.name);
        2
    })?;
    let mut inv = parsed.invocation;
    for (name, path) in &parsed.inputs {
        let blob = media::load_blob(&spec, name, path).map_err(|e| {
            eprintln!("{}: {e}", program.name);
            1
        })?;
        inv = inv.blob(name, blob);
    }
    // The action's own validation (types, bounds, required inputs) runs
    // before any weight is loaded, so a bad value costs nothing.
    let inv = spec.validate(inv).map_err(|e| {
        eprintln!("{} {action}: {e}", program.name);
        2
    })?;
    let mut progress = |p: Progress| {
        if p.total > 0 {
            eprint!("\r\x1b[2K{action} [{}/{}] {}", p.step, p.total, p.message);
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
    };
    let outcome = act.run(&inv, &mut progress).map_err(|e| {
        eprintln!("\n{} {action}: {e}", program.name);
        1
    })?;
    eprintln!();
    for (name, path) in &parsed.outputs {
        let Some(b) = outcome.blobs.get(name) else {
            eprintln!("{} {action}: the run made no output '{name}'", program.name);
            return Err(1);
        };
        // A clip carries the outcome's own sound as its track.
        let sound = (b.media == Media::Video).then(|| outcome.blobs.get("audio")).flatten();
        media::save_blob(b, path, sound).map_err(|e| {
            eprintln!("{} {action}: writing {path}: {e}", program.name);
            1
        })?;
        eprintln!("wrote {name} -> {path}");
    }
    if parsed.json {
        println!("{}", outcome.outputs);
    } else if let Some(obj) = outcome.outputs.as_object() {
        for (k, v) in obj {
            println!("{k}: {v}");
        }
    }
    Ok(outcome)
}

fn names(m: &Manifest) -> String {
    m.actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", ")
}

/// The action list `-h` prints.
pub fn usage(program: &Program, m: &Manifest) -> String {
    let mut s = format!("{} - {}\n\nUSAGE: {} <action> [--param value]... [--<input> PATH]... [--out name=PATH]...\n\nACTIONS:\n", program.name, m.summary, program.name);
    for a in &m.actions {
        s.push_str(&format!("  {:<18} {}\n", a.name, a.summary));
    }
    s.push_str(&format!("\n`{} <action> --help` lists every parameter of one action and its default.\n", program.name));
    s
}
