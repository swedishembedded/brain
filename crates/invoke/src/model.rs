// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One model's whole command line: `<program> <action> [flags]`.
//!
//! The order is what keeps it cheap and robust: help and the action's own
//! flags are answered from the static manifest, so `--help` and a bad value
//! cost no weight lookup; the weights are resolved next; the provider is built
//! last, and the action validates every value before it loads anything.

use std::collections::BTreeMap;
use std::sync::Arc;

use brain_modelstore::resolve::ArchSpec;
use capability::{Assembly, Manifest, ParamType, Provider};
use loader::resolver::ResolveFailure;

use crate::run::{run_with, usage, Program};

/// Builds a model's provider over its resolved weights.
pub type BuildProvider<'a> = dyn Fn(&Assembly) -> Result<Arc<dyn Provider>, String> + 'a;

/// How a model is found and built.
pub struct Model<'a> {
    /// The static manifest: every action and param, no weights needed.
    pub manifest: Manifest,
    /// The resolver architecture and spec its weights come from; `None` for a
    /// model that reads no weights through the model store.
    pub weights: Option<(&'a str, &'a dyn ArchSpec)>,
    /// Build the provider over resolved weights.
    pub build: &'a BuildProvider<'a>,
}

/// Run `argv` (`<action> [flags]`) against `model`. Returns the exit code.
pub fn run_model(program: &str, model: &Model, argv: &[String]) -> i32 {
    let p = Program { name: program };
    let help = argv.iter().any(|a| a == "-h" || a == "--help");
    let Some(action) = argv.first().filter(|a| !a.starts_with('-')) else {
        print!("{}", usage(&p, &model.manifest));
        if let Some((_, spec)) = model.weights {
            print!("\nWEIGHTS (resolved from the model store unless named):\n{}", crate::resolve::role_help(spec));
        }
        print!("\nGLOBAL:\n{GLOBAL_HELP}");
        return if help { 0 } else { 2 };
    };
    let Some(spec) = model.manifest.actions.iter().find(|a| a.name == *action) else {
        let names: Vec<&str> = model.manifest.actions.iter().map(|a| a.name.as_str()).collect();
        eprintln!("{program}: no action '{action}'; it has: {}", names.join(", "));
        return 2;
    };
    if help {
        let mut cmd = crate::args::command(&format!("{program} {action}"), spec);
        let _ = cmd.print_long_help();
        if let Some((_, arch_spec)) = model.weights {
            print!("\nWeights (resolved from the model store unless named):\n{}", crate::resolve::role_help(arch_spec));
        }
        return 0;
    }
    let (overrides, flags) = match model.weights {
        None => (BTreeMap::new(), argv[1..].to_vec()),
        Some((_, arch_spec)) => crate::resolve::role_overrides(arch_spec, &argv[1..]),
    };
    // The command line is checked against the action before any weight is
    // looked up, so a bad value is named as such rather than surfacing as a
    // checkpoint the store cannot pick.
    if let Err(code) = crate::args::check(&format!("{program} {action}"), spec, &flags) {
        return code;
    }
    let assembly = match model.weights {
        None => crate::resolve::empty_assembly(),
        Some((arch, arch_spec)) => match resolve_for(arch, arch_spec, overrides, spec, &flags) {
            Ok(a) => a,
            Err(e) => {
                eprint!("{}", e.message());
                return e.exit_code();
            }
        },
    };
    let provider = match (model.build)(&assembly) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{program}: {e}");
            return 1;
        }
    };
    match run_with(&p, provider, action, &flags) {
        Ok(_) => 0,
        Err(code) => code,
    }
}

/// Resolve the weights for one action. Two checkpoints of one model that tie
/// (a tiny and a large release side by side) are settled by the action's
/// `variant`: the one given on the command line, else the param's default -
/// only a checkpoint of that variant assembles, so the choice is the user's
/// stated one, never a guess.
fn resolve_for(arch: &str, spec: &dyn ArchSpec, mut overrides: BTreeMap<String, String>, action: &capability::ActionSpec, flags: &[String]) -> Result<Assembly, ResolveFailure> {
    let dir = loader::model_dir::resolve(None);
    let variant_param = action.params.iter().find(|p| p.name == "variant" && matches!(p.ty, ParamType::Enum(_) | ParamType::Str));
    let given = flags.iter().position(|f| f == "--variant").and_then(|i| flags.get(i + 1)).cloned();
    if let (Some(_), Some(v)) = (variant_param, &given) {
        overrides.entry("variant".to_string()).or_insert_with(|| v.clone());
    }
    match loader::resolver::try_resolve(dir.as_deref(), arch, spec, &overrides) {
        Err(ResolveFailure::Ambiguous(why)) => {
            let default = variant_param.and_then(|p| p.default.as_ref()).and_then(|d| d.as_str()).map(str::to_string);
            match default.filter(|_| !overrides.contains_key("variant")) {
                Some(v) => {
                    overrides.insert("variant".to_string(), v);
                    loader::resolver::try_resolve(dir.as_deref(), arch, spec, &overrides).map_err(|_| ResolveFailure::Ambiguous(why))
                }
                None => Err(ResolveFailure::Ambiguous(why)),
            }
        }
        other => other,
    }
}

/// The whole `main` of a per-model program: the global options every brain
/// program takes (`--device`, `--backend`, `--brain-data-dir`), then
/// [`run_model`], then a clean exit that lets the devices drain first.
pub fn main(program: &str, model: &Model) -> ! {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let code = match global_options(&argv) {
        Ok(rest) => run_model(program, model, &rest),
        Err(e) => {
            eprintln!("{program}: {e}");
            2
        }
    };
    // A device dropped while its driver call is still in flight can crash
    // the process at exit; drain them, then leave without running atexit.
    gpu_core::drain_all_devices(std::time::Duration::from_secs(5));
    gpu_core::set_process_exiting();
    gpu_core::hard_exit(code)
}

/// Take and apply the global options, returning the rest in order.
fn global_options(argv: &[String]) -> Result<Vec<String>, String> {
    use appopts::Options;
    let mut args = appopts::Args::new(argv);
    let hardware = appopts::Hardware::take(&mut args)?;
    if let Some(root) = args.take_str("--brain-data-dir") {
        brain_modelstore::publish_data_root(Some(root.into()));
    }
    if !hardware.device.is_all() || hardware.backend.is_some() {
        hardware.apply()?;
    }
    Ok(args.leftovers())
}

/// The global options' help lines.
pub const GLOBAL_HELP: &str = "  --device SPEC          cpu | gpu | gpu0 | gpu0,gpu1 | cpu0-7 ...   [every device]
  --backend NAME         wgpu | vulkan | cuda | cpu                  [the device's own]
  --brain-data-dir DIR   brain's data root; the model store is DIR/models
";
