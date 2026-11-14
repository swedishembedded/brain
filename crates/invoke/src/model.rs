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

/// A step a program builds on a model's actions (chaining clips, say),
/// offered beside the actions themselves.
pub struct Workflow<'a> {
    pub name: &'a str,
    pub summary: &'a str,
    /// The action it runs: its weights are resolved as for that action, and
    /// `--help` lists that action's flags after the workflow's own.
    pub action: &'a str,
    /// The workflow's own arguments, for `--help`.
    pub usage: &'a str,
    pub run: &'a dyn Fn(&WorkflowCall) -> Result<(), String>,
}

/// What a running workflow is handed.
pub struct WorkflowCall<'a> {
    /// `<program> <workflow>`, for messages.
    pub program: &'a str,
    pub provider: Arc<dyn Provider>,
    /// The checkpoints the provider was built from, for a workflow that
    /// drives the model directly (a live camera loop, say) rather than
    /// through one action call per step.
    pub assembly: &'a Assembly,
    /// Its arguments, the weight-role flags already taken out.
    pub args: &'a [String],
}

/// Run `argv` (`<action> [flags]`) against `model`. Returns the exit code.
pub fn run_model(program: &str, model: &Model, argv: &[String]) -> i32 {
    run_model_with(program, model, &[], argv)
}

/// [`run_model`], with `workflows` offered beside the actions.
pub fn run_model_with(program: &str, model: &Model, workflows: &[Workflow], argv: &[String]) -> i32 {
    // `--autofetch` is a run-level flag, not an action param: it decides
    // whether a checkpoint the store lacks may be downloaded.
    let fetch = argv.iter().any(|a| a == "--autofetch");
    let argv: Vec<String> = argv.iter().filter(|a| *a != "--autofetch").cloned().collect();
    let argv = &argv[..];
    let p = Program { name: program };
    let help = argv.iter().any(|a| a == "-h" || a == "--help");
    if let Some(wf) = argv.first().and_then(|a| workflows.iter().find(|w| w.name == a)) {
        return run_workflow(program, model, wf, &argv[1..], help, fetch);
    }
    let Some(action) = argv.first().filter(|a| !a.starts_with('-')) else {
        print!("{}", usage(&p, &model.manifest));
        if !workflows.is_empty() {
            println!("\nWORKFLOWS:");
            for w in workflows {
                println!("  {:<18} {}", w.name, w.summary);
            }
        }
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
        Some((arch, arch_spec)) => match resolve_for(arch, arch_spec, overrides, spec, &flags, fetch) {
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

fn run_workflow(program: &str, model: &Model, wf: &Workflow, args: &[String], help: bool, fetch: bool) -> i32 {
    let name = format!("{program} {}", wf.name);
    let Some(spec) = model.manifest.actions.iter().find(|a| a.name == wf.action) else {
        eprintln!("{name}: the model has no action '{}'", wf.action);
        return 2;
    };
    if help {
        println!("{name} - {}\n\nUSAGE: {name} {}\n\nEvery flag of `{program} {}` is taken too:", wf.summary, wf.usage, wf.action);
        let _ = crate::args::command(&format!("{program} {}", wf.action), spec).print_long_help();
        return 0;
    }
    let (assembly, rest) = match model.weights {
        None => (crate::resolve::empty_assembly(), args.to_vec()),
        Some((arch, arch_spec)) => {
            let (overrides, rest) = crate::resolve::role_overrides(arch_spec, args);
            match resolve_for(arch, arch_spec, overrides, spec, &rest, fetch) {
                Ok(a) => (a, rest),
                Err(e) => {
                    eprint!("{}", e.message());
                    return e.exit_code();
                }
            }
        }
    };
    let provider = match (model.build)(&assembly) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{name}: {e}");
            return 1;
        }
    };
    match (wf.run)(&WorkflowCall { program: &name, provider, assembly: &assembly, args: &rest }) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{name}: {e}");
            1
        }
    }
}

/// Resolve the weights for one action. Two checkpoints of one model that tie
/// (a tiny and a large release side by side) are settled by the action's
/// `variant`: the one given on the command line, else the param's default -
/// only a checkpoint of that variant assembles, so the choice is the user's
/// stated one, never a guess. With `fetch` (`--autofetch`), a store with no
/// checkpoint fetches the architecture's default one and resolves again.
fn resolve_for(arch: &str, spec: &dyn ArchSpec, overrides: BTreeMap<String, String>, action: &capability::ActionSpec, flags: &[String], fetch: bool) -> Result<Assembly, ResolveFailure> {
    let named = !overrides.is_empty();
    let first = match resolve_in_store(arch, spec, overrides.clone(), action, flags) {
        Err(ResolveFailure::Missing(_)) if fetch && !named => match loader::ensure_default_weights(arch, loader::DownloadPolicy::IfMissing) {
            Ok(_) => resolve_in_store(arch, spec, overrides, action, flags),
            Err(e) => {
                eprintln!("{arch}: --autofetch could not fetch a default checkpoint ({e})");
                resolve_in_store(arch, spec, overrides, action, flags)
            }
        },
        other => other,
    };
    match first {
        // A fresh model needs no checkpoint, so a store with none - or with several that tie, none of which it would
        // start from - does not stop it; one the caller named must exist.
        Err(ResolveFailure::Missing(_) | ResolveFailure::NoModelsDir(_) | ResolveFailure::Ambiguous(_)) if action.from_scratch && !named => Ok(crate::resolve::empty_assembly()),
        other => other,
    }
}

fn resolve_in_store(arch: &str, spec: &dyn ArchSpec, mut overrides: BTreeMap<String, String>, action: &capability::ActionSpec, flags: &[String]) -> Result<Assembly, ResolveFailure> {
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
                    under_default_variant(why, loader::resolver::try_resolve(dir.as_deref(), arch, spec, &overrides))
                }
                None => Err(ResolveFailure::Ambiguous(why)),
            }
        }
        other => other,
    }
}

/// What a store ambiguous without a variant answers once resolved again under
/// the action's default one: that resolution, or its own question when it
/// still asks one (it is the narrower question, the one left to answer), or
/// else the first question, since nothing of the default variant is there.
fn under_default_variant(first: String, retry: Result<Assembly, ResolveFailure>) -> Result<Assembly, ResolveFailure> {
    match retry {
        Err(ResolveFailure::Ambiguous(narrower)) => Err(ResolveFailure::Ambiguous(narrower)),
        Err(_) => Err(ResolveFailure::Ambiguous(first)),
        resolved => resolved,
    }
}

/// The whole `main` of a per-model program: the global options every brain
/// program takes (`--device`, `--backend`, `--brain-data-dir`), then
/// [`run_model`], then a clean exit that lets the devices drain first.
pub fn main(program: &str, model: &Model) -> ! {
    main_with(program, model, &[])
}

/// [`main`], with `workflows` offered beside the actions.
pub fn main_with(program: &str, model: &Model, workflows: &[Workflow]) -> ! {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let code = match global_options(&argv) {
        Ok(rest) => run_model_with(program, model, workflows, &rest),
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
  --autofetch            download the model's default checkpoint when the store has none
  --backend NAME         wgpu | vulkan | cuda | cpu                  [the device's own]
  --brain-data-dir DIR   brain's data root; the model store is DIR/models
";

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolving again under the default variant keeps the narrower question
    /// it still asks (a role left to choose once the variant settled the
    /// rest), and falls back to the first question only when the default
    /// variant has nothing in the store.
    #[test]
    fn the_default_variant_retry_reports_the_question_left_to_answer() {
        let first = || "which dit?".to_string();
        let assembly = || Assembly { id: "local/x".into(), arch: "x".into(), variant: None, roles: Default::default(), provenance: Vec::new() };
        let narrower = under_default_variant(first(), Err(ResolveFailure::Ambiguous("which vae?".into())));
        assert!(matches!(narrower, Err(ResolveFailure::Ambiguous(m)) if m == "which vae?"));
        let absent = under_default_variant(first(), Err(ResolveFailure::Missing("no 4B dit".into())));
        assert!(matches!(absent, Err(ResolveFailure::Ambiguous(m)) if m == "which dit?"));
        assert!(matches!(under_default_variant(first(), Ok(assembly())), Ok(a) if a.id == "local/x"));
    }
}
