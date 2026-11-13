// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision-model lifecycle as served actions: the provider that runs
//! `train`, `eval`, `promote`, `rollback` and `inspect` of `brain/decision`
//! (declared by [`decisions::lifecycle`]) over [`crate::decision_run`].
//!
//! A host attaches it to the family provider
//! (`DecisionProvider::with_lifecycle`), which is what makes
//! `brain decide train ...` work through the generic `brain <model> <action>`
//! dispatch. Every input is a file on the host, so a serving process does not
//! register these actions.
//!
//! A result comes back as typed outputs plus a `report` text blob holding the
//! table the command prints; `eval` and `promote` also return the evaluation
//! `record` the checkpoint stores beside its manifest.
//!
//! Swedish Embedded AB implements the tooling that turns recorded outcomes
//! into governed decision models, from the first training run to a promotion
//! gate. If your team needs expertise in decision-model lifecycle, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::PathBuf;
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, Invocation, Manifest, Media, Outcome, Progress, Provider};
use decisions::examples::{Dataset, Split};
use decisions::lifecycle;
use serde_json::json;

use crate::decision_run::{evaluate_model, promote_model, rollback_model, train, EvalRun, PromoteRun, TrainRun};
use crate::Error;

/// The provider of the lifecycle actions; attach it to
/// `decisions::provider::DecisionProvider::with_lifecycle`.
pub fn lifecycle_provider() -> Arc<dyn Provider> {
    Arc::new(LifecycleProvider)
}

struct LifecycleProvider;

impl Provider for LifecycleProvider {
    fn manifest(&self) -> Manifest {
        Manifest::new(decisions::caps::FAMILY, "The decision-model lifecycle: train, evaluate, promote, roll back and inspect.", lifecycle::specs())
    }

    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let spec = lifecycle::specs().into_iter().find(|s| s.name == name)?;
        Some(Arc::new(LifecycleAction { spec }))
    }
}

struct LifecycleAction {
    spec: ActionSpec,
}

impl Action for LifecycleAction {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.spec.name.as_str() {
            "train" => train_action(inv, progress),
            "eval" => eval_action(inv),
            "promote" => promote_action(inv),
            "rollback" => rollback_action(inv),
            "inspect" => inspect_action(inv),
            other => Err(format!("no action '{other}' in the decision lifecycle")),
        }
    }
}

fn required(inv: &Invocation, name: &str) -> Result<String, String> {
    inv.get_str(name).ok_or_else(|| format!("missing required param '{name}'"))
}

fn count(inv: &Invocation, name: &str, default: i64) -> usize {
    inv.get_i64(name).unwrap_or(default).max(1) as usize
}

fn report(text: String) -> Blob {
    Blob::new(Media::Text, text.into_bytes())
}

fn train_action(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let mut run = TrainRun::new(required(inv, "data")?, required(inv, "base")?, required(inv, "out")?);
    run.freeze_encoder = !inv.get_bool("finetune_encoder").unwrap_or(false);
    run.steps = count(inv, "steps", lifecycle::DEFAULT_STEPS);
    run.batch = count(inv, "batch", lifecycle::DEFAULT_BATCH);
    if let Some(seed) = inv.seed()? {
        run.seed = seed;
    }
    run.saved = inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose()?;
    run.brain_commit = code_revision();

    let steps = run.steps;
    let mut last = 0;
    let result = train(&run, &mut |step, loss| {
        last = step + 1;
        progress(Progress::step(last as u32, steps as u32, format!("loss {loss:.4}")));
        !inv.cancel.is_cancelled()
    });
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(Error::Cancelled) => return Err(format!("cancelled after step {last}")),
        Err(e) => return Err(e.to_string()),
    };
    let text = format!(
        "{}wrote {} (final loss {:.4}, unvalidated)\nevaluate with: brain decide eval --model {} --data {} --split validation\n",
        outcome.summary.render(),
        outcome.out.display(),
        outcome.final_loss,
        outcome.out.display(),
        run.data.display()
    );
    Ok(Outcome::new()
        .set("out", json!(outcome.out.display().to_string()))
        .set("final_loss", json!(outcome.final_loss))
        .set("steps", json!(steps))
        .set("batch", json!(run.batch))
        .set("seed", json!(run.seed))
        .set("validation", json!("unvalidated"))
        .blob("report", report(text)))
}

fn eval_action(inv: &Invocation) -> ActionResult {
    let split = required(inv, "split")?;
    let split = Split::parse(&split).filter(|s| *s != Split::Train).ok_or_else(|| format!("split must be `validation` or `test`; the train split is not held out, got {split:?}"))?;
    let mut run = EvalRun::new(required(inv, "model")?, required(inv, "data")?, split);
    run.costs = inv.get_str("costs").map(PathBuf::from);
    if let Some(seed) = inv.seed()? {
        run.order_seed = seed;
    }
    let evaluation = evaluate_model(&run).map_err(|e| e.to_string())?;
    let text = format!("model {}  weights {}\n{}", evaluation.model_id, evaluation.weights_sha256, evaluation.report.render());
    Ok(Outcome::new()
        .set("model", json!(evaluation.model_id))
        .set("weights_sha256", json!(evaluation.weights_sha256))
        .set("seed", json!(run.order_seed))
        .set("record", evaluation.to_json())
        .blob("report", report(text)))
}

fn promote_action(inv: &Invocation) -> ActionResult {
    let outcome = promote_model(&PromoteRun { model: required(inv, "model")?, data: required(inv, "data")?.into(), policy: required(inv, "policy")?.into() }).map_err(|e| e.to_string())?;
    let Some(pointer) = &outcome.pointer else {
        let failures: Vec<String> = outcome.verdict.failures().map(|gate| format!("  {}", gate.message)).collect();
        return Err(format!("{}{}not promoted: {} gate(s) failed; nothing was changed\n{}", outcome.evaluation.report.render(), outcome.verdict, failures.len(), failures.join("\n")));
    };
    let last_known_good = pointer.last_known_good.as_ref().map(|e| e.dir.clone());
    let text = format!(
        "{}{}promoted {} for profile {:?}; last-known-good: {}\n",
        outcome.evaluation.report.render(),
        outcome.verdict,
        pointer.current.dir,
        pointer.profile,
        last_known_good.as_deref().unwrap_or("none")
    );
    Ok(Outcome::new()
        .set("promoted", json!(true))
        .set("current", json!(pointer.current.dir))
        .set("profile", json!(pointer.profile))
        .set("last_known_good", json!(last_known_good))
        .set("record", outcome.evaluation.to_json())
        .set("verdict", outcome.verdict.to_json())
        .blob("report", report(text)))
}

fn rollback_action(inv: &Invocation) -> ActionResult {
    let pointer = rollback_model(&required(inv, "model")?).map_err(|e| e.to_string())?;
    let text = format!("rolled back: {} is validated again for profile {:?}\n", pointer.current.dir, pointer.profile);
    Ok(Outcome::new().set("current", json!(pointer.current.dir)).set("profile", json!(pointer.profile)).blob("report", report(text)))
}

fn inspect_action(inv: &Invocation) -> ActionResult {
    let data = PathBuf::from(required(inv, "data")?);
    let dataset = Dataset::read_file(&data).map_err(|e| format!("{}: {e}", data.display()))?;
    let summary = dataset.summary();
    Ok(Outcome::new().set("data_sha256", json!(dataset.digest)).set("summary", summary.to_json()).blob("report", report(summary.render())))
}

/// `git rev-parse --short HEAD` of the checkout this binary was built from,
/// `-dirty` when its tree has changes, `unknown` when git cannot say.
fn code_revision() -> String {
    let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(PathBuf::from)) else { return "unknown".into() };
    let git = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(&dir).args(args).output().ok().filter(|o| o.status.success());
    let Some(head) = git(&["rev-parse", "--short", "HEAD"]).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).filter(|s| !s.is_empty()) else {
        return "unknown".into();
    };
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|o| !o.stdout.is_empty());
    if dirty { format!("{head}-dirty") } else { head }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declared_activation_choices_are_the_ones_the_policy_parser_accepts() {
        assert_eq!(lifecycle::ACTIVATIONS, model::saved::SavedPolicy::ACTIVATIONS_CHOICES);
    }

    #[test]
    fn every_declared_action_has_a_runner() {
        let provider = LifecycleProvider;
        for name in lifecycle::ACTIONS {
            assert!(provider.action(name).is_some(), "{name}");
        }
    }
}
