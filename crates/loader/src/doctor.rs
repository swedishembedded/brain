// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-serving setups that explain what is
// wrong before a model is loaded, for its clients. If your team needs
// expertise in making local inference deployments diagnosable, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `brain doctor`: whether a model can be served here, answered from the model
//! store, the disk and the GPUs' free memory without loading any weights,
//! opening the network or allocating on a card.
//!
//! Each check is a pure function of what it is handed, so the readings
//! (free disk, per-card memory) are injected: the CLI takes the real ones, a
//! test hands in the ones it needs. A failed check says what to change.

use std::path::{Path, PathBuf};

use brain_modelstore::{Format, LocalError, LocalModel, Store};
use gpu_core::capacity::GpuMem;

/// Free space below which nothing can be pulled or cached.
pub const FAIL_BELOW_FREE_BYTES: u64 = 1 << 30;
/// Free space below which a pull of a mid-size model may run out.
pub const WARN_BELOW_FREE_BYTES: u64 = 10 << 30;

const GIB: u64 = 1 << 30;

/// How a check came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Nothing to do.
    Ok,
    /// Usable, with something worth knowing.
    Warn,
    /// Not usable as it is.
    Fail,
}

impl Verdict {
    /// The word a report prints and `--json` carries.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warn => "warn",
            Self::Fail => "fail",
        }
    }
}

/// One thing checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Check {
    /// What was checked: `store`, `disk`, `gpu`, `model`, `tokenizer`.
    pub name: &'static str,
    /// How it came out.
    pub verdict: Verdict,
    /// What was found.
    pub detail: String,
    /// What to change, when it is not `ok`.
    pub fix: Option<String>,
}

impl Check {
    fn ok(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, verdict: Verdict::Ok, detail: detail.into(), fix: None }
    }

    fn problem(verdict: Verdict, name: &'static str, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self { name, verdict, detail: detail.into(), fix: Some(fix.into()) }
    }

    /// The check as the JSON object `--json` prints.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "name": self.name, "verdict": self.verdict.word(), "detail": self.detail, "fix": self.fix })
    }
}

/// Every check made, in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// The checks.
    pub checks: Vec<Check>,
}

impl Report {
    /// Whether serving can start: no check failed. Warnings do not stop it.
    #[must_use]
    pub fn ready(&self) -> bool {
        self.checks.iter().all(|c| c.verdict != Verdict::Fail)
    }

    /// The report as the JSON document `--json` prints.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "ready": self.ready(), "checks": self.checks.iter().map(Check::to_json).collect::<Vec<_>>() })
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}

/// The model store at `root`: that it exists, can be read, and holds models.
#[must_use]
pub fn store_check(root: Option<&Path>) -> Check {
    let Some(root) = root else {
        return Check::problem(
            Verdict::Fail,
            "store",
            "no model store is configured",
            "pass --models-dir <directory>",
        );
    };
    if let Err(e) = std::fs::read_dir(root) {
        let fix = if e.kind() == std::io::ErrorKind::NotFound {
            "create it, or pass --models-dir <directory> naming the directory that holds the models"
        } else {
            "give this user read access to it"
        };
        return Check::problem(Verdict::Fail, "store", format!("{} cannot be read: {e}", root.display()), fix);
    }
    let models = Store::new(root).scan();
    if models.is_empty() {
        return Check::problem(
            Verdict::Warn,
            "store",
            format!("{} holds no servable model", root.display()),
            "fetch one with `brain pull <vendor>/<repo>`",
        );
    }
    Check::ok("store", format!("{} holds {} servable model(s)", root.display(), models.len()))
}

/// The free space under the store, as the caller read it.
#[must_use]
pub fn disk_check(free_bytes: Option<u64>) -> Check {
    let fix = "free space, or move the model store to a larger filesystem";
    match free_bytes {
        None => Check::problem(Verdict::Warn, "disk", "the free space of the model store could not be read", fix),
        Some(free) if free < FAIL_BELOW_FREE_BYTES => Check::problem(Verdict::Fail, "disk", format!("{} free", gib(free)), fix),
        Some(free) if free < WARN_BELOW_FREE_BYTES => Check::problem(Verdict::Warn, "disk", format!("{} free", gib(free)), fix),
        Some(free) => Check::ok("disk", format!("{} free", gib(free))),
    }
}

/// The GPUs, and when `weights_bytes` is known whether some card has room for
/// them right now. Brain places a model no card can hold on the CPU backend,
/// which turns minutes into hours; this says so first. An estimate: it counts
/// the weights, not the activations or the KV cache.
#[must_use]
pub fn gpu_check(cards: &[GpuMem], weights_bytes: Option<u64>) -> Check {
    let fix = "free GPU memory (stop the processes using it) before serving";
    if cards.is_empty() {
        return Check::problem(
            Verdict::Warn,
            "gpu",
            "no GPU found: models run on the CPU backend, far slower",
            "run on a machine with a GPU",
        );
    }
    if !cards.iter().any(|c| c.measured) {
        return Check::problem(
            Verdict::Warn,
            "gpu",
            "the free memory of the GPUs could not be read, so whether a model fits is unknown",
            "install the driver's management tool (nvidia-smi) so free memory can be measured",
        );
    }
    let listing = cards
        .iter()
        .filter(|c| c.measured)
        .map(|c| format!("gpu{} {} free of {}", c.index, gib(c.free), gib(c.total)))
        .collect::<Vec<_>>()
        .join(", ");
    let Some(need) = weights_bytes else {
        return Check::ok("gpu", listing);
    };
    match cards.iter().filter(|c| c.measured && c.free >= need).max_by_key(|c| c.free) {
        Some(best) => Check::ok("gpu", format!("the weights ({}) fit on gpu{} ({listing})", gib(need), best.index)),
        None => Check::problem(
            Verdict::Warn,
            "gpu",
            format!("the weights ({}) fit on no GPU right now ({listing}): brain would run the model on the CPU backend", gib(need)),
            fix,
        ),
    }
}

/// The bytes a model's weights take on disk: the weights file, or for a
/// compound model every file its roles name (a role may be a directory).
#[must_use]
pub fn weights_bytes(model: &LocalModel) -> u64 {
    match (&model.format, &model.roles) {
        (Format::Compound, Some(roles)) => roles.values().map(|p| path_bytes(p)).sum(),
        _ => path_bytes(&model.weights),
    }
}

fn path_bytes(path: &Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else { return 0 };
    if meta.is_file() {
        return meta.len();
    }
    std::fs::read_dir(path)
        .map(|entries| entries.flatten().map(|e| path_bytes(&e.path())).sum())
        .unwrap_or(0)
}

/// What is wrong with a model the store could not serve, from the reason the
/// store gave.
#[must_use]
pub fn missing_model_check(reference: &str, why: &LocalError) -> Check {
    match why {
        LocalError::NotFound => Check::problem(
            Verdict::Fail,
            "model",
            format!("{reference} is not in the model store, or its files are incomplete"),
            format!("fetch it with `brain pull {reference}`, or finish an interrupted pull by running it again"),
        ),
        LocalError::MissingRole { role, path } => Check::problem(
            Verdict::Fail,
            "model",
            format!("{reference} lacks its {role} component: {} is missing", path.display()),
            format!("run `brain pull {reference}` again to fetch the rest"),
        ),
        LocalError::UnsafeRolePath { role, rel } => Check::problem(
            Verdict::Fail,
            "model",
            format!("{reference}'s manifest names {rel} for {role}, outside the model's own directory"),
            "remove the model's directory and pull it again; do not serve a manifest you did not fetch",
        ),
    }
}

/// The checks of a model the store found: its weights are on disk, and a
/// text model has the tokenizer beside them.
#[must_use]
pub fn model_checks(model: &LocalModel) -> Vec<Check> {
    let reference = model.reference.to_string();
    let bytes = weights_bytes(model);
    let mut checks = Vec::new();
    if bytes == 0 {
        checks.push(Check::problem(
            Verdict::Fail,
            "model",
            format!("{reference}: its weights at {} are empty or unreadable", model.weights.display()),
            format!("remove them and run `brain pull {reference}` again"),
        ));
        return checks;
    }
    checks.push(Check::ok("model", format!("{reference}: {} of weights on disk", gib(bytes))));
    if model.format != Format::Compound {
        checks.push(match &model.tokenizer {
            Some(t) if t.is_file() => Check::ok("tokenizer", format!("{}", t.display())),
            _ => Check::problem(
                Verdict::Warn,
                "tokenizer",
                format!("no tokenizer.json beside {}", model.weights.display()),
                format!("run `brain pull {reference}` again, or place the model's tokenizer.json in {}", model.dir.display()),
            ),
        });
    }
    checks
}

/// Everything `brain doctor` checks: the store, the disk, the GPUs, and when
/// `model` names one, that model. `free_disk` and `cards` are the readings the
/// caller took; nothing here reads the driver or the filesystem's free space.
#[must_use]
pub fn diagnose(root: Option<&Path>, model: Option<&str>, free_disk: Option<u64>, cards: &[GpuMem]) -> Report {
    let mut checks = vec![store_check(root), disk_check(free_disk)];
    let mut need = None;
    if let (Some(reference), Some(root)) = (model, root) {
        match brain_modelstore::refurl::parse_model_arg(reference) {
            Err(e) => checks.push(Check::problem(
                Verdict::Fail,
                "model",
                format!("{reference:?} is not a model reference: {e}"),
                "name it as <vendor>/<repo>, for example Qwen/Qwen3-0.6B",
            )),
            Ok(parsed) => match Store::new(root).local_detailed(&parsed) {
                Err(why) => checks.push(missing_model_check(&parsed.to_string(), &why)),
                Ok(local) => {
                    need = Some(weights_bytes(&local)).filter(|b| *b > 0);
                    checks.extend(model_checks(&local));
                }
            },
        }
    }
    checks.push(gpu_check(cards, need));
    Report { checks }
}

/// The model-store directory `brain doctor` reports on, for a caller that
/// wants to name it: the `--models-dir` flag, else the default root.
#[must_use]
pub fn store_root(flag: Option<&str>) -> Option<PathBuf> {
    crate::model_dir::resolve(flag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelref::ModelRef;

    fn card(index: u32, free_gib: u64, measured: bool) -> GpuMem {
        GpuMem { index, total: 24 * GIB, free: free_gib * GIB, measured }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-doctor-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn local(dir: &Path, weights_len: usize, tokenizer: bool) -> LocalModel {
        let weights = dir.join("model.brain.safetensors");
        std::fs::write(&weights, vec![0u8; weights_len]).unwrap();
        let tokenizer_path = dir.join("tokenizer.json");
        if tokenizer {
            std::fs::write(&tokenizer_path, b"{}").unwrap();
        }
        LocalModel {
            reference: ModelRef::new("Qwen", "Qwen3-0.6B", None),
            dir: dir.to_path_buf(),
            weights,
            tokenizer: tokenizer.then_some(tokenizer_path),
            card: None,
            format: Format::Safetensors,
            adapter: None,
            roles: None,
        }
    }

    #[test]
    fn a_store_that_is_not_configured_or_not_there_fails_with_the_fix() {
        let none = store_check(None);
        assert_eq!(none.verdict, Verdict::Fail);
        assert!(none.fix.unwrap().contains("--models-dir"));
        let absent = store_check(Some(Path::new("/nonexistent/brain-doctor-store")));
        assert_eq!(absent.verdict, Verdict::Fail);
        assert!(absent.detail.contains("cannot be read"));
    }

    #[test]
    fn an_empty_store_warns_and_says_how_to_fill_it() {
        let root = scratch("empty");
        let check = store_check(Some(&root));
        assert_eq!(check.verdict, Verdict::Warn);
        assert!(check.fix.unwrap().contains("brain pull"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn disk_space_is_judged_against_the_two_thresholds_and_unknown_is_a_warning() {
        assert_eq!(disk_check(Some(WARN_BELOW_FREE_BYTES)).verdict, Verdict::Ok);
        assert_eq!(disk_check(Some(WARN_BELOW_FREE_BYTES - 1)).verdict, Verdict::Warn);
        assert_eq!(disk_check(Some(FAIL_BELOW_FREE_BYTES - 1)).verdict, Verdict::Fail);
        assert_eq!(disk_check(None).verdict, Verdict::Warn);
    }

    #[test]
    fn the_gpu_check_says_whether_the_weights_fit_and_never_presents_a_guess() {
        assert_eq!(gpu_check(&[], None).verdict, Verdict::Warn);
        assert_eq!(gpu_check(&[card(0, 24, false)], Some(GIB)).verdict, Verdict::Warn);
        assert_eq!(gpu_check(&[card(0, 2, true), card(1, 20, true)], None).verdict, Verdict::Ok);
        let fits = gpu_check(&[card(0, 2, true), card(1, 20, true)], Some(4 * GIB));
        assert_eq!(fits.verdict, Verdict::Ok);
        assert!(fits.detail.contains("gpu1"), "{}", fits.detail);
        let full = gpu_check(&[card(0, 2, true), card(1, 3, true)], Some(4 * GIB));
        assert_eq!(full.verdict, Verdict::Warn);
        assert!(full.detail.contains("CPU backend"), "{}", full.detail);
    }

    #[test]
    fn a_model_with_weights_and_a_tokenizer_is_ready_and_one_without_a_tokenizer_warns() {
        let dir = scratch("model");
        let ready = model_checks(&local(&dir, 1024, true));
        assert!(ready.iter().all(|c| c.verdict == Verdict::Ok), "{ready:?}");
        let bare = model_checks(&local(&dir, 1024, false));
        let tokenizer = bare.iter().find(|c| c.name == "tokenizer").unwrap();
        assert_eq!(tokenizer.verdict, Verdict::Warn);
        assert!(tokenizer.fix.as_ref().unwrap().contains("brain pull"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_weights_fail_the_model_check() {
        let dir = scratch("empty-weights");
        let checks = model_checks(&local(&dir, 0, true));
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].verdict, Verdict::Fail);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_model_that_is_not_pulled_or_is_malformed_fails_with_the_command_to_run() {
        let root = scratch("diagnose");
        let report = diagnose(Some(&root), Some("Qwen/Qwen3-0.6B"), Some(50 * GIB), &[card(0, 20, true)]);
        assert!(!report.ready());
        let model = report.checks.iter().find(|c| c.name == "model").unwrap();
        assert!(model.fix.as_ref().unwrap().contains("brain pull Qwen/Qwen3-0.6B"), "{model:?}");
        let bad = diagnose(Some(&root), Some("not a reference"), Some(50 * GIB), &[]);
        assert!(bad.checks.iter().any(|c| c.name == "model" && c.verdict == Verdict::Fail));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn warnings_do_not_make_the_report_unready_and_the_json_carries_every_check() {
        let report = Report { checks: vec![disk_check(Some(5 * GIB)), gpu_check(&[], None)] };
        assert!(report.ready());
        let json = report.to_json();
        assert_eq!(json["ready"], true);
        assert_eq!(json["checks"].as_array().unwrap().len(), 2);
        assert_eq!(json["checks"][0]["verdict"], "warn");
    }
}
