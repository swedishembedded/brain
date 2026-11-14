// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The actions of a forecasting model that need no resident instance, and the
//! one-shot providers that put them beside the served `forecast` and `predict`:
//! `import` (Chronos-2, FinCast, TimesFM-3) converts a released checkpoint into
//! a brain container, `finetune` (Kronos, TimesFM-3) is the gated fine-tune of
//! the checkpoint the store resolved.
//!
//! A fine-tune streams one [`Progress`] per optimizer step, stops at the next
//! step when the invocation's cancel token fires, and echoes the seed the LoRA
//! adapters were initialised from. It writes a checkpoint only when the fine-tune
//! beat the base on a held-out, embargoed split.
//!
//! Swedish Embedded AB implements forecasting model serving and gated
//! fine-tuning on its clients' own machines and data for its clients. If your
//! team needs expertise in time-series foundation models then you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Assembly, Invocation, Manifest, Outcome, ParamSpec, ParamType, Progress, Provider};
use forecast::finetune::{StepReport, Universe};
use forecastops::finetune::{finetune_spec, gate_line, generalization_line, load_universes, require_verdict, FinetuneRequest};
use residency::ResidentModel;
use serde_json::json;

use crate::resident_forecast::{self as rf, Chronos2Resident, FincastResident, KronosResident, Timesfm3Resident};
use crate::resident_provider::ResidentProvider;

/// Where a Kronos fine-tune writes the promoted decoder when the caller names none.
const KRONOS_SAVE: &str = "kronos-decoder-ft.safetensors";
/// Where a TimesFM-3 fine-tune writes the promoted checkpoint when the caller names none.
const TIMESFM3_SAVE: &str = "timesfm3-ft.safetensors";

fn activations_param() -> ParamSpec {
    ParamSpec::new(
        "activations",
        ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()),
        "what the backward keeps of the forward (default: the model's own)",
    )
}

/// The `import` action: a released checkpoint into a brain container, with no
/// checkpoint resolved.
pub fn import_spec(summary: &str) -> ActionSpec {
    ActionSpec::new("import", summary)
        .cancellable()
        .from_scratch()
        .param(ParamSpec::new("source", ParamType::Str, "the released checkpoint to convert: a Hugging Face directory or a .safetensors file").required().host_resolved())
        .param(ParamSpec::new("dest", ParamType::Str, "where to write the brain checkpoint (.safetensors)").required().host_resolved())
}

/// `import` for one model: `convert(source, dest)` is the model's own importer.
struct ImportAction {
    model: &'static str,
    summary: &'static str,
    convert: fn(&str, &str) -> Result<(), String>,
}

impl Action for ImportAction {
    fn spec(&self) -> ActionSpec {
        import_spec(self.summary)
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let source = inv.get_str("source").ok_or_else(|| format!("{} import: missing required param 'source'", self.model))?;
        let dest = inv.get_str("dest").ok_or_else(|| format!("{} import: missing required param 'dest'", self.model))?;
        (self.convert)(&source, &dest).map_err(|e| format!("{} import: {e}", self.model))?;
        Ok(Outcome::new().set("dest", json!(dest)))
    }
}

fn import_action(model: &'static str, summary: &'static str, convert: fn(&str, &str) -> Result<(), String>) -> Arc<dyn Action> {
    Arc::new(ImportAction { model, summary, convert })
}

/// One resident provider over `build`, with the model's one-shot `extra` actions.
fn provider_of(manifest: Manifest, resident: impl Fn() -> Arc<dyn ResidentModel> + Send + Sync + 'static, extra: Vec<Arc<dyn Action>>) -> Arc<dyn Provider> {
    let build = move |_: &Invocation| Ok(resident());
    Arc::new(ResidentProvider::new(manifest, Box::new(build), Box::new(|_| String::new()), extra))
}

/// Chronos-2 over `assembly`'s checkpoint, run in this process through its own
/// serving adapter.
pub fn chronos2_provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    let path = assembly.role_path("weights")?;
    let import = import_action("chronos2", "convert an amazon/chronos-2 directory into a brain checkpoint (one container with the config in its header)", chronos2::import::import);
    Ok(provider_of(rf::chronos2_direct_manifest(), move || Arc::new(Chronos2Resident::new(&path)), vec![import]))
}

/// FinCast over `assembly`'s checkpoint.
pub fn fincast_provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    let path = assembly.role_path("weights")?;
    let import = import_action("fincast", "convert the released FinCast safetensors into a brain checkpoint (one container with the config in its header)", fincast::import::import);
    Ok(provider_of(rf::fincast_direct_manifest(), move || Arc::new(FincastResident::new(&path)), vec![import]))
}

/// Kronos over `assembly`'s tokenizer and decoder.
pub fn kronos_provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    let (tokenizer, decoder) = (assembly.role_path("tokenizer")?, assembly.role_path("decoder")?);
    let finetune = Arc::new(KronosFinetune { tokenizer: tokenizer.clone(), decoder: decoder.clone() });
    Ok(provider_of(rf::kronos_direct_manifest(), move || Arc::new(KronosResident::new(&tokenizer, &decoder)), vec![finetune]))
}

/// TimesFM-3 over `assembly`'s checkpoint.
pub fn timesfm3_provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    let path = assembly.role_path("weights")?;
    let finetune = Arc::new(Timesfm3Finetune { weights: path.clone() });
    let import = import_action("timesfm3", "convert a google/timesfm-3.0-pytorch directory into a brain checkpoint (one container with the config in its header)", timesfm3::import::import);
    Ok(provider_of(rf::timesfm3_direct_manifest(), move || Arc::new(Timesfm3Resident::new(&path)), vec![finetune, import]))
}

/// The `finetune` schema of Kronos.
pub fn kronos_finetune_spec() -> ActionSpec {
    finetune_spec("fine-tune the Kronos decoder on a universe of OHLCV CSVs; promoted (and written) only if it beats the base on a held-out, embargoed split", KRONOS_SAVE).param(activations_param())
}

/// The `finetune` schema of TimesFM-3.
pub fn timesfm3_finetune_spec() -> ActionSpec {
    finetune_spec(
        "fine-tune TimesFM-3 on a universe of OHLCV CSVs (the close is forecast and scored); promoted (and written) only if it beats the base on a held-out, embargoed split. The result is a derivative of non-commercial weights and may not be redistributed",
        TIMESFM3_SAVE,
    )
    .param(activations_param())
}

/// One progress line per optimizer step.
fn step_progress(r: StepReport) -> Progress {
    Progress::step(r.step, r.total, format!("epoch {}/{} step {}/{} loss {:.4}", r.epoch, r.epochs, r.step, r.total, r.loss))
}

/// Everything a finetune action reports once the gate has spoken.
struct Verdict<'a> {
    req: &'a FinetuneRequest,
    report: forecast::FinetuneReport,
    /// Where the promoted checkpoint was written, `None` when the base was kept.
    saved: Option<&'a str>,
    /// `(names, base loss, tuned loss)` on the instruments never trained on.
    held_out: Option<(usize, f32, f32)>,
    license: Option<&'a str>,
    /// What the run said before it trained (the licence notice, what the data
    /// loader rejected, the configuration), repeated first in the verdict: a
    /// terminal shows only the latest progress line, and these must outlive it.
    preamble: Vec<String>,
}

impl Verdict<'_> {
    fn outcome(&self) -> Outcome {
        let rep = &self.report;
        let mut text = self.preamble.clone();
        text.push(gate_line(rep));
        match self.saved {
            Some(path) => text.push(format!("promoted checkpoint -> {path}")),
            None => text.push("not promoted -> no checkpoint written (base kept)".to_string()),
        }
        if let Some(seed) = rep.lora_seed {
            text.push(format!("LoRA adapters initialised from seed {seed} (pass seed {seed} to reproduce)"));
        }
        if let (Some(license), Some(_)) = (self.license, self.saved) {
            text.push(format!("license: {license} - NOT redistributable"));
        }
        let held_out = self.held_out.map(|(names, base, tuned)| {
            text.push(generalization_line(names, base, tuned));
            json!({"names": names, "base": base, "tuned": tuned, "generalizes": tuned < base})
        });
        Outcome::new()
            .set("seed", json!(self.req.seed))
            .set("promoted", json!(rep.promoted))
            .set("base_val", json!(rep.base_val))
            .set("ft_val", json!(rep.ft_val))
            .set("steps", json!(rep.steps))
            .set("saved", json!(self.saved))
            .set("held_out", json!(held_out))
            .set("verdict", json!(text.join("\n")))
    }
}

/// Said once per fine-tune before any work, in the form the model's reference
/// gives it.
fn announce(progress: &mut dyn FnMut(Progress), lines: impl IntoIterator<Item = String>) {
    for l in lines {
        progress(Progress::step(0, 0, l));
    }
}

struct KronosFinetune {
    tokenizer: String,
    decoder: String,
}

impl Action for KronosFinetune {
    fn spec(&self) -> ActionSpec {
        kronos_finetune_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let req = FinetuneRequest::parse(inv, KRONOS_SAVE).map_err(|e| format!("kronos {e}"))?;
        let saved = req.activations.as_deref().map(model::saved::SavedPolicy::parse_activations).transpose()?;
        let universes = load_universes(&req)?;
        let mut preamble = universes.notes.clone();
        announce(progress, preamble.clone());

        let (cfg, base) = kronos::import::load_decoder(&self.decoder).map_err(|e| format!("kronos finetune: load decoder: {e}"))?;
        let model = kronos::import::load_model(&self.tokenizer, &self.decoder).map_err(|e| format!("kronos finetune: load model: {e}"))?;
        let config = format!("finetune: {} names, context {}, horizon {}, epochs {}, lr {}{}", universes.series.len(), req.context, req.horizon, req.epochs, req.lr, if req.lora_rank > 0 { format!(", LoRA r{}", req.lora_rank) } else { ", full".into() });
        announce(progress, [config.clone()]);
        preamble.push(config);

        let lora = (req.lora_rank > 0).then(|| kronos::train::LoraCfg::attn(req.lora_rank, (req.lora_rank * 2) as f32).with_seed(req.seed));
        let opts = kronos::train::FinetuneOpts { epochs: req.epochs, lr: req.lr, wd: 0.1, clip: 3.0, lora, batch: req.batch, saved };
        let universe = Universe { series: &universes.series, context: req.context, horizon: req.horizon, split: req.split() };
        let (report, weights) = kronos::finetune::finetune_universe(&model, &base, &universe, &opts, &mut |r| {
            progress(step_progress(r));
            !inv.cancel.is_cancelled()
        })
        .map_err(|e| if e == forecast::finetune::CANCELLED { e } else { format!("kronos finetune: {e}") })?;
        require_verdict(&req, universes.series.len(), &report)?;
        let weights = weights.ok_or("kronos finetune: the trainer returned no weights")?;

        // Written BEFORE the held-out names are scored, so a slow
        // generalization pass cannot cost the checkpoint.
        let written = report.promoted.then(|| {
            kronos::finetune::save_decoder_weights(&cfg, &weights, &req.save);
            req.save.as_str()
        });
        let held_out = universes.holdout.as_ref().map(|hs| {
            let base_h = kronos::finetune::eval_universe_loss(&model, &base, hs, req.context, req.horizon);
            let tuned_h = kronos::finetune::eval_universe_loss(&model, &weights, hs, req.context, req.horizon);
            (hs.len(), base_h, tuned_h)
        });
        Ok(Verdict { req: &req, report, saved: written, held_out, license: None, preamble }.outcome())
    }
}

struct Timesfm3Finetune {
    weights: String,
}

impl Action for Timesfm3Finetune {
    fn spec(&self) -> ActionSpec {
        timesfm3_finetune_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        // Said before anything else, and before any work: an operator about to
        // spend hours producing an artifact they may not redistribute should
        // learn that first, not from the model card afterwards.
        let mut preamble = vec![timesfm3::finetune::LICENSE_NOTICE.to_string()];
        announce(progress, preamble.clone());
        let req = FinetuneRequest::parse(inv, TIMESFM3_SAVE).map_err(|e| format!("timesfm3 {e}"))?;
        let saved = req.activations.as_deref().map(model::saved::SavedPolicy::parse_activations).transpose()?;
        let universes = load_universes(&req)?;
        announce(progress, universes.notes.clone());
        preamble.extend(universes.notes.clone());

        let (cfg, base) = timesfm3::finetune::load_base(&self.weights).map_err(|e| format!("timesfm3 finetune: load {}: {e}", self.weights))?;
        // Checked against the LOADED config, not a constant: the limit is
        // `stitch_extract_len()`, which a differently configured checkpoint moves.
        timesfm3::finetune::supported_horizon(&cfg, req.horizon).map_err(|e| format!("timesfm3 finetune: {e}"))?;
        let config = format!("finetune: timesfm3, {} names, context {}, horizon {}, epochs {}, lr {}{}", universes.series.len(), req.context, req.horizon, req.epochs, req.lr, if req.lora_rank > 0 { format!(", LoRA r{}", req.lora_rank) } else { ", full".into() });
        announce(progress, [config.clone()]);
        preamble.push(config);

        // Only a LoRA run draws (its adapters' initial factors); a full
        // fine-tune has no seed to use, and its report says so.
        let lora = (req.lora_rank > 0).then(|| timesfm3::train::LoraCfg::attn(req.lora_rank, (req.lora_rank * 2) as f32).with_seed(req.seed));
        let opts = timesfm3::finetune::FinetuneOpts { epochs: req.epochs, lr: req.lr, wd: 0.1, clip: 3.0, lora, batch: req.batch, saved };
        let universe = Universe { series: &universes.series, context: req.context, horizon: req.horizon, split: req.split() };
        let (report, weights) = timesfm3::finetune::finetune_universe(&cfg, &base, &universe, &opts, &mut |r| {
            progress(step_progress(r));
            !inv.cancel.is_cancelled()
        })
        .map_err(|e| if e == forecast::finetune::CANCELLED { e } else { format!("timesfm3 finetune: {e}") })?;
        require_verdict(&req, universes.series.len(), &report)?;
        let weights = weights.ok_or("timesfm3 finetune: the trainer returned no weights")?;

        let written = report.promoted.then(|| {
            timesfm3::finetune::save_weights(&cfg, &weights, &req.save);
            req.save.as_str()
        });
        let held_out = universes.holdout.as_ref().map(|hs| {
            let base_h = timesfm3::finetune::eval_universe_loss(&cfg, &base, hs, req.context, req.horizon);
            let tuned_h = timesfm3::finetune::eval_universe_loss(&cfg, &weights, hs, req.context, req.horizon);
            (hs.len(), base_h, tuned_h)
        });
        let license = format!("{} (derived from {})", timesfm3::finetune::WEIGHTS_LICENSE, timesfm3::finetune::UPSTREAM);
        Ok(Verdict { req: &req, report, saved: written, held_out, license: Some(&license), preamble }.outcome())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Blob, CancelToken, Media};

    /// A tiny TimesFM-3 brain container in a fresh models root, where the model
    /// store finds it with no environment variable.
    fn tiny_timesfm3_store(tag: &str) -> (std::path::PathBuf, Assembly) {
        let root = std::env::temp_dir().join(format!("brain-forecast-actions-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("local");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = timesfm3::Timesfm3Config::tiny();
        let w = timesfm3::train::init_weights(&cfg, 7);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(n, shape)| (n.clone(), shape.iter().map(|&x| x as u64).collect(), w[&n].clone())).collect();
        let path = dir.join("timesfm3.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
        let assembly = Assembly { id: "local/timesfm3".into(), arch: "timesfm3".into(), variant: None, roles: [("weights".to_string(), path)].into(), provenance: Vec::new() };
        (root, assembly)
    }

    /// `n` valid daily bars with a trend and a seasonal component.
    fn csv(n: usize, phase: f32) -> String {
        let mut s = String::from("Date,open,high,low,close,volume\n");
        for i in 0..n {
            let (month, day) = (1 + i / 28, 1 + i % 28);
            let t = i as f32;
            let c = 100.0 + 0.05 * t + 3.0 * (t * 0.3 + phase).sin();
            s.push_str(&format!("2024-{month:02}-{day:02},{c:.2},{:.2},{:.2},{c:.2},1000\n", c + 2.0, c - 2.0));
        }
        s
    }

    fn universe(root: &std::path::Path, names: &[&str], rows: usize) -> String {
        let dir = root.join("universe");
        std::fs::create_dir_all(&dir).unwrap();
        for (i, n) in names.iter().enumerate() {
            std::fs::write(dir.join(format!("{n}.csv")), csv(rows, i as f32)).unwrap();
        }
        dir.to_string_lossy().into_owned()
    }

    /// The model store serves the checkpoint it holds (and nothing from an empty
    /// one), and the one-shot provider IS that resident: its `forecast` and
    /// `predict` run through the same instance, and the direct manifest adds
    /// only the actions that need no instance.
    #[test]
    fn a_store_checkpoint_is_served_and_the_provider_runs_the_residents_actions() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let (root, assembly) = tiny_timesfm3_store("serve");
        let store = loader::served::ServedStore::new(Some(root.clone()));
        assert!(Timesfm3Resident::from_store(&store).is_some(), "the store's checkpoint is served");
        assert!(Timesfm3Resident::from_store(&loader::served::ServedStore::new(Some(root.join("nowhere")))).is_none(), "nothing to serve from a store with no checkpoint");

        let provider = timesfm3_provider(&assembly).unwrap();
        let names: Vec<String> = provider.manifest().actions.into_iter().map(|a| a.name).collect();
        assert_eq!(names, ["forecast", "predict", "finetune", "import"]);

        let series: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let forecast = provider.action("forecast").unwrap().run(&Invocation::new().set("horizon", json!(4)).blob("context", Blob::new(capability::Media::Bytes, series)), &mut |_| {}).unwrap();
        assert_eq!(forecast.blobs["forecast"].meta["shape"], json!([4, 5]), "tiny() has 5 native quantile levels");

        let inv = Invocation::new().blob("csv", Blob::new(Media::Text, csv(120, 0.0).into_bytes())).set("horizon", json!(4)).set("context", json!(16)).set("seed", json!(3));
        let predict = provider.action("predict").unwrap().run(&inv, &mut |_| {}).unwrap();
        assert_eq!(predict.outputs["seed"], json!(3));
        let report = String::from_utf8(predict.blobs["report"].bytes.clone()).unwrap();
        assert!(report.contains("persistence (last close)"), "{report}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A TimesFM-3 fine-tune reports one progress line per optimizer step,
    /// reaches a gate verdict, echoes the seed, and a refused horizon is named.
    #[test]
    fn a_timesfm3_finetune_streams_its_steps_and_reaches_a_gate_verdict() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let (root, assembly) = tiny_timesfm3_store("finetune");
        let data = universe(&root, &["AAA", "BBB", "CCC"], 200);
        let save = root.join("ft.safetensors");
        let provider = timesfm3_provider(&assembly).unwrap();
        let action = provider.action("finetune").unwrap();
        let inv = |horizon: i64| {
            Invocation::new()
                .set("data", json!(data))
                .set("save", json!(save.to_string_lossy()))
                .set("context", json!(8))
                .set("horizon", json!(horizon))
                .set("epochs", json!(1))
                .set("batch", json!(2))
                .set("lr", json!(1e-3))
                .set("seed", json!(11))
        };
        let mut lines = Vec::new();
        let out = action.run(&inv(4), &mut |p| lines.push(p.message)).unwrap();
        assert!(lines[0].contains("timesfm-non-commercial-license-v1.0"), "the license notice comes first: {lines:?}");
        assert!(lines.iter().filter(|l| l.contains("epoch 1/1 step")).count() as u64 == out.outputs["steps"].as_u64().unwrap(), "one progress line per optimizer step: {lines:?}");
        assert_eq!(out.outputs["seed"], json!(11));
        let verdict = out.outputs["verdict"].as_str().unwrap();
        assert!(verdict.contains("PROMOTE") || verdict.contains("KEEP BASE"), "{verdict}");
        assert_eq!(save.exists(), out.outputs["promoted"] == json!(true), "a checkpoint is written exactly when the gate promoted");

        let e = action.run(&inv(9), &mut |_| {}).unwrap_err();
        assert!(e.contains("stitch_extract_len"), "the refusal names the limit: {e}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Cancelling stops the fine-tune at the next step with the literal
    /// `cancelled`, and nothing is written.
    #[test]
    fn a_cancelled_finetune_stops_and_writes_nothing() {
        if brain_testutil::gpu_tests_skipped() {
            return;
        }
        let (root, assembly) = tiny_timesfm3_store("cancel");
        let data = universe(&root, &["AAA", "BBB", "CCC"], 200);
        let save = root.join("never.safetensors");
        let token = CancelToken::armed();
        let inv = Invocation { cancel: token.clone(), ..Invocation::new() }
            .set("data", json!(data))
            .set("save", json!(save.to_string_lossy()))
            .set("context", json!(8))
            .set("horizon", json!(4))
            .set("epochs", json!(1))
            .set("batch", json!(2));
        let mut steps = 0;
        let e = timesfm3_provider(&assembly).unwrap().action("finetune").unwrap().run(&inv, &mut |p| {
            if p.message.contains("step 1/") {
                steps += 1;
                token.cancel();
            }
        });
        assert_eq!(e.unwrap_err(), "cancelled");
        assert_eq!(steps, 1, "it stopped at the step after the one that cancelled");
        assert!(!save.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `import` converts a released checkpoint into a brain container the store
    /// then finds: a tiny FinCast safetensors in, a container with its config out.
    #[test]
    fn import_writes_a_brain_container_the_store_classifies() {
        let root = std::env::temp_dir().join(format!("brain-forecast-actions-import-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        // The released FinCast file holds exactly the model's tensors and no config; at
        // the published size that is a gigabyte, so the converter is exercised through
        // chronos2's, whose tiny() directory is a few kilobytes.
        let cfg = chronos2::Chronos2Config::tiny();
        let src = root.join("amazon").join("chronos-2");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("config.json"), serde_json::to_vec(&cfg.to_json()).unwrap()).unwrap();
        let list = cfg.param_list();
        let plan: Vec<(String, Vec<u64>)> = list.iter().map(|(n, s)| (n.clone(), s.iter().map(|&d| d as u64).collect())).collect();
        let mut w = checkpoint::weightio::StWriter::create(src.join("model.safetensors").to_str().unwrap(), &plan, &serde_json::Value::Null, None).unwrap();
        for (name, shape) in &list {
            w.write(name, &vec![0.01; shape.iter().product()]).unwrap();
        }
        w.finish().unwrap();

        let dest = root.join("out").join("chronos2.safetensors");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let assembly = Assembly { id: "local/chronos2".into(), arch: "chronos2".into(), variant: None, roles: [("weights".to_string(), src.clone())].into(), provenance: Vec::new() };
        let inv = Invocation::new().set("source", json!(src.to_string_lossy())).set("dest", json!(dest.to_string_lossy()));
        let out = chronos2_provider(&assembly).unwrap().action("import").unwrap().run(&inv, &mut |_| {}).unwrap();
        assert_eq!(out.outputs["dest"], json!(dest.to_string_lossy()));

        let store = loader::served::ServedStore::new(Some(root.join("out")));
        let resident = Chronos2Resident::from_store(&store);
        assert!(resident.is_some(), "the written container is what the store serves for chronos2");
        let _ = std::fs::remove_dir_all(&root);
    }
}
