// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained speech
// synthesizers into smaller models that still load in their upstream tooling,
// for its clients. If your team needs expertise in compressing text-to-speech
// models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! [`SpeechSurgery`]: a Qwen3-TTS checkpoint made physically smaller by a
//! surgery plan, chosen and measured by calibration on recorded speech, and
//! written as a checkpoint in its base's upstream layout.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let plan = r#"[{"select_channels": {"dims": "talker.layers.*.mlp", "keep": 4608, "score": "wanda"}},
//!                {"remove_layers": {"stack": "talker.layers", "count": 4, "by": "kl"}}]"#;
//! let outcome = brain::SpeechSurgery::new("sten-tts", "validation.bttshrd")
//!     .plan_json(plan)?
//!     .out_dir("runs/surgery-1")
//!     .run()?;
//! println!("{}: {} -> {} parameters", outcome.checkpoint.unwrap().display(), outcome.params_before, outcome.params_after);
//! # Ok(()) }
//! ```
//!
//! **What runs.** The base and the calibration shard are read; one
//! teacher-forced pass of the unedited checkpoint over the calibration
//! records collects what the plan ranks by
//! ([`qwen3tts::speech_sft::calibrate::calibrate`]: per-layer ablation and
//! stream cosines, MLP and text-projection activations, the streams' second
//! moments), and the Fisher and group-Taylor channel estimates when the plan
//! scores by them. A plan that trims the text vocabulary keeps every token
//! the calibration shard holds and its utterance texts and the corpus encode
//! to ([`modelopt::surgery::bpe`]); the held-out shard's texts and the probe
//! measure how many texts outside it would tokenise differently. The ids the
//! held-out shard stores are kept too, so a recovery can read it; only
//! held-out records of kept tokens are measured on. The plan is applied
//! ([`modelopt::surgery::plan::apply`]; a `greedy` layer removal is first
//! chosen on the calibration records of the checkpoint as edited so far,
//! each round measured with the layers already chosen removed,
//! [`modelopt::surgery::greedy_removal`]), each edit's loss and KL divergence
//! from the unedited predictions measured on the held-out shard (the
//! calibration shard when none is given), and the result exported over the
//! base ([`qwen3tts::export::export_edited`]) with `surgery.json` (the
//! manifest) and `calibration.safetensors` beside it. [`SpeechSurgery::profile`]
//! instead measures every edit of the plan alone on the unedited checkpoint
//! and writes no checkpoint: a sensitivity profile.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use model::surgery::{Calibration, ChannelCurvature, SurgeryGraph};
use modelopt::surgery::bpe::{self, BpeReport, TokenizerFiles};
use modelopt::surgery::plan::{self, Edit, Inputs};
use modelopt::surgery::scores::Score;
use modelopt::surgery::{Checkpoint, EditRecord, Manifest, Measured};
use qwen3tts::speech_sft::calibrate::{self as cal, CalibrationOpts, Evaluator, Measurement};
use qwen3tts::speech_sft::{Base, GemmPrecision};
use serde::Serialize;

use crate::{Error, Result};

/// The checkpoint directory, the manifest and the calibration, inside the out directory.
const CHECKPOINT_DIR: &str = "checkpoint";
const MANIFEST_FILE: &str = "surgery.json";
const CALIBRATION_FILE: &str = "calibration.safetensors";

/// A surgery of a Qwen3-TTS checkpoint. See this module's doc.
#[derive(Clone, Debug)]
pub struct SpeechSurgery {
    base: PathBuf,
    calibration: PathBuf,
    held_out: Option<PathBuf>,
    plan: Vec<Edit>,
    corpus: Vec<String>,
    probe: Vec<String>,
    out_dir: Option<PathBuf>,
    id: String,
    records: usize,
    held_out_records: usize,
    batch: usize,
    sample_rows: usize,
    mtp_weight: f32,
    profile: bool,
    exact: bool,
}

/// One edit's measurement: what it was and what it cost.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EditCost {
    pub edit: Edit,
    pub params_before: u64,
    pub params_after: u64,
    pub loss: f64,
    pub kl: f64,
    pub talker_kl: f64,
    pub code_predictor_kl: f64,
    pub talker_ce: f64,
    pub code_predictor_ce: f64,
}

/// What a surgery did and measured.
#[derive(Clone, Debug, Serialize)]
pub struct SpeechSurgeryOutcome {
    /// The unedited checkpoint on the held-out records.
    pub baseline: MeasuredLoss,
    /// Per edit, in the order applied (alone on the base for a profile).
    pub edits: Vec<EditCost>,
    pub params_before: u64,
    pub params_after: u64,
    /// The vocabulary trim's report, when the plan trims one.
    pub vocabulary: Option<BpeReport>,
    /// Of the calibration shard's utterances that carry their text, how many and how many of them hold other token
    /// ids than the tokenizer files encode that text to (a shard tokenised by another tokenizer), when the plan
    /// trims a vocabulary.
    pub shard_texts: Option<(usize, usize)>,
    /// The exported checkpoint (none for a profile).
    pub checkpoint: Option<PathBuf>,
    pub manifest: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MeasuredLoss {
    pub loss: f64,
    pub talker_ce: f64,
    pub code_predictor_ce: f64,
}

fn backend(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("speech surgery: {e}"))
}

impl SpeechSurgery {
    /// A surgery of the Qwen3-TTS release directory `base`, calibrated on the
    /// training shard `calibration` (`brain-tts-shard-v1`).
    pub fn new(base: impl Into<PathBuf>, calibration: impl Into<PathBuf>) -> SpeechSurgery {
        SpeechSurgery {
            base: base.into(),
            calibration: calibration.into(),
            held_out: None,
            plan: Vec::new(),
            corpus: Vec::new(),
            probe: Vec::new(),
            out_dir: None,
            id: "local/qwen3tts-surgery".into(),
            records: 64,
            held_out_records: 64,
            batch: 8,
            sample_rows: 512,
            mtp_weight: 0.3,
            profile: false,
            exact: false,
        }
    }

    /// The edits, as [`Edit`]s.
    pub fn plan(mut self, edits: Vec<Edit>) -> Self {
        self.plan = edits;
        self
    }

    /// The edits as a JSON array ([`modelopt::surgery::plan`]); unknown fields are refused.
    pub fn plan_json(self, json: &str) -> Result<Self> {
        let edits: Vec<Edit> = serde_json::from_str(json).map_err(|e| Error::Backend(format!("speech surgery: plan: {e}")))?;
        Ok(self.plan(edits))
    }

    /// A shard the edits are measured on (default: the calibration shard).
    pub fn held_out(mut self, shard: impl Into<PathBuf>) -> Self {
        self.held_out = Some(shard.into());
        self
    }

    /// Texts whose tokens a vocabulary trim keeps beside the shards' own.
    pub fn corpus(mut self, texts: Vec<String>) -> Self {
        self.corpus = texts;
        self
    }

    /// Texts outside the corpus whose encodings the trim's report compares.
    pub fn probe(mut self, texts: Vec<String>) -> Self {
        self.probe = texts;
        self
    }

    /// Where the checkpoint, `surgery.json` and `calibration.safetensors` are written.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// The model id the exported checkpoint's manifest registers.
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Calibration records (default 64; 0: the whole shard).
    pub fn records(mut self, n: usize) -> Self {
        self.records = n;
        self
    }

    /// Held-out records the edits are measured on (default 64; 0: the whole shard).
    pub fn held_out_records(mut self, n: usize) -> Self {
        self.held_out_records = n;
        self
    }

    /// Records per calibration batch (default 8).
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = n;
        self
    }

    /// Rows per stream point and batch the statistics take (default 512).
    pub fn sample_rows(mut self, n: usize) -> Self {
        self.sample_rows = n;
        self
    }

    /// The code predictor's weight in the loss and the KL (default 0.3, the training objective's).
    pub fn mtp_weight(mut self, w: f32) -> Self {
        self.mtp_weight = w;
        self
    }

    /// Measure every edit alone on the unedited checkpoint, write no checkpoint.
    pub fn profile(mut self, on: bool) -> Self {
        self.profile = on;
        self
    }

    /// Run every product in exact fp32 instead of the device's tensor cores (the reference a parity check states
    /// its claim against).
    pub fn exact(mut self, on: bool) -> Self {
        self.exact = on;
        self
    }

    /// Calibrate, edit, measure, export.
    pub fn run(&self) -> Result<SpeechSurgeryOutcome> {
        self.run_with(|_| {})
    }

    /// [`Self::run`], reporting each stage to `on_progress`.
    pub fn run_with(&self, mut on_progress: impl FnMut(&str)) -> Result<SpeechSurgeryOutcome> {
        let out = self.out_dir.clone().ok_or_else(|| backend("name an out_dir"))?;
        if self.plan.is_empty() {
            return Err(backend("the plan names no edit"));
        }
        std::fs::create_dir_all(&out).map_err(|e| backend(format!("{}: {e}", out.display())))?;
        on_progress("reading the base");
        let ck = Checkpoint::read(&self.base).map_err(backend)?;
        let source = brain_modelstore::fetch::file_digest(&self.base.join("model.safetensors")).map_err(|e| backend(format!("{}: {e}", self.base.display())))?;
        let shard = qwen3tts::shard::Shard::open(&self.calibration).map_err(|e| backend(format!("{}: {e}", self.calibration.display())))?;
        let held_out = self.held_out.as_ref().map(|p| qwen3tts::shard::Shard::open(p).map_err(|e| backend(format!("{}: {e}", p.display())))).transpose()?;
        let opts = |records: usize| CalibrationOpts { records, batch: self.batch, mtp_weight: self.mtp_weight, layers: self.plan.iter().any(needs_ablation), sample_rows: self.sample_rows, gemm: if self.exact { GemmPrecision::Fp32Reference } else { GemmPrecision::Auto } };

        on_progress("calibrating");
        let (cal_ev, mut calibration) = cal::calibrate(base_of(&ck)?, &shard, &source, &opts(self.records), &mut |s| on_progress(s)).map_err(backend)?;
        let graph = qwen3tts_graph(&ck).map_err(backend)?;
        if self.plan.iter().any(needs_curvature) {
            on_progress("estimating the channels' curvature");
            calibration.curvature = curvature(&ck, &graph, &cal_ev)?;
        }
        modelopt::surgery::calibration::save(&out.join(CALIBRATION_FILE), &calibration).map_err(backend)?;
        // A vocabulary trim keeps what the calibration shard and the corpus need; the held-out records it would
        // re-tokenise are left out of the measurement and counted in its report.
        let (vocab_keep, trimmed) = self.vocabulary(&ck, &shard, held_out.as_ref())?;
        let measured_shard = held_out.as_ref().unwrap_or(&shard);
        let kept: Option<std::collections::HashSet<u32>> = vocab_keep.get("text").map(|k| k.iter().map(|&i| i as u32).collect());
        let candidates: Vec<usize> = (0..measured_shard.len()).filter(|&r| kept.as_ref().is_none_or(|k| measured_shard.record(r).text_ids.iter().all(|t| k.contains(t)))).collect();
        let measured_records: Vec<usize> = cal::spread(candidates.len(), self.held_out_records).into_iter().map(|i| candidates[i]).collect();
        let measured_ev = match &held_out {
            Some(h) => Some(Evaluator::on_records(base_of(&ck)?, h, &measured_records, &opts(self.held_out_records)).map_err(backend)?.0),
            None if kept.is_some() => Some(Evaluator::on_records(base_of(&ck)?, &shard, &measured_records, &opts(self.held_out_records)).map_err(backend)?.0),
            None => None,
        };
        let measure_on = measured_ev.as_ref().unwrap_or(&cal_ev);
        let baseline = measure_on.baseline();
        on_progress(&format!("held-out records, unedited: loss {:.4} over {} records ({} of {} held out records hold only kept tokens)", baseline.loss, measured_records.len(), candidates.len(), measured_shard.len()));

        let model = plan::Model { graph: &qwen3tts_graph, check: &|c: &Checkpoint| qwen3tts::surgery::check_upstream(&c.config, &|n| c.tensors.contains_key(n)) };
        let mut costs = Vec::new();
        let mut manifest = Manifest { source: source.clone(), calibration: Some(modelopt_digest(&out.join(CALIBRATION_FILE))?), edits: Vec::new() };
        let mut edited = ck.clone();
        let mut inputs = Inputs::default();
        // Every edit in the order a plan applies them; a profile applies each to the unedited checkpoint alone.
        let mut order: Vec<&Edit> = self.plan.iter().collect();
        order.sort_by_key(|e| e.rank());
        for edit in order {
            on_progress(&format!("applying {}", serde_json::to_string(edit).unwrap_or_default()));
            if self.profile {
                edited = ck.clone();
                inputs = Inputs::default();
            }
            let (resolved, greedy_method) = match edit {
                Edit::RemoveLayers { greedy: true, .. } => {
                    let (e, method) = resolve_greedy(edit, &edited, &inputs, &cal_ev, self.mtp_weight, &mut on_progress)?;
                    (e, Some(method))
                }
                _ => (edit.clone(), None),
            };
            let mut applied = plan::apply(&mut edited, std::slice::from_ref(&resolved), &calibration, &vocab_keep, &model).map_err(backend)?;
            if let Some(method) = greedy_method {
                applied.changes.iter_mut().for_each(|(_, m)| *m = method.clone());
            }
            inputs.vocabularies.extend(applied.inputs.vocabularies);
            for (name, b) in applied.inputs.streams {
                if inputs.streams.insert(name.clone(), b).is_some() {
                    return Err(backend(format!("the plan rotates stream {name} twice")));
                }
            }
            let measured = measure(measure_on, &edited, &inputs)?;
            on_progress(&format!("measured: loss {:.4} kl {:.4}", measured.loss, measured.kl(self.mtp_weight)));
            for (change, method) in applied.changes {
                costs.push(EditCost {
                    edit: edit.clone(),
                    params_before: change.params_before,
                    params_after: change.params_after,
                    loss: measured.loss,
                    kl: measured.kl(self.mtp_weight),
                    talker_kl: measured.talker_kl,
                    code_predictor_kl: measured.code_predictor_kl,
                    talker_ce: measured.talker_ce,
                    code_predictor_ce: measured.code_predictor_ce,
                });
                manifest.edits.push(EditRecord { change, method, measured: Some(Measured { loss: measured.loss, kl: measured.kl(self.mtp_weight) }) });
            }
        }
        let manifest_path = out.join(MANIFEST_FILE);
        let checkpoint = if self.profile {
            None
        } else {
            on_progress("exporting");
            let dir = out.join(CHECKPOINT_DIR);
            let list = edited.to_list();
            let files: Vec<(String, Vec<u8>)> = trimmed.as_ref().map(|(files, _, _)| tokenizer_bytes(files)).transpose()?.unwrap_or_default();
            qwen3tts::export::export_edited(&self.base, &dir, &self.id, &qwen3tts::export::Edited { config: &edited.config, tensors: &list, files: &files }, &mut || true).map_err(backend)?;
            Some(dir)
        };
        let outcome = SpeechSurgeryOutcome {
            baseline: MeasuredLoss { loss: baseline.loss, talker_ce: baseline.talker_ce, code_predictor_ce: baseline.code_predictor_ce },
            edits: costs,
            params_before: ck.params(),
            params_after: if self.profile { ck.params() } else { edited.params() },
            shard_texts: trimmed.as_ref().map(|(_, _, s)| *s),
            vocabulary: trimmed.map(|(_, r, _)| r),
            checkpoint,
            manifest: manifest_path.clone(),
        };
        let doc = serde_json::json!({"manifest": manifest, "outcome": outcome});
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&doc).map_err(backend)?).map_err(|e| backend(format!("{}: {e}", manifest_path.display())))?;
        Ok(outcome)
    }

    /// The ids every vocabulary trim of the plan keeps: every token the
    /// shards hold and the corpus encodes to, with the rewritten files.
    fn vocabulary(&self, ck: &Checkpoint, shard: &qwen3tts::shard::Shard, held_out: Option<&qwen3tts::shard::Shard>) -> Result<Trimmed> {
        let names: Vec<&String> = self.plan.iter().filter_map(|e| if let Edit::TrimVocabulary { vocabulary } = e { Some(vocabulary) } else { None }).collect();
        if names.is_empty() {
            return Ok((BTreeMap::new(), None));
        }
        if names.iter().any(|n| n.as_str() != "text") {
            return Err(backend("a Qwen3-TTS checkpoint has one vocabulary to trim, `text`"));
        }
        let read = |f: &str| std::fs::read_to_string(self.base.join(f)).map_err(|e| backend(format!("{f}: {e}")));
        let files = TokenizerFiles { vocab_json: read("vocab.json")?, merges_txt: read("merges.txt")?, tokenizer_config: serde_json::from_str(&read("tokenizer_config.json")?).map_err(backend)? };
        // The corpus: every utterance text of the calibration shard and the texts given; kept exactly: the token ids
        // both shards store (a recovery reads the held-out shard too, and the ids may come from another loader than
        // the files' encoding); the probe: the held-out shard's texts and the probe texts given.
        let stored = |s: &qwen3tts::shard::Shard| (0..s.len()).flat_map(|r| s.record(r).text_ids.iter().map(|&t| t as usize)).collect::<Vec<_>>();
        let mut extra: Vec<usize> = stored(shard);
        extra.extend(held_out.map(stored).unwrap_or_default());
        // The config's own special tokens stay whatever the corpus holds.
        for key in qwen3tts::surgery::TOKEN_ID_KEYS {
            if let Some(id) = ck.config.get(key).and_then(serde_json::Value::as_u64) {
                extra.push(id as usize);
            }
        }
        extra.sort_unstable();
        extra.dedup();
        let texts = |s: &qwen3tts::shard::Shard| -> Vec<String> { (0..s.len()).map(|r| s.record(r).id.text.clone()).filter(|t| !t.is_empty()).collect() };
        let shard_texts = texts(shard);
        let held_out_texts = held_out.map(texts).unwrap_or_default();
        let corpus: Vec<&str> = shard_texts.iter().chain(&self.corpus).map(String::as_str).collect();
        let probe: Vec<&str> = held_out_texts.iter().chain(&self.probe).map(String::as_str).collect();
        let t = bpe::trim(&files, &corpus, &extra, &probe).map_err(backend)?;
        let tok = data::qwen_tokenizer::QwenBpe::from_split_files(&serde_json::from_str(&files.vocab_json).map_err(backend)?, &files.merges_txt, added_records(&files.tokenizer_config)).map_err(backend)?;
        let (mut with_text, mut differ) = (0, 0);
        for r in 0..shard.len() {
            let rec = shard.record(r);
            if rec.id.text.is_empty() || rec.text_ids.len() < 8 {
                continue;
            }
            with_text += 1;
            // A record's ids frame its text: `<|im_start|>assistant\n` before, `<|im_end|>\n<|im_start|>assistant\n` after.
            differ += usize::from(data::tokenizer::Tokenizer::encode(&tok, &rec.id.text) != rec.text_ids[3..rec.text_ids.len() - 5]);
        }
        Ok((BTreeMap::from([("text".to_string(), t.keep.clone())]), Some((t.files, t.report, (with_text, differ)))))
    }
}

/// Per vocabulary the ids a trim keeps, and the rewritten tokenizer files with the trim's report.
type Trimmed = (BTreeMap<String, Vec<usize>>, Option<(TokenizerFiles, BpeReport, (usize, usize))>);

/// The added tokens of a `tokenizer_config.json` as `[{content, id}]` records.
fn added_records(config: &serde_json::Value) -> serde_json::Value {
    let records: Vec<serde_json::Value> = config["added_tokens_decoder"].as_object().map(|m| m.iter().filter_map(|(id, r)| Some(serde_json::json!({"content": r["content"].as_str()?, "id": id.parse::<u64>().ok()?}))).collect()).unwrap_or_default();
    serde_json::Value::Array(records)
}

/// A one-shot layer ranking reads the calibration's per-layer ablations; a greedy one measures its own.
fn needs_ablation(e: &Edit) -> bool {
    matches!(e, Edit::RemoveLayers { by: Some(plan::LayerScore::Kl | plan::LayerScore::Loss), greedy: false, .. })
}

fn needs_curvature(e: &Edit) -> bool {
    matches!(e, Edit::SelectChannels { score: Score::Fisher | Score::Taylor, .. } | Edit::FoldEmbedding { score: Score::Fisher | Score::Taylor, .. })
}

fn qwen3tts_graph(c: &Checkpoint) -> std::result::Result<SurgeryGraph, String> {
    qwen3tts::surgery::graph(&c.config, &|n| c.tensors.contains_key(n))
}

fn modelopt_digest(path: &Path) -> Result<String> {
    brain_modelstore::fetch::file_digest(path).map_err(|e| backend(format!("{}: {e}", path.display())))
}

/// The speech objective's base over the tensors of `ck`.
fn base_of(ck: &Checkpoint) -> Result<Base> {
    Base::from_parts(&ck.config, &|n| ck.get(n).ok().map(|t| t.shape.iter().map(|&d| d as u64).collect()), &|n| ck.get(n).ok().map(|t| t.data.clone())).map_err(backend)
}

/// Measure the edited `ck`, its inputs moved as the edits moved them.
/// Run `f` with the evaluator's view of how the edits so far moved the checkpoint's inputs.
fn with_inputs<T>(inputs: &Inputs, f: impl FnOnce(&cal::Inputs<'_>) -> T) -> T {
    let text_map: Option<std::collections::HashMap<u32, u32>> = inputs.vocabularies.get("text").map(|keep| keep.iter().enumerate().map(|(new, &old)| (old as u32, new as u32)).collect());
    let text = text_map.as_ref().map(|m| move |id: u32| m.get(&id).copied());
    let xvector = inputs.streams.get("talker").map(|(b, k)| move |v: &[f32]| b.project(v, *k));
    let text_ref: Option<&cal::TextMap> = text.as_ref().map(|f| f as &cal::TextMap);
    let xvec_ref: Option<&cal::XVectorMap> = xvector.as_ref().map(|f| f as &cal::XVectorMap);
    f(&cal::Inputs { text: text_ref, xvector: xvec_ref })
}

fn measure(ev: &Evaluator, ck: &Checkpoint, inputs: &Inputs) -> Result<Measurement> {
    with_inputs(inputs, |i| ev.measure(base_of(ck)?, i).map_err(backend))
}

/// A greedy `remove_layers` edit resolved into the layers it removes, chosen on the calibration records of the
/// checkpoint as edited so far (one model built, each candidate set masked on it), so the held-out records stay a
/// measurement only. Answers the resolved edit and the method the manifest names.
fn resolve_greedy(edit: &Edit, ck: &Checkpoint, inputs: &Inputs, ev: &Evaluator, mtp_weight: f32, on_progress: &mut dyn FnMut(&str)) -> Result<(Edit, String)> {
    let Edit::RemoveLayers { stack, count: Some(count), by: Some(by @ (plan::LayerScore::Kl | plan::LayerScore::Loss)), layers: None, greedy: true } = edit else {
        return Err(backend("remove_layers: a greedy choice takes count and by (kl or loss), not layers"));
    };
    let graph = qwen3tts_graph(ck).map_err(backend)?;
    let st = graph.stack(stack).map_err(backend)?;
    let m = ev.build(base_of(ck)?).map_err(backend)?;
    let mut cost = |set: &[usize]| -> std::result::Result<f64, String> {
        let measured = modelopt::surgery::with_layers_masked(&m, st, set, |m| with_inputs(inputs, |i| ev.measure_model(m, i)))?;
        Ok(if *by == plan::LayerScore::Kl { measured.kl(mtp_weight) } else { measured.loss })
    };
    let mut round = 0;
    let chosen = modelopt::surgery::greedy_removal(st.count, *count, &mut cost, &mut |l, c| {
        round += 1;
        on_progress(&format!("{stack}: greedy round {round}: remove layer {l}, {} {c:.4}", format!("{by:?}").to_lowercase()));
    })
    .map_err(backend)?;
    let layers: Vec<usize> = chosen.iter().map(|(l, _)| *l).collect();
    Ok((Edit::RemoveLayers { stack: stack.clone(), count: None, by: None, layers: Some(layers), greedy: false }, format!("greedy_ablation_{by:?}").to_lowercase()))
}

/// The Fisher and group-Taylor scores of every MLP and KV-group dimension of both stacks, over the calibration batches.
fn curvature(ck: &Checkpoint, graph: &SurgeryGraph, ev: &Evaluator) -> Result<BTreeMap<String, ChannelCurvature>> {
    let m = ev.build(base_of(ck)?).map_err(backend)?;
    let dims: Vec<&model::surgery::ChannelDim> = graph.dims.iter().collect();
    let mut load = |m: &qwen3tts::speech_sft::SpeechSftModel, i: usize| ev.load(m, i);
    let c = modelopt::surgery::scores::curvature(&m, ck, &dims, ev.batches(), &mut load).map_err(backend)?;
    Ok(dims.iter().zip(c.fisher.into_iter().zip(c.taylor)).map(|(d, (fisher, taylor))| (d.name.clone(), ChannelCurvature { fisher, taylor })).collect())
}

/// The rewritten tokenizer files, as the export writes them.
fn tokenizer_bytes(files: &TokenizerFiles) -> Result<Vec<(String, Vec<u8>)>> {
    Ok(vec![
        ("vocab.json".into(), files.vocab_json.clone().into_bytes()),
        ("merges.txt".into(), files.merges_txt.clone().into_bytes()),
        ("tokenizer_config.json".into(), serde_json::to_vec_pretty(&files.tokenizer_config).map_err(backend)?),
    ])
}

/// The calibration of a checkpoint is written beside its surgery; read one back.
pub fn read_calibration(path: &Path) -> Result<Calibration> {
    modelopt::surgery::calibration::load(path).map_err(backend)
}
