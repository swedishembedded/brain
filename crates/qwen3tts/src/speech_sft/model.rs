// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The Talker and its code predictor as ONE trainable [`model::Model`],
//! teacher forced on recorded speech.
//!
//! Where every parameter lives, and who steps it:
//!
//! | parameters | where | optimizer |
//! |---|---|---|
//! | Talker decoder layers and final norm | a [`qwen3::Qwen`], every row spliced in, its own head off | its `adamw_step` |
//! | code-predictor decoder layers and norm | a second [`qwen3::Qwen`] (batch = frames, block = one frame's codebooks) | its `adamw_step` |
//! | `codec_head`, the 15 residual heads, `small_to_mtp_projection` | a parameter store on a device of their own ([`model::head_ce::HeadCe`], [`model::projector::MlpProjector`]) | `optim::Optim` |
//! | Talker `codec_embedding`, the 15 residual `codec_embedding`s | host ([`HostTables`]) | [`model::hostopt::HostAdam`] |
//! | text embedding and text projection | host, frozen in this release | - |
//!
//! One global gradient norm spans all of them ([`model::Model::adamw_step`]);
//! each group has its own learning-rate multiplier ([`LrMultipliers`]).
//!
//! **The loss**, per batch: `CE_talker + mtp_weight * CE_code_predictor`, each a
//! mean over its supervised positions. The Talker predicts frame `t`'s first
//! codebook at row `prefix - 1 + t` and the end of speech at the row after the
//! last frame, only for a record that is complete. The code predictor reads,
//! per frame, `[talker_hidden, cb0, cb1 .. cb14]` (the Talker hidden being the
//! one at the row that predicted the frame's first codebook, as generation
//! feeds it) and head `k` at position `k` predicts codebook `k`.
//!
//! **The gradient routes**, all of them: the code predictor's input gradient
//! at position 0 joins the Talker head's gradient at that Talker row, position 1
//! lands on the Talker table's row of `cb0`, positions `2..` on the residual
//! table that embedded them; the Talker's input-row gradient lands on every
//! codec row its row summed (the frame's codes, the prompt's special, language
//! and voice rows), never on the frozen text path or the x-vector.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu};
use model::head_ce::{HeadCe, HeadCeConfig, HeadKernels, IGNORE};
use model::ops::{Ops, Weight};
use model::projector::{MlpProjector, ProjectorConfig, ProjectorKind};
use optim::gradnorm::{DeviceGradNorm, NormItem};
use optim::Optim;
use paramstore::precision::{GradDtype, ParamPrecision};
use paramstore::ParamStore;
use qwen3::{Qwen, QwenConfig};
use serde::{Deserialize, Serialize};

use super::tables::{HostGrads, HostTables};
use crate::config::{MtpConfig, TalkerConfig, SPLICED_VOCAB};
use crate::import::{mtp_brain_to_hf, mtp_hf_to_brain, talker_brain_to_hf, talker_hf_to_brain};
use crate::prompt::{Part, PromptPlan, TalkerHost, TtsSpecials};
use crate::talker::TextProjection;

/// The Talker's input table, under its upstream name.
pub const TALKER_TABLE: &str = "talker.model.codec_embedding.weight";
/// The Talker's head.
pub const TALKER_HEAD: &str = "talker.codec_head.weight";
/// The frozen text path.
pub const TEXT_EMBEDDING: &str = "talker.model.text_embedding.weight";
pub const TEXT_PROJECTION: [&str; 4] = ["talker.text_projection.linear_fc1.weight", "talker.text_projection.linear_fc1.bias", "talker.text_projection.linear_fc2.weight", "talker.text_projection.linear_fc2.bias"];
/// `small_to_mtp_projection`'s weight and bias (only where the widths differ).
pub const PROJECTION: [&str; 2] = ["talker.code_predictor.small_to_mtp_projection.weight", "talker.code_predictor.small_to_mtp_projection.bias"];

/// Code predictor residual table `i` (it embeds codebook `i + 1`).
pub fn residual_table(i: usize) -> String {
    format!("talker.code_predictor.model.codec_embedding.{i}.weight")
}

/// Code predictor head `i` (it predicts codebook `i + 1`).
pub fn residual_head(i: usize) -> String {
    format!("talker.code_predictor.lm_head.{i}.weight")
}

/// Learning-rate multipliers of the parameter groups, on the run's rate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LrMultipliers {
    /// The Talker decoder and its head.
    pub talker: f32,
    /// The code predictor: decoder, heads and projection.
    pub code_predictor: f32,
    /// Every codec table row the base checkpoint already trained.
    pub codec_tables: f32,
    /// The rows a run allocates for a new language or voice.
    pub new_rows: f32,
}

impl Default for LrMultipliers {
    fn default() -> Self {
        LrMultipliers { talker: 1.0, code_predictor: 1.0, codec_tables: 1.0, new_rows: 1.0 }
    }
}

impl LrMultipliers {
    /// Refuse a multiplier that is negative or not finite, naming it.
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [("talker", self.talker), ("code_predictor", self.code_predictor), ("codec_tables", self.codec_tables), ("new_rows", self.new_rows)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("lr_multipliers.{name} must be finite and not negative, got {v}"));
            }
        }
        Ok(())
    }
}

/// Everything a build is sized and wired by.
#[derive(Clone, Debug)]
pub struct SftShape {
    pub talker: TalkerConfig,
    pub mtp: MtpConfig,
    pub specials: TtsSpecials,
    /// Utterances per micro-batch.
    pub batch: u32,
    /// Talker rows per utterance: the longest prompt plus its frames.
    pub block: u32,
    /// Frames the code predictor takes per micro-batch.
    pub frames: u32,
    /// The weight of the code predictor's loss (upstream: 0.3).
    pub mtp_weight: f32,
    pub lr: LrMultipliers,
    /// Talker table rows allocated for new languages and voices ([`LrMultipliers::new_rows`]).
    pub new_rows: Vec<u32>,
    /// Hidden width of the text path's projection.
    pub text_inter: u32,
}

impl SftShape {
    /// The Talker width: of its rows, its codec tables and its x-vectors.
    pub fn d(&self) -> usize {
        self.talker.d_model as usize
    }

    fn talker_qwen(&self) -> QwenConfig {
        // Every input row is spliced in: the decoder's own token table is never read.
        QwenConfig { vocab: SPLICED_VOCAB, tie_embeddings: true, ..self.talker.to_qwen(self.block) }.with_defaults()
    }

    fn cp_qwen(&self) -> QwenConfig {
        self.mtp.to_qwen(self.mtp.num_code_groups)
    }

    fn residuals(&self) -> usize {
        self.mtp.n_residual() as usize
    }

    fn projected(&self) -> bool {
        self.mtp.embedding_dim != self.mtp.d_model
    }

    /// The device-held parameters outside the two decoders: the heads and the projection.
    fn aux_params(&self) -> Vec<(String, usize)> {
        let (d, v) = (self.d(), self.talker.vocab as usize);
        let (dc, cv) = (self.mtp.d_model as usize, self.mtp.vocab as usize);
        let mut out = vec![(TALKER_HEAD.to_string(), v * d)];
        out.extend((0..self.residuals()).map(|i| (residual_head(i), cv * dc)));
        if self.projected() {
            out.push((PROJECTION[0].to_string(), dc * d));
            out.push((PROJECTION[1].to_string(), dc));
        }
        out
    }

    /// The host-held trained tables.
    fn table_params(&self) -> Vec<(String, usize)> {
        let d = self.d();
        let mut out = vec![(TALKER_TABLE.to_string(), self.talker.vocab as usize * d)];
        out.extend((0..self.residuals()).map(|i| (residual_table(i), self.mtp.vocab as usize * d)));
        out
    }

    /// The decoders' trained parameters, upstream-named, with the decoder they belong to.
    fn decoder_params(&self) -> Vec<(String, usize, Decoder)> {
        let keep = |n: &String| n != "tok.weight" && n != "lm_head.weight";
        let talker = self.talker_qwen().param_list().into_iter().filter(|(n, _)| keep(n)).map(|(n, s)| (talker_brain_to_hf(&n).unwrap_or_else(|| panic!("talker {n} has no upstream name")), s, Decoder::Talker));
        let cp = self.cp_qwen().param_list().into_iter().filter(|(n, _)| keep(n)).map(|(n, s)| (mtp_brain_to_hf(&n).unwrap_or_else(|| panic!("code predictor {n} has no upstream name")), s, Decoder::CodePredictor));
        talker.chain(cp).collect()
    }

    /// The frozen tensors a build reads besides the trained ones.
    pub fn frozen_params(&self) -> Vec<(String, usize)> {
        let (th, inter, d) = (self.talker.text_hidden_size as usize, self.text_inter as usize, self.d());
        vec![
            (TEXT_EMBEDDING.to_string(), self.talker.text_vocab_size as usize * th),
            (TEXT_PROJECTION[0].to_string(), inter * th),
            (TEXT_PROJECTION[1].to_string(), inter),
            (TEXT_PROJECTION[2].to_string(), d * inter),
            (TEXT_PROJECTION[3].to_string(), d),
        ]
    }

    /// Where an upstream-named trained parameter lives.
    fn place(&self, name: &str) -> Option<Place> {
        if name.starts_with("talker.code_predictor.model.layers.") || name == "talker.code_predictor.model.norm.weight" {
            return mtp_hf_to_brain(name).map(Place::CodePredictor);
        }
        if name == TALKER_TABLE {
            return Some(Place::Table(0));
        }
        if let Some(i) = (0..self.residuals()).find(|&i| residual_table(i) == name) {
            return Some(Place::Table(i + 1));
        }
        if self.aux_params().iter().any(|(n, _)| n == name) {
            return Some(Place::Aux);
        }
        if name.starts_with("talker.model.layers.") || name == "talker.model.norm.weight" {
            return talker_hf_to_brain(name).map(Place::Talker);
        }
        None
    }
}

impl model::ModelConfig for SftShape {
    fn param_list(&self) -> Vec<(String, usize)> {
        let mut out: Vec<(String, usize)> = self.decoder_params().into_iter().map(|(n, s, _)| (n, s)).collect();
        out.extend(self.aux_params());
        out.extend(self.table_params());
        out
    }

    fn to_json(&self) -> serde_json::Value {
        let sp = &self.specials;
        serde_json::json!({
            "model": "qwen3_tts_speech_sft",
            "talker": self.talker.to_qwen(self.block).to_json(),
            "talker_text": {"text_hidden_size": self.talker.text_hidden_size, "text_vocab_size": self.talker.text_vocab_size, "text_inter": self.text_inter, "num_code_groups": self.talker.num_code_groups},
            "mtp": self.mtp.to_json(),
            "specials": [sp.tts_bos, sp.tts_eos, sp.tts_pad, sp.codec_nothink, sp.codec_think, sp.codec_think_bos, sp.codec_think_eos, sp.codec_pad, sp.codec_bos, sp.codec_eos],
            "batch": self.batch, "block": self.block, "frames": self.frames,
            "mtp_weight_bits": self.mtp_weight.to_bits(),
            "lr": self.lr,
            "new_rows": self.new_rows,
        })
    }

    fn from_json(v: &serde_json::Value) -> Self {
        let q = QwenConfig::from_json_checked(&v["talker"]).unwrap_or_else(|e| panic!("speech sft config: {e}"));
        let mut talker = TalkerConfig::from_qwen(&q);
        let tt = &v["talker_text"];
        let u = |o: &serde_json::Value, k: &str| o[k].as_u64().unwrap_or_else(|| panic!("speech sft config: no {k}")) as u32;
        talker.text_hidden_size = u(tt, "text_hidden_size");
        talker.text_vocab_size = u(tt, "text_vocab_size");
        talker.num_code_groups = u(tt, "num_code_groups");
        let ids: Vec<u32> = serde_json::from_value(v["specials"].clone()).unwrap_or_else(|e| panic!("speech sft config: specials: {e}"));
        let specials = TtsSpecials {
            tts_bos: ids[0],
            tts_eos: ids[1],
            tts_pad: ids[2],
            codec_nothink: ids[3],
            codec_think: ids[4],
            codec_think_bos: ids[5],
            codec_think_eos: ids[6],
            codec_pad: ids[7],
            codec_bos: ids[8],
            codec_eos: ids[9],
            lang: HashMap::new(),
            spk_id: HashMap::new(),
            checkpoint: None,
        };
        SftShape {
            talker,
            mtp: MtpConfig::from_brain_json(&v["mtp"]),
            specials,
            batch: u(v, "batch"),
            block: u(v, "block"),
            frames: u(v, "frames"),
            mtp_weight: f32::from_bits(u(v, "mtp_weight_bits")),
            lr: serde_json::from_value(v["lr"].clone()).unwrap_or_default(),
            new_rows: serde_json::from_value(v["new_rows"].clone()).unwrap_or_default(),
            text_inter: u(tt, "text_inter"),
        }
    }

    fn vocab(&self) -> u32 {
        self.talker.vocab
    }

    fn block_size(&self) -> u32 {
        self.block
    }

    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> Self {
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decoder {
    Talker,
    CodePredictor,
}

/// Where a trained parameter lives: a decoder (under its brain name), the
/// device store of heads and projection, or host table `i` (0 = the Talker's).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Place {
    Talker(String),
    CodePredictor(String),
    Aux,
    Table(usize),
}

/// One utterance of a batch: its prompt plan, its speaker x-vector (read only
/// when the plan has the slot), its `[frames, num_code_groups]` codes, and
/// whether the end of speech is supervised.
#[derive(Clone, Debug)]
pub struct Example {
    pub plan: PromptPlan,
    pub xvector: Vec<f32>,
    pub codes: Vec<u32>,
    pub complete: bool,
}

/// What a loaded batch supervises, fixed when it is loaded.
struct Loaded {
    examples: Vec<Example>,
    /// Talker rows with a target, as global rows `slot * block + row`, and their labels.
    talker_rows: Vec<usize>,
    talker_labels: Vec<u32>,
    /// Per code-predictor frame: `(slot, frame, the Talker row whose hidden it reads)`.
    frames: Vec<(usize, usize, usize)>,
}

/// What the latest forward left for the backward.
struct Forward {
    talker_hidden: Vec<f32>,
    cp_hidden: Vec<f32>,
    /// The code predictor's raw input rows (before the projection).
    cp_raw: Vec<f32>,
    talker_count: u32,
    cp_count: u32,
}

/// The latest forward's measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ForwardStats {
    /// Summed cross-entropy of the Talker's targets and how many there were.
    pub talker_sum: f64,
    pub talker_count: u32,
    /// Summed cross-entropy of the code predictor's targets and how many.
    pub cp_sum: f64,
    pub cp_count: u32,
    /// End-of-speech targets the Talker's greedy pick got right, of how many.
    pub eos_hits: u32,
    pub eos_count: u32,
    /// Frames the batch trained on.
    pub frames: u32,
}

impl ForwardStats {
    /// `CE_talker + mtp_weight * CE_code_predictor`.
    pub fn loss(&self, mtp_weight: f32) -> f32 {
        let mean = |s: f64, n: u32| if n == 0 { 0.0 } else { s / f64::from(n) };
        (mean(self.talker_sum, self.talker_count) + f64::from(mtp_weight) * mean(self.cp_sum, self.cp_count)) as f32
    }

    /// The sum of two measurements (a step's micro-batches, an evaluation's batches).
    pub fn add(&mut self, o: &ForwardStats) {
        self.talker_sum += o.talker_sum;
        self.talker_count += o.talker_count;
        self.cp_sum += o.cp_sum;
        self.cp_count += o.cp_count;
        self.eos_hits += o.eos_hits;
        self.eos_count += o.eos_count;
        self.frames += o.frames;
    }
}

/// The heads and the projection, on their own device.
struct Aux {
    gpu: Gpu,
    ops: Ops,
    ps: ParamStore,
    opt: Optim,
    norm: DeviceGradNorm,
    talker_head: HeadCe,
    cp_heads: Vec<HeadCe>,
    projector: Option<MlpProjector>,
    /// Gathered hidden rows and their gradient, for the Talker head and a residual head.
    talker_h: DeviceBuffer,
    talker_dh: DeviceBuffer,
    cp_h: DeviceBuffer,
    cp_dh: DeviceBuffer,
    /// The projection's input and its gradient's in/out buffers.
    proj_in: DeviceBuffer,
    proj_d_out: DeviceBuffer,
    proj_d_in: DeviceBuffer,
}

/// The kernels the device of the heads and the projection dispatches.
pub(crate) fn aux_kernels() -> Vec<(&'static str, &'static str)> {
    let mut list = model::ops::training_kernel_list().to_vec();
    model::ops::extend_kernel_list(&mut list, model::head_ce::kernels());
    model::ops::extend_kernel_list(&mut list, model::projector::PROJECTOR_PIPELINES);
    model::ops::extend_kernel_list(
        &mut list,
        &[
            ("adamw", kernels::ADAMW),
            ("gradnorm_sq", kernels::GRADNORM_SQ),
            ("grad_scale", kernels::GRAD_SCALE),
            ("clip_coef", kernels::CLIP_COEF),
            ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
            ("gradnorm_part", kernels::GRADNORM_PART),
            ("gradnorm_part_bf16", kernels::GRADNORM_PART_BF16),
            ("clip_coef_wg", kernels::CLIP_COEF_WG),
            ("argmax_part", kernels::ARGMAX_PART),
            ("argmax_final", kernels::ARGMAX_FINAL),
        ],
    );
    model::ops::extend_kernel_list(&mut list, optim::MOMENT_KERNELS);
    list
}

/// Rows of logits a head keeps alive for `n` tokens: all of them up to 256,
/// else 256 (chunk starts stay on 64-row binding boundaries).
fn head_rows(n: usize) -> u32 {
    n.clamp(1, 256) as u32
}

/// The host optimizer's tensors: every table whole, the Talker table cut at the
/// new rows so they take their own rate. `(table, first element, elements, new)`.
fn segments(shape: &SftShape, tables: &HostTables) -> Vec<(usize, usize, usize, bool)> {
    let d = shape.d();
    let mut new: Vec<u32> = shape.new_rows.clone();
    new.sort_unstable();
    new.dedup();
    let mut out = Vec::new();
    let mut at = 0usize;
    for r in new {
        let start = r as usize * d;
        if start > at {
            out.push((0, at, start - at, false));
        }
        out.push((0, start, d, true));
        at = start + d;
    }
    if at < tables.codec.len() {
        out.push((0, at, tables.codec.len() - at, false));
    }
    for (i, t) in tables.residual.iter().enumerate() {
        out.push((i + 1, 0, t.len(), false));
    }
    out
}

/// The composite trainable model. See the module documentation.
pub struct SpeechSftModel {
    shape: SftShape,
    talker: Qwen,
    cp: Qwen,
    talker_norm: DeviceGradNorm,
    cp_norm: DeviceGradNorm,
    aux: Aux,
    tables: RefCell<HostTables>,
    grads: RefCell<HostGrads>,
    host_opt: RefCell<model::hostopt::HostAdam>,
    segments: Vec<(usize, usize, usize, bool)>,
    loaded: RefCell<Option<Loaded>>,
    forward: RefCell<Option<Forward>>,
    stats: Cell<ForwardStats>,
    last_norm: Cell<Option<f32>>,
}

/// Move tensor `name` (of `numel` values) out of `init`.
fn take(init: &mut HashMap<String, Vec<f32>>, name: &str, numel: usize) -> Result<Vec<f32>, String> {
    let v = init.remove(name).ok_or_else(|| format!("speech sft: no tensor {name}"))?;
    if v.len() != numel {
        return Err(format!("speech sft: {name} holds {} values, the configuration needs {numel}", v.len()));
    }
    Ok(v)
}

/// The host tables and the frozen text path of `shape`, moved out of `init`.
pub(crate) fn host_tables(shape: &SftShape, init: &mut HashMap<String, Vec<f32>>) -> Result<HostTables, String> {
    let d = shape.d();
    let (th, inter) = (shape.talker.text_hidden_size as usize, shape.text_inter as usize);
    let frozen = shape.frozen_params();
    let mut get = |i: usize| take(init, &frozen[i].0, frozen[i].1);
    let text = TextProjection { text_embedding: Some(get(0)?), fc1_w: get(1)?, fc1_b: get(2)?, fc2_w: get(3)?, fc2_b: get(4)?, in_dim: th, inter, out: d, text_vocab: shape.talker.text_vocab_size as usize };
    let tparams = shape.table_params();
    let codec = take(init, &tparams[0].0, tparams[0].1)?;
    let residual = tparams[1..].iter().map(|(n, s)| take(init, n, *s)).collect::<Result<Vec<_>, _>>()?;
    Ok(HostTables::new(d, text, codec, residual))
}

impl SpeechSftModel {
    /// Build from upstream-named tensors: every [`model::ModelConfig::param_list`]
    /// entry and every [`SftShape::frozen_params`] one, each moved into the
    /// part that holds it (a release's tensors are never held twice). `seed`
    /// fills the decoders' unread token tables.
    pub fn build(shape: SftShape, mut init: HashMap<String, Vec<f32>>, seed: u64) -> Result<SpeechSftModel, String> {
        shape.lr.validate()?;
        let d = shape.d();
        if shape.mtp.embedding_dim as usize != d {
            return Err(format!("speech sft: the code predictor embeds into {} columns, the Talker is {d} wide", shape.mtp.embedding_dim));
        }
        if shape.mtp.num_code_groups != shape.talker.num_code_groups {
            return Err(format!("speech sft: the code predictor fills {} codebooks, the Talker has {}", shape.mtp.num_code_groups, shape.talker.num_code_groups));
        }
        let mut decoder = |cfg: &QwenConfig, which: Decoder| -> Result<Qwen, String> {
            let mut weights = qwen3::init_weights(cfg, seed);
            for (hf, numel, owner) in shape.decoder_params() {
                if owner != which {
                    continue;
                }
                let brain = match which {
                    Decoder::Talker => talker_hf_to_brain(&hf),
                    Decoder::CodePredictor => mtp_hf_to_brain(&hf),
                }
                .expect("a listed parameter maps");
                weights.insert(brain, take(&mut init, &hf, numel)?);
            }
            let (b, t) = match which {
                Decoder::Talker => (shape.batch, shape.block),
                Decoder::CodePredictor => (shape.frames, shape.mtp.num_code_groups),
            };
            let mut q = Qwen::new(cfg.clone(), b, t, &weights);
            q.enable_mm_splices(&[(0, b * t)]);
            q.enable_external_head();
            let store = model::Model::optimizer_state(&mut q).ok_or("speech sft: the decoder exposes no parameter store")?.store;
            store.freeze_where(|n| n == "tok.weight");
            // Every row is spliced: the token ids are never read, and no built-in head runs.
            q.set_batch(&vec![0u32; (b * t) as usize], &vec![IGNORE; (b * t) as usize]);
            Ok(q)
        };
        let talker = decoder(&shape.talker_qwen(), Decoder::Talker)?;
        let cp = decoder(&shape.cp_qwen(), Decoder::CodePredictor)?;
        let talker_norm = DeviceGradNorm::new(talker.gpu())?;
        let cp_norm = DeviceGradNorm::new(cp.gpu())?;

        if let Some(&r) = shape.new_rows.iter().find(|&&r| r >= shape.talker.vocab) {
            return Err(format!("speech sft: new row {r} is past the Talker table's {} rows", shape.talker.vocab));
        }
        let tables = host_tables(&shape, &mut init)?;
        let grads = HostGrads::zeros_like(&tables);
        let segments = segments(&shape, &tables);
        let host_params: Vec<(String, usize)> = segments.iter().map(|&(t, at, n, _)| (format!("table{t}@{at}"), n)).collect();
        let host_opt = model::hostopt::HostAdam::new(&host_params, ParamPrecision::FP32)?;

        let aux = Self::build_aux(&shape, &talker, &mut init)?;
        Ok(SpeechSftModel {
            shape,
            talker,
            cp,
            talker_norm,
            cp_norm,
            aux,
            tables: RefCell::new(tables),
            grads: RefCell::new(grads),
            host_opt: RefCell::new(host_opt),
            segments,
            loaded: RefCell::new(None),
            forward: RefCell::new(None),
            stats: Cell::new(ForwardStats::default()),
            last_norm: Cell::new(None),
        })
    }

    fn build_aux(shape: &SftShape, talker: &Qwen, init: &mut HashMap<String, Vec<f32>>) -> Result<Aux, String> {
        let gpu = talker.gpu().new_like(&aux_kernels());
        let mut ops = Ops::new(gpu.share())?;
        ops.set_gemm_precision(gpu_core::provider::library::GemmPrecision::Fp32Reference);
        let params = shape.aux_params();
        let mut weights = HashMap::new();
        for (n, s) in &params {
            weights.insert(n.clone(), take(init, n, *s)?);
        }
        let mut ps = ParamStore::new(&gpu, params.clone(), &weights);
        for (n, _) in &params {
            let mult = if n == TALKER_HEAD { shape.lr.talker } else { shape.lr.code_predictor };
            ps.set_lr_mult(n, mult);
        }
        let k = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("speech sft: kernel {name} is not registered"));
        let opt = Optim::new(k("adamw")?, k("gradnorm_sq")?, k("grad_scale")?, k("clip_coef")?, k("grad_scale_buf")?);
        let norm = DeviceGradNorm::new(&gpu)?;
        let (d, v) = (shape.d() as u32, shape.talker.vocab);
        let (dc, cv) = (shape.mtp.d_model, shape.mtp.vocab);
        let talker_rows = (shape.batch * shape.block) as usize;
        let head = |name: &str, n: u32, k: u32, rows: usize| -> Result<HeadCe, String> {
            let w = Weight::dense_f32(ps.w(name).clone(), n, k);
            HeadCe::new(&ops, k, n, &w, HeadCeConfig { rows_per_chunk: head_rows(rows), tile_budget_words: None }, HeadKernels::from_gpu(&gpu)?)
        };
        let talker_head = head(TALKER_HEAD, v, d, talker_rows)?;
        let cp_heads = (0..shape.residuals()).map(|i| head(&residual_head(i), cv, dc, shape.frames as usize)).collect::<Result<Vec<_>, _>>()?;
        let cp_rows = shape.frames * shape.mtp.num_code_groups;
        let projector = if shape.projected() {
            let cfg = ProjectorConfig { kind: ProjectorKind::Mlp, depth: 1, input_dim: d, n_embed: dc, out_dim: dc, bias: true };
            let p: HashMap<String, DeviceBuffer> = [("in.weight", PROJECTION[0]), ("in.bias", PROJECTION[1])].iter().map(|(k, n)| (k.to_string(), ps.w(n).clone())).collect();
            let g: HashMap<String, DeviceBuffer> = [("in.weight", PROJECTION[0]), ("in.bias", PROJECTION[1])].iter().map(|(k, n)| (k.to_string(), ps.g(n).clone())).collect();
            Some(MlpProjector::with_buffers(&gpu, cfg, cp_rows, p, g, true)?)
        } else {
            None
        };
        let st = |n: u64| gpu.storage(n.max(1));
        Ok(Aux {
            talker_h: st(talker_rows as u64 * u64::from(d)),
            talker_dh: st(talker_rows as u64 * u64::from(d)),
            cp_h: st(u64::from(shape.frames) * u64::from(dc)),
            cp_dh: st(u64::from(shape.frames) * u64::from(dc)),
            proj_in: st(u64::from(cp_rows) * u64::from(d)),
            proj_d_out: st(u64::from(cp_rows) * u64::from(dc)),
            proj_d_in: st(u64::from(cp_rows) * u64::from(d)),
            talker_head,
            cp_heads,
            projector,
            norm,
            opt,
            ps,
            ops,
            gpu,
        })
    }

    pub fn shape(&self) -> &SftShape {
        &self.shape
    }

    /// The latest forward's measurements.
    pub fn stats(&self) -> ForwardStats {
        self.stats.get()
    }

    /// Talker rows a plan of `prefix` rows and `frames` frames takes.
    pub fn rows_needed(prefix: usize, frames: usize) -> usize {
        prefix + frames
    }

    /// Load the next batch: at most [`SftShape::batch`] examples, each fitting the block.
    pub fn load(&self, examples: Vec<Example>) -> Result<(), String> {
        let s = &self.shape;
        let g = s.mtp.num_code_groups as usize;
        if examples.len() > s.batch as usize {
            return Err(format!("speech sft: {} examples for a batch of {}", examples.len(), s.batch));
        }
        let mut talker_rows = Vec::new();
        let mut talker_labels = Vec::new();
        let mut frames = Vec::new();
        for (slot, ex) in examples.iter().enumerate() {
            if ex.codes.is_empty() || !ex.codes.len().is_multiple_of(g) {
                return Err(format!("speech sft: example {slot}: {} codes are not whole frames of {g}", ex.codes.len()));
            }
            let (prefix, n) = (ex.plan.len(), ex.codes.len() / g);
            if Self::rows_needed(prefix, n) > s.block as usize {
                return Err(format!("speech sft: example {slot}: {prefix} prompt rows and {n} frames exceed the block of {}", s.block));
            }
            if ex.plan.rows.iter().flatten().any(|p| *p == Part::XVector) && ex.xvector.len() != s.d() {
                return Err(format!("speech sft: example {slot}: an x-vector of {} values, the Talker is {} wide", ex.xvector.len(), s.d()));
            }
            let base = slot * s.block as usize;
            for t in 0..n {
                talker_rows.push(base + prefix - 1 + t);
                talker_labels.push(ex.codes[t * g]);
                frames.push((slot, t, base + prefix - 1 + t));
            }
            if ex.complete {
                talker_rows.push(base + prefix - 1 + n);
                talker_labels.push(s.specials.codec_eos);
            }
        }
        if frames.len() > s.frames as usize {
            return Err(format!("speech sft: {} frames for a code-predictor capacity of {}", frames.len(), s.frames));
        }
        *self.loaded.borrow_mut() = Some(Loaded { examples, talker_rows, talker_labels, frames });
        *self.forward.borrow_mut() = None;
        Ok(())
    }

    /// The code of codebook `k` in frame `f` of the loaded batch.
    fn code(l: &Loaded, f: usize, k: usize, g: usize) -> u32 {
        let (slot, t, _) = l.frames[f];
        l.examples[slot].codes[t * g + k]
    }

    /// Rows `rows` of `[*, d]` `x`, concatenated.
    fn gather(x: &[f32], d: usize, rows: impl Iterator<Item = usize>) -> Vec<f32> {
        rows.flat_map(|r| x[r * d..(r + 1) * d].iter().copied()).collect()
    }

    fn forward_loss(&self) -> f32 {
        let s = &self.shape;
        let (d, dc, g) = (s.d(), s.mtp.d_model as usize, s.mtp.num_code_groups as usize);
        let (bt, fg) = ((s.batch * s.block) as usize, (s.frames * s.mtp.num_code_groups) as usize);
        let loaded = self.loaded.borrow();
        let l = loaded.as_ref().expect("speech sft: load a batch before the forward");
        let tables = self.tables.borrow();

        // The Talker's rows: the materialised plan, then one row per frame, as generation feeds them.
        let rows_range = gpu_core::markers::range(c"speech_sft.rows");
        let mut rows = vec![0.0f32; bt * d];
        for (slot, ex) in l.examples.iter().enumerate() {
            let prompt = ex.plan.materialise(&*tables, Some(&ex.xvector));
            let r = crate::train::policy_rows(&*tables, &*tables, &prompt, &ex.codes, g, None);
            let base = slot * s.block as usize * d;
            rows[base..base + r.rows.len()].copy_from_slice(&r.rows);
        }
        self.talker.write_img_embeds(&rows);
        drop(rows_range);
        let talker_range = gpu_core::markers::range(c"speech_sft.talker_fwd");
        let talker_hidden = self.talker.forward_hidden();
        drop(talker_range);
        let _heads = gpu_core::markers::range(c"speech_sft.talker_head_fwd");

        let aux = &self.aux;
        let n = l.talker_rows.len();
        aux.gpu.write_f32(&aux.talker_h, &Self::gather(&talker_hidden, d, l.talker_rows.iter().copied()));
        let (tl, picks) = aux.talker_head.forward_argmax(&aux.ops, &aux.talker_h, n as u32, &l.talker_labels, 1.0).unwrap_or_else(|e| panic!("speech sft: {e}"));
        let eos = s.specials.codec_eos;
        let eos_at: Vec<usize> = (0..n).filter(|&i| l.talker_labels[i] == eos).collect();
        let eos_hits = eos_at.iter().filter(|&&i| picks[i] == eos).count() as u32;

        // The code predictor's rows: [talker hidden, cb0 (the Talker's table), cb1 .. cb14 (its own)].
        drop(_heads);
        let cp_rows_range = gpu_core::markers::range(c"speech_sft.cp_rows");
        let mut raw = vec![0.0f32; fg * d];
        for (f, &(_, _, trow)) in l.frames.iter().enumerate() {
            let at = f * g * d;
            raw[at..at + d].copy_from_slice(&talker_hidden[trow * d..(trow + 1) * d]);
            raw[at + d..at + 2 * d].copy_from_slice(TalkerHost::codec_embed(&*tables, Self::code(l, f, 0, g)));
            for p in 2..g {
                let row = crate::prompt::MtpHost::codec_embed(&*tables, p - 2, Self::code(l, f, p - 1, g));
                raw[at + p * d..at + (p + 1) * d].copy_from_slice(row);
            }
        }
        let cp_in = match &aux.projector {
            Some(proj) => {
                aux.gpu.write_f32(&aux.proj_in, &raw);
                aux.gpu.submit(&[], &proj.forward_rows(&aux.gpu, &[&aux.proj_in], fg as u32));
                aux.gpu.read(proj.out(), fg * dc)
            }
            None => raw.clone(),
        };
        self.cp.write_img_embeds(&cp_in);
        drop(cp_rows_range);
        let cp_range = gpu_core::markers::range(c"speech_sft.cp_fwd");
        let cp_hidden = self.cp.forward_hidden();
        drop(cp_range);
        let _cp_heads = gpu_core::markers::range(c"speech_sft.cp_heads_fwd");

        let nf = l.frames.len();
        let mut cp_sum = 0.0f64;
        let mut cp_count = 0u32;
        for (k, head) in aux.cp_heads.iter().enumerate().map(|(i, h)| (i + 1, h)) {
            aux.gpu.write_f32(&aux.cp_h, &Self::gather(&cp_hidden, dc, (0..nf).map(|f| f * g + k)));
            let labels: Vec<u32> = (0..nf).map(|f| Self::code(l, f, k, g)).collect();
            let hl = head.forward(&aux.ops, &aux.cp_h, nf as u32, &labels, None, 1.0);
            cp_sum += hl.loss_sum;
            cp_count += hl.count;
        }
        let stats = ForwardStats { talker_sum: tl.loss_sum, talker_count: tl.count, cp_sum, cp_count, eos_hits, eos_count: eos_at.len() as u32, frames: nf as u32 };
        self.stats.set(stats);
        *self.forward.borrow_mut() = Some(Forward { talker_hidden, cp_hidden, cp_raw: raw, talker_count: tl.count, cp_count });
        stats.loss(s.mtp_weight)
    }

    fn backward_all(&self) {
        let s = &self.shape;
        let (d, dc, g) = (s.d(), s.mtp.d_model as usize, s.mtp.num_code_groups as usize);
        let (bt, fg) = ((s.batch * s.block) as usize, (s.frames * s.mtp.num_code_groups) as usize);
        let loaded = self.loaded.borrow();
        let l = loaded.as_ref().expect("speech sft: load a batch before the backward");
        let fwd = self.forward.borrow();
        let fw = fwd.as_ref().expect("speech sft: the backward follows a forward");
        let aux = &self.aux;
        let nf = l.frames.len();

        // Code predictor: every head's gradient into its position's rows, then the decoder.
        let heads_range = gpu_core::markers::range(c"speech_sft.cp_heads_bwd");
        let mut d_cp = vec![0.0f32; fg * dc];
        if fw.cp_count > 0 {
            let scale = s.mtp_weight / fw.cp_count as f32;
            for (i, head) in aux.cp_heads.iter().enumerate() {
                let k = i + 1;
                aux.gpu.write_f32(&aux.cp_h, &Self::gather(&fw.cp_hidden, dc, (0..nf).map(|f| f * g + k)));
                let labels: Vec<u32> = (0..nf).map(|f| Self::code(l, f, k, g)).collect();
                head.backward(&aux.ops, &aux.cp_h, &aux.cp_dh, nf as u32, &labels, None, scale, 1.0, Some(aux.ps.g(&residual_head(i))));
                let dh = aux.gpu.read(&aux.cp_dh, nf * dc);
                for f in 0..nf {
                    d_cp[(f * g + k) * dc..(f * g + k + 1) * dc].copy_from_slice(&dh[f * dc..(f + 1) * dc]);
                }
            }
        }
        self.cp.backward_hidden(&d_cp);
        drop(heads_range);
        let cp_range = gpu_core::markers::range(c"speech_sft.cp_bwd");
        let d_cp_in = self.cp.read_d_img_embeds();
        drop(cp_range);
        let proj_range = gpu_core::markers::range(c"speech_sft.cp_proj_bwd");
        let d_raw = match &aux.projector {
            Some(proj) => {
                aux.gpu.write_f32(&aux.proj_in, &fw.cp_raw);
                aux.gpu.write_f32(&aux.proj_d_out, &d_cp_in);
                aux.gpu.submit(&[], &proj.backward_rows(&aux.gpu, &[&aux.proj_in], &aux.proj_d_out, Some(&[&aux.proj_d_in]), fg as u32));
                aux.gpu.read(&aux.proj_d_in, fg * d)
            }
            None => d_cp_in,
        };

        // Talker: the head's gradient plus the code predictor's at position 0.
        drop(proj_range);
        let head_range = gpu_core::markers::range(c"speech_sft.talker_head_bwd");
        let mut d_hidden = vec![0.0f32; bt * d];
        if fw.talker_count > 0 {
            let n = l.talker_rows.len();
            aux.gpu.write_f32(&aux.talker_h, &Self::gather(&fw.talker_hidden, d, l.talker_rows.iter().copied()));
            aux.talker_head.backward(&aux.ops, &aux.talker_h, &aux.talker_dh, n as u32, &l.talker_labels, None, 1.0 / fw.talker_count as f32, 1.0, Some(aux.ps.g(TALKER_HEAD)));
            let dh = aux.gpu.read(&aux.talker_dh, n * d);
            for (i, &r) in l.talker_rows.iter().enumerate() {
                d_hidden[r * d..(r + 1) * d].copy_from_slice(&dh[i * d..(i + 1) * d]);
            }
        }
        for (f, &(_, _, trow)) in l.frames.iter().enumerate() {
            add(&mut d_hidden[trow * d..(trow + 1) * d], &d_raw[f * g * d..(f * g + 1) * d]);
        }
        self.talker.backward_hidden(&d_hidden);
        drop(head_range);
        let talker_range = gpu_core::markers::range(c"speech_sft.talker_bwd");
        let d_rows = self.talker.read_d_img_embeds();
        drop(talker_range);
        let _route = gpu_core::markers::range(c"speech_sft.route");

        // Every codec row a Talker input row summed; the text path and the x-vector are frozen.
        let mut grads = self.grads.borrow_mut();
        for (slot, ex) in l.examples.iter().enumerate() {
            let base = slot * s.block as usize;
            for (r, parts) in ex.plan.rows.iter().enumerate() {
                let dr = &d_rows[(base + r) * d..(base + r + 1) * d];
                for part in parts {
                    if let Part::Codec(id) = *part {
                        grads.add_codec(id, dr);
                    }
                }
            }
            let prefix = ex.plan.len();
            for t in 0..ex.codes.len() / g {
                let dr = &d_rows[(base + prefix + t) * d..(base + prefix + t + 1) * d];
                let frame = &ex.codes[t * g..(t + 1) * g];
                grads.add_codec(frame[0], dr);
                for (k, &code) in frame.iter().enumerate().skip(1) {
                    grads.add_residual(k - 1, code, dr);
                }
            }
        }
        // The code predictor's input rows: position 1 is the Talker's cb0 row, position p its own table p - 2.
        for f in 0..nf {
            let at = f * g * d;
            grads.add_codec(Self::code(l, f, 0, g), &d_raw[at + d..at + 2 * d]);
            for p in 2..g {
                grads.add_residual(p - 2, Self::code(l, f, p - 1, g), &d_raw[at + p * d..at + (p + 1) * d]);
            }
        }
    }

    fn device_sum_sq(norm: &DeviceGradNorm, m: &Qwen) -> f64 {
        let dg = model::Model::device_grads(m).unwrap_or_else(|e| panic!("speech sft: {e}"));
        let items: Vec<NormItem<'_>> = dg.tensors.iter().map(|(buf, numel, _)| NormItem { buf, numel: *numel as usize, dtype: GradDtype::F32 }).collect();
        norm.sum_sq(&dg.gpu, &items)
    }
}

fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

impl model::Model for SpeechSftModel {
    type Config = SftShape;

    fn new(cfg: SftShape, _b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        SpeechSftModel::build(cfg, init.clone(), data::rng::random_seed()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fresh weights for every tensor a build reads, upstream-named: the
    /// decoders as `qwen3` initialises them, the tables, heads, projection and
    /// text path drawn small.
    fn init_weights(cfg: &SftShape, seed: u64) -> HashMap<String, Vec<f32>> {
        let mut out = HashMap::new();
        for (q, to_hf) in [(cfg.talker_qwen(), talker_brain_to_hf as fn(&str) -> Option<String>), (cfg.cp_qwen(), mtp_brain_to_hf)] {
            for (n, v) in qwen3::init_weights(&q, seed) {
                if let Some(hf) = to_hf(&n).filter(|_| n != "tok.weight" && n != "lm_head.weight") {
                    out.insert(hf, v);
                }
            }
        }
        let mut rng = data::rng::Lcg::new(seed ^ 0x5eec_45f7);
        for (n, size) in cfg.aux_params().into_iter().chain(cfg.table_params()).chain(cfg.frozen_params()) {
            let scale = if n.ends_with(".bias") { 0.01 } else { 0.1 };
            out.insert(n, rng.vec_scaled(size, scale));
        }
        out
    }

    fn config(&self) -> &SftShape {
        &self.shape
    }

    fn set_batch(&self, _batch: model::Batch) {
        panic!("speech sft: a batch is a set of utterances; load it with SpeechSftModel::load");
    }

    fn forward(&self) -> f32 {
        self.forward_loss()
    }

    fn backward(&self) {
        self.backward_all();
    }

    fn zero_grads(&self) {
        self.talker.zero_grads();
        self.cp.zero_grads();
        self.aux.ps.zero_grads(&self.aux.gpu);
        self.grads.borrow_mut().clear();
    }

    /// One update of every group with one global gradient norm: the decoders'
    /// and the heads' measured on their devices, the tables' on the host, and
    /// the multiplier it gives (`extra_scale`, clipped) applied to all.
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        let _opt = gpu_core::markers::range(c"speech_sft.optimizer");
        let aux_items: Vec<NormItem<'_>> = self.aux.ps.opt_params().iter().map(|(n, numel)| NormItem { buf: self.aux.ps.g(n), numel: *numel, dtype: GradDtype::F32 }).collect();
        let sum_sq = Self::device_sum_sq(&self.talker_norm, &self.talker) + Self::device_sum_sq(&self.cp_norm, &self.cp) + self.aux.norm.sum_sq(&self.aux.gpu, &aux_items) + self.grads.borrow().sum_sq();
        self.last_norm.set(Some((sum_sq.sqrt() as f32) * extra_scale));
        let mult = model::grad_multiplier(sum_sq, clip, extra_scale);
        let m = self.shape.lr;
        self.talker.adamw_step(t, lr * m.talker, wd, adam, None, mult);
        self.cp.adamw_step(t, lr * m.code_predictor, wd, adam, None, mult);
        self.aux.opt.step(&self.aux.gpu, &self.aux.ps, t, lr, wd, adam, None, mult);
        let mut opt = self.host_opt.borrow_mut();
        let args = model::resident_ft::StepArgs { step: t, lr, wd, adam, clip: None, extra_scale: mult };
        let mult = opt.begin_step(&args, 0.0);
        let mut tables = self.tables.borrow_mut();
        let grads = self.grads.borrow();
        for (ti, &(table, at, n, new)) in self.segments.iter().enumerate() {
            let (value, grad) = match table {
                0 => (&mut tables.codec[at..at + n], &grads.codec[at..at + n]),
                i => (&mut tables.residual[i - 1][at..at + n], &grads.residual[i - 1][at..at + n]),
            };
            let group = if new { m.new_rows } else { m.codec_tables };
            opt.update_tensor(ti, value, grad, lr * group, mult).unwrap_or_else(|e| panic!("speech sft: host update: {e}"));
        }
    }

    fn last_grad_norm(&self) -> Option<f32> {
        self.last_norm.get()
    }

    fn poll_wait(&self) {
        self.talker.poll_wait();
        self.cp.poll_wait();
        self.aux.gpu.poll_wait();
    }

    fn pin_rounding_seed(&self, seed: u64) {
        model::Model::pin_rounding_seed(&self.talker, seed);
        model::Model::pin_rounding_seed(&self.cp, seed);
        self.aux.opt.set_rounding_seed(seed ^ 0x5eed_0000_1234_9876);
        self.host_opt.borrow_mut().pin_seed(seed);
    }

    /// Each decoder rebuilds its layers from their inputs in the backward.
    fn set_saved_policy(&mut self, policy: model::saved::SavedPolicy) -> Result<(), String> {
        model::Model::set_saved_policy(&mut self.talker, policy)?;
        model::Model::set_saved_policy(&mut self.cp, policy)
    }

    fn optimized_params(&self) -> Option<Vec<String>> {
        Some(self.param_names())
    }

    fn read_moments(&self, name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
        match self.shape.place(name)? {
            Place::Talker(b) => model::Model::read_moments(&self.talker, &b),
            Place::CodePredictor(b) => model::Model::read_moments(&self.cp, &b),
            Place::Aux => optim::read_moments(&self.aux.gpu, &self.aux.ps, name),
            Place::Table(i) => {
                let mut opt = self.host_opt.borrow_mut();
                let (mut m, mut v) = (Vec::new(), Vec::new());
                for (ti, _) in self.segments.iter().enumerate().filter(|(_, s)| s.0 == i) {
                    let (sm, sv) = opt.read_moments(ti).ok()?;
                    m.extend(sm);
                    v.extend(sv);
                }
                Some((m, v))
            }
        }
    }

    fn write_moments(&self, name: &str, m: &[f32], v: &[f32]) -> Result<(), String> {
        match self.shape.place(name).ok_or_else(|| format!("speech sft: no parameter {name}"))? {
            Place::Talker(b) => model::Model::write_moments(&self.talker, &b, m, v),
            Place::CodePredictor(b) => model::Model::write_moments(&self.cp, &b, m, v),
            Place::Aux => optim::write_moments(&self.aux.gpu, &self.aux.ps, name, m, v),
            Place::Table(i) => {
                let mut opt = self.host_opt.borrow_mut();
                let total: usize = self.segments.iter().filter(|s| s.0 == i).map(|s| s.2).sum();
                if m.len() != total || v.len() != total {
                    return Err(format!("speech sft: {name} holds {total} values, the moments {} and {}", m.len(), v.len()));
                }
                let mut at = 0;
                for (ti, seg) in self.segments.iter().enumerate().filter(|(_, s)| s.0 == i) {
                    opt.write_moments(ti, &m[at..at + seg.2], &v[at..at + seg.2]).map_err(|e| e.to_string())?;
                    at += seg.2;
                }
                Ok(())
            }
        }
    }

    fn param_names(&self) -> Vec<String> {
        model::ModelConfig::param_list(&self.shape).into_iter().map(|(n, _)| n).collect()
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        match self.shape.place(name).unwrap_or_else(|| panic!("speech sft: no parameter {name}")) {
            Place::Talker(b) => self.talker.read_weight(&b),
            Place::CodePredictor(b) => self.cp.read_weight(&b),
            Place::Aux => self.aux.ps.read_weight(&self.aux.gpu, name),
            Place::Table(0) => self.tables.borrow().codec.clone(),
            Place::Table(i) => self.tables.borrow().residual[i - 1].clone(),
        }
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        match self.shape.place(name).unwrap_or_else(|| panic!("speech sft: no parameter {name}")) {
            Place::Talker(b) => self.talker.write_weight(&b, data),
            Place::CodePredictor(b) => self.cp.write_weight(&b, data),
            Place::Aux => self.aux.gpu.write_f32(self.aux.ps.w(name), data),
            Place::Table(0) => self.tables.borrow_mut().codec.copy_from_slice(data),
            Place::Table(i) => self.tables.borrow_mut().residual[i - 1].copy_from_slice(data),
        }
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        match self.shape.place(name).unwrap_or_else(|| panic!("speech sft: no parameter {name}")) {
            Place::Talker(b) => self.talker.read_grad(&b),
            Place::CodePredictor(b) => self.cp.read_grad(&b),
            Place::Aux => self.aux.ps.read_grad(&self.aux.gpu, name),
            Place::Table(0) => self.grads.borrow().codec.clone(),
            Place::Table(i) => self.grads.borrow().residual[i - 1].clone(),
        }
    }

    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }

    /// Every trained tensor under its upstream name, as one safetensors file.
    fn save(&self, path: &str) {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self.param_names().into_iter().map(|n| {
            let w = self.read_weight(&n);
            (n, vec![w.len() as u64], w)
        }).collect();
        checkpoint::st::save_safetensors(path, &tensors, &model::ModelConfig::to_json(&self.shape), None).unwrap_or_else(|e| panic!("speech sft: save {path}: {e}"));
    }

    fn config_json(&self) -> serde_json::Value {
        model::ModelConfig::to_json(&self.shape)
    }
}

impl SpeechSftModel {
    /// [`model::Model::forward`] without differentiating: the loss of the loaded batch.
    pub fn score(&self) -> ForwardStats {
        self.forward_loss();
        self.stats.get()
    }
}
