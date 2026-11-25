// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The Talker and its code predictor as ONE trainable [`model::Model`],
//! teacher forced on recorded speech, every row and every gradient on the
//! device.
//!
//! Where every parameter lives, and who steps it:
//!
//! | parameters | where | optimizer |
//! |---|---|---|
//! | Talker decoder layers, final norm and `codec_head` (its own head) | a [`qwen3::Qwen`] fed input rows | its `adamw_step` |
//! | code-predictor decoder layers, norm and the 15 residual heads (its head, grouped by position) | a second [`qwen3::Qwen`] on the Talker's device (batch = frames, block = one frame's codebooks) | its `adamw_step` |
//! | Talker `codec_embedding` and the 15 residual `codec_embedding`s (one table), the new rows, `small_to_mtp_projection` | a parameter store on the same card | `optim::Optim` |
//! | text embedding and text projection | host, frozen in this release; each token's projected row is uploaded once | - |
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
//! **The rows** are assembled on the device: every Talker input row is a sum of
//! table rows (`rows_gather_sum3` over the trained tables, the new rows and the
//! frozen text and x-vector rows, in the order generation sums them), every code
//! predictor row one table row or the Talker's hidden state of its frame, copied
//! on the device. The host builds only the index plans.
//!
//! **The gradient routes**, all of them, on the device: the code predictor's
//! input gradient at position 0 joins the Talker's hidden-state gradient at
//! that Talker row ([`qwen3::Qwen::enable_hidden_grad`]), position 1 lands on
//! the Talker table's row of `cb0`, positions `2..` on the residual table that
//! embedded them; the Talker's input-row gradient lands on every codec row its
//! row summed (the frame's codes, the prompt's special, language and voice
//! rows), never on the frozen text path or the x-vector. Each table's gradient
//! is the transpose of its gather (`rows_gather_weighted_bwd` over a host-built
//! CSR): no atomics, a fixed summation order, so a step is reproducible.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use gpu_core::provider::library::GemmPrecision;
use gpu_core::{DeviceBuffer, Gpu, Step};
use model::head_ce::{HeadCe, HeadCeConfig, HeadGroups, HeadKernels, IGNORE};
use model::ops::{Ops, Weight};
use optim::gradnorm::{DeviceGradNorm, NormItem};
use optim::Optim;
use paramstore::precision::GradDtype;
use paramstore::ParamStore;
use qwen3::{Qwen, QwenConfig};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use super::tables::HostTables;
use crate::config::{MtpConfig, TalkerConfig};
use crate::import::{mtp_brain_to_hf, mtp_hf_to_brain, talker_brain_to_hf, talker_hf_to_brain};
use crate::prompt::{Part, PromptPlan, TtsSpecials};
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

/// The store's name of the trained codec tables, one after another: the
/// Talker's `codec_embedding`, then residual table 0, 1, ...
const TABLES: &str = "speech_sft.codec_tables";
/// The store's name of the rows a run allocates for new languages and voices.
const NEW_ROWS: &str = "speech_sft.new_rows";

/// A gather tap that adds nothing (`rows_gather_sum3`).
const NO_TAP: u32 = u32::MAX;
/// The table a tap names, in its top two bits: the trained tables, the new rows, the frozen text and x-vector rows.
const TAG_MASK: u32 = 3 << 30;
const TAG_TABLES: u32 = 0;
const TAG_NEW: u32 = 1 << 30;
const TAG_FROZEN: u32 = 2 << 30;

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
    /// The arithmetic of every GEMM and attention of the decoders and heads: tensor cores where the device has them
    /// ([`GemmPrecision::Auto`], TF32 products of the fp32 weights), or the exact fp32 kernels a derivative check
    /// states its claim against. A build input, not model data: never saved with the configuration.
    pub gemm: GemmPrecision,
}

impl SftShape {
    /// The Talker width: of its rows, its codec tables and its x-vectors.
    pub fn d(&self) -> usize {
        self.talker.d_model as usize
    }

    /// The Talker decoder: its own head is `codec_head`; its token table is never read (every row is an input row).
    fn talker_qwen(&self) -> QwenConfig {
        QwenConfig { tie_embeddings: false, ..self.talker.to_qwen(self.block) }.with_defaults()
    }

    /// The code predictor: its head is the residual heads stacked, one group of `codebook` columns per position.
    fn cp_qwen(&self) -> QwenConfig {
        QwenConfig { vocab: self.residuals() as u32 * self.mtp.vocab, tie_embeddings: false, ..self.mtp.to_qwen(self.mtp.num_code_groups) }.with_defaults()
    }

    fn residuals(&self) -> usize {
        self.mtp.n_residual() as usize
    }

    fn projected(&self) -> bool {
        self.mtp.embedding_dim != self.mtp.d_model
    }

    /// The code predictor's batch: whole groups of 64 sequences (its grouped head's binding windows).
    fn cp_batch(&self) -> u32 {
        self.frames.next_multiple_of(64)
    }

    /// Rows of the trained codec tables, one after another.
    fn table_rows(&self) -> usize {
        self.talker.vocab as usize + self.residuals() * self.mtp.vocab as usize
    }

    /// The first row of residual table `i` in the trained tables.
    fn residual_row0(&self, i: usize) -> usize {
        self.talker.vocab as usize + i * self.mtp.vocab as usize
    }

    /// The heads and the projection, under their upstream names.
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

    /// The trained codec tables, under their upstream names.
    fn table_params(&self) -> Vec<(String, usize)> {
        let d = self.d();
        let mut out = vec![(TALKER_TABLE.to_string(), self.talker.vocab as usize * d)];
        out.extend((0..self.residuals()).map(|i| (residual_table(i), self.mtp.vocab as usize * d)));
        out
    }

    /// The decoders' layer parameters, upstream-named, with the decoder they belong to (the heads are in
    /// [`Self::aux_params`]).
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
        if name == TALKER_HEAD {
            return Some(Place::Talker("lm_head.weight".into()));
        }
        if let Some(i) = (0..self.residuals()).find(|&i| residual_table(i) == name) {
            return Some(Place::Table(i + 1));
        }
        if let Some(i) = (0..self.residuals()).find(|&i| residual_head(i) == name) {
            return Some(Place::CpHead(i));
        }
        if self.projected() && PROJECTION.contains(&name) {
            return Some(Place::Projection);
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
            gemm: GemmPrecision::default(),
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

/// Where a trained parameter lives: a decoder (under its brain name), a residual
/// head (rows of the code predictor's stacked head), the projection, or codec
/// table `i` (0 = the Talker's) of the device tables.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Place {
    Talker(String),
    CodePredictor(String),
    CpHead(usize),
    Projection,
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

/// What a loaded batch supervises, fixed when it is loaded (its plans are on the device by then).
struct Loaded {
    talker_count: u32,
    /// The rows the Talker and the code predictor compute (`sequences * run`).
    talker_rows: u32,
    cp_rows: u32,
    /// Frames of the batch.
    frames: u32,
    /// End-of-speech labels, one per complete utterance.
    eos: u32,
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

/// A gather's transpose for one trained table, uploaded per batch: for table row
/// `i`, the edges `ptr[i] .. ptr[i + 1]` name the gathered rows that read it.
struct Csr {
    rows: u32,
    ptr: DeviceBuffer,
    edge_row: DeviceBuffer,
}

impl Csr {
    fn new(gpu: &Gpu, rows: usize, edges: usize) -> Csr {
        Csr { rows: rows as u32, ptr: gpu.storage(rows as u64 + 1), edge_row: gpu.storage(edges.max(1) as u64) }
    }

    /// Upload the transpose of `(gathered row, table row)` reads, listed in gathered-row order: each table row's
    /// edges keep that order, so its gradient is summed in a fixed one.
    fn upload(&self, gpu: &Gpu, reads: &[(u32, u32)]) {
        let mut ptr = vec![0u32; self.rows as usize + 1];
        for &(_, t) in reads {
            ptr[t as usize + 1] += 1;
        }
        for i in 0..self.rows as usize {
            ptr[i + 1] += ptr[i];
        }
        let mut next = ptr.clone();
        let mut edge_row = vec![0u32; reads.len()];
        for &(r, t) in reads {
            edge_row[next[t as usize] as usize] = r;
            next[t as usize] += 1;
        }
        gpu.write(&self.ptr, &ptr);
        if !edge_row.is_empty() {
            gpu.write(&self.edge_row, &edge_row);
        }
    }

    /// `dtable += transpose * dout` for `width`-wide rows.
    fn grad_step(&self, gpu: &Gpu, kernel: usize, ones: &DeviceBuffer, dout: &DeviceBuffer, dtable: &DeviceBuffer, width: u32) -> Step {
        gpu.step(kernel, &[&self.ptr, &self.edge_row, ones, dout, dtable], &[self.rows, width], self.rows * width)
    }
}

/// The aux device's kernels, by slot.
struct AuxKernels {
    gather: usize,
    gather_bwd: usize,
    copy: usize,
    bias_add: usize,
    bias_grad: usize,
}

/// The tables, the projection and their optimizer, with the per-batch plans that feed and drain the decoders.
struct Aux {
    gpu: Gpu,
    ops: Ops,
    ps: ParamStore,
    opt: Optim,
    norm: DeviceGradNorm,
    k: AuxKernels,
    /// The frozen rows: every text token's projected row at its id, then one x-vector per batch slot.
    frozen: DeviceBuffer,
    /// Which token rows of [`Self::frozen`] are uploaded.
    text_ready: RefCell<Vec<bool>>,
    text: TextProjection,
    /// The new rows by Talker table row: their row in [`NEW_ROWS`].
    new_slot: HashMap<u32, u32>,
    /// A one-word stand-in for an absent new-row table.
    none: DeviceBuffer,
    /// The Talker's gather plan ([`frame_taps`] taps a row) and the code predictor's (one tap a row).
    talker_idx: DeviceBuffer,
    cp_idx: DeviceBuffer,
    /// Transposes: Talker rows into the tables and into the new rows, code-predictor rows into the tables.
    talker_tables: Csr,
    talker_new: Option<Csr>,
    cp_tables: Csr,
    /// Unit edge weights for [`Csr::grad_step`].
    ones: DeviceBuffer,
    /// The code predictor's rows before the projection and their gradient (the decoder's own buffers when unprojected).
    raw: Option<(DeviceBuffer, DeviceBuffer)>,
    /// `(talker row, code-predictor row)` of every frame's hidden state, and the same reversed for its gradient.
    hidden_plan: DeviceBuffer,
    hidden_back: DeviceBuffer,
    /// The code predictor's gradient of the Talker's hidden states ([`Qwen::enable_hidden_grad`]).
    hidden_grad: DeviceBuffer,
    /// The end-of-speech rows' hidden states, gathered for the greedy pick, and their plan.
    eos_plan: DeviceBuffer,
    eos_h: DeviceBuffer,
    eos_head: HeadCe,
    eos_labels: RefCell<Vec<u32>>,
}

/// The kernels the device of the tables and the projection dispatches.
pub(crate) fn aux_kernels() -> Vec<(&'static str, &'static str)> {
    let mut list = model::ops::training_kernel_list().to_vec();
    model::ops::extend_kernel_list(&mut list, model::head_ce::kernels());
    model::ops::extend_kernel_list(
        &mut list,
        &[
            ("rows_gather_sum3", kernels::ROWS_GATHER_SUM3),
            ("rows_gather_weighted_bwd", kernels::ROWS_GATHER_WEIGHTED_BWD),
            ("anchor_rows_copy", kernels::ANCHOR_ROWS_COPY),
            ("bias_add", kernels::BIAS_ADD),
            ("bias_grad", kernels::BIAS_GRAD),
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

/// The composite trainable model. See the module documentation.
pub struct SpeechSftModel {
    shape: SftShape,
    talker: Qwen,
    cp: Qwen,
    talker_norm: DeviceGradNorm,
    cp_norm: DeviceGradNorm,
    aux: Aux,
    loaded: RefCell<Option<Loaded>>,
    stats: Cell<ForwardStats>,
    last_norm: Cell<Option<f32>>,
    /// The sum of squares of every accumulated gradient, once measured for the current gradients (the update gate
    /// asks for their finiteness, the update for their norm: one reduction serves both).
    grad_sum_sq: Cell<Option<f64>>,
}

/// Move tensor `name` (of `numel` values) out of `init`.
fn take(init: &mut HashMap<String, Vec<f32>>, name: &str, numel: usize) -> Result<Vec<f32>, String> {
    let v = init.remove(name).ok_or_else(|| format!("speech sft: no tensor {name}"))?;
    if v.len() != numel {
        return Err(format!("speech sft: {name} holds {} values, the configuration needs {numel}", v.len()));
    }
    Ok(v)
}

/// The frozen text path of `shape`, moved out of `init`.
fn text_projection(shape: &SftShape, init: &mut HashMap<String, Vec<f32>>) -> Result<TextProjection, String> {
    let (th, inter) = (shape.talker.text_hidden_size as usize, shape.text_inter as usize);
    let frozen = shape.frozen_params();
    let mut get = |i: usize| take(init, &frozen[i].0, frozen[i].1);
    Ok(TextProjection { text_embedding: Some(get(0)?), fc1_w: get(1)?, fc1_b: get(2)?, fc2_w: get(3)?, fc2_b: get(4)?, in_dim: th, inter, out: shape.d(), text_vocab: shape.talker.text_vocab_size as usize })
}

/// The host tables and the frozen text path of `shape`, moved out of `init`: what generation reads, for checking
/// that training rows are generation's rows.
#[cfg(test)]
pub(crate) fn host_tables(shape: &SftShape, init: &mut HashMap<String, Vec<f32>>) -> Result<HostTables, String> {
    let text = text_projection(shape, init)?;
    let tparams = shape.table_params();
    let codec = take(init, &tparams[0].0, tparams[0].1)?;
    let residual = tparams[1..].iter().map(|(n, s)| take(init, n, *s)).collect::<Result<Vec<_>, _>>()?;
    Ok(HostTables::new(shape.d(), text, codec, residual))
}

/// The taps of a Talker frame row: its residual codebooks, its first codebook and its text, in the order generation
/// sums them (`train::policy_rows`).
const fn frame_taps(groups: usize) -> usize {
    groups + 1
}

/// The text tokens a plan's rows read.
fn plan_text(plan: &PromptPlan) -> impl Iterator<Item = u32> + Clone + '_ {
    plan.rows.iter().flatten().filter_map(|p| if let Part::Text(id) = p { Some(*id) } else { None }).chain(plan.trailing.iter().copied()).chain([plan.tts_pad])
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
        if let Some(&r) = shape.new_rows.iter().find(|&&r| r >= shape.talker.vocab) {
            return Err(format!("speech sft: new row {r} is past the Talker table's {} rows", shape.talker.vocab));
        }
        let (dc, cv) = (shape.mtp.d_model as usize, shape.mtp.vocab as usize);
        // The heads join their decoders: the Talker's is its own head, the residual heads the code predictor's stacked one.
        let talker_head = take(&mut init, TALKER_HEAD, shape.talker.vocab as usize * d)?;
        let mut cp_head = Vec::with_capacity(shape.residuals() * cv * dc);
        for i in 0..shape.residuals() {
            cp_head.extend(take(&mut init, &residual_head(i), cv * dc)?);
        }
        let mut heads = [Some(talker_head), Some(cp_head)];
        let mut decoder = |cfg: &QwenConfig, which: Decoder, parent: Option<&Gpu>| -> Result<Qwen, String> {
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
            weights.insert("lm_head.weight".into(), heads[which as usize].take().expect("each head is placed once"));
            let (b, t) = match which {
                Decoder::Talker => (shape.batch, shape.block),
                Decoder::CodePredictor => (shape.cp_batch(), shape.mtp.num_code_groups),
            };
            // One device for the run: the code predictor is built on the Talker's handle.
            let mut q = Qwen::new_training_on(parent, cfg.clone(), b, t, &weights, shape.gemm, &qwen3::model::FullFtBuild::ambient())?;
            // Every row is an input row: the token table is never read and takes no gradient.
            q.freeze_params(|n| n == "tok.weight");
            q.enable_input_rows();
            q.enable_batch_trimming();
            Ok(q)
        };
        let mut talker = decoder(&shape.talker_qwen(), Decoder::Talker, None)?;
        let mut cp = decoder(&shape.cp_qwen(), Decoder::CodePredictor, Some(talker.gpu()))?;
        cp.set_head_groups(HeadGroups { period: shape.mtp.num_code_groups, offset: 1, width: shape.mtp.vocab, heads: shape.residuals() as u32, capacity: shape.cp_batch() })?;
        let aux = Self::build_aux(&shape, &talker, &mut init)?;
        talker.enable_hidden_grad(&aux.hidden_grad);
        let talker_norm = DeviceGradNorm::new(talker.gpu())?;
        let cp_norm = DeviceGradNorm::new(cp.gpu())?;
        Ok(SpeechSftModel { shape, talker, cp, talker_norm, cp_norm, aux, loaded: RefCell::new(None), stats: Cell::new(ForwardStats::default()), last_norm: Cell::new(None), grad_sum_sq: Cell::new(None) })
    }

    fn build_aux(shape: &SftShape, talker: &Qwen, init: &mut HashMap<String, Vec<f32>>) -> Result<Aux, String> {
        let gpu = talker.gpu().new_like(&aux_kernels());
        let mut ops = Ops::new(gpu.share())?;
        ops.set_gemm_precision(shape.gemm);
        let d = shape.d();
        let (dc, g) = (shape.mtp.d_model as usize, shape.mtp.num_code_groups as usize);

        // The trained tables as one, the new rows beside them (their own rate), the projection.
        let mut tables = Vec::with_capacity(shape.table_rows() * d);
        for (n, s) in shape.table_params() {
            tables.extend(take(init, &n, s)?);
        }
        let mut new_ids: Vec<u32> = shape.new_rows.clone();
        new_ids.sort_unstable();
        new_ids.dedup();
        let new_slot: HashMap<u32, u32> = new_ids.iter().enumerate().map(|(i, &r)| (r, i as u32)).collect();
        let new_rows: Vec<f32> = new_ids.iter().flat_map(|&r| tables[r as usize * d..(r as usize + 1) * d].to_vec()).collect();
        let mut params = vec![(TABLES.to_string(), tables.len())];
        let mut weights = HashMap::from([(TABLES.to_string(), tables)]);
        if !new_ids.is_empty() {
            params.push((NEW_ROWS.to_string(), new_rows.len()));
            weights.insert(NEW_ROWS.to_string(), new_rows);
        }
        if shape.projected() {
            for (n, s) in [(PROJECTION[0], dc * d), (PROJECTION[1], dc)] {
                params.push((n.to_string(), s));
                weights.insert(n.to_string(), take(init, n, s)?);
            }
        }
        let mut ps = ParamStore::new(&gpu, params, &weights);
        drop(weights);
        ps.set_lr_mult(TABLES, shape.lr.codec_tables);
        if !new_ids.is_empty() {
            ps.set_lr_mult(NEW_ROWS, shape.lr.new_rows);
        }
        if shape.projected() {
            for n in PROJECTION {
                ps.set_lr_mult(n, shape.lr.code_predictor);
            }
        }
        let k = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("speech sft: kernel {name} is not registered"));
        let opt = Optim::new(k("adamw")?, k("gradnorm_sq")?, k("grad_scale")?, k("clip_coef")?, k("grad_scale_buf")?);
        let norm = DeviceGradNorm::new(&gpu)?;
        let kernels = AuxKernels { gather: k("rows_gather_sum3")?, gather_bwd: k("rows_gather_weighted_bwd")?, copy: k("anchor_rows_copy")?, bias_add: k("bias_add")?, bias_grad: k("bias_grad")? };

        let text = text_projection(shape, init)?;
        let text_vocab = shape.talker.text_vocab_size as usize;
        let batch = shape.batch as usize;
        let frozen = gpu.storage(((text_vocab + batch) * d) as u64);

        let talker_cap = (shape.batch * shape.block) as usize;
        let cp_cap = shape.cp_batch() as usize * g;
        let st = |n: usize| gpu.storage(n.max(1) as u64);
        let talker_edges = talker_cap * frame_taps(g);
        let ones = gpu.storage_init("speech_sft.ones", &vec![1.0f32; talker_edges.max(cp_cap)]);
        let raw = shape.projected().then(|| (st(cp_cap * d), st(cp_cap * d)));
        let head = model::Model::optimizer_parts(talker).ok_or("speech sft: the Talker exposes no parameter store")?.store.w("lm_head.weight").clone();
        let head_w = Weight::dense_f32(head, shape.talker.vocab, d as u32);
        let eos_head = HeadCe::new(&ops, d as u32, shape.talker.vocab, &head_w, HeadCeConfig { rows_per_chunk: shape.batch.clamp(1, 256), tile_budget_words: None }, HeadKernels::from_gpu(&gpu)?)?;
        Ok(Aux {
            frozen,
            text_ready: RefCell::new(vec![false; text_vocab]),
            text,
            none: st(1),
            talker_idx: st(talker_edges),
            cp_idx: st(cp_cap),
            talker_tables: Csr::new(&gpu, shape.table_rows(), talker_edges),
            talker_new: (!new_ids.is_empty()).then(|| Csr::new(&gpu, new_ids.len(), talker_cap * 2)),
            cp_tables: Csr::new(&gpu, shape.table_rows(), cp_cap),
            new_slot,
            ones,
            raw,
            hidden_plan: st(shape.frames as usize * 2),
            hidden_back: st(shape.frames as usize * 2),
            hidden_grad: talker.gpu().storage((talker_cap * d) as u64),
            eos_plan: st(batch * 2),
            eos_h: st(batch * d),
            eos_head,
            eos_labels: RefCell::new(Vec::new()),
            k: kernels,
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

    /// The rows the loaded batch computes, Talker and code predictor together (each trimmed to what the batch holds).
    pub fn computed_rows(&self) -> u32 {
        self.loaded.borrow().as_ref().map_or(0, |l| l.talker_rows + l.cp_rows)
    }

    /// Talker rows a plan of `prefix` rows and `frames` frames takes.
    pub fn rows_needed(prefix: usize, frames: usize) -> usize {
        prefix + frames
    }

    /// The tap of a Talker table row: the new rows sit in their own table.
    fn codec_tap(&self, id: u32) -> u32 {
        match self.aux.new_slot.get(&id) {
            Some(&slot) => TAG_NEW | slot,
            None => TAG_TABLES | id,
        }
    }

    /// Upload the projected rows of the text tokens `ids` the device does not hold yet: each token is projected once
    /// per run (the text path is frozen), on the host, as generation projects it.
    fn upload_text(&self, ids: impl Iterator<Item = u32>) {
        let mut ready = self.aux.text_ready.borrow_mut();
        let mut fresh: Vec<u32> = ids.filter(|&id| !ready[id as usize]).collect();
        fresh.sort_unstable();
        fresh.dedup();
        if fresh.is_empty() {
            return;
        }
        let text = &self.aux.text;
        let rows = text.project(&text.embed_text(&fresh));
        let d = self.shape.d();
        for (i, &id) in fresh.iter().enumerate() {
            self.aux.gpu.write_f32_at(&self.aux.frozen, id as u64 * d as u64, &rows[i * d..(i + 1) * d]);
            ready[id as usize] = true;
        }
    }

    /// Load the next batch: at most [`SftShape::batch`] examples, each fitting the block. Every plan the forward and
    /// backward read is on the device when this returns.
    pub fn load(&self, examples: Vec<Example>) -> Result<(), String> {
        let s = &self.shape;
        let (g, d) = (s.mtp.num_code_groups as usize, s.d());
        let text_vocab = s.talker.text_vocab_size;
        if examples.len() > s.batch as usize {
            return Err(format!("speech sft: {} examples for a batch of {}", examples.len(), s.batch));
        }
        let mut frames = 0usize;
        for (slot, ex) in examples.iter().enumerate() {
            if ex.codes.is_empty() || !ex.codes.len().is_multiple_of(g) {
                return Err(format!("speech sft: example {slot}: {} codes are not whole frames of {g}", ex.codes.len()));
            }
            let (prefix, n) = (ex.plan.len(), ex.codes.len() / g);
            if Self::rows_needed(prefix, n) > s.block as usize {
                return Err(format!("speech sft: example {slot}: {prefix} prompt rows and {n} frames exceed the block of {}", s.block));
            }
            if ex.plan.rows.iter().flatten().any(|p| *p == Part::XVector) && ex.xvector.len() != d {
                return Err(format!("speech sft: example {slot}: an x-vector of {} values, the Talker is {} wide", ex.xvector.len(), d));
            }
            if let Some(bad) = plan_text(&ex.plan).find(|&id| id >= text_vocab) {
                return Err(format!("speech sft: example {slot}: text token {bad} is past the text vocabulary of {text_vocab}"));
            }
            if let Some((i, c)) = ex.codes.iter().enumerate().find(|&(i, &c)| c >= if i % g == 0 { s.talker.vocab } else { s.mtp.vocab }) {
                return Err(format!("speech sft: example {slot}: code {c} at {i} is past its codebook"));
            }
            if ex.plan.rows.iter().any(|parts| parts.len() > frame_taps(g)) {
                return Err(format!("speech sft: example {slot}: a prompt row sums more than {} parts", frame_taps(g)));
            }
            frames += n;
        }
        if frames > s.frames as usize {
            return Err(format!("speech sft: {frames} frames for a code-predictor capacity of {}", s.frames));
        }
        let aux = &self.aux;

        // The Talker's targets decide which rows it computes; its plans are laid out at that run.
        let (b, t) = (s.batch as usize, s.block as usize);
        let mut targets = vec![IGNORE; b * t];
        for (slot, ex) in examples.iter().enumerate() {
            let (base, prefix, n) = (slot * t, ex.plan.len(), ex.codes.len() / g);
            for f in 0..n {
                targets[base + prefix - 1 + f] = ex.codes[f * g];
            }
            if ex.complete {
                targets[base + prefix - 1 + n] = s.specials.codec_eos;
            }
        }
        let talker_count = targets.iter().filter(|&&y| y != IGNORE).count() as u32;
        self.talker.set_batch(&vec![0u32; b * t], &targets);
        let (run, seqs) = (self.talker.run_len() as usize, self.talker.batch_run() as usize);

        self.upload_text(examples.iter().flat_map(|ex| plan_text(&ex.plan)).collect::<Vec<_>>().into_iter());
        for (slot, ex) in examples.iter().enumerate() {
            if ex.plan.rows.iter().flatten().any(|p| *p == Part::XVector) {
                aux.gpu.write_f32_at(&aux.frozen, (u64::from(text_vocab) + slot as u64) * d as u64, &ex.xvector);
            }
        }

        // Talker rows: each prompt row its parts in order; each frame row its residual codes, its first code and its
        // text. Reads are listed row by row, so every table row's edges come in row order.
        let taps = frame_taps(g);
        let mut idx = vec![NO_TAP; seqs * run * taps];
        let mut table_reads: Vec<(u32, u32)> = Vec::new();
        let mut new_reads: Vec<(u32, u32)> = Vec::new();
        let mut hidden: Vec<(u32, u32)> = Vec::with_capacity(frames);
        let mut eos_rows: Vec<u32> = Vec::new();
        let mut read = |row: usize, tap: u32| match tap & TAG_MASK {
            TAG_TABLES => table_reads.push((row as u32, tap)),
            TAG_NEW => new_reads.push((row as u32, tap & !TAG_MASK)),
            _ => {}
        };
        for (slot, ex) in examples.iter().enumerate() {
            let base = slot * run;
            let prefix = ex.plan.len();
            for (r, parts) in ex.plan.rows.iter().enumerate() {
                for (k, part) in parts.iter().enumerate() {
                    let tap = match *part {
                        Part::Text(id) => TAG_FROZEN | id,
                        Part::Codec(id) => self.codec_tap(id),
                        Part::XVector => TAG_FROZEN | (text_vocab + slot as u32),
                    };
                    idx[(base + r) * taps + k] = tap;
                    read(base + r, tap);
                }
            }
            let n = ex.codes.len() / g;
            for f in 0..n {
                let row = base + prefix + f;
                let frame = &ex.codes[f * g..(f + 1) * g];
                let mut put = |k: usize, tap: u32| {
                    idx[row * taps + k] = tap;
                    read(row, tap);
                };
                for (k, &code) in frame.iter().enumerate().skip(1) {
                    put(k - 1, TAG_TABLES | (s.residual_row0(k - 1) as u32 + code));
                }
                put(g - 1, self.codec_tap(frame[0]));
                put(g, TAG_FROZEN | ex.plan.trailing.get(f).copied().unwrap_or(ex.plan.tts_pad));
                hidden.push(((base + prefix - 1 + f) as u32, (hidden.len() * g) as u32));
            }
            if ex.complete {
                eos_rows.push((base + prefix - 1 + n) as u32);
            }
        }
        aux.gpu.write(&aux.talker_idx, &idx);
        aux.talker_tables.upload(&aux.gpu, &table_reads);
        if let Some(csr) = &aux.talker_new {
            csr.upload(&aux.gpu, &new_reads);
        }

        // The code predictor: frame f's rows are f * g + p; position 0 is the Talker's hidden state, copied in.
        let cpb = s.cp_batch() as usize;
        let mut cp_targets = vec![IGNORE; cpb * g];
        for (f, frame) in examples.iter().flat_map(|ex| ex.codes.chunks(g)).enumerate() {
            cp_targets[f * g + 1..(f + 1) * g].copy_from_slice(&frame[1..]);
        }
        self.cp.set_batch(&vec![0u32; cpb * g], &cp_targets);
        let cp_seqs = self.cp.batch_run() as usize;
        assert_eq!(self.cp.run_len() as usize, g, "speech sft: every frame's last position is supervised, so the code predictor runs whole frames");
        let mut cp_idx = vec![NO_TAP; cp_seqs * g];
        let mut cp_reads: Vec<(u32, u32)> = Vec::with_capacity(frames * (g - 1));
        for (f, frame) in examples.iter().flat_map(|ex| ex.codes.chunks(g)).enumerate() {
            for p in 1..g {
                let row = if p == 1 { frame[0] } else { s.residual_row0(p - 2) as u32 + frame[p - 1] };
                cp_idx[f * g + p] = TAG_TABLES | row;
                cp_reads.push(((f * g + p) as u32, row));
            }
        }
        aux.gpu.write(&aux.cp_idx, &cp_idx);
        aux.cp_tables.upload(&aux.gpu, &cp_reads);
        let plan: Vec<u32> = hidden.iter().flat_map(|&(trow, crow)| [trow, crow]).collect();
        let back: Vec<u32> = hidden.iter().flat_map(|&(trow, crow)| [crow, trow]).collect();
        aux.gpu.write(&aux.hidden_plan, &plan);
        aux.gpu.write(&aux.hidden_back, &back);
        if !eos_rows.is_empty() {
            let pairs: Vec<u32> = eos_rows.iter().enumerate().flat_map(|(i, &r)| [r, i as u32]).collect();
            aux.gpu.write(&aux.eos_plan, &pairs);
        }
        *aux.eos_labels.borrow_mut() = vec![s.specials.codec_eos; eos_rows.len()];
        *self.loaded.borrow_mut() = Some(Loaded { talker_count, talker_rows: (seqs * run) as u32, cp_rows: (cp_seqs * g) as u32, frames: frames as u32, eos: eos_rows.len() as u32 });
        Ok(())
    }

    /// The gather of `rows` rows of `taps` taps over the three tables: trained, new rows (or a stand-in), frozen.
    fn gather_step(&self, idx: &DeviceBuffer, rows: u32, taps: u32, out: &DeviceBuffer) -> Step {
        let aux = &self.aux;
        let new = if aux.talker_new.is_some() { aux.ps.w(NEW_ROWS) } else { &aux.none };
        let d = self.shape.d() as u32;
        aux.gpu.step(aux.k.gather, &[idx, aux.ps.w(TABLES), new, &aux.frozen, out], &[rows, taps, d], rows * d)
    }

    /// The code predictor's rows before the projection and their gradient: the raw buffers, or the decoder's own
    /// input rows when there is no projection.
    fn cp_raw(&self) -> (&DeviceBuffer, &DeviceBuffer) {
        match &self.aux.raw {
            Some((raw, d_raw)) => (raw, d_raw),
            None => (self.cp.input_rows_buffer(), self.cp.input_rows_grad_buffer()),
        }
    }

    fn forward_loss(&self) -> f32 {
        let s = &self.shape;
        let (d, g) = (s.d() as u32, s.mtp.num_code_groups);
        let loaded = self.loaded.borrow();
        let l = loaded.as_ref().expect("speech sft: load a batch before the forward");
        let aux = &self.aux;

        let rows = gpu_core::markers::range(c"speech_sft.rows");
        aux.gpu.submit(&[], &[self.gather_step(&aux.talker_idx, l.talker_rows, frame_taps(g as usize) as u32, self.talker.input_rows_buffer())]);
        drop(rows);
        let talker_range = gpu_core::markers::range(c"speech_sft.talker_fwd");
        let talker_mean = self.talker.forward();
        drop(talker_range);

        let cp_rows_range = gpu_core::markers::range(c"speech_sft.cp_rows");
        let (raw, _) = self.cp_raw();
        let mut steps = vec![self.gather_step(&aux.cp_idx, l.cp_rows, 1, raw), aux.gpu.step(aux.k.copy, &[&aux.hidden_plan, self.talker.final_hidden_buffer(), raw], &[l.frames, d], l.frames * d)];
        if aux.raw.is_some() {
            let dc = s.mtp.d_model;
            let w = Weight::dense_f32(aux.ps.w(PROJECTION[0]).clone(), dc, d);
            let out = self.cp.input_rows_buffer();
            aux.ops.matmul(&mut steps, &w, &aux.ops.act_f32(raw, 0, l.cp_rows, d), out, 0);
            steps.push(aux.gpu.step(aux.k.bias_add, &[out, aux.ps.w(PROJECTION[1])], &[l.cp_rows, dc], l.cp_rows * dc));
        }
        aux.gpu.submit(&[], &steps);
        drop(cp_rows_range);
        let cp_range = gpu_core::markers::range(c"speech_sft.cp_fwd");
        let cp_count = l.frames * (g - 1);
        // The code predictor's gradient carries its weight in the loss: its mean is taken over `count / mtp_weight`.
        let weighted = s.mtp_weight > 0.0;
        if weighted {
            self.cp.set_loss_denominator(cp_count as f32 / s.mtp_weight);
        }
        let cp_out = self.cp.forward();
        let cp_mean = if weighted { cp_out / s.mtp_weight } else { cp_out };
        drop(cp_range);

        let eos_range = gpu_core::markers::range(c"speech_sft.eos");
        let eos_hits = if l.eos > 0 {
            aux.gpu.submit(&[], &[aux.gpu.step(aux.k.copy, &[&aux.eos_plan, self.talker.final_hidden_buffer(), &aux.eos_h], &[l.eos, d], l.eos * d)]);
            let labels = aux.eos_labels.borrow();
            let (_, picks) = aux.eos_head.forward_argmax(&aux.ops, &aux.eos_h, l.eos, &labels, 1.0).unwrap_or_else(|e| panic!("speech sft: {e}"));
            picks.iter().filter(|&&p| p == s.specials.codec_eos).count() as u32
        } else {
            0
        };
        drop(eos_range);
        let stats = ForwardStats {
            talker_sum: f64::from(talker_mean) * f64::from(l.talker_count),
            talker_count: l.talker_count,
            cp_sum: f64::from(cp_mean) * f64::from(cp_count),
            cp_count,
            eos_hits,
            eos_count: l.eos,
            frames: l.frames,
        };
        self.stats.set(stats);
        stats.loss(s.mtp_weight)
    }

    fn backward_all(&self) {
        let s = &self.shape;
        let (d, dc) = (s.d() as u32, s.mtp.d_model);
        let loaded = self.loaded.borrow();
        let l = loaded.as_ref().expect("speech sft: load a batch before the backward");
        let aux = &self.aux;

        // A code predictor weighted out of the loss sends no gradient anywhere.
        if s.mtp_weight > 0.0 {
            let cp_range = gpu_core::markers::range(c"speech_sft.cp_bwd");
            self.cp.backward();
            drop(cp_range);
            let _route = gpu_core::markers::range(c"speech_sft.cp_route");
            let (raw, d_raw) = self.cp_raw();
            let mut steps = Vec::new();
            if aux.raw.is_some() {
                let dy = self.cp.input_rows_grad_buffer();
                let w = Weight::dense_f32(aux.ps.w(PROJECTION[0]).clone(), dc, d);
                aux.ops.matmul_dw(&mut steps, raw, dy, l.cp_rows, dc, d, aux.ps.g(PROJECTION[0]));
                steps.push(aux.gpu.step(aux.k.bias_grad, &[dy, aux.ps.g(PROJECTION[1])], &[l.cp_rows, dc], dc));
                aux.ops.matmul_dx(&mut steps, &w, dy, l.cp_rows, d_raw, false);
            }
            // Position 0 back to the Talker's hidden state of its frame, positions 1.. into the table rows they read.
            steps.push(aux.gpu.step(aux.k.copy, &[&aux.hidden_back, d_raw, &aux.hidden_grad], &[l.frames, d], l.frames * d));
            steps.push(aux.cp_tables.grad_step(&aux.gpu, aux.k.gather_bwd, &aux.ones, d_raw, aux.ps.g(TABLES), d));
            aux.gpu.submit(&[], &steps);
        }
        let talker_range = gpu_core::markers::range(c"speech_sft.talker_bwd");
        self.talker.backward();
        drop(talker_range);
        let _route = gpu_core::markers::range(c"speech_sft.route");
        let d_rows = self.talker.input_rows_grad_buffer();
        let mut steps = vec![aux.talker_tables.grad_step(&aux.gpu, aux.k.gather_bwd, &aux.ones, d_rows, aux.ps.g(TABLES), d)];
        if let Some(csr) = &aux.talker_new {
            steps.push(csr.grad_step(&aux.gpu, aux.k.gather_bwd, &aux.ones, d_rows, aux.ps.g(NEW_ROWS), d));
        }
        aux.gpu.submit(&[], &steps);
    }

    /// The sum of squares of every accumulated gradient, measured on the device, once per set of gradients.
    fn grad_sum_sq(&self) -> f64 {
        if let Some(v) = self.grad_sum_sq.get() {
            return v;
        }
        let aux_items: Vec<NormItem<'_>> = self.aux.ps.opt_params().iter().map(|(n, numel)| NormItem { buf: self.aux.ps.g(n), numel: *numel, dtype: GradDtype::F32 }).collect();
        let v = Self::device_sum_sq(&self.talker_norm, &self.talker) + Self::device_sum_sq(&self.cp_norm, &self.cp) + self.aux.norm.sum_sq(&self.aux.gpu, &aux_items);
        self.grad_sum_sq.set(Some(v));
        v
    }

    fn device_sum_sq(norm: &DeviceGradNorm, m: &Qwen) -> f64 {
        let dg = model::Model::device_grads(m).unwrap_or_else(|e| panic!("speech sft: {e}"));
        let items: Vec<NormItem<'_>> = dg.tensors.iter().map(|(buf, numel, _)| NormItem { buf, numel: *numel as usize, dtype: GradDtype::F32 }).collect();
        norm.sum_sq(&dg.gpu, &items)
    }

    /// Codec table `i` (0 = the Talker's) within the trained tables: its first word and its length.
    fn table_span(&self, i: usize) -> (usize, usize) {
        let s = &self.shape;
        let d = s.d();
        match i {
            0 => (0, s.talker.vocab as usize * d),
            i => (s.residual_row0(i - 1) * d, s.mtp.vocab as usize * d),
        }
    }

    /// Codec table `i` read out of `all` (the trained tables, their gradient or a moment), the Talker's with its new
    /// rows taken from `new` (the same quantity of the new rows).
    fn compose_table(&self, i: usize, all: &[f32], new: Option<&[f32]>) -> Vec<f32> {
        let (at, n) = self.table_span(i);
        self.with_new_rows(i, all[at..at + n].to_vec(), new)
    }

    /// Codec table `i` read off the device out of `tables` (the trained tables' weights or gradient; only its own
    /// span is copied), the Talker's with its new rows read out of `new`.
    fn read_table(&self, i: usize, tables: &DeviceBuffer, new: Option<&DeviceBuffer>) -> Vec<f32> {
        let (at, n) = self.table_span(i);
        let mut out = vec![0.0; n];
        self.aux.gpu.read_into(tables, at as u64, &mut out);
        let new = new.filter(|_| i == 0).map(|b| self.aux.gpu.read(b, self.aux.new_slot.len() * self.shape.d()));
        self.with_new_rows(i, out, new.as_deref())
    }

    /// `out`, codec table `i`, with the new rows of `new` written over theirs when it is the Talker's.
    fn with_new_rows(&self, i: usize, mut out: Vec<f32>, new: Option<&[f32]>) -> Vec<f32> {
        if let (0, Some(new)) = (i, new) {
            let d = self.shape.d();
            for (&row, &slot) in &self.aux.new_slot {
                out[row as usize * d..(row as usize + 1) * d].copy_from_slice(&new[slot as usize * d..(slot as usize + 1) * d]);
            }
        }
        out
    }

    /// The new rows of Talker-table-shaped `data`, in their store's order.
    fn new_rows_of(&self, data: &[f32]) -> Vec<f32> {
        let d = self.shape.d();
        let mut out = vec![0.0; self.aux.new_slot.len() * d];
        for (&row, &slot) in &self.aux.new_slot {
            out[slot as usize * d..(slot as usize + 1) * d].copy_from_slice(&data[row as usize * d..(row as usize + 1) * d]);
        }
        out
    }

    /// Residual head `i` within the code predictor's stacked head, in words.
    fn head_span(&self, i: usize) -> std::ops::Range<usize> {
        let n = self.shape.mtp.vocab as usize * self.shape.mtp.d_model as usize;
        i * n..(i + 1) * n
    }

    /// The input rows the Talker read in the latest forward, `[sequences * run, d]`, and the run (tests).
    #[cfg(test)]
    pub(crate) fn talker_input_rows(&self) -> (Vec<f32>, usize) {
        let rows = self.loaded.borrow().as_ref().expect("a loaded batch").talker_rows as usize;
        (self.talker.gpu().read(self.talker.input_rows_buffer(), rows * self.shape.d()), self.talker.run_len() as usize)
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
        self.grad_sum_sq.set(None);
        self.backward_all();
    }

    fn zero_grads(&self) {
        self.grad_sum_sq.set(None);
        self.talker.zero_grads();
        self.cp.zero_grads();
        self.aux.ps.zero_grads(&self.aux.gpu);
    }

    /// One update of every group with one global gradient norm, measured on the
    /// device, and the multiplier it gives (`extra_scale`, clipped) applied to all.
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, adam: model::Adam, clip: Option<f32>, extra_scale: f32) {
        let _opt = gpu_core::markers::range(c"speech_sft.optimizer");
        let sum_sq = self.grad_sum_sq();
        self.grad_sum_sq.set(None);
        self.last_norm.set(Some((sum_sq.sqrt() as f32) * extra_scale));
        let mult = model::grad_multiplier(sum_sq, clip, extra_scale);
        let m = self.shape.lr;
        self.talker.adamw_step(t, lr * m.talker, wd, adam, None, mult);
        self.cp.adamw_step(t, lr * m.code_predictor, wd, adam, None, mult);
        self.aux.opt.step(&self.aux.gpu, &self.aux.ps, t, lr, wd, adam, None, mult);
    }

    /// Every gradient is finite exactly when their sum of squares is, measured on the device (a host scan reads
    /// back every gradient each step); the update reuses the measurement.
    fn gradients_finite(&self) -> Option<bool> {
        Some(self.grad_sum_sq().is_finite())
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
        let aux = &self.aux;
        match self.shape.place(name)? {
            Place::Talker(b) => model::Model::read_moments(&self.talker, &b),
            Place::CodePredictor(b) => model::Model::read_moments(&self.cp, &b),
            Place::CpHead(i) => {
                let (m, v) = model::Model::read_moments(&self.cp, "lm_head.weight")?;
                Some((m[self.head_span(i)].to_vec(), v[self.head_span(i)].to_vec()))
            }
            Place::Projection => optim::read_moments(&aux.gpu, &aux.ps, name),
            Place::Table(i) => {
                let (m, v) = optim::read_moments(&aux.gpu, &aux.ps, TABLES)?;
                let new = aux.talker_new.as_ref().and_then(|_| optim::read_moments(&aux.gpu, &aux.ps, NEW_ROWS));
                Some((self.compose_table(i, &m, new.as_ref().map(|n| n.0.as_slice())), self.compose_table(i, &v, new.as_ref().map(|n| n.1.as_slice()))))
            }
        }
    }

    fn write_moments(&self, name: &str, m: &[f32], v: &[f32]) -> Result<(), String> {
        let aux = &self.aux;
        match self.shape.place(name).ok_or_else(|| format!("speech sft: no parameter {name}"))? {
            Place::Talker(b) => model::Model::write_moments(&self.talker, &b, m, v),
            Place::CodePredictor(b) => model::Model::write_moments(&self.cp, &b, m, v),
            Place::CpHead(i) => {
                let span = self.head_span(i);
                if m.len() != span.len() || v.len() != span.len() {
                    return Err(format!("speech sft: {name} holds {} values, the moments {} and {}", span.len(), m.len(), v.len()));
                }
                let (mut am, mut av) = model::Model::read_moments(&self.cp, "lm_head.weight").ok_or("speech sft: the code predictor's head has no moments")?;
                am[span.clone()].copy_from_slice(m);
                av[span].copy_from_slice(v);
                model::Model::write_moments(&self.cp, "lm_head.weight", &am, &av)
            }
            Place::Projection => optim::write_moments(&aux.gpu, &aux.ps, name, m, v),
            Place::Table(i) => {
                let (at, n) = self.table_span(i);
                if m.len() != n || v.len() != n {
                    return Err(format!("speech sft: {name} holds {n} values, the moments {} and {}", m.len(), v.len()));
                }
                let (mut am, mut av) = optim::read_moments(&aux.gpu, &aux.ps, TABLES).ok_or("speech sft: the tables have no moments")?;
                am[at..at + n].copy_from_slice(m);
                av[at..at + n].copy_from_slice(v);
                optim::write_moments(&aux.gpu, &aux.ps, TABLES, &am, &av)?;
                if i == 0 && aux.talker_new.is_some() {
                    optim::write_moments(&aux.gpu, &aux.ps, NEW_ROWS, &self.new_rows_of(m), &self.new_rows_of(v))?;
                }
                Ok(())
            }
        }
    }

    fn param_names(&self) -> Vec<String> {
        model::ModelConfig::param_list(&self.shape).into_iter().map(|(n, _)| n).collect()
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        let aux = &self.aux;
        match self.shape.place(name).unwrap_or_else(|| panic!("speech sft: no parameter {name}")) {
            Place::Talker(b) => self.talker.read_weight(&b),
            Place::CodePredictor(b) => self.cp.read_weight(&b),
            Place::CpHead(i) => self.cp.read_weight("lm_head.weight")[self.head_span(i)].to_vec(),
            Place::Projection => aux.ps.read_weight(&aux.gpu, name),
            Place::Table(i) => self.read_table(i, aux.ps.w(TABLES), aux.talker_new.as_ref().map(|_| aux.ps.w(NEW_ROWS))),
        }
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        let aux = &self.aux;
        match self.shape.place(name).unwrap_or_else(|| panic!("speech sft: no parameter {name}")) {
            Place::Talker(b) => self.talker.write_weight(&b, data),
            Place::CodePredictor(b) => self.cp.write_weight(&b, data),
            Place::CpHead(i) => {
                let mut all = self.cp.read_weight("lm_head.weight");
                all[self.head_span(i)].copy_from_slice(data);
                self.cp.write_weight("lm_head.weight", &all);
            }
            Place::Projection => aux.gpu.write_f32(aux.ps.w(name), data),
            Place::Table(i) => {
                let (at, n) = self.table_span(i);
                assert_eq!(data.len(), n, "speech sft: {name} holds {n} values");
                aux.gpu.write_f32_at(aux.ps.w(TABLES), at as u64, data);
                if i == 0 && aux.talker_new.is_some() {
                    aux.gpu.write_f32(aux.ps.w(NEW_ROWS), &self.new_rows_of(data));
                }
            }
        }
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        let aux = &self.aux;
        match self.shape.place(name).unwrap_or_else(|| panic!("speech sft: no parameter {name}")) {
            Place::Talker(b) => self.talker.read_grad(&b),
            Place::CodePredictor(b) => self.cp.read_grad(&b),
            Place::CpHead(i) => self.cp.read_grad("lm_head.weight")[self.head_span(i)].to_vec(),
            Place::Projection => aux.ps.read_grad(&aux.gpu, name),
            Place::Table(i) => self.read_table(i, aux.ps.g(TABLES), aux.talker_new.as_ref().map(|_| aux.ps.g(NEW_ROWS))),
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
