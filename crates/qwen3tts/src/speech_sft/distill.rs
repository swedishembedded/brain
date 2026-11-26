// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the distillation of large speech synthesizers
// into small ones that keep their voices for its clients. If your team needs
// expertise in compressing text-to-speech models, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Distillation of a speech fine-tune from a frozen teacher: the student
//! (a checkpoint of the same upstream layout, possibly narrower, shallower or
//! with a trimmed text vocabulary) learns the teacher's distributions on the
//! same batch, teacher forced on identical histories.
//!
//! **The terms**, per micro-batch, beside the ground-truth objective
//! ([`super::model`]):
//!
//! * `talker_kd_weight * T² * mean_r KL(q_r || p_r)` over the Talker's
//!   supervised rows (every frame's first codebook and the end of speech), `p`
//!   and `q` the student's and the teacher's codebook-0 distributions at
//!   temperature `T`;
//! * `mtp_kd_weight * Σ_k T² * mean_f KL(q_{f,k} || p_{f,k})` over the code
//!   predictor's residual codebooks `k = 1 ..`: a sum over the codebooks of each
//!   one's mean over the frames;
//! * `hidden_kd_weight * Σ_(s, t) mean_r mean_c (P h^s_r - h^t_r)²` for each
//!   mapped pair of a student layer `s` and a teacher layer `t` (the residual
//!   stream entering the layer; the layer count names the stream after the last
//!   layer), over the same rows, `P` a learned `[d_teacher, d_student]`
//!   projection that trains with the run and is never exported.
//!
//! **Where it runs.** The teacher ([`SpeechTeacher`]) is a forward-only build
//! of the same composite model on the student's card: its linears at bf16, no
//! gradient, no optimizer state. It writes its logits (and mapped hidden
//! states) into the student's buffers ([`Kd`]), and the student's forward
//! computes each KL and its logit gradient in one pass (`kd_kl_rows`); the
//! backward sends those gradients through the student's heads into its hidden
//! states, where the decoders' own backward takes them
//! ([`qwen3::Qwen::enable_hidden_grad`], [`qwen3::Qwen::enable_residual_grad`]).
//!
//! **The text vocabulary.** A trimmed student numbers its text tokens
//! differently from the teacher: [`TextVocabMap`] maps every student id to the
//! teacher's id of the same token, built from both tokenizers' files and
//! refused at job start when a student token is not the teacher's; its
//! languages and voices are the teacher's rows of the same names
//! ([`TeacherMap`]), so a student may have made them new rows of its own.

use std::collections::HashMap;
use std::path::Path;

use gpu_core::{DeviceBuffer, Dispatch, Gpu, Step};
use model::ops::{Ops, Weight};
use serde::{Deserialize, Serialize};

use super::model::{Csr, Example, SftShape, SpeechSftModel};
use crate::prompt::{Part, TtsSpecials};

/// What a distilling build is sized and weighted by.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KdShape {
    /// The weight of the Talker's codebook-0 KL.
    pub talker: f32,
    /// The weight of the code predictor's residual-codebook KLs (summed over the codebooks).
    pub mtp: f32,
    /// The weight of the hidden-state matching.
    pub hidden: f32,
    /// The softmax temperature of both distributions.
    pub temperature: f32,
    /// `(student layer, teacher layer)` pairs whose residual streams are matched.
    pub layers: Vec<(u32, u32)>,
    /// The teacher's Talker width: the projections' output.
    pub teacher_d: u32,
}

/// A hidden-state projection's name: a parameter of the run that is never exported.
pub fn projection(j: usize) -> String {
    format!("{PRIVATE_PREFIX}hidden_kd.{j}.weight")
}

/// The prefix of a run's own parameters, which are not checkpoint tensors.
pub const PRIVATE_PREFIX: &str = "speech_sft.";

impl KdShape {
    /// Refuse what no build can honour, naming it.
    pub fn validate(&self, student: &SftShape) -> Result<(), String> {
        for (name, v) in [("talker_kd_weight", self.talker), ("mtp_kd_weight", self.mtp), ("hidden_kd_weight", self.hidden)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("{name} must be finite and not negative, got {v}"));
            }
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(format!("kd_temperature must be finite and positive, got {}", self.temperature));
        }
        if (self.hidden > 0.0) != !self.layers.is_empty() {
            return Err("hidden_kd_weight and hidden_kd_layers go together: name both or neither".into());
        }
        let n = student.talker.n_layers;
        if let Some((s, _)) = self.layers.iter().find(|(s, _)| *s > n) {
            return Err(format!("hidden_kd_layers: student layer {s} is past the student's {n} layers"));
        }
        let mut seen: Vec<u32> = self.layers.iter().map(|(s, _)| *s).collect();
        seen.sort_unstable();
        if seen.windows(2).any(|w| w[0] == w[1]) {
            return Err("hidden_kd_layers: a student layer is matched twice".into());
        }
        Ok(())
    }

    /// The projections, under their names, `[teacher_d, d]` each.
    pub fn params(&self, d: usize) -> Vec<(String, usize)> {
        (0..self.layers.len()).map(|j| (projection(j), self.teacher_d as usize * d)).collect()
    }

    /// A projection's starting value, `[teacher_d, d]`: the identity on the widths both share.
    pub fn initial_projection(&self, d: usize) -> Vec<f32> {
        let td = self.teacher_d as usize;
        let mut w = vec![0.0; td * d];
        for i in 0..td.min(d) {
            w[i * d + i] = 1.0;
        }
        w
    }
}

/// Every token of a tokenizer directory, by its string: `vocab.json` and the added tokens of `tokenizer_config.json`.
fn tokens(dir: &Path) -> Result<HashMap<String, u32>, String> {
    let at = |f: &str, e: &dyn std::fmt::Display| format!("{}: {f}: {e}", dir.display());
    let vocab: HashMap<String, u32> = serde_json::from_slice(&std::fs::read(dir.join("vocab.json")).map_err(|e| at("vocab.json", &e))?).map_err(|e| at("vocab.json", &e))?;
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json")).map_err(|e| at("tokenizer_config.json", &e))?).map_err(|e| at("tokenizer_config.json", &e))?;
    let mut out = vocab;
    if let Some(added) = cfg["added_tokens_decoder"].as_object() {
        for (id, info) in added {
            let id: u32 = id.parse().map_err(|_| at("tokenizer_config.json", &format!("added token id {id:?} is not a number")))?;
            let content = info["content"].as_str().ok_or_else(|| at("tokenizer_config.json", &format!("added token {id} has no content")))?;
            out.insert(content.to_string(), id);
        }
    }
    Ok(out)
}

/// Every student text id's teacher id. See the module documentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextVocabMap {
    /// Indexed by student id; [`Self::ABSENT`] where the student's table row is no token (rows past its tokenizer).
    to_teacher: Vec<u32>,
}

impl TextVocabMap {
    const ABSENT: u32 = u32::MAX;

    /// The map from the student's tokens (`student_vocab` table rows) to the teacher's (`teacher_vocab` rows): each
    /// student token is the teacher's token of the same string. Refused: a student token the teacher's vocabulary does
    /// not hold, a token id past either table, and two student tokens of one teacher id.
    pub fn from_tokens(student: &HashMap<String, u32>, teacher: &HashMap<String, u32>, student_vocab: u32, teacher_vocab: u32) -> Result<TextVocabMap, String> {
        let mut to_teacher = vec![Self::ABSENT; student_vocab as usize];
        let mut by_teacher: HashMap<u32, &str> = HashMap::new();
        let mut named: Vec<(&String, &u32)> = student.iter().collect();
        named.sort_by_key(|(_, &id)| id);
        for (token, &id) in named {
            let slot = to_teacher.get_mut(id as usize).ok_or_else(|| format!("text vocabulary map: student token {token:?} is id {id}, past the student's {student_vocab} text rows"))?;
            let t = *teacher.get(token).ok_or_else(|| format!("text vocabulary map: student token {token:?} (id {id}) is not in the teacher's vocabulary"))?;
            if t >= teacher_vocab {
                return Err(format!("text vocabulary map: teacher token {token:?} is id {t}, past the teacher's {teacher_vocab} text rows"));
            }
            if let Some(other) = by_teacher.insert(t, token) {
                return Err(format!("text vocabulary map: student tokens {other:?} and {token:?} are both teacher id {t}"));
            }
            *slot = t;
        }
        Ok(TextVocabMap { to_teacher })
    }

    /// [`Self::from_tokens`] over the tokenizer files of two checkpoint directories.
    pub fn from_dirs(student: &Path, teacher: &Path, student_vocab: u32, teacher_vocab: u32) -> Result<TextVocabMap, String> {
        Self::from_tokens(&tokens(student)?, &tokens(teacher)?, student_vocab, teacher_vocab)
    }

    /// The student's id of each teacher id (`teacher_vocab` of them), [`crate::speech_sft::RowMap::ABSENT`] where the
    /// student has no such token: how shards tokenised by the teacher's tokenizer read for the student.
    pub fn to_student(&self, teacher_vocab: u32) -> Vec<u32> {
        let mut out = vec![super::objective::RowMap::ABSENT; teacher_vocab as usize];
        for (s, &t) in self.to_teacher.iter().enumerate() {
            if t != Self::ABSENT {
                out[t as usize] = s as u32;
            }
        }
        out
    }

    /// The teacher's id of student token `id`, or an error naming it.
    pub fn get(&self, id: u32) -> Result<u32, String> {
        match self.to_teacher.get(id as usize) {
            Some(&t) if t != Self::ABSENT => Ok(t),
            _ => Err(format!("text token {id} has no teacher token (it is no token of the student's tokenizer)")),
        }
    }


    /// Refuse a teacher whose prompt specials are not the student's: its text specials the map of the student's, its
    /// codec specials the same rows.
    pub fn check_specials(&self, student: &TtsSpecials, teacher: &TtsSpecials) -> Result<(), String> {
        for (name, s, t) in [("tts_bos", student.tts_bos, teacher.tts_bos), ("tts_eos", student.tts_eos, teacher.tts_eos), ("tts_pad", student.tts_pad, teacher.tts_pad)] {
            let mapped = self.get(s).map_err(|e| format!("{name}: {e}"))?;
            if mapped != t {
                return Err(format!("the student's {name} (text id {s}) is the teacher's text id {mapped}, but the teacher's {name} is {t}"));
            }
        }
        let codec = |sp: &TtsSpecials| [sp.codec_nothink, sp.codec_think, sp.codec_think_bos, sp.codec_think_eos, sp.codec_pad, sp.codec_bos, sp.codec_eos];
        if codec(student) != codec(teacher) {
            return Err(format!("the student's codec specials {:?} are not the teacher's {:?}", codec(student), codec(teacher)));
        }
        Ok(())
    }
}

/// How the teacher reads a student's batch: its text tokens through the vocabulary map, its language and voice rows as
/// the teacher's rows of the same names (the student may hold them at other rows, or have made them new).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeacherMap {
    pub text: TextVocabMap,
    /// Student codec row -> the teacher's row of the same language or voice; every other codec row is the same row.
    pub codec_rows: HashMap<u32, u32>,
}

impl TeacherMap {
    /// The text map alone: every codec row the teacher's own.
    pub fn text_only(text: TextVocabMap) -> TeacherMap {
        TeacherMap { text, codec_rows: HashMap::new() }
    }

    /// `ex` as the teacher reads it: every text token renumbered, its language and voice rows the teacher's, its codes
    /// kept.
    pub fn example(&self, ex: &Example) -> Result<Example, String> {
        let mut out = ex.clone();
        for part in out.plan.rows.iter_mut().flatten() {
            match part {
                Part::Text(id) => *id = self.text.get(*id)?,
                Part::Codec(id) => *id = self.codec_rows.get(id).copied().unwrap_or(*id),
                Part::XVector => {}
            }
        }
        for id in &mut out.plan.trailing {
            *id = self.text.get(*id)?;
        }
        out.plan.tts_pad = self.text.get(out.plan.tts_pad)?;
        // The teacher reads the x-vector the shard stores, not the student's projection of it.
        out.xvector = ex.source_xvector.clone();
        Ok(out)
    }
}

/// The kernels a distilling build dispatches on its auxiliary device, beside [`super::model::aux_kernels`]'s.
pub(crate) fn kernels() -> [(&'static str, &'static str); 3] {
    [("kd_kl_rows", kernels::KD_KL_ROWS), ("mse_value_w", kernels::MSE_VALUE_W), ("mse_grad_w", kernels::MSE_GRAD_W)]
}

/// The kernels [`Kd`] dispatches by slot.
struct KdKernels {
    kl: usize,
    copy: usize,
    gather_bwd: usize,
    mse_value: usize,
    mse_grad: usize,
    matmul_dw: usize,
    matmul_dw_reg: Option<usize>,
}

/// One mapped hidden-state pair's buffers.
struct HiddenPair {
    student_layer: usize,
    /// The student's rows, `[t_cap, d]`, their projection, `[t_cap, teacher_d]`, the teacher's rows (written by the
    /// teacher), the projection's gradient and the rows' gradient.
    rows: DeviceBuffer,
    pred: DeviceBuffer,
    target: DeviceBuffer,
    d_pred: DeviceBuffer,
    d_rows: DeviceBuffer,
    /// Per-row squared error.
    err: DeviceBuffer,
    /// The gradient of the residual stream entering the student layer, `[batch * block, d]`, which the Talker adds in
    /// its backward ([`qwen3::Qwen::enable_residual_grad`]).
    grad: DeviceBuffer,
}

/// What a batch distils: the rows each term reads, set when the batch is loaded.
#[derive(Clone, Copy, Default)]
struct KdLoaded {
    talker_rows: u32,
    frames: u32,
    /// The load the teacher taught last, against the student's: a forward never reads a previous batch's teacher.
    load: u64,
    taught: u64,
}

/// The measurements of one distilling forward: sums, so micro-batches and held-out batches add up.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KdStats {
    /// Summed Talker KL and how many rows.
    pub talker_sum: f64,
    pub talker_count: u32,
    /// Summed residual-codebook KL over every frame and codebook, and the frames.
    pub cp_sum: f64,
    pub cp_frames: u32,
    /// Summed per-row mean squared error over every mapped pair, and the rows.
    pub hidden_sum: f64,
    pub hidden_count: u32,
}

impl KdStats {
    pub fn add(&mut self, o: &KdStats) {
        self.talker_sum += o.talker_sum;
        self.talker_count += o.talker_count;
        self.cp_sum += o.cp_sum;
        self.cp_frames += o.cp_frames;
        self.hidden_sum += o.hidden_sum;
        self.hidden_count += o.hidden_count;
    }

    /// The mean Talker KL.
    pub fn talker_kl(&self) -> f64 {
        mean(self.talker_sum, self.talker_count)
    }

    /// The residual codebooks' KLs, each a mean over the frames, summed.
    pub fn code_predictor_kl(&self) -> f64 {
        mean(self.cp_sum, self.cp_frames)
    }

    /// The mapped pairs' mean squared errors, summed.
    pub fn hidden_mse(&self) -> f64 {
        mean(self.hidden_sum, self.hidden_count)
    }

    /// The distillation terms of the loss under `kd`'s weights.
    pub fn loss(&self, kd: &KdShape) -> f64 {
        let t2 = f64::from(kd.temperature) * f64::from(kd.temperature);
        f64::from(kd.talker) * t2 * self.talker_kl() + f64::from(kd.mtp) * t2 * self.code_predictor_kl() + f64::from(kd.hidden) * self.hidden_mse()
    }
}

fn mean(sum: f64, n: u32) -> f64 {
    if n == 0 {
        0.0
    } else {
        sum / f64::from(n)
    }
}

/// The student's side of a distillation: the rows every term reads, the teacher's logits and hidden states the teacher
/// writes, the student's logits and their gradient, all on the student's card. See the module documentation.
pub struct Kd {
    pub(super) shape: KdShape,
    k: KdKernels,
    /// Rows of the Talker's supervised rows a batch may hold, and of each residual codebook's frames (whole groups of
    /// 64, so every codebook's rows start on a binding boundary).
    t_cap: u32,
    stride: u32,
    /// `(talker row, kd row)` of every supervised Talker row, and its transpose into the Talker's rows.
    pub(super) t_plan: DeviceBuffer,
    t_csr: Csr,
    t_rows: DeviceBuffer,
    t_logits: DeviceBuffer,
    pub(super) t_teacher: DeviceBuffer,
    t_kl: DeviceBuffer,
    t_dh: DeviceBuffer,
    /// `(code-predictor row, kd row)`: frame `f`'s codebook `k + 1` at kd row `k * stride + f`, and its transpose.
    pub(super) c_plan: DeviceBuffer,
    c_csr: Csr,
    c_rows: DeviceBuffer,
    /// Per residual codebook: the student's logits (then their gradient) and the teacher's, `[stride, codebook]`.
    c_logits: Vec<DeviceBuffer>,
    pub(super) c_teacher: Vec<DeviceBuffer>,
    c_kl: DeviceBuffer,
    c_dh: DeviceBuffer,
    /// The gradient of the code predictor's hidden states, which its backward adds ([`qwen3::Qwen::enable_hidden_grad`]).
    pub(super) cp_hidden_grad: DeviceBuffer,
    hidden: Vec<HiddenPair>,
    ones: DeviceBuffer,
    loaded: std::cell::Cell<KdLoaded>,
}

impl Kd {
    /// The buffers of `shape`'s distillation on the student's auxiliary device `gpu` (the Talker's card).
    pub(super) fn new(shape: &SftShape, kd: &KdShape, gpu: &Gpu) -> Result<Kd, String> {
        let k = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("speech sft distillation: kernel {name} is not registered"));
        let kernels = KdKernels { kl: k("kd_kl_rows")?, copy: k("anchor_rows_copy")?, gather_bwd: k("rows_gather_weighted_bwd")?, mse_value: k("mse_value_w")?, mse_grad: k("mse_grad_w")?, matmul_dw: k("matmul_dw")?, matmul_dw_reg: gpu.kernel_index("matmul_dw_reg") };
        let (d, v) = (shape.d() as u64, u64::from(shape.talker.vocab));
        let (dc, cv) = (u64::from(shape.mtp.d_model), u64::from(shape.mtp.vocab));
        let heads = shape.mtp.n_residual() as u64;
        let t_cap = shape.frames + shape.batch;
        let stride = shape.frames.next_multiple_of(64);
        let talker_cap = (shape.batch * shape.block) as usize;
        let cp_cap = shape.cp_batch() as usize * shape.mtp.num_code_groups as usize;
        let st = |n: u64| gpu.storage(n.max(1));
        let tc = u64::from(t_cap);
        let rows = heads * u64::from(stride);
        let td = u64::from(kd.teacher_d);
        let hidden = kd
            .layers
            .iter()
            .map(|&(s, _)| HiddenPair { student_layer: s as usize, rows: st(tc * d), pred: st(tc * td), target: st(tc * td), d_pred: st(tc * td), d_rows: st(tc * d), err: st(tc), grad: gpu.storage((talker_cap as u64 * d).max(1)) })
            .collect();
        Ok(Kd {
            shape: kd.clone(),
            k: kernels,
            t_cap,
            stride,
            t_plan: st(tc * 2),
            t_csr: Csr::new(gpu, talker_cap, t_cap as usize),
            t_rows: st(tc * d),
            t_logits: st(tc * v),
            t_teacher: st(tc * v),
            t_kl: st(tc),
            t_dh: st(tc * d),
            c_plan: st(rows * 2),
            c_csr: Csr::new(gpu, cp_cap, rows as usize),
            c_rows: st(rows * dc),
            c_logits: (0..heads).map(|_| st(u64::from(stride) * cv)).collect(),
            c_teacher: (0..heads).map(|_| st(u64::from(stride) * cv)).collect(),
            c_kl: st(rows),
            c_dh: st(rows * dc),
            cp_hidden_grad: gpu.storage((cp_cap as u64 * dc).max(1)),
            hidden,
            ones: gpu.storage_init("speech_sft.kd_ones", &vec![1.0f32; t_cap.max(rows as u32) as usize]),
            loaded: Default::default(),
        })
    }

    /// The pairs' residual-gradient buffers, by student layer, for the Talker to add ([`qwen3::Qwen::enable_residual_grad`]).
    pub(super) fn residual_grads(&self) -> impl Iterator<Item = (usize, &DeviceBuffer)> {
        self.hidden.iter().map(|p| (p.student_layer, &p.grad))
    }

    /// The teacher's buffer of pair `j`'s target rows.
    pub(super) fn hidden_target(&self, j: usize) -> &DeviceBuffer {
        &self.hidden[j].target
    }

    /// Rows of every residual codebook's block: the kd row of frame `f`, codebook `k + 1` is `k * stride + f`.
    pub(super) fn stride(&self) -> u32 {
        self.stride
    }

    /// The rows the loaded batch distils: Talker rows and frames.
    pub(super) fn loaded_rows(&self) -> (u32, u32) {
        let l = self.loaded.get();
        (l.talker_rows, l.frames)
    }

    /// Upload the plans of a batch: its supervised Talker rows (in the Talker's computed layout) and its frames, frame
    /// `f`'s code-predictor rows starting at `f * groups`.
    pub(super) fn load(&self, gpu: &Gpu, talker_rows: &[u32], frames: u32, groups: u32) {
        assert!(talker_rows.len() <= self.t_cap as usize && frames <= self.stride, "speech sft distillation: a batch past the build's rows");
        let t: Vec<u32> = talker_rows.iter().enumerate().flat_map(|(i, &r)| [r, i as u32]).collect();
        if !t.is_empty() {
            gpu.write(&self.t_plan, &t);
        }
        let reads: Vec<(u32, u32)> = talker_rows.iter().enumerate().map(|(i, &r)| (i as u32, r)).collect();
        self.t_csr.upload(gpu, &reads);
        let heads = groups - 1;
        let mut plan = Vec::with_capacity((heads * frames * 2) as usize);
        let mut c_reads = Vec::with_capacity((heads * frames) as usize);
        for h in 0..heads {
            for f in 0..frames {
                let (src, dst) = (f * groups + 1 + h, h * self.stride + f);
                plan.extend([src, dst]);
                c_reads.push((dst, src));
            }
        }
        // The transpose lists its edges in code-predictor row order, as the gathered rows are read.
        c_reads.sort_unstable_by_key(|&(_, src)| src);
        if !plan.is_empty() {
            gpu.write(&self.c_plan, &plan);
        }
        self.c_csr.upload(gpu, &c_reads);
        let mut l = self.loaded.get();
        l.talker_rows = talker_rows.len() as u32;
        l.frames = frames;
        l.load += 1;
        self.loaded.set(l);
    }

    /// The teacher has written its side of the loaded batch.
    pub(super) fn mark_taught(&self) {
        let mut l = self.loaded.get();
        l.taught = l.load;
        self.loaded.set(l);
    }

    /// The student's terms of the loaded batch, its logit gradients left for [`Self::backward_code_predictor`] and
    /// [`Self::backward_talker`]: `talker_hidden` and `cp_hidden` are the decoders' final-norm hidden states,
    /// `residual(layer)` the Talker's residual stream entering a layer.
    pub(super) fn forward(&self, ops: &Ops, heads: &KdHeads<'_>, talker_hidden: &DeviceBuffer, cp_hidden: &DeviceBuffer, residual: &dyn Fn(usize) -> DeviceBuffer, store: &paramstore::ParamStore) -> KdStats {
        let l = self.loaded.get();
        assert_eq!(l.taught, l.load, "speech sft distillation: the teacher has not taught the loaded batch");
        let gpu = ops.gpu();
        let (nt, nf) = (l.talker_rows, l.frames);
        let kd = &self.shape;
        let inv_t = (1.0 / kd.temperature).to_bits();
        let mut steps = Vec::new();
        let (d, v) = (heads.d, heads.vocab);
        if nt > 0 {
            steps.push(gpu.step(self.k.copy, &[&self.t_plan, talker_hidden, &self.t_rows], &[nt, d], nt * d));
            ops.matmul(&mut steps, &heads.talker, &ops.act_f32(&self.t_rows, 0, nt, d), &self.t_logits, 0);
            let coef = kd.talker * kd.temperature / nt as f32;
            steps.push(gpu.step(self.k.kl, &[&self.t_logits, &self.t_teacher, &self.t_kl], &[nt, v, inv_t, coef.to_bits()], nt));
        }
        let (dc, cv) = (heads.cp_d, heads.cp_vocab);
        if nf > 0 {
            steps.push(gpu.step(self.k.copy, &[&self.c_plan, cp_hidden, &self.c_rows], &[heads.residuals * nf, dc], heads.residuals * nf * dc));
            let coef = kd.mtp * kd.temperature / nf as f32;
            for h in 0..heads.residuals {
                ops.matmul_rows(&mut steps, &heads.cp, (h * cv, cv), &ops.act_f32(&self.c_rows, h * self.stride, nf, dc), &self.c_logits[h as usize], 0);
                let at = (u64::from(h * self.stride), u64::from(nf));
                steps.push(gpu.step_sliced(self.k.kl, &[&self.c_logits[h as usize], &self.c_teacher[h as usize], &self.c_kl], &[(0, 0), (0, 0), at], &[nf, cv, inv_t, coef.to_bits()], nf));
            }
        }
        for (j, p) in self.hidden.iter().enumerate().filter(|_| nt > 0) {
            let td = kd.teacher_d;
            steps.push(gpu.step(self.k.copy, &[&self.t_plan, &residual(p.student_layer), &p.rows], &[nt, d], nt * d));
            ops.matmul(&mut steps, &Weight::dense_f32(store.w(&projection(j)).clone(), td, d), &ops.act_f32(&p.rows, 0, nt, d), &p.pred, 0);
            steps.push(gpu.step(self.k.mse_value, &[&p.pred, &p.target, &self.ones, &p.err], &[nt, td], nt));
            let scale = kd.hidden / nt as f32;
            steps.push(gpu.step(self.k.mse_grad, &[&p.pred, &p.target, &self.ones, &p.d_pred], &[nt * td, td, scale.to_bits()], nt * td));
        }
        gpu.submit(&[], &steps);
        let sum = |buf: &DeviceBuffer, at: u64, n: u32| {
            let mut out = vec![0f32; n as usize];
            gpu.read_into(buf, at, &mut out);
            out.iter().map(|&x| f64::from(x)).sum::<f64>()
        };
        let mut stats = KdStats { talker_count: nt, cp_frames: nf, ..Default::default() };
        if nt > 0 {
            stats.talker_sum = sum(&self.t_kl, 0, nt);
            stats.hidden_count = if self.hidden.is_empty() { 0 } else { nt };
            stats.hidden_sum = self.hidden.iter().map(|p| sum(&p.err, 0, nt)).sum();
        }
        if nf > 0 {
            stats.cp_sum = (0..heads.residuals).map(|h| sum(&self.c_kl, u64::from(h * self.stride), nf)).sum();
        }
        stats
    }

    /// The code predictor's logit gradients through its heads: into its heads' gradient and its hidden states'
    /// ([`Self::cp_hidden_grad`]). Before the code predictor's backward.
    pub(super) fn backward_code_predictor(&self, ops: &Ops, heads: &KdHeads<'_>) {
        let nf = self.loaded.get().frames;
        if nf == 0 || self.shape.mtp == 0.0 {
            return;
        }
        let gpu = ops.gpu();
        let (dc, cv) = (heads.cp_d, heads.cp_vocab);
        let grad = heads.cp_grad.expect("a trained build's heads have gradients");
        let mut steps = Vec::new();
        for h in 0..heads.residuals {
            let x = (u64::from(h * self.stride) * u64::from(dc), u64::from(nf) * u64::from(dc));
            let window = (u64::from(h * cv) * u64::from(dc), u64::from(cv) * u64::from(dc));
            steps.push(self.dw_step(ops, &self.c_logits[h as usize], &self.c_rows, x, grad, window, nf, cv, dc));
            ops.matmul_dx_rows(&mut steps, &heads.cp, (h * cv, cv), &self.c_logits[h as usize], nf, &self.c_dh, h * self.stride, false);
        }
        steps.push(self.c_csr.grad_step(gpu, self.k.gather_bwd, &self.ones, &self.c_dh, &self.cp_hidden_grad, dc));
        gpu.submit(&[], &steps);
    }

    /// The Talker's logit gradients into its head's gradient and its hidden states' (`hidden_grad`, added to), and the
    /// hidden-state pairs' into their projections and the Talker's residual streams. After the code predictor's route
    /// into `hidden_grad`, before the Talker's backward.
    pub(super) fn backward_talker(&self, ops: &Ops, heads: &KdHeads<'_>, hidden_grad: &DeviceBuffer, store: &paramstore::ParamStore) {
        let nt = self.loaded.get().talker_rows;
        if nt == 0 {
            return;
        }
        let gpu = ops.gpu();
        let d = heads.d;
        let mut steps = Vec::new();
        if self.shape.talker > 0.0 {
            ops.matmul_dw(&mut steps, &self.t_rows, &self.t_logits, nt, heads.vocab, d, heads.talker_grad.expect("a trained build's heads have gradients"));
            ops.matmul_dx(&mut steps, &heads.talker, &self.t_logits, nt, &self.t_dh, false);
            steps.push(self.t_csr.grad_step(gpu, self.k.gather_bwd, &self.ones, &self.t_dh, hidden_grad, d));
        }
        let td = self.shape.teacher_d;
        for (j, p) in self.hidden.iter().enumerate() {
            let name = projection(j);
            ops.matmul_dw(&mut steps, &p.rows, &p.d_pred, nt, td, d, store.g(&name));
            ops.matmul_dx(&mut steps, &Weight::dense_f32(store.w(&name).clone(), td, d), &p.d_pred, nt, &p.d_rows, false);
            steps.push(self.t_csr.grad_step(gpu, self.k.gather_bwd, &self.ones, &p.d_rows, &p.grad, d));
        }
        gpu.submit(&[], &steps);
    }

    /// `dW[window] += dy^T x[x]` for `rows` rows of `cols` outputs of width `d`: the register-tiled kernel where it fills
    /// its tile, the portable one elsewhere.
    #[allow(clippy::too_many_arguments)]
    fn dw_step(&self, ops: &Ops, dy: &DeviceBuffer, x: &DeviceBuffer, x_at: (u64, u64), dw: &DeviceBuffer, window: (u64, u64), rows: u32, cols: u32, d: u32) -> Step {
        let (kind, grid) = match self.k.matmul_dw_reg {
            Some(reg) if ops.fills_tile(cols, d) => (reg, Dispatch::Workgroups(cols.div_ceil(128) * d.div_ceil(128))),
            _ => (self.k.matmul_dw, Dispatch::Threads(cols * d)),
        };
        ops.gpu().dispatch_sliced(kind, &[dy, x, dw], &[(0, 0), x_at, window], &[rows, d, cols], grid)
    }
}

/// A composite model's heads as the distillation reads them: the Talker's `[vocab, d]` and the code predictor's
/// stacked `[residuals * cp_vocab, cp_d]`, with the student's gradients (unused on a teacher).
pub(super) struct KdHeads<'a> {
    pub talker: Weight,
    pub talker_grad: Option<&'a DeviceBuffer>,
    pub cp: Weight,
    pub cp_grad: Option<&'a DeviceBuffer>,
    pub d: u32,
    pub vocab: u32,
    pub cp_d: u32,
    pub cp_vocab: u32,
    pub residuals: u32,
}

/// The teacher's gathered rows: its final-norm hidden states at the student's supervised rows.
struct TeacherRows {
    talker: DeviceBuffer,
    cp: DeviceBuffer,
}

/// A frozen teacher on the student's card. See the module documentation.
pub struct SpeechTeacher {
    model: SpeechSftModel,
    map: TeacherMap,
    /// `(student layer index into the pairs, teacher layer)` of the hidden-state pairs.
    layers: Vec<usize>,
    rows: TeacherRows,
    /// `anchor_rows_copy` on the teacher's auxiliary device.
    copy: usize,
}

impl SpeechTeacher {
    /// The teacher of a student `student` from the upstream-named tensors `init` of the teacher checkpoint (`shape` its
    /// configuration, sized as the student's batch), its text tokens read through `map`. Refused: a teacher of other
    /// codebooks, codebook sizes or Talker codec rows than the student's, a mapped teacher layer past its layers.
    pub fn new(student: &SpeechSftModel, shape: SftShape, init: HashMap<String, Vec<f32>>, map: TeacherMap) -> Result<SpeechTeacher, String> {
        let s = student.shape();
        let kd = s.kd.as_ref().ok_or("speech sft distillation: the student is not built to distil")?;
        if (shape.talker.vocab, shape.mtp.vocab, shape.mtp.num_code_groups) != (s.talker.vocab, s.mtp.vocab, s.mtp.num_code_groups) {
            return Err(format!(
                "speech sft distillation: the teacher codes {} codebooks of {} codes with {} Talker rows, the student {} of {} with {}",
                shape.mtp.num_code_groups, shape.mtp.vocab, shape.talker.vocab, s.mtp.num_code_groups, s.mtp.vocab, s.talker.vocab
            ));
        }
        if (shape.batch, shape.block, shape.frames) != (s.batch, s.block, s.frames) {
            return Err("speech sft distillation: the teacher is sized as another batch than the student's".into());
        }
        if shape.d() as u32 != kd.teacher_d {
            return Err(format!("speech sft distillation: the student projects onto a {}-wide teacher, the teacher is {} wide", kd.teacher_d, shape.d()));
        }
        let layers = kd.layers.iter().map(|&(_, t)| if t > shape.talker.n_layers { Err(format!("hidden_kd_layers: teacher layer {t} is past the teacher's {} layers", shape.talker.n_layers)) } else { Ok(t as usize) }).collect::<Result<Vec<_>, _>>()?;
        map.text.check_specials(&s.specials, &shape.specials)?;
        let model = SpeechSftModel::build_frozen(shape, init, student.device())?;
        let k = student.kd().expect("checked: the student distils");
        let gpu = model.aux_gpu();
        let rows = TeacherRows { talker: gpu.storage(u64::from(k.t_cap) * model.shape().d() as u64), cp: gpu.storage((u64::from(model.shape().mtp.n_residual()) * u64::from(k.stride) * u64::from(model.shape().mtp.d_model)).max(1)) };
        let copy = gpu.kernel_index("anchor_rows_copy").ok_or("speech sft distillation: kernel anchor_rows_copy is not registered")?;
        Ok(SpeechTeacher { model, map, layers, rows, copy })
    }

    /// The map the teacher reads the student's batches through.
    pub fn map(&self) -> &TeacherMap {
        &self.map
    }

    /// The parameters the teacher would train: none (no gradient and no optimizer state is held for any).
    pub fn trained_params(&self) -> usize {
        self.model.trained_param_count()
    }

    /// Run the teacher on `examples` (the student's loaded batch, as [`SpeechSftModel::load`] took it) and write its
    /// logits and mapped hidden states into the student's buffers.
    pub fn teach(&self, student: &SpeechSftModel, examples: &[Example]) -> Result<(), String> {
        let k = student.kd().ok_or("speech sft distillation: the student is not built to distil")?;
        let mapped = examples.iter().map(|e| self.map.example(e)).collect::<Result<Vec<_>, _>>()?;
        self.model.load(mapped)?;
        if self.model.computed_layout() != student.computed_layout() {
            return Err(format!("speech sft distillation: the teacher computes the batch as {:?}, the student as {:?}", self.model.computed_layout(), student.computed_layout()));
        }
        self.model.run_decoders();
        let (nt, nf) = k.loaded_rows();
        let ops = self.model.aux_ops();
        let gpu = ops.gpu();
        let heads = self.model.kd_heads();
        let mut steps = Vec::new();
        if nt > 0 {
            steps.push(gpu.step(self.copy, &[&k.t_plan, self.model.talker_hidden(), &self.rows.talker], &[nt, heads.d], nt * heads.d));
            ops.matmul(&mut steps, &heads.talker, &ops.act_f32(&self.rows.talker, 0, nt, heads.d), &k.t_teacher, 0);
            for (j, &layer) in self.layers.iter().enumerate() {
                steps.push(gpu.step(self.copy, &[&k.t_plan, &self.model.talker_residual(layer), k.hidden_target(j)], &[nt, heads.d], nt * heads.d));
            }
        }
        if nf > 0 {
            let (dc, cv) = (heads.cp_d, heads.cp_vocab);
            steps.push(gpu.step(self.copy, &[&k.c_plan, self.model.cp_hidden(), &self.rows.cp], &[heads.residuals * nf, dc], heads.residuals * nf * dc));
            for h in 0..heads.residuals {
                ops.matmul_rows(&mut steps, &heads.cp, (h * cv, cv), &ops.act_f32(&self.rows.cp, h * k.stride(), nf, dc), &k.c_teacher[h as usize], 0);
            }
        }
        gpu.submit(&[], &steps);
        k.mark_taught();
        Ok(())
    }
}
