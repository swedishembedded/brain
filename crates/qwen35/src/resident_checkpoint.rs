// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements serving of released LLM checkpoints in the
// precision they were published in for its clients. If your team needs
// expertise in serving GGUF or FP8 safetensors releases on its own GPUs, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The checkpoint a resident Qwen3.5/3.8 engine is served from, as ONE seam:
//! a released GGUF or a HuggingFace safetensors directory (the FP8 release
//! `Qwen/Qwen3.8-27B-FP8`, or a bf16 one), both of the same architecture.
//!
//! [`crate::int8_gguf_resident`] asks this type for everything it used to read
//! from a `MmapGguf` directly: the model config, the tokenizer and stop ids,
//! the sampling defaults, the per-stage [`TensorSource`] in brain-canonical
//! names, the embedding rows, the three endpoint tensors, and the weight tier
//! that holds each linear exactly as it is stored. Everything past this seam -
//! the paged KV, the prefix cache, the scheduler, the kernels - is one
//! implementation for both formats.
//!
//! What a format offers differs in two places, and both are stated here
//! instead of being discovered as a missing tensor:
//!
//! * the GGUF holds its linears as Q8_0/K-quant blocks ([`gguf::held_tier_of`])
//!   and the head as int8; an HF directory holds them in whatever
//!   [`model::ops::stored_tier`] names (FP8 e4m3 with 128x128 block scales,
//!   bf16 for the tensors the release keeps unconverted), and the head and
//!   embedding in bf16, served as stored;
//! * the multi-token-prediction head is the GGUF's `blk.{n_layers}` block, or an
//!   HF directory's `mtp.*` tensors (`mtp.safetensors`), served in the encoding
//!   they are stored in ([`ResidentCheckpoint::mtp_source`]).

use std::path::{Path, PathBuf};

use checkpoint::gguf::MmapGguf;
use checkpoint::packed::{PackedSource, QuantEncoding};
use checkpoint::TensorSource;
use data::generation::SamplingDefaults;
use data::qwen_tokenizer::QwenBpe;
use gpu_core::select::Dtype;
use model::ops::TierPolicy;

use crate::config::Qwen35Config;
use model::shard::Shard;

/// An embedding table held in host memory as the bf16 it is stored in: a row is a copy, never a page fault.
///
/// A drafter that reads the embeddings of tokens it did not choose (a whole prompt's, in a feed) would otherwise touch one
/// cold page of the mapped checkpoint per token, which is milliseconds each on a network file system.
pub struct HostEmbedding {
    bits: Vec<u16>,
    d: usize,
}

impl HostEmbedding {
    /// Rows `tokens` (`d` wide) as f32, `None` when a token is outside the table.
    pub fn rows(&self, tokens: &[u32]) -> Option<Vec<f32>> {
        let vocab = self.bits.len() / self.d;
        let mut out = Vec::with_capacity(tokens.len() * self.d);
        for &t in tokens {
            let row = self.bits.get(t as usize * self.d..(t as usize + 1) * self.d).filter(|_| (t as usize) < vocab)?;
            out.extend(row.iter().map(|&b| f32::from_bits(u32::from(b) << 16)));
        }
        Some(out)
    }
}

/// A checkpoint of the Qwen3.5/3.8 dense architecture a resident engine can be
/// built from. See the module doc.
pub enum ResidentCheckpoint {
    /// A released GGUF: Q8_0/K-quant tensors in llama.cpp's names.
    Gguf(MmapGguf),
    /// A HuggingFace directory read lazily under brain-canonical names.
    Hf(Box<HfCheckpoint>),
}

/// One stage's tensors, borrowed from a [`ResidentCheckpoint`].
pub enum ShardSource<'a> {
    Gguf(gguf::GdnFixSource<'a>),
    Hf(&'a PackedSource),
}

impl ShardSource<'_> {
    /// The source as the trait object model builders take.
    pub fn as_source(&self) -> &dyn TensorSource {
        match self {
            ShardSource::Gguf(s) => s,
            ShardSource::Hf(s) => *s,
        }
    }
}

/// The HF half of [`ResidentCheckpoint`]: the directory and its lazy source.
pub struct HfCheckpoint {
    dir: PathBuf,
    cfg: Qwen35Config,
    src: PackedSource,
    /// The multi-token-prediction head's tensors under a one-layer stage's names ([`crate::fp8_source::open_mtp`]).
    mtp: Option<PackedSource>,
}

/// The ids that end a generation: what the checkpoint declares (a GGUF's `eos`, an HF release's
/// `generation_config.json`) plus the chat markup terminators, resolved through the vocabulary rather than
/// written down. Duplicates are dropped so `eos.contains` stays a plain scan.
fn stop_ids(dir: Option<&Path>, tok: &QwenBpe, declared: Option<u32>) -> Result<Vec<u32>, String> {
    let mut ids = data::generation::stop_ids(dir, tok, declared, Some("<|im_end|>"))?;
    if let Some(id) = tok.special_id("<|endoftext|>").filter(|id| !ids.contains(id)) {
        ids.push(id);
    }
    Ok(ids)
}

impl ResidentCheckpoint {
    /// Open the checkpoint at `path`: a directory is an HF checkpoint, anything
    /// else a GGUF file. Reads headers only.
    pub fn open(path: &str) -> Result<ResidentCheckpoint, String> {
        if Path::new(path).is_dir() {
            let dir = PathBuf::from(path);
            let cfg = crate::fp8_source::hf_config(&dir)?;
            let block = crate::fp8_source::hf_fp8_block_size(&dir)?;
            let src = crate::fp8_source::open_dir(&dir, &cfg, block)?;
            let mtp = crate::fp8_source::open_mtp(&dir, &cfg, block)?;
            Ok(ResidentCheckpoint::Hf(Box::new(HfCheckpoint { dir, cfg, src, mtp })))
        } else {
            MmapGguf::open(path).map(ResidentCheckpoint::Gguf)
        }
    }

    /// Whether `path` names a checkpoint [`Self::open`] reads as an HF directory.
    pub fn is_hf_dir(path: &str) -> bool {
        Path::new(path).is_dir()
    }

    /// The model config for a resident whose per-sequence context is `cap`
    /// tokens: the MTP block is never imported by the resident (it loads its
    /// own), and `block_size` is the context the stages are sized for.
    pub fn config(&self, cap: u32) -> Result<Qwen35Config, String> {
        let mut cfg = match self {
            ResidentCheckpoint::Gguf(mg) => crate::gguf_import::config_from_gguf(mg)?,
            ResidentCheckpoint::Hf(hf) => hf.cfg.clone(),
        };
        cfg.mtp = false;
        cfg.block_size = cap;
        Ok(cfg)
    }

    /// The tokenizer and the stop ids a generation ends on.
    pub fn tokenizer(&self) -> Result<(QwenBpe, Vec<u32>), String> {
        match self {
            ResidentCheckpoint::Gguf(mg) => {
                let gtok = mg.tokenizer().ok_or("it carries no embedded tokenizer (tokenizer.ggml.* KV)")?;
                let tok = QwenBpe::from_gguf(&gtok).map_err(|e| format!("embedded tokenizer: {e}"))?;
                let stop = stop_ids(None, &tok, gtok.eos)?;
                Ok((tok, stop))
            }
            ResidentCheckpoint::Hf(hf) => {
                let file = hf.dir.join("tokenizer.json");
                let tok = QwenBpe::from_file(file.to_str().ok_or("tokenizer path is not UTF-8")?)?;
                let stop = stop_ids(Some(&hf.dir), &tok, None)?;
                Ok((tok, stop))
            }
        }
    }

    /// The checkpoint's chat template (a GGUF's `tokenizer.chat_template`, an HF directory's `chat_template.jinja` or
    /// `tokenizer_config.json`), for tools that render conversations themselves.
    pub fn chat_template(&self) -> Result<data::chat_template::ChatTemplate, String> {
        match self {
            ResidentCheckpoint::Gguf(mg) => {
                let gtok = mg.tokenizer().ok_or("it carries no embedded tokenizer (tokenizer.ggml.* KV)")?;
                data::chat_template::ChatTemplate::from_gguf(&gtok).map_err(|e| format!("embedded chat template: {e}"))
            }
            ResidentCheckpoint::Hf(hf) => data::chat_template::ChatTemplate::from_model_dir(&hf.dir).map_err(|e| format!("chat template: {e}")),
        }
    }

    /// Whether the tokenizer can be built, for the cheap "is this servable" check.
    pub fn has_tokenizer(&self) -> bool {
        match self {
            ResidentCheckpoint::Gguf(mg) => mg.tokenizer().is_some(),
            ResidentCheckpoint::Hf(hf) => hf.dir.join("tokenizer.json").is_file(),
        }
    }

    /// The sampling settings the release declares for requests that name none.
    pub fn sampling_defaults(&self) -> Result<SamplingDefaults, String> {
        match self {
            ResidentCheckpoint::Gguf(mg) => SamplingDefaults::from_gguf(|key| mg.kv().get(key).and_then(|v| v.as_f64())),
            ResidentCheckpoint::Hf(hf) => SamplingDefaults::from_generation_config(&hf.dir),
        }
    }

    /// The tensor names `(embedding table, final norm, lm_head)` in THIS
    /// checkpoint's spelling, resolved through the same name maps the stages use.
    pub fn endpoint_names(&self, cfg: &Qwen35Config) -> Result<(String, String, String), String> {
        match self {
            ResidentCheckpoint::Gguf(mg) => crate::int8_gguf_resident::endpoint_names(mg, cfg),
            ResidentCheckpoint::Hf(_) => Ok(("tok.weight".to_string(), "norm.weight".to_string(), cfg.head_weight().to_string())),
        }
    }

    /// The brain-canonical [`TensorSource`] of ONE stage: exactly the tensors
    /// the stage's layers (and endpoints, if it owns them) ask for, validated
    /// against their declared sizes before any byte is uploaded.
    pub fn shard_source<'a>(&'a self, cfg: &Qwen35Config, shard: &Shard) -> Result<ShardSource<'a>, String> {
        match self {
            ResidentCheckpoint::Gguf(mg) => Ok(ShardSource::Gguf(crate::int8_gguf_resident::shard_source(mg, cfg, shard)?)),
            // The HF source already serves every canonical name, validated
            // two-way against the config at open.
            ResidentCheckpoint::Hf(hf) => Ok(ShardSource::Hf(&hf.src)),
        }
    }

    /// The source the three endpoint tensors are read from (names from
    /// [`Self::endpoint_names`]).
    pub fn endpoint_source(&self) -> &dyn TensorSource {
        match self {
            ResidentCheckpoint::Gguf(mg) => mg,
            ResidentCheckpoint::Hf(hf) => &hf.src,
        }
    }

    /// The tier the lm_head is held at: int8 for a GGUF (its tensor is a
    /// quantized one and a fp32 table cannot be a device buffer on small
    /// cards), as stored (bf16) for an HF release, which keeps it unconverted.
    pub fn head_tier(&self) -> Dtype {
        match self {
            ResidentCheckpoint::Gguf(_) => Dtype::I8,
            ResidentCheckpoint::Hf(_) => Dtype::BF16,
        }
    }

    /// The tier the embedding table can be held at on a card without changing a value: the 16-bit float the
    /// checkpoint stores it in. `None` for a GGUF (a quantized table, decoded row by row from the mapping) and for any
    /// HF encoding with scales, which the host decodes.
    pub fn embed_tier(&self) -> Option<Dtype> {
        match self {
            ResidentCheckpoint::Gguf(_) => None,
            ResidentCheckpoint::Hf(hf) => {
                let enc = &hf.src.descriptor("tok.weight")?.encoding;
                [(QuantEncoding::bf16(), Dtype::BF16), (QuantEncoding::f16(), Dtype::F16)].into_iter().find(|(e, _)| e == enc).map(|(_, t)| t)
            }
        }
    }

    /// Embedding rows `rows` (token ids) of table `name` (`d` wide), `[rows.len(), d]` in the order asked.
    pub fn embed_rows(&self, name: &str, rows: &[usize], d: usize) -> Option<Result<Vec<f32>, String>> {
        match self {
            ResidentCheckpoint::Gguf(mg) => mg.tensor_rows(name, rows, d),
            ResidentCheckpoint::Hf(hf) => hf.src.read_rows(name, rows, d).transpose(),
        }
    }

    /// The embedding table `name` (`d` wide) in host memory as stored, for a table the checkpoint keeps in bf16; `None` for
    /// every other encoding and for a GGUF, whose rows are decoded from the mapping on demand ([`Self::embed_rows`]).
    pub fn host_embedding(&self, name: &str, d: usize) -> Option<Result<HostEmbedding, String>> {
        let ResidentCheckpoint::Hf(hf) = self else { return None };
        let desc = hf.src.descriptor(name)?;
        if desc.encoding != QuantEncoding::bf16() {
            return None;
        }
        let vocab = hf.src.numel(name)? / d;
        let mut bits = Vec::with_capacity(vocab * d);
        // Rows are decoded in parallel, a batch at a time (bounded host staging).
        const ROWS_PER_BATCH: usize = 8192;
        for first in (0..vocab).step_by(ROWS_PER_BATCH) {
            let rows: Vec<usize> = (first..vocab.min(first + ROWS_PER_BATCH)).collect();
            match hf.src.read_rows(name, &rows, d) {
                // A bf16 value widened to f32 has no bits below the high half, so the narrowing loses nothing.
                Ok(Some(f)) => bits.extend(f.iter().map(|v| (v.to_bits() >> 16) as u16)),
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(HostEmbedding { bits, d }))
    }

    /// The final-norm gain, `[d]`.
    pub fn final_norm(&self, name: &str) -> Option<Result<Vec<f32>, String>> {
        match self {
            ResidentCheckpoint::Gguf(mg) => mg.tensor(name),
            ResidentCheckpoint::Hf(hf) => {
                let mut out = Vec::with_capacity(hf.src.numel(name)?);
                hf.src.with_tensor_chunks(name, 1 << 20, &mut |_, chunk| out.extend_from_slice(chunk)).then_some(Ok(out))
            }
        }
    }

    /// The GGUF handle, for the parts that exist only there (the MTP head).
    pub fn gguf(&self) -> Option<&MmapGguf> {
        match self {
            ResidentCheckpoint::Gguf(mg) => Some(mg),
            ResidentCheckpoint::Hf(_) => None,
        }
    }

    /// Whether the checkpoint carries an MTP head this engine can load.
    pub fn has_mtp(&self, cfg: &Qwen35Config) -> bool {
        match self {
            ResidentCheckpoint::Gguf(mg) => crate::int8_gguf_resident::checkpoint_has_mtp(mg, cfg),
            ResidentCheckpoint::Hf(hf) => hf.mtp.is_some(),
        }
    }

    /// The MTP head of an HF directory (`None` for a GGUF, which reads its block through [`Self::gguf`]).
    pub fn mtp_source(&self) -> Option<&PackedSource> {
        match self {
            ResidentCheckpoint::Gguf(_) => None,
            ResidentCheckpoint::Hf(hf) => hf.mtp.as_ref(),
        }
    }

    /// The weight tier that holds every linear of the checkpoint as it is
    /// stored: a GGUF's Q4_K as a K-quant and the rest at int8
    /// ([`gguf::held_tier_of`]), an HF directory's FP8 as FP8 and its
    /// unconverted tensors as bf16 ([`model::ops::stored_tier`]).
    pub fn held_tiers(&self, cfg: &Qwen35Config) -> Result<TierPolicy, String> {
        match self {
            ResidentCheckpoint::Gguf(mg) => crate::int8_gguf_resident::held_tiers(mg, cfg),
            ResidentCheckpoint::Hf(hf) => {
                let mut leaves = Vec::new();
                for (l, ty) in cfg.layer_types().into_iter().enumerate() {
                    for (suffix, n, k) in cfg.layer_leaves(ty) {
                        let name = format!("blocks.{l}.{suffix}");
                        let tier = model::ops::stored_tier(&hf.src, &name).ok_or_else(|| format!("{name}: the checkpoint stores it in an encoding with no resident tier"))?;
                        leaves.push((name, tier, n as u64 * k as u64));
                    }
                }
                if let Some(mtp) = &hf.mtp {
                    let layer = (cfg.full_attention_interval - 1) as usize;
                    for (suffix, n, k) in cfg.layer_leaves(crate::config::LayerType::Full) {
                        let tier = model::ops::stored_tier(mtp, &format!("blocks.{layer}.{suffix}")).ok_or_else(|| format!("{}{suffix}: the checkpoint stores it in an encoding with no resident tier", crate::int8_gguf_resident::MTP_PREFIX))?;
                        leaves.push((format!("{}{suffix}", crate::int8_gguf_resident::MTP_PREFIX), tier, n as u64 * k as u64));
                    }
                }
                Ok(TierPolicy::per_leaf(leaves))
            }
        }
    }
}
