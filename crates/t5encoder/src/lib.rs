// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **T5-XXL encoder** — the text conditioning FLUX.1 needs alongside CLIP-L.
//!
//! One config-driven encoder stack (`T5EncoderModel` in transformers' terms),
//! imported 1:1 from the released `FLUX.1-*/text_encoder_2/` safetensors shards
//! and gated stage by stage against goldens dumped by
//! `tools/goldens/t5encoder_dump_reference.py`.
//!
//! T5 is unlike every decoder already in this workspace, and each difference is
//! a correctness trap rather than a stylistic one. All four are verified in the
//! reference dump (which records the measurements in its manifest) and
//! documented at the point of use in [`model`]:
//!
//! * **RMSNorm without a bias and without mean subtraction**, `eps = 1e-6`,
//!   and **no rescaling of the residual stream**;
//! * a learned **relative position bias** — bucketed on `key - query`, computed
//!   once in block 0 and added to the attention scores of every block. There is
//!   **no RoPE and no absolute position embedding**. brain's `rel_shift` is
//!   Transformer-XL's shift of an existing score slab, a different mechanism
//!   (see [`hostbias`]);
//! * **no `1/sqrt(d_kv)` attention scaling** — T5 folds it into the
//!   initialisation. Using the scaled kernel is silently wrong, not a crash
//!   (the dump measures the wrong variant at max|d| 7.0e+01);
//! * a **gated-GELU** FFN (T5 v1.1 / XXL): `gelu_new(wi_0(x)) * wi_1(x)`, with
//!   `wo` untied from the embedding.
//!
//! ## Scope
//!
//! Forward ([`model`]) + **training** ([`train`]), and **unmasked** — which is
//! the FLUX contract:
//! `FluxPipeline._get_t5_prompt_embeds` passes no `attention_mask`, so right-pad
//! positions are attended as ordinary keys. Unlike CLIP's causal isolation this
//! is *not* a no-op (the dumper measures a 4.5 max|d| difference on content rows
//! between the masked and unmasked runs), so a masked variant would be a real
//! feature; it is not implemented. Also not implemented: the sentencepiece
//! tokenizer (ids come from the goldens; the tokenizer belongs in `crates/data`
//! next to the GPT-2/Qwen/CLIP BPEs), INT8, and the serving contract.
//!
//! [`train`] adds the SSA forward + hand-written reverse over **every** tensor
//! in the manifest (including the learned relative-position bias), gated by
//! `gradcheck::check_t5` on both the P40 and `backend-cpu`. It is honest about
//! two limits: it caches the softmax probabilities **per block** (the inference
//! graph shares one slab, which the reverse cannot read), so T = 512 needs
//! `block::chunked_bidir_bwd`'s per-chunk recompute before a real XXL finetune;
//! and it re-records the forward rather than sharing [`model`]'s private step
//! builder.
//!
//! ## Size
//!
//! 4.762 B parameters: **19.05 GB in fp32**. That still fits one 24 GB card
//! *at the parity shape* — measured: the B=2, T=128 gate runs on a single
//! Tesla P40 with ~1.9 GB of activations. It stops fitting at FLUX's real
//! T=512, where the activations scale to ~7.5 GB (the two shared `[B,64,T,T]`
//! score slabs alone are 134 MB each) and the total passes 26 GB.
//! Per-channel symmetric INT8 (`model::int8`, the path `qwen3::q8` and
//! `s3dit::int8` already take) puts the weights at **~4.77 GB** plus ~2.4 MB
//! of scales — comfortably single-card at T=512, and the reason INT8 is the
//! first tool here rather than sharding.

pub mod config;
pub mod hostbias;
pub mod import;
pub mod model;
pub mod train;
