// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **T5-XXL / umT5-XXL encoder** - the text conditioning FLUX.1 needs alongside
//! CLIP-L, and the text tower Wan2.1 video conditions on.
//!
//! One config-driven encoder stack (`T5EncoderModel` in transformers' terms) in
//! two variants:
//!
//! | | [`config::T5Config::xxl`] | [`config::T5Config::umt5_xxl`] |
//! |---|---|---|
//! | used by | FLUX.1 / FLUX.2 | Wan2.1 / 2.2 video |
//! | source | `FLUX.1-*/text_encoder_2/` safetensors | `models_t5_umt5-xxl-enc-bf16.pth` |
//! | vocabulary | 32128 | **256384** (multilingual) |
//! | relative bias | ONE table, shared by 24 blocks | **one table PER block** |
//! | attention mask | none (right pad is attended) | **key padding, 512-token window** |
//! | parameters | 4.762 B | **5.681 B** |
//! | importer | [`import::import_hf`] | [`import::import_wan`] |
//! | goldens | `tools/goldens/t5encoder_dump_reference.py` | `tools/goldens/wan_t5_dump_reference.py` |
//!
//! Both are imported 1:1 with two-way coverage and gated stage by stage.
//!
//! T5 is unlike every decoder already in this workspace, and each difference is
//! a correctness trap rather than a stylistic one. All four are verified in the
//! reference dump (which records the measurements in its manifest) and
//! documented at the point of use in [`model`]:
//!
//! * **RMSNorm without a bias and without mean subtraction**, `eps = 1e-6`,
//!   and **no rescaling of the residual stream**;
//! * a learned **relative position bias** - bucketed on `key - query` and added
//!   to the attention scores of every block; computed once in block 0 for T5
//!   v1.1, and from **one table per block** for umT5 (`shared_pos=False`).
//!   There is **no RoPE and no absolute position embedding**. brain's
//!   `rel_shift` is Transformer-XL's shift of an existing score slab, a
//!   different mechanism (see [`hostbias`]);
//! * **no `1/sqrt(d_kv)` attention scaling** — T5 folds it into the
//!   initialisation. Using the scaled kernel is silently wrong, not a crash
//!   (the dump measures the wrong variant at max|d| 7.0e+01);
//! * a **gated-GELU** FFN (T5 v1.1 / XXL): `gelu_new(wi_0(x)) * wi_1(x)`, with
//!   `wo` untied from the embedding.
//!
//! ## Scope
//!
//! Forward ([`model`]) for both variants; **training** ([`train`]) for the T5
//! v1.1 variant only.
//!
//! **Masking is a config choice, not a global.** FLUX passes no
//! `attention_mask` (`FluxPipeline._get_t5_prompt_embeds`), so right-pad
//! positions are attended as ordinary keys; Wan passes one and pads to 512.
//! Neither is a no-op the way CLIP's causal isolation is - the two dumpers
//! measure 4.5 and 1.5 max|d| respectively on *content* rows between the masked
//! and unmasked runs - so `T5Config::masked` selects between two genuinely
//! different answers and the graph for an unmasked config records no mask step
//! at all.
//!
//! [`train`] refuses a per-block-bias or masked config rather than producing a
//! gradient for a graph it does not implement: its reverse folds ONE
//! `rel_bias.weight` gradient across the block stack and attends over every key.
//! Also not implemented: INT8 and the serving contract. The SentencePiece
//! unigram tokenizer umT5 needs now exists as `data::unigram` (next to the
//! GPT-2/Qwen/CLIP BPEs, per the same rule that put it there);
//! `tests/umt5_parity.rs` gates the two against one golden so the ids brain
//! encodes with and the ids the reference ran on cannot drift.
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
//! **T5-XXL v1.1: 4.762 B parameters = 19.05 GB in fp32.** That still fits one
//! 24 GB card *at the parity shape* - measured: the B=2, T=128 gate runs on a
//! single Tesla P40 with ~1.9 GB of activations. It stops fitting at FLUX's
//! real T=512, where the activations scale to ~7.5 GB (the two shared
//! `[B,64,T,T]` score slabs alone are 134 MB each) and the total passes 26 GB.
//!
//! **umT5-XXL: 5.681 B parameters = 22.72 GB in fp32.** The whole 918 M
//! difference is the embedding table - 224256 extra rows of 4096 - so the
//! vocabulary alone costs **+3.67 GB** (0.53 -> 4.20 GB) before a single block
//! is allocated, and the 24 relative-position tables add 47 KB between them.
//! It does **not** fit a 24 GB card at any shape: Wan's T=512 window needs
//! ~4 GB of activations at B=2 on top, so the umT5 gate is a `BRAIN_DEVICE=cpu`
//! test today. Its per-block bias slabs are another 67 MB each at T=512
//! (1.6 GB across 24 blocks), kept per block so a shared-bias regression is
//! observable - see [`model`].
//!
//! Per-channel symmetric INT8 (`model::int8`, the path `qwen3::q8` and
//! `s3dit::int8` already take) puts the weights at **~4.77 GB / ~5.69 GB** plus
//! ~2.4 MB of scales - comfortably single-card at T=512 for either variant, and
//! the reason INT8 is the first tool here rather than sharding.

pub mod config;
pub mod hostbias;
pub mod import;
pub mod model;
pub mod train;
