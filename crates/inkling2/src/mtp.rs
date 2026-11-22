// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers and the state rollback a looped stack
// needs - for clients serving large models on modest hardware. If your team needs a released drafter wired to its
// target then you can procure our services by sending an email to info@swedishembedded.com.

//! The release's optional drafter (`mtp.safetensors`, `mtp_config`): a DSpark block drafter that reads the target's
//! hidden states at logical depths of the loop, loaded as a [`dflash2::Dflash2`].
//!
//! `mtp_config.target_layer_ids` are visits of the target, the last of them its full depth: the drafter reads the
//! carried state entering each of the other visits and the final normalised row
//! ([`crate::serve::Engine::enable_taps`]). A drafter declared for another width, another vocabulary or depths the loop
//! does not have is refused by name, as are the head types and biases the forward does not implement.

use std::path::Path;

use checkpoint::packed::{HfQuant, PackedReader};
use dflash2::{Dflash2, Dflash2Config, DraftCache, DsparkShape, HfDraft};
use gpu_core::Gpu;

use crate::config::{CheckpointConfig, MtpConfig};
use crate::Inkling2Config;

/// The drafter's file beside the checkpoint.
pub const MTP_FILE: &str = "mtp.safetensors";

/// The drafter's shape for the target `target` from its declaration `mtp`.
pub fn shape(mtp: &MtpConfig, target: &Inkling2Config) -> Result<DsparkShape, String> {
    let bad = |field: &str, why: String| Err(format!("inkling2 mtp_config.{field}: {why}"));
    if mtp.markov_head_type != "rnn_hidden" {
        return bad("markov_head_type", format!("{:?}; the drafter implements the recurrent head (\"rnn_hidden\")", mtp.markov_head_type));
    }
    if mtp.q_bias || mtp.o_bias {
        return bad("q_bias/o_bias", "the drafter's attention has no biases".into());
    }
    if mtp.hidden_size != target.hidden {
        return bad("hidden_size", format!("{}, the target's width is {}", mtp.hidden_size, target.hidden));
    }
    let visits = crate::model::visit_plan(target)?.visits().len();
    let depths: Vec<usize> = mtp.target_layer_ids.iter().map(|&v| v as usize).collect();
    if depths.is_empty() || depths.windows(2).any(|w| w[0] >= w[1]) || depths.last() != Some(&visits) {
        return bad("target_layer_ids", format!("{depths:?}: ascending visits of the target followed by its full depth {visits}"));
    }
    let local: Vec<usize> = mtp.local_layer_ids.iter().map(|&l| l as usize).collect();
    Ok(DsparkShape {
        d_model: mtp.hidden_size,
        n_layers: mtp.num_hidden_layers as usize,
        n_heads: mtp.num_attention_heads,
        n_kv_heads: mtp.num_key_value_heads,
        head_dim: mtp.head_dim,
        d_ff: mtp.intermediate_size,
        eps: mtp.rms_norm_eps as f32,
        rope_theta: mtp.rope_theta as f32,
        local_layers: local,
        window: mtp.sliding_window_size,
        block_size: mtp.block_size,
        markov_rank: mtp.markov_rank,
        target_depths: depths,
        vocab: target.vocab,
    })
}

/// The drafter `dir` declares for `target` (`config.json`'s `mtp_config`), or `None` when it declares none.
pub fn draft_config(dir: &Path, target: &Inkling2Config) -> Result<Option<Dflash2Config>, String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let raw: CheckpointConfig = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let Some(mtp) = raw.mtp_config.as_ref() else { return Ok(None) };
    Ok(Some(Dflash2Config::dspark(&shape(mtp, target)?)?))
}

/// Device bytes the drafter of `dir` takes for `slots` sequences of up to `cap` positions: its weights as stored and its
/// K/V caches. What a memory plan charges before anything loads.
pub fn device_bytes(dir: &Path, target: &Inkling2Config, slots: u32, cap: u32) -> Result<u64, String> {
    let cfg = draft_config(dir, target)?.ok_or_else(|| format!("inkling2: {} declares no mtp_config", dir.join("config.json").display()))?;
    let file = dir.join(MTP_FILE);
    let reader = PackedReader::open(&file).map_err(|e| format!("inkling2: the drafter {}: {e}", file.display()))?.with_declaration(HfQuant::from_dir(dir)?);
    let weights = HfDraft::from_reader(reader).device_weight_bytes(&cfg);
    Ok(weights + DraftCache::dense(&cfg, slots, cap).bytes(&cfg, slots, cap))
}

/// Load the drafter of `dir` (`config.json` and [`MTP_FILE`], read under the release's quantization declaration when it
/// has one) onto a device of its own (the device the calling thread is scoped to), with a cache slot for each of `slots`
/// sequences of up to `cap` positions.
pub fn load(dir: &Path, target: &Inkling2Config, slots: u32, cap: u32) -> Result<Dflash2, String> {
    let cfg = draft_config(dir, target)?.ok_or_else(|| format!("inkling2: {} declares no mtp_config", dir.join("config.json").display()))?;
    let file = dir.join(MTP_FILE);
    let reader = PackedReader::open(&file).map_err(|e| format!("inkling2: the drafter {}: {e}", file.display()))?.with_declaration(HfQuant::from_dir(dir)?);
    let gpu = Gpu::new(&dflash2::pipelines(&cfg));
    let cache = DraftCache::dense(&cfg, slots, cap);
    Dflash2::load_with(gpu, cfg, Box::new(HfDraft::from_reader(reader)), slots, cap, cache)
}
