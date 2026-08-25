// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The real 22B AV checkpoint as something a generation loop can call: load
//! [`crate::dit::av_dit_tensor_manifest`]'s tensors off a GGUF, hand them to
//! [`LtxAvDit`], and run ONE joint audio+video forward per denoise step.
//!
//! The whole checkpoint is audio-visual - two thirds of its tensors are the
//! audio stream and the bidirectional A<->V cross-attention - so producing
//! sound is not a matter of adding a decoder. It is a matter of running the
//! model that is already in the file, instead of only its video half.
//!
//! ## Why this holds the model as host fp32, and what that costs
//!
//! The video-only production path ([`crate::dit::forward_q_streamed_in`])
//! keeps only the non-block "head" tensors resident and streams each
//! transformer block from the GGUF, quantized to int8, cached per checkpoint
//! and left resident on the card between steps. **No such path exists for the
//! audio-extended block.** `LtxAvBlockQ`, an AV `CachedQBlockWeights`, and an
//! AV `DitSession` are all absent, and every one of them lives in
//! `crate::block`/`crate::dit`.
//!
//! So this module takes the only route the current code offers: the eager,
//! host-fp32 [`LtxAvDit`] that the tiny-config parity suite already proves
//! (`crates/ltxv/tests/av_dit_parity.rs`), driven at the real config through
//! its SHARDING entry points rather than [`LtxAvDit::forward`] - because
//! [`LtxAvDit::run_stage_forward`] discards the per-block activation taps
//! that `forward` returns for parity bisection, and at a real token count
//! those taps are the difference between a forward that fits and one that
//! does not.
//!
//! The consequences are real and are not hidden here: the model occupies
//! host memory proportional to its full fp32 expansion rather than its
//! quantized size, and every forward re-uploads every block's fp32 weights to
//! the device, because `LtxAvBlock` is constructed per layer per call. That
//! makes an audio-visual generation markedly more expensive per step than the
//! video-only one at the same shape. It is correct - the same op sequence,
//! the same weights, at higher precision than the int8 video path - and it is
//! the honest cost of the AV stream having no streamed/quantized/resident
//! implementation yet. Closing that gap is what would make audio cheap; it is
//! not a change this module can make from outside `crate::block`.
//!
//! [`AvWeights::fits_in_host_memory`] checks the machine BEFORE anything is
//! read, so a box that cannot hold the expansion is told so in one line
//! instead of being discovered by the OOM killer part way through a load.

use std::time::Instant;

use vae::blocks::Tensors;

use crate::config::LtxAvDitConfig;
use crate::dit::{av_dit_tensor_manifest, AvDitBatch, LtxAvDit};

/// The AV DiT's weights, expanded to host fp32.
pub struct AvWeights {
    pub cfg: LtxAvDitConfig,
    pub tensors: Tensors,
}

/// How many f32 values [`av_dit_tensor_manifest`] describes at `cfg` - the
/// host expansion this module needs, derived from the manifest rather than
/// from any number written down here.
pub fn host_floats(cfg: &LtxAvDitConfig) -> u64 {
    av_dit_tensor_manifest(cfg).iter().map(|(_, s)| s.iter().product::<usize>() as u64).sum()
}

impl AvWeights {
    /// Refuse before reading anything if this machine cannot hold the fp32
    /// expansion, with the two numbers that decide it.
    ///
    /// A margin is kept over the bare weight bytes because a forward also
    /// needs both streams' activations, each block's own output copy, and
    /// whatever the caller is holding - a load that exactly fits the weights
    /// and nothing else is a load that dies later, further from the cause.
    pub fn fits_in_host_memory(cfg: &LtxAvDitConfig) -> Result<(), String> {
        let want = host_floats(cfg) * 4;
        let Some(avail) = available_host_bytes() else { return Ok(()) };
        let margin = want / 8;
        if avail < want + margin {
            return Err(format!(
                "ltxv: audio generation needs the audio+video DiT resident as host fp32 ({} GiB, plus headroom) but only {} GiB is available - the AV stream has no streamed/quantized path yet (see crate::av_stream's module doc)",
                want / (1 << 30),
                avail / (1 << 30)
            ));
        }
        Ok(())
    }

    /// Read every manifest tensor from `src` into host fp32.
    ///
    /// Streams one tensor at a time out of the source's own dequantizer, so
    /// the transient peak above the final map is a single tensor - the same
    /// discipline `crate::dit::load_head_tensors_from_source` uses, applied to
    /// the whole manifest rather than its non-block part.
    pub fn load(src: &dyn checkpoint::TensorSource, cfg: LtxAvDitConfig) -> Result<AvWeights, String> {
        AvWeights::fits_in_host_memory(&cfg)?;
        let manifest = av_dit_tensor_manifest(&cfg);
        let t0 = Instant::now();
        let mut tensors = Tensors::new();
        for (name, shape) in manifest {
            let mut data = Vec::new();
            if !src.with_tensor(&name, &mut |d| data = d.to_vec()) {
                return Err(format!("ltxv av: the checkpoint has no tensor {name}"));
            }
            let want: usize = shape.iter().product();
            if data.len() != want {
                return Err(format!("ltxv av: {name} has {} values, expected {want}", data.len()));
            }
            tensors.insert(name, (shape, data));
        }
        tracing::info!(tensors = tensors.len(), secs = t0.elapsed().as_secs_f32(), gib = host_floats(&cfg) * 4 / (1 << 30), "audio+video DiT expanded to host fp32");
        Ok(AvWeights { cfg, tensors })
    }
}

/// `MemAvailable` from `/proc/meminfo`, in bytes - what the kernel says can
/// be had without swapping, which is the number that decides whether a large
/// allocation survives. `None` on any platform or format that does not
/// provide it, which makes the check above skip rather than guess.
fn available_host_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Both streams' inputs for one joint forward. The two are deliberately one
/// struct: the A<->V cross-attention means neither stream's answer exists
/// without the other's input, so there is no such thing as "the audio
/// forward" to pass separately.
pub struct AvStepInputs<'a> {
    pub v_latent: &'a [f32],
    pub v_timesteps: &'a [f32],
    pub v_positions: &'a [f32],
    pub v_keyframes_mask: &'a [f32],
    pub tv: usize,
    pub a_latent: &'a [f32],
    pub a_timesteps: &'a [f32],
    pub a_positions: &'a [f32],
    pub ta: usize,
    /// The schedule's scalar sigma for this step. Each stream's AV gate is
    /// driven by the OTHER stream's sigma; both streams run one schedule in
    /// lockstep (`ltx_pipelines.utils.samplers.euler_denoising_loop` steps
    /// them together off one `sigmas` tensor), so one value serves both.
    pub sigma: f32,
    pub context_len: usize,
    pub context_valid: &'a [f32],
}

/// The real AV checkpoint's joint denoiser.
pub struct AvDenoiser {
    dit: LtxAvDit,
}

impl AvDenoiser {
    pub fn new(w: AvWeights, device: Option<&str>) -> AvDenoiser {
        AvDenoiser { dit: LtxAvDit::new(w.cfg, w.tensors, device) }
    }

    /// One joint forward: `(video velocity, audio velocity)`.
    ///
    /// Routed through `load_shard_batch` + `run_stage_forward` +
    /// `take_stage_output` rather than [`LtxAvDit::forward`] - a whole-model
    /// `Shard` is `embed && head`, so the stage runs patchify, both adaLN
    /// tables, all four RoPE tables, both embeddings connectors, every block
    /// and both output stages exactly as `forward` does, but without
    /// retaining a per-block tap set nothing here reads.
    ///
    /// `v_context` and `a_context` are the two streams' OWN text projections,
    /// not one caption reused: the checkpoint carries
    /// `text_embedding_projection.{video,audio}_aggregate_embed` side by
    /// side, and each stream's embeddings connector - which
    /// `LtxAvDit::run_stage_forward` runs internally - is built for its own
    /// head's output width.
    pub fn forward(&self, i: &AvStepInputs, v_context: &[f32], a_context: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.dit.load_shard_batch(AvDitBatch {
            v_latent: i.v_latent.to_vec(),
            v_timesteps: i.v_timesteps.to_vec(),
            v_positions: i.v_positions.to_vec(),
            v_keyframes_mask: i.v_keyframes_mask.to_vec(),
            v_context: v_context.to_vec(),
            v_context_len: i.context_len,
            tv: i.tv,
            v_sigma: i.sigma,
            v_context_valid: i.context_valid.to_vec(),
            a_latent: i.a_latent.to_vec(),
            a_timesteps: i.a_timesteps.to_vec(),
            a_positions: i.a_positions.to_vec(),
            a_context: a_context.to_vec(),
            a_context_len: i.context_len,
            ta: i.ta,
            a_sigma: i.sigma,
            a_context_valid: i.context_valid.to_vec(),
            v_target: None,
            a_target: None,
        });
        self.dit.run_stage_forward();
        self.dit.take_stage_output()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host expansion is derived from the manifest, so a config change
    /// that grew the model could never leave this check reading a stale size.
    /// Pinned against the tiny config, whose manifest is small enough to sum
    /// by hand in a test rather than asserted against a magnitude.
    #[test]
    fn host_floats_is_the_manifests_own_sum() {
        let cfg = LtxAvDitConfig::tiny();
        let by_hand: u64 = av_dit_tensor_manifest(&cfg).iter().map(|(_, s)| s.iter().product::<usize>() as u64).sum();
        assert_eq!(host_floats(&cfg), by_hand);
        assert!(by_hand > 0);
    }

    /// A tiny config must never be refused for memory on any machine that can
    /// run this test at all - the guard is there to catch the 22B case early,
    /// not to become a flaky precondition on small ones.
    #[test]
    fn the_memory_guard_admits_a_tiny_config() {
        AvWeights::fits_in_host_memory(&LtxAvDitConfig::tiny()).expect("a tiny AV config must fit anywhere");
    }
}
