// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for turning generative video models
// into interactive, replayable simulators for its clients. If your team needs
// expertise in action-conditioned world models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! GenieRedux as a playable [`wm_core::WorldModel`].
//!
//! The session state is a window of the last `max_context_frames` frames as
//! tokens plus the action taken at each. One [`WorldModel::step`]:
//!
//! 1. sets the newest frame's action (the one passed in);
//! 2. runs the dynamics over the frames the cache has not seen
//!    ([`dynamics_forward_extend`]: normally just the newest frame) and takes
//!    the logits of the newest frame's positions, which predict the next frame
//!    (the reference's `cat(prime, mask)[:, :-1]` drop of the last frame, see
//!    [`crate::maskgit`]);
//! 3. samples the next frame's tokens with [`maskgit_sample_with`];
//! 4. decodes the new frame with the tokenizer decoder, itself cached
//!    ([`decode_extend`]);
//! 5. appends the tokens to the window. Generated tokens are not re-encoded:
//!    the token window is the model's state, as in the reference loop.
//!
//! Context frames given to [`WorldModel::reset`] are pixels in `[0, 1]`,
//! encoded as one video by the tokenizer (so its first frame is the window's
//! first frame); more frames than the window keep the newest. An empty context
//! is one mid-grey frame. `ctx_actions[f]` is the action taken at context frame
//! `f` (missing entries are action 0); the newest frame's entry is replaced by
//! the first `step`'s action.
//!
//! When the window is full, the oldest frame is dropped and both caches are
//! rebuilt on the next step, because the dynamics' positional embedding is
//! absolute and the decoder's first frame changes; the incremental cost
//! therefore applies while the window grows.
//!
//! Randomness: every step draws its sampler seed from a stream seeded at
//! construction and re-seeded by `reset`/`reset_initial`, so the same context
//! and actions replay the same frames bit for bit. The seed is a constructor
//! argument; there is no default. With the checkpoint's single inference pass
//! the sampler is greedy and the seed has no effect.
//!
//! An action `>= num_actions()` is clamped to the last action, as DIAMOND does.

use crate::import::{DynamicsConfig, TokenizerConfig};
use crate::maskgit::SplitMix64;
use crate::{
    bias, decode_extend, dynamics_forward_extend, dynamics_spatial_bias, maskgit_sample_with,
    tokenizer_encode, DynamicsWeights, MaskgitSettings, StCache, TokenizerWeights,
};
use gpu_core::Gpu;
use wm_core::WorldModel;

/// Token grid of the released CoinRun checkpoints: 64x64 frames, 4x4 patches.
pub const COINRUN_GRID: (u32, u32) = (16, 16);

/// Context window the CoinRun parity run primes the dynamics with (the
/// checkpoint's position table allows up to 15 frames of 256 tokens).
pub const COINRUN_CONTEXT_FRAMES: usize = 4;

/// A GenieRedux tokenizer + guided-MaskGIT dynamics pair playing one session.
pub struct GenieReduxWorldModel {
    gpu: Gpu,
    tw: TokenizerWeights,
    tc: TokenizerConfig,
    dw: DynamicsWeights,
    dc: DynamicsConfig,
    grid: (u32, u32),
    max_context_frames: usize,
    settings: MaskgitSettings,
    seed: u64,
    dyn_sbias: Vec<f32>,
    tok_sbias: Vec<f32>,
    tokens: Vec<u32>,
    actions: Vec<u32>,
    init_tokens: Vec<u32>,
    init_actions: Vec<u32>,
    rng: SplitMix64,
    dyn_cache: StCache,
    dec_cache: StCache,
}

impl GenieReduxWorldModel {
    /// Build a session over loaded weights. `grid` is the token grid `(h, w)`
    /// (frames are `h * patch` by `w * patch` pixels); `max_context_frames` is
    /// the window. Fails when the two checkpoints disagree (codebook size or
    /// width) or the window does not fit the dynamics' position table.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: Gpu, tw: TokenizerWeights, tc: TokenizerConfig, dw: DynamicsWeights, dc: DynamicsConfig,
        grid: (u32, u32), max_context_frames: usize, settings: MaskgitSettings, seed: u64,
    ) -> Result<Self, String> {
        let hw = (grid.0 * grid.1) as usize;
        if hw == 0 || max_context_frames == 0 {
            return Err("grid and max_context_frames must be non-zero".into());
        }
        if settings.inference_steps == 0 {
            return Err("inference_steps must be at least 1".into());
        }
        if dc.n_codes != tc.n_codes || dc.code_dim != tc.code_dim || dc.dim != tc.dim {
            return Err(format!(
                "tokenizer (codes {}, code_dim {}, dim {}) and dynamics (codes {}, code_dim {}, dim {}) disagree",
                tc.n_codes, tc.code_dim, tc.dim, dc.n_codes, dc.code_dim, dc.dim
            ));
        }
        if max_context_frames * hw > dc.max_seq_len as usize {
            return Err(format!(
                "{max_context_frames} frames of {hw} tokens exceed the dynamics position table ({})",
                dc.max_seq_len
            ));
        }
        let dyn_sbias = dynamics_spatial_bias(&dw, &dc, grid.0, grid.1);
        let tok_sbias = bias::cpb_bias(&tw.cpb_net, grid.0 as usize, grid.1 as usize, tc.heads as usize);
        let mut m = GenieReduxWorldModel {
            gpu, tw, tc, dw, dc, grid, max_context_frames, settings, seed, dyn_sbias, tok_sbias,
            tokens: Vec::new(), actions: Vec::new(), init_tokens: Vec::new(), init_actions: Vec::new(),
            rng: SplitMix64(seed), dyn_cache: StCache::new(), dec_cache: StCache::new(),
        };
        m.reset(&[], &[]);
        Ok(m)
    }

    fn hw(&self) -> usize {
        (self.grid.0 * self.grid.1) as usize
    }

    fn frame_len(&self) -> usize {
        let (c, h, w) = self.frame_shape();
        (c * h * w) as usize
    }
}

impl WorldModel for GenieReduxWorldModel {
    fn frame_shape(&self) -> (u32, u32, u32) {
        (self.tc.channels, self.grid.0 * self.tc.patch, self.grid.1 * self.tc.patch)
    }

    fn num_actions(&self) -> u32 {
        self.dc.num_actions
    }

    fn reset(&mut self, ctx_frames: &[f32], ctx_actions: &[u32]) {
        let fl = self.frame_len();
        let (c, hh, ww) = self.frame_shape();
        let available = ctx_frames.len() / fl;
        let skip = available.saturating_sub(self.max_context_frames);
        let (frames, n) = if available == 0 {
            (vec![0.5f32; fl], 1)
        } else {
            (ctx_frames[skip * fl..available * fl].to_vec(), available - skip)
        };
        // frame-major [n][c][H][W] -> the tokenizer's channels-first [c][n][H][W]
        let plane = (hh * ww) as usize;
        let mut video = vec![0.0f32; frames.len()];
        for f in 0..n {
            for ch in 0..c as usize {
                let src = (f * c as usize + ch) * plane;
                let dst = (ch * n + f) * plane;
                video[dst..dst + plane].copy_from_slice(&frames[src..src + plane]);
            }
        }
        let tbias = bias::alibi_bias(self.tc.heads as usize, n);
        let (_, tokens) = tokenizer_encode(
            &self.gpu, &video, &self.tw, 1, c, n as u32, hh, ww, self.tc.patch, self.tc.dim,
            self.tc.heads, self.tc.head_dim, self.tc.code_dim, self.tc.n_codes, &self.tok_sbias, &tbias,
        );
        let last = self.dc.num_actions - 1;
        self.init_actions = (0..n)
            .map(|f| ctx_actions.get(if available == 0 { f } else { skip + f }).copied().unwrap_or(0).min(last))
            .collect();
        self.init_tokens = tokens;
        self.reset_initial();
    }

    fn reset_initial(&mut self) {
        self.tokens = self.init_tokens.clone();
        self.actions = self.init_actions.clone();
        self.rng = SplitMix64(self.seed);
        self.dyn_cache.clear();
        self.dec_cache.clear();
    }

    fn step(&mut self, action: u32) -> Vec<f32> {
        let (hw, nc) = (self.hw(), self.dc.n_codes as usize);
        let (h, w) = self.grid;
        let t = self.tokens.len() / hw;
        self.actions[t - 1] = action.min(self.dc.num_actions - 1);

        // Dynamics: the frames the cache has not seen (the newest one, once warm).
        let from = self.dyn_cache.frames();
        let one_hot = one_hot(&self.actions[from..t], self.dc.num_actions as usize);
        let logits = dynamics_forward_extend(
            &self.gpu, &mut self.dyn_cache, &self.tokens[from * hw..], &one_hot, &self.dw, &self.dc,
            h, w, &self.dyn_sbias,
        );
        let newest = &logits[logits.len() - hw * nc..];
        let new_tokens = maskgit_sample_with(hw, self.dc.n_codes, &self.settings, self.rng.next_u64(), |_| newest.to_vec());

        // The model sees the window without the frame being generated, so every
        // pass of the sampler reads the same logits (see `crate::maskgit`).
        self.tokens.extend_from_slice(&new_tokens);
        self.actions.push(0);

        let from = self.dec_cache.frames();
        let pixels = decode_extend(
            &self.gpu, &mut self.dec_cache, &self.tokens[from * hw..], &self.tw, &self.tc, h, w, &self.tok_sbias,
        );
        let fl = self.frame_len();
        let frame = pixels[pixels.len() - fl..].iter().map(|v| v.clamp(0.0, 1.0)).collect();

        if self.tokens.len() / hw > self.max_context_frames {
            self.tokens.drain(..hw);
            self.actions.remove(0);
            self.dyn_cache.clear();
            self.dec_cache.clear();
        }
        frame
    }
}

/// One-hot rows `[actions.len(), num_actions]`.
fn one_hot(actions: &[u32], num_actions: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; actions.len() * num_actions];
    for (i, &a) in actions.iter().enumerate() {
        v[i * num_actions + a as usize] = 1.0;
    }
    v
}
