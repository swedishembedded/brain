// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rolling-window arithmetic for a clip longer than one denoising window can
//! hold: how the request is cut into windows, and what crosses each seam.
//!
//! Swedish Embedded AB implements rolling-window latent video diffusion for
//! its clients. If your team needs a generative video pipeline whose motion
//! survives a window boundary, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Two callers, one plan. [`crate::pipeline::generate_long`] windows a clip it
//! is INVENTING; [`crate::pipeline::upscale`] windows a clip that already
//! exists, whose refinement will not fit one pass. The token ceiling, the
//! carried prefix and the emitted-frame arithmetic are the same question in
//! both, so they are the same code - the only thing refinement adds is
//! [`Window::source_first_frame`], which says where in the input clip a window
//! reads from, and [`fitted_context`], which says what a four-times-denser
//! grid can afford to carry.
//!
//! # What crosses a seam, and why it is not a picture
//!
//! Chaining windows by decoding window `n`, taking its last RGB frame and
//! VAE-encoding that frame as window `n+1`'s conditioning still is continuous
//! in POSITION and discontinuous in VELOCITY: a single frame carries no
//! information about what was moving, in which direction, or how fast, so the
//! model re-invents the motion at every boundary. That is the defect this
//! module exists to remove.
//!
//! What crosses a seam here is [`carry_tail`]'s output: the last
//! [`CONTEXT_LATENT_FRAMES`] latent frames of window `n`'s own final denoised
//! latent, sliced out before anything was ever decoded, and pinned at the
//! head of window `n+1`'s sequence as clean content at sigma 0 - the same
//! `denoise_mask == 0` / `timesteps_from_mask` freezing
//! [`crate::pipeline`] already applies to one latent frame for
//! `--start-frame` image conditioning. Eight latent frames span 57 pixel
//! frames of real motion history, which is what the model extrapolates from.
//!
//! # Where the context size comes from
//!
//! [`CONTEXT_LATENT_FRAMES`] is the reference's own number for exactly this
//! task, not a derivation of ours: `packages/ltx-trainer/configs/
//! video_extend_lora.yaml`'s `training_strategy.video.conditions[0]` is
//! `{type: prefix, temporal_boundary: 8}`, and that file's own comment spells
//! out both halves of the arithmetic - "For prefix conditioning, N latent
//! frames correspond to (N - 1) * 8 + 1 pixel frames. temporal_boundary=8
//! means 57 pixel frames are used as prefix" - with validation samples that
//! repeat it as `num_frames: 57`. `ltx_trainer.training_strategies.flexible.
//! PrefixConditionConfig` documents `temporal_boundary` as "Number of
//! temporal units for prefix region. For video: number of latent frames", and
//! `_compute_temporal_mask` places those units at the FRONT of the token
//! sequence (`mask[:, :num_tokens] = 1.0`, `num_tokens = num_frames *
//! tokens_per_frame`), which is the layout [`window_plan`] produces.
//!
//! **The VAE's temporal receptive field is NOT what sets this, and cannot
//! be.** That was the first hypothesis and it is dead: this checkpoint's
//! decoder is `causal_decoder: false`, so every one of its 42 kernel-3
//! temporal convolutions pads symmetrically. Summing them at the temporal
//! resolution each runs at (6 convs at the latent grid, 5 at the 2:1 grid, 9
//! at 4:1, 22 at 8:1) puts decoding one latent frame in a window of roughly `6 + 5/2 +
//! 9/4 + 22/8 = 13.5` latent frames on EACH side - an exact index walk gives
//! +/-14. A rolling window cannot supply 14 latent frames of lookahead at
//! all, at any cost, because those frames do not exist yet. Nothing this
//! module does makes a seam decode *exactly*; [`crate::vae3d::LtxVaeTiling`]'s
//! own temporal overlap (3 latent frames, upstream's `_CONV_AUTO_FRAMES`)
//! already carries the same admission for the same reason. What
//! [`CONTEXT_LATENT_FRAMES`] buys is the DIFFUSION model's motion history,
//! which is a conditioning question, not a convolution one - and there the
//! reference has an answer and it is 8.
//!
//! The DiT imposes no additional minimum: its attention is global over the
//! whole window (`crate::block`), so a frozen prefix is visible to every
//! generated token regardless of how far away it sits.
//!
//! # Where a seam is deliberately NOT crossed
//!
//! Everything above keeps one SHOT continuous. A request can also be a
//! sequence of [`Scene`]s ([`scene_plan`]), and there the carry is what has to
//! stop: a scene conditioned on the previous scene's real content is not free
//! to become a different subject, setting or action. So a scene boundary
//! resets the context to nothing - the new scene's first window is an ordinary
//! `context == 0` window, exactly like the first window of a whole plan - and
//! the caller drives what happens next with that scene's own prompt.

/// The clean latent-frame prefix each continuation window carries from its
/// predecessor - the reference's `temporal_boundary` for video extension.
/// See this module's doc for the citation and for the hypothesis it replaced.
pub const CONTEXT_LATENT_FRAMES: usize = 8;

/// [`CONTEXT_LATENT_FRAMES`] in pixel frames, `(N - 1) * 8 + 1` - the
/// reference's own `num_frames: 57` for the same prefix. Exposed because a
/// caller choosing a context asks in the units its `--frames` are in.
pub const CONTEXT_FRAMES: usize = 1 + 8 * (CONTEXT_LATENT_FRAMES - 1);

/// The largest per-window video-token count [`window_plan`] will plan a
/// window at.
///
/// **Derived the same way [`crate::pipeline::REFINE_MAX_TOKENS`] is, then
/// pinned to a measured point rather than left at the derivation.** The DiT's
/// per-forward adaLN table is `[t, 9 * inner_dim]` fp32 - 147456 bytes per
/// token at the real checkpoint's `inner_dim = 4096` - against a
/// `max_storage_buffer_binding_size` this box's Tesla P40 reports as 2047
/// MiB, which that one table crosses at `t ~= 14556`. The largest window this
/// crate has a RECORDED real generation at is 113 frames at 1280x704: 15
/// latent frames x 22 x 40 = **13200** tokens. This constant is that number,
/// so every window shape already known to run keeps running unsplit and no
/// window is ever planned at a token count nothing has ever run at. Where in
/// `(13200, 14556)` the real limit sits is not measured and this constant
/// does not pretend to know.
///
/// It is not [`crate::pipeline::SINGLE_STAGE_MAX_TOKENS`], which is a QUALITY
/// ceiling on building structure from noise in ONE stage; a window above that
/// takes the reference's two-stage shape exactly as a single-window request
/// of the same size already does.
///
/// `BRAIN_LTXV_LONGFORM_MAX_TOKENS` overrides it, which is also how a card
/// with a different binding size gets a usable plan.
pub const LONGFORM_MAX_TOKENS: usize = 13200;

/// [`LONGFORM_MAX_TOKENS`], or `BRAIN_LTXV_LONGFORM_MAX_TOKENS` when it names
/// a usable number. A value that does not parse, or is zero, is ignored with
/// a warning rather than silently taken as "no windows are legal".
pub fn max_window_tokens_from_env() -> usize {
    match std::env::var("BRAIN_LTXV_LONGFORM_MAX_TOKENS").ok().as_deref().map(str::trim) {
        Some(v) if !v.is_empty() => match v.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                tracing::warn!(value = v, "BRAIN_LTXV_LONGFORM_MAX_TOKENS is not a positive integer; using the built-in ceiling");
                LONGFORM_MAX_TOKENS
            }
        },
        _ => LONGFORM_MAX_TOKENS,
    }
}

/// One denoising window of a long-form generation.
///
/// The window's token sequence is `context + new` latent frames long: the
/// first `context` of them are the previous window's own final latent frames,
/// frozen at sigma 0, and the remaining `new` are what this window actually
/// generates. The first window has `context == 0` and is an ordinary
/// generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    /// Clean latent frames carried in from the previous window, at the head
    /// of the sequence. `0` for the first window.
    pub context: usize,
    /// Latent frames this window denoises from noise and contributes to the
    /// output.
    pub new: usize,
    /// The output pixel frame this window's first emitted frame becomes.
    pub first_frame: usize,
}

impl Window {
    /// Latent frames in this window's token sequence - what its token count
    /// and its VAE decode are both sized by.
    pub fn latent_frames(&self) -> usize {
        self.context + self.new
    }

    /// Video tokens one DiT forward over this window costs.
    pub fn tokens(&self, lh: usize, lw: usize) -> usize {
        self.latent_frames() * lh * lw
    }

    /// Pixel frames a VAE decode of the WHOLE window produces - always
    /// `1 + 8k`, because the decoder is handed the window's full latent.
    ///
    /// The carried context is decoded too, and has to be: a latent frame
    /// cannot be decoded without the frames around it (see this module's doc
    /// on the decoder's receptive field). Its pixels are then dropped -
    /// [`Self::dropped_frames`] of them - not emitted twice and never
    /// re-encoded.
    pub fn decoded_frames(&self) -> usize {
        1 + 8 * (self.latent_frames() - 1)
    }

    /// Pixel frames this window contributes to the output clip.
    ///
    /// The first window emits everything it decoded. A continuation window
    /// emits `8` frames per NEW latent frame - not `1 + 8 * new` - because
    /// the "+1" of the `1 + 8k` rule is the decode's own first pixel frame,
    /// which belongs to the carried context and was already emitted by the
    /// window that generated it.
    pub fn emitted_frames(&self) -> usize {
        if self.context == 0 {
            1 + 8 * (self.new - 1)
        } else {
            8 * self.new
        }
    }

    /// Leading decoded pixel frames that belong to the carried context and
    /// are discarded.
    pub fn dropped_frames(&self) -> usize {
        self.decoded_frames() - self.emitted_frames()
    }

    /// The first frame of the SOURCE clip this window's whole decode covers,
    /// for a plan applied to a clip that already exists
    /// ([`crate::pipeline::upscale`]) rather than to frames being generated.
    ///
    /// A generating window has no source: it invents [`Self::new`] latent
    /// frames and drops the leading pixels its carried context decoded to. A
    /// window refining an existing clip has to READ those leading pixels from
    /// somewhere, and the somewhere is exactly [`Self::dropped_frames`] before
    /// its own first emitted frame - which makes the range it reads end where
    /// its own output ends.
    pub fn source_first_frame(&self) -> usize {
        self.first_frame - self.dropped_frames()
    }
}

/// Latent frames one `max_tokens` forward holds on an `lh` x `lw` grid.
///
/// The one line [`window_plan`] and [`fitted_context`] share, and the only
/// thing either of them knows about a token ceiling.
pub fn max_latent_frames(lh: usize, lw: usize, max_tokens: usize) -> Result<usize, String> {
    let per_frame = lh.checked_mul(lw).filter(|&n| n > 0).ok_or_else(|| format!("a {lh}x{lw} latent grid is not a grid"))?;
    Ok(max_tokens / per_frame)
}

/// The largest context an `lh` x `lw` plan can actually carry under
/// `max_tokens`, capped at `want`.
///
/// [`window_plan`] REFUSES a grid with no room for `want + 1` latent frames
/// once a clip needs a second window, and for generation that is right: the
/// caller picked the resolution, so a smaller one is available and a
/// truncated motion history is not what [`CONTEXT_LATENT_FRAMES`] cites the
/// reference for. (A clip that fits ONE window is never refused for it - it
/// carries no context at all.)
///
/// Refinement has neither escape. Its grid is the input clip's grid times the
/// upscale factor squared - four times the tokens per latent frame - and its
/// resolution is the whole request, so "generate at a smaller size" is not
/// advice, it is a refusal. At 2560x1408 a 12288-token pass holds three latent
/// frames total and eight carried ones is arithmetically impossible. So a
/// refinement plan takes the most history the ceiling leaves room for and
/// keeps one frame to refine. That is a quality COMPROMISE and the caller is
/// told; carrying nothing is not a compromise, it is the defect - a refinement
/// starting at sigma 0.909 with no history re-imagines the clip from almost
/// nothing, once per pass.
pub fn fitted_context(lh: usize, lw: usize, want: usize, max_tokens: usize) -> Result<usize, String> {
    let max_lat = max_latent_frames(lh, lw, max_tokens)?;
    if max_lat < 2 {
        return Err(format!(
            "a {}x{} grid ({} tokens per latent frame) leaves room for {max_lat} latent frames under the {max_tokens}-token ceiling, and a continued pass needs at least one carried frame plus one new one - work at a smaller size",
            lw * 32,
            lh * 32,
            lh * lw
        ));
    }
    Ok(want.min(max_lat - 1))
}

/// Cut a `frames`-frame request into windows that each fit `max_tokens`,
/// carrying `context` latent frames across every seam.
///
/// The arithmetic that makes this close: window 0 contributes `new - 1` to
/// the request's own `k` (`frames == 1 + 8k`) and every later window
/// contributes its whole `new`, because the leading pixel frame of a
/// continuation window's decode belongs to the context. So the windows sum to
/// the request exactly, for any number of windows - which disjoint `1 + 8k`
/// segments cannot do, since `n` of them sum to `n + 8 * sum(k_i)` and that is
/// a legal clip length only for `n == 1`.
///
/// **Window 0 takes the whole budget; the continuation windows are equal to
/// each other.** Two reasons, and neither is aesthetic. It is the only window
/// that builds structure from noise with no history at all, so it is the one
/// that benefits most from length; and it is the only window whose budget is
/// not already spent on a context, so making it the largest is what
/// guarantees it can hand its successor a FULL `context` frames - an even
/// split across all windows cannot (at a 15-latent-frame budget and an
/// 8-frame context, an even three-way split gives window 0 six latent frames
/// and window 1 would carry a silently truncated context). Splitting the
/// remainder equally then keeps the clip from ending on a stub window.
pub fn window_plan(frames: usize, lh: usize, lw: usize, context: usize, max_tokens: usize) -> Result<Vec<Window>, String> {
    if frames == 0 || !(frames - 1).is_multiple_of(8) {
        return Err(format!("{frames} frames is not of the form 1 + 8k (the causal VAE gives the first frame its own latent frame)"));
    }
    if context == 0 {
        return Err("a long-form window plan needs at least one carried latent frame: a zero-frame context is independent windows, which is the discontinuity this path exists to remove".into());
    }
    let max_lat = max_latent_frames(lh, lw, max_tokens)?;
    let k_total = (frames - 1) / 8;
    // A clip that fits is one window carrying NOTHING, so `context` is not a
    // cost it pays and not a reason to refuse it - `k_total < max_lat` makes
    // its `k_total + 1` latent frames fit the ceiling on their own. The
    // context-fits check below therefore has to come after this, not before:
    // it is a statement about a CONTINUATION window, and this clip has none.
    if k_total < max_lat {
        return Ok(vec![Window { context: 0, new: k_total + 1, first_frame: 0 }]);
    }
    if max_lat < context + 1 {
        return Err(format!(
            "a {}x{} request ({} tokens per latent frame) leaves room for {max_lat} latent frames under the {max_tokens}-token ceiling, and a continuation window needs {context} carried frames plus at least one new one - generate at a smaller size, or lower the context",
            lw * 32,
            lh * 32,
            lh * lw
        ));
    }
    // Every window after the first spends `context` of its budget before it
    // generates anything, so that - not `max_lat` - is what sets their count.
    let cap_new = max_lat - context;
    let remaining = k_total - (max_lat - 1);
    let n_cont = remaining.div_ceil(cap_new);
    let (base, rem) = (remaining / n_cont, remaining % n_cont);
    let mut plan = Vec::with_capacity(n_cont + 1);
    let mut first_frame = 0usize;
    let head = Window { context: 0, new: max_lat, first_frame };
    first_frame += head.emitted_frames();
    plan.push(head);
    for i in 0..n_cont {
        let w = Window { context, new: base + usize::from(i < rem), first_frame };
        first_frame += w.emitted_frames();
        plan.push(w);
    }
    Ok(plan)
}

/// One scene of a multi-scene request: how long it runs, and what it shows.
///
/// A scene is the unit a rolling latent context is carried WITHIN. Windows
/// inside one scene chain exactly as [`window_plan`] describes; at a scene
/// boundary the context resets to nothing, so the next scene is free to be a
/// different subject, setting or action rather than a forced continuation of
/// the one before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scene {
    /// Pixel frames this scene contributes, `1 + 8k` like any clip.
    pub frames: usize,
    /// What this scene shows. Each scene denoises against its own text
    /// context; nothing else about a scene differs.
    pub prompt: String,
}

impl Scene {
    /// `<frames>:<prompt>` - the shell-writable spelling.
    ///
    /// The separator is the FIRST colon and only the first: a prompt is
    /// ordinary English and "then: the camera pans" is a sentence, not a
    /// syntax error.
    pub fn parse(spec: &str) -> Result<Scene, String> {
        let (count, prompt) = spec.split_once(':').ok_or_else(|| format!("a scene is <frames>:<prompt>, and {spec:?} has no ':' to separate them"))?;
        let frames = count.trim().parse::<usize>().map_err(|e| format!("a scene's frame count comes first: {:?} in {spec:?} is not a number ({e})", count.trim()))?;
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return Err(format!("scene {spec:?} has no prompt - the prompt is the only thing that makes a scene a DIFFERENT scene"));
        }
        Ok(Scene { frames, prompt: prompt.to_string() })
    }
}

/// What a scene's index is folded into its seed with, so two scenes of one
/// clip never draw the same initial noise.
///
/// Multiplied by the scene index rather than XORed with it, so scene 0 - the
/// only scene a single-scene request has - keeps the caller's seed EXACTLY and
/// a one-scene call stays the run it already was. Public because the gate that
/// reproduces a scene on its own has to spell the same seed the multi-scene
/// run gave it.
pub const SCENE_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Cut a sequence of scenes into per-scene window plans, one list per scene,
/// with every window's [`Window::first_frame`] an offset into the WHOLE clip.
///
/// This is [`window_plan`] run once per scene and nothing else: inside a scene
/// the rolling context is exactly what that function plans, and the boundary
/// between two scenes is a reset by construction, because a fresh plan's first
/// window carries `context == 0`. That reset is the point - a new scene
/// conditioned on the old scene's real content cannot become a different
/// scene, which is the failure this plan shape exists to remove.
///
/// Resolved before any weight is read, over ALL scenes, so a five-scene
/// request whose fourth scene is unplannable fails now rather than three
/// scenes of device time later.
pub fn scene_plan(scenes: &[Scene], lh: usize, lw: usize, context: usize, max_tokens: usize) -> Result<Vec<Vec<Window>>, String> {
    if scenes.is_empty() {
        return Err("a generation needs at least one scene".into());
    }
    let mut out = Vec::with_capacity(scenes.len());
    let mut origin = 0usize;
    for (si, s) in scenes.iter().enumerate() {
        let mut plan = window_plan(s.frames, lh, lw, context, max_tokens).map_err(|e| format!("scene {} ({:?}): {e}", si + 1, s.prompt))?;
        for w in &mut plan {
            w.first_frame += origin;
        }
        origin += s.frames;
        out.push(plan);
    }
    Ok(out)
}

/// The last `k` latent frames of a `[C, lat_t, lh, lw]` channel-major latent,
/// as a `[C, k, lh, lw]` latent of the same layout.
///
/// A slice and a copy, nothing else - no decode, no re-encode, no rescale.
/// That is the whole point: what window `n + 1` freezes is bit-identical to
/// what window `n` produced.
pub fn carry_tail(latent_chw: &[f32], channels: usize, lat_t: usize, lh: usize, lw: usize, k: usize) -> Vec<f32> {
    assert!(k <= lat_t, "carry_tail: cannot carry {k} of {lat_t} latent frames");
    let plane = lh * lw;
    assert_eq!(latent_chw.len(), channels * lat_t * plane, "carry_tail: {} values, expected {}", latent_chw.len(), channels * lat_t * plane);
    let mut out = vec![0f32; channels * k * plane];
    for c in 0..channels {
        let src = (c * lat_t + (lat_t - k)) * plane;
        out[c * k * plane..(c + 1) * k * plane].copy_from_slice(&latent_chw[src..src + k * plane]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window's own accounting has to be self-consistent: what a decode
    /// produces is what is emitted plus what is dropped, and a continuation
    /// window drops exactly the pixel frames its carried context covers.
    #[test]
    fn a_continuation_window_drops_exactly_its_carried_contexts_own_frames() {
        let w = Window { context: 8, new: 5, first_frame: 100 };
        assert_eq!(w.latent_frames(), 13);
        assert_eq!(w.decoded_frames(), 1 + 8 * 12);
        assert_eq!(w.emitted_frames(), 40);
        assert_eq!(w.dropped_frames(), 1 + 8 * 7, "the dropped range is exactly a 8-latent-frame clip's own length");
        assert_eq!(w.dropped_frames() + w.emitted_frames(), w.decoded_frames());
    }

    /// A first window emits everything it decodes, so the `1 + 8k` request
    /// length survives even when the plan is one window.
    #[test]
    fn a_first_window_emits_everything_it_decodes() {
        let w = Window { context: 0, new: 15, first_frame: 0 };
        assert_eq!(w.emitted_frames(), w.decoded_frames());
        assert_eq!(w.dropped_frames(), 0);
    }
}
