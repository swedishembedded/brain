// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The prompt has to reach the picture.** A real-weight gate on plain
//! text-to-video: two unrelated prompts, everything else identical - same
//! seed, same shape, same schedule, no image conditioning - must not produce
//! the same clip.
//!
//! Swedish Embedded AB implements text-conditioning pipelines and the
//! numerical gates that keep them honest for its clients. If your team needs
//! expertise in diffusion text encoders and their conditioning paths then you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! ## Why this gate exists
//!
//! Every parity gate in this crate was green while `brain ltxv t2v` ignored
//! its prompt completely. `crates/gemma4`'s `AggregateEmbed` - LTX's own
//! 49-hidden-state `text_embedding_projection` - was built from the real
//! checkpoint's tensor SHAPE alone, on the recorded assumption that the
//! module's internals were not derivable from a header. They were: the module
//! is `ltx_core.text_encoders.gemma.feature_extractor.FeatureExtractorV2`,
//! and it differs from a plain concatenate-then-project in three ways at
//! once - a per-token, per-state RMS normalization, an interleaved
//! (`d*n_states + k`, not `k*hidden + d`) column order, and a
//! `sqrt(out_dim/hidden)` rescale.
//!
//! What that produced, measured on the real 22B Q8_0 DiT + real Gemma-4 Q8_0
//! encoder + real conv VAE, one Tesla P40, 512x512 / 25 frames / seed 7:
//! *"a bright red vintage convertible car driving fast through an empty
//! desert highway at sunset"* and *"a slow pan across a snowbound pine
//! forest"* both decoded to the same close-up of the same woman's face, mean
//! absolute pixel delta **6.36** of 255 between the two mid-clip frames. The
//! two prompts' encoded contexts had mean rows at cosine **0.9998** of each
//! other and every row within one context sat at cosine 0.996 of every other:
//! the caption survived only as a ~5% residual on a near-constant vector, so
//! the DiT was sampling its unconditional prior whatever it was asked for.
//!
//! The fast, weight-free gates on the same defect are
//! `gemma4::model`'s `the_projection_rms_normalizes_every_layer_slice_and_rescales`
//! and the `aggregate_out` tap of `gemma4`'s `gemma4_tiny_matches_reference`
//! parity suite; this file is the perceptual half, which is what proves those
//! are testing the thing a viewer actually sees.
//!
//! `#[ignore]`d: two full real 22B generations. Run explicitly:
//!
//! ```text
//! BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
//! BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
//! BRAIN_LTXV_TEXT_ENCODER=<...gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf> \
//! cargo test -p brain-ltxv --test prompt_adherence_real -- --ignored --nocapture
//! ```

use ltxv::pipeline::{generate, GenOpts, Paths};

/// The shape `anchor_real.rs` runs at, for the same reason: the smallest real
/// clip this pipeline can produce, so the gate costs the least real weight
/// streaming that still exercises the whole conditioning path.
const FRAMES: usize = 9;
const WIDTH: usize = 384;
const HEIGHT: usize = 192;
const SEED: u64 = 7;
const FPS: usize = 8;

/// Two prompts with no subject, no setting, no palette and no motion in
/// common - so a clip that follows EITHER of them cannot resemble a clip that
/// follows the other. Deliberately not two variations on one scene: this gate
/// asks whether the caption reaches the model at all, not how finely it is
/// resolved.
const PROMPT_A: &str = "a bright red vintage convertible car driving fast through an empty desert highway at sunset, dust clouds behind it";
const PROMPT_B: &str = "a slow pan across a snowbound pine forest";

/// Mean absolute per-channel difference in 0-255 pixel units, over a whole
/// clip. Same metric `anchor_real.rs` corroborates with.
fn mean_abs_delta(a: &[Vec<u8>], b: &[Vec<u8>]) -> f64 {
    assert_eq!(a.len(), b.len(), "clips have different frame counts");
    let (mut acc, mut n) = (0.0f64, 0usize);
    for (fa, fb) in a.iter().zip(b) {
        assert_eq!(fa.len(), fb.len());
        acc += fa.iter().zip(fb).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>();
        n += fa.len();
    }
    acc / n as f64
}

fn real_paths() -> Option<Paths> {
    let p = Paths::resolve(None, None, None).ok()?;
    p.dit.as_ref()?;
    p.text_encoder.as_ref()?;
    Some(p)
}

fn base_opts() -> GenOpts {
    GenOpts {
        frames: FRAMES,
        width: WIDTH,
        height: HEIGHT,
        seed: SEED,
        fps: FPS,
        // The distilled checkpoint's own sampler, same as `anchor_real.rs`:
        // `ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)`, and reproducible from
        // `seed` regardless (`data::rng::Rng`).
        eta: 1.0,
        // No CFG. With `guidance > 1.0` a broken conditional branch could be
        // masked by the unconditional one being subtracted from it; one
        // forward per step means the clip below is the conditional branch's
        // own answer and nothing else.
        guidance: 1.0,
        dit_config: "ltx25_22b".into(),
        device: Some("gpu".into()),
        // Explicitly no image conditioning - the whole point is that the text
        // context is the ONLY thing steering these two runs apart.
        start_frame: None,
        end_frame: None,
        ..GenOpts::default()
    }
}

/// A floor on how far apart two unrelated prompts must push the decoded clip,
/// in 0-255 units, measured on the real 22B Q8_0 DiT + real Gemma-4 Q8_0
/// encoder + real conv VAE, one Tesla P40. Both ends of the calibration are
/// real numbers from these same two prompts:
///
/// | | shape | whole-clip delta |
/// |---|---|---:|
/// | defective projection | 512x512 / 25f / seed 7 | **6.67** |
/// | fixed projection | 512x512 / 25f / seed 7 | **100.57** |
/// | fixed projection | THIS test's 384x192 / 9f | **80.04** |
///
/// The defect was measured at 512x512 rather than at this test's shape
/// because that is the shape it was reported and investigated at, and it is
/// not a resolution-dependent defect: the two prompts' encoded contexts sat
/// at cosine 0.9998 of each other, so the clips were near-identical at any
/// size. The floor sits 4x below what a correct run produces here and 3x
/// above what the defect produced - deliberately loose, because this gate is
/// a liveness check on the conditioning path, not a similarity metric, and a
/// tight bound would only make it fragile against sampler or schedule changes
/// that are nobody's bug.
const MIN_PROMPT_DELTA: f64 = 20.0;

#[test]
#[ignore = "two full real 22B generations"]
fn two_unrelated_prompts_do_not_produce_the_same_clip() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE + BRAIN_LTXV_TEXT_ENCODER to the real LTX-2.5 checkpoints");
    };
    let cancel = capability::CancelToken::default();

    let (a, ta) = generate(&paths, PROMPT_A, &base_opts(), &cancel, |_, _, _| {}).expect("text-to-video run A");
    let (b, tb) = generate(&paths, PROMPT_B, &base_opts(), &cancel, |_, _, _| {}).expect("text-to-video run B");
    assert_eq!(a.frames.len(), FRAMES);
    assert_eq!(b.frames.len(), FRAMES);

    let delta = mean_abs_delta(&a.frames, &b.frames);
    eprintln!("t2v A: {:.1}s, t2v B: {:.1}s, mean absolute pixel delta between the two clips: {delta:.2}", ta.total(), tb.total());

    assert!(
        delta >= MIN_PROMPT_DELTA,
        "two unrelated prompts produced near-identical clips: mean absolute pixel delta {delta:.2} of 255 (floor {MIN_PROMPT_DELTA:.1}). \
         Everything but the caption is identical between these two runs, so this says the text context is not reaching - or not discriminating at - the DiT's cross-attention. \
         A delta near 6 is the signature of `gemma4::AggregateEmbed` projecting an un-normalized, wrongly-ordered hidden-state concatenation - see this file's module doc."
    );
}
