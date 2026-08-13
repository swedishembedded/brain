// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Thinker → Talker prefill assembly: the exact row-by-row construction
//! `Qwen3OmniMoeForConditionalGeneration.generate` builds (in
//! `_get_talker_user_parts`/`_get_talker_assistant_parts`, before calling
//! `talker.generate(...)`) before this module existed, no golden or doc in
//! this repo pinned it down — this is transcribed directly from the
//! installed `transformers` source
//! (`transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`,
//! `Qwen3OmniMoeForConditionalGeneration`, lines ~3770-3848 and ~3976-4020 at
//! the time of writing), not inferred from Omni's own goldens (none exercise
//! the composed splice).
//!
//! **Scope**: single-turn only — one user segment + the current assistant
//! segment, no multi-turn conversation history (matches `qwen3omnimoe::caps::generate`'s
//! own single-turn-per-call scope — the real reference supports resuming a
//! multi-turn conversation's Talker context, which brain does not track
//! across calls). The user segment now DOES select, per position, between
//! `talker.hidden_projection(thinker_hidden_at_accept_layer)` (multimodal
//! positions: audio/image/video) and `talker.text_projection(thinker_embed_tokens
//! (id))` (plain text positions) — [`UserMediaSplice`], passed as `None` for
//! the plain-text case that predates it (bit-identical to the original
//! text-only-only behavior; see the regression test at the bottom of this
//! file).
//!
//! The composed row template (`build_talker_prompt`'s assistant segment) is
//! a FIXED 9-row structure: `[assistant_hidden[0..3], tts_pad×4, tts_bos,
//! assistant_hidden[3..4]]` (the projected text) summed elementwise with
//! `[zeros×3, codec_embed(nothink, think_bos, think_eos, speaker, pad, bos)]`
//! (the codec conditioning) — see `build_talker_prompt`'s body for exactly
//! why each row is what it is; do not "simplify" this shape, it is copied
//! from the reference bit for bit, not derived.

use tts::talker::TextProjection;

/// Special token ids `build_talker_prompt` needs beyond `speaker_id` itself —
/// all read from `qwen3omnimoe::config::{OmniConfig, TalkerConfig}` (real checkpoint
/// values: `tts_bos/eos/pad` 151672/151673/151671, `codec_nothink` 2155,
/// `codec_think_bos` 2156, `codec_think_eos` 2157, `codec_pad` 2148,
/// `codec_bos` 2149).
pub struct TalkerPromptSpecials {
    pub tts_bos_id: u32,
    pub tts_eos_id: u32,
    pub tts_pad_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_pad_id: u32,
    pub codec_bos_id: u32,
}

/// The assembled Talker prefill: `embeds` (`[n, d]`) feeds Talker's own
/// KV-cache prefill (`talker::layer_fwd(..., cache: Some(..))` per layer);
/// `trailing` (`[t_trail, d]`) is added, one row per decode step, onto the
/// codec feedback embedding during generation — `tts_pad`'s own projected
/// embedding once `trailing` is exhausted (see the real reference's
/// `prepare_inputs_for_generation`, `generation_step < trailing_text_hidden
/// .shape[1]`, mirrored by `crate::talker_generate`).
pub struct TalkerPrompt {
    pub embeds: Vec<f32>,
    pub trailing: Vec<f32>,
    /// `text_projection(thinker_embed(tts_pad_id))` — the feedback row used
    /// once `trailing` runs out. Precomputed here since building it needs
    /// the same `text_proj`/`thinker_embed_table` this function already has.
    pub tts_pad_embed: Vec<f32>,
}

/// Per-position multimodal splice for the user segment: at a media position
/// (`media_mask[i]`), the row is `hidden_proj.project(hidden[i])` — the
/// Thinker's accept-layer hidden state at that position
/// (`crate::generate::thinker_hidden_at_layer`), not the placeholder token's
/// embedding — instead of `text_proj`'s usual `text_projection(thinker_embed
/// (id))`. At a text position, [`build_talker_prompt`] uses the ordinary text
/// branch unchanged; `user_text_ids[i]` at a media position is never read
/// (the real placeholder token id, e.g. `audio_token_id`, is harmless to pass
/// through — it's simply skipped).
pub struct UserMediaSplice<'a> {
    /// `talker.hidden_projection` — same shape as `talker.text_projection`
    /// (`tts::talker::TextProjection`, `codec_bridge::load_talker_projection`
    /// already loads it by name), a DIFFERENT set of weights from `text_proj`.
    pub hidden_proj: &'a TextProjection,
    /// Thinker's hidden state at `TalkerConfig::accept_hidden_layer`, `[n_user,
    /// thinker_d]`, row-aligned to `user_text_ids` (same length, same order —
    /// e.g. `crate::mm::MultimodalPrompt::token_ids`'s user-turn span and the
    /// matching rows of a `thinker_hidden_at_layer` capture over that span).
    pub hidden: &'a [f32],
    /// `true` at a media (audio/image/video placeholder) position, `false` at
    /// a plain text position. Length must equal `user_text_ids.len()`.
    pub media_mask: &'a [bool],
}

/// Build the Talker prefill for ONE user turn + the current assistant turn.
///
/// `text_proj` is `talker.text_projection` (a 2-layer `fc2(silu(fc1(x)))`
/// MLP, `thinker_hidden_size -> talker_d`; `tts::talker::TextProjection`
/// reused unchanged — same shape, per `crate::config::TalkerConfig`'s doc).
/// `codec_embed(id)` looks up Talker's OWN `codec_embedding` table (a plain
/// row gather, not a projection). `thinker_embed_table`/`thinker_d` are
/// Thinker's `embed_tokens` table (`user_text_ids`/`assistant_text_ids` are
/// looked up there, matching the reference's `thinker_embed = embed_tokens
/// (sequences)` — layer-0 hidden state IS the embedding-table output, no
/// separate capture needed for the text-only path this function implements).
/// `user_media`: `None` for a plain-text user turn (bit-identical to this
/// function's original text-only behavior); `Some(UserMediaSplice { .. })`
/// to select, per user-segment position, between the text branch and
/// `hidden_projection` — see [`UserMediaSplice`]'s doc.
///
/// `assistant_text_ids` must have at least 4 tokens — the fixed template
/// needs `assistant_hidden[0..4]` (real assistant turns, even one word plus
/// punctuation, comfortably clear this; a caller with a shorter response
/// should pad or fall back to text-only output rather than call this).
#[allow(clippy::too_many_arguments)]
pub fn build_talker_prompt(
    text_proj: &TextProjection,
    codec_embed: &dyn Fn(u32) -> Vec<f32>,
    specials: &TalkerPromptSpecials,
    speaker_id: u32,
    thinker_embed_table: &[f32],
    thinker_d: usize,
    user_text_ids: &[u32],
    assistant_text_ids: &[u32],
    user_media: Option<UserMediaSplice>,
) -> TalkerPrompt {
    assert!(assistant_text_ids.len() >= 4, "assistant response ({} tokens) too short for the fixed 9-row prefill template (need >= 4)", assistant_text_ids.len());
    let d = text_proj.out;
    let embed_row = |id: u32| thinker_embed_table[id as usize * thinker_d..(id as usize + 1) * thinker_d].to_vec();

    // User segment: text_projection(thinker_embed(id)) at a text position,
    // hidden_projection(accept-layer hidden) at a media position -- the
    // multimodal branch this module's doc describes. `None` reproduces the
    // original text-only-only behavior exactly (one batched `project` call,
    // same as before this parameter existed).
    let user_part: Vec<f32> = match user_media {
        None => {
            let user_hidden: Vec<f32> = user_text_ids.iter().flat_map(|&id| embed_row(id)).collect();
            text_proj.project(&user_hidden) // [n_user, d]
        }
        Some(m) => {
            assert_eq!(m.media_mask.len(), user_text_ids.len(), "media_mask len {} != user_text_ids len {}", m.media_mask.len(), user_text_ids.len());
            assert_eq!(m.hidden.len(), user_text_ids.len() * thinker_d, "hidden len {} != n_user*thinker_d {}", m.hidden.len(), user_text_ids.len() * thinker_d);
            let mut out = Vec::with_capacity(user_text_ids.len() * d);
            for (i, &id) in user_text_ids.iter().enumerate() {
                if m.media_mask[i] {
                    let h = &m.hidden[i * thinker_d..(i + 1) * thinker_d];
                    out.extend(m.hidden_proj.project(h));
                } else {
                    out.extend(text_proj.project(&embed_row(id)));
                }
            }
            out
        }
    };

    // Assistant segment's own projected text hidden states.
    let asst_hidden: Vec<f32> = assistant_text_ids.iter().flat_map(|&id| embed_row(id)).collect();
    let assistant_hidden = text_proj.project(&asst_hidden); // [n_asst, d]

    // tts_bos/eos/pad: text_projection(thinker_embed(id)) each, one row.
    let tts_row = |id: u32| -> Vec<f32> { text_proj.project(&embed_row(id)) };
    let tts_bos = tts_row(specials.tts_bos_id);
    let tts_eos = tts_row(specials.tts_eos_id);
    let tts_pad_embed = tts_row(specials.tts_pad_id);

    // assistant_text_hidden = [assistant_hidden[0..3], tts_pad x4, tts_bos, assistant_hidden[3..4]] (9 rows).
    let mut assistant_text_hidden = Vec::with_capacity(9 * d);
    assistant_text_hidden.extend_from_slice(&assistant_hidden[0..3 * d]);
    for _ in 0..4 {
        assistant_text_hidden.extend_from_slice(&tts_pad_embed);
    }
    assistant_text_hidden.extend_from_slice(&tts_bos);
    assistant_text_hidden.extend_from_slice(&assistant_hidden[3 * d..4 * d]);

    // assistant_codec_hidden = [zeros x3, codec_embed(nothink, think_bos, think_eos, speaker, pad, bos)] (9 rows).
    let codec_ids = [specials.codec_nothink_id, specials.codec_think_bos_id, specials.codec_think_eos_id, speaker_id, specials.codec_pad_id, specials.codec_bos_id];
    let mut assistant_codec_hidden = vec![0f32; 3 * d];
    for id in codec_ids {
        assistant_codec_hidden.extend(codec_embed(id));
    }

    let assistant_part: Vec<f32> = assistant_text_hidden.iter().zip(&assistant_codec_hidden).map(|(a, b)| a + b).collect();

    // trailing_text_hidden = [assistant_hidden[4..], tts_eos] -- fed in during decode.
    let mut trailing = assistant_hidden[4 * d..].to_vec();
    trailing.extend_from_slice(&tts_eos);

    let mut embeds = user_part;
    embeds.extend(assistant_part);

    TalkerPrompt { embeds, trailing, tts_pad_embed }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic tiny `TextProjection` (`in=out=inter=3`, non-identity
    /// weights so `silu`'s nonlinearity is actually exercised) — pins down
    /// `build_talker_prompt`'s ROW ASSEMBLY (order, zero-padding, elementwise
    /// sum), not `TextProjection::project`'s own math (that's `tts`'s test
    /// surface, reused unchanged here).
    fn tiny_proj() -> TextProjection {
        TextProjection {
            text_embedding: None,
            fc1_w: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            fc1_b: vec![0.1, 0.2, 0.3],
            fc2_w: vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            fc2_b: vec![0.0, 0.0, 0.0],
            in_dim: 3,
            inter: 3,
            out: 3,
            text_vocab: 16,
        }
    }

    #[test]
    fn assembles_the_fixed_nine_row_assistant_template() {
        let proj = tiny_proj();
        let d = 3usize;
        let thinker_vocab = 16u32;
        // thinker_embed_table[id] = [id, id, id] (as f32) -- trivially traceable.
        let thinker_embed_table: Vec<f32> = (0..thinker_vocab).flat_map(|id| vec![id as f32; d]).collect();

        let specials = TalkerPromptSpecials { tts_bos_id: 10, tts_eos_id: 11, tts_pad_id: 12, codec_nothink_id: 1, codec_think_bos_id: 2, codec_think_eos_id: 3, codec_pad_id: 4, codec_bos_id: 5 };
        let speaker_id = 6u32;
        let codec_embed = |id: u32| -> Vec<f32> { vec![100.0 + id as f32; d] }; // distinguishable from text rows

        let user_ids = [7u32, 8];
        let assistant_ids = [1u32, 2, 3, 4, 9]; // >= 4 tokens, per the function's own precondition

        let got = build_talker_prompt(&proj, &codec_embed, &specials, speaker_id, &thinker_embed_table, d, &user_ids, &assistant_ids, None);

        // Shapes: user (2 rows) + the fixed 9-row assistant template.
        assert_eq!(got.embeds.len(), (2 + 9) * d);
        // trailing = assistant_hidden[4..] (1 row, since assistant has 5 tokens) + tts_eos (1 row) = 2 rows.
        assert_eq!(got.trailing.len(), 2 * d);

        // Independently recompute every referenced row via the SAME
        // TextProjection::project / codec_embed primitives, then assemble by
        // hand, and compare -- this is the test's actual assertion surface.
        let row = |id: u32| proj.project(&thinker_embed_table[id as usize * d..(id as usize + 1) * d]);
        let user_want: Vec<f32> = user_ids.iter().flat_map(|&id| row(id)).collect();
        assert_eq!(&got.embeds[0..2 * d], user_want.as_slice(), "user segment");

        let asst: Vec<Vec<f32>> = assistant_ids.iter().map(|&id| row(id)).collect();
        let tts_bos = row(specials.tts_bos_id);
        let tts_eos = row(specials.tts_eos_id);
        let tts_pad = row(specials.tts_pad_id);
        assert_eq!(got.tts_pad_embed, tts_pad);

        // assistant_text_hidden rows 0..9, by the fixed template.
        let text_hidden: Vec<Vec<f32>> = vec![asst[0].clone(), asst[1].clone(), asst[2].clone(), tts_pad.clone(), tts_pad.clone(), tts_pad.clone(), tts_pad.clone(), tts_bos.clone(), asst[3].clone()];
        let codec_hidden: Vec<Vec<f32>> = vec![
            vec![0.0; d],
            vec![0.0; d],
            vec![0.0; d],
            codec_embed(specials.codec_nothink_id),
            codec_embed(specials.codec_think_bos_id),
            codec_embed(specials.codec_think_eos_id),
            codec_embed(speaker_id),
            codec_embed(specials.codec_pad_id),
            codec_embed(specials.codec_bos_id),
        ];
        for r in 0..9 {
            let want_row: Vec<f32> = text_hidden[r].iter().zip(&codec_hidden[r]).map(|(a, b)| a + b).collect();
            let got_row = &got.embeds[(2 + r) * d..(2 + r + 1) * d];
            assert_eq!(got_row, want_row.as_slice(), "assistant row {r}");
        }

        // trailing = [asst[4], tts_eos].
        assert_eq!(&got.trailing[0..d], asst[4].as_slice(), "trailing row 0");
        assert_eq!(&got.trailing[d..2 * d], tts_eos.as_slice(), "trailing row 1 (tts_eos)");
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn rejects_an_assistant_response_shorter_than_four_tokens() {
        let proj = tiny_proj();
        let thinker_embed_table = vec![0.0; 16 * 3];
        let specials = TalkerPromptSpecials { tts_bos_id: 0, tts_eos_id: 0, tts_pad_id: 0, codec_nothink_id: 0, codec_think_bos_id: 0, codec_think_eos_id: 0, codec_pad_id: 0, codec_bos_id: 0 };
        build_talker_prompt(&proj, &|_| vec![0.0; 3], &specials, 0, &thinker_embed_table, 3, &[1, 2], &[1, 2, 3], None);
    }

    /// `Some(UserMediaSplice { media_mask: all false, .. })` must reproduce
    /// `None`'s output bit-for-bit — the property that makes adding this
    /// parameter safe for every existing caller (`OmniInner::speak` today
    /// passes `None`; nothing about that call site's numeric output may move).
    #[test]
    fn all_text_media_mask_matches_the_none_path_exactly() {
        let proj = tiny_proj();
        let hidden_proj = TextProjection {
            text_embedding: None,
            fc1_w: vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
            fc1_b: vec![0.0, 0.0, 0.0],
            fc2_w: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            fc2_b: vec![0.0, 0.0, 0.0],
            in_dim: 3,
            inter: 3,
            out: 3,
            text_vocab: 0,
        };
        let d = 3usize;
        let thinker_embed_table: Vec<f32> = (0..16u32).flat_map(|id| vec![id as f32; d]).collect();
        let specials = TalkerPromptSpecials { tts_bos_id: 10, tts_eos_id: 11, tts_pad_id: 12, codec_nothink_id: 1, codec_think_bos_id: 2, codec_think_eos_id: 3, codec_pad_id: 4, codec_bos_id: 5 };
        let codec_embed = |id: u32| -> Vec<f32> { vec![100.0 + id as f32; d] };
        let user_ids = [7u32, 8, 9];
        let assistant_ids = [1u32, 2, 3, 4, 9];

        let want = build_talker_prompt(&proj, &codec_embed, &specials, 6, &thinker_embed_table, d, &user_ids, &assistant_ids, None);

        // hidden rows are deliberately NOT what the text branch would embed
        // (large, distinguishable values) -- if media_mask=false didn't
        // actually skip them, this test would fail.
        let hidden = vec![9999.0f32; user_ids.len() * d];
        let media_mask = vec![false; user_ids.len()];
        let splice = UserMediaSplice { hidden_proj: &hidden_proj, hidden: &hidden, media_mask: &media_mask };
        let got = build_talker_prompt(&proj, &codec_embed, &specials, 6, &thinker_embed_table, d, &user_ids, &assistant_ids, Some(splice));

        assert_eq!(got.embeds, want.embeds, "all-false media_mask must match the None path exactly");
        assert_eq!(got.trailing, want.trailing);
        assert_eq!(got.tts_pad_embed, want.tts_pad_embed);
    }

    /// A real mixed user turn (text, media, text): the media row must come
    /// from `hidden_proj.project(hidden[i])`, NOT `text_proj.project(thinker_embed
    /// (user_text_ids[i]))` -- proving the branch selection is load-bearing,
    /// not merely wired and ignored (the exact shape of gap this splice
    /// closes: `talker_prompt`'s user segment used to take the text branch
    /// unconditionally).
    #[test]
    fn media_position_uses_hidden_projection_not_text_projection() {
        let proj = tiny_proj();
        let hidden_proj = TextProjection {
            text_embedding: None,
            fc1_w: vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
            fc1_b: vec![0.0, 0.0, 0.0],
            fc2_w: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            fc2_b: vec![0.0, 0.0, 0.0],
            in_dim: 3,
            inter: 3,
            out: 3,
            text_vocab: 0,
        };
        let d = 3usize;
        let thinker_embed_table: Vec<f32> = (0..16u32).flat_map(|id| vec![id as f32; d]).collect();
        let specials = TalkerPromptSpecials { tts_bos_id: 10, tts_eos_id: 11, tts_pad_id: 12, codec_nothink_id: 1, codec_think_bos_id: 2, codec_think_eos_id: 3, codec_pad_id: 4, codec_bos_id: 5 };
        let codec_embed = |id: u32| -> Vec<f32> { vec![100.0 + id as f32; d] };
        // [text(7), media(placeholder id 151655, never read), text(8)]
        let user_ids = [7u32, 151655, 8];
        let assistant_ids = [1u32, 2, 3, 4, 9];
        let media_hidden_row = vec![42.0f32, 43.0, 44.0];
        let mut hidden = vec![0.0f32; user_ids.len() * d];
        hidden[d..2 * d].copy_from_slice(&media_hidden_row);
        let media_mask = [false, true, false];
        let splice = UserMediaSplice { hidden_proj: &hidden_proj, hidden: &hidden, media_mask: &media_mask };

        let got = build_talker_prompt(&proj, &codec_embed, &specials, 6, &thinker_embed_table, d, &user_ids, &assistant_ids, Some(splice));

        let text_row = |id: u32| proj.project(&thinker_embed_table[id as usize * d..(id as usize + 1) * d]);
        assert_eq!(&got.embeds[0..d], text_row(7).as_slice(), "position 0 (text) uses text_projection");
        assert_eq!(&got.embeds[d..2 * d], hidden_proj.project(&media_hidden_row).as_slice(), "position 1 (media) uses hidden_projection");
        assert_eq!(&got.embeds[2 * d..3 * d], text_row(8).as_slice(), "position 2 (text) uses text_projection");
    }
}
