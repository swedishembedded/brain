// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LLaVA's `vicuna_v1` conversation template.
//!
//! Transcribed from upstream `llava/conversation.py`'s `conv_vicuna_v1`
//! (`SeparatorStyle.TWO`) and its `Conversation::get_prompt` - not guessed:
//!
//! ```text
//! conv_vicuna_v1 = Conversation(
//!     system="A chat between a curious user and an artificial intelligence "
//!            "assistant. The assistant gives helpful, detailed, and polite "
//!            "answers to the user's questions.",
//!     roles=("USER", "ASSISTANT"),
//!     sep=" ", sep2="</s>",
//! )
//! ```
//!
//! `get_prompt` for `SeparatorStyle.TWO`, specialised to one user turn with no
//! assistant reply yet (the shape `llava/eval/model_vqa.py`'s single-image
//! captioning path builds - `conv.append_message(roles[1], None)` before
//! `get_prompt()`):
//!
//! ```text
//! ret = system + " "
//! ret += "USER" + ": " + message + " "   # message is non-empty -> trailing sep
//! ret += "ASSISTANT" + ":"                # message is None -> NO trailing space
//! ```
//!
//! and the user's `message` is `DEFAULT_IMAGE_TOKEN + "\n" + question`
//! (`"<image>\n" + question`) when `mm_use_im_start_end` is false - the
//! setting every released `llava-v1.5-*` checkpoint ships.
//!
//! SUPIR only ever issues one caption call per image (no prior assistant turn
//! to continue), so [`caption_prompt`] is the single entry point this crate
//! needs; multi-turn conversation is out of scope (recorded in the roadmap).

/// `conv_vicuna_v1.system`.
pub const SYSTEM: &str = "A chat between a curious user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions.";
pub const ROLE_USER: &str = "USER";
pub const ROLE_ASSISTANT: &str = "ASSISTANT";
/// `DEFAULT_IMAGE_TOKEN` - the literal placeholder [`crate::prompt`] splices
/// on. NOT a vocab token (see that module's docs).
pub const IMAGE_TOKEN: &str = "<image>";

/// Build the single-turn captioning prompt: system message, one USER turn
/// carrying the image placeholder + `question`, then an empty ASSISTANT turn
/// (the model is about to generate one) - byte-for-byte what upstream's
/// `get_prompt()` produces for this shape.
pub fn caption_prompt(question: &str) -> String {
    format!("{SYSTEM} {ROLE_USER}: {IMAGE_TOKEN}\n{question} {ROLE_ASSISTANT}:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caption_prompt_matches_the_reference_get_prompt_shape() {
        let got = caption_prompt("Describe this image and its style in a very detailed manner.");
        let want = "A chat between a curious user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions. \
USER: <image>\nDescribe this image and its style in a very detailed manner. ASSISTANT:";
        assert_eq!(got, want);
    }

    #[test]
    fn the_assistant_turn_has_no_trailing_space() {
        let got = caption_prompt("q");
        assert!(got.ends_with("ASSISTANT:"), "empty message -> role + ':' with no trailing space: {got:?}");
    }

    #[test]
    fn the_image_placeholder_is_on_its_own_line_before_the_question() {
        let got = caption_prompt("what is this?");
        assert!(got.contains("USER: <image>\nwhat is this?"), "{got:?}");
    }
}
