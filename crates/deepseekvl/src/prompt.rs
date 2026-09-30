// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `MultiModalityCausalLM` conversation format and the image splice.
//!
//! Neither DeepSeek-VL nor Janus-Pro ships a chat template; their processors
//! render the `deepseek` conversation style: the system prompt, then user
//! turns closed by a blank line and assistant turns closed by the end-of-
//! sentence token, ending in the bare assistant role for the reply. The two
//! differ only in the role names ([`Style`]). Each image is one
//! `<image_placeholder>` in the text; after tokenizing, each placeholder id
//! becomes the image's rows of the aligner's output, which Janus-Pro wraps in
//! begin- and end-of-image tokens ([`ImageSplice`]).

use qwen3::model::PrefillInput;

/// The system prompt `VLChatProcessor` renders every conversation with.
pub const SYSTEM_PROMPT: &str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.";

/// The text an image occupies in a message.
pub const IMAGE_TAG: &str = "<image_placeholder>";

/// The role names a model was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    pub user: &'static str,
    pub assistant: &'static str,
}

/// DeepSeek-VL: `User:` / `Assistant:`.
pub const DEEPSEEK_VL: Style = Style { user: "User", assistant: "Assistant" };
/// Janus-Pro: `<|User|>:` / `<|Assistant|>:`.
pub const JANUS: Style = Style { user: "<|User|>", assistant: "<|Assistant|>" };

/// Who spoke a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One conversation turn. Images are [`IMAGE_TAG`]s inside `content`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

/// Render `turns` as the model was trained to read them, ending with the
/// open `Assistant:` the reply continues. Turns must alternate starting with
/// the user and end with a user turn.
pub fn render(style: &Style, system: &str, turns: &[Turn], eos: &str) -> Result<String, String> {
    if turns.last().map(|t| t.role) != Some(Role::User) {
        return Err("a DeepSeek-VL conversation ends with a user turn".into());
    }
    let mut out = String::new();
    if !system.is_empty() {
        out.push_str(system);
        out.push_str("\n\n");
    }
    for (i, t) in turns.iter().enumerate() {
        let (want, name, sep) = if i % 2 == 0 { (Role::User, style.user, "\n\n") } else { (Role::Assistant, style.assistant, eos) };
        if t.role != want {
            return Err(format!("turn {i} is from the {:?}; turns alternate starting with the user", t.role));
        }
        let content = t.content.trim();
        // The processor renders an empty message as the bare role; it never
        // separates it.
        if content.is_empty() {
            out.push_str(name);
            out.push(':');
        } else {
            out.push_str(&format!("{name}: {content}{sep}"));
        }
    }
    out.push_str(style.assistant);
    out.push(':');
    Ok(out.trim().to_string())
}

/// How an image's rows replace its placeholder in the token stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSplice {
    /// `<image_placeholder>`'s id.
    pub image_id: u32,
    /// Aligner rows per image.
    pub rows: usize,
    /// Janus-Pro's `(<begin_of_image>, <end_of_image>)` around the rows.
    pub wrap: Option<(u32, u32)>,
}

impl ImageSplice {
    /// The token ids the processor feeds the model: every placeholder
    /// expanded to `rows` copies (wrapped, when the model wraps images).
    pub fn expand_ids(&self, ids: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(ids.len());
        for &t in ids {
            if t != self.image_id {
                out.push(t);
                continue;
            }
            out.extend(self.wrap.map(|w| w.0));
            out.extend(std::iter::repeat(t).take(self.rows));
            out.extend(self.wrap.map(|w| w.1));
        }
        out
    }

    /// The prefill inputs for `ids`: each placeholder becomes the next image's
    /// `rows` rows of `embeds` (`[n_images * rows, width]`), every other id
    /// stays a token. The number of placeholders must equal the number of
    /// images.
    pub fn inputs<'a>(&self, ids: &[u32], embeds: &'a [f32], width: usize) -> Result<Vec<PrefillInput<'a>>, String> {
        let per_image = self.rows * width;
        let images = embeds.len() / per_image;
        if embeds.len() != images * per_image {
            return Err(format!("{} image-embedding values are not whole images of {} x {width}", embeds.len(), self.rows));
        }
        let placeholders = ids.iter().filter(|&&t| t == self.image_id).count();
        if placeholders != images {
            return Err(format!("the prompt has {placeholders} image placeholders for {images} images"));
        }
        let mut rows = embeds.chunks(width);
        let mut out = Vec::with_capacity(ids.len() + images * (self.rows + 1));
        for &t in ids {
            if t != self.image_id {
                out.push(PrefillInput::Token(t));
                continue;
            }
            out.extend(self.wrap.map(|w| PrefillInput::Token(w.0)));
            out.extend(rows.by_ref().take(self.rows).map(PrefillInput::Embed));
            out.extend(self.wrap.map(|w| PrefillInput::Token(w.1)));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: Role, content: &str) -> Turn {
        Turn { role, content: content.into() }
    }

    #[test]
    fn a_single_question_renders_as_the_processor_does() {
        let got = render(&DEEPSEEK_VL, SYSTEM_PROMPT, &[turn(Role::User, "<image_placeholder>Describe this image.")], "</s>").unwrap();
        assert_eq!(got, format!("{SYSTEM_PROMPT}\n\nUser: <image_placeholder>Describe this image.\n\nAssistant:"));
        let got = render(&JANUS, "", &[turn(Role::User, "Hi")], "</s>").unwrap();
        assert_eq!(got, "<|User|>: Hi\n\n<|Assistant|>:");
    }

    #[test]
    fn earlier_replies_close_with_the_eos_text() {
        let turns = [turn(Role::User, "Hi"), turn(Role::Assistant, " Hello! "), turn(Role::User, "Again")];
        assert_eq!(render(&DEEPSEEK_VL, "", &turns, "<eos>").unwrap(), "User: Hi\n\nAssistant: Hello!<eos>User: Again\n\nAssistant:");
        assert!(render(&DEEPSEEK_VL, "", &turns[..2], "<eos>").is_err(), "a conversation ending on the assistant has nothing to reply to");
    }

    #[test]
    fn each_placeholder_takes_its_own_images_rows() {
        let embeds: Vec<f32> = (0..8).map(|v| v as f32).collect(); // 2 images x 2 rows x width 2
        let splice = ImageSplice { image_id: 9, rows: 2, wrap: None };
        let inputs = splice.inputs(&[5, 9, 6, 9, 7], &embeds, 2).unwrap();
        let rows: Vec<Option<&[f32]>> = inputs.iter().map(|i| if let PrefillInput::Embed(r) = i { Some(*r) } else { None }).collect();
        assert_eq!(inputs.len(), 7);
        assert_eq!(rows[1..3], [Some(&[0.0, 1.0][..]), Some(&[2.0, 3.0][..])]);
        assert_eq!(rows[4..6], [Some(&[4.0, 5.0][..]), Some(&[6.0, 7.0][..])]);
        assert!(splice.inputs(&[5, 9], &embeds, 2).err().unwrap().contains("1 image placeholders for 2 images"));
    }

    #[test]
    fn a_wrapped_image_sits_between_its_markers() {
        let splice = ImageSplice { image_id: 9, rows: 3, wrap: Some((1, 2)) };
        assert_eq!(splice.expand_ids(&[5, 9, 6]), vec![5, 1, 9, 9, 9, 2, 6]);
        let embeds = [0.0f32; 3];
        let inputs = splice.inputs(&[5, 9, 6], &embeds, 1).unwrap();
        let tokens: Vec<Option<u32>> = inputs.iter().map(|i| if let PrefillInput::Token(t) = i { Some(*t) } else { None }).collect();
        assert_eq!(tokens, [Some(5), Some(1), None, None, None, Some(2), Some(6)]);
    }
}
