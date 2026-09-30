// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-VL's conversation format and the image splice.
//!
//! The checkpoint ships no chat template; its processor renders the
//! `deepseek` conversation style: the system prompt, then `User: ...` turns
//! closed by a blank line and `Assistant: ...` turns closed by the end-of-
//! sentence token, ending in a bare `Assistant:` for the reply. Each image is
//! one `<image_placeholder>` in the text; after tokenizing, each placeholder
//! id becomes `image_tokens` rows of the aligner's output.

use qwen3::model::PrefillInput;

/// The system prompt `VLChatProcessor` renders every conversation with.
pub const SYSTEM_PROMPT: &str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.";

/// The text an image occupies in a message.
pub const IMAGE_TAG: &str = "<image_placeholder>";

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
pub fn render(system: &str, turns: &[Turn], eos: &str) -> Result<String, String> {
    if turns.last().map(|t| t.role) != Some(Role::User) {
        return Err("a DeepSeek-VL conversation ends with a user turn".into());
    }
    let mut out = String::new();
    if !system.is_empty() {
        out.push_str(system);
        out.push_str("\n\n");
    }
    for (i, t) in turns.iter().enumerate() {
        let (want, name, sep) = if i % 2 == 0 { (Role::User, "User", "\n\n") } else { (Role::Assistant, "Assistant", eos) };
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
    out.push_str("Assistant:");
    Ok(out.trim().to_string())
}

/// The token ids with every `image_id` expanded to `image_tokens` copies,
/// what the processor feeds the model (and what its image mask marks).
pub fn expand_image_ids(ids: &[u32], image_id: u32, image_tokens: usize) -> Vec<u32> {
    ids.iter().flat_map(|&t| std::iter::repeat(t).take(if t == image_id { image_tokens } else { 1 })).collect()
}

/// The prefill inputs for `ids`: each `image_id` becomes the next image's
/// `image_tokens` rows of `embeds` (`[n_images * image_tokens, width]`), every
/// other id stays a token. The number of placeholders must equal the number
/// of images.
pub fn splice<'a>(ids: &[u32], image_id: u32, embeds: &'a [f32], image_tokens: usize, width: usize) -> Result<Vec<PrefillInput<'a>>, String> {
    let per_image = image_tokens * width;
    let images = embeds.len() / per_image;
    if embeds.len() != images * per_image {
        return Err(format!("{} image-embedding values are not whole images of {image_tokens} x {width}", embeds.len()));
    }
    let placeholders = ids.iter().filter(|&&t| t == image_id).count();
    if placeholders != images {
        return Err(format!("the prompt has {placeholders} image placeholders for {images} images"));
    }
    let mut rows = embeds.chunks(width);
    let mut out = Vec::with_capacity(ids.len() + images * (image_tokens - 1));
    for &t in ids {
        if t == image_id {
            out.extend(rows.by_ref().take(image_tokens).map(PrefillInput::Embed));
        } else {
            out.push(PrefillInput::Token(t));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: Role, content: &str) -> Turn {
        Turn { role, content: content.into() }
    }

    #[test]
    fn a_single_question_renders_as_the_processor_does() {
        let got = render(SYSTEM_PROMPT, &[turn(Role::User, "<image_placeholder>Describe this image.")], "</s>").unwrap();
        assert_eq!(got, format!("{SYSTEM_PROMPT}\n\nUser: <image_placeholder>Describe this image.\n\nAssistant:"));
    }

    #[test]
    fn earlier_replies_close_with_the_eos_text() {
        let turns = [turn(Role::User, "Hi"), turn(Role::Assistant, " Hello! "), turn(Role::User, "Again")];
        assert_eq!(render("", &turns, "<eos>").unwrap(), "User: Hi\n\nAssistant: Hello!<eos>User: Again\n\nAssistant:");
        assert!(render("", &turns[..2], "<eos>").is_err(), "a conversation ending on the assistant has nothing to reply to");
    }

    #[test]
    fn each_placeholder_takes_its_own_images_rows() {
        let embeds: Vec<f32> = (0..8).map(|v| v as f32).collect(); // 2 images x 2 rows x width 2
        let inputs = splice(&[5, 9, 6, 9, 7], 9, &embeds, 2, 2).unwrap();
        let rows: Vec<Option<&[f32]>> = inputs.iter().map(|i| if let PrefillInput::Embed(r) = i { Some(*r) } else { None }).collect();
        assert_eq!(inputs.len(), 7);
        assert_eq!(rows[1..3], [Some(&[0.0, 1.0][..]), Some(&[2.0, 3.0][..])]);
        assert_eq!(rows[4..6], [Some(&[4.0, 5.0][..]), Some(&[6.0, 7.0][..])]);
        assert_eq!(expand_image_ids(&[5, 9, 6], 9, 3), vec![5, 9, 9, 9, 6]);
        assert!(splice(&[5, 9], 9, &embeds, 2, 2).err().unwrap().contains("1 image placeholders for 2 images"));
    }
}
