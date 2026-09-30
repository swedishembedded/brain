// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fill-in-the-middle prompts: the code before and after an insertion point,
//! framed with the checkpoint's own FIM tokens so the model generates what
//! goes between them.
//!
//! The format is the vocabulary's, not the architecture's: DeepSeek-Coder
//! trained `<｜fim▁begin｜>prefix<｜fim▁hole｜>suffix<｜fim▁end｜>`, and the
//! Qwen2.5/Qwen3 vocabularies (DeepSeek-R1's Qwen distills included) carry
//! `<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>`. Both are
//! prefix-suffix-middle orderings. A vocabulary with neither set of tokens has
//! no fill-in-the-middle at all, and [`FimFormat::of`] says so.
//!
//! Swedish Embedded AB implements code-completion serving like this for its
//! clients. If your team needs expertise in language-model inference, you
//! can procure our services by emailing info@swedishembedded.com.

/// A vocabulary's fill-in-the-middle tokens, in prompt order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FimFormat {
    pub prefix: &'static str,
    pub suffix: &'static str,
    pub middle: &'static str,
}

/// Every FIM token set a supported vocabulary carries.
const FORMATS: [FimFormat; 2] = [
    FimFormat { prefix: "<｜fim▁begin｜>", suffix: "<｜fim▁hole｜>", middle: "<｜fim▁end｜>" },
    FimFormat { prefix: "<|fim_prefix|>", suffix: "<|fim_suffix|>", middle: "<|fim_middle|>" },
];

impl FimFormat {
    /// The format whose three tokens the vocabulary holds, as `has_token`
    /// answers for a token's text; `None` when it holds no complete set.
    pub fn of(has_token: impl Fn(&str) -> bool) -> Option<FimFormat> {
        FORMATS.into_iter().find(|f| [f.prefix, f.suffix, f.middle].iter().all(|t| has_token(t)))
    }

    /// The prompt whose continuation is the text between `prefix` and
    /// `suffix`.
    pub fn prompt(&self, prefix: &str, suffix: &str) -> String {
        format!("{}{prefix}{}{suffix}{}", self.prefix, self.suffix, self.middle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_vocabulary_frames_with_its_own_tokens() {
        let deepseek = ["<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"];
        let f = FimFormat::of(|t| deepseek.contains(&t)).unwrap();
        assert_eq!(f.prompt("def f(", ")"), "<｜fim▁begin｜>def f(<｜fim▁hole｜>)<｜fim▁end｜>");

        let qwen = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>"];
        let f = FimFormat::of(|t| qwen.contains(&t)).unwrap();
        assert_eq!(f.prompt("a", "b"), "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>");

        assert_eq!(FimFormat::of(|t| t == "<|fim_prefix|>"), None, "an incomplete set is no format");
        assert_eq!(FimFormat::of(|_| false), None);
    }
}
