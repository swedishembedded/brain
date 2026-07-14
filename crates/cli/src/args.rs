// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared CLI argument grammar for the model commands (`gpt`/`qwen`/`glm`).
//!
//! Replaces the per-command hand-rolled `while i < len { match … }` + `val(…)`
//! loops with one take-and-remove parser: each command declares exactly the
//! flags it wants (`take_*`), and [`Args::finish`] warns on anything left over —
//! so unknown flags are surfaced instead of silently ignored, and every command
//! shares the same value/parse/positional handling.

/// A consuming argument parser: `take_*` scans the remaining tokens, removes the
/// matched flag (and its value), and returns it. Positional (non-`--`) tokens are
/// pulled in order by [`Args::positional`].
pub struct Args {
    toks: Vec<String>,
    used: Vec<bool>,
}

impl Args {
    pub fn new(args: &[String]) -> Args {
        Args { toks: args.to_vec(), used: vec![false; args.len()] }
    }

    /// Value flag `--name VALUE`: returns the value and marks both tokens used.
    pub fn take_str(&mut self, name: &str) -> Option<String> {
        for i in 0..self.toks.len() {
            if !self.used[i] && self.toks[i] == name {
                self.used[i] = true;
                if i + 1 < self.toks.len() {
                    self.used[i + 1] = true;
                    return Some(self.toks[i + 1].clone());
                }
                eprintln!("{name} requires a value");
                std::process::exit(2);
            }
        }
        None
    }
    pub fn str_or(&mut self, name: &str, default: &str) -> String {
        self.take_str(name).unwrap_or_else(|| default.to_string())
    }
    pub fn u32_or(&mut self, name: &str, default: u32) -> u32 {
        self.take_str(name).and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    pub fn usize_or(&mut self, name: &str, default: usize) -> usize {
        self.take_str(name).and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    pub fn u64_or(&mut self, name: &str, default: u64) -> u64 {
        self.take_str(name).and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    pub fn f32_or(&mut self, name: &str, default: f32) -> f32 {
        self.take_str(name).and_then(|s| s.parse().ok()).unwrap_or(default)
    }
    pub fn char_opt(&mut self, name: &str) -> Option<char> {
        self.take_str(name).and_then(|s| s.chars().next())
    }
    pub fn opt_u32(&mut self, name: &str) -> Option<u32> {
        self.take_str(name).and_then(|s| s.parse().ok())
    }
    /// Presence flag `--name` (no value): true if present, marks it used.
    pub fn take_flag(&mut self, name: &str) -> bool {
        for i in 0..self.toks.len() {
            if !self.used[i] && self.toks[i] == name {
                self.used[i] = true;
                return true;
            }
        }
        false
    }
    /// The next unused positional (non-`--`) token, in order.
    pub fn positional(&mut self) -> Option<String> {
        for i in 0..self.toks.len() {
            if !self.used[i] && !self.toks[i].starts_with("--") {
                self.used[i] = true;
                return Some(self.toks[i].clone());
            }
        }
        None
    }
    /// Warn about any leftover (unrecognised) tokens. Call once at the end.
    pub fn finish(&self) {
        let extra: Vec<&String> = self.toks.iter().zip(&self.used).filter(|(_, u)| !**u).map(|(t, _)| t).collect();
        if !extra.is_empty() {
            eprintln!("ignoring unrecognised args: {extra:?}");
        }
    }
}

/// Canonicalise a verb, accepting back-compat aliases so old invocations keep
/// working across the unified model CLIs (`gen`→`infer`, `fine-tune`→`finetune`,
/// `sample`→`infer`).
pub fn canon_verb(v: &str) -> &str {
    match v {
        "gen" | "sample" | "generate" => "infer",
        "fine-tune" => "finetune",
        other => other,
    }
}
