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

/// `--out`'s value, accepting either a dedicated command's own bare-path
/// spelling (`--out out.png`) or the generic capability-manifest convention
/// `brain caps <arch>` documents (`--out <blob_name>=<path>`, what `brain
/// do`/D-Bus callers actually type) - the manifest line is the only place a
/// caller learns an action's `--out` syntax from, and typing it against a
/// dedicated command whose own hand-rolled parser only knew the bare form
/// used to silently write a file literally named `image=out.png` (or worse,
/// turn a path VALUE into a nested `image=/...` directory tree), with no
/// error at all - found live on `brain flux2 generate`. Strips `blob_name=`
/// when present; anything else (including a value that merely starts with
/// the name as part of its own filename, not followed by `=`) passes
/// through unchanged.
///
/// A malformed value exits the process here, naming the flag, exactly as a
/// value-less `--flag` does in [`Args::take_str`] above - the mistake is the
/// caller's typing and this is the last point at which it is still
/// attributable to it.
pub fn strip_out_name_prefix<'a>(raw: &'a str, blob_name: &str) -> &'a str {
    parse_out_name_prefix(raw, blob_name).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    })
}

/// [`strip_out_name_prefix`]'s fallible core.
///
/// `--out image=` names the blob and then nothing at all. Stripping the prefix
/// leaves an EMPTY path, which fails unpredictably deep inside whatever opens
/// the output file - a "No such file or directory" on a path the caller never
/// typed, or, before the prefix form was understood at all, a file literally
/// named `image=`. Neither is something a caller can act on, so the empty
/// value is refused here instead, quoting what was typed.
pub fn parse_out_name_prefix<'a>(raw: &'a str, blob_name: &str) -> Result<&'a str, String> {
    let path = raw.strip_prefix(blob_name).and_then(|r| r.strip_prefix('=')).unwrap_or(raw);
    if path.is_empty() {
        return Err(format!("--out {raw:?} names no output path; write it as '--out <path>' or '--out {blob_name}=<path>'"));
    }
    Ok(path)
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

#[cfg(test)]
mod out_name_prefix_tests {
    use super::{parse_out_name_prefix, strip_out_name_prefix};

    #[test]
    fn accepts_either_the_bare_path_or_the_documented_name_equals_path_form() {
        assert_eq!(strip_out_name_prefix("out.png", "image"), "out.png");
        assert_eq!(strip_out_name_prefix("image=out.png", "image"), "out.png");
        assert_eq!(strip_out_name_prefix("image=out/a/b.png", "image"), "out/a/b.png");
        assert_eq!(strip_out_name_prefix("imageX=out.png", "image"), "imageX=out.png");
        assert_eq!(strip_out_name_prefix("adapter=my.brain", "adapter"), "my.brain");
        assert_eq!(strip_out_name_prefix("video=clip.mp4", "video"), "clip.mp4");
    }

    /// `--out image=` names the blob and then no path at all. Stripping the
    /// prefix leaves an EMPTY path, which fails unpredictably deep inside
    /// whatever opens the file (and on some platforms creates nothing at all)
    /// rather than telling the caller what they typed wrong. Refuse it here,
    /// where the mistake is still attributable to the flag.
    #[test]
    fn a_name_equals_with_no_path_is_refused_rather_than_becoming_an_empty_path() {
        let e = parse_out_name_prefix("image=", "image").unwrap_err();
        assert!(e.contains("--out"), "the refusal must name the flag: {e}");
        assert!(e.contains("image="), "and quote what was typed: {e}");
        // A bare empty value is the same mistake without the prefix.
        assert!(parse_out_name_prefix("", "image").is_err());
        // Everything the accepted forms above allow still parses.
        assert_eq!(parse_out_name_prefix("image=out.png", "image").unwrap(), "out.png");
        assert_eq!(parse_out_name_prefix("out.png", "image").unwrap(), "out.png");
        assert_eq!(parse_out_name_prefix("imageX=", "image").unwrap(), "imageX=");
    }
}
