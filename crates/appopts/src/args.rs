// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A consuming flag grammar.
//!
//! Moved here from `crates/cli/src/args.rs` - unchanged, and still the CLI's
//! own parser, which is the point: a sample that needs `--device` should be
//! parsing it with the same code the `brain` binary parses it with, not with
//! a second loop that happens to agree on the easy cases.

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
    /// `--seed` for a run whose seed matters: the value given, or a freshly
    /// drawn random one that is printed to stderr so the run can be reproduced.
    /// `what` names the run in that line. A `--seed` that is not a seed
    /// (`abc`, `-1`) exits 2 naming the flag: it never falls back to a default.
    pub fn seed(&mut self, what: &str) -> u64 {
        self.seed_flag(what, "--seed")
    }

    /// [`Args::seed`] for a differently named seed flag (`--order-seed`): the
    /// value given, or a freshly drawn one printed with `flag` in the
    /// reproduce hint; a value that is not a seed exits 2 naming the flag.
    pub fn seed_flag(&mut self, what: &str, flag: &str) -> u64 {
        self.seed_opt(flag).unwrap_or_else(|| {
            let seed = seed::random_seed();
            eprintln!("{}", drawn_seed_notice(what, flag, seed));
            seed
        })
    }

    /// A seed flag whose absence means "unseeded": `Some` when given, `None`
    /// when not, and a value that is not a seed exits 2 naming the flag.
    pub fn seed_opt(&mut self, flag: &str) -> Option<u64> {
        match self.try_seed_opt(flag) {
            Ok(seed) => seed,
            Err(message) => {
                eprintln!("{message}");
                std::process::exit(2);
            }
        }
    }

    fn try_seed_opt(&mut self, flag: &str) -> Result<Option<u64>, String> {
        match self.take_str(flag) {
            None => Ok(None),
            Some(text) => seed::parse_seed(&text).map(Some).map_err(|e| format!("{flag}: {e}")),
        }
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
    /// The tokens nothing claimed.
    pub fn leftovers(&self) -> Vec<String> {
        self.toks
            .iter()
            .zip(&self.used)
            .filter(|(_, u)| !**u)
            .map(|(t, _)| t.clone())
            .collect()
    }
    /// Refuse to run with a flag nobody understood. Call once at the end.
    ///
    /// This used to print a line and carry on, which is the same failure mode
    /// `--device` has its own test about: the job runs, it finishes, it prints
    /// numbers, and the numbers are for a configuration nobody asked for.
    /// `--warmup-episodes` for `--warmup` cost a training run that way, and
    /// nothing in its output said so.
    pub fn finish(&self) {
        let extra = self.leftovers();
        if !extra.is_empty() {
            eprintln!(
                "unrecognised: {}\n  nothing here understands that flag, and running \
                 without it would run something other than what was asked for. \
                 --help lists what there is.",
                extra.join(" ")
            );
            std::process::exit(2);
        }
    }
}

/// The line printed when a run drew its own seed: says it was drawn and the
/// exact flag that reproduces it.
fn drawn_seed_notice(what: &str, flag: &str, seed: u64) -> String {
    format!("{what}: no {flag} given, using random seed {seed} (pass {flag} {seed} to reproduce)")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(tokens: &[&str]) -> Args {
        Args::new(&tokens.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn a_given_seed_is_taken_and_leaves_the_other_flags() {
        let mut a = args(&["--steps", "3", "--seed", "0x2a"]);
        assert_eq!(a.seed("train"), 42);
        assert_eq!(a.usize_or("--steps", 0), 3);
        a.finish();
    }

    #[test]
    fn an_absent_seed_is_drawn_afresh_each_time() {
        let first = args(&[]).seed("train");
        let second = args(&[]).seed("train");
        assert_ne!(first, second, "an unseeded run must not repeat a default");
    }

    #[test]
    fn a_differently_named_seed_flag_is_taken_or_drawn_like_seed() {
        assert_eq!(args(&["--order-seed", "5"]).seed_flag("eval", "--order-seed"), 5);
        let first = args(&[]).seed_flag("eval", "--order-seed");
        let second = args(&[]).seed_flag("eval", "--order-seed");
        assert_ne!(first, second, "an unseeded flag must not repeat a default");
    }

    #[test]
    fn a_malformed_seed_is_an_error_that_names_the_flag() {
        for bad in ["abc", "-1", "1.5"] {
            let err = args(&["--seed", bad]).try_seed_opt("--seed").expect_err(bad);
            assert!(err.starts_with("--seed: "), "{err}");
            assert!(err.contains(bad), "{err}");
        }
        assert_eq!(args(&["--data-seed", "7"]).try_seed_opt("--data-seed"), Ok(Some(7)));
        assert_eq!(args(&[]).try_seed_opt("--seed"), Ok(None));
    }

    #[test]
    fn the_drawn_seed_notice_says_how_to_reproduce() {
        assert_eq!(
            drawn_seed_notice("qwen3 finetune", "--seed", 9),
            "qwen3 finetune: no --seed given, using random seed 9 (pass --seed 9 to reproduce)"
        );
    }
}
