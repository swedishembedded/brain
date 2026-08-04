// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! String Reverser dataset generator.
//!
//! Faithful port of nanogpt's `data_generators/reverser.py`. Produces lines of
//! the form `STRING=REVERSED\n`, each at most 64 characters including the
//! trailing newline. Four string flavors are chosen with ~25% probability each:
//!
//! * lowercase letters (length 3-20),
//! * mixed-case ASCII letters (length 3-20),
//! * alphanumeric (length 3-25),
//! * letters/digits/spaces (length 3-20, guaranteeing at least one space).
//!
//! Whitespace is normalized with the equivalent of Python's
//! `" ".join(s.split())` before reversing, and the reversed string is the
//! per-character reverse (Python `s[::-1]`).

use crate::rng::Rng;

/// Maximum characters per line, including the trailing newline.
const MAX_CHARS_PER_LINE: usize = 64;

const ASCII_LOWERCASE: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const ASCII_LETTERS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const ALPHANUMERIC: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const LETTERS_DIGITS_SPACE: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ";

/// Build a random string of `len` characters drawn from `char_set`.
fn random_string(char_set: &[u8], len: usize, rng: &mut Rng) -> String {
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        s.push(*rng.choice(char_set) as char);
    }
    s
}

/// Lowercase string, length 3-20.
fn generate_simple_string(rng: &mut Rng) -> String {
    let length = rng.gen_range_inclusive(3, 20) as usize;
    random_string(ASCII_LOWERCASE, length, rng)
}

/// Mixed-case string, length 3-20.
fn generate_mixed_case_string(rng: &mut Rng) -> String {
    let length = rng.gen_range_inclusive(3, 20) as usize;
    random_string(ASCII_LETTERS, length, rng)
}

/// Alphanumeric string, length 3-25.
fn generate_alphanumeric_string(rng: &mut Rng) -> String {
    let length = rng.gen_range_inclusive(3, 25) as usize;
    random_string(ALPHANUMERIC, length, rng)
}

/// String including spaces, length 3-20, guaranteed to contain at least one
/// space.
fn generate_string_with_spaces(rng: &mut Rng) -> String {
    let length = rng.gen_range_inclusive(3, 20) as usize;
    let mut result = random_string(LETTERS_DIGITS_SPACE, length, rng);

    // Ensure at least one space is included.
    if !result.contains(' ') {
        // Insert a space at a random position in `1..=len-1`, matching the
        // reference's `random.randint(1, len(result) - 1)`. All chars so far
        // are single-byte ASCII, so byte and char positions coincide.
        let pos = rng.gen_range_inclusive(1, result.len() as i64 - 1) as usize;
        result.insert(pos, ' ');
    }
    result
}

/// Normalize whitespace like Python's `" ".join(s.split())`: split on any run
/// of ASCII whitespace, discard empties, and rejoin with single spaces.
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Reverse a string by Unicode scalar (Python `s[::-1]`).
fn reverse_chars(s: &str) -> String {
    s.chars().rev().collect()
}

/// Generate one valid `STRING=REVERSED\n` line within the character limit.
///
/// Retries up to `max_attempts` times; if none fits, falls back to a truncated
/// lowercase string (matching the reference's fallback).
fn generate_line_with_char_limit(rng: &mut Rng) -> String {
    const MAX_ATTEMPTS: usize = 50;

    for _ in 0..MAX_ATTEMPTS {
        let rand = rng.next_f64();
        let input_str = if rand < 0.25 {
            generate_simple_string(rng)
        } else if rand < 0.5 {
            generate_mixed_case_string(rng)
        } else if rand < 0.75 {
            generate_alphanumeric_string(rng)
        } else {
            generate_string_with_spaces(rng)
        };

        // Normalize whitespace but preserve interior spaces.
        let input_str = normalize_whitespace(&input_str);
        if input_str.is_empty() {
            continue;
        }

        let reversed_str = reverse_chars(&input_str);
        let line = format!("{input_str}={reversed_str}\n");

        if line.chars().count() <= MAX_CHARS_PER_LINE {
            return line;
        }
    }

    // Fallback: truncate a fresh lowercase string to 10 chars.
    let simple = generate_simple_string(rng);
    let input_str: String = simple.chars().take(10).collect();
    let reversed_str = reverse_chars(&input_str);
    format!("{input_str}={reversed_str}\n")
}

/// Generate `n_examples` string-reversal lines (`s=reverse(s)\n`), concatenated.
/// Every line is <= 64 chars incl. newline. Deterministic for a fixed `rng`.
pub fn generate(n_examples: usize, rng: &mut Rng) -> String {
    let mut out = String::new();
    for _ in 0..n_examples {
        out.push_str(&generate_line_with_char_limit(rng));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_well_formed() {
        let mut rng = Rng::new(123);
        let data = generate(5000, &mut rng);

        let mut count = 0;
        for line in data.split_inclusive('\n') {
            count += 1;

            // <= 64 chars including newline.
            assert!(
                line.chars().count() <= MAX_CHARS_PER_LINE,
                "line too long: {line:?}"
            );

            // Must end with a newline.
            assert!(line.ends_with('\n'), "line missing newline: {line:?}");

            let body = line.trim_end_matches('\n');

            // Exactly one '='.
            assert_eq!(
                body.matches('=').count(),
                1,
                "line must contain exactly one '=': {line:?}"
            );

            // Right side equals char-reverse of left side.
            let (left, right) = body.split_once('=').unwrap();
            
            
            assert_eq!(
                right,
                reverse_chars(left),
                "right side is not the char-reverse of the left: {line:?}"
            );

            // Left side is non-empty.
            assert!(!left.is_empty(), "left side empty: {line:?}");
        }

        assert_eq!(count, 5000, "wrong number of lines");
    }

    #[test]
    fn correct_line_count() {
        let mut rng = Rng::new(7);
        for &n in &[0usize, 1, 10, 100] {
            let data = generate(n, &mut rng);
            assert_eq!(data.matches('\n').count(), n);
        }
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut a = Rng::new(2024);
        let mut b = Rng::new(2024);
        assert_eq!(generate(1000, &mut a), generate(1000, &mut b));
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize_whitespace("  a   b  c "), "a b c");
        assert_eq!(normalize_whitespace("   "), "");
        assert_eq!(normalize_whitespace("abc"), "abc");
    }

    #[test]
    fn reverse_is_per_char() {
        assert_eq!(reverse_chars("abc"), "cba");
        assert_eq!(reverse_chars("a b"), "b a");
        assert_eq!(reverse_chars(""), "");
    }
}
