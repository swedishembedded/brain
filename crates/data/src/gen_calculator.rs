// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Synthetic calculator dataset generator.
//!
//! Ported from nanogpt's `data_generators/calculator.py`. Each emitted line has
//! the form `EXPRESSION=RESULT\n` and is at most [`MAX_CHARS_PER_LINE`] long
//! (including the trailing newline). Expressions use 1-2 of the operators
//! `+ - * /` over numbers of varied precision, and the result is the arithmetic
//! value rounded to 4 decimals.
//!
//! We cannot byte-reproduce Python's `random` stream, so this is *functionally*
//! equivalent to the reference (same task, format, 64-char invariant, and
//! distributions) rather than bit-identical. For a fixed [`crate::rng::Rng`]
//! the output is fully deterministic.

use crate::rng::Rng;

/// Maximum characters per line, including the trailing newline.
const MAX_CHARS_PER_LINE: usize = 64;

/// The four operators we draw from.
const OPS: [char; 4] = ['+', '-', '*', '/'];

/// Generate `n_examples` calculator lines (`expr=result\n`), concatenated.
/// Every line is <= 64 chars including the trailing newline. Deterministic for a fixed `rng`.
pub fn generate(n_examples: usize, rng: &mut Rng) -> String {
    // Roughly 24 bytes/line on average; pre-reserve to avoid reallocations.
    let mut out = String::with_capacity(n_examples * 24);
    for _ in 0..n_examples {
        out.push_str(&generate_expression_with_char_limit(rng, 50));
    }
    out
}

/// Generate a number with varied precision (Python `generate_number`).
///
/// ~40% are integers; the rest are rounded to 1/2/3/4 decimal places with
/// decreasing probability.
fn generate_number(min_val: f64, max_val: f64, rng: &mut Rng) -> f64 {
    let rand = rng.next_f64();
    if rand < 0.4 {
        // Python: random.randint(int(max(1, min_val)), int(max_val)).
        let lo = (min_val.max(1.0)) as i64;
        let hi = max_val as i64;
        rng.gen_range_inclusive(lo, hi) as f64
    } else if rand < 0.6 {
        round_to(rng.uniform(min_val, max_val), 1)
    } else if rand < 0.8 {
        round_to(rng.uniform(min_val, max_val), 2)
    } else if rand < 0.9 {
        round_to(rng.uniform(min_val, max_val), 3)
    } else {
        round_to(rng.uniform(min_val, max_val), 4)
    }
}

/// Round to `decimals` decimal places (matches Python's `round(x, n)` closely
/// enough for formatting; ties are not load-bearing here).
fn round_to(x: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (x * factor).round() / factor
}

/// Format a number consistently (Python `format_number`).
///
/// Values that are (within tolerance) integers print without a decimal point;
/// otherwise trailing zeros — and a trailing dot — are stripped.
fn format_number(num: f64) -> String {
    if (num - num.round()).abs() < 1e-9 {
        return format!("{}", num.round() as i64);
    }
    // Python uses f"{num:.10f}" then strips trailing zeros and a trailing '.'.
    let s = format!("{num:.10}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

/// Generate a simple 1-2 operation expression (Python `generate_simple_expression`).
fn generate_simple_expression(rng: &mut Rng) -> String {
    let num_ops = rng.gen_range_inclusive(1, 2) as usize;

    let mut numbers = Vec::with_capacity(num_ops + 1);
    for _ in 0..(num_ops + 1) {
        // Magnitude buckets: 0.1-10 / 10-100 / 100-1000.
        let mut num = if rng.next_f64() < 0.3 {
            generate_number(0.1, 10.0, rng)
        } else if rng.next_f64() < 0.6 {
            generate_number(10.0, 100.0, rng)
        } else {
            generate_number(100.0, 1000.0, rng)
        };

        // ~30% negative.
        if rng.next_f64() < 0.3 {
            num = -num;
        }
        numbers.push(num);
    }

    let operations: Vec<char> = (0..num_ops).map(|_| *rng.choice(&OPS)).collect();

    let mut expr = format_number(numbers[0]);
    for (i, op) in operations.iter().enumerate() {
        expr.push(*op);
        expr.push_str(&format_number(numbers[i + 1]));
    }
    expr
}

/// Generate a valid expression line within the character limit (Python
/// `generate_expression_with_char_limit`), retrying on division-by-zero,
/// non-finite, overlarge, or over-long results. Falls back to a guaranteed-safe
/// two-operand expression if all attempts fail.
fn generate_expression_with_char_limit(rng: &mut Rng, max_attempts: usize) -> String {
    for _ in 0..max_attempts {
        let expression = generate_simple_expression(rng);

        if let Some(result) = eval_expression(&expression) {
            if !result.is_finite() || result.abs() > 1e10 {
                continue;
            }
            let result = round_to(result, 4);
            let result_str = format_number(result);
            let line = format!("{expression}={result_str}\n");
            if line.len() <= MAX_CHARS_PER_LINE {
                return line;
            }
        }
        // eval failed (e.g. division by zero) -> retry.
    }

    // Fallback: only + - * over small numbers, so the result is always finite
    // and the line is always short enough.
    let a = generate_number(1.0, 10.0, rng);
    let b = generate_number(1.0, 10.0, rng);
    let op = *rng.choice(&['+', '-', '*']);
    let expression = format!("{}{}{}", format_number(a), op, format_number(b));
    // This expression cannot fail to evaluate.
    let result = eval_expression(&expression).unwrap_or(0.0);
    let result_str = format_number(round_to(result, 4));
    format!("{expression}={result_str}\n")
}

/// Evaluate a simple arithmetic expression with standard `* /` over `+ -`
/// precedence, left to right. Returns `None` on division by zero (matching
/// Python's `ZeroDivisionError` retry path).
///
/// The grammar is restricted to what the generators emit: signed numeric
/// literals separated by single binary operators. We therefore parse with a
/// two-pass shunting approach: first split into terms/operators, multiply and
/// divide within additive runs, then sum.
fn eval_expression(expr: &str) -> Option<f64> {
    let tokens = tokenize(expr)?;

    // First pass: collapse '*' and '/' so we're left with additive terms.
    // `terms` holds the running additive operands; the last one is folded into
    // as we encounter multiplicative operators.
    let mut terms: Vec<f64> = Vec::new();
    let mut signs: Vec<f64> = Vec::new(); // +1.0 / -1.0 preceding each term
    let mut iter = tokens.into_iter();

    // First token is always a number.
    let first = match iter.next()? {
        Token::Num(n) => n,
        Token::Op(_) => return None,
    };
    terms.push(first);
    signs.push(1.0);

    while let Some(tok) = iter.next() {
        let op = match tok {
            Token::Op(c) => c,
            Token::Num(_) => return None,
        };
        let rhs = match iter.next()? {
            Token::Num(n) => n,
            Token::Op(_) => return None,
        };
        match op {
            '+' => {
                terms.push(rhs);
                signs.push(1.0);
            }
            '-' => {
                terms.push(rhs);
                signs.push(-1.0);
            }
            '*' => {
                let last = terms.last_mut().unwrap();
                *last *= rhs;
            }
            '/' => {
                if rhs == 0.0 {
                    return None; // division by zero -> retry
                }
                let last = terms.last_mut().unwrap();
                *last /= rhs;
            }
            _ => return None,
        }
    }

    // Second pass: apply the additive signs.
    let mut acc = 0.0;
    for (term, sign) in terms.iter().zip(signs.iter()) {
        acc += sign * term;
    }
    Some(acc)
}

/// A token in a simple arithmetic expression.
enum Token {
    Num(f64),
    Op(char),
}

/// Tokenize a generator-produced expression into numbers and binary operators.
///
/// Handles a leading unary minus and unary minus directly after a binary
/// operator (the generators produce e.g. `5*-3`), folding the sign into the
/// numeric literal so the evaluator only sees binary operators.
fn tokenize(expr: &str) -> Option<Vec<Token>> {
    let bytes = expr.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    // True when the next '-' (or '+') is a unary sign on a literal rather than a
    // binary operator: at the start, or right after a binary operator.
    let mut expect_number = true;

    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '+' || c == '-' || c == '*' || c == '/' {
            if expect_number && (c == '-' || c == '+') {
                // Unary sign: fold into the following number literal.
                let start = i;
                i += 1;
                let num_start = i;
                while i < bytes.len() && is_num_char(bytes[i] as char) {
                    i += 1;
                }
                if i == num_start {
                    return None; // dangling sign
                }
                let lit = &expr[start..i];
                let n: f64 = lit.parse().ok()?;
                tokens.push(Token::Num(n));
                expect_number = false;
            } else {
                tokens.push(Token::Op(c));
                expect_number = true;
                i += 1;
            }
        } else if is_num_char(c) {
            let num_start = i;
            while i < bytes.len() && is_num_char(bytes[i] as char) {
                i += 1;
            }
            let n: f64 = expr[num_start..i].parse().ok()?;
            tokens.push(Token::Num(n));
            expect_number = false;
        } else {
            return None; // unexpected character
        }
    }
    Some(tokens)
}

/// Whether `c` can appear inside a numeric literal (digits and decimal point).
fn is_num_char(c: char) -> bool {
    c.is_ascii_digit() || c == '.'
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluate the left-hand side of a `lhs=rhs` line for spot-checking.
    fn eval_lhs(line: &str) -> f64 {
        let lhs = line.split('=').next().unwrap();
        eval_expression(lhs).unwrap()
    }

    #[test]
    fn lines_respect_char_limit_and_have_one_equals() {
        let mut rng = Rng::new(123);
        let text = generate(5000, &mut rng);
        for line in text.lines() {
            // `lines()` strips the '\n', so add it back for the limit check.
            assert!(
                line.len() + 1 <= MAX_CHARS_PER_LINE,
                "line too long: {line:?}"
            );
            assert_eq!(
                line.matches('=').count(),
                1,
                "expected exactly one '=' in {line:?}"
            );
        }
    }

    #[test]
    fn produces_exactly_n_lines() {
        let mut rng = Rng::new(7);
        let n = 1234;
        let text = generate(n, &mut rng);
        assert!(text.ends_with('\n'));
        assert_eq!(text.lines().count(), n);
        // Every line ends with a newline -> trailing-newline count equals n.
        assert_eq!(text.matches('\n').count(), n);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut a = Rng::new(2024);
        let mut b = Rng::new(2024);
        assert_eq!(generate(2000, &mut a), generate(2000, &mut b));
    }

    #[test]
    fn lhs_evaluates_to_rhs() {
        let mut rng = Rng::new(99);
        let text = generate(2000, &mut rng);
        for line in text.lines() {
            let mut parts = line.split('=');
            let lhs = parts.next().unwrap();
            let rhs: f64 = parts.next().unwrap().parse().unwrap();
            let computed = round_to(eval_expression(lhs).unwrap(), 4);
            assert!(
                (computed - rhs).abs() < 1e-3,
                "lhs {lhs:?} = {computed} but rhs is {rhs}"
            );
        }
    }

    #[test]
    fn evaluator_respects_precedence() {
        // 2 + 3 * 4 = 14, not 20.
        assert!((eval_expression("2+3*4").unwrap() - 14.0).abs() < 1e-9);
        // 10 / 2 - 3 = 2.
        assert!((eval_expression("10/2-3").unwrap() - 2.0).abs() < 1e-9);
        // Unary minus literal: 5 * -3 = -15.
        assert!((eval_expression("5*-3").unwrap() + 15.0).abs() < 1e-9);
        // Division by zero -> None (retry path).
        assert!(eval_expression("4/0").is_none());
        // Leading negative.
        assert!((eval_expression("-7+2").unwrap() + 5.0).abs() < 1e-9);
    }

    #[test]
    fn format_number_matches_reference_behavior() {
        assert_eq!(format_number(5.0), "5");
        assert_eq!(format_number(-3.0), "-3");
        assert_eq!(format_number(2.5), "2.5");
        assert_eq!(format_number(2.50), "2.5");
        // Spot check a generated line evaluates sanely.
        let mut rng = Rng::new(1);
        let text = generate(10, &mut rng);
        let first = text.lines().next().unwrap();
        let _ = eval_lhs(first); // does not panic
    }
}
