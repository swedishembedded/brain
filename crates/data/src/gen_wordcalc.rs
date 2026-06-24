// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Word-calculator dataset generator.
//!
//! Port of nanogpt's `data_generators/wordcalc.py`. Produces lines of the form
//! `WORDS=MATH\n`, where the left side is a natural-language rendering of a math
//! expression (e.g. `twentythree plus five`) and the right side is the compact
//! digit/operator form (e.g. `23+5`). Every line is at most 64 chars including
//! the trailing newline.
//!
//! This is *functionally* equivalent to the Python source rather than
//! byte-identical: the same task, format, and distributions, but driven by
//! brain's deterministic [`crate::rng::Rng`] instead of Python's `random`.

/// Maximum line length, including the trailing newline.
const MAX_CHARS_PER_LINE: usize = 64;

/// Convert a number to its word representation, without spaces or hyphens.
///
/// e.g. `23 -> "twentythree"`, `105 -> "onehundredfive"`, `-5 -> "negativefive"`.
fn number_to_words(num: i64) -> String {
    if num == 0 {
        return "zero".to_string();
    }
    if num < 0 {
        // `-num` can't overflow for any value we generate (well within i64 range).
        return format!("negative{}", number_to_words(-num));
    }

    const ONES: [&str; 10] = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    const TEENS: [&str; 10] = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    if num < 10 {
        ONES[num as usize].to_string()
    } else if num < 20 {
        TEENS[(num - 10) as usize].to_string()
    } else if num < 100 {
        let tens_digit = (num / 10) as usize;
        let ones_digit = (num % 10) as usize;
        if ones_digit == 0 {
            TENS[tens_digit].to_string()
        } else {
            format!("{}{}", TENS[tens_digit], ONES[ones_digit])
        }
    } else if num < 1000 {
        let hundreds_digit = (num / 100) as usize;
        let remainder = num % 100;
        let mut result = format!("{}hundred", ONES[hundreds_digit]);
        if remainder > 0 {
            result.push_str(&number_to_words(remainder));
        }
        result
    } else if num < 1_000_000 {
        let thousands = num / 1000;
        let remainder = num % 1000;
        let mut result = format!("{}thousand", number_to_words(thousands));
        if remainder > 0 {
            result.push_str(&number_to_words(remainder));
        }
        result
    } else {
        // Fallback for very large numbers (never reached for generated inputs).
        num.to_string()
    }
}

/// Convert an operator character to its word representation.
fn operator_to_word(op: char) -> &'static str {
    match op {
        '+' => "plus",
        '-' => "minus",
        '*' => "times",
        '/' => "divided by",
        _ => unreachable!("unsupported operator: {op}"),
    }
}

/// Generate a simple expression, returning `(math_expr, word_expr)`.
///
/// Uses 1-3 operators. Numbers are drawn from buckets (1-20 ~50%, 21-100, and
/// 100-999), with ~15% made negative. The math form joins numbers and operators
/// with no spaces; the word form joins their word renderings with single spaces.
fn generate_simple_expression(rng: &mut crate::rng::Rng) -> (String, String) {
    let num_ops = rng.gen_range_inclusive(1, 3) as usize;

    let mut numbers: Vec<i64> = Vec::with_capacity(num_ops + 1);
    for _ in 0..num_ops + 1 {
        let mut num = if rng.next_f64() < 0.5 {
            // Small numbers (1-20).
            rng.gen_range_inclusive(1, 20)
        } else if rng.next_f64() < 0.8 {
            // Medium numbers (21-100).
            rng.gen_range_inclusive(21, 100)
        } else {
            // Larger numbers (100-999).
            rng.gen_range_inclusive(100, 999)
        };

        // Occasionally make negative.
        if rng.next_f64() < 0.15 {
            num = -num;
        }

        numbers.push(num);
    }

    const OPS: [char; 4] = ['+', '-', '*', '/'];
    let operations: Vec<char> = (0..num_ops).map(|_| *rng.choice(&OPS)).collect();

    // Build math expression: "n0 op0 n1 op1 n2 ..." with no separators.
    let mut math_expr = numbers[0].to_string();
    for (i, op) in operations.iter().enumerate() {
        math_expr.push(*op);
        math_expr.push_str(&numbers[i + 1].to_string());
    }

    // Build word expression: number-words and operator-words joined by spaces.
    let mut word_parts: Vec<String> = Vec::with_capacity(2 * num_ops + 1);
    word_parts.push(number_to_words(numbers[0]));
    for (i, op) in operations.iter().enumerate() {
        word_parts.push(operator_to_word(*op).to_string());
        word_parts.push(number_to_words(numbers[i + 1]));
    }
    let word_expr = word_parts.join(" ");

    (math_expr, word_expr)
}

/// Outcome of evaluating a math expression.
enum EvalError {
    /// A division by zero occurred (mirrors Python's `ZeroDivisionError`).
    DivisionByZero,
    /// The result was not finite (NaN or infinity).
    NotFinite,
}

/// Evaluate a math expression built from integers and the operators `+ - * /`.
///
/// Implements standard precedence: `*` and `/` bind tighter than `+` and `-`.
/// Division is floating-point (Python's `eval` on `int/int` yields a float).
/// Returns the finite numeric result, or an [`EvalError`] on division by zero
/// or a non-finite result.
fn eval_expression(expr: &str) -> Result<f64, EvalError> {
    // Tokenize into numbers and operators. A leading or post-operator '-' is a
    // sign for the following number, which matches how the expressions are
    // generated (negative operands are emitted as e.g. "5+-3").
    enum Token {
        Num(f64),
        Op(char),
    }

    let mut tokens: Vec<Token> = Vec::new();
    let bytes = expr.as_bytes();
    let mut i = 0;
    // `expect_number` is true when the next token must be (the start of) a number,
    // i.e. at the start of the expression or immediately after a binary operator.
    let mut expect_number = true;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '+' || c == '-' || c == '*' || c == '/' {
            if expect_number && (c == '-' || c == '+') {
                // Unary sign on the upcoming number.
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let n: f64 = expr[start..i].parse().map_err(|_| EvalError::NotFinite)?;
                tokens.push(Token::Num(n));
                expect_number = false;
            } else {
                tokens.push(Token::Op(c));
                i += 1;
                expect_number = true;
            }
        } else if c.is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let n: f64 = expr[start..i].parse().map_err(|_| EvalError::NotFinite)?;
            tokens.push(Token::Num(n));
            expect_number = false;
        } else {
            // Unexpected character.
            return Err(EvalError::NotFinite);
        }
    }

    // First pass: collapse `*` and `/` into a list of additive terms.
    let mut terms: Vec<f64> = Vec::new();
    let mut signs: Vec<char> = Vec::new(); // sign joining successive terms ('+' / '-')
    let mut iter = tokens.into_iter();
    let mut current = match iter.next() {
        Some(Token::Num(n)) => n,
        _ => return Err(EvalError::NotFinite),
    };
    while let Some(tok) = iter.next() {
        match tok {
            Token::Op(op @ ('*' | '/')) => {
                let rhs = match iter.next() {
                    Some(Token::Num(n)) => n,
                    _ => return Err(EvalError::NotFinite),
                };
                if op == '*' {
                    current *= rhs;
                } else {
                    if rhs == 0.0 {
                        return Err(EvalError::DivisionByZero);
                    }
                    current /= rhs;
                }
            }
            Token::Op(op @ ('+' | '-')) => {
                terms.push(current);
                signs.push(op);
                current = match iter.next() {
                    Some(Token::Num(n)) => n,
                    _ => return Err(EvalError::NotFinite),
                };
            }
            _ => return Err(EvalError::NotFinite),
        }
    }
    terms.push(current);

    // Second pass: sum the additive terms.
    let mut result = terms[0];
    for (sign, &term) in signs.iter().zip(terms.iter().skip(1)) {
        if *sign == '+' {
            result += term;
        } else {
            result -= term;
        }
    }

    if result.is_finite() {
        Ok(result)
    } else {
        Err(EvalError::NotFinite)
    }
}

/// Generate one valid expression line within the character limit.
///
/// Retries up to `max_attempts` times, skipping expressions that divide by zero,
/// produce a non-finite result, or exceed the 64-char limit. Falls back to a
/// simple addition if no attempt succeeds.
fn generate_expression_with_char_limit(rng: &mut crate::rng::Rng, max_attempts: usize) -> String {
    for _ in 0..max_attempts {
        let (math_expr, word_expr) = generate_simple_expression(rng);

        // Validate the math expression evaluates to a finite number.
        match eval_expression(&math_expr) {
            Ok(_) => {}
            Err(_) => continue,
        }

        let line = format!("{word_expr}={math_expr}\n");
        if line.len() <= MAX_CHARS_PER_LINE {
            return line;
        }
    }

    // Fallback: simple addition.
    let a = rng.gen_range_inclusive(1, 10);
    let b = rng.gen_range_inclusive(1, 10);
    format!(
        "{} plus {}={}+{}\n",
        number_to_words(a),
        number_to_words(b),
        a,
        b
    )
}

/// Generate `n_examples` word-calculator lines (`words=math\n`), concatenated.
/// Every line is <= 64 chars incl. newline. Deterministic for a fixed `rng`.
pub fn generate(n_examples: usize, rng: &mut crate::rng::Rng) -> String {
    let mut out = String::new();
    for _ in 0..n_examples {
        out.push_str(&generate_expression_with_char_limit(rng, 100));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_to_words_known_cases() {
        assert_eq!(number_to_words(0), "zero");
        assert_eq!(number_to_words(7), "seven");
        assert_eq!(number_to_words(13), "thirteen");
        assert_eq!(number_to_words(20), "twenty");
        assert_eq!(number_to_words(23), "twentythree");
        assert_eq!(number_to_words(100), "onehundred");
        assert_eq!(number_to_words(105), "onehundredfive");
        assert_eq!(number_to_words(-5), "negativefive");
    }

    #[test]
    fn evaluator_respects_precedence() {
        assert_eq!(eval_expression("2+3*4").ok(), Some(14.0));
        assert_eq!(eval_expression("10-2-3").ok(), Some(5.0));
        assert_eq!(eval_expression("6/2+1").ok(), Some(4.0));
        assert_eq!(eval_expression("5+-3").ok(), Some(2.0));
        assert_eq!(eval_expression("-5+3").ok(), Some(-2.0));
        assert!(matches!(
            eval_expression("1/0"),
            Err(EvalError::DivisionByZero)
        ));
    }

    #[test]
    fn lines_are_well_formed() {
        let mut rng = crate::rng::Rng::new(12345);
        let text = generate(5000, &mut rng);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 5000);
        for line in &lines {
            // `lines()` strips the '\n'; add it back to check full length.
            assert!(
                line.len() + 1 <= MAX_CHARS_PER_LINE,
                "line too long: {line:?}"
            );
            assert_eq!(
                line.matches('=').count(),
                1,
                "expected exactly one '=': {line:?}"
            );
        }
    }

    #[test]
    fn correct_line_count() {
        let mut rng = crate::rng::Rng::new(1);
        let text = generate(123, &mut rng);
        assert_eq!(text.matches('\n').count(), 123);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut a = crate::rng::Rng::new(99);
        let mut b = crate::rng::Rng::new(99);
        assert_eq!(generate(1000, &mut a), generate(1000, &mut b));
    }
}
