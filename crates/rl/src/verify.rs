// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Programmatic verifiers for reinforcement learning on verifiable answers.
//!
//! [`MathAnswer`] grades a math completion the way the R1 recipe's reward
//! does: the final answer is what follows the reasoning (after the last
//! `</think>`), preferably inside the last `\boxed{...}`, else the last number
//! in it. Answers compare as exact rationals, so `0.5`, `1/2` and
//! `\frac{1}{2}` are one answer and no float tolerance can let a wrong one
//! through.
//!
//! Swedish Embedded AB implements reward modelling and reinforcement
//! fine-tuning of reasoning models for its clients. If your team needs
//! expertise in training models on verifiable rewards, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;

use crate::env::{Reward, Step, Task, Verifier};

/// An exact rational `num / den` in lowest terms, `den > 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    pub num: i128,
    pub den: i128,
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Rational {
    /// `num / den` reduced, or `None` for a zero denominator.
    pub fn new(num: i128, den: i128) -> Option<Rational> {
        if den == 0 {
            return None;
        }
        let g = gcd(num, den).max(1);
        let sign = if den < 0 { -1 } else { 1 };
        Some(Rational { num: sign * num / g, den: sign * den / g })
    }

    /// An answer as written: an integer (`-12`, `1,234`), a decimal
    /// (`0.125`, `.5`), a fraction (`3/4`, `\frac{3}{4}`, `\dfrac34`), with
    /// surrounding `$`, `\(...\)`, a trailing full stop, `\!`/`\,` spacing and
    /// a leading `x =` tolerated. `None` for anything else - never a guess.
    pub fn parse(text: &str) -> Option<Rational> {
        let s = normalize(text);
        let s = s.as_str();
        if s.is_empty() {
            return None;
        }
        if let Some(r) = parse_frac(s) {
            return Some(r);
        }
        if let Some((a, b)) = s.split_once('/') {
            let (a, b) = (parse_decimal(a)?, parse_decimal(b)?);
            return Rational::new(a.num.checked_mul(b.den)?, a.den.checked_mul(b.num)?);
        }
        parse_decimal(s)
    }
}

/// Strip what surrounds a number without changing its value.
fn normalize(text: &str) -> String {
    let mut s = text.trim().to_string();
    for (open, close) in [("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")] {
        while s.starts_with(open) && s.ends_with(close) && s.len() >= open.len() + close.len() {
            s = s[open.len()..s.len() - close.len()].trim().to_string();
        }
    }
    if let Some((lhs, rhs)) = s.split_once('=') {
        // `x = 4`: the value, when the left side is a bare name.
        if lhs.trim().chars().all(|c| c.is_ascii_alphabetic() || c == '_' || c == ' ') && !lhs.trim().is_empty() {
            s = rhs.trim().to_string();
        }
    }
    let s = s.trim_end_matches('.').replace("\\!", "").replace("\\,", "").replace("{,}", ",");
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// `\frac{a}{b}`, `\dfrac{a}{b}`, `\tfrac{a}{b}` and the brace-less
/// `\frac34`, with an optional leading sign.
fn parse_frac(s: &str) -> Option<Rational> {
    let (neg, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let body = ["\\frac", "\\dfrac", "\\tfrac"].iter().find_map(|p| s.strip_prefix(p))?;
    let (a, rest) = frac_arg(body)?;
    let (b, rest) = frac_arg(rest)?;
    if !rest.is_empty() {
        return None;
    }
    let (a, b) = (Rational::parse(a)?, Rational::parse(b)?);
    let r = Rational::new(a.num.checked_mul(b.den)?, a.den.checked_mul(b.num)?)?;
    Some(if neg { Rational { num: -r.num, ..r } } else { r })
}

/// One `\frac` argument: a braced group, or a single character.
fn frac_arg(s: &str) -> Option<(&str, &str)> {
    if let Some(rest) = s.strip_prefix('{') {
        let end = matching_brace(rest)?;
        Some((&rest[..end], &rest[end + 1..]))
    } else {
        let c = s.chars().next()?;
        Some((&s[..c.len_utf8()], &s[c.len_utf8()..]))
    }
}

/// The index of the `}` closing a group whose `{` precedes `s`.
fn matching_brace(s: &str) -> Option<usize> {
    let mut depth = 1usize;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// An optionally signed decimal with optional thousands commas
/// (`1,234.5`): each comma group after the first must be three digits, so
/// `1,23` is not a number.
fn parse_decimal(s: &str) -> Option<Rational> {
    let (neg, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (int, frac) = match s.split_once('.') {
        Some((i, f)) => (i, f),
        None => (s, ""),
    };
    if int.is_empty() && frac.is_empty() {
        return None;
    }
    let groups: Vec<&str> = int.split(',').collect();
    if groups.len() > 1 && (groups[0].is_empty() || groups[0].len() > 3 || groups[1..].iter().any(|g| g.len() != 3)) {
        return None;
    }
    let digits: String = groups.concat();
    if !digits.chars().all(|c| c.is_ascii_digit()) || !frac.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut num: i128 = 0;
    for c in digits.chars().chain(frac.chars()) {
        num = num.checked_mul(10)?.checked_add(c.to_digit(10)? as i128)?;
    }
    let den = 10i128.checked_pow(frac.len() as u32)?;
    Rational::new(if neg { -num } else { num }, den)
}

/// Where a completion's final answer is, and how it was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extracted {
    pub text: String,
    /// Found inside `\boxed{...}` rather than taken as the last number.
    pub boxed: bool,
}

/// The final answer in `completion`: the text after the last `</think>`
/// (a completion whose reasoning never closed has no answer), then the
/// content of the last `\boxed{...}` or `\fbox{...}` in it, else its last
/// number.
pub fn extract_answer(completion: &str) -> Option<Extracted> {
    let answer = match completion.rfind("</think>") {
        Some(i) => &completion[i + "</think>".len()..],
        None if completion.contains("<think>") => return None,
        None => completion,
    };
    let boxed = ["\\boxed{", "\\fbox{"].iter().filter_map(|m| answer.rfind(m).map(|i| (i, m.len()))).max_by_key(|(i, _)| *i);
    if let Some((i, len)) = boxed {
        let rest = &answer[i + len..];
        let end = matching_brace(rest)?;
        return Some(Extracted { text: rest[..end].to_string(), boxed: true });
    }
    last_number(answer).map(|text| Extracted { text, boxed: false })
}

/// The last number-shaped run in `text`: digits with an optional sign,
/// decimal point, thousands commas, or a `/` between two such.
fn last_number(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let is_part = |c: char| c.is_ascii_digit() || matches!(c, '.' | ',' | '/');
    let mut end = chars.len();
    while end > 0 && !chars[end - 1].is_ascii_digit() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut start = end;
    while start > 0 && is_part(chars[start - 1]) {
        start -= 1;
    }
    while start < end && !chars[start].is_ascii_digit() && chars[start] != '.' {
        start += 1;
    }
    if start > 0 && chars[start - 1] == '-' {
        start -= 1;
    }
    let s: String = chars[start..end].iter().collect();
    Some(s.trim_start_matches([',', '/']).to_string())
}

/// Rewards a math completion 1.0 when its final answer equals the task's,
/// as exact rationals, and 0.0 otherwise - including when there is no
/// final answer or it is not a number. `Task::answer` is the expected
/// answer as a JSON number or string. `parts` records `correct` and
/// `boxed` (the answer was given in `\boxed{}`).
pub struct MathAnswer<D: Fn(&[u32]) -> String> {
    decode: D,
}

impl<D: Fn(&[u32]) -> String> MathAnswer<D> {
    /// A verifier that reads completions through `decode` (the policy's
    /// detokenizer).
    pub fn new(decode: D) -> MathAnswer<D> {
        MathAnswer { decode }
    }

    /// The grade of one decoded completion against an expected answer.
    pub fn grade(completion: &str, expected: &serde_json::Value) -> Reward {
        let want = match expected {
            serde_json::Value::String(s) => Rational::parse(s),
            serde_json::Value::Number(n) => Rational::parse(&n.to_string()),
            _ => None,
        };
        let got = extract_answer(completion);
        let correct = matches!((&want, got.as_ref().and_then(|g| Rational::parse(&g.text))), (Some(w), Some(g)) if *w == g);
        let boxed = got.is_some_and(|g| g.boxed);
        let parts = BTreeMap::from([("correct".to_string(), correct as u8 as f32), ("boxed".to_string(), boxed as u8 as f32)]);
        Reward { value: correct as u8 as f32, parts }
    }
}

impl<D: Fn(&[u32]) -> String> Verifier for MathAnswer<D> {
    fn verify(&self, task: &Task, _transcript: &[Step], completion: &[u32]) -> Reward {
        MathAnswer::<D>::grade(&(self.decode)(completion), &task.answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn r(num: i128, den: i128) -> Option<Rational> {
        Rational::new(num, den)
    }

    #[test]
    fn answers_parse_as_exact_rationals() {
        let cases: &[(&str, Option<Rational>)] = &[
            ("42", r(42, 1)),
            ("-7", r(-7, 1)),
            ("+3", r(3, 1)),
            ("0.5", r(1, 2)),
            (".5", r(1, 2)),
            ("2.50", r(5, 2)),
            ("1,234", r(1234, 1)),
            ("1,234,567.25", r(4938269, 4)),
            ("3/4", r(3, 4)),
            ("-6/8", r(-3, 4)),
            ("\\frac{3}{4}", r(3, 4)),
            ("\\dfrac{10}{4}", r(5, 2)),
            ("\\tfrac12", r(1, 2)),
            ("-\\frac{1}{3}", r(-1, 3)),
            ("\\frac{-2}{6}", r(-1, 3)),
            ("$42$", r(42, 1)),
            ("\\(7\\)", r(7, 1)),
            ("x = 4", r(4, 1)),
            ("12.", r(12, 1)),
            ("1\\,000", r(1000, 1)),
            ("1{,}000", r(1000, 1)),
            // Not numbers: refused, never guessed.
            ("1,23", None),
            ("12a", None),
            ("", None),
            ("\\frac{1}{0}", None),
            ("3/0", None),
            ("x^2", None),
            ("\\sqrt{2}", None),
            ("1.2.3", None),
            ("99999999999999999999999999999999999999999", None),
        ];
        for (text, want) in cases {
            assert_eq!(Rational::parse(text), *want, "{text:?}");
        }
    }

    #[test]
    fn the_final_answer_is_the_last_box_after_the_reasoning() {
        let cases: &[(&str, Option<(&str, bool)>)] = &[
            ("<think>maybe \\boxed{3}</think>So \\boxed{42}.", Some(("42", true))),
            ("<think>\\boxed{3}</think> The answer is 42.", Some(("42", false))),
            ("<think>still thinking about 42", None),
            ("First \\boxed{1}, then \\boxed{2}", Some(("2", true))),
            ("\\boxed{\\frac{1}{2}}", Some(("\\frac{1}{2}", true))),
            ("\\fbox{7}", Some(("7", true))),
            ("\\boxed{unclosed", None),
            ("The total is 1,234 apples.", Some(("1,234", false))),
            ("It drops to -5 degrees", Some(("-5", false))),
            ("ratio 3/4", Some(("3/4", false))),
            ("no number here", None),
            ("</think>", None),
        ];
        for (text, want) in cases {
            let got = extract_answer(text);
            assert_eq!(got.as_ref().map(|e| (e.text.as_str(), e.boxed)), *want, "{text:?}");
        }
    }

    #[test]
    fn a_completion_is_rewarded_only_for_the_expected_answer() {
        let cases: &[(&str, serde_json::Value, f32)] = &[
            ("<think>17+25</think>\\boxed{42}", json!(42), 1.0),
            ("<think>17+25</think>\\boxed{42.0}", json!("42"), 1.0),
            ("\\boxed{\\frac{1}{2}}", json!("0.5"), 1.0),
            ("\\boxed{2/4}", json!("1/2"), 1.0),
            ("The answer is 42.", json!(42), 1.0),
            ("<think>\\boxed{42}</think>\\boxed{41}", json!(42), 0.0),
            ("<think>the answer is 42", json!(42), 0.0),
            ("\\boxed{42.0001}", json!(42), 0.0),
            ("\\boxed{forty-two}", json!(42), 0.0),
            ("\\boxed{42}", json!("forty-two"), 0.0),
            ("\\boxed{42}", json!(null), 0.0),
        ];
        for (text, want, value) in cases {
            assert_eq!(MathAnswer::<fn(&[u32]) -> String>::grade(text, want).value, *value, "{text:?} vs {want}");
        }
        let boxed = MathAnswer::<fn(&[u32]) -> String>::grade("\\boxed{42}", &json!(42));
        assert_eq!((boxed.parts["correct"], boxed.parts["boxed"]), (1.0, 1.0));
    }

    /// Through the `Verifier` seam, completions are read with the policy's
    /// own detokenizer.
    #[test]
    fn verify_decodes_the_completion_first() {
        let v = MathAnswer::new(|ids: &[u32]| ids.iter().map(|&i| char::from_digit(i, 10).unwrap()).collect());
        let task = Task { id: "t".into(), prompt: vec![], answer: json!(42) };
        assert_eq!(v.verify(&task, &[], &[4, 2]).value, 1.0);
        assert_eq!(v.verify(&task, &[], &[4, 3]).value, 0.0);
    }
}
