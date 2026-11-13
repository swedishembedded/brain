// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The server workflow's own flags, taken off the command line one at a time:
//! a flag left out is a value left to its default, an unknown one is refused.

/// The value of `flag`, taken off `args`; `None` when it is not there.
pub fn take_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            let v = args.get(i + 1).cloned().ok_or_else(|| format!("{flag} needs a value"))?;
            args.drain(i..=i + 1);
            Ok(Some(v))
        }
        None => Ok(None),
    }
}

/// Whether the switch `flag` was given; taken off `args`.
pub fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    }
}

/// `flag`'s value as a positive count.
pub fn take_count(args: &mut Vec<String>, flag: &str) -> Result<Option<usize>, String> {
    take_value(args, flag)?
        .map(|v| v.parse::<usize>().ok().filter(|&n| n > 0).ok_or_else(|| format!("{flag} {v}: not a positive count")))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flag_is_taken_off_with_its_value() {
        let mut args: Vec<String> = ["--cap", "384", "--turbo", "--x"].map(String::from).to_vec();
        assert_eq!(take_count(&mut args, "--cap").unwrap(), Some(384));
        assert!(take_flag(&mut args, "--turbo") && !take_flag(&mut args, "--turbo"));
        assert_eq!(args, ["--x"]);
        assert!(take_value(&mut vec!["--socket".to_string()], "--socket").is_err());
        assert!(take_count(&mut vec!["--cap".to_string(), "0".to_string()], "--cap").is_err());
        assert!(take_count(&mut vec!["--cap".to_string(), "many".to_string()], "--cap").is_err());
    }
}
