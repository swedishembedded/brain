// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The windows' own flags, taken off the command line before the rest is
//! read as `infer`'s, so a value is typed and bounded exactly as there.

use zipdepth::view::View;

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

pub fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    }
}

pub fn take_count(args: &mut Vec<String>, flag: &str) -> Result<Option<u64>, String> {
    take_value(args, flag)?.map(|v| v.parse::<u64>().map_err(|_| format!("{flag} {v}: not a count"))).transpose()
}

/// `WxH` as a pair of positive sizes.
pub fn size(v: &str, flag: &str) -> Result<(u32, u32), String> {
    let (w, h) = v.split_once('x').ok_or_else(|| format!("{flag} {v}: expected WxH"))?;
    match (w.parse::<u32>(), h.parse::<u32>()) {
        (Ok(w), Ok(h)) if w > 0 && h > 0 => Ok((w, h)),
        _ => Err(format!("{flag} {v}: expected WxH")),
    }
}

/// What a window starts on: `infer`'s view, or the photo beside its depth
/// when `infer` would draw none.
pub fn start_view(inv: &capability::Invocation) -> Result<View, String> {
    match inv.get_str("view").as_deref() {
        None | Some("none") => Ok(View::Side),
        Some(name) => View::parse(name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_flag_is_taken_off_with_its_value() {
        let mut args: Vec<String> = ["--scale", "3", "--npu", "--view", "fog"].map(String::from).to_vec();
        assert_eq!(take_count(&mut args, "--scale").unwrap(), Some(3));
        assert!(take_flag(&mut args, "--npu"));
        assert_eq!(args, ["--view", "fog"]);
        assert!(take_value(&mut vec!["--res".to_string()], "--res").is_err());
        assert_eq!(size("640x480", "--res"), Ok((640, 480)));
        assert!(size("640", "--res").is_err() && size("0x480", "--res").is_err());
    }
}
