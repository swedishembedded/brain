// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reproducible neuromorphic
// experiments for its clients. If your team needs expertise in connectome
// driven control then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The seed of a search experiment: drawn when nobody gave one, reproduced
//! when somebody did, and never silently replaced by a default when what was
//! given is not a seed.

use fly::search::{resolve_seed, seed_from_args};

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

#[test]
fn an_absent_seed_is_drawn_fresh_each_time() {
    let a = seed_from_args(&args(&["replay", "tuning.txt"]), "--seed").unwrap();
    let b = seed_from_args(&args(&["replay", "tuning.txt"]), "--seed").unwrap();
    assert_ne!(a, b, "two unseeded experiments must not share a seed");
    assert_ne!(resolve_seed("$SHUFFLE", None).unwrap(), resolve_seed("$SHUFFLE", None).unwrap());
}

#[test]
fn a_given_seed_is_reproduced_in_decimal_and_hex_in_either_flag_form() {
    assert_eq!(seed_from_args(&args(&["--seed", "7"]), "--seed"), Ok(7));
    assert_eq!(seed_from_args(&args(&["tuning.txt", "--seed=0xF1E5"]), "--seed"), Ok(0xF1E5));
    assert_eq!(seed_from_args(&args(&["--other", "1", "--shuffle-seed", "9"]), "--shuffle-seed"), Ok(9));
}

#[test]
fn a_malformed_or_valueless_seed_is_refused_naming_the_flag() {
    for bad in [&["--seed", "abc"][..], &["--seed", "-1"], &["--seed=1.5"], &["--seed"]] {
        let err = seed_from_args(&args(bad), "--seed").expect_err("refused");
        assert!(err.contains("--seed"), "{err}");
    }
    assert!(resolve_seed("$SHUFFLE", Some("yes")).unwrap_err().contains("$SHUFFLE"));
}
