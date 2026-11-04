// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One-off structural inspection of the GenieRedux tokenizer checkpoint (in
//! gitignored scratch). Skipped unless the file is present. Run with:
//!   cargo test -p brain-wm-genie --test import_inspect -- --nocapture --ignored
#![allow(clippy::print_stderr)]
#![allow(non_snake_case)] // uppercase test-path locals (AGENTS.md: no absolute paths)

#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}


#[test]
#[ignore = "needs the 1.2GB checkpoint in scratch; run manually"]
fn inspect_tokenizer_checkpoint() {
        let CK = repo_path("scratchpad/wm-checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0.pt");
    if !std::path::Path::new(&CK).exists() {
        brain_testutil::skip(&format!("{CK} absent"));
        return;
    }
    let rep = checkpoint::torchpt::read_report(&CK).expect("read");
    eprintln!("total tensors: {}  skipped-non-tensor: {}", rep.tensors.len(), rep.skipped_non_tensor);
    // group by top-level prefix segment
    use std::collections::BTreeMap;
    let mut by_prefix: BTreeMap<String, usize> = BTreeMap::new();
    for t in &rep.tensors {
        let p = t.name.split('.').next().unwrap_or("").to_string();
        *by_prefix.entry(p).or_default() += 1;
    }
    eprintln!("top-level prefixes: {by_prefix:?}");
    // show first 30 names that look like model params (contain encoder/decoder/vq/to_patch)
    let mut shown = 0;
    for t in &rep.tensors {
        if t.name.contains("encoder") || t.name.contains("to_patch") || t.name.contains("vq")
            || t.name.contains("spatial_rel_pos") || t.name.contains("to_pixels") {
            eprintln!("  {}  {:?}", t.name, t.shape);
            shown += 1;
            if shown >= 30 { break; }
        }
    }
}
