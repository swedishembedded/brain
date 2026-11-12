// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The import walk: each released graph imports against its own config with
//! every tensor bound and every node visited, and refuses another's.

use openseeface::config::{DETECTOR, LANDMARKS_BEST, LANDMARKS_FAST};
use openseeface::import::import;

#[test]
fn every_released_graph_imports_with_full_coverage() {
    let Some(dir) = crate::weights() else { return };
    for (name, cfg) in crate::NETS {
        let t = import(&dir.join(cfg.file), &cfg).unwrap_or_else(|e| panic!("{name}: {e}"));
        let params: usize = t.values().map(|(s, _)| s.iter().product::<usize>()).sum();
        println!("  {name}: {} tensors, {params} parameters", t.len());
        assert!(t.contains_key("stem.conv.weight") && t.contains_key("head.out.conv.bias"));
    }
}

#[test]
fn a_graph_is_refused_under_another_graphs_config() {
    let Some(dir) = crate::weights() else { return };
    for (file_cfg, wrong) in [(LANDMARKS_FAST, LANDMARKS_BEST), (LANDMARKS_BEST, LANDMARKS_FAST), (DETECTOR, LANDMARKS_FAST)] {
        let e = import(&dir.join(file_cfg.file), &wrong).expect_err("a mismatched config must not import");
        println!("  {} as {}: {e}", file_cfg.file, wrong.file);
    }
}
