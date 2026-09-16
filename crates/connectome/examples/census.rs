// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! What is actually in an export, measured through the production loader.
//!
//! Run it on a new or re-converted dataset before writing anything about it
//! down. Every number a roadmap or a gate quotes for a connectome should come
//! from here rather than from the portal's own summary page: the portal counts
//! rows, this counts what `crates/connectome` kept, and the difference between
//! those two is exactly the thing an import gate exists to pin.
//!
//! ```text
//! BRAIN_CONNECTOME_DIR=... cargo run --release -p brain-connectome --example census -- banc manc
//! ```
fn main() {
    let root = std::path::PathBuf::from(std::env::var("BRAIN_CONNECTOME_DIR").unwrap_or_else(|_| {
        eprintln!("$BRAIN_CONNECTOME_DIR is not set");
        std::process::exit(2)
    }));
    let names: Vec<String> = std::env::args().skip(1).collect();
    let names = if names.is_empty() { vec!["manc".to_string()] } else { names };

    for name in names {
        let (neurons, edges) = match connectome::find(&root, &name) {
            Ok(p) => p,
            Err(e) => {
                println!("{name}: {e}");
                continue;
            }
        };
        let c = match connectome::load(&name, &neurons, &edges) {
            Ok(c) => c,
            Err(e) => {
                println!("{name}: {e}");
                continue;
            }
        };
        let pop = |p: &str| c.population(|n| n.super_class == p).len();
        let sizes = c.sizes();
        let sized = sizes.iter().filter(|s| **s > 0.0).count();
        let (mean, median, p99, max) = c.in_degree_stats();
        println!("{name}");
        println!("  {}", c.coverage.summary());
        println!("  neurons {}  edges {}  synapses {}", c.coverage.neurons_kept, c.coverage.edges, c.coverage.synapses);
        println!(
            "  motor {}  descending {}  ascending {}  sensory {}",
            pop("motor"),
            pop("descending"),
            pop("ascending"),
            pop("sensory")
        );
        println!("  sized {sized} of {}", c.neurons.len());
        println!("  in-degree mean {mean:.1} median {median} p99 {p99} max {max}");
        println!("  CSR {:.1} MB", c.csc.nnz() as f64 * 8.0 / 1e6);
        let (v, p, u) = c.sign_census();
        println!("  transmitter: {v} verified, {p} predicted, {u} unknown");
    }
}
