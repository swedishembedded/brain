// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Codex CSV export reader.
//!
//! Codex publishes every dataset it hosts through one schema, so this reader
//! serves BANC, MANC and the rest without a per-dataset branch. Two files:
//!
//! * `neurons.csv[.gz]` -- one row per neuron, `Root ID` plus annotation;
//! * `connections_princeton.csv[.gz]` -- one row per (pre, post, NEUROPIL),
//!   so the same neuron pair appears once per region its synapses fall in and
//!   must be aggregated into a single edge.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::csv::{split_line, Header};
use crate::{Connectome, Neuron, Nt, NtPrior};

/// Every input row is either kept or rejected with a reason, and this proves
/// it. A connectome that silently drops rows is one nobody can compare against
/// a published figure, which is exactly the comparison an import has to
/// survive.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    pub neuron_rows: usize,
    pub neurons_kept: usize,
    pub neuron_rejects: BTreeMap<String, usize>,
    /// Rows in the connections file, before aggregation.
    pub edge_rows: usize,
    pub edge_rows_kept: usize,
    pub edge_rejects: BTreeMap<String, usize>,
    /// Distinct `(pre, post)` pairs after aggregating across neuropils.
    pub edges: usize,
    pub synapses: u64,
}

impl Coverage {
    /// Both files balance: kept + rejected == rows, on each side.
    ///
    /// Called by `load`, so an unbalanced import cannot be returned at all.
    pub fn check(&self) -> Result<(), String> {
        let nr: usize = self.neuron_rejects.values().sum();
        if self.neurons_kept + nr != self.neuron_rows {
            return Err(format!(
                "neuron rows do not balance: {} kept + {} rejected != {} rows",
                self.neurons_kept, nr, self.neuron_rows
            ));
        }
        let er: usize = self.edge_rejects.values().sum();
        if self.edge_rows_kept + er != self.edge_rows {
            return Err(format!(
                "edge rows do not balance: {} kept + {} rejected != {} rows",
                self.edge_rows_kept, er, self.edge_rows
            ));
        }
        Ok(())
    }

    /// A one-line summary for a log or a test failure.
    pub fn summary(&self) -> String {
        format!(
            "{} neurons ({} rejected), {} edges from {} rows ({} rejected), {} synapses",
            self.neurons_kept,
            self.neuron_rejects.values().sum::<usize>(),
            self.edges,
            self.edge_rows,
            self.edge_rejects.values().sum::<usize>(),
            self.synapses
        )
    }
}

fn reject(map: &mut BTreeMap<String, usize>, why: &str) {
    *map.entry(why.to_string()).or_insert(0) += 1;
}

/// Open a path, transparently decompressing `.gz`.
fn open(path: &Path) -> Result<Box<dyn Read>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if path.extension().is_some_and(|e| e == "gz") {
        Ok(Box::new(flate2::read::GzDecoder::new(f)))
    } else {
        Ok(Box::new(f))
    }
}

/// Load a Codex export from two paths. `.gz` is handled transparently.
pub fn load(dataset: &str, neurons_csv: &Path, connections_csv: &Path) -> Result<Connectome, String> {
    let n = open(neurons_csv)?;
    let c = open(connections_csv)?;
    load_readers(dataset, n, c)
}

/// The same, from any two readers. Separated so tests can drive it from
/// in-memory fixtures with no files and no network.
pub fn load_readers(dataset: &str, neurons: impl Read, connections: impl Read) -> Result<Connectome, String> {
    let mut cov = Coverage::default();
    let (index, rows) = read_neurons(neurons, &mut cov)?;
    let edges = read_edges(connections, &index, &mut cov)?;

    let csc = neuro::Csc::from_edges(rows.len() as u32, &edges)?;
    cov.edges = edges.len();
    cov.check()?;

    Ok(Connectome { dataset: dataset.to_string(), neurons: rows, csc, coverage: cov })
}

fn read_neurons(r: impl Read, cov: &mut Coverage) -> Result<(BTreeMap<u64, u32>, Vec<Neuron>), String> {
    let mut lines = BufReader::new(r).lines();
    let header = match lines.next() {
        Some(Ok(h)) => Header::new(&h),
        _ => return Err("neurons: empty file".to_string()),
    };
    // Looked up by NAME, so a column added upstream moves the rest rather
    // than corrupting them.
    let c_id = header.need("Root ID")?;
    let c_flow = header.find("Flow");
    let c_super = header.find("Super Class");
    let c_class = header.find("Class");
    let c_nerve = header.find("Nerve");
    let c_side = header.find("Soma side");
    let c_type = header.find("Primary Cell Type");
    let c_pred = header.find("Predicted NT type");
    let c_conf = header.find("Predicted NT confidence");
    let c_ver = header.find("Verified NT type");

    let mut index = BTreeMap::new();
    let mut out = Vec::new();
    let mut fields = Vec::new();
    for line in lines {
        let line = line.map_err(|e| format!("neurons: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        cov.neuron_rows += 1;
        split_line(&line, &mut fields);
        if fields.len() != header.len() {
            reject(&mut cov.neuron_rejects, "wrong field count");
            continue;
        }
        let Ok(root_id) = fields[c_id].trim().parse::<u64>() else {
            reject(&mut cov.neuron_rejects, "unparseable Root ID");
            continue;
        };
        if index.contains_key(&root_id) {
            // Two rows for one neuron would make `index_of` ambiguous and
            // silently drop one row's annotation.
            reject(&mut cov.neuron_rejects, "duplicate Root ID");
            continue;
        }
        let get = |c: Option<usize>| c.map(|i| fields[i].trim().to_string()).unwrap_or_default();
        let nt = NtPrior {
            predicted: c_pred.and_then(|i| Nt::parse(&fields[i])),
            confidence: c_conf.and_then(|i| fields[i].trim().parse::<f32>().ok()).unwrap_or(0.0),
            verified: c_ver.and_then(|i| Nt::parse(&fields[i])),
        };
        index.insert(root_id, out.len() as u32);
        out.push(Neuron {
            root_id,
            flow: get(c_flow),
            super_class: get(c_super),
            class: get(c_class),
            nerve: get(c_nerve),
            soma_side: get(c_side),
            cell_type: get(c_type),
            nt,
        });
        cov.neurons_kept += 1;
    }
    Ok((index, out))
}

/// Read the connections file and aggregate it into neuron-pair edges.
///
/// The published file is one row per (pre, post, neuropil), so a pair whose
/// synapses straddle three regions appears three times. Summing them is not a
/// convenience: leaving them separate would put three parallel edges between
/// the same two neurons into the CSC, and the gather would count their
/// weights independently, which is the same total but three times the memory
/// and a degree statistic that no longer matches the published one.
fn read_edges(
    r: impl Read,
    index: &BTreeMap<u64, u32>,
    cov: &mut Coverage,
) -> Result<Vec<(u32, u32, f32)>, String> {
    let mut lines = BufReader::new(r).lines();
    let header = match lines.next() {
        Some(Ok(h)) => Header::new(&h),
        _ => return Err("connections: empty file".to_string()),
    };
    let c_pre = header.need("pre_root_id")?;
    let c_post = header.need("post_root_id")?;
    let c_syn = header.need("syn_count")?;

    let mut pairs: Vec<(u32, u32, u32)> = Vec::new();
    let mut fields = Vec::new();
    for line in lines {
        let line = line.map_err(|e| format!("connections: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        cov.edge_rows += 1;
        split_line(&line, &mut fields);
        if fields.len() != header.len() {
            reject(&mut cov.edge_rejects, "wrong field count");
            continue;
        }
        let (Ok(pre), Ok(post)) = (fields[c_pre].trim().parse::<u64>(), fields[c_post].trim().parse::<u64>()) else {
            reject(&mut cov.edge_rejects, "unparseable root id");
            continue;
        };
        let Ok(syn) = fields[c_syn].trim().parse::<u32>() else {
            reject(&mut cov.edge_rejects, "unparseable syn_count");
            continue;
        };
        // An edge naming a neuron with no annotation row cannot be placed in
        // the graph at all: there is no index for it, and inventing one would
        // put an unannotated neuron into every population query.
        let (Some(&p), Some(&q)) = (index.get(&pre), index.get(&post)) else {
            reject(&mut cov.edge_rejects, "endpoint has no neuron row");
            continue;
        };
        pairs.push((q, p, syn));
        cov.edge_rows_kept += 1;
        cov.synapses += syn as u64;
    }

    // Sort by (post, pre) and merge: deterministic, and it yields the edges
    // already grouped the way the CSC wants them.
    pairs.sort_unstable();
    let mut out: Vec<(u32, u32, f32)> = Vec::with_capacity(pairs.len());
    let mut it = pairs.into_iter();
    if let Some((mut post, mut pre, mut syn)) = it.next() {
        for (q, p, s) in it {
            if q == post && p == pre {
                syn += s;
            } else {
                out.push((pre, post, syn as f32));
                (post, pre, syn) = (q, p, s);
            }
        }
        out.push((pre, post, syn as f32));
    }
    Ok(out)
}
