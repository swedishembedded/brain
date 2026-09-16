// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Joining a brain to a nerve cord at the neurons that cross between them.
//!
//! ## Why there are two datasets and not one
//!
//! BANC reconstructs a whole central nervous system, brain and cord together,
//! which sounds like it removes the need for this file. Measured, its cord is
//! four to five times more sparsely reconstructed than MANC's on MANC's own
//! territory - in-degree onto motor neurons 110 against 224 - and it is the
//! cord that has to drive the legs. So the cord stays MANC, the brain is
//! BANC's, and the two have to be made into one nervous system.
//!
//! ## What a crossing neuron IS
//!
//! Not a connection. A descending neuron is ONE cell with its soma and
//! dendrites in the brain and its axon terminals in the cord, and the two
//! datasets have reconstructed different halves of it. So the join
//! IDENTIFIES the two copies rather than wiring one to the other: the merged
//! cell integrates what the brain puts into it and releases what the cord
//! reads out, and it fires once. Modelling it as a synapse between two
//! neurons would insert a spike's worth of delay and a threshold that no
//! axon has, and would double the cell's count in every census.
//!
//! The identity is PUBLISHED, not inferred. BANC's own metadata carries a
//! `manc_match` column giving the MANC root id of the same cell, for 1,261 of
//! its 1,316 descending neurons and 1,767 of its 1,849 ascending ones.
//! Matching on cell-type NAME instead - the obvious alternative - is a
//! heuristic that silently mispairs the types whose names differ between
//! datasets, and silently drops the ones annotated in only one.
//!
//! ## What stops the cord being counted twice
//!
//! BANC's nerve cord is dropped, and the crossing cells are what keep their
//! edges anyway. A BANC descending neuron's inputs are brain neurons, which
//! survive; its outputs are BANC cord neurons, which do not, so those edges
//! fall away with their endpoint and MANC supplies the cord side of the same
//! cell. An ascending neuron is the mirror image: its soma is in the cord, so
//! MANC has its inputs, and BANC has the brain neurons it terminates on.
//! Neither half is represented twice and neither is missing.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::csv::{split_line, Header};
use crate::{Connectome, Neuron};

/// One published cross-dataset identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Crossing {
    /// The neuron's id in the brain dataset.
    pub brain_root_id: u64,
    /// The same cell's id in the cord dataset.
    pub cord_root_id: u64,
    /// `descending`, `ascending`, ... as the brain dataset classes it. This
    /// is what gives an unmerged crossing its DIRECTION.
    pub flow: String,
    /// The brain dataset's independent morphology match, where it has one.
    /// Used only to break a many-to-one tie; see [`Policy`].
    pub morphology_match: Option<u64>,
}

impl Crossing {
    /// Whether this cell carries signal from the brain down, as opposed to
    /// from the cord up. Read off the flow the brain dataset annotates.
    pub fn is_descending(&self) -> bool {
        self.flow.contains("descending")
    }
}

/// How to treat a crossing the data does not pin down.
///
/// `manc_match` is many-to-one on 391 of MANC's cells - up to eight BANC
/// neurons naming the same one - because a curator recorded one exemplar for
/// a cell TYPE whose members the other dataset did not resolve separately.
/// Merging such a group would fuse distinct brain neurons into one cell, which
/// is a stronger claim than the annotation makes and is wrong in the 297
/// groups whose members share a soma side. So exactly one member of each group
/// is merged - the one whose independent morphology match agrees, else the
/// lowest id, so the choice is deterministic - and the others are WIRED to the
/// same cord cell instead, which asserts correspondence without asserting
/// identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    /// Synapse count for the axonal link that stands in for an ambiguous
    /// crossing. `0` drops those pathways entirely, which is the control for
    /// whether they matter.
    ///
    /// The default is the 90th percentile of a REAL descending connection in
    /// this cord, measured over all 465,033 descending-neuron output pairs in
    /// MANC: median 2 synapses, p75 6, p90 16, max 642. Strong enough to
    /// survive `fly::Wiring`'s synapse floor and to carry a spike the way an
    /// axon does, and drawn from the same distribution as every other edge in
    /// the graph rather than invented.
    pub ambiguous_axon_synapses: u32,
}

impl Default for Policy {
    fn default() -> Self {
        Policy { ambiguous_axon_synapses: 16 }
    }
}

/// What a [`join`] actually joined, so a caller can assert on it.
///
/// A silent join is the failure this is here to prevent: a bridge whose ids
/// match nothing produces a perfectly valid graph in which the brain and the
/// cord are two disconnected components, and every downstream measurement
/// still runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Joined {
    /// Crossings whose BOTH ends were found and merged into one cell.
    pub merged: usize,
    /// Crossings the data does not pin down, wired instead of merged. See
    /// [`Policy`].
    pub linked: usize,
    /// Rows whose brain id is not in the brain graph.
    pub brain_missing: usize,
    /// Rows whose cord id is not in the cord graph.
    pub cord_missing: usize,
    /// Neurons in the joined graph.
    pub neurons: usize,
    /// Edges that survived the join.
    pub edges: usize,
    /// Edges dropped because an endpoint was not kept.
    pub edges_dropped: usize,
}

impl Joined {
    pub fn summary(&self) -> String {
        format!(
            "{} neurons, {} edges ({} dropped), {} crossing cells merged, {} wired \
             ({} brain ids and {} cord ids unmatched)",
            self.neurons,
            self.edges,
            self.edges_dropped,
            self.merged,
            self.linked,
            self.brain_missing,
            self.cord_missing
        )
    }
}

/// Read a bridge file: `banc_root_id,manc_root_id,flow,...`.
///
/// Written by `tools/convert/banc_codex.py` out of BANC's own `manc_match`
/// column. Deliberately NOT a Codex file and deliberately not folded into the
/// neuron schema: it is one dataset's statement about another, and the two
/// exports are versioned separately.
pub fn read_bridge(path: impl AsRef<Path>) -> Result<Vec<Crossing>, String> {
    let path = path.as_ref();
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(flate2::read::GzDecoder::new(f))
    } else {
        Box::new(f)
    };
    read_bridge_reader(reader).map_err(|e| format!("{}: {e}", path.display()))
}

/// The same, from any reader, so a test needs no file.
pub fn read_bridge_reader(r: impl Read) -> Result<Vec<Crossing>, String> {
    let mut lines = BufReader::new(r).lines();
    let head = lines.next().ok_or("the bridge file is empty")?.map_err(|e| e.to_string())?;
    let header = Header::new(&head);
    let (col_brain, col_cord) = (header.need("banc_root_id")?, header.need("manc_root_id")?);
    let col_flow = header.find("flow");
    let col_morph = header.find("manc_nblast_match");
    let mut out = Vec::new();
    let mut fields = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        split_line(&line, &mut fields);
        let at = |i: usize| fields.get(i).map(|s| s.trim()).unwrap_or("");
        let (brain, cord) = (at(col_brain).to_string(), at(col_cord).to_string());
        // A row that names no cell on one side is a malformed row, not an
        // unmatched one, and the two have to stay distinguishable: an
        // unmatched neuron is a fact about the datasets, a malformed row is a
        // fact about the file.
        let (brain, cord) = match (brain.parse::<u64>(), cord.parse::<u64>()) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return Err(format!("row {}: ids must both be integers, got {brain:?} and {cord:?}", i + 2)),
        };
        out.push(Crossing {
            brain_root_id: brain,
            cord_root_id: cord,
            flow: col_flow.map(|i| at(i).to_string()).unwrap_or_default(),
            morphology_match: col_morph.and_then(|i| at(i).parse::<u64>().ok()),
        });
    }
    Ok(out)
}

/// One nervous system out of a brain and a cord.
///
/// `keep_brain` selects which of the brain dataset's neurons survive - in
/// practice its brain REGIONS, since its own cord is the half being replaced.
/// A neuron named by the bridge is kept whether or not the predicate takes
/// it, because that is what a crossing cell is for; see this module's header
/// for why that does not double-count the cord.
///
/// Index order in the result is every kept brain neuron, then every cord
/// neuron that was not merged into one. A merged cell keeps the BRAIN's
/// annotation row: it carries the soma, the transmitter and the cell type,
/// and the cord's copy of a descending neuron annotates the same cell the
/// same way.
pub fn join(
    brain: &Connectome,
    cord: &Connectome,
    bridge: &[Crossing],
    policy: Policy,
    keep_brain: impl Fn(&Neuron) -> bool,
) -> Result<(Connectome, Joined), String> {
    let mut report = Joined::default();

    let brain_by_id: HashMap<u64, u32> =
        brain.neurons.iter().enumerate().map(|(i, n)| (n.root_id, i as u32)).collect();
    let cord_by_id: HashMap<u64, u32> = cord.neurons.iter().enumerate().map(|(i, n)| (n.root_id, i as u32)).collect();

    // Resolve both ends first, so a group can be seen whole before any of it
    // is acted on. A crossing whose ids resolve is `(brain index, cord index,
    // crossing)`; one that does not is counted and dropped.
    let mut resolved: Vec<(u32, u32, &Crossing)> = Vec::new();
    for x in bridge {
        match (brain_by_id.get(&x.brain_root_id), cord_by_id.get(&x.cord_root_id)) {
            (Some(&b), Some(&c)) => resolved.push((b, c, x)),
            (None, _) => report.brain_missing += 1,
            (_, None) => report.cord_missing += 1,
        }
    }

    // Group by the CORD cell, because that is the side the ambiguity is on:
    // the brain side is one-to-one in this data and a brain cell naming two
    // cord cells would be a different defect, caught below.
    let mut groups: HashMap<u32, Vec<(u32, &Crossing)>> = HashMap::new();
    let mut seen_brain: HashMap<u32, u32> = HashMap::new();
    for (b, c, x) in resolved {
        if let Some(&prev) = seen_brain.get(&b) {
            if prev != c {
                return Err(format!(
                    "brain neuron {} is matched to two different cord neurons ({} and {})",
                    x.brain_root_id, cord.neurons[prev as usize].root_id, x.cord_root_id
                ));
            }
        }
        seen_brain.insert(b, c);
        groups.entry(c).or_default().push((b, x));
    }

    // cord index -> brain index, for the cells that MERGE; and the axonal
    // links that stand in for the rest.
    let mut cord_is_brain: HashMap<u32, u32> = HashMap::new();
    let mut links: Vec<(u32, u32, f32, bool)> = Vec::new();
    for (c, mut members) in groups {
        // Deterministic: the morphology match decides, then the lower id.
        // Anything order-dependent here would make the joined graph depend on
        // the order of rows in a CSV.
        members.sort_by_key(|(b, x)| {
            (x.morphology_match != Some(cord.neurons[c as usize].root_id), brain.neurons[*b as usize].root_id)
        });
        let (keep, rest) = members.split_first().expect("a group has at least one member");
        cord_is_brain.insert(c, keep.0);
        report.merged += 1;
        for (b, x) in rest {
            report.linked += 1;
            if policy.ambiguous_axon_synapses > 0 {
                links.push((*b, c, policy.ambiguous_axon_synapses as f32, x.is_descending()));
            }
        }
    }
    let bridged: std::collections::HashSet<u32> = cord_is_brain.values().copied().collect();

    let mut neurons: Vec<Neuron> = Vec::new();
    let mut brain_remap = vec![u32::MAX; brain.neurons.len()];
    for (i, n) in brain.neurons.iter().enumerate() {
        if keep_brain(n) || bridged.contains(&(i as u32)) {
            brain_remap[i] = neurons.len() as u32;
            neurons.push(n.clone());
        }
    }
    let mut cord_remap = vec![u32::MAX; cord.neurons.len()];
    for (i, n) in cord.neurons.iter().enumerate() {
        match cord_is_brain.get(&(i as u32)) {
            // Merged: this cell already has a row, the brain's.
            Some(&b) if brain_remap[b as usize] != u32::MAX => cord_remap[i] = brain_remap[b as usize],
            _ => {
                cord_remap[i] = neurons.len() as u32;
                neurons.push(n.clone());
            }
        }
    }

    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    let mut synapses = 0u64;
    let mut take = |src: &Connectome, remap: &[u32], edges: &mut Vec<(u32, u32, f32)>, report: &mut Joined| {
        for post in 0..src.csc.n as usize {
            let to = remap[post];
            let (a, b) = (src.csc.indptr[post] as usize, src.csc.indptr[post + 1] as usize);
            for k in a..b {
                let from = remap[src.csc.pre[k] as usize];
                if to == u32::MAX || from == u32::MAX {
                    report.edges_dropped += 1;
                    continue;
                }
                edges.push((from, to, src.csc.w[k]));
                synapses += src.csc.w[k] as u64;
            }
        }
    };
    take(brain, &brain_remap, &mut edges, &mut report);
    take(cord, &cord_remap, &mut edges, &mut report);

    // The axonal stand-ins, in the direction the crossing runs: a descending
    // cell carries the brain's cell onto the cord's, an ascending one the
    // other way.
    for (b, c, w, descending) in links {
        let (from, to) = (brain_remap[b as usize], cord_remap[c as usize]);
        if from == u32::MAX || to == u32::MAX {
            report.edges_dropped += 1;
            continue;
        }
        let (from, to) = if descending { (from, to) } else { (to, from) };
        edges.push((from, to, w));
        synapses += w as u64;
    }

    report.neurons = neurons.len();
    report.edges = edges.len();

    let csc = neuro::Csc::from_edges(neurons.len() as u32, &edges)
        .map_err(|e| format!("joining {} to {}: {e}", brain.dataset, cord.dataset))?;
    let coverage = crate::codex::Coverage {
        neuron_rows: neurons.len(),
        neurons_kept: neurons.len(),
        neuron_rejects: Default::default(),
        edge_rows: edges.len(),
        edge_rows_kept: edges.len(),
        edge_rejects: Default::default(),
        edges: edges.len(),
        synapses,
    };
    let dataset = format!("{}+{}", brain.dataset, cord.dataset);
    Ok((Connectome { dataset, neurons, csc, coverage }, report))
}

/// The brain half of a BANC-shaped export: everything outside its nerve cord.
///
/// Keyed on the export's own `region` column, whose three values are
/// `optic_lobe`, `central_brain` and `ventral_nerve_cord`. A neuron with no
/// region is kept out: 368 of 188,313 rows have none, and a cell nobody could
/// place is not evidence for a brain circuit.
pub fn is_brain(n: &Neuron) -> bool {
    n.region == "optic_lobe" || n.region == "central_brain"
}
