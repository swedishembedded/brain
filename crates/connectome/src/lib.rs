// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Connectome import: a published wiring diagram, read into something the
//! engine can run.
//!
//! One reader serves BANC, MANC and every other Codex dataset, because Codex
//! exports one schema for all of them. What comes out is a [`neuro::Csc`] plus
//! a per-neuron annotation row, and a [`Coverage`] report that accounts for
//! every line of both input files: kept, or rejected with a typed reason.
//!
//! Swedish Embedded AB implements connectome and large-graph import pipelines
//! for its clients, including the coverage discipline that makes a silently
//! dropped row impossible. If your team needs a scientific dataset turned into
//! something a runtime can execute, you can procure our services by sending an
//! email to info@swedishembedded.com.

pub mod bridge;
pub mod codex;
pub mod csv;
pub mod mushroom_body;
pub mod retinotopy;

use std::path::{Path, PathBuf};

pub use bridge::{join, read_bridge, Crossing, Joined};
pub use codex::{load, load_readers, Coverage};
pub use mushroom_body::{Compartment, MushroomBody};

/// A neurotransmitter, as published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nt {
    Acetylcholine,
    Glutamate,
    Gaba,
    Dopamine,
    Serotonin,
    Octopamine,
    Histamine,
    Tyramine,
}

impl Nt {
    /// One transmitter name, in any spelling either dataset uses.
    ///
    /// `HA` and `TYR` are here because BANC uses them and their absence was
    /// not a small gap: `HA` is how 928 of its photoreceptors spell histamine,
    /// and a name this function does not recognise is not treated as unknown,
    /// it falls through to the machine prediction instead.
    pub fn parse(s: &str) -> Option<Nt> {
        Some(match s.trim().to_ascii_uppercase().as_str() {
            "ACH" | "ACETYLCHOLINE" => Nt::Acetylcholine,
            "GLUT" | "GLUTAMATE" => Nt::Glutamate,
            "GABA" => Nt::Gaba,
            "DA" | "DOPAMINE" => Nt::Dopamine,
            "SER" | "SEROTONIN" | "5HT" => Nt::Serotonin,
            "OCT" | "OCTOPAMINE" => Nt::Octopamine,
            "HIST" | "HISTAMINE" | "HA" => Nt::Histamine,
            "TYR" | "TYRAMINE" => Nt::Tyramine,
            _ => return None,
        })
    }

    /// Whether this transmitter gates a fast ionotropic synapse.
    ///
    /// The distinction decides which member of a co-transmitting cell's list
    /// sets the sign of its synapses in a model whose only currency is fast
    /// current. Dopamine, serotonin, octopamine and tyramine act through
    /// G-protein-coupled receptors over hundreds of milliseconds; they are
    /// carried because they identify a cell and because a neuromodulatory
    /// pathway needs them, not because they push a membrane this tick.
    pub fn is_fast(self) -> bool {
        matches!(self, Nt::Acetylcholine | Nt::Glutamate | Nt::Gaba | Nt::Histamine)
    }

    /// A verified transmitter field, which may name several.
    ///
    /// BANC's verified column is a LIST: `GABA,NITRIC_OXIDE` on 2,878 cells,
    /// `HISTAMINE,ACETYLCHOLINE` on 914, `ACETYLCHOLINE,NITRIC_OXIDE,DOPAMINE`
    /// on 666. Parsing it as one name fails on all of them, and failure here
    /// is not "unknown" - [`NtPrior::best`] then falls back to the machine
    /// PREDICTION, so a human annotation is discarded in favour of a guess it
    /// sometimes contradicts. Across BANC that is 5,872 neurons, 315 of which
    /// end up with the opposite sign, and essentially every photoreceptor in
    /// the dataset. MANC has none, which is why running only the cord never
    /// showed it.
    ///
    /// Resolution, in order:
    ///
    /// * names that are not transmitters at all are dropped. Nitric oxide is
    ///   a gas that diffuses through membranes and has no synapse to sign;
    ///   listing it says something true about the cell and nothing about its
    ///   synaptic sign.
    /// * if one FAST transmitter remains, it sets the sign.
    /// * if several do, the cell genuinely co-releases two fast transmitters
    ///   and the pair is returned so the caller can see the ambiguity rather
    ///   than inherit a silent choice. Histamine precedes acetylcholine
    ///   because that is the photoreceptor case this arises in, and the
    ///   histamine-gated chloride channel is the fast synapse there.
    /// * if none do, the cell is modulatory and the first name is returned.
    pub fn parse_verified(s: &str) -> (Option<Nt>, bool) {
        let named: Vec<Nt> = s.split(',').filter_map(Nt::parse).collect();
        let mut fast: Vec<Nt> = named.iter().copied().filter(|n| n.is_fast()).collect();
        // Precedence among co-released fast transmitters, documented above.
        fast.sort_by_key(|n| match n {
            Nt::Histamine => 0,
            Nt::Gaba => 1,
            Nt::Glutamate => 2,
            _ => 3,
        });
        match (fast.first(), named.first()) {
            (Some(&f), _) => (Some(f), fast.len() > 1),
            (None, Some(&m)) => (Some(m), false),
            (None, None) => (None, false),
        }
    }

    /// The sign this transmitter is USUALLY given in a *Drosophila* model.
    ///
    /// Read the word "usually" as load-bearing. Sign is a property of the
    /// POSTsynaptic receptor, which electron microscopy cannot see at all, so
    /// this is a convention and not a measurement:
    ///
    /// * acetylcholine excites through nicotinic receptors;
    /// * GABA inhibits;
    /// * glutamate mostly INHIBITS in the fly, through the glutamate-gated
    ///   chloride channel GluCl - the opposite of the vertebrate default, and
    ///   the single easiest sign to get wrong here;
    /// * histamine inhibits (it is the photoreceptor transmitter);
    /// * dopamine, serotonin and octopamine are modulatory rather than
    ///   fast-acting at all. They are given +1 because that is what the
    ///   published whole-brain leaky integrate-and-fire models do, not because
    ///   a modulatory synapse is an excitatory one.
    ///
    /// Nothing downstream should treat this as fixed: it is the centre of a
    /// prior whose width is [`NtPrior::strength`], and the sign is fitted.
    pub fn conventional_sign(self) -> f32 {
        match self {
            Nt::Acetylcholine | Nt::Dopamine | Nt::Serotonin | Nt::Octopamine | Nt::Tyramine => 1.0,
            Nt::Glutamate | Nt::Gaba | Nt::Histamine => -1.0,
        }
    }
}

/// What is known about one neuron's transmitter, and how well.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NtPrior {
    /// The machine-learning prediction, where there is one.
    pub predicted: Option<Nt>,
    /// The prediction's own confidence, `0.0` when there is no prediction.
    pub confidence: f32,
    /// A human-verified transmitter. BANC has one for about 41% of its
    /// neurons; Janelia's MANC export has none at all.
    pub verified: Option<Nt>,
    /// The verification named more than one FAST transmitter, so the sign
    /// here is a choice among them rather than a reading of the data. See
    /// [`Nt::parse_verified`].
    pub co_released: bool,
}

impl NtPrior {
    /// The transmitter to believe: verified if there is one, else predicted.
    pub fn best(&self) -> Option<Nt> {
        self.verified.or(self.predicted)
    }

    /// The centre of the sign prior, `0.0` when nothing is known.
    pub fn sign(&self) -> f32 {
        self.best().map_or(0.0, Nt::conventional_sign)
    }

    /// How much to believe [`Self::sign`], in `[0, 1]`.
    ///
    /// A verified transmitter is `1.0`; a prediction is worth its own
    /// confidence; nothing known is `0.0`. This is the number that keeps a
    /// fitted sign honest, and it matters more than it looks: across both
    /// datasets only about half of all PREDICTIONS clear 0.8, so a model that
    /// hard-codes sign from the prediction is asserting something the data
    /// does not support for half the animal.
    pub fn strength(&self) -> f32 {
        if self.verified.is_some() {
            // A cell verified to co-release two fast transmitters of opposite
            // sign is not fully known, whatever the precedence rule picked.
            if self.co_released {
                0.5
            } else {
                1.0
            }
        } else if self.predicted.is_some() {
            self.confidence.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// One neuron's published annotation. Strings are kept verbatim rather than
/// parsed into enums: the vocabularies differ per dataset and per release, and
/// an unrecognised value must not become a silent `Other`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Neuron {
    /// The dataset's own identifier. Too large for `u32`, and not dense, which
    /// is why the graph is indexed positionally instead.
    pub root_id: u64,
    /// `intrinsic` / `afferent` / `efferent`.
    pub flow: String,
    /// `motor`, `descending`, `sensory`, `optic_lobe_intrinsic`, ...
    pub super_class: String,
    /// `leg_motor_neuron`, `wing_motor_neuron`, ... In MANC this is the short
    /// form (`fl`, `ml`, `hl`, `wm`).
    pub class: String,
    /// The finer annotation, and the one that carries the MUSCLE a motor
    /// neuron innervates: `MN-LegNpT2-Ti_flexor` names both the thoracic
    /// segment and the muscle. Worth knowing that this exists, because the
    /// obvious alternative is Janelia's per-neuron feather, and going that way
    /// would have put an Arrow dependency in the import path for a field the
    /// CSV already carries.
    pub sub_class: String,
    /// The nerve a motor or sensory neuron runs in, e.g.
    /// `left_prothoracic_leg_nerve`. This is the handle a body attaches to.
    pub nerve: String,
    pub soma_side: String,
    pub cell_type: String,
    /// The neuropil this neuron has most of its connectivity in, e.g.
    /// `LEGNP_T1` for the front-leg neuropil. This is the handle a
    /// per-neuropil subnetwork is selected by.
    pub region: String,
    /// Membrane surface area in square nanometres, as reconstructed. `0.0`
    /// where the export does not carry it.
    ///
    /// Not a curiosity. A leaky integrate-and-fire membrane obeys
    /// `C dV/dt = -g_L (V - V_rest) + I`, and BOTH `C` and `g_L` scale with
    /// membrane area - so the time constant `C/g_L` does not depend on size,
    /// but the voltage a given synaptic current produces goes as `1/g_L`, and
    /// therefore as one over the area. A large neuron is genuinely less
    /// excitable per unit of input than a small one, and a model that gives
    /// every cell the same threshold has quietly made the largest cells in the
    /// cord hundreds of times more excitable than the smallest.
    pub surface_area_nm2: f64,
    /// Reconstructed volume in cubic nanometres. `0.0` where absent.
    ///
    /// Carried because it is what is actually POPULATED: the MANC Codex export
    /// has a surface-area column and leaves it empty on every row, while the
    /// volume column is filled. For a neurite the membrane area and the
    /// enclosed volume differ by a factor of `2/r`, so at roughly constant
    /// neurite radius the two are proportional and volume serves as the size
    /// measure. See [`Connectome::size`].
    pub volume_nm3: f64,
    pub nt: NtPrior,
}

/// Find a dataset's two files under `root`, wherever it was unpacked.
///
/// A Codex export arrives as a directory of two gzipped CSVs, and what that
/// directory is CALLED depends on how it was downloaded: the bare dataset
/// name, the name with the portal's `-codex` suffix, or the files sitting at
/// the root with no directory at all. None of those is more correct than the
/// others and a caller should not have to know which one they have - the
/// failure otherwise is a path error naming one spelling, from which it is not
/// obvious that two others would have worked.
///
/// Returns `(neurons, connections)`. The error names every layout that was
/// tried, because "no such file" without the list is the least actionable
/// message a data dependency can produce.
pub fn find(root: impl AsRef<Path>, dataset: &str) -> Result<(PathBuf, PathBuf), String> {
    let root = root.as_ref();
    let mut tried = Vec::new();
    for dir in [root.to_path_buf(), root.join(dataset), root.join(format!("{dataset}-codex"))] {
        let neurons = dir.join("neurons.csv.gz");
        let edges = dir.join("connections_princeton.csv.gz");
        if neurons.is_file() && edges.is_file() {
            return Ok((neurons, edges));
        }
        tried.push(dir.display().to_string());
    }
    Err(format!(
        "no {dataset} export found. Looked for neurons.csv.gz and \
         connections_princeton.csv.gz in: {}",
        tried.join(", ")
    ))
}

/// A loaded connectome: the graph, the annotations, and the proof that every
/// input row was accounted for.
#[derive(Clone, Debug)]
pub struct Connectome {
    /// Which dataset this came from, for error messages and provenance.
    pub dataset: String,
    /// Annotations, positionally parallel to the graph's neuron indices.
    pub neurons: Vec<Neuron>,
    /// The graph. Indices are `0..neurons.len()`.
    pub csc: neuro::Csc,
    pub coverage: Coverage,
}

impl Connectome {
    /// Position of a published id, or `None` if it is not in this graph.
    pub fn index_of(&self, root_id: u64) -> Option<u32> {
        self.neurons.iter().position(|n| n.root_id == root_id).map(|i| i as u32)
    }

    /// Neurons whose every synapse is multiplied by zero, and the synapses
    /// they carry.
    ///
    /// Sign comes from the presynaptic transmitter, and a neuron whose
    /// transmitter nobody has predicted gets a sign of `0.0`, which means its
    /// output is not weak: it is ABSENT. The cell stays in the graph, keeps
    /// its in-degree, counts in every population, and contributes nothing.
    /// That is a deletion, and a silent one, so it is worth a number.
    ///
    /// The aggregate is reassuring and the distribution need not be: in BANC
    /// 17.7% of neurons have no transmitter and they carry 1.4% of all
    /// synapses, but the loss sits wherever the annotation was hardest rather
    /// than spread evenly, so a single well-connected cell can matter far more
    /// than its share of the graph.
    ///
    /// Counted AFTER [`Nt::parse_verified`] resolves the verified column, and
    /// that ordering is the point. Reading that column as a single name left
    /// 5,872 BANC neurons falling back to a machine prediction, which is a
    /// worse failure than silence because it is invisible here: the cell has
    /// a transmitter, so it is not counted, and the transmitter is a guess.
    pub fn silenced(&self) -> (usize, u64) {
        // One pass over the edge list, not one per silent neuron. BANC has
        // 33,272 cells with no predicted transmitter and 13.6M edges, and
        // asking the question the other way round is 4.5e11 comparisons: it
        // ran for ten minutes on one core, allocating nothing, before anything
        // printed.
        let mute: Vec<bool> = self.neurons.iter().map(|n| n.nt.best().is_none()).collect();
        let mut synapses = 0u64;
        for (pre, w) in self.csc.pre.iter().zip(&self.csc.w) {
            if mute.get(*pre as usize).copied().unwrap_or(false) {
                synapses += *w as u64;
            }
        }
        (mute.iter().filter(|m| **m).count(), synapses)
    }

    /// Assert a transmitter for cells this dataset's predictor abstained on.
    ///
    /// Returns how many neurons it filled in. Gaps ONLY: a neuron that already
    /// has a verified or predicted transmitter is never overwritten, so this
    /// can add knowledge and cannot contradict the data.
    ///
    /// This exists because "no prediction" and "no transmitter" are different
    /// statements that the sign convention collapses into the same number, and
    /// for a handful of cells the literature is simply more certain than the
    /// classifier. It is deliberately NOT a table applied at import: which
    /// cells those are is a scientific judgement that belongs at the call
    /// site, in the open, next to the reason - not in a curated list that
    /// silently goes stale with the next release.
    pub fn assume_transmitter(&mut self, cell_type: &str, nt: Nt) -> usize {
        let mut filled = 0;
        for n in &mut self.neurons {
            if n.cell_type == cell_type && n.nt.best().is_none() {
                n.nt.predicted = Some(nt);
                n.nt.confidence = 1.0;
                filled += 1;
            }
        }
        filled
    }

    /// Every neuron whose annotation satisfies `pred`, as graph indices.
    ///
    /// This is how a body finds its motor neurons: `population(|n|
    /// n.class == "leg_motor_neuron" && n.nerve.starts_with("left_pro"))`.
    /// Deliberately a predicate rather than a fixed set of named groups - the
    /// vocabularies differ per dataset, and hard-coding them here would make
    /// this crate wrong for the next release.
    pub fn population(&self, pred: impl Fn(&Neuron) -> bool) -> Vec<u32> {
        self.neurons.iter().enumerate().filter(|(_, n)| pred(n)).map(|(i, _)| i as u32).collect()
    }

    /// The graph with each edge SIGNED by its presynaptic neuron's
    /// transmitter and scaled.
    ///
    /// [`Self::csc`] carries raw synapse counts, which are unsigned: a count
    /// cannot be negative. Sign belongs to the presynaptic neuron, not the
    /// edge, so it is applied here rather than at import - and applying it is
    /// not optional, because a network in which every synapse excites has no
    /// inhibition at all and saturates immediately.
    ///
    /// `scale` converts counts to membrane current. It matters more than it
    /// looks: in-degree averages in the hundreds and synapse counts run to
    /// tens, so an unscaled network drives every neuron thousands of
    /// millivolts past threshold. Nothing here fits it; it is the caller's
    /// dial.
    ///
    /// A neuron whose transmitter is unknown contributes ZERO rather than a
    /// guessed sign. That is deliberate: an unknown sign is not a coin flip,
    /// it is an absence of evidence, and 0.0 is what
    /// [`NtPrior::sign`] already returns for it.
    pub fn signed_csc(&self, scale: f32) -> neuro::Csc {
        let mut csc = self.csc.clone();
        for (k, w) in csc.w.iter_mut().enumerate() {
            let pre = self.csc.pre[k] as usize;
            let sign = self.neurons.get(pre).map_or(0.0, |n| n.nt.sign());
            *w *= sign * scale;
        }
        csc
    }

    /// A connectome restricted to the neurons `keep` selects, with the edges
    /// between them.
    ///
    /// Published circuit models of this nerve cord work on a SUBNETWORK - one
    /// leg neuropil and the neurons that reach it - rather than on the whole
    /// cord, and that is not only a compute saving. A large recurrent network
    /// carries many feedback loops of many different lengths at once, and a
    /// network oscillator's period is set by its loop delay; running them all
    /// together smears the rhythm across every period the graph contains.
    ///
    /// Edges are kept only where BOTH endpoints survive. An edge from a
    /// dropped neuron is not evidence of anything about the remaining ones, and
    /// carrying its weight in as a constant would be inventing a tonic drive
    /// the data does not describe.
    ///
    /// The coverage report on the result describes the SUBSET, and says so: it
    /// is derived from a balanced import rather than measured against an input
    /// file, so its row counts equal its kept counts by construction and a
    /// balance check on it proves nothing new.
    pub fn subgraph(&self, keep: impl Fn(&Neuron) -> bool) -> Connectome {
        let mut remap = vec![u32::MAX; self.neurons.len()];
        let mut neurons = Vec::new();
        for (i, n) in self.neurons.iter().enumerate() {
            if keep(n) {
                remap[i] = neurons.len() as u32;
                neurons.push(n.clone());
            }
        }
        let mut edges: Vec<(u32, u32, f32)> = Vec::new();
        let mut synapses = 0u64;
        for post in 0..self.csc.n as usize {
            let to = remap[post];
            if to == u32::MAX {
                continue;
            }
            let (a, b) = (self.csc.indptr[post] as usize, self.csc.indptr[post + 1] as usize);
            for k in a..b {
                let from = remap[self.csc.pre[k] as usize];
                if from != u32::MAX {
                    edges.push((from, to, self.csc.w[k]));
                    synapses += self.csc.w[k] as u64;
                }
            }
        }
        let n = neurons.len() as u32;
        // Cannot fail: every endpoint was produced by `remap` and is therefore
        // below `n` by construction.
        let csc = neuro::Csc::from_edges(n, &edges).expect("remapped endpoints are in range");
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
        Connectome { dataset: format!("{} (subgraph)", self.dataset), neurons, csc, coverage }
    }

    /// The graph as a network: signed, scaled, optionally size-normalised, and
    /// optionally pruned of the weakest connections.
    ///
    /// `min_synapses` drops every pair connected by fewer than that many
    /// synapses. This is not tidying. A reconstruction assigns a great many
    /// one- and two-synapse pairs that are at the edge of what the imaging can
    /// resolve, and they are numerous enough to dominate a neuron's input
    /// count while carrying almost none of its drive - so including them
    /// spends the network's whole dynamic range on connections nobody would
    /// defend. Published connectome circuit models routinely impose a floor of
    /// around five for exactly this reason. `0` or `1` keeps everything, which
    /// is the control.
    pub fn network(&self, scale: f32, size_limit: Option<f32>, min_synapses: u32) -> neuro::Csc {
        self.network_keeping(scale, size_limit, min_synapses, &std::collections::HashSet::new())
    }

    /// The same, with `keep` exempt from the synapse floor.
    ///
    /// The floor exists because reconstruction assigns a great many one- and
    /// two-synapse pairs at the resolution limit, numerous enough to dominate
    /// a neuron's input count while carrying almost none of its drive. That
    /// argument is sound for ordinary neuropil and it is FALSE for a pathway
    /// that is sparse by design. In BANC, 9,483 of the 17,789 Kenyon-cell
    /// synapses onto mushroom-body output neurons are single-synapse pairs,
    /// and a floor of 5 deletes 79% of that pathway's synapse mass while
    /// keeping 52% of everything else arriving at the same cells. Those
    /// synapses are not noise: a Kenyon cell is MEANT to contribute almost
    /// nothing on its own, the odour is carried by which two thousand of them
    /// fire together, and applying a per-pair threshold to a population code
    /// removes the code and leaves the cells.
    ///
    /// So the exemption is a statement about one pathway, made by whoever
    /// knows which pathway it is, rather than a weaker floor everywhere.
    pub fn network_keeping(
        &self,
        scale: f32,
        size_limit: Option<f32>,
        min_synapses: u32,
        keep: &std::collections::HashSet<(u32, u32)>,
    ) -> neuro::Csc {
        let mut csc = if min_synapses > 1 {
            let mut edges: Vec<(u32, u32, f32)> = Vec::new();
            for post in 0..self.csc.n as usize {
                let (a, b) = (self.csc.indptr[post] as usize, self.csc.indptr[post + 1] as usize);
                for k in a..b {
                    if self.csc.w[k] >= min_synapses as f32 || keep.contains(&(self.csc.pre[k], post as u32)) {
                        edges.push((self.csc.pre[k], post as u32, self.csc.w[k]));
                    }
                }
            }
            // Cannot fail: every index came out of a valid graph of the same
            // size, so `from_edges` sees only in-range endpoints.
            neuro::Csc::from_edges(self.csc.n, &edges).unwrap_or_else(|_| self.csc.clone())
        } else {
            self.csc.clone()
        };
        let signs: Vec<f32> = csc.pre.iter().map(|p| self.neurons.get(*p as usize).map_or(0.0, |n| n.nt.sign())).collect();
        for (w, sign) in csc.w.iter_mut().zip(&signs) {
            *w *= sign * scale;
        }
        if let Some(limit) = size_limit {
            let _ = csc.scale_by_post(&self.excitability(limit));
        }
        csc
    }

    /// Per-neuron excitability, from reconstructed membrane area.
    ///
    /// A leaky integrate-and-fire membrane obeys `C dV/dt = -g_L(V - V_rest) + I`.
    /// Both the capacitance and the leak conductance scale with membrane area,
    /// so the time constant `C/g_L` is size-independent but the voltage a given
    /// current produces goes as `1/g_L`, and therefore as one over the area.
    /// Giving every cell the same threshold and the same input resistance
    /// therefore makes the largest cells in the cord hundreds of times more
    /// excitable than the smallest - which is not a small modelling liberty,
    /// it is the difference between a network that oscillates and one that
    /// saturates.
    ///
    /// Returned as a factor to MULTIPLY a neuron's input by, normalised so the
    /// median neuron gets exactly `1.0`: the population keeps whatever overall
    /// gain the caller chose, and only the spread changes. Clamped to
    /// `[1/limit, limit]` because reconstruction leaves a long tail of
    /// fragments and giant cells, and an unclamped factor turns one badly
    /// reconstructed neuron into a silent one or a runaway one.
    ///
    /// A neuron with no recorded area gets the median, not zero: an unknown
    /// size is an absence of evidence, and silencing the cell would be a
    /// strong claim made by accident.
    pub fn excitability(&self, limit: f32) -> Vec<f32> {
        let sizes = self.sizes();
        let mut known: Vec<f64> = sizes.iter().copied().filter(|a| *a > 0.0).collect();
        if known.is_empty() {
            return vec![1.0; self.neurons.len()];
        }
        known.sort_by(f64::total_cmp);
        let median = known[known.len() / 2];
        sizes
            .iter()
            .map(|s| if *s <= 0.0 { 1.0 } else { ((median / s) as f32).clamp(1.0 / limit, limit) })
            .collect()
    }

    /// Every neuron's size, in whatever unit the export actually populated.
    ///
    /// Surface area is the quantity the conductance scales with and is
    /// preferred where it exists. Where it does not - and in the MANC Codex
    /// export it exists as a COLUMN and is empty on every row, which is the
    /// failure mode a column-name lookup alone will not catch - volume stands
    /// in for it. The two are proportional at constant neurite radius, and a
    /// proportionality constant is absorbed by the median normalisation in
    /// [`Self::excitability`], so the substitution costs nothing as long as it
    /// is the same choice for every neuron. Mixed units across neurons would
    /// be silently wrong, which is why this returns 0 rather than falling back
    /// per row.
    pub fn sizes(&self) -> Vec<f64> {
        let area = self.neurons.iter().any(|n| n.surface_area_nm2 > 0.0);
        self.neurons
            .iter()
            .map(|n| if area { n.surface_area_nm2 } else { n.volume_nm3 })
            .collect()
    }

    /// [`Self::signed_csc`], with every neuron's inputs scaled by its own
    /// [`Self::excitability`].
    pub fn signed_csc_sized(&self, scale: f32, limit: f32) -> neuro::Csc {
        let mut csc = self.signed_csc(scale);
        // Cannot fail: `excitability` returns one factor per neuron by
        // construction, and the graph has that many columns.
        let _ = csc.scale_by_post(&self.excitability(limit));
        csc
    }

    /// How many edges [`Self::signed_csc`] would silence, and how many it
    /// would make inhibitory. Reported rather than inferred, because "the
    /// network went quiet" has too many possible causes to guess between.
    pub fn sign_census(&self) -> (usize, usize, usize) {
        let (mut exc, mut inh, mut zero) = (0, 0, 0);
        for &p in &self.csc.pre {
            match self.neurons.get(p as usize).map_or(0.0, |n| n.nt.sign()) {
                s if s > 0.0 => exc += 1,
                s if s < 0.0 => inh += 1,
                _ => zero += 1,
            }
        }
        (exc, inh, zero)
    }

    /// `(mean, median, p99, max)` of the in-degree, the statistic a published
    /// connectome is most cheaply compared against.
    pub fn in_degree_stats(&self) -> (f64, u32, u32, u32) {
        let mut d = self.csc.in_degrees();
        if d.is_empty() {
            return (0.0, 0, 0, 0);
        }
        let mean = d.iter().map(|&x| x as f64).sum::<f64>() / d.len() as f64;
        d.sort_unstable();
        (mean, d[d.len() / 2], d[(d.len() as f64 * 0.99) as usize], *d.last().unwrap())
    }
}
