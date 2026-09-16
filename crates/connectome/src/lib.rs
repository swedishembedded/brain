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

pub mod codex;
pub mod csv;

pub use codex::{load, load_readers, Coverage};

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
}

impl Nt {
    /// Codex spells these `ACH`, `GLUT`, `GABA`, `DA`, `SER`, `OCT`, `HIST`.
    pub fn parse(s: &str) -> Option<Nt> {
        Some(match s.trim().to_ascii_uppercase().as_str() {
            "ACH" | "ACETYLCHOLINE" => Nt::Acetylcholine,
            "GLUT" | "GLUTAMATE" => Nt::Glutamate,
            "GABA" => Nt::Gaba,
            "DA" | "DOPAMINE" => Nt::Dopamine,
            "SER" | "SEROTONIN" => Nt::Serotonin,
            "OCT" | "OCTOPAMINE" => Nt::Octopamine,
            "HIST" | "HISTAMINE" => Nt::Histamine,
            _ => return None,
        })
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
            Nt::Acetylcholine | Nt::Dopamine | Nt::Serotonin | Nt::Octopamine => 1.0,
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
            1.0
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
    /// `leg_motor_neuron`, `wing_motor_neuron`, ...
    pub class: String,
    /// The nerve a motor or sensory neuron runs in, e.g.
    /// `left_prothoracic_leg_nerve`. This is the handle a body attaches to.
    pub nerve: String,
    pub soma_side: String,
    pub cell_type: String,
    pub nt: NtPrior,
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
