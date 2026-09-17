// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The fly's learning organ, located in a published wiring diagram.
//!
//! A connectome is not a trained model. It is an anatomical measurement, and
//! the parameters that make it behave are not in it. The usual response is to
//! put dynamics on top and optimise everything, which works and quietly stops
//! being about the connectome: a rule free to change sixteen million edges
//! will find a solution that the wiring is not doing any of the work in. The
//! measurement that catches this is a degree-matched shuffle, and when a
//! result survives it, the result was never about the animal.
//!
//! The alternative is the one the biology already specifies. In *Drosophila*
//! the best understood learning happens at an identified population of
//! synapses in the mushroom body, and it is gated by identified dopaminergic
//! cells:
//!
//! ```text
//!     odour -> receptor neurons -> projection neurons
//!                                        |
//!                                        v
//!                                  Kenyon cells          (sparse odour code)
//!                                        |
//!                                   [ PLASTIC ]          KC -> MBON
//!                                        |
//!                                        v
//!                              output neurons (MBON)     approach / avoid
//!                                        ^
//!                                        |
//!                         dopaminergic neurons (PAM / PPL1)
//!                                   reinforcement
//! ```
//!
//! Each dopaminergic cell innervates ONE compartment of the mushroom body and
//! modulates only the Kenyon-cell synapses there, which is what makes sugar
//! and shock different signals rather than one number with two signs. This
//! module finds those populations and that compartment structure, and hands
//! back a [`neuro::Sites`] map restricting plasticity to them: in BANC, 0.1%
//! of the edges. The other 99.9% stay exactly as measured.
//!
//! Nothing here is curated by hand. The populations come from the dataset's
//! own class labels and the compartments come from the graph - a compartment
//! is an output neuron together with the dopaminergic cells that synapse onto
//! it. That the result reproduces the published compartment map (MBON11 with
//! PPL101, MBON01 with PAM01) is a test in this crate, not an assumption.
//!
//! Swedish Embedded AB implements connectome-grounded learning systems for its
//! clients, including the population identification that keeps a plasticity
//! rule answerable to the anatomy it claims to model. If your team needs this,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::{Connectome, Neuron, Nt};

/// BANC/FlyWire class labels for the three populations.
const KENYON: &str = "kenyon_cell";
const OUTPUT: &str = "mushroom_body_output_neuron";
const DOPAMINERGIC: &str = "mushroom_body_dopaminergic_neuron";

/// How a mushroom body is identified. All of it is a judgement call, so all of
/// it is a knob with a stated default rather than a constant in a loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    /// A dopaminergic cell joins an output neuron's compartment only if it
    /// makes at least this many synapses onto it.
    ///
    /// Reconstruction assigns a long tail of one- and two-synapse pairs at the
    /// resolution limit. Left in, they would put nearly every dopaminergic
    /// cell in nearly every compartment, which is precisely the broadcast
    /// signal that compartments exist to avoid.
    pub min_dan_synapses: u32,
    /// Require a dopaminergic cell's transmitter prediction to agree with its
    /// class label.
    ///
    /// The class is an anatomical call and the transmitter is a separate
    /// machine-learning prediction, so they can disagree. Off by default: the
    /// class label is the stronger evidence for these cells, and about one in
    /// six has no confident prediction at all.
    pub require_dopamine: bool,
}

impl Default for Policy {
    fn default() -> Self {
        Policy { min_dan_synapses: 5, require_dopamine: false }
    }
}

/// One compartment: an output neuron and the cells that reinforce it.
#[derive(Clone, Debug, PartialEq)]
pub struct Compartment {
    /// The output neuron whose incoming Kenyon-cell synapses are plastic here.
    pub mbon: u32,
    /// Its published cell type, e.g. `MBON11`.
    pub name: String,
    /// The dopaminergic neurons innervating it.
    pub dans: Vec<u32>,
    /// How many synapses they make onto it. The evidence for the compartment.
    pub dan_synapses: u32,
}

/// The learning organ, as found in one dataset.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MushroomBody {
    pub kc: Vec<u32>,
    pub mbon: Vec<u32>,
    pub dan: Vec<u32>,
    /// One per output neuron that has dopaminergic input above the policy.
    pub compartments: Vec<Compartment>,
}

impl MushroomBody {
    /// Find it, or return an empty one if this dataset has no mushroom body.
    ///
    /// An empty result is normal and is not an error: MANC is a nerve cord,
    /// the fly equivalent of a spinal cord, and contains no mushroom body at
    /// all. A caller that needs one should say so itself, because "the animal
    /// cannot learn" and "you loaded the wrong quarter of the animal" are
    /// different problems with the same symptom.
    pub fn find(c: &Connectome, policy: Policy) -> MushroomBody {
        let is_dan = |n: &Neuron| {
            n.class == DOPAMINERGIC && (!policy.require_dopamine || n.nt.best() == Some(Nt::Dopamine))
        };
        let kc = c.population(|n| n.class == KENYON);
        let mbon = c.population(|n| n.class == OUTPUT);
        let dan = c.population(is_dan);

        // A compartment is read off the graph: the dopaminergic cells that
        // actually synapse onto this output neuron, at this dataset's own
        // synapse counts.
        let dan_set: std::collections::HashSet<u32> = dan.iter().copied().collect();
        let mut compartments = Vec::new();
        for &m in &mbon {
            let (lo, hi) = (c.csc.indptr[m as usize] as usize, c.csc.indptr[m as usize + 1] as usize);
            let mut dans = Vec::new();
            let mut synapses = 0.0f32;
            for k in lo..hi {
                let pre = c.csc.pre[k];
                if dan_set.contains(&pre) && c.csc.w[k] >= policy.min_dan_synapses as f32 {
                    dans.push(pre);
                    synapses += c.csc.w[k];
                }
            }
            if dans.is_empty() {
                continue;
            }
            let name = c.neurons[m as usize].cell_type.clone();
            compartments.push(Compartment { mbon: m, name, dans, dan_synapses: synapses as u32 });
        }
        MushroomBody { kc, mbon, dan, compartments }
    }

    pub fn is_empty(&self) -> bool {
        self.compartments.is_empty()
    }

    /// Restrict plasticity to the Kenyon-cell output synapses, compartment by
    /// compartment.
    ///
    /// Built against `csc` rather than against the import, because the network
    /// a creature runs is a PRUNED and signed derivative of the import and its
    /// edge numbering is its own. A site map built against the wrong graph
    /// would mark real edges, just not the intended ones, and would do it
    /// silently.
    ///
    /// `gain` is signed and is the direction of the rule. It defaults negative
    /// at the call site for a reason: dopamine paired with Kenyon-cell
    /// activity DEPRESSES that cell's output synapse. A fly does not learn
    /// that sugar is good by strengthening the path to approach; it learns by
    /// weakening the path to the output neuron that drives avoidance, and the
    /// behaviour follows from the imbalance between opposing output neurons.
    pub fn sites(&self, csc: &neuro::Csc, gain: f32, decay: f32) -> neuro::Sites {
        let mut sites = neuro::Sites::new(csc.nnz());
        sites.decay = decay;
        let kc: std::collections::HashSet<u32> = self.kc.iter().copied().collect();
        for comp in &self.compartments {
            let c = sites.compartment(&comp.dans, gain);
            let m = comp.mbon as usize;
            if m + 1 >= csc.indptr.len() {
                continue;
            }
            let (lo, hi) = (csc.indptr[m] as usize, csc.indptr[m + 1] as usize);
            let edges: Vec<u32> = (lo..hi).filter(|&k| kc.contains(&csc.pre[k])).map(|k| k as u32).collect();
            // Cannot fail: the compartment was just created and every edge
            // index came from this graph's own column range.
            let _ = sites.assign(&edges, c);
        }
        sites
    }

    /// The dopaminergic synapses onto output neurons, as edges of `csc`.
    ///
    /// These need taking OUT of the fast pathway by whoever builds the
    /// network, and the reason is a modelling error worth naming. A
    /// connectome's transmitter prediction is turned into a sign by
    /// convention, and the convention gives dopamine `+1` because that is what
    /// published whole-brain models do. So a dopaminergic synapse currently
    /// injects excitatory current like any other. That is wrong twice over:
    /// dopamine acts through G-protein-coupled receptors on a timescale of
    /// seconds rather than gating a fast channel, and here it would also mean
    /// the reinforcement signal DRIVES the output neuron it is supposed to be
    /// teaching - so a post-training change in that neuron's odour response
    /// could be nothing but the reinforcement still arriving.
    pub fn modulatory_edges(&self, csc: &neuro::Csc) -> Vec<u32> {
        let dan: std::collections::HashSet<u32> = self.dan.iter().copied().collect();
        let mut out = Vec::new();
        for &m in &self.mbon {
            let m = m as usize;
            if m + 1 >= csc.indptr.len() {
                continue;
            }
            let (lo, hi) = (csc.indptr[m] as usize, csc.indptr[m + 1] as usize);
            out.extend((lo..hi).filter(|&k| dan.contains(&csc.pre[k])).map(|k| k as u32));
        }
        out
    }

    /// One line, for a run to print before it claims anything.
    pub fn summary(&self) -> String {
        format!(
            "{} Kenyon cells, {} output neurons, {} dopaminergic neurons, {} compartments",
            self.kc.len(),
            self.mbon.len(),
            self.dan.len(),
            self.compartments.len()
        )
    }
}

impl MushroomBody {
    /// The Kenyon-cell-to-output-neuron pairs, for
    /// [`Connectome::network_keeping`].
    ///
    /// These are the pairs a synapse floor must not remove, and the reason is
    /// documented there: this pathway is sparse per pair by design, so a
    /// per-pair threshold deletes the population code rather than cleaning it.
    pub fn plastic_pairs(&self, c: &Connectome) -> std::collections::HashSet<(u32, u32)> {
        let kc: std::collections::HashSet<u32> = self.kc.iter().copied().collect();
        let mut out = std::collections::HashSet::new();
        for comp in &self.compartments {
            let m = comp.mbon as usize;
            let (lo, hi) = (c.csc.indptr[m] as usize, c.csc.indptr[m + 1] as usize);
            for k in lo..hi {
                if kc.contains(&c.csc.pre[k]) {
                    out.insert((c.csc.pre[k], comp.mbon));
                }
            }
        }
        out
    }
}
