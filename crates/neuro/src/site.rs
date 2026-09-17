// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a network is allowed to learn, and what tells it to.
//!
//! A three-factor rule needs a third factor, and the usual way to supply one
//! is a scalar the host sets when it judges the animal to have done well.
//! That makes every synapse plastic and every reinforcer the same signal,
//! which is a fine instrument and a poor animal. Two things go wrong with it
//! at connectome scale. A rule that may change all 16 million edges can solve
//! a task by leaving the measured wiring behind entirely, so the result stops
//! being about the connectome. And one broadcast number cannot distinguish
//! sugar from shock, so it can teach "do more" but not "approach this and
//! avoid that".
//!
//! [`Sites`] is the restriction. Each synapse names a COMPARTMENT, and each
//! compartment names the neurons whose spiking is its modulator. Compartment
//! `0` is reserved and permanently inert: a synapse assigned to it never
//! learns, exactly, which is what makes "only these synapses are plastic" a
//! bit-identical claim about every other edge rather than a hope.
//!
//! Nothing here knows what a mushroom body is. The compartments are built by
//! whoever has the annotations - `connectome::mushroom_body` does it from
//! published cell types - because the identities are the dataset's business
//! and the arithmetic is this crate's.
//!
//! Swedish Embedded AB implements biologically sited learning rules over
//! large graphs for its clients. If your team needs plasticity that is
//! restricted to identified populations and driven by the network's own
//! activity rather than by an external optimiser, you can procure our
//! services by sending an email to info@swedishembedded.com.

/// The compartment that means "this synapse does not learn".
pub const INERT: u32 = 0;

/// Which synapses are plastic, and which neurons modulate each of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Sites {
    /// Compartment of each edge, parallel to the weight array. [`INERT`]
    /// unless assigned.
    of_edge: Vec<u32>,
    /// Source-list starts, CSC-style over compartments.
    indptr: Vec<u32>,
    /// Neuron index of each modulatory source.
    source: Vec<u32>,
    /// Signed gain on each compartment's summed drive. Negative for a rule
    /// that depresses, which is what dopamine does to a Kenyon cell's output
    /// synapse.
    gain: Vec<f32>,
    /// Modulator decay per tick. The modulator is a chemical level rather
    /// than a spike: it outlasts the firing that produced it, and how long it
    /// does so is how late a reinforcer may arrive and still teach anything.
    pub decay: f32,
}

impl Sites {
    /// A map in which nothing is plastic yet.
    pub fn new(nnz: usize) -> Sites {
        Sites { of_edge: vec![INERT; nnz], indptr: vec![0, 0], source: Vec::new(), gain: vec![0.0], decay: 0.9 }
    }

    /// Every synapse plastic under one experimenter-driven compartment.
    ///
    /// The global rule, expressed in this vocabulary rather than beside it, so
    /// there is one learning path and not two. [`crate::Plastic::modulate`]
    /// drives it.
    pub fn everywhere(nnz: usize) -> Sites {
        let mut s = Sites::new(nnz);
        let c = s.compartment(&[], 1.0);
        s.of_edge = vec![c; nnz];
        s
    }

    /// Add a compartment modulated by `sources`, and return its index.
    ///
    /// An empty `sources` means the host owns the level: `modulate` writes it
    /// and [`crate::SpikingNet`] clears it after the tick, so a reward
    /// delivered once is applied once.
    pub fn compartment(&mut self, sources: &[u32], gain: f32) -> u32 {
        let c = self.gain.len() as u32;
        self.source.extend_from_slice(sources);
        self.indptr.push(self.source.len() as u32);
        self.gain.push(gain);
        c
    }

    /// Put `edges` in `compartment`.
    pub fn assign(&mut self, edges: &[u32], compartment: u32) -> Result<(), String> {
        if compartment as usize >= self.gain.len() {
            return Err(format!("compartment {compartment} does not exist ({} defined)", self.gain.len()));
        }
        let nnz = self.of_edge.len();
        for &k in edges {
            let slot =
                self.of_edge.get_mut(k as usize).ok_or_else(|| format!("edge {k} is outside a graph of {nnz} edges"))?;
            *slot = compartment;
        }
        Ok(())
    }

    /// Number of compartments, counting the inert one.
    pub fn len(&self) -> usize {
        self.gain.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plastic_edges() == 0
    }

    /// How many synapses may learn.
    pub fn plastic_edges(&self) -> usize {
        self.of_edge.iter().filter(|&&c| c != INERT).count()
    }

    /// Compartment of each edge.
    pub fn of_edge(&self) -> &[u32] {
        &self.of_edge
    }

    /// The modulatory sources of one compartment.
    pub fn sources(&self, c: u32) -> &[u32] {
        let (lo, hi) = (self.indptr[c as usize] as usize, self.indptr[c as usize + 1] as usize);
        &self.source[lo..hi]
    }

    pub(crate) fn parts(&self) -> (&[u32], &[u32], &[u32], &[f32]) {
        (&self.of_edge, &self.indptr, &self.source, &self.gain)
    }

    /// Whether a compartment's level is written by the host rather than
    /// computed from spikes. The inert compartment is neither.
    pub(crate) fn host_driven(&self, c: u32) -> bool {
        c != INERT && self.sources(c).is_empty()
    }

    pub fn validate(&self, nnz: usize, n: u32) -> Result<(), String> {
        if self.of_edge.len() != nnz {
            return Err(format!("site map is {} edges, the graph is {nnz}", self.of_edge.len()));
        }
        if !(0.0..1.0).contains(&self.decay) {
            return Err(format!("modulator decay must be in [0, 1), got {} - a level of 1 never clears", self.decay));
        }
        if let Some(&bad) = self.source.iter().find(|&&s| s >= n) {
            return Err(format!("modulatory source {bad} is outside a network of {n} neurons"));
        }
        if !self.sources(INERT).is_empty() {
            return Err("compartment 0 is reserved as inert and cannot have sources".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_plastic_until_it_is_assigned() {
        let mut s = Sites::new(10);
        assert_eq!(s.plastic_edges(), 0);
        assert_eq!(s.len(), 1, "only the inert compartment exists");
        let c = s.compartment(&[3, 4], -1.0);
        assert_eq!(c, 1, "compartment 0 is reserved");
        s.assign(&[0, 2, 9], c).unwrap();
        assert_eq!(s.plastic_edges(), 3);
        assert_eq!(s.sources(c), &[3, 4]);
        assert!(!s.host_driven(c));
    }

    #[test]
    fn a_source_outside_the_network_is_refused_rather_than_read() {
        let mut s = Sites::new(4);
        let c = s.compartment(&[7], 1.0);
        assert!(s.validate(4, 5).is_err(), "source 7 is not a neuron of a 5-neuron network");
        assert!(s.assign(&[99], c).is_err(), "edge 99 is not an edge of a 4-edge graph");
        assert!(Sites::everywhere(4).validate(4, 5).is_ok());
    }
}
