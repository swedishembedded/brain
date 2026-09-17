// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The parameters a wiring diagram does not contain, shared the way the
//! anatomy shares them.
//!
//! A connectome edge says `A contacts B, N times`. That is not enough to run
//! anything. Executing it needs a membrane time constant, an excitability, a
//! tonic input and a conductance per synapse, and none of those are in any
//! file. They are what a connectome-constrained model INFERS, and the choice
//! that decides whether the result still says anything about the connectome is
//! how many of them there are.
//!
//! Two wrong answers bracket the right one. Ours was the first: eleven gains,
//! one per super class, which is too coarse to express that a descending
//! neuron's effect on an inhibitory interneuron differs from its effect on a
//! motor neuron - so the search compensates by pushing whole populations
//! around and lands somewhere the anatomy is not doing the work. The other is
//! a free parameter per synapse, which is millions of them, and at that point
//! the measured graph is a sparsity mask on a fitted recurrent network and a
//! degree-matched shuffle will do just as well.
//!
//! The answer that works elsewhere is to share a parameter across every
//! connection of the same TYPE: one non-negative multiplier per
//! (presynaptic class, postsynaptic class), one time constant per class, one
//! excitability per class, one bias per class. Sign and synapse count stay
//! exactly as measured; only the strength scaling is fitted, and it is fitted
//! per connection type rather than per connection.
//!
//! On MANC that is:
//!
//! ```text
//!   36 published classes
//!   693 connection types that actually occur, covering 95% of edges
//!   36 x 3 per-class parameters
//!   ~800 numbers, against 1.4 million synapses
//! ```
//!
//! which is the same order as the published connectome-constrained visual
//! model, arrived at from this dataset's own annotation rather than by
//! copying a number.
//!
//! Swedish Embedded AB implements connectome-constrained models for its
//! clients, including the parameter-sharing design that keeps a fitted model
//! answerable to the anatomy it is built on. If your team needs this, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;

use connectome::Connectome;

use crate::Wiring;

/// A neuron with no published class still has to go somewhere.
const UNNAMED: &str = "<unnamed>";

/// The free parameters, all of them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params {
    /// One non-negative multiplier per connection type, indexed by
    /// [`Physiology::types`].
    ///
    /// Non-negative is not a convenience. The sign of a synapse is anatomy -
    /// it follows from the presynaptic transmitter - and an optimiser allowed
    /// to flip it is no longer constrained by the connectome at all, because
    /// any graph can become any other graph if the signs are free.
    pub gain: Vec<f32>,
    /// Membrane speed per class, as a multiplier on `dt/tau`.
    pub tau: Vec<f32>,
    /// Excitability per class, as a multiplier on the input resistance.
    pub excitability: Vec<f32>,
    /// Tonic input per class, in the same units as a synaptic current.
    ///
    /// Carried here and applied through the drive port rather than through the
    /// kernel, because a per-neuron current already exists and a second way to
    /// inject one would be a second thing to keep consistent.
    pub bias: Vec<f32>,
}

/// Anatomy frozen, physiology free and shared per class.
#[derive(Clone, Debug)]
pub struct Physiology {
    classes: Vec<String>,
    /// Class of each neuron.
    of_neuron: Vec<u16>,
    /// Connection type of each edge, indexed into `types`.
    of_edge: Vec<u32>,
    types: Vec<(u16, u16)>,
    /// Signed, scaled weights at unit gain: the measurement itself.
    base: Vec<f32>,
}

impl Physiology {
    /// Build against the network a creature with this `wiring` will run.
    pub fn new(c: &Connectome, wiring: Wiring) -> Physiology {
        let mut classes: Vec<String> = Vec::new();
        let mut index: BTreeMap<&str, u16> = BTreeMap::new();
        let mut of_neuron = Vec::with_capacity(c.neurons.len());
        for n in &c.neurons {
            let key = if n.class.is_empty() { UNNAMED } else { n.class.as_str() };
            let id = *index.entry(key).or_insert_with(|| {
                classes.push(key.to_string());
                (classes.len() - 1) as u16
            });
            of_neuron.push(id);
        }

        let net = c.network(wiring.weight_scale, wiring.size_limit, wiring.min_synapses);
        let mut types: Vec<(u16, u16)> = Vec::new();
        let mut seen: BTreeMap<(u16, u16), u32> = BTreeMap::new();
        let mut of_edge = Vec::with_capacity(net.nnz());
        for post in 0..net.n as usize {
            let b = of_neuron[post];
            for k in net.indptr[post]..net.indptr[post + 1] {
                let a = of_neuron[net.pre[k as usize] as usize];
                let id = *seen.entry((a, b)).or_insert_with(|| {
                    types.push((a, b));
                    (types.len() - 1) as u32
                });
                of_edge.push(id);
            }
        }
        Physiology { classes, of_neuron, of_edge, types, base: net.w }
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// The connection types that actually occur, as `(pre class, post class)`.
    pub fn types(&self) -> &[(u16, u16)] {
        &self.types
    }

    /// A readable name for one connection type.
    pub fn type_name(&self, i: usize) -> String {
        let (a, b) = self.types[i];
        format!("{} -> {}", self.classes[a as usize], self.classes[b as usize])
    }

    /// How many free parameters this parameterisation has in total.
    pub fn len(&self) -> usize {
        self.types.len() + 3 * self.classes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    pub fn edges(&self) -> usize {
        self.of_edge.len()
    }

    /// The neutral point: the connectome exactly as measured.
    ///
    /// Every gain 1, every time constant and excitability 1, no tonic input.
    /// This is the control every fitted result is read against, and it has to
    /// reproduce the imported weights BIT for bit rather than closely.
    pub fn unit(&self) -> Params {
        Params {
            gain: vec![1.0; self.types.len()],
            tau: vec![1.0; self.classes.len()],
            excitability: vec![1.0; self.classes.len()],
            bias: vec![0.0; self.classes.len()],
        }
    }

    pub fn validate(&self, p: &Params) -> Result<(), String> {
        let (t, c) = (self.types.len(), self.classes.len());
        for (name, got, want) in [
            ("gain", p.gain.len(), t),
            ("tau", p.tau.len(), c),
            ("excitability", p.excitability.len(), c),
            ("bias", p.bias.len(), c),
        ] {
            if got != want {
                return Err(format!("{name} has {got} values, this parameterisation has {want}"));
            }
        }
        if let Some(bad) = p.gain.iter().find(|g| !g.is_finite() || **g < 0.0) {
            return Err(format!("a connection gain must be finite and non-negative, got {bad} - sign is anatomy"));
        }
        Ok(())
    }

    /// The weight vector these parameters imply.
    pub fn weights(&self, p: &Params) -> Vec<f32> {
        self.base
            .iter()
            .zip(&self.of_edge)
            .map(|(w, &t)| w * p.gain.get(t as usize).copied().unwrap_or(1.0).max(0.0))
            .collect()
    }

    /// Per-neuron `(tau_scale, gain_scale)` for
    /// [`neuro::SpikingNet::set_cell_scales`].
    pub fn cell_scales(&self, p: &Params) -> (Vec<f32>, Vec<f32>) {
        let pick = |v: &[f32]| -> Vec<f32> {
            self.of_neuron.iter().map(|&c| v.get(c as usize).copied().unwrap_or(1.0).max(1e-4)).collect()
        };
        (pick(&p.tau), pick(&p.excitability))
    }

    /// Per-neuron tonic current, to be added to whatever a body is driving.
    pub fn bias_drive(&self, p: &Params) -> Vec<f32> {
        self.of_neuron.iter().map(|&c| p.bias.get(c as usize).copied().unwrap_or(0.0)).collect()
    }

    /// The knobs an optimiser searches, in the order [`Self::from_knobs`]
    /// reads them back.
    ///
    /// Ranges rather than free reals, for the reason the rest of this crate
    /// gives: these have physical meaning, and values outside these bounds are
    /// not merely worse but meaningless. A gain of zero deletes a connection
    /// type the anatomy says exists; a membrane a hundred times too fast is
    /// not a membrane.
    pub fn knobs(&self) -> Vec<crate::search::Knob> {
        let mut k = Vec::with_capacity(self.len());
        for i in 0..self.types.len() {
            k.push(crate::search::Knob::new(format!("w:{}", self.type_name(i)), 0.0, 4.0));
        }
        for (what, lo, hi) in [("tau", 0.1f32, 4.0f32), ("exc", 0.1, 4.0), ("bias", -1.0, 1.0)] {
            for c in &self.classes {
                k.push(crate::search::Knob::new(format!("{what}:{c}"), lo, hi));
            }
        }
        k
    }

    /// Read a knob vector back into parameters.
    pub fn from_knobs(&self, values: &[f32]) -> Params {
        let (t, c) = (self.types.len(), self.classes.len());
        let take = |from: usize, n: usize, default: f32| -> Vec<f32> {
            (0..n).map(|i| values.get(from + i).copied().unwrap_or(default)).collect()
        };
        Params {
            gain: take(0, t, 1.0),
            tau: take(t, c, 1.0),
            excitability: take(t + c, c, 1.0),
            bias: take(t + 2 * c, c, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEURONS: &str = "\
Root ID,Flow,Super Class,Class,Nerve,Soma side,Primary Cell Type,Predicted NT type,Predicted NT confidence,Verified NT type
1,afferent,descending,descending_neuron,,right,DNp01,ACH,0.99,
2,intrinsic,ventral_nerve_cord_intrinsic,intrinsic_neuron,,left,IN01,GABA,0.99,
3,efferent,motor,fl,left_prothoracic_leg_nerve,left,MN01,ACH,0.99,
4,efferent,motor,fl,right_prothoracic_leg_nerve,right,MN02,ACH,0.99,
";
    const EDGES: &str = "\
pre_root_id,post_root_id,neuropil,syn_count,nt_type
1,2,LegNp_T1,10,ACH
1,3,LegNp_T1,20,ACH
2,3,LegNp_T1,30,GABA
2,4,LegNp_T1,40,GABA
";

    fn fixture() -> Connectome {
        connectome::load_readers("test", NEURONS.as_bytes(), EDGES.as_bytes()).expect("loads")
    }

    #[test]
    fn parameters_are_shared_per_connection_type_not_per_synapse() {
        let c = fixture();
        let ph = Physiology::new(&c, Wiring { min_synapses: 1, size_limit: None, ..Wiring::default() });
        // Three classes, and three connection types among four edges: the two
        // intrinsic-to-motor edges share one parameter.
        assert_eq!(ph.classes().len(), 3, "{:?}", ph.classes());
        assert_eq!(ph.types().len(), 3, "{:?}", (0..ph.types().len()).map(|i| ph.type_name(i)).collect::<Vec<_>>());
        assert_eq!(ph.edges(), 4);
        assert_eq!(ph.len(), 3 + 3 * 3);
    }

    #[test]
    fn unit_parameters_reproduce_the_connectome_bit_for_bit() {
        let c = fixture();
        let w = Wiring { min_synapses: 1, ..Wiring::default() };
        let ph = Physiology::new(&c, w);
        let p = ph.unit();
        ph.validate(&p).unwrap();
        assert_eq!(ph.weights(&p), c.network(w.weight_scale, w.size_limit, w.min_synapses).w);
        let (tau, exc) = ph.cell_scales(&p);
        assert!(tau.iter().all(|&x| x == 1.0) && exc.iter().all(|&x| x == 1.0));
        assert!(ph.bias_drive(&p).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn a_gain_reaches_every_edge_of_its_type_and_no_other() {
        let c = fixture();
        let ph = Physiology::new(&c, Wiring { min_synapses: 1, size_limit: None, ..Wiring::default() });
        let base = ph.weights(&ph.unit());
        let inhibitory = (0..ph.types().len())
            .find(|&i| ph.type_name(i) == "intrinsic_neuron -> fl")
            .expect("the inhibitory interneuron reaches both motor neurons");
        let mut p = ph.unit();
        p.gain[inhibitory] = 0.0;
        let w = ph.weights(&p);
        let zeroed: Vec<usize> = (0..w.len()).filter(|&k| w[k] == 0.0 && base[k] != 0.0).collect();
        assert_eq!(zeroed.len(), 2, "both edges of that type, and only those");
        for k in 0..w.len() {
            if !zeroed.contains(&k) {
                assert_eq!(w[k], base[k], "edge {k} is of another type and must not move");
            }
        }
    }

    #[test]
    fn a_gain_may_not_flip_a_sign_because_sign_is_anatomy() {
        let c = fixture();
        let ph = Physiology::new(&c, Wiring { min_synapses: 1, ..Wiring::default() });
        let mut p = ph.unit();
        p.gain[0] = -1.0;
        assert!(ph.validate(&p).is_err(), "a negative gain would turn an inhibitory synapse excitatory");
        // And even unvalidated, the arithmetic refuses.
        assert!(ph.weights(&p).iter().zip(&ph.weights(&ph.unit())).all(|(a, b)| a.signum() == b.signum() || *a == 0.0));
    }

    #[test]
    fn knobs_round_trip_through_the_parameters_they_name() {
        let c = fixture();
        let ph = Physiology::new(&c, Wiring { min_synapses: 1, ..Wiring::default() });
        let knobs = ph.knobs();
        assert_eq!(knobs.len(), ph.len());
        let values: Vec<f32> = knobs.iter().enumerate().map(|(i, _)| 0.5 + i as f32 * 0.01).collect();
        let p = ph.from_knobs(&values);
        assert_eq!(p.gain.len(), ph.types().len());
        assert_eq!(p.bias.len(), ph.classes().len());
        assert_eq!(p.gain[0], values[0]);
        assert_eq!(p.bias[0], values[ph.types().len() + 2 * ph.classes().len()]);
    }
}
