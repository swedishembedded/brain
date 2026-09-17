// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Teaching a connectome that one smell means something, the way a fly does.
//!
//! Everything else in this crate optimises parameters against a behaviour: a
//! search proposes gains, an episode scores them, and the best set is kept.
//! That is machine learning wearing a connectome, and the shuffle control
//! keeps saying so. This module is the other thing. Nothing optimises
//! anything here. The animal is placed in a protocol, its own dopaminergic
//! cells fire, and the synapses that happened to be active change - and
//! whether that produces a memory is a measurement rather than an objective.
//!
//! The protocol is differential conditioning, which is what the behavioural
//! literature actually runs:
//!
//! ```text
//!   test      odour A alone, then odour B alone      (what it thinks now)
//!   train     odour A together with reinforcement    (the pairing)
//!             odour B alone                          (the discrimination)
//!   test      odour A alone, then odour B alone      (what it thinks after)
//! ```
//!
//! The reinforcement is delivered by stimulating identified dopaminergic
//! neurons, exactly as the optogenetic experiments do, rather than by adding
//! a term to a loss. What changes is the Kenyon-cell synapses onto that
//! compartment's output neuron, and only those: 0.05% of the graph.
//!
//! The control that matters is UNPAIRED, not "no reinforcement". A fly given
//! the same odour and the same dopamine, with the two separated in time,
//! receives identical total stimulation and learns nothing. If the paired and
//! unpaired conditions come out the same here, then whatever moved was not an
//! association, and no amount of it moving is evidence of learning.
//!
//! Swedish Embedded AB implements biologically grounded learning systems for
//! its clients, including the protocol design and controls that separate a
//! system that learned from one that merely changed. If your team needs this,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::HashSet;

use connectome::mushroom_body::{MushroomBody, Policy};
use connectome::Connectome;
use neuro::{DynamicalSystem, LifParams, PlasticityParams, Port, SpikingNet};

use crate::Wiring;

/// Transmitters the mushroom body needs and BANC's classifier abstained on.
///
/// Sign comes from a predicted transmitter, and a cell with no prediction gets
/// a sign of zero, so it is not weakened but DELETED while still counting as a
/// neuron. Across BANC that costs 1.4% of synapses and is fair enough. It is
/// not fair for APL.
///
/// APL is one neuron per hemisphere that receives from the whole Kenyon-cell
/// population and inhibits the whole Kenyon-cell population, and it is the
/// reason the odour code is sparse: it is a gain control that holds the number
/// of responding cells roughly constant however strong the odour. It is
/// GABAergic, which is not in doubt - it is the fly's counterpart of the
/// locust giant GABAergic neuron, and the sparseness of Kenyon-cell responses
/// has been shown to depend on it. BANC's predictor returns nothing for it,
/// so its 22,430 output synapses are multiplied by zero, and the mushroom body
/// runs with no normalisation at all. That is 0.05% of the graph and all of
/// the odour code.
///
/// Returned rather than applied silently, and applied here rather than at
/// import, because this is knowledge added to a dataset and every use of it
/// should have to say so.
pub fn restore_known_transmitters(c: &mut Connectome) -> Vec<(String, usize)> {
    [("APL", connectome::Nt::Gaba), ("DPM", connectome::Nt::Serotonin)]
        .into_iter()
        .map(|(t, nt)| (t.to_string(), c.assume_transmitter(t, nt)))
        .filter(|(_, n)| *n > 0)
        .collect()
}

/// An odour, as the glomeruli its receptor neurons project to.
///
/// Not an arbitrary input vector. BANC names every olfactory receptor neuron
/// by its glomerulus - `ORN_DM1`, `ORN_DA2`, 56 of them - and a glomerulus is
/// the unit a receptor's ligand spectrum is published in, so an odour
/// expressed this way is a real and checkable object rather than a vector
/// somebody chose.
///
/// What is NOT claimed: that a particular named chemical produces exactly this
/// set. A real odour drives a distributed, graded, overlapping pattern across
/// many glomeruli, and that tuning is published elsewhere and is not in this
/// dataset. Two odours here are two sparse patterns over real glomeruli, which
/// is the property the mushroom body operates on, and the discrimination being
/// tested is between the patterns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Odour {
    pub name: String,
    pub glomeruli: Vec<String>,
    /// The receptor neurons, as graph indices.
    pub orns: Vec<u32>,
}

impl Odour {
    pub fn new(c: &Connectome, name: &str, glomeruli: &[&str]) -> Odour {
        let want: HashSet<String> = glomeruli.iter().map(|g| format!("ORN_{g}")).collect();
        let orns = c.population(|n| n.class == "olfactory_receptor_neuron" && want.contains(&n.cell_type));
        Odour {
            name: name.to_string(),
            glomeruli: glomeruli.iter().map(|g| g.to_string()).collect(),
            orns,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.orns.is_empty()
    }
}

/// When the reinforcement arrives relative to the odour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pairing {
    /// Dopamine during the odour. The experimental condition.
    Paired,
    /// Dopamine between odour presentations, never during one. The control
    /// that isolates association from exposure: identical total odour,
    /// identical total dopamine, no coincidence.
    Unpaired,
    /// Odour with no reinforcement at all.
    OdourOnly,
    /// Reinforcement with no odour.
    ReinforcerOnly,
}

/// The shape of a conditioning session, in ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Protocol {
    /// Ticks of no input before each measurement, so a test measures the
    /// response to its own odour rather than the tail of the last one.
    pub settle: u32,
    /// Ticks of odour in a test presentation.
    pub test: u32,
    pub trials: u32,
    /// Ticks of odour per training trial.
    pub trial: u32,
    /// Ticks between training trials. The unpaired condition puts its
    /// dopamine here.
    pub gap: u32,
    /// Current injected into a receptor neuron while its odour is present.
    pub odour_current: f32,
    /// Current injected into the compartment's dopaminergic neurons.
    pub reinforcer_current: f32,
}

impl Default for Protocol {
    fn default() -> Self {
        // 12 trials of 60 ticks: a real fly is conditioned in 1 to 10 pairings
        // of about a minute, and the ratio that matters is trials to
        // eligibility-trace length rather than absolute time.
        Protocol {
            settle: 40,
            test: 120,
            trials: 12,
            trial: 60,
            gap: 60,
            odour_current: 4.0,
            reinforcer_current: 8.0,
        }
    }
}

/// What the output neurons did during one presentation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Response {
    /// Spikes per output neuron, parallel to [`MushroomBody::mbon`].
    pub mbon: Vec<f64>,
    /// Spikes summed over the Kenyon cells.
    pub kc: f64,
    /// How many DISTINCT Kenyon cells fired at least once.
    ///
    /// The number that says whether there is an odour code at all, and the
    /// reason it is counted separately from the spike total. A mushroom body
    /// represents an odour by WHICH few percent of its Kenyon cells respond;
    /// the published figure is of the order of 5%. A population where a
    /// quarter of the cells fire on every tick has a large spike total and no
    /// code, because every odour looks the same, and nothing downstream can
    /// learn a distinction that the representation does not make.
    pub kc_active: usize,
    /// Spikes summed over the dopaminergic neurons. Should be near zero unless
    /// this presentation was reinforced: a network whose teaching signal is on
    /// all the time is not teaching anything.
    pub dan: f64,
    /// Spikes summed over the receptor neurons this odour drives.
    pub orn: f64,
    pub ticks: u32,
}

impl Response {
    pub fn total(&self) -> f64 {
        self.mbon.iter().sum()
    }

    /// The fraction of the Kenyon-cell population that responded.
    pub fn sparseness(&self, kc: usize) -> f64 {
        if kc == 0 {
            0.0
        } else {
            self.kc_active as f64 / kc as f64
        }
    }
}

/// A brain in a conditioning apparatus: no body, no task, no objective.
pub struct Apparatus {
    net: SpikingNet,
    pub mb: MushroomBody,
    n: usize,
    /// Scratch for the per-tick spike readback.
    spike: Vec<f32>,
}

impl Apparatus {
    /// Build one over a connectome.
    ///
    /// Fails loudly when the connectome has no mushroom body, because the
    /// alternative is an apparatus that runs a full protocol, changes nothing,
    /// and reports that the animal did not learn. MANC is a nerve cord and has
    /// no mushroom body; this needs BANC.
    pub fn new(
        gpu: gpu_core::Gpu,
        c: &Connectome,
        wiring: Wiring,
        lif: LifParams,
        plast: PlasticityParams,
        gain: f32,
        policy: Policy,
    ) -> Result<Apparatus, String> {
        let mb = MushroomBody::find(c, policy);
        if mb.is_empty() {
            return Err(format!(
                "{} has no mushroom body ({}), so there is nowhere for an association to be stored - \
                 this needs a connectome that includes the brain",
                c.dataset,
                mb.summary()
            ));
        }
        // The Kenyon-cell pathway is exempt from the synapse floor. See
        // `Connectome::network_keeping`: it is sparse per pair by design, and
        // a floor removes the odour code rather than reconstruction noise.
        let exempt = mb.plastic_pairs(c);
        let mut graph = c.network_keeping(wiring.weight_scale, wiring.size_limit, wiring.min_synapses, &exempt);
        if let Some(seed) = wiring.shuffle_seed {
            graph = graph.shuffled_sources(seed);
        }

        // Dopamine leaves the fast pathway. A connectome's transmitter
        // prediction is signed by a convention that gives dopamine +1, so
        // without this every reinforcement would also be a barrage of
        // excitatory current into the very neuron it is teaching - and a
        // changed response after training could be nothing but that current.
        let modulatory = mb.modulatory_edges(&graph);
        for &k in &modulatory {
            graph.w[k as usize] = 0.0;
        }

        let sites = mb.sites(&graph, gain, plast_decay(&plast));
        if sites.plastic_edges() == 0 {
            return Err("no Kenyon-cell output synapse survived into the network".into());
        }
        let n = graph.n as usize;
        let mut net = SpikingNet::new(gpu, &graph, lif)?;
        net.enable_plasticity_at(plast, sites)?;
        Ok(Apparatus { net, mb, n, spike: vec![0.0; n] })
    }

    pub fn sites(&self) -> &neuro::Sites {
        self.net.sites().expect("plasticity was enabled in the constructor")
    }

    pub fn weights(&self) -> Vec<f32> {
        self.net.weights()
    }

    pub fn set_plasticity(&mut self, on: bool) {
        use neuro::Plastic;
        self.net.set_plasticity(on);
    }

    /// The compartment reinforced by a named dopaminergic cell type.
    ///
    /// `PPL101` is the aversive one in the literature: it innervates the
    /// gamma-1-pedc compartment whose output neuron drives avoidance, and it
    /// is what an electric shock recruits.
    pub fn compartment_driven_by(&self, c: &Connectome, dan_type: &str) -> Option<usize> {
        self.mb.compartments.iter().position(|comp| {
            comp.dans.iter().any(|&d| c.neurons[d as usize].cell_type.starts_with(dan_type))
        })
    }

    fn run(&mut self, odour: Option<&Odour>, reinforce: &[u32], p: &Protocol, ticks: u32) -> Response {
        let mut fired = vec![false; self.mb.kc.len()];
        let mut drive = vec![0.0f32; self.n];
        if let Some(o) = odour {
            for &i in &o.orns {
                drive[i as usize] = p.odour_current;
            }
        }
        for &d in reinforce {
            drive[d as usize] = p.reinforcer_current;
        }
        // Cannot fail: the vector is built at the network's own width.
        let _ = self.net.drive(Port::Drive, &drive);

        let mut r = Response { mbon: vec![0.0; self.mb.mbon.len()], ticks, ..Response::default() };
        for _ in 0..ticks {
            self.net.step();
            // Cannot fail: same width again.
            let _ = self.net.read(Port::Spike, &mut self.spike);
            for (slot, &m) in r.mbon.iter_mut().zip(&self.mb.mbon) {
                *slot += self.spike[m as usize] as f64;
            }
            for (slot, &k) in fired.iter_mut().zip(&self.mb.kc) {
                let s = self.spike[k as usize];
                r.kc += s as f64;
                *slot |= s != 0.0;
            }
            r.dan += self.mb.dan.iter().map(|&d| self.spike[d as usize] as f64).sum::<f64>();
            if let Some(o) = odour {
                r.orn += o.orns.iter().map(|&i| self.spike[i as usize] as f64).sum::<f64>();
            }
        }
        r.kc_active = fired.iter().filter(|&&f| f).count();
        r
    }

    /// Present nothing, and see what the animal does anyway.
    ///
    /// The measurement that says whether a response is a response. A network
    /// at 25% population activity produces a large number for every odour and
    /// the same large number for silence.
    pub fn spontaneous(&mut self, p: &Protocol) -> Response {
        self.net.reset_state();
        self.run(None, &[], p, p.test)
    }

    /// Present an odour and count what the output neurons do, WITHOUT
    /// learning from it.
    ///
    /// A test that taught would confound the thing being measured with the
    /// measurement, and the confound has a direction: every test presentation
    /// of the trained odour would drive the association further, so a protocol
    /// would appear to learn from being examined.
    pub fn test(&mut self, odour: &Odour, p: &Protocol) -> Response {
        let was = {
            use neuro::Plastic;
            self.net.plasticity()
        };
        self.set_plasticity(false);
        // Clear the dynamical state, KEEPING the weights. Without this a test
        // measures the tail of whatever came before it, which puts a drift
        // into the before/after comparison that has nothing to do with
        // learning - and the drift is what the frozen control was picking up.
        self.net.reset_state();
        self.run(None, &[], p, p.settle);
        let r = self.run(Some(odour), &[], p, p.test);
        self.set_plasticity(was);
        r
    }

    /// Run the training phase.
    ///
    /// `compartment` indexes [`MushroomBody::compartments`] and chooses which
    /// reinforcer this is: the compartment IS the identity of the teaching
    /// signal, which is the whole reason for wiring dopamine per compartment
    /// rather than broadcasting a scalar.
    pub fn train(&mut self, odour: &Odour, other: &Odour, compartment: usize, pairing: Pairing, p: &Protocol) {
        let dans = self.mb.compartments[compartment].dans.clone();
        for _ in 0..p.trials {
            match pairing {
                Pairing::Paired => {
                    self.run(Some(odour), &dans, p, p.trial);
                    self.run(None, &[], p, p.gap);
                }
                Pairing::Unpaired => {
                    // Same odour, same dopamine, never at the same time.
                    self.run(Some(odour), &[], p, p.trial);
                    self.run(None, &dans, p, p.gap);
                }
                Pairing::OdourOnly => {
                    self.run(Some(odour), &[], p, p.trial);
                    self.run(None, &[], p, p.gap);
                }
                Pairing::ReinforcerOnly => {
                    self.run(None, &dans, p, p.trial);
                    self.run(None, &[], p, p.gap);
                }
            }
            // The discrimination: the other odour, always unreinforced.
            self.run(Some(other), &[], p, p.trial);
        }
    }

    pub fn reset(&mut self) {
        self.net.reset(0);
    }
}

fn plast_decay(p: &PlasticityParams) -> f32 {
    // The modulator outlives the spikes that made it, on the same scale as the
    // eligibility trace it multiplies: a dopamine level that cleared faster
    // than the trace could only ever reinforce what was happening in the same
    // few ticks.
    p.elig_decay
}

/// The change one odour's output response underwent, as a signed index.
///
/// `(after - before) / (after + before)`, which is bounded, symmetric, and
/// defined when a response is zero. Reported per odour so that the comparison
/// is between the trained and untrained odour in the SAME animal: an index
/// that moved for both is a drift in excitability, not a memory.
pub fn index(before: f64, after: f64) -> f64 {
    let sum = before + after;
    if sum == 0.0 {
        0.0
    } else {
        (after - before) / sum
    }
}
