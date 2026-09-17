// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates for locating the fly's learning organ.
//!
//! The synthetic tests fix the rules: which class labels name which
//! population, that a compartment needs evidence to exist, and that the site
//! map is built against the graph a creature will actually run. The
//! real-dataset test is the one that matters scientifically, and it does not
//! check a count - it checks that the compartments derived from the graph
//! REPRODUCE the published mushroom-body compartment map. If they do, the
//! derivation is sound and no hand-curated table is needed. It skips cleanly
//! when BANC is not on the machine.

use connectome::mushroom_body::{MushroomBody, Policy};
use connectome::{load, load_readers};

/// KC 1 and 2 -> MBON 3 and 4; PPL101 (5) reinforces MBON 3 only; PAM01 (6)
/// makes a single stray synapse onto MBON 4, which is below the evidence bar.
const NEURONS: &str = "\
Root ID,Flow,Super Class,Class,Nerve,Soma side,Primary Cell Type,Predicted NT type,Predicted NT confidence,Verified NT type
1,intrinsic,central_brain_intrinsic,kenyon_cell,,right,KCg,ACH,0.99,
2,intrinsic,central_brain_intrinsic,kenyon_cell,,left,KCab,ACH,0.99,
3,intrinsic,central_brain_intrinsic,mushroom_body_output_neuron,,right,MBON11,GABA,0.95,
4,intrinsic,central_brain_intrinsic,mushroom_body_output_neuron,,left,MBON01,GLUT,0.95,
5,intrinsic,central_brain_intrinsic,mushroom_body_dopaminergic_neuron,,right,PPL101,DA,0.97,
6,intrinsic,central_brain_intrinsic,mushroom_body_dopaminergic_neuron,,left,PAM01,ACH,0.30,
7,intrinsic,central_brain_intrinsic,,,right,CB0001,ACH,0.90,
";

const EDGES: &str = "\
pre_root_id,post_root_id,neuropil,syn_count,nt_type
1,3,MB_CA,20,ACH
2,3,MB_CA,14,ACH
1,4,MB_CA,11,ACH
5,3,MB_ML,40,DA
6,4,MB_ML,1,DA
7,3,MB_ML,30,ACH
1,7,MB_CA,9,ACH
";

fn fixture() -> connectome::Connectome {
    load_readers("test", NEURONS.as_bytes(), EDGES.as_bytes()).expect("fixture loads")
}

#[test]
fn the_three_populations_come_from_the_datasets_own_class_labels() {
    let c = fixture();
    let mb = MushroomBody::find(&c, Policy::default());
    assert_eq!(mb.kc.len(), 2, "two Kenyon cells");
    assert_eq!(mb.mbon.len(), 2, "two output neurons");
    assert_eq!(mb.dan.len(), 2, "two dopaminergic neurons");
}

/// A compartment is an output neuron plus the cells that demonstrably
/// reinforce it. One synapse is not a demonstration.
#[test]
fn a_compartment_needs_evidence_and_a_single_synapse_is_not_evidence() {
    let c = fixture();
    let mb = MushroomBody::find(&c, Policy::default());
    assert_eq!(mb.compartments.len(), 1, "only MBON11 has dopaminergic input above the bar");
    let comp = &mb.compartments[0];
    assert_eq!(comp.name, "MBON11");
    assert_eq!(comp.dans, vec![4], "PPL101 is neuron index 4, and the cholinergic input to MBON11 is not a DAN");
    assert_eq!(comp.dan_synapses, 40);

    // Drop the bar and the stray synapse creates a second compartment, which
    // is the failure mode the bar exists to prevent.
    let loose = MushroomBody::find(&c, Policy { min_dan_synapses: 1, ..Policy::default() });
    assert_eq!(loose.compartments.len(), 2);

    // Requiring the transmitter prediction to agree drops PAM01, whose
    // prediction here says acetylcholine at low confidence.
    let strict = MushroomBody::find(&c, Policy { require_dopamine: true, min_dan_synapses: 1 });
    assert_eq!(strict.dan.len(), 1);
    assert_eq!(strict.compartments.len(), 1);
}

/// The site map marks Kenyon-cell synapses and nothing else, and it is built
/// against the graph the creature runs rather than against the import.
#[test]
fn only_kenyon_cell_synapses_onto_an_output_neuron_are_made_plastic() {
    let c = fixture();
    let mb = MushroomBody::find(&c, Policy::default());
    let csc = c.network(1.0, None, 1);
    let sites = mb.sites(&csc, -1.0, 0.9);

    // MBON11 receives from KC 1, KC 2, PPL101 and one ordinary cholinergic
    // cell. Only the first two may learn.
    assert_eq!(sites.plastic_edges(), 2, "two Kenyon-cell synapses onto MBON11");
    for (k, &comp) in sites.of_edge().iter().enumerate() {
        let post = (0..csc.n as usize).find(|&p| (csc.indptr[p]..csc.indptr[p + 1]).contains(&(k as u32))).unwrap();
        let pre = csc.pre[k];
        let plastic = comp != neuro::INERT;
        let expected = mb.kc.contains(&pre) && mb.compartments.iter().any(|x| x.mbon as usize == post);
        assert_eq!(plastic, expected, "edge {pre} -> {post} plastic={plastic}, expected {expected}");
    }
    sites.validate(csc.nnz(), csc.n).expect("the map fits the graph it was built against");

    // The dopaminergic synapses are found separately, because they have to
    // leave the fast pathway rather than drive the cells they are teaching.
    // ALL of them do, including PAM01's single synapse: the evidence bar
    // decides what counts as a compartment, not what counts as dopaminergic,
    // and a synapse too weak to define a compartment is still not a fast
    // excitatory one.
    let modulatory = mb.modulatory_edges(&csc);
    let from: Vec<u32> = modulatory.iter().map(|&k| csc.pre[k as usize]).collect();
    assert_eq!(from, vec![4, 5], "PPL101 -> MBON11 and PAM01 -> MBON01");

    // Pruning renumbers every edge. A map built against the import would mark
    // real edges here, just the wrong ones, and would do it silently.
    let pruned = c.network(1.0, None, 16);
    let on_pruned = mb.sites(&pruned, -1.0, 0.9);
    assert_eq!(on_pruned.plastic_edges(), 1, "only the 20-synapse KC -> MBON11 pair survives a floor of 16");
    on_pruned.validate(pruned.nnz(), pruned.n).expect("the map fits the pruned graph");
}

#[test]
fn a_nerve_cord_has_no_mushroom_body_and_that_is_not_an_error() {
    let c = load_readers(
        "cord",
        "Root ID,Flow,Super Class,Class,Nerve,Soma side,Primary Cell Type,Predicted NT type,Predicted NT confidence,Verified NT type\n1,efferent,motor,leg_motor_neuron,left_prothoracic_leg_nerve,left,MN01,ACH,0.9,\n".as_bytes(),
        "pre_root_id,post_root_id,neuropil,syn_count,nt_type\n1,1,LegNp_T1,3,ACH\n".as_bytes(),
    )
    .expect("loads");
    let mb = MushroomBody::find(&c, Policy::default());
    assert!(mb.is_empty());
    assert_eq!(mb.sites(&c.network(1.0, None, 1), -1.0, 0.9).plastic_edges(), 0);
}

/// The scientific gate: compartments derived from the graph must reproduce the
/// published mushroom-body compartment map.
///
/// These pairings are the textbook ones and were not used to build the
/// derivation: PPL101 innervates the gamma-1-pedc compartment, whose output
/// neuron MBON11 drives avoidance, and it is the cell aversive conditioning
/// recruits. PAM01 innervates gamma-5-beta-prime-2a, whose output neuron
/// MBON01 is on the appetitive side. If reading the graph reproduces both,
/// then the compartment structure is in the connectome and does not need a
/// curated table that would go stale with the next release.
#[test]
fn the_derived_compartments_reproduce_the_published_map() {
    let Ok(root) = std::env::var("BRAIN_CONNECTOME_DIR") else {
        eprintln!("skipping: set BRAIN_CONNECTOME_DIR to the directory holding banc/");
        return;
    };
    let Ok((neurons, edges)) = connectome::find(&root, "banc") else {
        eprintln!("skipping: no banc dataset under {root}");
        return;
    };
    let c = load("banc", &neurons, &edges).expect("BANC loads");
    let mb = MushroomBody::find(&c, Policy::default());
    eprintln!("BANC mushroom body: {}", mb.summary());

    assert!(mb.kc.len() > 4000, "BANC reports about 4,550 Kenyon cells, found {}", mb.kc.len());
    assert!(mb.mbon.len() >= 100, "about 104 output neurons, found {}", mb.mbon.len());
    assert!(mb.dan.len() >= 280, "about 293 dopaminergic neurons, found {}", mb.dan.len());

    let dan_types = |name: &str| -> Vec<String> {
        let mut t: Vec<String> = mb
            .compartments
            .iter()
            .filter(|x| x.name == name)
            .flat_map(|x| x.dans.iter().map(|&d| c.neurons[d as usize].cell_type.clone()))
            .collect();
        t.sort();
        t.dedup();
        t
    };
    for (mbon, dan) in [("MBON11", "PPL101"), ("MBON01", "PAM01"), ("MBON12", "PPL103")] {
        let found = dan_types(mbon);
        assert!(
            found.iter().any(|t| t == dan),
            "{mbon}'s compartment should contain {dan}, found {found:?}"
        );
    }

    // And the map has to be sparse: a compartment that recruited every
    // dopaminergic cell in the brain would not be a compartment.
    let widest = mb.compartments.iter().map(|x| x.dans.len()).max().unwrap_or(0);
    assert!(widest < mb.dan.len() / 4, "the widest compartment has {widest} of {} DANs", mb.dan.len());

    let csc = c.network(0.6, Some(10.0), 5);
    let sites = mb.sites(&csc, -1.0, 0.9);
    sites.validate(csc.nnz(), csc.n).expect("the map fits the network");
    let fraction = sites.plastic_edges() as f64 / csc.nnz() as f64;
    eprintln!(
        "plastic: {} of {} edges ({:.3}%), {} compartments",
        sites.plastic_edges(),
        csc.nnz(),
        100.0 * fraction,
        mb.compartments.len()
    );
    assert!(sites.plastic_edges() > 0, "no Kenyon-cell output synapse survived the synapse floor");
    assert!(fraction < 0.01, "plasticity is meant to be confined, got {:.2}% of the graph", 100.0 * fraction);
}

/// The synapse floor must not delete the pathway the learning happens on.
///
/// A Kenyon cell makes one or two synapses onto an output neuron and is
/// supposed to: the odour is in which cells fire together, not in how hard any
/// one of them pushes. A per-pair threshold is the right tool for
/// reconstruction noise in ordinary neuropil and the wrong tool here, and
/// getting it wrong is silent - the cells remain, the compartments remain, and
/// the odour code is gone.
#[test]
fn the_synapse_floor_does_not_delete_the_sparse_kenyon_cell_pathway() {
    let c = fixture();
    let mb = MushroomBody::find(&c, Policy::default());
    // KC 2 -> MBON11 carries 14 synapses, KC 1 -> MBON11 carries 20.
    let floored = mb.sites(&c.network(1.0, None, 16), -1.0, 0.9);
    assert_eq!(floored.plastic_edges(), 1, "the floor removed one of the two Kenyon-cell synapses");

    let exempt = mb.plastic_pairs(&c);
    let kept = c.network_keeping(1.0, None, 16, &exempt);
    let sites = mb.sites(&kept, -1.0, 0.9);
    assert_eq!(sites.plastic_edges(), 2, "the exemption keeps the whole Kenyon-cell pathway");

    // And it exempts that pathway ONLY: the ordinary cholinergic input to
    // MBON11 carries 30 synapses and survives on its own merits, while the
    // 9-synapse KC -> non-MBON edge is still removed.
    let mut survivors: Vec<(u32, u32)> = Vec::new();
    for post in 0..kept.n as usize {
        for k in kept.indptr[post]..kept.indptr[post + 1] {
            survivors.push((kept.pre[k as usize], post as u32));
        }
    }
    assert!(survivors.contains(&(0, 2)), "KC 1 -> MBON11 is exempt");
    assert!(survivors.contains(&(1, 2)), "KC 2 -> MBON11 is exempt");
    assert!(!survivors.contains(&(0, 6)), "KC 1 -> an ordinary cell is not exempt and is below the floor");
}
