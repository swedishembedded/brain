// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Joining a brain to a cord.
//!
//! The synthetic fixtures are four neurons a side, because the property being
//! checked is structural and a real dataset would only make it slower to read:
//! a crossing cell has to end up as ONE neuron carrying the brain's inputs and
//! the cord's outputs, and the cord it came from must not be counted twice.
//! The real datasets are checked at the bottom, against numbers measured
//! through the same loader.

use connectome::bridge::{is_brain, join, read_bridge_reader, Crossing, Policy};
use connectome::load_readers;

/// A brain: two intrinsic cells and a descending one they drive.
const BRAIN_NEURONS: &str = "\
Root ID,Top in/out region,Predicted NT type,Predicted NT confidence,Flow,Super Class,Class,Sub Class,Nerve,Soma side,Primary Cell Type,Volume (nm^3)
100,central_brain,ACH,0.9,intrinsic,central_brain_intrinsic,,,,left,CB1,10
101,optic_lobe,ACH,0.9,intrinsic,optic_lobe_intrinsic,,,,left,OL1,10
102,central_brain,ACH,0.9,efferent,descending,,,,left,DNx01,10
103,ventral_nerve_cord,ACH,0.9,intrinsic,ventral_nerve_cord_intrinsic,,,,left,VNC1,10
";

/// The brain's own cord copy (103) is what must NOT survive the join.
const BRAIN_EDGES: &str = "\
pre_root_id,post_root_id,neuropil,syn_count,nt_type
100,102,CB,7,ACH
101,100,OL,5,ACH
102,103,VNC,11,ACH
";

/// A cord: the same descending cell, an interneuron, a motor neuron.
const CORD_NEURONS: &str = "\
Root ID,Top in/out region,Predicted NT type,Predicted NT confidence,Flow,Super Class,Class,Sub Class,Nerve,Soma side,Primary Cell Type,Volume (nm^3)
200,LEGNP_T1,ACH,0.9,efferent,descending,,,,left,DNx01,10
201,LEGNP_T1,ACH,0.9,intrinsic,intrinsic,,,,left,IN1,10
202,LEGNP_T1,ACH,0.9,efferent,motor,fl,MN-Ti_flexor,left_pro,left,MN1,10
";

const CORD_EDGES: &str = "\
pre_root_id,post_root_id,neuropil,syn_count,nt_type
200,201,LegNp_T1,13,ACH
201,202,LegNp_T1,3,ACH
";

const BRIDGE: &str = "\
banc_root_id,manc_root_id,flow,banc_cell_type,manc_cell_type
102,200,descending,DNx01,DNx01
";

fn parts() -> (connectome::Connectome, connectome::Connectome, Vec<Crossing>) {
    let brain = load_readers("brain", BRAIN_NEURONS.as_bytes(), BRAIN_EDGES.as_bytes()).expect("brain loads");
    let cord = load_readers("cord", CORD_NEURONS.as_bytes(), CORD_EDGES.as_bytes()).expect("cord loads");
    let bridge = read_bridge_reader(BRIDGE.as_bytes()).expect("the bridge parses");
    (brain, cord, bridge)
}

#[test]
fn a_crossing_neuron_becomes_one_cell_carrying_both_halves() {
    let (brain, cord, bridge) = parts();
    let (cns, report) = join(&brain, &cord, &bridge, Policy::default(), is_brain).expect("the join succeeds");

    assert_eq!(report.merged, 1, "the descending cell should have been merged");
    assert_eq!((report.brain_missing, report.cord_missing), (0, 0));

    // brain 100, 101, 102 (the DN) + cord 201, 202. The brain's own cord copy
    // (103) is dropped and the cord's copy of the DN (200) is not a second row.
    assert_eq!(cns.neurons.len(), 5, "{:?}", cns.neurons.iter().map(|n| n.root_id).collect::<Vec<_>>());
    assert_eq!(cns.population(|n| n.super_class == "descending").len(), 1, "the DN is one cell, not two");
    assert_eq!(cns.population(|n| n.super_class == "motor").len(), 1);
    assert!(cns.index_of(103).is_none(), "the brain dataset's own cord neuron survived the join");
    assert!(cns.index_of(200).is_none(), "the cord's copy of the crossing cell is a second row");

    // The merged cell has the BRAIN's input and the CORD's output: that is the
    // whole point, and either one alone is a plausible-looking half-join.
    let dn = cns.index_of(102).expect("the merged cell keeps the brain's id");
    let inputs: Vec<u64> = incoming(&cns, dn).iter().map(|&i| cns.neurons[i as usize].root_id).collect();
    assert_eq!(inputs, vec![100], "the merged cell lost its brain input");
    let targets: Vec<u64> = outgoing(&cns, dn).iter().map(|&i| cns.neurons[i as usize].root_id).collect();
    assert_eq!(targets, vec![201], "the merged cell lost its cord output");
}

#[test]
fn the_brain_datasets_own_cord_is_dropped_rather_than_counted_twice() {
    let (brain, cord, bridge) = parts();
    let (cns, report) = join(&brain, &cord, &bridge, Policy::default(), is_brain).expect("the join succeeds");
    // brain: 100->102 and 101->100 survive, 102->103 dies with its endpoint.
    // cord:  both survive. So four edges and one dropped.
    assert_eq!(report.edges, 4);
    assert_eq!(report.edges_dropped, 1, "the brain's cord-side edge was kept");
    assert_eq!(cns.coverage.synapses, 7 + 5 + 13 + 3);
}

/// A bridge nobody can match produces a perfectly valid graph in two
/// disconnected halves, and every downstream measurement still runs. So the
/// report says so and a caller can refuse it.
#[test]
fn a_bridge_that_matches_nothing_is_reported_rather_than_silently_empty() {
    let (brain, cord, _) = parts();
    let bogus = read_bridge_reader("banc_root_id,manc_root_id,flow\n999,998,descending\n".as_bytes()).unwrap();
    let (cns, report) = join(&brain, &cord, &bogus, Policy::default(), is_brain).expect("the join still succeeds");
    assert_eq!(report.merged, 0);
    assert_eq!((report.brain_missing, report.cord_missing), (1, 0));
    assert_eq!(cns.population(|n| n.super_class == "descending").len(), 2, "two unmerged copies of one cell");
}

/// A cord cell claimed by several brain cells is what the REAL data looks
/// like - 391 of MANC's cells, up to eight claimants each - and it is not an
/// error. Exactly one of them becomes the same cell; the others keep their own
/// identity and are wired to it, so no pathway is lost and no two distinct
/// brain neurons are fused.
#[test]
fn a_cord_cell_claimed_twice_merges_one_and_wires_the_other() {
    let (brain, cord, _) = parts();
    // 102 carries the morphology match, so it is the member that becomes the
    // same cell; 100 does not and is the one wired to it. Without that column
    // the tie would fall to the lower id, which the next test pins.
    let two = read_bridge_reader(
        "banc_root_id,manc_root_id,flow,manc_nblast_match\n102,200,descending,200\n100,200,descending,\n".as_bytes(),
    )
    .unwrap();
    let (cns, report) = join(&brain, &cord, &two, Policy::default(), is_brain).expect("ambiguity is not an error");
    assert_eq!((report.merged, report.linked), (1, 1));
    // Both brain cells still exist, and both reach the cord's interneuron -
    // one by being the descending cell, one through the axonal link.
    let a = cns.index_of(100).expect("the runner-up keeps its own row");
    let b = cns.index_of(102).expect("the merged cell keeps the brain's id");
    assert_ne!(a, b, "two distinct brain neurons were fused into one");
    let merged_targets = outgoing(&cns, b);
    assert!(merged_targets.contains(&cns.index_of(201).unwrap()), "the merged cell lost the cord");
    assert!(outgoing(&cns, a).contains(&b), "the runner-up was not wired to the cell it corresponds to");
}

/// The same, with the link turned off: the control for whether those
/// pathways matter at all.
#[test]
fn an_ambiguous_crossing_can_be_dropped_instead_of_wired() {
    let (brain, cord, _) = parts();
    // 101 is the optic-lobe cell, which has no edge of its own to 102 - so
    // whether it reaches the cord is entirely the link's doing.
    let two = read_bridge_reader(
        "banc_root_id,manc_root_id,flow,manc_nblast_match\n102,200,descending,200\n101,200,descending,\n".as_bytes(),
    )
    .unwrap();
    let on = join(&brain, &cord, &two, Policy::default(), is_brain).expect("the join succeeds").0;
    let policy = Policy { ambiguous_axon_synapses: 0 };
    let (off, report) = join(&brain, &cord, &two, policy, is_brain).expect("the join succeeds");
    assert_eq!((report.merged, report.linked), (1, 1), "it is still REPORTED as a crossing");
    assert!(
        outgoing(&on, on.index_of(101).unwrap()).contains(&on.index_of(102).unwrap()),
        "with the link on, the runner-up should reach the cell it corresponds to"
    );
    assert!(
        !outgoing(&off, off.index_of(101).unwrap()).contains(&off.index_of(102).unwrap()),
        "the link was supposed to be off"
    );
}

/// With nothing to choose between them, the lower id wins - so the joined
/// graph does not depend on the order of rows in a CSV.
#[test]
fn an_unbroken_tie_is_settled_deterministically_by_id() {
    let (brain, cord, _) = parts();
    let two = read_bridge_reader(
        "banc_root_id,manc_root_id,flow\n102,200,descending\n100,200,descending\n".as_bytes(),
    )
    .unwrap();
    let reversed: Vec<Crossing> = two.iter().rev().cloned().collect();
    for rows in [two.as_slice(), reversed.as_slice()] {
        let (cns, _) = join(&brain, &cord, rows, Policy::default(), is_brain).expect("the join succeeds");
        let merged = cns.index_of(100).expect("the lower id is the one that merged");
        assert!(outgoing(&cns, merged).contains(&cns.index_of(201).unwrap()), "100 should carry the cord");
    }
}

#[test]
fn a_brain_neuron_matched_to_two_cord_cells_is_an_error() {
    let (brain, cord, _) = parts();
    let bad = read_bridge_reader(
        "banc_root_id,manc_root_id,flow\n102,200,descending\n102,201,descending\n".as_bytes(),
    )
    .unwrap();
    let e = join(&brain, &cord, &bad, Policy::default(), is_brain).expect_err("a fan-out match is corrupt");
    assert!(e.contains("two different"), "{e}");
}

#[test]
fn a_bridge_row_without_two_ids_is_an_error_naming_the_row() {
    let e = read_bridge_reader("banc_root_id,manc_root_id,flow\n102,,descending\n".as_bytes())
        .expect_err("half a pair is not a crossing");
    assert!(e.contains("row 2"), "{e}");
}

fn incoming(c: &connectome::Connectome, post: u32) -> Vec<u32> {
    let (a, b) = (c.csc.indptr[post as usize] as usize, c.csc.indptr[post as usize + 1] as usize);
    c.csc.pre[a..b].to_vec()
}

fn outgoing(c: &connectome::Connectome, pre: u32) -> Vec<u32> {
    (0..c.csc.n).filter(|&post| incoming(c, post).contains(&pre)).collect()
}

/// The real thing, when it is on the machine.
#[test]
fn banc_and_manc_join_into_one_nervous_system() {
    let Some(root) = std::env::var_os("BRAIN_CONNECTOME_DIR").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_CONNECTOME_DIR unset");
        return;
    };
    let root = std::path::PathBuf::from(root);
    let bridge_path = root.join("banc").join("bridge_manc.csv.gz");
    if !bridge_path.is_file() {
        brain_testutil::skip("no BANC export with a bridge under BRAIN_CONNECTOME_DIR");
        return;
    }
    let load = |name: &str| {
        let (n, e) = connectome::find(&root, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        connectome::load(name, &n, &e).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let brain = load("banc");
    let cord = load("manc");
    let bridge = connectome::read_bridge(&bridge_path).expect("the bridge reads");
    let (cns, report) = join(&brain, &cord, &bridge, Policy::default(), is_brain).expect("the join succeeds");
    eprintln!("{}", report.summary());

    // Measured through this same loader. The count that matters is `merged`:
    // it is how many of the published cross-dataset identities actually
    // resolved on BOTH sides, and a change in it means one of the two exports
    // moved under the bridge.
    assert_eq!(report.merged + report.linked, 3_530, "every published crossing should resolve");
    assert_eq!(report.merged, 2_954, "one merged cell per distinct cord match");
    assert_eq!(report.linked, 576, "the runners-up in the 391 ambiguous groups, wired rather than fused");
    assert_eq!((report.brain_missing, report.cord_missing), (0, 0));

    // Descending neurons after the join: BANC's 1,316 (each now carrying its
    // cord half) plus the 255 MANC descending cells that have no counterpart
    // in BANC at all. Those are kept rather than dropped - they are real cells
    // with real cord connectivity, they simply have no brain half in this
    // reconstruction and therefore no brain input. The number to watch is that
    // it is well below the 2,644 a plain concatenation would give.
    let dns = cns.population(|n| n.super_class == "descending").len();
    assert_eq!(dns, 1_571, "descending cells after merging");
    assert!(dns < 1_316 + 1_328, "the two descending populations were concatenated, not identified");

    // Every motor neuron is still there and still annotated - the body
    // attaches to those by name. 699 of BANC's 805 are in its nerve cord,
    // which is the half that was dropped, so what survives is MANC's 737 plus
    // the 106 in BANC's brain: proboscis, neck, pharynx, antenna, crop, eye
    // and salivary muscles, which a nerve cord does not contain and which the
    // flybody model has no actuators for either.
    assert_eq!(cns.population(|n| n.super_class == "motor").len(), 737 + 106, "motor neurons");

    // And the halves are actually connected: a path exists from the optic lobe
    // into the cord. Checked as REACHABILITY rather than as an edge count,
    // because a join that merged nothing still has plenty of edges.
    let reach = reachable_from(&cns, &cns.population(|n| n.super_class == "optic_lobe_intrinsic"));
    let motor: Vec<u32> = cns.population(|n| n.super_class == "motor");
    let reached = motor.iter().filter(|m| reach[**m as usize]).count();
    assert!(
        reached * 2 > motor.len(),
        "only {reached} of {} motor neurons are reachable from the optic lobe; the brain and the cord are not joined",
        motor.len()
    );
}

/// Forward reachability over the joined graph, from a seed set.
fn reachable_from(c: &connectome::Connectome, seed: &[u32]) -> Vec<bool> {
    // The graph is stored by POSTsynaptic neuron, so walking forwards needs
    // the transpose built once rather than a scan per step.
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); c.csc.n as usize];
    for post in 0..c.csc.n as usize {
        let (a, b) = (c.csc.indptr[post] as usize, c.csc.indptr[post + 1] as usize);
        for k in a..b {
            out[c.csc.pre[k] as usize].push(post as u32);
        }
    }
    let mut seen = vec![false; c.csc.n as usize];
    let mut stack: Vec<u32> = seed.to_vec();
    for &s in seed {
        seen[s as usize] = true;
    }
    while let Some(i) = stack.pop() {
        for &j in &out[i as usize] {
            if !seen[j as usize] {
                seen[j as usize] = true;
                stack.push(j);
            }
        }
    }
    seen
}
