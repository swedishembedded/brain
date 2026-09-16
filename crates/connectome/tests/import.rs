// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import gates.
//!
//! The checkpoint-free tests run everywhere and cover the properties that make
//! a published figure reproducible: every row accounted for, neuropil rows
//! aggregated into one edge per pair, and nothing silently dropped. The
//! real-dataset test reproduces the statistics this repo's roadmap quotes, and
//! skips cleanly when the data is not on the machine.

use connectome::{load, load_readers, Nt};

const NEURONS: &str = "\
Root ID,Flow,Super Class,Class,Nerve,Soma side,Primary Cell Type,Predicted NT type,Predicted NT confidence,Verified NT type
1,afferent,descending,,,right,DNp01,ACH,0.51,
2,intrinsic,ventral_nerve_cord_intrinsic,,,left,IN01,GABA,0.93,
3,efferent,motor,leg_motor_neuron,left_prothoracic_leg_nerve,left,MN01,GLUT,0.40,ACH
4,intrinsic,central_brain_intrinsic,\"a,b\",,right,\"CB,01\",,,
";

/// Pair (1 -> 3) is split across two neuropils, which is exactly the shape the
/// published file has and the thing a naive reader gets wrong.
const EDGES: &str = "\
pre_root_id,post_root_id,neuropil,syn_count,nt_type
1,3,LegNp_T1,7,ACH
1,3,IntTct,5,ACH
2,3,LegNp_T1,4,GABA
1,2,CV,9,ACH
";

fn fixture() -> connectome::Connectome {
    load_readers("test", NEURONS.as_bytes(), EDGES.as_bytes()).expect("fixture loads")
}

#[test]
fn every_row_is_accounted_for() {
    let c = fixture();
    let cov = &c.coverage;
    assert_eq!(cov.neuron_rows, 4);
    assert_eq!(cov.neurons_kept, 4);
    assert_eq!(cov.edge_rows, 4);
    assert_eq!(cov.edge_rows_kept, 4);
    assert_eq!(cov.synapses, 25);
    // `load` calls this itself, so an unbalanced import cannot be returned;
    // asserting it here pins the arithmetic rather than the call.
    cov.check().expect("coverage balances");
}

#[test]
fn neuropil_rows_aggregate_into_one_edge_per_pair() {
    let c = fixture();
    // Four rows, three distinct pairs: (1->3) appeared twice.
    assert_eq!(c.coverage.edges, 3, "rows were not aggregated across neuropils");
    assert_eq!(c.csc.nnz(), 3);

    let (i1, i3) = (c.index_of(1).unwrap(), c.index_of(3).unwrap());
    let (lo, hi) = (c.csc.indptr[i3 as usize] as usize, c.csc.indptr[i3 as usize + 1] as usize);
    let w = (lo..hi).find(|&k| c.csc.pre[k] == i1).map(|k| c.csc.w[k]);
    assert_eq!(w, Some(12.0), "7 + 5 synapses across two neuropils should be one edge of 12");
}

#[test]
fn a_verified_transmitter_outranks_a_prediction_and_carries_full_strength() {
    let c = fixture();
    let mn = &c.neurons[c.index_of(3).unwrap() as usize];
    // Row 3 predicts GLUT at 0.40 but is VERIFIED as ACH.
    assert_eq!(mn.nt.predicted, Some(Nt::Glutamate));
    assert_eq!(mn.nt.verified, Some(Nt::Acetylcholine));
    assert_eq!(mn.nt.best(), Some(Nt::Acetylcholine));
    assert_eq!(mn.nt.sign(), 1.0, "the verified transmitter decides the sign");
    assert_eq!(mn.nt.strength(), 1.0, "a verified transmitter is not a prior");

    let inh = &c.neurons[c.index_of(2).unwrap() as usize];
    assert_eq!(inh.nt.sign(), -1.0);
    assert!((inh.nt.strength() - 0.93).abs() < 1e-6, "a prediction is worth its own confidence");

    let none = &c.neurons[c.index_of(4).unwrap() as usize];
    assert_eq!(none.nt.sign(), 0.0);
    assert_eq!(none.nt.strength(), 0.0, "an unknown transmitter must not imply a sign");
}

#[test]
fn glutamate_is_inhibitory_in_the_fly() {
    // The single easiest sign to get wrong: glutamate excites in vertebrates
    // and mostly inhibits in Drosophila, through GluCl.
    assert_eq!(Nt::Glutamate.conventional_sign(), -1.0);
    assert_eq!(Nt::Acetylcholine.conventional_sign(), 1.0);
    assert_eq!(Nt::Gaba.conventional_sign(), -1.0);
    assert_eq!(Nt::Histamine.conventional_sign(), -1.0);
}

#[test]
fn quoted_annotation_fields_do_not_shift_the_columns() {
    let c = fixture();
    // Row 4's Class is "a,b" and its cell type is "CB,01" - if the reader
    // split on commas, both would shift every later column and the neuron
    // would appear to have a transmitter it does not have.
    let n = &c.neurons[c.index_of(4).unwrap() as usize];
    assert_eq!(n.class, "a,b");
    assert_eq!(n.cell_type, "CB,01");
    assert_eq!(n.nt.best(), None);
}

#[test]
fn a_population_is_selected_by_annotation() {
    let c = fixture();
    let legs = c.population(|n| n.class == "leg_motor_neuron");
    assert_eq!(legs, vec![c.index_of(3).unwrap()]);
    let left_pro = c.population(|n| n.nerve == "left_prothoracic_leg_nerve");
    assert_eq!(left_pro.len(), 1, "the nerve is the handle a body attaches to");
}

#[test]
fn malformed_rows_are_rejected_with_a_reason_never_dropped() {
    let neurons = format!("{NEURONS}5,intrinsic\n1,intrinsic,dup,,,,,,,\nnotanid,intrinsic,x,,,,,,,\n");
    let edges = format!("{EDGES}1,999,X,3,\n1,3,X,notanumber,\n");
    let c = load_readers("test", neurons.as_bytes(), edges.as_bytes()).unwrap();

    assert_eq!(c.coverage.neuron_rows, 7);
    assert_eq!(c.coverage.neurons_kept, 4);
    assert_eq!(c.coverage.neuron_rejects.get("wrong field count"), Some(&1));
    assert_eq!(c.coverage.neuron_rejects.get("duplicate Root ID"), Some(&1));
    assert_eq!(c.coverage.neuron_rejects.get("unparseable Root ID"), Some(&1));

    assert_eq!(c.coverage.edge_rows, 6);
    assert_eq!(c.coverage.edge_rejects.get("endpoint has no neuron row"), Some(&1));
    assert_eq!(c.coverage.edge_rejects.get("unparseable syn_count"), Some(&1));
    c.coverage.check().expect("rejects still balance");
}

#[test]
fn a_missing_required_column_is_an_error_naming_the_header() {
    let err = load_readers("test", "Wrong,Header\n1,2\n".as_bytes(), EDGES.as_bytes()).unwrap_err();
    assert!(err.contains("Root ID"), "the error should name the column it wanted: {err}");
    assert!(err.contains("Wrong"), "and show the header it actually got: {err}");
}

/// Reproduce the published statistics on the real datasets.
///
/// Skips when `$BRAIN_CONNECTOME_DIR` is unset or the files are absent: the
/// exports are hundreds of megabytes and are not in this repo. Set it to the
/// directory holding `banc-codex/` and `manc-codex/`.
#[test]
fn real_datasets_reproduce_their_published_statistics() {
    let Some(root) = std::env::var_os("BRAIN_CONNECTOME_DIR").filter(|v| !v.is_empty()) else {
        brain_testutil::skip("BRAIN_CONNECTOME_DIR unset (Codex exports are not in this repo)");
        return;
    };
    let root = std::path::PathBuf::from(root);

    // (dataset, neurons, edges, synapses, motor, descending)
    let expect = [
        ("banc-codex", 158_262usize, 3_037_361usize, 23_556_214u64, 805usize, 1_316usize),
        ("manc-codex", 23_665, 5_305_638, 30_934_610, 737, 1_328),
    ];
    for (name, neurons, edges, synapses, motor, descending) in expect {
        let dir = root.join(name);
        if !dir.join("neurons.csv.gz").is_file() {
            brain_testutil::skip(&format!("{name}: not present under BRAIN_CONNECTOME_DIR"));
            continue;
        }
        let c = load(name, &dir.join("neurons.csv.gz"), &dir.join("connections_princeton.csv.gz"))
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        assert_eq!(c.coverage.neurons_kept, neurons, "{name}: neuron count");
        assert_eq!(c.coverage.edges, edges, "{name}: aggregated edge count");
        assert_eq!(c.coverage.synapses, synapses, "{name}: synapse total");
        assert_eq!(c.population(|n| n.super_class == "motor").len(), motor, "{name}: motor neurons");
        assert_eq!(c.population(|n| n.super_class == "descending").len(), descending, "{name}: descending neurons");
        assert_eq!(c.coverage.neuron_rejects.len(), 0, "{name}: {:?}", c.coverage.neuron_rejects);

        let (mean, median, p99, max) = c.in_degree_stats();
        eprintln!("{name}: {}", c.coverage.summary());
        eprintln!("  in-degree mean {mean:.1} median {median} p99 {p99} max {max}");
        eprintln!("  CSR {:.1} MB", c.csc.nnz() as f64 * 8.0 / 1e6);
    }
}
