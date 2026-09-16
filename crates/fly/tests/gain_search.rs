// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates for the ceiling instrument.
//!
//! A search whose parameterisation is wrong reports a ceiling that is not the
//! ceiling, and a wrong ceiling is worse than none: it would license the
//! conclusion "the structure cannot do this" when what actually happened is
//! that the gains never reached the weights.

use fly::learn::GainSearch;

const NEURONS: &str = "\
Root ID,Flow,Super Class,Class,Sub Class,Nerve,Soma side,Primary Cell Type,Predicted NT type,Predicted NT confidence,Verified NT type
1,afferent,descending,,,,right,DN1,ACH,0.9,
2,intrinsic,intrinsic_neuron,,,,left,IN1,GABA,0.9,
3,efferent,motor,fl,MN-LegNpT1-Ti_flexor,ProLN_L,left,MN1,ACH,0.9,
";
const EDGES: &str = "\
pre_root_id,post_root_id,neuropil,syn_count,nt_type
1,3,X,10,
2,3,X,4,
1,2,X,6,
";

fn fixture() -> connectome::Connectome {
    connectome::load_readers("t", NEURONS.as_bytes(), EDGES.as_bytes()).unwrap()
}

#[test]
fn unit_gains_reproduce_the_connectome_exactly() {
    let c = fixture();
    let s = GainSearch::new(&c, 2.0);
    // The search must start from the imported connectome and nothing else. A
    // parameterisation whose neutral point is not the real graph makes every
    // comparison against the local rule meaningless, because the two would not
    // start from the same animal.
    assert_eq!(s.weights(&s.unit_gains()), c.signed_csc(2.0).w);
}

#[test]
fn a_zero_gain_silences_exactly_one_cell_type() {
    let c = fixture();
    let s = GainSearch::new(&c, 1.0);
    let groups = s.groups().to_vec();
    let desc = groups.iter().position(|g| g == "descending").expect("descending group");
    let intr = groups.iter().position(|g| g == "intrinsic_neuron").expect("intrinsic group");

    let mut gains = s.unit_gains();
    gains[desc] = 0.0;
    let w = s.weights(&gains);
    let base = c.signed_csc(1.0).w;

    // Edges FROM the descending neuron are silenced; every other edge is
    // untouched. A gain applied by postsynaptic class instead of presynaptic
    // would silence the wrong set and still look plausible.
    let mut silenced = 0;
    let mut untouched = 0;
    for (k, (a, b)) in base.iter().zip(&w).enumerate() {
        let pre = c.csc.pre[k] as usize;
        if c.neurons[pre].super_class == "descending" {
            assert_eq!(*b, 0.0, "edge {k} from a descending neuron should be silenced");
            silenced += 1;
        } else {
            assert_eq!(a, b, "edge {k} should be untouched");
            untouched += 1;
        }
    }
    assert!(silenced > 0 && untouched > 0, "the fixture must exercise both sides: {silenced}/{untouched}");

    // And a gain of 2 on a different group doubles only that group.
    let mut gains = s.unit_gains();
    gains[intr] = 2.0;
    let w2 = s.weights(&gains);
    for (k, (a, b)) in base.iter().zip(&w2).enumerate() {
        let pre = c.csc.pre[k] as usize;
        let want = if c.neurons[pre].super_class == "intrinsic_neuron" { a * 2.0 } else { *a };
        assert_eq!(*b, want, "edge {k}");
    }
}

#[test]
fn every_neuron_belongs_to_exactly_one_group() {
    let c = fixture();
    let s = GainSearch::new(&c, 1.0);
    // A neuron missing from the partition would have its edges fall through to
    // the default gain and be invisible to the search: the optimiser would be
    // unable to reach part of the graph and would report a ceiling that is too
    // low, which is the failure mode this instrument must not have.
    let named: std::collections::BTreeSet<&str> =
        c.neurons.iter().map(|n| if n.super_class.is_empty() { "<none>" } else { n.super_class.as_str() }).collect();
    let groups: std::collections::BTreeSet<&str> = s.groups().iter().map(|g| g.as_str()).collect();
    assert_eq!(named, groups, "the gain partition must cover exactly the super classes present");

    // Scaling every group by the same factor must scale every weight, with no
    // edge left behind.
    let k = 3.0;
    let all = s.weights(&vec![k; s.groups().len()]);
    let base = c.signed_csc(1.0).w;
    for (a, b) in base.iter().zip(&all) {
        assert_eq!(*b, a * k);
    }
}
