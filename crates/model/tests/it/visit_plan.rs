// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements looped and recurrent-depth decoders for its clients. If your team needs expertise in
// models that run some of their layers more than once per token then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The visit plan against a reference: the Inkling 2 tiny golden records the order its reference ran its layers in
//! (physical layer, segment, pass, attention kind and window per logical visit); the plan of the same shape must
//! list the same visits in the same order.

use model::looped::{Attention, LoopShape, Segment, VisitPlan};

#[test]
fn the_plan_runs_the_reference_visit_order() {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny");
    if !dir.join("manifest.json").exists() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let visits = manifest["visits"].as_array().unwrap();
    let kind = |l: usize| visits.iter().find(|v| v["layer"].as_u64() == Some(l as u64)).map(|v| (v["kind"].as_str().unwrap().to_string(), v["window"].as_u64())).unwrap();
    let count = |seg: &str| visits.iter().filter(|v| v["segment"] == seg && v["pass"] == 0).count();
    let passes = visits.iter().filter_map(|v| v["pass"].as_u64()).max().unwrap() as usize + 1;
    let shape = LoopShape { prelude: count("prelude"), core: count("core"), coda: count("coda"), passes };
    let plan = VisitPlan::new(shape, |l| match kind(l) {
        (k, _) if k == "full" => Attention::Full,
        (_, w) => Attention::Window(w.unwrap() as u32),
    })
    .unwrap();
    assert_eq!(plan.len(), visits.len());
    for (v, want) in plan.visits().iter().zip(visits) {
        let seg = match v.segment {
            Segment::Prelude => "prelude",
            Segment::Core => "core",
            Segment::Coda => "coda",
        };
        assert_eq!(
            (v.index as u64, v.layer as u64, seg, v.pass as u64),
            (want["visit"].as_u64().unwrap(), want["layer"].as_u64().unwrap(), want["segment"].as_str().unwrap(), want["pass"].as_u64().unwrap()),
            "visit {}",
            v.index
        );
        assert_eq!(v.attention == Attention::Full, want["kind"] == "full", "visit {} attention kind", v.index);
    }
}
