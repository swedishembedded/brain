// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Does the fly land, and which way is forward?
use mujoco::{Data, Model, MuJoCo, StateSpec};
fn main() {
    let mj = MuJoCo::load().unwrap();
    let m = Model::from_xml(&mj, std::env::var("BRAIN_FLYBODY_XML").unwrap()).unwrap();
    let mut d = Data::new(&m).unwrap();
    println!("nq={} nv={} nu={}", m.nq(), m.nv(), m.nu());
    for k in 0..12 {
        for _ in 0..500 { d.step(&m); }
        let q = d.get(&m, StateSpec::QPOS);
        let v = d.get(&m, StateSpec::QVEL);
        println!("t={:5.3}s  root xyz [{:+.4} {:+.4} {:+.4}]  vel [{:+.4} {:+.4} {:+.4}]",
            d.time(&m), q[0], q[1], q[2], v[0], v[1], v[2]);
        if k == 11 {
            let settled = v[..3].iter().all(|x| x.abs() < 0.5);
            println!("\nsettled: {settled}  (root height {:+.4})", q[2]);
        }
    }
}
