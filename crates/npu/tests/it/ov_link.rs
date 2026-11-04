// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Link/visibility probe: only runs when BRAIN_OV_PROBE is set.
#[test]
fn lists_devices() {
    if std::env::var("BRAIN_OV_PROBE").is_err() { return; }
    match npu::openvino::available_devices() {
        Ok(d) => { println!("OV_DEVICES {d:?}"); assert!(!d.is_empty()); }
        Err(e) => panic!("openvino unavailable: {e:?}"),
    }
}
