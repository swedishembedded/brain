// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reference dump of the released Qwen3.8-27B reading one picture (`tools/goldens/qwen35_vl_dump_reference.py`),
//! shared by the real-weight tests that compare against it.

use std::collections::HashMap;
use std::path::Path;

pub fn golden() -> Option<HashMap<String, Vec<f32>>> {
    let path = brain_testutil::testdata("golden/qwen35/vl/qwen35_vl.safetensors");
    if !Path::new(&path).exists() {
        brain_testutil::skip(&format!("fixture {path} absent - run tools/goldens/qwen35_vl_dump_reference.py"));
        return None;
    }
    Some(checkpoint::safetensors::read(&path).expect("read the golden").into_iter().map(|t| (t.name, t.data)).collect())
}

pub fn ints(v: &[f32]) -> Vec<u32> {
    v.iter().map(|&x| x.round() as u32).collect()
}
