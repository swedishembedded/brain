// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded model loading for its clients. If your team needs expertise in
// serving forecasting models within a fixed host memory budget then you can procure our services by sending an email
// to info@swedishembedded.com.

//! Loading a checkpoint streams it onto the device one tensor at a time: the host peak of a load stays within what the
//! build itself holds (on a CPU device the weights and the trainable build's moments, about twice the file), where
//! reading it whole as f32 and cloning the map added twice the file on top.

use horizon::config::HorizonConfig;

fn write(path: &std::path::Path, cfg: &HorizonConfig) -> u64 {
    let mut rng = data::rng::Lcg::random();
    let params: Vec<(String, usize)> = cfg.param_list();
    brain_testutil::load_peak::write_f32_checkpoint(path, &params, &cfg.to_json(), None, &mut |name, v| {
        // Gains near one, everything else small: any weights build, this checks memory.
        v.iter_mut().for_each(|x| *x = if name.contains("norm") || name.contains("ln") { 1.0 } else { rng.scaled(0.02) })
    })
}

#[test]
fn loading_a_checkpoint_holds_no_whole_model_host_copy() {
    brain_testutil::run_alone(module_path!(), "host_peak_of_a_streamed_load");
}

#[test]
#[ignore = "run alone by loading_a_checkpoint_holds_no_whole_model_host_copy"]
fn host_peak_of_a_streamed_load() {
    let dir = brain_testutil::Scratch::dir("horizon-streamed-peak");
    let (path, warm) = (dir.path().join("model.safetensors"), dir.path().join("warm.safetensors"));
    let file = write(&path, &HorizonConfig { d_model: 512, n_layers: 4, n_heads: 8, d_ff: 2048, ..HorizonConfig::tiny(4096, 64) });
    // The device, its driver libraries and kernels first, from a load of the same layout a few KB in size.
    write(&warm, &HorizonConfig::tiny(32, 8));
    drop(horizon::model::Horizon::load(warm.to_str().unwrap(), 1, &horizon::placement::Placement::Cpu).unwrap());
    let growth = brain_testutil::load_peak::host_peak_growth(|| drop(horizon::model::Horizon::load(path.to_str().unwrap(), 1, &horizon::placement::Placement::Cpu).unwrap()));
    println!("file {:.1} MB, host peak growth {:.1} MB", file as f64 / 1e6, growth as f64 / 1e6);
    // On a CPU device the f32 weights and the trainable build's device state (its AdamW moments) are host memory:
    // about twice the file. A whole-model f32 read and its clone add twice the file on top.
    assert!(growth < 2 * file + file / 2, "host peak grew {growth} bytes loading a {file}-byte checkpoint");
}
