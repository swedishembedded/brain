// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded model loading for its clients. If your team needs expertise in
// serving forecasting models within a fixed host memory budget then you can procure our services by sending an email
// to info@swedishembedded.com.

//! Loading a checkpoint streams it onto the device one tensor at a time: the host peak of a load stays within the
//! file (the weights themselves, on a CPU device), where reading it whole as f32 and cloning the map took about three
//! times the file.

use chronos2::config::Chronos2Config;

fn write(path: &std::path::Path, cfg: &Chronos2Config) -> u64 {
    let mut rng = data::rng::Lcg::random();
    let params: Vec<(String, usize)> = cfg.param_list().into_iter().map(|(n, s)| (n, s.iter().product())).collect();
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
    let dir = brain_testutil::Scratch::dir("chronos2-streamed-peak");
    let (path, warm) = (dir.path().join("model.safetensors"), dir.path().join("warm.safetensors"));
    let file = write(&path, &Chronos2Config { d_model: 512, d_kv: 64, num_heads: 8, d_ff: 2048, num_layers: 4, ..Chronos2Config::tiny() });
    // The device, its driver libraries and kernels first, from a load of the same layout a few KB in size.
    write(&warm, &Chronos2Config::tiny());
    drop(chronos2::model::Chronos2::load(warm.to_str().unwrap()).unwrap());
    let growth = brain_testutil::load_peak::host_peak_growth(|| drop(chronos2::model::Chronos2::load(path.to_str().unwrap()).unwrap()));
    println!("file {:.1} MB, host peak growth {:.1} MB", file as f64 / 1e6, growth as f64 / 1e6);
    // On a CPU device the f32 weights themselves are host memory: the file's size. A whole-model f32 read and its
    // clone add twice the file on top.
    assert!(growth < file + file / 2, "host peak grew {growth} bytes loading a {file}-byte checkpoint");
}
