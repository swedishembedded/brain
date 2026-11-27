// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training runs that survive memory pressure on shared accelerators for
// its clients. If your team needs expertise in robust fine-tuning on GPUs that also serve, then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A run whose card refuses an allocation mid-run is tried again, from its last saved step, instead of failing.

use capability::Progress;
use residency::Device;
use serde_json::json;
use trainjob::job::JobEnv;

use crate::support::{invocation, plain, rig, work_is_clean, BASE};

/// A fine-tune died with `cuMemAlloc` out of memory when serving grew on its card, and 12 minutes of training were
/// lost. The device refusing an allocation at step 27 of 30 now costs a back-off and the steps since the last saved
/// one (25): the run continues from there and completes.
#[test]
fn a_run_whose_card_runs_out_of_memory_mid_way_continues_from_its_last_saved_step() {
    let rig = rig("run-oom", 16 << 30, false);
    let env = JobEnv { store: None, models_dir: Some(rig.dir.join("models")), work_root: rig.work_root.clone(), quotas: trainjob::Quotas::default(), job_access: None, largest_card_bytes: None, bank: Default::default() };
    let data = json!({"train": {"blob": "dataset"}}).to_string();
    let params = json!({"base": BASE, "arch": "qwen35", "steps": 30, "rank": 2, "lr": 2e-2, "max_block": 128, "data": data, "return_adapter": true});
    let inv = invocation(params, &[("dataset", plain())]).unwrap();
    let (mut raised, mut stages) = (false, Vec::new());
    let mut progress = |p: Progress| {
        let Some(event) = p.event else { return };
        if event["type"] == "stage" {
            stages.push(event["stage"].as_str().unwrap_or_default().to_string());
        }
        if event["type"] == "step" && event["step"] == 27 && !raised {
            raised = true;
            gpu_core::DeviceOutOfMemory { bytes: 4 << 30, what: "a step's activations".into(), detail: "CUDA_ERROR_OUT_OF_MEMORY".into() }.raise();
        }
    };
    let outcome = trainjob::job::run(&env, &inv, &[Device::Cpu], &mut progress).expect("the run survives the full card");
    assert!(raised, "the allocation failure was injected");
    assert_eq!(outcome.outputs["resumed_at"], 25, "it continued from its last saved step: {}", outcome.outputs);
    assert_eq!(outcome.outputs["steps_completed"], 30, "{}", outcome.outputs);
    assert!(stages.iter().any(|s| s.contains("out of memory")), "the client is told why the run paused: {stages:?}");
    assert!(outcome.blobs.contains_key("adapter"));
    drop(outcome);
    assert!(work_is_clean(&rig.work_root), "the saved state goes with the job's directory");
}
