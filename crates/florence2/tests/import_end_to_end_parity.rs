// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `florence2::import::build_param_source` against the real checkpoint,
//! chained through the FULL pipeline (DaViT -> `ImageProject` ->
//! `Florence2Lm`) rather than each stage fed the previous stage's own
//! golden in isolation (as `davit_*_parity.rs`/`text_lm_parity.rs` do) -
//! this is the actual shipped path a real `ground` call takes, and the one
//! place a mismatch between `import.rs`'s tensor names/transforms and what
//! each module's tests were separately built against would surface.

use std::collections::HashMap;

use gpu_core::Gpu;
use model::hostmath::cosine;
use paramstore::{ParamStore, Role};

use florence2::import::{all_tensor_names, build_param_source};
use florence2::text::{BartConfig, Florence2Lm, Florence2LmKernelIds};
use florence2::vision::pipelines::PIPELINES;
use florence2::vision::{Davit, DavitConfig, DavitKernelIds, ImageProject, ImageProjectKernelIds};

const PROMPT_IDS: [u32; 6] = [100, 200, 300, 400, 500, 600];
const DECODER_IDS: [u32; 4] = [2, 10, 20, 30];

#[test]
fn import_then_full_pipeline_matches_real_checkpoint_end_to_end() {
    let Some(ckpt_dir) = std::env::var("FLORENCE2_DIR").ok() else {
        brain_testutil::skip("FLORENCE2_DIR unset");
        return;
    };
    let davit_golden_path = brain_testutil::testdata("florence2/davit/stages.safetensors");
    let encdec_golden_path = brain_testutil::testdata("florence2/encdec/step.safetensors");
    if !std::path::Path::new(&davit_golden_path).exists() || !std::path::Path::new(&encdec_golden_path).exists() {
        brain_testutil::skip("florence2 golden fixtures absent - regenerate with tools/goldens/florence2_dump_reference.py");
        return;
    }

    let source = build_param_source(std::path::Path::new(&ckpt_dir)).expect("build_param_source");

    let davit_cfg = DavitConfig::florence2_base();
    let bart_cfg = BartConfig::florence2_base();
    let roles: Vec<_> = all_tensor_names(&davit_cfg, &bart_cfg)
        .into_iter()
        .map(|name| {
            let numel = source.get(&name).unwrap_or_else(|| panic!("missing source tensor {name}")).1.len();
            (name, numel, Role::Frozen)
        })
        .collect();

    let gpu = Gpu::new_cpu(PIPELINES);
    let ps = ParamStore::new_with_roles_src(&gpu, roles, &source);

    let davit_k = DavitKernelIds::resolve(PIPELINES);
    let davit = Davit::new(&gpu, &davit_k, "vision_tower", &davit_cfg, 4, 1e-5, false);
    // `final_tokens_and_dim` is `forward_features_unpool`'s own 576-token
    // output count - `ImageProject` then pools one MORE token on top
    // (`spatial_avg_pool`), so the actual vision-token count the encoder
    // sees is 577, not 576.
    let (davit_tokens, davit_dim) = davit_cfg.final_tokens_and_dim();
    let t_vision = davit_tokens + 1;
    let proj_k = ImageProjectKernelIds::resolve(PIPELINES);
    let proj = ImageProject::new(&gpu, florence2::import::VISION_PROJECTOR_PREFIX, 24, 24, davit_dim, bart_cfg.d_model, 1e-5);

    let davit_golden = checkpoint::safetensors::read(&davit_golden_path).expect("read davit golden");
    let dgmap: HashMap<String, checkpoint::safetensors::StTensor> = davit_golden.into_iter().map(|t| (t.name.clone(), t)).collect();

    let px_host = &dgmap["pixel_values"].data;
    let px_buf = gpu.storage(px_host.len() as u64);
    gpu.write_f32(&px_buf, px_host);

    let unpooled = davit.forward_features_unpool(&gpu, &davit_k, &ps, &px_buf);
    let projected = proj.forward(&gpu, &proj_k, &ps, unpooled);
    gpu.poll_wait();

    let want_projected = &dgmap["projected"].data;
    let got_projected = gpu.read(projected, want_projected.len());
    let c_proj = cosine(&got_projected, want_projected);
    assert!(c_proj >= 0.999, "vision pipeline (import -> DaViT -> ImageProject) cosine {c_proj:.6} (want >= 0.999)");

    let image_features = gpu.storage((t_vision * bart_cfg.d_model) as u64);
    gpu.write_f32(&image_features, &got_projected);

    let lm_k = Florence2LmKernelIds::resolve(PIPELINES);
    let lm = Florence2Lm::new(&gpu, bart_cfg, t_vision, PROMPT_IDS.len() as u32, DECODER_IDS.len() as u32);
    let enc_out = lm.encode(&gpu, &lm_k, &ps, &image_features, &PROMPT_IDS);
    gpu.poll_wait();

    let encdec_golden = checkpoint::safetensors::read(&encdec_golden_path).expect("read encdec golden");
    let egmap: HashMap<String, checkpoint::safetensors::StTensor> = encdec_golden.into_iter().map(|t| (t.name.clone(), t)).collect();

    let want_enc = &egmap["encoder_out"].data;
    let got_enc = gpu.read(enc_out, want_enc.len());
    let c_enc = cosine(&got_enc, want_enc);
    assert!(c_enc >= 0.999, "encoder (fed from the just-computed vision pipeline, not the golden's inputs_embeds slice) cosine {c_enc:.6} (want >= 0.999)");

    let logits = lm.decode(&gpu, &lm_k, &ps, enc_out, &DECODER_IDS);
    gpu.poll_wait();
    let want_logits = &egmap["decoder_logits"].data;
    let got_logits = gpu.read(logits, want_logits.len());
    let c_logits = cosine(&got_logits, want_logits);
    assert!(c_logits >= 0.999, "decoder logits cosine {c_logits:.6} (want >= 0.999)");
}
