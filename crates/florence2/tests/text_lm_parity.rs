// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Florence2Lm` (encoder + decoder + lm_head) against the real checkpoint:
//! `inputs_embeds`'s first 577 rows (already-projected vision tokens, no
//! DaViT/`ImageProject` recomputation needed here - that ladder is verified
//! independently in `davit_image_project_parity.rs`) plus the fixed
//! `PROMPT_IDS` text prompt feed the encoder, and the fixed `DECODER_IDS`
//! teacher-forced target feeds the decoder. Golden captured via
//! `tools/goldens/florence2_dump_reference.py::dump_encdec` - see that
//! function's own doc for why `PROMPT_IDS`/`DECODER_IDS` are arbitrary
//! fixed in-vocab integers rather than real tokenizer output, and for the
//! real HF weight-tying bug this dump works around (`encoder.embed_tokens`/
//! `decoder.embed_tokens`/`lm_head` do NOT auto-tie to `shared.weight` on
//! load - confirmed via a direct `torch.equal` check - so the dump script
//! force-ties them before running, matching what `brain`'s own import
//! always does by reading `shared.weight` directly for all three uses).

use std::collections::HashMap;

use gpu_core::Gpu;
use model::hostmath::cosine;
use paramstore::{ParamStore, Role};

use florence2::text::{BartConfig, Florence2Lm, Florence2LmKernelIds};
use florence2::vision::pipelines::PIPELINES;

const PROMPT_IDS: [u32; 6] = [100, 200, 300, 400, 500, 600];
const DECODER_IDS: [u32; 4] = [2, 10, 20, 30];
const T_VISION: u32 = 577;

fn linear_names(prefix: &str) -> Vec<String> {
    vec![format!("{prefix}.weight"), format!("{prefix}.bias")]
}

fn attn_names(prefix: &str) -> Vec<String> {
    ["q_proj", "k_proj", "v_proj", "out_proj"].iter().flat_map(|p| linear_names(&format!("{prefix}.{p}"))).collect()
}

fn all_tensor_names(cfg: &BartConfig) -> Vec<String> {
    let mut names = vec!["language_model.model.shared.weight".to_string(), "language_model.final_logits_bias".to_string()];

    for side in ["encoder", "decoder"] {
        let base = format!("language_model.model.{side}");
        names.push(format!("{base}.embed_positions.weight"));
        names.extend(linear_names(&format!("{base}.layernorm_embedding")));
        let layers = if side == "encoder" { cfg.encoder_layers } else { cfg.decoder_layers };
        for i in 0..layers {
            let lp = format!("{base}.layers.{i}");
            names.extend(attn_names(&format!("{lp}.self_attn")));
            names.extend(linear_names(&format!("{lp}.self_attn_layer_norm")));
            if side == "decoder" {
                names.extend(attn_names(&format!("{lp}.encoder_attn")));
                names.extend(linear_names(&format!("{lp}.encoder_attn_layer_norm")));
            }
            names.extend(linear_names(&format!("{lp}.fc1")));
            names.extend(linear_names(&format!("{lp}.fc2")));
            names.extend(linear_names(&format!("{lp}.final_layer_norm")));
        }
    }
    names
}

#[test]
fn florence2_lm_encoder_and_decoder_match_real_checkpoint() {
    let Some(ckpt_dir) = std::env::var("FLORENCE2_DIR").ok() else {
        brain_testutil::skip("FLORENCE2_DIR unset");
        return;
    };
    let golden_path = brain_testutil::testdata("florence2/encdec/step.safetensors");
    if !std::path::Path::new(&golden_path).exists() {
        brain_testutil::skip("florence2 encdec golden fixture absent - regenerate with tools/goldens/florence2_dump_reference.py");
        return;
    }

    let ckpt = checkpoint::safetensors::read(&format!("{ckpt_dir}/model.safetensors")).expect("read checkpoint");
    let mut source: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for t in ckpt {
        source.insert(t.name, (t.shape, t.data));
    }

    let cfg = BartConfig::florence2_base();
    let roles: Vec<_> = all_tensor_names(&cfg)
        .into_iter()
        .map(|name| {
            let numel = source.get(&name).unwrap_or_else(|| panic!("missing checkpoint tensor {name}")).1.len();
            (name, numel, Role::Frozen)
        })
        .collect();

    let gpu = Gpu::new_cpu(PIPELINES);
    let ps = ParamStore::new_with_roles_src(&gpu, roles, &source);
    let k = Florence2LmKernelIds::resolve(PIPELINES);

    let golden = checkpoint::safetensors::read(&golden_path).expect("read golden");
    let gmap: HashMap<String, checkpoint::safetensors::StTensor> = golden.into_iter().map(|t| (t.name.clone(), t)).collect();

    let inputs_embeds = &gmap["inputs_embeds"].data;
    let d = cfg.d_model as usize;
    let image_features_host = &inputs_embeds[..T_VISION as usize * d];
    let image_features = gpu.storage(image_features_host.len() as u64);
    gpu.write_f32(&image_features, image_features_host);

    let lm = Florence2Lm::new(&gpu, cfg, T_VISION, PROMPT_IDS.len() as u32, DECODER_IDS.len() as u32);
    let enc_out = lm.encode(&gpu, &k, &ps, &image_features, &PROMPT_IDS);
    gpu.poll_wait();

    let want_enc = &gmap["encoder_out"].data;
    let got_enc = gpu.read(enc_out, want_enc.len());
    let c_enc = cosine(&got_enc, want_enc);
    assert!(c_enc >= 0.999, "encoder cosine {c_enc:.6} (want >= 0.999); got[0..4]={:?} want[0..4]={:?}", &got_enc[..4], &want_enc[..4]);

    let logits = lm.decode(&gpu, &k, &ps, enc_out, &DECODER_IDS);
    gpu.poll_wait();
    let want_logits = &gmap["decoder_logits"].data;
    let got_logits = gpu.read(logits, want_logits.len());
    let c_logits = cosine(&got_logits, want_logits);
    assert!(c_logits >= 0.999, "decoder logits cosine {c_logits:.6} (want >= 0.999); got[0..4]={:?} want[0..4]={:?}", &got_logits[..4], &want_logits[..4]);

    // Florence2Lm::generate's first step is a t=1 decode from just
    // decoder_start_token_id (== DECODER_IDS[0]=2). Causal self-attention
    // means row 0 of a t=4 decode must depend on nothing past position 0,
    // so its prediction should be IDENTICAL to a t=1 decode's only row -
    // this catches a causal-masking bug the cosine check above cannot (a
    // leak from a later position would still leave the OVERALL 4-row
    // cosine near 1.0 while silently corrupting row 0 specifically).
    let vocab = cfg.vocab_size as usize;
    let want_argmax = want_logits[..vocab].iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap();
    let generated = lm.generate(&gpu, &k, &ps, enc_out, 1);
    assert_eq!(generated, vec![want_argmax], "generate()'s first greedy token must match the golden's row-0 argmax");
}
