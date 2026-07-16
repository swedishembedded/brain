// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the real GenieRedux tokenizer checkpoint (gitignored scratch) with
//! full-coverage validation. Ignored by default; run manually with the file:
//!   cargo test -p brain-wm-genie --test import -- --ignored --nocapture
use wm_genie::import::import_tokenizer;

const CK: &str = "/data/workspace/applications/edgeai/brain/scratchpad/wm-checkpoints/GenieRedux_Tokenizer_CoinRun_100mln_v1.0.pt";

#[test]
#[ignore = "needs the 1.2GB checkpoint in scratch; run manually"]
fn import_tokenizer_full_coverage() {
    if !std::path::Path::new(CK).exists() {
        eprintln!("SKIP: {CK} absent");
        return;
    }
    let (w, cfg) = import_tokenizer(CK).expect("import must succeed with full coverage");
    // spot-check structure against the fixed config
    assert_eq!(cfg.dim, 512);
    assert_eq!(w.encoder.layers.len(), cfg.enc_layers);
    assert_eq!(w.decoder.layers.len(), cfg.dec_layers);
    assert_eq!(w.encoder.norm_out_gamma.len(), 512);
    // fused to_kv split
    let a = &w.encoder.layers[0].spatial_attn;
    assert_eq!(a.to_q.len(), 512 * 512);
    assert_eq!(a.to_k.len(), 512 * 512);
    assert_eq!(a.to_v.len(), 512 * 512);
    assert_eq!(a.q_scale.len(), 64);
    // GEGLU in-proj split
    let f = &w.encoder.layers[0].spatial_ff;
    assert_eq!(f.w_x.len(), cfg.ff_inner as usize * 512);
    assert_eq!(f.w_gate.len(), cfg.ff_inner as usize * 512);
    // VQ codebook 1024x32 (stored [1,K,cd])
    assert_eq!(w.vq.codebook.len(), 1024 * 32);
    // patch embed 48->512
    assert_eq!(w.patch_first.lin_w.len(), 512 * 48);
    // CPB net
    assert_eq!(w.cpb_net.len(), 3);
    assert_eq!(w.cpb_net[2].out_dim, 8);
    eprintln!("import OK: {} enc + {} dec blocks, all 514 model.* tensors consumed",
        w.encoder.layers.len(), w.decoder.layers.len());
}
