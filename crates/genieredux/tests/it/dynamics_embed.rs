// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying neural-network ports
// without their checkpoints for its clients. If your team needs expertise in
// parity-gated model porting then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The dynamics input embedding at kept and masked tokens, against the
//! reference rule: `token_emb(id) + 10 * project_out(codebook[id])` at kept
//! tokens, `token_emb(mask) + 0` at masked ones (the reference's
//! `mask_token_template` is `torch.zeros(dim)`), plus the absolute positional
//! embedding and the action one-hot.

use crate::tiny;
use gpu_core::Gpu;
use genieredux::{dynamics_embed, kernel_sources};

#[test]
fn kept_and_masked_tokens_embed_by_the_reference_rule() {
    let gpu = Gpu::new_cpu(&kernel_sources());
    let dw = tiny::dynamics(&tiny::tokenizer());
    let dc = tiny::dynamics_config();
    let (dim, cd, na, hw) = (dc.dim as usize, dc.code_dim as usize, dc.num_actions as usize, tiny::HW);
    let mask = dc.n_codes;
    // one frame, second frame starting at absolute frame 3: a mask and three kept tokens
    let ids = [mask, 5, 0, 15];
    let actions = tiny::one_hot(&[2], na);
    let first_frame = 3usize;
    let got = dynamics_embed(&gpu, &ids, &actions, &dw, 1, 1, 2, 2, dc.dim, dc.n_codes, dc.code_dim, dc.num_actions, first_frame as u32);

    for (p, &id) in ids.iter().enumerate() {
        let row = &got[p * (dim + na)..(p + 1) * (dim + na)];
        let pos = first_frame * hw + p;
        for c in 0..dim {
            let blend = if id == mask {
                0.0
            } else {
                let code = &dw.codebook[id as usize * cd..(id as usize + 1) * cd];
                let mut acc = dw.project_out_b[c];
                for k in 0..cd {
                    acc += code[k] * dw.project_out_w[c * cd + k];
                }
                10.0 * acc
            };
            let want = dw.token_emb[id as usize * dim + c] + blend + dw.pos_emb[pos * dim + c];
            assert!((row[c] - want).abs() < 1e-4, "token {p} ({id}) channel {c}: {} vs {want}", row[c]);
        }
        assert_eq!(&row[dim..], &actions[..], "action one-hot at token {p}");
    }
}
