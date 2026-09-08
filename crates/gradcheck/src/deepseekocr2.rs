// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for `crates/deepseekocr2`'s resampler - the
//! prefix-LM-masked Qwen2 GQA tower plus linear projector that DeepSeek-OCR-2
//! substitutes for v1's CLIP tower. SAM's own gradients are v1's concern
//! ([`crate::deepseekocr`], `crates/sam1/tests/gradcheck.rs`) and the decoder
//! is unchanged from v1's ([`crate`]'s `check_deepseek2*` family) - this file
//! covers exactly the new surface: every trainable weight/bias/norm/query-bank
//! tensor in the tower, AND the SAM-token-grid input the tower's own gradient
//! must flow back into for a real SAM+encoder joint fine-tune to make sense.
//!
//! ## Why a bespoke [`CheckModel`], not the blanket `model::Model` impl
//!
//! `Resampler` runs one view at a time and has no token stream, no batch, and
//! no loss of its own - `model::Model::forward`'s "return a scalar loss"
//! contract has nothing natural to mean here. The harness below supplies the
//! missing piece itself: a fixed random linear probe against the projected
//! output (`loss = <projected, target>`), which differentiates to exactly
//! `target` - simple enough that the harness cannot itself be the bug, while
//! still exercising the tower's real backward end to end.
//!
//! ## Bridging `loss()`/`backward()` across a two-call trait
//!
//! [`CheckModel::loss`] and [`CheckModel::backward`] are separate `&self`
//! calls, but `Resampler::forward_train`/`backward_train` are a single
//! forward-then-consume pair threaded through an owned [`deepseekocr2::
//! encoder::TrainState`]. [`Harness`] bridges the two with a `RefCell` slot -
//! `loss()` fills it, `backward()` takes it - which is exactly what
//! `directional_check`'s own call order needs (`zero_grads` → one `loss` +
//! `backward` pair → repeated `loss`-only calls for the finite-difference
//! sweep, each of which harmlessly refills the slot with a run `backward()`
//! never drains again).

use std::cell::RefCell;
use std::collections::HashMap;

use data::rng::Rng;
use deepseekocr2::config::{DeepseekOcr2VisionConfig, Qwen2EncoderConfig};
use deepseekocr2::encoder::{Resampler, TrainState};

use crate::{directional_check, CheckModel, Report};

/// The synthetic input's name in [`CheckModel::param_names`] - not a real
/// `ParamStore` entry, so [`Harness`] special-cases it in every accessor.
const SAM_INPUT: &str = "sam_input";

struct Harness {
    r: Resampler,
    local: bool,
    sam0: RefCell<Vec<f32>>,
    target: Vec<f32>,
    state: RefCell<Option<TrainState>>,
    d_sam: RefCell<Option<Vec<f32>>>,
}

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        let mut names = self.r.param_names();
        names.push(SAM_INPUT.to_string());
        names
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        if name == SAM_INPUT {
            self.sam0.borrow().clone()
        } else {
            self.r.read_weight(name)
        }
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        if name == SAM_INPUT {
            *self.sam0.borrow_mut() = data.to_vec();
        } else {
            self.r.write_weight(name, data);
        }
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        if name == SAM_INPUT {
            self.d_sam.borrow().clone().expect("read_grad(\"sam_input\") called before loss()+backward()")
        } else {
            self.r.read_grad(name)
        }
    }
    fn zero_grads(&self) {
        self.r.zero_grads();
    }
    fn loss(&self) -> f32 {
        let sam = self.sam0.borrow().clone();
        let (projected, state) = self.r.forward_train(&sam, self.local);
        *self.state.borrow_mut() = Some(state);
        projected.iter().zip(&self.target).map(|(p, t)| p * t).sum()
    }
    fn backward(&self) {
        let state = self.state.borrow_mut().take().expect("backward() called before loss()");
        let d_sam = self.r.backward_train(state, &self.target, self.local);
        *self.d_sam.borrow_mut() = Some(d_sam);
    }
}

/// Gradient-check the resampler: every encoder-block weight/bias, both norm
/// weights, the projector's weight/bias, both learned query banks, and the
/// SAM-token-grid input, over the tower's LOCAL (768x768-tile) view.
///
/// Dims are deliberately tiny and mutually distinct (an even `head_dim` so
/// RoPE's pairing is well-formed, a `rms_eps` matched to the fixed kernel
/// constant `Qwen2EncoderConfig::check` enforces) - independent of
/// `tools/goldens/deepseekocr2_dump_reference.py`'s own tiny fixture, since
/// this check needs no golden at all, only a finite-difference comparison
/// against itself.
pub fn check_deepseekocr2(seed: u64) -> Report {
    let sam = sam1::SamViTConfig { compress_out: 8, ..sam1::SamViTConfig::tiny() };
    let encoder = Qwen2EncoderConfig {
        d_model: 8,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        ffn_hidden: 13,
        rms_eps: model::block::RMSNORM_EPS,
        rope_theta: 10_000.0,
        n_query_local: 3,
        n_query_global: 5,
    };
    let cfg = DeepseekOcr2VisionConfig { sam, encoder, decoder_hidden: 6 };

    let mut rng = Rng::new(seed);
    let mut init: HashMap<String, Vec<f32>> = HashMap::new();
    let mut names = cfg.encoder.param_list();
    names.extend(cfg.projector_param_list());
    for (name, numel) in names {
        init.insert(name, (0..numel).map(|_| (rng.next_f32() - 0.5) * 0.2).collect());
    }

    let gpu = gpu_core::testgpu::dev(deepseekocr2::encoder::PIPELINES);
    let r = Resampler::new_on(gpu, cfg.clone(), &init, true);

    let n_query = cfg.encoder.n_query_local as usize;
    let d = cfg.encoder.d_model as usize;
    let sam0: Vec<f32> = (0..n_query * d).map(|_| rng.next_f32() - 0.5).collect();
    let target: Vec<f32> = (0..n_query * cfg.decoder_hidden as usize).map(|_| rng.next_f32() - 0.5).collect();

    let harness = Harness { r, local: true, sam0: RefCell::new(sam0), target, state: RefCell::new(None), d_sam: RefCell::new(None) };
    directional_check(&harness, 5e-3, 4, seed ^ 0x2ee2)
}
