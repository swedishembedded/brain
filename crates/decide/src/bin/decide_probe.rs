// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What actually reaches the score - a probe for a model that answers the same
//! thing whatever it is asked.
//!
//! A decision model returning a near-constant probability has one of three
//! problems, and they need completely different fixes:
//!
//! 1. **The encoder is not separating the states.** Then nothing downstream
//!    can, and the fix is upstream of the head entirely.
//! 2. **The options are not separating the queries.** The head takes each
//!    option's slot representation as its attention QUERY, so two options
//!    whose text is nearly identical start out nearly parallel, produce nearly
//!    equal scores, and leave almost no gradient to separate them.
//! 3. **The head is not using what it is given** - both of the above are fine
//!    and the score still does not move.
//!
//! This prints the measurements that tell them apart, so the next change is
//! chosen rather than guessed. It also compares the `[CLS]` gather against
//! MEAN pooling of the same slot, because the released checkpoint this runs on
//! is a sentence-transformer: its sentence representation is the mean, and its
//! `[CLS]` was never trained to be one.
//!
//! Swedish Embedded AB diagnoses and repairs neural network training failures
//! for its clients. If your team needs a model that is not learning turned into
//! one that is, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

use decide::decide::{Decide, Limits};
use decide::primitives::Question;

/// Deliberately extreme, so a model with ANY signal must separate them.
const CONVERSATIONS: &[(&str, &str)] = &[
    (
        "obvious close",
        "customer: this looks perfect, exactly what we need\n\
         rep: great, I can send the contract over today\n\
         customer: yes please send it, I have budget approved and I want to sign this week",
    ),
    (
        "obvious loss",
        "customer: honestly this is way outside our budget\n\
         rep: I understand, we do have a smaller tier\n\
         customer: no, we have decided to go with a competitor, please stop contacting me",
    ),
    (
        "undecided",
        "customer: interesting, tell me more about the integrations\n\
         rep: sure, it connects to most CI systems\n\
         customer: ok, I will think about it and get back to you",
    ),
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

/// Mean of the rows of one span - what this checkpoint's sentence embedding
/// actually is.
#[allow(dead_code)]
fn span_mean(hidden: &[f32], row0: usize, len: usize, h: usize) -> Vec<f32> {
    let mut v = vec![0f32; h];
    for r in row0..row0 + len {
        for (c, vc) in v.iter_mut().enumerate() {
            *vc += hidden[r * h + c];
        }
    }
    for vc in &mut v {
        *vc /= len as f32;
    }
    v
}

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")
    });
    let head_path = std::env::args().nth(2).filter(|p| p != "-");
    let dir = Path::new(&dir);
    let cfg = decide::import::config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let tensors = checkpoint::safetensors::read(dir.join("model.safetensors").to_str().unwrap()).unwrap();
    let enc_init = decide::import::brain_init_from_hf(tensors, &cfg).unwrap();
    let tok = data::wordpiece::WordPiece::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let head_init = match &head_path {
        Some(p) => checkpoint::safetensors::read(p).unwrap().into_iter().map(|x| (x.name, x.data)).collect(),
        None => decide::init::init_head(&cfg, 0),
    };
    let gpu = gpu_core::Gpu::new(decide::kern::PIPELINES);
    let limits = Limits { cap_rows: 8192, cap_slots: 8, max_span: 256, overlap: 32 };
    let mut m = Decide::new_on(gpu, cfg.clone(), tok, limits, &enc_init, &head_init, true);
    println!("head: {}\n", head_path.as_deref().unwrap_or("(fresh, untrained)"));

    // The question as the sales pipeline asks it, and a contrastive rewrite
    // with no shared instructions at all.
    let questions: Vec<(&str, Question)> = vec![
        (
            "long shared instructions",
            Question::Noul {
                instructions: "will this sales conversation end in a closed deal".into(),
                yes: Some("the deal closes".into()),
                no: Some("the deal is lost".into()),
            },
        ),
        (
            "bare contrastive options",
            Question::Noul { instructions: String::new(), yes: Some("closed won".into()), no: Some("closed lost".into()) },
        ),
    ];

    let h = cfg.d_model as usize;
    for (qname, q) in &questions {
        println!("=== question: {qname} ===");
        let mut probs = Vec::new();
        let mut state_emb = Vec::new();
        let mut reported_slots = false;
        for (name, text) in CONVERSATIONS {
            let req = m.pack_request(text, std::slice::from_ref(q)).expect("pack");
            let scores = m.run_packed(&req);
            let p = decide::primitives::sigmoid(scores[0]);
            probs.push(p);
            state_emb.push(m.enc.pooled_mean()[..h].to_vec());
            if !reported_slots {
                reported_slots = true;
                let slots = req.packed.slot_spans();
                println!("  slots: {} (a noul scores the proposition alone)", slots.len());
            }
            println!("  {name:<16} P(closes) = {:.4}   score {:+.4}", p, scores[0]);
        }
        let spread = probs.iter().cloned().fold(f32::MIN, f32::max) - probs.iter().cloned().fold(f32::MAX, f32::min);
        println!("  P spread over three very different conversations: {spread:.4}");
        println!(
            "  state embedding cosine: close/loss {:.4}, close/undecided {:.4}",
            cosine(&state_emb[0], &state_emb[1]),
            cosine(&state_emb[0], &state_emb[2])
        );
        println!();
    }
}
