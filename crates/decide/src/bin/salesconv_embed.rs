// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dump this encoder's sentence embedding of each conversation, so the ceiling
//! it supports can be measured the same way the paper's own embeddings were.
//!
//! The question this answers: of the gap between `samples/decision/salesagent`
//! and the best a linear model gets on the published Azure OpenAI embeddings,
//! how much is the ENCODER and how much is everything downstream of it? Fitting
//! the same classifier on both representations splits the difference, and
//! nothing else does - a lower score could otherwise always be blamed on the
//! head, the objective, or the training budget.
//!
//! Writes TSV: `outcome` then the pooled embedding. Usage:
//!
//! ```text
//! salesconv_embed [DATA_DIR] [OUT_TSV] [MAX_ROWS_PER_SPLIT]
//! ```
//!
//! Swedish Embedded AB measures where a model's accuracy is actually lost
//! before anyone optimizes for its clients. If your team needs that, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::io::Write;
use std::path::Path;

use decide::decide::{Decide, Limits};
use decide::primitives::Question;

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    let enc_dir = std::env::var("BRAIN_MINILM_DIR").unwrap_or_else(|_| {
        format!("{home}/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2")
    });
    let data = std::env::args().nth(1).unwrap_or_else(|| "testdata/decide/salesconv".into());
    // Repo-relative, so the default resolves wherever the checkout is.
    let out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../out/salesconv-embeddings.tsv").into());
    let limit: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(4000);

    let dir = Path::new(&enc_dir);
    let cfg = decide::import::config_from_hf(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let tensors = checkpoint::safetensors::read(dir.join("model.safetensors").to_str().unwrap()).unwrap();
    let init = decide::import::brain_init_from_hf(tensors, &cfg).unwrap();
    let tok = data::wordpiece::WordPiece::from_file(dir.join("tokenizer.json").to_str().unwrap()).unwrap();
    let head = decide::init::init_head(&cfg, 0);
    let gpu = gpu_core::Gpu::new(decide::kern::PIPELINES);
    let limits = Limits { cap_rows: 8192, cap_slots: 8, max_span: 256, overlap: 32 };
    let mut m = Decide::new_on(gpu, cfg, tok, limits, &init, &head, true);

    let convs = decide::salesconv::SalesConversations::load(Path::new(&data)).expect("dataset");
    // The head is untrained and its scores are ignored; only the encoder's
    // pooled state embedding is read.
    let q = Question::Noul { instructions: "does this convert".into(), yes: None, no: None };
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).expect("create"));
    let mut n = 0usize;
    for split in [&convs.train, &convs.test] {
        for c in split.iter().take(limit) {
            // The WHOLE conversation, which is what the Azure embeddings in the
            // published dataset were computed over.
            let state = c.prefix(c.len() - 1);
            let req = m.pack_request(&state, std::slice::from_ref(&q)).expect("pack");
            let _ = m.run_packed(&req);
            let e = m.state_embedding();
            write!(f, "{}", u8::from(c.outcome)).unwrap();
            for v in &e {
                write!(f, "\t{v:.5}").unwrap();
            }
            writeln!(f).unwrap();
            n += 1;
            if n.is_multiple_of(500) {
                eprint!("  {n}\r");
            }
        }
    }
    eprintln!("\nwrote {n} rows of {} dims to {out}", 384);
}
