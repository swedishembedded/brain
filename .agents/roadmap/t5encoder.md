# t5encoder - roadmap

T5-XXL text encoder port (`crates/t5encoder`), the text conditioning FLUX.1
runs on: bidirectional encoder-only T5 (RMSNorm, no bias, **no** `1/sqrt(d_kv)`
attention scale, a learned relative-position bucket bias shared by every
layer, gated-GELU `wi_0`/`wi_1` FFN). Imported 219 → 171 tensors, two-way
covered; `relative_position_bucket` exact over all 16384 entries.

Forward parity is gated at 42/42, worst cosine 0.9999999992 (B=2, T=128),
plus a **checkpoint-free** `tiny_ref` gate at deliberately distinct dims
(`heads != d_kv`, `heads*d_kv != d_model`) at cosine 1.0000000000 - because
at XXL those three numbers are all equal and a head-count/head-width swap
would be invisible.

Backward exists (`train.rs`), gated by `gradcheck::check_t5`,
`check_t5_one_block`, `check_t5_tiled` and `check_t5_rel_bias_elementwise`.
The serving contract is met: `t5encoder::caps` (`encode`, `variant` selecting
flux_xxl / wan_umt5, tokenized via `data::unigram::UnigramTokenizer`),
`resident_t5encoder::T5encoderResident`, a `catalog.rs` entry, D-Bus `Run`,
`examples/embedding/t5_embed.py`. `crates/flux1::pipeline` calls it for real
(`t5encoder::import::read_encoder`).

## Not yet done

- [ ] **Sequence length 512 - the length FLUX.1 actually uses.** Forward
      parity is gated only at length 128. At 512 the fp32 activations do not
      fit on one 24 GiB GPU, *and* the attention-score kernel's dispatch
      shape crosses into a code path shorter sequences never exercise, so it
      needs its own verification pass. This is the single most important gap
      here: the served path runs at a length nothing has verified.
- [ ] An end-to-end fixture for the served `encode` path - `caps` has no
      checked-in test data in this workspace to verify tokenize → encode
      against
- [ ] `run_batch` batches rows into one forward at a shared context length
      only; ragged/mixed-length batching is not implemented

The CPU backend's reduction order depends on Rayon's runtime work-splitting,
so its low-order output bits are not deterministic run-to-run - it cannot be
used as an exact numeric fingerprint the way the GPU path can.
