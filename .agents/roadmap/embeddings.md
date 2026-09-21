# embeddings - roadmap

Long-context text embedding, retrieval, retrieval-augmented generation, and
contrastive fine-tuning over frozen embeddings, reachable through the public
`brain` SDK. Written in terms of "a caller" throughout - nothing here names a
downstream project.

## What shipped

- [x] **A 32768-token embedding backbone.** `qwen3::caps`'s new `embed`
      action pools the Qwen3 decoder's last-token final-norm hidden state
      through `Qwen::prefill`'s decode-only KV-cache path (`O(T)` memory,
      never the batched forward's `O(T^2)` `scores`/`probs`), L2-normalizes
      on the host, and supports the Qwen3-Embedding instruction-prefix query
      convention. A latent `u32` overflow in the batched-forward attention
      score buffer sizing (`b * n_heads * t * t` computed before widening to
      `u64`) was found and fixed while sizing this action's allocations -
      real for any long-context batched caller, not only this one.
- [x] **`brain::EmbeddingPipeline` becomes multi-architecture**: CLIP's text
      towers (`vision` feature, unchanged default behavior) and the Qwen3
      backbone above (`text` feature), dispatched from the resolved
      architecture per sdk-design rule 2. `EmbeddingOptions` (instruction,
      max_tokens, dimensions, normalize) and `Embedding::cosine_similarity`.
- [x] **`brain::EmbeddingTrainer`**: a host-side, symmetric (CLIP-style)
      InfoNCE-trained linear projection over frozen embeddings, near-identity
      initialized. Does NOT fine-tune either backbone - see the seeded-backward
      gap below. Gradient-checked against finite differences directly (no
      `crates/gradcheck` integration - this objective has no GPU `ParamStore`
      for that trait's `CheckModel` shape to describe).
- [x] Four samples: `samples/text/embed` (inference, including a real
      long-document 32k example), `samples/text/retrieve` (exact
      brute-force top-k, explicitly NOT an ANN index), `samples/text/retrieve-json`
      (retrieval feeding a decoder for prompt-validated, NOT
      grammar-constrained, JSON output), `samples/text/embed-train`
      (before/after recall@1 over `EmbeddingTrainer`).
- [x] `qwen3::caps` moved off `BRAIN_QWEN_WEIGHTS`/`BRAIN_QWEN_TOKENIZER`
      (`host_env`) onto the model-store resolver (`ParamSpec::host_resolved()`,
      `qwen3::spec::Qwen3Spec`) for `generate`/`lora_train`/`lora_gate`/`embed`
      alike - see `.agents/roadmap/continuous-learning.md`'s B3a update. The
      separate `brain serve` residency-scheduler hot-swap resident
      (`crate::resident_llm::QwenResident`) is unaffected, a different
      mechanism, out of this track's scope.
- [x] LFM2.5-Encoder gains YaRN long-context RoPE scaling
      (`LfmConfig::rope_scaling`) - see `.agents/roadmap/lfm2.md` for the
      detail.
- [x] LFM2.5-Encoder gains a seeded backward pass (`Lfm::seed_buf`/
      `backward_seeded`) - an external objective's gradient on the final
      hidden states now reaches every encoder parameter, gradient-checked
      directly against finite differences
      (`lfm_seeded_analytic_grads_match_finite_differences`). This is the
      primitive full-encoder contrastive fine-tuning needs; the SDK-level
      training loop that actually DRIVES it with `EmbeddingTrainer`'s InfoNCE
      objective is not yet built - see `.agents/roadmap/lfm2.md`.
- [x] `lfm2::caps`'s `embed` action gains a `normalize` param, defaulting to
      `false` so the existing raw-mean output stays byte-identical for every
      caller that predates it.
- [x] `brain::EmbeddingPipeline` resolves LFM2 as a third backend
      (`crates/lfm2/src/spec.rs`'s new `Lfm2Spec`, the same seam
      `qwen3::spec::Qwen3Spec` gives Qwen3). A local `.safetensors` file
      routes by `ModelCard.family`; a hub id tries Qwen3's resolver first
      (the pre-existing default) and falls back to LFM2's only on a genuine
      `Missing`. The SDK's LFM2 arm always L2-normalizes (a different
      default from the capability action above, deliberately - the SDK
      surface has no pre-existing caller to stay byte-identical for, and
      matching Qwen3's own always-normalized SDK contract is what makes
      `Embedding::cosine_similarity` mean the same thing regardless of which
      backbone resolved).

## Explicitly out of scope for this track

- Token-level (grammar/schema) constrained decoding. `samples/text/retrieve-json`
  validates a decoder's JSON output after the fact and retries; the real hook
  point is named (`qwen3::sample::sample_logits`, following the
  `model::serve::apply_no_repeat_ngram` precedent) but not built.
- An ANN/vector index. `samples/text/retrieve`'s flat scan is the right size
  for a sample; a real corpus needs a real index (HNSW, IVF, ...) over the
  same `Embedding` vectors this SDK already produces - a data-structure
  concern, not a model capability this crate should own.
- A `Domain::Embedding` architecture-vocabulary variant. Both backbones
  already have a registered domain (CLIP's `Vision`, Qwen3's `Text`); adding
  one would be a new capability-vocabulary concept for no behavior gained.
- Matryoshka-style nested-dimension re-projection. `EmbeddingTrainer`'s
  projection head is deliberately same-dimension (a learned refinement, not
  a truncation scheme).

## Not yet done

- [ ] An SDK-level training loop over LFM2's new seeded backward - neither
      `brain::EmbeddingTrainer` nor a sibling type yet unfreezes and steps a
      live `lfm2::Lfm::new_train` through `seed_buf`/`backward_seeded` with
      an InfoNCE (or other) objective; today `EmbeddingTrainer` only trains a
      frozen-embedding projection head, over either backbone. Needed before a
      SAMPLE can demonstrate full-encoder fine-tuning at all: samples may
      only depend on `brain`, never `brain-lfm2` directly.
- [ ] LFM2 at 32768 tokens is unvalidated extrapolation to 4x its native
      8192-token training extent - real quality there needs continued
      pretraining, not just the RoPE math being correct (which is tested).
- [ ] The Qwen3-Embedding-0.6B int8 paged KV cache: fp32 is roughly 7.5 GiB
      at 32768 tokens for a 0.6B model; the existing int8 paged-KV serving
      path (`crates/qwen3/src/serve.rs`) was not wired into the `embed`
      action, which stays fp32 and decode-only (no paged batching).
