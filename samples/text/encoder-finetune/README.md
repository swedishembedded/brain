# sample: text/encoder-finetune

Full-encoder contrastive fine-tuning of LFM2.5-Encoder with
`brain::EncoderFineTuner`, and report retrieval quality (recall@1 on a
held-out split) BEFORE and AFTER training - via `brain::EmbeddingPipeline`,
once against the original checkpoint and once against the fine-tuned one this
sample writes out.

```bash
make samples/text/encoder-finetune/run ARGS="--weights /models/lfm2-encoder.safetensors \
                                              --tokenizer /models/lfm2-encoder/tokenizer.json \
                                              --steps 200"
```

## What it demonstrates

* **The checkpoint's own weights change, not a bolted-on head.**
  `samples/text/embed-train` freezes the backbone and trains only a small
  projection head on top of embeddings computed once, up front. This sample
  is the other half of that story: `EncoderFineTuner` re-runs LFM2's own
  forward AND backward every step, driving `lfm2::model::Lfm`'s seeded
  backward pass (`Lfm::seed_buf`/`backward_seeded`, gradient-checked directly
  in `crates/gradcheck/src/lfm2_seeded.rs`) with the exact same InfoNCE
  objective `EmbeddingTrainer` uses.
* **The before/after number is the deliverable**, the same reasoning
  `samples/text/embed-train/README.md` gives: a training sample that only
  prints a falling loss proves nothing about whether retrieval actually
  improved.
* **Every training text is truncated to a fixed `--seq-len`.** LFM2's
  bidirectional attention has no padding mask, so `EncoderFineTuner` cannot
  mix sequence lengths in one training batch the way a causal decoder can -
  see that type's own module doc. This sample truncates every training
  phrase to `--seq-len` tokens (default 8); the before/after EVALUATION calls
  go through `EmbeddingPipeline` instead, which embeds each held-out phrase
  at its own natural length.
* **A small, fictional, self-contained dataset - no fetch script.** The same
  eight topics `samples/text/embed-train` uses, for the same reason
  (fictional, so recall@1 measures generalization, not the checkpoint's own
  pretraining having memorized the pairing).

## What the numbers mean (and don't)

With a real LFM2.5-Encoder checkpoint, recall@1 should measurably rise after
training. With a tiny, randomly-initialized checkpoint (the CI / no-GPU path
this repo's own tests exercise), the embeddings carry no real semantic signal
to begin with, so recall@1 before and after both land near chance (`1/8` =
12.5%) - that is expected, not a bug. What a run against a toy checkpoint
proves is that the forward -> InfoNCE -> seeded-backward -> AdamW -> save ->
re-embed -> re-measure loop runs correctly end to end, not that the encoder
learned anything useful.

## Usage

```text
--weights FILE     LFM2.5-Encoder checkpoint (a carded .safetensors, e.g.
                    from `brain lfm2 import`)
--tokenizer FILE   tokenizer.json (LFM2 carries no embedded tokenizer)
--out FILE         where to write the fine-tuned checkpoint (default
                    lfm2-finetuned.safetensors)
--seq-len N        fixed training sequence length in tokens - every
                    training phrase is truncated to this length (default 8)
--capacity N       EmbeddingPipeline context to build for when measuring
                    recall@1 before/after (default 64)
--steps N          training steps (default 200)
--lr X             AdamW learning rate (default 3e-5 - a full-encoder
                    fine-tune needs a much smaller step than a projection
                    head, the same reason `brain lfm2 finetune` defaults to
                    this value)
--temperature X    InfoNCE softmax temperature (default 0.05)
```

## What it needs

* An LFM2.5-Encoder checkpoint (imported via `brain lfm2 import`, which
  writes the `ModelCard` `EmbeddingPipeline`'s LFM2 routing and
  `EncoderFineTuner::save`'s own output both rely on) plus its
  `tokenizer.json`.
* A GPU - both the forward and the backward pass run for real, every step,
  unlike `samples/text/embed-train`'s frozen-backbone loop.

## Why this is a different objective from `brain lfm2 finetune`

`brain lfm2 finetune` continues the checkpoint's own masked-LM pretraining
objective (predict masked tokens). This sample trains a DIFFERENT objective
entirely - symmetric InfoNCE over `(query, answer)` pairs - which is what
actually optimizes retrieval quality; an MLM loss falling is not the same
claim as embeddings that retrieve better, the same distinction
`samples/text/embed-train/README.md` draws for its own frozen-head training
loop.

## Why it is a sample rather than a `brain` subcommand

`EncoderFineTuner` is library machinery, the same reasoning
`samples/text/embed-train/README.md` gives for `EmbeddingTrainer`: fine-tuning
retrieval on a caller's own data is a function a caller's own service calls,
not a CLI verb the engine would have to own a dataset format and evaluation
policy for.
