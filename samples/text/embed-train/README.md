# sample: text/embed-train

Contrastively fine-tune a projection head over frozen embeddings with
`brain::EmbeddingTrainer`, and report retrieval quality (recall@1 on a
held-out split) BEFORE and AFTER training.

```bash
make samples/text/embed-train/run ARGS="--weights /models/qwen3-embedding-0.6b.safetensors \
                                         --tokenizer /models/qwen3-embedding-0.6b/tokenizer.json \
                                         --steps 200"
```

## What it demonstrates

* **The before/after number is the deliverable.** A training sample that
  only prints a falling loss proves nothing about whether retrieval
  actually improved - `EmbeddingTrainer`'s own finite-difference test
  already proves the loss math is correct; that is a different claim from
  "and this made retrieval better." This sample measures recall@1 on a
  held-out split with the SAME two calls, once before training and once
  after, so the comparison is apples to apples.
* **The backbone stays frozen; only the head trains.** Every embedding is
  computed ONCE, up front (`pipe.embed_batch(...)`, four calls total), and
  cached as plain `Embedding` values. `EmbeddingTrainer::step` never touches
  the backbone again - see its own module doc for why (no engine crate has
  a seeded backward pass for either embedding backbone yet). This is what
  keeps training tractable at a real embedding checkpoint's cost: the
  expensive forward runs once per document, not once per training step.
* **A small, fictional, self-contained dataset - no fetch script.** Eight
  topics, each with a training phrasing and a differently-worded held-out
  phrasing of the same fact, built into `main.rs` (`TOPICS`). Fictional so
  there is no risk of the underlying model's own pretraining having
  memorized the exact pairing, which would make recall@1 measure
  memorization instead of what the projection head generalizes to. Swap in
  a real dataset (your own query/passage pairs) for a real result.

## What the numbers mean (and don't)

With a real, trained embedding checkpoint, recall@1 should measurably rise
after training - the projection head is learning to pull each topic's
query and answer phrasing closer together relative to the other seven
topics' in-batch negatives. With a tiny, randomly-initialized checkpoint
(the CI / no-GPU path this repo's own tests exercise), the embeddings carry
no real semantic signal to begin with, so recall@1 before and after both
land near chance (`1/8` = 12.5%) and training may not move the number at
all - that is expected, not a bug. What a run against a toy checkpoint
proves is that the embed -> cache -> train -> re-embed -> re-measure loop
runs correctly end to end, not that the projection head learned anything
useful.

## Usage

```text
--weights BASE     checkpoint path, model directory, or vendor/repo ref
--tokenizer FILE   tokenizer.json (required for a Qwen3 .safetensors checkpoint)
--capacity N       KV-cache context to build for (default 2048)
--steps N          training steps (default 200)
--lr X             Adam learning rate (default 0.05)
--temperature X    InfoNCE softmax temperature (default 0.05)
--seed S           projection head init seed (default 1)
```

## What it needs

* A Qwen3-Embedding-shaped checkpoint (`Qwen/Qwen3-Embedding-0.6B` or
  similar) plus its `tokenizer.json`.
* A GPU for anything beyond a toy fixture - the backbone still runs a real
  forward pass per embedding call, even though the head trains on the host.

## Why it is a sample rather than a `brain` subcommand

`EmbeddingTrainer` is architecture-agnostic library machinery, the same
reasoning `samples/study/document/README.md` and `samples/text/retrieve/README.md`
give for their own surfaces: fine-tuning retrieval on a caller's own data is
a function a caller's own service calls, not a CLI verb the engine would
have to own a dataset format and evaluation policy for.
