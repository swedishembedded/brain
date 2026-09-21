# sample: text/retrieve

Exact, brute-force top-k retrieval over a text corpus: `brain::EmbeddingPipeline`
embeds every passage once, embeds the query, and this program scores every
passage against it with `Embedding::cosine_similarity` and sorts.

```bash
make samples/text/retrieve/run ARGS="--weights /models/qwen3-embedding-0.6b.safetensors \
                                      --tokenizer /models/qwen3-embedding-0.6b/tokenizer.json \
                                      --corpus passages.txt \
                                      --query 'what does the report say about Q3 revenue'"
```

## What it demonstrates

* **This is a flat scan, not an ANN index - said plainly, not implied
  otherwise.** Every passage is scored against the query, every call:
  `O(n)` per query, `O(n)` memory for the index. That is the right, honest
  scope for a sample and for a corpus of a few thousand passages; past that,
  the right next step is a real approximate-nearest-neighbor index (HNSW,
  IVF, ...) over the SAME `Embedding` vectors this program already
  produces - not something `brain` ships, since an ANN index is a data
  structure over vectors, not a model capability.
* **Asymmetric retrieval, as the underlying model expects it.** Passages
  carry no instruction; the query does (`--instruction`, defaulting to
  "Given a query, retrieve relevant passages" - the Qwen3-Embedding
  convention). Embedding a query and a passage the SAME way measurably hurts
  retrieval quality on an instruction-tuned embedding model, so this sample
  does not offer a "no instruction" shortcut for the query side.
* **The corpus is embedded once, not once per query.** `embed_batch` runs
  the whole corpus through one call; a real service would cache `index` and
  only re-embed on ingest, not on every request - this sample's `main`
  recomputes it every run purely because it has no place to persist it.

## Usage

```text
--weights BASE     checkpoint path, model directory, or vendor/repo ref
--tokenizer FILE   tokenizer.json (required for a Qwen3 .safetensors checkpoint)
--capacity N       KV-cache context to build for (default 2048)
--corpus FILE      one passage per line
--query TEXT       the query string
--top-k N          how many results to print (default 5)
--instruction TEXT the Qwen3-Embedding query instruction (default "Given a
                    query, retrieve relevant passages")
```

## What it needs

* A Qwen3-Embedding-shaped checkpoint (`Qwen/Qwen3-Embedding-0.6B` or
  similar) plus its `tokenizer.json` beside it.
* A GPU for anything beyond a toy fixture.

## Why it is a sample rather than a `brain` subcommand

Retrieval built on `brain::EmbeddingPipeline` is exactly the kind of
architecture-agnostic library machinery `samples/study/document/README.md`
already argues belongs in the SDK, not the engine's own command line: the
useful form of "search my corpus" is a function a caller's own service
calls, not a CLI verb `brain` would have to own the index format for.
