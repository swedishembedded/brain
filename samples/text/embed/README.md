# sample: text/embed

Embed text with `brain::EmbeddingPipeline`, and print a pairwise cosine
similarity matrix over the results.

```bash
make samples/text/embed/run ARGS="--weights /models/qwen3-embedding-0.6b.safetensors \
                                   --tokenizer /models/qwen3-embedding-0.6b/tokenizer.json \
                                   'a whale submarine' 'a submarine shaped like a whale' 'tax law in 1998'"
```

## What it demonstrates

* **One SDK type, two backbones.** `EmbeddingPipeline` resolves either to
  CLIP's text towers (`vision` feature, unaffected by this sample) or to a
  Qwen3-Embedding-shaped checkpoint (`text` feature, what this sample links):
  last-token pooled, L2-normalized, over a real 32768-token native context. A
  literal local checkpoint path always resolves to Qwen3 - see the SDK's own
  module doc for why a bare file can never be CLIP's release-directory shape.
* **A real long document, in ONE forward pass.** `--document FILE` embeds the
  whole file's contents through the decode-only KV-cache path
  (`Qwen::prefill`), not a batched `O(T^2)` forward - the only shape that
  stays sane at a real 32k-token document. Pass `--capacity 32768` to size
  the pipeline for it.
* **`Embedding::cosine_similarity` is the whole retrieval primitive this SDK
  ships.** Everything past a pairwise score - an index, a corpus, top-k - is
  a caller's own data structure; see `samples/text/retrieve` for the
  smallest one that is still real.

## Usage

```text
--weights BASE     checkpoint path, model directory, or vendor/repo ref
--tokenizer FILE   tokenizer.json (required for a Qwen3 .safetensors checkpoint)
--capacity N       KV-cache context to build for (default 2048; pass 32768
                    to embed a real long document with --document)
--tower NAME       CLIP tower ("clip_l"/"openclip_bigg"), when --weights
                    resolves to CLIP instead of Qwen3
--document FILE    embed this file's full contents too, in ONE forward pass
TEXT...            strings to embed and compare pairwise (at least one,
                    unless --document alone is given)
```

## What it needs

* A Qwen3-Embedding-shaped checkpoint (any Qwen3 decoder works; a checkpoint
  actually trained for embedding, like `Qwen/Qwen3-Embedding-0.6B`, is what
  produces meaningful similarity scores) plus its `tokenizer.json` beside it,
  OR a released CLIP/SDXL-layout directory for the `--tower` path.
* A GPU for anything beyond a toy fixture.

## Why it is a sample rather than a `brain` subcommand

Embedding is architecture-agnostic library machinery
(`brain::EmbeddingPipeline`), the same reasoning
`samples/study/document/README.md` gives for `DocumentStudy`: the thing that
makes it useful to a caller is that it is *embeddable* - code that wants to
compute a similarity score reaches for a library call, not a subprocess.
