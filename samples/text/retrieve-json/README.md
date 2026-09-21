# sample: text/retrieve-json

Retrieval-augmented JSON generation: embed a query, retrieve the top-k
passages from a corpus (the same exact flat scan `samples/text/retrieve`
runs), build a prompt from them, and ask a decoder for a validated
`{"answer", "sources"}` JSON object - retrying a bounded number of times
when the completion is not valid JSON of that shape.

```bash
make samples/text/retrieve-json/run ARGS="--embed-weights /models/qwen3-embedding-0.6b.safetensors \
                                           --embed-tokenizer /models/qwen3-embedding-0.6b/tokenizer.json \
                                           --gen-weights /models/qwen3-4b-instruct.safetensors \
                                           --gen-tokenizer /models/qwen3-4b-instruct/tokenizer.json \
                                           --corpus passages.txt \
                                           --query 'what does the report say about Q3 revenue'"
```

## What it demonstrates

* **An embedding backbone feeding a decoder, in one program.** Two
  `brain::EmbeddingPipeline`/`brain::TextGenerationPipeline` instances, both
  built from the `text` surface - retrieval decides WHAT the decoder sees,
  the decoder decides WHAT TO SAY about it, and the boundary between the two
  is exactly the numbered passage list in the prompt.
* **brain has no grammar-constrained or JSON-schema-constrained decoding,
  and this sample does not pretend otherwise.** It prompts for JSON, parses
  whatever comes back (recovering from a code-fence or a stray sentence
  around the object), and validates the parsed value against an exact
  `{"answer": string, "sources": [int]}` shape - the same prompt-then-validate
  approach brain's own `tools`/`tool_choice` function-calling support uses.
  A malformed completion is retried, with a fresh seed, up to `--max-attempts`
  times; exhausting them is reported honestly, not silently swallowed.
* **Where a real constraint would go, named rather than built.** A
  token-level JSON grammar would hook into `qwen3::sample::sample_logits`,
  the one place the full `[vocab]` logits row is visible on the host before
  sampling - the same seam `model::serve::apply_no_repeat_ngram` already
  uses for an in-place logits mask. That is real, unbuilt work; this sample
  exists to make its absence a scoped, honest thing you can see rather than
  something quietly implied to already work.

## Usage

```text
--embed-weights BASE   embedding checkpoint path, directory, or vendor/repo ref
--embed-tokenizer FILE tokenizer.json for the embedding checkpoint
--embed-capacity N     embedding pipeline KV-cache context (default 2048)
--gen-weights BASE     generation checkpoint path, directory, or vendor/repo ref
--gen-tokenizer FILE   tokenizer.json for the generation checkpoint
--gen-capacity N       generation pipeline context budget (default 4096)
--corpus FILE          one passage per line
--query TEXT           the question to answer
--top-k N              how many passages to retrieve into the prompt (default 4)
--max-attempts N       generation attempts before giving up (default 3)
--instruction TEXT     the Qwen3-Embedding query instruction (default
                        "Given a query, retrieve relevant passages")
```

`--embed-weights` and `--gen-weights` are deliberately separate: a
checkpoint trained for embedding (`Qwen/Qwen3-Embedding-0.6B`) and one
trained for chat (`Qwen/Qwen3-4B-Instruct-2507`, or any registered
architecture `brain::TextGenerationPipeline` loads) are different files, and
this sample never assumes they are the same one.

## What it needs

* A Qwen3-Embedding-shaped checkpoint plus its `tokenizer.json`.
* A Qwen3-family instruct/chat checkpoint plus its `tokenizer.json` and a
  `tokenizer_config.json` carrying a chat template (the same requirement
  `samples/study/document` documents).
* A GPU for anything beyond a toy fixture - both pipelines run a real
  forward pass.

## Why it is a sample rather than a `brain` subcommand

Both pipelines it composes are architecture-agnostic library machinery
(`brain::EmbeddingPipeline`, `brain::TextGenerationPipeline`), the same
reasoning `samples/study/document/README.md` and `samples/text/retrieve/README.md`
give for their own surfaces: a retrieval-augmented answer is a function call
a caller's own service makes, not a CLI verb the engine would have to own
the prompt format and retry policy for.
