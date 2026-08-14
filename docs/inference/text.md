# Text: chat, tool-calling, and language models from scratch

brain runs local language models for chat and tool-calling, gives you
architectures to train a decoder from scratch or study how one generalizes,
and produces text embeddings for search and similarity. These are separate
model ids with their own `brain <model> ...` commands; pick the one that
matches what you're trying to do.

## Capabilities

### Chat and tool-calling - `brain/qwen3`

brain's flagship served LLM: a dense instruct/chat decoder you can run
inference against, train from scratch, or LoRA-finetune into a named
adapter. It's the model behind brain's OpenAI/Anthropic/OpenRouter-compatible
HTTP endpoints, with concurrent request batching and a paged KV cache. See
[the Qwen3 page](../models/qwen3.md) for the model, and
[the HTTP API page](../using/http-api.md) for the served API surface.

### An alternative decoder architecture - `brain/glm`

A from-scratch GLM-5.2 decoder: compressed low-rank attention plus a
mixture-of-experts routing layer (a sparse set of expert MLPs selected per
token, with one always-on shared expert). Reach for it to train, finetune,
evaluate, or run inference against a from-scratch MoE decoder, or to import
official GLM-5.2 weights. See [the GLM page](../models/glm.md).

### A simple baseline decoder - `brain/gpt`

A dense, decoder-only Transformer built to nanoGPT parity - brain's simplest
reference decoder, with no grouped-query attention, MoE, quantization, or
paged serving to get in the way. Reach for it when you want a straightforward
train/evaluate/generate loop on your own dataset, for learning or
experimenting with training from scratch. See
[the GPT page](../models/gpt.md).

### Text embeddings

For turning text into vectors for search or similarity: `brain/lfm2` is a
dedicated bidirectional text encoder (also does cloze-style mask-filling),
and `brain/clip` produces text embeddings from the same encoder its image
tower shares, for text-image comparison. See
[the LFM2.5 page](../models/lfm.md) and
[the CLIP page](../models/clip.md).

### Studying routing and generalization - the toy Sparse MoE

A small sparse Mixture-of-Experts decoder trained from scratch on a
synthetic task with a known ground truth, purpose-built to study
memorization vs. generalization rather than to serve as a production model.
See [the Sparse MoE page](../models/moe.md).

### Sequence-to-sequence - the encoder-decoder architecture

A general encoder-decoder Transformer for tasks shaped like translation -
mapping one token sequence to another, rather than continuing a single
stream. See [the seq2seq page](../models/seq2seq.md).
