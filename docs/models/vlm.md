# Vision-Language Models

Three image-understanding architectures share one shape (vision encoder ->
connector/projector -> autoregressive text decoder, image embeddings
spliced into the decoder's residual stream) and one validation ladder. Each
has its own reference page with getting-started commands and options.

| Architecture | Solves | Status |
|---|---|---|
| [FastVLM](fastvlm.md) | dedicated image captioning - fast, single-purpose | servable |
| [Qwen3-VL](qwen3vl.md) | general image + text -> text - ask a question, not just "describe this" | servable |
| [Moondream 3](moondream3.md) | a third architecture, SigLIP + MoE decoder | not yet servable |

Reach for FastVLM when you want a caption and nothing else; reach for
Qwen3-VL when you need to prompt about an image's content.
