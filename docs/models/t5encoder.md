# T5-XXL encoder (not yet servable)

The text encoder [FLUX.1](flux1.md) conditions on: a bidirectional
encoder-only T5 (RMSNorm, no bias, learned relative-position bucket bias
shared by every layer, gated-GELU FFN).

This is a real, verified port - imported (219 -> 171 tensors, two-way
covered), forward-parity-gated (42/42 stages, worst cosine 0.9999999992),
and has a full hand-written backward gated by gradcheck - but T=512 (the
length FLUX.1 actually uses) is untested, there's no tokenizer, and the
serving contract is deferred. Not something you can run as a model today.

Package: `brain-t5encoder`.
