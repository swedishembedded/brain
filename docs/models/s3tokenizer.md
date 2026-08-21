# S3Tokenizer (speech tokenizer, component)

The supervised-semantic FSQ speech tokenizer [CosyVoice](cosyvoice.md)
conditions its LM on - turns 16 kHz reference audio into 25 Hz FSQ token ids
(`3^8 = 6561` codes) for zero-shot voice cloning. Two codebook versions, v2
(CosyVoice 2) and v3 (CosyVoice 3), sharing the same codebook size. Not
independently servable: it has no capability manifest or CLI verb of its own,
reached only as part of CosyVoice's own actions.

Weights env vars, one per codebook version, resolved via
[CosyVoice](cosyvoice.md)'s consumption of this crate (not yet read by any
code - reserved alongside the architecture id): `BRAIN_S3TOKENIZER_V2`,
`BRAIN_S3TOKENIZER_V3`.

Package: `brain-s3tokenizer`.
