# CAM++ (speaker encoder, component)

The 192-d x-vector speaker encoder [CosyVoice](cosyvoice.md) uses for
zero-shot voice cloning - turns a reference clip's 80-dim kaldi-style fbank
into the x-vector CosyVoice's flow model conditions its CFM decoder on. Not
independently servable: it has no capability manifest or CLI verb of its own,
reached only as part of CosyVoice's own actions.

Package: `brain-campplus`.

## Status

Import (two-way coverage against the released `campplus.onnx`, 617
initializers) and forward are implemented (`crates/campplus/src/{config,import,model}.rs`).
Forward parity vs the real ONNX checkpoint, replayed through `onnxruntime`:
cosine 1.0000000000, `rel_l2` 2.3e-6, `max_abs` 5.3e-6
(`crates/campplus/tests/parity.rs`, gated on `BRAIN_CAMPPLUS_DIR`/`testdata/golden/cosyvoice`).
No training/gradcheck path yet - this component is inference-only until a
future milestone adds one.
