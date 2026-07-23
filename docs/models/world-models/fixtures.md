# Parity fixtures — regeneration & provenance

brain has no Python in the build/test path, and fixture binaries are NOT
committed to git. Reference activations from the original world-model
implementations are regenerated on demand:

```bash
make wm-fixtures   # -> crates/wm-diamond/tests/fixtures/diamond/ (gitignored)
```

- Producer scripts live in `scripts/parity-dump/` and run the ORIGINAL
  implementation from `/data/workspace/resources/world-models/repos/<repo>`
  (read-only reference material) on a tiny fixed-seed config/input, dumping
  per-module weights, inputs, activations, and output as raw little-endian
  f32 blobs plus a `manifest.json` carrying shapes, per-blob SHA-256, and
  provenance (reference repo commit, torch version, seed, command line).
- Requirements: python3 + torch (system torch works; omegaconf-free — the
  scripts import the reference model modules directly, bypassing the repos'
  heavyweight `__init__` chains).
- Determinism: fixed seeds; identical torch versions reproduce bit-identical
  blobs (manifest SHA-256s verify this). Across torch versions, tiny fp32
  drift is possible — brain's parity tests compare with tolerances
  (~1e-4 at the first block, relaxing with depth), so regeneration on a
  different torch is fine.
- Tests that consume fixtures SKIP with a clear message when the fixture
  directory is absent (run `make wm-fixtures` first); CI without torch stays
  green.

## Current producers
- `scripts/parity-dump/diamond.py` — tiny random-weight DIAMOND `InnerModel`
  (img_ch 3, ctx 2, cond 16, depths [1,1], ch [8,8], mid attention, 4
  actions, 8x8; seed 7; zero-init params re-randomized so those paths are
  exercised). ~245 KiB, 239 files.
