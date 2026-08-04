# Long-context embeddings over D-Bus (LFM2.5-Encoder)

`embed_document.py` drives brain's generic D-Bus interface
(`com.swedishembedded.Brain1`, see `examples/dbus/README.md` for the protocol):
the document travels **as a file descriptor** (sealed memfd) and the per-token
hidden states come back the same way — no bytes marshalled through D-Bus.
Concurrency goes through brain's residency executor: equal-length requests are
batched into one true batched forward on a device lane; the tokenizer runs on
the dispatcher thread so lane time is pure forward.

## Run

```bash
# deps: jeepney (pip install -e brain-py)
dbus-run-session -- bash -c '
  BRAIN_LFM=out/lfm-230m.weights \
  BRAIN_LFM_TOKENIZER=/path/to/LFM2.5-Encoder-230M/tokenizer.json \
  ./target/release/brain serve --dbus & sleep 2
  python3 examples/embedding/embed_document.py --input README.md --concurrent 4'
```

Environment knobs: `BRAIN_LFM_BATCH` (batched-forward slots per instance,
default 2), `BRAIN_DEVICE` (which compute is schedulable).

Expected output shape (numbers are hardware-dependent):

```
document: doc.txt (11 KiB)
warm-up:
  [warm] 3547 tokens x 1024 dim over memfd (14188 KiB) in …s; mean[0]=-0.7685
4 concurrent request(s):
  [req0] … [req3]   ← equal completion times = they ran in batched groups
wall …s for 4 requests (batching/lanes = wall < sum)
```

## Notes

- The instance key is the **exact token length**: identical-length documents
  share one built graph and batch together; a new length builds (weight upload
  + graph) once and stays resident under the LRU/budget machinery.
- Bidirectional attention makes unmasked token-padding unsound, so requests are
  never padded with pad tokens — batch tails repeat a real sequence instead
  (exact results, some redundant compute).
- The benchmark twin of this example is `make perf/lfm` — the same executor
  measured by the brain perf suite (`brain perf run sweep --target
  lfm:<weights>:<tokenizer> --input 8192 --ladder 1,2,4,8`).
