# sample: embedding/embed-document

Long-context embeddings over brain's D-Bus interface (LFM2.5-Encoder).
`embed_document.py` drives the generic `com.swedishembedded.Brain1` surface
(protocol in `samples/python/dbus/brain-dbus/README.md`): the document
travels **as a file descriptor** (sealed memfd) and the per-token hidden
states come back the same way - no bytes marshalled through D-Bus.
Concurrency goes through brain's residency executor: equal-length requests
are batched into one true batched forward on a device lane; the tokenizer
runs on the dispatcher thread so lane time is pure forward.

```bash
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/embedding/embed-document/embed_document.py --input README.md --concurrent 4
```

Nothing to pre-fetch: `--model` defaults to `LiquidAI/LFM2.5-350M`, a
fully-qualified `<vendor>/<repo>` reference - brain's transparent auto-fetch
downloads and converts it on the first request that names it (that first
call is as slow as the cold fetch; every one after is instant). Point
`--model` at `LiquidAI/LFM2.5-230M` for the smaller encoder, or `brain/mock`
for a weight-free smoke test.

Prefer an already-converted local checkpoint instead? Set `BRAIN_LFM2`/
`BRAIN_LFM2_TOKENIZER` before `brain serve --dbus` and pass `--model
brain/lfm2` - the env-loaded-checkpoint fallback (the `brain/` table),
unchanged from before auto-fetch existed.

## What it demonstrates

* The instance key is the **exact token length**: identical-length documents
  share one built graph and batch together; a new length builds (weight
  upload + graph) once and stays resident under the LRU/budget machinery.
* Bidirectional attention makes unmasked token-padding unsound, so requests
  are never padded with pad tokens - batch tails repeat a real sequence
  instead (exact results, some redundant compute).
* Expected output shape (numbers are hardware-dependent):

  ```
  document: doc.txt (11 KiB)
  warm-up:
    [warm] 3547 tokens x 1024 dim over memfd (14188 KiB) in ...s; mean[0]=-0.7685
  4 concurrent request(s):
    [req0] ... [req3]   <- equal completion times = they ran in batched groups
  wall ...s for 4 requests (batching/lanes = wall < sum)
  ```

## What it needs

- Nothing pre-fetched, or `BRAIN_LFM2`/`BRAIN_LFM2_TOKENIZER` for a local
  checkpoint.
- `jeepney` - `pip install -e brain-py`.

## Options

| flag / env | default |
|---|---|
| `--input FILE` | *required* - text file to embed (long context welcome) |
| `--concurrent N` | `1` - issue N identical requests concurrently |
| `BRAIN_LFM2_BATCH` | `2` - batched-forward slots per instance |
| `BRAIN_DEVICE` | unset - which compute is schedulable |

The benchmark twin of this sample is `make perf/lfm` - the same executor
measured by the brain perf suite (`brain perf run sweep --target
lfm:<weights>:<tokenizer> --input 8192 --ladder 1,2,4,8`).
