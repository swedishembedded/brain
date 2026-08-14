# brain-py

The Python client for the **`brain`** edge-AI runtime. brain serves a
**capability model** — every model advertises *actions* (`generate`, `embed`,
`text2image`, `transcribe`, …) that take typed params and named binary blobs and
return an **outcome** (scalar outputs + output blobs). brain-py speaks that model
over two transports that share **one** high-level API:

| Transport | Class | Backend | Default? |
|-----------|-------|---------|----------|
| **D-Bus** | `BrainDBus` | `com.swedishembedded.Brain1` (`brain serve --dbus`), bulk data as **fds** | **yes** |
| JSONL/stdio | `BrainStdio` | a `brain serve --stdio` subprocess, correlated by `req_id`, blobs base64'd | no |

Because both implement the same `run` / `subscribe` primitives and the same
convenience wrappers, you can switch transport without rewriting anything.

## Install

```bash
pip install -e brain-py     # pulls in pillow + jeepney (both required)
```

`jeepney` is a **required** dependency now that D-Bus is the default transport.
The JSONL transport adds no dependency of its own. You also need the `brain`
binary; `BrainStdio` locates it via `brain_bin=`, `$BRAIN_BIN`,
`./target/release/brain`, `./target/debug/brain`, then `brain` on `$PATH`.

## Quick start

```python
from brain_py import Brain

# D-Bus by default — connects to com.swedishembedded.Brain1 (brain serve --dbus)
with Brain() as brain:
    print(brain.models())                                  # discovery
    print(brain.generate(prompt="hello", model="mock"))    # text
    vec = brain.embed("hello world", model="mock")         # embedding vector
    img = brain.text2image("a red cube", model="mock")     # -> PIL Image
    text = brain.transcribe(pcm_f32le_16k, model="nemotronasr")  # live ASR
```

Select the JSONL/stdio transport explicitly — the API is identical:

```python
with Brain(transport="jsonl") as brain:   # spawns `brain serve --stdio`
    print(brain.generate(prompt="hello", model="mock"))
```

`Brain(...)` is a factory: `transport="dbus"` (default) builds `BrainDBus`,
`transport="jsonl"` builds `BrainStdio`. Construct either class directly if you
prefer. (`BrainClient` remains as an alias of `BrainStdio`.)

## The capability API (both transports)

Everything is built on two primitives, plus thin convenience wrappers over them
(see `brain_py/base.py`):

```python
out = brain.run(model, action, params, blobs={"image": raw_bytes})   # one-shot
#   -> Outcome(outputs: dict, blobs: dict[str, bytes], meta: dict)

out = brain.subscribe(model, action, params, on_progress=cb)         # streaming
#   cb(step, total, message) fires as it advances; returns the final Outcome

brain.models()                    # served model names
brain.model_for("generate")       # first model advertising an action
brain.generate(prompt=…, messages=…, on_progress=…) -> str
brain.chat("hi") -> str           # sugar over generate
brain.embed(text) -> list[float]
brain.text2image(prompt, width=…, height=…) -> PIL.Image   # alias: image()
```

`Outcome` materialises blobs to `bytes` for you (over D-Bus the returned fds are
read + closed; over JSONL the base64 is decoded). Helpers: `out.text()`,
`out.image("image")`, `out.get(key)`.

`transcribe(pcm, …)` is **D-Bus only** (`StreamTranscribe` over a live pipe); it
returns the transcript and calls `on_segment(text, final)` as windows decode.

### D-Bus low-level (fds, zero-copy)

`BrainDBus` also exposes the raw fd path: `run_fds(...) -> RunResult` (output fds
left open — read them with `read_fd`), `stream_frames(...)` /
`stream_frames_with_job(...)`, `cancel(job)`, `stream_transcribe(...)`,
`version()`, `active_jobs()`, `stats()`, `stats_snapshot()`. `sealed_memfd(bytes)`
builds an input fd.

### JSONL/stdio extras

`BrainStdio` keeps the `brain serve --stdio` legacy verbs: `detect(image)` (object
detection), `converse(text)` (the `user_text` → `brain_text_chunk` chat path,
distinct from the capability `generate`), and — with `BrainStdio(forecast=True)`
— `forecast(...)`, `backtest(...)`, `capabilities()`. It can also `connect()` to
a long-lived `brain forecast serve` socket.

## Tests

```bash
python -m pytest brain-py/tests -q
```

* `tests/test_dbus.py` — bus-free: the memfd/fd plumbing plus the
  transport-agnostic capability layer (`Outcome` + the convenience wrappers)
  exercised against a fake transport. **No server, no D-Bus needed.**
* `tests/test_client.py`, `tests/test_forecast.py` - drive a real `brain serve --stdio`
  subprocess (fake detector / echo model); **skipped** when the `brain` binary is
  absent.

A full live D-Bus round-trip runs `BRAIN_MOCK=1 BRAIN_DEVICE=cpu brain serve
--dbus` (see `tests/e2e/scheduler.bats` in the repo).
```
