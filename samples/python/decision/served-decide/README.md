<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: decision/served-decide

Typed decisions from a decision model you do not link: plain HTTP, Python
standard library only, against a running `brain serve`. The model answers
`choice`, `score` and `noul` questions about one state with a calibrated
distribution over options the **request** supplies.

```bash
brain pull convaiinnovations/laya
brain serve --openai 8788 --api-keys-out /tmp/keys.json &
python3 samples/python/decision/served-decide/decide_client.py \
    --base-url http://127.0.0.1:8788 --api-key "$KEY"
```

With no server, or a server holding no decision model, it exits **77** and says
what to do (start `brain serve`, or `brain pull convaiinnovations/laya`); it
needs no weights of its own.

## What it shows

1. `models` - `POST /v1/run` on `brain/decision` lists the decision models the
   server has, each `ready`, `not-installed` or `incompatible`, and whether a
   validation is recorded for it. It loads nothing.
2. `POST /v1/systemone` - the shared wire shape: questions keyed by name,
   probabilities keyed by option, `usage.output_tokens` always 0.
   `"model": "auto"` picks a configured model, else a validated Laya model,
   else a validated `decide` model - and when none qualifies the client asks
   an installed one by name, because trusting an unvalidated model is the
   operator's call.
3. `POST /v1/run` on the model's own id, action `decide` - the full ranked
   distribution with every candidate's id and position, the backend, latency in
   milliseconds, and whether the state was truncated to the model's context.

The failure body of every call carries a machine-readable `code`
(`not_installed`, `failed_to_load`, `unsupported_capability`, `timeout`,
`invalid_request`, `backend_failure`).

---

Swedish Embedded AB implements typed decision endpoints - a model behind a
schema a service can rely on, rather than free text a caller has to parse and
hope about - for its clients. If your team needs judgment in the loop with a
contract around it, you can procure our services by sending an email to
info@swedishembedded.com.
