<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# json - a decision endpoint on a pipe: JSON in, JSON out

A decision model returns a **calibrated distribution over options the caller
supplies at request time**, instead of text. Every other sample here spells its
own question out in Rust. This one spells out nothing: the state, the
questions, their types and their options all arrive at run time as JSON, and
the answers go back as JSON.

```bash
echo '{"state": "my replacement card never arrived",
       "questions": {"intent": {"type": "choice",
                                "instructions": "what is this about",
                                "criteria": {"card": "cards and deliveries",
                                             "fx": "exchange rates"}}}}' \
  | ./target/release/sample-decision-json \
  | jq -r .answers.intent.choice
```

(The binary directly, not `make samples/decision/json/run`, whenever the
output is piped: that rule prints which binary it is about to run, and it
prints it on stdout.)

The request format is **[Jev](https://docs.aimlapi.com/api-references/decision-models/typesafe/jev)-style**:
a `state` plus a `questions` map built from three primitives, which brain's own
decision vocabulary already matches one for one (`decide::primitives`).

| type | `criteria` | answer |
|---|---|---|
| `choice` | an **ordered** object, option name -> description (`null`/`""` = the name says it all) | `choice`, `confidence`, `probabilities` per option name |
| `score` | an **ordered list** of 2-10 level descriptions, worst first | `score` (the 0-based expectation over the levels), `confidence`, `legend`, `probabilities` per level index |
| `noul` | optional `{"true": ..., "false": ...}` readings | `noul`, a probability on `[0, 1]` - and nothing else, because a two-outcome probability already **is** its own confidence |

## Run it

```bash
brain pull convaiinnovations/laya          # the default model, once (807 MiB)

make samples/decision/json/build
make samples/decision/json/run ARGS=--help

./target/release/sample-decision-json < request.json
./target/release/sample-decision-json --model minilm --device cpu < request.json
```

Flags: `--model NAME|DIR` (`laya` by default, `minilm` for a
`crates/decide`-shaped encoder, or any `<vendor>/<repo>` id or checkpoint
directory), `--head FILE` (trained head weights for an encoder that ships
without one - `samples/decision/triage` writes one; a Laya checkpoint carries
its own), `--jsonl`, plus the usual `--device`/`--backend`. The model
directory is found under `$BRAIN_MODELS_DIR`, `$XDG_DATA_HOME/brain/models`, or
`$HOME/.local/share/brain/models` - wherever `brain pull` put it.

**stdout carries nothing but responses**, one JSON object per request, so it
pipes straight into `jq`. Progress, the model that loaded and the device it
ran on go to stderr. A request that fails is still a document
(`{"model": ..., "error": {"message": ...}}`) and the process exits non-zero -
so in `--jsonl` mode, where the model is loaded once and answers a stream of
requests line by line, output stays aligned with input even when one request is
malformed.

## A measured run

The request from Jev's own documentation, answered by `convaiinnovations/laya`
zero-shot on one Intel Arc iGPU:

```json
{"model": "laya",
 "answers": {"is_urgent":   {"type": "noul", "noul": 0.780172},
             "department":  {"type": "choice", "choice": "billing", "confidence": 0.915133,
                             "probabilities": {"billing": 0.983965, "technical": 0.008701,
                                               "sales": 0.007335}},
             "frustration": {"type": "score", "score": 1.317004, "confidence": 0.375038,
                             "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                             "probabilities": {"0": 0.010521, "1": 0.661955, "2": 0.327524}}}}
```

Three questions nobody had written down when the model was loaded - a routing
decision, an urgency check and a 3-level rubric - answered in one request.

## What this checkpoint knows, and what it does not

Measured, because the format working is not the same as the answer being right.
22 fixed questions with obvious answers were put through this sample AND
through Laya's own released Python serving code (`rl_agent_api.py`) on the same
weights:

- **22/22 identical answers, worst difference 3.2e-4 on any published number.**
  So a wrong answer from this sample is the CHECKPOINT, not the port. (The
  `temperature_by_options` calibration table the reference consults is applied
  here too - see `brain::DecisionPipeline`'s own docs.)
- **17/22 match the obvious human answer.** It is right on the decisions it was
  trained for: which team owns a ticket, is this a refund request, did
  something go wrong, is this an emergency, how bad was this experience. It is
  wrong when the options are bare nouns carrying no relation to the state
  ("dog" / "cat" / "fish" for *it barks and fetches sticks* comes back `fish`).

A temperature sweep is the sharpest way to see the edge of it. One `score`
question (`hot`..`freezing`), one `noul` ("Do I need a jacket?") and one
`choice` (t-shirt / jacket / winter coat), asked at eleven temperatures:

| deg C | 35 | 25 | 15 | 5 | 0 | -20 | -40 |
|---|---|---|---|---|---|---|---|
| `howcold` score (0 hot .. 4 freezing) | 1.79 | 1.70 | 1.77 | 2.03 | 3.61 | 3.57 | 3.55 |
| `noul` need a jacket | 0.08 | 0.08 | 0.08 | 0.05 | 0.00 | 0.04 | 0.04 |
| `choice` what to wear | t-shirt | t-shirt | t-shirt | t-shirt | t-shirt | t-shirt | t-shirt |

The `score` question has a coarse sense of sign - above zero against below it -
and no ordering inside either range. The other two are flat: it will tell you
to wear a t-shirt at minus forty. The released Python reference produces those
same numbers to within 5e-4, so this is the model, not the wiring.

Read that as a boundary, not a defect: Laya is a System-1 decision model
trained on agent and routing decisions over text, not a world model. Use it to
choose among actions your system defines, to triage, and to decide whether to
escalate. Do not ask it to do arithmetic, and do not put a threshold rule in
the criteria (`"wear jacket": "if temperature is less than 8 degrees"` is
comparing text, not numbers - compare the number in your own code and ask the
model only the judgement call). When the question is outside what it knows,
`confidence` is usually the tell: 0.85 on the banking routing above, 0.09 on
the animal question it got wrong.

## A second measured run

The same request with `--model minilm` returns the same SHAPE and a `confidence`
of 0.011 on the routing question, near-uniform probabilities, and a different
answer: that encoder arrives with an untrained head, and the number says so
rather than hiding it behind a confident-looking argmax. Point `--head` at a
head `samples/decision/triage` trained and it answers for real. Reading the
distribution rather than the argmax is the whole point of the contract, and it
is why `laya` - which ships its own head and answers all three question types
zero-shot - is the default here.

## What it shows

**The output space lives in the request.** One loaded model answers a
three-option question and a ten-level rubric with no reload, because there is
no final layer whose width is the answer space. A service can ask a question
invented this morning.

**Questions are independent and answered per request.** Nothing conditions an
answer on another answer. On a `crates/decide`-backed model the state is
encoded once and every question scored against that encoding - the economics
that architecture exists for; on Laya each question is a full re-encode, which
is a real per-call cost difference and is stated in
`brain::DecisionPipeline::decide`'s own documentation rather than hidden.

**Key order survives the trip.** The state reaches the model as the exact JSON
text the caller wrote, in the caller's own key order, because a sorted
re-serialization would tokenize different bytes than the reference
implementation does - see `brain::decision::OrderedJson`.

**A malformed request is refused by field name**, with the offending question
named, before any model runs.

## Cost

26 brain crates - it names one surface (`decision`) and the SDK links only what
that surface needs, exactly like `samples/decision/triage`.

---

Swedish Embedded AB implements typed decision endpoints - a model behind a
schema a service can rely on, rather than free text a caller has to parse and
hope about - for its clients. If your team needs judgment in the loop with a
contract around it, you can procure our services by sending an email to
info@swedishembedded.com.
