# PID control policy

A small Transformer trained via DAgger to imitate a pole-placed PID control
oracle: it reads a CBOR-encoded stream of event/effect records from a
simulated plant and, at each decision point, picks the actuator action the
oracle would have taken. This is brain's control-systems demo model — a
concrete example of the engine doing sequential decision-making rather than
text generation. It's also the model behind brain's WebGPU browser demo:
this model can run entirely client-side, in a browser, with no server round
trip.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| INT8                   | [ ] |

## Getting the weights

There's no published checkpoint or model id — `brain pid train` produces
your own from simulated plant data; there's nothing to fetch.

## Running it

```bash
brain pid train    --out pid.safetensors
brain pid rollout  --weights pid.safetensors
brain pid profile  --weights pid.safetensors
```

`train` runs DAgger — an exploration policy drives a grid of simulated
plants, every visited state is relabeled with the oracle's action, and the
Transformer is trained on the result — then prints a closed-loop
model-vs-oracle report across both the training and a disjoint held-out set
of plants. `rollout` reproduces that same report from a saved checkpoint.
`profile` times a single forward pass to the decision point and reports
throughput.

### WebGPU browser demo

`make web/dev` serves the browser demo. It runs two closed control loops
side by side on the same simulated plant and setpoint schedule — the
Transformer driving the plant, and that plant's PID oracle — and renders
tracking and control-signal charts plus a generalization badge for
off-training-grid plants. Plant physics, the control loop, and both
controllers run compiled to WebAssembly and WebGPU; the browser page itself
only renders. It needs a secure context (localhost or https) and a
WebGPU-capable browser (Chrome/Edge 113+).

## Hardware and limits

The browser build is inference-only — there's no training or loss readback
in-wasm, only native training. There's no `import` or `serve` subcommand:
unlike brain's imported models, there's no upstream checkpoint for this
model, so weights only ever come from `brain pid train` and are consumed by
`rollout`, `profile`, and the web demo.
