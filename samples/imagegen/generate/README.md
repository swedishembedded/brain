# sample: imagegen/generate

Text prompt in, PNG out, through the public `brain` SDK - the smallest
complete application you can build on brain.

```bash
make samples/imagegen/generate/build
make samples/imagegen/generate/run ARGS="--prompt 'a whale submarine surfacing at dawn'"
```

## What it demonstrates

* `brain::ImagePipeline::from_pretrained` resolves a model out of the local
  model store using the same resolver the CLI uses - no parallel path, no
  environment variable required.
* `generate_with` + `Image::save`: the whole generation surface is three calls.
* No CLI process and no capability-dispatch server in the loop. This binary
  links `brain` the way a product would.

## What it needs

Real weights. The default model is `black-forest-labs/FLUX.2-klein-4B`; if it
is not in the local store the sample prints the `brain pull` command to run and
exits non-zero rather than downloading several gigabytes behind your back.

```bash
brain pull black-forest-labs/FLUX.2-klein-4B
```

## Options

| flag | default |
|---|---|
| `--model ID` | `black-forest-labs/FLUX.2-klein-4B` |
| `--prompt TEXT` | `a whale submarine surfacing at dawn` |
| `--out PATH` | `out/sample-imagegen.png` |
| `--steps N` | model default |
| `--seed N` | model default |
