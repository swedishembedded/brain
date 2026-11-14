# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Capabilities, Providers and APIs".

Plain Python 3, standard library only, written from the chapter's own description of each rule. Each public function without
a leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: capabilities_and_apis.<name> -->` markers. Helpers with a leading underscore are shared with the chapter's
figure.

Everything here is exact: string rules, integer counts and 32-bit wrapping arithmetic. No generator is used, so another
implementation reproduces every printed character.

Swedish Embedded AB implements solutions for presenting many machine learning models through a few stable, self-describing
interfaces for its clients. If your team needs expertise in inference APIs, model catalogs and serving contracts then you
can procure our services by sending an email to info@swedishembedded.com.
"""
import base64
import json
import math

import number_formats as nf

_table = nf._table

MAX_STR_PARAM_BYTES = 16 << 20          # the longest text a caller may pass as one param, in bytes
MAX_LINE_BYTES = 64 << 20               # the longest line of the event protocol, newline excluded
DIALECT_BODY_BYTES = 8 * 1024 * 1024    # the request body limit of the chat, embeddings and image routes
RUN_BODY_BYTES = 64 * 1024 * 1024       # the request body limit of /v1/run and /v1/jobs
FD_BLOB_BYTES = 256 << 20               # the largest input descriptor the bus service reads


# ------------------------------------------------------------------ the schema and its validation

def _p(name, ty, required=False, default=None, lo=None, hi=None, values=None, host_env=None, host_resolved=False, help=""):
    return {"name": name, "ty": ty, "required": required, "default": default, "min": lo, "max": hi, "values": values,
            "host_env": host_env, "host_resolved": host_resolved, "help": help}


def _b(name, media, required=False):
    return {"name": name, "media": media, "required": required}


def paint_spec():
    """The action of the validation example: one of every kind of param, a host-configured checkpoint and an optional
    input image."""
    return {
        "name": "paint",
        "params": [
            _p("prompt", "str", required=True, help="what to draw"),
            _p("steps", "int", default=8, lo=1, hi=1000, help="denoising steps"),
            _p("guidance", "float", default=3.5, lo=0, hi=30, help="guidance scale"),
            _p("style", "enum", default="photo", values=["photo", "sketch"], help="rendering style"),
            _p("seed", "int", lo=0, help="seed; drawn when absent"),
            _p("weights", "str", required=True, host_env="BRAIN_TOY_WEIGHTS", help="checkpoint file"),
            _p("adapter", "str", host_resolved=True, help="adapter file"),
        ],
        "inputs": [_b("init", "image")],
        "outputs": [_b("image", "image")],
    }


def _js(v):
    """A JSON value as compact text, keys sorted (the order of the engine's ordered maps)."""
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jsp(v):
    """The same JSON printed for a table, with a space after every comma and colon so that the line can break."""
    return json.dumps(v, sort_keys=True, separators=(", ", ": "), ensure_ascii=False)


def _rf(x):
    """A bound as the engine prints a 64-bit float: no fraction when it is whole."""
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(x)


def _kind(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "text"
    return "other"


def _type_ok(p, v):
    k = _kind(v)
    return {"str": k == "text", "int": k == "int", "float": k in ("int", "float"), "bool": k == "bool",
            "enum": k == "text" and v in (p["values"] or [])}[p["ty"]]


def validate(spec, params, blobs, env):
    """The validation rule of the chapter. Returns (True, normalized params) or (False, error text)."""
    name = spec["name"]
    known = [p["name"] for p in spec["params"]]
    for k in sorted(params):
        if k not in known:
            return False, f"unknown param '{k}' for action '{name}'"
    out = dict(params)
    for p in spec["params"]:
        if p["name"] in params:
            v = params[p["name"]]
            if isinstance(v, str) and len(v.encode()) > MAX_STR_PARAM_BYTES:
                return False, f"param '{p['name']}' is longer than {MAX_STR_PARAM_BYTES} bytes"
            if not _type_ok(p, v):
                return False, f"param '{p['name']}' must be {p['ty']} (got {_js(v)})"
            if _kind(v) in ("int", "float"):
                if p["min"] is not None and v < p["min"]:
                    return False, f"param '{p['name']}' must be at least {_rf(p['min'])} (got {_js(v)})"
                if p["max"] is not None and v > p["max"]:
                    return False, f"param '{p['name']}' must be at most {_rf(p['max'])} (got {_js(v)})"
        else:
            hv = env.get(p["host_env"], "") if p["host_env"] else ""
            if hv:
                out[p["name"]] = hv
            elif p["default"] is not None:
                out[p["name"]] = p["default"]
            elif p["required"]:
                if p["host_env"]:
                    return False, (f"action '{name}' needs '{p['name']}', which this machine has not configured: "
                                   f"set ${p['host_env']} (or pass '{p['name']}' explicitly)")
                return False, f"missing required param '{p['name']}' for action '{name}'"
    for b in spec["inputs"]:
        if b["name"] in blobs:
            if blobs[b["name"]] != b["media"]:
                return False, f"input '{b['name']}' expects {b['media']} but got {blobs[b['name']]}"
        elif b["required"]:
            return False, f"missing required input '{b['name']}' for action '{name}'"
    for k in sorted(blobs):
        if k not in [b["name"] for b in spec["inputs"]]:
            return False, f"unknown input '{k}' for action '{name}'"
    return True, out


def for_serving(spec):
    s = dict(spec)
    s["params"] = [p for p in spec["params"] if p["host_env"] is None and not p["host_resolved"]]
    return s


def validate_served(spec, params, blobs, env):
    ok, r = validate(for_serving(spec), params, blobs, env)
    if not ok:
        return ok, r
    return validate(spec, r, blobs, env)


def spec_table():
    s = paint_spec()
    rows = []
    for p in s["params"]:
        rng = "" if p["min"] is None and p["max"] is None else f"[{'' if p['min'] is None else _rf(p['min'])}, {'' if p['max'] is None else _rf(p['max'])}]"
        host = f"from \\${p['host_env']}" if p["host_env"] else ("host-resolved" if p["host_resolved"] else "")
        ty = p["ty"] if p["ty"] != "enum" else "enum: " + ", ".join(p["values"])
        served = "no" if (p["host_env"] or p["host_resolved"]) else "yes"
        rows.append([f"`{p['name']}`", ty, "yes" if p["required"] else "no", "" if p["default"] is None else f"`{_js(p['default'])}`",
                     rng, host, served])
    return ("The action `paint` of a hypothetical model. Ranges are inclusive. A param marked `from $VAR` is filled from that "
            "environment variable of the machine that runs the action when the caller gives none; a host-resolved param "
            "is answered by the host in some other way. Input blobs: `init` (image, optional). Output blobs: `image` "
            "(image).\n\n"
            + _table(["param", "type", "required", "default", "range", "host", "in the served view"], rows, "lllllll"))


def validate_table():
    s = paint_spec()
    env_set = {"BRAIN_TOY_WEIGHTS": "/models/toy.safetensors"}
    cases = [
        ("local", {"prompt": "a cat"}, {}, env_set),
        ("local", {"prompt": "a cat"}, {}, {}),
        ("local", {"prompt": "a cat", "steps": 0}, {}, env_set),
        ("local", {"prompt": "a cat", "steps": 2.0}, {}, env_set),
        ("local", {"prompt": "a cat", "guidance": 30}, {}, env_set),
        ("local", {"prompt": "a cat", "style": "oil"}, {}, env_set),
        ("local", {"prompt": "a cat", "colour": "red", "zoom": 2}, {}, env_set),
        ("local", {}, {}, env_set),
        ("local", {"prompt": "a cat"}, {"init": "mask"}, env_set),
        ("local", {"prompt": "a cat", "weights": "/etc/passwd"}, {}, env_set),
        ("served", {"prompt": "a cat"}, {}, env_set),
        ("served", {"prompt": "a cat", "weights": "/etc/passwd"}, {}, env_set),
        ("served", {"prompt": "a cat", "adapter": "/srv/a.safetensors"}, {}, env_set),
        ("served", {"prompt": "a cat"}, {}, {}),
    ]
    rows = []
    for i, (where, params, blobs, env) in enumerate(cases, start=1):
        f = validate if where == "local" else validate_served
        ok, r = f(s, params, blobs, env)
        res = f"accepted: {_jsp(r)}" if ok else "refused: " + r.replace("$", "\\$")
        bl = ", ".join(f"`{k}` ({v})" for k, v in sorted(blobs.items())) or "none"
        rows.append([str(i), where, _jsp(params), bl, "set" if env else "unset", res])
    return ("Each call is validated against `paint` by the local rule (`validate`, for a caller on the machine that holds the "
            "weights) or the served rule (`validate_served`, for every caller over a network or a bus). "
            "`BRAIN_TOY_WEIGHTS`, when set, is `/models/toy.safetensors`. Params print as JSON with the keys sorted and a "
            "space after each comma and colon; the first failing check names the refusal.\n\n"
            + _table(["call", "rule", "params", "blobs", "variable", "result"], rows, "rlllll"))


# ------------------------------------------------------------------ counting surfaces

def adapter_counts():
    rows = []
    for m in (10, 100, 300):
        for s in (3, 6):
            rows.append([f"{m:,}", str(s), f"{m * s:,}", f"{m + s:,}", str(s), "1"])
    return ("An engine with M models and S surfaces (a command line, an HTTP dialect, a message bus, ...). Without a shared "
            "description, every pair needs its own glue: M x S pieces. With one description per model that every surface "
            "reads, there are M descriptions and S readers. The last two columns count the places one new parameter of "
            "one model has to be written.\n\n"
            + _table(["models M", "surfaces S", "glue per pair: M x S", "descriptions and readers: M + S",
                      "new param, glue per pair", "new param, one description"], rows, "rrrrrr"))


# ------------------------------------------------------------------ cancellation is counted in steps

def _sci(x):
    """A count as a mantissa of up to 3 significant digits times a power of ten, or 0."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(x)))
    m = x / 10**e
    ms = f"{m:.3g}"
    return f"{ms} x 10^{e}" if ms != "1" else f"10^{e}"


def cancel_work():
    rows = []
    cases = [
        ("decode, 8 x 10^9 params", 2 * 8 * 10**9, 512, 0),
        ("decode, 8 x 10^9 params", 2 * 8 * 10**9, 512, 10**11),
        ("decode, 8 x 10^9 params", 2 * 8 * 10**9, 512, 10**13),
        ("denoise, hypothetical", 4 * 10**12, 28, 10**13),
    ]
    for name, c, n, w in cases:
        if w >= n * c:
            done, after, note = n, 0, "ran to the end before the request"
        else:
            done = math.ceil(w / c)
            after, note = done * c - w, "stopped at the next poll"
        rows.append([name, f"{_sci(c)} FLOPs", str(n), f"{_sci(w)} FLOPs", str(done), f"{_sci(after)} FLOPs", note])
    load = 16 * 10**9
    w = 10**9
    rows.append(["first call: load a checkpoint", f"{_sci(load)} bytes read", "512", f"{_sci(w)} bytes read", "0",
                 f"{_sci(load - w)} bytes read", "the load has no poll; the poll before step 0 stops it"])
    return ("An action polls its cancellation flag before each of its n steps and once after the last. Each step costs "
            "c FLOPs (floating-point operations); a decode step of a model of 8 x 10^9 parameters is counted as 2 FLOPs "
            "per parameter, one multiply and one add. The request arrives after w FLOPs of steps (or w bytes of a load) "
            "have been done; the steps completed are ceil(w / c), at most n, and the work after the request is the "
            "completed steps' work minus w.\n\n"
            + _table(["work", "per step c", "steps n", "request at w", "steps completed", "work after the request", "outcome"],
                     rows, "lrrrrrl"))


# ------------------------------------------------------------------ one catalog instead of three lists

def catalog_drift():
    rows = []
    for mask in range(8):
        lists = [i + 1 for i in range(3) if mask >> i & 1]
        listed, runnable, served = (1 in lists), (2 in lists), (3 in lists)
        name = ", ".join(str(i) for i in lists) or "none"
        rows.append([name, "yes" if listed else "no", "yes" if runnable else "no", "yes" if served else "no",
                     "consistent" if listed == runnable == served else "silently partial"])
    rows.append(["one catalog entry", "yes", "yes", "yes", "consistent by construction"])
    return ("Three hand-kept lists: 1, the manifests a listing command prints; 2, the constructors the run command looks a "
            "model up in; 3, the registrations the server schedules. A model reaches a surface only through its list. The "
            "last row derives all three from one entry.\n\n"
            + _table(["lists the model was added to", "listed", "runnable by name", "served", "state"], rows, "lllll"))


# ------------------------------------------------------------------ reading API capabilities from manifests

def _a(name, params=(), inputs=(), outputs=(), streaming=False, maxes=None):
    return {"name": name, "params": list(params), "inputs": list(inputs), "outputs": list(outputs),
            "streaming": streaming, "maxes": maxes or {}}


def api_manifests():
    return [
        ("x/chat", [_a("generate", ["messages", "max_new", "tools"], [("image", "image", False)], [("text", "text")], True,
                       {"max_new": 262144})]),
        ("x/oneshot", [_a("generate", ["prompt"], [], [("text", "text")], False)]),
        ("x/captioner", [_a("caption", ["prompt"], [("image", "image", True)], [("text", "text")], True)]),
        ("x/textembed", [_a("embed", ["text"], [], [("mean", "bytes")])]),
        ("x/faceembed", [_a("embed", [], [("image", "image", True)], [("embedding", "bytes")])]),
        ("x/painter", [_a("text2image", ["prompt", "steps"], [("init", "image", False)], [("image", "image")])]),
        ("x/restorer", [_a("restore", ["strength"], [("image", "image", True)], [("image", "image")])]),
        ("x/listener", [_a("transcribe", ["language"], [("audio", "audio", True)], [("text", "text")])]),
        ("x/speaker", [_a("speak", ["text", "voice"], [], [("audio", "audio")])]),
    ]


def api_caps(actions):
    def has(a, p):
        return p in a["params"]
    chat = any(a["name"] == "generate" and a["streaming"] and any(has(a, p) for p in ("prompt", "messages", "text"))
               and any(m == "text" for _, m in a["outputs"]) for a in actions)
    emb = any(a["name"] == "embed" and has(a, "text") for a in actions)
    image = any(any(m == "image" for _, m in a["outputs"]) and has(a, "prompt") and not any(r for _, _, r in a["inputs"])
                for a in actions)
    tools = chat and any(a["name"] == "generate" and has(a, "tools") for a in actions)
    vision = chat and any(a["name"] == "generate" and any(n == "image" for n, _, _ in a["inputs"]) for a in actions)
    trans = any(a["name"] == "transcribe" and any(n == "audio" and m == "audio" for n, m, _ in a["inputs"])
                and any(m == "text" for _, m in a["outputs"]) for a in actions)
    speech = any(a["name"] == "speak" and has(a, "text") and any(n == "audio" and m == "audio" for n, m in a["outputs"])
                 for a in actions)
    return {"chat": chat, "tools": tools, "vision": vision, "embeddings": emb, "images": image,
            "transcription": trans, "speech": speech}


def api_capabilities():
    rows = []
    for model, actions in api_manifests():
        caps = api_caps(actions)
        names = [k for k in ("chat", "tools", "vision", "embeddings", "images", "transcription", "speech") if caps[k]]
        anyc = caps["chat"] or caps["embeddings"] or caps["images"] or caps["transcription"] or caps["speech"]
        mo = ""
        for a in actions:
            if a["name"] == "generate" and "max_new" in a["maxes"]:
                mo = f"{a['maxes']['max_new']:,}"
        desc = "; ".join(
            f"`{a['name']}`" + (" (streaming)" if a["streaming"] else "") + ": params " + (", ".join(a["params"]) or "none")
            + "; inputs " + (", ".join(f"{n} {m}" + (" required" if r else "") for n, m, r in a["inputs"]) or "none")
            + "; outputs " + ", ".join(f"{n} {m}" for n, m in a["outputs"]) for a in actions)
        rows.append([f"`{model}`", desc, ", ".join(names) or "none", "yes" if caps["chat"] else "no", "yes" if anyc else "no",
                     mo or "not stated"])
    return ("Nine one-action manifests and what the shape rules read from them. The chat dialect lists a model only for "
            "chat; the other dialects list it for any capability.\n\n"
            + _table(["model", "action", "capabilities", "listed by the chat-only dialect", "listed by the others",
                      "most new tokens per request"], rows, "llllll"))


# ------------------------------------------------------------------ which model a request names

ALIASES = {"mock": "brain/mock", "qwen": "brain/qwen3"}
SERVED = {"brain/mock", "brain/qwen3", "Qwen/Qwen3-0.6B"}


def candidates(dialect, body):
    out = []
    strip = dialect == "openrouter"

    def push(i):
        if not i:
            return
        out.append(i)
        if i in ALIASES:
            out.append(ALIASES[i])
        if strip and "/" in i:
            rest = i.split("/", 1)[1]
            if rest:
                out.append(rest)
    if isinstance(body.get("model"), str):
        push(body["model"])
    if strip:
        for m in body.get("models", []):
            if isinstance(m, str):
                push(m)
    return out


def model_candidates():
    bodies = [
        ("anthropic", {"model": "brain/mock"}),
        ("openai", {"model": "qwen"}),
        ("openai", {"model": "openrouter/brain/mock"}),
        ("openrouter", {"model": "openrouter/brain/mock"}),
        ("openrouter", {"model": "x/unknown", "models": ["qwen", "brain/mock"]}),
        ("openrouter", {"model": "Qwen/Qwen3-0.6B"}),
    ]
    rows = []
    for d, b in bodies:
        c = candidates(d, b)
        hit = next((i for i in c if i in SERVED), None)
        rows.append([d, _jsp(b), ", ".join(f"`{i}`" for i in c), f"`{hit}`" if hit else "none: model not found"])
    return ("The served models are `brain/mock`, `brain/qwen3` and `Qwen/Qwen3-0.6B`; the alias table maps `mock` to "
            "`brain/mock` and `qwen` to `brain/qwen3`. Each name a request gives is tried as written, then its alias, "
            "and, for the routing dialect only, without its first path segment; the first candidate that is served "
            "wins.\n\n"
            + _table(["dialect", "request body", "candidates in order", "answers"], rows, "llll"))


# ------------------------------------------------------------------ failures a client can act on

SLUGS = ["not-installed", "failed-to-load", "unsupported-capability", "timeout", "invalid-request", "backend-failure",
         "unavailable"]
CLASS_STATUS = {"not-installed": 404, "failed-to-load": 500, "backend-failure": 500, "unsupported-capability": 422,
                "invalid-request": 422, "timeout": 504, "unavailable": 503}


def _classify_here(e):
    if not e.startswith("["):
        return None
    rest = e[1:]
    if "]" not in rest:
        return None
    slug, msg = rest.split("]", 1)
    if slug not in SLUGS:
        return None
    return slug, msg[1:] if msg.startswith(" ") else msg


def classify(e):
    rest = e
    while True:
        found = _classify_here(rest)
        if found:
            return found
        i = rest.find(": ")
        if i < 0:
            return None
        rest = rest[i + 2:]


def reply(e, model):
    """(kind, client message, hides detail, HTTP status, bus reply)."""
    c = classify(e)
    if c:
        slug, msg = c
        return f"classified {slug}", msg, False, CLASS_STATUS[slug], f"[{slug}] {msg}" if msg else f"[{slug}]"
    if ": panicked while running: " in e:
        m = "the model faulted while running the request"
        return "faulted", m, True, 500, m
    if e == "cancelled" or e.endswith(": cancelled"):
        return "cancelled", "the request was cancelled", False, 503, "cancelled"
    if e.startswith("no model") or "no action" in e:
        m = f"no model '{model}'"
        return "no model", m, False, 404, m
    if e.endswith("has no fill-in-the-middle tokens"):
        m = ("'suffix' (fill-in-the-middle) is not supported: this model's vocabulary has no fill-in-the-middle tokens")
        return "no fill-in-the-middle", m, False, 400, m
    if "needs " in e:
        rest = e.split("needs ", 1)[1]
        sep = " tokens, engine capacity is "
        if sep in rest:
            need, cap = rest.split(sep, 1)
            if need.strip().isdigit() and cap.strip().isdigit():
                m = (f"the request needs {int(need)} tokens but this model's serving capacity is {int(cap)} tokens; "
                     "reduce the prompt or lower max_tokens")
                return "capacity", m, True, 400, m
    m = "the model failed to process the request"
    return "failed", m, True, 400, m


def failure_replies():
    errors = [
        "[not-installed] weights for 'x/y' are not installed: pull them first",
        "activate x/y/default: [failed-to-load] the checkpoint is corrupt",
        "[timeout]",
        "x/y: panicked while running: index out of bounds at src/decode.rs:10",
        "cancelled",
        "no model 'x/z'",
        "qwen: needs 14995 tokens, engine capacity is 2048",
        "backend error at /srv/m/w.gguf: kernel failed",
        "clip says [timeout] in prose",
    ]
    rows = []
    for e in errors:
        model = "x/z" if e.startswith("no model") else "x/y"
        kind, msg, hides, status, bus = reply(e, model)
        rows.append([e, kind, msg or "(empty)", "yes" if hides else "no", str(status), bus])
    return ("What one action error becomes on each surface, for a request to the model named in the second column of each "
            "error (`x/z` for the sixth, `x/y` otherwise). The rules are tried in the order of the chapter; an error that "
            "hides detail is written to the server's log and never sent.\n\n"
            + _table(["error text the action returned", "sorted as", "message to the caller", "detail logged only",
                      "HTTP status", "bus reply"], rows, "llllrl"))


# ------------------------------------------------------------------ bytes on each transport

def _b64(n):
    return 4 * ((n + 2) // 3)


def transport_bytes():
    cases = [
        ("frame 1920 x 1080, RGB 8-bit", 1920 * 1080 * 3),
        ("frame 3840 x 2160, RGB 8-bit", 3840 * 2160 * 3),
        ("image 3840 x 2160, RGB 32-bit float", 3840 * 2160 * 3 * 4),
        ("audio, 160,000 samples (ten seconds at 16 kHz), mono 32-bit float", 10 * 16000 * 4),
        ("image 8192 x 8192, RGB 32-bit float", 8192 * 8192 * 3 * 4),
    ]
    rows = []
    for name, n in cases:
        b = _b64(n)
        rows.append([name, f"{n:,}", f"{b:,}", "yes" if b <= DIALECT_BODY_BYTES else "no",
                     "yes" if b <= MAX_LINE_BYTES else "no", "yes" if n <= FD_BLOB_BYTES else "no"])
    return ("The raw bytes of a payload and its base64 text, 4 characters for every 3 bytes, the last group padded: "
            "4 x ceil(n / 3). A text transport carries the base64; the bus carries a file descriptor and the receiver "
            "reads the raw bytes from it. The limits are 8,388,608 bytes for a dialect's request body, 67,108,864 bytes "
            "for one event line and for a `/v1/run` body, and 268,435,456 bytes for one descriptor. 'Fits' compares the "
            "payload alone; the JSON around it needs a few bytes more.\n\n"
            + _table(["payload", "raw bytes", "base64 bytes", "fits a dialect body", "fits an event line or /v1/run",
                      "fits one bus descriptor"], rows, "lrrlll"))


# ------------------------------------------------------------------ the event protocol's lines

def _envelope(ev, req_id=None):
    v = dict(ev)
    if req_id is not None:
        v["req_id"] = req_id
    return v


def event_lines():
    out_b64 = base64.b64encode(b"hihihi").decode()
    enc = [
        ("a request to run an action, correlated", _envelope({"event": "action_request", "model": "brain/demo",
                                                             "action": "echo", "params": {"text": "hi", "times": 3},
                                                             "blobs": []}, "r1")),
        ("its progress", _envelope({"event": "action_progress", "step": 1, "total": 1, "message": "echoing"}, "r1")),
        ("its result", _envelope({"event": "action_result", "outputs": {"chars": 6},
                                  "blobs": [{"name": "result", "media": "text", "b64": out_b64, "meta": None}]}, "r1")),
        ("a text chunk", _envelope({"event": "brain_text_chunk", "text": "Hel", "seq": 0, "done": False})),
        ("the last text chunk, with its seed", _envelope({"event": "brain_text_chunk", "text": "lo", "seq": 1, "done": True,
                                                          "seed": 42})),
        ("cancel the operation in flight", _envelope({"event": "cancel"})),
        ("a line over the limit", ({"event": "error", "code": "line_too_long",
                                       "message": f"line exceeds {MAX_LINE_BYTES} bytes", "retryable": False})),
        ("a connection over the cap", ({"event": "error", "code": "server_busy", "message": "connection limit reached",
                                           "retryable": True})),
    ]
    dec = [
        ('{"event":"brain_text_chunk","text":"a"}', "text chunk: text `a`, seq 0, done false, no seed"),
        ('{"event":"user_text"}', "user text: the empty text"),
        ('{"event":"cancel","req_id":7}', "cancel, with no correlation id (a non-text `req_id` reads as absent)"),
        ('{"text":"x"}', 'error: missing "event" tag'),
        ('{"event":"nope"}', 'error: unknown event tag: "nope"'),
    ]
    t1 = _table(["what", "line, spaced"], [[w, _jsp(v)] for w, v in enc], "ll")
    t2 = _table(["line received", "decoded as"], [[f"`{line}`", r] for line, r in dec], "ll")
    return ("Each event is one line of JSON; keys are written in byte order, because the maps that build the lines are "
            "ordered maps. The first table prints each line with a space after each comma and colon so that it can break; "
            "the wire has none. `aGloaWhp` is the base64 of the 6 bytes `hihihi`. The second table decodes lines: a missing "
            "field takes its default (the empty text, 0, false), and an unknown or missing tag is an error.\n\n"
            + t1 + "\n\n" + t2)


# ------------------------------------------------------------------ a command line read from the schema

def sketch_spec():
    return {
        "name": "sketch",
        "params": [
            _p("text", "str", required=True, help="the text"),
            _p("times", "int", default=1, lo=0, hi=1000, help="repeat count"),
            _p("mode", "enum", default="as-is", values=["as-is", "upper", "lower"], help="casing"),
            _p("held_out", "int", default=4, help="rows held out"),
            _p("preview", "bool", default=False, help="show a preview"),
            _p("seed", "int", help="seed"),
        ],
        "inputs": [_b("image", "image", required=True)],
        "outputs": [_b("result", "text")],
    }


def _flag_of(p):
    names = f"`--{p['name']}`" + (f", `--{p['name'].replace('_', '-')}`" if "_" in p["name"] else "")
    if p["ty"] == "bool":
        value = "`BOOL`, optional: absent value means true; one of true, false, 1, 0"
    elif p["ty"] == "enum":
        value = "`ENUM`: one of " + ", ".join(p["values"])
    elif p["ty"] in ("int", "float"):
        lo = p["min"] if p["min"] is not None else (0 if p["name"] == "seed" else None)
        bound = []
        if lo is not None:
            bound.append(f"at least {_rf(lo)}")
        if p["max"] is not None:
            bound.append(f"at most {_rf(p['max'])}")
        value = f"`{p['ty'].upper()}`: " + ("a whole number" if p["ty"] == "int" else "a finite number") + (
            ", " + " and ".join(bound) if bound else "")
    else:
        value = "`STR`"
    req = "yes" if (p["required"] and p["default"] is None and p["ty"] != "bool") else "no"
    hlp = p["help"] + (f" [default: {_js(p['default'])}]" if p["default"] is not None else "")
    return [names, value, req, hlp]


def _parse_sketch(argv):
    """The flag reading of the chapter for `sketch`, before the action's own validation. Returns (params, inputs, error)."""
    s = sketch_spec()
    by = {}
    for p in s["params"]:
        by["--" + p["name"]] = p
        by["--" + p["name"].replace("_", "-")] = p
    params, inputs, i = {}, [], 0
    while i < len(argv):
        t = argv[i]
        if t in by:
            p = by[t]
            if p["ty"] == "bool":
                if i + 1 < len(argv) and argv[i + 1] in ("true", "false", "1", "0"):
                    params[p["name"]] = argv[i + 1] in ("true", "1")
                    i += 2
                else:
                    params[p["name"]] = True
                    i += 1
                continue
            v = argv[i + 1]
            if p["ty"] == "int":
                lo = p["min"] if p["min"] is not None else (0 if p["name"] == "seed" else None)
                try:
                    x = int(v)
                except ValueError:
                    return None, None, f"--{p['name']}: '{v}' is not a whole number"
                if lo is not None and x < lo:
                    return None, None, f"--{p['name']}: {v} is below the least allowed, {_rf(lo)}"
                if p["max"] is not None and x > p["max"]:
                    return None, None, f"--{p['name']}: {v} is above the most allowed, {_rf(p['max'])}"
                params[p["name"]] = x
            else:
                params[p["name"]] = v
            i += 2
        elif t == "--image":
            inputs.append(("image", argv[i + 1]))
            i += 2
        elif t == "--in":
            n, path = argv[i + 1].split("=", 1)
            inputs.append((n, path))
            i += 2
        elif t == "--out":
            n, path = argv[i + 1].split("=", 1)
            if n != "result":
                return None, None, f"--out \"{argv[i + 1]}\": the action has no output '{n}' (it has: result)"
            i += 2
        else:
            return None, None, f"unexpected argument {t}"
    for p in s["params"]:
        if p["required"] and p["default"] is None and p["name"] not in params:
            return None, None, f"the flag --{p['name']} is required"
    return params, inputs, None


def flags_from_spec():
    s = sketch_spec()
    t1 = _table(["param", "flags", "value", "required on the command line", "help line"],
                [[f"`{p['name']}`"] + _flag_of(p) for p in s["params"]], "lllll")
    runs = [
        ["--text", "hi", "--image", "a.png"],
        ["--text", "hi", "--held-out", "2", "--in", "image=b.png", "--preview"],
        ["--text", "hi", "--image", "a.png", "--preview", "false"],
        ["--text", "hi"],
        ["--image", "a.png"],
        ["--text", "hi", "--image", "a.png", "--times", "1001"],
        ["--text", "hi", "--image", "a.png", "--seed", "-1"],
        ["--text", "hi", "--image", "a.png", "--out", "picture=o.txt"],
    ]
    rows = []
    blob_media = {"image": "image"}
    for argv in runs:
        params, inputs, err = _parse_sketch(argv)
        if err:
            rows.append([f"`{' '.join(argv)}`", f"refused while parsing: {err}", "2"])
            continue
        ok, r = validate(s, params, {n: blob_media[n] for n, _ in inputs}, {})
        if not ok:
            rows.append([f"`{' '.join(argv)}`", f"parsed; refused by the action's validation: {r}", "2"])
            continue
        ins = ", ".join(f"{n} from {p}" for n, p in inputs)
        rows.append([f"`{' '.join(argv)}`", f"params `{_js(params)}`, inputs {ins}; validated as `{_js(r)}`", "0"])
    t2 = _table(["flags after `<program> sketch`", "what is read", "exit code"], rows, "llr")
    return ("The action `sketch` declares six params, one required input blob `image` and one output blob `result`. "
            "Every param becomes a flag of its own; an input blob becomes `--image PATH` (or `--in image=PATH`) and an "
            "output `--out result=PATH`. A flag left out sets nothing, so the action's own default applies when it "
            "validates. A refusal is printed as `--flag: reason`, a simplified framing of the longer message the flag-parsing library prints; the reason is the program's own wording for a number out of bounds and for an unknown output, and the refusal of a missing required flag is paraphrased.\n\n" + t1 + "\n\n" + t2)


# ------------------------------------------------------------------ the consuming flag grammar

class _Args:
    def __init__(self, toks):
        self.toks, self.used = list(toks), [False] * len(toks)

    def take_str(self, name):
        for i, t in enumerate(self.toks):
            if not self.used[i] and t == name:
                self.used[i] = True
                if i + 1 < len(self.toks):
                    self.used[i + 1] = True
                    return self.toks[i + 1]
                return "exit 2: requires a value"
        return None

    def take_flag(self, name):
        for i, t in enumerate(self.toks):
            if not self.used[i] and t == name:
                self.used[i] = True
                return True
        return False

    def positional(self):
        for i, t in enumerate(self.toks):
            if not self.used[i] and not t.startswith("--"):
                self.used[i] = True
                return t
        return None

    def leftovers(self):
        return [t for t, u in zip(self.toks, self.used) if not u]


def consuming_grammar():
    runs = [
        (["--steps", "7", "--device", "cpu", "--backend", "vulkan", "--window"],
         [("take_str", "--device"), ("take_str", "--backend"), ("take_str", "--steps"), ("take_flag", "--window")]),
        (["--device", "cpu", "--warmup-episodes", "16"], [("take_str", "--device"), ("take_str", "--warmup")]),
        (["--seed", "5", "in.png"], [("take_str", "--seed"), ("positional", None)]),
        (["--seed", "5", "in.png"], [("positional", None), ("take_str", "--seed")]),
        (["--model", "--device", "cpu"], [("take_str", "--model"), ("take_str", "--device")]),
    ]
    rows = []
    for toks, calls in runs:
        a = _Args(toks)
        got = []
        for kind, name in calls:
            if kind == "take_str":
                v = a.take_str(name)
                got.append(f"take `{name}` -> " + ("nothing" if v is None else f"`{v}`"))
            elif kind == "take_flag":
                got.append(f"flag `{name}` -> {'present' if a.take_flag(name) else 'absent'}")
            else:
                v = a.positional()
                got.append("positional -> " + ("nothing" if v is None else f"`{v}`"))
        left = a.leftovers()
        fin = "runs" if not left else "exit 2: unrecognised: " + " ".join(left)
        rows.append([f"`{' '.join(toks)}`", "; ".join(got), ", ".join(f"`{t}`" for t in left) or "none", fin])
    return ("Each run gives the tokens of a command line and the calls an application makes on them, in order. A call "
            "marks what it takes as used; `finish` refuses to run while any token is left unused.\n\n"
            + _table(["tokens", "calls in order and what each returns", "left over", "finish"], rows, "llll"))


# ------------------------------------------------------------------ a weight-free stand-in

def fold_seed(model, action, extra):
    h = 0x811C9DC5
    for byte in (model + "/" + action).encode():
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    h ^= extra & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    return h


def mock_seeds():
    cases = [("mock/vision", "go", 0), ("mock/vision", "go", 1), ("mock/vision", "go", 2), ("mock/vision", "edit", 0),
             ("real/model", "text2image", 0)]
    rows = []
    for m, a, e in cases:
        s = fold_seed(m, a, e)
        px = [(0 + s) % 256, (0 + (s >> 8)) % 256, (0 + 0 + s) % 256]
        px1 = [(1 + s) % 256, (0 + (s >> 8)) % 256, (1 + 0 + s) % 256]
        rows.append([f"`{m}`", f"`{a}`", str(e), f"{s:,}", f"0x{s:08x}", f"({px[0]}, {px[1]}, {px[2]}) / 255",
                     f"({px1[0]}, {px1[1]}, {px1[2]}) / 255", f"{220 + s % 440}"])
    return ("The seed of a stand-in action folds the model's name, a `/`, the action's name and the request's `seed` "
            "(0 when absent) in 32-bit wrapping arithmetic: h = 0x811C9DC5; for each byte, h = (h xor byte) x 0x01000193; "
            "then h = h xor seed (its low 32 bits); h = h xor (h >> 15); h = h x 0x85EBCA6B; h = h xor (h >> 13). The "
            "pixel at column x, row y of a stand-in image is ((x + h) mod 256, (y + (h >> 8)) mod 256, "
            "(x + y + h) mod 256) / 255, and a stand-in tone has the frequency 220 + (h mod 440) Hz.\n\n"
            + _table(["model", "action", "seed", "h", "h in hex", "pixel (0, 0)", "pixel (1, 0)", "tone (Hz)"], rows,
                     "llrrrllr"))
