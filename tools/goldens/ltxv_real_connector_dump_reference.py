#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""DIAGNOSTIC (not committed to the gate suite): dump the REAL-WEIGHT
`video_embeddings_connector` reference output, at real width, from the
actual Q8_0 GGUF - the one real-generation component whose real-weight
forward has never been checked against the reference at all (tracked gap,
`crates/ltxv/tests/dit_parity.rs`'s real-weight test explicitly runs with
`use_embeddings_connector: false`).

Reuses `ltxv_real_dit_dump_reference.py`'s GgufLite/deq_q8_0 machinery
(imported, not copied) and instantiates the reference's own
`Embeddings1DConnector` directly (`text_encoders/gemma/embeddings_connector.py`)
with the real config values read straight off the GGUF's own `config` KV.
"""

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ltxv_real_dit_dump_reference as _dit_dump  # noqa: E402
from ltxv_real_dit_dump_reference import GgufLite  # noqa: E402

# `learnable_registers` ships as GGUF type 1 (F16), which the DiT dumper's
# `_BLOCK_GEOM`/`load_real_tensor` never needed (the DiT's own real-weight
# subset is F32/Q8_0 only). Extend both here rather than in the shared file,
# since this is this script's own tensor subset's requirement, not the DiT
# dumper's.
_dit_dump._BLOCK_GEOM[1] = (1, 2)


def load_real_tensor(g, name):
    raw, ty, numel = g.raw(name)
    if ty == 1:
        import numpy as np
        flat = torch.from_numpy(np.frombuffer(raw, dtype="<f2")[:numel].astype(np.float32).copy())
        return flat.reshape(g.shape_torch(name))
    return _dit_dump.load_real_tensor(g, name)

_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "resources" / "ltxv" / "source")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

import importlib.util as _ilu  # noqa: E402


def _load_module_direct(mod_name, file_path):
    """Load a submodule straight from its file, bypassing the parent
    package's `__init__.py` (`text_encoders/gemma/__init__.py` eagerly
    imports `base_encoder.py`, which needs a `transformers` this env's
    `jinja2` is incompatible with - `embeddings_connector.py` itself needs
    none of that, only `attention`/`feed_forward`/`rope`/`utils.rms_norm`)."""
    spec = _ilu.spec_from_file_location(mod_name, file_path)
    module = _ilu.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_ec_path = _REFERENCE_ROOT / "packages" / "ltx-core" / "src" / "ltx_core" / "text_encoders" / "gemma" / "embeddings_connector.py"
Embeddings1DConnector = _load_module_direct("ltx_core.text_encoders.gemma.embeddings_connector", str(_ec_path)).Embeddings1DConnector
from ltx_core.model.transformer.rope import LTXRopeType  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

DIM = 32 * 128  # connector_num_attention_heads * connector_attention_head_dim = 4096
NUM_LAYERS = 8
NUM_REGISTERS = 128
S = 128  # one full multiple of NUM_REGISTERS - smallest real-shaped, valid test length
N_VALID = 20  # a plausible short real-prompt token count


def connector_tensor_names(num_layers):
    names = ["video_embeddings_connector.learnable_registers"]
    for l in range(num_layers):
        p = f"video_embeddings_connector.transformer_1d_blocks.{l}"
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            names.append(f"{p}.attn1.{proj}.weight")
            names.append(f"{p}.attn1.{proj}.bias")
        names += [f"{p}.attn1.q_norm.weight", f"{p}.attn1.k_norm.weight",
                  f"{p}.attn1.to_gate_logits.weight", f"{p}.attn1.to_gate_logits.bias"]
        names += [f"{p}.ff.net.0.proj.weight", f"{p}.ff.net.0.proj.bias",
                  f"{p}.ff.net.2.weight", f"{p}.ff.net.2.bias"]
    return names


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    g = GgufLite(args.gguf)
    print(f"opened {args.gguf}: v{g.version}, {len(g.index)} tensors", flush=True)

    connector = Embeddings1DConnector(
        attention_head_dim=128,
        num_attention_heads=32,
        num_layers=NUM_LAYERS,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4096],
        num_learnable_registers=NUM_REGISTERS,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        apply_gated_attention=True,
        ff_bias=True,
    )
    names = connector_tensor_names(NUM_LAYERS)
    prefix = "video_embeddings_connector."
    # The standalone `Embeddings1DConnector` instance has no parent module
    # supplying that prefix (unlike the real pipeline's `LTXModel`, which
    # never even holds a connector submodule at all - this crate's own doc:
    # the connector is a standalone preprocessing step, not part of
    # `BasicAVTransformerBlock`/`LTXModel`) - strip it to match.
    sd = {name[len(prefix):]: load_real_tensor(g, name) for name in names}
    missing, unexpected = connector.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    connector.eval().requires_grad_(False)
    print(f"connector built: dim={connector.inner_dim}, layers={NUM_LAYERS}, registers={NUM_REGISTERS}", flush=True)

    torch.manual_seed(args.seed)
    hidden = 0.5 * torch.randn(1, S, DIM)
    # additive mask: 0.0 valid, -finfo.max pad - first N_VALID valid, rest padded
    # (right-padded, matching `EmbeddingsProcessor.create_embeddings`'s own
    # `_compute_right_pad_order` normalization before the connector ever runs).
    binary = torch.zeros(1, S)
    binary[:, :N_VALID] = 1.0
    additive_mask = (binary.to(torch.int64) - 1).to(hidden.dtype).reshape(1, 1, 1, S) * torch.finfo(hidden.dtype).max

    out, out_mask = connector(hidden, additive_attention_mask=additive_mask)
    assert not out.isnan().any(), "connector output contains NaN"

    # self-validation: fresh instantiation, bit-identical
    connector2 = Embeddings1DConnector(
        attention_head_dim=128, num_attention_heads=32, num_layers=NUM_LAYERS,
        positional_embedding_theta=10000.0, positional_embedding_max_pos=[4096],
        num_learnable_registers=NUM_REGISTERS, rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True, apply_gated_attention=True, ff_bias=True,
    )
    connector2.load_state_dict(sd, strict=True)
    connector2.eval().requires_grad_(False)
    out2, _ = connector2(hidden, additive_attention_mask=additive_mask)
    d = (out2.double() - out.double()).abs().max().item()
    print(f"self-validate fresh-instantiation: max_abs={d:.3e}", flush=True)
    assert d == 0.0

    tensors = {
        "hidden": hidden[0],
        "valid": binary[0],
        "connector_out": out[0],
    }
    save_file({k: v.detach().to(torch.float32).contiguous() for k, v in tensors.items()},
              os.path.join(args.out, "connector_real.safetensors"))
    manifest = {"run": {"seed": args.seed, "s": S, "n_valid": N_VALID, "dim": DIM,
                        "num_layers": NUM_LAYERS, "num_registers": NUM_REGISTERS}}
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        identity={"dim": DIM, "num_layers": NUM_LAYERS, "num_registers": NUM_REGISTERS},
    )
    with open(os.path.join(args.out, "manifest_connector.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out}/connector_real.safetensors", flush=True)


if __name__ == "__main__":
    main()
